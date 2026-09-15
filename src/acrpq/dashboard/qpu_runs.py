"""Persistent, event-sourced store for external (IBM QPU) run records.

A QPU run is *mutable* — it advances through a state machine and accumulates IBM
job IDs — unlike the write-once scientific records in :mod:`persist`. To keep that
mutable history crash-safe and inter-process-safe **without a second, incoherent
persistence abstraction**, each run is an append-only event log built on the same
atomic-exclusive primitive as :mod:`persist`: every event is a distinct file
created once (``os.link``, POSIX) and never overwritten, so no event — in
particular no *received IBM job ID* — can be lost or clobbered, even across
processes. The current state is a fold over the ordered events; a restart simply
replays them.

Integrity & safety:

* every event carries a typed :class:`EventKind` and is validated against a
  per-kind schema; reserved fields (seq/stamp_utc/prev_sha256/content_sha256)
  cannot be injected;
* a **mode-dependent** table of allowed transitions is enforced; hardware modes
  require an explicit ``awaiting_confirmation`` step (no ``prepared→submitting``);
  ``submission_unknown`` can only terminate through a determined
  ``reconciliation_required``, never straight to ``failed``;
* a terminal run (completed/failed/cancelled) is immutable for state/result, but
  still accepts **probative, append-only** evidence events (late_job_id,
  provider_metrics, provider_observation, reconciliation_evidence) that can never
  change the state or the result;
* parameters are deep-copied before hashing/persistence, validated by a strict
  per-mode schema, and refused if they contain secrets (recursively), tuples,
  NaN/Infinity or any non-JSON-native value (no silent ``default=str``);
* integrity is verified by default on every load, and a corrupt run is surfaced
  as ``corrupted`` — the reconciler never receives a corrupt/incomplete run.

Honest limits: each event carries a self-hash (catches a naive edit anywhere,
incl. the tail) and a ``prev_sha256`` chain link (cryptographically binds every
non-tail event). It is tamper-*evident*, not tamper-proof: a fully-consistent
rewrite of the tail event would need a signature to detect.
"""

from __future__ import annotations

import copy
import json
import math
import re
from enum import Enum
from pathlib import Path
from typing import Any

from .persist import (
    _now_utc,
    _publish_atomic_exclusive,
    _run_id,
    _sha256,
    _validate_run_id,
    config_hash,
)

QPU_SCHEMA_VERSION = "acrpq-qpu-run/1"
_KIND_DIR = "qpu"

QPU_MODES = frozenset(
    {"aer_simulation", "hardware_validation", "full_qaoa_hardware", "recorded_hardware_result"}
)
_HARDWARE_MODES = frozenset({"hardware_validation", "full_qaoa_hardware"})

_EVENT_WIDTH = 8
MAX_EVENTS = 1_000_000
MAX_JOB_ID_LEN = 256

_RESERVED_EVENT_FIELDS = frozenset({"seq", "stamp_utc", "prev_sha256", "content_sha256"})
_SECRET_MARKERS = frozenset(
    {"token", "secret", "password", "apikey", "credential", "authorization", "bearer", "privatekey"}
)
_SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class QpuState(str, Enum):
    PREPARED = "prepared"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    SUBMISSION_UNKNOWN = "submission_unknown"
    RECONCILIATION_REQUIRED = "reconciliation_required"


_STATE_VALUES = frozenset(s.value for s in QpuState)
TERMINAL_STATES = frozenset({QpuState.COMPLETED, QpuState.FAILED, QpuState.CANCELLED})
_TERMINAL_VALUES = frozenset(s.value for s in TERMINAL_STATES)


class EventKind(str, Enum):
    TRANSITION = "transition"
    JOB_ID = "job_id"
    RESULT = "result"
    ERROR = "error"
    BATCH_IDENTITY = "batch_identity"   # binds the full batch_config_hash into the chain
    # probative, append-only evidence allowed even after a terminal state:
    LATE_JOB_ID = "late_job_id"
    PROVIDER_METRICS = "provider_metrics"
    PROVIDER_OBSERVATION = "provider_observation"
    RECONCILIATION_EVIDENCE = "reconciliation_evidence"


_KIND_VALUES = frozenset(k.value for k in EventKind)
# Evidence kinds accepted after a terminal state; they may never carry state/result.
_PROBATIVE_KINDS = frozenset(
    {EventKind.LATE_JOB_ID.value, EventKind.PROVIDER_METRICS.value,
     EventKind.PROVIDER_OBSERVATION.value, EventKind.RECONCILIATION_EVIDENCE.value}
)

# Strict per-kind field schema over *caller* fields. ``event_kind`` is allowed
# implicitly; seq/stamp_utc/prev_sha256/content_sha256 are reserved (added by the
# log, never by callers). Any field outside a kind's ``allowed`` set is refused,
# so no arbitrary field can ride along — and only TRANSITION may carry ``state``,
# only RESULT may carry ``result`` (plus the *documented* atomic completed+result
# where a TRANSITION to ``completed`` may include the result in one event).
_EVENT_SCHEMA: dict[str, dict[str, frozenset[str]]] = {
    "transition": {"required": frozenset({"state"}), "allowed": frozenset({"state", "result"})},
    "result": {"required": frozenset({"result"}), "allowed": frozenset({"result"})},
    "error": {"required": frozenset({"error"}), "allowed": frozenset({"error", "error_category"})},
    "job_id": {"required": frozenset({"job_id"}), "allowed": frozenset({"job_id"})},
    "late_job_id": {"required": frozenset({"job_id"}), "allowed": frozenset({"job_id"})},
    "provider_metrics": {"required": frozenset({"metrics"}), "allowed": frozenset({"metrics"})},
    "provider_observation": {
        "required": frozenset({"observation"}), "allowed": frozenset({"observation"})
    },
    "reconciliation_evidence": {
        "required": frozenset({"evidence"}), "allowed": frozenset({"evidence"})
    },
    "batch_identity": {
        "required": frozenset({"batch_config_hash"}),
        "allowed": frozenset({"batch_config_hash"}),
    },
}

# Base transition table (state value -> allowed next state values). Terminal
# states map to the empty set. submission_unknown intentionally omits 'failed':
# an ambiguous submission must first be reconciled.
_ALLOWED: dict[str, frozenset[str]] = {
    "prepared": frozenset(
        {"awaiting_confirmation", "submitting", "submission_unknown", "failed",
         "cancelled", "cancel_requested"}
    ),
    "awaiting_confirmation": frozenset(
        # 'submission_unknown' is reachable here only via the pre-submit tag search
        # discovering a prior remote job for this local_run_id (an ambiguity).
        {"submitting", "submission_unknown", "cancelled", "cancel_requested", "failed"}
    ),
    "submitting": frozenset(
        {"submitted", "submission_unknown", "failed", "cancel_requested", "cancelled"}
    ),
    "submitted": frozenset(
        {"queued", "running", "completed", "failed", "cancel_requested",
         "reconciliation_required"}
    ),
    "queued": frozenset(
        {"running", "completed", "failed", "cancel_requested", "cancelled",
         "reconciliation_required"}
    ),
    "running": frozenset(
        {"completed", "failed", "cancel_requested", "reconciliation_required"}
    ),
    "cancel_requested": frozenset(
        {"cancelled", "completed", "failed", "reconciliation_required"}
    ),
    "submission_unknown": frozenset(
        {"reconciliation_required", "submitted", "cancelled"}  # NOT 'failed' directly
    ),
    "reconciliation_required": frozenset(
        {"submitted", "queued", "running", "completed", "failed", "cancelled",
         "submission_unknown"}
    ),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


class IllegalTransitionError(ValueError):
    """A requested state transition is not permitted from the current state."""


class CorruptRunError(Exception):
    """A run's integrity chain/hashes do not verify."""


def _allowed_next(mode: str | None, current: str | None) -> frozenset[str]:
    base = _ALLOWED.get(current or "", frozenset())
    # hardware modes must pass through awaiting_confirmation before submitting
    if mode in _HARDWARE_MODES and current == "prepared":
        base = base - {"submitting"}
    return base


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _scan_secrets(obj: Any, path: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(m in _norm_key(k) for m in _SECRET_MARKERS):
                raise ValueError(f"refusing to persist a secret-like field: {path}{k!r}")
            _scan_secrets(v, f"{path}{k}.")
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _scan_secrets(item, f"{path}{i}.")


def _assert_json_native_finite(obj: Any, path: str = "") -> None:
    """Refuse non-JSON-native types, tuples, non-str keys and NaN/Infinity."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite number at {path or '<root>'}")
        return
    if isinstance(obj, tuple):
        raise ValueError(
            f"tuple not accepted at {path or '<root>'} (would be silently coerced to a list)"
        )
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError(f"non-string dict key {k!r} at {path or '<root>'}")
            _assert_json_native_finite(v, f"{path}{k}.")
        return
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_json_native_finite(item, f"{path}{i}.")
        return
    raise ValueError(f"non-JSON-native value of type {type(obj).__name__} at {path or '<root>'}")


def _clean(obj: dict[str, Any]) -> None:
    _scan_secrets(obj)
    _assert_json_native_finite(obj)


def _json_strict(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)


def _canonical(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _clean_job_id(job_id: Any) -> str:
    if not isinstance(job_id, str):
        raise ValueError("job_id must be a string")
    jid = job_id.strip()
    if not jid:
        raise ValueError("job_id must be non-empty after stripping")
    if len(jid) > MAX_JOB_ID_LEN:
        raise ValueError(f"job_id exceeds {MAX_JOB_ID_LEN} characters")
    return jid


def _require_int(params: dict, key: str, lo: int, hi: int, *, required: bool) -> None:
    if key not in params:
        if required:
            raise ValueError(f"missing required param {key!r}")
        return
    v = params[key]
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"param {key!r} must be an integer")
    if not lo <= v <= hi:
        raise ValueError(f"param {key!r} must be in [{lo}, {hi}]")


def validate_params(params: dict[str, Any]) -> dict[str, Any]:
    """Strictly validate prepared-run params (per-mode schema); return them unchanged."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")
    _clean(params)  # recursive: no secrets, JSON-native, finite, no tuples

    mode = params.get("mode")
    if mode not in QPU_MODES:
        raise ValueError(f"param 'mode' must be one of {sorted(QPU_MODES)}")
    inst = params.get("instance")
    if not isinstance(inst, str) or not inst.strip():
        raise ValueError("param 'instance' must be a non-empty string")
    _require_int(params, "n_theta", 1, 25, required=True)
    _require_int(params, "shots", 1, 1_000_000, required=True)
    _require_int(params, "n_q", 1, 10, required=False)
    _require_int(params, "reps", 1, 100, required=False)
    _require_int(params, "maxiter", 1, 1_000_000, required=False)
    if "objective_id" in params:
        from ..objectives import coerce_objective_id

        coerce_objective_id(params["objective_id"])

    hardware = mode in _HARDWARE_MODES
    _require_int(params, "transpiler_seed", 0, 2**31 - 1, required=hardware)
    _require_int(params, "max_jobs", 1, 10_000, required=hardware)
    if hardware or "max_quantum_seconds" in params:
        v = params.get("max_quantum_seconds")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError("param 'max_quantum_seconds' must be a finite number > 0")
    if hardware:
        if "qubo_sha256" not in params:
            raise ValueError("hardware modes require param 'qubo_sha256'")
        br = params.get("backend_requested")
        if not isinstance(br, str) or not br.strip():
            raise ValueError("hardware modes require a non-empty 'backend_requested'")

    subset = params.get("subset")
    if subset is not None:
        if (
            not isinstance(subset, list) or not subset
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 1 for i in subset)
            or len(set(subset)) != len(subset)
        ):
            raise ValueError("param 'subset' must be a non-empty list of unique aircraft ids >= 1")
    for hkey in ("qubo_sha256", "source_sha256"):
        if hkey in params and not (isinstance(params[hkey], str) and _SHA_RE.match(params[hkey])):
            raise ValueError(f"param {hkey!r} must match 'sha256:<64 hex>'")
    return params


def _validate_event(event: dict[str, Any]) -> None:
    """Reserved-field, secret, native and STRICT per-kind schema checks.

    Every event is validated against :data:`_EVENT_SCHEMA`: required fields must
    be present and no field outside the kind's whitelist is accepted, so arbitrary
    fields (and cross-kind fields like ``state`` on an error) are impossible.
    """
    if not isinstance(event, dict):
        raise ValueError("event must be a dict")
    reserved = _RESERVED_EVENT_FIELDS & set(event)
    if reserved:
        raise ValueError(f"reserved event field(s) may not be set by callers: {sorted(reserved)}")
    _clean(event)  # secrets / native / finite / no tuples — before kind checks

    kind = event.get("event_kind")
    if kind not in _KIND_VALUES:
        raise ValueError(f"event_kind must be one of {sorted(_KIND_VALUES)}; got {kind!r}")
    schema = _EVENT_SCHEMA[kind]
    keys = set(event) - {"event_kind"}
    missing = schema["required"] - keys
    if missing:
        raise ValueError(f"{kind} event is missing required field(s): {sorted(missing)}")
    extra = keys - schema["allowed"]
    if extra:
        raise ValueError(f"{kind} event has unexpected field(s): {sorted(extra)}")

    if kind == "transition":
        if event["state"] not in _STATE_VALUES:
            raise ValueError(f"invalid transition state {event['state']!r}")
        if "result" in event and event["state"] != QpuState.COMPLETED.value:
            raise ValueError("a transition may carry 'result' only when state == 'completed'")
    elif kind in ("job_id", "late_job_id"):
        _clean_job_id(event["job_id"])
    elif kind == "provider_metrics":
        if not isinstance(event["metrics"], dict) or not event["metrics"]:
            raise ValueError("provider_metrics requires a non-empty dict 'metrics'")
    elif kind == "reconciliation_evidence":
        if not isinstance(event["evidence"], dict) or not event["evidence"]:
            raise ValueError("reconciliation_evidence requires a non-empty dict 'evidence'")


# --------------------------------------------------------------------------- #
# Layout / low-level log
# --------------------------------------------------------------------------- #
def _run_dir(root: str, local_run_id: str) -> Path:
    return Path(root) / _KIND_DIR / local_run_id


def _event_files(run_dir: Path) -> list[Path]:
    files = []
    for p in run_dir.glob("*.event.json"):
        stem = p.name.split(".", 1)[0]
        if stem.isdigit():
            files.append(p)
    return sorted(files, key=lambda p: int(p.name.split(".", 1)[0]))


def _tail_text(run_dir: Path, events: list[Path]) -> str:
    tail = events[-1] if events else run_dir / "manifest.json"
    return tail.read_text(encoding="utf-8")


def _current_state(events: list[Path]) -> str | None:
    for p in reversed(events):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if "state" in rec:
            return rec["state"]
    return None


def _manifest_mode(run_dir: Path) -> str | None:
    try:
        return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["params"]["mode"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def _append(run_dir: Path, event: dict[str, Any], *, max_retries: int = 256) -> int:
    """Assign the next seq, re-validate the transition, and write atomically.

    Validation and write are atomic w.r.t. concurrent appenders: on a sequence
    collision the current state is re-folded and the rules re-checked, so two
    incompatible concurrent transitions cannot both succeed.
    """
    local_run_id = run_dir.name
    mode = _manifest_mode(run_dir)
    kind = event.get("event_kind")
    for _ in range(max_retries):
        events = _event_files(run_dir)
        if len(events) >= MAX_EVENTS:
            raise RuntimeError(f"run {local_run_id!r} reached the {MAX_EVENTS}-event cap")
        seq = int(events[-1].name.split(".", 1)[0]) + 1 if events else 0
        if events:  # not the seed event
            current = _current_state(events)
            if current in _TERMINAL_VALUES:
                # terminal is immutable for state/result; only probative evidence allowed
                if kind not in _PROBATIVE_KINDS:
                    raise IllegalTransitionError(
                        f"run {local_run_id!r} is terminal ({current}); only probative "
                        f"evidence events are allowed, not {kind!r}"
                    )
            elif kind == "transition":
                target = event["state"]
                if target not in _allowed_next(mode, current):
                    raise IllegalTransitionError(
                        f"transition {current} -> {target} not allowed"
                        + (f" for mode {mode}" if mode in _HARDWARE_MODES else "")
                    )
        record = {
            "seq": seq,
            "stamp_utc": _now_utc(),
            "prev_sha256": _sha256(_tail_text(run_dir, events)),
            **event,
        }
        record["content_sha256"] = _sha256(_canonical({k: v for k, v in record.items()}))
        try:
            _publish_atomic_exclusive(
                run_dir / f"{seq:0{_EVENT_WIDTH}d}.event.json", _json_strict(record)
            )
            return seq
        except FileExistsError:
            continue  # concurrent appender took this seq: re-fold, re-validate, retry
    raise RuntimeError(f"could not append event for {local_run_id!r} after {max_retries} tries")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def prepare(params: dict[str, Any], *, root: str = "runs") -> str:
    """Create a run and return its id. The initial event is the commit point.

    ``params`` is deep-copied before validation/hash/persist, so a later mutation
    of the caller's dict cannot change the stored snapshot.
    """
    params = validate_params(copy.deepcopy(params))
    created = _now_utc()
    chash = config_hash(dict(params))
    local_run_id = _run_id(created, chash)
    run_dir = _run_dir(root, local_run_id)
    manifest = {
        "schema": QPU_SCHEMA_VERSION,
        "local_run_id": local_run_id,
        "created_utc": created,
        "config_hash": chash,
        "params": params,
    }
    _publish_atomic_exclusive(run_dir / "manifest.json", _json_strict(manifest))
    try:
        _append(run_dir, {"event_kind": "transition", "state": QpuState.PREPARED.value})
    except BaseException:
        try:
            (run_dir / "manifest.json").unlink()
        except OSError:
            pass
        raise
    return local_run_id


def _committed(run_dir: Path) -> bool:
    return (run_dir / "manifest.json").is_file() and bool(_event_files(run_dir))


def append_event(local_run_id: str, event: dict[str, Any], *, root: str = "runs") -> int:
    """Append a validated, typed event. Rejects reserved/secret fields and bad kinds."""
    _validate_run_id(local_run_id)
    _validate_event(event)
    run_dir = _run_dir(root, local_run_id)
    if not _committed(run_dir):
        raise FileNotFoundError(f"no committed QPU run {local_run_id!r} under {root}")
    return _append(run_dir, event)


def transition(
    local_run_id: str, state: QpuState, *, root: str = "runs", result: Any = None
) -> int:
    """Append a validated state-transition event.

    ``result`` may be supplied ONLY for a transition to ``completed`` (the
    documented atomic completed+result event); any other state with a result is
    rejected. Errors/job-ids/observations are separate typed events.
    """
    if not isinstance(state, QpuState):
        raise TypeError("state must be a QpuState")
    event: dict[str, Any] = {"event_kind": "transition", "state": state.value}
    if result is not None:
        event["result"] = result
    return append_event(local_run_id, event, root=root)


def record_result(local_run_id: str, result: Any, *, root: str = "runs") -> int:
    """Append a standalone RESULT event."""
    return append_event(local_run_id, {"event_kind": "result", "result": result}, root=root)


def record_error(
    local_run_id: str, error: Any, *, root: str = "runs", category: str | None = None
) -> int:
    """Append an ERROR event (its error plus an optional normalised category)."""
    event: dict[str, Any] = {"event_kind": "error", "error": error}
    if category is not None:
        event["error_category"] = category
    return append_event(local_run_id, event, root=root)


def record_job_id(local_run_id: str, job_id: str, *, root: str = "runs") -> int:
    """Persist a received IBM job ID immediately (append-only → never lost)."""
    return append_event(
        local_run_id, {"event_kind": "job_id", "job_id": _clean_job_id(job_id)}, root=root
    )


_BATCH_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def record_batch_identity(local_run_id: str, batch_config_hash: str, *, root: str = "runs") -> int:
    """Bind the FULL batch_config_hash into the authoritative append-only chain. Appended
    once, before AWAITING_CONFIRMATION, so a run is provably tied to its protocol identity.
    Idempotent for the SAME hash; a conflicting hash is refused by :func:`batch_identity_of`.
    A malformed hash is refused up front."""
    if not isinstance(batch_config_hash, str) or not _BATCH_HASH_RE.match(batch_config_hash):
        raise ValueError("batch_config_hash must be 'sha256:<64 hex>'")
    existing = batch_identity_of(local_run_id, root=root)
    if existing is not None:
        if existing != batch_config_hash:
            raise IllegalTransitionError(
                f"run {local_run_id!r} already bound to batch identity {existing}; "
                f"refusing to rebind to {batch_config_hash}")
        return -1                                       # idempotent: already bound identically
    return append_event(
        local_run_id, {"event_kind": "batch_identity", "batch_config_hash": batch_config_hash},
        root=root)


def batch_identity_of(local_run_id: str, *, root: str = "runs") -> str | None:
    """The batch_config_hash bound in the run's chain, or None if never bound (legacy).
    Fail-closed (CorruptRunError) if the chain carries two conflicting identities."""
    state = load(local_run_id, root=root, verify_integrity=True)
    return state.get("batch_config_hash")


def record_late_job_id(local_run_id: str, job_id: str, *, root: str = "runs") -> int:
    """Attach a job ID discovered *after* the run reached a terminal state (probative)."""
    return append_event(
        local_run_id, {"event_kind": "late_job_id", "job_id": _clean_job_id(job_id)}, root=root
    )


def record_provider_metrics(local_run_id: str, metrics: dict[str, Any], *, root: str = "runs") -> int:
    """Attach provider metrics (e.g. quantum seconds) as probative evidence."""
    return append_event(
        local_run_id, {"event_kind": "provider_metrics", "metrics": metrics}, root=root
    )


def record_provider_observation(local_run_id: str, observation: Any, *, root: str = "runs") -> int:
    return append_event(
        local_run_id, {"event_kind": "provider_observation", "observation": observation}, root=root
    )


def record_reconciliation_evidence(local_run_id: str, evidence: Any, *, root: str = "runs") -> int:
    return append_event(
        local_run_id, {"event_kind": "reconciliation_evidence", "evidence": evidence}, root=root
    )


def verify(local_run_id: str, *, root: str = "runs") -> dict[str, Any]:
    """Recompute self-hash + ``prev_sha256`` chain; raise ``CorruptRunError`` on mismatch."""
    _validate_run_id(local_run_id)
    run_dir = _run_dir(root, local_run_id)
    if not (run_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"no QPU run {local_run_id!r} under {root}")
    files = [run_dir / "manifest.json", *_event_files(run_dir)]
    for i in range(1, len(files)):
        record = json.loads(files[i].read_text(encoding="utf-8"))
        expected_prev = _sha256(files[i - 1].read_text(encoding="utf-8"))
        if record.get("prev_sha256") != expected_prev:
            raise CorruptRunError(f"integrity chain broken at {files[i].name} for {local_run_id!r}")
        stored = record.pop("content_sha256", None)
        if stored != _sha256(_canonical(record)):
            raise CorruptRunError(
                f"event content hash mismatch at {files[i].name} for {local_run_id!r}"
            )
    return {"local_run_id": local_run_id, "ok": True, "n_files": len(files)}


def load(local_run_id: str, *, root: str = "runs", verify_integrity: bool = True) -> dict[str, Any]:
    """Fold the manifest + ordered events into the current run state.

    Integrity is verified by default; pass ``verify_integrity=False`` only for
    diagnostics. A corrupt run raises ``CorruptRunError`` so it is never acted on.
    """
    _validate_run_id(local_run_id)
    run_dir = _run_dir(root, local_run_id)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no QPU run {local_run_id!r} under {root}")
    events = _event_files(run_dir)
    if not events:
        raise ValueError(f"run {local_run_id!r} is incomplete (no initial event)")
    if verify_integrity:
        verify(local_run_id, root=root)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state: str | None = None
    job_ids: list[str] = []
    late_job_ids: list[str] = []
    transitions: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    result: Any = None
    error: Any = None
    batch_config_hash: str | None = None
    for p in events:
        ev = json.loads(p.read_text(encoding="utf-8"))
        kind = ev.get("event_kind")
        # Fold STRICTLY by event_kind — never by the mere presence of a key, so a
        # non-transition event can never change the state and a non-result event
        # can never change the result.
        if kind == "transition":
            state = ev["state"]
            transitions.append({"seq": ev["seq"], "state": ev["state"], "stamp_utc": ev["stamp_utc"]})
            if "result" in ev:  # the documented atomic completed+result
                result = ev["result"]
        elif kind == "result":
            result = ev["result"]
        elif kind == "error":
            error = ev["error"]
        elif kind == "job_id":
            if ev["job_id"] not in job_ids:
                job_ids.append(ev["job_id"])
        elif kind == "late_job_id":
            if ev["job_id"] not in late_job_ids:
                late_job_ids.append(ev["job_id"])
        elif kind in ("provider_metrics", "provider_observation", "reconciliation_evidence"):
            observations.append(
                {"seq": ev["seq"], "stamp_utc": ev["stamp_utc"], "event_kind": kind,
                 "payload": ev.get("metrics") or ev.get("observation") or ev.get("evidence")}
            )
        elif kind == "batch_identity":
            bch = ev["batch_config_hash"]
            if batch_config_hash is not None and batch_config_hash != bch:
                raise CorruptRunError(
                    f"run {local_run_id!r} has conflicting batch identities "
                    f"{batch_config_hash} != {bch}")
            batch_config_hash = bch
    return {
        "schema": manifest["schema"],
        "local_run_id": local_run_id,
        "created_utc": manifest["created_utc"],
        "config_hash": manifest["config_hash"],
        "params": manifest["params"],
        "state": state,
        "batch_config_hash": batch_config_hash,
        "ibm_job_ids": job_ids,
        "late_ibm_job_ids": late_job_ids,
        "transitions": transitions,
        "provider_observations": observations,
        "result": result,
        "error": error,
        "n_events": len(events),
        "terminal": state in _TERMINAL_VALUES,
    }


def raw_events(local_run_id: str, *, root: str = "runs", after_seq: int = -1,
               limit: int | None = None) -> dict[str, Any]:
    """Return the full ordered append-only event stream (ALL kinds), sequenced.

    Each event: ``{seq, stamp_utc, event_kind, payload}`` where payload is the
    kind's caller fields (already secret-scanned + JSON-native at write time).
    ``after_seq``/``limit`` give safe forward pagination; ``total`` is the true
    event count so a caller can assert ``total == n_events``.
    """
    _validate_run_id(local_run_id)
    run_dir = _run_dir(root, local_run_id)
    if not _committed(run_dir):
        raise FileNotFoundError(f"no committed QPU run {local_run_id!r} under {root}")
    files = _event_files(run_dir)
    reserved = {"seq", "stamp_utc", "prev_sha256", "content_sha256", "event_kind"}
    events: list[dict[str, Any]] = []
    for p in files:
        ev = json.loads(p.read_text(encoding="utf-8"))
        seq = int(ev["seq"])
        if seq <= after_seq:
            continue
        events.append({
            "seq": seq, "stamp_utc": ev["stamp_utc"], "event_kind": ev["event_kind"],
            "payload": {k: v for k, v in ev.items() if k not in reserved},
        })
    total = len(files)
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive int")
        events = events[:limit]
    return {"local_run_id": local_run_id, "total": total, "returned": len(events),
            "events": events}


def list_runs(*, root: str = "runs") -> list[dict[str, Any]]:
    """All committed QPU runs (most recent first), integrity-verified.

    Incomplete runs (manifest without a commit event) are invisible here. A run
    that fails verification is surfaced with ``state="corrupted"``.
    """
    base = Path(root) / _KIND_DIR
    if not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for run_dir in base.iterdir():
        if not _committed(run_dir):
            continue
        name = run_dir.name
        try:
            out.append(load(name, root=root, verify_integrity=True))
        except CorruptRunError as exc:
            out.append(_corrupt_marker(name, str(exc), _manifest_config_hash(run_dir)))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            out.append(_corrupt_marker(name, f"{type(exc).__name__}: {exc}",
                                       _manifest_config_hash(run_dir)))
    out.sort(key=lambda r: r["local_run_id"], reverse=True)
    return out


def _manifest_config_hash(run_dir: Path) -> str | None:
    """Read a run's config_hash from its manifest — but ONLY when the manifest is
    provably intact, so a *tampered* manifest can never masquerade as a different
    (or valid) config. The manifest seeds the hash chain: the first event's
    ``prev_sha256`` equals ``_sha256(manifest_text)``. We recompute that link and
    return the config_hash only if it matches; otherwise (manifest tampered/desynced,
    no anchoring event, or unreadable) we return None so the caller fails closed.

    This survives event-log corruption (the manifest and event0 can still be intact
    while a later event rots) yet does NOT trust a manifest whose own bytes changed."""
    try:
        manifest_text = (run_dir / "manifest.json").read_text(encoding="utf-8")
        events = _event_files(run_dir)
        if not events:
            return None                                  # no chain to anchor -> untrusted
        first = json.loads(events[0].read_text(encoding="utf-8"))
        if first.get("prev_sha256") != _sha256(manifest_text):
            return None                                  # manifest tampered/desynced
        ch = json.loads(manifest_text).get("config_hash")
        return ch if isinstance(ch, str) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _corrupt_marker(name: str, reason: str, config_hash: str | None = None) -> dict[str, Any]:
    return {
        "local_run_id": name, "state": "corrupted", "terminal": False,
        "corrupt_reason": reason, "config_hash": config_hash,
        "ibm_job_ids": [], "late_ibm_job_ids": [],
        "transitions": [], "provider_observations": [],
    }


def non_terminal_runs(*, root: str = "runs") -> list[dict[str, Any]]:
    """Every non-terminal run, INCLUDING corrupted ones (broad view)."""
    return [r for r in list_runs(root=root) if not r.get("terminal")]


# States a reconciler may act on: a submission was attempted (or is ambiguous).
# 'prepared'/'awaiting_confirmation' are excluded (no submission → nothing to
# reconcile), as are corrupted/incomplete/terminal runs.
_RECONCILABLE_STATES = frozenset(
    {"submitting", "submitted", "queued", "running", "submission_unknown",
     "reconciliation_required", "cancel_requested"}
)


_PRE_SUBMIT_STATES = frozenset({"prepared", "awaiting_confirmation"})


def reconciliation_candidates(*, root: str = "runs") -> list[dict[str, Any]]:
    """Runs the reconciler may safely act on (a submission was attempted/ambiguous).

    A prepared/awaiting_confirmation run is normally excluded (no submission), but
    if it carries an IBM job id — a job adopted or received just before a crash,
    leaving the state un-advanced — it MUST become a candidate so the crash is
    recoverable. Corrupted runs are always excluded (they go to
    :func:`attention_runs`).
    """
    out = []
    for r in list_runs(root=root):
        st = r.get("state")
        if st in _RECONCILABLE_STATES:
            out.append(r)
        elif st in _PRE_SUBMIT_STATES and r.get("ibm_job_ids"):
            out.append(r)  # crash-adopted job -> recoverable
    return out


def attention_runs(*, root: str = "runs") -> list[dict[str, Any]]:
    """Runs needing human attention: corrupted (bad integrity) or incomplete."""
    base = Path(root) / _KIND_DIR
    out: list[dict[str, Any]] = []
    if not base.is_dir():
        return out
    for run_dir in sorted(base.iterdir(), reverse=True):
        if not (run_dir / "manifest.json").is_file():
            continue
        name = run_dir.name
        if not _event_files(run_dir):
            out.append({"local_run_id": name, "state": "incomplete", "terminal": False,
                        "reason": "manifest without a commit event"})
            continue
        try:
            load(name, root=root, verify_integrity=True)
        except CorruptRunError as exc:
            out.append(_corrupt_marker(name, str(exc)))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            out.append(_corrupt_marker(name, f"{type(exc).__name__}: {exc}"))
    return out
