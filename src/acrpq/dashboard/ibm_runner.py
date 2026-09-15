"""Idempotent IBM QPU submission + reconciliation over the persistent run store.

A QPU submission is a **potentially paid, non-idempotent** operation, so this
layer fails *closed* and never contacts IBM by itself in tests: all IBM access
goes through an injectable :class:`IbmGateway` (structural Protocol). No IBM
token is handled here; the real gateway reads it only from the environment, and
error text is sanitised before persistence so a leaked credential in an
exception is never written to disk.

Safety properties (see the mission invariants):

* the ``local_run_id`` is persisted (by :func:`qpu_runs.prepare`) *before* any
  IBM call and is used as the IBM job **tag**;
* a valid, matching ``expected_config_hash`` is **mandatory** before any gateway
  call — an unconfirmed submission is refused;
* the pre-submit tag search is skew-adjusted and read several times; a job that
  already exists is adopted and **nothing is resubmitted**; a ``max_jobs`` budget
  is enforced before ``run_sampler`` is ever called;
* on submission the job id is persisted immediately; any ordinary exception from
  ``run_sampler`` yields ``submission_unknown`` (never a resubmit), while
  ``KeyboardInterrupt``/``SystemExit`` propagate and leave the run reconcilable;
* reconciliation retries only *reads* (network/timeout), with bounded
  backoff+jitter; an unknown provider status never becomes ``failed``.
"""

from __future__ import annotations

import functools
import hmac
import math
import random
import re
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from . import qpu_runs
from .persist import _now_utc

# Canonical remote job statuses. UNKNOWN is explicit: an unrecognised provider
# status must NEVER be silently treated as a terminal error.
JOB_QUEUED = "QUEUED"
JOB_RUNNING = "RUNNING"
JOB_DONE = "DONE"
JOB_ERROR = "ERROR"
JOB_CANCELLED = "CANCELLED"
JOB_INITIALIZING = "INITIALIZING"
JOB_UNKNOWN = "UNKNOWN"
# Raw provider status (upper-cased) -> canonical. Anything absent here -> UNKNOWN.
_STATUS_ALIASES = {
    "QUEUED": JOB_QUEUED, "RUNNING": JOB_RUNNING, "DONE": JOB_DONE,
    "COMPLETED": JOB_DONE, "ERROR": JOB_ERROR, "FAILED": JOB_ERROR,
    "CANCELLED": JOB_CANCELLED, "CANCELED": JOB_CANCELLED,
    "INITIALIZING": JOB_INITIALIZING,
}

ERROR_CATEGORIES = frozenset(
    {"auth", "quota", "validation", "backend_unavailable", "network", "timeout",
     "ibm_job", "decode", "unknown"}
)
_RETRYABLE_CATEGORIES = frozenset({"network", "timeout"})

_SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_MSG_LEN = 500
_MAX_JOB_ID_LEN = 256


@runtime_checkable
class JobHandle(Protocol):
    def job_id(self) -> str: ...
    def status(self) -> Any: ...
    def cancel(self) -> None: ...


@runtime_checkable
class IbmGateway(Protocol):
    def run_sampler(
        self, *, isa_circuit: Any, shots: int, tags: list[str], max_execution_time: float | None
    ) -> JobHandle: ...
    def find_jobs(self, *, tags: list[str], created_after_iso: str) -> list[JobHandle]: ...
    def get_job(self, job_id: str) -> JobHandle: ...


class SubmissionRefused(RuntimeError):
    """A pre-submit guard failed; no potentially paid call was made."""


# --------------------------------------------------------------------------- #
# Error sanitisation (never persist raw exception text)
# --------------------------------------------------------------------------- #
_REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]+=*"), "Bearer <redacted>"),
    (re.compile(r"(?i)\b(authorization)\b\s*[:=]\s*['\"]?[^\s,;'\"}\]]+"), r"\1: <redacted>"),
    (re.compile(
        r'(?i)"(api[_-]?key|secret|password|token|access[_-]?token|refresh[_-]?token'
        r'|client[_-]?secret|authorization|bearer|credential|private[_-]?key)"\s*:\s*"[^"]*"'
    ), r'"\1": "<redacted>"'),
    (re.compile(
        r"(?i)\b(api[_-]?key|apikey|secret|password|access[_-]?token|refresh[_-]?token"
        r"|client[_-]?secret|token|signature|sig|private[_-]?key)\b\s*[:=]\s*['\"]?[^\s,;'\"}\]&]+"
    ), r"\1=<redacted>"),
    (re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^/\s:@]+:[^/\s@]+@"), r"\1<redacted>@"),
    (re.compile(
        r"(?i)([?&](?:token|api[_-]?key|apikey|access[_-]?token|sig|signature|key|secret|password)=)"
        r"[^&\s]+"
    ), r"\1<redacted>"),
]


def redact_secrets(text: Any) -> str:
    """Redact bearer/authorization/api-key/url-credential/query-token material."""
    s = str(text)
    for pattern, repl in _REDACTIONS:
        s = pattern.sub(repl, s)
    return s


def sanitize_error(exc: BaseException) -> dict[str, Any]:
    """Normalised, secret-free, bounded description of a remote/transport error."""
    return {
        "category": classify_remote_error(exc),
        "type": type(exc).__name__,
        "message": redact_secrets(str(exc))[:_MAX_MSG_LEN],
    }


def classify_remote_error(exc: BaseException) -> str:
    """Best-effort normalised category for a remote/transport error (no qiskit import)."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "auth" in name or "unauthorized" in msg or "invalid token" in msg or "forbidden" in msg:
        return "auth"
    if "quota" in msg or "too many" in msg or ("limit" in msg and "exceed" in msg):
        return "quota"
    if "timeout" in name or "timed out" in msg:
        return "timeout"
    if (
        "connection" in name or "connection" in msg or "network" in msg
        or "temporarily unavailable" in msg or "unreachable" in msg
    ):
        return "network"
    if "backend" in msg and ("unavailable" in msg or "not operational" in msg or "maintenance" in msg):
        return "backend_unavailable"
    if "validation" in name or "invalid" in msg or "not supported" in msg or "unsupported" in msg:
        return "validation"
    if "decode" in msg or "deserial" in msg:
        return "decode"
    if "job" in msg and "ibm" in msg:
        return "ibm_job"
    return "unknown"


def normalise_status(raw: Any) -> str:
    """Map a provider status to the canonical set; anything unrecognised -> UNKNOWN."""
    s = str(getattr(raw, "name", raw)).strip().upper()
    return _STATUS_ALIASES.get(s, JOB_UNKNOWN)


_MAX_RAW_STATUS_LEN = 64


def _raw_status_text(raw: Any) -> str:
    """Sanitised, bounded string form of a raw provider status (str/enum/object).

    Uses the enum ``.name`` when present and never emits a dangerous ``repr``; the
    text is redacted so a status object that stringifies to include a credential
    cannot leak.
    """
    text = getattr(raw, "name", None)
    if not isinstance(text, str):
        text = str(raw)
    return redact_secrets(text)[:_MAX_RAW_STATUS_LEN]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _iso_now() -> str:
    return _now_utc()


def _skewed_iso(created_utc: str, skew_s: float) -> str:
    try:
        return (datetime.fromisoformat(created_utc) - timedelta(seconds=skew_s)).isoformat()
    except ValueError:
        return created_utc


def _validate_retry_params(
    *, reads: int, read_attempts: int, clock_skew_s: float, backoff_base_s: float,
    jitter: Callable[[], float],
) -> None:
    """Validate all retry/search knobs before ANY gateway access (fails closed).

    Rejects zero, negative, boolean, NaN, Inf and wrong types deterministically.
    """
    for name, iv in (("reads", reads), ("read_attempts", read_attempts)):
        if isinstance(iv, bool) or not isinstance(iv, int) or iv < 1:
            raise ValueError(f"{name} must be a non-boolean integer >= 1")
    for name, fv in (("clock_skew_s", clock_skew_s), ("backoff_base_s", backoff_base_s)):
        if isinstance(fv, bool) or not isinstance(fv, (int, float)) or not math.isfinite(fv) or fv < 0:
            raise ValueError(f"{name} must be a finite number >= 0")
    if not callable(jitter):
        raise ValueError("jitter must be callable")
    sample = jitter()  # one sample: must honour the finite, non-negative contract
    if isinstance(sample, bool) or not isinstance(sample, (int, float)) \
            or not math.isfinite(sample) or sample < 0:
        raise ValueError("jitter() must return a finite number >= 0")


def _exec_time_seconds(v: float | None) -> int | None:
    """Whole-second execution cap; ceil so a positive sub-second budget never -> 0."""
    if v is None:
        return None
    return max(1, math.ceil(v))


def _valid_job_id(raw: Any) -> str:
    if not isinstance(raw, str):
        raise ValueError("gateway returned a non-string job id")
    jid = raw.strip()
    if not jid or len(jid) > _MAX_JOB_ID_LEN:
        raise ValueError("gateway returned an empty or over-long job id")
    return jid


def _require_confirmation(expected_config_hash: Any, run: dict[str, Any]) -> None:
    """Mandatory, strict, constant-time confirmation check before any gateway call."""
    if not isinstance(expected_config_hash, str) or not expected_config_hash:
        raise SubmissionRefused("expected_config_hash is required (non-empty string)")
    if not _SHA_RE.match(expected_config_hash):
        raise SubmissionRefused("expected_config_hash must match 'sha256:<64 hex>'")
    if not hmac.compare_digest(expected_config_hash, str(run["config_hash"])):
        raise SubmissionRefused("confirmation is stale: config hash mismatch")


def _retry_read(
    fn: Callable[[], Any], *, attempts: int, backoff_base_s: float,
    sleeper: Callable[[float], None], jitter: Callable[[], float],
) -> Any:
    """Call a READ-ONLY fn, retrying only network/timeout errors, bounded."""
    for i in range(max(1, attempts)):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised unless retryable
            if classify_remote_error(exc) not in _RETRYABLE_CATEGORIES or i == attempts - 1:
                raise
            sleeper(max(0.0, backoff_base_s * (2**i) + jitter()))
    raise RuntimeError("unreachable")  # pragma: no cover


def _spaced_search(
    gateway: IbmGateway, tags: list[str], created_after_iso: str, *,
    reads: int, backoff_base_s: float, sleeper: Callable[[float], None],
    jitter: Callable[[], float], read_attempts: int,
) -> list[JobHandle]:
    """Search the tag several times (empty-tolerant) with per-read network retry."""
    for i in range(reads):
        jobs = _retry_read(
            lambda: gateway.find_jobs(tags=tags, created_after_iso=created_after_iso),
            attempts=read_attempts, backoff_base_s=backoff_base_s, sleeper=sleeper, jitter=jitter,
        )
        if jobs:
            return jobs
        if i < reads - 1:
            sleeper(max(0.0, backoff_base_s * (2**i) + jitter()))
    return []


def _valid_ids(jobs: list[JobHandle]) -> list[str]:
    return list(dict.fromkeys(_valid_job_id(j.job_id()) for j in jobs))


def _adopt_new(local_run_id: str, ids: list[str], already: list[str], *, root: str) -> None:
    for jid in ids:
        if jid not in already:
            qpu_runs.record_job_id(local_run_id, jid, root=root)


def _record_sanitised_error(local_run_id: str, exc: BaseException, *, root: str) -> dict[str, Any]:
    info = sanitize_error(exc)
    qpu_runs.record_error(
        local_run_id, {"type": info["type"], "message": info["message"]},
        category=info["category"], root=root,
    )
    return info


# --------------------------------------------------------------------------- #
# Submission
# --------------------------------------------------------------------------- #
def submit_run(
    local_run_id: str,
    gateway: IbmGateway,
    *,
    isa_circuit: Any,
    expected_config_hash: str | None = None,
    root: str = "runs",
    clock_skew_s: float = 60.0,
    reads: int = 3,
    backoff_base_s: float = 1.0,
    sleeper: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] | None = None,
    read_attempts: int = 3,
    hardware_authorizer: Any = None,
) -> dict[str, Any]:
    """Idempotently submit a prepared+confirmed run. Fails closed; returns a verdict.

    NB: an empty pre-submit search cannot *prove* no job exists — a job created
    remotely but not yet visible (eventual consistency) may still exist. The skew
    window + repeated reads shrink, but do not eliminate, that residual window;
    that is why any ``run_sampler`` failure becomes ``submission_unknown``.
    """
    jitter = jitter or (lambda: random.uniform(0.0, backoff_base_s))
    _validate_retry_params(reads=reads, read_attempts=read_attempts,
                           clock_skew_s=clock_skew_s, backoff_base_s=backoff_base_s, jitter=jitter)

    run = qpu_runs.load(local_run_id, root=root, verify_integrity=True)  # raises if corrupt
    if run["state"] != qpu_runs.QpuState.AWAITING_CONFIRMATION.value:
        raise SubmissionRefused(f"run must be awaiting_confirmation to submit; is {run['state']!r}")
    _require_confirmation(expected_config_hash, run)  # mandatory, before any gateway call

    params = run["params"]
    shots = params["shots"]
    max_qs = params.get("max_quantum_seconds")
    max_jobs = params.get("max_jobs")
    tags = [local_run_id]
    local_ids = list(dict.fromkeys(run["ibm_job_ids"]))

    # Skew-adjusted, repeated pre-submit search. If reads are exhausted (network/
    # timeout) or the error is non-retryable (auth/quota/validation), NO paid call
    # is made: stay awaiting_confirmation, persist a sanitised error, and return a
    # typed verdict so the submission can be retried manually.
    try:
        found = _spaced_search(
            gateway, tags, _skewed_iso(run["created_utc"], clock_skew_s),
            reads=reads, backoff_base_s=backoff_base_s, sleeper=sleeper, jitter=jitter,
            read_attempts=read_attempts,
        )
    except Exception as exc:  # noqa: BLE001 - ordinary read failure; base excs propagate
        info = _record_sanitised_error(local_run_id, exc, root=root)
        return {"verdict": "pre_submit_search_failed", "error_category": info["category"],
                "resubmitted": False, "state": run["state"]}
    found_ids = _valid_ids(found)
    known = list(dict.fromkeys(local_ids + found_ids))
    _adopt_new(local_run_id, found_ids, local_ids, root=root)

    if len(known) > 1:  # never adopt several jobs silently
        qpu_runs.record_reconciliation_evidence(
            local_run_id, {"anomaly": "duplicate_remote_jobs", "job_ids": known}, root=root
        )
        qpu_runs.transition(local_run_id, qpu_runs.QpuState.SUBMISSION_UNKNOWN, root=root)
        return {"verdict": "duplicate_remote_jobs", "job_ids": known, "resubmitted": False}
    if known:  # a job already exists for this tag -> reconcile, never resubmit
        qpu_runs.record_reconciliation_evidence(
            local_run_id, {"stage": "pre_submit", "found_job_ids": known}, root=root
        )
        qpu_runs.transition(local_run_id, qpu_runs.QpuState.SUBMISSION_UNKNOWN, root=root)
        return {"verdict": "existing_job_found", "job_ids": known, "resubmitted": False}

    # Budget: refuse before any paid call if this submission would exceed max_jobs.
    if isinstance(max_jobs, int) and (len(known) + 1) > max_jobs:
        qpu_runs.record_reconciliation_evidence(
            local_run_id, {"anomaly": "max_jobs_budget", "known": len(known), "max_jobs": max_jobs},
            root=root,
        )
        raise SubmissionRefused(f"max_jobs budget {max_jobs} would be exceeded")

    # HARDWARE AUTHORIZATION (Codex items 1-2): require the gate from the persisted
    # destination as well as the concrete gateway type.  Checking RuntimeGateway alone is
    # insufficient: a composition wrapper can delegate to a real RuntimeGateway without
    # being an instance of it.  Local fake-provider backends are the only explicit offline
    # exception; every other destination must present our exact HardwareAuthorizer type.
    # The gate may be skipped ONLY for a provably-local destination. A self-declared
    # "fake*" string is not proof: a composition wrapper delegating to a real
    # RuntimeGateway (so not an isinstance of it) plus a persisted
    # backend_requested="fake_manila" would otherwise reach run_sampler with zero
    # authorization (MoE Expert E, IMPORTANT-1). Require ALL of: an allow-listed local
    # backend name, a gateway that declares itself dry-run-safe, and not a RuntimeGateway.
    backend_requested = str(params.get("backend_requested", "")).strip().casefold()
    # A destination is provably local only if the NAME says fake/local AND the gateway
    # is not a real RuntimeGateway AND it does not expose a real service handle. The
    # previous check trusted the persisted name alone, so a composition wrapper around a
    # real gateway (not an isinstance of it) submitted with zero authorization.
    _looks_local_name = backend_requested.startswith("fake") or backend_requested.startswith("aer")
    _wraps_real_service = any(
        isinstance(getattr(gateway, attr, None), RuntimeGateway)
        or getattr(getattr(gateway, attr, None), "_service", None) is not None
        for attr in ("_inner", "_gateway", "_real", "_delegate", "inner", "gateway")
    ) or getattr(gateway, "_service", None) is not None
    offline_fake_destination = (
        _looks_local_name
        and not isinstance(gateway, RuntimeGateway)
        and not _wraps_real_service
    )
    requires_hardware_authorization = not offline_fake_destination
    if requires_hardware_authorization:
        from .qpu_submit_gate import ApiHardwareAuthorizer, HardwareAuthorizer

        if type(hardware_authorizer) not in (HardwareAuthorizer, ApiHardwareAuthorizer):
            raise SubmissionRefused(
                "non-fake hardware submission requires an exact HardwareAuthorizer type"
            )
        # Bind the authorizer to THIS run: a type check alone let an authorizer minted
        # for run B authorize run A (MoE Expert E, IMPORTANT-2).
        if getattr(hardware_authorizer, "local_run_id", None) != local_run_id:
            raise SubmissionRefused(
                "hardware_authorizer is not bound to this local_run_id"
            )
        _auth_tags = hardware_authorizer.authorize_and_tag()   # runs gate + fresh dedup
        if not (isinstance(_auth_tags, list) and _auth_tags
                and all(isinstance(t, str) and t for t in _auth_tags)):
            raise SubmissionRefused("hardware_authorizer did not return a valid deterministic tag")
        # Submit under the authorizer tag(s) AND the local_run_id: the pre-submit search
        # and reconcile_run both search by [local_run_id], so a crash between run_sampler
        # and job-id persistence must still be discoverable by tag reconciliation
        # (Agent D finding MEDIUM-2 — otherwise the batch-authorizer path's job would be
        # invisible to reconciliation and invite an operator resubmit).
        tags = list(dict.fromkeys([*_auth_tags, local_run_id]))

    qpu_runs.transition(local_run_id, qpu_runs.QpuState.SUBMITTING, root=root)
    try:
        job = gateway.run_sampler(
            isa_circuit=isa_circuit, shots=shots, tags=tags, max_execution_time=max_qs
        )
        job_id = _valid_job_id(job.job_id())  # capture immediately
    except Exception as exc:  # ordinary failure only; KeyboardInterrupt/SystemExit propagate
        info = _record_sanitised_error(local_run_id, exc, root=root)
        qpu_runs.transition(local_run_id, qpu_runs.QpuState.SUBMISSION_UNKNOWN, root=root)
        return {"verdict": "submission_unknown", "error_category": info["category"],
                "resubmitted": False}

    qpu_runs.record_job_id(local_run_id, job_id, root=root)
    qpu_runs.transition(local_run_id, qpu_runs.QpuState.SUBMITTED, root=root)
    return {"verdict": "submitted", "job_id": job_id, "resubmitted": False}


# --------------------------------------------------------------------------- #
# Reconciliation + cancellation
# --------------------------------------------------------------------------- #
_STATUS_TO_STATE = {
    JOB_DONE: qpu_runs.QpuState.COMPLETED,
    JOB_RUNNING: qpu_runs.QpuState.RUNNING,
    JOB_QUEUED: qpu_runs.QpuState.QUEUED,
    JOB_INITIALIZING: qpu_runs.QpuState.QUEUED,
    JOB_ERROR: qpu_runs.QpuState.FAILED,
    JOB_CANCELLED: qpu_runs.QpuState.CANCELLED,
}
_HUB_SOURCES = frozenset(
    {qpu_runs.QpuState.SUBMISSION_UNKNOWN.value, qpu_runs.QpuState.SUBMITTED.value,
     qpu_runs.QpuState.QUEUED.value, qpu_runs.QpuState.RUNNING.value,
     qpu_runs.QpuState.CANCEL_REQUESTED.value}
)


# States routed to submission_unknown first (they never go straight to an IBM
# state): a crash-adopted awaiting/prepared run, and an ambiguous submitting run.
_AMBIGUOUS_SOURCES = frozenset(
    {qpu_runs.QpuState.PREPARED.value, qpu_runs.QpuState.AWAITING_CONFIRMATION.value,
     qpu_runs.QpuState.SUBMITTING.value}
)


def _to_reconciliation_hub(local_run_id: str, state: str, *, root: str) -> str:
    """Route any candidate into reconciliation_required WITHOUT an illegal jump.

    prepared/awaiting_confirmation (a job was adopted just before a crash) and
    submitting all pass through submission_unknown first — never directly to an
    IBM state such as running.
    """
    if state in _AMBIGUOUS_SOURCES:
        qpu_runs.transition(local_run_id, qpu_runs.QpuState.SUBMISSION_UNKNOWN, root=root)
        state = qpu_runs.QpuState.SUBMISSION_UNKNOWN.value
    if state in _HUB_SOURCES:
        qpu_runs.transition(local_run_id, qpu_runs.QpuState.RECONCILIATION_REQUIRED, root=root)
        state = qpu_runs.QpuState.RECONCILIATION_REQUIRED.value
    return state


def reconcile_run(
    local_run_id: str,
    gateway: IbmGateway,
    *,
    root: str = "runs",
    reads: int = 3,
    clock_skew_s: float = 60.0,
    backoff_base_s: float = 1.0,
    sleeper: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] | None = None,
    read_attempts: int = 3,
) -> dict[str, Any]:
    """Determine a run's true state from IBM by tag/job-id. Never resubmits."""
    jitter = jitter or (lambda: random.uniform(0.0, backoff_base_s))
    _validate_retry_params(reads=reads, read_attempts=read_attempts,
                           clock_skew_s=clock_skew_s, backoff_base_s=backoff_base_s, jitter=jitter)

    run = qpu_runs.load(local_run_id, root=root, verify_integrity=True)
    state = run["state"]
    if run["terminal"]:
        return {"verdict": "already_terminal", "state": state}
    # prepared / awaiting_confirmation with NO job is not an active job; WITH a job
    # (adopted just before a crash) it IS reconcilable and routed via the hub.
    if state in {qpu_runs.QpuState.PREPARED.value, qpu_runs.QpuState.AWAITING_CONFIRMATION.value} \
            and not run["ibm_job_ids"]:
        return {"verdict": "not_submitted", "state": state}

    # Fast idempotent poll path: a stable provider status is an observation, not
    # a state transition.  Do not churn queued/running through the reconciliation
    # hub on every UI polling tick.  Any read error, status change or duplicate
    # job falls through to the full recovery path below.
    initial_known = list(dict.fromkeys(run["ibm_job_ids"]))
    if state in {qpu_runs.QpuState.QUEUED.value, qpu_runs.QpuState.RUNNING.value} \
            and len(initial_known) == 1:
        try:
            stable_job = _retry_read(
                functools.partial(gateway.get_job, initial_known[0]),
                attempts=read_attempts, backoff_base_s=backoff_base_s,
                sleeper=sleeper, jitter=jitter)
            stable_raw = _retry_read(
                lambda: stable_job.status(), attempts=read_attempts,
                backoff_base_s=backoff_base_s, sleeper=sleeper, jitter=jitter)
            stable_status = normalise_status(stable_raw)
            stable_target = _STATUS_TO_STATE.get(stable_status)
            if stable_target is not None and stable_target.value == state:
                return {"verdict": "state_unchanged", "state": state,
                        "job_status": stable_status, "job_id": initial_known[0]}
        except Exception:  # noqa: BLE001 - full recovery path records a sanitised failure
            pass

    state = _to_reconciliation_hub(local_run_id, state, root=root)
    run = qpu_runs.load(local_run_id, root=root, verify_integrity=True)
    known = list(dict.fromkeys(run["ibm_job_ids"]))

    try:
        if known:
            jobs: list[JobHandle] = [
                _retry_read(functools.partial(gateway.get_job, j), attempts=read_attempts,
                            backoff_base_s=backoff_base_s, sleeper=sleeper, jitter=jitter)
                for j in known
            ]
        else:
            jobs = _spaced_search(
                gateway, [local_run_id], _skewed_iso(run["created_utc"], clock_skew_s),
                reads=reads, backoff_base_s=backoff_base_s, sleeper=sleeper, jitter=jitter,
                read_attempts=read_attempts,
            )
    except Exception as exc:  # noqa: BLE001 - reads exhausted / non-retryable
        info = _record_sanitised_error(local_run_id, exc, root=root)
        return {"verdict": "read_retry_exhausted", "error_category": info["category"],
                "state": qpu_runs.QpuState.RECONCILIATION_REQUIRED.value}

    if not jobs:
        qpu_runs.record_reconciliation_evidence(
            local_run_id, {"stage": "search", "found": 0, "reads": reads}, root=root
        )
        return {"verdict": "no_remote_job", "resubmitted": False,
                "needs_explicit_reconfirmation": True,
                "state": qpu_runs.QpuState.RECONCILIATION_REQUIRED.value}

    unique = _valid_ids(jobs)
    _adopt_new(local_run_id, unique, known, root=root)
    if len(unique) > 1:
        qpu_runs.record_reconciliation_evidence(
            local_run_id, {"anomaly": "duplicate_remote_jobs", "job_ids": unique}, root=root
        )
        return {"verdict": "duplicate_remote_jobs", "job_ids": unique,
                "state": qpu_runs.QpuState.RECONCILIATION_REQUIRED.value}

    try:
        raw_status = _retry_read(lambda: jobs[0].status(), attempts=read_attempts,
                                 backoff_base_s=backoff_base_s, sleeper=sleeper, jitter=jitter)
    except Exception as exc:  # noqa: BLE001
        info = _record_sanitised_error(local_run_id, exc, root=root)
        return {"verdict": "read_retry_exhausted", "error_category": info["category"],
                "state": qpu_runs.QpuState.RECONCILIATION_REQUIRED.value}
    status = normalise_status(raw_status)

    if status == JOB_UNKNOWN:  # never mark failed on an unrecognised status
        # keep BOTH the canonical UNKNOWN and the sanitised, bounded raw value so a
        # later observation/reconciliation can act on it (e.g. VALIDATING -> RUNNING)
        qpu_runs.record_provider_observation(
            local_run_id,
            {"canonical": JOB_UNKNOWN, "raw_status": _raw_status_text(raw_status)},
            root=root,
        )
        return {"verdict": "unknown_provider_status", "job_id": unique[0],
                "raw_status": _raw_status_text(raw_status),
                "state": qpu_runs.QpuState.RECONCILIATION_REQUIRED.value}

    target = _STATUS_TO_STATE[status]
    qpu_runs.transition(local_run_id, target, root=root)
    return {"verdict": "reconciled", "state": target.value, "job_status": status,
            "job_id": unique[0]}


def request_cancel(
    local_run_id: str, gateway: IbmGateway, *, root: str = "runs",
    read_attempts: int = 3, backoff_base_s: float = 1.0,
    sleeper: Callable[[float], None] = time.sleep, jitter: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Best-effort cancel. Only IBM's own report moves the run to cancelled/completed.

    Retry discipline: ``get_job`` and ``status()`` are reads (bounded network/
    timeout retry); ``cancel()`` is a remote mutation and is called **exactly
    once** — never retried — so a transient error can never fan out into repeated
    cancel attempts.
    """
    jitter = jitter or (lambda: random.uniform(0.0, backoff_base_s))
    _validate_retry_params(reads=1, read_attempts=read_attempts, clock_skew_s=0.0,
                           backoff_base_s=backoff_base_s, jitter=jitter)
    run = qpu_runs.load(local_run_id, root=root, verify_integrity=True)
    if run["terminal"]:
        return {"verdict": "already_terminal", "state": run["state"]}
    if run["state"] not in {
        qpu_runs.QpuState.SUBMITTED.value, qpu_runs.QpuState.QUEUED.value,
        qpu_runs.QpuState.RUNNING.value,
    }:
        raise qpu_runs.IllegalTransitionError(f"cannot request cancel from state {run['state']!r}")

    job_ids = list(dict.fromkeys(run["ibm_job_ids"]))
    if len(job_ids) > 1:  # never guess which of several jobs to cancel
        qpu_runs.record_reconciliation_evidence(
            local_run_id, {"anomaly": "duplicate_remote_jobs", "job_ids": job_ids}, root=root
        )
        return {"verdict": "duplicate_remote_jobs_requires_resolution", "job_ids": job_ids,
                "state": run["state"]}
    if not job_ids:
        raise qpu_runs.IllegalTransitionError("no IBM job id to cancel")

    job_id = job_ids[0]
    qpu_runs.transition(local_run_id, qpu_runs.QpuState.CANCEL_REQUESTED, root=root)

    # 1) obtain the handle — a READ, retryable on network/timeout
    job: Any = None
    try:
        job = _retry_read(functools.partial(gateway.get_job, job_id), attempts=read_attempts,
                          backoff_base_s=backoff_base_s, sleeper=sleeper, jitter=jitter)
    except Exception as exc:  # noqa: BLE001
        qpu_runs.record_reconciliation_evidence(
            local_run_id, {"cancel_get_job_error": redact_secrets(str(exc))[:_MAX_MSG_LEN]}, root=root
        )
    # 2) request cancellation — a MUTATION, called at most once, NEVER retried
    if job is not None:
        try:
            job.cancel()
        except Exception as exc:  # noqa: BLE001 - best-effort, single attempt
            qpu_runs.record_reconciliation_evidence(
                local_run_id, {"cancel_error": redact_secrets(str(exc))[:_MAX_MSG_LEN]}, root=root
            )

    # 3) read the resulting status — a READ, retryable on network/timeout
    try:
        status = normalise_status(
            _retry_read(lambda: gateway.get_job(job_id).status(), attempts=read_attempts,
                        backoff_base_s=backoff_base_s, sleeper=sleeper, jitter=jitter)
        )
    except Exception as exc:  # noqa: BLE001 - status unreadable -> stay cancel_requested
        qpu_runs.record_reconciliation_evidence(
            local_run_id, {"cancel_status_error": redact_secrets(str(exc))[:_MAX_MSG_LEN]}, root=root
        )
        return {"verdict": "cancel_requested", "job_id": job_id,
                "note": "status unreadable; a running computation may finish server-side"}

    if status == JOB_CANCELLED:
        qpu_runs.transition(local_run_id, qpu_runs.QpuState.CANCELLED, root=root)
        return {"verdict": "cancelled", "job_id": job_id}
    if status == JOB_DONE:  # IBM already finished -> completed, not cancelled
        qpu_runs.transition(local_run_id, qpu_runs.QpuState.COMPLETED, root=root)
        return {"verdict": "completed", "job_id": job_id}
    return {"verdict": "cancel_requested", "job_id": job_id, "job_status": status,
            "note": "a running computation may finish server-side; not claimed as stopped"}


# --------------------------------------------------------------------------- #
# Real gateway (lazy qiskit imports; module stays qiskit-free at import)
# --------------------------------------------------------------------------- #
class RuntimeGateway:
    """Real gateway over qiskit-ibm-runtime. All imports are lazy.

    ``run_sampler`` works from either an explicit ``backend`` (e.g. a
    ``fake_provider`` backend, for local testing without an account) or a
    ``service`` + ``backend_name``. ``find_jobs``/``get_job`` require a
    ``QiskitRuntimeService``. No token is handled here.
    """

    def __init__(self, *, service: Any = None, backend: Any = None, backend_name: str | None = None):
        self._service = service
        self._backend = backend
        self._backend_name = backend_name

    def _resolve_backend(self) -> Any:
        if self._backend is not None:
            return self._backend
        if self._service is not None and self._backend_name:
            return self._service.backend(self._backend_name)
        raise RuntimeError("RuntimeGateway needs a backend or a service + backend_name")

    def run_sampler(
        self, *, isa_circuit: Any, shots: int, tags: list[str], max_execution_time: float | None
    ) -> Any:
        from qiskit_ibm_runtime import SamplerV2

        sampler = SamplerV2(mode=self._resolve_backend())
        sampler.options.default_shots = shots
        sampler.options.environment.job_tags = list(tags)
        exec_s = _exec_time_seconds(max_execution_time)
        if exec_s is not None:
            sampler.options.max_execution_time = exec_s
        return sampler.run([isa_circuit])

    def find_jobs(self, *, tags: list[str], created_after_iso: str) -> list[Any]:
        if self._service is None:
            raise RuntimeError("find_jobs requires a QiskitRuntimeService")
        after = datetime.fromisoformat(created_after_iso)
        return list(self._service.jobs(job_tags=list(tags), created_after=after))

    def get_job(self, job_id: str) -> Any:
        if self._service is None:
            raise RuntimeError("get_job requires a QiskitRuntimeService")
        return self._service.job(job_id)

    def get_result(self, job: Any) -> dict[str, Any]:
        """Read a finished job's counts per the Runtime 0.47.0 SamplerV2 contract.

        Returns ``{"counts": {bitstring: int}, "bit_order": "qiskit", "kind": "counts",
        "n_bits": int}``. Never fabricates counts: an unreadable/empty result raises
        so the caller keeps the run 'completed but result not retrieved'.
        """
        result = job.result()          # SamplerV2 PrimitiveResult (list of pub results)
        pubs = list(result)
        if not pubs:
            raise RuntimeError("sampler result contained no pub results")
        data = pubs[0].data
        # the classical register name varies; take the first BitArray-like field
        reg = next((getattr(data, n) for n in getattr(data, "__dict__", {})
                    if hasattr(getattr(data, n), "get_counts")), None)
        if reg is None:
            for n in ("meas", "c", "cr"):
                if hasattr(data, n) and hasattr(getattr(data, n), "get_counts"):
                    reg = getattr(data, n)
                    break
        if reg is None or not hasattr(reg, "get_counts"):
            raise RuntimeError("could not locate a counts register in the sampler result")
        counts = {str(k): int(v) for k, v in reg.get_counts().items()}
        if not counts:
            raise RuntimeError("sampler result had empty counts")
        n_bits = len(next(iter(counts)))
        return {"counts": counts, "bit_order": "qiskit", "kind": "counts", "n_bits": n_bits}
