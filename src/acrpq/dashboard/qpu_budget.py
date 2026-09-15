"""Budget policy, atomic reservation, and single-use confirmation (Phase 5.4) — no IBM.

Nothing here submits or contacts IBM; it decides and reserves *locally* so a future real
submission can be gated. All state transitions are inter-process atomic (``os.link``), so
concurrent processes can never over-commit or race a commit against a release.

* :class:`BudgetPolicy` — immutable, versioned caps. ``evaluate`` returns a structured
  ``allowed`` / ``blocked`` / ``requires_explicit_confirmation`` decision.
* :class:`BudgetLedger` — a reservation atomically takes one slot in EACH capped dimension
  (global jobs, per-backend jobs, per-instance jobs; total shots and total QPU-seconds are
  bound through ``effective_max_slots``). Commit and release are a single mutually-exclusive
  atomic transition on a ``.final`` marker (no read/check/unlink race). A reservation is
  identified by id; its slots are derived from the content-hash-verified record, never from
  a caller-supplied path.
* :class:`ConfirmationStore` — a durable single-use nonce bound to the exact identity, with
  content-hash + schema + finite validation, expiry, replay refusal, and atomic consume.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .persist import _publish_atomic_exclusive, _sha256

BUDGET_SCHEMA = "acrpq-qpu-budget/1"
RESERVATION_SCHEMA = "acrpq-qpu-reservation/1"
FINAL_SCHEMA = "acrpq-qpu-reservation-final/1"
CONFIRM_SCHEMA = "acrpq-qpu-confirm/1"
CONFIRM_PENDING_SCHEMA = "acrpq-qpu-confirm-pending/1"


class BudgetError(RuntimeError):
    """A budget/reservation/confirmation invariant was violated; fail closed."""


class BudgetExhausted(BudgetError):
    """No reservation slot is available within some policy cap (message names the dimension)."""


def _canon(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hashed(body: dict) -> dict:
    b = dict(body)
    b["content_sha256"] = _sha256(_canon(b))
    return b


def _verify_hashed(obj: dict, name: str) -> dict:
    if not isinstance(obj, dict):
        raise BudgetError(f"corrupt {name}: not an object")
    stored = obj.get("content_sha256")
    recomputed = _sha256(_canon({k: obj[k] for k in obj if k != "content_sha256"}))
    if stored != recomputed:
        raise BudgetError(f"corrupt {name}: content hash mismatch (tampered/truncated)")
    return obj


def _sanitise(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in s)


# --- policy ---------------------------------------------------------------
@dataclass(frozen=True)
class BudgetPolicy:
    max_jobs: int
    shots_per_job: int
    total_shot_budget: int
    qpu_seconds_per_job: float
    max_total_qpu_seconds: float
    max_wait_seconds: float
    max_jobs_per_backend: int
    max_jobs_per_instance: int
    stop_after_consecutive_errors: int
    schema_version: str = BUDGET_SCHEMA

    def __post_init__(self) -> None:
        for n, v in (("max_jobs", self.max_jobs), ("shots_per_job", self.shots_per_job),
                     ("total_shot_budget", self.total_shot_budget),
                     ("max_jobs_per_backend", self.max_jobs_per_backend),
                     ("max_jobs_per_instance", self.max_jobs_per_instance),
                     ("stop_after_consecutive_errors", self.stop_after_consecutive_errors)):
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise BudgetError(f"{n} must be an int >= 1; got {v!r}")
        for fn, fv in (("qpu_seconds_per_job", self.qpu_seconds_per_job),
                       ("max_total_qpu_seconds", self.max_total_qpu_seconds),
                       ("max_wait_seconds", self.max_wait_seconds)):
            if not isinstance(fv, (int, float)) or isinstance(fv, bool) \
                    or not math.isfinite(fv) or fv <= 0:
                raise BudgetError(f"{fn} must be a finite number > 0; got {fv!r}")

    def effective_max_slots(self) -> int:
        """The binding global cap: no more jobs than the job/shot/QPU-time budgets allow."""
        return min(self.max_jobs,
                   self.total_shot_budget // self.shots_per_job,
                   int(self.max_total_qpu_seconds // self.qpu_seconds_per_job))


@dataclass(frozen=True)
class BudgetDecision:
    action: str                     # allowed | blocked | requires_explicit_confirmation
    reasons: tuple[str, ...]

    @property
    def allowed(self) -> bool:
        return self.action == "allowed"


def evaluate(policy: BudgetPolicy, *, requested_jobs: int, requested_shots_per_job: int,
             backend_jobs_used: int = 0, instance_jobs_used: int = 0,
             consecutive_errors: int = 0) -> BudgetDecision:
    """Structured budget decision. A within-caps billable submission is never silently
    ``allowed`` — it is ``requires_explicit_confirmation``."""
    blocked: list[str] = []
    if requested_jobs < 1:
        blocked.append("requested_jobs<1")
    if requested_shots_per_job != policy.shots_per_job:
        blocked.append("shots_per_job_mismatch")
    if requested_jobs > policy.effective_max_slots():
        blocked.append("exceeds_effective_max_slots")
    if backend_jobs_used + requested_jobs > policy.max_jobs_per_backend:
        blocked.append("exceeds_max_jobs_per_backend")
    if instance_jobs_used + requested_jobs > policy.max_jobs_per_instance:
        blocked.append("exceeds_max_jobs_per_instance")
    if consecutive_errors >= policy.stop_after_consecutive_errors:
        blocked.append("stop_after_consecutive_errors")
    if blocked:
        return BudgetDecision("blocked", tuple(blocked))
    return BudgetDecision("requires_explicit_confirmation",
                          ("within_caps_but_billable_requires_confirmation",))


# --- atomic reservation ---------------------------------------------------
@dataclass(frozen=True)
class Reservation:
    """A reservation handle — an id + the bound identity. Carries NO filesystem path; the
    ledger derives everything from the content-hash-verified record keyed by the id."""

    reservation_id: str
    batch_config_hash: str
    job_config_hash: str
    backend: str
    instance: str
    shots: int
    qpu_seconds: float


class BudgetLedger:
    def __init__(self, root: str, policy: BudgetPolicy) -> None:
        self.root = Path(root)
        self.policy = policy
        (self.root / "reservations").mkdir(parents=True, exist_ok=True)

    # slot namespaces ------------------------------------------------------
    def _slot_dir(self, dim: str, key: str = "_") -> Path:
        return self.root / "slots" / dim / _sanitise(key)

    def _take_slot(self, dim: str, key: str, cap: int) -> int | None:
        d = self._slot_dir(dim, key)
        d.mkdir(parents=True, exist_ok=True)
        for idx in range(cap):
            try:
                _publish_atomic_exclusive(d / f"slot-{idx:06d}", "1")
                return idx
            except FileExistsError:
                continue
        return None

    def _free_slot(self, dim: str, key: str, idx: int) -> None:
        try:
            (self._slot_dir(dim, key) / f"slot-{idx:06d}").unlink()
        except OSError:
            pass

    def slots_used(self, dim: str = "global", key: str = "_") -> int:
        d = self._slot_dir(dim, key)
        return len(list(d.glob("slot-*"))) if d.is_dir() else 0

    # reservation records --------------------------------------------------
    def _record_path(self, rid: str) -> Path:
        return self.root / "reservations" / f"{_sanitise(rid)}.json"

    def _final_path(self, rid: str) -> Path:
        return self.root / "reservations" / f"{_sanitise(rid)}.final"

    def _load_record(self, rid: str) -> dict:
        p = self._record_path(rid)
        if not p.exists():
            raise BudgetError(f"no reservation record for {rid!r}")
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BudgetError(f"corrupt reservation record {rid!r}: {exc}") from exc
        _verify_hashed(obj, f"reservation record {rid!r}")
        if obj.get("schema") != RESERVATION_SCHEMA:
            raise BudgetError(f"reservation record {rid!r} has foreign schema {obj.get('schema')!r}")
        if obj.get("reservation_id") != rid:
            raise BudgetError(f"reservation record id mismatch for {rid!r}")
        return obj

    def reserve(self, *, reservation_id: str, batch_config_hash: str, job_config_hash: str,
                backend: str, instance: str) -> Reservation:
        """Atomically take one slot in EACH capped dimension (global, per-backend,
        per-instance). Rolls back and fails closed if any cap is reached, so concurrent
        callers can never over-commit any dimension."""
        for n, v in (("reservation_id", reservation_id), ("batch_config_hash", batch_config_hash),
                     ("job_config_hash", job_config_hash), ("backend", backend),
                     ("instance", instance)):
            if not isinstance(v, str) or not v.strip():
                raise BudgetError(f"{n} must be a non-empty string")
        shots, qpu = self.policy.shots_per_job, self.policy.qpu_seconds_per_job

        gidx = self._take_slot("global", "_", self.policy.effective_max_slots())
        if gidx is None:
            raise BudgetExhausted("global job/shot/QPU budget exhausted")
        bidx = self._take_slot("backend", backend, self.policy.max_jobs_per_backend)
        if bidx is None:
            self._free_slot("global", "_", gidx)
            raise BudgetExhausted(f"per-backend job cap for {backend!r} exhausted")
        iidx = self._take_slot("instance", instance, self.policy.max_jobs_per_instance)
        if iidx is None:
            self._free_slot("backend", backend, bidx)
            self._free_slot("global", "_", gidx)
            raise BudgetExhausted(f"per-instance job cap for {instance!r} exhausted")

        record = _hashed({
            "schema": RESERVATION_SCHEMA, "reservation_id": reservation_id,
            "batch_config_hash": batch_config_hash, "job_config_hash": job_config_hash,
            "backend": backend, "instance": instance, "shots": shots, "qpu_seconds": qpu,
            "global_slot": gidx, "backend_slot": bidx, "instance_slot": iidx})
        try:
            _publish_atomic_exclusive(self._record_path(reservation_id), _canon(record))
        except FileExistsError:
            self._free_slot("instance", instance, iidx)
            self._free_slot("backend", backend, bidx)
            self._free_slot("global", "_", gidx)
            raise BudgetError(f"reservation_id {reservation_id!r} already exists") from None
        return Reservation(reservation_id, batch_config_hash, job_config_hash, backend,
                           instance, shots, qpu)

    def _transition(self, reservation: Reservation, target: str) -> None:
        rec = self._load_record(reservation.reservation_id)
        # verify the handle matches the persisted, content-hashed record
        for k, v in (("batch_config_hash", reservation.batch_config_hash),
                     ("job_config_hash", reservation.job_config_hash),
                     ("backend", reservation.backend), ("instance", reservation.instance)):
            if rec.get(k) != v:
                raise BudgetError(f"reservation handle {k} does not match the record")
        final = self._final_path(reservation.reservation_id)
        payload = _hashed({"schema": FINAL_SCHEMA,
                           "reservation_id": reservation.reservation_id, "state": target})
        try:
            _publish_atomic_exclusive(final, _canon(payload))
            won = True
        except FileExistsError:
            won = False
        cur = _verify_hashed(json.loads(final.read_text(encoding="utf-8")), "reservation final")
        if cur.get("schema") != FINAL_SCHEMA or cur.get("reservation_id") != reservation.reservation_id:
            raise BudgetError("reservation final marker is foreign/mismatched")
        if cur.get("state") != target:
            raise BudgetError(f"reservation already {cur.get('state')!r}; cannot {target}")
        if won and target == "released":
            # release RETURNS the budget: free every dimension's slot (winner only)
            self._free_slot("instance", rec["instance"], rec["instance_slot"])
            self._free_slot("backend", rec["backend"], rec["backend_slot"])
            self._free_slot("global", "_", rec["global_slot"])

    def is_committed(self, reservation: Reservation) -> bool:
        """True iff this reservation has an intact ``committed`` final marker. Fail-closed:
        a missing/corrupt marker returns False (the submission gate then refuses)."""
        final = self._final_path(reservation.reservation_id)
        if not final.exists():
            return False
        try:
            cur = _verify_hashed(json.loads(final.read_text(encoding="utf-8")), "reservation final")
        except (BudgetError, OSError, ValueError):
            return False
        return (cur.get("schema") == FINAL_SCHEMA
                and cur.get("reservation_id") == reservation.reservation_id
                and cur.get("state") == "committed")

    def commit(self, reservation: Reservation) -> None:
        """Mark the reservation committed (a submission is happening). Atomic and mutually
        exclusive with release; fails closed if already released."""
        self._transition(reservation, "committed")

    def release(self, reservation: Reservation) -> None:
        """Return the reservation's budget (no submission happened). Atomic and mutually
        exclusive with commit; fails closed if already committed."""
        self._transition(reservation, "released")


# --- single-use confirmation ----------------------------------------------
@dataclass(frozen=True)
class Confirmation:
    batch_config_hash: str
    job_config_hash: str
    instance: str
    objective_id: str
    backend: str
    shots: int
    n_qubits: int
    reserved_shots: int
    reserved_qpu_seconds: float
    qubo_sha256: str
    isa_sha256: str
    nonce: str
    expires_epoch: float
    schema_version: str = CONFIRM_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CONFIRM_SCHEMA:
            raise BudgetError(f"unknown confirmation schema {self.schema_version!r}")
        for n, v in (("batch_config_hash", self.batch_config_hash),
                     ("job_config_hash", self.job_config_hash), ("instance", self.instance),
                     ("objective_id", self.objective_id), ("backend", self.backend),
                     ("qubo_sha256", self.qubo_sha256), ("isa_sha256", self.isa_sha256),
                     ("nonce", self.nonce)):
            if not isinstance(v, str) or not v.strip():
                raise BudgetError(f"confirmation {n} must be a non-empty string")
        for iname, iv in (("shots", self.shots), ("n_qubits", self.n_qubits),
                          ("reserved_shots", self.reserved_shots)):
            if not isinstance(iv, int) or isinstance(iv, bool) or iv < 1:
                raise BudgetError(f"confirmation {iname} must be an int >= 1")
        for fn, fv in (("reserved_qpu_seconds", self.reserved_qpu_seconds),
                       ("expires_epoch", self.expires_epoch)):
            if not isinstance(fv, (int, float)) or isinstance(fv, bool) or not math.isfinite(fv):
                raise BudgetError(f"confirmation {fn} must be finite")

    def identity_sha256(self) -> str:
        d = {k: v for k, v in asdict(self).items() if k not in ("nonce", "expires_epoch")}
        return _sha256(_canon(d))


class ConfirmationStore:
    def __init__(self, root: str) -> None:
        self.dir = Path(root) / "confirmations"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _stem(self, nonce: str) -> str:
        return _sanitise(nonce)

    def _pending(self, nonce: str) -> Path:
        return self.dir / f"{self._stem(nonce)}.pending.json"

    def _consumed(self, nonce: str) -> Path:
        return self.dir / f"{self._stem(nonce)}.consumed.json"

    def issue(self, confirmation: Confirmation) -> None:
        """Persist the pending confirmation's identity, nonce and expiry (content-hashed),
        so consume can bind against exactly this record."""
        body = _hashed({"schema": CONFIRM_PENDING_SCHEMA,
                        "identity_sha256": confirmation.identity_sha256(),
                        "nonce": confirmation.nonce,
                        "expires_epoch": float(confirmation.expires_epoch)})
        try:
            _publish_atomic_exclusive(self._pending(confirmation.nonce), _canon(body))
        except FileExistsError:
            raise BudgetError("a confirmation for this nonce was already issued") from None

    def is_consumed(self, confirmation: Confirmation) -> bool:
        """True iff a single-use consumed marker exists AND binds exactly this confirmation
        identity. Fail-closed: missing/corrupt/foreign returns False."""
        p = self._consumed(confirmation.nonce)
        if not p.exists():
            return False
        try:
            cur = _verify_hashed(json.loads(p.read_text(encoding="utf-8")), "consumed confirmation")
        except (BudgetError, OSError, ValueError):
            return False
        return (cur.get("schema") == CONFIRM_SCHEMA and cur.get("nonce") == confirmation.nonce
                and cur.get("identity_sha256") == confirmation.identity_sha256()
                and cur.get("consumed") is True)

    def consume(self, confirmation: Confirmation, *, now_epoch: float) -> None:
        """Atomically consume the nonce for THIS exact confirmation. Fails closed on a
        tampered/truncated/foreign-schema pending record, an identity or nonce mismatch,
        expiry, replay, or an unissued nonce."""
        pending = self._pending(confirmation.nonce)
        if not pending.exists():
            raise BudgetError("no pending confirmation for this nonce (never issued/already gone)")
        try:
            rec = json.loads(pending.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BudgetError(f"corrupt pending confirmation: {exc}") from exc
        _verify_hashed(rec, "pending confirmation")
        if rec.get("schema") != CONFIRM_PENDING_SCHEMA:
            raise BudgetError(f"pending confirmation has foreign schema {rec.get('schema')!r}")
        if rec.get("nonce") != confirmation.nonce:
            raise BudgetError("pending confirmation nonce mismatch")
        if rec.get("identity_sha256") != confirmation.identity_sha256():
            raise BudgetError("confirmation identity does not match the issued nonce "
                              "(backend/shots/config changed)")
        exp = rec.get("expires_epoch")
        if not isinstance(exp, (int, float)) or not math.isfinite(exp):
            raise BudgetError("pending confirmation has a non-finite expiry")
        if now_epoch > float(exp):
            raise BudgetError("confirmation nonce has expired")
        marker = _hashed({"schema": CONFIRM_SCHEMA, "nonce": confirmation.nonce,
                          "identity_sha256": confirmation.identity_sha256(), "consumed": True})
        try:
            _publish_atomic_exclusive(self._consumed(confirmation.nonce), _canon(marker))
        except FileExistsError:
            raise BudgetError("confirmation nonce already consumed (replay refused)") from None
