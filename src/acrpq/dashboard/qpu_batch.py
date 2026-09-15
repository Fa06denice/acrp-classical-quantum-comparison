"""Offline IBM-batch orchestration (Phase 5) — fake gateways ONLY, never real IBM.

This drives the *entire* safe QPU submission pipeline — prepare → confirm →
submit → reconcile → completed → decode — for a small batch of instances, but
exclusively through the fenced ``qpu_orchestrator.simulate_lifecycle`` seam with a
**fake** gateway and a **local** fake backend. It NEVER constructs a real
``RuntimeGateway``, NEVER reads a token, and NEVER contacts IBM.

Four independent safety layers make a real submission impossible from here:

1. ``assert_offline`` refuses to run at all if the environment is configured for
   real submission (``qpu_flags.submission_allowed`` true) or if the real runtime
   factory is enabled (``qpu_ibm_factory.runtime_factory_enabled`` true).
2. The lifecycle driver fences the gateway by EXACT TYPE
   (``require_offline_exact_gateway``): only a registered offline type
   (:class:`FakeBatchGateway`) passes — NOT a self-declared ``dry_run_safe``
   marker (which a composition wrapper around a real gateway can trivially fake),
   NOT a subclass, and never a real ``RuntimeGateway``.
3. The real ``RealIbmGatewayFactory`` / ``RuntimeGateway`` / ``QiskitRuntimeService``
   are never referenced; the only gateway ever built is a ``FakeBatchGateway`` that
   holds pre-computed local data and cannot reach a real backend.
4. The ISA circuit is transpiled against a LOCAL qiskit ``fake_provider`` backend
   (default :class:`FakeGuadalupeV2`, allow-listed to ``Fake*`` names), so no
   service/account is ever constructed.

Counts from ``gateway.get_result()`` flow through the SAME normalisation adapter a
real IBM provider result would (:func:`normalize_result` →
``hardware_validation._normalise_distribution``). The batch is persistent, atomic
and RESUMABLE (:class:`BatchJobStore`): each job's ``local_run_id`` is stored before
submission, so a crash at any boundary resumes onto the same run and a job is not
submitted twice — subject to the durability conditions below
(``ibm_runner.submit_run`` is only ever reached for an awaiting-confirmation run
with no job).

**Durability of the no-double-submit guarantee.** The authority on what has already
been prepared/submitted is the durable run store (``runs_dir``), NOT the batch
index. Before preparing a fresh run the driver scans ``qpu_runs.list_runs`` for an
existing run with the same ``config_hash`` (:func:`_find_run_by_config_hash`) and
reuses it — so even if the batch index is lost *after* a submission, the existing
run is reconciled, never resubmitted; and a run orphaned by a crash before the
first index write is adopted and submitted exactly once. A run whose event log has
been CORRUPTED is not mistaken for "never submitted": the scan matches it by its
CHAIN-VERIFIED manifest ``config_hash`` — the manifest seeds the run's hash chain
(``event0.prev_sha256 == sha256(manifest)``), so an accidentally-corrupted or naively
single-field-tampered manifest is detected and its hash rejected — and fails CLOSED,
refusing to prepare a possible duplicate, when that hash matches, or when the manifest
cannot be chain-verified so a match cannot be ruled out. Only a corrupt run whose
manifest is chain-verified intact AND whose config differs is ignored (it cannot be a
prior submission of this job). This inherits ``qpu_runs``' honest limit — tamper
*evident*, not tamper *proof*: an adversary who ALSO re-anchors ``event0.prev_sha256``
to a co-tampered manifest is the signature-required rewrite class, out of scope for
this offline phase (and offline-harmless: a synthetic job, never a real submission).
**Config-identity guard.** The run-store scan keys on ``bundle.prepared.config_hash``,
which — inherited from the hardware-validation param model — OMITS gammas/betas/
optimization_level. Two batch configs differing only in those axes share that run hash,
so the scan alone could adopt the WRONG config's run. :func:`_claimed_by_other_config`
closes this whenever the batch index is intact (the common case): a run already claimed
by a different :func:`batch_config_hash` is never adopted, so each config prepares its
own run. With a LOST index those axes cannot be distinguished (the run hash inherits the
un-modifiable hardware-validation contract) — the same residual class as losing the
runs_dir, and offline-harmless (synthetic counts, no real submission).

The one residual DURABILITY case (distinct from that adversarial one) is losing
the ``runs_dir`` **itself** together with the index: offline that only forfeits
resume (a fresh run is a new synthetic job, no real duplicate); for a FUTURE real
provider it would require provider-side job-tag dedup (out of scope for this offline
phase, flagged in the artifact safety block; the same dedup would also cover a
concurrent-prepare race, which the sequential offline CLI does not hit). Cross-restart
resume therefore requires a persistent ``runs_dir`` (``--runs-dir``); the default
ephemeral tempdir forfeits it.

The ``synthetic_counts`` that stand in for what a Sampler would return are a
delta on the CLASSICAL optimum bitstring (from exhaustive enumeration), so the
decoded result can be checked against the known optimum — demonstrating the
decode/reconcile path end-to-end. They are always labelled synthetic; no quantum
sampling occurs.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

from ..classical.discrete_enum import exhaustive_discrete_optimum
from ..io.loader import InstanceLoader
from ..quantum.qubo import build_qubo
from . import ibm_runner, qpu_claim, qpu_flags, qpu_ibm_factory, qpu_orchestrator, qpu_runs
from . import hardware_validation as H

BATCH_SCHEMA = "acrpq-qpu-batch-offline/2"
BATCH_PROTOCOL_VERSION = BATCH_SCHEMA   # protocol identity token in the batch config hash
DEFAULT_FAKE_BACKEND = "FakeGuadalupeV2"   # local, 16 qubits — no network/token

# Fields excluded from the batch content hash: volatile run-store identifiers
# (timestamped run ids) and wall-clock timings. Everything scientific is kept.
_VOLATILE = ("local_run_id", "wall_time_s", "cpu_time_s", "created_utc")


class OfflineSafetyError(RuntimeError):
    """The environment is configured for real submission; the offline batch refuses."""


def batch_config_identity(
    *, instance: str, objective_id: str, n_theta: int, n_q: int,
    gammas: Any, betas: Any, shots: int, transpiler_seed: int,
    optimization_level: int, backend_name: str, qubo_sha256: str,
    max_quantum_seconds: float,
) -> dict:
    """FULL protocol identity of an offline batch job — captures EVERY axis that
    changes the job or its provenance, so a stored result can never be reused across a
    config change: protocol version, instance, objective, grid (n_theta/n_q), angles
    (gammas/betas), shots, transpiler seed, transpilation level, backend, the QUBO hash
    (which itself binds objective+grid+instance), and the quantum-time budget
    (``max_quantum_seconds`` — it flows into the plan/artefact hashes). ``qubo_sha256``
    is passed in so the caller computes it once. All values coerced to JSON-native so
    the hash is stable."""
    return {
        "protocol_version": BATCH_PROTOCOL_VERSION,
        "instance": str(instance),
        "objective_id": str(objective_id),
        "n_theta": int(n_theta),
        "n_q": int(n_q),
        "gammas": [float(g) for g in gammas],
        "betas": [float(b) for b in betas],
        "shots": int(shots),
        "transpiler_seed": int(transpiler_seed),
        "optimization_level": int(optimization_level),
        "backend_name": str(backend_name),
        "qubo_sha256": str(qubo_sha256),
        "max_quantum_seconds": float(max_quantum_seconds),
    }


def batch_config_hash(identity: dict) -> str:
    """Canonical sha256 over a :func:`batch_config_identity` dict (sorted keys, no NaN)."""
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _assert_matching_config(store: BatchJobStore, job_key: str, config_hash: str) -> None:
    """Fail-closed: refuse to reuse OR overwrite a store entry that was created for a
    DIFFERENT protocol config. An existing entry whose recorded ``config_hash`` differs
    from (or is absent for) the current one is a hard :class:`OfflineSafetyError` — the
    old result is never returned and never clobbered. A change to shots / angles / seed /
    transpilation level / backend / objective / grid / protocol version therefore cannot
    silently ride on a stale ``job_key``."""
    prior = store.load(job_key)              # already fails closed on a corrupt entry
    if prior is None:
        return
    stored = prior.get("config_hash")
    if stored != config_hash:
        raise OfflineSafetyError(
            f"batch store entry {job_key!r} was created for a different protocol config "
            f"(stored config_hash={stored!r} != current {config_hash!r}); refusing to reuse "
            f"or overwrite it. A config change (shots/angles/seed/transpile-level/backend/"
            f"objective/grid/protocol-version) needs a distinct job_key — derive job_key "
            f"from the config hash, or clear the store."
        )


def assert_offline(*, env: Callable[[str], str | None] | None = None) -> dict:
    """Fail-closed: refuse to run if the environment could permit a real submission.

    Returns a snapshot of the safety flags (all expected False) for the artifact.
    """
    kw = {} if env is None else {"env": env}
    snapshot = qpu_flags.flags_snapshot(**kw)
    factory_enabled = qpu_ibm_factory.runtime_factory_enabled(**kw)
    if snapshot["submission_allowed"]:
        raise OfflineSafetyError(
            "refusing to run the OFFLINE batch: qpu_flags.submission_allowed() is True "
            "(the environment is configured for real IBM submission)")
    if factory_enabled:
        raise OfflineSafetyError(
            "refusing to run the OFFLINE batch: the real IBM runtime factory is enabled "
            "(ACRPQ_IBM_RUNTIME_FACTORY_ENABLED) — this CLI is fake-gateway-only")
    return {**snapshot, "runtime_factory_enabled": factory_enabled}


class FakeBatchGateway:
    """A deterministic, local fake gateway (never contacts IBM).

    Returns a single DONE job whose result is the caller-supplied synthetic counts.
    This module's driver fences it by EXACT TYPE (not the ``dry_run_safe`` marker,
    which is kept only for backward compatibility with the shared simulate seam and
    is NOT the security basis — a composition wrapper can fake it).
    """

    dry_run_safe = True

    def __init__(self, counts: dict[str, int], *, bit_order: str, n_bits: int,
                 job_id: str = "offline-fake-job-0") -> None:
        self._counts = dict(counts)
        self._bit_order = bit_order
        self._n_bits = n_bits
        self._job_id = job_id
        self.run_calls = 0

    class _Job:
        def __init__(self, jid: str) -> None:
            self._id = jid

        def job_id(self) -> str:
            return self._id

        def status(self) -> str:
            return "DONE"

        def cancel(self) -> None:
            return None

    def run_sampler(self, *, isa_circuit: Any, shots: int, tags: list[str],
                    max_execution_time: float | None) -> "FakeBatchGateway._Job":
        self.run_calls += 1
        return self._Job(self._job_id)

    def find_jobs(self, *, tags: list[str], created_after_iso: str) -> list:
        return []

    def get_job(self, job_id: str) -> "FakeBatchGateway._Job":
        return self._Job(job_id)

    def get_result(self, job: Any) -> dict:
        return {"counts": dict(self._counts), "bit_order": self._bit_order,
                "kind": "counts", "n_bits": self._n_bits}


# The set of EXACT offline gateway types this module's driver accepts. This is a
# TYPE fence, not a self-declared ``dry_run_safe`` marker: a self-attested boolean
# is trivially defeated by composition (a wrapper that forwards run_sampler to a
# REAL gateway while setting ``dry_run_safe = True`` passes the marker check — see
# tests/test_qpu_batch_offline.py::test_composition_wrapper_defeats_marker...). An
# exact ``type(gateway) in _OFFLINE_GATEWAY_TYPES`` check refuses that wrapper (its
# type is the wrapper, not FakeBatchGateway) AND any subclass, and FakeBatchGateway
# holds only pre-computed local data — it cannot wrap or reach a real gateway.
_OFFLINE_GATEWAY_TYPES: tuple[type, ...] = (FakeBatchGateway,)


def require_offline_exact_gateway(gateway: Any) -> None:
    """Fail-closed: accept ONLY an exact registered offline gateway type. Refuses a
    real RuntimeGateway (belt-and-suspenders) and anything else — including a
    composition wrapper that fakes a marker."""
    if isinstance(gateway, ibm_runner.RuntimeGateway):
        raise OfflineSafetyError("refusing a real RuntimeGateway")
    if type(gateway) not in _OFFLINE_GATEWAY_TYPES:
        raise OfflineSafetyError(
            f"gateway must be an exact offline type {[t.__name__ for t in _OFFLINE_GATEWAY_TYPES]}; "
            f"got {type(gateway).__name__} (a self-declared marker is not accepted)")


def normalize_result(envelope: Any, *, expected_n_bits: int) -> dict:
    """Normalise a gateway.get_result() envelope into a canonical
    ``{qubo_bitstring: probability}`` map via the SAME adapter the real IBM
    provider result flows through (hardware_validation._normalise_distribution).

    Handles counts and quasi-distributions; applies endianness (bit_order); and
    fails closed on an empty / malformed / non-finite result. A fake gateway thus
    exercises the exact normalisation a real Sampler result would.
    """
    if not isinstance(envelope, dict):
        raise H.HardwareValidationError("result envelope must be a dict")
    raw = envelope.get("counts")
    if raw is None or not isinstance(raw, dict):
        raise H.HardwareValidationError("result envelope missing a 'counts'/'quasi' dict")
    kind = envelope.get("kind", "counts")
    if kind not in ("counts", "quasi"):
        raise H.HardwareValidationError("kind must be 'counts' or 'quasi'")
    bit_order = envelope.get("bit_order", "qiskit")
    if bit_order not in ("qiskit", "qubo"):
        raise H.HardwareValidationError("bit_order must be 'qiskit' or 'qubo'")
    if not raw:
        raise H.HardwareValidationError("empty result: no bitstrings returned")
    try:  # fold every downstream validation failure into one fail-closed type
        prob, n_shots = H._normalise_distribution(
            raw, expected_n_bits, bit_order=bit_order, kind=kind, shots=None)
    except H.HardwareValidationError:
        raise
    except (ValueError, TypeError) as exc:              # e.g. non-finite probability
        raise H.HardwareValidationError(f"malformed gateway result: {exc}") from exc
    return {"prob_qubo_order": prob, "n_shots": n_shots, "kind": kind,
            "bit_order_in": bit_order, "n_unique": len(raw)}


def optimum_onehot_bits(inst: Any, qubo: Any, objective_id: str, n_theta: int, n_q: int) -> dict:
    """The exhaustive-optimum solution as a one-hot bitstring (qubo bit order).

    Returns ``{bitstring, optimum, feasible, n_optima}`` or an ``unavailable``
    marker when the search space exceeds the enumeration cap (then no synthetic
    delta can be pinned to a certified optimum)."""
    res = exhaustive_discrete_optimum(inst, objective_id=objective_id, n_theta=n_theta, n_q=n_q)
    if res.status != "optimal":
        return {"status": res.status, "bitstring": None, "optimum": None, "feasible": None}
    choice = res.representative_choice
    d = qubo.discrete
    bits = ["0"] * qubo.n_qubits
    for i in inst.aircraft():
        bits[d.var_index(i, choice[i - 1])] = "1"
    return {"status": "optimal", "bitstring": "".join(bits), "optimum": res.optimum,
            "feasible": res.feasible, "n_optima": res.n_optima}


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(obj, sort_keys=True, allow_nan=False))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class BatchJobStore:
    """Persistent, atomic, resumable per-job index for a batch. One JSON file per
    job records the assigned ``local_run_id`` (persisted BEFORE submission), the
    stage, and — once finished — the exported record. Atomic (temp + os.replace)
    so a crash at any point leaves a consistent, recoverable file. On resume the
    stored ``local_run_id`` is reused, so a job is NEVER prepared/submitted twice."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, job_key: str) -> Path:
        safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in job_key)
        return self.root / f"{safe}.json"

    def load(self, job_key: str) -> dict | None:
        p = self._path(job_key)
        if not p.exists():
            return None
        try:
            obj = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            # Fail-closed: a torn/corrupt index entry is operator-actionable, not a
            # fresh job. Returning None here would look like "never submitted" and
            # could drive a re-prepare/re-submit — refuse instead.
            raise OfflineSafetyError(
                f"corrupt batch-store entry {p.name}: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(obj, dict):
            raise OfflineSafetyError(
                f"corrupt batch-store entry {p.name}: expected object, got {type(obj).__name__}"
            )
        return obj

    def record(self, job_key: str, data: dict) -> None:
        cur = self.load(job_key) or {}
        cur.update(data)
        cur["job_key"] = job_key
        _atomic_write_json(self._path(job_key), cur)

    def is_exported(self, job_key: str) -> bool:
        d = self.load(job_key)
        return bool(d and d.get("stage") == "exported")

    def iter_entries(self) -> "Iterator[dict]":
        """Yield every stored entry dict (fail-closed on a corrupt file, like load())."""
        for p in sorted(self.root.glob("*.json")):
            try:
                obj = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                raise OfflineSafetyError(
                    f"corrupt batch-store entry {p.name}: {type(exc).__name__}: {exc}"
                ) from exc
            if isinstance(obj, dict):
                yield obj


def _claimed_by_other_config(store: BatchJobStore, local_run_id: str, config_hash: str) -> bool:
    """True if a DIFFERENT batch config already claims this run (a store entry records
    this ``local_run_id`` under a config_hash != the current one). The run-store scan
    (:func:`_find_run_by_config_hash`) keys on ``bundle.prepared.config_hash``, which —
    inherited from the hardware-validation param model — OMITS gammas/betas/
    optimization_level. So two batch configs differing only in those axes share one run
    hash; without this guard the scan would wrongly ADOPT the other config's run. This
    closes that hole whenever the batch index is intact (the common case). A LOST index
    cannot distinguish those axes — the same residual class as losing the runs_dir, and
    offline-harmless (synthetic counts)."""
    for entry in store.iter_entries():
        if entry.get("local_run_id") != local_run_id:
            continue
        ch = entry.get("config_hash")
        if ch is None:
            # An entry references this run but records no config_hash (only reachable via
            # a legacy/hand-written/foreign file — every write here stamps one). We cannot
            # confirm it is THIS config's run; fail closed rather than silently adopt it
            # (which could be a different config) or prepare a fresh one (which could double
            # a submission). This keeps the guard symmetric with _assert_matching_config.
            raise OfflineSafetyError(
                f"batch store entry {entry.get('job_key')!r} claims run {local_run_id!r} "
                f"without a config_hash; cannot confirm ownership — refusing to adopt or "
                f"resubmit ambiguously (inspect/repair the store)"
            )
        if ch != config_hash:
            return True
    return False


def _claims_dir(root: str) -> str:
    """Sibling of the qpu run directory holding the single-winner claim/runid files."""
    return str(Path(root) / "qpu-claims")


def _parse_utc(ts: str) -> float | None:
    from datetime import datetime
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None


def list_orphan_runs(root: str, batch_config_hash: str, *, now_epoch: float | None = None,
                     lease_seconds: float = 3600.0) -> dict:
    """READ-ONLY inventory of orphan candidate runs for a batch config.

    An orphan is a run that shares the DESIGNATED run's run-store ``config_hash`` but is
    not the designated ``local_run_id``, carries no ``ibm_job_ids``, and is non-terminal
    and non-corrupt — i.e. a candidate that a crash-time recovery prepared but never
    designated (see :mod:`qpu_claim`). Each is classified against a safety lease:
    ``safe_to_gc`` if older than ``lease_seconds`` (provably not the in-flight winner),
    else ``possibly_active`` (leave alone). This function never mutates or deletes
    anything — cleanup is a separate, operator-gated step and must NEVER remove an
    active (submitted/terminal/leased) run."""
    from datetime import datetime, timezone
    now = now_epoch if now_epoch is not None else datetime.now(timezone.utc).timestamp()
    designated = qpu_claim.read_designated(batch_config_hash, _claims_dir(root))
    out: dict = {"batch_config_hash": batch_config_hash, "designated_local_run_id": designated,
                 "orphans": []}
    if designated is None:
        return out
    runs = qpu_runs.list_runs(root=root)
    by_id = {r["local_run_id"]: r for r in runs}
    des = by_id.get(designated)
    if des is None:
        out["warning"] = "designated run named by the claim is absent from the run store"
        return out
    pch = des.get("config_hash")
    out["designated_run_config_hash"] = pch
    for r in runs:
        rid = r["local_run_id"]
        if rid == designated or r.get("config_hash") != pch:
            continue
        if r.get("state") == "corrupted" or r.get("terminal") or r.get("ibm_job_ids"):
            continue                                    # active/terminal/corrupt -> never an orphan-GC target
        age = None
        created = _parse_utc(r.get("created_utc", ""))
        if created is not None:
            age = max(0.0, now - created)
        classification = ("safe_to_gc" if (age is not None and age >= lease_seconds)
                          else "possibly_active")
        out["orphans"].append({"local_run_id": rid, "state": r.get("state"),
                               "age_seconds": age, "classification": classification})
    return out


def read_batch_identity(root: str, local_run_id: str) -> str | None:
    """The run's batch_config_hash from its AUTHORITATIVE append-only chain (5.2.B / Codex
    item 4), or None for a legacy run that never bound one. Fail-closed (CorruptRunError)
    if the chain is tampered — the integrity chain protects it."""
    return qpu_runs.batch_identity_of(local_run_id, root=root)


def verify_batch_identity(root: str, local_run_id: str, expected: str) -> str:
    """Verify a run's chain-bound batch identity against ``expected``.

    - ``"verified"`` — the chain binds exactly ``expected``.
    - ``"identity_completed"`` — the chain has NO identity but the single-winner claim
      DESIGNATES this run for ``expected`` (a crash between prepare and the identity commit,
      Codex item 5): the transaction is safely completed by binding ``expected`` now, so the
      run is never left as legacy.
    - ``"legacy_unverified"`` — no chain identity and not claim-designated (a genuine
      pre-5.2.B run): inspectable offline, but refused for hardware submission.
    Raises ``OfflineSafetyError`` on a concrete mismatch."""
    got = read_batch_identity(root, local_run_id)
    if got is not None:
        if got != expected:
            raise OfflineSafetyError(
                f"run {local_run_id} was built for batch config {got} != current {expected}; "
                f"refusing to reuse it under a different protocol identity")
        return "verified"
    designated = qpu_claim.read_designated(expected, _claims_dir(root)) == local_run_id
    if designated:
        qpu_runs.record_batch_identity(local_run_id, expected, root=root)   # complete the txn
        return "identity_completed"
    return "legacy_unverified"


def assert_hardware_submittable(root: str, local_run_id: str, expected: str) -> str:
    """Gate a (future) HARDWARE submission on an authoritative batch identity. A
    ``legacy_unverified`` run is refused — legacy runs stay inspectable offline but are
    never submittable to real hardware (Codex item 4). Returns the (verified/completed)
    status. NOTE: the real submit path lives in the WIP ``ibm_runner`` and must call this
    before ``run_sampler``; see docs/NIGHT_SHIFT_OPEN_ISSUES.md (D1)."""
    status = verify_batch_identity(root, local_run_id, expected)
    if status == "legacy_unverified":
        raise OfflineSafetyError(
            f"run {local_run_id} has no authoritative batch identity in its chain; "
            f"refusing hardware submission (legacy runs are inspectable but not submittable)")
    return status


def _prepare_awaiting_run(prepared: Any, root: str, batch_config_hash: str | None = None) -> str:
    """Prepare a run, bind its full batch identity into the authoritative chain BEFORE
    moving to AWAITING_CONFIRMATION (Codex item 4), and return its local_run_id. This is
    the ``prepare_fn`` the single-winner claim invokes at most once."""
    params = H._expected_phase3_params(prepared)
    local_run_id = qpu_runs.prepare(params, root=root)
    if batch_config_hash is not None:
        qpu_runs.record_batch_identity(local_run_id, batch_config_hash, root=root)  # before AWAITING
    qpu_runs.transition(local_run_id, qpu_runs.QpuState.AWAITING_CONFIRMATION, root=root)
    return local_run_id


def _find_run_by_config_hash(root: str, config_hash: str) -> str | None:
    """Return the local_run_id of an EXISTING run for this exact config_hash in the
    durable run store, or None. The run store — not the batch index — is the
    authority on what has already been prepared/submitted, so this closes the
    lost-index resubmission hole.

    Fails CLOSED (never silently prepares a possible duplicate) when:
    - two intact runs share one config_hash (impossible by construction; refuse to
      guess which to resume), or
    - a CORRUPTED run could be a prior submission for this config — i.e. its manifest
      config_hash matches, OR is unreadable so we cannot rule it out. A corrupted run
      that provably belongs to a DIFFERENT config is ignored. This mirrors the
      fail-closed handling of a corrupt batch index: a damaged authority is
      operator-actionable, not 'never submitted'."""
    runs = qpu_runs.list_runs(root=root)
    matches = [
        r["local_run_id"]
        for r in runs
        if r.get("config_hash") == config_hash and r.get("state") != "corrupted"
    ]
    if len(matches) > 1:
        raise OfflineSafetyError(
            f"multiple runs for config_hash {config_hash}: {sorted(matches)} — refusing to guess"
        )
    for r in runs:
        if r.get("state") != "corrupted":
            continue
        ch = r.get("config_hash")            # manifest hash, surfaced even when corrupt
        if ch is None or ch == config_hash:  # matches, or cannot be ruled out
            raise OfflineSafetyError(
                f"corrupted run {r.get('local_run_id')!r} may be a prior submission for "
                f"config_hash {config_hash} (manifest hash={ch!r}); refusing to prepare a "
                f"possible duplicate — operator must inspect/repair the run store"
            )
    return matches[0] if matches else None


def _run_offline_lifecycle(
    bundle: Any, qubo: Any, gateway: Any, *, root: str, reference_objective: float | None,
    store: BatchJobStore, job_key: str, base_record: dict, config_hash: str,
    crash_hook: Callable[[str], None] | None = None,
) -> dict:
    """Resumable, exact-type-fenced offline lifecycle: prepare -> submit(once) ->
    reconcile -> decode. Reuses the stored local_run_id on resume so a crash at any
    boundary can never cause a second submission (submit_run is only called on an
    awaiting_confirmation run with no job).

    ``config_hash`` is the FULL batch protocol identity (:func:`batch_config_hash`) —
    distinct from ``bundle.prepared.config_hash`` (the run-store hash used for
    durability). It is recorded in every store entry and re-checked on resume, so a
    resume onto a store entry created for a DIFFERENT config is refused fail-closed,
    never silently reusing or overwriting the old result."""
    require_offline_exact_gateway(gateway)              # exact type, not a marker
    p = bundle.prepared
    _assert_matching_config(store, job_key, config_hash)  # refuse a config-drifted reuse

    def _crash(point: str) -> None:
        if crash_hook is not None:
            crash_hook(point)

    prior = store.load(job_key)
    if prior is not None and prior.get("local_run_id"):
        local_run_id = prior["local_run_id"]            # RESUME via the batch index
        resumed = True
    else:
        # DURABILITY: the run store (runs_dir) — not the batch index — is the
        # authority. Even if the batch-store entry was lost AFTER a submission, an
        # existing run for this exact config_hash is REUSED (reconciled), never
        # prepared+submitted a second time, so a lost index can never cause a
        # double submission. This also recovers a run orphaned by a crash before
        # the first index write. (For a FUTURE real provider, losing the runs_dir
        # itself would still require provider-side job-tag dedup — out of scope for
        # this offline phase and flagged in the safety block.)
        existing = _find_run_by_config_hash(root, p.config_hash)
        if existing is not None and _claimed_by_other_config(store, existing, config_hash):
            existing = None                 # that run belongs to a DIFFERENT batch config
        if existing is not None:
            local_run_id = existing
            store.record(job_key, {"local_run_id": local_run_id, "stage": "recovered",
                                   "config_hash": config_hash})
            resumed = True
        else:
            # GENUINELY FRESH: serialise concurrent preparers of the SAME config through
            # the atomic single-winner claim (keyed by the FULL batch config_hash). Two
            # processes that both scanned an empty runs_dir now converge on ONE run — the
            # winner prepares it, losers adopt it — so a real provider can never receive
            # a double submission from the prepare race (5.2.C / see qpu_claim).
            claim = qpu_claim.claim_or_adopt(
                config_hash,
                claims_dir=_claims_dir(root),
                prepare_fn=lambda: _prepare_awaiting_run(p, root, config_hash),
            )
            local_run_id = claim.local_run_id
            resumed = claim.role != "winner"
            store.record(job_key, {
                "local_run_id": local_run_id,
                "stage": "prepared" if claim.role == "winner" else "recovered",
                "config_hash": config_hash})

    # 5.2.B: the run's persisted batch identity must match this protocol config (or be a
    # flagged legacy run) — a concrete mismatch refuses to reuse it under a different config.
    identity_status = verify_batch_identity(root, local_run_id, config_hash)

    _crash("before_submit")
    run = qpu_runs.load(local_run_id, root=root)
    submit_verdict: dict | None = None
    # submit ONLY a run that is awaiting_confirmation with no job yet; a run that
    # already has a job (or advanced) is reconciled, never resubmitted.
    if run["state"] == qpu_runs.QpuState.AWAITING_CONFIRMATION.value and not run.get("ibm_job_ids"):
        submit_verdict = ibm_runner.submit_run(
            local_run_id, gateway, isa_circuit=bundle.isa_circuit,
            expected_config_hash=p.config_hash, root=root,
            sleeper=lambda _s: None, jitter=lambda: 0.0)
        store.record(job_key, {"local_run_id": local_run_id, "stage": "submitted",
                               "config_hash": config_hash})

    _crash("after_job_id")
    state = qpu_runs.load(local_run_id, root=root)["state"]
    reconcile_verdict: dict | None = None
    if state not in qpu_runs._TERMINAL_VALUES:
        reconcile_verdict = ibm_runner.reconcile_run(
            local_run_id, gateway, root=root, sleeper=lambda _s: None, jitter=lambda: 0.0)
        state = qpu_runs.load(local_run_id, root=root)["state"]

    _crash("during_wait")
    run = qpu_runs.load(local_run_id, root=root)
    job_ids = tuple(run.get("ibm_job_ids", []))
    anomalies: list[str] = []
    if identity_status != "verified":
        anomalies.append(f"batch_identity:{identity_status}")   # e.g. legacy_unverified
    decoded_summary: dict = {}
    normalized: dict | None = None
    if state == qpu_runs.QpuState.COMPLETED.value:
        # counts come from gateway.get_result() and flow through the SAME normaliser
        # a real IBM provider result would (hardware_validation._normalise_distribution).
        job = gateway.get_job(job_ids[0]) if job_ids else None
        envelope = gateway.get_result(job)
        normalized = normalize_result(envelope, expected_n_bits=qubo.n_qubits)
        decoded = H.decode_samples(qubo, envelope["counts"], bit_order=envelope["bit_order"],
                                   kind=envelope["kind"], reference_objective=reference_objective)
        anomalies.extend(f"decode:{w}" for w in decoded.warnings)
        decoded_summary = _decoded_summary(decoded)
    else:
        anomalies.append(f"non_completed:{state}")

    _crash("after_result_before_export")
    obj = base_record["objective_id"]
    record = {
        **base_record,
        "final_state": state, "job_ids": list(job_ids), "resumed": resumed,
        "gateway_run_calls": getattr(gateway, "run_calls", None),
        "submit_verdict": submit_verdict, "reconcile_verdict": reconcile_verdict,
        "normalized_n_shots": (normalized or {}).get("n_shots"),
        "anomalies": anomalies,
        "decoded_best_bitstring": decoded_summary.get("best_bitstring"),
        "decoded_best_energy": decoded_summary.get("best_energy"),
        "decoded_best_feasible": decoded_summary.get("best_feasible"),
        "decoded_matches_reference": _matches(decoded_summary.get("best_energy"),
                                              base_record["reference_optimum"], obj),
        "local_run_id": local_run_id,   # volatile (excluded from the hash)
        "batch_identity_status": identity_status,
        "no_ibm": True,
    }
    store.record(job_key, {"local_run_id": local_run_id, "stage": "exported",
                           "final_state": state, "record": record, "config_hash": config_hash})
    return record


def run_offline_qpu_job(
    loader: InstanceLoader, instance: str, *, objective_id: str, n_theta: int, n_q: int = 1,
    gammas: tuple[float, ...] = (0.4,), betas: tuple[float, ...] = (0.3,),
    shots: int = 512, transpiler_seed: int = 1, optimization_level: int = 0,
    max_quantum_seconds: float = 20.0, backend_name: str = DEFAULT_FAKE_BACKEND,
    runs_dir: str, store: BatchJobStore | None = None, job_key: str | None = None,
    crash_hook: Callable[[str], None] | None = None,
) -> dict:
    """Run ONE instance through the full offline QPU pipeline and return a JSON-native
    record. Fake gateway + local fake backend only; no IBM, no token.

    The FULL protocol identity (:func:`batch_config_identity` — protocol version,
    instance, objective, grid, angles, shots, seed, transpile level, backend, QUBO
    hash) is computed BEFORE any store lookup and recorded in the store entry. Resume
    via ``store``/``job_key`` returns an already-exported job only when its recorded
    ``config_hash`` matches the current one; a store entry created for a DIFFERENT
    config is refused fail-closed (:class:`OfflineSafetyError`) — never silently reused
    or overwritten. The default ``job_key`` is derived from the config hash, so distinct
    configs never collide. An interrupted job reuses its run (never resubmits)."""
    from qiskit_ibm_runtime import fake_provider as _fp

    # Allow-list FIRST: only a LOCAL ``Fake*`` provider backend may be named (defense-in-depth
    # so ``backend_name`` can't reach a real service/network class or a dunder via getattr).
    if not (isinstance(backend_name, str) and backend_name.startswith("Fake")
            and backend_name in {n for n in dir(_fp) if n.startswith("Fake")}):
        raise ValueError(
            f"backend must be a local qiskit fake_provider 'Fake*' backend; got {backend_name!r}")

    inst = loader.load(instance)
    qubo = build_qubo(inst, n_theta=n_theta, n_q=n_q, objective_id=objective_id)
    n = qubo.n_qubits
    qubo_sha = H.canonical_qubo_hash(qubo)

    # FULL protocol identity + hash computed BEFORE any store return, so a config change
    # (shots/angles/seed/transpile-level/backend/objective/grid/protocol-version) can never
    # ride a stale job_key into a reused/overwritten result.
    cfg_hash = batch_config_hash(batch_config_identity(
        instance=instance, objective_id=objective_id, n_theta=n_theta, n_q=n_q,
        gammas=gammas, betas=betas, shots=shots, transpiler_seed=transpiler_seed,
        optimization_level=optimization_level, backend_name=backend_name, qubo_sha256=qubo_sha,
        max_quantum_seconds=max_quantum_seconds))
    job_key = job_key or f"{instance}__{objective_id}__{cfg_hash.split(':')[-1][:16]}"
    store = store or BatchJobStore(tempfile.mkdtemp(prefix="acrpq-batchstore-"))
    _assert_matching_config(store, job_key, cfg_hash)   # fail-closed on config drift
    prior = store.load(job_key)
    if prior is not None and prior.get("stage") == "exported":
        return prior["record"]                          # SAME config, fully done -> no re-run

    opt = optimum_onehot_bits(inst, qubo, objective_id, n_theta, n_q)

    backend = getattr(_fp, backend_name)()   # LOCAL fake backend — no service/token
    if int(backend.num_qubits) < n:
        raise ValueError(f"{backend_name} has {backend.num_qubits} qubits < {n} needed")
    backend_id = getattr(backend, "name", backend_name)   # e.g. "fake_guadalupe"

    req = H.HardwareValidationRequest(
        instance_id=instance, qubo_sha256=qubo_sha, n_binary_vars=n,
        variable_meaning=tuple(f"x{b}" for b in range(n)), qaoa_reps=len(gammas),
        gammas=tuple(gammas), betas=tuple(betas), shots=shots, transpiler_seed=transpiler_seed,
        backend_requested=backend_id, max_jobs=1, max_quantum_seconds=max_quantum_seconds,
        n_theta=n_theta, optimization_level=optimization_level, decision_ts=0.0)
    bundle = H.prepare_hardware_bundle(req, qubo, backend, allow_synthetic=True)
    plan = qpu_orchestrator.build_execution_plan(bundle)
    dry = qpu_orchestrator.dry_run(plan, bundle)

    # synthetic counts: a delta on the certified optimum (labelled synthetic).
    bitstr = opt["bitstring"]
    synthetic: dict[str, int] = ({str(bitstr): shots} if bitstr is not None else {"0" * n: shots})
    gateway = FakeBatchGateway(synthetic, bit_order="qubo", n_bits=n)

    base_record = {
        "instance": instance, "family": inst.family.value, "n_aircraft": inst.n,
        "objective_id": objective_id, "n_theta": n_theta, "n_q": n_q, "n_qubits": n,
        "backend": backend_id, "backend_class": backend_name,
        "shots": shots, "gammas": list(gammas), "betas": list(betas),
        "optimization_level": optimization_level,
        "reference_optimum": opt["optimum"], "reference_feasible": opt["feasible"],
        "reference_status": opt["status"],
        "synthetic_counts_source": "classical_optimum_delta",
        "dry_run_notes": list(dry.notes),
        "plan_max_jobs": plan.max_jobs, "plan_jobs": len(plan.jobs),
        "plan_sha256": plan.plan_sha256, "artefact_sha256": bundle.prepared.artefact_sha256,
        "decision_sha256": bundle.prepared.decision_sha256,
    }
    return _run_offline_lifecycle(
        bundle, qubo, gateway, root=runs_dir, reference_objective=opt["optimum"],
        store=store, job_key=job_key, base_record=base_record, config_hash=cfg_hash,
        crash_hook=crash_hook)


def _decoded_summary(decoded: Any) -> dict:
    best = getattr(decoded, "best_raw", None)
    if best is None:
        return {}
    return {"best_bitstring": getattr(best, "bitstring", None),
            "best_energy": getattr(best, "energy", None),
            "best_feasible": getattr(best, "feasible", None)}


def _matches(value: float | None, ref: float | None, objective_id: str) -> bool | None:
    if value is None or ref is None:
        return None
    tol = 0.0 if objective_id == "maneuver_count_v1" else 1e-9
    return abs(value - ref) <= tol


# ---------------------------------------------------------------------------
# Aggregation helpers (JSON + CSV parity, content hash) — mirrors Phase 3/4
# ---------------------------------------------------------------------------
CSV_COLS = [
    "instance", "family", "n_aircraft", "objective_id", "n_theta", "n_qubits", "backend",
    "shots", "reference_optimum", "reference_status", "final_state", "job_ids",
    "gateway_run_calls", "decoded_best_bitstring", "decoded_best_energy",
    "decoded_best_feasible", "decoded_matches_reference", "no_ibm",
]


def _csv_repr(v) -> str:
    return "" if v is None else (",".join(map(str, v)) if isinstance(v, list) else str(v))


def rows_to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLS, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({c: _csv_repr(r.get(c)) for c in CSV_COLS})
    return buf.getvalue()


def assert_json_csv_parity(rows: list[dict], csv_text: str) -> None:
    csv_rows = list(csv.DictReader(io.StringIO(csv_text)))
    if len(csv_rows) != len(rows):
        raise ValueError(f"JSON/CSV row-count mismatch: {len(rows)} vs {len(csv_rows)}")
    for jr, cr in zip(rows, csv_rows):
        for col in CSV_COLS:
            if _csv_repr(jr.get(col)) != cr[col]:
                raise ValueError(f"JSON/CSV parity mismatch at {jr.get('instance')} col={col}: "
                                 f"{jr.get(col)!r} != {cr[col]!r}")


def _canonical(obj, excluded: set[str]):
    if isinstance(obj, dict):
        return {k: _canonical(v, excluded) for k, v in obj.items() if k not in excluded}
    if isinstance(obj, list):
        return [_canonical(v, excluded) for v in obj]
    return obj


_HASH_EXCLUDED = set(_VOLATILE) | {"batch_sha256", "batch_sha256_note", "repository_dirty"}


def canonical_batch_sha256(batch: dict) -> str:
    """Content hash over the batch (volatile run ids/timings + transient env state out)."""
    blob = json.dumps(_canonical(batch, _HASH_EXCLUDED), sort_keys=True,
                      separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
