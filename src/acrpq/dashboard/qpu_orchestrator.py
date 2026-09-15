"""Crash-safe local QPU orchestration (Phase 6) — never submits a real job.

Given a Phase-5 :class:`PreparedCircuitBundle` (itself bound to a verified Phase-4
decision), this builds an immutable, hashed :class:`QpuExecutionPlan`, enforces
budgets and artefact coherence, and offers two safe execution paths:

* :func:`dry_run` — validates the plan and reports exactly *what would be
  submitted*, calling **no** gateway and never transitioning a run to
  ``submitting``.
* :func:`simulate_lifecycle` — drives the real Phase-2 run store
  (:mod:`acrpq.dashboard.qpu_runs`) and the real Phase-3 submission/reconciliation
  logic (:mod:`acrpq.dashboard.ibm_runner`) through a **fake gateway**, exercising
  prepare → confirm → submit → reconcile → completed → decode. A real
  ``RuntimeGateway`` is refused, so this can never contact IBM.

Design invariants (enforced + tested):
  * the plan is immutable and hashed; any field change moves the hash;
  * budgets (``max_jobs`` / total shots / quantum-seconds) and no-duplicate-job
    tags are enforced BEFORE any execution;
  * a fake/synthetic artefact stays NOT submittable — the plan records it and the
    real-submission path is only ever reachable through the Phase-7 confirmation
    + server feature flag (never here);
  * submissions are never retried; only reads are (handled by ``ibm_runner``);
  * every step goes through the append-only, crash-safe event store, so a crash at
    any boundary leaves a recoverable run and never a silent double submission.

Qiskit is never imported at module load.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

from . import ibm_runner, qpu_runs
from .backend_scoring import _canonical, _json_native
from .hardware_validation import (
    HW_VALIDATION_SCHEMA_VERSION,
    MAX_HW_MAX_JOBS,
    MAX_HW_MAX_QUANTUM_SECONDS,
    PreparedCircuitBundle,
    _expected_phase3_params,
    decode_samples,
    verify_bundle_integrity,
)

QPU_PLAN_SCHEMA_VERSION = "acrpq-qpu-plan/1"
QPU_FULL_QAOA_SCHEMA_VERSION = "acrpq-qpu-full-qaoa/1"

# A single hardware-validation job runs exactly one angle point.
_SINGLE_ANGLE_PROTOCOL = "hardware_validation"
_MULTI_ANGLE_PROTOCOL = "full_qaoa_hardware"

# Bounded full-QAOA caps (a real full study is a later, separately-confirmed phase).
MAX_ANGLE_POINTS = MAX_HW_MAX_JOBS          # never plan more angle points than the job cap


class OrchestrationError(ValueError):
    """Base error for plan construction / dry-run / simulation."""


class BudgetError(OrchestrationError):
    """A job/shot/quantum-second budget would be exceeded."""


class RealGatewayRefused(OrchestrationError):
    """A real RuntimeGateway was passed where only a fake/dry gateway is allowed."""


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Immutable, hashed execution plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QpuExecutionPlan:
    schema: str
    protocol: str
    backend: str
    artefact_sha256: str
    config_hash: str
    qubo_sha256: str
    decision_sha256: str | None
    submittable: bool
    jobs: tuple[dict[str, Any], ...]     # planned jobs (index, tag, shots, gammas, betas)
    max_jobs: int
    total_shots: int
    max_quantum_seconds: float
    strategy: str
    transpiler_seed: int
    recovery_policy: str
    provenance: dict[str, Any] | None
    warnings: tuple[str, ...]
    plan_sha256: str = ""

    def to_manifest(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("plan_sha256")
        return _json_native(d)


def _hash_plan(manifest: dict[str, Any]) -> str:
    return _sha(_canonical(manifest))


def verify_plan_hash(plan: QpuExecutionPlan) -> bool:
    """Recompute the plan hash over everything except ``plan_sha256``."""
    return plan.plan_sha256 == _hash_plan(plan.to_manifest())


def build_execution_plan(
    bundle: PreparedCircuitBundle,
    *,
    strategy: str = "single_angle_point",
    n_angle_evals: int = 1,
    recovery_policy: str = "reconcile_never_resubmit",
    provenance: dict[str, Any] | None = None,
) -> QpuExecutionPlan:
    """Build an immutable, budget-checked execution plan bound to ``bundle``.

    Only ``hardware_validation`` (a single angle point → one job) is planned here.
    ``full_qaoa_hardware`` (multiple angle evaluations) is explicitly refused: it
    belongs to a later, separately-confirmed phase and must never be planned as if
    it were a validated single-point run.
    """
    if not isinstance(bundle, PreparedCircuitBundle):
        raise OrchestrationError("build_execution_plan requires a PreparedCircuitBundle")
    problems = verify_bundle_integrity(bundle)
    if problems:
        raise OrchestrationError(f"bundle integrity check failed: {problems}")

    p = bundle.prepared
    req = p.request
    mode = req["mode"]
    if mode == _MULTI_ANGLE_PROTOCOL:
        raise OrchestrationError(
            "full_qaoa_hardware planning is disabled here; it needs a separate confirmed phase")
    if mode != _SINGLE_ANGLE_PROTOCOL:
        raise OrchestrationError(f"unsupported mode for planning: {mode!r}")
    if p.schema != HW_VALIDATION_SCHEMA_VERSION:
        raise OrchestrationError("artefact schema mismatch")
    if not isinstance(n_angle_evals, int) or isinstance(n_angle_evals, bool) or n_angle_evals != 1:
        raise OrchestrationError("hardware_validation must plan exactly one angle evaluation")

    max_jobs = int(req["max_jobs"])
    shots = int(req["shots"])
    jobs: tuple[dict[str, Any], ...] = (
        {"job_index": 0, "local_tag": "job-0", "shots": shots,
         "gammas": list(req["gammas"]), "betas": list(req["betas"])},)
    # --- budgets (fail closed before any execution) ---
    if len(jobs) > max_jobs:
        raise BudgetError(f"{len(jobs)} jobs > max_jobs {max_jobs}")
    tags = [j["local_tag"] for j in jobs]
    if len(set(tags)) != len(tags):
        raise BudgetError("duplicate job tags in plan")
    total_shots = sum(int(j["shots"]) for j in jobs)
    if any(int(j["shots"]) < 1 for j in jobs):
        raise BudgetError("every job needs shots >= 1")
    max_qs = float(req["max_quantum_seconds"])

    warnings: list[str] = []
    if not p.submittable:
        warnings.append("artefact_not_submittable_dry_run_only")

    manifest: dict[str, Any] = {
        "schema": QPU_PLAN_SCHEMA_VERSION,
        "protocol": mode,
        "backend": p.backend,
        "artefact_sha256": p.artefact_sha256,
        "config_hash": p.config_hash,
        "qubo_sha256": p.qubo_sha256,
        "decision_sha256": p.decision_sha256,
        "submittable": p.submittable,
        "jobs": [dict(j) for j in jobs],
        "max_jobs": max_jobs,
        "total_shots": total_shots,
        "max_quantum_seconds": max_qs,
        "strategy": strategy,
        "transpiler_seed": int(req["transpiler_seed"]),
        "recovery_policy": recovery_policy,
        "provenance": _json_native(provenance) if provenance is not None else None,
        "warnings": list(warnings),
    }
    plan_sha = _hash_plan(manifest)
    return QpuExecutionPlan(
        schema=manifest["schema"], protocol=manifest["protocol"], backend=manifest["backend"],
        artefact_sha256=manifest["artefact_sha256"], config_hash=manifest["config_hash"],
        qubo_sha256=manifest["qubo_sha256"], decision_sha256=manifest["decision_sha256"],
        submittable=manifest["submittable"], jobs=tuple(manifest["jobs"]),
        max_jobs=manifest["max_jobs"], total_shots=manifest["total_shots"],
        max_quantum_seconds=manifest["max_quantum_seconds"], strategy=manifest["strategy"],
        transpiler_seed=manifest["transpiler_seed"], recovery_policy=manifest["recovery_policy"],
        provenance=manifest["provenance"], warnings=tuple(warnings), plan_sha256=plan_sha)


def _assert_plan_matches_bundle(plan: QpuExecutionPlan, bundle: PreparedCircuitBundle) -> None:
    p = bundle.prepared
    if not verify_plan_hash(plan):
        raise OrchestrationError("plan hash does not verify")
    problems = verify_bundle_integrity(bundle)
    if problems:
        raise OrchestrationError(f"bundle integrity check failed: {problems}")
    for field_name, a, b in (
        ("artefact_sha256", plan.artefact_sha256, p.artefact_sha256),
        ("config_hash", plan.config_hash, p.config_hash),
        ("backend", plan.backend, p.backend),
        ("qubo_sha256", plan.qubo_sha256, p.qubo_sha256),
    ):
        if a != b:
            raise OrchestrationError(f"plan/artefact {field_name} mismatch")


# --------------------------------------------------------------------------- #
# Dry-run (no gateway, no submission)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DryRunReport:
    plan_sha256: str
    submittable: bool
    would_submit: tuple[dict[str, Any], ...]
    budget: dict[str, Any]
    notes: tuple[str, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)


def dry_run(plan: QpuExecutionPlan, bundle: PreparedCircuitBundle) -> DryRunReport:
    """Validate the plan and report what WOULD be submitted — no gateway, no submit.

    The per-job summary describes the intended Phase-3 call arguments (backend,
    shots, expected config hash, ISA fingerprint) without building a payload and
    without any side effect. A not-submittable (fake/synthetic) artefact yields a
    clearly-marked dry-run-only report.
    """
    _assert_plan_matches_bundle(plan, bundle)
    p = bundle.prepared
    would: list[dict[str, Any]] = []
    for j in plan.jobs:
        would.append({
            "job_index": j["job_index"],
            "local_tag": j["local_tag"],
            "backend": plan.backend,
            "shots": j["shots"],
            "expected_config_hash": plan.config_hash,
            "qubo_sha256": plan.qubo_sha256,
            "isa_fingerprint_sha256": p.isa_circuit["fingerprint_sha256"],
            "max_quantum_seconds": plan.max_quantum_seconds,
            "submittable": p.submittable,
        })
    notes = ["no gateway called", "no state transitioned to submitting", "no token read"]
    warnings = list(plan.warnings)
    if not p.submittable:
        warnings.append("dry_run_only_not_submittable")
    return DryRunReport(
        plan_sha256=plan.plan_sha256, submittable=p.submittable, would_submit=tuple(would),
        budget={"max_jobs": plan.max_jobs, "planned_jobs": len(plan.jobs),
                "total_shots": plan.total_shots, "max_quantum_seconds": plan.max_quantum_seconds},
        notes=tuple(notes), warnings=tuple(warnings))


# --------------------------------------------------------------------------- #
# Fake-gateway lifecycle simulation (never contacts IBM)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LifecycleReport:
    local_run_id: str
    final_state: str
    submit_verdict: dict[str, Any]
    reconcile_verdict: dict[str, Any] | None
    job_ids: tuple[str, ...]
    decoded: Any
    plan_sha256: str
    artefact_sha256: str
    decision_sha256: str | None
    anomalies: tuple[str, ...]


def _require_safe_gateway(gateway: Any) -> None:
    """Allow-list: only a gateway that POSITIVELY asserts it is dry-run-safe passes.

    A blocklist ("not literally RuntimeGateway") is trivially defeated by a
    pass-through wrapper, so instead the gateway must expose
    ``dry_run_safe is True`` — a deliberate marker a genuine fake/test double sets
    and that a real gateway (or a naive wrapper around one) does not. As a second
    guard, a real ``RuntimeGateway`` (or subclass) is refused outright even if it
    somehow carried the marker.
    """
    if isinstance(gateway, ibm_runner.RuntimeGateway):
        raise RealGatewayRefused("simulate_lifecycle refuses a real RuntimeGateway")
    if getattr(gateway, "dry_run_safe", False) is not True:
        raise RealGatewayRefused(
            "simulate_lifecycle requires a gateway that declares `dry_run_safe = True` "
            "(a fake/test double); refusing an unmarked gateway")


def simulate_lifecycle(
    plan: QpuExecutionPlan,
    bundle: PreparedCircuitBundle,
    qubo: Any,
    *,
    gateway: Any,
    synthetic_counts: dict[str, float],
    root: str,
    decode_bit_order: str = "qiskit",
    reference_objective: float | None = None,
) -> LifecycleReport:
    """Drive prepare → confirm → submit → reconcile → completed → decode against a
    FAKE gateway, through the real run store + submission logic. Never real IBM.

    ``synthetic_counts`` stand in for what a Sampler *would* return; they are
    decoded by the Phase-5 decoder. The run store records every event, so a crash
    at any boundary leaves a recoverable run (see :func:`qpu_runs.reconciliation_candidates`).
    """
    _require_safe_gateway(gateway)
    _assert_plan_matches_bundle(plan, bundle)
    p = bundle.prepared

    # prepare (commit point) + explicit confirm (prepared -> awaiting_confirmation)
    params = _expected_phase3_params(p)
    local_run_id = qpu_runs.prepare(params, root=root)
    qpu_runs.transition(local_run_id, qpu_runs.QpuState.AWAITING_CONFIRMATION, root=root)

    anomalies: list[str] = []
    # submit through the REAL idempotent path with a FAKE gateway (deterministic,
    # no sleeping, no jitter). Submissions are never retried by ibm_runner.
    verdict = ibm_runner.submit_run(
        local_run_id, gateway, isa_circuit=bundle.isa_circuit,
        expected_config_hash=p.config_hash, root=root,
        sleeper=lambda _s: None, jitter=lambda: 0.0)

    reconcile_verdict: dict[str, Any] | None = None
    state = qpu_runs.load(local_run_id, root=root)["state"]
    if state not in qpu_runs._TERMINAL_VALUES:
        # reconcile ambiguous/non-terminal submissions to a true state
        reconcile_verdict = ibm_runner.reconcile_run(
            local_run_id, gateway, root=root, sleeper=lambda _s: None, jitter=lambda: 0.0)
        state = qpu_runs.load(local_run_id, root=root)["state"]

    run = qpu_runs.load(local_run_id, root=root)
    job_ids = tuple(run.get("ibm_job_ids", []))

    decoded = None
    if state == qpu_runs.QpuState.COMPLETED.value:
        decoded = decode_samples(
            qubo, synthetic_counts, bit_order=decode_bit_order, kind="counts",
            reference_objective=reference_objective)
        # surface decode-level warnings (e.g. empty_result) so a "completed" run
        # decoded from nothing does not look clean
        anomalies.extend(f"decode:{w}" for w in decoded.warnings)
    else:
        anomalies.append(f"non_completed_terminal_or_pending_state:{state}")

    return LifecycleReport(
        local_run_id=local_run_id, final_state=state, submit_verdict=verdict,
        reconcile_verdict=reconcile_verdict, job_ids=job_ids, decoded=decoded,
        plan_sha256=plan.plan_sha256, artefact_sha256=p.artefact_sha256,
        decision_sha256=p.decision_sha256, anomalies=tuple(anomalies))


# --------------------------------------------------------------------------- #
# Phase 6.3 — bounded full-QAOA (multiple angle points, never submits)
# --------------------------------------------------------------------------- #
def generate_angle_points(
    strategy: str, *, reps: int, provided: list[tuple[tuple[float, ...], tuple[float, ...]]] | None = None,
    max_points: int = MAX_ANGLE_POINTS,
) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    """Produce a bounded, DETERMINISTIC list of (gammas, betas) angle points.

    ``provided``: use the caller's explicit list. ``deterministic_sweep``: a fixed,
    RNG-free grid derived only from the point index (reproducible). Both are capped
    at ``max_points`` (fails closed above ``MAX_ANGLE_POINTS``).
    """
    if not isinstance(max_points, int) or isinstance(max_points, bool) or not 1 <= max_points <= MAX_ANGLE_POINTS:
        raise BudgetError(f"max_points must be an int in 1..{MAX_ANGLE_POINTS}")
    if reps < 1:
        raise OrchestrationError("reps must be >= 1")
    if strategy == "provided_points":
        if not provided:
            raise OrchestrationError("provided_points strategy needs a non-empty 'provided' list")
        if len(provided) > max_points:
            raise BudgetError(f"{len(provided)} angle points > cap {max_points}")
        pts = [(tuple(float(g) for g in gs), tuple(float(b) for b in bs)) for gs, bs in provided]
        for gs, bs in pts:
            if len(gs) != reps or len(bs) != reps:
                raise OrchestrationError("each angle point needs exactly `reps` gammas and betas")
        return pts
    if strategy == "deterministic_sweep":
        pts = []
        for i in range(max_points):
            g = round(0.2 * (i + 1), 6)
            b = round(0.15 * (i + 1), 6)
            pts.append((tuple(g for _ in range(reps)), tuple(b for _ in range(reps))))
        return pts
    raise OrchestrationError(f"unknown angle strategy {strategy!r}")


@dataclass(frozen=True)
class FullQaoaPlan:
    schema: str
    protocol: str
    backend: str
    qubo_sha256: str
    submittable: bool
    jobs: tuple[dict[str, Any], ...]     # one per angle point (tag, gammas, betas, shots, hashes)
    strategy: str
    max_jobs: int
    total_shots: int
    total_shots_cap: int
    max_quantum_seconds: float
    max_angle_points: int
    recovery_policy: str
    provenance: dict[str, Any] | None
    warnings: tuple[str, ...]
    plan_sha256: str = ""

    def to_manifest(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("plan_sha256")
        return _json_native(d)


def verify_full_qaoa_plan_hash(plan: FullQaoaPlan) -> bool:
    return plan.plan_sha256 == _hash_plan(plan.to_manifest())


def build_full_qaoa_plan(
    bundles: list[PreparedCircuitBundle],
    *,
    strategy: str = "provided_points",
    total_shots_cap: int,
    max_quantum_seconds: float,
    max_jobs: int = MAX_HW_MAX_JOBS,
    max_angle_points: int = MAX_ANGLE_POINTS,
    recovery_policy: str = "reconcile_never_resubmit",
    provenance: dict[str, Any] | None = None,
) -> FullQaoaPlan:
    """Build an immutable, hard-capped multi-angle-point plan from N bundles.

    Each bundle is one angle point (its own artefact/config/ISA); all must share the
    same backend and QUBO. Fails closed above every cap (jobs, angle points, total
    shots, quantum-seconds), on duplicate tags, or on any bundle-integrity problem.
    A fake/synthetic bundle keeps ``submittable = False``.
    """
    if not bundles:
        raise OrchestrationError("build_full_qaoa_plan needs at least one bundle")
    if not 1 <= max_jobs <= MAX_HW_MAX_JOBS:
        raise BudgetError(f"max_jobs must be 1..{MAX_HW_MAX_JOBS}")
    if not 1 <= max_angle_points <= MAX_ANGLE_POINTS:
        raise BudgetError(f"max_angle_points must be 1..{MAX_ANGLE_POINTS}")
    if len(bundles) > min(max_jobs, max_angle_points):
        raise BudgetError(f"{len(bundles)} angle points exceed the cap "
                          f"{min(max_jobs, max_angle_points)}")
    mqs = float(max_quantum_seconds)
    if not 0 < mqs <= MAX_HW_MAX_QUANTUM_SECONDS:
        raise BudgetError(f"max_quantum_seconds must be in (0, {MAX_HW_MAX_QUANTUM_SECONDS}]")

    backend = bundles[0].prepared.backend
    qubo_sha = bundles[0].prepared.qubo_sha256
    jobs: list[dict[str, Any]] = []
    submittable = True
    for i, b in enumerate(bundles):
        problems = verify_bundle_integrity(b)
        if problems:
            raise OrchestrationError(f"bundle {i} integrity check failed: {problems}")
        p = b.prepared
        if p.backend != backend:
            raise OrchestrationError("all full-QAOA bundles must share one backend")
        if p.qubo_sha256 != qubo_sha:
            raise OrchestrationError("all full-QAOA bundles must share one QUBO")
        submittable = submittable and p.submittable
        jobs.append({
            "job_index": i, "local_tag": f"aqp-{i}", "gammas": list(p.request["gammas"]),
            "betas": list(p.request["betas"]), "shots": int(p.request["shots"]),
            "config_hash": p.config_hash, "artefact_sha256": p.artefact_sha256,
            "isa_fingerprint_sha256": p.isa_circuit["fingerprint_sha256"],
            "status": "planned", "deps": [],
        })
    tags = [j["local_tag"] for j in jobs]
    if len(set(tags)) != len(tags):
        raise BudgetError("duplicate job tags in full-QAOA plan")
    total_shots = sum(int(j["shots"]) for j in jobs)
    if total_shots > int(total_shots_cap):
        raise BudgetError(f"total shots {total_shots} > cap {total_shots_cap}")

    warnings: list[str] = []
    if not submittable:
        warnings.append("artefact_not_submittable_dry_run_only")

    manifest: dict[str, Any] = {
        "schema": QPU_FULL_QAOA_SCHEMA_VERSION, "protocol": _MULTI_ANGLE_PROTOCOL,
        "backend": backend, "qubo_sha256": qubo_sha, "submittable": submittable,
        "jobs": [dict(j) for j in jobs], "strategy": strategy, "max_jobs": max_jobs,
        "total_shots": total_shots, "total_shots_cap": int(total_shots_cap),
        "max_quantum_seconds": mqs, "max_angle_points": max_angle_points,
        "recovery_policy": recovery_policy,
        "provenance": _json_native(provenance) if provenance is not None else None,
        "warnings": list(warnings),
    }
    plan_sha = _hash_plan(manifest)
    return FullQaoaPlan(
        schema=manifest["schema"], protocol=manifest["protocol"], backend=backend,
        qubo_sha256=qubo_sha, submittable=submittable, jobs=tuple(manifest["jobs"]),
        strategy=strategy, max_jobs=max_jobs, total_shots=total_shots,
        total_shots_cap=int(total_shots_cap), max_quantum_seconds=mqs,
        max_angle_points=max_angle_points, recovery_policy=recovery_policy,
        provenance=manifest["provenance"], warnings=tuple(warnings), plan_sha256=plan_sha)


@dataclass(frozen=True)
class FullQaoaReport:
    plan_sha256: str
    jobs: tuple[dict[str, Any], ...]      # per-job: tag, run_id, state, resumed, best_feasible_energy
    best_point: dict[str, Any] | None
    budget_used: dict[str, Any]
    complete: bool
    anomalies: tuple[str, ...]


def simulate_full_qaoa(
    plan: FullQaoaPlan,
    bundles: list[PreparedCircuitBundle],
    qubo: Any,
    *,
    gateway: Any,
    counts_by_tag: dict[str, dict[str, float]],
    root: str,
    existing_run_ids: dict[str, str] | None = None,
    stop_after_jobs: int | None = None,
    decode_bit_order: str = "qiskit",
) -> FullQaoaReport:
    """Drive N angle-point jobs through the real run store + fake gateway.

    Honest partial results, hard budget stop, crash-resume WITHOUT re-submitting an
    already-known job (no double submission), cancellation via ``stop_after_jobs``,
    aggregation and best-angle-point selection. Never contacts IBM.
    """
    _require_safe_gateway(gateway)
    if not verify_full_qaoa_plan_hash(plan):
        raise OrchestrationError("full-QAOA plan hash does not verify")
    if len(bundles) != len(plan.jobs):
        raise OrchestrationError("bundles do not match the plan jobs")
    existing = dict(existing_run_ids or {})
    anomalies: list[str] = []
    job_reports: list[dict[str, Any]] = []
    shots_used = 0
    best: dict[str, Any] | None = None
    completed_jobs = 0

    for i, jobspec in enumerate(plan.jobs):
        tag = jobspec["local_tag"]
        if stop_after_jobs is not None and completed_jobs >= stop_after_jobs:
            anomalies.append(f"stopped_after_{stop_after_jobs}_jobs")
            break
        bundle = bundles[i]
        p = bundle.prepared
        # budget: never exceed the total-shots cap
        if shots_used + int(jobspec["shots"]) > plan.total_shots_cap:
            anomalies.append(f"budget_reached_before_{tag}")
            break

        run_id = existing.get(tag)
        resumed = False
        if run_id is not None:
            # crash-resume: adopt an already-known run; NEVER re-prepare/re-submit
            state = qpu_runs.load(run_id, root=root)["state"]
            resumed = True
        else:
            params = _expected_phase3_params(p)
            run_id = qpu_runs.prepare(params, root=root)
            qpu_runs.transition(run_id, qpu_runs.QpuState.AWAITING_CONFIRMATION, root=root)
            ibm_runner.submit_run(run_id, gateway, isa_circuit=bundle.isa_circuit,
                                  expected_config_hash=p.config_hash, root=root,
                                  sleeper=lambda _s: None, jitter=lambda: 0.0)
            state = qpu_runs.load(run_id, root=root)["state"]
            if state not in qpu_runs._TERMINAL_VALUES:
                ibm_runner.reconcile_run(run_id, gateway, root=root,
                                         sleeper=lambda _s: None, jitter=lambda: 0.0)
                state = qpu_runs.load(run_id, root=root)["state"]
            shots_used += int(jobspec["shots"])

        decoded_energy = None
        if state == qpu_runs.QpuState.COMPLETED.value and tag in counts_by_tag:
            summary = decode_samples(qubo, counts_by_tag[tag], bit_order=decode_bit_order,
                                     kind="counts")
            if summary.best_feasible is not None:
                decoded_energy = summary.best_feasible.energy
                cand = {"local_tag": tag, "gammas": jobspec["gammas"], "betas": jobspec["betas"],
                        "energy": decoded_energy, "run_id": run_id}
                if best is None or decoded_energy < best["energy"]:
                    best = cand
            completed_jobs += 1
        elif state != qpu_runs.QpuState.COMPLETED.value:
            anomalies.append(f"{tag}_not_completed:{state}")

        job_reports.append({"local_tag": tag, "run_id": run_id, "state": state,
                            "resumed": resumed, "best_feasible_energy": decoded_energy})
        existing[tag] = run_id

    complete = (len(job_reports) == len(plan.jobs)
                and all(j["state"] == "completed" for j in job_reports))
    return FullQaoaReport(
        plan_sha256=plan.plan_sha256, jobs=tuple(job_reports), best_point=best,
        budget_used={"jobs": len(job_reports), "shots": shots_used}, complete=complete,
        anomalies=tuple(anomalies))


def run_ask_tell_full_qaoa(
    ask_tell: Any,
    bundle_for_point: Any,
    counts_for_point: Any,
    qubo: Any,
    *,
    gateway: Any,
    root: str,
    max_points: int = MAX_ANGLE_POINTS,
    total_shots_cap: int,
    max_quantum_seconds: float,
    decode_bit_order: str = "qiskit",
) -> FullQaoaReport:
    """Bounded, LOCAL ask/tell optimiser loop over angle points (no IBM).

    ``ask_tell(history) -> (gammas, betas) | None`` proposes the next point from the
    history of ``{gammas, betas, energy}`` (None ends the loop). Each point is built
    into a fake bundle by ``bundle_for_point(gammas, betas)`` and run as a 1-job
    full-QAOA; ``counts_for_point(gammas, betas)`` supplies its synthetic result.
    Hard-capped by ``max_points`` and ``total_shots_cap``; never contacts IBM.
    """
    _require_safe_gateway(gateway)
    if not 1 <= max_points <= MAX_ANGLE_POINTS:
        raise BudgetError(f"max_points must be 1..{MAX_ANGLE_POINTS}")
    history: list[dict[str, Any]] = []
    job_reports: list[dict[str, Any]] = []
    anomalies: list[str] = []
    shots_used = 0
    best: dict[str, Any] | None = None

    for step in range(max_points):
        point = ask_tell(list(history))
        if point is None:
            break
        gammas, betas = point
        bundle = bundle_for_point(gammas, betas)
        sub_plan = build_full_qaoa_plan(
            [bundle], strategy="ask_tell", total_shots_cap=total_shots_cap,
            max_quantum_seconds=max_quantum_seconds, max_jobs=1, max_angle_points=1)
        shots = int(bundle.prepared.request["shots"])
        if shots_used + shots > total_shots_cap:
            anomalies.append(f"budget_reached_at_step_{step}")
            break
        tag = "aqp-0"
        rep = simulate_full_qaoa(
            sub_plan, [bundle], qubo, gateway=gateway,
            counts_by_tag={tag: counts_for_point(gammas, betas)}, root=root,
            decode_bit_order=decode_bit_order)
        shots_used += shots
        energy = rep.best_point["energy"] if rep.best_point else None
        history.append({"gammas": list(gammas), "betas": list(betas), "energy": energy})
        job_reports.append({"step": step, "gammas": list(gammas), "betas": list(betas),
                            "energy": energy, "run_id": rep.jobs[0]["run_id"] if rep.jobs else None})
        if energy is not None and (best is None or energy < best["energy"]):
            best = {"gammas": list(gammas), "betas": list(betas), "energy": energy}

    return FullQaoaReport(
        plan_sha256="ask_tell:dynamic", jobs=tuple(job_reports), best_point=best,
        budget_used={"jobs": len(job_reports), "shots": shots_used},
        complete=bool(job_reports), anomalies=tuple(anomalies))
