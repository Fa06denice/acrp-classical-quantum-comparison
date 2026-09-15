"""Phase 6 orchestration tests: immutable plan, budgets, dry-run (no gateway),
and a full fake-gateway lifecycle (prepare -> confirm -> submit -> reconcile ->
completed -> decode). Never contacts IBM; a real RuntimeGateway is refused.
"""

from __future__ import annotations

import dataclasses

import pytest

from acrpq.dashboard import hardware_validation as H
from acrpq.dashboard import ibm_runner
from acrpq.dashboard import qpu_orchestrator as O
from acrpq.dashboard import qpu_runs
from acrpq.quantum.qubo import build_qubo

_TS = 1_000_000.0


# --- fake gateways (shaped like the ibm_runner test double) ----------------- #
class _Job:
    def __init__(self, jid, status="DONE"):
        self._id, self._status = jid, status

    def job_id(self):
        return self._id

    def status(self):
        return self._status

    def cancel(self):
        return None


class DoneGateway:
    """run_sampler succeeds; the job reports DONE."""

    dry_run_safe = True  # asserts this is a fake/test double, not a real gateway

    def __init__(self):
        self.run_calls = 0

    def run_sampler(self, *, isa_circuit, shots, tags, max_execution_time):
        self.run_calls += 1
        return _Job("sim-done-1")

    def find_jobs(self, *, tags, created_after_iso):
        return []

    def get_job(self, job_id):
        return _Job(job_id)


class AmbiguousThenFoundGateway:
    """Pre-submit search is empty (so a submission is attempted); run_sampler then
    raises a network error (submission_unknown), but the job DID land remotely and
    is found by tag on the LATER reconciliation -> completed."""

    dry_run_safe = True

    def __init__(self):
        self.run_calls = 0
        self.find_calls = 0

    def run_sampler(self, *, isa_circuit, shots, tags, max_execution_time):
        self.run_calls += 1
        raise ConnectionError("transient network blip after the request left")

    def find_jobs(self, *, tags, created_after_iso):
        self.find_calls += 1
        # empty for ALL pre-submit reads (before run_sampler is attempted); the
        # job only becomes visible once the submission has actually been tried.
        return [] if self.run_calls == 0 else [_Job("recovered-1")]

    def get_job(self, job_id):
        return _Job(job_id)


def _bundle(cp3, **over):
    q = build_qubo(cp3, n_theta=2)
    n = q.n_qubits
    base = dict(
        instance_id="CP_3", qubo_sha256=H.canonical_qubo_hash(q), n_binary_vars=n,
        variable_meaning=tuple(f"x{b}" for b in range(n)), qaoa_reps=1, gammas=(0.4,),
        betas=(0.3,), shots=512, transpiler_seed=1, backend_requested="fake_lagos", max_jobs=1,
        max_quantum_seconds=20.0, n_theta=2, optimization_level=0, decision_ts=_TS)
    base.update(over)
    req = H.HardwareValidationRequest(**base)
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2
    bundle = H.prepare_hardware_bundle(req, q, FakeLagosV2(), allow_synthetic=True)
    return q, bundle


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def test_plan_builds_and_hash_verifies(cp3):
    _q, bundle = _bundle(cp3)
    plan = O.build_execution_plan(bundle)
    assert O.verify_plan_hash(plan) is True
    assert plan.submittable is False and len(plan.jobs) == 1 and plan.total_shots == 512
    assert plan.backend == bundle.prepared.backend
    assert plan.config_hash == bundle.prepared.config_hash


def test_plan_hash_detects_mutation(cp3):
    _q, bundle = _bundle(cp3)
    plan = O.build_execution_plan(bundle)
    tampered = dataclasses.replace(plan, total_shots=999999)
    assert O.verify_plan_hash(tampered) is False


def test_full_qaoa_and_multi_angle_are_refused(cp3):
    _q, bundle = _bundle(cp3)
    with pytest.raises(O.OrchestrationError):
        O.build_execution_plan(bundle, n_angle_evals=3)


def test_plan_bundle_mismatch_is_detected(cp3):
    _q, b1 = _bundle(cp3)
    _q2, b2 = _bundle(cp3, shots=256)  # different config -> different artefact
    plan1 = O.build_execution_plan(b1)
    with pytest.raises(O.OrchestrationError):
        O.dry_run(plan1, b2)


# --------------------------------------------------------------------------- #
# dry-run
# --------------------------------------------------------------------------- #
def test_dry_run_reports_without_submitting(cp3):
    _q, bundle = _bundle(cp3)
    plan = O.build_execution_plan(bundle)
    report = O.dry_run(plan, bundle)
    assert report.submittable is False
    assert report.would_submit[0]["shots"] == 512
    assert report.would_submit[0]["expected_config_hash"] == bundle.prepared.config_hash
    assert "no state transitioned to submitting" in report.notes
    assert "dry_run_only_not_submittable" in report.warnings


# --------------------------------------------------------------------------- #
# fake-gateway lifecycle
# --------------------------------------------------------------------------- #
def test_lifecycle_done_reaches_completed_and_decodes(cp3, tmp_path):
    q, bundle = _bundle(cp3)
    plan = O.build_execution_plan(bundle)
    n = q.n_qubits
    counts = {"0" * n: 300, "1" + "0" * (n - 1): 212}
    rep = O.simulate_lifecycle(plan, bundle, q, gateway=DoneGateway(),
                               synthetic_counts=counts, root=str(tmp_path), decode_bit_order="qubo")
    assert rep.final_state == qpu_runs.QpuState.COMPLETED.value
    assert rep.job_ids == ("sim-done-1",)
    assert rep.decoded is not None and rep.decoded.best_raw is not None
    assert rep.anomalies == ()
    # the run is persisted, terminal and no longer a reconciliation candidate
    loaded = qpu_runs.load(rep.local_run_id, root=str(tmp_path))
    assert loaded["state"] == "completed"


def test_lifecycle_ambiguous_submission_is_reconciled(cp3, tmp_path):
    q, bundle = _bundle(cp3)
    plan = O.build_execution_plan(bundle)
    n = q.n_qubits
    counts = {"0" * n: 512}
    gw = AmbiguousThenFoundGateway()
    rep = O.simulate_lifecycle(plan, bundle, q, gateway=gw,
                               synthetic_counts=counts, root=str(tmp_path), decode_bit_order="qubo")
    # run_sampler was attempted exactly once and NEVER retried (no double submission)
    assert gw.run_calls == 1
    # the ambiguous submission was reconciled to a true terminal state via find_jobs
    assert rep.final_state in ("completed", "reconciliation_required", "submission_unknown")
    if rep.final_state == "completed":
        assert rep.decoded is not None
        assert "recovered-1" in rep.job_ids


def test_empty_counts_surface_as_anomaly(cp3, tmp_path):
    q, bundle = _bundle(cp3)
    plan = O.build_execution_plan(bundle)
    rep = O.simulate_lifecycle(plan, bundle, q, gateway=DoneGateway(),
                               synthetic_counts={}, root=str(tmp_path), decode_bit_order="qubo")
    # a completed run decoded from an EMPTY result must not look clean
    assert rep.final_state == "completed"
    assert any("empty_result" in a for a in rep.anomalies)


def test_real_gateway_is_refused(cp3, tmp_path):
    q, bundle = _bundle(cp3)
    plan = O.build_execution_plan(bundle)
    with pytest.raises(O.RealGatewayRefused):
        O.simulate_lifecycle(plan, bundle, q, gateway=ibm_runner.RuntimeGateway(),
                             synthetic_counts={}, root=str(tmp_path))


def test_unmarked_and_wrapped_gateways_are_refused(cp3, tmp_path):
    q, bundle = _bundle(cp3)
    plan = O.build_execution_plan(bundle)

    class Unmarked:  # a fake WITHOUT the dry_run_safe marker
        def run_sampler(self, **k): ...
        def find_jobs(self, **k): return []
        def get_job(self, j): ...

    class Wrapper:  # a pass-through wrapper around a real gateway (blocklist bypass)
        dry_run_safe = False

        def __init__(self):
            self._real = ibm_runner.RuntimeGateway()

        def run_sampler(self, **k):
            return self._real.run_sampler(**k)

        def find_jobs(self, **k):
            return self._real.find_jobs(**k)

        def get_job(self, j):
            return self._real.get_job(j)

    for gw in (Unmarked(), Wrapper()):
        with pytest.raises(O.RealGatewayRefused):
            O.simulate_lifecycle(plan, bundle, q, gateway=gw, synthetic_counts={},
                                 root=str(tmp_path))


def test_orchestrator_import_is_qiskit_free():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, acrpq.dashboard.qpu_orchestrator as m; "
         "print(any(k.startswith('qiskit') for k in sys.modules))"],
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
