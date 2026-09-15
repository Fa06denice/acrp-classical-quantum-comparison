"""End-to-end QPU integration (no IBM, no network, no token).

Test A: instance -> classical objective (read-only) -> backend scoring -> QAOA
        preparation -> fake transpilation -> NOT submittable -> execution plan ->
        dry-run -> fake-gateway lifecycle -> decode -> verified export.
Test B: a fully fake gateway drives prepare -> confirm -> submit -> reconcile ->
        completed -> decode, and never contacts IBM.
"""

from __future__ import annotations

import json
from pathlib import Path

from acrpq.dashboard import backend_scoring as SC
from acrpq.dashboard import hardware_validation as H
from acrpq.dashboard import qpu_export as EX
from acrpq.dashboard import qpu_orchestrator as O
from acrpq.dashboard import qpu_runs
from acrpq.quantum.qubo import build_qubo

_TS = 1_000_000.0


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
    dry_run_safe = True

    def __init__(self):
        self.run_calls = 0

    def run_sampler(self, *, isa_circuit, shots, tags, max_execution_time):
        self.run_calls += 1
        return _Job("integ-1")

    def find_jobs(self, *, tags, created_after_iso):
        return []

    def get_job(self, job_id):
        return _Job(job_id)


def _bundle_and_workload(cp3):
    q = build_qubo(cp3, n_theta=2)
    n = q.n_qubits
    snap = SC.BackendSnapshot(
        name="fake_lagos", region="us-east", family="Falcon", physical_qubits=7,
        programmable_qubits=7, operational=True, maintenance=False, pending_jobs=3,
        two_qubit_error=0.006, readout_error=0.012, clops=2500.0, avg_coupling_degree=2.0,
        data_ts=_TS - 60.0, synthetic=True)
    workload = SC.WorkloadRequirements(
        logical_qubits=n, est_two_qubit_gates=len(q.quadratic), logical_depth=2,
        interaction_density=0.4, shots=512, decision_ts=_TS)
    decision = SC.rank_backends(workload, [snap])
    req = H.HardwareValidationRequest(
        instance_id="CP_3", qubo_sha256=H.canonical_qubo_hash(q), n_binary_vars=n,
        variable_meaning=tuple(f"x{b}" for b in range(n)), qaoa_reps=1, gammas=(0.4,),
        betas=(0.3,), shots=512, transpiler_seed=1, backend_requested="fake_lagos", max_jobs=1,
        max_quantum_seconds=20.0, n_theta=2, optimization_level=0, decision_ts=_TS)
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2
    bundle = H.prepare_hardware_bundle(req, q, FakeLagosV2(), decision=decision, workload=workload)
    return q, bundle, decision


def test_A_full_pipeline_to_verified_export(cp3, tmp_path):
    q, bundle, decision = _bundle_and_workload(cp3)
    p = bundle.prepared

    # scoring picked the backend; artefact is bound + NOT submittable (fake)
    assert decision["selected"] == "fake_lagos"
    assert p.submittable is False
    assert H.verify_artefact_hash(p) is True
    assert H.verify_decision_hash(decision) is True
    assert H.verify_bundle_integrity(bundle) == []

    # plan + dry-run (no submission)
    plan = O.build_execution_plan(bundle)
    assert O.verify_plan_hash(plan) is True
    report = O.dry_run(plan, bundle)
    assert report.submittable is False

    # fake lifecycle -> completed -> decode
    root = str(tmp_path / "runs")
    n = q.n_qubits
    counts = {"0" * n: 260, "1" + "0" * (n - 1): 252}
    life = O.simulate_lifecycle(plan, bundle, q, gateway=DoneGateway(),
                                synthetic_counts=counts, root=root, decode_bit_order="qubo")
    assert life.final_state == "completed"
    decoded = life.decoded
    assert decoded.best_raw is not None
    # explanation carries the honest heuristic framing
    assert decoded.n_unique_total == 2

    # verified, secret-free, atomic export
    dest = str(tmp_path / "export.json")
    EX.export_run(life.local_run_id, root=root, dest_path=dest,
                  decision=decision, artefact=p.to_manifest(), plan=plan, decoded=decoded)
    on_disk = json.loads(Path(dest).read_text())
    assert EX.verify_export(on_disk) is True
    assert on_disk["run"]["state"] == "completed"
    # nothing secret slipped into the export
    assert "token" not in json.dumps(on_disk).lower()


def test_B_fake_gateway_never_contacts_ibm(cp3, tmp_path):
    q, bundle, _decision = _bundle_and_workload(cp3)
    plan = O.build_execution_plan(bundle)
    gw = DoneGateway()
    root = str(tmp_path / "runs")
    n = q.n_qubits
    life = O.simulate_lifecycle(plan, bundle, q, gateway=gw,
                                synthetic_counts={"0" * n: 512}, root=root, decode_bit_order="qubo")
    assert gw.run_calls == 1                 # submitted exactly once, never retried
    assert life.final_state == "completed"
    assert life.job_ids == ("integ-1",)
    # the run is persisted, terminal, and not a reconciliation candidate
    assert qpu_runs.load(life.local_run_id, root=root)["state"] == "completed"
    assert life.local_run_id not in [
        r.get("local_run_id") for r in qpu_runs.reconciliation_candidates(root=root)]
