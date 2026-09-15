"""Phase 6.1 tests: durable per-run artefact store (crash boundaries, tamper,
schema, ISA QPY roundtrip, cross-hash coherence)."""

from __future__ import annotations

import json

import pytest

from acrpq.dashboard import qpu_runs, qpu_store
from acrpq.quantum.qubo import build_qubo


def _params(instance="CP_3"):
    return {
        "mode": "hardware_validation", "instance": instance, "n_theta": 2, "n_q": 1,
        "shots": 512, "transpiler_seed": 1, "max_jobs": 1, "max_quantum_seconds": 20.0,
        "qubo_sha256": "sha256:" + "a" * 64, "backend_requested": "fake_lagos",
    }


def _run(root):
    return qpu_runs.prepare(_params(), root=root)


def test_save_load_roundtrip(tmp_path):
    root = str(tmp_path / "runs")
    rid = _run(root)
    dsha = qpu_store.save_decision(rid, {"selected": "fake_lagos", "decision_sha256": "sha256:x"},
                                   root=root)
    assert dsha.startswith("sha256:")
    got = qpu_store.load_decision(rid, root=root)
    assert got["selected"] == "fake_lagos"


def test_no_overwrite(tmp_path):
    root = str(tmp_path / "runs")
    rid = _run(root)
    qpu_store.save_plan(rid, {"plan_sha256": "sha256:1"}, root=root)
    with pytest.raises(FileExistsError):
        qpu_store.save_plan(rid, {"plan_sha256": "sha256:2"}, root=root)


def test_tamper_detected(tmp_path):
    root = str(tmp_path / "runs")
    rid = _run(root)
    qpu_store.save_artefact(rid, {"artefact_sha256": "sha256:z"}, root=root)
    p = qpu_store._doc_path(root, rid, "artefact")
    doc = json.loads(p.read_text())
    doc["payload"]["artefact_sha256"] = "sha256:tampered"   # change content, keep old hash
    p.write_text(json.dumps(doc))
    with pytest.raises(qpu_store.StoreError):
        qpu_store.load_artefact(rid, root=root)


def test_schema_mismatch_refused(tmp_path):
    root = str(tmp_path / "runs")
    rid = _run(root)
    qpu_store.save_decision(rid, {"selected": "x"}, root=root)
    p = qpu_store._doc_path(root, rid, "decision")
    doc = json.loads(p.read_text())
    doc["schema"] = "acrpq-qpu-store/999"
    from acrpq.dashboard.backend_scoring import _canonical
    from acrpq.dashboard.persist import _sha256
    payload_only = {k: v for k, v in doc.items() if k != "doc_sha256"}
    doc["doc_sha256"] = _sha256(_canonical(payload_only))   # re-hash so only schema differs
    p.write_text(json.dumps(doc))
    with pytest.raises(qpu_store.StoreError):
        qpu_store.load_decision(rid, root=root)


def test_secret_refused(tmp_path):
    root = str(tmp_path / "runs")
    rid = _run(root)
    with pytest.raises(ValueError):
        qpu_store.save_decision(rid, {"ibm_token": "shh"}, root=root)


def test_bad_run_id_refused(tmp_path):
    root = str(tmp_path / "runs")
    with pytest.raises(ValueError):
        qpu_store.save_decision("../etc/passwd", {"x": 1}, root=root)


def test_save_requires_committed_run(tmp_path):
    root = str(tmp_path / "runs")
    # a well-formed but nonexistent run id
    rid = _run(root)
    fake = rid[:-1] + ("0" if rid[-1] != "0" else "1")
    with pytest.raises(qpu_store.StoreError):
        qpu_store.save_decision(fake, {"x": 1}, root=root)


# --- crash boundaries: verify_full_run reports the reached stage (structural) --
def test_verify_full_run_stage_boundaries(tmp_path):
    root = str(tmp_path / "runs")
    rid = _run(root)
    r0 = qpu_store.verify_full_run(rid, root=root)
    assert r0["stage_reached"] is None and r0["present"]["decision"] is False
    qpu_store.save_decision(rid, {"decision_sha256": "sha256:d"}, root=root)
    r1 = qpu_store.verify_full_run(rid, root=root)
    assert r1["stage_reached"] == "decision" and r1["present"]["decision"] is True
    qpu_store.save_artefact(rid, {"artefact_sha256": "sha256:a"}, root=root)
    r2 = qpu_store.verify_full_run(rid, root=root)
    assert r2["stage_reached"] == "artefact"
    # synthetic hand-built docs are (correctly) NOT semantically coherent/submittable
    assert r2["real_submittable"] is False


def test_verify_full_run_detects_incoherent_chain(tmp_path):
    root = str(tmp_path / "runs")
    rid = _run(root)
    qpu_store.save_decision(rid, {"decision_sha256": "sha256:d"}, root=root)
    # artefact references a DIFFERENT decision hash
    qpu_store.save_artefact(rid, {"artefact_sha256": "sha256:a", "decision_sha256": "sha256:OTHER"},
                            root=root)
    r = qpu_store.verify_full_run(rid, root=root)
    assert r["coherent"] is False
    assert any("decision_sha256" in p for p in r["problems"])


# --- ISA QPY roundtrip (real circuit) ----------------------------------------
def test_isa_circuit_qpy_roundtrip(cp3, tmp_path):
    root = str(tmp_path / "runs")
    rid = _run(root)
    from acrpq.dashboard import hardware_validation as H
    q = build_qubo(cp3, n_theta=2)
    req = H.HardwareValidationRequest(
        instance_id="CP_3", qubo_sha256=H.canonical_qubo_hash(q), n_binary_vars=q.n_qubits,
        variable_meaning=tuple(f"x{b}" for b in range(q.n_qubits)), qaoa_reps=1, gammas=(0.4,),
        betas=(0.3,), shots=512, transpiler_seed=1, backend_requested="fake_lagos", max_jobs=1,
        max_quantum_seconds=20.0, n_theta=2, optimization_level=0)
    circ = H.build_bound_qaoa(q, req)
    payload = qpu_store.dump_isa_circuit(circ)
    qpu_store.save_isa_circuit(rid, payload, root=root)
    reloaded = qpu_store.load_isa_circuit(rid, root=root, as_circuit=True)
    # the reloaded circuit re-fingerprints to the stored value (verified inside load)
    fp, _ = H._circuit_fingerprint(reloaded)
    assert fp == payload["fingerprint_sha256"]


def _real_chain(cp3, root):
    """Persist a REAL decision/artefact/plan/ISA chain for a prepared run."""
    import dataclasses
    from acrpq.dashboard import backend_scoring as SC
    from acrpq.dashboard import hardware_validation as H
    from acrpq.dashboard import qpu_orchestrator as O
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2
    q = build_qubo(cp3, n_theta=2)
    n = q.n_qubits
    snap = SC.BackendSnapshot(name="fake_lagos", region="us-east", family="F", physical_qubits=7,
                              programmable_qubits=7, operational=True, maintenance=False,
                              pending_jobs=3, two_qubit_error=0.006, readout_error=0.012,
                              clops=2500.0, avg_coupling_degree=2.0, data_ts=0.0, synthetic=True)
    wl = SC.WorkloadRequirements(logical_qubits=n, est_two_qubit_gates=len(q.quadratic),
                                 logical_depth=2, interaction_density=0.4, shots=512,
                                 max_quantum_seconds=20.0, decision_ts=0.0)
    decision = SC.rank_backends(wl, [snap])
    req = H.HardwareValidationRequest(
        instance_id="CP_3", qubo_sha256=H.canonical_qubo_hash(q), n_binary_vars=n,
        variable_meaning=tuple(f"x{b}" for b in range(n)), qaoa_reps=1, gammas=(0.4,), betas=(0.3,),
        shots=512, transpiler_seed=1, backend_requested="fake_lagos", max_jobs=1,
        max_quantum_seconds=20.0, n_theta=2, optimization_level=0)
    bundle = H.prepare_hardware_bundle(req, q, FakeLagosV2(), decision=decision, workload=wl)
    plan = O.build_execution_plan(bundle)
    rid = qpu_runs.prepare(H._expected_phase3_params(bundle.prepared), root=root)
    qpu_store.save_decision(rid, decision, root=root)
    qpu_store.save_artefact(rid, dataclasses.asdict(bundle.prepared), root=root)
    qpu_store.save_plan(rid, dataclasses.asdict(plan), root=root)
    qpu_store.save_isa_circuit(rid, qpu_store.dump_isa_circuit(bundle.isa_circuit), root=root)
    return rid


def test_verify_full_run_real_chain_is_complete_and_coherent(cp3, tmp_path):
    root = str(tmp_path / "runs")
    rid = _real_chain(cp3, root)
    r = qpu_store.verify_full_run(rid, root=root)
    assert r["complete"] is True and r["coherent"] is True
    # fake backend -> NOT real_submittable even with a valid chain
    assert r["real_submittable"] is False
    assert "artefact_sha256" in r["verified_hashes"] and "plan_sha256" in r["verified_hashes"]


@pytest.mark.parametrize("kind,field,value", [
    ("artefact", "submittable", True),          # forged submittable
    ("artefact", "backend", "ibm_real_device"),  # backend swap
    ("plan", "total_shots", 999999),             # budget swap
])
def test_verify_full_run_detects_semantic_corruption(cp3, tmp_path, kind, field, value):
    import json
    from acrpq.dashboard.backend_scoring import _canonical
    from acrpq.dashboard.persist import _sha256
    root = str(tmp_path / "runs")
    rid = _real_chain(cp3, root)
    p = qpu_store._doc_path(root, rid, kind)
    doc = json.loads(p.read_text())
    doc["payload"][field] = value
    # correctly RE-HASH the store envelope so only the BUSINESS hash is wrong
    payload_only = {k: v for k, v in doc.items() if k != "doc_sha256"}
    doc["doc_sha256"] = _sha256(_canonical(payload_only))
    p.write_text(json.dumps(doc))
    r = qpu_store.verify_full_run(rid, root=root)
    assert r["coherent"] is False and r["real_submittable"] is False


def test_import_is_qiskit_free():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, acrpq.dashboard.qpu_store as m; "
         "print(any(k.startswith('qiskit') for k in sys.modules))"],
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
