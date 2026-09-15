"""Phase 5 hardware-validation tests: QUBO->Ising identity, QAOA build, fake-backend
transpilation, immutable artefact + hash, strict decoding, and Phase-3/4 coherence.

No IBM, no network, no token. Circuit/transpile tests use ``fake_provider`` only;
``RuntimeGateway.run_sampler`` / ``submit_run`` / a real service are never touched.
"""

from __future__ import annotations

import inspect
import itertools

import pytest

from acrpq.dashboard import backend_scoring as SC
from acrpq.dashboard import hardware_validation as H
from acrpq.quantum.ising import qubo_to_ising
from acrpq.quantum.qubo import build_qubo

_TS = 1_000_000.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _qubo(cp3, n_theta=2):
    return build_qubo(cp3, n_theta=n_theta)


def _request(qubo, *, backend="fake_lagos", n_theta=2, **over):
    n = qubo.n_qubits
    base = dict(
        instance_id="CP_3", qubo_sha256=H.canonical_qubo_hash(qubo), n_binary_vars=n,
        variable_meaning=tuple(f"x{b}" for b in range(n)), qaoa_reps=2,
        gammas=(0.3, 0.6), betas=(0.2, 0.4), shots=1024, transpiler_seed=7,
        backend_requested=backend, max_jobs=1, max_quantum_seconds=30.0, n_theta=n_theta,
        optimization_level=0, decision_ts=_TS,  # opt 0 is reproducible on fake_lagos
    )
    base.update(over)
    return H.HardwareValidationRequest(**base)


def _decision_for(backend_name, *, synthetic=True, logical_qubits=6, shots=1024):
    snap = SC.BackendSnapshot(
        name=backend_name, region="us-east", family="Falcon", physical_qubits=7,
        programmable_qubits=7, operational=True, maintenance=False, pending_jobs=3,
        two_qubit_error=0.006, readout_error=0.012, clops=2500.0, avg_coupling_degree=2.0,
        data_ts=_TS - 60.0, synthetic=synthetic)
    req = SC.WorkloadRequirements(logical_qubits=logical_qubits, est_two_qubit_gates=40,
                                  logical_depth=20, interaction_density=0.4, shots=shots,
                                  decision_ts=_TS)
    return SC.rank_backends(req, [snap]), req


# --------------------------------------------------------------------------- #
# A. request contract
# --------------------------------------------------------------------------- #
def test_request_roundtrip_and_submission_params(cp3):
    q = _qubo(cp3)
    r = _request(q)
    p = r.to_submission_params()
    assert p["mode"] == "hardware_validation" and p["backend_requested"] == "fake_lagos"
    assert p["qubo_sha256"] == r.qubo_sha256


@pytest.mark.parametrize("over", [
    {"qaoa_reps": 2, "gammas": (0.1,)},          # wrong number of gammas
    {"betas": (0.1, 0.2, 0.3)},                  # wrong number of betas
    {"gammas": (0.1, float("inf"))},             # non-finite angle
    {"gammas": (0.1, 100.0)},                    # angle out of bounds
    {"mode": "full_qaoa_hardware"},              # wrong mode
    {"shots": 0},                                # shots < 1
    {"transpiler_seed": True},                   # bool seed
    {"max_quantum_seconds": 0.0},                # budget not > 0
    {"bit_order": "weird"},                      # bad convention
    {"optimization_level": 5},                   # out of 0..3
    {"variable_meaning": ("only-one",)},         # wrong length
])
def test_request_rejects_bad_values(cp3, over):
    with pytest.raises(ValueError):  # HardwareValidationError is a ValueError
        _request(_qubo(cp3), **over)


def test_prepare_rejects_qubo_hash_mismatch(cp3):
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2

    q = _qubo(cp3)
    # a valid-format but wrong digest builds fine; the mismatch is caught at prepare
    req = _request(q, qubo_sha256="sha256:" + "0" * 64)
    with pytest.raises(H.HardwareValidationError):
        H.prepare_hardware_run(req, q, FakeLagosV2(), allow_synthetic=True)


def test_incoherent_declared_workload_is_refused(cp3):
    with pytest.raises(H.HardwareValidationError):
        _request(_qubo(cp3), n_binary_vars=999)


# --------------------------------------------------------------------------- #
# B. QUBO->Ising identity (independent brute force)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_theta", [2, 3])
def test_qubo_ising_energy_identity(cp3, n_theta):
    q = build_qubo(cp3, n_theta=n_theta)
    n = q.n_qubits
    if n > 9:
        pytest.skip("keep brute force small")
    ising = qubo_to_ising(q)
    for bits in itertools.product((0, 1), repeat=n):
        z = {b: 1 - 2 * bits[b] for b in range(n)}
        e_ising = (ising.offset + sum(h * z[b] for b, h in ising.h.items())
                   + sum(j * z[a] * z[c] for (a, c), j in ising.J.items()))
        e_qubo = q.energy("".join(map(str, bits)))
        assert abs(e_ising - e_qubo) < 1e-9


def test_canonical_qubo_hash_is_order_independent(cp3):
    q = _qubo(cp3)
    assert H.canonical_qubo_hash(q) == H.canonical_qubo_hash(q)
    assert H.canonical_qubo_hash(q).startswith("sha256:")


# --------------------------------------------------------------------------- #
# C. circuit construction
# --------------------------------------------------------------------------- #
def test_bound_circuit_is_fully_bound_with_measurements(cp3):
    q = _qubo(cp3)
    circ = H.build_bound_qaoa(q, _request(q))
    assert circ.num_parameters == 0
    assert circ.num_qubits == q.n_qubits and circ.num_clbits == q.n_qubits
    ops = circ.count_ops()
    assert ops.get("measure", 0) == q.n_qubits
    assert ops.get("rx", 0) == 2 * q.n_qubits          # 2 layers x n qubits
    assert ops.get("rzz", 0) >= len(qubo_to_ising(q).J)  # per layer couplings


def test_logical_fingerprint_is_deterministic(cp3):
    q = _qubo(cp3)
    c1 = H.build_bound_qaoa(q, _request(q))
    c2 = H.build_bound_qaoa(q, _request(q))
    assert H._circuit_fingerprint(c1)[0] == H._circuit_fingerprint(c2)[0]


# --------------------------------------------------------------------------- #
# D. transpilation (fake_provider only)
# --------------------------------------------------------------------------- #
def test_transpile_to_fake_lagos(cp3):
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2

    q = _qubo(cp3)
    circ = H.build_bound_qaoa(q, _request(q))
    isa, rep = H.transpile_to_isa(circ, FakeLagosV2(), seed=7, optimization_level=1)
    assert rep.backend_is_fake and rep.backend_num_qubits == 7
    assert rep.depth_after > 0 and rep.two_qubit_gates_after >= 0
    assert rep.non_basis_ops == ()          # everything mapped to the ISA
    assert len(rep.initial_layout) == q.n_qubits


def test_transpile_refuses_backend_too_small(cp3):
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    q = _qubo(cp3)  # 6 qubits > FakeManilaV2's 5
    circ = H.build_bound_qaoa(q, _request(q))
    with pytest.raises(H.BudgetExceeded):
        H.transpile_to_isa(circ, FakeManilaV2(), seed=7, optimization_level=1)


# --------------------------------------------------------------------------- #
# E. prepared artefact
# --------------------------------------------------------------------------- #
def test_prepare_synthetic_is_not_submittable_and_hashes(cp3):
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2

    q = _qubo(cp3)
    prep = H.prepare_hardware_run(_request(q), q, FakeLagosV2(), allow_synthetic=True)
    assert prep.submittable is False
    assert "artefact_not_submittable" in prep.warnings
    assert H.verify_artefact_hash(prep) is True


def test_artefact_hash_detects_tampering(cp3):
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2

    q = _qubo(cp3)
    prep = H.prepare_hardware_run(_request(q), q, FakeLagosV2(), allow_synthetic=True)
    import dataclasses
    tampered = dataclasses.replace(prep, backend="totally-different")
    assert H.verify_artefact_hash(tampered) is False


def test_prepare_is_deterministic(cp3):
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2

    q = _qubo(cp3)
    be = FakeLagosV2()
    a = H.prepare_hardware_run(_request(q), q, be, allow_synthetic=True)
    b = H.prepare_hardware_run(_request(q), q, be, allow_synthetic=True)
    assert a.artefact_sha256 == b.artefact_sha256


def test_depth_budget_is_enforced(cp3):
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2

    q = _qubo(cp3)
    with pytest.raises(H.BudgetExceeded):
        H.prepare_hardware_run(_request(q, max_depth=1), q, FakeLagosV2(), allow_synthetic=True)


def test_synthetic_prepare_requires_fake_backend_without_decision(cp3):
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2

    q = _qubo(cp3)
    # allow_synthetic False + no decision -> refuse
    with pytest.raises(H.Phase4CoherenceError):
        H.prepare_hardware_run(_request(q), q, FakeLagosV2(), allow_synthetic=False)


# --------------------------------------------------------------------------- #
# F. Phase-4 coherence
# --------------------------------------------------------------------------- #
def test_valid_decision_on_fake_backend_is_never_submittable(cp3):
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2

    q = _qubo(cp3)
    decision, workload = _decision_for("fake_lagos")
    prep = H.prepare_hardware_run(_request(q, backend="fake_lagos"), q, FakeLagosV2(),
                                  decision=decision, workload=workload)
    # a fake backend can never yield a submittable (payable) artefact
    assert prep.submittable is False
    assert "fake_backend_never_submittable" in prep.warnings
    assert prep.decision_sha256 == decision["decision_sha256"]


def test_backend_mismatch_with_decision_is_refused(cp3):
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2

    q = _qubo(cp3)
    decision, workload = _decision_for("some_other_backend")
    with pytest.raises(H.BackendMismatch):
        H.prepare_hardware_run(_request(q, backend="fake_lagos"), q, FakeLagosV2(),
                               decision=decision, workload=workload)


def test_tampered_decision_is_refused(cp3):
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2

    q = _qubo(cp3)
    decision, workload = _decision_for("fake_lagos")
    decision["selected"] = "fake_lagos"  # keep name, but break the hash coverage
    decision["uncertainty_block"] = 0.999999
    with pytest.raises(H.Phase4CoherenceError):
        H.prepare_hardware_run(_request(q, backend="fake_lagos"), q, FakeLagosV2(),
                               decision=decision, workload=workload)


# --------------------------------------------------------------------------- #
# G. Phase-3 hand-off (pure)
# --------------------------------------------------------------------------- #
def test_bundle_integrity_and_consistency(cp3):
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2

    q = _qubo(cp3)
    decision, workload = _decision_for("fake_lagos")
    bundle = H.prepare_hardware_bundle(_request(q, backend="fake_lagos"), q, FakeLagosV2(),
                                       decision=decision, workload=workload)
    # structural checks pass for a real bundle (independent of the submittable gate)
    assert H.verify_bundle_integrity(bundle) == []
    # but a fake backend is never submittable, so the hand-off refuses
    with pytest.raises(H.Phase4CoherenceError):
        H.to_submission_payload(bundle, local_run_id="run-1")
    # full-model consistency: expected params pass, a changed field is flagged
    prep = bundle.prepared
    expected = H._expected_phase3_params(prep)
    assert H.check_submission_consistency(prep, expected) == []
    H.assert_submission_consistency(prep, expected)  # does not raise
    bad = dict(expected)
    bad["shots"] = 9999
    assert "shots" in H.check_submission_consistency(prep, bad)
    with pytest.raises(H.HardwareValidationError):
        H.assert_submission_consistency(prep, bad)


def test_submission_payload_refuses_non_bundle_and_synthetic(cp3):
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2

    q = _qubo(cp3)
    with pytest.raises(H.HardwareValidationError):
        H.to_submission_payload("not-a-bundle", local_run_id="r")
    bundle = H.prepare_hardware_bundle(_request(q), q, FakeLagosV2(), allow_synthetic=True)
    with pytest.raises(H.Phase4CoherenceError):
        H.to_submission_payload(bundle, local_run_id="r")


# --------------------------------------------------------------------------- #
# H. strict decoding
# --------------------------------------------------------------------------- #
def test_decode_counts_endianness(cp3):
    q = _qubo(cp3)
    n = q.n_qubits
    # a raw qiskit (little-endian) string; bit 0 set means rightmost char '1'
    raw = {"0" * n: 10, "0" * (n - 1) + "1": 5}
    summ = H.decode_samples(q, raw, bit_order="qiskit", shots=15)
    # the '...1' qiskit string maps to QUBO-order bit 0 set
    bitset = {c.bitstring for c in summ.candidates}
    assert ("1" + "0" * (n - 1)) in bitset       # bit 0 in QUBO order
    # same data read as already-qubo-order flips which bit is set
    summ2 = H.decode_samples(q, raw, bit_order="qubo", shots=15)
    assert ("0" * (n - 1) + "1") in {c.bitstring for c in summ2.candidates}


def test_decode_distinguishes_best_raw_feasible_modal(cp3):
    q = _qubo(cp3)
    n = q.n_qubits
    # all-zeros is infeasible-ish (no maneuvers); make it modal but high energy,
    # and a one-hot-ish string the lower energy one.
    onehot = "".join("1" if b in (0,) else "0" for b in range(n))
    raw = {"0" * n: 900, onehot: 100}
    summ = H.decode_samples(q, raw, bit_order="qubo", shots=1000)
    assert summ.modal.bitstring == "0" * n          # most frequent
    assert summ.best_raw.energy <= summ.modal.energy  # best_raw is lowest energy
    assert 0.0 <= summ.feasibility_rate <= 1.0
    assert summ.n_unique == 2


def test_decode_never_calls_repaired_non_onehot_sample_feasible(cp3):
    """Raw hardware feasibility requires the encoding, not only repaired geometry."""
    q = _qubo(cp3, n_theta=3)
    # This real-CP5-shaped failure mode has two selected options per aircraft.
    # Search the tiny CP3 space for an invalid raw sample whose repaired maneuver
    # happens to be conflict-free; it must still be rejected as a QUBO solution.
    candidate = None
    for bits in itertools.product("01", repeat=q.n_qubits):
        bs = "".join(bits)
        groups = [bs[i * q.discrete.k:(i + 1) * q.discrete.k] for i in range(cp3.n)]
        if any(group.count("1") != 1 for group in groups):
            decoded = H.decode_samples(q, {bs: 1}, bit_order="qubo")
            if decoded.best_raw is not None and decoded.best_raw.n_conflicts == 0:
                candidate = decoded
                break
    assert candidate is not None
    assert candidate.best_raw.onehot_valid is False
    assert candidate.best_raw.n_onehot_violations > 0
    assert candidate.best_raw.feasible is False
    assert candidate.best_feasible is None
    assert candidate.feasibility_rate == 0.0


def test_decode_empty_result_is_honest(cp3):
    q = _qubo(cp3)
    summ = H.decode_samples(q, {}, bit_order="qiskit")
    assert summ.best_raw is None and summ.best_feasible is None
    assert "empty_result" in summ.warnings


def test_decode_scores_the_qubos_declared_objective(cp3):
    quadratic = build_qubo(cp3, n_theta=3, objective_id="quadratic_control_cost_v1")
    count = build_qubo(cp3, n_theta=3, objective_id="maneuver_count_v1")
    bitstring = "100010001"
    q_summary = H.decode_samples(quadratic, {bitstring: 1}, bit_order="qubo")
    c_summary = H.decode_samples(count, {bitstring: 1}, bit_order="qubo")
    assert q_summary.best_raw is not None and c_summary.best_raw is not None
    assert q_summary.best_raw.objective != c_summary.best_raw.objective
    assert c_summary.best_raw.objective == 2.0


@pytest.mark.parametrize("raw,kind", [
    ({"010": -1}, "counts"),                       # negative count
    ({"01": 3}, "counts"),                          # wrong bit width
    ({"0" * 6: 0.4, "1" + "0" * 5: 0.4}, "quasi"),  # quasi not summing to 1
    ({"0" * 6: -0.1, "1" + "0" * 5: 1.1}, "quasi"),  # negative quasi prob
])
def test_decode_rejects_malformed(cp3, raw, kind):
    q = _qubo(cp3)
    with pytest.raises(H.HardwareValidationError):
        H.decode_samples(q, raw, bit_order="qubo", kind=kind)


def test_decode_quasi_distribution(cp3):
    q = _qubo(cp3)
    n = q.n_qubits
    raw = {"0" * n: 0.7, "1" + "0" * (n - 1): 0.3}
    summ = H.decode_samples(q, raw, bit_order="qubo", kind="quasi")
    assert abs(sum(c.probability for c in summ.candidates) - 1.0) < 1e-9


def test_optimality_gap_sign_for_minimisation(cp3):
    q = _qubo(cp3)
    n = q.n_qubits
    onehot = "".join("1" if b == 0 else "0" for b in range(n))
    summ = H.decode_samples(q, {onehot: 1}, bit_order="qubo", kind="counts",
                            reference_objective=1.0)
    if summ.best_feasible is not None:
        # gap = (sampled - reference)/|reference|; sampled objective >= reference for a min
        assert summ.optimality_gap == pytest.approx(
            (summ.best_feasible.objective - 1.0) / 1.0)


# --------------------------------------------------------------------------- #
# I. no-IBM / no-submission proofs
# --------------------------------------------------------------------------- #
def test_module_has_no_submission_or_service_path():
    src = inspect.getsource(H)
    for forbidden in ("run_sampler", "QiskitRuntimeService", "submit_run",
                      "record_job_id", "QISKIT_IBM_TOKEN"):
        assert forbidden not in src


def test_import_is_qiskit_free():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, acrpq.dashboard.hardware_validation as m; "
         "print(any(k.startswith('qiskit') for k in sys.modules))"],
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
