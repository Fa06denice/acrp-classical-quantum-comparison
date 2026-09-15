"""Phase 5.1 trust-chain tests: bind decision->workload->backend->circuit->payload,
and decode aggregates computed BEFORE truncation. No IBM/network/token.
"""

from __future__ import annotations

import dataclasses

import pytest

from acrpq.dashboard import backend_scoring as SC
from acrpq.dashboard import hardware_validation as H
from acrpq.quantum.qubo import build_qubo

_TS = 1_000_000.0


def _qubo(cp3, n_theta=2):
    return build_qubo(cp3, n_theta=n_theta)


def _request(qubo, *, backend="fake_lagos", n_theta=2, **over):
    n = qubo.n_qubits
    base = dict(
        instance_id="CP_3", qubo_sha256=H.canonical_qubo_hash(qubo), n_binary_vars=n,
        variable_meaning=tuple(f"x{b}" for b in range(n)), qaoa_reps=1, gammas=(0.4,),
        betas=(0.3,), shots=1024, transpiler_seed=1, backend_requested=backend, max_jobs=1,
        max_quantum_seconds=30.0, n_theta=n_theta, optimization_level=0, decision_ts=_TS)
    base.update(over)
    return H.HardwareValidationRequest(**base)


def _decision(name="fake_lagos", *, logical_qubits=6, shots=1024, decision_ts=_TS,
              synthetic=True):
    snap = SC.BackendSnapshot(
        name=name, region="us-east", family="Falcon", physical_qubits=7, programmable_qubits=7,
        operational=True, maintenance=False, pending_jobs=3, two_qubit_error=0.006,
        readout_error=0.012, clops=2500.0, avg_coupling_degree=2.0, data_ts=decision_ts - 60.0,
        synthetic=synthetic)
    req = SC.WorkloadRequirements(logical_qubits=logical_qubits, est_two_qubit_gates=40,
                                  logical_depth=20, interaction_density=0.4, shots=shots,
                                  decision_ts=decision_ts)
    return SC.rank_backends(req, [snap]), req


# --------------------------------------------------------------------------- #
# A. workload binding
# --------------------------------------------------------------------------- #
def test_decision_records_workload_commitment(cp3):
    decision, workload = _decision()
    assert decision["workload_sha256"] == SC.workload_commitment(workload)


def test_decision_for_other_workload_is_refused(cp3):
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2

    q = _qubo(cp3)  # 6 qubits, 1024 shots
    # a decision made for a 1-qubit / 1-shot workload must not bind a 6-qubit prep
    decision, workload = _decision(logical_qubits=1, shots=1)
    with pytest.raises(H.Phase4CoherenceError):
        H.prepare_hardware_run(_request(q), q, FakeLagosV2(), decision=decision, workload=workload)


def test_decision_requires_matching_workload_object(cp3):
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2

    q = _qubo(cp3)
    decision, _workload = _decision()
    other = SC.WorkloadRequirements(logical_qubits=6, est_two_qubit_gates=999, logical_depth=20,
                                    interaction_density=0.4, shots=1024, decision_ts=_TS)
    with pytest.raises(H.Phase4CoherenceError):  # commitment mismatch
        H.prepare_hardware_run(_request(q), q, FakeLagosV2(), decision=decision, workload=other)


def test_decision_without_workload_is_refused(cp3):
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2

    q = _qubo(cp3)
    decision, _workload = _decision()
    with pytest.raises(H.Phase4CoherenceError):
        H.prepare_hardware_run(_request(q), q, FakeLagosV2(), decision=decision, workload=None)


def test_stale_decision_is_refused(cp3):
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2

    q = _qubo(cp3)
    decision, workload = _decision(decision_ts=_TS - 10 * 24 * 3600.0)  # 10 days old
    with pytest.raises(H.Phase4CoherenceError):
        H.prepare_hardware_run(_request(q), q, FakeLagosV2(), decision=decision, workload=workload,
                               decision_max_age_s=3600.0)


# --------------------------------------------------------------------------- #
# Red-team F1: a fake backend disguised by name is still detected as fake
# --------------------------------------------------------------------------- #
def test_disguised_fake_backend_is_still_not_submittable(cp3):
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2

    class AcrpqLocalTestBackend(FakeLagosV2):  # innocuous name, but still a fake
        pass

    q = _qubo(cp3)
    decision, workload = _decision("fake_lagos")   # name still resolves to fake_lagos
    prep = H.prepare_hardware_run(_request(q, backend="fake_lagos"), q, AcrpqLocalTestBackend(),
                                  decision=decision, workload=workload)
    assert prep.submittable is False               # isinstance(FakeBackendV2) catches the subclass
    assert "fake_backend_never_submittable" in prep.warnings


# Red-team F3: billing-relevant budgets are hard-capped
@pytest.mark.parametrize("over", [
    {"max_jobs": 10_000},                 # over MAX_HW_MAX_JOBS
    {"max_quantum_seconds": 315360000.0}, # 10 years, over MAX_HW_MAX_QUANTUM_SECONDS
])
def test_budget_hard_caps(cp3, over):
    with pytest.raises(H.HardwareValidationError):
        _request(_qubo(cp3), **over)


# --------------------------------------------------------------------------- #
# B/C. backend identity
# --------------------------------------------------------------------------- #
def test_backend_name_must_match_request(cp3):
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2

    q = _qubo(cp3)
    # request claims a different backend than the real object (fake_lagos)
    with pytest.raises(H.BackendMismatch):
        H.prepare_hardware_run(_request(q, backend="ibm_real_device"), q, FakeLagosV2(),
                               allow_synthetic=True)


# --------------------------------------------------------------------------- #
# D/F. bundle integrity vs tampering / swapped circuit
# --------------------------------------------------------------------------- #
def test_bundle_detects_swapped_isa_circuit(cp3):
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2

    q = _qubo(cp3)
    bundle = H.prepare_hardware_bundle(_request(q), q, FakeLagosV2(), allow_synthetic=True)
    assert H.verify_bundle_integrity(bundle) == []
    # swap in a different circuit -> fingerprint mismatch
    other_logical = H.build_bound_qaoa(q, _request(q, gammas=(1.23,), betas=(0.77,)))
    swapped = dataclasses.replace(bundle, isa_circuit=other_logical)
    problems = H.verify_bundle_integrity(swapped)
    assert any("isa" in p for p in problems)
    # an arbitrary object is rejected, not fingerprinted
    junk = dataclasses.replace(bundle, isa_circuit=object())
    assert H.verify_bundle_integrity(junk)


def test_bundle_detects_tampered_artefact(cp3):
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2

    q = _qubo(cp3)
    bundle = H.prepare_hardware_bundle(_request(q), q, FakeLagosV2(), allow_synthetic=True)
    tampered_prep = dataclasses.replace(bundle.prepared, backend="evil")
    tampered = dataclasses.replace(bundle, prepared=tampered_prep)
    assert "artefact_hash_invalid" in H.verify_bundle_integrity(tampered)


# --------------------------------------------------------------------------- #
# G. decode aggregates computed BEFORE truncation
# --------------------------------------------------------------------------- #
def test_best_raw_survives_detail_truncation(cp3):
    q = _qubo(cp3)
    n = q.n_qubits
    # Build many high-probability high-energy strings + one rare low-energy string.
    # With max_candidates=1 the old code truncated by probability first and would
    # drop the rare low-energy sample from best_raw. It must NOT now.
    import itertools
    strings = ["".join(b) for b in itertools.islice(itertools.product("01", repeat=n), 40)]
    energies = {s: q.energy(s) for s in strings}
    rare_low = min(energies, key=energies.get)          # the globally lowest energy
    raw = {}
    for s in strings:
        raw[s] = 1 if s == rare_low else 100            # rare_low is the LEAST probable
    summ = H.decode_samples(q, raw, bit_order="qubo", kind="counts", max_candidates=1)
    assert summ.best_raw.bitstring == rare_low          # not dropped by truncation
    assert summ.n_unique_total == len(strings)
    assert summ.n_candidates_returned == 1
    assert 0.0 < summ.probability_mass_returned < 1.0
    assert any(w.startswith("truncated_detail_list") for w in summ.warnings)


# --------------------------------------------------------------------------- #
# H. counts vs shots
# --------------------------------------------------------------------------- #
def test_counts_shots_must_match_sum(cp3):
    q = _qubo(cp3)
    n = q.n_qubits
    raw = {"0" * n: 600, "1" + "0" * (n - 1): 400}   # sum = 1000
    ok = H.decode_samples(q, raw, bit_order="qubo", kind="counts", shots=1000)
    assert ok.n_shots == 1000
    with pytest.raises(H.HardwareValidationError):
        H.decode_samples(q, raw, bit_order="qubo", kind="counts", shots=999)


# --------------------------------------------------------------------------- #
# I. optimality gap with zero reference
# --------------------------------------------------------------------------- #
def test_zero_reference_gives_no_relative_gap(cp3):
    q = _qubo(cp3)
    n = q.n_qubits
    onehot = "".join("1" if b == 0 else "0" for b in range(n))
    summ = H.decode_samples(q, {onehot: 1}, bit_order="qubo", kind="counts",
                            reference_objective=0.0)
    if summ.best_feasible is not None:
        assert summ.optimality_gap is None
        assert summ.absolute_gap is not None
        assert "relative_gap_undefined_zero_reference" in summ.warnings


# --------------------------------------------------------------------------- #
# J. decode limit validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kw", [{"max_candidates": 0}, {"n_explain": -1}])
def test_decode_limit_validation(cp3, kw):
    q = _qubo(cp3)
    n = q.n_qubits
    with pytest.raises(ValueError):  # HardwareValidationError is a ValueError
        H.decode_samples(q, {"0" * n: 1}, bit_order="qubo", kind="counts", **kw)


def test_decode_refuses_too_many_raw_entries(cp3, monkeypatch):
    q = _qubo(cp3)
    n = q.n_qubits
    monkeypatch.setattr(H, "MAX_DECODED_CANDIDATES", 2)
    raw = {f"{i:0{n}b}": 1 for i in range(3)}
    with pytest.raises(H.BudgetExceeded):
        H.decode_samples(q, raw, bit_order="qubo", kind="counts")
