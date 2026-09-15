"""Phase 5.3 — robust IBM Sampler result adapter + scientific decode (no IBM).

Anonymised, clearly-synthetic fixtures only. Covers normalisation shapes/failure modes,
the negative-quasi preservation policy, and the distinct (never-conflated) decode
candidates.
"""
from __future__ import annotations

import warnings as _w

import pytest

from acrpq.dashboard import qpu_result as R
from acrpq.io.loader import InstanceLoader
from acrpq.dashboard import qpu_batch

_w.filterwarnings("ignore")


# --- normalise: happy shapes ----------------------------------------------
def test_counts_qubo_order():
    n = R.normalize_ibm_sampler_result(
        {"kind": "counts", "bit_order": "qubo", "counts": {"000": 6, "111": 2}},
        expected_n_bits=3)
    assert n.prob_qubo_order == {"000": 0.75, "111": 0.25} and n.n_shots == 8
    assert n.negative_quasi_policy == "not_applicable"


def test_counts_qiskit_order_is_reversed():
    n = R.normalize_ibm_sampler_result(
        {"kind": "counts", "bit_order": "qiskit", "counts": {"001": 4}}, expected_n_bits=3)
    assert list(n.prob_qubo_order) == ["100"] and n.endianness_applied == "reversed"


def test_hex_and_spaced_keys_are_cleaned():
    n = R.normalize_ibm_sampler_result(
        {"kind": "counts", "bit_order": "qubo", "counts": {"0x5": 3, "1 1 0": 1}},
        expected_n_bits=3)
    assert set(n.prob_qubo_order) == {"101", "110"}


def test_quasi_preserves_negatives_without_clamping():
    n = R.normalize_ibm_sampler_result(
        {"kind": "quasi", "bit_order": "qubo", "counts": {"000": 1.2, "111": -0.2}},
        expected_n_bits=3)
    assert n.prob_qubo_order == {"000": 1.2, "111": -0.2}     # NOT clamped
    assert "negative_quasi_preserved" in n.warnings and n.negative_quasi_policy == "preserved"


def test_single_pub_single_register_extracted():
    n = R.normalize_ibm_sampler_result(
        {"kind": "counts", "bit_order": "qubo",
         "pubs": [{"registers": {"c": {"00": 2}}}]}, expected_n_bits=2)
    assert n.prob_qubo_order == {"00": 1.0}


# --- normalise: fail closed -----------------------------------------------
@pytest.mark.parametrize("env", [
    {"kind": "counts", "bit_order": "qubo", "counts": {}},                    # empty
    {"kind": "nope", "bit_order": "qubo", "counts": {"0": 1}},                 # unknown kind
    {"kind": "counts", "bit_order": "weird", "counts": {"0": 1}},              # bad order
    {"kind": "counts", "bit_order": "qubo", "counts": {"00": -1}},            # negative count
    {"kind": "counts", "bit_order": "qubo", "counts": {"00": 1.5}},           # non-int count
    {"kind": "counts", "bit_order": "qubo", "counts": {"0": 0}},              # zero sum (1-bit)
    {"kind": "counts", "bit_order": "qubo", "counts": {"0000": 1}},           # wrong length
    {"kind": "quasi", "bit_order": "qubo", "counts": {"00": float("nan")}},   # non-finite
    {"kind": "quasi", "bit_order": "qubo", "counts": {"00": 0.0}},            # near-zero sum
    {"kind": "counts", "bit_order": "qubo",
     "pubs": [{"registers": {"a": {"00": 1}}}, {"registers": {"b": {"00": 1}}}]},  # multi PUB
    {"kind": "counts", "bit_order": "qubo",
     "registers": {"a": {"00": 1}, "b": {"11": 1}}},                          # multi register
    "not-a-dict",
])
def test_normalise_fails_closed(env):
    with pytest.raises(R.ResultAdapterError):
        R.normalize_ibm_sampler_result(env, expected_n_bits=2)


# --- scientific decode: distinct candidates -------------------------------
@pytest.fixture(scope="module")
def cp3():
    from acrpq.quantum.qubo import build_qubo
    inst = InstanceLoader().load("CP_3")
    q = build_qubo(inst, n_theta=3, n_q=1, objective_id="maneuver_count_v1")
    o = qpu_batch.optimum_onehot_bits(inst, q, "maneuver_count_v1", 3, 1)
    return q, o


def test_decode_keeps_candidates_distinct(cp3):
    q, o = cp3
    opt = o["bitstring"]                                  # feasible, one-hot, optimal
    allzero = "0" * q.n_qubits                            # not one-hot; repaired to all-noop
    # make the non-optimal all-zero bitstring the MOST probable -> modal != best_raw
    env = {"kind": "counts", "bit_order": "qubo", "counts": {opt: 10, allzero: 90}}
    d = R.decode_scientific(q, env, objective_id="maneuver_count_v1",
                            reference_objective=o["optimum"])
    assert d.best_raw_bitstring == opt                   # lowest energy
    assert d.modal_bitstring == allzero                  # most probable (different!)
    assert d.best_onehot_valid_bitstring == opt          # only opt is raw one-hot
    assert d.best_feasible_bitstring is not None
    assert d.comparability == "comparable"
    assert d.delta_vs_reference is not None
    # feasibility vs one-hot validity vs energy-min are reported SEPARATELY
    assert 0.0 <= d.feasibility_rate <= 1.0
    assert 0.0 <= d.onehot_valid_mass <= 1.0


def test_decode_no_reference_is_not_comparable(cp3):
    q, o = cp3
    env = {"kind": "counts", "bit_order": "qubo", "counts": {o["bitstring"]: 4}}
    d = R.decode_scientific(q, env, objective_id="maneuver_count_v1", reference_objective=None)
    assert d.comparability == "no_reference" and d.delta_vs_reference is None


def test_decode_negative_quasi_is_not_comparable(cp3):
    q, o = cp3
    env = {"kind": "quasi", "bit_order": "qubo",
           "counts": {o["bitstring"]: 1.1, "0" * q.n_qubits: -0.1}}
    d = R.decode_scientific(q, env, objective_id="maneuver_count_v1", reference_objective=o["optimum"])
    assert d.comparability == "not_comparable" and "negative_quasi" in d.comparability_reason
