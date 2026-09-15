"""Phase 4.1 adversarial hardening tests for the backend scorer.

Covers the eight reproduced defects (A-H) plus the Phase 4 audit checklist (I):
strict model validation, full-record+provenance hashing with a public verifier,
backend-name uniqueness, uncertainty-block validation, hard-threshold-requires-
metric, weighted (contribution-based) explanations, strict JSON-native
serialisation, and defensive immutability. No IBM, no network, no qiskit.
"""

from __future__ import annotations

import math

import pytest

from acrpq.dashboard import backend_scoring as S
from acrpq.dashboard.backend_scoring import (
    BackendSnapshot,
    ScoringWeights,
    TranspilationMetrics,
    WorkloadRequirements,
    verify_decision_hash,
)

_NOW = 1_000_000.0


def _wl(**over):
    base = dict(logical_qubits=5, est_two_qubit_gates=40, logical_depth=20,
                interaction_density=0.5, shots=1024, decision_ts=_NOW)
    base.update(over)
    return WorkloadRequirements(**base)


def _snap(name, **over):
    base = dict(
        name=name, region="us-east", family="Heron", physical_qubits=127,
        programmable_qubits=127, operational=True, maintenance=False,
        pending_jobs=10, two_qubit_error=0.005, two_qubit_error_layered=0.008,
        readout_error=0.01, clops=3000.0, avg_coupling_degree=2.5,
        data_ts=_NOW - 3600.0, synthetic=True,
    )
    base.update(over)
    return BackendSnapshot(**base)


# ---- A. strict model validation ------------------------------------------- #
@pytest.mark.parametrize("over", [
    {"two_qubit_error": -0.1},          # negative
    {"two_qubit_error": True},          # bool as number
    {"two_qubit_error": float("nan")},  # NaN
    {"two_qubit_error": float("inf")},  # +Inf
    {"two_qubit_error": "0.01"},        # wrong type
    {"two_qubit_error": 1.5},           # error > 1
    {"readout_error": 2.0},             # error > 1
    {"physical_qubits": 5, "programmable_qubits": 10},  # programmable > physical
    {"data_ts": float("nan")},          # invalid timestamp
    {"data_ts": -1.0},                  # negative timestamp
    {"operational": 1},                 # flag not a real bool
    {"clops": -3.0},                    # negative
    {"avg_coupling_degree": float("inf")},
    {"region": ""},                     # empty string
    {"pending_jobs": True},             # bool as count
    {"pending_jobs": -1},               # negative count
])
def test_snapshot_rejects_bad_values(over):
    with pytest.raises(ValueError):
        _snap("b", **over)


@pytest.mark.parametrize("over", [
    {"logical_qubits": 0},              # < 1
    {"shots": 0},                       # < 1
    {"interaction_density": 1.5},       # out of [0,1]
    {"interaction_density": True},      # bool
    {"max_quantum_seconds": 0.0},       # not strictly positive
    {"max_quantum_seconds": -5.0},      # negative budget
    {"decision_ts": -1.0},              # negative timestamp
    {"max_two_qubit_error": 1.5},       # threshold out of [0,1]
    {"needs_dynamic_circuits": 1},      # flag not a bool
    {"allowed_regions": ["us-east"]},   # list, not tuple
    {"allowed_regions": ("us", "us")},  # duplicate
    {"blocked_backends": ("",)},        # empty element
    {"blocked_backends": (5,)},         # non-string element
    {"est_two_qubit_gates": -1},        # negative count
])
def test_workload_rejects_bad_values(over):
    with pytest.raises(ValueError):
        _wl(**over)


@pytest.mark.parametrize("over", [
    {"provenance": {"k": {1, 2}}},           # set value
    {"provenance": {"k": b"x"}},             # bytes value
    {"provenance": {1: "v"}},                # non-string key
    {"provenance": {"k": float("inf")}},     # non-finite value
    {"unknown_fields": ("two_qubit_error",), "two_qubit_error": 0.01},  # listed but present
    {"unknown_fields": ("bogus",)},          # not an optional metric
    {"unknown_fields": ("clops", "clops"), "clops": None},  # duplicate
])
def test_snapshot_provenance_and_unknown_fields_are_strict(over):
    with pytest.raises(ValueError):
        _snap("b", **over)


@pytest.mark.parametrize("bad_name", ["   ", "", "\t\n"])
def test_snapshot_rejects_empty_name(bad_name):
    with pytest.raises(ValueError):
        BackendSnapshot(name=bad_name, region="us", family="Heron", physical_qubits=5,
                        programmable_qubits=5, operational=True, maintenance=False)


def test_unknown_fields_coherent_when_metric_is_absent():
    s = _snap("b", clops=None, unknown_fields=("clops",))
    assert "clops" in s.unknown_fields


@pytest.mark.parametrize("kw", [
    dict(isa_depth=-1, two_qubit_gate_count=1, swap_count=0, layout_ok=True),
    dict(isa_depth=1, two_qubit_gate_count=-1, swap_count=0, layout_ok=True),
    dict(isa_depth=1, two_qubit_gate_count=1, swap_count=0, layout_ok=1),  # not bool
    dict(isa_depth=True, two_qubit_gate_count=1, swap_count=0, layout_ok=True),
])
def test_transpilation_metrics_validation(kw):
    with pytest.raises(ValueError):
        TranspilationMetrics(**kw)


@pytest.mark.parametrize("kw", [
    dict(two_qubit_quality=float("nan")),
    dict(two_qubit_quality=float("inf")),
    dict(two_qubit_quality=True, readout_quality=-0.7 + 0.7),  # bool weight
])
def test_weights_reject_nonfinite_or_bool(kw):
    with pytest.raises(ValueError):
        ScoringWeights(**kw)


# ---- B. hash covers the whole decision + provenance ----------------------- #
def test_hash_covers_provenance():
    ra = S.rank_backends(_wl(), [_snap("a")], provenance={"v": "A"})
    rb = S.rank_backends(_wl(), [_snap("a")], provenance={"v": "B"})
    assert ra["decision_sha256"] != rb["decision_sha256"]


def test_hash_changes_when_any_field_mutates():
    base = S.rank_backends(_wl(), [_snap("a")])
    diff_weight = S.rank_backends(_wl(), [_snap("a")],
                                  weights=ScoringWeights(two_qubit_quality=0.35, readout_quality=0.10,
                                                         topological_fit=0.15, qubit_margin=0.05,
                                                         queue_pressure=0.15, throughput=0.05,
                                                         data_freshness=0.05, metric_completeness=0.05,
                                                         region_preference=0.05))
    diff_metric = S.rank_backends(_wl(), [_snap("a", two_qubit_error=0.001)])
    diff_block = S.rank_backends(_wl(), [_snap("a")], uncertainty_block=0.9)
    hashes = {base["decision_sha256"], diff_weight["decision_sha256"],
              diff_metric["decision_sha256"], diff_block["decision_sha256"]}
    assert len(hashes) == 4  # every mutation moves the hash


def test_verify_decision_hash_roundtrip_and_tamper():
    r = S.rank_backends(_wl(), [_snap("a"), _snap("b", pending_jobs=40)])
    assert verify_decision_hash(r) is True
    tampered = dict(r)
    tampered["selected"] = "does-not-exist"
    assert verify_decision_hash(tampered) is False
    tampered2 = dict(r)
    tampered2["ranking"] = []  # drop the ranking silently
    assert verify_decision_hash(tampered2) is False


def test_verify_decision_hash_refuses_non_native_payload():
    r = S.rank_backends(_wl(), [_snap("a")])
    poisoned = dict(r)
    poisoned["extra"] = {1, 2, 3}  # a set is not JSON-native
    with pytest.raises(ValueError):
        verify_decision_hash(poisoned)


# ---- C. backend-name uniqueness ------------------------------------------- #
def test_duplicate_backend_names_are_refused():
    with pytest.raises(ValueError):
        S.rank_backends(_wl(), [_snap("dup"), _snap("dup")])
    with pytest.raises(ValueError):  # even with different metrics -> never merged
        S.rank_backends(_wl(), [_snap("dup", two_qubit_error=0.001),
                                _snap("dup", two_qubit_error=0.01)])


def test_name_is_whitespace_stripped_and_case_sensitive():
    assert _snap("  edge  ").name == "edge"
    # different case = different identity (documented policy): not a duplicate
    r = S.rank_backends(_wl(), [_snap("Alpha"), _snap("alpha")])
    assert {row["name"] for row in r["ranking"]} == {"Alpha", "alpha"}


# ---- D. uncertainty_block validation -------------------------------------- #
@pytest.mark.parametrize("block", [float("nan"), float("inf"), -0.1, 1.5, True, "0.5"])
def test_uncertainty_block_is_validated(block):
    with pytest.raises(ValueError):
        S.rank_backends(_wl(), [_snap("a")], uncertainty_block=block)


@pytest.mark.parametrize("block", [0.0, 1.0])
def test_uncertainty_block_accepts_bounds(block):
    r = S.rank_backends(_wl(), [_snap("a")], uncertainty_block=block)
    assert "decision_sha256" in r


# ---- E. hard threshold requires the metric -------------------------------- #
def test_missing_metric_cannot_bypass_a_hard_cap():
    req = _wl(max_two_qubit_error=0.01)
    a = S.assess_backend(req, _snap("m", two_qubit_error=None))
    assert not a.applicable
    assert "two_qubit_error_required_for_threshold" in {r["code"] for r in a.rejection_reasons}


def test_missing_readout_cannot_bypass_readout_cap():
    req = _wl(max_readout_error=0.02)
    a = S.assess_backend(req, _snap("m", readout_error=None))
    assert not a.applicable
    assert "readout_error_required_for_threshold" in {r["code"] for r in a.rejection_reasons}


# ---- F. weighted (contribution-based) explanations ------------------------ #
def test_favourable_is_ranked_by_contribution_not_raw_subscore():
    # region_preference has subscore 1.0 (weight .05); an excellent 2Q backend has a
    # far bigger contribution (~.30). A raw-subscore ranking would wrongly float
    # region to the top; a contribution ranking must not.
    a = S.assess_backend(_wl(preferred_region="us-east"), _snap("b", two_qubit_error=0.0005))
    top = a.explanation["top_favourable"]
    assert top[0]["dimension"] == "two_qubit_quality"
    assert top[0]["contribution"] >= top[-1]["contribution"]
    # every entry exposes the full breakdown
    for e in top:
        assert set(e) == {"dimension", "subscore", "weight", "contribution"}


def test_zero_weight_dimension_never_appears_as_favourable():
    w = ScoringWeights(two_qubit_quality=0.35, readout_quality=0.15, topological_fit=0.15,
                       qubit_margin=0.05, queue_pressure=0.15, throughput=0.05,
                       data_freshness=0.05, metric_completeness=0.05, region_preference=0.0)
    a = S.assess_backend(_wl(preferred_region="us-east"), _snap("b"), weights=w)
    dims = {e["dimension"] for e in a.explanation["top_favourable"]}
    assert "region_preference" not in dims  # zero weight -> zero contribution


def test_contribution_sum_matches_total_score():
    a = S.assess_backend(_wl(), _snap("b"))
    assert math.isclose(a.explanation["contribution_sum"], a.total_score, abs_tol=1e-3)
    assert math.isclose(sum(a.contributions.values()), a.total_score, abs_tol=1e-3)


# ---- G. strict JSON-native serialisation ---------------------------------- #
def test_canonical_is_order_independent():
    assert S._canonical({"a": 1, "b": 2}) == S._canonical({"b": 2, "a": 1})


@pytest.mark.parametrize("bad", [{1, 2}, b"bytes", {1: "x"}, float("nan"), float("inf")])
def test_json_native_refuses_non_native(bad):
    with pytest.raises(ValueError):
        S._canonical({"x": bad})


def test_tuple_serialises_as_list_deterministically():
    assert S._canonical({"t": (1, 2, 3)}) == S._canonical({"t": [1, 2, 3]})


# ---- H. defensive immutability -------------------------------------------- #
def test_snapshot_provenance_is_defensively_copied():
    p = {"src": "live"}
    s = _snap("b", provenance=p)
    p["src"] = "mutated"
    assert s.provenance["src"] == "live"  # external mutation does not leak in


def test_record_provenance_is_defensively_copied():
    prov = {"git_dirty": True}
    r = S.rank_backends(_wl(), [_snap("a")], provenance=prov)
    prov["git_dirty"] = False
    assert r["provenance"]["git_dirty"] is True


def test_identical_calls_are_byte_identical():
    r1 = S.rank_backends(_wl(), [_snap("a"), _snap("b")])
    r2 = S.rank_backends(_wl(), [_snap("b"), _snap("a")])
    assert r1["decision_sha256"] == r2["decision_sha256"]
    assert S._canonical({k: v for k, v in r1.items() if k != "decision_sha256"}) == \
        S._canonical({k: v for k, v in r2.items() if k != "decision_sha256"})


# ---- I. audit: future timestamp + rejected-never-selected ----------------- #
def test_future_timestamp_is_not_silently_fresh():
    a = S.assess_backend(_wl(decision_ts=1000.0), _snap("f", data_ts=5000.0))
    assert a.subscores["data_freshness"] == 0.0
    assert "future_data_timestamp" in a.explanation["warnings"]


def test_rejected_backend_is_never_selected():
    req = _wl(blocked_backends=("hi",))
    r = S.rank_backends(req, [_snap("hi", two_qubit_error=0.0001), _snap("ok")])
    assert r["selected"] == "ok"
    assert "hi" not in {row["name"] for row in r["ranking"]}
    assert "hi" in {row["name"] for row in r["rejected"]}
