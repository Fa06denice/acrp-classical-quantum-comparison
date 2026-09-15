"""Explainable backend scoring — pure, deterministic, no IBM/network/qiskit."""

from __future__ import annotations


import pytest

from acrpq.dashboard import backend_scoring as S
from acrpq.dashboard.backend_scoring import (
    BackendSnapshot,
    ScoringWeights,
    WorkloadRequirements,
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


# --- model validation ------------------------------------------------------
def test_models_reject_nonfinite_and_bad_values():
    with pytest.raises(ValueError):
        _snap("x", two_qubit_error=float("inf"))
    with pytest.raises(ValueError):
        _snap("x", programmable_qubits=-1)
    with pytest.raises(ValueError):
        _wl(logical_qubits=0)
    with pytest.raises(ValueError):
        _wl(interaction_density=1.5)
    with pytest.raises(ValueError):
        ScoringWeights(two_qubit_quality=0.9)  # weights don't sum to 1
    with pytest.raises(ValueError):
        ScoringWeights(two_qubit_quality=-0.3, readout_quality=0.45)  # negative weight


# --- hard filters ----------------------------------------------------------
@pytest.mark.parametrize("over,code", [
    ({"operational": False}, "offline"),
    ({"maintenance": True}, "maintenance"),
    ({"programmable_qubits": 3}, "insufficient_qubits"),
    ({"region": "eu-west"}, "region_not_allowed"),
    ({"two_qubit_error": 0.5}, "two_qubit_error_too_high"),
])
def test_hard_filters_reject_with_stable_code(over, code):
    req = _wl(allowed_regions=("us-east",), max_two_qubit_error=0.1)
    a = S.assess_backend(req, _snap("b", **over))
    assert a.applicable is False
    assert code in {r["code"] for r in a.rejection_reasons}
    assert a.total_score is None  # never scored


def test_blocked_and_feature_filters():
    req = _wl(blocked_backends=("bad",), needs_dynamic_circuits=True)
    assert not S.assess_backend(req, _snap("bad", supports_dynamic=True)).applicable
    assert not S.assess_backend(req, _snap("nodyn", supports_dynamic=False)).applicable


def test_require_all_metrics_policy_rejects_missing():
    req = _wl(require_all_metrics=True)
    a = S.assess_backend(req, _snap("m", two_qubit_error=None))
    assert not a.applicable and "missing_required_metrics" in {r["code"] for r in a.rejection_reasons}


# --- bounded + monotone ----------------------------------------------------
def test_all_subscores_and_total_are_bounded():
    a = S.assess_backend(_wl(), _snap("b"))
    assert 0.0 <= a.total_score <= 1.0
    assert all(0.0 <= v <= 1.0 for v in a.subscores.values())


def test_lower_two_qubit_error_never_lowers_quality():
    req = _wl()
    good = S.assess_backend(req, _snap("g", two_qubit_error=0.001))
    bad = S.assess_backend(req, _snap("b", two_qubit_error=0.015))
    assert good.subscores["two_qubit_quality"] >= bad.subscores["two_qubit_quality"]
    assert good.total_score >= bad.total_score


def test_lower_queue_never_lowers_operational():
    req = _wl()
    empty = S.assess_backend(req, _snap("e", pending_jobs=0))
    busy = S.assess_backend(req, _snap("b", pending_jobs=180))
    assert empty.subscores["queue_pressure"] >= busy.subscores["queue_pressure"]
    assert empty.operational_score >= busy.operational_score


def test_older_data_never_scores_fresher():
    req = _wl()
    fresh = S.assess_backend(req, _snap("f", data_ts=_NOW - 60.0))
    stale = S.assess_backend(req, _snap("s", data_ts=_NOW - 20 * 3600.0))
    assert fresh.subscores["data_freshness"] >= stale.subscores["data_freshness"]


def test_missing_metric_never_beats_known_excellent():
    req = _wl()
    known = S.assess_backend(req, _snap("known", two_qubit_error=0.0005))
    missing = S.assess_backend(req, _snap("missing", two_qubit_error=None))
    assert known.subscores["two_qubit_quality"] > missing.subscores["two_qubit_quality"]
    assert "two_qubit_quality" in missing.metrics_missing
    assert missing.uncertainty > known.uncertainty


# --- determinism, tie-break, provenance ------------------------------------
def test_ranking_is_byte_deterministic_with_hash():
    req = _wl()
    snaps = [_snap("b-two", pending_jobs=5), _snap("a-one", pending_jobs=50)]
    r1 = S.rank_backends(req, snaps)
    r2 = S.rank_backends(req, list(reversed(snaps)))
    assert r1["decision_sha256"] == r2["decision_sha256"]  # order-independent, stable
    assert r1["decision_sha256"].startswith("sha256:")


def test_perfect_tie_breaks_by_name():
    req = _wl()
    a = _snap("alpha")
    b = _snap("bravo")  # identical metrics
    r = S.rank_backends(req, [b, a])
    order = [row["name"] for row in r["ranking"]]
    assert order == ["alpha", "bravo"]  # equal score -> name asc
    assert r["ranking"][0]["total_score"] == r["ranking"][1]["total_score"]


def test_provenance_attached_when_supplied():
    r = S.rank_backends(_wl(), [_snap("b")], provenance={"git_dirty": True, "schema": "x"})
    assert r["provenance"]["git_dirty"] is True


def test_injected_timestamp_drives_freshness():
    snap = _snap("b", data_ts=500.0)
    young = S.assess_backend(_wl(decision_ts=1000.0), snap)
    old = S.assess_backend(_wl(decision_ts=500.0 + 30 * 3600.0), snap)
    assert young.subscores["data_freshness"] > old.subscores["data_freshness"]


# --- selection safety ------------------------------------------------------
def test_zero_backends_selects_nothing():
    r = S.rank_backends(_wl(), [])
    assert r["selected"] is None and r["selection_note"] == "no_applicable_backend"
    assert r["ranking"] == []


def test_no_applicable_backend_selects_nothing():
    req = _wl(logical_qubits=200)  # nothing has 200 programmable qubits
    r = S.rank_backends(req, [_snap("a"), _snap("b")])
    assert r["selected"] is None and r["selection_note"] == "no_applicable_backend"
    assert len(r["rejected"]) == 2


def test_excessive_uncertainty_withholds_selection():
    # a backend missing every optional metric + stale -> high uncertainty
    req = _wl()
    blind = _snap("blind", two_qubit_error=None, readout_error=None,
                  avg_coupling_degree=None, pending_jobs=None, clops=None,
                  data_ts=_NOW - 40 * 3600.0)
    r = S.rank_backends(req, [blind], uncertainty_block=0.5)
    assert r["selected"] is None and "uncertainty" in r["selection_note"]
    assert r["ranking"][0]["name"] == "blind"  # still ranked + explained


# --- explainability + compare ----------------------------------------------
def test_explanation_has_actionable_structure():
    a = S.assess_backend(_wl(), _snap("b", two_qubit_error=None))
    e = a.explanation
    assert e["compatible"] is True
    assert len(e["top_favourable"]) <= 3 and len(e["top_penalties"]) <= 3
    assert "two_qubit_quality" in e["missing_metrics"]
    assert "not a probability" in e["heuristic_warning"].lower()
    assert e["weights"]


def test_compare_explains_ordering():
    req = _wl()
    a = S.assess_backend(req, _snap("a", two_qubit_error=0.001, pending_jobs=5))
    b = S.assess_backend(req, _snap("b", two_qubit_error=0.015, pending_jobs=150))
    c = S.compare(a, b)
    assert c["ahead"] == "a" and c["total_delta"] > 0
    assert len(c["top_dimension_deltas"]) == 3


# --- topology + post-transpilation hook ------------------------------------
def test_topological_fit_rewards_higher_coupling_degree():
    req = _wl(interaction_density=0.8)
    good = S.assess_backend(req, _snap("g", avg_coupling_degree=4.0))
    poor = S.assess_backend(req, _snap("p", avg_coupling_degree=1.0))
    assert good.subscores["topological_fit"] >= poor.subscores["topological_fit"]


def test_post_transpilation_hook_records_without_rewriting_score():
    a = S.assess_backend(_wl(), _snap("b"))
    refined = S.refine_with_transpilation(
        a, S.TranspilationMetrics(isa_depth=120, two_qubit_gate_count=88, swap_count=14, layout_ok=True)
    )
    assert refined.total_score == a.total_score  # heuristic score unchanged
    assert refined.explanation["post_transpilation"]["swap_count"] == 14


# --- eight synthetic backends + realistic mix ------------------------------
def test_eight_synthetic_backends_rank_sensibly():
    req = _wl(logical_qubits=5, preferred_region="us-east")
    snaps = [
        _snap("hi-fidelity-busy", two_qubit_error=0.001, pending_jobs=190),
        _snap("mid-available", two_qubit_error=0.008, pending_jobs=2),
        _snap("too-small", programmable_qubits=3),
        _snap("in-maintenance", maintenance=True),
        _snap("missing-metrics", two_qubit_error=None, readout_error=None),
        _snap("stale", data_ts=_NOW - 48 * 3600.0),
        _snap("bad-topology", avg_coupling_degree=0.5),
        _snap("eu-region", region="eu-west"),
    ]
    r = S.rank_backends(req, snaps)
    names_ranked = {row["name"] for row in r["ranking"]}
    assert "too-small" not in names_ranked and "in-maintenance" not in names_ranked
    assert {"too-small", "in-maintenance"} <= {row["name"] for row in r["rejected"]}
    assert r["selected"] in names_ranked
    # every applicable row is fully explained
    for row in r["ranking"]:
        assert 0.0 <= row["total_score"] <= 1.0 and row["explanation"]["heuristic_warning"]


# --- no submission path -----------------------------------------------------
def test_scorer_has_no_submission_or_gateway_path():
    import inspect

    src = inspect.getsource(S)
    for forbidden in ("run_sampler", "QiskitRuntimeService", "submit_run", "record_job_id"):
        assert forbidden not in src  # scoring never touches submission/token paths


def test_scorer_import_is_qiskit_free():
    # a fresh interpreter: importing the scorer alone must not pull in qiskit
    # (checking this process's sys.modules is unreliable — other tests load qiskit)
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, acrpq.dashboard.backend_scoring as m; "
         "print(any(k.startswith('qiskit') for k in sys.modules))"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "False"
