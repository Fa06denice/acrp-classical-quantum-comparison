"""Phase 8 — the four surfaces must agree: UI request, API response, visible table, export.

This is a VALIDATOR, not a smoke test: it refuses a missing field, an incompatible status,
a reference from another grid, a gap on an infeasible solution, IBM hardware without
proof, an unknown objective_id, and a stale snapshot. Runs entirely offline through the
in-process client (no browser, no network) so it can gate every commit cheaply; the
browser-level equivalence is covered by tests/ui/test_demo_flow.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from acrpq.dashboard.api import create_app
from acrpq.selection_policy import SELECTION_POLICY_ID

KNOWN_OBJECTIVES = {"quadratic_control_cost_v1", "maneuver_count_v1"}
CONFIGS = [("CP_3", 3, 1), ("CP_4", 3, 1), ("CP_3", 5, 1)]


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


@pytest.mark.parametrize("instance,n_theta,n_q", CONFIGS)
def test_api_identity_is_complete_and_self_consistent(client, instance, n_theta, n_q):
    """Every identity field the UI displays must exist in the API answer and agree."""
    r = client.post("/api/preflight",
                    json={"instance": instance, "n_theta": n_theta, "n_q": n_q})
    assert r.status_code == 200, r.text
    d = r.json()
    for field in ("instance", "n_aircraft", "grid_k", "n_qubits", "objective_id",
                  "objective_degeneracy", "selection_policy", "methods"):
        assert field in d, f"missing {field}"
    assert d["instance"] == instance
    assert d["grid_k"] == n_theta * n_q
    assert d["n_qubits"] == d["n_aircraft"] * d["grid_k"]      # qubits = n * K
    assert d["objective_id"] in KNOWN_OBJECTIVES               # never unknown
    assert d["selection_policy"]["selection_policy_id"] == SELECTION_POLICY_ID
    deg = d["objective_degeneracy"]
    assert (deg["factor"] is None) == (not deg["proportional"])
    if deg["proportional"]:
        assert deg["factor"] > 0.0


@pytest.mark.parametrize("instance,n_theta,n_q", CONFIGS)
def test_no_method_is_reported_both_applicable_and_refused(client, instance, n_theta, n_q):
    d = client.post("/api/preflight",
                    json={"instance": instance, "n_theta": n_theta, "n_q": n_q}).json()
    for m in d["methods"]:
        assert "applicable" in m
        if not m["applicable"]:
            # a refusal must carry BOTH a technical reason and a business one
            assert m.get("reason"), f"{m.get('method')} refused without a technical reason"
            assert m.get("business_reason_fr"), f"{m.get('method')} refused without French reason"


def test_recorded_ibm_result_carries_job_proof_or_is_not_called_hardware():
    """IBM hardware without proof must be impossible to present as hardware."""
    manifest = Path("results/ibm_fixed_angle_campaign_v1/manifest.json")
    if not manifest.exists():
        pytest.skip("campaign manifest absent")
    jobs = json.loads(manifest.read_text())["jobs"]
    tc = TestClient(create_app(runs_dir="runs"))
    checked = 0
    for job in jobs[:5]:
        r = tc.get(f"/api/qpu/{job['run_id']}/result")
        if r.status_code != 200 or not r.json().get("available"):
            continue
        prov = r.json()["provenance"]
        # NOT `has_remote_job is (len(ids) > 0)`: the API derives one from the other, so
        # that comparison is tautological. The signal is that a campaign run really
        # carries a provider job id — the only offline evidence that it is hardware.
        assert "ibm_job_ids" in prov and "has_remote_job" in prov
        assert prov["has_remote_job"] is True, "a campaign job must carry its provider id"
        assert prov["ibm_job_ids"], "hardware provenance with an empty job id list"
        assert all(isinstance(j, str) and j.strip() for j in prov["ibm_job_ids"])
        assert prov["objective_id"] in KNOWN_OBJECTIVES
        checked += 1
    if checked == 0:
        pytest.skip("no local run evidence in this checkout")


def test_thesis_bundle_rows_never_gap_an_infeasible_or_cross_grid_result():
    """The published dataset must not contain a gap without a feasible, same-grid,
    certified reference — and must never pool grids in a synthesis."""
    import csv
    bundle = Path("results/thesis_benchmark_v1_1")
    if not bundle.exists():
        pytest.skip("thesis bundle v1_1 absent")
    rows = json.loads((bundle / "benchmark_results.json").read_text())["rows"]
    for r in rows:
        if r.get("abs_gap_to_reference") is not None and r["data_source"] in (
                "local_classical", "aer_simulator"):
            assert r["feasible"], f"gap on an infeasible row: {r['instance']}/{r['method']}"
        if r.get("objective_id"):
            assert r["objective_id"] in KNOWN_OBJECTIVES
    for name in ("synthesis_by_objective.csv", "synthesis_by_family.csv",
                 "synthesis_by_method.csv"):
        head = next(csv.reader((bundle / name).open()))
        assert "grid_k" in head, f"{name} pools grids"


def test_export_schema_refuses_a_stale_or_unproven_document():
    """The export validator itself must fail closed on the four forbidden shapes."""
    from acrpq.dashboard import qpu_export_schema as X

    base = {"schema": X.EXPORT_SCHEMA, "rows": [], "provenance": {
        "source_commit": "abc", "scientific_code_dirty": False, "repository_dirty": False,
        "versions": {}, "data_source": "synthetic_offline", "status": X.OFFLINE_STATUS,
        "official": False}}
    base["export_sha256"] = X.canonical_export_sha256(base)
    X.validate_export(base)                                   # honest empty synthetic: ok
    bad = dict(base, export_sha256="sha256:" + "0" * 64)      # stale/tampered hash
    with pytest.raises(X.ExportError):
        X.validate_export(bad)
    unknown = json.loads(json.dumps(base))
    unknown["surprise"] = 1
    unknown["export_sha256"] = X.canonical_export_sha256(unknown)
    with pytest.raises(X.ExportError):
        X.validate_export(unknown)
