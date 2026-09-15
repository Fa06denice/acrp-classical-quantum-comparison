"""Validate the thesis benchmark bundle builder (Codex freeze deliverable)."""
from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path

_P = Path(__file__).resolve().parent.parent / "scripts" / "build_thesis_benchmark.py"
_spec = importlib.util.spec_from_file_location("build_thesis_benchmark", _P)
tb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tb)


def test_bundle_builds_and_is_internally_consistent():
    bundle = tb.build()                                  # raises on dup key / parity / secret / NaN
    m = bundle["manifest"]
    assert m["schema"] == tb.BUNDLE_SCHEMA and m["n_canonical_rows"] > 0
    assert m["data_source_counts"]["ibm_hardware"] == 0   # no hardware yet


def test_no_duplicate_canonical_key():
    rows = json.loads(bundle_json())["rows"]
    keys = [(r["data_source"], r["instance"], r["grid_k"], r["objective_id"], r["method"], r["seed"])
            for r in rows]
    assert len(keys) == len(set(keys))


def bundle_json():
    return tb.build()["files"]["benchmark_results.json"]


def test_full_json_csv_parity():
    files = tb.build()["files"]
    rows = json.loads(files["benchmark_results.json"])["rows"]
    tb._assert_parity(tb.CANON_COLUMNS, rows, files["benchmark_results.csv"])   # no raise


def test_synthetic_offline_excluded_from_scientific_aggregates():
    files = tb.build()["files"]
    for name in ("synthesis_by_method.csv", "synthesis_by_family.csv", "synthesis_by_objective.csv"):
        assert "synthetic" not in files[name] and "qpu_offline" not in files[name]
    # but the synthetic rows ARE preserved in the canonical results, tagged
    rows = json.loads(files["benchmark_results.json"])["rows"]
    synth = [r for r in rows if r["data_source"] == "synthetic_offline"]
    assert synth and all(r["reason"] == "synthetic_plumbing_not_quantum_evidence" for r in synth)


def test_objectives_never_mixed_in_synthesis():
    files = tb.build()["files"]
    reader = list(csv.reader(io.StringIO(files["synthesis_by_objective.csv"])))
    head, rows = reader[0], reader[1:]
    objs = {row[0] for row in rows}
    assert objs == {"maneuver_count_v1", "quadratic_control_cost_v1"}   # never pooled together
    # ...and each row is scoped to ONE grid, so no (objective, K) pair repeats
    assert head[0] == "objective_id" and head[1] == "grid_k"
    keys = [(row[0], row[1]) for row in rows]
    assert len(keys) == len(set(keys))


def test_synthesis_never_pools_across_grids():
    """MoE Expert B, CRITICAL: gaps are defined against a same-grid certified reference,
    so a mean pooled over K=3/5/7 compares quantities on different feasible sets. Every
    synthesis table must therefore carry grid_k in its key."""
    files = tb.build()["files"]
    for name in ("synthesis_by_objective.csv", "synthesis_by_family.csv",
                 "synthesis_by_method.csv"):
        reader = list(csv.reader(io.StringIO(files[name])))
        assert "grid_k" in reader[0], f"{name} lost its grid_k scoping"
    # the manifest must state the claim explicitly
    manifest = json.loads(files["manifest.json"])
    assert manifest["claims"]["grids_aggregated_together"] is False
    assert manifest["claims"]["objectives_aggregated_together"] is False


def test_gap_only_from_certified_feasible_reference():
    rows = json.loads(bundle_json())["rows"]
    # every scientific row that contributed a gap must be feasible with a certified reference
    for r in rows:
        if r["data_source"] in ("local_classical", "aer_simulator") and tb._gap_eligible(r):
            assert r["feasible"] and r["reference_status"] in tb._CERTIFIED_REFERENCE


def test_no_secrets_paths_or_nonfinite():
    files = tb.build()["files"]
    blob = "".join(files.values())
    for bad in ("/Users/", "/home/", "QISKIT_IBM_TOKEN", "Bearer ", "api_token", "NaN", "Infinity"):
        assert bad not in blob
