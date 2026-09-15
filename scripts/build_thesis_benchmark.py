#!/usr/bin/env python3
"""Consolidate the versioned artefacts into the frozen thesis benchmark bundle.

Reuses existing artefacts (no new campaign). One canonical row per
instance × grid × objective × method × seed. Strictly separates data sources
(local_classical / aer_simulator / synthetic_offline / ibm_hardware); synthetic offline
plumbing NEVER enters the scientific aggregates. Non-applicable / timeout / infeasible /
unavailable rows are preserved with explicit reasons; missing results are never fabricated.

Emits results/thesis_benchmark_v1/{benchmark_results.json,benchmark_results.csv,
applicability_matrix.csv,synthesis_by_method.csv,synthesis_by_family.csv,
synthesis_by_objective.csv,manifest.json,README.md} and validates the bundle. --verify
recomputes the manifest hash.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path

BUNDLE_SCHEMA = "acrpq-thesis-benchmark/1.1"   # 1.1: synthesis scoped per grid_k
OUT = Path("results/thesis_benchmark_v1_1")   # v1 is historical: never rewritten
SOURCES = {
    "discrete_benchmark": "results/discrete_benchmark/benchmark.json",
    "discrete_benchmark_multifamily_v1": "results/discrete_benchmark_multifamily_v1/benchmark.json",
    "family_matrix": "results/family_matrix/matrix.json",
    "equivalence_proof": "results/equivalence_proof/proof.json",
    "qpu_batch_offline_v1": "results/qpu_batch_offline_v1/batch.json",
}
_LOCAL = {"gurobi_milp", "reference", "qubo_exact", "qubo_annealing", "dwave_sa"}
_CERTIFIED_REFERENCE = {"optimal", "certified", "certified_optimum"}

CANON_COLUMNS = (
    "data_source", "instance", "family", "grid_k", "n_theta", "n_q", "n_qubits",
    "objective_id", "method", "seed", "result_class", "status", "feasible",
    "decoded_feasible", "raw_onehot_valid", "matches_reference_optimum", "objective_value",
    "reference_optimum", "reference_status", "abs_gap_to_reference", "comparison_tolerance",
    "qubo_energy", "n_conflicts", "is_qpu", "reason",
)
_FORBIDDEN_STRINGS = ("/Users/", "/home/", "/private/", "QISKIT_IBM_TOKEN", "api_token",
                      "Bearer ", "@")


class BundleError(RuntimeError):
    pass


def _data_source(method: str) -> str:
    if method == "qaoa_aer":
        return "aer_simulator"
    if method in _LOCAL:
        return "local_classical"
    raise BundleError(f"unknown method {method!r}")


def _canonical_key(r: dict) -> tuple:
    return (r["data_source"], r["instance"], r["grid_k"], r["objective_id"], r["method"], r["seed"])


def _to_canon(rec: dict) -> dict:
    method = rec["method"]
    ds = _data_source(method)
    feasible = bool(rec.get("decoded_feasible", rec.get("feasible")))
    reason = None
    if rec.get("status") not in (None, "ok", "completed"):
        reason = str(rec.get("status"))
    elif not feasible:
        reason = "infeasible_incumbent"
    return {
        "data_source": ds, "instance": rec["instance"], "family": rec["family"],
        "grid_k": rec["grid_k"], "n_theta": rec["n_theta"], "n_q": rec["n_q"],
        "n_qubits": rec["n_qubits"], "objective_id": rec["objective_id"], "method": method,
        "seed": rec.get("seed"), "result_class": rec.get("result_class"),
        "status": rec.get("status"), "feasible": feasible,
        "decoded_feasible": bool(rec.get("decoded_feasible", feasible)),
        "raw_onehot_valid": rec.get("raw_onehot_valid"),
        "matches_reference_optimum": rec.get("matches_reference_optimum"),
        "objective_value": rec.get("objective_value"), "reference_optimum": rec.get("reference_optimum"),
        "reference_status": rec.get("reference_status"),
        "abs_gap_to_reference": rec.get("abs_gap_to_reference"),
        "comparison_tolerance": rec.get("comparison_tolerance"), "qubo_energy": rec.get("qubo_energy"),
        "n_conflicts": rec.get("n_conflicts"), "is_qpu": bool(rec.get("is_qpu", False)),
        "reason": reason,
    }


def _load_rows() -> list[dict]:
    rows: list[dict] = []
    for key in ("discrete_benchmark", "discrete_benchmark_multifamily_v1"):
        doc = json.loads(Path(SOURCES[key]).read_text())
        for rec in doc.get("results", []):
            rows.append(_to_canon(rec))
    # synthetic offline plumbing rows — SEPARATE, tagged, never in scientific aggregates
    batch = json.loads(Path(SOURCES["qpu_batch_offline_v1"]).read_text())
    for rec in batch.get("results", []):
        rows.append({
            **{c: None for c in CANON_COLUMNS},
            "data_source": "synthetic_offline", "instance": rec.get("instance"),
            "family": (rec.get("instance") or "").split("_")[0], "grid_k": rec.get("n_theta"),
            "n_theta": rec.get("n_theta"), "n_q": rec.get("n_q"), "n_qubits": rec.get("n_qubits"),
            "objective_id": rec.get("objective_id"), "method": "qpu_offline_synthetic",
            "seed": "synthetic", "result_class": "synthetic_plumbing",
            "status": rec.get("final_state"), "feasible": bool(rec.get("decoded_best_feasible")),
            "decoded_feasible": bool(rec.get("decoded_best_feasible")), "is_qpu": False,
            "reason": "synthetic_plumbing_not_quantum_evidence",
        })
    return rows


def _dedup(rows: list[dict]) -> list[dict]:
    seen: dict[tuple, dict] = {}
    for r in rows:
        k = _canonical_key(r)
        if k in seen and seen[k] != r:
            raise BundleError(f"duplicate canonical key with differing values: {k}")
        seen[k] = r
    return list(seen.values())


def _csv(cols, rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for r in rows:
        w.writerow(["" if r.get(c) is None else ("true" if r.get(c) is True else
                    ("false" if r.get(c) is False else r.get(c))) for c in cols])
    return buf.getvalue()


def _assert_parity(cols, rows, csv_text):
    reader = list(csv.reader(io.StringIO(csv_text)))
    if tuple(reader[0]) != tuple(cols):
        raise BundleError("CSV header mismatch")
    if len(reader) - 1 != len(rows):
        raise BundleError("CSV/JSON row count mismatch")
    for i, (r, cells) in enumerate(zip(rows, reader[1:])):
        for c, cell in zip(cols, cells):
            v = r.get(c)
            proj = "" if v is None else ("true" if v is True else ("false" if v is False else str(v)))
            if proj != cell:
                raise BundleError(f"parity row {i} col {c}: {proj!r} != {cell!r}")


def _scientific(rows):
    return [r for r in rows if r["data_source"] in ("local_classical", "aer_simulator")]


def _gap_eligible(r) -> bool:
    return (r["feasible"] and r.get("abs_gap_to_reference") is not None
            and r.get("reference_status") in _CERTIFIED_REFERENCE)


def _synthesis(rows, key_fields):
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for r in _scientific(rows):
        groups[tuple(r[k] for k in key_fields)].append(r)
    out = []
    for gk, rs in sorted(groups.items(), key=lambda x: tuple(str(v) for v in x[0])):
        feas = [r for r in rs if r["feasible"]]
        gap_rows = [r for r in rs if _gap_eligible(r)]
        gaps = [r["abs_gap_to_reference"] for r in gap_rows]
        matches = [r for r in rs if r.get("matches_reference_optimum") is True]
        row = dict(zip(key_fields, gk))
        row.update(n_rows=len(rs), n_feasible=len(feas),
                   feasibility_rate=round(len(feas) / len(rs), 4) if rs else None,
                   reference_match_rate=round(len(matches) / len(rs), 4) if rs else None,
                   n_gap_eligible=len(gap_rows),
                   mean_abs_gap=(round(sum(gaps) / len(gaps), 6) if gaps else None),
                   max_abs_gap=(round(max(gaps), 6) if gaps else None))
        out.append(row)
    return out


def _no_nan_secrets_paths(obj, where=""):
    if isinstance(obj, float) and not math.isfinite(obj):
        raise BundleError(f"non-finite at {where}")
    if isinstance(obj, str):
        for bad in _FORBIDDEN_STRINGS:
            if bad in obj:
                raise BundleError(f"forbidden string {bad!r} at {where}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _no_nan_secrets_paths(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _no_nan_secrets_paths(v, f"{where}[{i}]")


def _sha_file(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def _provenance() -> dict:
    from dataclasses import asdict
    from acrpq.benchmark.export import ReproInfo
    from acrpq.classical.ampl_campaign import git_info
    repro = asdict(ReproInfo.capture())
    git = git_info()
    return {"source_commit": git["git_commit"], "scientific_code_dirty": git["scientific_code_dirty"],
            "repository_dirty": git["repository_dirty"], "python_version": repro["python_version"],
            "package_version": repro["package_version"],
            "versions": {k: repro["deps"].get(k) for k in ("qiskit", "numpy")},
            "source_artifacts": {k: _sha_file(Path(v)) for k, v in SOURCES.items()}}


def build() -> dict:
    rows = _dedup(_load_rows())
    rows.sort(key=lambda r: (r["data_source"], r["objective_id"], r["family"], r["instance"],
                             r["grid_k"], r["method"], str(r["seed"])))
    for r in rows:
        _no_nan_secrets_paths(r, "row")
    results_csv = _csv(CANON_COLUMNS, rows)
    _assert_parity(CANON_COLUMNS, rows, results_csv)

    # objective/grid/reference consistency + no official gap without certified same-grid ref
    for r in _scientific(rows):
        if r.get("abs_gap_to_reference") is not None and not _gap_eligible(r):
            # a gap is present but not from a certified same-grid feasible comparison
            if r["feasible"] and r.get("reference_status") not in _CERTIFIED_REFERENCE:
                raise BundleError(f"gap without a certified reference: {_canonical_key(r)}")

    # grid_k is part of EVERY synthesis key: a gap is only meaningful against a
    # same-grid reference, so pooling K=3/5/7 into one mean would compare quantities
    # defined on different feasible sets (MoE Expert B, CRITICAL).
    syn_method = _synthesis(rows, ("data_source", "objective_id", "grid_k", "method"))
    syn_family = _synthesis(rows, ("family", "objective_id", "grid_k"))
    syn_obj = _synthesis(rows, ("objective_id", "grid_k"))
    m_cols = ("data_source", "objective_id", "grid_k", "method", "n_rows", "n_feasible",
              "feasibility_rate", "reference_match_rate", "n_gap_eligible", "mean_abs_gap",
              "max_abs_gap")
    f_cols = ("family", "objective_id", "grid_k", "n_rows", "n_feasible", "feasibility_rate",
              "reference_match_rate", "n_gap_eligible", "mean_abs_gap", "max_abs_gap")
    o_cols = ("objective_id", "grid_k", "n_rows", "n_feasible", "feasibility_rate",
              "reference_match_rate", "n_gap_eligible", "mean_abs_gap", "max_abs_gap")

    # applicability matrix (flatten family_matrix per method)
    mm = json.loads(Path(SOURCES["family_matrix"]).read_text())
    appl_rows = []
    for cell in mm.get("rows", []):
        for method, info in (cell.get("methods") or {}).items():
            appl_rows.append({"instance": cell["instance"], "family": cell["family"],
                              "grid_k": cell["grid_k"], "n_qubits": cell.get("n_vars"),
                              "method": method, "applicable": info.get("applicable"),
                              "resolution": info.get("resolution"), "reason": info.get("reason")})
    a_cols = ("instance", "family", "grid_k", "n_qubits", "method", "applicable", "resolution", "reason")

    manifest = {"schema": BUNDLE_SCHEMA, "n_canonical_rows": len(rows),
                "data_source_counts": {ds: sum(1 for r in rows if r["data_source"] == ds)
                                       for ds in ("local_classical", "aer_simulator",
                                                  "synthetic_offline", "ibm_hardware")},
                "provenance": _provenance(),
                "files": {}, "note": "synthetic_offline rows are plumbing validation, NOT quantum "
                                     "evidence, and are excluded from all scientific aggregates",
                "caveats": [
                    "every synthesis row is scoped to a single grid_k: gaps are defined "
                    "against a same-grid certified reference and must never be pooled "
                    "across K (the feasible set changes with the grid)",
                    "a gap is reported only for a feasible result against a certified "
                    "same-grid reference; infeasible rows are excluded from gap aggregates",
                    "the two objectives are never aggregated together",
                ],
                "claims": {"grids_aggregated_together": False,
                           "objectives_aggregated_together": False}}
    files = {
        "benchmark_results.json": json.dumps({"schema": BUNDLE_SCHEMA, "rows": rows},
                                             indent=2, sort_keys=True, allow_nan=False),
        "benchmark_results.csv": results_csv,
        "applicability_matrix.csv": _csv(a_cols, appl_rows),
        "synthesis_by_method.csv": _csv(m_cols, syn_method),
        "synthesis_by_family.csv": _csv(f_cols, syn_family),
        "synthesis_by_objective.csv": _csv(o_cols, syn_obj),
    }
    manifest["files"] = {name: "sha256:" + hashlib.sha256(txt.encode()).hexdigest()
                         for name, txt in files.items()}
    manifest["manifest_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps({k: v for k, v in manifest.items() if k != "manifest_sha256"},
                   sort_keys=True).encode()).hexdigest()
    _no_nan_secrets_paths(manifest, "manifest")
    files["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
    return {"manifest": manifest, "files": files}


_README = """# Thesis benchmark bundle (results/thesis_benchmark_v1)

Frozen consolidation of the versioned artefacts. One canonical row per
instance x grid x objective x method x seed. Data sources are strictly separated:
`local_classical`, `aer_simulator`, `synthetic_offline`, `ibm_hardware` (none yet).

**synthetic_offline rows are plumbing validation, NOT quantum evidence, and are excluded
from every scientific aggregate.** No real IBM hardware result exists yet.

Files: benchmark_results.json/csv (canonical rows, full parity), applicability_matrix.csv
(per-method applicability + reason), synthesis_by_method/family/objective.csv (aggregates
over scientific rows only; a gap is reported ONLY for a feasible result against a certified
same-grid reference; infeasible rows are excluded from gap aggregates), manifest.json
(schema, counts, provenance, source-artefact + file hashes).

Objectives (`quadratic_control_cost_v1`, `maneuver_count_v1`) are never aggregated together.
An incumbent is never labelled an optimum. No quantum speed-up is claimed. Timings live in
the source artefacts and are excluded from the reproducible hashes.
"""


def _write(bundle: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, txt in bundle["files"].items():
        (OUT / name).write_text(txt)
    (OUT / "README.md").write_text(_README)


def _verify() -> int:
    manifest = json.loads((OUT / "manifest.json").read_text())
    ok = True
    for name, h in manifest["files"].items():
        got = "sha256:" + hashlib.sha256((OUT / name).read_bytes()).hexdigest()
        if got != h:
            print(f"TAMPERED: {name}")
            ok = False
    recon = "sha256:" + hashlib.sha256(json.dumps(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}, sort_keys=True).encode()
    ).hexdigest()
    ok = ok and recon == manifest["manifest_sha256"]
    print("OK manifest verified" if ok else "MANIFEST TAMPERED")
    return 0 if ok else 2


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        return _verify()
    bundle = build()
    _write(bundle)
    print(f"[thesis] {bundle['manifest']['n_canonical_rows']} canonical rows -> {OUT}")
    print(f"  data_source_counts: {bundle['manifest']['data_source_counts']}")
    print(f"  manifest_sha256={bundle['manifest']['manifest_sha256'][:24]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
