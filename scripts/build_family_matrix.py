#!/usr/bin/env python3
"""Emit the CP/RCP/FP/GP applicability matrix (Phase 3A).

Pure metadata (no solver runs), so the artifact is fully deterministic. Writes
JSON + CSV (parity-checked) with provenance and a content hash.

Run:    python scripts/build_family_matrix.py [--out results/family_matrix]
        python scripts/build_family_matrix.py --verify results/family_matrix/matrix.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from acrpq.benchmark.family_matrix import (
    METHOD_KEYS as _METHODS,
    assert_json_csv_parity,
    build_matrix,
    canonical_matrix_sha256,
    rows_to_csv,
)
from acrpq.io.loader import InstanceLoader

_HARNESS_FILES = ["src/acrpq/benchmark/family_matrix.py", "scripts/build_family_matrix.py"]


def _provenance() -> dict:
    from acrpq.benchmark.export import ReproInfo
    from acrpq.classical.ampl_campaign import git_info
    from dataclasses import asdict
    repro = asdict(ReproInfo.capture())
    git = git_info()
    h = hashlib.sha256()
    for f in _HARNESS_FILES:
        h.update(Path(f).read_bytes())
    return {
        "git_commit": git["git_commit"], "source_commit": git["git_commit"],
        "generated_artifact_commit": None,
        "scientific_code_dirty": git["scientific_code_dirty"],
        "repository_dirty": git["repository_dirty"],
        "python_version": repro["python_version"], "package_version": repro["package_version"],
        "versions": {"amplpy": repro["deps"].get("amplpy"), "numpy": repro["deps"].get("numpy")},
        "harness_files": list(_HARNESS_FILES), "harness_sha256": h.hexdigest(),
    }


def _verify(path: str) -> int:
    m = json.loads(Path(path).read_text())
    ok = m.get("matrix_sha256") == canonical_matrix_sha256(m)
    print(f"{'OK matrix_sha256 verified' if ok else 'TAMPERED'}: {m.get('matrix_sha256')}")
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/family_matrix")
    ap.add_argument("--verify", metavar="MATRIX_JSON")
    ap.add_argument("--rcp-fl-limit", type=int, default=90,
                    help="cap on the large RCP_FL family (n up to 150 is ~1s/cell)")
    args = ap.parse_args()
    if args.verify:
        return _verify(args.verify)

    ld = InstanceLoader()
    # CP/FP/GP/RCP covered in full; only the very large RCP_FL family is sampled
    # (documented in coverage). RCP_FL is homogeneous random-circle-with-levels.
    limits = {"RCP_FL": args.rcp_fl_limit}
    rows, coverage = build_matrix(ld, per_family_limit=limits)
    matrix = {
        "schema": "acrpq-family-matrix/1",
        "methods": _METHODS,
        "protocol": {"k_all": 3, "k_sensitivity": [5, 7], "sensitivity_max_aircraft": 8,
                     "per_family_limit": limits},
        "coverage": coverage,
        "n_rows": len(rows),
        "provenance": _provenance(),
        "rows": rows,
    }
    matrix["matrix_sha256"] = canonical_matrix_sha256(matrix)
    matrix["matrix_sha256_note"] = "deterministic sha over the full matrix content"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "matrix.json").write_text(json.dumps(matrix, indent=2, sort_keys=True, allow_nan=False))
    csv_text = rows_to_csv(rows)
    (out / "matrix.csv").write_text(csv_text)
    assert_json_csv_parity(rows, csv_text)  # fail-closed JSON/CSV parity

    for c in coverage:
        note = "" if c["dropped"] == 0 else f"  (DROPPED {c['dropped']} — capped)"
        print(f"  {c['family']:7} covered {c['covered']}/{c['available']}{note}")
    print(f"\n{len(rows)} rows -> {out/'matrix.json'} (+ .csv, parity OK)  "
          f"sha256={matrix['matrix_sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
