#!/usr/bin/env python3
"""Compare le paysage AVANT et APRÈS la correction du scoring des références.

Condition Codex : les angles, les hashes de QUBO, les masses faisables et les masses
one-hot ne doivent PAS être supposés inchangés — ils doivent être vérifiés. Ce script
échoue (code 3) si l'un d'eux bouge, ce qui impose un arrêt et une nouvelle contre-review.

Ce qui DOIT changer : la référence certifiée, le nombre d'optima, la masse sur l'optimum
certifié et l'objectif atteignable des cellules dont l'objective_id n'est pas le coût
quadratique.

Usage : python scripts/verify_landscape_correction.py AVANT.json APRES.json [--emit]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

OUT = Path("results/offline_angle_landscape_v2/correction_verification.json")

# Invariants : tout changement ici est un ARRÊT.
INVARIANT_ARM_FIELDS = ("gamma", "beta", "feasible_mass", "onehot_valid_mass",
                        "expected_energy", "conditional_geometry_feasibility")
INVARIANT_CELL_FIELDS = ("qubo_sha256", "n_qubits", "n_states", "n_onehot_states",
                         "n_feasible_states", "penalty_conflict")
# Champs dont la correction est ATTENDUE.
EXPECTED_TO_CHANGE_CELL = ("certified_reference_objective", "n_certified_optima")
EXPECTED_TO_CHANGE_ARM = ("certified_optimum_mass", "best_reachable_objective",
                          "certified_optimum_reachable")
ARMS = ("P0", "P1_minH", "P1_feas")


def _sha(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _key(row: dict) -> str:
    return f"{row['instance']}|K{row['grid_k']}|{row['objective_id']}"


def compare(before: dict, after: dict) -> dict[str, Any]:
    b = {_key(r): r for r in before["rows"]}
    a = {_key(r): r for r in after["rows"]}
    violations: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    invariants_checked = 0

    if set(b) != set(a):
        violations.append({"cell": "*", "field": "cell_set",
                           "before": sorted(b), "after": sorted(a)})

    for cell in sorted(set(b) & set(a)):
        rb, ra = b[cell], a[cell]
        for f in INVARIANT_CELL_FIELDS:
            invariants_checked += 1
            if rb.get(f) != ra.get(f):
                violations.append({"cell": cell, "field": f,
                                   "before": rb.get(f), "after": ra.get(f)})
        for f in EXPECTED_TO_CHANGE_CELL:
            if rb.get(f) != ra.get(f):
                corrections.append({"cell": cell, "field": f,
                                    "before": rb.get(f), "after": ra.get(f)})
        for arm in ARMS:
            ab, aa = rb[arm], ra[arm]
            for f in INVARIANT_ARM_FIELDS:
                invariants_checked += 1
                # comparaison BIT À BIT : pas de tolérance, ce sont des invariants
                if ab.get(f) != aa.get(f):
                    violations.append({"cell": cell, "arm": arm, "field": f,
                                       "before": ab.get(f), "after": aa.get(f)})
            for f in EXPECTED_TO_CHANGE_ARM:
                if ab.get(f) != aa.get(f):
                    corrections.append({"cell": cell, "arm": arm, "field": f,
                                        "before": ab.get(f), "after": aa.get(f)})

    doc = {
        "schema": "acrpq-landscape-correction-verification/1",
        "before_content_sha256": before["content_sha256"],
        "after_content_sha256": after["content_sha256"],
        "invariant_fields_checked": invariants_checked,
        "invariant_violations": violations,
        "angles_bit_identical": not any(
            v.get("field") in ("gamma", "beta") for v in violations),
        "qubo_hashes_identical": not any(
            v.get("field") == "qubo_sha256" for v in violations),
        "feasible_masses_identical": not any(
            v.get("field") == "feasible_mass" for v in violations),
        "onehot_masses_identical": not any(
            v.get("field") == "onehot_valid_mass" for v in violations),
        "n_corrections": len(corrections),
        "corrections": corrections,
        "files_changed": ["results/offline_angle_landscape_v2/angle_landscape.json"],
        "fields_changed": sorted({c["field"] for c in corrections}),
        "verdict": "SAFE_METADATA_ONLY" if not violations else "STOP_INVARIANT_VIOLATED",
    }
    doc["content_sha256"] = _sha({k: v for k, v in doc.items() if k != "content_sha256"})
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--emit", action="store_true")
    args = ap.parse_args()
    doc = compare(json.loads(Path(args.before).read_text()),
                  json.loads(Path(args.after).read_text()))
    print(f"invariants vérifiés : {doc['invariant_fields_checked']}")
    print(f"  angles bit-identiques      : {doc['angles_bit_identical']}")
    print(f"  hashes QUBO identiques     : {doc['qubo_hashes_identical']}")
    print(f"  masses faisables identiques: {doc['feasible_masses_identical']}")
    print(f"  masses one-hot identiques  : {doc['onehot_masses_identical']}")
    print(f"corrections attendues appliquées : {doc['n_corrections']} "
          f"sur les champs {doc['fields_changed']}")
    if doc["invariant_violations"]:
        print(f"\nARRÊT — {len(doc['invariant_violations'])} violation(s) d'invariant :")
        for v in doc["invariant_violations"][:10]:
            print(f"  {v}")
        return 3
    if args.emit:
        OUT.write_text(json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(f"\n-> {OUT} ({doc['content_sha256'][:26]}…)")
    print(f"\nverdict : {doc['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
