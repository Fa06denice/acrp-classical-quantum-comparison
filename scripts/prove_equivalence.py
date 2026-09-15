#!/usr/bin/env python3
"""Emit the machine-readable three-way discrete equivalence proof (Phase 2D / 2.1).

For every applicable small instance x grid x objective, prove
    exhaustive == Gurobi discrete MILP (certified) == QUBO global minimum
where the QUBO global minimum is established by full 2^(nK) enumeration on small
configs and by the formally-validated one-hot reduction theorem otherwise.

Writes a strict, deterministic JSON artifact (no NaN/Inf) with full provenance
and a hashed manifest. HARD STOP: exits non-zero on any divergence.

Run:  python scripts/prove_equivalence.py [--out results/equivalence_proof]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from acrpq.benchmark.equivalence import FULL_BINARY_CAP, prove_equivalence
from acrpq.io.loader import InstanceLoader
from acrpq.model import ManeuverGrid

CASES = [
    ("CP_3", 3), ("CP_4", 3), ("CP_5", 3), ("CP_6", 3), ("CP_7", 3), ("CP_8", 3),
    ("CP_3", 5), ("CP_4", 5), ("CP_5", 5),
    ("CP_3", 7),
]
OBJECTIVES = ["quadratic_control_cost_v1", "maneuver_count_v1"]

_HARNESS_FILES = [
    "src/acrpq/benchmark/equivalence.py",
    "src/acrpq/classical/discrete_enum.py",
    "src/acrpq/classical/discrete_milp.py",
    "src/acrpq/quantum/qubo.py",
    "src/acrpq/objectives.py",
    "scripts/prove_equivalence.py",
]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_proof_sha256(proof: dict) -> str:
    """Deterministic sha over the proof's SCIENTIFIC content.

    Strips (a) the sha fields themselves and (b) wall-clock ``solve_time_s``
    (the only nondeterministic field, kept in the artifact for transparency),
    then hashes the canonical JSON. Generation and ``--verify`` share this
    function so a stored hash can be recomputed and any tampering refused.
    """
    def strip(obj):
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items()
                    if k not in ("solve_time_s", "proof_sha256", "proof_sha256_note")}
        if isinstance(obj, list):
            return [strip(v) for v in obj]
        return obj

    blob = json.dumps(strip(proof), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def verify_proof(path: str) -> int:
    """Recompute proof_sha256 from a JSON artifact and refuse any alteration."""
    proof = json.loads(Path(path).read_text())
    stored = proof.get("proof_sha256")
    recomputed = canonical_proof_sha256(proof)
    if stored == recomputed:
        print(f"OK proof_sha256 verified: {recomputed}")
        return 0
    print(f"TAMPERED: stored {stored} != recomputed {recomputed}")
    return 2


def _grid_identity(inst, k: int) -> str:
    grid = ManeuverGrid.build(n_theta=k, n_q=1, instance=inst)
    canon = json.dumps({
        "q_levels": list(grid.q_levels), "theta_levels": list(grid.theta_levels),
        "noop_index": grid.noop_index(),
        "options": [[o.idx, o.q, o.theta, o.label] for o in grid.options],
    }, sort_keys=True, allow_nan=False)
    return _sha256_text(canon)


def _provenance(results, ld) -> dict:
    from acrpq.benchmark.export import ReproInfo
    from acrpq.classical.ampl_campaign import git_info

    repro = asdict(ReproInfo.capture())
    git = git_info()
    h = hashlib.sha256()
    for f in _HARNESS_FILES:  # order fixed above -> deterministic
        h.update(Path(f).read_bytes())
    gurobi_ver = ""
    for r in results:
        gv = r["legs"].get("gurobi_milp", {}).get("detail", {}).get("solver_version")
        if gv:
            gurobi_ver = gv
            break
    return {
        "git_commit": git["git_commit"],
        # the exact code commit this proof was GENERATED from (the scientific
        # source). The artifact is stored in a LATER, separate commit — that
        # storing commit is never the scientific source (see the repo git log /
        # the commit that adds this file). Left null here (cannot self-reference).
        "source_commit": git["git_commit"],
        "generated_artifact_commit": None,
        "scientific_code_dirty": git["scientific_code_dirty"],
        "repository_dirty": git["repository_dirty"],
        "scientific_code_paths": git["scientific_code_paths"],
        "package_version": repro["package_version"],
        "python_version": repro["python_version"],
        "platform": repro["platform"],
        "versions": {
            "amplpy": repro["deps"].get("amplpy"),
            "gurobi": gurobi_ver,
            "qiskit": repro["deps"].get("qiskit"),
            "numpy": repro["deps"].get("numpy"),
        },
        "reproinfo_source_hash": repro["source_hash"],
        "harness_files": list(_HARNESS_FILES),
        "harness_sha256": h.hexdigest(),
        "tolerance_policy": {
            "maneuver_count_v1": "exact (abs_tol=0, integer cardinality)",
            "quadratic_control_cost_v1": "abs_tol=1e-9 (float grouping + Gurobi gap)",
        },
        "full_binary_cap": FULL_BINARY_CAP,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/equivalence_proof")
    ap.add_argument("--verify", metavar="PROOF_JSON",
                    help="recompute proof_sha256 from an existing artifact and exit")
    args = ap.parse_args()
    if args.verify:
        return verify_proof(args.verify)

    ld = InstanceLoader()
    results = []
    diverged = []
    for name, k in CASES:
        inst = ld.load(name)
        dat_text, dat_source = ld.read_dat_text(name)
        inst_sha = _sha256_text(dat_text)
        grid_sha = _grid_identity(inst, k)
        for oid in OBJECTIVES:
            r = prove_equivalence(inst, objective_id=oid, n_theta=k, n_q=1)
            row = r.to_json()
            row["instance_sha256"] = inst_sha
            row["instance_source"] = dat_source
            row["grid_identity_sha256"] = grid_sha
            results.append(row)
            print(f"{'OK' if r.equivalent else 'DIVERGE':8} {name} K{k} {oid:26} "
                  f"opt={r.optimum} n_optima={r.n_optima} ({r.tie_kind}) "
                  f"qubo={r.qubo_search}")
            if not r.equivalent:
                diverged.append({"instance": name, "grid_k": k, "objective_id": oid,
                                 "mismatches": r.mismatches})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    proof = {
        "schema": "acrpq-equivalence-proof/2",
        "claim": "exhaustive == gurobi_discrete_milp(certified) == qubo_global_min",
        "objectives": OBJECTIVES,
        "n_cases": len(results),
        "all_equivalent": not diverged,
        "diverged": diverged,
        "provenance": _provenance(results, ld),
        "results": results,
    }
    # proof_sha256 covers the DETERMINISTIC scientific content only (wall-clock
    # solve_time_s excluded; kept in the artifact for transparency). Recompute
    # and verify any time with `--verify`.
    proof["proof_sha256"] = canonical_proof_sha256(proof)
    proof["proof_sha256_note"] = "sha over scientific content; wall-clock solve_time_s excluded"
    (out / "proof.json").write_text(json.dumps(proof, indent=2, sort_keys=True, allow_nan=False))
    print(f"\n{len(results)} cases -> {out/'proof.json'}  sha256={proof['proof_sha256'][:16]}…")
    if diverged:
        print(f"HARD STOP: {len(diverged)} case(s) diverged")
        return 1
    print("ALL EQUIVALENT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
