#!/usr/bin/env python
"""EXPERIMENTAL final LOCAL benchmark of the adaptive K=3 heading grid (Phase 11).

CP3..CP8, quadratic_control_cost_v1, initialisations NOOP and continuous-q1
(theta-only provisional incumbent when available), rho in {0.5, 0.65, 0.8},
rounds 1..8 (one chain per rho, per-round archive reported), exact local solver
and SA (3 seeds), references K3/K7 (+K33/K65 where computable), gaps, feasibility,
minimum safety margin, moved aircraft, width, evaluations, duration, monotonicity,
reach failure and classical assistance.

maneuver_count_v1 is run ONLY as a negative control (no gain claimed).

Flags: experimental=true, official_benchmark=false, real_qpu=false. No QPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

from acrpq import geometry
from acrpq.experimental import adaptive_grid as ag
from acrpq.experimental import provenance as prov
from acrpq.io.loader import InstanceLoader

ROOT = Path(__file__).resolve().parents[1]
FLAGS = {"experimental": True, "official_benchmark": False, "real_qpu": False}
RHOS = (0.5, 0.65, 0.8)
MAX_ROUNDS = 8
SA_SEEDS = (0, 1, 2)
FINE_KS = {3: (33, 65), 4: (33, 65), 5: (33, 65), 6: (33,), 7: (), 8: ()}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def min_safety_margin(inst, theta) -> float | None:
    if theta is None:
        return None
    reps = geometry.conflict_reports(inst, (1.0,) * inst.n, tuple(theta))
    return min((r.min_separation - r.required) for r in reps) if reps else None


def load_q1_reference(path: Path | None, name: str) -> dict | None:
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text())
    entry = data.get(name) if isinstance(data, dict) else None
    return entry


def load_anchor(name: str) -> dict | None:
    p = ROOT / "results" / "ampl_gurobi_campaign_v3" / f"{name}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if d.get("status") != "optimal" or not d.get("is_official_anchor"):
        return {"available": False, "status": d.get("status")}
    return {"available": True, "objective_q_theta": d["rescored_objective"], "theta": d["theta"], "q": d["q"],
            "path": str(p.relative_to(ROOT)), "sha256": _sha(p),
            "domain": "continuous q+theta (NOT the same control domain as q=1 legs)"}


def global_leg(inst, oid, k, fine=False) -> dict:
    grid = ag.global_grid(inst, k)
    table = ag.conflict_table(inst, grid)
    t0 = time.perf_counter()
    rs = ag.exact_local_solve(inst, oid, grid, table, max_states=10**15, branch_order="cost" if fine else "index")
    return {"leg": f"global_K{k}", "k": k, "certificate": {
        "algorithm": "exact branch-and-bound (conflict + cost pruning), equivalence with complete enumeration tested",
        "lower_bound": rs.objective, "upper_bound": rs.objective, "gap": 0.0, "nodes": rs.n_evaluations,
        "status": "certified_exact_on_grid" if rs.theta is not None else "infeasible_on_grid",
        "wall_time_s": time.perf_counter() - t0, "finite": rs.objective is not None and math.isfinite(rs.objective),
        "feasible": rs.theta is not None,
        "hash": hashlib.sha256(json.dumps({"instance": inst.name, "objective_id": oid, "k": k, "theta": rs.theta,
                                           "objective": rs.objective}, sort_keys=True).encode()).hexdigest()},
        "objective": rs.objective, "theta": rs.theta, "n_qubits": grid.n_qubits,
        "n_moved": None if rs.theta is None else sum(1 for t in rs.theta if t != 0.0),
        "min_safety_margin": min_safety_margin(inst, rs.theta), "evaluations": rs.n_evaluations}


def adaptive_leg(inst, oid, rho, solver, seed, init_mode, init_theta, refs) -> list[dict]:
    cfg = ag.AdaptiveConfig(objective_id=oid, rho=rho, max_rounds=MAX_ROUNDS, include_noop=(oid == "maneuver_count_v1"),
                            solver=solver, seed=seed, stagnation_patience=MAX_ROUNDS)  # report all rounds
    res = ag.run_adaptive(inst, cfg, init_mode=init_mode, init_theta=init_theta)
    rows = []
    reach = ag.reach_bound(res.delta_trajectory[0], rho)
    for rr in res.rounds:
        arch = rr.archive_objective_after
        rows.append({
            "instance": inst.name, "n": inst.n, "objective_id": oid, "leg": "adaptive_k3", "rho": rho,
            "solver": solver, "seed": seed, "init_mode": init_mode, "round": rr.round,
            "rounds_used": rr.round + 1, "delta": rr.delta, "feasible_round": rr.feasible,
            "round_objective": rr.objective, "archive_objective": arch,
            "gap_vs_q1_reference": None if (arch is None or refs.get("q1") is None) else arch - refs["q1"],
            "gap_vs_K3": None if (arch is None or refs.get(3) is None) else arch - refs[3],
            "gap_vs_K7": None if (arch is None or refs.get(7) is None) else arch - refs[7],
            "gap_vs_K65": None if (arch is None or refs.get(65) is None) else arch - refs[65],
            "n_moved": None if rr.theta is None else rr.n_moved,
            "min_safety_margin": min_safety_margin(inst, rr.theta),
            "max_qubits": rr.n_qubits, "cumulative_qubit_rounds": sum(x.n_qubits for x in res.rounds[: rr.round + 1]),
            "cumulative_evaluations": sum(x.n_evaluations for x in res.rounds[: rr.round + 1]),
            "cumulative_wall_time_s": sum(x.wall_time_s for x in res.rounds[: rr.round + 1]),
            "previous_solution_in_grid": rr.previous_solution_in_grid,
            "reach_bound": reach, "reach_failure_possible": reach < (inst.hmax - inst.hmin),
            "classical_assistance": "none (NOOP centre)" if init_mode == "noop" else "strong (continuous q=1 incumbent as centre)",
            "stop_reason": res.stop_reason,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "experimental_adaptive_grid_v1" / "local_benchmark_final"))
    ap.add_argument("--q1-reference", default=None, help="JSON {instance: {theta, objective, status,...}} (theta-only q=1 incumbents)")
    ap.add_argument("--min-n", type=int, default=3)
    ap.add_argument("--max-n", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    q1_path = Path(args.q1_reference) if args.q1_reference else None
    loader = InstanceLoader()
    rows: list[dict] = []
    refs_out: dict[str, dict] = {}
    t_start = time.perf_counter()
    for n in range(args.min_n, args.max_n + 1):
        name = f"CP_{n}"
        inst = loader.load(name)
        for oid in ("quadratic_control_cost_v1", "maneuver_count_v1"):
            legs = {k: global_leg(inst, oid, k) for k in (3, 7)}
            for k in FINE_KS.get(n, ()):
                legs[k] = global_leg(inst, oid, k, fine=True)
            q1 = load_q1_reference(q1_path, name) if oid == "quadratic_control_cost_v1" else None
            anchor = load_anchor(name) if oid == "quadratic_control_cost_v1" else None
            refs = {k: v["objective"] for k, v in legs.items()}
            refs["q1"] = None if q1 is None else q1.get("objective")
            refs_out[f"{name}__{oid}"] = {"global": legs, "theta_only_q1_reference": q1,
                                          "continuous_q_theta_anchor": anchor, **FLAGS}
            print(name, oid, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in refs.items()}, flush=True)
            inits = [("noop", None)]
            if oid == "quadratic_control_cost_v1" and q1 is not None and q1.get("theta"):
                inits.append(("continuous", tuple(float(t) for t in q1["theta"])))
            for init_mode, th in inits:
                for rho in RHOS:
                    rows += adaptive_leg(inst, oid, rho, "exact", 0, init_mode, th, refs)
                    if oid == "quadratic_control_cost_v1":
                        for seed in SA_SEEDS:
                            rows += adaptive_leg(inst, oid, rho, "sa", seed, init_mode, th, refs)
    fields = list(rows[0].keys())
    with (out / "adaptive_rounds.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    (out / "adaptive_rounds.json").write_text(json.dumps({"schema": "acrpq-experimental-adaptive-local-benchmark/1", **FLAGS, "rows": rows}, indent=1, default=str))
    (out / "references.json").write_text(json.dumps({"schema": "acrpq-experimental-adaptive-references/1", **FLAGS, "references": refs_out,
                                                     "note": "global_K* are certified exact on their grid; theta_only_q1_reference is a provisional incumbent unless its status says otherwise; the q+theta anchor is a different control domain"}, indent=1, default=str))
    manifest = {"schema": "acrpq-experimental-manifest/2", **FLAGS, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "provenance": prov.provenance_block(ROOT), "python": platform.python_version(), "script": "scripts/experimental_adaptive_local_benchmark.py",
                "rhos": list(RHOS), "max_rounds": MAX_ROUNDS, "sa_seeds": list(SA_SEEDS), "fine_ks": {str(k): list(v) for k, v in FINE_KS.items()},
                "q1_reference_path": None if q1_path is None else str(q1_path), "q1_reference_sha256": None if (q1_path is None or not q1_path.exists()) else _sha(q1_path),
                "files": {p.name: _sha(p) for p in out.iterdir() if p.is_file() and p.name != "manifest.json"},
                "total_wall_time_s": time.perf_counter() - t_start, "n_rows": len(rows)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"done in {manifest['total_wall_time_s']:.1f}s, {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
