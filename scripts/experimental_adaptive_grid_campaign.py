#!/usr/bin/env python
"""EXPERIMENTAL local campaign: adaptive iterative K=3 grid vs global one-hot K3/K5/K7.

Writes results/experimental_adaptive_grid_v1/ (flags: experimental=true,
official_benchmark=false, real_qpu=false). Pure local computation: no QPU, no
Qiskit, no token, no modification of any official artefact.

Usage: .venv/bin/python scripts/experimental_adaptive_grid_campaign.py [--out DIR] [--max-n 8]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from acrpq.experimental import adaptive_grid as ag
from acrpq.experimental import provenance as prov
from acrpq.io.loader import InstanceLoader

ROOT = Path(__file__).resolve().parents[1]
OBJECTIVES = ("quadratic_control_cost_v1", "maneuver_count_v1")
GLOBAL_KS = (3, 5, 7)
# fine uniform references (exact branch-and-bound, cost-ordered); only where it finishes
# in seconds: K=33 for n<=6, K=65 for n<=5. Not computable in reasonable time for CP_7/CP_8.
FINE_KS = {3: (33, 65), 4: (33, 65), 5: (33, 65), 6: (33,), 7: (), 8: ()}
STEP_POLICIES = {"rho_0.5": 0.5, "rho_0.35": 0.35, "rho_0.7": 0.7}
SA_SEEDS = (0, 1, 2)
MAX_ROUNDS = {"rho_0.5": 8, "rho_0.35": 6, "rho_0.7": 12}  # roughly equal final resolution
FLAGS = dict(ag.ARTIFACT_FLAGS)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_anchor(name: str, objective_id: str) -> dict | None:
    """Read-only continuous reference (Gurobi) when an official anchor exists."""
    if objective_id == "quadratic_control_cost_v1":
        p = ROOT / "results" / "ampl_gurobi_campaign_v3" / f"{name}.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        if d.get("status") != "optimal" or not d.get("is_official_anchor"):
            return {"available": False, "status": d.get("status"), "path": str(p.relative_to(ROOT))}
        return {"available": True, "objective": d["rescored_objective"], "theta": d["theta"],
                "q": d["q"], "path": str(p.relative_to(ROOT)), "sha256": _sha(p),
                "note": "continuous optimum uses q AND theta; discrete legs fix q=1"}
    p = ROOT / "results" / "ampl_gurobi_maneuver_count_v1" / f"{name}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    ok = d.get("status") == "optimal" and d.get("certified_integer_optimum", False)
    return {"available": bool(ok), "objective": d.get("objective"), "theta": d.get("theta"),
            "q": d.get("q"), "status": d.get("status"), "path": str(p.relative_to(ROOT)),
            "sha256": _sha(p)}


def global_leg(inst, oid: str, k: int, *, fine: bool = False) -> dict:
    grid = ag.global_grid(inst, k)
    table = ag.conflict_table(inst, grid)
    t0 = time.perf_counter()
    rs = ag.exact_local_solve(inst, oid, grid, table, max_states=10**15,
                              branch_order="cost" if fine else "index")
    wall = time.perf_counter() - t0
    nmoved = None if rs.theta is None else sum(1 for t in rs.theta if t != 0.0)
    return {
        "kind": "global_onehot_fine" if fine else "global_onehot", "k": k,
        "solver": "exact_bb_cost_order" if fine else "exact_bb", "exact": True,
        "n_qubits": grid.n_qubits, "search_space": grid.search_space(),
        "n_onehot_quadratic_terms": grid.n_onehot_quadratic_terms(),
        "n_conflict_quadratic_terms": ag.n_conflict_terms(table),
        "feasible": rs.theta is not None, "objective": rs.objective,
        "theta": rs.theta, "n_moved": nmoved, "n_evaluations": rs.n_evaluations,
        "solver_calls": 1, "wall_time_s": wall, "rounds": 1,
        "width_times_rounds": grid.n_qubits,
    }


def adaptive_leg(inst, oid: str, policy: str, rho: float, solver: str, seed: int,
                 init_mode: str, init_theta=None, delta0: float | None = None) -> dict:
    cfg = ag.AdaptiveConfig(objective_id=oid, rho=rho, max_rounds=MAX_ROUNDS[policy], delta0=delta0,
                            include_noop=(oid == "maneuver_count_v1"), solver=solver, seed=seed,
                            stagnation_patience=3)
    res = ag.run_adaptive(inst, cfg, init_mode=init_mode, init_theta=init_theta)
    d = res.to_dict()
    d.update({
        "kind": "adaptive_k3", "policy": policy, "rho": rho, "solver": solver, "seed": seed,
        "delta0": res.delta_trajectory[0] if res.delta_trajectory else None,
        "reach_bound": ag.reach_bound(res.delta_trajectory[0], rho) if res.delta_trajectory else None,
        "total_quadratic_terms_all_rounds": sum(
            rr.n_onehot_quadratic_terms + rr.n_conflict_quadratic_terms for rr in res.rounds),
        "round_monotone": ag.round_solutions_monotone(res),
        "archive_monotone": ag.archive_is_monotone(res),
        "all_previous_in_grid": all(rr.previous_solution_in_grid for rr in res.rounds),
        "final_delta": res.delta_trajectory[-1] if res.delta_trajectory else None,
        "n_moved": None if res.best_theta is None else sum(1 for t in res.best_theta if t != 0.0),
    })
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "experimental_adaptive_grid_v1"))
    ap.add_argument("--min-n", type=int, default=3)
    ap.add_argument("--max-n", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    loader = InstanceLoader()
    t_start = time.perf_counter()
    rows: list[dict] = []
    runs: list[dict] = []
    for n in range(args.min_n, args.max_n + 1):
        name = f"CP_{n}"
        inst = loader.load(name)
        for oid in OBJECTIVES:
            anchor = _load_anchor(name, oid)
            legs: list[dict] = []
            for k in GLOBAL_KS:
                legs.append(global_leg(inst, oid, k))
                print(f"{name} {oid} global K{k}: obj={legs[-1]['objective']} "
                      f"qubits={legs[-1]['n_qubits']} t={legs[-1]['wall_time_s']:.2f}s", flush=True)
            for k in FINE_KS.get(n, ()):
                legs.append(global_leg(inst, oid, k, fine=True))
                print(f"{name} {oid} FINE global K{k}: obj={legs[-1]['objective']} "
                      f"qubits={legs[-1]['n_qubits']} t={legs[-1]['wall_time_s']:.2f}s", flush=True)
            for policy, rho in STEP_POLICIES.items():
                legs.append(adaptive_leg(inst, oid, policy, rho, "exact", 0, "noop"))
                print(f"{name} {oid} adaptive {policy} exact: obj={legs[-1]['best_objective']} "
                      f"rounds={legs[-1]['n_rounds']} maxq={legs[-1]['max_qubits']} "
                      f"stop={legs[-1]['stop_reason']}", flush=True)
                for seed in SA_SEEDS:
                    legs.append(adaptive_leg(inst, oid, policy, rho, "sa", seed, "noop"))
            # init B: continuous anchor as GRID/control warm-start (read-only)
            if anchor and anchor.get("available"):
                th = tuple(float(t) for t in anchor["theta"])
                legs.append(adaptive_leg(inst, oid, "rho_0.5", 0.5, "exact", 0, "continuous", th))
                # red-team requirement: small-Delta0 continuous init (bounded reach)
                legs.append(adaptive_leg(inst, oid, "rho_0.5", 0.5, "exact", 0, "continuous", th,
                                         delta0=0.05))
                for k in GLOBAL_KS + tuple(FINE_KS.get(n, ())):
                    snap = ag.snap_to_grid_baseline(inst, oid, th, k)
                    snap.update({"kind": "classical_snap_baseline", "solver": "snap_nearest_level",
                                 "init_mode": "continuous"})
                    legs.append(snap)
                    rep = ag.snap_and_greedy_repair_baseline(inst, oid, th, k)
                    rep.update({"kind": "classical_snap_repair_baseline",
                                "solver": "snap_plus_greedy_repair", "init_mode": "continuous"})
                    legs.append(rep)
            rec = {"schema": "acrpq-experimental-adaptive-grid/1", **FLAGS, "instance": name,
                   "n": n, "objective_id": oid, "continuous_anchor": anchor, "legs": legs}
            p = out / "runs" / f"{name}__{oid}.json"
            p.write_text(json.dumps(rec, indent=1, default=str))
            runs.append({"path": str(p.relative_to(out)), "sha256": _sha(p)})
            g3 = next(x for x in legs if x["kind"] == "global_onehot" and x["k"] == 3)
            g5 = next(x for x in legs if x["kind"] == "global_onehot" and x["k"] == 5)
            g7 = next(x for x in legs if x["kind"] == "global_onehot" and x["k"] == 7)
            for leg in legs:
                if leg["kind"] == "adaptive_k3":
                    best = leg["best_objective"]
                    rows.append({
                        "instance": name, "n": n, "objective_id": oid, "leg": "adaptive_k3",
                        "policy": leg["policy"] + (f"_d0={leg['delta0']:.3f}" if leg["init_mode"] == "continuous" else ""),
                        "solver": leg["solver"], "seed": leg["seed"],
                        "init_mode": leg["init_mode"], "feasible": leg["best_feasible"],
                        "best_objective": best, "n_moved": leg["n_moved"],
                        "rounds": leg["n_rounds"], "max_qubits": leg["max_qubits"],
                        "width_times_rounds": leg["width_times_rounds"],
                        "solver_calls": leg["n_solver_calls"], "evaluations": leg["total_evaluations"],
                        "wall_time_s": round(leg["total_wall_time_s"], 4),
                        "final_delta": leg["final_delta"], "stop_reason": leg["stop_reason"],
                        "round_monotone": leg["round_monotone"], "archive_monotone": leg["archive_monotone"],
                        "gap_vs_K3": None if best is None or g3["objective"] is None else best - g3["objective"],
                        "gap_vs_K5": None if best is None or g5["objective"] is None else best - g5["objective"],
                        "gap_vs_K7": None if best is None or g7["objective"] is None else best - g7["objective"],
                        "anchor_objective": anchor.get("objective") if anchor and anchor.get("available") else None,
                    })
                elif leg["kind"] in ("global_onehot", "global_onehot_fine"):
                    rows.append({
                        "instance": name, "n": n, "objective_id": oid, "leg": f"global_K{leg['k']}",
                        "policy": "", "solver": leg["solver"], "seed": "", "init_mode": "",
                        "feasible": leg["feasible"], "best_objective": leg["objective"],
                        "n_moved": leg["n_moved"], "rounds": 1, "max_qubits": leg["n_qubits"],
                        "width_times_rounds": leg["n_qubits"], "solver_calls": 1,
                        "evaluations": leg["n_evaluations"], "wall_time_s": round(leg["wall_time_s"], 4),
                        "final_delta": (inst.hmax - inst.hmin) / (leg["k"] - 1), "stop_reason": "",
                        "round_monotone": "", "archive_monotone": "", "gap_vs_K3": "", "gap_vs_K5": "",
                        "gap_vs_K7": "", "anchor_objective": anchor.get("objective") if anchor and anchor.get("available") else None,
                    })
                elif leg["kind"] in ("classical_snap_baseline", "classical_snap_repair_baseline"):
                    tag = "snap" if leg["kind"] == "classical_snap_baseline" else "snap_repair"
                    rows.append({
                        "instance": name, "n": n, "objective_id": oid, "leg": f"{tag}_K{leg['k']}",
                        "policy": "", "solver": leg["solver"], "seed": "", "init_mode": "continuous",
                        "feasible": leg["feasible"], "best_objective": leg["objective"],
                        "n_moved": sum(1 for t in leg["theta"] if t != 0.0), "rounds": 0, "max_qubits": 0,
                        "width_times_rounds": 0, "solver_calls": 0, "evaluations": 1, "wall_time_s": 0.0,
                        "final_delta": "", "stop_reason": "", "round_monotone": "", "archive_monotone": "",
                        "gap_vs_K3": "", "gap_vs_K5": "", "gap_vs_K7": "",
                        "anchor_objective": anchor.get("objective") if anchor and anchor.get("available") else None,
                    })
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({"schema": "acrpq-experimental-adaptive-grid-summary/1",
                                                  **FLAGS, "rows": rows}, indent=1, default=str))
    manifest = {
        "schema": "acrpq-experimental-manifest/2", **FLAGS,
        "title": "Experimental adaptive iterative K=3 heading grid vs global one-hot K3/K5/K7 (LOCAL, exact + SA)",
        "not_for_citation_as_benchmark": True,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": prov.provenance_block(ROOT), "python": platform.python_version(),
        "script": "scripts/experimental_adaptive_grid_campaign.py",
        "module": "src/acrpq/experimental/adaptive_grid.py",
        "instances": [f"CP_{n}" for n in range(args.min_n, args.max_n + 1)],
        "objectives": list(OBJECTIVES), "global_ks": list(GLOBAL_KS), "fine_ks": {str(k): list(v) for k, v in FINE_KS.items()},
        "step_policies": STEP_POLICIES, "max_rounds": MAX_ROUNDS, "sa_seeds": list(SA_SEEDS),
        "speed_model": "q fixed to 1 (heading-only), identical to official n_q=1 grids",
        "continuous_anchor_usage": "read-only; used as GRID/control warm-start centre (init B) and for gap reporting; never modified",
        # per-run JSON files are RECONSTRUCTIBLE by this script and are NOT versioned (the repo
        # .gitignore excludes runs/); their hashes are recorded for local verification only.
        "runs": runs, "runs_versioned": False,
        "runs_reconstruction": "PYTHONPATH=src python scripts/experimental_adaptive_grid_campaign.py --max-n 8",
        "total_wall_time_s": time.perf_counter() - t_start,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"done in {manifest['total_wall_time_s']:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
