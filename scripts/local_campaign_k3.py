#!/usr/bin/env python3
"""Small reproducible LOCAL scientific campaign — CP_3/CP_4/CP_5, grid K=3.

Reuses the existing pipeline end to end — there is NO second persistence or
schema here:
  api.scientific_run  -> seed-repeated solves (dispersion)
  persist.build_scientific_record + persist.save_run
                      -> per-run JSON + per-seed CSV + integrity + meta sidecar
This script only *orchestrates* those calls and writes an aggregate manifest +
summary on top of the artifacts they produce.

Methods per instance (same K=3 grid for all — never cross-grid):
  reference       discrete in-grid optimum (proven if under cap)  — 1 seed
  qubo-exact      exact min H(x)                                   — 1 seed
  qubo-annealing  simulated annealing                             — 10 seeds
  qaoa (aer_sim)  QAOA on the local Aer simulator                 — 10 seeds

Never called: any real QPU. mode is always aer_sim. Deterministic methods use a
single seed (stdev 0 by construction).

Outputs (nothing existing is overwritten; a fresh, timestamped dir is used):
  <out>/runs/scientific/<run_id>.json|.csv|.meta.json   (via save_run)
  <out>/manifest.json      hashed list of every saved run
  <out>/summary.json       per (instance, method) stats  (strict: no NaN/Inf)
  <out>/summary.csv        the same, flat

Usage:
  python scripts/local_campaign_k3.py [--out DIR] [--sa-seeds N] [--qaoa-seeds N]
                                      [--max-minutes M] [--estimate-only]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import time
from pathlib import Path

from acrpq.dashboard.api import InstanceLoader, scientific_run
from acrpq.dashboard.persist import build_scientific_record, save_run
from acrpq.classical.ampl_protocol import AnchorRejected, accept_official_anchor
from acrpq.classical.ampl_campaign import git_info

INSTANCES = ["CP_3", "CP_4", "CP_5"]
K = 3
N_Q = 1
# per-seed cost model (s), from a local K=3 probe; only used for the estimate
_COST = {"reference": 0.2, "qubo-exact": 0.05, "qubo-annealing": 0.05, "qaoa": 2.5}
# continuous certified anchors, read for the discretisation-cost context column
_ANCHOR_DIR = Path("results/ampl_gurobi_campaign_v3")


def _finite(x):
    """Refuse NaN/Inf; pass through finite numbers, keep None/other as-is."""
    if isinstance(x, float) and not math.isfinite(x):
        raise ValueError(f"non-finite value refused: {x!r}")
    return x


def _stats(objectives: list[float]) -> dict:
    if not objectives:
        return {"n": 0, "median": None, "mean": None, "stdev": None,
                "min": None, "max": None}
    return {
        "n": len(objectives),
        "median": _finite(statistics.median(objectives)),
        "mean": _finite(statistics.fmean(objectives)),
        "stdev": _finite(statistics.pstdev(objectives) if len(objectives) > 1 else 0.0),
        "min": _finite(min(objectives)),
        "max": _finite(max(objectives)),
    }


def _continuous_anchor(instance: str) -> tuple[float | None, bool]:
    """(rescored continuous objective, is_official_anchor) or (None, False)."""
    path = _ANCHOR_DIR / f"{instance}.json"
    manifest_path = _ANCHOR_DIR / "campaign_manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        return None, False
    try:
        d = json.loads(path.read_text())
        manifest = json.loads(manifest_path.read_text())
        expected = (manifest.get("artifact_hashes") or {}).get(instance)
        actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is None or expected != actual:
            return None, False
        accept_official_anchor(d, _ANCHOR_DIR)
    except (OSError, ValueError, AnchorRejected):
        return None, False
    return d.get("rescored_objective"), True


def _plan(sa_seeds: int, qaoa_seeds: int) -> list[dict]:
    plan = []
    for inst in INSTANCES:
        plan.append({"instance": inst, "method": "reference", "solver": "reference",
                     "seeds": [1]})
        plan.append({"instance": inst, "method": "qubo-exact", "solver": "qubo-exact",
                     "seeds": [1]})
        plan.append({"instance": inst, "method": "qubo-annealing",
                     "solver": "qubo-annealing", "seeds": list(range(1, sa_seeds + 1))})
        plan.append({"instance": inst, "method": "qaoa", "solver": "qaoa",
                     "seeds": list(range(1, qaoa_seeds + 1))})
    return plan


def _estimate_seconds(plan: list[dict]) -> float:
    return sum(_COST.get(p["solver"], 1.0) * len(p["seeds"]) for p in plan)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/local_campaign_k3")
    ap.add_argument("--sa-seeds", type=int, default=10)
    ap.add_argument("--qaoa-seeds", type=int, default=10)
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--estimate-only", action="store_true")
    args = ap.parse_args()

    sa_seeds, qaoa_seeds = args.sa_seeds, args.qaoa_seeds
    plan = _plan(sa_seeds, qaoa_seeds)
    est = _estimate_seconds(plan)
    print(f"planned methods: {len(plan)}  ·  estimated ~{est/60:.1f} min")

    # honest cap: if over budget, shrink the QAOA seed count and log the trade-off
    while est > args.max_minutes * 60 and qaoa_seeds > 1:
        qaoa_seeds -= 1
        plan = _plan(sa_seeds, qaoa_seeds)
        est = _estimate_seconds(plan)
        print(f"  over budget -> reduce QAOA seeds to {qaoa_seeds}  (~{est/60:.1f} min)")
    if est > args.max_minutes * 60:
        print(f"  still over budget at {qaoa_seeds} QAOA seed(s); proceeding minimally")
    print(f"final plan: SA {sa_seeds} seeds, QAOA {qaoa_seeds} seeds  ·  ~{est/60:.1f} min")
    if args.estimate_only:
        return 0

    git = git_info()
    if git.get("scientific_code_dirty") is not False:
        raise RuntimeError(
            "refusing a citable campaign from dirty scientific code; commit/review "
            "the code first, then run into a fresh --out directory"
        )

    out = Path(args.out)
    runs_root = out / "runs"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite or mix campaign artifacts in non-empty {out}; "
            "choose a fresh --out directory"
        )
    out.mkdir(parents=True, exist_ok=True)

    loader = InstanceLoader()
    manifest_entries: list[dict] = []
    summary_rows: list[dict] = []
    # reference objective per instance, filled as we run the reference first
    ref_obj: dict[str, float | None] = {}
    ref_proven: dict[str, bool] = {}

    t_start = time.time()
    for p in plan:
        inst, method, solver, seeds = p["instance"], p["method"], p["solver"], p["seeds"]
        t0 = time.time()
        try:
            sci = scientific_run(
                loader, inst, solver=solver, seeds=seeds, n_theta=K, n_q=N_Q,
                mode="aer_sim", reps=1, maxiter=100, shots=1024,
            )
        except Exception as e:  # a not-applicable method never aborts the campaign
            print(f"  {inst:5} {method:15} SKIPPED — {type(e).__name__}: {str(e)[:70]}")
            summary_rows.append({"instance": inst, "method": method,
                                 "status": "skipped", "reason": str(e)[:120]})
            continue
        dt = time.time() - t0

        config = {"n_theta": K, "n_q": N_Q, "solver": solver, "mode": "aer_sim",
                  "reps": 1, "maxiter": 100, "shots": 1024, "seeds": seeds}
        record = build_scientific_record(sci, instance=inst, subset=None, config=config)
        receipt = save_run(record, root=str(runs_root))
        manifest_entries.append({
            "instance": inst, "method": method, "run_id": record["run_id"],
            "json": receipt["path"], "csv": receipt["csv_path"],
            "json_sha256": receipt["json_sha256"],
            "csv_sha256": receipt["csv_sha256"],
            "citable": receipt["citable"],
        })

        agg = sci["aggregate"]
        objectives = [
            r["solution"]["objective"] for r in sci["runs"]
            if r["solution"].get("feasible") and r["solution"].get("objective") is not None
        ]
        walls = [r["solution"].get("wall_time_s") for r in sci["runs"]
                 if isinstance(r["solution"].get("wall_time_s"), (int, float))]
        qpu = [r["solution"].get("qpu_time_s") for r in sci["runs"]
               if isinstance(r["solution"].get("qpu_time_s"), (int, float))]
        st = _stats(objectives)

        if method == "reference":
            ref_obj[inst] = st["median"]
            # proven in-grid optimum iff the solver reports gap_allowed
            ref_proven[inst] = bool(
                sci["runs"][0]["solution"].get("proof_status", {}).get("gap_allowed"))

        cont, cont_official = _continuous_anchor(inst)
        robj = ref_obj.get(inst)
        algo_difference = (_finite(st["median"] - robj)
                    if (st["median"] is not None and robj is not None) else None)
        disc_cost = (_finite(robj - cont)
                     if (robj is not None and cont is not None) else None)
        algorithmic_gap_official = bool(
            algo_difference is not None and ref_proven.get(inst)
            and agg["feasible_rate"] > 0
        )
        discretisation_cost_official = bool(
            disc_cost is not None and cont_official and ref_proven.get(inst)
        )

        summary_rows.append({
            "instance": inst, "method": method, "status": "ok",
            "grid": f"K{K}", "n_binary_variables": sci["runs"][0]["solution"].get("n_qubits"),
            "n_seeds": len(seeds),
            "feasible_rate": _finite(agg["feasible_rate"]),
            "objective_median": st["median"], "objective_mean": st["mean"],
            "objective_stdev": st["stdev"], "objective_min": st["min"],
            "objective_max": st["max"],
            "reference_k3_objective": robj,
            "reference_k3_proven": ref_proven.get(inst),
            "algorithmic_difference_vs_ref": algo_difference,
            "algorithmic_gap_official": algorithmic_gap_official,
            "continuous_gurobi_objective": cont,
            "continuous_anchor_official": cont_official,
            "discretisation_cost": disc_cost,
            "discretisation_cost_official": discretisation_cost_official,
            "objective_population": "feasible_runs_only",
            "wall_s_median": _finite(statistics.median(walls)) if walls else None,
            "qpu_s_median": _finite(statistics.median(qpu)) if qpu else None,
        })
        print(f"  {inst:5} {method:15} ok  median={st['median']}  "
              f"feas={agg['feasible_rate']:.2f}  {dt:.1f}s")

    wall = time.time() - t_start

    # ---- manifest (hashed list of every saved run) -------------------------
    man_blob = json.dumps(manifest_entries, sort_keys=True, separators=(",", ":"))
    manifest = {
        "schema": "acrpq-local-campaign/1",
        "grid": f"K{K}", "instances": INSTANCES,
        "methods": ["reference", "qubo-exact", "qubo-annealing", "qaoa"],
        "sa_seeds": sa_seeds, "qaoa_seeds": qaoa_seeds,
        "backend": "aer_simulator", "real_qpu_used": False,
        "git_commit": git.get("git_commit"),
        "scientific_code_dirty": git.get("scientific_code_dirty"),
        "repository_dirty": git.get("repository_dirty"),
        "wall_seconds": round(wall, 2),
        "n_runs": len(manifest_entries),
        "runs": manifest_entries,
        "manifest_sha256": hashlib.sha256(man_blob.encode()).hexdigest(),
    }
    # strict JSON everywhere in the aggregate: no NaN/Inf, no default=str fallback
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    (out / "summary.json").write_text(
        json.dumps({"grid": f"K{K}", "rows": summary_rows}, indent=2, allow_nan=False))

    buf = io.StringIO()
    cols = ["instance", "method", "status", "grid", "n_binary_variables", "n_seeds",
            "feasible_rate", "objective_median", "objective_mean", "objective_stdev",
            "objective_min", "objective_max", "reference_k3_objective",
            "reference_k3_proven", "algorithmic_difference_vs_ref",
            "algorithmic_gap_official",
            "continuous_gurobi_objective", "continuous_anchor_official",
            "discretisation_cost", "discretisation_cost_official",
            "objective_population", "wall_s_median", "qpu_s_median", "reason"]
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in summary_rows:
        w.writerow(r)
    (out / "summary.csv").write_text(buf.getvalue())

    print(f"\ncampaign done in {wall:.1f}s · {len(manifest_entries)} runs saved · out={out}")
    print(f"manifest_sha256={manifest['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
