#!/usr/bin/env python
"""EXPERIMENTAL local benchmark of exact conflict-graph decomposition (Phase 7).

For selected multi-family instances and BOTH objectives:
  1. monolithic one-hot K3 exact optimum (branch-and-bound, timeout);
  2. exact decomposition: components of the any-option K3 conflict graph built from the
     OFFICIAL DiscreteACRP conflict tables, each solved exactly on its sub-instance,
     recomposed by aircraft id and RESCORED on the full geometry;
  3. adaptive K=3 refinement per component (quadratic objective; the maneuver-count
     objective is kept as a negative control), recomposed and rescored;
  4. resource accounting: max width, cumulative width, sub-problems, rounds, evaluations,
     time, statevector memory of the widest sub-problem.

Flags: experimental=true, official_benchmark=false, real_qpu=false. No QPU, no network.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import signal
import sys
import time
from pathlib import Path

from acrpq import geometry
from acrpq.discretize import DiscreteACRP
from acrpq.experimental import adaptive_grid as ag
from acrpq.experimental import provenance as prov
from acrpq.io.loader import InstanceLoader
from acrpq.model import ManeuverGrid
from acrpq.objectives import primary_objective

ROOT = Path(__file__).resolve().parents[1]
FLAGS = {"experimental": True, "official_benchmark": False, "real_qpu": False}
OBJECTIVES = ("quadratic_control_cost_v1", "maneuver_count_v1")
DEFAULT_INSTANCES = ("CP_5", "CP_8", "GP_4", "FP_4", "FP_5", "FP_6", "FP_7", "FP_15",
                     "RCP_10_1", "RCP_10_2", "RCP_20_1", "RCP_50_1", "RCP_50_1@FL5", "RCP_100_1@FL5")


class Timeout(Exception):
    pass


def _alarm(signum, frame):  # noqa: ARG001
    raise Timeout()


def with_timeout(seconds: int, fn, *args, **kwargs):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        return fn(*args, **kwargs), None
    except Timeout:
        return None, f"timeout>{seconds}s"
    except ValueError as exc:  # e.g. search-space cap: recorded as an explicit refusal
        return None, f"refused: {exc}"
    finally:
        signal.alarm(0)


def components_k3(inst):
    """Any-option K3 components from the OFFICIAL conflict tables (independent of the analyst module)."""
    grid = ManeuverGrid.build(n_theta=3, n_q=1, instance=inst)
    d = DiscreteACRP.build(inst, grid)
    parent = list(range(inst.n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = 0
    for (i, j), mat in d.conflict.items():
        if any(any(row) for row in mat):
            edges += 1
            parent[find(i)] = find(j)
    comps: dict[int, list[int]] = {}
    for i in inst.aircraft():
        comps.setdefault(find(i), []).append(i)
    return sorted(comps.values(), key=lambda c: (-len(c), c)), edges


def rescore(inst, oid, theta):
    q = (1.0,) * inst.n
    return geometry.count_conflicts(inst, q, tuple(theta)), primary_objective(oid, q, tuple(theta), inst.w)


def sv_bytes(qubits: int) -> float:
    return 16.0 * (2.0 ** qubits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "experimental_instance_scalability_v1" / "decomposition_benchmark"))
    ap.add_argument("--instances", nargs="*", default=list(DEFAULT_INSTANCES))
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--rounds", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    loader = InstanceLoader()
    rows: list[dict] = []
    t_all = time.perf_counter()
    for name in args.instances:
        inst = loader.load(name)
        comps, edges = components_k3(inst)
        largest = len(comps[0])
        for oid in OBJECTIVES:
            row: dict = {"instance": name, "family": inst.family.value, "n": inst.n, "objective_id": oid,
                         "n_edges_K3": edges, "n_components": len(comps), "component_sizes": [len(c) for c in comps],
                         "largest_component": largest, "Q_global_K3": 3 * inst.n, "Q_largest_component": 3 * largest,
                         "statevector_bytes_global": sv_bytes(3 * inst.n), "statevector_bytes_largest": sv_bytes(3 * largest)}
            # 1) monolithic exact
            t0 = time.perf_counter()
            res, err = with_timeout(args.timeout, ag.exact_local_solve, inst, oid, ag.global_grid(inst, 3), max_states=10**18)
            row["mono_time_s"] = time.perf_counter() - t0
            if err or res is None or res.theta is None:
                row.update({"mono_status": err or "infeasible", "mono_objective": None, "mono_evaluations": None if res is None else res.n_evaluations})
            else:
                nconf, obj = rescore(inst, oid, res.theta)
                row.update({"mono_status": "exact", "mono_objective": obj, "mono_conflicts": nconf, "mono_evaluations": res.n_evaluations,
                            "mono_moved": sum(1 for t in res.theta if t != 0.0)})
            # 2) exact decomposition
            t0 = time.perf_counter()
            theta = [0.0] * inst.n
            evals = 0
            ok = True
            comp_status = []
            for ids in comps:
                sub = inst.subset(ids)
                r_sub, err_sub = with_timeout(args.timeout, ag.exact_local_solve, sub, oid, ag.global_grid(sub, 3), max_states=10**18)
                if err_sub or r_sub is None or r_sub.theta is None:
                    ok = False
                    comp_status.append(err_sub or "infeasible")
                    continue
                evals += r_sub.n_evaluations
                comp_status.append("exact")
                for k, aid in enumerate(ids):
                    theta[aid - 1] = r_sub.theta[k]
            row["decomp_time_s"] = time.perf_counter() - t0
            row["decomp_component_status"] = comp_status
            if ok:
                nconf, obj = rescore(inst, oid, theta)
                row.update({"decomp_status": "exact_recomposed", "decomp_objective": obj, "decomp_conflicts": nconf,
                            "decomp_evaluations": evals, "decomp_moved": sum(1 for t in theta if t != 0.0),
                            "decomp_equals_mono": (None if row.get("mono_objective") is None else
                                                   (math.isclose(obj, row["mono_objective"], rel_tol=0, abs_tol=1e-12) and nconf == 0)),
                            "decomp_cumulative_width": 3 * inst.n, "decomp_max_width": 3 * largest})
            else:
                row.update({"decomp_status": "incomplete", "decomp_objective": None})
            # 3) adaptive per component
            t0 = time.perf_counter()
            theta_a = [0.0] * inst.n
            ok_a = True
            rounds_max = 0
            evals_a = 0
            cum_width = 0
            for ids in comps:
                sub = inst.subset(ids)
                cfg = ag.AdaptiveConfig(objective_id=oid, rho=0.5, max_rounds=args.rounds,
                                        include_noop=(oid == "maneuver_count_v1"), stagnation_patience=args.rounds,
                                        max_states=10**18)  # branch-and-bound prunes; timeout guards the run
                r_ad, err_ad = with_timeout(args.timeout, ag.run_adaptive, sub, cfg)
                if err_ad or r_ad is None or r_ad.best_theta is None:
                    ok_a = False
                    row.setdefault("adaptive_refusals", []).append(err_ad or "no_feasible")
                    continue
                rounds_max = max(rounds_max, r_ad.n_rounds)
                evals_a += r_ad.total_evaluations
                cum_width += r_ad.width_times_rounds
                for k, aid in enumerate(ids):
                    theta_a[aid - 1] = r_ad.best_theta[k]
            row["adaptive_time_s"] = time.perf_counter() - t0
            if ok_a:
                nconf, obj = rescore(inst, oid, theta_a)
                row.update({"adaptive_status": "recomposed", "adaptive_objective": obj, "adaptive_conflicts": nconf,
                            "adaptive_rounds_max": rounds_max, "adaptive_evaluations": evals_a,
                            "adaptive_cumulative_width": cum_width, "adaptive_max_width": 3 * largest,
                            "adaptive_moved": sum(1 for t in theta_a if t != 0.0),
                            # DIFFERENT GRIDS: the adaptive value lives on refined local grids, the
                            # decomposition value on the fixed K3 grid. Reported as a difference of
                            # resolution, never as a comparable gap.
                            "adaptive_minus_decompK3_objective_DIFFERENT_GRIDS": None if row.get("decomp_objective") is None else row["decomp_objective"] - obj,
                            "domains_comparable_adaptive_vs_K3": False})
            else:
                row.update({"adaptive_status": "incomplete", "adaptive_objective": None})
            rows.append(row)
            print(name, oid, {k: row.get(k) for k in ("largest_component", "mono_status", "mono_objective", "decomp_status", "decomp_objective",
                                                     "decomp_conflicts", "decomp_equals_mono", "adaptive_objective", "adaptive_conflicts")}, flush=True)
    fields = sorted({k for r in rows for k in r})
    out.mkdir(parents=True, exist_ok=True)  # a concurrent writer may have recreated the parent directory
    with (out / "benchmark.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    (out / "benchmark.json").write_text(json.dumps({"schema": "acrpq-experimental-decomposition-benchmark/2", **FLAGS, "rows": rows},
                                                   sort_keys=True, separators=(",", ":"), allow_nan=False, default=str) + "\n")
    manifest = {"schema": "acrpq-experimental-manifest/2", **FLAGS,
                "provenance": prov.provenance_block(ROOT), "script": "scripts/experimental_decomposition_benchmark.py",
                "scientific_hash_rows": prov.scientific_hash({"rows": rows}),
                "instances": args.instances, "timeout_s": args.timeout, "adaptive_rounds": args.rounds,
                "component_graph": "any-option K3 from official DiscreteACRP conflict tables (q=1)",
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name != "manifest.json"},
                "total_wall_time_s": time.perf_counter() - t_all}
    prov.write_compact_json(out / "manifest.json", manifest)
    print(f"done in {manifest['total_wall_time_s']:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
