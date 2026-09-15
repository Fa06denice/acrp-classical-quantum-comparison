#!/usr/bin/env python3
"""Run the bounded six-leg discrete benchmark (Phase 3B / 3.1).

Compares — per objective, per (instance, grid), on the identical QUBO — the legs
gurobi_milp, reference (exhaustive), qubo_exact, qubo_annealing, dwave_sa (Ocean
CPU), qaoa_aer, running ONLY the legs genuinely applicable to each instance
(applicability is reported honestly; skips are logged). Each leg is scored on its
OWN objective axis and compared only to a CERTIFIED baseline for that same
objective — the exhaustive optimum (ground truth) when enumeration fits, and/or
the certified Gurobi-MILP optimum. An incumbent is never labelled an optimum;
absent any certified baseline, only the incumbent/bound is reported.

Idempotent, fail-closed resume: one atomically-written JSON file per
(instance, grid, objective, method, seed) under ``jobs/``. Each file embeds the
canonical ``config_hash`` of the whole protocol, the instance/grid identity
hashes, the source commit and a ``result_sha256``. A resume revalidates ALL of
these and REFUSES any old/incompatible/corrupt file (never reuses it silently);
``--force`` recomputes. Aggregation refuses unexpected/missing/duplicate job
files and checks certified-baseline consistency within each group.

HONESTY OF THE HASH: ``benchmark_sha256`` covers the scientific content with
wall/CPU/provider timings excluded; ``--verify`` detects post-hoc edits. It is
NOT a cross-run reproducibility claim (sampler legs are only seed-pinned).

Never overwrites a differing historical aggregate unless ``--force``.

Run:  python scripts/run_discrete_benchmark.py [--preset cp|multifamily] [--out DIR]
      python scripts/run_discrete_benchmark.py --verify DIR/benchmark.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from acrpq.benchmark.discrete_benchmark import (
    DEFAULT_INSTANCES,
    DEFAULT_OBJECTIVES,
    DEFAULT_SEEDS,
    MULTIFAMILY_INSTANCES,
    SCHEMA_BENCH,
    JobValidationError,
    LegConfig,
    assert_json_csv_parity,
    build_baselines,
    build_protocol,
    canonical_benchmark_sha256,
    finalize_job_record,
    grid_identity_sha256,
    instance_sha256,
    method_applicability,
    plan_jobs,
    probe_gurobi,
    result_sha256,
    rows_to_csv,
    run_job,
    summarize,
    validate_aggregate,
    validate_job_record,
)
from acrpq.io.loader import InstanceLoader

# Every module whose bytes affect a leg's result is hashed into harness_sha256
# (which in turn feeds config_hash), including the load-bearing modules the legs
# transitively import — so the fingerprint answers "which exact code produced
# this", not only the top-level entry points.
_HARNESS_FILES = [
    "src/acrpq/benchmark/discrete_benchmark.py",
    "src/acrpq/benchmark/family_matrix.py",
    "src/acrpq/classical/discrete_milp.py",
    "src/acrpq/classical/discrete_enum.py",
    "src/acrpq/quantum/qubo.py",
    "src/acrpq/quantum/qubo_solvers.py",
    "src/acrpq/quantum/dwave_solver.py",
    "src/acrpq/quantum/qaoa.py",
    "src/acrpq/quantum/decode.py",
    "src/acrpq/quantum/explain.py",
    "src/acrpq/quantum/backends.py",
    "src/acrpq/quantum/memguard.py",
    "src/acrpq/objectives.py",
    "src/acrpq/geometry.py",
    "src/acrpq/model.py",
    "src/acrpq/discretize.py",
    "src/acrpq/io/loader.py",
    # dashboard/api.py hosts preflight_payload — the authoritative cap source for
    # 5 of the 6 legs' applicability — so its bytes MUST be in the fingerprint
    # (and thus config_hash), else a cap-logic change there would not bust resume.
    "src/acrpq/dashboard/api.py",
    "scripts/run_discrete_benchmark.py",
]


def _harness_sha256() -> str:
    h = hashlib.sha256()
    for f in _HARNESS_FILES:
        h.update(Path(f).read_bytes())
    return h.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp in same dir + fsync + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _provenance(harness_sha: str) -> dict:
    from dataclasses import asdict as _asdict

    from acrpq.benchmark.export import ReproInfo
    from acrpq.classical.ampl_campaign import git_info
    repro = _asdict(ReproInfo.capture())
    git = git_info()
    return {
        "git_commit": git["git_commit"], "source_commit": git["git_commit"],
        "generated_artifact_commit": None,
        "scientific_code_dirty": git["scientific_code_dirty"],
        "scientific_code_paths": git["scientific_code_paths"],
        "repository_dirty": git["repository_dirty"],
        "python_version": repro["python_version"], "package_version": repro["package_version"],
        "platform": repro["platform"],
        "versions": {k: repro["deps"].get(k)
                     for k in ("amplpy", "qiskit", "qiskit_aer", "dimod", "dwave", "numpy")},
        "harness_files": list(_HARNESS_FILES), "harness_sha256": harness_sha,
        "tolerance_policy": {
            "maneuver_count_v1": "exact (abs_tol=0, integer cardinality)",
            "quadratic_control_cost_v1": "abs_tol=1e-9 (float grouping + Gurobi gap)",
        },
        "timing_semantics": {
            "wall_time_s": "time.perf_counter around the leg call",
            "cpu_time_s": "time.process_time (THIS process; excludes subprocess "
                          "solvers such as AMPL/Gurobi). May EXCEED wall_time_s for "
                          "qaoa_aer: the Aer statevector simulator is multi-threaded, "
                          "so process_time sums CPU across threads.",
            "provider_time_s": "solver's own reported time (gurobi _solve_time for "
                               "the MILP leg; None for the Aer/CPU legs)",
        },
        "reproducibility": {
            "deterministic_legs": ["reference", "gurobi_milp", "qubo_exact",
                                   "qubo_annealing"],
            "seed_pinned_sampler_legs": ["dwave_sa", "qaoa_aer"],
            "note": "benchmark_sha256 is an integrity hash of this artifact "
                    "(timings excluded); --verify detects edits. Sampler legs are "
                    "seed-pinned but a re-run may differ by library/platform.",
        },
    }


def _verify(path: str) -> int:
    bench = json.loads(Path(path).read_text())
    ok = bench.get("benchmark_sha256") == canonical_benchmark_sha256(bench)
    print(f"{'OK benchmark_sha256 verified' if ok else 'TAMPERED'}: "
          f"{bench.get('benchmark_sha256')}")
    return 0 if ok else 2


def _certified_baselines_from_rows(rows: list[dict]) -> list[dict]:
    """Derive the per-(instance,grid,objective) certified baselines from the
    persisted rows (resume-stable; validate_aggregate already checked they are
    consistent within each group)."""
    seen: dict[tuple, dict] = {}
    for r in rows:
        g = (r["instance"], r["grid_k"], r["n_q"], r["objective_id"])
        if g not in seen:
            seen[g] = {
                "instance": r["instance"], "grid_k": r["grid_k"], "n_q": r["n_q"],
                "objective_id": r["objective_id"],
                "exhaustive_optimum": r.get("exhaustive_optimum"),
                "exhaustive_status": r.get("exhaustive_status"),
                "certified_milp_optimum": r.get("certified_milp_optimum"),
                "milp_certified": r.get("milp_certified"),
            }
    return [seen[k] for k in sorted(seen)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=("cp", "multifamily"), default="cp")
    ap.add_argument("--out")
    ap.add_argument("--verify", metavar="BENCHMARK_JSON")
    ap.add_argument("--force", action="store_true",
                    help="recompute existing job files and overwrite a differing aggregate")
    ap.add_argument("--instances", nargs="*")
    ap.add_argument("--qaoa-maxiter", type=int, default=100)
    ap.add_argument("--qaoa-shots", type=int, default=1024)
    ap.add_argument("--dwave-reads", type=int, default=200)
    ap.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
        help="explicit unique seeds for every stochastic leg (part of config_hash)",
    )
    args = ap.parse_args()
    if args.verify:
        return _verify(args.verify)

    # Preset selects the instance set, grid policy and default output dir.
    if args.preset == "multifamily":
        instances = tuple(args.instances) if args.instances else MULTIFAMILY_INSTANCES
        k_sensitivity: tuple[int, ...] = ()          # K=3 only across families
        default_out = "results/discrete_benchmark_multifamily_v1"
    else:
        instances = tuple(args.instances) if args.instances else DEFAULT_INSTANCES
        k_sensitivity = (5, 7)
        default_out = "results/discrete_benchmark"
    out = Path(args.out or default_out)
    objectives = DEFAULT_OBJECTIVES
    seeds = tuple(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        ap.error("--seeds must contain at least one unique integer")
    n_q, k_primary, sensitivity_max = 1, 3, 5

    ld = InstanceLoader()
    cfg = LegConfig(qaoa_maxiter=args.qaoa_maxiter, qaoa_shots=args.qaoa_shots,
                    dwave_num_reads=args.dwave_reads)
    # Provenance + protocol captured BEFORE any output is written (a clean worktree
    # then reads scientific_code_dirty AND repository_dirty as False).
    harness_sha = _harness_sha256()
    provenance = _provenance(harness_sha)
    gurobi_probe = probe_gurobi()
    from acrpq.benchmark.discrete_benchmark import ALL_METHODS
    protocol = build_protocol(
        instances=instances, objectives=objectives, methods=ALL_METHODS, seeds=seeds,
        leg_config=cfg, harness_sha256=harness_sha, k_primary=k_primary,
        k_sensitivity=k_sensitivity, n_q=n_q, sensitivity_max_aircraft=sensitivity_max)
    source_commit = provenance["source_commit"]

    jobs_dir = out / "jobs"
    plan = plan_jobs(ld, instances=instances, objectives=objectives,
                     k_primary=k_primary, k_sensitivity=k_sensitivity,
                     sensitivity_max_aircraft=sensitivity_max, n_q=n_q, seeds=seeds,
                     cfg=cfg, gurobi_probe=gurobi_probe)

    # Certified baselines + identity hashes, cached per (instance, grid, objective).
    base_cache: dict[tuple, dict] = {}
    id_cache: dict[tuple, tuple[str, str]] = {}

    def baselines_for(inst: str, k: int, nq: int, oid: str) -> dict:
        key = (inst, k, nq, oid)
        if key not in base_cache:
            base_cache[key] = build_baselines(ld, inst, k, nq, oid)
        return base_cache[key]

    def identity_for(inst: str, k: int, nq: int) -> tuple[str, str]:
        key = (inst, k, nq)
        if key not in id_cache:
            id_cache[key] = (instance_sha256(ld, inst), grid_identity_sha256(ld, inst, k, nq))
        return id_cache[key]

    ran = resumed = 0
    for job in plan.jobs:
        jp = jobs_dir / f"{job.key()}.json"
        ish, gsh = identity_for(job.instance, job.grid_k, job.n_q)
        if jp.exists() and not args.force:
            record = json.loads(jp.read_text())
            try:  # fail-closed: refuse an old/incompatible/corrupt file, never reuse silently
                validate_job_record(record, expected_job=job, protocol=protocol,
                                    instance_sha=ish, grid_identity_sha=gsh)
            except JobValidationError as exc:
                print(f"REFUSING to resume {jp.name}: {exc}\n"
                      f"Use a fresh --out or --force to recompute.")
                return 4
            resumed += 1
            continue
        base = baselines_for(job.instance, job.grid_k, job.n_q, job.objective_id)
        record = run_job(ld, job, baselines=base, cfg=cfg)
        record = finalize_job_record(record, protocol=protocol, instance_sha=ish,
                                     grid_identity_sha=gsh, source_commit=source_commit)
        _atomic_write(jp, json.dumps(record, sort_keys=True, allow_nan=False))
        ran += 1
        print(f"  ran {job.key()}  class={record['result_class']} "
              f"obj={record['objective_value']} gap_ex={record['gap_to_exhaustive_optimum']} "
              f"gap_milp={record['gap_to_certified_milp']}")

    # Aggregate: read every job file (resume-safe, sorted for determinism).
    rows = [json.loads(p.read_text()) for p in sorted(jobs_dir.glob("*.json"))]
    try:  # membership / missing / duplicate / baseline consistency — fail-closed
        validate_aggregate(rows, plan)
    except ValueError as exc:
        print(f"REFUSING to aggregate: {exc}\n"
              f"The jobs/ directory does not match the current plan "
              f"(unexpected/missing/duplicate/inconsistent). Use a fresh --out.")
        return 6
    for r in rows:  # integrity of every aggregated row (not just resumed ones)
        if r.get("result_sha256") != result_sha256(r):
            print(f"CORRUPT job record in aggregate: {r.get('job_key')}")
            return 5

    # Honest applicability, one entry per (instance, grid) x method.
    applicability = []
    for inst_name in instances:
        n_aircraft = ld.load(inst_name).n
        grids = [k_primary] + [k for k in k_sensitivity if n_aircraft <= sensitivity_max]
        for k in grids:
            for method, info in method_applicability(ld, inst_name, k, n_q, cfg,
                                                     gurobi_probe).items():
                applicability.append({"instance": inst_name, "grid_k": k,
                                      "method": method, **info})

    # A summary/link to the untouched CP artifact (never re-run here).
    related = []
    cp_path = Path("results/discrete_benchmark/benchmark.json")
    if cp_path.exists():
        cp = json.loads(cp_path.read_text())
        related.append({"artifact": str(cp_path), "schema": cp.get("schema"),
                        "benchmark_sha256": cp.get("benchmark_sha256"),
                        "n_jobs": cp.get("n_jobs"),
                        "note": "CP core benchmark (kept intact; not re-run here)"})

    bench = {
        "schema": SCHEMA_BENCH,
        "preset": args.preset,
        "protocol": protocol.descriptor(),
        "legs_present": sorted({r["method"] for r in rows}),
        "n_jobs": len(rows),
        "n_skipped_not_applicable": len(plan.skipped),
        "skipped_not_applicable": plan.skipped,
        "applicability": applicability,
        "certified_baselines": _certified_baselines_from_rows(rows),
        "synthesis": summarize(rows),
        "related_artifacts": related,
        "provenance": provenance,
        "results": rows,
    }
    bench["benchmark_sha256"] = canonical_benchmark_sha256(bench)
    bench["benchmark_sha256_note"] = "sha over scientific content; wall/cpu/provider times excluded"

    bench_path = out / "benchmark.json"
    if bench_path.exists() and not args.force:
        existing = json.loads(bench_path.read_text())
        if existing.get("benchmark_sha256") != bench["benchmark_sha256"]:
            print(f"REFUSING to overwrite differing historical {bench_path} "
                  f"(existing {existing.get('benchmark_sha256', '')[:16]}… != "
                  f"new {bench['benchmark_sha256'][:16]}…). Use --force or a fresh --out.")
            return 3

    csv_text = rows_to_csv(rows)
    assert_json_csv_parity(rows, csv_text)
    _atomic_write(bench_path, json.dumps(bench, indent=2, sort_keys=True, allow_nan=False))
    _atomic_write(out / "benchmark.csv", csv_text)

    n_opt = sum(1 for r in rows if r["result_class"] in
                ("certified_optimum", "qubo_optimum", "one_hot_optimum"))
    n_inc = sum(1 for r in rows if r["result_class"] == "incumbent")
    n_unavail = sum(1 for r in rows if r["result_class"] == "unavailable")
    print(f"\n[{args.preset}] {len(rows)} jobs ({ran} ran, {resumed} resumed), "
          f"{len(plan.skipped)} not-applicable skipped; "
          f"{n_opt} optimum-class, {n_inc} incumbent, {n_unavail} unavailable.")
    print(f"-> {bench_path} (+ .csv, parity OK)  sha256={bench['benchmark_sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
