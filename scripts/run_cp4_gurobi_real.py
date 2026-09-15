#!/usr/bin/env python
"""Single REAL demonstration run: CP_4 through the historical NC_ACRP.mod with Gurobi.

This is the one explicit real execution (not a campaign). It:
  * shows the exact ampl binary + the generated temp .run it will execute;
  * runs the verbatim whole-instance CP_4.dat via the secure OriginalAMPLSolver
    (Code/ untouched, bounded timeout, minimal env, process-group kill);
  * rescoring / integrity / status all come from the runner;
  * writes a dedicated artifact under results/original_ampl_real/ with git SHA,
    dirty state, source/model/data hashes, versions and parameters.

Local AMPL/Gurobi execution is authorised; it is NOT an IBM QPU operation and costs
no cloud money. The licence file is never read, copied or logged.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from acrpq.classical.original_ampl import OriginalAMPLSolver, _repo_code_dir
from acrpq.io.loader import InstanceLoader

TIMEOUT_S = 120.0
INSTANCE = "CP_4"
SOLVER = "gurobi"


def _git_state() -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5)
        return (sha.stdout.strip() or None,
                bool(dirty.stdout.strip()) if dirty.returncode == 0 else None)
    except (OSError, subprocess.SubprocessError):
        return None, None


def main() -> int:
    loader = InstanceLoader()
    inst = loader.load(INSTANCE)
    dat_text, source = loader.read_dat_text(INSTANCE)
    solver = OriginalAMPLSolver(solver=SOLVER, timeout_s=TIMEOUT_S)

    binary = solver._resolve_binary()
    print("=== CP_4 real run — pre-flight ===")
    print(f"instance         : {INSTANCE} (n={inst.n}, family={inst.family.value})")
    print(f"dat source       : {source} ({len(dat_text.encode())} bytes, verbatim)")
    print(f"ampl solver      : {SOLVER} (supervised)")
    print(f"ampl binary      : {binary}")
    print(f"timeout          : {TIMEOUT_S:.0f}s")
    if binary is None:
        print("AMPL not resolvable — aborting (nothing fabricated).")
        return 4
    # Exact temp .run that WILL be generated (paths shown as they will be resolved):
    mod, pre = solver._code_paths()
    preview = solver._run_script(mod, pre, Path("<tmpdir>") / f"{INSTANCE}.dat", inst)
    print("--- generated .run (preview; solver only set in this TEMP file) ---")
    print(preview)
    print(f"argv             : [{binary}, <tmpdir>/acrpq_run.run]")
    print("=== executing ===")

    res = solver.solve(inst, dat_text=dat_text)

    e = res.extra
    prov = json.loads(res.message) if res.message else {}
    sha, dirty = _git_state()

    # --- post-run scientific verification ---
    sol = res.solution
    cardinality_ok = sol is not None and len(sol.q) == inst.n and len(sol.theta) == inst.n
    finite_ok = sol is not None and all(
        v == v and abs(v) != float("inf") for v in (*sol.q, *sol.theta))

    artifact = {
        "kind": "original_ampl_real_run",
        "instance": INSTANCE,
        "n": inst.n,
        "ampl_solver_requested": SOLVER,
        "status": res.status.value,
        "feasible_rescored": res.feasible,
        "n_initial_conflicts": res.n_initial_conflicts,
        "n_residual_conflicts_rescored": res.n_conflicts,
        "cardinality_ok": cardinality_ok,
        "all_values_finite": finite_ok,
        "provider_objective": e.get("provider_objective"),
        "rescored_objective": e.get("rescored_objective"),
        "objective_abs_delta": e.get("objective_abs_delta"),
        "objective_rel_delta": e.get("objective_rel_delta"),
        "integrity_ok": bool(e.get("integrity_ok", 0.0)),
        "optimality_proven": bool(e.get("optimality_proven", 0.0)),
        "solve_result_num": e.get("solve_result_num"),
        "mip_abs_gap": e.get("mip_abs_gap"),
        "mip_rel_gap": e.get("mip_rel_gap"),
        "solver_time_s": e.get("solver_time_s"),
        "wall_time_s": res.wall_time_s,
        "solution_q": list(sol.q) if sol else None,
        "solution_theta": list(sol.theta) if sol else None,
        "provenance": prov,
        "git_commit": sha,
        "git_dirty": dirty,
        "code_dir": str(_repo_code_dir()),
        "timeout_s": TIMEOUT_S,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    out_dir = Path("results") / "original_ampl_real"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{INSTANCE}_{SOLVER}.json"
    out_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")

    print("=== result ===")
    print(f"status                 : {res.status.value}")
    print(f"cardinality_ok         : {cardinality_ok} | all_finite: {finite_ok}")
    print(f"provider objective     : {e.get('provider_objective')}")
    print(f"rescored objective     : {e.get('rescored_objective')}")
    print(f"|Δ| abs / rel          : {e.get('objective_abs_delta')} / {e.get('objective_rel_delta')}")
    print(f"integrity_ok           : {bool(e.get('integrity_ok', 0.0))}")
    print(f"optimality_proven      : {bool(e.get('optimality_proven', 0.0))}")
    print(f"solve_result_num       : {e.get('solve_result_num')}")
    print(f"mip gap abs/rel        : {e.get('mip_abs_gap')} / {e.get('mip_rel_gap')}")
    print(f"solver_time_s / wall   : {e.get('solver_time_s')} / {res.wall_time_s:.3f}")
    print(f"residual conflicts     : {res.n_conflicts} / {res.n_initial_conflicts} initial")
    print(f"solver_version         : {prov.get('solver_version')} | ampl: {prov.get('ampl_version')}")
    print(f"git                    : {sha} (dirty={dirty})")
    print(f"artifact               : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
