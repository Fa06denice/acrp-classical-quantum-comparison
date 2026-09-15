#!/usr/bin/env python
"""CP_4-only diagnostic campaign: isolate the source of the borderline separation.

Runs the historical NC_ACRP.mod on verbatim CP_4.dat with AMPL + Gurobi under four
solver-option variants (separate artifacts), capturing per-pair separation *depths*
(not just conflict counts), the three feasibility regimes (strict / solver-tolerance
/ operational), the effective solver options echoed by the driver, and a PI-literal-
vs-math.pi sensitivity check. It NEVER touches Code/ and runs one small instance
only — NOT a multi-instance campaign.

Variants:
  A control            : (no options)                 — witness for the default
  B funcnonlinear      : pre:funcnonlinear=1          — force TRUE nonlinear funcs
  C funcnl + feastol   : pre:funcnonlinear=1 alg:feastol=1e-9
  D + strict MP check  : C + sol:chk:feastol=1e-9 sol:chk:fail  (flag, no argument!)
  E C + mip:gap=1e-6   : tighten the relative optimality gap to the protocol threshold
  F C + mip:gap=1e-8   : tighten it further

Local AMPL/Gurobi only; not an IBM/QPU operation. The licence file is never read.
"""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from acrpq.benchmark.separation_diag import pi_convention_check, separation_diagnostics
from acrpq.classical.original_ampl import OriginalAMPLSolver, _repo_code_dir
from acrpq.io.loader import InstanceLoader

INSTANCE = "CP_4"
SOLVER = "gurobi"
TIMEOUT_S = 120.0

_C = "pre:funcnonlinear=1 alg:feastol=1e-9"
VARIANTS = {
    "A_control": "",
    "B_funcnonlinear": "pre:funcnonlinear=1",
    "C_funcnonlinear_feastol": _C,
    # NOTE: sol:chk:fail is a FLAG (no '=1'); an argument is rejected and fails the solve.
    "D_strict_mpcheck": f"{_C} sol:chk:feastol=1e-9 sol:chk:fail",
    "E_mipgap_1e-6": f"{_C} mip:gap=1e-6",
    "F_mipgap_1e-8": f"{_C} mip:gap=1e-8",
}

# Proposed protocol threshold for the OFFICIAL relative optimality gap.
PROTOCOL_GAP = 1e-6


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


def _run_variant(inst, dat_text, name, opts, git):
    solver = OriginalAMPLSolver(solver=SOLVER, solver_options=opts, timeout_s=TIMEOUT_S)
    res = solver.solve(inst, dat_text=dat_text)
    e = res.extra
    prov = json.loads(res.message) if res.message else {}
    sol = res.solution
    # On solver failure the runner returns solution=None -> q/theta stay null, NEVER
    # fabricated zeros.
    q = list(sol.q) if sol else None
    theta = list(sol.theta) if sol else None

    sep = separation_diagnostics(inst, sol.q, sol.theta) if sol else None
    pi = pi_convention_check(inst, sol.q, sol.theta) if sol else None

    provider_obj = e.get("provider_objective")
    abs_gap = e.get("mip_abs_gap")
    rel_gap = e.get("mip_rel_gap")
    best_bound = (provider_obj - abs_gap) if (provider_obj is not None and abs_gap is not None) \
        else None
    residual = e.get("max_model_constraint_residual")  # None => below the solver's check tol

    feasible_exact = bool(sep and sep["feasible_exact_theoretical"])
    feasible_common = bool(sep and sep["feasible_common_numeric"])
    min_margin = sep["minimum_safety_margin"] if sep else None
    cardinality_ok = sol is not None and len(sol.q) == inst.n and len(sol.theta) == inst.n
    finite_ok = sol is not None and all(math.isfinite(v) for v in (*sol.q, *sol.theta))

    # Proposed OFFICIAL-anchor criteria (a proposal, evaluated per variant).
    anchor = {
        "status_optimal": res.status.value == "optimal",
        "gap_le_protocol": rel_gap is not None and rel_gap <= PROTOCOL_GAP,
        "finite_and_cardinality_ok": cardinality_ok and finite_ok,
        "provider_matches_rescored": bool(e.get("integrity_ok", 0.0)),
        "accepted_by_common_geometry": feasible_common,
        "no_material_constraint_residual": residual is None or residual <= PROTOCOL_GAP,
        "provenance_complete": all(prov.get(k) for k in
                                   ("model_sha256", "data_sha256", "ampl_version")),
    }
    meets_anchor = all(anchor.values())

    artifact = {
        "variant": name,
        "solver_options_requested": opts,
        "solver_options_echo": prov.get("solver_options_echo"),
        "instance": INSTANCE,
        "status": res.status.value,
        "solve_result": prov.get("ampl_solve_result"),
        "solve_result_num": e.get("solve_result_num"),
        "optimality_proven": bool(e.get("optimality_proven", 0.0)),
        "provider_objective": provider_obj,
        "rescored_objective": e.get("rescored_objective"),
        "objective_abs_delta": e.get("objective_abs_delta"),
        "objective_rel_delta": e.get("objective_rel_delta"),
        "integrity_ok": bool(e.get("integrity_ok", 0.0)),
        "best_bound": best_bound,
        "mip_abs_gap": abs_gap,
        "optimality_gap": rel_gap,
        "solver_time_s": e.get("solver_time_s"),
        "wall_time_s": res.wall_time_s,
        "q": q,
        "theta": theta,
        # Kept separately per the counter-review terminology:
        "feasible_exact_theoretical": feasible_exact,
        "feasible_common_numeric": feasible_common,
        "minimum_safety_margin": min_margin,
        "max_model_constraint_residual": residual,
        "mp_solution_check": prov.get("mp_solution_check"),
        "solver_constraint_acceptance": {
            "definition": "no model-constraint residual above the solver's own check "
                          "tolerance (residual is in constraint units, NOT a distance)",
            "max_model_constraint_residual": residual,
            "residual_reported": residual is not None,
            "acceptable_at_protocol_gap": residual is None or residual <= PROTOCOL_GAP,
        },
        "separation_diagnostics": sep,
        "pi_convention_check": pi,
        "meets_official_anchor_criteria": meets_anchor,
        "anchor_criteria_detail": anchor,
        "ampl_version": prov.get("ampl_version"),
        "solver_version": prov.get("solver_version"),
        "solver_driver_version": prov.get("solver_driver_version"),
        "provenance": prov,
        "git_commit": git[0],
        "git_dirty": git[1],
        "code_dir": str(_repo_code_dir()),
        "timeout_s": TIMEOUT_S,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return res, artifact


def main() -> int:
    loader = InstanceLoader()
    inst = loader.load(INSTANCE)
    dat_text, source = loader.read_dat_text(INSTANCE)
    git = _git_state()

    out_dir = Path("results") / "original_ampl_diag"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== CP_4 diagnostic campaign ({source}, d={inst.d}) ===")
    print(f"solver: {SOLVER} | timeout: {TIMEOUT_S:.0f}s | protocol_gap={PROTOCOL_GAP:g} | "
          f"git: {git[0]} (dirty={git[1]})\n")
    print(f"{'variant':26s} {'status':9s} {'exact':6s} {'common':7s} "
          f"{'min_margin':12s} {'gap':10s} {'residual':10s} {'anchor':6s}")

    summary = []
    for name, opts in VARIANTS.items():
        _, art = _run_variant(inst, dat_text, name, opts, git)
        (out_dir / f"{INSTANCE}_variant_{name}.json").write_text(
            json.dumps(art, indent=2, default=str), encoding="utf-8")
        mm = art["minimum_safety_margin"]
        gap = art["optimality_gap"]
        res_ = art["max_model_constraint_residual"]
        print(f"{name:26s} {art['status']:9s} "
              f"{str(art['feasible_exact_theoretical']):6s} "
              f"{str(art['feasible_common_numeric']):7s} "
              f"{(f'{mm:+.2e}' if mm is not None else '-'):12s} "
              f"{(f'{gap:.2e}' if gap is not None else '-'):10s} "
              f"{(f'{res_:.2e}' if res_ is not None else '<tol'):10s} "
              f"{str(art['meets_official_anchor_criteria']):6s}")
        summary.append({
            "variant": name, "status": art["status"],
            "feasible_exact_theoretical": art["feasible_exact_theoretical"],
            "feasible_common_numeric": art["feasible_common_numeric"],
            "minimum_safety_margin": mm,
            "optimality_gap": gap,
            "max_model_constraint_residual": res_,
            "meets_official_anchor_criteria": art["meets_official_anchor_criteria"],
            "options": opts,
        })

    (out_dir / f"{INSTANCE}_summary.json").write_text(
        json.dumps({"summary": summary, "protocol_gap": PROTOCOL_GAP,
                    "git_commit": git[0], "git_dirty": git[1]}, indent=2), encoding="utf-8")
    print(f"\nartifacts -> {out_dir}/{INSTANCE}_variant_*.json (+ summary)")
    anchors = [s["variant"] for s in summary if s["meets_official_anchor_criteria"]]
    cands = [s["variant"] for s in summary
             if s["feasible_common_numeric"] and s["status"] == "optimal"]
    print(f"common-numeric-feasible candidate config(s): {cands or 'NONE'}")
    print(f"meets ALL proposed official-anchor criteria: {anchors or 'NONE yet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
