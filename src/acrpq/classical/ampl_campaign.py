"""Shared per-instance logic for the AMPL/Gurobi protocol sweep and campaign.

Builds ONE fully-provenanced artifact per instance under a frozen protocol:

* the primary v3 solve (fatal MP check at 1e-8);
* a warn-mode residual MEASUREMENT solve whose PROOF is serialised — options,
  echo, per-variable deltas, match tolerance, measured residual, violation table,
  times and hashes (A). No ``Δ=0`` is claimed unless the deltas are published;
* the grid-aligned discrete references K3/K5/K7, each recording its exhaustive
  proof (only when ``K^n`` is within the search cap; otherwise an explicit
  non-proven status and NO official delta);
* the fail-closed anchor decision (``evaluate_anchor``), which consumes the
  measurement block and the scientific-code cleanliness (C).

This module never touches ``Code/`` or any historical/quantum result.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from typing import Any

from ..benchmark.separation_diag import pi_convention_check, separation_diagnostics
from ..io.loader import InstanceLoader
from .ampl_protocol import OFFICIAL_PROTOCOL, AmplProtocol, evaluate_anchor, grid_key
from .original_ampl import _repo_code_dir

SCIENTIFIC_CODE_PATHS = ["src", "scripts", "Code", "tests", "pyproject.toml"]
DISCRETE_GRIDS = (3, 5, 7)  # K = n_theta with n_q=1; K3 is the demo protocol


def git_info() -> dict[str, Any]:
    """Short SHA + SCIENTIFIC-code cleanliness (C).

    ``scientific_code_dirty`` is scoped to the reproducibility-relevant paths; the
    full-tree ``repository_dirty`` is published SEPARATELY and never conflated with it.
    """
    def _run(args: list[str]) -> tuple[int, str]:
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=5)
            return p.returncode, p.stdout
        except (OSError, subprocess.SubprocessError):
            return 1, ""

    rc_sha, sha = _run(["git", "rev-parse", "--short", "HEAD"])
    rc_sci, sci = _run(["git", "status", "--porcelain", "--", *SCIENTIFIC_CODE_PATHS])
    rc_repo, repo = _run(["git", "status", "--porcelain"])
    return {
        "git_commit": sha.strip() or None,
        "scientific_code_dirty": bool(sci.strip()) if rc_sci == 0 else None,
        "scientific_code_paths": list(SCIENTIFIC_CODE_PATHS),
        "repository_dirty": bool(repo.strip()) if rc_repo == 0 else None,
    }


def check_license_not_expired(expiry: str, *, now: datetime | None = None) -> tuple[bool, str]:
    """Return (ok, iso_today). ``ok`` is False once ``today >= expiry`` (YYYY-MM-DD)."""
    today = (now or datetime.now(timezone.utc)).date()
    exp = datetime.strptime(expiry, "%Y-%m-%d").date()
    return today < exp, today.isoformat()


def measure_residual(inst, dat_text: str, primary_sol, protocol: AmplProtocol) -> dict[str, Any]:
    """Warn-mode MEASUREMENT solve; serialises the full reproduction proof (A)."""
    tol = protocol.measurement_match_tol
    base = {"performed": False, "solve_options_note": protocol.mp_measure_options,
            "solution_match_tolerance": tol, "solution_matches_primary": False}
    if protocol.mp_measure_options is None or primary_sol is None:
        return base
    r = protocol.build_solver(solver_options=protocol.mp_measure_options).solve(
        inst, dat_text=dat_text)
    prov = json.loads(r.message) if r.message else {}
    sol = r.solution
    report: dict[str, Any] = {
        "performed": True,
        "status": r.status.value,
        "solve_result_num": r.extra.get("solve_result_num"),
        "solver_options_requested": protocol.mp_measure_options,
        "solver_options_echo": prov.get("solver_options_echo"),
        "solution_match_tolerance": tol,
        "measured_max_abs_mp_residual": r.extra.get("max_model_constraint_residual"),
        "mp_violation_table": prov.get("mp_solution_check"),
        "solver_time_s": r.extra.get("solver_time_s"),
        "wall_time_s": r.wall_time_s,
        "model_sha256": prov.get("model_sha256"),
        "data_sha256": prov.get("data_sha256"),
        "preprocessing_sha256": prov.get("preprocessing_sha256"),
    }
    if sol is None or len(sol.q) != len(primary_sol.q) or len(sol.theta) != len(primary_sol.theta):
        report.update({"solution_matches_primary": False,
                       "max_abs_delta_q": None, "max_abs_delta_theta": None})
        return report
    dq = max((abs(a - b) for a, b in zip(sol.q, primary_sol.q)), default=None)
    dt = max((abs(a - b) for a, b in zip(sol.theta, primary_sol.theta)), default=None)
    report["max_abs_delta_q"] = dq
    report["max_abs_delta_theta"] = dt
    report["solution_matches_primary"] = (
        dq is not None and dt is not None and dq <= tol and dt <= tol)
    return report


def discrete_reference(inst, n_theta: int, n_q: int = 1) -> dict[str, Any]:
    """Grid-aligned discrete optimum with an EXPLICIT exhaustive-search proof.

    When ``K^n`` exceeds the exact-search cap, the status is a non-proven heuristic
    and ``exhaustive_optimality_proven`` is false (no official delta will be built).
    """
    from ..model import ManeuverGrid
    from .ampl_protocol import discrete_optimum_proven
    from .reference import DiscreteReferenceSolver

    grid = ManeuverGrid.build(n_theta=n_theta, n_q=n_q, instance=inst)
    solver = DiscreteReferenceSolver(n_theta=n_theta, n_q=n_q)
    search_space = grid.k ** inst.n
    exhaustive = search_space <= solver.max_search_space
    res = solver.solve(inst)
    sol = res.solution
    proven = discrete_optimum_proven(
        status=res.status.value, search_space_size=search_space,
        search_cap=solver.max_search_space)
    return {
        "grid_key": grid_key(n_theta, n_q),
        "n_theta": n_theta, "n_q": n_q, "K": grid.k,
        "theta_levels": list(grid.theta_levels), "q_levels": list(grid.q_levels),
        "search_space_size": search_space, "search_space_cap": solver.max_search_space,
        "within_search_cap": exhaustive,
        "mode": "exact" if exhaustive else "heuristic",
        "states_explored": None,
        "status": res.status.value,
        "objective": res.objective if sol else None,
        "feasible_common_numeric": (res.n_conflicts == 0 and res.feasible) if sol else None,
        "n_residual_conflicts": res.n_conflicts if sol else None,
        "exhaustive_optimality_proven": proven,
    }


def build_instance_artifact(inst, dat_text: str, source: str, git: dict[str, Any],
                            protocol: AmplProtocol = OFFICIAL_PROTOCOL) -> dict[str, Any]:
    """Run one instance under ``protocol`` and return its full artifact dict."""
    solver = protocol.build_solver()
    res = solver.solve(inst, dat_text=dat_text)     # PRIMARY (fatal MP check)
    e = dict(res.extra)
    prov = json.loads(res.message) if res.message else {}
    sol = res.solution
    q = tuple(sol.q) if sol else None
    theta = tuple(sol.theta) if sol else None

    measurement = measure_residual(inst, dat_text, sol, protocol) \
        if (res.status.value == "optimal" and sol is not None) else {"performed": False}
    if measurement.get("measured_max_abs_mp_residual") is not None:
        e["max_model_constraint_residual"] = measurement["measured_max_abs_mp_residual"]

    sep = separation_diagnostics(inst, sol.q, sol.theta) if sol else None
    pi = pi_convention_check(inst, sol.q, sol.theta) if sol else None
    prov_for_anchor = {**prov, "protocol_id": protocol.protocol_id}
    anchor = evaluate_anchor(
        status=res.status.value, extra=e, provenance=prov_for_anchor,
        feasible_common_numeric=(sep["feasible_common_numeric"] if sep else None),
        n=inst.n, q=q, theta=theta, measurement=measurement,
        scientific_code_dirty=git["scientific_code_dirty"], protocol=protocol,
    )
    provider_obj = e.get("provider_objective")
    abs_gap = e.get("mip_abs_gap")
    best_bound = (provider_obj - abs_gap) if (provider_obj is not None and abs_gap is not None) \
        else None
    discretes = {f"K{k}_q1": discrete_reference(inst, k) for k in DISCRETE_GRIDS}

    return {
        "protocol_id": protocol.protocol_id,
        "protocol": protocol.as_dict(),
        "instance": inst.name, "n": inst.n, "n_pairs": inst.n_pairs(),
        "dat_source": source,
        "ampl_solve_result": prov.get("ampl_solve_result"),
        "solve_result_num": e.get("solve_result_num"),
        "status": res.status.value,
        "status_kind": prov.get("status_kind"),
        "anchor_kind": anchor["anchor_kind"],
        "is_official_anchor": anchor["is_anchor"],
        "anchor_criteria": anchor["criteria"],
        "anchor_qualification": anchor["anchor_qualification"],
        "incumbent_objective_provider": provider_obj,
        "rescored_objective": e.get("rescored_objective"),
        "objective_abs_delta": e.get("objective_abs_delta"),
        "objective_rel_delta": e.get("objective_rel_delta"),
        "integrity_ok": bool(e.get("integrity_ok", 0.0)),
        "best_bound": best_bound,
        "mip_abs_gap": abs_gap,
        "optimality_gap": e.get("mip_rel_gap"),
        "solver_time_s": e.get("solver_time_s"),
        "wall_time_s": res.wall_time_s,
        "q": list(q) if q else None,
        "theta": list(theta) if theta else None,
        "n_initial_conflicts": res.n_initial_conflicts,
        "n_residual_conflicts": res.n_conflicts if sol else None,
        "separation_diagnostics": sep,
        "minimum_safety_margin": sep["minimum_safety_margin"] if sep else None,
        "feasible_common_numeric": sep["feasible_common_numeric"] if sep else None,
        "feasible_exact_theoretical": sep["feasible_exact_theoretical"] if sep else None,
        "max_model_constraint_residual": e.get("max_model_constraint_residual"),
        "residual_measurement": measurement,
        "max_abs_mp_residual_threshold": protocol.max_abs_mp_residual,
        "max_rel_optimality_gap_threshold": protocol.max_rel_optimality_gap,
        "pi_convention_check": pi,
        "discrete_references": discretes,
        "ampl_version": prov.get("ampl_version"),
        "solver_requested": prov.get("solver_requested"),
        "solver_effective": prov.get("solver_effective"),
        "solver_version": prov.get("solver_version"),
        "solver_driver_version": prov.get("solver_driver_version"),
        "solver_driver_version_available": prov.get("solver_driver_version") is not None,
        "solver_options": prov.get("solver_options"),
        "solver_options_echo": prov.get("solver_options_echo"),
        "model_sha256": prov.get("model_sha256"),
        "preprocessing_sha256": prov.get("preprocessing_sha256"),
        "data_sha256": prov.get("data_sha256"),
        "git_commit": git["git_commit"],
        "scientific_code_dirty": git["scientific_code_dirty"],
        "scientific_code_paths": git["scientific_code_paths"],
        "repository_dirty": git["repository_dirty"],
        "code_dir": str(_repo_code_dir()),
        "timeout_s": protocol.timeout_s,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def discretisation_costs(artifact: dict[str, Any]) -> dict[str, Any]:
    """Named per-grid discretisation deltas; official only for a valid anchor + proven grid."""
    anchored = artifact["is_official_anchor"]
    orig = artifact["rescored_objective"] if anchored else None
    out: dict[str, Any] = {}
    for key, disc in artifact["discrete_references"].items():
        proven = disc["exhaustive_optimality_proven"] and disc["objective"] is not None
        if orig is not None and proven:
            out[f"discretisation_cost_{key}"] = {
                "grid_key": disc["grid_key"], "value": disc["objective"] - orig,
                "ratio": (disc["objective"] / orig) if orig not in (0.0, None) else None,
                "official": True, "discrete_objective": disc["objective"],
                "search_space_size": disc["search_space_size"], "mode": disc["mode"],
            }
        else:
            out[f"discretisation_cost_{key}"] = {
                "grid_key": disc["grid_key"], "value": None, "official": False,
                "reason": ("original leg not a valid anchor" if orig is None
                           else "discrete reference not an exhaustively-proven optimum "
                                f"(within_cap={disc['within_search_cap']}, status={disc['status']})"),
                "discrete_status": disc["status"], "mode": disc["mode"],
            }
    return out


def load_instance(name: str, loader: InstanceLoader | None = None):
    loader = loader or InstanceLoader()
    inst = loader.load(name)
    dat_text, source = loader.read_dat_text(name)
    return inst, dat_text, source
