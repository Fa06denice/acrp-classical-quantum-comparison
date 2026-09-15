#!/usr/bin/env python
"""NON-OFFICIAL timeout-recovery for CP_8..CP_20 (analysis only — never an anchor).

Runs ``ampl_gurobi_v3_timeout_recovery`` (v3 scientific options + an INTERNAL
``lim:time=110`` below the external 125 s timeout) so Gurobi terminates gracefully and
its incumbent survives. This protocol is EXPLICITLY non-official: ``accept_official_
anchor`` refuses it by ``protocol_id``, and a feasible incumbent is only ever a
``provisional_feasible_incumbent`` — never an optimum, official baseline or anchor.

Rules: CP_8..CP_20 only, sequential, idempotent resume, atomic write; capture status,
termination, incumbent q/theta, provider+rescored objective, best bound, abs/rel gap,
solver+wall time, common-geometry feasibility, residual conflicts, minimum safety
margin, MP residual (if the warn-mode check ran) and full provenance. No serialized
incumbent => ``no_serialized_incumbent_before_limit`` (null, explicit). Writes to
results/ampl_gurobi_timeout_recovery/; never touches the v3 campaign or any historical
result. No official delta is produced — only clearly-labelled provisional comparisons.
"""

from __future__ import annotations

import json
import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from acrpq.benchmark.separation_diag import separation_diagnostics
from acrpq.classical.ampl_campaign import discrete_reference, git_info, load_instance
from acrpq.classical.ampl_protocol import AMPL_GUROBI_V3_TIMEOUT_RECOVERY as PROTOCOL
from acrpq.classical.original_ampl import _repo_code_dir

INSTANCES = [f"CP_{n}" for n in range(8, 21)]  # CP_8 .. CP_20
OUT_DIR = Path("results") / "ampl_gurobi_timeout_recovery"


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=2, default=str))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _valid_existing(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        art = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return art if art.get("protocol_id") == PROTOCOL.protocol_id else None


def _recover_instance(name: str, git: dict[str, Any]) -> dict[str, Any]:
    inst, dat_text, source = load_instance(name)
    t0 = time.perf_counter()
    res = PROTOCOL.build_solver().solve(inst, dat_text=dat_text)
    wall = time.perf_counter() - t0
    e = res.extra
    prov = json.loads(res.message) if res.message else {}
    sol = res.solution

    art: dict[str, Any] = {
        "protocol_id": PROTOCOL.protocol_id,
        "protocol_is_official": False,
        "protocol": PROTOCOL.as_dict(),
        "instance": name, "n": inst.n, "n_pairs": inst.n_pairs(),
        "dat_source": source,
        "is_official_anchor": False,          # invariant: recovery NEVER anchors
        "anchor_kind": None, "anchor_qualification": None,
        "status": res.status.value,
        "status_kind": prov.get("status_kind"),
        "ampl_solve_result": prov.get("ampl_solve_result"),
        "termination_reason": prov.get("termination_reason") or prov.get("ampl_solve_result"),
        "solve_result_num": e.get("solve_result_num"),
        "solver_time_s": e.get("solver_time_s"),
        "wall_time_s": wall,
        "official_exclusion_reason": None,
        "ampl_version": prov.get("ampl_version"),
        "solver_requested": prov.get("solver_requested"),
        "solver_effective": prov.get("solver_effective"),
        "solver_version": prov.get("solver_version"),
        "solver_options": prov.get("solver_options"),
        "solver_options_echo": prov.get("solver_options_echo"),
        "model_sha256": prov.get("model_sha256"),
        "preprocessing_sha256": prov.get("preprocessing_sha256"),
        "data_sha256": prov.get("data_sha256"),
        "git_commit": git["git_commit"],
        "scientific_code_dirty": git["scientific_code_dirty"],
        "repository_dirty": git["repository_dirty"],
        "code_dir": str(_repo_code_dir()),
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if sol is None:
        art.update({
            "incumbent_kind": "no_serialized_incumbent_before_limit",
            "provisional_feasible_incumbent": None,
            "q": None, "theta": None,
            "provider_objective": None, "rescored_objective": None,
            "best_bound": None, "mip_abs_gap": None, "optimality_gap": None,
            "feasible_common_numeric": None, "n_residual_conflicts": None,
            "minimum_safety_margin": None, "max_model_constraint_residual": None,
            "official_exclusion_reason":
                "no incumbent serialized before the internal solver limit",
        })
        return art

    sep = separation_diagnostics(inst, sol.q, sol.theta)
    provider_obj = e.get("provider_objective")
    abs_gap = e.get("mip_abs_gap")
    rel_gap = e.get("mip_rel_gap")
    best_bound = (provider_obj - abs_gap) if (provider_obj is not None and abs_gap is not None) \
        else None
    feasible_common = sep["feasible_common_numeric"]
    # A limit-time feasible point is ONLY ever a provisional feasible incumbent.
    incumbent_kind = "provisional_feasible_incumbent" if feasible_common \
        else "provisional_infeasible_incumbent"
    art.update({
        "incumbent_kind": incumbent_kind,
        "provisional_feasible_incumbent": {
            "objective": e.get("rescored_objective"),
            "feasible_common_numeric": feasible_common,
            "optimality_gap": rel_gap,
            "note": "NOT an optimum / baseline / anchor — provisional feasible incumbent "
                    "captured at the internal solver time limit",
        } if feasible_common else None,
        "q": list(sol.q), "theta": list(sol.theta),
        "provider_objective": provider_obj,
        "rescored_objective": e.get("rescored_objective"),
        "objective_abs_delta": e.get("objective_abs_delta"),
        "integrity_ok": bool(e.get("integrity_ok", 0.0)),
        "best_bound": best_bound,
        "mip_abs_gap": abs_gap, "optimality_gap": rel_gap,
        "feasible_common_numeric": feasible_common,
        "n_residual_conflicts": res.n_conflicts,
        "minimum_safety_margin": sep["minimum_safety_margin"],
        "feasible_exact_theoretical": sep["feasible_exact_theoretical"],
        "max_model_constraint_residual": e.get("max_model_constraint_residual"),
        "mp_solution_check": prov.get("mp_solution_check"),
        "separation_diagnostics": sep,
        "official_exclusion_reason":
            (f"timeout: relative gap {rel_gap} > official threshold 1e-6 (not certified)"
             if rel_gap is not None else "timeout: no certified gap"),
    })
    # OPTIONAL provisional (NON-official) comparison vs an exhaustively-proven discrete grid.
    if feasible_common:
        disc = discrete_reference(inst, 3)  # K3 demo grid
        if disc["exhaustive_optimality_proven"] and disc["objective"] is not None:
            art["provisional_vs_feasible_incumbent"] = {
                "grid_key": disc["grid_key"],
                "discrete_objective": disc["objective"],
                "provisional_incumbent_objective": e.get("rescored_objective"),
                "provisional_discretisation_cost":
                    disc["objective"] - (e.get("rescored_objective") or 0.0),
                "official": False,
                "label": "provisional_vs_feasible_incumbent",
                "warning": "NON-OFFICIAL: the incumbent is not a certified anchor; this is "
                           "not an official discretisation delta",
            }
    return art


def main() -> int:
    git = git_info()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"NON-OFFICIAL timeout recovery: {PROTOCOL.protocol_id}")
    print(f"instances: {INSTANCES[0]}..{INSTANCES[-1]} | options: {PROTOCOL.solver_options}")
    print(f"git {git['git_commit']} (scientific_code_dirty={git['scientific_code_dirty']}, "
          f"repository_dirty={git['repository_dirty']})")

    rows = []
    for i, name in enumerate(INSTANCES, 1):
        path = OUT_DIR / f"{name}.json"
        existing = _valid_existing(path)
        if existing is not None:
            print(f"[{i}/{len(INSTANCES)}] {name}: resume — skipped")
            art = existing
        else:
            art = _recover_instance(name, git)
            _atomic_write_json(path, art)
            print(f"[{i}/{len(INSTANCES)}] {name}: kind={art['incumbent_kind']} "
                  f"obj={art.get('rescored_objective')} gap={art.get('optimality_gap')} "
                  f"common_feasible={art.get('feasible_common_numeric')} "
                  f"wall={art['wall_time_s']:.0f}s")
        rows.append({
            "instance": name, "n": art.get("n"),
            "incumbent_available": art.get("q") is not None,
            "incumbent_kind": art.get("incumbent_kind"),
            "objective": art.get("rescored_objective"),
            "best_bound": art.get("best_bound"),
            "optimality_gap": art.get("optimality_gap"),
            "feasible_common_numeric": art.get("feasible_common_numeric"),
            "minimum_safety_margin": art.get("minimum_safety_margin"),
            "wall_time_s": art.get("wall_time_s"),
            "official_exclusion_reason": art.get("official_exclusion_reason"),
            "is_official_anchor": False,
        })

    summary = {
        "protocol_id": PROTOCOL.protocol_id, "protocol_is_official": False,
        "instances": INSTANCES, "rows": rows,
        "note": "NON-OFFICIAL analysis. No official anchor or delta. CP_3..CP_7 remain the "
                "only official anchors (results/ampl_gurobi_campaign_v3/).",
        "git_commit": git["git_commit"],
        "scientific_code_dirty": git["scientific_code_dirty"],
        "repository_dirty": git["repository_dirty"],
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _atomic_write_json(OUT_DIR / "recovery_summary.json", summary)
    import csv
    cols = list(rows[0].keys())
    tmp = OUT_DIR / "recovery_summary.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, OUT_DIR / "recovery_summary.csv")
    # Pin every file consumed by the read-only dashboard.  This is tamper-evident
    # (not signed/tamper-proof), but prevents a partial or accidental edit from
    # silently becoming a displayed scientific number.
    names = [f"{name}.json" for name in INSTANCES]
    names += ["recovery_summary.json", "recovery_summary.csv"]
    hashes = {
        name: "sha256:" + hashlib.sha256((OUT_DIR / name).read_bytes()).hexdigest()
        for name in names
    }
    _atomic_write_json(OUT_DIR / "integrity_manifest.json", {
        "schema": "acrpq-recovery-integrity/1",
        "protocol_id": PROTOCOL.protocol_id,
        "sha256": hashes,
    })
    print(f"\nNON-OFFICIAL recovery artifacts -> {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
