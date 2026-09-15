#!/usr/bin/env python
"""Official AMPL/Gurobi protocol mini-sweep — CP_3, CP_4, CP_5 ONLY (no campaign).

Thin wrapper over :mod:`acrpq.classical.ampl_campaign`: runs the frozen
``ampl_gurobi_supervised_v3`` protocol on the three demo instances, one at a time,
serialising the residual-measurement proof (A) and the scientific-code cleanliness
(C), and writes an official artifact per instance plus a comparison with the
grid-aligned discretisation costs K3/K5/K7. Never overwrites the A–F diagnostics,
the v1/v2 artifacts, or any historical result.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from acrpq.classical.ampl_campaign import (
    DISCRETE_GRIDS,
    build_instance_artifact,
    discretisation_costs,
    git_info,
    load_instance,
)
from acrpq.classical.ampl_protocol import OFFICIAL_PROTOCOL as PROTOCOL

INSTANCES = ["CP_3", "CP_4", "CP_5"]
OUT_DIR = Path("results") / PROTOCOL.protocol_id


def main() -> int:
    git = git_info()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"AMPL/Gurobi protocol mini-sweep: {PROTOCOL.protocol_id}")
    print(f"instances: {INSTANCES} | grids K={list(DISCRETE_GRIDS)} | "
          f"git {git['git_commit']} (scientific_code_dirty={git['scientific_code_dirty']}, "
          f"repository_dirty={git['repository_dirty']})")

    rows = []
    for name in INSTANCES:
        inst, dat_text, source = load_instance(name)
        art = build_instance_artifact(inst, dat_text, source, git, PROTOCOL)
        art["discretisation_costs"] = discretisation_costs(art)
        (OUT_DIR / f"{name}.json").write_text(json.dumps(art, indent=2, default=str),
                                              encoding="utf-8")
        m = art["residual_measurement"]
        print(f"  {name}: status={art['status']} anchor={art['is_official_anchor']} "
              f"gap={art['optimality_gap']} residual={art['max_model_constraint_residual']} "
              f"meas_matches={m.get('solution_matches_primary')} "
              f"Δq={m.get('max_abs_delta_q')} Δθ={m.get('max_abs_delta_theta')}")
        rows.append({"instance": name, **art["discretisation_costs"],
                     "original_ampl_gurobi": {
                         "objective": art["rescored_objective"], "status": art["status"],
                         "is_official_anchor": art["is_official_anchor"],
                         "optimality_gap": art["optimality_gap"],
                         "minimum_safety_margin": art["minimum_safety_margin"]},
                     "adaptation_cost": {"value": None,
                                         "reason": "pyomo_minlp unavailable"},
                     "algorithmic_gap": {"value": None, "official": False,
                                         "reason": "no comparable approximate; quantum not re-run"}})

    comparison = {"protocol_id": PROTOCOL.protocol_id, "protocol": PROTOCOL.as_dict(),
                  "instances": INSTANCES, "rows": rows,
                  "git_commit": git["git_commit"],
                  "scientific_code_dirty": git["scientific_code_dirty"],
                  "repository_dirty": git["repository_dirty"],
                  "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (OUT_DIR / "comparison_CP3_CP4_CP5.json").write_text(
        json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    print(f"\nofficial artifacts -> {OUT_DIR}/<instance>.json (+ comparison)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
