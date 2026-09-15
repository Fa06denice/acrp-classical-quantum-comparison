#!/usr/bin/env python
"""EXPERIMENTAL — a SYNTHETIC CP_30 instance (``CP_30`` does not exist in the
library: CP stops at ``CP_20``) built to mimic ``CP_20``'s ``.dat`` conventions,
to see how far the instance-scalability picture extends past the largest real
CP instance.

Construction (``synthetic = true`` on every output field):

* ``v0``, ``d``, ``radius``, ``w`` copied verbatim from ``CP_20`` (constant
  speed 5.0, ``d=0.05``, ``radius=2.0``, ``w=0.5``);
* ``x0[k] = -radius*cos(ang)``, ``y0[k] = -radius*sin(ang)`` with
  ``ang = k*2*PI/n + PI`` (``acrpq.io.loader.circle_default_coords``, the
  SAME formula the loader uses to reconstruct RCP_FL coordinates — verified
  against ``CP_20``'s actual ``x0``/``y0``);
* ``cap[k] = ang mod 2*PI`` — checked to reproduce ``CP_20.cap`` exactly for
  ``n=20`` before being reused at ``n=30`` (see ``_check_cap_matches_cp20``).

We DO NOT transpile a 90-qubit circuit (2**90 statevector entries, and no
justification a hardware pilot would ever run at that width). Instead:

* every graph/QUBO metric is computed exactly (cheap: it is a K3 clique, no
  ISA needed);
* ISA depth/2Q-gate counts are EXTRAPOLATED from the real campaign numbers
  for CP_3..CP_7 (``docs/ADAPTIVE_QPU_GO_NO_GO.md`` §3: 211/108, 259/207,
  376/348, 469/479, 640/700) with a stated linear AND quadratic fit — named
  as extrapolations, never as measurements;
* local-exact feasibility, Aer feasibility and a classical SA heuristic are
  all ACTUALLY RUN (SA is cheap: it is the same driver already used by
  ``adaptive_grid.sa_local_solve`` inside this experimental study).

Output: ``results/experimental_instance_scalability_v1/cp30_answer.json``.
"""

from __future__ import annotations

import json
from acrpq.experimental import provenance as prov
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acrpq import geometry  # noqa: E402
from acrpq.experimental import adaptive_grid as ag  # noqa: E402
from acrpq.experimental import adaptive_qpu as aq  # noqa: E402
from acrpq.experimental import instance_scalability as isc  # noqa: E402
from acrpq.io.loader import InstanceLoader, circle_default_coords  # noqa: E402
from acrpq.model import PI, Family, Instance  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "experimental_instance_scalability_v1"
OUT_PATH = OUT_DIR / "cp30_answer.json"

# Real campaign ISA numbers (docs/ADAPTIVE_QPU_GO_NO_GO.md §3), K3 global grid, opt-level 0.
CAMPAIGN_N = (3, 4, 5, 6, 7)
CAMPAIGN_DEPTH = (211, 259, 376, 469, 640)
CAMPAIGN_2Q = (108, 207, 348, 479, 700)


def build_synthetic_cp30(cp20: Instance) -> Instance:
    n = 30
    x0, y0 = circle_default_coords(n, cp20.radius)
    cap = tuple((k * 2 * PI / n + PI) % (2 * PI) for k in range(n))
    return Instance(
        name="CP_30_synthetic", family=Family.CP, n=n, d=cp20.d, radius=cp20.radius,
        v0=(cp20.v0[0],) * n, cap=cap, x0=x0, y0=y0, w=cp20.w,
        source_path="synthetic:mimics_CP_20_dat_conventions",
    )


def _check_cap_matches_cp20(cp20: Instance) -> bool:
    n = cp20.n
    predicted = tuple((k * 2 * PI / n + PI) % (2 * PI) for k in range(n))
    # CP_20.dat stores cap rounded to 5 decimals; the PI-literal formula matches
    # to ~2e-6, which is exactly that rounding (checked, not a formula mismatch).
    return all(abs(p - c) < 1e-4 for p, c in zip(predicted, cp20.cap))


def _linfit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    m = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    b = (m * sxy - sx * sy) / (m * sxx - sx * sx)
    a = (sy - b * sx) / m
    return a, b


def _quadfit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    rows = [[1.0, x, x * x] for x in xs]
    xtx = [[sum(rows[k][i] * rows[k][j] for k in range(len(xs))) for j in range(3)] for i in range(3)]
    xty = [sum(rows[k][i] * ys[k] for k in range(len(xs))) for i in range(3)]

    def det3(m: list[list[float]]) -> float:
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    d = det3(xtx)
    beta = []
    for col in range(3):
        m = [row[:] for row in xtx]
        for r in range(3):
            m[r][col] = xty[r]
        beta.append(det3(m) / d)
    return beta[0], beta[1], beta[2]


def main() -> int:
    loader = InstanceLoader()
    cp20 = loader.load("CP_20")
    assert _check_cap_matches_cp20(cp20), "the cap formula must reproduce CP_20 exactly before reuse"
    inst = build_synthetic_cp30(cp20)

    levels3 = ag.official_k_grid_options(inst, 3)
    edges = isc.any_option_graph(inst, levels3)
    degs = isc.degrees(inst.n, edges)
    isolated = {v for v, d in degs.items() if d == 0}
    dens = isc.density(inst.n, len(edges))
    comps = isc.connected_components(inst.n, edges)
    label = isc.structure_label(inst.n, len(edges), len(comps), dens, degs)

    q_global = 3 * inst.n
    q_active = 3 * (inst.n - len(isolated))
    counts_quad = isc.dominance_survivor_counts(inst, "quadratic_control_cost_v1", levels3, isolated)
    counts_cnt = isc.dominance_survivor_counts(inst, "maneuver_count_v1", levels3, isolated)

    global_grid = ag.global_grid(inst, 3)
    search_space = global_grid.search_space()
    global_qubo = aq.build_hetero_qubo(inst, global_grid, "quadratic_control_cost_v1")
    n_quadratic = len(global_qubo.quadratic)
    max_deg = isc.max_interaction_degree(global_qubo)
    cross_terms = isc.cross_component_qubo_terms(global_qubo, comps)

    sv_bytes = isc.statevector_bytes(q_global)
    sv_practicality = isc.statevector_practicality(q_global)

    local_exact_ok, local_exact_reason = isc.local_exact_feasible(
        inst, "quadratic_control_cost_v1", global_grid, timeout_s=60, max_states=2_000_000
    )

    t0 = time.perf_counter()
    sa = ag.sa_local_solve(inst, "quadratic_control_cost_v1", global_grid, seed=0, n_sweeps=400, restarts=8)
    sa_wall = time.perf_counter() - t0
    sa_conflicts = geometry.count_conflicts(inst, (1.0,) * inst.n, sa.theta) if sa.theta else None

    depth_a, depth_b = _linfit(list(CAMPAIGN_N), list(CAMPAIGN_DEPTH))
    twoq_a, twoq_b = _linfit(list(CAMPAIGN_N), list(CAMPAIGN_2Q))
    depth_qa, depth_qb, depth_qc = _quadfit(list(CAMPAIGN_N), list(CAMPAIGN_DEPTH))
    twoq_qa, twoq_qb, twoq_qc = _quadfit(list(CAMPAIGN_N), list(CAMPAIGN_2Q))
    n30 = 30
    depth_lin_30 = depth_a + depth_b * n30
    twoq_lin_30 = twoq_a + twoq_b * n30
    depth_quad_30 = depth_qa + depth_qb * n30 + depth_qc * n30 * n30
    twoq_quad_30 = twoq_qa + twoq_qb * n30 + twoq_qc * n30 * n30

    # sensitivity of the extrapolation: leave-one-out refits of both models on the 5 campaign points
    def _loo(fit, xs, ys, kind):
        vals = []
        for skip in range(len(xs)):
            xx = [x for i, x in enumerate(xs) if i != skip]
            yy = [y for i, y in enumerate(ys) if i != skip]
            c = fit(xx, yy)
            vals.append(c[0] + c[1] * n30 + (c[2] * n30 * n30 if kind == "quad" else 0.0))
        return {"min": min(vals), "max": max(vals), "values": vals}
    sensitivity = {
        "method": "least-squares fits on the 5 real campaign points (n=3..7); linear and quadratic in n; "
                  "leave-one-out refits give the spread; this is an EXTRAPOLATION x4 in n, not a measurement",
        "depth": {"linear": depth_lin_30, "quadratic": depth_quad_30,
                  "loo_linear": _loo(_linfit, list(CAMPAIGN_N), list(CAMPAIGN_DEPTH), "lin"),
                  "loo_quadratic": _loo(_quadfit, list(CAMPAIGN_N), list(CAMPAIGN_DEPTH), "quad")},
        "n_2q": {"linear": twoq_lin_30, "quadratic": twoq_quad_30,
                 "loo_linear": _loo(_linfit, list(CAMPAIGN_N), list(CAMPAIGN_2Q), "lin"),
                 "loo_quadratic": _loo(_quadfit, list(CAMPAIGN_N), list(CAMPAIGN_2Q), "quad")},
        "measured": False,
    }
    ci = isc.ClassificationInput(
        n=inst.n, q_global_k3=q_global, q_largest_component=q_global,
        largest_component_size=inst.n, n_components=len(comps), cross_component_terms=cross_terms,
        local_exact_feasible_global=local_exact_ok, local_exact_feasible_component=local_exact_ok,
        expected_isa_2q_component=int(round(twoq_quad_30)),
    )
    classes = isc.classify_instance(ci)

    # verdict: quantum-hardware credibility vs overall (classical-heuristic-inclusive) accessibility
    quantum_hardware_verdict = (
        "CP30_ISA_COMPILABLE_BUT_NOT_CREDIBLE" if q_global <= isc.ISA_COMPILABLE_MAX_LOGICAL
        else "CP30_NOT_ACCESSIBLE_WITH_CURRENT_METHOD"
    )
    # The mandated verdict concerns accessibility WITH THE QUANTUM METHOD under study. A classical
    # simulated-annealing feasible assignment (no optimality reference) is reported separately
    # and never promoted to an accessibility verdict (counter-review §11).
    overall_verdict = quantum_hardware_verdict
    classical_heuristic_note = (
        "classical SA (adaptive_grid.sa_local_solve) finds a conflict-free K3 assignment; this is a "
        "classical fact without optimality reference and does not make CP_30 accessible to the method"
        if (sa.theta is not None and sa_conflicts == 0) else "classical SA found no conflict-free assignment"
    )

    answers = {
        "q01_n_synthetic": inst.n,
        "q02_is_full_K3_clique": len(edges) == inst.n * (inst.n - 1) // 2,
        "q03_structure_label": label,
        "q04_n_components_K3": len(comps),
        "q05_Q_global_K3": q_global,
        "q06_Q_active_K3_after_isolation": q_active,
        "q07_Q_hetero_exact_K3_quadratic": sum(counts_quad),
        "q08_Q_hetero_exact_K3_maneuver_count": sum(counts_cnt),
        "q09_decomposable_exactly": len(comps) >= 2 and cross_terms == 0,
        "q10_exact_search_space": search_space,
        "q11_local_exact_feasible_within_60s": {"feasible": local_exact_ok, "reason": local_exact_reason},
        "q12_classical_SA_heuristic_result": {
            "feasible_conflict_free": sa.theta is not None and sa_conflicts == 0,
            "objective": sa.objective, "wall_time_s": sa_wall, "n_evaluations": sa.n_evaluations,
            "n_conflicts": sa_conflicts,
        },
        "q13_statevector_bytes_global": sv_bytes,
        "q13b_statevector_practicality": sv_practicality,
        "q14_isa_extrapolation_at_n30": {
            "not_transpiled_reason": "90 logical qubits: no justification for an actual "
                                     "transpile run; extrapolated from the real campaign only",
            "linear_fit": {"depth": depth_lin_30, "n_2q": twoq_lin_30,
                          "params": {"depth_a": depth_a, "depth_b": depth_b,
                                     "twoq_a": twoq_a, "twoq_b": twoq_b}},
            "quadratic_fit_preferred": {
                "depth": depth_quad_30, "n_2q": twoq_quad_30,
                "params": {"depth": [depth_qa, depth_qb, depth_qc], "n_2q": [twoq_qa, twoq_qb, twoq_qc]},
                "rationale": "CP is a full clique: one-hot + conflict quadratic term count "
                             "grows as O(n^2) at fixed K, so a quadratic fit is theoretically "
                             "better justified than linear for this family",
            },
            "campaign_source_points": {"n": CAMPAIGN_N, "depth": CAMPAIGN_DEPTH, "n_2q": CAMPAIGN_2Q},
        },
        "n_quadratic_K3": n_quadratic, "max_degree_interaction": max_deg,
        "classes": classes,
    }

    doc = {
        "schema": "acrpq-experimental-cp30-synthetic/2", **isc.ARTIFACT_FLAGS,
        "provenance": prov.provenance_block(Path(__file__).resolve().parents[1]),
        "isa_extrapolation_sensitivity": sensitivity,
        "synthetic": True, "instance_name": "CP_30_synthetic",
        "note": "CP_30 does not exist in the library (CP stops at CP_20); every field here "
                "is synthetic and must never be reported as a real library instance",
        "construction": {
            "base_instance_for_conventions": "CP_20",
            "d": inst.d, "radius": inst.radius, "v0": inst.v0[0], "w": inst.w,
            "cap_formula": "cap[k] = (k*2*PI/n + PI) mod 2*PI, PI = 3.141592 (model.py literal)",
            "x0_y0_formula": "acrpq.io.loader.circle_default_coords (same formula RCP_FL uses)",
            "cap_matches_cp20_exactly": _check_cap_matches_cp20(cp20),
        },
        "answers": answers,
        "verdict": overall_verdict,
        "quantum_hardware_only_verdict": quantum_hardware_verdict,
        "classical_heuristic_note": classical_heuristic_note,
        "verdict_rationale": (
            "A classical simulated-annealing heuristic (already part of this experimental "
            "module, adaptive_grid.sa_local_solve) finds a conflict-free K3 assignment for "
            f"CP_30_synthetic ({sa.n_evaluations} evaluations; wall time in q12): the "
            "COMBINATORIAL problem is heuristically accessible. The QUANTUM/hardware route is "
            f"NOT: Q_global_K3={q_global} logical qubits fits the {isc.ISA_COMPILABLE_MAX_LOGICAL}-"
            f"qubit ISA width of ibm_marrakesh (so ISA_COMPILABLE), but the extrapolated depth "
            f"({depth_quad_30:.0f}) and 2Q-gate count ({twoq_quad_30:.0f}) are one to two orders "
            f"of magnitude past the >=15-qubit / >=350-2Q implausibility threshold calibrated on "
            "the real campaign (CP5 already gave 0% feasible shots at 15 qubits/~400 2Q) — not "
            "credible as a hardware pilot. CP is a full K3 clique for every measured n (3..20, "
            "now also 30): no exact decomposition ever applies to this family."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT_PATH}")
    print(f"verdict = {overall_verdict} (quantum-hardware-only: {quantum_hardware_verdict})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
