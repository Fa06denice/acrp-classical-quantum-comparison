"""Matplotlib visualisations for the thesis (OPTIONAL — ``[viz]`` extra).

Each function imports matplotlib lazily and returns the created ``Figure`` so
callers can further customise or save it. Plots cover the brief's recommended
charts: aircraft trajectories, the pairwise conflict matrix, classical-vs-quantum
objective comparison, and problem size vs qubit count.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .._optional import require
from ..model import Instance, Solution


def plot_trajectories(
    inst: Instance,
    solution: Solution | None = None,
    *,
    horizon: float = 1.0,
    ax=None,
):
    """Plot initial positions and resolved trajectories (resolved if a solution)."""
    require("matplotlib")
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    q = solution.q if solution else (1.0,) * inst.n
    theta = solution.theta if solution else (0.0,) * inst.n
    theta0 = inst.theta0
    for i in inst.aircraft():
        x0, y0 = inst.x0[i - 1], inst.y0[i - 1]
        ang = theta[i - 1] + theta0[i - 1]
        v = q[i - 1] * inst.v0[i - 1]
        dx, dy = v * math.cos(ang) * horizon, v * math.sin(ang) * horizon
        ax.plot([x0], [y0], "ko", markersize=6)
        ax.annotate(str(i), (x0, y0), textcoords="offset points", xytext=(5, 5))
        ax.arrow(x0, y0, dx, dy, head_width=0.05, length_includes_head=True, alpha=0.7)
    ax.set_aspect("equal")
    ax.set_title(f"{inst.name}: trajectories" + ("" if solution else " (no maneuvers)"))
    ax.set_xlabel("x (NM/100)")
    ax.set_ylabel("y (NM/100)")
    return ax.figure


def plot_conflict_matrix(inst: Instance, solution: Solution | None = None, *, ax=None):
    """Heat-map of pairwise minimum separation (red = conflict)."""
    from .. import geometry

    require("matplotlib")
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    q = solution.q if solution else (1.0,) * inst.n
    theta = solution.theta if solution else (0.0,) * inst.n
    n = inst.n
    mat = [[float("nan")] * n for _ in range(n)]
    for r in geometry.conflict_reports(inst, q, theta):
        val = min(r.min_separation, inst.d * 3)
        mat[r.i - 1][r.j - 1] = val
        mat[r.j - 1][r.i - 1] = val
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=inst.d * 3)
    ax.figure.colorbar(im, ax=ax, label="min separation (capped at 3d)")
    ax.set_title(f"{inst.name}: pairwise separation")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(1, n + 1))
    ax.set_yticklabels(range(1, n + 1))
    return ax.figure


def plot_benchmark_objective(records, *, ax=None):
    """Grouped bar chart of objective by scenario and solver."""
    require("matplotlib")
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    by_scenario: dict[str, dict[str, float]] = {}
    for rec in records:
        by_scenario.setdefault(rec.scenario_id, {})[rec.solver_kind] = rec.common.objective
    scenarios = list(by_scenario)
    solvers = sorted({s for d in by_scenario.values() for s in d})
    width = 0.8 / max(len(solvers), 1)
    for k, solver in enumerate(solvers):
        xs = [i + k * width for i in range(len(scenarios))]
        ys = [by_scenario[s].get(solver, 0.0) for s in scenarios]
        ax.bar(xs, ys, width=width, label=solver)
    ax.set_xticks([i + width * (len(solvers) - 1) / 2 for i in range(len(scenarios))])
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("objective")
    ax.set_title("Classical vs quantum objective")
    ax.legend()
    return ax.figure


def plot_problem_size_vs_qubits(records, *, ax=None):
    """Scatter of number of aircraft vs qubit count for quantum runs."""
    require("matplotlib")
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    pts: list[tuple[int, int]] = [
        (rec.n, rec.quantum.n_qubits) for rec in records if rec.quantum
    ]
    if pts:
        xs, ys = zip(*sorted(pts))
        ax.plot(xs, ys, "o-")
    ax.set_xlabel("number of aircraft")
    ax.set_ylabel("number of qubits")
    ax.set_title("Problem size vs qubit count")
    return ax.figure


def save_campaign_figures(campaign, out_dir: str = "results/figures") -> list[str]:
    """Render the thesis figures from a :class:`CampaignResult`."""
    require("matplotlib")
    import matplotlib.pyplot as plt
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pts = campaign.points
    saved: list[str] = []

    # 1) number of aircraft vs qubits (size scaling)
    fig, ax = plt.subplots(figsize=(6, 4))
    seen = sorted({(p.n, p.n_qubits) for p in pts})
    if seen:
        xs, ys = zip(*seen)
        ax.plot(xs, ys, "o-")
    ax.set_xlabel("number of aircraft")
    ax.set_ylabel("number of qubits")
    ax.set_title("Problem size vs qubit count")
    p = out / "size_vs_qubits.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    saved.append(str(p))

    # 2) feasibility rate vs QAOA depth, one line per scenario
    fig, ax = plt.subplots(figsize=(7, 4))
    by_sc: dict[str, list] = {}
    for pt in pts:
        by_sc.setdefault(pt.scenario_id, []).append(pt)
    for sc, cells in by_sc.items():
        cells = sorted(cells, key=lambda c: c.reps)
        ax.plot([c.reps for c in cells], [c.feasible_rate * 100 for c in cells], "o-", label=sc)
    ax.set_xlabel("QAOA depth p (reps)")
    ax.set_ylabel("feasibility rate (%)")
    ax.set_title("QAOA feasibility vs circuit depth")
    ax.legend(fontsize=8)
    p = out / "feasibility_vs_depth.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    saved.append(str(p))

    # 3) mean approximation ratio vs QAOA depth
    fig, ax = plt.subplots(figsize=(7, 4))
    for sc, cells in by_sc.items():
        cells = sorted(cells, key=lambda c: c.reps)
        ratio_xs = [c.reps for c in cells if c.mean_approx_ratio is not None]
        ratio_ys = [c.mean_approx_ratio for c in cells if c.mean_approx_ratio is not None]
        if ratio_xs:
            ax.plot(ratio_xs, ratio_ys, "o-", label=sc)
    ax.axhline(1.0, color="grey", ls="--", lw=1, label="classical optimum")
    ax.set_xlabel("QAOA depth p (reps)")
    ax.set_ylabel("mean approximation ratio (1 = optimal)")
    ax.set_title("QAOA solution quality vs circuit depth")
    ax.legend(fontsize=8)
    p = out / "approx_ratio_vs_depth.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    saved.append(str(p))

    return saved


def save_all(records: Sequence, out_dir: str = "results/figures") -> list[str]:
    """Render and save the benchmark summary figures; returns saved paths."""
    require("matplotlib")
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    fig1 = plot_benchmark_objective(list(records))
    p1 = out / "objective_by_scenario.png"
    fig1.savefig(p1, dpi=120, bbox_inches="tight")
    saved.append(str(p1))
    fig2 = plot_problem_size_vs_qubits(list(records))
    p2 = out / "size_vs_qubits.png"
    fig2.savefig(p2, dpi=120, bbox_inches="tight")
    saved.append(str(p2))
    return saved
