"""Objective-aware EXHAUSTIVE discrete optimum — the ground-truth for Phase 2.

This lives OUTSIDE the protected ``reference.py`` (which is quadratic-only and
must not be modified). It enumerates every ``K**n`` one-hot grid assignment and
returns the objective-optimal CONFLICT-FREE assignment for a selected
``objective_id``, together with the number of optima (exact, because it is a
complete enumeration), the decoded controls, the common-geometry conflict count,
and the secondary metrics.

It uses the SAME discrete model as the QUBO (``DiscreteACRP``: identical grid,
option order, NOOP and conflict table), so it is a faithful reference leg for the
three-way equivalence proof ``exhaustive == Gurobi-discrete == Exact-QUBO``.

The discrete classical problem solved here is::

    minimise   primary_objective(objective_id, controls(choice))
    subject to the assignment being CONFLICT-FREE (a hard constraint)

If no conflict-free assignment exists in the grid the instance is discrete-
infeasible (reported honestly, never relaxed).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .. import geometry
from ..discretize import DiscreteACRP
from ..model import Instance, ManeuverGrid
from ..objectives import (
    ObjectiveId,
    coerce_objective_id,
    pick_optimum,
    primary_objective,
    secondary_metrics,
)

# A complete enumeration of K**n; kept modest so the exhaustive proof stays fast.
DEFAULT_MAX_SEARCH = 2_000_000


@dataclass(frozen=True)
class DiscreteEnumResult:
    """Result of an exhaustive objective-aware discrete optimisation."""

    objective_id: str
    status: str  # "optimal" | "infeasible" | "search_too_large"
    feasible: bool
    optimum: float | None
    representative_choice: tuple[int, ...] | None
    # for maneuver_count (integer) this is an EXACT count; for the continuous
    # objective it is the count of optima WITHIN tie_tolerance (see tie_kind).
    n_optima: int
    tie_kind: str  # "exact" | "within_absolute_tolerance"
    tie_tolerance: float
    q: tuple[float, ...]
    theta: tuple[float, ...]
    n_conflicts: int
    grid_k: int
    search_space: int
    secondary: dict[str, float] = field(default_factory=dict)


def exhaustive_discrete_optimum(
    inst: Instance,
    *,
    objective_id: ObjectiveId | str,
    grid: ManeuverGrid | None = None,
    discrete: DiscreteACRP | None = None,
    n_theta: int = 3,
    n_q: int = 1,
    max_search: int = DEFAULT_MAX_SEARCH,
    tol: float = 0.0,
) -> DiscreteEnumResult:
    """Enumerate ``K**n`` grid assignments and return the discrete optimum.

    ``tol`` is the tie tolerance passed to :func:`~acrpq.objectives.pick_optimum`;
    it defaults to ``0.0`` (EXACT optima — required for the integer maneuver-count
    objective and correct for counting exactly-degenerate quadratic optima).
    """
    oid = coerce_objective_id(objective_id)
    if discrete is None:
        if grid is None:
            grid = ManeuverGrid.build(n_theta=n_theta, n_q=n_q, instance=inst)
        discrete = DiscreteACRP.build(inst, grid)
    elif discrete.instance != inst:
        raise ValueError("discrete model was built for a different instance")

    k = discrete.k
    n = inst.n
    space = k**n
    empty = (0.0,) * n
    tie_kind = "exact" if tol == 0.0 else "within_absolute_tolerance"
    if space > max_search:
        return DiscreteEnumResult(
            objective_id=oid.value, status="search_too_large", feasible=False,
            optimum=None, representative_choice=None, n_optima=0,
            tie_kind=tie_kind, tie_tolerance=tol, q=empty,
            theta=empty, n_conflicts=-1, grid_k=k, search_space=space,
        )

    feasible: list[tuple[tuple[int, ...], float]] = []
    for choice in itertools.product(range(k), repeat=n):
        if discrete.assignment_conflicts(choice) == 0:  # conflict-free (feasible)
            q, theta = discrete.maneuvers(choice)
            feasible.append((choice, primary_objective(oid, q, theta, inst.w)))

    if not feasible:
        return DiscreteEnumResult(
            objective_id=oid.value, status="infeasible", feasible=False,
            optimum=None, representative_choice=None, n_optima=0,
            tie_kind=tie_kind, tie_tolerance=tol, q=empty,
            theta=empty, n_conflicts=-1, grid_k=k, search_space=space,
        )

    best = min(value for _, value in feasible)
    rep, n_optima = pick_optimum(feasible, tol=tol)
    assert rep is not None
    q, theta = discrete.maneuvers(rep)
    # mandatory common-geometry rescore (conflicts recomputed by the shared kernel)
    n_conflicts = geometry.count_conflicts(inst, q, theta)
    return DiscreteEnumResult(
        objective_id=oid.value,
        status="optimal",
        feasible=n_conflicts == 0,
        optimum=best,
        representative_choice=rep,
        n_optima=n_optima,
        tie_kind=tie_kind,
        tie_tolerance=tol,
        q=q,
        theta=theta,
        n_conflicts=n_conflicts,
        grid_k=k,
        search_space=space,
        secondary=secondary_metrics(q, theta, inst.w),
    )
