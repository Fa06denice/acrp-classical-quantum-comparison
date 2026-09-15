"""Exact discrete reference solver — pure stdlib, always runnable.

Solves the *discretised* ACRP exactly: each aircraft picks one option from the
maneuver grid; the solver finds the assignment of minimum total maneuver cost
with **zero** residual conflicts. This is the ground-truth anchor for the
benchmark — QAOA is measured against it (``approx_ratio``), not the other way
round.

It is exact only on the discretised model: it can report INFEASIBLE when no
grid combination resolves all conflicts even though a continuous solution
exists, and when feasible its objective is generally >= the true continuous
(Couenne) optimum because the grid cannot hit the exact continuous
``(q, theta)``. Both behaviours are expected and documented; the optional
:class:`~acrpq.classical.pyomo_minlp.PyomoMINLPSolver` provides the true
continuous reference when a MINLP solver is available.

Algorithm: depth-first branch-and-bound over aircraft, ordering options by cost
(cheapest first, so NOOP is tried first), pruning a partial assignment as soon
as (a) a placed pair already conflicts or (b) the partial cost reaches the
incumbent. A search-space guard (``K ** n``) avoids blowing up on large
instances; above it the solver falls back to a documented greedy + local-search
heuristic and marks the status accordingly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .. import geometry
from ..discretize import DiscreteACRP
from ..model import (
    Instance,
    ManeuverGrid,
    Result,
    Solution,
    SolverKind,
    Status,
)

_DEFAULT_MAX_SEARCH = 10_000_000  # K**n cap before falling back to heuristic


@dataclass
class DiscreteReferenceSolver:
    """Exact brute-force solver over a maneuver grid.

    Parameters
    ----------
    n_theta, n_q:
        Grid resolution if no grid/discrete model is supplied to :meth:`solve`.
    max_search_space:
        If ``K ** n`` exceeds this, use the greedy heuristic instead of exact
        search (recorded as ``Status.FEASIBLE``/``INFEASIBLE`` with a note).
    """

    n_theta: int = 5
    n_q: int = 1
    max_search_space: int = _DEFAULT_MAX_SEARCH

    def solve(
        self,
        inst: Instance,
        *,
        time_limit_s: float | None = None,
        discrete: DiscreteACRP | None = None,
    ) -> Result:
        t0 = time.perf_counter()
        if discrete is None:
            grid = ManeuverGrid.build(n_theta=self.n_theta, n_q=self.n_q, instance=inst)
            discrete = DiscreteACRP.build(inst, grid)

        n = inst.n
        k = discrete.k
        n_initial = geometry.count_initial_conflicts(inst)

        exact_feasible = k**n <= self.max_search_space if n > 0 else True
        if exact_feasible:
            choice, status, note = self._exact(discrete, time_limit_s, t0)
        else:
            choice, status, note = self._greedy(discrete)

        q, theta = discrete.maneuvers(choice)
        n_conf = geometry.count_conflicts(inst, q, theta)
        obj = geometry.objective(inst, q, theta)
        feasible = n_conf == 0
        sol = Solution(
            q=q,
            theta=theta,
            choice=choice,
            objective=obj,
            n_conflicts=n_conf,
            feasible=feasible,
        )
        if status is None:
            status = Status.OPTIMAL if feasible else Status.INFEASIBLE
        return Result(
            instance_name=inst.name,
            solver=SolverKind.CLASSICAL_DISCRETE,
            status=status,
            solution=sol,
            objective=obj,
            n_conflicts=n_conf,
            n_initial_conflicts=n_initial,
            feasible=feasible,
            wall_time_s=time.perf_counter() - t0,
            backend="discrete",
            n_qubits=discrete.n_vars,
            message=note,
            extra={"grid_k": float(k)},
        )

    # ------------------------------------------------------------------ #
    def _exact(
        self,
        d: DiscreteACRP,
        time_limit_s: float | None,
        t0: float,
    ) -> tuple[tuple[int, ...], Status | None, str]:
        """Exact branch-and-bound; returns (choice, status_override, note)."""
        n = d.n
        k = d.k
        # Option order: cheapest first (NOOP is cost 0 -> tried first).
        order = sorted(range(k), key=lambda o: d.cost[o])
        # Pair lookup by aircraft for incremental conflict checks.
        pairs_with = {i: [] for i in range(1, n + 1)}
        for (i, j), mat in d.conflict.items():
            pairs_with[i].append((j, mat))
            pairs_with[j].append((i, mat))

        best_cost = [float("inf")]
        best_choice: list[tuple[int, ...]] = [tuple([d.grid.noop_index()] * n)]
        found_feasible = [False]
        assign = [-1] * (n + 1)  # 1-based; assign[i] = option for aircraft i

        def conflicts_with_assigned(i: int, oi: int) -> bool:
            for j, mat in pairs_with[i]:
                oj = assign[j]
                if oj == -1:
                    continue
                # mat is indexed [option_of_i'th_aircraft][option_of_j'th] where
                # the matrix key was the ordered pair; orientation handled below.
                if _pair_conflict(d, i, j, oi, oj):
                    return True
            return False

        def recurse(i: int, cost: float) -> None:
            if time_limit_s is not None and (time.perf_counter() - t0) > time_limit_s:
                return
            if cost >= best_cost[0]:
                return
            if i > n:
                # complete, conflict-free (pruning guaranteed feasibility)
                best_cost[0] = cost
                best_choice[0] = tuple(assign[1:])
                found_feasible[0] = True
                return
            for o in order:
                new_cost = cost + d.cost[o]
                if new_cost >= best_cost[0]:
                    break  # options are cost-sorted; rest are no better
                if conflicts_with_assigned(i, o):
                    continue
                assign[i] = o
                recurse(i + 1, new_cost)
                assign[i] = -1

        recurse(1, 0.0)
        if found_feasible[0]:
            return best_choice[0], Status.OPTIMAL, "exact search"
        # No conflict-free assignment exists in this grid: report the
        # min-conflict assignment for diagnostics (INFEASIBLE).
        choice = self._min_conflict(d)
        return choice, Status.INFEASIBLE, "no conflict-free assignment in grid (exact)"

    def _min_conflict(self, d: DiscreteACRP) -> tuple[int, ...]:
        """Greedy minimum-conflict assignment used as an infeasible fallback."""
        return self._greedy(d)[0]

    def _greedy(self, d: DiscreteACRP) -> tuple[tuple[int, ...], Status | None, str]:
        """Greedy + local search: minimise conflicts, then cost. Heuristic."""
        n = d.n
        k = d.k
        noop = d.grid.noop_index()
        assign = [noop] * (n + 1)  # 1-based

        def residual_for(i: int, oi: int) -> int:
            c = 0
            for (a, b), mat in d.conflict.items():
                if a == i:
                    if _pair_conflict(d, i, b, oi, assign[b]):
                        c += 1
                elif b == i:
                    if _pair_conflict(d, a, i, assign[a], oi):
                        c += 1
            return c

        # Local search: repeatedly move the most-conflicted aircraft to its
        # best option (fewest conflicts, then cheapest), until stable.
        improved = True
        passes = 0
        while improved and passes < 50 * n:
            improved = False
            passes += 1
            for i in range(1, n + 1):
                best_o = assign[i]
                best_key = (residual_for(i, assign[i]), d.cost[assign[i]])
                for o in range(k):
                    key = (residual_for(i, o), d.cost[o])
                    if key < best_key:
                        best_key, best_o = key, o
                if best_o != assign[i]:
                    assign[i] = best_o
                    improved = True

        choice = tuple(assign[1:])
        total = d.assignment_conflicts(choice)
        status = Status.FEASIBLE if total == 0 else Status.INFEASIBLE
        note = "greedy/local-search heuristic (search space exceeded cap)"
        return choice, status, note


def _pair_conflict(d: DiscreteACRP, i: int, j: int, oi: int, oj: int) -> bool:
    """Conflict lookup honouring the canonical ``i < j`` table orientation."""
    if i < j:
        return d.conflict[(i, j)][oi][oj]
    return d.conflict[(j, i)][oj][oi]
