"""Discrete reformulation of the ACRP shared by the classical reference solver
and the quantum QUBO.

Each aircraft chooses exactly one maneuver from a small grid of ``K`` options
(:class:`~acrpq.model.ManeuverGrid`). This collapses the continuous control box
``(q, theta)`` to ``K`` options per aircraft — the *only* simplification needed
to make the problem quantum-tractable. The ACRP's separation structure is
already strictly pairwise in ``Code/NC_ACRP.mod`` (one disjunction per pair,
depending only on the two aircraft's controls) and its objective is separable,
so a pairwise conflict table captures the constraint coupling *exactly*; nothing
many-body is lost, only the continuous control resolution.

Binary variable layout (canonical, load-bearing for qubit indexing):

    x[i, o] in {0, 1}    aircraft i in 1..n chooses option o in 0..K-1
    bit_index(i, o) = (i - 1) * K + o
    total binary variables = n * K

The conflict table ``conflict[(i, j)][oi][oj]`` is a ``K x K`` boolean matrix
per aircraft pair, precomputed once via the geometry kernel; it is the sole
source of geometry the quantum layer ever consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import geometry
from .model import Instance, ManeuverGrid


@dataclass(frozen=True, slots=True)
class DiscreteACRP:
    """The discretised ACRP: a maneuver grid plus a precomputed conflict table."""

    instance: Instance
    grid: ManeuverGrid
    n_vars: int
    # (i, j) with i < j -> K x K matrix; conflict[(i,j)][oi][oj] is True iff
    # aircraft i taking option oi and j taking option oj are in conflict.
    conflict: dict[tuple[int, int], tuple[tuple[bool, ...], ...]]
    cost: tuple[float, ...]  # cost[o] = per-aircraft objective term for option o

    @property
    def n(self) -> int:
        return self.instance.n

    @property
    def k(self) -> int:
        return self.grid.k

    def var_index(self, i: int, o: int) -> int:
        """Bit index of ``x[i, o]`` (aircraft ``i`` 1-based, option ``o``)."""
        return (i - 1) * self.grid.k + o

    def inv_index(self, b: int) -> tuple[int, int]:
        """Inverse of :meth:`var_index`: bit index -> (aircraft id, option idx)."""
        k = self.grid.k
        return b // k + 1, b % k

    def maneuvers(self, choice: tuple[int, ...]) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Map a per-aircraft option choice to ``(q, theta)`` tuples."""
        if len(choice) != self.n:
            raise ValueError("choice must have length n")
        opts = self.grid.options
        q = tuple(opts[o].q for o in choice)
        theta = tuple(opts[o].theta for o in choice)
        return q, theta

    def assignment_conflicts(self, choice: tuple[int, ...]) -> int:
        """Conflicts implied by ``choice`` *using the discrete table* (fast).

        This counts conflicts from the precomputed ``K x K`` tables, which is
        exactly the geometry of the chosen maneuvers. (The canonical scoring of
        a final solution still goes through :mod:`acrpq.geometry`, which is
        identical for these discrete maneuvers.)
        """
        total = 0
        for (i, j), mat in self.conflict.items():
            if mat[choice[i - 1]][choice[j - 1]]:
                total += 1
        return total

    @staticmethod
    def build(inst: Instance, grid: ManeuverGrid) -> "DiscreteACRP":
        """Precompute the conflict table and per-option cost for ``inst``."""
        opts = grid.options
        k = grid.k
        conflict: dict[tuple[int, int], tuple[tuple[bool, ...], ...]] = {}
        for i, j in inst.pairs():
            rows: list[tuple[bool, ...]] = []
            for oi in range(k):
                a = opts[oi]
                row = tuple(
                    geometry.in_conflict(
                        inst, i, j, a.q, a.theta, opts[oj].q, opts[oj].theta
                    )
                    for oj in range(k)
                )
                rows.append(row)
            conflict[(i, j)] = tuple(rows)
        cost = tuple(grid.cost(o, inst.w) for o in opts)
        return DiscreteACRP(
            instance=inst,
            grid=grid,
            n_vars=inst.n * k,
            conflict=conflict,
            cost=cost,
        )


def discretize(inst: Instance, *, n_theta: int = 5, n_q: int = 1) -> DiscreteACRP:
    """Convenience: build a maneuver grid for ``inst`` and discretise it."""
    grid = ManeuverGrid.build(n_theta=n_theta, n_q=n_q, instance=inst)
    return DiscreteACRP.build(inst, grid)
