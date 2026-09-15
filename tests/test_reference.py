"""Tests for the exact discrete reference solver."""

from __future__ import annotations

from acrpq.classical.reference import DiscreteReferenceSolver
from acrpq.model import SolverKind, Status


def test_cp3_resolved(cp3):
    res = DiscreteReferenceSolver(n_theta=5).solve(cp3)
    assert res.solver == SolverKind.CLASSICAL_DISCRETE
    assert res.status == Status.OPTIMAL
    assert res.feasible
    assert res.n_conflicts == 0
    assert res.n_initial_conflicts == cp3.n_pairs()


def test_cp4_resolved(cp4):
    res = DiscreteReferenceSolver(n_theta=5).solve(cp4)
    assert res.feasible
    assert res.objective > 0  # some maneuver was needed


def test_finer_grid_not_worse(cp4):
    coarse = DiscreteReferenceSolver(n_theta=3).solve(cp4)
    fine = DiscreteReferenceSolver(n_theta=7).solve(cp4)
    if coarse.feasible and fine.feasible:
        # a finer grid contains the coarse options, so its optimum is <=
        assert fine.objective <= coarse.objective + 1e-9


def test_greedy_fallback_on_budget():
    # tiny search budget forces the greedy heuristic path
    from acrpq.io.loader import InstanceLoader

    inst = InstanceLoader().load("RCP_10_1")
    res = DiscreteReferenceSolver(n_theta=5, max_search_space=10).solve(inst)
    assert "heuristic" in res.message
    # heuristic should still resolve this sparse instance
    assert res.n_conflicts <= res.n_initial_conflicts
