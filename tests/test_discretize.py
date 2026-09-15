"""Tests for the discrete reformulation."""

from __future__ import annotations

from acrpq.discretize import discretize
from acrpq.model import ManeuverGrid


def test_grid_has_noop():
    grid = ManeuverGrid.build(n_theta=5, n_q=3)
    assert any(o.is_noop for o in grid.options)
    assert grid.k == 15


def test_grid_noop_only_when_single_level():
    grid = ManeuverGrid.build(n_theta=1, n_q=1)
    assert grid.k == 1
    assert grid.options[0].is_noop


def test_var_index_bijection(cp3):
    d = discretize(cp3, n_theta=5)
    seen = set()
    for i in cp3.aircraft():
        for o in range(d.k):
            b = d.var_index(i, o)
            assert d.inv_index(b) == (i, o)
            seen.add(b)
    assert seen == set(range(d.n_vars))


def test_conflict_table_symmetry(cp4):
    d = discretize(cp4, n_theta=3)
    for (i, j), mat in d.conflict.items():
        assert len(mat) == d.k
        assert all(len(row) == d.k for row in mat)


def test_assignment_conflicts_consistent_with_geometry(cp4):
    from acrpq import geometry

    d = discretize(cp4, n_theta=5)
    # all-NOOP choice
    noop = d.grid.noop_index()
    choice = tuple(noop for _ in cp4.aircraft())
    q, theta = d.maneuvers(choice)
    assert d.assignment_conflicts(choice) == geometry.count_conflicts(cp4, q, theta)
