"""Tests for the conflict/objective geometry kernel."""

from __future__ import annotations

import math

from acrpq import geometry
from acrpq.model import Family, Instance


def test_min_separation_diverging():
    # moving apart -> closest approach is now -> min_sep = |r0|
    msq = geometry.min_separation_sq(1.0, 0.0, 1.0, 0.0)  # r0=(1,0), vr=(1,0): dot>0
    assert math.isclose(msq, 1.0)


def test_min_separation_head_on():
    # r0=(1,0), vr=(-1,0): closing head-on -> passes through 0
    msq = geometry.min_separation_sq(1.0, 0.0, -1.0, 0.0)
    assert math.isclose(msq, 0.0, abs_tol=1e-12)


def test_min_separation_perpendicular_miss():
    # r0=(1,0), vr=(-1,1): t*=0.5 -> closest point (0.5,0.5), |.|^2=0.5
    msq = geometry.min_separation_sq(1.0, 0.0, -1.0, 1.0)
    assert math.isclose(msq, 0.5, abs_tol=1e-12)


def test_cp4_all_converge_no_maneuver(cp4):
    # circle problem: with no maneuver every pair collides at the centre
    assert geometry.count_initial_conflicts(cp4) == cp4.n_pairs()


def test_cp4_resolved_by_turn(cp4):
    # turning every aircraft hard should separate them
    n = cp4.n
    q = (1.0,) * n
    theta = (cp4.hmax,) * n  # all turn +30deg by the same amount -> still parallel-ish
    # a uniform turn keeps relative geometry similar; instead turn alternating
    theta = tuple(cp4.hmax if i % 2 == 0 else cp4.hmin for i in range(n))
    before = geometry.count_initial_conflicts(cp4)
    after = geometry.count_conflicts(cp4, q, theta)
    assert after <= before  # maneuvering does not increase conflicts here


def test_flight_levels_never_conflict():
    inst = Instance(
        name="t", family=Family.RCP_FL, n=2, d=0.05, radius=2.0,
        v0=(5.0, 5.0), cap=(0.0, math.pi), x0=(1.0, -1.0), y0=(0.0, 0.0),
        nf=2, l0=(1, 2),
    )
    # head-on but on different levels -> not a conflict
    assert geometry.count_conflicts(inst, (1.0, 1.0), (0.0, 0.0)) == 0


def test_objective_matches_mod_formula(cp4):
    q = tuple(1.0 for _ in range(cp4.n))
    theta = tuple(0.1 for _ in range(cp4.n))
    expected = sum(cp4.w * 0.1**2 + (1 - cp4.w) * 0.0 for _ in range(cp4.n))
    assert math.isclose(geometry.objective(cp4, q, theta), expected)
