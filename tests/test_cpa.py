"""Closest-point-of-approach geometry tests (pure, analytic cases)."""

from __future__ import annotations

import math

from acrpq import geometry
from acrpq.model import Family, Instance


def _pair_instance(cap_i, cap_j, x, y, *, d=0.05, v0=5.0, nf=None, l0=()):
    """Two-aircraft instance with explicit headings/positions for CPA tests."""
    return Instance(
        name="t", family=Family.CP, n=2, d=d, radius=2.0,
        v0=(v0, v0), cap=(cap_i, cap_j), x0=(x[0], x[1]), y0=(y[0], y[1]),
        nf=nf, l0=l0,
    )


# ---- pure closest_approach_time ------------------------------------------ #
def test_head_on_closing():
    t, closing = geometry.closest_approach_time(1.0, 0.0, -1.0, 0.0)
    assert closing is True
    assert math.isclose(t, 1.0)
    assert math.isclose(geometry.min_separation_sq(1.0, 0.0, -1.0, 0.0), 0.0, abs_tol=1e-12)


def test_parallel_not_closing():
    # relative velocity perpendicular to r0 -> dot = 0 -> not closing, t*=0
    t, closing = geometry.closest_approach_time(1.0, 0.0, 0.0, 1.0)
    assert closing is False
    assert t == 0.0


def test_diverging():
    t, closing = geometry.closest_approach_time(1.0, 0.0, 1.0, 0.0)
    assert closing is False
    assert t == 0.0


def test_near_zero_relative_velocity():
    t, closing = geometry.closest_approach_time(1.0, 0.0, 1e-9, 0.0)
    assert closing is False
    assert t == 0.0


def test_perpendicular_miss():
    # r0=(1,0), vr=(-1,1): t* = 0.5, min_sep^2 = 0.5
    t, closing = geometry.closest_approach_time(1.0, 0.0, -1.0, 1.0)
    assert closing is True
    assert math.isclose(t, 0.5)
    assert math.isclose(geometry.min_separation_sq(1.0, 0.0, -1.0, 1.0), 0.5)


# ---- pair_diagnostic ------------------------------------------------------ #
def test_pair_diagnostic_head_on_positions():
    # i at (1,0) heading west (pi), j at (-1,0) heading east (0): head-on, meet at origin
    inst = _pair_instance(math.pi, 0.0, (1.0, -1.0), (0.0, 0.0))
    d = geometry.pair_diagnostic(inst, 1, 2, (1.0, 1.0), (0.0, 0.0))
    assert d.closing is True
    assert d.resolved_conflict is True  # they collide (min_sep 0 < d)
    # both reach ~origin at CPA
    assert abs(d.xi_cpa) < 1e-3 and abs(d.xj_cpa) < 1e-3


def test_pair_diagnostic_excluded_flight_levels():
    inst = _pair_instance(math.pi, 0.0, (1.0, -1.0), (0.0, 0.0), nf=2, l0=(1, 2))
    d = geometry.pair_diagnostic(inst, 1, 2, (1.0, 1.0), (0.0, 0.0))
    assert d.excluded is True
    assert d.resolved_conflict is False  # different FL never conflicts
    assert d.initial_conflict is False


def test_pair_diagnostic_exactly_at_limit_is_not_conflict():
    # place a perpendicular miss whose min separation equals d exactly
    # r0=(d,0) with j offset so CPA separation == d; use a clean miss of exactly d
    inst = _pair_instance(0.0, 0.0, (0.0, 0.0), (0.0, 0.05))  # parallel, static offset d
    d = geometry.pair_diagnostic(inst, 1, 2, (1.0, 1.0), (0.0, 0.0))
    # same heading/speed -> parallel, separation constant = d -> strict '<' => no conflict
    assert d.resolved_conflict is False
    assert math.isclose(d.min_separation, 0.05, abs_tol=1e-9)


def test_pair_diagnostics_matches_conflict_reports(cp4):
    q = (1.0,) * cp4.n
    theta = (0.0,) * cp4.n
    diags = geometry.pair_diagnostics(cp4, q, theta)
    reports = geometry.conflict_reports(cp4, q, theta)
    assert len(diags) == len(reports) == cp4.n_pairs()
    for dg, rp in zip(diags, reports):
        assert dg.resolved_conflict == rp.in_conflict
