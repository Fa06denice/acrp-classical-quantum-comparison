"""Separation diagnostics — sign-correct feasibility terminology.

Controlled 2-aircraft instances with zero relative velocity (identical-velocity
tracks) so the minimum separation equals the initial distance |r0| exactly and the
violation depth / safety margin are analytically known.
"""

from __future__ import annotations

import math

from acrpq.benchmark.separation_diag import (
    pi_convention_check,
    separation_diagnostics,
)
from acrpq.model import Family, Instance


def _two_ac(d: float, gap: float = 0.05) -> Instance:
    # both aircraft same velocity (cap=0, v0=1, theta=0) -> vr=0 -> min_sep = gap
    return Instance(name="T2", family=Family.CP, n=2, d=d, radius=2.0,
                    v0=(1.0, 1.0), cap=(0.0, 0.0), x0=(0.0, gap), y0=(0.0, 0.0))


_STILL = (1.0, 1.0), (0.0, 0.0)  # q, theta with no maneuver


def test_violation_depth_is_exact() -> None:
    diag = separation_diagnostics(_two_ac(d=0.05 + 5e-7), *_STILL)
    p = diag["pairs"][0]
    assert math.isclose(p["min_separation"], 0.05, abs_tol=1e-12)
    assert math.isclose(p["violation_depth"], 5e-7, rel_tol=1e-6)
    assert math.isclose(diag["max_violation_depth"], 5e-7, rel_tol=1e-6)


def test_positive_depth_is_never_exact_theoretical() -> None:
    # even a 6.7e-12 shortfall is NOT exact theoretical feasibility
    for depth in (5e-7, 6.7e-12, 1e-3):
        diag = separation_diagnostics(_two_ac(d=0.05 + depth), *_STILL)
        assert diag["max_violation_depth"] > 0.0
        assert diag["feasible_exact_theoretical"] is False
        assert diag["feasibility_regimes"]["exact_theoretical"]["feasible"] is False


def test_exact_separation_is_exact_theoretical() -> None:
    # required strictly below the separation -> no shortfall in any pair
    diag = separation_diagnostics(_two_ac(d=0.05 - 1e-6), *_STILL)
    assert diag["max_violation_depth"] == 0.0
    assert diag["feasible_exact_theoretical"] is True
    assert diag["minimum_safety_margin"] > 0.0


def test_safety_margin_sign_matches_separation_vs_d() -> None:
    # margin > 0 only when separation > d; = boundary; < 0 when short of d
    assert separation_diagnostics(_two_ac(d=0.05 - 1e-4), *_STILL)["minimum_safety_margin"] > 0
    at_boundary = separation_diagnostics(_two_ac(d=0.05), *_STILL)["minimum_safety_margin"]
    assert math.isclose(at_boundary, 0.0, abs_tol=1e-12)
    assert separation_diagnostics(_two_ac(d=0.05 + 1e-4), *_STILL)["minimum_safety_margin"] < 0


def test_common_numeric_can_pass_while_exact_fails() -> None:
    # a 6.7e-12 shortfall: passes the scorer's documented numeric tolerance but is
    # NOT exact-theoretical feasible and has a (marginally) negative safety margin
    diag = separation_diagnostics(_two_ac(d=0.05 + 6.7e-12), *_STILL)
    assert diag["feasible_common_numeric"] is True     # within scorer tolerance
    assert diag["feasible_exact_theoretical"] is False  # positive depth
    assert diag["minimum_safety_margin"] < 0.0          # never called "positive margin"


def test_large_violation_fails_common_numeric() -> None:
    diag = separation_diagnostics(_two_ac(d=0.05 + 5e-3), *_STILL)
    assert diag["feasible_common_numeric"] is False
    assert diag["feasible_exact_theoretical"] is False
    assert diag["depth_distribution_counts"]["depth_gt_1e-3"] == 1


def test_pi_convention_check_quantifies_theta0_shift() -> None:
    inst = Instance(name="T2p", family=Family.CP, n=2, d=0.05, radius=2.0,
                    v0=(5.0, 5.0), cap=(4.71239, 0.0), x0=(2.0, -2.0), y0=(0.0, 0.0))
    pi = pi_convention_check(inst, (1.0, 1.0), (0.0, 0.0))
    assert 1e-7 < pi["max_abs_theta0_diff"] < 1e-5
    assert pi["theta0_literal_pi"] != pi["theta0_math_pi"]
