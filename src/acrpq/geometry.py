"""The single ACRP geometry kernel: conflict detection and objective.

This is the **only** place in the package where separation and objective are
computed. Classical-MINLP, classical-discrete and quantum-decoded results all
route through these functions, which is what makes the classical-vs-quantum
comparison valid: every solution is scored on identical geometry.

Faithfulness to ``Code/NC_ACRP.mod``
------------------------------------
Under the 2D uniform-motion model, aircraft ``i`` at position
``(x0_i, y0_i)`` moves with velocity

    V_i = q_i * v0_i,   alpha_i = theta_i + theta0_i
    vel_i = (V_i cos(alpha_i), V_i sin(alpha_i))

(``NC_ACRP.mod`` lines 70-71, with ``theta0`` from line 16). For a pair the
relative position is ``r0 = (x0_i - x0_j, y0_i - y0_j)`` and the relative
velocity is ``vr = vel_i - vel_j``. The minimum separation over future time
``t >= 0`` along the straight-line trajectories ``r(t) = r0 + t*vr`` is

    if |vr|^2 ~ 0:          min_sep^2 = |r0|^2          (no closing speed)
    elif r0 . vr >= 0:      min_sep^2 = |r0|^2          (diverging: closest now)
    else: t* = -(r0.vr)/|vr|^2 ;  min_sep^2 = |r0 + t* vr|^2

A pair is in conflict iff ``min_sep^2 < d^2``.

This closed form was verified numerically (1M+ random samples) to be equivalent
to the ``bin11/bin12 + bin3xx`` big-M tangent-cone disjunction of the ``.mod``
everywhere except the degenerate region ``|r0| < d`` (two aircraft already
inside each other's protection zone at ``t = 0``). That region never occurs in
the library instances (``radius = 2.0`` >> ``d = 0.05``); where it would, the
closed form conservatively flags the existing violation, which is the safer
direction. See ``docs/LIMITATIONS.md``.

Flight levels: if both aircraft carry an initial flight level and the levels
differ, the pair is vertically separated and can never conflict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .model import Instance

_VR_EPS = 1e-15  # below this |vr|^2 the pair has effectively no closing speed
_TOL = 1e-9  # strict-inequality tolerance for the conflict test


def aircraft_velocity(inst: Instance, i: int, q_i: float, theta_i: float) -> tuple[float, float]:
    """Absolute velocity vector of aircraft ``i`` (1-based) under control ``(q_i, theta_i)``."""
    theta0 = inst.theta0
    alpha = theta_i + theta0[i - 1]
    v = q_i * inst.v0[i - 1]
    return v * math.cos(alpha), v * math.sin(alpha)


def relative_velocity(
    inst: Instance, i: int, j: int, q_i: float, t_i: float, q_j: float, t_j: float
) -> tuple[float, float]:
    """Relative velocity ``vel_i - vel_j`` (matches ``cvrx``/``cvry``)."""
    vix, viy = aircraft_velocity(inst, i, q_i, t_i)
    vjx, vjy = aircraft_velocity(inst, j, q_j, t_j)
    return vix - vjx, viy - vjy


def relative_position(inst: Instance, i: int, j: int) -> tuple[float, float]:
    """Relative initial position ``r0 = (x0_i - x0_j, y0_i - y0_j)``."""
    return inst.x0[i - 1] - inst.x0[j - 1], inst.y0[i - 1] - inst.y0[j - 1]


def min_separation_sq(xr0: float, yr0: float, vrx: float, vry: float) -> float:
    """Squared minimum separation over ``t >= 0`` for one pair.

    Pure function of the relative position/velocity so it is trivially testable.
    """
    r0_sq = xr0 * xr0 + yr0 * yr0
    vv = vrx * vrx + vry * vry
    if vv <= _VR_EPS:
        return r0_sq  # parallel / no closing speed
    dot = xr0 * vrx + yr0 * vry
    if dot >= 0.0:
        return r0_sq  # diverging: closest approach is now
    t_star = -dot / vv
    dx = xr0 + t_star * vrx
    dy = yr0 + t_star * vry
    return dx * dx + dy * dy


def levels_separated(inst: Instance, i: int, j: int) -> bool:
    """True if aircraft ``i`` and ``j`` are on different flight levels."""
    if not inst.has_flight_levels:
        return False
    return inst.l0[i - 1] != inst.l0[j - 1]


def in_conflict(
    inst: Instance,
    i: int,
    j: int,
    q_i: float,
    t_i: float,
    q_j: float,
    t_j: float,
    tol: float = _TOL,
) -> bool:
    """True iff pair ``(i, j)`` is in conflict under the given controls."""
    if levels_separated(inst, i, j):
        return False
    xr0, yr0 = relative_position(inst, i, j)
    vrx, vry = relative_velocity(inst, i, j, q_i, t_i, q_j, t_j)
    msq = min_separation_sq(xr0, yr0, vrx, vry)
    return msq < inst.d * inst.d - tol


@dataclass(frozen=True, slots=True)
class ConflictReport:
    """Per-pair conflict diagnostics for a given assignment."""

    i: int
    j: int
    min_separation: float
    required: float  # d
    in_conflict: bool

    @property
    def depth(self) -> float:
        """How far the separation falls short of ``d`` (0 if no conflict)."""
        return max(0.0, self.required - self.min_separation)


def conflict_reports(
    inst: Instance, q: tuple[float, ...], theta: tuple[float, ...]
) -> tuple[ConflictReport, ...]:
    """Conflict report for every pair under the assignment ``(q, theta)``."""
    if len(q) != inst.n or len(theta) != inst.n:
        raise ValueError("q and theta must each have length n")
    reports: list[ConflictReport] = []
    for i, j in inst.pairs():
        if levels_separated(inst, i, j):
            reports.append(ConflictReport(i, j, math.inf, inst.d, False))
            continue
        xr0, yr0 = relative_position(inst, i, j)
        vrx, vry = relative_velocity(inst, i, j, q[i - 1], theta[i - 1], q[j - 1], theta[j - 1])
        msq = min_separation_sq(xr0, yr0, vrx, vry)
        sep = math.sqrt(max(msq, 0.0))
        reports.append(ConflictReport(i, j, sep, inst.d, msq < inst.d * inst.d - _TOL))
    return tuple(reports)


def count_conflicts(inst: Instance, q: tuple[float, ...], theta: tuple[float, ...]) -> int:
    """Number of conflicting pairs under the assignment ``(q, theta)``."""
    return sum(1 for r in conflict_reports(inst, q, theta) if r.in_conflict)


def count_initial_conflicts(inst: Instance) -> int:
    """Conflicts in the do-nothing assignment (``q = 1``, ``theta = 0``)."""
    q = (1.0,) * inst.n
    theta = (0.0,) * inst.n
    return count_conflicts(inst, q, theta)


def objective(inst: Instance, q: tuple[float, ...], theta: tuple[float, ...]) -> float:
    """Total deviation cost (``NC_ACRP.mod`` line 67)."""
    if len(q) != inst.n or len(theta) != inst.n:
        raise ValueError("q and theta must each have length n")
    w = inst.w
    return sum(w * t * t + (1.0 - w) * (1.0 - qq) ** 2 for qq, t in zip(q, theta))


# --------------------------------------------------------------------------- #
# Closest point of approach (CPA) — the single source for trajectory analysis.
# The dashboard renders these values; it never re-derives the geometry itself.
# --------------------------------------------------------------------------- #
def closest_approach_time(xr0: float, yr0: float, vrx: float, vry: float) -> tuple[float, bool]:
    """Time ``t* >= 0`` of minimum separation, and whether the pair is closing.

    ``t* = -(r0 . vr)/|vr|^2`` when the pair is approaching; ``0`` when parallel
    (no closing speed) or already diverging (closest approach is at ``t = 0``).
    The second value is ``True`` iff the pair is closing (``r0 . vr < 0`` and
    ``|vr|^2 > 0``). This is the same relative-velocity geometry as
    :func:`min_separation_sq`; nothing new is introduced.
    """
    vv = vrx * vrx + vry * vry
    if vv <= _VR_EPS:
        return 0.0, False
    dot = xr0 * vrx + yr0 * vry
    if dot >= 0.0:
        return 0.0, False
    return -dot / vv, True


@dataclass(frozen=True, slots=True)
class PairDiagnostic:
    """Full closest-approach diagnostics for one aircraft pair under a solution.

    All times are in the model's abstract time unit (positions in NM/100,
    speeds in (NM/100)/h, so ``t`` carries hours; the dashboard labels it
    "model time unit" unless a unit is proven). ``excluded`` marks a pair on
    different flight levels, which the scorer never counts as a conflict.
    """

    i: int
    j: int
    excluded: bool  # different flight levels -> never a conflict
    initial_conflict: bool  # conflict with no maneuver (q=1, theta=0)
    resolved_conflict: bool  # conflict under the provided (q, theta)
    min_separation: float  # resolved minimum separation
    required: float  # separation norm d
    violation_depth: float  # max(0, d - min_separation)
    t_cpa: float  # time of closest approach (>= 0)
    closing: bool  # True iff the pair is approaching
    xi_cpa: float  # aircraft i position at t_cpa
    yi_cpa: float
    xj_cpa: float  # aircraft j position at t_cpa
    yj_cpa: float


def pair_diagnostic(
    inst: Instance, i: int, j: int, q: tuple[float, ...], theta: tuple[float, ...]
) -> PairDiagnostic:
    """Closest-approach diagnostic for pair ``(i, j)`` under solution ``(q, theta)``."""
    excluded = levels_separated(inst, i, j)
    xr0, yr0 = relative_position(inst, i, j)

    # initial (do-nothing) conflict state
    init_conf = False if excluded else in_conflict(inst, i, j, 1.0, 0.0, 1.0, 0.0)

    # resolved state under the provided controls
    vrx, vry = relative_velocity(inst, i, j, q[i - 1], theta[i - 1], q[j - 1], theta[j - 1])
    msq = min_separation_sq(xr0, yr0, vrx, vry)
    sep = math.sqrt(max(msq, 0.0))
    resolved_conf = (not excluded) and (msq < inst.d * inst.d - _TOL)
    t_cpa, closing = closest_approach_time(xr0, yr0, vrx, vry)

    # absolute positions of each aircraft at t_cpa
    vix, viy = aircraft_velocity(inst, i, q[i - 1], theta[i - 1])
    vjx, vjy = aircraft_velocity(inst, j, q[j - 1], theta[j - 1])
    xi = inst.x0[i - 1] + t_cpa * vix
    yi = inst.y0[i - 1] + t_cpa * viy
    xj = inst.x0[j - 1] + t_cpa * vjx
    yj = inst.y0[j - 1] + t_cpa * vjy

    return PairDiagnostic(
        i=i, j=j, excluded=excluded,
        initial_conflict=init_conf, resolved_conflict=resolved_conf,
        min_separation=sep, required=inst.d,
        violation_depth=max(0.0, inst.d - sep),
        t_cpa=t_cpa, closing=closing,
        xi_cpa=xi, yi_cpa=yi, xj_cpa=xj, yj_cpa=yj,
    )


def pair_diagnostics(
    inst: Instance, q: tuple[float, ...], theta: tuple[float, ...]
) -> tuple[PairDiagnostic, ...]:
    """CPA diagnostics for every aircraft pair under solution ``(q, theta)``."""
    if len(q) != inst.n or len(theta) != inst.n:
        raise ValueError("q and theta must each have length n")
    return tuple(pair_diagnostic(inst, i, j, q, theta) for i, j in inst.pairs())
