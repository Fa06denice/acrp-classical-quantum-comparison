"""Versioned, strictly-separated optimisation objectives for the discrete ACRP.

There are exactly two objectives, each carrying an immutable string *identity*
(``ObjectiveId``). That identity MUST travel with every request, result,
artifact and hash so results computed under different objectives are never
mixed, compared, or ranked as if they were the same problem.

``quadratic_control_cost_v1`` — the historical NC_ACRP objective (``model.py``
line references NC_ACRP.mod line 67)::

    J(q, theta) = sum_i [ w * theta_i^2 + (1 - w) * (1 - q_i)^2 ]

Kept because it measures the discretisation cost and enables the
classical-discrete <-> QUBO comparison at an identical objective.

``maneuver_count_v1`` — the supervisor's cardinality objective::

    C = sum_i y_i,   y_i = 0  iff  (q_i, theta_i) == (1, 0)   else   1

i.e. the number of aircraft given ANY non-NOOP command, independent of the
command's type or magnitude. In the one-hot QUBO this is exactly
``C(x) = sum_i sum_{o != noop} x[i, o]`` (each non-NOOP option costs 1).

Design rules enforced here:

* One-hot and conflict conditions are CONSTRAINTS / penalties, never objective
  terms — this module only defines the *objective*; penalties live in
  :mod:`acrpq.quantum.qubo`.
* No epsilon tie-break term is ever added to separate ties. Ties are resolved
  by an explicit, deterministic representative (:func:`pick_optimum`), and the
  number of optima is reported separately when it can be enumerated.
* Secondary/descriptive metrics (quadratic cost, maneuvered count, angle and
  speed statistics) are reported by :func:`secondary_metrics`, always separate
  from the selected primary objective.

Identity-propagation status: the QUBO, canonical QUBO hash, hardware-validation
request/artefact/config hash, persistent QPU run params and hardware decoder all
carry ``objective_id``.  The generic historical ``model.Result`` remains scoped
to the quadratic legacy solver; objective-aware benchmark/QPU exports use their
own explicit schemas and must not be silently converted to that legacy result.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum


def _check_real(value: object, name: str) -> float:
    """Fail-closed: value must be a real, finite number (bools rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


def _check_weight(w: object) -> float:
    """Fail-closed: objective weight must be a finite number in [0, 1]."""
    w = _check_real(w, "w")
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"objective weight w must be in [0, 1], got {w!r}")
    return w


def _check_tolerance(tol: object) -> float:
    """Fail-closed: a tie tolerance must be finite and non-negative."""
    tol = _check_real(tol, "tol")
    if tol < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tol!r}")
    return tol

# The exact NOOP control (q=1, theta=0). Mirrors ManeuverOption.is_noop in
# model.py; grid.build() guarantees an exact NOOP option exists, so grid-derived
# (discrete / quantum) solutions compare exactly.
NOOP_Q = 1.0
NOOP_THETA = 0.0


class ObjectiveId(str, Enum):
    """Immutable identity of an optimisation objective. Never rename a value."""

    QUADRATIC_CONTROL_COST_V1 = "quadratic_control_cost_v1"
    MANEUVER_COUNT_V1 = "maneuver_count_v1"


DEFAULT_OBJECTIVE = ObjectiveId.QUADRATIC_CONTROL_COST_V1
ALL_OBJECTIVES: tuple[ObjectiveId, ...] = tuple(ObjectiveId)


def coerce_objective_id(objective_id: ObjectiveId | str) -> ObjectiveId:
    """Accept an ``ObjectiveId`` or its exact string value; reject anything else."""
    if isinstance(objective_id, ObjectiveId):
        return objective_id
    try:
        return ObjectiveId(objective_id)
    except ValueError as exc:
        raise ValueError(
            f"unknown objective_id {objective_id!r}; "
            f"expected one of {[o.value for o in ObjectiveId]}"
        ) from exc


def is_noop(q: float, theta: float) -> bool:
    """True iff the control is exactly the NOOP (q=1, theta=0)."""
    return q == NOOP_Q and theta == NOOP_THETA


def option_cost(objective_id: ObjectiveId | str, q: float, theta: float, w: float) -> float:
    """Per-aircraft objective contribution of a single control ``(q, theta)``."""
    oid = coerce_objective_id(objective_id)
    q = _check_real(q, "q")
    theta = _check_real(theta, "theta")
    w = _check_weight(w)
    if oid is ObjectiveId.QUADRATIC_CONTROL_COST_V1:
        return w * theta * theta + (1.0 - w) * (1.0 - q) ** 2
    if oid is ObjectiveId.MANEUVER_COUNT_V1:
        return 0.0 if is_noop(q, theta) else 1.0
    raise ValueError(f"unhandled objective {oid!r}")  # pragma: no cover


def option_cost_vector(
    objective_id: ObjectiveId | str, grid, w: float
) -> tuple[float, ...]:
    """Per-grid-option linear cost vector used as the QUBO's linear term.

    For ``quadratic_control_cost_v1`` this DELEGATES to ``grid.cost(o, w)`` so
    the QUBO's linear term is byte-for-byte identical to the historical
    ``discrete.cost`` (which is ``grid.cost``) — the QUBO keeps its exact
    coefficients and hash. Note ``grid.cost`` groups the θ term as ``w*θ**2``
    whereas :func:`option_cost` (used by :func:`primary_objective`) groups it as
    ``w*θ*θ`` to match the canonical geometry scorer; the two differ only at the
    ULP for non-dyadic ``w``. The QUBO uses the grid grouping; reported objective
    values always come from the geometry kernel — this split is intentional.

    For ``maneuver_count_v1`` every non-NOOP option costs 1 and the NOOP costs 0.
    """
    oid = coerce_objective_id(objective_id)
    if oid is ObjectiveId.QUADRATIC_CONTROL_COST_V1:
        return tuple(grid.cost(o, w) for o in grid.options)
    return tuple(option_cost(oid, o.q, o.theta, w) for o in grid.options)


def primary_objective(
    objective_id: ObjectiveId | str,
    q_seq: Sequence[float],
    theta_seq: Sequence[float],
    w: float,
) -> float:
    """The selected primary objective value of a full assignment."""
    if len(q_seq) != len(theta_seq):
        raise ValueError("q and theta sequences must have equal length")
    oid = coerce_objective_id(objective_id)
    return float(sum(option_cost(oid, q, t, w) for q, t in zip(q_seq, theta_seq)))


def secondary_metrics(
    q_seq: Sequence[float], theta_seq: Sequence[float], w: float
) -> dict[str, float]:
    """Descriptive metrics reported ALONGSIDE (never instead of) the objective.

    Objective-agnostic: the same dict is reported whichever objective is being
    optimised, so a maneuver-count run still exposes its quadratic cost and a
    quadratic run still exposes how many aircraft it moved.
    """
    if len(q_seq) != len(theta_seq):
        raise ValueError("q and theta sequences must have equal length")
    _check_weight(w)
    q_seq = [_check_real(q, "q") for q in q_seq]
    theta_seq = [_check_real(t, "theta") for t in theta_seq]
    n = len(q_seq)
    abs_theta = [abs(t) for t in theta_seq]
    abs_dq = [abs(1.0 - q) for q in q_seq]
    n_maneuvered = sum(0 if is_noop(q, t) else 1 for q, t in zip(q_seq, theta_seq))
    return {
        "quadratic_control_cost": primary_objective(
            ObjectiveId.QUADRATIC_CONTROL_COST_V1, q_seq, theta_seq, w
        ),
        "maneuver_count": float(n_maneuvered),
        "n_maneuvered": float(n_maneuvered),
        "n_unchanged": float(n - n_maneuvered),
        "mean_abs_theta": (sum(abs_theta) / n) if n else 0.0,
        "max_abs_theta": max(abs_theta, default=0.0),
        "mean_abs_dq": (sum(abs_dq) / n) if n else 0.0,
        "max_abs_dq": max(abs_dq, default=0.0),
    }


def pick_optimum(
    candidates: Sequence[tuple[tuple[int, ...], float]],
    *,
    tol: float = 0.0,
) -> tuple[tuple[int, ...] | None, int]:
    """Deterministically pick one optimum and count the tied optima.

    ``candidates`` is a sequence of ``(choice_tuple, objective_value)``. Returns
    ``(representative_choice, n_optima)`` where the representative is the
    lexicographically smallest ``choice_tuple`` among the optima — a stable,
    epsilon-free tie-break.

    ``tol`` defaults to ``0.0`` (EXACT equality), which is the only correct
    setting for an integer-valued objective such as ``maneuver_count_v1`` — two
    assignments are tied iff their objectives are bit-for-bit equal. For a
    continuous objective (``quadratic_control_cost_v1``) a caller MAY pass an
    explicit, finite, non-negative ``tol`` to treat near-equal values as tied;
    this is a deliberate numerical tolerance and is reported as such — with
    ``tol > 0`` the returned count is "optima within ``tol``", NOT an exact tie.
    A solution ``tol`` away from the best is never silently called an exact tie.

    ``n_optima`` is exact only when ``candidates`` is a complete enumeration.
    Objective values must be finite (NaN/Inf are refused). ``(None, 0)`` for an
    empty input.
    """
    tol = _check_tolerance(tol)
    if not candidates:
        return None, 0
    values = [_check_real(value, "objective value") for _, value in candidates]
    best = min(values)
    optima = [choice for (choice, _), value in zip(candidates, values)
              if value <= best + tol]
    optima.sort()
    return optima[0], len(optima)
