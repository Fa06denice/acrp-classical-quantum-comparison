"""Per-pair separation diagnostics with sign-correct feasibility terminology.

A conflict *count* is numerically misleading without the *depth* of each violation.
But a positive shortfall — even ``6.7e-12`` — is still a violation of the exact
constraint ``min_sep >= d``, NOT strict mathematical feasibility. This module keeps
the following distinctions explicit and never merges them:

* **exact_theoretical** — feasible iff ``min_separation >= d`` for EVERY pair
  (i.e. ``max violation depth <= 0``). Any positive depth fails this, always.
* **common_numeric_acceptance** — the shared geometry scorer's documented numeric
  criterion (``min_sep^2 < d^2 - 1e-9``). A solution may pass this while failing
  ``exact_theoretical`` by a sub-tolerance amount. This is the repository's scorer.
* **minimum_safety_margin** — the SIGNED quantity ``min_over_pairs(min_sep - d)``:
  positive ⇒ separated with margin, zero ⇒ exactly at the boundary, negative ⇒ a
  violation. This is reported as a number, never re-labelled as "feasible with
  margin" unless it is genuinely positive.

Note: ``solver_constraint_acceptance`` (feasibility to the solver's own constraint
residuals) is a SEPARATE notion — it is based on the model constraints' residuals
reported by the solver, not on the geometric separation depth — and is assembled by
the caller from the runner's ``max_model_constraint_residual``. This module makes no
claim converting a solver feasibility tolerance into a distance.

All separations are computed by the single :mod:`acrpq.geometry` kernel.
:func:`pi_convention_check` re-computes them under the historical ``PI = 3.141592``
(scorer/AMPL) and ``math.pi`` to show whether the PI convention explains a result.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .. import geometry
from ..model import Instance

# Informational depth-distribution buckets (shortfall ``d - min_separation``). These
# are COUNTS of how many pairs fall past each depth — a distribution, NOT feasibility
# verdicts (feasibility is exact_theoretical / common_numeric_acceptance only).
DEPTH_DISTRIBUTION_BUCKETS: dict[str, float] = {
    "depth_gt_0": 0.0,
    "depth_gt_1e-12": 1e-12,
    "depth_gt_1e-9": 1e-9,
    "depth_gt_1e-6": 1e-6,
    "depth_gt_1e-4": 1e-4,
    "depth_gt_1e-3": 1e-3,
}


@dataclass(frozen=True)
class PairSeparation:
    i: int
    j: int
    excluded: bool           # different flight levels -> never a conflict
    min_separation: float
    required: float          # d
    violation_depth: float   # max(0, d - min_separation)
    scorer_conflict: bool    # the strict geometry scorer's verdict for this pair


def _pairs(inst: Instance, q, theta) -> list[PairSeparation]:
    out: list[PairSeparation] = []
    for pd in geometry.pair_diagnostics(inst, q, theta):
        out.append(PairSeparation(
            i=pd.i, j=pd.j, excluded=pd.excluded,
            min_separation=pd.min_separation, required=pd.required,
            violation_depth=pd.violation_depth, scorer_conflict=pd.resolved_conflict,
        ))
    return out


def separation_diagnostics(inst: Instance, q, theta) -> dict:
    """Full separation diagnostics for one solution, ready to serialise.

    ``minimum_safety_margin`` is the SIGNED ``min(min_sep - d)``: positive only when
    every pair is strictly separated with margin. ``feasible_exact_theoretical`` is
    true only when NO pair falls short of ``d`` (max depth <= 0); a positive depth,
    however small, makes it false.
    """
    pairs = _pairs(inst, q, theta)
    active = [p for p in pairs if not p.excluded]
    depths = [p.violation_depth for p in active]
    max_depth = max(depths, default=0.0)
    sum_depth = float(sum(depths))
    n_scorer = sum(1 for p in active if p.scorer_conflict)
    # signed margin: min over pairs of (min_separation - d)
    min_safety_margin = min((p.min_separation - p.required for p in active), default=math.inf)

    feasible_exact = max_depth <= 0.0
    feasible_common = n_scorer == 0

    counts = {name: sum(1 for d in depths if d > thr)
              for name, thr in DEPTH_DISTRIBUTION_BUCKETS.items()}
    counts["scorer_conflicts"] = n_scorer

    regimes = {
        "exact_theoretical": {
            "definition": "min_separation >= d for every pair (max violation depth <= 0)",
            "feasible": feasible_exact,
            "n_pairs_short_of_d": sum(1 for d in depths if d > 0.0),
            "worst_depth": max_depth,
        },
        "common_numeric_acceptance": {
            "definition": "shared geometry scorer: min_sep^2 < d^2 - 1e-9 (documented tol)",
            "feasible": feasible_common,
            "n_violations": n_scorer,
        },
    }

    return {
        "required_separation_d": inst.d,
        "n_pairs": len(pairs),
        "n_pairs_active": len(active),
        "pairs": [asdict(p) for p in pairs],
        "max_violation_depth": max_depth,
        "sum_violation_depth": sum_depth,
        "minimum_safety_margin": min_safety_margin,
        "feasible_exact_theoretical": feasible_exact,
        "feasible_common_numeric": feasible_common,
        "depth_distribution_counts": counts,
        "feasibility_regimes": regimes,
    }


def _seps_with_theta0(inst: Instance, q, theta, theta0: tuple[float, ...]) -> list[dict]:
    """Per-pair minimum separations using an explicit ``theta0`` array.

    Mirrors the geometry kernel but lets us substitute an alternative angular
    convention (e.g. ``math.pi``-based ``theta0``) for the PI sensitivity check.
    """
    out: list[dict] = []
    for i, j in inst.pairs():
        if geometry.levels_separated(inst, i, j):
            out.append({"i": i, "j": j, "excluded": True, "min_separation": math.inf})
            continue
        xr0, yr0 = geometry.relative_position(inst, i, j)
        vix = q[i - 1] * inst.v0[i - 1] * math.cos(theta[i - 1] + theta0[i - 1])
        viy = q[i - 1] * inst.v0[i - 1] * math.sin(theta[i - 1] + theta0[i - 1])
        vjx = q[j - 1] * inst.v0[j - 1] * math.cos(theta[j - 1] + theta0[j - 1])
        vjy = q[j - 1] * inst.v0[j - 1] * math.sin(theta[j - 1] + theta0[j - 1])
        msq = geometry.min_separation_sq(xr0, yr0, vix - vjx, viy - vjy)
        out.append({"i": i, "j": j, "excluded": False,
                    "min_separation": math.sqrt(max(msq, 0.0))})
    return out


def pi_convention_check(inst: Instance, q, theta) -> dict:
    """Re-score separations under PI=3.141592 (scorer/AMPL) vs math.pi.

    ``theta0`` is derived from ``cap`` via a ``cap - 2*PI`` wraparound, so the PI
    literal shifts some angles by ~1e-6. This quantifies whether that convention
    could account for a borderline separation shortfall. It does NOT modify any
    historical constant.
    """
    from ..model import PI as PI_LITERAL

    theta0_lit = tuple(c - 2 * PI_LITERAL if c >= PI_LITERAL else c for c in inst.cap)
    theta0_math = tuple(c - 2 * math.pi if c >= math.pi else c for c in inst.cap)
    seps_lit = _seps_with_theta0(inst, q, theta, theta0_lit)
    seps_math = _seps_with_theta0(inst, q, theta, theta0_math)

    def _depths(seps):
        return [max(0.0, inst.d - s["min_separation"]) for s in seps if not s["excluded"]]

    d_lit, d_math = _depths(seps_lit), _depths(seps_math)
    max_abs_sep_diff = max(
        (abs(a["min_separation"] - b["min_separation"])
         for a, b in zip(seps_lit, seps_math) if not a["excluded"]),
        default=0.0,
    )
    return {
        "theta0_literal_pi": list(theta0_lit),
        "theta0_math_pi": list(theta0_math),
        "max_abs_theta0_diff": max((abs(a - b) for a, b in zip(theta0_lit, theta0_math)),
                                   default=0.0),
        "max_violation_depth_literal_pi": max(d_lit, default=0.0),
        "max_violation_depth_math_pi": max(d_math, default=0.0),
        "max_abs_separation_diff_between_conventions": max_abs_sep_diff,
        "pairs_literal_pi": seps_lit,
        "pairs_math_pi": seps_math,
    }
