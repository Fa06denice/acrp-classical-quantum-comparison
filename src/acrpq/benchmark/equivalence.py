"""Machine-readable three-way discrete equivalence proof (Phase 2D / 2.1).

For a given instance + grid + objective, prove that three independent classical
discrete solvers agree on ``min primary_objective s.t. conflict-free``:

    exhaustive enumeration  ==  Gurobi discrete MILP  ==  QUBO global minimum

QUBO leg — how the GLOBAL minimum of ``H`` over the full ``{0,1}^(nK)`` space is
established (never silently "exact_qubo"):

* ``full_2^nK`` — for small configs (``2**nK <= FULL_BINARY_CAP``) we enumerate
  the ENTIRE binary space, take the global minimum, and CHECK it lies in the
  one-hot subspace (empirically validating the reduction theorem).
* ``one_hot_reduction`` — for larger configs we invoke the reduction theorem and
  only enumerate the one-hot subspace. Its hypotheses are checked fail-closed by
  ``verify_qubo_wellformed`` (full-coefficient rebuild + penalty dominance
  ``onehot_optimum_guaranteed``); the theorem and its conditions are recorded in
  the artifact. If dominance is not guaranteed the proof is REFUSED.

Reduction theorem (recorded, empirically validated by the ``full_2^nK`` cases):
if ``lam_pen > n*cost_max`` and ``lam_oh > lam_pen*max_conflict_degree`` then every
non-one-hot ``x`` has ``H(x)`` strictly greater than the best one-hot assignment,
so ``argmin_{{0,1}^(nK)} H`` lies in the one-hot subspace.

Tolerance / tie policy (explicit, never relaxed to force a pass):
* ``maneuver_count_v1``: EXACT — integer values, zero tolerance, exact optima
  cardinality; the Gurobi leg is certified as an exact integer optimum.
* ``quadratic_control_cost_v1``: one finite documented ``abs_tol`` (default 1e-9)
  applied consistently to optimum comparison, tie counting (``tie_kind =
  within_absolute_tolerance``) and Gurobi gap certification.

Any divergence — including a non-completing exhaustive reference, an
uncertified Gurobi solve, or a reduction-theorem violation — sets
``equivalent=False`` with a tagged diagnosis. Callers must hard-stop.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

from .. import geometry
from ..classical.discrete_enum import exhaustive_discrete_optimum
from ..classical.discrete_milp import solve_discrete_milp
from ..discretize import DiscreteACRP
from ..model import Instance, ManeuverGrid
from ..objectives import (
    ObjectiveId,
    coerce_objective_id,
    primary_objective,
    secondary_metrics,
)
from ..quantum.qubo import build_qubo, verify_qubo_wellformed

# "small accessible" full-binary bound: 2^18 bitstrings enumerated directly.
FULL_BINARY_CAP = 1 << 18

_ABS_TOL = {
    ObjectiveId.MANEUVER_COUNT_V1: 0.0,
    ObjectiveId.QUADRATIC_CONTROL_COST_V1: 1e-9,
}


@dataclass(frozen=True)
class LegResult:
    method: str
    status: str
    feasible: bool
    optimum: float | None
    representative_choice: tuple[int, ...] | None
    n_optima: int | None  # None => the method does not enumerate the optimal set
    tie_kind: str | None
    tie_tolerance: float | None
    n_conflicts: int
    certified: bool | None = None      # Gurobi leg only
    qubo_search: str | None = None     # QUBO leg only
    secondary: dict[str, float] = field(default_factory=dict)
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EquivalenceResult:
    instance: str
    objective_id: str
    grid_k: int
    n_aircraft: int
    n_qubits: int
    search_space: int
    abs_tol: float
    tie_kind: str
    equivalent: bool
    optimum: float | None
    feasible: bool
    n_optima: int | None
    qubo_search: str | None
    reduction_theorem: dict
    legs: dict[str, LegResult]
    mismatches: list[dict[str, str]] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        d["legs"] = {k: asdict(v) for k, v in self.legs.items()}
        return d


def _mask_energy(qubo, mask: int) -> float:
    """H(x) for the bitstring encoded by integer ``mask`` (bit b == qubit b)."""
    e = qubo.constant
    lin = qubo.linear
    m, b = mask, 0
    while m:
        if m & 1:
            e += lin.get(b, 0.0)
        m >>= 1
        b += 1
    for (a, c), coeff in qubo.quadratic.items():
        if (mask >> a) & 1 and (mask >> c) & 1:
            e += coeff
    return e


def _is_one_hot(discrete: DiscreteACRP, mask: int) -> bool:
    for i in discrete.instance.aircraft():
        if sum((mask >> discrete.var_index(i, o)) & 1 for o in range(discrete.k)) != 1:
            return False
    return True


def _mask_choice(discrete: DiscreteACRP, mask: int) -> tuple[int, ...]:
    return tuple(next(o for o in range(discrete.k) if (mask >> discrete.var_index(i, o)) & 1)
                 for i in discrete.instance.aircraft())


def _reduction_theorem(qubo) -> dict:
    m = qubo.meta
    return {
        "claim": "argmin over {0,1}^(nK) of H(x) lies in the one-hot subspace",
        "lam_pen_gt_n_cost_max": qubo.penalty_conflict > m["n_aircraft"] * m["cost_max"],
        "lam_oh_gt_lam_pen_maxdeg":
            qubo.penalty_onehot > qubo.penalty_conflict * m["max_conflict_degree"],
        "onehot_optimum_guaranteed": m.get("onehot_optimum_guaranteed") == 1.0,
    }


def _failed_qubo_leg(status, tie_kind, abs_tol, qubo_search, detail) -> LegResult:
    """A diagnosed, non-decodable QUBO leg (fail-closed, never crashes downstream)."""
    return LegResult(
        method="qubo_global_min", status=status, feasible=False, optimum=None,
        representative_choice=None, n_optima=None, tie_kind=tie_kind,
        tie_tolerance=abs_tol, n_conflicts=-1, qubo_search=qubo_search, detail=detail)


def _qubo_leg(inst, discrete, oid, abs_tol, tie_kind) -> tuple[LegResult, dict, list]:
    """Establish the QUBO global minimum; returns (leg, reduction_theorem, mismatches)."""
    problems: list[dict[str, str]] = []
    qubo = build_qubo(inst, discrete=discrete, objective_id=oid, max_qubits=10_000)
    theorem = _reduction_theorem(qubo)
    try:
        verify_qubo_wellformed(qubo)  # full-coefficient + fail-closed penalty dominance
    except ValueError as exc:
        # inadmissible QUBO (falsified coefficients or non-dominant penalties):
        # do NOT attempt a reduction proof — fail closed, no crash.
        problems.append({"field": "qubo_wellformed", "detail": str(exc), "cause": "penalty"})
        return (_failed_qubo_leg("not_wellformed", tie_kind, abs_tol, None,
                                 {"error": str(exc)}), theorem, problems)
    nq = qubo.n_qubits

    if (1 << nq) <= FULL_BINARY_CAP:
        # enumerate the ENTIRE binary space and prove the global min is one-hot
        best_e = None
        min_masks: list[int] = []
        for mask in range(1 << nq):
            e = _mask_energy(qubo, mask)
            if best_e is None or e < best_e - abs_tol:
                best_e, min_masks = e, [mask]
            elif abs(e - best_e) <= abs_tol:
                min_masks.append(mask)
        onehot_min = [m for m in min_masks if _is_one_hot(discrete, m)]
        non_onehot = [m for m in min_masks if not _is_one_hot(discrete, m)]
        theorem["empirically_validated_full_binary"] = not non_onehot
        if non_onehot:
            problems.append({"field": "reduction_theorem",
                             "detail": f"{len(non_onehot)} global-min bitstring(s) are NOT one-hot",
                             "cause": "penalty"})
        if not onehot_min:
            # reduction theorem violated: no one-hot global minimum to decode.
            return (_failed_qubo_leg("reduction_violated", tie_kind, abs_tol, "full_2^nK",
                                     {"n_qubits": nq, "min_energy": best_e,
                                      "non_onehot_global_minima": len(non_onehot)}),
                    theorem, problems)
        rep_mask = min(onehot_min)
        qubo_search = "full_2^nK"
        n_optima = len(min_masks)
    else:
        # reduction theorem: enumerate only the one-hot subspace (dominance verified)
        best_e = None
        min_masks = []
        for choice in itertools.product(range(discrete.k), repeat=inst.n):
            mask = 0
            for i in inst.aircraft():
                mask |= 1 << discrete.var_index(i, choice[i - 1])
            e = _mask_energy(qubo, mask)
            if best_e is None or e < best_e - abs_tol:
                best_e, min_masks = e, [mask]
            elif abs(e - best_e) <= abs_tol:
                min_masks.append(mask)
        rep_mask = min(min_masks)
        qubo_search = "one_hot_reduction"
        n_optima = len(min_masks)

    rep = _mask_choice(discrete, rep_mask)
    q, theta = discrete.maneuvers(rep)
    n_conf = geometry.count_conflicts(inst, q, theta)
    leg = LegResult(
        method="qubo_global_min", status="optimal", feasible=n_conf == 0,
        optimum=primary_objective(oid, q, theta, inst.w), representative_choice=rep,
        n_optima=n_optima, tie_kind=tie_kind, tie_tolerance=abs_tol, n_conflicts=n_conf,
        qubo_search=qubo_search, secondary=secondary_metrics(q, theta, inst.w),
        detail={"n_qubits": nq, "min_energy": best_e})
    return leg, theorem, problems


def _leg_from_enum(r) -> LegResult:
    return LegResult(
        method="exhaustive", status=r.status, feasible=r.feasible, optimum=r.optimum,
        representative_choice=r.representative_choice, n_optima=r.n_optima,
        tie_kind=r.tie_kind, tie_tolerance=r.tie_tolerance, n_conflicts=r.n_conflicts,
        secondary=dict(r.secondary))


def _leg_from_milp(r) -> LegResult:
    return LegResult(
        method="gurobi_milp", status=r.status, feasible=r.feasible, optimum=r.optimum,
        representative_choice=r.choice, n_optima=None, tie_kind=None, tie_tolerance=None,
        n_conflicts=r.n_conflicts, certified=r.certified, secondary=dict(r.secondary),
        detail={"solve_result": r.solve_result, "solve_result_num": r.solve_result_num,
                "best_bound": r.best_bound, "abs_gap": r.abs_gap, "rel_gap": r.rel_gap,
                "solve_time_s": r.solve_time_s, "solver_version": r.solver_version,
                "solver_options": r.solver_options, "message": r.message})


def prove_equivalence(
    inst: Instance,
    *,
    objective_id: ObjectiveId | str,
    n_theta: int = 3,
    n_q: int = 1,
    max_search: int = 2_000_000,
    milp_solver: str = "gurobi",
) -> EquivalenceResult:
    """Run the three legs and compare them. ``equivalent=False`` on any divergence."""
    oid = coerce_objective_id(objective_id)
    abs_tol = _ABS_TOL[oid]
    tie_kind = "exact" if abs_tol == 0.0 else "within_absolute_tolerance"
    grid = ManeuverGrid.build(n_theta=n_theta, n_q=n_q, instance=inst)
    discrete = DiscreteACRP.build(inst, grid)
    nq = discrete.k * inst.n

    enum = exhaustive_discrete_optimum(inst, objective_id=oid, discrete=discrete,
                                       max_search=max_search, tol=abs_tol)
    milp = solve_discrete_milp(inst, objective_id=oid, discrete=discrete,
                               solver=milp_solver, abs_tol=abs_tol)
    legs: dict[str, LegResult] = {"exhaustive": _leg_from_enum(enum),
                                  "gurobi_milp": _leg_from_milp(milp)}
    mismatches: list[dict[str, str]] = []
    theorem: dict = {}
    qubo_search: str | None = None

    def close(a, b) -> bool:
        return a is not None and b is not None and abs(a - b) <= abs_tol

    def diag(field_: str, detail: str, cause: str) -> None:
        mismatches.append({"field": field_, "detail": detail, "cause": cause})

    if enum.status == "search_too_large":
        diag("reference", f"exhaustive reference could not enumerate "
             f"(K^n={discrete.k**inst.n} > max_search={max_search})", "reference-incomplete")
    elif enum.status == "infeasible":
        qubo_leg, theorem, probs = _qubo_leg(inst, discrete, oid, abs_tol, tie_kind)
        legs["qubo_global_min"] = qubo_leg
        qubo_search = qubo_leg.qubo_search
        mismatches.extend(probs)
        if milp.status != "infeasible":
            diag("feasibility", f"exhaustive infeasible but MILP status={milp.status}", "conflict")
        if qubo_leg.feasible:
            diag("feasibility", "exhaustive infeasible but QUBO decoded a conflict-free solution", "penalty")
    elif enum.status == "optimal":
        qubo_leg, theorem, probs = _qubo_leg(inst, discrete, oid, abs_tol, tie_kind)
        legs["qubo_global_min"] = qubo_leg
        qubo_search = qubo_leg.qubo_search
        mismatches.extend(probs)
        # feasibility agreement
        feas = {m: leg.feasible for m, leg in legs.items() if leg.status in ("optimal", "infeasible")}
        if len(set(feas.values())) > 1:
            diag("feasibility", f"{feas}", "conflict")
        # Gurobi must be CERTIFIED optimal (never a bare "solved")
        if milp.status != "optimal":
            diag("gurobi_milp", f"MILP status={milp.status}: {milp.message}", "solver-unavailable")
        elif not milp.certified:
            diag("gurobi_milp", f"MILP not certified optimal (gap policy): "
                 f"abs_gap={milp.abs_gap} solve_result={milp.solve_result}", "solver-tolerance")
        # optimum value pairwise within tolerance
        for m, leg in legs.items():
            if m == "exhaustive" or leg.status != "optimal":
                continue
            if not close(leg.optimum, enum.optimum):
                diag(f"optimum:{m}", f"{leg.optimum} vs exhaustive {enum.optimum}",
                     "solver-tolerance" if oid is ObjectiveId.QUADRATIC_CONTROL_COST_V1 else "objective")
        # each representative must itself be a valid, geometry-feasible optimum
        for m, leg in legs.items():
            if leg.status != "optimal":
                continue
            if not leg.feasible:
                diag(f"representative:{m}", "representative is not conflict-free", "conflict")
            if not close(leg.optimum, enum.optimum):
                diag(f"representative:{m}", "representative does not attain the optimum", "objective")
        # optimal-set cardinality where enumerable (exhaustive vs QUBO), same tie policy
        if qubo_leg.status == "optimal" and qubo_leg.n_optima != enum.n_optima:
            diag("n_optima", f"qubo {qubo_leg.n_optima} vs exhaustive {enum.n_optima} "
                 f"(tie_kind={tie_kind})",
                 "penalty" if oid is ObjectiveId.MANEUVER_COUNT_V1 else "solver-tolerance")
    else:
        diag("reference", f"exhaustive reference status={enum.status}", "reference-incomplete")

    equivalent = (not mismatches) and enum.status in ("optimal", "infeasible")
    return EquivalenceResult(
        instance=inst.name, objective_id=oid.value, grid_k=discrete.k, n_aircraft=inst.n,
        n_qubits=nq, search_space=discrete.k**inst.n, abs_tol=abs_tol, tie_kind=tie_kind,
        equivalent=equivalent, optimum=enum.optimum, feasible=enum.feasible,
        n_optima=enum.n_optima, qubo_search=qubo_search, reduction_theorem=theorem,
        legs=legs, mismatches=mismatches)
