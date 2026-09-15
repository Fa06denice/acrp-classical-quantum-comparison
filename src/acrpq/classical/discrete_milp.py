"""Classical discrete COMPATIBILITY MILP (Gurobi via AMPL) — Phase 2A.

A genuine mixed-integer *linear* program over the SAME discrete grid + conflict
table as the QUBO, solved by Gurobi through amplpy. This is NOT a QUBO and NOT
the continuous NC_ACRP model: it is the textbook assignment/compatibility
formulation the supervisor asked to compare against.

    binary x[i,o]                                    (aircraft i takes option o)
    minimise  sum_{i,o} cost[o] * x[i,o]             (cost[o] = option_cost(objective_id, o))
    s.t.      sum_o x[i,o] = 1        for each i     (exactly one option per aircraft)
              x[i,o] + x[j,p] <= 1    for each incompatible option pair (i,o)-(j,p)

The incompatible pairs come from ``DiscreteACRP.conflict`` (built by the common
geometry kernel), so the grid, option order, NOOP and conflicts are identical to
the QUBO's. The returned controls are always re-scored by the common geometry
(mandatory), never trusted from the solver.

``Code/`` is never read or written here — this is a standalone supplemental
model, not a copy or mutation of the historical NC_ACRP model.

If AMPL/Gurobi is unavailable the result status is ``"unavailable"`` (honest,
never a silent success).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import geometry
from ..discretize import DiscreteACRP
from ..model import Instance, ManeuverGrid
from ..objectives import (
    ObjectiveId,
    coerce_objective_id,
    option_cost_vector,
    secondary_metrics,
)

_MODEL = """
reset;
set AIRCRAFT;
set OPTIONS;
set INCOMPAT within {AIRCRAFT, OPTIONS, AIRCRAFT, OPTIONS};
param cost{OPTIONS};
var x{AIRCRAFT, OPTIONS} binary;
minimize Obj: sum{i in AIRCRAFT, o in OPTIONS} cost[o] * x[i,o];
subject to OneHot{i in AIRCRAFT}: sum{o in OPTIONS} x[i,o] = 1;
subject to Incompat{(i,o,j,p) in INCOMPAT}: x[i,o] + x[j,p] <= 1;
"""


@dataclass(frozen=True)
class DiscreteMILPResult:
    objective_id: str
    status: str  # "optimal" | "infeasible" | "timeout" | "unavailable" | "error"
    feasible: bool
    optimum: float | None
    choice: tuple[int, ...] | None
    q: tuple[float, ...]
    theta: tuple[float, ...]
    n_conflicts: int
    grid_k: int
    solver: str
    # certification (never a bare "solved" -> "optimal"): only True when Gurobi
    # PROVED optimality within a gap compatible with the objective's policy.
    certified: bool = False
    solve_result: str = ""
    solve_result_num: int | None = None
    best_bound: float | None = None
    abs_gap: float | None = None
    rel_gap: float | None = None
    solve_time_s: float | None = None
    solver_version: str = ""
    solver_options: str = ""
    message: str = ""
    secondary: dict[str, float] = field(default_factory=dict)


def _incompatible_pairs(discrete: DiscreteACRP) -> list[tuple[int, int, int, int]]:
    """(i, oi, j, op) for every option pair that conflicts — from the shared table."""
    out: list[tuple[int, int, int, int]] = []
    k = discrete.k
    for (i, j), matrix in discrete.conflict.items():
        for oi in range(k):
            row = matrix[oi]
            for oj in range(k):
                if row[oj]:
                    out.append((i, oi, j, oj))
    return out


def solve_discrete_milp(
    inst: Instance,
    *,
    objective_id: ObjectiveId | str,
    grid: ManeuverGrid | None = None,
    discrete: DiscreteACRP | None = None,
    n_theta: int = 3,
    n_q: int = 1,
    solver: str = "gurobi",
    time_limit_s: float | None = None,
    abs_tol: float | None = None,
) -> DiscreteMILPResult:
    """Solve the discrete compatibility MILP for ``objective_id`` with Gurobi/AMPL.

    ``abs_tol`` is the certification tolerance for the continuous objective
    (defaults to 1e-9); maneuver_count is certified as an exact integer optimum.
    """
    oid = coerce_objective_id(objective_id)
    if discrete is None:
        if grid is None:
            grid = ManeuverGrid.build(n_theta=n_theta, n_q=n_q, instance=inst)
        discrete = DiscreteACRP.build(inst, grid)
    elif discrete.instance != inst:
        raise ValueError("discrete model was built for a different instance")

    k = discrete.k
    n = inst.n
    empty = (0.0,) * n
    cost_vec = option_cost_vector(oid, discrete.grid, inst.w)

    # gap policy: prove optimality tightly, then certify per objective.
    if abs_tol is None:
        abs_tol = 0.0 if oid is ObjectiveId.MANEUVER_COUNT_V1 else 1e-9
    opts = "mipgap=1e-12 mipgapabs=1e-12 return_mipgap=7"
    if time_limit_s is not None:
        opts += f" timelim={float(time_limit_s)}"

    def _fail(status: str, msg: str) -> DiscreteMILPResult:
        return DiscreteMILPResult(
            objective_id=oid.value, status=status, feasible=False, optimum=None,
            choice=None, q=empty, theta=empty, n_conflicts=-1, grid_k=k,
            solver=solver, solver_options=opts, message=msg)

    try:
        from amplpy import AMPL
    except Exception as exc:  # pragma: no cover - environment dependent
        return _fail("unavailable", f"amplpy not importable: {exc}")

    try:
        ampl = AMPL()
        ampl.eval(_MODEL)
        ampl.set["AIRCRAFT"] = list(range(1, n + 1))
        ampl.set["OPTIONS"] = list(range(k))
        ampl.param["cost"] = {o: float(cost_vec[o]) for o in range(k)}
        ampl.set["INCOMPAT"] = _incompatible_pairs(discrete)
        ampl.option["solver"] = solver
        ampl.option[f"{solver}_options"] = opts
        banner = ampl.get_output("solve;")  # runs the solve, captures the driver banner
        solve_result = str(ampl.get_value("solve_result"))
        solve_result_num = int(ampl.get_value("solve_result_num"))
        solve_time = float(ampl.get_value("_solve_time"))
    except Exception as exc:
        return _fail("error", f"AMPL/Gurobi error: {exc}")

    ver = ""
    for line in banner.splitlines():
        if line.startswith(solver.capitalize()) or line.lower().startswith(solver):
            ver = line.split(":")[0].strip()
            break

    if solve_result != "solved" or solve_result_num != 0:
        # incumbent / timeout / infeasible / incoherent -> NEVER a certified optimum
        low = solve_result.lower()
        status = ("infeasible" if "infeasible" in low
                  else "timeout" if ("limit" in low or "interrupt" in low) else "error")
        r = _fail(status, f"solve_result={solve_result} ({solve_result_num})")
        return DiscreteMILPResult(**{**r.__dict__,
                                     "solve_result": solve_result,
                                     "solve_result_num": solve_result_num,
                                     "solver_version": ver, "solve_time_s": solve_time})

    # read the binary assignment -> exactly one option per aircraft
    xvals = ampl.get_variable("x").get_values().to_dict()
    choice_list: list[int] = []
    for i in range(1, n + 1):
        picked = [o for o in range(k) if xvals.get((i, o), 0.0) > 0.5]
        if len(picked) != 1:
            return _fail("error", f"aircraft {i} not one-hot in MILP solution: {picked}")
        choice_list.append(picked[0])
    choice = tuple(choice_list)
    q, theta = discrete.maneuvers(choice)
    n_conflicts = geometry.count_conflicts(inst, q, theta)  # mandatory geometry rescore
    optimum = float(ampl.get_objective("Obj").value())
    abs_gap: float | None
    rel_gap: float | None
    try:
        abs_gap = float(ampl.get_value("Obj.absmipgap"))
        rel_gap = float(ampl.get_value("Obj.relmipgap"))
    except Exception:
        abs_gap = rel_gap = None
    best_bound = (optimum - abs_gap) if abs_gap is not None else None

    # OBJECTIVE-AWARE certification (never a bare "solved" -> "optimal")
    certified = True
    if abs_gap is None:
        certified = False
    elif oid is ObjectiveId.MANEUVER_COUNT_V1:
        # integer objective: value integral AND gap proves it (< 1 => no better int)
        certified = abs(optimum - round(optimum)) <= 1e-9 and abs_gap < 0.5
    else:
        certified = abs_gap <= abs_tol  # quadratic: gap within the declared tolerance

    return DiscreteMILPResult(
        objective_id=oid.value, status="optimal", feasible=n_conflicts == 0,
        optimum=optimum, choice=choice, q=q, theta=theta, n_conflicts=n_conflicts,
        grid_k=k, solver=solver, certified=certified, solve_result=solve_result,
        solve_result_num=solve_result_num, best_bound=best_bound, abs_gap=abs_gap,
        rel_gap=rel_gap, solve_time_s=solve_time, solver_version=ver,
        solver_options=opts, secondary=secondary_metrics(q, theta, inst.w))
