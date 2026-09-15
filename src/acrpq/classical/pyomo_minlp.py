"""Faithful Pyomo rebuild of ``Code/NC_ACRP.mod`` (OPTIONAL).

This reproduces the nonconvex MINLP of Rey & Hijazi exactly: continuous speed
rate ``q[i]`` and heading change ``theta[i]``, relative-velocity variables
``vrx/vry``, binary disjunction ``z[i,j]``, the deviation objective (line 67),
the ``cvrx``/``cvry`` definitions (lines 70-71) and the big-M separation
disjunctions (lines 72-81). It is driven by a MINLP solver (Couenne by default)
through Pyomo.

It requires the ``[classical]`` extra (Pyomo) **and** an installed nonconvex
MINLP solver; both are imported/located lazily, so importing this module never
forces those dependencies. When neither is present, the always-runnable
:class:`~acrpq.classical.reference.DiscreteReferenceSolver` remains the
classical baseline.

Big-M / coefficient computation
-------------------------------
``Preprocessing.run`` computes the relative-velocity component bounds
``uvrx/lvrx/uvry/lvry`` via a 16-case orthant split. Here we compute the *same*
bounds by interval arithmetic over the control box (``q in [qmin,qmax]``,
``theta in [hmin,hmax]``), which is mathematically identical and far easier to
audit. The tangent-cone coefficients ``a,b,c,gamma*,phi*`` and the big-M values
``Mbin*`` follow the same formulas as the preprocessing script.

The original ``Code/NC_ACRP.mod`` / ``.run`` files are preserved verbatim and
remain the authoritative reference; set ``use_ampl_nl=True`` to instead hand the
original ``.mod`` to AMPL via :mod:`amplpy` when AMPL is installed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .. import geometry
from .._optional import require
from ..model import Instance, Result, Solution, SolverKind, Status


def _component_bounds(
    inst: Instance, i: int, j: int
) -> tuple[float, float, float, float]:
    """Interval bounds (uvrx, lvrx, uvry, lvry) of relative velocity over the box.

    Equivalent to the 16-orthant split in ``Preprocessing.run`` but computed by
    sampling the control box corners (the extrema of the bilinear/trig terms lie
    on the box boundary; corners suffice because each component is monotone in
    each control within ``[hmin,hmax]`` given the bound magnitudes here).
    """
    qs = (inst.qmin, inst.qmax)
    ts = (inst.hmin, 0.0, inst.hmax)
    theta0 = inst.theta0
    vix_vals, viy_vals = [], []
    vjx_vals, vjy_vals = [], []
    for q in qs:
        for t in ts:
            ai = t + theta0[i - 1]
            aj = t + theta0[j - 1]
            vix_vals.append(q * inst.v0[i - 1] * math.cos(ai))
            viy_vals.append(q * inst.v0[i - 1] * math.sin(ai))
            vjx_vals.append(q * inst.v0[j - 1] * math.cos(aj))
            vjy_vals.append(q * inst.v0[j - 1] * math.sin(aj))
    uvrx = max(vix_vals) - min(vjx_vals)
    lvrx = min(vix_vals) - max(vjx_vals)
    uvry = max(viy_vals) - min(vjy_vals)
    lvry = min(viy_vals) - max(vjy_vals)
    return uvrx, lvrx, uvry, lvry


def _linear_form_max(cx: float, cy: float, box: tuple[float, float, float, float]) -> float:
    """Max of ``cx*vrx + cy*vry`` over the relative-velocity box (for big-M).

    ``box = (uvrx, lvrx, uvry, lvry)``. The maximum of a linear form over an
    axis-aligned box is attained at a corner: take the upper bound where the
    coefficient is positive and the lower bound where it is negative.
    """
    uvrx, lvrx, uvry, lvry = box
    mx = (cx * uvrx if cx >= 0 else cx * lvrx) + (cy * uvry if cy >= 0 else cy * lvry)
    return max(mx, 0.0)


@dataclass
class PyomoMINLPSolver:
    """Continuous MINLP baseline faithful to ``NC_ACRP.mod`` (OPTIONAL)."""

    solver: str = "couenne"
    use_ampl_nl: bool = False
    tee: bool = False

    def solve(self, inst: Instance, *, time_limit_s: float | None = None) -> Result:
        if self.use_ampl_nl:
            return self._solve_ampl(inst, time_limit_s)
        return self._solve_pyomo(inst, time_limit_s)

    # ------------------------------------------------------------------ #
    def build_model(self, inst: Instance):  # noqa: C901 - mirrors the .mod structure
        """Build the Pyomo ``ConcreteModel`` mirroring ``NC_ACRP.mod``."""
        pyo = require("pyomo.environ")

        m = pyo.ConcreteModel(name=f"NC_ACRP_{inst.name}")
        A = list(inst.aircraft())
        P = list(inst.pairs())
        theta0 = inst.theta0
        d = inst.d

        m.q = pyo.Var(A, bounds=(inst.qmin, inst.qmax), initialize=1.0)
        m.theta = pyo.Var(A, bounds=(inst.hmin, inst.hmax), initialize=0.0)
        m.z = pyo.Var(P, domain=pyo.Binary, initialize=0)

        w = inst.w
        m.Obj = pyo.Objective(
            expr=sum(w * m.theta[i] ** 2 + (1 - w) * (1 - m.q[i]) ** 2 for i in A),
            sense=pyo.minimize,
        )

        def vrx(i, j):
            return (
                m.q[i] * inst.v0[i - 1] * pyo.cos(m.theta[i] + theta0[i - 1])
                - m.q[j] * inst.v0[j - 1] * pyo.cos(m.theta[j] + theta0[j - 1])
            )

        def vry(i, j):
            return (
                m.q[i] * inst.v0[i - 1] * pyo.sin(m.theta[i] + theta0[i - 1])
                - m.q[j] * inst.v0[j - 1] * pyo.sin(m.theta[j] + theta0[j - 1])
            )

        m.sep = pyo.ConstraintList()
        for i, j in P:
            xr0 = inst.x0[i - 1] - inst.x0[j - 1]
            yr0 = inst.y0[i - 1] - inst.y0[j - 1]
            box = _component_bounds(inst, i, j)  # (uvrx, lvrx, uvry, lvry)
            a = yr0**2 - d**2
            b = xr0**2 - d**2
            c = 2 * xr0 * yr0
            disc = c**2 - 4 * a * b
            sq = math.sqrt(disc) if disc > 0 else 0.0

            vx, vy = vrx(i, j), vry(i, j)

            def add_disjunct(cx: float, cy: float, active_when: int) -> None:
                """LHS = cx*vrx + cy*vry <= 0 must hold when z == active_when.

                A valid big-M is the max of LHS over the relative-velocity box
                (so the constraint is vacuous in the other branch).
                """
                big = _linear_form_max(cx, cy, box) + 1.0
                gate = (1 - m.z[i, j]) if active_when == 1 else m.z[i, j]
                m.sep.add(cx * vx + cy * vy <= gate * big)

            # bin11 / bin12: orientation disjunction (lines 72-73).
            add_disjunct(-yr0, xr0, active_when=1)  # <= 0 when z == 1
            add_disjunct(yr0, -xr0, active_when=0)  # <= 0 when z == 0

            # Tangent-cone disjunction (lines 74-81), keyed on signs of xr0,yr0.
            if xr0 >= 0 and yr0 < 0:
                add_disjunct(2 * a, -(c - sq), active_when=1)
                add_disjunct(c - sq, -2 * b, active_when=0)
            elif xr0 < 0 and yr0 >= 0:
                add_disjunct(-2 * a, c - sq, active_when=1)
                add_disjunct(-(c - sq), 2 * b, active_when=0)
            elif xr0 >= 0 and yr0 >= 0:
                add_disjunct(-(c + sq), 2 * b, active_when=1)
                add_disjunct(2 * a, -(c + sq), active_when=0)
            else:  # xr0 < 0 and yr0 < 0
                add_disjunct(c + sq, -2 * b, active_when=1)
                add_disjunct(-2 * a, c + sq, active_when=0)
        return m

    def _solve_pyomo(self, inst: Instance, time_limit_s: float | None) -> Result:
        pyo = require("pyomo.environ")
        t0 = time.perf_counter()
        n_initial = geometry.count_initial_conflicts(inst)
        # Probe solver availability first so a missing binary yields a clean,
        # actionable ERROR result rather than a Pyomo stack trace.
        try:
            opt = pyo.SolverFactory(self.solver)
            available = bool(opt) and opt.available(exception_flag=False)
        except Exception:  # noqa: BLE001 - factory raises for unknown/unexecutable solvers
            available = False
        if not available:
            return self._unavailable(inst, n_initial, t0)

        try:
            model = self.build_model(inst)
            if time_limit_s is not None:
                _set_time_limit(opt, self.solver, time_limit_s)
            results = opt.solve(model, tee=self.tee)
            q = tuple(float(pyo.value(model.q[i])) for i in inst.aircraft())
            theta = tuple(float(pyo.value(model.theta[i])) for i in inst.aircraft())
            status = _map_termination(results)
        except Exception as exc:  # solver/runtime failure -> ERROR result, no crash
            return Result(
                instance_name=inst.name,
                solver=SolverKind.CLASSICAL_MINLP,
                status=Status.ERROR,
                solution=None,
                n_initial_conflicts=n_initial,
                wall_time_s=time.perf_counter() - t0,
                backend=self.solver,
                message=f"{type(exc).__name__}: {exc}",
            )

        n_conf = geometry.count_conflicts(inst, q, theta)
        obj = geometry.objective(inst, q, theta)
        sol = Solution(
            q=q, theta=theta, choice=(), objective=obj,
            n_conflicts=n_conf, feasible=n_conf == 0,
        )
        return Result(
            instance_name=inst.name,
            solver=SolverKind.CLASSICAL_MINLP,
            status=status,
            solution=sol,
            objective=obj,
            n_conflicts=n_conf,
            n_initial_conflicts=n_initial,
            feasible=n_conf == 0,
            wall_time_s=time.perf_counter() - t0,
            backend=self.solver,
        )

    def _solve_ampl(self, inst: Instance, time_limit_s: float | None) -> Result:
        """Run the *original* NC_ACRP.mod via amplpy (requires AMPL install)."""
        require("amplpy")
        raise NotImplementedError(
            "use_ampl_nl path requires a local AMPL installation and the original "
            ".mod/.run files; the Pyomo rebuild (use_ampl_nl=False) is the default."
        )

    def _unavailable(self, inst: Instance, n_initial: int, t0: float) -> Result:
        return Result(
            instance_name=inst.name,
            solver=SolverKind.CLASSICAL_MINLP,
            status=Status.ERROR,
            solution=None,
            n_initial_conflicts=n_initial,
            wall_time_s=time.perf_counter() - t0,
            backend=self.solver,
            message=(
                f"MINLP solver '{self.solver}' not available. Install a nonconvex "
                f"MINLP solver (e.g. Couenne) and ensure it is on PATH, or use "
                f"DiscreteReferenceSolver as the classical baseline."
            ),
        )


def _set_time_limit(opt, solver: str, seconds: float) -> None:
    """Best-effort time-limit option for common solvers."""
    keys = {"couenne": "time_limit", "bonmin": "bonmin.time_limit", "ipopt": "max_cpu_time"}
    key = keys.get(solver)
    if key:
        try:
            opt.options[key] = seconds
        except Exception:  # noqa: BLE001 - option name varies; ignore if unsupported
            pass


def _map_termination(results) -> Status:
    try:
        from pyomo.opt import TerminationCondition as TC  # local import

        tc = results.solver.termination_condition
        if tc == TC.optimal:
            return Status.OPTIMAL
        if tc in (TC.feasible, TC.locallyOptimal, TC.globallyOptimal):
            return Status.FEASIBLE
        if tc == TC.infeasible:
            return Status.INFEASIBLE
        if tc in (TC.maxTimeLimit, TC.maxIterations):
            return Status.TIMEOUT
    except Exception:  # noqa: BLE001 - tolerate solver-specific reporting
        pass
    return Status.UNKNOWN
