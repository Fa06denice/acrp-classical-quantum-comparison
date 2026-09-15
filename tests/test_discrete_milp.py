"""Fake-AMPL certification tests for the discrete MILP leg (Phase 2.1-2).

No real AMPL/Gurobi is contacted: amplpy.AMPL is monkeypatched with a fake whose
solve outcome is scripted, so we can exercise every certification path —
solved+zero-gap, solved+nonzero-gap, timeout+incumbent, infeasible, unavailable,
and an incoherent status — deterministically and everywhere.
"""

from __future__ import annotations

import pytest

from acrpq.classical import discrete_milp
from acrpq.io.loader import InstanceLoader


class _Dictish:
    def __setitem__(self, k, v):  # ampl.set[...] = ..., ampl.option[...] = ...
        pass


class _Values:
    def __init__(self, d):
        self._d = d

    def to_dict(self):
        return self._d


class _Var:
    def __init__(self, d):
        self._d = d

    def get_values(self):
        return _Values(self._d)


class _Obj:
    def __init__(self, v):
        self._v = v

    def value(self):
        return self._v


class FakeAMPL:
    """Scripted amplpy.AMPL stand-in. Configure via class attributes per test."""

    solve_result = "solved"
    solve_result_num = 0
    obj = 2.0
    absgap = 0.0
    relgap = 0.0
    xvals: dict = {}

    def __init__(self):
        self.set = _Dictish()
        self.param = _Dictish()
        self.option = _Dictish()

    def eval(self, _s):
        pass

    def get_output(self, _cmd):
        return "Gurobi 13.0.2: optimal solution; objective X\n0 simplex iterations\n"

    def get_value(self, key):
        return {
            "solve_result": self.solve_result,
            "solve_result_num": self.solve_result_num,
            "_solve_time": 0.01,
            "Obj.absmipgap": self.absgap,
            "Obj.relmipgap": self.relgap,
        }[key]

    def get_variable(self, _n):
        return _Var(self.xvals)

    def get_objective(self, _n):
        return _Obj(self.obj)


@pytest.fixture(scope="module")
def cp3():
    return InstanceLoader().load("CP_3")


def _valid_xvals(n: int) -> dict:
    # every aircraft picks option 0 (a valid one-hot the decoder accepts)
    return {(i, 0): 1.0 for i in range(1, n + 1)}


def _patch(monkeypatch, **attrs):
    import amplpy

    fake = type("F", (FakeAMPL,), attrs)
    monkeypatch.setattr(amplpy, "AMPL", fake)


def test_solved_zero_gap_is_certified(cp3, monkeypatch):
    _patch(monkeypatch, solve_result="solved", solve_result_num=0, obj=2.0,
           absgap=0.0, relgap=0.0, xvals=_valid_xvals(cp3.n))
    r = discrete_milp.solve_discrete_milp(cp3, objective_id="maneuver_count_v1", n_theta=3)
    assert r.status == "optimal" and r.certified is True
    assert r.solver_version == "Gurobi 13.0.2" and r.best_bound == 2.0


def test_solved_with_nonzero_gap_is_not_certified_for_quadratic(cp3, monkeypatch):
    # a real but gap-tolerant "solved": abs gap 1e-3 exceeds the 1e-9 policy
    _patch(monkeypatch, solve_result="solved", solve_result_num=0, obj=0.5,
           absgap=1e-3, relgap=2e-3, xvals=_valid_xvals(cp3.n))
    r = discrete_milp.solve_discrete_milp(cp3, objective_id="quadratic_control_cost_v1", n_theta=3)
    assert r.status == "optimal" and r.certified is False


def test_maneuver_count_rejects_nonintegral_objective(cp3, monkeypatch):
    _patch(monkeypatch, solve_result="solved", solve_result_num=0, obj=2.4,
           absgap=0.0, relgap=0.0, xvals=_valid_xvals(cp3.n))
    r = discrete_milp.solve_discrete_milp(cp3, objective_id="maneuver_count_v1", n_theta=3)
    assert r.certified is False  # 2.4 is not an integer optimum


def test_timeout_with_incumbent_is_not_certified(cp3, monkeypatch):
    _patch(monkeypatch, solve_result="limit: time", solve_result_num=400, obj=2.0,
           xvals=_valid_xvals(cp3.n))
    r = discrete_milp.solve_discrete_milp(cp3, objective_id="maneuver_count_v1", n_theta=3)
    assert r.status == "timeout" and r.certified is False and r.optimum is None


def test_infeasible_status(cp3, monkeypatch):
    _patch(monkeypatch, solve_result="infeasible", solve_result_num=200)
    r = discrete_milp.solve_discrete_milp(cp3, objective_id="maneuver_count_v1", n_theta=3)
    assert r.status == "infeasible" and r.certified is False


def test_incoherent_status_is_not_certified(cp3, monkeypatch):
    # "solved" string but a non-zero solve_result_num => incoherent, never certified
    _patch(monkeypatch, solve_result="solved", solve_result_num=100,
           xvals=_valid_xvals(cp3.n))
    r = discrete_milp.solve_discrete_milp(cp3, objective_id="maneuver_count_v1", n_theta=3)
    assert r.certified is False and r.status in ("error", "timeout", "infeasible")


def test_unavailable_when_amplpy_absent(cp3, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "amplpy":
            raise ImportError("amplpy not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    r = discrete_milp.solve_discrete_milp(cp3, objective_id="maneuver_count_v1", n_theta=3)
    assert r.status == "unavailable" and r.certified is False
