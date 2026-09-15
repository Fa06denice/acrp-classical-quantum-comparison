"""Alternative AMPL objective: original geometry, minimum moved aircraft."""
from __future__ import annotations

import json
import importlib.util
import os
import signal
import time
from pathlib import Path

import pytest

from acrpq.classical.discrete_enum import exhaustive_discrete_optimum
from acrpq.classical import original_ampl
from acrpq.classical.original_ampl_maneuver_count import (
    MODEL_VARIANT,
    OBJECTIVE_ID,
    OriginalAMPLManeuverCountSolver,
    _addon_path,
)


def _block(name: str, body: str) -> str:
    return f"ACRPQ_{name}_BEGIN\n{body}\nACRPQ_{name}_END\n"


def _valid_output(cp3) -> str:
    ref = exhaustive_discrete_optimum(cp3, objective_id=OBJECTIVE_ID, n_theta=3)
    assert ref.feasible
    moved = [int(q != 1.0 or theta != 0.0) for q, theta in zip(ref.q, ref.theta, strict=True)]
    def indexed(values):
        return "\n".join(f"{i},{v:.12g}" for i, v in enumerate(values, 1))
    return (
        "Gurobi 13.0.2: optimal solution\n"
        "absmipgap=0, relmipgap=0\n"
        + _block("STATUS", "solved")
        + _block("SRNUM", "0")
        + _block("OBJECTIVE", str(sum(moved)))
        + _block("SOLVETIME", "0.1")
        + _block("Q", indexed(ref.q))
        + _block("THETA", indexed(ref.theta))
        + _block("MOVED", indexed(moved))
    )


def test_addon_only_adds_indicator_objective_and_links():
    text = _addon_path().read_text()
    assert "var moved{i in A} binary" in text
    assert "minimize ManeuverCount: sum{i in A} moved[i]" in text
    assert all(name in text for name in (
        "LinkThetaUpper", "LinkThetaLower", "LinkSpeedUpper", "LinkSpeedLower"
    ))
    assert "bin11" not in text and "cvrx" not in text  # historical geometry is not copied


def test_run_script_loads_historical_model_then_addon(cp3, tmp_path):
    solver = OriginalAMPLManeuverCountSolver()
    script = solver._run_script(
        tmp_path / "NC_ACRP.mod", tmp_path / "Preprocessing.run", tmp_path / "CP_3.dat", cp3
    )
    assert script.index("NC_ACRP.mod") < script.index("NC_ACRP_maneuver_count.mod")
    assert "problem NC_ACRP_COUNT: ManeuverCount" in script
    assert "solve NC_ACRP_COUNT" in script
    assert "printf \"%.12g\\n\", ManeuverCount" in script
    assert " Obj," not in script


def test_parser_rescores_count_and_preserves_quadratic_as_secondary(cp3):
    solver = OriginalAMPLManeuverCountSolver()
    result = solver._parse_and_score(
        cp3,
        3,
        time.perf_counter(),
        {"stdout": _valid_output(cp3), "stderr": "", "returncode": 0, "timed_out": False},
        {"timeout_s": 120.0, "objective_id": OBJECTIVE_ID, "model_variant": MODEL_VARIANT},
    )
    meta = json.loads(result.message)
    assert result.status.value == "optimal"
    assert result.feasible and result.n_conflicts == 0
    assert result.objective == result.extra["maneuver_count"] == 2.0
    assert result.extra["objective_abs_delta"] == 0.0
    assert result.extra["integer_optimality_certified"] == 1.0
    assert result.extra["historical_quadratic_objective"] > 0.0
    assert meta["objective_id"] == OBJECTIVE_ID
    assert meta["integer_optimality_certified"] is True


def test_missing_indicator_block_fails_closed(cp3):
    solver = OriginalAMPLManeuverCountSolver()
    text = _valid_output(cp3).split("ACRPQ_MOVED_BEGIN", 1)[0]
    result = solver._parse_and_score(
        cp3,
        3,
        time.perf_counter(),
        {"stdout": text, "stderr": "", "returncode": 0, "timed_out": False},
        {"timeout_s": 120.0, "objective_id": OBJECTIVE_ID, "model_variant": MODEL_VARIANT},
    )
    assert result.status.value == "error"
    assert result.solution is None and not result.feasible


@pytest.mark.parametrize("value", [None, float("inf"), float("-inf"), float("nan"), True])
def test_campaign_serialises_failed_numeric_sentinels_as_null(value):
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_ampl_maneuver_count_campaign.py"
    spec = importlib.util.spec_from_file_location("maneuver_count_campaign", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._finite_or_none(value) is None


def test_operator_interrupt_kills_solver_process_group(monkeypatch, tmp_path):
    class InterruptedProcess:
        pid = 123

        def __init__(self):
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt
            return "", ""

    proc = InterruptedProcess()
    killed = []
    monkeypatch.setattr(original_ampl.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(original_ampl.os, "getpgid", lambda pid: 777)
    monkeypatch.setattr(original_ampl.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
    with pytest.raises(KeyboardInterrupt):
        OriginalAMPLManeuverCountSolver()._invoke("ampl", tmp_path / "x.run", tmp_path, 120)
    assert killed == [(777, signal.SIGTERM)]


@pytest.mark.ampl
def test_real_cp3_matches_discrete_count_when_enabled(loader):
    if os.environ.get("ACRPQ_RUN_REAL_AMPL") != "1":
        pytest.skip("set ACRPQ_RUN_REAL_AMPL=1 for the real licensed AMPL/Gurobi test")
    inst = loader.load("CP_3")
    dat, _ = loader.read_dat_text("CP_3")
    result = OriginalAMPLManeuverCountSolver(timeout_s=120).solve(inst, dat_text=dat)
    discrete = exhaustive_discrete_optimum(inst, objective_id=OBJECTIVE_ID, n_theta=3)
    assert result.status.value == "optimal" and result.feasible
    assert result.extra["integer_optimality_certified"] == 1.0
    assert result.objective == discrete.optimum
