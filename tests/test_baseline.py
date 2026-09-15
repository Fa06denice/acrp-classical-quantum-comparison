"""Scientific baseline deltas — the maths and the honest-None edge cases."""

from __future__ import annotations

import math

import pytest

from acrpq.benchmark.baseline import compute_baseline
from acrpq.benchmark.export import ReproInfo, RunRecord
from acrpq.benchmark.metrics import CommonMetrics


def _repro(backend: str | None = None) -> ReproInfo:
    return ReproInfo(
        seed=0,
        package_version="test",
        python_version="3.10",
        backend=backend,
        shots=None,
        deps={},
        timestamp_utc="1970-01-01T00:00:00+00:00",
        git_commit=None,
        git_dirty=None,
        source_hash=None,
        platform="test",
    )


def _rec(
    kind: str,
    status: str,
    objective: float,
    *,
    feasible: bool = True,
    extra: dict | None = None,
    backend: str | None = None,
) -> RunRecord:
    common = CommonMetrics(
        objective=objective,
        feasible=feasible,
        n_initial_conflicts=1,
        n_resolved_conflicts=1,
        n_residual_conflicts=0,
        resolution_rate=1.0,
        max_violation_depth=0.0,
        total_heading_change=0.0,
        total_speed_change=0.0,
        solve_time_s=0.0,
    )
    return RunRecord(
        scenario_id="Q4",
        instance_name="CP_4",
        family="CP",
        n=4,
        n_pairs=6,
        grid_k=5,
        solver_kind=kind,
        status=status,
        common=common,
        repro=_repro(backend),
        extra=extra or {},
    )


def test_all_three_deltas_and_ratios() -> None:
    recs = [
        _rec("original_ampl", "optimal", 0.10, extra={"provider_objective": 0.10, "integrity_ok": 1.0}),
        _rec("classical_minlp", "optimal", 0.12),
        _rec("classical_discrete", "optimal", 0.20),
        _rec("qubo_annealing", "feasible", 0.25),
    ]
    bl = compute_baseline(recs)
    assert bl is not None
    # adaptation = pyomo - original
    assert math.isclose(bl.delta_adaptation.delta, 0.02, abs_tol=1e-12)
    assert math.isclose(bl.delta_adaptation.ratio, 0.12 / 0.10, abs_tol=1e-12)
    # discretization = discrete - original
    assert math.isclose(bl.delta_discretization.delta, 0.10, abs_tol=1e-12)
    assert math.isclose(bl.delta_discretization.ratio, 0.20 / 0.10, abs_tol=1e-12)
    # algorithm = approx - discrete
    assert len(bl.delta_algorithm) == 1
    alg = bl.delta_algorithm[0]
    assert alg.value == "qubo_annealing" and alg.base == "discrete_reference"
    assert math.isclose(alg.delta, 0.05, abs_tol=1e-12)
    assert math.isclose(alg.ratio, 0.25 / 0.20, abs_tol=1e-12)
    # the only note identifies the official anchor; no problem notes
    assert any("official continuous anchor" in n for n in bl.notes)
    assert not any(("null" in n or "diverged" in n or "FEASIBLE" in n) for n in bl.notes)


def test_unavailable_original_suppresses_continuous_deltas() -> None:
    recs = [
        _rec("original_ampl", "unavailable", math.inf, feasible=False),
        _rec("classical_minlp", "optimal", 0.12),
        _rec("classical_discrete", "optimal", 0.20),
    ]
    bl = compute_baseline(recs)
    assert bl is not None
    assert bl.delta_adaptation.delta is None
    assert bl.delta_discretization.delta is None
    assert any("UNAVAILABLE" in n for n in bl.notes)


def test_zero_denominator_gives_none_ratio_but_keeps_delta() -> None:
    recs = [
        _rec("original_ampl", "optimal", 0.0),
        _rec("classical_minlp", "optimal", 0.05),
    ]
    bl = compute_baseline(recs)
    assert bl is not None
    assert math.isclose(bl.delta_adaptation.delta, 0.05, abs_tol=1e-12)
    assert bl.delta_adaptation.ratio is None
    assert "zero denominator" in bl.delta_adaptation.note


def test_integrity_divergence_flag_noted() -> None:
    recs = [
        _rec(
            "original_ampl",
            "optimal",
            0.10,
            extra={"provider_objective": 0.90, "integrity_ok": 0.0},
        ),
        _rec("classical_discrete", "optimal", 0.20),
    ]
    bl = compute_baseline(recs)
    assert bl is not None
    assert bl.legs["original_ampl"].integrity_ok is False
    assert any("diverged" in n for n in bl.notes)


def test_infeasible_approx_not_anchored() -> None:
    recs = [
        _rec("classical_discrete", "optimal", 0.20),
        _rec("qubo_annealing", "infeasible", math.inf, feasible=False),
    ]
    bl = compute_baseline(recs)
    assert bl is not None
    assert len(bl.delta_algorithm) == 1
    assert bl.delta_algorithm[0].delta is None


def test_empty_records_returns_none() -> None:
    assert compute_baseline([]) is None


# --- proven-optimum vs feasible-incumbent rigour --------------------------- #
def test_feasible_original_gives_null_official_but_provisional_metrics() -> None:
    recs = [
        _rec("original_ampl", "feasible", 0.10,
             extra={"provider_objective": 0.10, "integrity_ok": 1.0}),
        _rec("classical_minlp", "optimal", 0.12),
        _rec("classical_discrete", "optimal", 0.20),
    ]
    bl = compute_baseline(recs)
    assert bl is not None
    # original_ampl is only FEASIBLE -> NOT a proven continuous optimum
    assert bl.legs["original_ampl"].is_incumbent is True
    assert bl.legs["original_ampl"].is_optimal_anchor is False
    # official continuous deltas are null
    assert bl.delta_adaptation.delta is None
    assert bl.delta_discretization.delta is None
    # but provisional metrics vs the feasible incumbent ARE recorded and labelled
    bases = {d.base for d in bl.provisional_vs_feasible_incumbent}
    assert bases == {"original_ampl_feasible_incumbent"}
    prov_adapt = next(d for d in bl.provisional_vs_feasible_incumbent
                      if d.value == "pyomo_minlp")
    assert prov_adapt.delta == pytest.approx(0.02)
    assert any("FEASIBLE incumbent" in n for n in bl.notes)


def test_optimal_original_with_integrity_false_is_not_official_anchor() -> None:
    recs = [
        _rec("original_ampl", "optimal", 0.10,
             extra={"provider_objective": 0.90, "integrity_ok": 0.0}),
        _rec("classical_discrete", "optimal", 0.20),
    ]
    bl = compute_baseline(recs)
    assert bl is not None
    assert bl.legs["original_ampl"].is_optimal_anchor is False
    assert bl.delta_discretization.delta is None


def test_heuristic_feasible_discrete_gives_null_algorithm_delta() -> None:
    recs = [
        _rec("original_ampl", "optimal", 0.10,
             extra={"provider_objective": 0.10, "integrity_ok": 1.0}),
        _rec("classical_discrete", "feasible", 0.20),  # heuristic mode, not proven
        _rec("qubo_annealing", "feasible", 0.25),
    ]
    bl = compute_baseline(recs)
    assert bl is not None
    assert bl.legs["classical_discrete"].is_optimal_anchor is False
    # official algorithm delta is null because the discrete reference isn't proven
    assert all(d.delta is None for d in bl.delta_algorithm)
    # provisional algorithm metric vs the feasible discrete incumbent is recorded
    prov = [d for d in bl.provisional_vs_feasible_incumbent
            if d.base == "discrete_reference_feasible_incumbent"]
    assert prov and prov[0].value == "qubo_annealing"
    assert prov[0].delta == pytest.approx(0.05)
    assert any("heuristic incumbent" in n for n in bl.notes)


def test_couenne_leg_is_separate_and_not_the_official_anchor() -> None:
    # A couenne original_ampl run is kept as a distinct validation leg, NEVER merged
    # under `original_ampl` and never the official continuous anchor (gurobi is).
    recs = [
        _rec("original_ampl", "optimal", 0.10, backend="ampl:gurobi",
             extra={"provider_objective": 0.10, "integrity_ok": 1.0}),
        _rec("original_ampl", "optimal", 0.11, backend="ampl:couenne",
             extra={"provider_objective": 0.11, "integrity_ok": 1.0}),
        _rec("classical_discrete", "optimal", 0.20),
    ]
    bl = compute_baseline(recs)
    assert bl is not None
    assert "original_ampl" in bl.legs and "original_ampl_couenne" in bl.legs
    assert bl.legs["original_ampl"].ampl_solver == "gurobi"
    assert bl.legs["original_ampl_couenne"].ampl_solver == "couenne"
    # discretisation anchors on the GUROBI optimum (0.10), not couenne
    assert bl.delta_discretization.delta == pytest.approx(0.10)
    assert any("via gurobi" in n for n in bl.notes)
    assert any("original_ampl_couenne present as an OPTIONAL validation leg" in n
               for n in bl.notes)


def test_couenne_only_is_not_official_anchor() -> None:
    # Only couenne ran (no supervised gurobi) -> no official continuous anchor.
    recs = [
        _rec("original_ampl", "optimal", 0.11, backend="ampl:couenne",
             extra={"provider_objective": 0.11, "integrity_ok": 1.0}),
        _rec("classical_discrete", "optimal", 0.20),
    ]
    bl = compute_baseline(recs)
    assert bl is not None
    assert "original_ampl" not in bl.legs
    assert bl.delta_discretization.delta is None  # no supervised anchor


def test_proven_optima_still_produce_official_deltas() -> None:
    recs = [
        _rec("original_ampl", "optimal", 0.10,
             extra={"provider_objective": 0.10, "integrity_ok": 1.0}),
        _rec("classical_minlp", "optimal", 0.12),
        _rec("classical_discrete", "optimal", 0.20),
        _rec("qubo_annealing", "feasible", 0.25),
    ]
    bl = compute_baseline(recs)
    assert bl is not None
    assert bl.delta_adaptation.delta == pytest.approx(0.02)
    assert bl.delta_discretization.delta == pytest.approx(0.10)
    assert bl.delta_algorithm[0].delta == pytest.approx(0.05)
    assert bl.provisional_vs_feasible_incumbent == []
