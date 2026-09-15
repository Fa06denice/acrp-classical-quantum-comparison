"""Smoke test for the experimental campaign (skipped without the quantum extra)."""

from __future__ import annotations

import pytest

from acrpq._optional import have

pytestmark = pytest.mark.skipif(
    not (have("qiskit") and have("qiskit_optimization") and have("qiskit_aer")),
    reason="quantum extra not installed",
)


def test_campaign_smoke(tmp_path):
    from acrpq.benchmark.campaign import run_campaign

    result = run_campaign(
        ["Q3f"],
        reps_values=(1,),
        repetitions=2,
        maxiter=30,
        out_dir=tmp_path,
        base_seed=7,
    )
    assert len(result.runs) == 2
    assert len(result.points) == 1
    pt = result.points[0]
    assert pt.scenario_id == "Q3f"
    assert 0.0 <= pt.feasible_rate <= 1.0
    assert (tmp_path / "campaign_summary.csv").exists()
    assert (tmp_path / "campaign_runs.csv").exists()


def test_campaign_over_budget_is_logged(tmp_path):
    """A cell exceeding the qubit budget is recorded in `skipped`, not hidden."""
    from acrpq.benchmark.campaign import run_campaign

    result = run_campaign(
        ["Q4"],  # 20 qubits
        reps_values=(1,),
        repetitions=1,
        maxiter=20,
        max_qubits=8,  # force over-budget
        out_dir=tmp_path,
        base_seed=7,
    )
    assert result.skipped
    assert result.skipped[0]["scenario_id"] == "Q4"
