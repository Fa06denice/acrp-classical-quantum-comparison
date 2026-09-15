"""Sim-vs-hardware comparison test (simulator leg only; skipped without quantum)."""

from __future__ import annotations

import json

import pytest

from acrpq._optional import have

pytestmark = pytest.mark.skipif(
    not (have("qiskit") and have("qiskit_optimization") and have("qiskit_aer")),
    reason="quantum extra not installed",
)


def test_sim_only_comparison(tmp_path):
    from acrpq.benchmark.sim_vs_hardware import format_table, run_sim_vs_hardware

    result = run_sim_vs_hardware(
        ["Q3f"], reps=1, maxiter=30, include_hardware=False, out_dir=tmp_path
    )
    assert len(result.cells) == 1
    cell = result.cells[0]
    assert cell.backend_kind == "aer_sim"
    assert cell.is_simulator is True
    assert cell.error == ""
    assert (tmp_path / "sim_vs_hardware.csv").exists()
    row = json.loads((tmp_path / "sim_vs_hardware.json").read_text())[0]
    assert row["seed"] == 42
    assert row["source_hash"]
    assert isinstance(row["git_dirty"], bool)
    assert "scenario" in format_table(result)


def test_hardware_failure_is_captured_not_raised(tmp_path, monkeypatch):
    """A backend error on the hardware leg becomes a recorded cell, not a crash."""
    from acrpq.benchmark import sim_vs_hardware as svh

    # Force the quantum run to raise only for the hardware leg by pointing
    # ibm_real at a bogus backend name (no account call succeeds offline).
    result = svh.run_sim_vs_hardware(
        ["Q3f"],
        reps=1,
        maxiter=30,
        include_hardware=True,
        hardware_mode="ibm_real",
        backend_name="__nonexistent__",
        out_dir=tmp_path,
    )
    kinds = {c.backend_kind for c in result.cells}
    assert "aer_sim" in kinds and "ibm_real" in kinds
    hw = next(c for c in result.cells if c.backend_kind == "ibm_real")
    assert hw.error  # captured, not raised
