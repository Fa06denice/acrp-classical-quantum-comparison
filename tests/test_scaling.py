"""Qubit-scaling study tests."""

from __future__ import annotations

import csv
import json

import pytest

from acrpq._optional import have
from acrpq.benchmark.scaling import _CSV_COLS, SizeSpec, run_scaling_study


def test_dry_run_build_only(tmp_path):
    """Dry run builds QUBOs without QAOA; safe with no quantum extra."""
    result = run_scaling_study(
        [SizeSpec("CP_4", 3), SizeSpec("CP_6", 5)],
        methods=("statevector",),
        run_qaoa=False,
        out_dir=tmp_path,
    )
    assert len(result.cells) == 2
    assert all(c.status == "ok(build)" for c in result.cells)
    # CP_6 @ K=5 = 30 qubits -> flagged over budget
    big = next(c for c in result.cells if c.n_qubits == 30)
    assert big.within_budget is False
    with (tmp_path / "scaling.csv").open() as f:
        assert next(csv.reader(f)) == _CSV_COLS
    row = json.loads((tmp_path / "scaling.json").read_text())[0]
    assert row["seed"] == 42
    assert row["source_hash"]
    assert isinstance(row["git_dirty"], bool)


@pytest.mark.skipif(not have("qiskit"), reason="quantum extra not installed")
def test_over_budget_statevector_refused_without_alloc(tmp_path):
    result = run_scaling_study(
        [SizeSpec("CP_6", 5)],  # 30 qubits, over the statevector budget
        methods=("statevector",),
        run_qaoa=True,
        out_dir=tmp_path,
    )
    cell = result.cells[0]
    assert cell.status == "refused_memory"
    assert cell.sim_time_s is None  # never allocated a 30q state
