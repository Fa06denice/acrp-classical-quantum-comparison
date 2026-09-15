from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "build_thesis_benchmark_v2.py"
    spec = importlib.util.spec_from_file_location("build_thesis_benchmark_v2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cp_size_is_available_for_timeout_rows() -> None:
    module = _module()
    assert module._cp_size("CP_8") == 8
    with pytest.raises(ValueError, match="unexpected"):
        module._cp_size("RCP_10_1")


def test_validation_rejects_infeasible_gap() -> None:
    module = _module()
    row = {
        "job_key": "bad",
        "objective_id": "maneuver_count_v1",
        "feasible": False,
        "gap_to_exhaustive_optimum": 0.0,
        "gap_to_certified_milp": None,
    }
    with pytest.raises(ValueError, match="infeasible row carries a gap"):
        module._validate([row])


def test_csv_parity_detects_tampering(tmp_path: Path) -> None:
    module = _module()
    rows = [{"job_key": "a", "value": 1}, {"job_key": "b", "value": None}]
    path = tmp_path / "rows.csv"
    module._write_csv(path, rows)
    module._assert_csv_parity(path, rows)
    path.write_text(path.read_text().replace("1", "2", 1))
    with pytest.raises(ValueError, match="CSV rows differ"):
        module._assert_csv_parity(path, rows)
