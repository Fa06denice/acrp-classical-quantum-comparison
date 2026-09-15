"""End-to-end benchmark tests that run with the reference solver only."""

from __future__ import annotations

import json

from acrpq.benchmark.export import write_csv, write_json
from acrpq.benchmark.runner import BenchmarkRunner


def test_reference_only_compare(tmp_path):
    runner = BenchmarkRunner(out_dir=tmp_path, seed=1)
    records = runner.compare("Q3f", with_quantum=False)
    assert len(records) == 1
    rec = records[0]
    assert rec.solver_kind == "classical_discrete"
    assert rec.common.feasible
    assert (tmp_path / "benchmark_Q3f.csv").exists()
    assert (tmp_path / "benchmark_Q3f.json").exists()


def test_export_roundtrip(tmp_path):
    runner = BenchmarkRunner(out_dir=tmp_path, seed=1)
    records = runner.compare("Q2", with_quantum=False, write=False)
    jpath = write_json(records, tmp_path / "r.json")
    cpath = write_csv(records, tmp_path / "r.csv")
    data = json.loads(jpath.read_text())
    assert data[0]["repro"]["seed"] == 1
    assert "package_version" in data[0]["repro"]
    assert data[0]["repro"]["source_hash"]
    # git_dirty reflects the working tree: True in dev, False on a clean CI checkout
    assert isinstance(data[0]["repro"]["git_dirty"], bool)
    # CSV has a header + one row
    rows = cpath.read_text().strip().splitlines()
    assert len(rows) == 2
    assert rows[0].startswith("scenario_id,")


def test_record_uses_actual_solver_seed(tmp_path):
    from acrpq.model import Result, SolverKind, Status

    runner = BenchmarkRunner(out_dir=tmp_path, seed=1)
    sc, inst, discrete = runner._resolve("Q2")
    result = Result(
        instance_name=inst.name,
        solver=SolverKind.QUANTUM_QAOA,
        status=Status.ERROR,
        solution=None,
        seed=987,
    )
    record = runner._record(sc, inst, discrete, result)
    assert record.repro.seed == 987


def test_metrics_resolution_rate(loader):
    from acrpq.benchmark.metrics import common_metrics
    from acrpq.classical.reference import DiscreteReferenceSolver

    inst = loader.load("CP_3")
    res = DiscreteReferenceSolver(n_theta=5).solve(inst)
    m = common_metrics(res, inst)
    assert m.n_initial_conflicts == inst.n_pairs()
    assert m.n_residual_conflicts == 0
    assert m.resolution_rate == 1.0
