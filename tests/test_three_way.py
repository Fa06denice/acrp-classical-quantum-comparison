"""Three-way comparison tests (dwave leg skipped without the extra)."""

from __future__ import annotations

import csv
import json

import pytest

from acrpq._optional import have


def test_anchor_equals_discrete_reference(cp3):
    """The QUBO-exact anchor must equal the discrete-reference optimum."""
    from acrpq.classical.reference import DiscreteReferenceSolver
    from acrpq.discretize import discretize
    from acrpq.quantum.qubo import build_qubo
    from acrpq.quantum.qubo_solvers import QuboExactSolver

    d = discretize(cp3, n_theta=3)
    qubo = build_qubo(cp3, discrete=d)
    anchor = QuboExactSolver().solve(qubo)
    ref = DiscreteReferenceSolver().solve(cp3, discrete=d)
    assert abs(anchor.objective - ref.objective) < 1e-9


@pytest.mark.skipif(not (have("dimod") and have("dwave")), reason="[dwave] extra not installed")
def test_three_way_exact_plus_dwave(tmp_path):
    from acrpq.benchmark.three_way import _CSV_COLS, run_three_way

    result = run_three_way(
        ["Q2"], include_ibm=False, include_dwave=True, num_reads=200, out_dir=tmp_path
    )
    routes = {c.route for c in result.cells}
    assert "classical_exact" in routes and "dwave" in routes
    assert all(c.feasible for c in result.cells)
    # CSV header matches the declared columns
    with (tmp_path / "three_way.csv").open() as f:
        header = next(csv.reader(f))
    assert header == _CSV_COLS
    row = json.loads((tmp_path / "three_way.json").read_text())[0]
    assert row["seed"] == 42
    assert row["source_hash"]
    assert isinstance(row["git_dirty"], bool)
