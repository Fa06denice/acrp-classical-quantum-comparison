"""D-Wave solver tests (skipped without the [dwave] extra)."""

from __future__ import annotations

import math

import pytest

from acrpq._optional import have

pytestmark = pytest.mark.skipif(
    not (have("dimod") and have("dwave")), reason="[dwave] extra not installed"
)


def test_bqm_offset_equals_constant(cp3):
    from acrpq.quantum.dwave_solver import DWaveAnnealingSolver
    from acrpq.quantum.qubo import build_qubo

    qubo = build_qubo(cp3, n_theta=3)
    bqm = DWaveAnnealingSolver().to_bqm(qubo)
    assert bqm.offset == qubo.constant


def test_dwave_sa_feasible_and_energy_invariant(cp4):
    from acrpq.model import SolverKind
    from acrpq.quantum.dwave_solver import (
        DWaveAnnealingSolver,
        DWaveConfig,
        DWaveMode,
    )
    from acrpq.quantum.qubo import build_qubo

    qubo = build_qubo(cp4, n_theta=5)
    solver = DWaveAnnealingSolver(DWaveConfig(mode=DWaveMode.NEAL_SIM, num_reads=300, seed=1))
    res = solver.solve(qubo)
    assert res.solver == SolverKind.QUBO_DWAVE_SA
    assert res.feasible
    # the load-bearing gate: reported energy reproduces H(x)
    raw = solver.solve_raw(qubo)
    assert math.isclose(raw.best_energy, qubo.energy(raw.best_bitstring), abs_tol=1e-6)


def test_dwave_sa_matches_exact(cp3):
    from acrpq.quantum.dwave_solver import DWaveAnnealingSolver, DWaveConfig, DWaveMode
    from acrpq.quantum.qubo import build_qubo
    from acrpq.quantum.qubo_solvers import QuboExactSolver

    qubo = build_qubo(cp3, n_theta=5)
    ex = QuboExactSolver().solve(qubo)
    dw = DWaveAnnealingSolver(DWaveConfig(mode=DWaveMode.NEAL_SIM, num_reads=500, seed=1)).solve(qubo)
    assert dw.objective >= ex.objective - 1e-9  # SA reaches (or beats-equal) exact here
