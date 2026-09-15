"""Tests for the classical QUBO solvers (pure Python, no qiskit)."""

from __future__ import annotations

import pytest

from acrpq.classical.reference import DiscreteReferenceSolver
from acrpq.quantum.qubo import build_qubo
from acrpq.quantum.qubo_solvers import QuboAnnealingSolver, QuboExactSolver


def test_qubo_exact_matches_reference(cp4):
    qubo = build_qubo(cp4, n_theta=5)
    ref = DiscreteReferenceSolver(n_theta=5).solve(cp4)
    ex = QuboExactSolver().solve(qubo)
    assert ex.feasible
    assert abs(ex.objective - ref.objective) < 1e-9
    # the energy equals the ACRP objective at the optimum (NOOP-cost convention)
    assert "qubo_energy" in ex.extra
    assert ex.extra["global_unconstrained_qubo"] == 0.0
    assert ex.extra["states_enumerated"] == 5**4


def test_qubo_exact_brute_force_bits_is_onehot(cp3):
    """Brute-forcing all 2**n_qubits states yields the same one-hot optimum."""
    qubo = build_qubo(cp3, n_theta=3)  # 9 qubits -> 512 states, fine
    block = QuboExactSolver(brute_force_bits=False).solve(qubo)
    bits = QuboExactSolver(brute_force_bits=True).solve(qubo)
    assert bits.extra["global_unconstrained_qubo"] == 1.0
    assert bits.extra["states_enumerated"] == 2**qubo.n_qubits
    assert bits.feasible == block.feasible
    assert abs(bits.objective - block.objective) < 1e-9


def test_qubo_annealing_resolves(cp4):
    qubo = build_qubo(cp4, n_theta=5)
    sa = QuboAnnealingSolver(seed=1).solve(qubo)
    assert sa.feasible
    # annealing should reach (or get close to) the exact optimum on this small case
    ex = QuboExactSolver().solve(qubo)
    assert sa.objective >= ex.objective - 1e-9


def test_qubo_annealing_deterministic(cp3):
    qubo = build_qubo(cp3, n_theta=5)
    a = QuboAnnealingSolver(seed=42).solve(qubo)
    b = QuboAnnealingSolver(seed=42).solve(qubo)
    assert a.objective == b.objective
    assert a.solution.choice == b.solution.choice


def test_qubo_solver_configuration_is_validated():
    with pytest.raises(ValueError, match="max_states"):
        QuboExactSolver(max_states=0)
    with pytest.raises(ValueError, match="restarts"):
        QuboAnnealingSolver(restarts=0)
    with pytest.raises(ValueError, match="t_start"):
        QuboAnnealingSolver(t_start=0)


def test_backend_configurations_are_validated():
    from acrpq.quantum.backends import BackendConfig
    from acrpq.quantum.dwave_solver import DWaveConfig

    with pytest.raises(ValueError, match="shots"):
        BackendConfig(shots=0)
    with pytest.raises(ValueError, match="optimization_level"):
        BackendConfig(optimization_level=4)
    with pytest.raises(ValueError, match="num_reads"):
        DWaveConfig(num_reads=0)
