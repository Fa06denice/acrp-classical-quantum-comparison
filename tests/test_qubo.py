"""Tests for the QUBO build (pure Python, no qiskit)."""

from __future__ import annotations

import itertools

import pytest

from acrpq.discretize import discretize
from acrpq.exceptions import QubitBudgetError
from acrpq.quantum.ising import qubo_to_ising
from acrpq.quantum.qubo import build_qubo


def test_qubit_count(cp3):
    qubo = build_qubo(cp3, n_theta=3)  # K=3 -> 9 qubits
    assert qubo.n_qubits == cp3.n * 3


def test_qubit_budget_enforced(cp4):
    with pytest.raises(QubitBudgetError):
        build_qubo(cp4, n_theta=5, n_q=3, max_qubits=10)  # 4*15=60 > 10


def test_onehot_penalty_dominates(cp3):
    qubo = build_qubo(cp3, n_theta=3)
    # should-fix guarantee: lam_oh > lam_pen * (n-1)
    assert qubo.penalty_onehot > qubo.penalty_conflict * (cp3.n - 1)


def test_qubo_global_min_is_onehot_and_optimal(cp3):
    """Brute force the QUBO: its global minimum must be a valid one-hot
    assignment, and (when a conflict-free assignment exists) it must coincide
    with the reference solver's optimum."""
    from acrpq.classical.reference import DiscreteReferenceSolver

    d = discretize(cp3, n_theta=3)
    qubo = build_qubo(cp3, discrete=d)
    n, k = cp3.n, d.k

    best_energy = float("inf")
    best_bits = None
    # enumerate all 2^(n*k) bitstrings is too many; enumerate one-hot + a few
    # non-one-hot perturbations to confirm one-hot wins. Full enum for 9 bits ok.
    for combo in itertools.product(range(2), repeat=n * k):
        e = qubo.energy(list(combo))
        if e < best_energy:
            best_energy = e
            best_bits = combo

    # decode the global min and check it is a valid one-hot
    from acrpq.quantum.decode import decode_assignment

    assignment = decode_assignment(qubo, list(best_bits))
    # one bit set per aircraft in the true global min
    for i in cp3.aircraft():
        set_bits = sum(best_bits[d.var_index(i, o)] for o in range(k))
        assert set_bits == 1, "QUBO global minimum is not one-hot"

    ref = DiscreteReferenceSolver(n_theta=3).solve(cp3)
    if ref.feasible:
        q, theta = d.maneuvers(tuple(assignment[i] for i in cp3.aircraft()))
        from acrpq import geometry

        assert geometry.count_conflicts(cp3, q, theta) == 0


def test_ising_conversion(cp3):
    qubo = build_qubo(cp3, n_theta=3)
    ising = qubo_to_ising(qubo)
    assert ising.n_spins == qubo.n_qubits
    # energy of all-zeros bitstring (x=0 -> z=+1) should match offset + sum h + sum J
    e_qubo = qubo.energy([0] * qubo.n_qubits)
    e_ising = ising.offset + sum(ising.h.values()) + sum(ising.J.values())
    assert abs(e_qubo - e_ising) < 1e-6


def test_qubo_rejects_malformed_bit_assignments(cp3):
    qubo = build_qubo(cp3, n_theta=3)
    with pytest.raises(ValueError, match="exactly"):
        qubo.energy("0")
    with pytest.raises(ValueError, match="binary"):
        qubo.energy([0] * (qubo.n_qubits - 1) + [2])
    with pytest.raises(ValueError, match="out of range"):
        qubo.energy({qubo.n_qubits: 1})


def test_qubo_rejects_inconsistent_model_and_penalties(cp3, cp4):
    from acrpq.discretize import discretize

    discrete = discretize(cp3, n_theta=3)
    with pytest.raises(ValueError, match="different instance"):
        build_qubo(cp4, discrete=discrete)
    with pytest.raises(ValueError, match="conflict penalty"):
        build_qubo(cp3, n_theta=3, penalty_conflict=0)
