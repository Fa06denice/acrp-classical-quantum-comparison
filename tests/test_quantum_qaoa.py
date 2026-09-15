"""QAOA end-to-end test — skipped automatically when the [quantum] extra is absent.

This keeps the default (dependency-free) test run green while still exercising
the real quantum path in environments that have Qiskit installed.
"""

from __future__ import annotations

import pytest

from acrpq._optional import have

pytestmark = pytest.mark.skipif(
    not (have("qiskit") and have("qiskit_optimization") and have("qiskit_aer")),
    reason="quantum extra (qiskit, qiskit-optimization, qiskit-aer) not installed",
)


def test_qaoa_resolves_cp3(cp3):
    from acrpq.quantum.backends import BackendConfig, BackendMode
    from acrpq.quantum.decode import decode
    from acrpq.quantum.qaoa import QAOASolver
    from acrpq.quantum.qubo import build_qubo

    qubo = build_qubo(cp3, n_theta=3, max_qubits=24)
    cfg = BackendConfig(mode=BackendMode.AER_SIM, shots=2048, seed=42)
    qres = QAOASolver(backend=cfg, reps=2, maxiter=120).solve(qubo)
    res = decode(qubo, qres.best_bitstring, seed=42, shots=cfg.shots, qaoa_reps=2)
    # On CP_3 with K=3, QAOA reliably finds a conflict-free assignment.
    assert res.feasible
    assert res.n_conflicts == 0
    assert qres.circuit_depth > 0


def test_qaoa_matches_reference_objective(cp3):
    from acrpq.benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(seed=42)
    ref, _ = runner.run_classical_reference("Q3f")
    res, rec = runner.run_quantum("Q3f", reps=2, maxiter=120, reference_obj=ref.objective)
    if res.feasible and ref.feasible:
        assert rec.quantum.approx_ratio is not None
        # QAOA should reach the in-grid optimum on this tiny case
        assert rec.quantum.approx_ratio <= 1.0 + 1e-6
