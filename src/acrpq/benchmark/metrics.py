"""Benchmark metrics — common to both approaches, plus quantum-specific ones."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .. import geometry
from ..model import Instance, Result


@dataclass(frozen=True)
class CommonMetrics:
    """Metrics computable for any :class:`Result` (classical or quantum)."""

    objective: float
    feasible: bool
    n_initial_conflicts: int
    n_resolved_conflicts: int
    n_residual_conflicts: int
    resolution_rate: float  # resolved / initial (1.0 if no initial conflicts)
    max_violation_depth: float  # worst shortfall below d among residual conflicts
    total_heading_change: float  # sum |theta|
    total_speed_change: float  # sum |1 - q|
    solve_time_s: float


def common_metrics(result: Result, inst: Instance) -> CommonMetrics:
    sol = result.solution
    n_init = result.n_initial_conflicts if result.n_initial_conflicts >= 0 else (
        geometry.count_initial_conflicts(inst)
    )
    if sol is None:
        return CommonMetrics(
            objective=math.inf,
            feasible=False,
            n_initial_conflicts=n_init,
            n_resolved_conflicts=0,
            n_residual_conflicts=n_init,
            resolution_rate=0.0,
            max_violation_depth=math.inf,
            total_heading_change=0.0,
            total_speed_change=0.0,
            solve_time_s=result.wall_time_s,
        )

    residual = result.n_conflicts if result.n_conflicts >= 0 else (
        geometry.count_conflicts(inst, sol.q, sol.theta)
    )
    resolved = max(0, n_init - residual)
    rate = 1.0 if n_init == 0 else resolved / n_init
    reports = geometry.conflict_reports(inst, sol.q, sol.theta)
    max_depth = max((r.depth for r in reports if r.in_conflict), default=0.0)
    return CommonMetrics(
        objective=result.objective,
        feasible=result.feasible,
        n_initial_conflicts=n_init,
        n_resolved_conflicts=resolved,
        n_residual_conflicts=residual,
        resolution_rate=rate,
        max_violation_depth=max_depth,
        total_heading_change=sum(abs(t) for t in sol.theta),
        total_speed_change=sum(abs(1.0 - q) for q in sol.q),
        solve_time_s=result.wall_time_s,
    )


@dataclass(frozen=True)
class QuantumMetrics:
    """QAOA-specific metrics for the quantum results table."""

    n_aircraft: int
    n_pairs: int
    maneuvers_per_aircraft: int
    n_qubits: int
    n_onehot_terms: int
    n_conflict_terms: int
    qaoa_reps: int
    circuit_depth: int
    circuit_size: int
    two_qubit_gates: int
    mode: str
    backend_name: str
    is_simulator: bool
    shots: int
    seed: int
    optimizer: str
    optimizer_evals: int
    wall_time_s: float
    lambda_pen: float
    lambda_oh: float
    best_energy: float
    feasible: bool
    n_residual_conflicts: int
    acrp_objective: float
    approx_ratio: float | None  # acrp_objective / reference optimum (if both feasible)
    best_prob: float | None


def quantum_metrics(
    *,
    inst: Instance,
    discrete,
    qubo,
    qaoa_result,
    backend_config,
    decoded: Result,
    optimizer: str,
    reps: int,
    reference_obj: float | None,
    best_prob: float | None,
) -> QuantumMetrics:
    """Assemble :class:`QuantumMetrics` from a QAOA run and its decoded result."""
    approx = None
    if reference_obj is not None and decoded.feasible and reference_obj > 0:
        approx = decoded.objective / reference_obj
    name = qaoa_result.backend_meta.get("name", backend_config.label)
    return QuantumMetrics(
        n_aircraft=inst.n,
        n_pairs=inst.n_pairs(),
        maneuvers_per_aircraft=discrete.k,
        n_qubits=qubo.n_qubits,
        n_onehot_terms=qubo.n_onehot_terms,
        n_conflict_terms=qubo.n_conflict_terms,
        qaoa_reps=reps,
        circuit_depth=qaoa_result.circuit_depth,
        circuit_size=qaoa_result.circuit_size,
        two_qubit_gates=qaoa_result.two_qubit_gates,
        mode=backend_config.mode.value,
        backend_name=name,
        is_simulator=bool(qaoa_result.backend_meta.get("is_simulator", True)),
        shots=backend_config.shots,
        seed=backend_config.seed,
        optimizer=optimizer,
        optimizer_evals=qaoa_result.eval_count,
        wall_time_s=qaoa_result.wall_time_s,
        lambda_pen=qubo.penalty_conflict,
        lambda_oh=qubo.penalty_onehot,
        best_energy=qaoa_result.best_energy,
        feasible=decoded.feasible,
        n_residual_conflicts=decoded.n_conflicts,
        acrp_objective=decoded.objective,
        approx_ratio=approx,
        best_prob=best_prob,
    )
