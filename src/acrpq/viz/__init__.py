"""Plotting helpers (OPTIONAL — requires the ``[viz]`` extra, matplotlib)."""

from .plots import (
    plot_benchmark_objective,
    plot_conflict_matrix,
    plot_problem_size_vs_qubits,
    plot_trajectories,
)

__all__ = [
    "plot_trajectories",
    "plot_conflict_matrix",
    "plot_benchmark_objective",
    "plot_problem_size_vs_qubits",
]
