"""Experimental campaign: sweep scenarios x QAOA depth x repetitions.

A single benchmark run is one data point; a *campaign* repeats runs across
several scenarios, QAOA depths (``reps`` = the circuit-layer count ``p``) and
random seeds, then aggregates them into statistics suitable for the thesis
figures and tables:

* feasibility rate over repetitions (how often QAOA finds a conflict-free
  solution),
* mean / best approximation ratio versus the exact in-grid optimum,
* mean QAOA circuit depth and two-qubit-gate count,
* mean wall-clock time.

The campaign reuses :class:`~acrpq.benchmark.runner.BenchmarkRunner` (so the
classical anchor and the geometry-scored comparison are identical to a single
run) and writes two artefacts:

* ``campaign_runs.{csv,json}``  — every individual run (tidy, one row per run),
* ``campaign_summary.csv``      — one aggregated row per (scenario, reps).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..exceptions import QubitBudgetError
from .export import ReproInfo, _repro_row
from .runner import BenchmarkRunner


@dataclass
class CampaignPoint:
    """Aggregated statistics for one (scenario, reps) cell over repetitions."""

    scenario_id: str
    instance_name: str
    n: int
    n_pairs: int
    grid_k: int
    n_qubits: int
    reps: int
    repetitions: int
    feasible_rate: float  # fraction of repetitions that reached 0 conflicts
    mean_approx_ratio: float | None  # mean over feasible runs (>=1; 1 == optimal)
    best_approx_ratio: float | None
    mean_residual_conflicts: float
    mean_circuit_depth: float
    mean_two_qubit_gates: float
    mean_wall_time_s: float
    classical_objective: float
    classical_feasible: bool


@dataclass
class CampaignResult:
    points: list[CampaignPoint] = field(default_factory=list)
    runs: list[dict] = field(default_factory=list)  # tidy per-run rows
    skipped: list[dict] = field(default_factory=list)  # over-budget cells, logged not hidden


def run_campaign(
    scenarios: list[str],
    *,
    reps_values: tuple[int, ...] = (1, 2, 3),
    repetitions: int = 5,
    maxiter: int = 100,
    shots: int = 1024,
    base_seed: int = 1000,
    max_qubits: int = 24,
    out_dir: str | Path = "results",
    runner: BenchmarkRunner | None = None,
) -> CampaignResult:
    """Run a full simulator campaign and export tidy + summary artefacts.

    Each (scenario, reps) cell is run ``repetitions`` times with distinct seeds
    so feasibility/approx-ratio statistics are meaningful. The classical
    reference is computed once per scenario and reused as the anchor.
    """
    from ..quantum.backends import BackendConfig, BackendMode

    runner = runner or BenchmarkRunner(out_dir=out_dir, seed=base_seed)
    result = CampaignResult()

    for scenario_id in scenarios:
        # classical anchor (exact, deterministic) — computed once per scenario
        ref_result, ref_record = runner.run_classical_reference(scenario_id)
        ref_obj = ref_result.objective

        for reps in reps_values:
            approx_ratios: list[float] = []
            residuals: list[int] = []
            depths: list[int] = []
            twoq: list[int] = []
            times: list[float] = []
            feasible_count = 0
            cell_skipped = False

            for rep in range(repetitions):
                seed = base_seed + reps * 100 + rep
                cfg = BackendConfig(mode=BackendMode.AER_SIM, shots=shots, seed=seed)
                try:
                    q_result, q_record = runner.run_quantum(
                        scenario_id,
                        backend=cfg,
                        reps=reps,
                        maxiter=maxiter,
                        max_qubits=max_qubits,
                        reference_obj=ref_obj,
                    )
                except QubitBudgetError as exc:
                    result.skipped.append(
                        {"scenario_id": scenario_id, "reps": reps, "reason": str(exc)}
                    )
                    cell_skipped = True
                    break

                qm = q_record.quantum
                residuals.append(q_result.n_conflicts)
                if qm is not None:
                    depths.append(qm.circuit_depth)
                    twoq.append(qm.two_qubit_gates)
                times.append(q_result.wall_time_s)
                if q_result.feasible:
                    feasible_count += 1
                    if qm is not None and qm.approx_ratio is not None:
                        approx_ratios.append(qm.approx_ratio)

                result.runs.append(
                    {
                        "scenario_id": scenario_id,
                        "instance_name": q_record.instance_name,
                        "n": q_record.n,
                        "n_qubits": qm.n_qubits if qm else None,
                        "reps": reps,
                        "repetition": rep,
                        "seed": seed,
                        "feasible": q_result.feasible,
                        "objective": _clean(q_result.objective),
                        "approx_ratio": _clean(qm.approx_ratio) if qm else None,
                        "residual_conflicts": q_result.n_conflicts,
                        "circuit_depth": qm.circuit_depth if qm else None,
                        "two_qubit_gates": qm.two_qubit_gates if qm else None,
                        "wall_time_s": _clean(q_result.wall_time_s),
                        "classical_objective": _clean(ref_obj),
                        "backend": qm.backend_name if qm else None,
                        "shots": shots,
                        "lambda_pen": qm.lambda_pen if qm else None,
                        "lambda_oh": qm.lambda_oh if qm else None,
                        "package_version": q_record.repro.package_version,
                        "python_version": q_record.repro.python_version,
                        "git_commit": q_record.repro.git_commit,
                        "git_dirty": q_record.repro.git_dirty,
                        "source_hash": q_record.repro.source_hash,
                    }
                )

            if cell_skipped:
                continue

            ran = len(residuals)
            qm0 = q_record.quantum if ran else None
            result.points.append(
                CampaignPoint(
                    scenario_id=scenario_id,
                    instance_name=ref_record.instance_name,
                    n=ref_record.n,
                    n_pairs=ref_record.n_pairs,
                    grid_k=ref_record.grid_k,
                    n_qubits=qm0.n_qubits if qm0 else ref_record.grid_k * ref_record.n,
                    reps=reps,
                    repetitions=ran,
                    feasible_rate=feasible_count / ran if ran else 0.0,
                    mean_approx_ratio=_mean(approx_ratios),
                    best_approx_ratio=min(approx_ratios) if approx_ratios else None,
                    mean_residual_conflicts=_mean(residuals) or 0.0,
                    mean_circuit_depth=_mean(depths) or 0.0,
                    mean_two_qubit_gates=_mean(twoq) or 0.0,
                    mean_wall_time_s=_mean(times) or 0.0,
                    classical_objective=_clean(ref_obj) or math.inf,
                    classical_feasible=ref_result.feasible,
                )
            )

    _export(result, Path(out_dir), ReproInfo.capture(seed=base_seed))
    return result


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return None
    return v


def _export(result: CampaignResult, out: Path, repro: ReproInfo) -> None:
    out.mkdir(parents=True, exist_ok=True)
    provenance = _repro_row(repro)
    # tidy per-run (each run already carries the same provenance fields inline)
    (out / "campaign_runs.json").write_text(
        json.dumps(result.runs, indent=2), encoding="utf-8"
    )
    if result.runs:
        with (out / "campaign_runs.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(result.runs[0].keys()))
            w.writeheader()
            w.writerows(result.runs)
    # summary per cell — attach the same provenance so the aggregate is traceable
    rows = [{**asdict(p), **provenance} for p in result.points]
    (out / "campaign_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with (out / "campaign_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
