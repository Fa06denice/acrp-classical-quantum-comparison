"""Simulator-vs-hardware comparison for the QAOA ACRP benchmark.

Runs the *same* reduced scenarios on the local Aer simulator and on a real IBM
QPU, scoring both with the identical geometry kernel, and tabulates the gap
attributable to hardware noise — the headline experimental contribution for the
thesis (brief §7.3, §13).

Each (scenario, backend) cell records feasibility, the approximation ratio
versus the exact in-grid optimum, the QAOA circuit depth, and the backend name.
Results are written incrementally to ``sim_vs_hardware.{json,csv}`` so a
partially-completed run (e.g. if a hardware job is cancelled) is not lost.

The classical reference (exact, deterministic) is computed once per scenario and
used as the shared anchor for both backends.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .export import _REPRO_COLUMNS, ReproInfo, _repro_row
from .runner import BenchmarkRunner


@dataclass
class CompareCell:
    """One (scenario, backend) measurement."""

    scenario_id: str
    instance_name: str
    n: int
    n_qubits: int
    reps: int
    backend_kind: str  # "aer_sim" | "ibm_real" | "ibm_runtime"
    backend_name: str
    is_simulator: bool
    feasible: bool
    n_residual_conflicts: int
    objective: float | None
    approx_ratio: float | None
    circuit_depth: int
    two_qubit_gates: int
    shots: int
    wall_time_s: float  # connect + queue + execute (queue dominates on open plan)
    qpu_time_s: float | None  # real quantum-processor seconds (IBM only)
    n_jobs: int
    classical_objective: float | None
    error: str = ""


@dataclass
class CompareResult:
    cells: list[CompareCell] = field(default_factory=list)


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return None
    return v


def run_sim_vs_hardware(
    scenarios: list[str],
    *,
    reps: int = 1,
    shots: int = 1024,
    seed: int = 42,
    maxiter: int = 100,
    include_hardware: bool = True,
    hardware_mode: str = "ibm_real",  # or "ibm_runtime" for the cloud simulator
    backend_name: str | None = None,
    out_dir: str | Path = "results",
    runner: BenchmarkRunner | None = None,
) -> CompareResult:
    """Run each scenario on the Aer simulator and (optionally) real IBM hardware.

    Writes ``sim_vs_hardware.{json,csv}`` incrementally. Hardware jobs may queue
    for a long time; the simulator cells are always computed first so partial
    output is useful even if a hardware job never returns.
    """
    from ..quantum.backends import BackendConfig, BackendMode

    runner = runner or BenchmarkRunner(out_dir=out_dir, seed=seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = CompareResult()
    repro = ReproInfo.capture(seed=seed)

    backends = [("aer_sim", BackendMode.AER_SIM)]
    if include_hardware:
        backends.append((hardware_mode, BackendMode(hardware_mode)))

    for scenario_id in scenarios:
        ref_result, ref_record = runner.run_classical_reference(scenario_id)
        ref_obj = ref_result.objective

        for kind, mode in backends:
            cfg = BackendConfig(
                mode=mode, shots=shots, seed=seed, backend_name=backend_name
            )
            cell = _run_cell(runner, scenario_id, cfg, kind, reps, maxiter, ref_obj, ref_record)
            result.cells.append(cell)
            _export(result, out, repro)  # incremental save after every cell

    return result


def _run_cell(runner, scenario_id, cfg, kind, reps, maxiter, ref_obj, ref_record) -> CompareCell:
    base = dict(
        scenario_id=scenario_id,
        instance_name=ref_record.instance_name,
        n=ref_record.n,
        reps=reps,
        backend_kind=kind,
        shots=cfg.shots,
        classical_objective=_clean(ref_obj),
    )
    fallback_qubits = ref_record.grid_k * ref_record.n
    try:
        q_result, q_record = runner.run_quantum(
            scenario_id, backend=cfg, reps=reps, maxiter=maxiter, reference_obj=ref_obj
        )
        qm = q_record.quantum
        return CompareCell(
            **base,
            n_qubits=qm.n_qubits if qm else fallback_qubits,
            backend_name=qm.backend_name if qm else cfg.label,
            is_simulator=bool(qm.is_simulator) if qm else True,
            feasible=q_result.feasible,
            n_residual_conflicts=q_result.n_conflicts,
            objective=_clean(q_result.objective),
            approx_ratio=_clean(qm.approx_ratio) if qm else None,
            circuit_depth=qm.circuit_depth if qm else 0,
            two_qubit_gates=qm.two_qubit_gates if qm else 0,
            wall_time_s=_clean(q_result.wall_time_s) or 0.0,
            qpu_time_s=_clean(q_result.extra.get("qpu_time_s")),
            n_jobs=int(q_result.extra.get("n_jobs", 0)),
        )
    except Exception as exc:  # a hardware failure must not lose the other cells
        return CompareCell(
            **base,
            n_qubits=fallback_qubits,
            backend_name=cfg.label,
            is_simulator=(kind == "aer_sim"),
            feasible=False,
            n_residual_conflicts=-1,
            objective=None,
            approx_ratio=None,
            circuit_depth=0,
            two_qubit_gates=0,
            wall_time_s=0.0,
            qpu_time_s=None,
            n_jobs=0,
            error=f"{type(exc).__name__}: {exc}",
        )


_CSV_COLS = [
    "scenario_id", "instance_name", "n", "n_qubits", "reps",
    "backend_kind", "backend_name", "is_simulator",
    "feasible", "n_residual_conflicts", "objective", "approx_ratio",
    "circuit_depth", "two_qubit_gates", "shots",
    "wall_time_s", "qpu_time_s", "n_jobs",
    "classical_objective", "error",
] + _REPRO_COLUMNS


def _export(result: CompareResult, out: Path, repro: ReproInfo) -> None:
    provenance = _repro_row(repro)
    rows = [{**asdict(c), **provenance} for c in result.cells]
    (out / "sim_vs_hardware.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out / "sim_vs_hardware.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in _CSV_COLS})


def format_table(result: CompareResult) -> str:
    """Render a compact text table (simulator vs hardware side by side).

    ``wall(s)`` is total wall-clock (incl. queue); ``qpu(s)`` is the real
    quantum-processor time when the backend reports usage (IBM only).
    """
    lines = [
        f"{'scenario':9s} {'n':>2s} {'qubits':>6s} {'backend':16s} {'sim?':4s} "
        f"{'feas':4s} {'approx':>7s} {'depth':>6s} {'wall(s)':>8s} {'qpu(s)':>7s}",
        "-" * 84,
    ]
    for c in result.cells:
        ar = f"{c.approx_ratio:.3f}" if c.approx_ratio is not None else "  -  "
        feas = "yes" if c.feasible else "NO"
        sim = "sim" if c.is_simulator else "QPU"
        qpu = f"{c.qpu_time_s:.2f}" if c.qpu_time_s is not None else "  -  "
        note = f"  ! {c.error[:40]}" if c.error else ""
        lines.append(
            f"{c.scenario_id:9s} {c.n:>2d} {c.n_qubits:>6d} {c.backend_name:16.16s} {sim:4s} "
            f"{feas:4s} {ar:>7s} {c.circuit_depth:>6d} {c.wall_time_s:>8.1f} {qpu:>7s}{note}"
        )
    return "\n".join(lines)
