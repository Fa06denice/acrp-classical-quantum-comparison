"""Three-way comparison: classical-exact vs IBM-QAOA vs D-Wave, one QUBO.

Runs the *same* ``ManeuverQUBO`` per scenario through:

* the exact in-grid optimum (:class:`QuboExactSolver`) — the anchor,
* IBM QAOA (Aer simulator and/or a real IBM QPU),
* D-Wave (reference simulated annealing and/or a real D-Wave QPU),

all decoded through the identical geometry kernel. The purest cross-hardware
metric is ``energy_ratio`` (the routes minimise *identical* ``H(x)``); the
ACRP-level metric is ``approx_ratio`` (decoded objective vs the exact in-grid
optimum). Each cell is wrapped so a missing Leap token / IBM queue failure never
kills the other routes, and results are written incrementally.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypedDict

from .export import _REPRO_COLUMNS, ReproInfo, _repro_row
from .runner import BenchmarkRunner


@dataclass
class ThreeWayCell:
    scenario_id: str
    instance_name: str
    n: int
    n_qubits: int
    grid_k: int
    route: str  # classical_exact | ibm_qaoa | dwave
    method: str  # qubo_exact | aer_sim | ibm_real | dwave_sa | dwave_qpu
    backend_name: str
    is_quantum_hw: bool
    feasible: bool
    n_residual_conflicts: int
    acrp_objective: float | None
    qubo_energy: float | None
    approx_ratio: float | None
    energy_ratio: float | None
    best_feasible_prob: float | None = None
    wall_time_s: float = 0.0
    reps: int | None = None
    circuit_depth: int | None = None
    two_qubit_gates: int | None = None
    shots: int | None = None
    qpu_time_s: float | None = None
    num_reads: int | None = None
    qpu_access_time_s: float | None = None
    chain_break_mean: float | None = None
    chain_break_max: float | None = None
    n_physical_qubits: int | None = None
    embedding_max_chain: int | None = None
    anchor_objective: float | None = None
    anchor_qubo_energy: float | None = None
    error: str = ""


@dataclass
class ThreeWayResult:
    cells: list[ThreeWayCell] = field(default_factory=list)


class _CellBase(TypedDict):
    scenario_id: str
    instance_name: str
    n: int
    n_qubits: int
    grid_k: int


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return None
    return v


def run_three_way(
    scenarios: list[str],
    *,
    include_ibm: bool = False,
    include_ibm_real: bool = False,
    include_dwave: bool = True,
    include_dwave_real: bool = False,
    reps: int = 2,
    shots: int = 1024,
    maxiter: int = 100,
    num_reads: int = 1000,
    seed: int = 42,
    max_qubits: int = 29,
    ibm_backend_name: str | None = None,
    dwave_solver_name: str | None = None,
    out_dir: str | Path = "results",
    runner: BenchmarkRunner | None = None,
) -> ThreeWayResult:
    """Run the enabled routes on the same QUBO per scenario; export CSV/JSON."""
    from ..classical.reference import DiscreteReferenceSolver
    from ..quantum.qubo import build_qubo
    from ..quantum.qubo_solvers import QuboExactSolver

    runner = runner or BenchmarkRunner(out_dir=out_dir, seed=seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = ThreeWayResult()
    repro = ReproInfo.capture(seed=seed)

    for scenario_id in scenarios:
        sc, inst, discrete = runner._resolve(scenario_id)
        base: _CellBase = {
            "scenario_id": sc.id,
            "instance_name": inst.name,
            "n": inst.n,
            "n_qubits": discrete.n_vars,
            "grid_k": discrete.k,
        }
        try:
            qubo = build_qubo(inst, discrete=discrete, max_qubits=max_qubits)
        except Exception as exc:
            result.cells.append(_error_cell(base, "classical_exact", "qubo_exact", exc))
            _export(result, out, repro)
            continue

        # --- anchor: exact in-grid optimum on H(x) ---
        anchor = QuboExactSolver().solve(qubo)
        anchor_obj = _clean(anchor.objective)
        anchor_energy = _clean(anchor.extra.get("qubo_energy"))
        # sanity: exact QUBO optimum == discrete reference optimum
        ref = DiscreteReferenceSolver().solve(inst, discrete=discrete)
        result.cells.append(
            ThreeWayCell(
                **base, route="classical_exact", method="qubo_exact",
                backend_name="cpu", is_quantum_hw=False, feasible=anchor.feasible,
                n_residual_conflicts=anchor.n_conflicts, acrp_objective=anchor_obj,
                qubo_energy=anchor_energy, approx_ratio=1.0 if anchor.feasible else None,
                energy_ratio=1.0, wall_time_s=_clean(anchor.wall_time_s) or 0.0,
                anchor_objective=anchor_obj, anchor_qubo_energy=anchor_energy,
                error="" if abs(ref.objective - anchor.objective) < 1e-9 else
                      "anchor != discrete reference",
            )
        )
        _export(result, out, repro)

        # --- IBM QAOA cells (same qubo object) ---
        if include_ibm:
            result.cells.append(_qaoa_cell(base, qubo, anchor_obj, anchor_energy,
                                           "aer_sim", reps, shots, maxiter, seed))
            _export(result, out, repro)
        if include_ibm_real:
            result.cells.append(_qaoa_cell(base, qubo, anchor_obj, anchor_energy,
                                           "ibm_real", reps, shots, maxiter, seed,
                                           backend_name=ibm_backend_name))
            _export(result, out, repro)

        # --- D-Wave cells ---
        if include_dwave:
            result.cells.append(_dwave_cell(base, qubo, anchor_obj, anchor_energy,
                                            "neal_sim", num_reads, seed))
            _export(result, out, repro)
        if include_dwave_real:
            result.cells.append(_dwave_cell(base, qubo, anchor_obj, anchor_energy,
                                            "qpu_real", num_reads, seed,
                                            solver_name=dwave_solver_name))
            _export(result, out, repro)

    return result


def _error_cell(base, route, method, exc) -> ThreeWayCell:
    return ThreeWayCell(
        **base, route=route, method=method, backend_name="", is_quantum_hw=False,
        feasible=False, n_residual_conflicts=-1, acrp_objective=None, qubo_energy=None,
        approx_ratio=None, energy_ratio=None, error=f"{type(exc).__name__}: {exc}",
    )


def _qaoa_cell(base, qubo, anchor_obj, anchor_energy, mode, reps, shots, maxiter, seed,
               *, backend_name=None) -> ThreeWayCell:
    is_hw = mode == "ibm_real"
    try:
        from ..quantum.backends import BackendConfig, BackendMode
        from ..quantum.decode import best_feasible_bitstring, decode
        from ..quantum.qaoa import QAOASolver

        cfg = BackendConfig(mode=BackendMode(mode), shots=shots, seed=seed,
                            backend_name=backend_name)
        qres = QAOASolver(backend=cfg, reps=reps, maxiter=maxiter).solve(qubo)
        res = decode(qubo, qres.best_bitstring, wall_time_s=qres.wall_time_s,
                     seed=seed, backend=qres.backend_meta.get("name", cfg.label),
                     shots=shots, qaoa_reps=reps)
        _, prob = best_feasible_bitstring(qubo, qres.counts)
        energy = qubo.energy(qres.best_bitstring)
        return ThreeWayCell(
            **base, route="ibm_qaoa", method=mode,
            backend_name=qres.backend_meta.get("name", cfg.label), is_quantum_hw=is_hw,
            feasible=res.feasible, n_residual_conflicts=res.n_conflicts,
            acrp_objective=_clean(res.objective), qubo_energy=_clean(energy),
            approx_ratio=_ratio(res.objective, anchor_obj, res.feasible),
            energy_ratio=_ratio(energy, anchor_energy, True),
            best_feasible_prob=_clean(prob), wall_time_s=_clean(qres.wall_time_s) or 0.0,
            reps=reps, circuit_depth=qres.circuit_depth, two_qubit_gates=qres.two_qubit_gates,
            shots=shots, qpu_time_s=_clean(qres.qpu_time_s),
            anchor_objective=anchor_obj, anchor_qubo_energy=anchor_energy,
        )
    except Exception as exc:
        c = _error_cell(base, "ibm_qaoa", mode, exc)
        c.is_quantum_hw = is_hw
        return c


def _dwave_cell(base, qubo, anchor_obj, anchor_energy, mode, num_reads, seed,
                *, solver_name=None) -> ThreeWayCell:
    is_hw = mode == "qpu_real"
    try:
        from ..quantum.decode import best_feasible_bitstring, decode
        from ..quantum.dwave_solver import DWaveAnnealingSolver, DWaveConfig, DWaveMode

        cfg = DWaveConfig(mode=DWaveMode(mode), num_reads=num_reads, seed=seed,
                          solver_name=solver_name)
        raw = DWaveAnnealingSolver(cfg).solve_raw(qubo)
        res = decode(qubo, raw.best_bitstring, kind=cfg.kind, wall_time_s=raw.wall_time_s,
                     seed=seed, backend=raw.backend_name, shots=raw.num_reads)
        _, prob = best_feasible_bitstring(qubo, raw.counts)
        return ThreeWayCell(
            **base, route="dwave", method=("dwave_qpu" if is_hw else "dwave_sa"),
            backend_name=raw.backend_name, is_quantum_hw=is_hw,
            feasible=res.feasible, n_residual_conflicts=res.n_conflicts,
            acrp_objective=_clean(res.objective), qubo_energy=_clean(raw.best_energy),
            approx_ratio=_ratio(res.objective, anchor_obj, res.feasible),
            energy_ratio=_ratio(raw.best_energy, anchor_energy, True),
            best_feasible_prob=_clean(prob), wall_time_s=_clean(raw.wall_time_s) or 0.0,
            num_reads=raw.num_reads, qpu_time_s=_clean(raw.qpu_access_time_s),
            qpu_access_time_s=_clean(raw.qpu_access_time_s),
            chain_break_mean=_clean(raw.chain_break_mean),
            chain_break_max=_clean(raw.chain_break_max),
            n_physical_qubits=raw.n_physical_qubits,
            embedding_max_chain=raw.embedding_max_chain,
            anchor_objective=anchor_obj, anchor_qubo_energy=anchor_energy,
        )
    except Exception as exc:
        c = _error_cell(base, "dwave", ("dwave_qpu" if is_hw else "dwave_sa"), exc)
        c.is_quantum_hw = is_hw
        return c


def _ratio(value, anchor, feasible) -> float | None:
    if not feasible or anchor is None or anchor == 0:
        return None
    return value / anchor


_CSV_COLS = [
    "scenario_id", "instance_name", "n", "n_qubits", "grid_k", "route", "method",
    "backend_name", "is_quantum_hw", "feasible", "n_residual_conflicts",
    "acrp_objective", "qubo_energy", "approx_ratio", "energy_ratio",
    "best_feasible_prob", "wall_time_s", "reps", "circuit_depth", "two_qubit_gates",
    "shots", "qpu_time_s", "num_reads", "qpu_access_time_s", "chain_break_mean",
    "chain_break_max", "n_physical_qubits", "embedding_max_chain", "anchor_objective",
    "anchor_qubo_energy", "error",
] + _REPRO_COLUMNS


def _export(result: ThreeWayResult, out: Path, repro: ReproInfo) -> None:
    provenance = _repro_row(repro)
    rows = [{**asdict(c), **provenance} for c in result.cells]
    (out / "three_way.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out / "three_way.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in _CSV_COLS})


def format_table(result: ThreeWayResult) -> str:
    """Grouped side-by-side table: one row per route per scenario."""
    lines = [
        f"{'scenario':9s} {'route':15s} {'method':10s} {'hw':3s} {'feas':4s} "
        f"{'approx':>7s} {'energy_r':>8s} {'wall(s)':>8s} {'qpu(s)':>7s} {'chain_brk':>9s}",
        "-" * 92,
    ]
    for c in result.cells:
        ar = f"{c.approx_ratio:.3f}" if c.approx_ratio is not None else "  -  "
        er = f"{c.energy_ratio:.3f}" if c.energy_ratio is not None else "  -  "
        qp = f"{c.qpu_time_s:.2f}" if c.qpu_time_s is not None else "  -  "
        cb = f"{c.chain_break_mean:.3f}" if c.chain_break_mean is not None else "  -  "
        feas = "yes" if c.feasible else "NO"
        hw = "QPU" if c.is_quantum_hw else "sim"
        note = f"  ! {c.error[:36]}" if c.error else ""
        lines.append(
            f"{c.scenario_id:9s} {c.route:15s} {c.method:10s} {hw:3s} {feas:4s} "
            f"{ar:>7s} {er:>8s} {c.wall_time_s:>8.1f} {qp:>7s} {cb:>9s}{note}"
        )
    return "\n".join(lines)
