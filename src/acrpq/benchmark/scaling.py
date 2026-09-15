"""Qubit-scaling study: how far can we simulate, and at what cost.

Sweeps ACRP instance size (aircraft × maneuver options) to build QUBOs of
increasing qubit count ``n_qubits = n * K``, runs QAOA on the Aer simulator, and
records build time, simulation wall time, feasibility, predicted statevector
memory and measured peak RSS. This makes the ``2**N`` memory wall empirical: it
shows both where statevector refuses (guarded, never allocates) and how MPS
reaches further on low-entanglement circuits.

Peak RSS via stdlib ``resource.getrusage`` — bytes on macOS (darwin), KiB on
Linux (branch on ``sys.platform``).
"""

from __future__ import annotations

import csv
import json
import math
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..quantum.memguard import check_memory, max_safe_qubits, statevector_bytes
from .export import _REPRO_COLUMNS, ReproInfo, _repro_row

_GIB = 1024**3


def _peak_rss_gib() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / _GIB  # bytes on macOS
    return rss * 1024 / _GIB  # KiB on Linux


@dataclass(frozen=True)
class SizeSpec:
    instance_name: str
    n_theta: int
    n_q: int = 1
    subset: tuple[int, ...] | None = None

    @property
    def label(self) -> str:
        sub = f"[{len(self.subset)}]" if self.subset else ""
        return f"{self.instance_name}{sub}_t{self.n_theta}q{self.n_q}"


@dataclass
class ScalingCell:
    label: str
    instance_name: str
    n: int
    grid_k: int
    n_qubits: int
    n_conflict_terms: int
    n_onehot_terms: int
    sim_method: str
    reps: int
    predicted_sv_gib: float | None
    within_budget: bool
    build_time_s: float | None = None
    sim_time_s: float | None = None
    circuit_depth: int | None = None
    two_qubit_gates: int | None = None
    optimizer_evals: int | None = None
    feasible: bool | None = None
    n_residual_conflicts: int | None = None
    approx_ratio: float | None = None
    peak_rss_gib: float | None = None
    status: str = "ok"  # ok | ok(build) | refused_memory | error
    error: str = ""


@dataclass
class ScalingResult:
    cells: list[ScalingCell] = field(default_factory=list)


# default ladders: A raises aircraft at K=5; B raises K at n=4; MPS reach via subset
DEFAULT_SPECS: list[SizeSpec] = [
    SizeSpec("CP_3", 5),  # 15q
    SizeSpec("CP_4", 5),  # 20q
    SizeSpec("CP_5", 5),  # 25q
    SizeSpec("CP_6", 5),  # 30q -> statevector refused, MPS reaches
    SizeSpec("CP_4", 3),  # 12q
    SizeSpec("CP_4", 7),  # 28q (statevector ceiling)
]


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return None
    return v


def run_scaling_study(
    specs: list[SizeSpec] | None = None,
    *,
    methods: tuple[str, ...] = ("statevector", "matrix_product_state"),
    reps: int = 1,
    maxiter: int = 60,
    shots: int = 1024,
    seed: int = 42,
    run_qaoa: bool = True,
    out_dir: str | Path = "results",
) -> ScalingResult:
    """Sweep sizes × sim methods; refuse over-budget statevector without allocating.

    With ``run_qaoa=False`` (dry run) it only builds QUBOs and records structure
    + predicted memory — safe to run anywhere.
    """
    from ..discretize import DiscreteACRP
    from ..io.loader import InstanceLoader
    from ..model import ManeuverGrid
    from ..quantum.qubo import build_qubo

    specs = specs or DEFAULT_SPECS
    loader = InstanceLoader()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = ScalingResult()
    repro = ReproInfo.capture(seed=seed)

    for spec in specs:
        inst = loader.load(spec.instance_name)
        if spec.subset:
            inst = inst.subset(spec.subset)
        grid = ManeuverGrid.build(n_theta=spec.n_theta, n_q=spec.n_q, instance=inst)
        discrete = DiscreteACRP.build(inst, grid)
        n_qubits = discrete.n_vars
        sv_bytes = statevector_bytes(n_qubits)

        for method in methods:
            verdict = check_memory(n_qubits, method)
            cell = ScalingCell(
                label=spec.label,
                instance_name=inst.name,
                n=inst.n,
                grid_k=discrete.k,
                n_qubits=n_qubits,
                n_conflict_terms=0,
                n_onehot_terms=0,
                sim_method=method,
                reps=reps,
                predicted_sv_gib=(sv_bytes / _GIB) if method in ("statevector", "automatic") else None,
                within_budget=verdict.ok,
            )
            # build the QUBO (guard bypassed here; memguard is authoritative)
            t0 = time.perf_counter()
            try:
                qubo = build_qubo(inst, discrete=discrete, max_qubits=10_000)
            except Exception as exc:  # noqa: BLE001
                cell.status = "error"
                cell.error = f"build: {type(exc).__name__}: {exc}"
                result.cells.append(cell)
                _export(result, out, repro)
                continue
            cell.build_time_s = time.perf_counter() - t0
            cell.n_conflict_terms = qubo.n_conflict_terms
            cell.n_onehot_terms = qubo.n_onehot_terms

            # refuse over-budget statevector WITHOUT allocating a circuit
            if not run_qaoa:
                cell.status = "ok(build)"
                result.cells.append(cell)
                _export(result, out, repro)
                continue
            if method in ("statevector", "automatic") and not verdict.ok:
                cell.status = "refused_memory"
                cell.error = verdict.message
                result.cells.append(cell)
                _export(result, out, repro)
                continue

            _run_qaoa_leg(cell, inst, discrete, qubo, method, reps, maxiter, shots, seed)
            result.cells.append(cell)
            _export(result, out, repro)

    return result


def _run_qaoa_leg(cell, inst, discrete, qubo, method, reps, maxiter, shots, seed) -> None:
    try:
        from ..classical.reference import DiscreteReferenceSolver
        from ..quantum.backends import AerMethod, BackendConfig, BackendMode
        from ..quantum.decode import decode
        from ..quantum.memguard import safe_budget_bytes
        from ..quantum.qaoa import QAOASolver

        cfg = BackendConfig(
            mode=BackendMode.AER_SIM,
            sim_method=AerMethod(method),
            shots=shots,
            seed=seed,
            max_memory_mb=int(safe_budget_bytes() / (1024**2)),
        )
        t0 = time.perf_counter()
        qres = QAOASolver(backend=cfg, reps=reps, maxiter=maxiter).solve(qubo)
        cell.sim_time_s = time.perf_counter() - t0
        res = decode(qubo, qres.best_bitstring)
        ref = DiscreteReferenceSolver().solve(inst, discrete=discrete)
        cell.circuit_depth = qres.circuit_depth
        cell.two_qubit_gates = qres.two_qubit_gates
        cell.optimizer_evals = qres.eval_count
        cell.feasible = res.feasible
        cell.n_residual_conflicts = res.n_conflicts
        if res.feasible and ref.objective > 0:
            cell.approx_ratio = _clean(res.objective / ref.objective)
        cell.peak_rss_gib = round(_peak_rss_gib(), 3)
        cell.status = "ok"
    except Exception as exc:  # noqa: BLE001 - a too-big run fails here, recorded not raised
        cell.status = "error"
        cell.error = f"{type(exc).__name__}: {exc}"


_CSV_COLS = [
    "label", "instance_name", "n", "grid_k", "n_qubits", "n_conflict_terms",
    "n_onehot_terms", "sim_method", "reps", "predicted_sv_gib", "within_budget",
    "build_time_s", "sim_time_s", "circuit_depth", "two_qubit_gates",
    "optimizer_evals", "feasible", "n_residual_conflicts", "approx_ratio",
    "peak_rss_gib", "status", "error",
] + _REPRO_COLUMNS


def _export(result: ScalingResult, out: Path, repro: ReproInfo) -> None:
    provenance = _repro_row(repro)
    rows = [{**asdict(c), **provenance} for c in result.cells]
    (out / "scaling.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out / "scaling.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in _CSV_COLS})


def plot_scaling(result: ScalingResult, out_dir: str | Path = "results/figures"):
    """Semilog-y qubits vs sim time, per method, with the 24 GiB wall marked."""
    from .._optional import require

    require("matplotlib")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = sorted({c.sim_method for c in result.cells})
    for m in methods:
        pts = sorted(
            [(c.n_qubits, c.sim_time_s) for c in result.cells
             if c.sim_method == m and c.sim_time_s is not None],
        )
        if pts:
            xs, ys = zip(*pts)
            ax.semilogy(xs, ys, "o-", label=m)
    wall = max_safe_qubits()
    ax.axvline(wall + 0.5, color="#ff4d5e", ls="--", lw=1.5,
               label=f"statevector wall ({wall}q / 24 GiB)")
    ax.set_xlabel("number of qubits (n aircraft × K options)")
    ax.set_ylabel("QAOA simulation time (s, log)")
    ax.set_title("Simulation cost vs qubit count")
    ax.legend(fontsize=8)
    p = out / "scaling_qubits_vs_time.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    return str(p)


def format_table(result: ScalingResult) -> str:
    lines = [
        f"{'label':16s} {'qubits':>6s} {'method':20s} {'sv_GiB':>7s} {'budget':>6s} "
        f"{'sim(s)':>8s} {'feas':>4s} {'status':16s}",
        "-" * 92,
    ]
    for c in result.cells:
        svg = f"{c.predicted_sv_gib:.2f}" if c.predicted_sv_gib is not None else "  -  "
        st = f"{c.sim_time_s:.1f}" if c.sim_time_s is not None else "  -  "
        feas = "" if c.feasible is None else ("yes" if c.feasible else "NO")
        lines.append(
            f"{c.label:16s} {c.n_qubits:>6d} {c.sim_method:20s} {svg:>7s} "
            f"{'yes' if c.within_budget else 'NO':>6s} {st:>8s} {feas:>4s} {c.status:16s}"
        )
    return "\n".join(lines)
