"""Benchmark runner: classical vs quantum on the same scenarios.

The :class:`DiscreteReferenceSolver` always runs and is the ground-truth anchor.
QAOA runs when the ``[quantum]`` extra is installed; the optional Pyomo MINLP
runs when requested and a solver is available. Each run becomes a
:class:`~acrpq.benchmark.export.RunRecord`; ``compare`` writes CSV + JSON.

``approx_ratio`` for a quantum run is ``quantum.objective / reference.objective``
when both are feasible — i.e. how close QAOA gets to the *in-grid* optimum, not
the continuous objective (which the MINLP / original-AMPL legs report separately;
see :mod:`acrpq.benchmark.baseline` for the proven-optimum-anchored deltas).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .. import geometry
from ..discretize import DiscreteACRP
from ..io.loader import InstanceLoader
from ..model import Instance, ManeuverGrid, Result, SolverKind, Status
from .baseline import (
    BASELINE_LEG_ORDER,
    compute_baseline,
    write_baseline_csv,
    write_baseline_json,
)
from .export import ReproInfo, RunRecord, write_csv, write_json
from .metrics import QuantumMetrics, common_metrics, quantum_metrics
from .scenarios import Scenario, get_scenario


@dataclass
class BenchmarkRunner:
    """Run and compare solvers on scenarios, exporting results."""

    loader: InstanceLoader = field(default_factory=InstanceLoader)
    out_dir: str | Path = "results"
    seed: int = 42

    # ------------------------------------------------------------------ #
    def _resolve(self, scenario: Scenario | str) -> tuple[Scenario, Instance, DiscreteACRP]:
        sc = get_scenario(scenario) if isinstance(scenario, str) else scenario
        inst = self.loader.load(sc.instance_name, w=sc.w)
        if sc.subset:
            inst = inst.subset(sc.subset)
        grid = ManeuverGrid.build(instance=inst, **sc.grid_kwargs)
        discrete = DiscreteACRP.build(inst, grid)
        return sc, inst, discrete

    def _record(
        self,
        sc: Scenario,
        inst: Instance,
        discrete: DiscreteACRP,
        result: Result,
        *,
        quantum: QuantumMetrics | None = None,
        backend: str | None = None,
        shots: int | None = None,
    ) -> RunRecord:
        repro_seed = result.seed if result.seed is not None else self.seed
        repro = ReproInfo.capture(seed=repro_seed, backend=backend, shots=shots)
        return RunRecord(
            scenario_id=sc.id,
            instance_name=inst.name,
            family=inst.family.value,
            n=inst.n,
            n_pairs=inst.n_pairs(),
            grid_k=discrete.k,
            solver_kind=result.solver.value,
            status=result.status.value,
            common=common_metrics(result, inst),
            repro=repro,
            quantum=quantum,
            extra=dict(result.extra),
        )

    # ------------------------------------------------------------------ #
    def run_classical_reference(self, scenario: Scenario | str) -> tuple[Result, RunRecord]:
        from ..classical.reference import DiscreteReferenceSolver

        sc, inst, discrete = self._resolve(scenario)
        result = DiscreteReferenceSolver().solve(inst, discrete=discrete)
        return result, self._record(sc, inst, discrete, result, backend="discrete")

    def run_pyomo_minlp(
        self, scenario: Scenario | str, *, solver: str = "couenne"
    ) -> tuple[Result, RunRecord]:
        """Run the Python (Pyomo) rebuild of the continuous NC_ACRP model.

        This is NOT the historical AMPL model — see :meth:`run_original_ampl`.
        """
        from ..classical.pyomo_minlp import PyomoMINLPSolver

        sc, inst, discrete = self._resolve(scenario)
        result = PyomoMINLPSolver(solver=solver).solve(inst)
        return result, self._record(sc, inst, discrete, result, backend=f"pyomo:{solver}")

    def run_minlp(self, scenario: Scenario | str, *, solver: str = "couenne") -> tuple[Result, RunRecord]:
        """Deprecated alias for :meth:`run_pyomo_minlp` (kept for callers/tests)."""
        return self.run_pyomo_minlp(scenario, solver=solver)

    def run_original_ampl(
        self, scenario: Scenario | str, *, solver: str = "gurobi", timeout_s: float = 300.0
    ) -> tuple[Result, RunRecord]:
        """Run the historical Code/NC_ACRP.mod directly via AMPL + an authorised solver.

        The supervised protocol solver is Gurobi (the default); Couenne is an
        optional open-source cross-check, never an automatic fallback. The original
        instance ``.dat`` is loaded verbatim (preserving ``PI := 3.141592``) and fed
        to :class:`OriginalAMPLSolver`. When AMPL or the requested solver is absent
        the result carries ``Status.UNAVAILABLE`` / ``solver_unavailable`` — the
        benchmark still records a leg, it is simply never scored as executed.
        """
        import json

        from ..classical.original_ampl import OriginalAMPLSolver

        sc, inst, discrete = self._resolve(scenario)
        if sc.subset:
            # The historical model runs the verbatim whole-instance .dat. A subset
            # would require synthesising a reduced .dat (reconstruction) — which we
            # never do — so the original continuous baseline is not representable
            # here. Record an honest, non-executed leg rather than a fake number.
            result = Result(
                instance_name=inst.name,
                solver=SolverKind.CLASSICAL_ORIGINAL_AMPL,
                status=Status.UNAVAILABLE,
                solution=None,
                n_initial_conflicts=geometry.count_initial_conflicts(inst),
                message=json.dumps(
                    {
                        "status_kind": "not_representable",
                        "reason": (
                            "subset scenario: the original NC_ACRP model runs the "
                            "verbatim whole-instance .dat; a subset is never reconstructed"
                        ),
                        "instance": sc.instance_name,
                        "subset": list(sc.subset),
                    }
                ),
            )
            return result, self._record(sc, inst, discrete, result, backend=f"ampl:{solver}")

        dat_text, _ = self.loader.read_dat_text(sc.instance_name)
        result = OriginalAMPLSolver(solver=solver, timeout_s=timeout_s).solve(
            inst, dat_text=dat_text
        )
        return result, self._record(sc, inst, discrete, result, backend=f"ampl:{solver}")

    def run_dwave(
        self, scenario: Scenario | str, *, config=None, max_qubits: int = 29
    ) -> tuple[Result, RunRecord]:
        """Solve the scenario's QUBO on D-Wave (reference SA or real QPU)."""
        from ..quantum.dwave_solver import DWaveAnnealingSolver, DWaveConfig
        from ..quantum.qubo import build_qubo

        sc, inst, discrete = self._resolve(scenario)
        qubo = build_qubo(inst, discrete=discrete, max_qubits=max_qubits)
        cfg = config or DWaveConfig(seed=self.seed)
        result = DWaveAnnealingSolver(cfg).solve(qubo)
        return result, self._record(sc, inst, discrete, result, backend=result.backend)

    def run_quantum(
        self,
        scenario: Scenario | str,
        *,
        backend=None,
        reps: int = 1,
        maxiter: int = 100,
        max_qubits: int = 29,
        reference_obj: float | None = None,
    ) -> tuple[Result, RunRecord]:
        # Lazy imports keep the quantum stack optional.
        from ..quantum.backends import BackendConfig
        from ..quantum.decode import best_feasible_bitstring, decode
        from ..quantum.qaoa import QAOASolver
        from ..quantum.qubo import build_qubo

        sc, inst, discrete = self._resolve(scenario)
        cfg = backend or BackendConfig(seed=self.seed)
        qubo = build_qubo(inst, discrete=discrete, max_qubits=max_qubits)
        solver = QAOASolver(backend=cfg, reps=reps, maxiter=maxiter)
        qres = solver.solve(qubo)

        decode_extra: dict[str, float] = {"n_jobs": float(qres.n_jobs)}
        if qres.qpu_time_s is not None:
            decode_extra["qpu_time_s"] = qres.qpu_time_s
        result = decode(
            qubo,
            qres.best_bitstring,
            kind=SolverKind.QUANTUM_QAOA,
            wall_time_s=qres.wall_time_s,
            seed=cfg.seed,
            backend=qres.backend_meta.get("name", cfg.label),
            shots=cfg.shots,
            qaoa_reps=reps,
            extra=decode_extra,
        )
        _, best_prob = best_feasible_bitstring(qubo, qres.counts)

        qm = quantum_metrics(
            inst=inst,
            discrete=discrete,
            qubo=qubo,
            qaoa_result=qres,
            backend_config=cfg,
            decoded=result,
            optimizer=solver.optimizer,
            reps=reps,
            reference_obj=reference_obj,
            best_prob=best_prob,
        )
        record = self._record(
            sc, inst, discrete, result, quantum=qm,
            backend=qm.backend_name, shots=cfg.shots,
        )
        return result, record

    # ------------------------------------------------------------------ #
    def compare(
        self,
        scenario: Scenario | str,
        *,
        backend=None,
        reps: int = 1,
        maxiter: int = 100,
        with_quantum: bool = True,
        with_original_ampl: bool = False,
        with_pyomo_minlp: bool = False,
        with_minlp: bool | None = None,
        ampl_solver: str = "gurobi",
        max_qubits: int = 29,
        write: bool = True,
    ) -> list[RunRecord]:
        """Run reference (always), optionally the two continuous baselines and QAOA.

        ``with_minlp`` is a deprecated alias for ``with_pyomo_minlp``. Records are
        emitted in canonical scientific order: original_ampl, pyomo_minlp,
        discrete_reference, then the approximate legs. A requested continuous leg
        that fails or is unavailable still yields an explicit leg (never dropped),
        so the reader always sees which baselines were attempted.
        """
        sc = get_scenario(scenario) if isinstance(scenario, str) else scenario
        if with_minlp is not None:  # back-compat alias
            with_pyomo_minlp = with_pyomo_minlp or with_minlp
        records: list[RunRecord] = []

        if with_original_ampl:
            _, ampl_record = self.run_original_ampl(sc, solver=ampl_solver)
            records.append(ampl_record)

        ref_result, ref_record = self.run_classical_reference(sc)

        if with_pyomo_minlp:
            try:
                _, pyomo_record = self.run_pyomo_minlp(sc)
                records.append(pyomo_record)
            except Exception as exc:  # optional path: note on the reference leg
                ref_record.extra["pyomo_minlp_error"] = 1.0
                _ = exc

        records.append(ref_record)

        if with_quantum:
            _, q_record = self.run_quantum(
                sc, backend=backend, reps=reps, maxiter=maxiter,
                max_qubits=max_qubits, reference_obj=ref_result.objective,
            )
            records.append(q_record)

        records = self._order_records(records)
        if write:
            out = Path(self.out_dir)
            write_csv(records, out / f"benchmark_{sc.id}.csv")
            write_json(records, out / f"benchmark_{sc.id}.json")
            baseline = compute_baseline(records)
            if baseline is not None:
                write_baseline_csv([baseline], out / f"baseline_{sc.id}.csv")
                write_baseline_json([baseline], out / f"baseline_{sc.id}.json")
        return records

    @staticmethod
    def _order_records(records: list[RunRecord]) -> list[RunRecord]:
        """Sort records into canonical scientific leg order (stable within a kind)."""
        rank = {k: i for i, k in enumerate(BASELINE_LEG_ORDER)}
        return sorted(
            records, key=lambda r: rank.get(r.solver_kind, len(BASELINE_LEG_ORDER))
        )

    def compare_all(self, scenario_ids: list[str] | None = None, **kwargs) -> list[RunRecord]:
        """Run :meth:`compare` over many scenarios into one CSV/JSON pair."""
        from .scenarios import list_scenarios

        scs = (
            [get_scenario(s) for s in scenario_ids]
            if scenario_ids
            else list_scenarios()
        )
        all_records: list[RunRecord] = []
        baselines = []
        for sc in scs:
            recs = self.compare(sc, write=False, **kwargs)
            all_records.extend(recs)
            baseline = compute_baseline(recs)
            if baseline is not None:
                baselines.append(baseline)
        out = Path(self.out_dir)
        write_csv(all_records, out / "benchmark_all.csv")
        write_json(all_records, out / "benchmark_all.json")
        if baselines:
            write_baseline_csv(baselines, out / "baseline_all.csv")
            write_baseline_json(baselines, out / "baseline_all.json")
        return all_records


# convenience: confirm geometry is the shared scorer (used in tests/docs)
_SHARED_SCORER = geometry.objective
