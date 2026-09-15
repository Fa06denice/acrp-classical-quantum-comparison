"""``acrpq`` command-line interface.

Subcommands:

    acrpq list       [--family CP|FP|GP|RCP|RCP_FL]    list available instances
    acrpq show       <instance>                         parsed instance summary
    acrpq scenarios                                     list reduced benchmark scenarios
    acrpq classical  <instance|scenario> [--solver reference|couenne] [--n-theta N] [--n-q N]
    acrpq quantum    <scenario> [--mode aer_sim|ibm_runtime|ibm_real] [--reps R] [--shots N] [--seed S]
    acrpq bench      <scenario...|--all> [--mode ...] [--reps R] [--minlp] [--out DIR]
    acrpq campaign   [<scenario...>] [--reps p...] [--repetitions N] [--figures] [--out DIR]
    acrpq dashboard  [--host H] [--port P]            launch the radar web app

Pure subcommands (``list``, ``show``, ``scenarios``, ``classical --solver
reference``) run with zero third-party dependencies. ``quantum`` /
``bench --mode`` and ``classical --solver couenne`` load optional dependencies
lazily and print the exact ``pip install`` line if they are missing.
"""

from __future__ import annotations

import argparse
import sys

from ..exceptions import ACRPError, OptionalDependencyError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="acrpq", description=__doc__.split("\n")[0])
    p.add_argument("--data-root", default=None, help="path to the dir holding Code/ and Data/")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("list", help="list available instances")
    sp.add_argument("--family", default=None)

    sp = sub.add_parser("show", help="show a parsed instance summary")
    sp.add_argument("instance")

    sub.add_parser("scenarios", help="list reduced benchmark scenarios")

    sp = sub.add_parser("classical", help="run a classical / QUBO solver")
    sp.add_argument("target", help="instance name or scenario id")
    sp.add_argument(
        "--solver", default="reference",
        choices=["reference", "qubo-exact", "qubo-annealing", "dwave-sa",
                 "couenne", "bonmin", "ipopt", "original-ampl"],
        help="reference=exact grid; qubo-*=solve the QUBO energy classically; "
             "dwave-sa=D-Wave reference simulated annealing on the QUBO; "
             "couenne/bonmin/ipopt=Pyomo rebuild of the continuous MINLP (needs Pyomo+solver); "
             "original-ampl=direct run of the historical Code/NC_ACRP.mod via AMPL "
             "(solver from --ampl-solver, default gurobi)",
    )
    sp.add_argument("--ampl-solver", default="gurobi",
                    choices=["gurobi", "couenne", "cplex", "xpress", "highs"],
                    help="AMPL solver for --solver original-ampl (supervised default: gurobi; "
                         "couenne = optional open-source cross-check, never an auto fallback)")
    sp.add_argument("--n-theta", type=int, default=5)
    sp.add_argument("--n-q", type=int, default=1)
    sp.add_argument("--seed", type=int, default=1234)
    sp.add_argument("--max-qubits", type=int, default=29)
    sp.add_argument("--time-limit", type=float, default=None)

    sp = sub.add_parser("quantum", help="run QAOA (or D-Wave) on a scenario")
    sp.add_argument("scenario")
    sp.add_argument("--mode", default="aer_sim", choices=["aer_sim", "ibm_runtime", "ibm_real"])
    sp.add_argument("--sim-method", default="statevector",
                    choices=["statevector", "matrix_product_state", "automatic"],
                    help="Aer local-sim method (statevector=exact 2^N; MPS reaches more qubits)")
    sp.add_argument("--reps", type=int, default=1)
    sp.add_argument("--shots", type=int, default=1024)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--maxiter", type=int, default=100)
    sp.add_argument("--backend-name", default=None)
    sp.add_argument("--max-qubits", type=int, default=29)
    sp.add_argument("--dwave", action="store_true", help="use D-Wave instead of QAOA")
    sp.add_argument("--dwave-mode", default="auto", choices=["auto", "neal", "qpu"],
                    help="auto: real QPU if DWAVE_API_TOKEN set, else neal (SA)")
    sp.add_argument("--num-reads", type=int, default=1000, help="D-Wave num_reads")

    sp = sub.add_parser("bench", help="compare classical vs quantum and export results")
    sp.add_argument("scenarios", nargs="*", help="scenario ids (omit with --all)")
    sp.add_argument("--all", action="store_true", help="run all reduced scenarios")
    sp.add_argument("--mode", default="aer_sim", choices=["aer_sim", "ibm_runtime", "ibm_real"])
    sp.add_argument("--reps", type=int, default=1)
    sp.add_argument("--shots", type=int, default=1024)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--maxiter", type=int, default=100)
    sp.add_argument("--minlp", action="store_true",
                    help="deprecated alias for --pyomo-minlp")
    sp.add_argument("--pyomo-minlp", action="store_true",
                    help="also run the Pyomo (Python) rebuild of the continuous MINLP")
    sp.add_argument("--original-ampl", action="store_true",
                    help="also run the historical Code/NC_ACRP.mod via AMPL "
                         "(--ampl-solver, default gurobi; records 'unavailable' if AMPL absent)")
    sp.add_argument("--ampl-solver", default="gurobi",
                    choices=["gurobi", "couenne", "cplex", "xpress", "highs"],
                    help="AMPL solver for --original-ampl (supervised default: gurobi)")
    sp.add_argument("--no-quantum", action="store_true", help="reference only (no QAOA)")
    sp.add_argument("--out", default="results")
    sp.add_argument("--max-qubits", type=int, default=24)

    sp = sub.add_parser(
        "check-ampl-solver",
        help="verify AMPL + a solver + licence usability (cheap, no scientific solve)")
    sp.add_argument("solver", nargs="?", default="gurobi",
                    choices=["gurobi", "couenne", "cplex", "xpress", "highs"],
                    help="AMPL solver to verify (default: gurobi)")

    sp = sub.add_parser(
        "campaign",
        help="experimental sweep: scenarios x QAOA depth x repetitions (stats + figures)",
    )
    sp.add_argument("scenarios", nargs="*", help="scenario ids (default: all reduced)")
    sp.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3], help="QAOA depths to sweep")
    sp.add_argument("--repetitions", type=int, default=5, help="random seeds per cell")
    sp.add_argument("--maxiter", type=int, default=100)
    sp.add_argument("--shots", type=int, default=1024)
    sp.add_argument("--seed", type=int, default=1000)
    sp.add_argument("--max-qubits", type=int, default=24)
    sp.add_argument("--out", default="results")
    sp.add_argument("--figures", action="store_true", help="also render PNG figures (needs [viz])")

    sp = sub.add_parser("dashboard", help="launch the ATC-style radar dashboard (needs [dashboard])")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--results", default="results", help="dir holding campaign_summary.json")

    sp = sub.add_parser(
        "compare-hardware",
        help="run scenarios on Aer simulator AND a real IBM QPU, tabulate the noise gap",
    )
    sp.add_argument("scenarios", nargs="*", default=["Q2", "Q3"], help="scenario ids")
    sp.add_argument("--reps", type=int, default=1)
    sp.add_argument("--shots", type=int, default=1024)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--maxiter", type=int, default=100)
    sp.add_argument(
        "--hardware-mode", default="ibm_real", choices=["ibm_real", "ibm_runtime"],
        help="ibm_real = physical QPU; ibm_runtime = IBM cloud simulator",
    )
    sp.add_argument("--backend-name", default=None, help="explicit IBM backend (else least-busy)")
    sp.add_argument("--sim-only", action="store_true", help="skip the hardware leg")
    sp.add_argument("--out", default="results")

    sp = sub.add_parser(
        "compare-quantum",
        help="run the SAME QUBO through classical-exact vs IBM-QAOA vs D-Wave",
    )
    sp.add_argument("scenarios", nargs="*", default=["Q2", "Q3"], help="scenario ids")
    sp.add_argument("--ibm", action="store_true", help="add IBM QAOA on the Aer simulator")
    sp.add_argument("--ibm-real", action="store_true", help="add IBM QAOA on a real QPU")
    sp.add_argument("--dwave", dest="dwave", action="store_true", default=True,
                    help="add D-Wave simulated annealing (default on)")
    sp.add_argument("--no-dwave", dest="dwave", action="store_false")
    sp.add_argument("--dwave-real", action="store_true", help="add real D-Wave QPU")
    sp.add_argument("--reps", type=int, default=2)
    sp.add_argument("--shots", type=int, default=1024)
    sp.add_argument("--maxiter", type=int, default=100)
    sp.add_argument("--num-reads", type=int, default=1000)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--max-qubits", type=int, default=29)
    sp.add_argument("--ibm-backend-name", default=None)
    sp.add_argument("--dwave-solver-name", default=None)
    sp.add_argument("--out", default="results")

    sp = sub.add_parser(
        "scaling", help="qubit-scaling study: sim time & memory vs qubit count (+figure)"
    )
    sp.add_argument("--methods", nargs="+", default=["statevector", "matrix_product_state"],
                    choices=["statevector", "matrix_product_state", "automatic"])
    sp.add_argument("--reps", type=int, default=1)
    sp.add_argument("--maxiter", type=int, default=60)
    sp.add_argument("--shots", type=int, default=1024)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--dry-run", action="store_true", help="build QUBOs only, no QAOA")
    sp.add_argument("--figures", action="store_true", help="render the scaling figure (needs [viz])")
    sp.add_argument("--out", default="results")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except OptionalDependencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ACRPError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _loader(args):
    from ..io.loader import InstanceLoader

    return InstanceLoader(data_root=args.data_root)


def _dispatch(args) -> int:  # noqa: C901 - simple subcommand router
    if args.command == "list":
        return _cmd_list(args)
    if args.command == "show":
        return _cmd_show(args)
    if args.command == "scenarios":
        return _cmd_scenarios(args)
    if args.command == "classical":
        return _cmd_classical(args)
    if args.command == "quantum":
        return _cmd_quantum(args)
    if args.command == "bench":
        return _cmd_bench(args)
    if args.command == "check-ampl-solver":
        return _cmd_check_ampl_solver(args)
    if args.command == "campaign":
        return _cmd_campaign(args)
    if args.command == "dashboard":
        return _cmd_dashboard(args)
    if args.command == "compare-hardware":
        return _cmd_compare_hardware(args)
    if args.command == "compare-quantum":
        return _cmd_compare_quantum(args)
    if args.command == "scaling":
        return _cmd_scaling(args)
    return 1


def _cmd_list(args) -> int:
    names = _loader(args).list_names(args.family)
    print(f"{len(names)} instances" + (f" (family {args.family})" if args.family else ""))
    for name in names:
        print(" ", name)
    return 0


def _cmd_show(args) -> int:
    from .. import geometry

    inst = _loader(args).load(args.instance)
    ic = geometry.count_initial_conflicts(inst)
    print(f"instance:   {inst.name}")
    print(f"family:     {inst.family.value}")
    print(f"aircraft:   {inst.n}")
    print(f"pairs:      {inst.n_pairs()}")
    print(f"d:          {inst.d}")
    print(f"radius:     {inst.radius}")
    print(f"w (weight): {inst.w}")
    if inst.has_flight_levels:
        print(f"levels:     {inst.nf}")
    print(f"initial conflicts (no maneuver): {ic}")
    print(f"source:     {inst.source_path}")
    return 0


def _cmd_scenarios(args) -> int:
    from ..benchmark.scenarios import list_scenarios

    for sc in list_scenarios():
        print(f"{sc.id:5s} {sc.instance_name:10s} ~{sc.expected_qubits():>2d} qubits  {sc.description}")
    return 0


def _cmd_classical(args) -> int:
    inst = _resolve_target(args, args.target)
    if args.solver == "reference":
        from ..classical.reference import DiscreteReferenceSolver

        res = DiscreteReferenceSolver(n_theta=args.n_theta, n_q=args.n_q).solve(
            inst, time_limit_s=args.time_limit
        )
    elif args.solver in {"qubo-exact", "qubo-annealing", "dwave-sa"}:
        from ..quantum.qubo import build_qubo

        qubo = build_qubo(inst, n_theta=args.n_theta, n_q=args.n_q, max_qubits=args.max_qubits)
        if args.solver == "dwave-sa":
            from ..quantum.dwave_solver import DWaveAnnealingSolver, DWaveConfig, DWaveMode

            res = DWaveAnnealingSolver(
                DWaveConfig(mode=DWaveMode.NEAL_SIM, seed=args.seed)
            ).solve(qubo)
        else:
            from ..quantum.qubo_solvers import QuboAnnealingSolver, QuboExactSolver

            solver = (
                QuboExactSolver()
                if args.solver == "qubo-exact"
                else QuboAnnealingSolver(seed=args.seed)
            )
            res = solver.solve(qubo)
        _print_result(res)
        if "qubo_energy" in res.extra:
            print(f"QUBO energy: {res.extra['qubo_energy']:.4f} | qubits: {res.n_qubits}")
        return 0 if res.feasible else 3
    elif args.solver == "original-ampl":
        return _cmd_classical_original_ampl(args)
    else:
        from ..classical.pyomo_minlp import PyomoMINLPSolver

        res = PyomoMINLPSolver(solver=args.solver).solve(inst, time_limit_s=args.time_limit)
    _print_result(res)
    return 0 if res.feasible else 3


def _cmd_classical_original_ampl(args) -> int:
    """Run the historical NC_ACRP.mod directly (AMPL + --ampl-solver) on the .dat."""
    from ..benchmark.scenarios import REDUCED_SCENARIOS
    from ..classical.original_ampl import OriginalAMPLSolver

    loader = _loader(args)
    if args.target in REDUCED_SCENARIOS:
        sc = REDUCED_SCENARIOS[args.target]
        if sc.subset:
            print(
                "error: scenario is a subset; the original model runs the "
                "whole-instance .dat verbatim (never a reconstructed subset). "
                f"Use the whole instance '{sc.instance_name}' instead.",
                file=sys.stderr,
            )
            return 2
        inst_name, w = sc.instance_name, sc.w
    else:
        inst_name, w = args.target, None

    inst = loader.load(inst_name) if w is None else loader.load(inst_name, w=w)
    dat_text, source = loader.read_dat_text(inst_name)
    print(f"original .dat source: {source}")
    print(f"ampl solver: {args.ampl_solver}")
    solver = OriginalAMPLSolver(solver=args.ampl_solver, timeout_s=args.time_limit or 300.0)
    res = solver.solve(inst, dat_text=dat_text)
    _print_result(res)
    if "provider_objective" in res.extra:
        e = res.extra
        print(
            f"provider objective: {e['provider_objective']:.6f} | "
            f"rescored: {e['rescored_objective']:.6f} | "
            f"|Δ|={e.get('objective_abs_delta', float('nan')):.2e} | "
            f"integrity_ok={bool(e.get('integrity_ok', 0.0))} | "
            f"optimality_proven={bool(e.get('optimality_proven', 0.0))}"
        )
        if "mip_rel_gap" in e:
            print(f"mip gap: abs={e.get('mip_abs_gap', float('nan')):.2e} "
                  f"rel={e['mip_rel_gap']:.2e} | solver_time="
                  f"{e.get('solver_time_s', float('nan')):.3f}s")
    if res.status.value == "unavailable":
        print(f"note: AMPL unavailable — recorded as unavailable, not executed "
              f"(solver requested: {args.ampl_solver}).")
        return 4
    return 0 if res.feasible else 3


def _cmd_check_ampl_solver(args) -> int:
    """Verify AMPL + the chosen solver + licence usability (cheap 1-var QP)."""
    import json

    from ..classical.original_ampl import OriginalAMPLSolver

    report = OriginalAMPLSolver(solver=args.solver).check_solver()
    print(json.dumps(report, indent=2, default=str))
    print("\nNOTE: this checks the toolchain + licence usability only; it does NOT "
          "validate the ACRP scientific model.")
    return 0 if report.get("ok") else 5


def _cmd_quantum(args) -> int:
    from ..benchmark.runner import BenchmarkRunner
    from ..benchmark.scenarios import get_scenario

    runner = BenchmarkRunner(loader=_loader(args), seed=args.seed)

    if args.dwave:
        return _cmd_quantum_dwave(args, runner)

    from ..quantum.backends import AerMethod, BackendConfig, BackendMode

    cfg = BackendConfig(
        mode=BackendMode(args.mode), shots=args.shots, seed=args.seed,
        backend_name=args.backend_name, sim_method=AerMethod(args.sim_method),
    )
    ref_res, _ = runner.run_classical_reference(get_scenario(args.scenario))
    res, rec = runner.run_quantum(
        args.scenario, backend=cfg, reps=args.reps, maxiter=args.maxiter,
        max_qubits=args.max_qubits, reference_obj=ref_res.objective,
    )
    _print_result(res)
    if rec.quantum:
        q = rec.quantum
        print(
            f"qubits={q.n_qubits} depth={q.circuit_depth} 2q-gates={q.two_qubit_gates} "
            f"backend={q.backend_name} shots={q.shots} approx_ratio={q.approx_ratio}"
        )
    return 0 if res.feasible else 3


def _cmd_quantum_dwave(args, runner) -> int:
    import os

    from ..io.env import load_dotenv
    from ..quantum.dwave_solver import DWaveConfig, DWaveMode

    # auto: real QPU when a Leap token is present, else the token-free SA sampler
    if args.dwave_mode == "qpu":
        mode = DWaveMode.QPU_REAL
    elif args.dwave_mode == "neal":
        mode = DWaveMode.NEAL_SIM
    else:  # auto
        load_dotenv()
        mode = DWaveMode.QPU_REAL if os.environ.get("DWAVE_API_TOKEN") else DWaveMode.NEAL_SIM
    print(f"D-Wave mode: {mode.value}")
    cfg = DWaveConfig(mode=mode, num_reads=args.num_reads, seed=args.seed)
    res, _ = runner.run_dwave(args.scenario, config=cfg, max_qubits=args.max_qubits)
    _print_result(res)
    e = res.extra
    extra = f"energy={e.get('qubo_energy', float('nan')):.4f} num_reads={int(e.get('num_reads', 0))}"
    if "chain_break_mean" in e:
        extra += f" chain_break_mean={e['chain_break_mean']:.3f}"
    if "qpu_access_time_s" in e:
        extra += f" qpu_time={e['qpu_access_time_s']:.4f}s"
    print(extra)
    return 0 if res.feasible else 3


def _cmd_bench(args) -> int:
    from ..benchmark.runner import BenchmarkRunner
    from ..quantum.backends import BackendConfig, BackendMode

    runner = BenchmarkRunner(loader=_loader(args), out_dir=args.out, seed=args.seed)
    cfg = BackendConfig(mode=BackendMode(args.mode), shots=args.shots, seed=args.seed)
    common = dict(
        backend=cfg, reps=args.reps, maxiter=args.maxiter,
        with_quantum=not args.no_quantum,
        with_pyomo_minlp=args.pyomo_minlp or args.minlp,
        with_original_ampl=args.original_ampl,
        ampl_solver=args.ampl_solver,
        max_qubits=args.max_qubits,
    )
    if args.all:
        records = runner.compare_all(**common)
        print(f"wrote {args.out}/benchmark_all.csv/.json + baseline_all.csv/.json ({len(records)} runs)")
    elif args.scenarios:
        records = []
        for sc in args.scenarios:
            records.extend(runner.compare(sc, **common))
        print(f"wrote per-scenario benchmark_*/baseline_* CSV/JSON to {args.out}/ ({len(records)} runs)")
    else:
        print("error: provide scenario ids or --all", file=sys.stderr)
        return 1
    return 0


def _cmd_campaign(args) -> int:
    from ..benchmark.campaign import run_campaign
    from ..benchmark.runner import BenchmarkRunner
    from ..benchmark.scenarios import list_scenarios

    scenarios = args.scenarios or [sc.id for sc in list_scenarios()]
    runner = BenchmarkRunner(loader=_loader(args), out_dir=args.out, seed=args.seed)
    print(f"campaign: {len(scenarios)} scenarios x reps={args.reps} x {args.repetitions} reps each")
    result = run_campaign(
        scenarios,
        reps_values=tuple(args.reps),
        repetitions=args.repetitions,
        maxiter=args.maxiter,
        shots=args.shots,
        base_seed=args.seed,
        max_qubits=args.max_qubits,
        out_dir=args.out,
        runner=runner,
    )
    print(f"  {len(result.runs)} individual runs, {len(result.points)} summary cells")
    if result.skipped:
        print(f"  {len(result.skipped)} cells skipped (over qubit budget):")
        for s in result.skipped:
            print(f"    {s['scenario_id']} reps={s['reps']}")
    print(f"  wrote {args.out}/campaign_summary.csv, campaign_runs.csv (+ .json)")
    # compact summary table
    print("\n  scenario  n  qubits  reps  feasible%  mean_approx  mean_depth")
    for p in result.points:
        ar = f"{p.mean_approx_ratio:.3f}" if p.mean_approx_ratio is not None else "  -  "
        print(
            f"  {p.scenario_id:8s} {p.n:2d} {p.n_qubits:6d}  {p.reps:4d}  "
            f"{p.feasible_rate*100:7.0f}%  {ar:>10s}  {p.mean_circuit_depth:9.0f}"
        )
    if args.figures:
        try:
            from ..viz.plots import save_campaign_figures

            paths = save_campaign_figures(result, out_dir=f"{args.out}/figures")
            print(f"  figures: {paths}")
        except Exception as exc:  # viz is optional
            print(f"  (figures skipped: {exc})")
    return 0


def _cmd_compare_hardware(args) -> int:
    from ..benchmark.runner import BenchmarkRunner
    from ..benchmark.sim_vs_hardware import format_table, run_sim_vs_hardware

    scenarios = args.scenarios or ["Q2", "Q3"]
    runner = BenchmarkRunner(loader=_loader(args), out_dir=args.out, seed=args.seed)
    print(
        f"sim-vs-hardware: {scenarios} | sim=aer + "
        f"{'(hardware skipped)' if args.sim_only else args.hardware_mode}"
    )
    if not args.sim_only:
        print("  NOTE: real-hardware jobs may queue for minutes to over an hour.")
    result = run_sim_vs_hardware(
        scenarios,
        reps=args.reps,
        shots=args.shots,
        seed=args.seed,
        maxiter=args.maxiter,
        include_hardware=not args.sim_only,
        hardware_mode=args.hardware_mode,
        backend_name=args.backend_name,
        out_dir=args.out,
        runner=runner,
    )
    print("\n" + format_table(result))
    print(f"\nwrote {args.out}/sim_vs_hardware.csv (+ .json)")
    return 0


def _cmd_compare_quantum(args) -> int:
    from ..benchmark.runner import BenchmarkRunner
    from ..benchmark.three_way import format_table, run_three_way

    scenarios = args.scenarios or ["Q2", "Q3"]
    runner = BenchmarkRunner(loader=_loader(args), out_dir=args.out, seed=args.seed)
    routes = ["classical-exact"]
    if args.ibm:
        routes.append("ibm-qaoa(sim)")
    if args.ibm_real:
        routes.append("ibm-qaoa(QPU)")
    if args.dwave:
        routes.append("dwave-SA")
    if args.dwave_real:
        routes.append("dwave-QPU")
    print(f"compare-quantum: {scenarios} | routes: {', '.join(routes)}")
    if args.ibm_real or args.dwave_real:
        print("  NOTE: real-hardware jobs may queue.")
    result = run_three_way(
        scenarios,
        include_ibm=args.ibm,
        include_ibm_real=args.ibm_real,
        include_dwave=args.dwave,
        include_dwave_real=args.dwave_real,
        reps=args.reps,
        shots=args.shots,
        maxiter=args.maxiter,
        num_reads=args.num_reads,
        seed=args.seed,
        max_qubits=args.max_qubits,
        ibm_backend_name=args.ibm_backend_name,
        dwave_solver_name=args.dwave_solver_name,
        out_dir=args.out,
        runner=runner,
    )
    print("\n" + format_table(result))
    print(f"\nwrote {args.out}/three_way.csv (+ .json)")
    return 0


def _cmd_scaling(args) -> int:
    from ..benchmark.scaling import format_table, plot_scaling, run_scaling_study

    print(f"scaling study: methods={args.methods} dry_run={args.dry_run}")
    result = run_scaling_study(
        methods=tuple(args.methods),
        reps=args.reps,
        maxiter=args.maxiter,
        shots=args.shots,
        seed=args.seed,
        run_qaoa=not args.dry_run,
        out_dir=args.out,
    )
    print("\n" + format_table(result))
    print(f"\nwrote {args.out}/scaling.csv (+ .json)")
    if args.figures:
        try:
            print(f"figure: {plot_scaling(result, out_dir=f'{args.out}/figures')}")
        except Exception as exc:  # viz optional
            print(f"(figure skipped: {exc})")
    return 0


def _cmd_dashboard(args) -> int:
    from ..dashboard.api import run

    print(f"ACRP radar dashboard -> http://{args.host}:{args.port}  (Ctrl-C to stop)")
    run(host=args.host, port=args.port, data_root=args.data_root, results_dir=args.results)
    return 0


def _resolve_target(args, target: str):
    """Resolve a CLI target to an Instance (scenario id or instance name)."""
    from ..benchmark.scenarios import REDUCED_SCENARIOS
    from ..model import ManeuverGrid  # noqa: F401 - ensure model importable

    loader = _loader(args)
    if target in REDUCED_SCENARIOS:
        sc = REDUCED_SCENARIOS[target]
        inst = loader.load(sc.instance_name, w=sc.w)
        return inst.subset(sc.subset) if sc.subset else inst
    return loader.load(target)


def _print_result(res) -> None:
    print(f"solver:     {res.solver.value}")
    print(f"status:     {res.status.value}")
    print(f"feasible:   {res.feasible}")
    print(f"objective:  {res.objective:.6f}")
    print(f"conflicts:  {res.n_conflicts} residual / {res.n_initial_conflicts} initial")
    print(f"time:       {res.wall_time_s * 1000:.1f} ms")
    if res.message:
        print(f"note:       {res.message}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
