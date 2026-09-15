"""FastAPI backend for the ACRP radar dashboard.

Every endpoint is a thin JSON adapter over the existing ``acrpq`` package — no
ACRP logic lives here. The same geometry kernel scores classical and quantum
results, so the radar shows comparable numbers for both.

Endpoints
---------
GET  /api/health                      -> capabilities (which extras are present)
GET  /api/instances?family=CP         -> list of instance names
GET  /api/instances/catalog           -> instances grouped by local applicability
GET  /api/scenarios                   -> reduced benchmark scenarios
GET  /api/instance/{name}             -> aircraft geometry + initial conflicts
POST /api/solve                       -> solve with reference or QAOA, return radar payload
GET  /api/campaign                    -> aggregated campaign summary (if present)
GET  /                                -> the radar front-end (static)
"""

from __future__ import annotations

import json
import math
import re
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .._optional import have, require
from ..io.loader import InstanceLoader
from ..selection_policy import selection_policy_doc
from ..model import Instance

_STATIC = Path(__file__).resolve().parent / "static"
_ALLOWED_SOLVERS = {
    "reference",
    "qubo-exact",
    "qubo-annealing",
    "dwave",
    "dwave-qpu",
    "qaoa",
}
_ALLOWED_MODES = {"aer_sim", "ibm_runtime", "ibm_real"}
# Host/Origin values accepted for non-QPU state-changing POSTs (same loopback
# allowlist as the QPU routes — DNS-rebinding CSRF defense; no proxy trust).
_ALLOWED_LOCAL_HOSTS = frozenset(
    {"127.0.0.1", "localhost", "testclient", "testserver", "[::1]", "::1"})
_MAX_LOCAL_BODY_BYTES = 262_144  # 256 KiB cap for local mutation bodies

# Absolute server filesystem paths (>= 2 segments) must never reach a client:
# exception text (e.g. the loader's data_root) is sanitised at the API boundary.
_ABS_PATH_RE = re.compile(r"/[\w.\-]+(?:/[\w.\-]+)+")


def _sanitize_error_text(msg: str) -> str:
    """Strip absolute server filesystem paths from client-visible error text."""
    return _ABS_PATH_RE.sub("<server-path>", msg)
_MAX_GRID_OPTIONS = 25
_MAX_DISCRETE_VARS = 1_024
_MAX_QUBO_VARS = 1_024

DEFAULT_OBJECTIVE_ID = "quadratic_control_cost_v1"
_OBJECTIVE_ID_FR = {
    "quadratic_control_cost_v1": "objectif quadratique historique",
    "maneuver_count_v1": "minimum d'avions déviés",
}

# Two explicit execution profiles. DEMO is the fast, single-seed interactive
# run used for exploration and the live comparison — it is deliberately *not*
# citable. SCIENTIFIC is a validated, seed-repeated measurement carrying full
# provenance and is the only profile eligible to be exported as a scientific run.
_PROFILES = {
    "demo": {
        "label": "DEMO — quick interactive run",
        "citable": False,
        "note": (
            "Bounded single-seed interactive run for exploration and teaching; "
            "wall times are host-dependent and this is not a citable measurement."
        ),
    },
    "scientific": {
        "label": "SCIENTIFIC — reproducible measurement",
        "citable": True,
        "note": (
            "Validated parameters repeated across seeds with full provenance; "
            "eligible to be saved/exported as a scientific run."
        ),
    },
}
# Bound on how many seeds a single scientific run may sweep (machine-overload guard).
_MAX_SCIENTIFIC_SEEDS = 12


def _proof_status(solver: str, result: Any, extra: dict[str, Any]) -> dict[str, Any]:
    """Classify how strong a result's guarantee is, and whether a gap term applies.

    ``gap_allowed`` is True only for a proven optimum: it is the guard the UI
    reads to FORBID the words "optimality gap" whenever the anchor is merely a
    feasible heuristic or an approximate sample.
    """
    if not result.feasible:
        return {
            "status": "infeasible",
            "label": "infeasible — residual conflicts remain",
            "gap_allowed": False,
            "caveat": "The decoded assignment still has conflicts; not a valid resolution.",
        }
    if solver == "reference":
        if extra.get("exact_method"):
            return {
                "status": "exact_optimum",
                "label": "exact in-grid optimum",
                "gap_allowed": True,
                "caveat": "Global optimum over the enumerated discrete maneuver grid.",
            }
        return {
            "status": "feasible_heuristic",
            "label": "feasible heuristic — grid too large to enumerate exactly",
            "gap_allowed": False,
            "caveat": (
                "Greedy/local-search fallback: feasible but NOT proven optimal, so "
                "'optimality gap' does not apply to this anchor."
            ),
        }
    if solver == "qubo-exact":
        domain = extra.get("exact_domain", "one_hot_subspace")
        full = domain == "all_bitstrings"
        return {
            "status": "exact_optimum",
            "label": (
                "exact unconstrained QUBO minimum"
                if full else "exact one-hot QUBO minimum"
            ),
            "gap_allowed": True,
            "caveat": (
                "Global minimum of H(x) over every bitstring."
                if full
                else "Exact minimum of H(x) over the one-hot subspace; not a claim "
                "about every unconstrained bitstring."
            ),
        }
    if solver in {"qubo-annealing", "dwave", "dwave-qpu"}:
        return {
            "status": "feasible_heuristic",
            "label": "feasible — stochastic annealing sample",
            "gap_allowed": False,
            "caveat": "Sampled heuristic: feasible but not proven optimal.",
        }
    if solver == "qaoa":
        return {
            "status": "approximate_sampled",
            "label": "feasible — approximate QAOA sample",
            "gap_allowed": False,
            "caveat": "Variational approximate optimizer: feasible but not proven optimal.",
        }
    return {"status": "not_applicable", "label": "n/a", "gap_allowed": False, "caveat": ""}


def _profile_meta(profile: str, params: dict[str, Any]) -> dict[str, Any]:
    """Resolve a profile name to its label/citability plus the params actually used."""
    spec = _PROFILES.get(profile)
    if spec is None:
        raise ValueError(
            f"unknown profile {profile!r}; expected one of {sorted(_PROFILES)}"
        )
    return {
        "name": profile,
        "label": spec["label"],
        "citable": spec["citable"],
        "note": spec["note"],
        "params": params,
    }


# --------------------------------------------------------------------------- #
# Payload builders (pure, testable without a server)
# --------------------------------------------------------------------------- #
def _resolve_instance(loader: InstanceLoader, name: str, subset=None, w=None) -> Instance:
    inst = loader.load(name, w=w)
    if subset:
        inst = inst.subset(subset)
    return inst


# Deriving the model from a configuration is deterministic, so bounded caches let
# the five auto-comparison routes (and the QUBO tab) share a single build instead
# of re-deriving per request. Measured medians (host-dependent): DiscreteACRP.build
# is the dominant stage (~0.1 ms at 12 qubits to ~29 ms at 140 qubits, ~O(n²K²)),
# an order of magnitude above build_qubo — so the discretisation, not just the
# QUBO, must be cached. Keys are the *complete* configuration (instance, ordered
# subset, grid, weight), so nothing is reused across different configurations; a
# lock makes the caches safe under the concurrent job-pool threads.
_MODEL_CACHE: dict[tuple, dict[str, Any]] = {}  # key -> {"ig": (inst, grid), "discrete": ...}
_QUBO_CACHE: dict[tuple, Any] = {}
_CACHE_MAX = 16
_CACHE_LOCK = threading.Lock()


def _weight_key(w):
    """Stable, hashable key component for the scalar maneuver weight (or None).

    ``w`` is a scalar per :func:`_validate_solve_values` (a fraction in [0, 1]),
    so the previous ``tuple(w)`` raised ``TypeError: 'float' object is not
    iterable``. Normalise to ``float`` so ``w=0.25`` and ``w=0.5`` never collide
    and ``w=None`` (instance default) stays distinct from ``w=0.0``.
    """
    if w is None:
        return None
    if isinstance(w, bool) or not isinstance(w, (int, float)):
        raise ValueError("w must be a finite number in [0, 1] or None")
    return float(w)


def _model_key(name, subset, n_theta, n_q, w):
    return (name, tuple(subset) if subset else None, n_theta, n_q, _weight_key(w))


def _evict(cache: dict) -> None:
    while len(cache) > _CACHE_MAX:
        cache.pop(next(iter(cache)))


def _get_instance_grid(loader, name, subset, n_theta, n_q, w):
    """Cached ``(instance, grid)`` — cheap, but avoids re-reading the archive."""
    from ..model import ManeuverGrid

    key = _model_key(name, subset, n_theta, n_q, w)
    with _CACHE_LOCK:
        entry = _MODEL_CACHE.get(key)
        if entry is not None:
            return entry["ig"]
    inst = _resolve_instance(loader, name, subset, w)
    grid = ManeuverGrid.build(n_theta=n_theta, n_q=n_q, instance=inst)
    with _CACHE_LOCK:
        _MODEL_CACHE.setdefault(key, {"ig": (inst, grid), "discrete": None})
        _evict(_MODEL_CACHE)
    return inst, grid


def _get_discrete(loader, name, subset, n_theta, n_q, w):
    """Cached ``(instance, grid, DiscreteACRP)`` — caches the expensive stage."""
    from ..discretize import DiscreteACRP

    key = _model_key(name, subset, n_theta, n_q, w)
    inst, grid = _get_instance_grid(loader, name, subset, n_theta, n_q, w)
    with _CACHE_LOCK:
        entry = _MODEL_CACHE.get(key)
        if entry is not None and entry["discrete"] is not None:
            return inst, grid, entry["discrete"]
    discrete = DiscreteACRP.build(inst, grid)
    with _CACHE_LOCK:
        entry = _MODEL_CACHE.setdefault(key, {"ig": (inst, grid), "discrete": None})
        entry["discrete"] = discrete
        _evict(_MODEL_CACHE)
    return inst, grid, discrete


def _get_qubo(inst, discrete, name, subset, n_theta, n_q, w, max_qubits):
    """Return ``build_qubo(inst, discrete)`` from cache or build and store it."""
    key = (*_model_key(name, subset, n_theta, n_q, w), max_qubits)
    with _CACHE_LOCK:
        cached = _QUBO_CACHE.get(key)
    if cached is not None:
        return cached
    from ..quantum.qubo import build_qubo

    qubo = build_qubo(inst, discrete=discrete, max_qubits=max_qubits)
    with _CACHE_LOCK:
        _QUBO_CACHE[key] = qubo
        _evict(_QUBO_CACHE)
    return qubo


def instance_payload(
    inst: Instance, q: tuple[float, ...] | None = None, theta: tuple[float, ...] | None = None
) -> dict[str, Any]:
    """Radar payload: per-aircraft state + pairwise conflict diagnostics.

    ``q``/``theta`` default to the do-nothing assignment so the payload shows
    the *initial* (unresolved) picture; pass a solution's controls to show the
    resolved trajectories.
    """
    from .. import geometry

    n = inst.n
    q = q or (1.0,) * n
    theta = theta or (0.0,) * n
    theta0 = inst.theta0

    aircraft = []
    for i in inst.aircraft():
        vx, vy = geometry.aircraft_velocity(inst, i, q[i - 1], theta[i - 1])
        vx0, vy0 = geometry.aircraft_velocity(inst, i, 1.0, 0.0)  # do-nothing baseline
        aircraft.append(
            {
                "id": i,
                "x": inst.x0[i - 1],
                "y": inst.y0[i - 1],
                "vx": vx,
                "vy": vy,
                "vx0": vx0,
                "vy0": vy0,
                "speed": q[i - 1] * inst.v0[i - 1],
                "speed_initial": inst.v0[i - 1],
                "q": q[i - 1],
                "theta": theta[i - 1],
                "heading_deg": math.degrees(theta[i - 1] + theta0[i - 1]),
                "heading_initial_deg": math.degrees(theta0[i - 1]),
                "level": inst.l0[i - 1] if inst.has_flight_levels else None,
            }
        )

    conflicts = []
    for r in geometry.conflict_reports(inst, q, theta):
        if r.in_conflict:
            conflicts.append(
                {"i": r.i, "j": r.j, "min_separation": r.min_separation, "depth": r.depth}
            )

    # full CPA diagnostics per pair (the dashboard renders these; the geometry
    # is computed once here, never re-derived in JavaScript)
    pairs = []
    horizon = 0.0
    for pd in geometry.pair_diagnostics(inst, q, theta):
        if pd.closing and pd.t_cpa > horizon:
            horizon = pd.t_cpa
        pairs.append(
            {
                "i": pd.i, "j": pd.j,
                "excluded": pd.excluded,
                "initial_conflict": pd.initial_conflict,
                "resolved_conflict": pd.resolved_conflict,
                "min_separation": _clean(pd.min_separation),
                "required": pd.required,
                "violation_depth": _clean(pd.violation_depth),
                "t_cpa": _clean(pd.t_cpa),
                "closing": pd.closing,
                "xi_cpa": _clean(pd.xi_cpa), "yi_cpa": _clean(pd.yi_cpa),
                "xj_cpa": _clean(pd.xj_cpa), "yj_cpa": _clean(pd.yj_cpa),
            }
        )
    # a sensible animation horizon that comfortably covers every CPA
    horizon = round(horizon * 1.3, 4) if horizon > 0 else 1.0

    return {
        "name": inst.name,
        "family": inst.family.value,
        "n": n,
        "d": inst.d,
        "radius": inst.radius,
        "has_flight_levels": inst.has_flight_levels,
        "time_unit": "model time unit",  # unproven physical unit -> labelled honestly
        "suggested_horizon": horizon,
        "aircraft": aircraft,
        "conflicts": conflicts,
        "pairs": pairs,
        "n_conflicts": len(conflicts),
    }


def _clean(v):
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return None
    return v


def solve_payload(
    loader: InstanceLoader,
    name: str,
    *,
    solver: str = "reference",
    n_theta: int = 5,
    n_q: int = 1,
    mode: str = "aer_sim",
    reps: int = 2,
    maxiter: int = 100,
    shots: int = 1024,
    seed: int = 42,
    subset=None,
    w=None,
    profile: str = "demo",
) -> dict[str, Any]:
    """Solve an instance and return a radar payload for the resolved state."""
    profile_meta = _profile_meta(  # validates the profile name up front
        profile,
        {
            "solver": solver, "mode": mode, "n_theta": n_theta, "n_q": n_q,
            "reps": reps, "maxiter": maxiter, "shots": shots, "seed": seed,
        },
    )

    params = _validate_solve_values(
        {
            "instance": name,
            "solver": solver,
            "n_theta": n_theta,
            "n_q": n_q,
            "mode": mode,
            "reps": reps,
            "maxiter": maxiter,
            "shots": shots,
            "seed": seed,
            "subset": subset,
            "w": w,
        }
    )
    name = params["instance"]
    solver = params["solver"]
    n_theta = params["n_theta"]
    n_q = params["n_q"]
    mode = params["mode"]
    reps = params["reps"]
    maxiter = params["maxiter"]
    shots = params["shots"]
    seed = params["seed"]
    subset = params["subset"]
    w = params["w"]

    inst, grid = _get_instance_grid(loader, name, subset, n_theta, n_q, w)
    if inst.n * grid.k > _MAX_DISCRETE_VARS:
        raise ValueError(
            f"request needs {inst.n * grid.k} discrete variables; "
            f"dashboard limit is {_MAX_DISCRETE_VARS}"
        )
    inst, grid, discrete = _get_discrete(loader, name, subset, n_theta, n_q, w)
    qubo: Any = None
    raw_bitstring: str | None = None
    sample_counts: dict[str, int] = {}

    if solver == "reference":
        from ..classical.reference import DiscreteReferenceSolver

        result = DiscreteReferenceSolver().solve(inst, discrete=discrete)
        extra: dict[str, Any] = {
            "search_space": grid.k**inst.n,
            "exact_method": result.message == "exact search",
        }
    elif solver in {"qubo-exact", "qubo-annealing"}:
        from ..quantum.qubo_solvers import QuboAnnealingSolver, QuboExactSolver

        qubo = _get_qubo(inst, discrete, name, subset, n_theta, n_q, w, _MAX_DISCRETE_VARS)
        qsolver = (
            QuboExactSolver()
            if solver == "qubo-exact"
            else QuboAnnealingSolver(seed=seed)
        )
        result = qsolver.solve(qubo)
        extra = {
            "n_qubits": qubo.n_qubits,
            "qubo_energy": _clean(result.extra.get("qubo_energy")),
            "states_enumerated": result.extra.get("states_enumerated"),
            "exact_domain": (
                "all_bitstrings" if result.backend == "qubo_exact_bits"
                else "one_hot_subspace"
            ) if solver == "qubo-exact" else None,
            "global_unconstrained_qubo": (
                result.backend == "qubo_exact_bits"
            ) if solver == "qubo-exact" else None,
            "lambda_pen": qubo.penalty_conflict,
            "lambda_oh": qubo.penalty_onehot,
        }
    elif solver in {"dwave", "dwave-qpu"}:
        from ..quantum.dwave_solver import DWaveAnnealingSolver, DWaveConfig, DWaveMode

        qubo = _get_qubo(inst, discrete, name, subset, n_theta, n_q, w, _MAX_DISCRETE_VARS)
        mode = DWaveMode.QPU_REAL if solver == "dwave-qpu" else DWaveMode.NEAL_SIM
        result = DWaveAnnealingSolver(
            DWaveConfig(mode=mode, num_reads=shots, seed=seed)
        ).solve(qubo)
        extra = {
            "n_qubits": qubo.n_qubits,
            "qubo_energy": _clean(result.extra.get("qubo_energy")),
            "num_reads": _clean(result.extra.get("num_reads")),
            "is_qpu": _clean(result.extra.get("is_qpu")),
            "backend": result.backend,
            "chain_break_mean": _clean(result.extra.get("chain_break_mean")),
            "qpu_access_time_s": _clean(result.extra.get("qpu_access_time_s")),
        }
    elif solver == "qaoa":
        from ..quantum.backends import BackendConfig, BackendMode
        from ..quantum.decode import best_feasible_bitstring, decode
        from ..quantum.qaoa import QAOASolver

        cfg = BackendConfig(mode=BackendMode(mode), shots=shots, seed=seed)
        qubo = _get_qubo(inst, discrete, name, subset, n_theta, n_q, w, _MAX_DISCRETE_VARS)
        qaoa_solver = QAOASolver(backend=cfg, reps=reps, maxiter=maxiter)
        from ..quantum.explain import resource_analysis

        resources = resource_analysis(qubo)
        use_decomposition = (
            mode == "aer_sim"
            and not resources["raw_fits_statevector"]
            and resources["exact_decomposition_available"]
            and resources["decomposed_fits_statevector"]
        )
        qres = (
            qaoa_solver.solve_decomposed(qubo)
            if use_decomposition
            else qaoa_solver.solve(qubo)
        )
        raw_bitstring = qres.best_bitstring
        sample_counts = qres.counts
        result = decode(
            qubo, qres.best_bitstring, wall_time_s=qres.wall_time_s,
            seed=seed, backend=qres.backend_meta.get("name", cfg.label),
            shots=shots, qaoa_reps=reps,
        )
        _, best_prob = best_feasible_bitstring(qubo, qres.counts)
        extra = {
            "n_qubits": qubo.n_qubits,
            "circuit_depth": qres.circuit_depth,
            "two_qubit_gates": qres.two_qubit_gates,
            "backend": qres.backend_meta.get("name"),
            "is_simulator": qres.backend_meta.get("is_simulator"),
            "shots": shots,
            "reps": reps,
            "best_feasible_prob": best_prob,
            "qaoa_energy": qres.best_energy,
            "optimizer_evaluations": qres.eval_count,
            "optimal_parameters": qres.optimal_params,
            "backend_jobs": qres.n_jobs,
            "qpu_time_s": qres.qpu_time_s,
            "simulation_method": qres.sim_method,
            "exact_component_decomposition": use_decomposition,
            "n_components": (
                resources["n_exact_components"] if use_decomposition else 1
            ),
            "largest_executed_circuit_qubits": (
                resources["largest_component_qubits"]
                if use_decomposition
                else qubo.n_qubits
            ),
        }
    else:
        raise ValueError(f"unknown solver {solver!r}")

    sol = result.solution
    payload = instance_payload(inst, sol.q if sol else None, sol.theta if sol else None)

    # attach the chosen discrete option + its individual maneuver cost per aircraft
    if sol and sol.choice:
        from ..objectives import is_noop

        for ac, o in zip(payload["aircraft"], sol.choice):
            opt = grid.options[o]
            ac["option_idx"] = o
            ac["option_label"] = opt.label
            ac["maneuver_cost"] = _clean(grid.cost(opt, inst.w))
            # deviated == chosen option is not the exact NOOP (q=1, θ=0)
            ac["deviated"] = not is_noop(opt.q, opt.theta)

    from ..benchmark.export import ReproInfo

    repro = ReproInfo.capture(seed=seed, backend=str(extra.get("backend") or ""))
    payload["solution"] = {
        "solver_requested": solver,
        "solver_executed": result.solver.value,
        "solver": result.solver.value,  # back-compat
        "status": result.status.value,
        "feasible": result.feasible,
        "objective": _clean(result.objective),
        "n_initial_conflicts": result.n_initial_conflicts,
        "n_residual_conflicts": result.n_conflicts,
        "n_resolved": max(0, result.n_initial_conflicts - max(0, result.n_conflicts)),
        "total_heading_change": _clean(sum(abs(t) for t in sol.theta)) if sol else None,
        "total_speed_change": _clean(sum(abs(1.0 - qq) for qq in sol.q)) if sol else None,
        "wall_time_s": _clean(result.wall_time_s),
        "choice": list(sol.choice) if sol else [],
        "grid_k": grid.k,
        "method_note": result.message,
        "proof_status": _proof_status(solver, result, extra),
        **extra,
    }
    # Descriptive reassignment metrics ("avions déviés vs planning initial"):
    # reassignments = #{aircraft whose chosen option is not the exact NOOP}.
    if sol:
        from ..objectives import secondary_metrics

        sm = secondary_metrics(sol.q, sol.theta, inst.w)
        payload["solution"]["n_maneuvered"] = int(sm["n_maneuvered"])
        payload["solution"]["n_unchanged"] = int(sm["n_unchanged"])
    # The explanation is generated from the exact QUBO and the exact raw sample
    # when available.  Classical/annealing choices are encoded one-hot; no
    # presentation-layer approximation of H(x) is used.
    if sol and sol.choice:
        if qubo is None and discrete.n_vars <= _MAX_QUBO_VARS:
            qubo = _get_qubo(inst, discrete, name, subset, n_theta, n_q, w, _MAX_QUBO_VARS)
        if qubo is not None:
            from ..quantum.explain import (
                bitstring_explanation,
                energy_breakdown,
                resource_analysis,
                sample_explanations,
            )

            if raw_bitstring is None:
                chosen = {
                    discrete.var_index(i, sol.choice[i - 1]): 1
                    for i in inst.aircraft()
                }
                raw_bitstring = "".join(
                    "1" if chosen.get(bit) else "0"
                    for bit in range(qubo.n_qubits)
                )
            energy = energy_breakdown(qubo, raw_bitstring)
            encoding = bitstring_explanation(qubo, raw_bitstring)
            terms = {
                "maneuver_cost": energy["maneuver_cost"],
                "conflict_penalty": energy["conflict_penalty"],
                "onehot_penalty": energy["onehot_penalty"],
            }
            dominant = max(terms, key=lambda k: terms[k])
            costed = encoding["aircraft"]
            top_ac = max(costed, key=lambda r: r["cost"]) if costed else None
            payload["solution"]["explanation"] = {
                "energy": energy,
                "encoding": encoding,
                "resources": resource_analysis(qubo),
                "top_samples": sample_explanations(qubo, sample_counts, limit=10),
                "samples_available": bool(sample_counts),
                # dominant contributors — the "which/why" answers, never invented
                "summary": {
                    "dominant_energy_term": dominant,
                    "dominant_energy_value": terms[dominant],
                    "top_cost_aircraft": top_ac["aircraft"] if top_ac else None,
                    "top_cost_value": top_ac["cost"] if top_ac else None,
                    "active_conflict_pairs": len(energy["active_conflicts"]),
                    "raw_onehot_valid": encoding["raw_onehot_valid"],
                    "repair_applied": encoding["repair_applied"],
                },
            }
    payload["profile"] = profile_meta
    payload["provenance"] = {
        "git_commit": repro.git_commit,
        "git_dirty": repro.git_dirty,
        "source_hash": repro.source_hash,
        "python_version": repro.python_version,
        "package_version": repro.package_version,
        "seed": seed,
        "deps": repro.deps,
    }
    return payload


def scientific_run(
    loader: InstanceLoader, name: str, *, solver: str = "reference",
    seeds: Sequence[int] = (1, 2, 3, 4, 5), n_theta: int = 5, n_q: int = 1,
    mode: str = "aer_sim", reps: int = 2, maxiter: int = 100, shots: int = 1024,
    subset=None, w=None,
) -> dict[str, Any]:
    """Repeat a solve across seeds and report dispersion — the SCIENTIFIC profile.

    Each seed is an independent ``solve_payload(profile='scientific')``. The
    aggregate reports the feasibility rate and the objective mean/stdev/min/max
    so the UI can show robustness across seeds instead of fabricating a single
    'the' answer. Deterministic solvers simply return identical runs (stdev 0).
    """
    import statistics

    seeds = list(seeds)
    if not seeds:
        raise ValueError("scientific_run needs at least one seed")
    if len(seeds) > _MAX_SCIENTIFIC_SEEDS:
        raise ValueError(
            f"scientific_run limited to {_MAX_SCIENTIFIC_SEEDS} seeds; got {len(seeds)}"
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError("scientific_run seeds must be distinct")

    runs = [
        solve_payload(
            loader, name, solver=solver, n_theta=n_theta, n_q=n_q, mode=mode,
            reps=reps, maxiter=maxiter, shots=shots, seed=s, subset=subset, w=w,
            profile="scientific",
        )
        for s in seeds
    ]
    feasible_flags = [r["solution"]["feasible"] for r in runs]
    objectives = [
        r["solution"]["objective"]
        for r in runs
        if r["solution"]["feasible"] and r["solution"]["objective"] is not None
    ]
    aggregate = {
        "solver": solver,
        "seeds": seeds,
        "n_runs": len(runs),
        "feasible_count": sum(feasible_flags),
        "feasible_rate": sum(feasible_flags) / len(runs),
        "objective_mean": statistics.fmean(objectives) if objectives else None,
        "objective_stdev": statistics.pstdev(objectives) if len(objectives) > 1 else (
            0.0 if objectives else None
        ),
        "objective_min": min(objectives) if objectives else None,
        "objective_max": max(objectives) if objectives else None,
        "objective_spread": (max(objectives) - min(objectives)) if objectives else None,
        "deterministic": len(set(objectives)) <= 1 if objectives else None,
    }
    return {"profile": "scientific", "aggregate": aggregate, "runs": runs}


def campaign_payload(out_dir: str = "results") -> dict[str, Any]:
    """Read the campaign summary if it exists, else return an empty marker."""
    path = Path(out_dir) / "campaign_summary.json"
    if not path.is_file():
        return {"available": False, "points": []}
    return {"available": True, "points": json.loads(path.read_text(encoding="utf-8"))}


def baseline_payload(out_dir: str = "results") -> dict[str, Any]:
    """Read precomputed original-AMPL baseline comparisons (read-only).

    The dashboard only *displays* baselines produced by the CLI (``acrpq bench
    --original-ampl``); it never runs AMPL from the browser and never accepts a
    browser-supplied model/data path. The file is read from the server-side
    ``results`` directory only. Missing file -> honest empty marker.
    """
    path = Path(out_dir) / "baseline_all.json"
    if not path.is_file():
        return {"available": False, "comparisons": []}
    return {
        "available": True,
        "comparisons": json.loads(path.read_text(encoding="utf-8")),
    }


_AMPL_CAMPAIGN_SUBDIR = "ampl_gurobi_campaign_v3"
_AMPL_RECOVERY_SUBDIR = "ampl_gurobi_timeout_recovery"


def _ampl_gurobi_discrete_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """K3/K5/K7 discrete rows for one instance (K3=Demo grid, K5/K7=Sensitivity only)."""
    refs = artifact.get("discrete_references") or {}
    costs = artifact.get("discretisation_costs") or {}
    rows = []
    for k in (3, 5, 7):
        ref = refs.get(f"K{k}_q1") or {}
        cost = costs.get(f"discretisation_cost_K{k}_q1") or {}
        rows.append({
            "grid_key": f"K{k}_q1", "K": k,
            "role": "demo_grid" if k == 3 else "sensitivity_only",
            "label": "Demo grid" if k == 3 else "Sensitivity only",
            "objective": ref.get("objective"),
            "exhaustive_optimality_proven": ref.get("exhaustive_optimality_proven"),
            "search_space_size": ref.get("search_space_size"),
            "discretisation_cost": cost.get("value"),
            "discretisation_cost_official": bool(cost.get("official")),
        })
    return rows


def ampl_gurobi_payload(out_dir: str = "results") -> dict[str, Any]:
    """Read-only view of the official AMPL/Gurobi v3 campaign + non-official recovery.

    Reads ONLY fixed server paths under ``out_dir`` (never a client-supplied path),
    runs no AMPL/Gurobi, writes nothing, and never falls back to v1/v2. Officialness
    is gated through :func:`accept_official_anchor` on the hash-verified per-instance
    artifact — a stored ``is_official_anchor`` flag is never trusted on its own.
    CP_3..CP_7 surface as *Certified numerical anchor*; CP_8..CP_20 as *Provisional
    feasible incumbent — not certified*. Missing artifacts -> honest ``available:false``.
    """
    import hashlib

    from ..classical.ampl_protocol import (
        ANCHOR_QUALIFICATION,
        OFFICIAL_PROTOCOL,
        AnchorRejected,
        accept_official_anchor,
    )

    base = Path(out_dir)
    camp = base / _AMPL_CAMPAIGN_SUBDIR
    manifest_path = camp / "campaign_manifest.json"
    summary_path = camp / "campaign_summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        return {"available": False, "instances": [],
                "reason": "AMPL/Gurobi campaign artifacts not present"}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"available": False, "instances": [],
                "error": f"invalid/corrupt campaign artifact: {exc}"}
    if manifest.get("protocol_id") != OFFICIAL_PROTOCOL.protocol_id:
        return {"available": False, "instances": [],
                "error": "manifest protocol_id is not the official protocol"}

    recovery: dict[str, dict[str, Any]] = {}
    rec_dir = base / _AMPL_RECOVERY_SUBDIR
    rec_path = rec_dir / "recovery_summary.json"
    rec_manifest_path = rec_dir / "integrity_manifest.json"
    if rec_path.is_file() and rec_manifest_path.is_file():
        try:
            rec_manifest = json.loads(rec_manifest_path.read_text(encoding="utf-8"))
            rec_hashes = rec_manifest.get("sha256") or {}
            expected_summary = rec_hashes.get("recovery_summary.json")
            actual_summary = "sha256:" + hashlib.sha256(rec_path.read_bytes()).hexdigest()
            if expected_summary != actual_summary:
                raise ValueError("recovery summary hash missing or mismatched")
            rec_summary = json.loads(rec_path.read_text(encoding="utf-8"))
            for r in (rec_summary.get("rows") or []):
                rec_name = r.get("instance")
                if not isinstance(rec_name, str) or not re.fullmatch(r"CP_[0-9]+", rec_name):
                    raise ValueError("invalid recovery instance name")
                rec_art = rec_dir / f"{rec_name}.json"
                expected = rec_hashes.get(rec_art.name)
                actual = "sha256:" + hashlib.sha256(rec_art.read_bytes()).hexdigest()
                if expected != actual:
                    raise ValueError(f"recovery artifact hash missing or mismatched: {rec_name}")
                if rec_name in recovery:
                    raise ValueError(f"duplicate recovery instance: {rec_name}")
                rec_artifact = json.loads(rec_art.read_text(encoding="utf-8"))
                if rec_artifact.get("instance") != rec_name:
                    raise ValueError("recovery artifact instance does not match filename")
                recovery[rec_name] = {**r, "_verified_artifact": rec_artifact}
        except (OSError, ValueError, KeyError):
            recovery = {}

    art_hashes = manifest.get("artifact_hashes") or {}
    instances: list[dict[str, Any]] = []
    verified_ampl_version: str | None = None
    seen_names: set[str] = set()
    for row in (summary.get("rows") or []):
        name = row.get("instance")
        if not isinstance(name, str) or not re.fullmatch(r"CP_[0-9]+", name):
            return {"available": False, "instances": [],
                    "error": "invalid instance name in campaign summary"}
        if name in seen_names:
            return {"available": False, "instances": [],
                    "error": f"duplicate instance in campaign summary: {name}"}
        seen_names.add(name)
        art_path = camp / f"{name}.json"
        official, artifact, err = False, None, None
        if art_path.is_file():
            actual = "sha256:" + hashlib.sha256(art_path.read_bytes()).hexdigest()
            if art_hashes.get(name) is None:
                err = "artifact hash missing from manifest (unverified)"
            elif actual != art_hashes[name]:
                err = "artifact hash mismatch (corrupt)"
            else:
                try:
                    artifact = json.loads(art_path.read_text(encoding="utf-8"))
                    if artifact.get("instance") != name:
                        raise ValueError("artifact instance does not match filename")
                    if artifact.get("protocol_id") != OFFICIAL_PROTOCOL.protocol_id:
                        raise ValueError("artifact protocol is not official v3")
                    if verified_ampl_version is None:
                        verified_ampl_version = artifact.get("ampl_version")
                    accept_official_anchor(artifact, camp)  # raises unless truly official
                    official = True
                except AnchorRejected:
                    official = False
                except (OSError, ValueError) as exc:
                    err = f"unreadable artifact: {exc}"

        rec = recovery.get(name) or {}
        if official and artifact is not None:
            meas = artifact.get("residual_measurement") or {}
            entry = {
                "classification": "certified_numerical_anchor",
                "badge": "Certified numerical anchor", "badge_color": "green",
                "is_official_anchor": True,
                "anchor_qualification": ANCHOR_QUALIFICATION,
                "measurement": {
                    "solution_matches_primary": meas.get("solution_matches_primary"),
                    "max_abs_delta_q": meas.get("max_abs_delta_q"),
                    "max_abs_delta_theta": meas.get("max_abs_delta_theta"),
                    "measured_max_abs_mp_residual": meas.get("measured_max_abs_mp_residual"),
                },
                "ampl_version": artifact.get("ampl_version"),
                "solver_version": artifact.get("solver_version"),
                "original_ampl_gurobi_objective": artifact.get("rescored_objective"),
                "best_bound": artifact.get("best_bound"),
                "optimality_gap": artifact.get("optimality_gap"),
                "mp_residual": artifact.get("max_model_constraint_residual"),
                "wall_time_s": artifact.get("wall_time_s"),
                "feasible_common_numeric": artifact.get("feasible_common_numeric"),
                "minimum_safety_margin": artifact.get("minimum_safety_margin"),
                "continuous_controls": {
                    "q": artifact.get("q"), "theta": artifact.get("theta"),
                },
                "exclusion_reason": None,
                "discrete": _ampl_gurobi_discrete_rows(artifact),
            }
        else:
            # provisional / not certified — the continuous figure comes from recovery
            rec_artifact = rec.get("_verified_artifact") or {}
            entry = {
                "classification": "provisional_feasible_incumbent" if rec else "no_incumbent",
                "badge": "Provisional feasible incumbent — not certified",
                "badge_color": "orange",
                "is_official_anchor": False,
                "anchor_qualification": None,
                "original_ampl_gurobi_objective": rec.get("objective"),
                "best_bound": rec.get("best_bound"),
                "optimality_gap": rec.get("optimality_gap"),
                "mp_residual": None,
                "wall_time_s": rec.get("wall_time_s") or row.get("wall_time_s"),
                "feasible_common_numeric": rec.get("feasible_common_numeric"),
                "minimum_safety_margin": rec.get("minimum_safety_margin"),
                "continuous_controls": {
                    "q": rec_artifact.get("q"), "theta": rec_artifact.get("theta"),
                } if rec else None,
                "exclusion_reason": rec.get("official_exclusion_reason")
                    or "not certified within the time budget",
                "incumbent_kind": rec.get("incumbent_kind"),
                "discrete": _ampl_gurobi_discrete_rows(artifact) if artifact else [],
            }
        entry.update({
            "instance": name, "n": artifact.get("n") if artifact else None,
            "continuous_status": artifact.get("status") if artifact else "unverified",
            "protocol_id": manifest.get("protocol_id"),
            "artifact_error": err,
        })
        instances.append(entry)

    proto = manifest.get("protocol") or {}
    return {
        "available": True,
        "protocol_id": manifest.get("protocol_id"),
        "official_protocol_id": OFFICIAL_PROTOCOL.protocol_id,
        "protocol_options": proto.get("solver_options"),
        "ampl_version": verified_ampl_version,
        "counts": manifest.get("counts"),
        "official_anchors": [e["instance"] for e in instances if e["is_official_anchor"]],
        "provisional_incumbents": [e["instance"] for e in instances
                                   if e["classification"] == "provisional_feasible_incumbent"],
        "instances": instances,
    }


def qubo_payload(
    loader: InstanceLoader, name: str, *, n_theta: int = 5, n_q: int = 1,
    subset=None, w=None, max_matrix: int = 32,
) -> dict[str, Any]:
    """Expose the QUBO structure: qubits, penalty weights, term counts, matrix.

    The Q matrix (linear on the diagonal, quadratic off-diagonal) is returned
    only when ``n_qubits <= max_matrix`` to keep the payload small; the labels
    name each qubit as ``A<i>·opt<o>``.
    """
    from ..quantum.explain import coefficient_explanation, resource_analysis

    _validate_grid_size(n_theta, n_q)
    inst, grid = _get_instance_grid(loader, name, subset, n_theta, n_q, w)
    if inst.n * grid.k > _MAX_QUBO_VARS:
        raise ValueError(
            f"QUBO needs {inst.n * grid.k} variables; dashboard limit is {_MAX_QUBO_VARS}"
        )
    inst, grid, discrete = _get_discrete(loader, name, subset, n_theta, n_q, w)
    qubo = _get_qubo(inst, discrete, name, subset, n_theta, n_q, w, _MAX_QUBO_VARS)

    nq = qubo.n_qubits
    labels = []
    for b in range(nq):
        i, o = discrete.inv_index(b)
        opt = grid.options[o]
        labels.append(f"A{i}·{opt.label}")

    out: dict[str, Any] = {
        "instance": inst.name,
        "n_qubits": nq,
        "grid_k": grid.k,
        "n_aircraft": inst.n,
        "lambda_conflict": qubo.penalty_conflict,
        "lambda_onehot": qubo.penalty_onehot,
        "constant": qubo.constant,
        "n_conflict_terms": qubo.n_conflict_terms,
        "n_onehot_terms": qubo.n_onehot_terms,
        "n_linear_terms": len(qubo.linear),
        "n_quadratic_terms": len(qubo.quadratic),
        "labels": labels,
        "matrix_included": nq <= max_matrix,
        "resources": resource_analysis(qubo),
    }
    if nq <= max_matrix:
        # dense upper-triangular Q: diagonal = linear, off-diagonal = quadratic
        matrix = [[0.0] * nq for _ in range(nq)]
        for b, coeff in qubo.linear.items():
            matrix[b][b] = round(coeff, 4)
        for (a, c), coeff in qubo.quadratic.items():
            matrix[a][c] = round(coeff, 4)
        out["matrix"] = matrix
        out["coefficient_explanations"] = coefficient_explanation(qubo)
    return out


# Honest French method labels for the demo view. A simulation is NEVER called
# "quantique réel"; anything QPU-bound is "circuit préparé pour hardware, non
# soumis". These exact phrasings are load-bearing for the UI and its tests.
_METHOD_LABELS_FR = {
    "reference": "référence classique",
    "qubo-exact": "recherche exacte",
    "qubo-annealing": "heuristique classique (recuit simulé)",
    "dwave": "heuristique classique (recuit simulé — Ocean, CPU local)",
    "qaoa": "QAOA simulé localement (Aer)",
}
_HARDWARE_LABEL_FR = "circuit préparé pour hardware, non soumis"


def _fr_approx(x: int) -> str:
    """French human-scale approximation: 9_765_625 -> '≈ 9,8 millions'."""
    for unit, div in (("milliards", 1_000_000_000), ("millions", 1_000_000),
                      ("milliers", 1_000)):
        if x >= div:
            v = x / div
            s = f"{v:.1f}".rstrip("0").rstrip(".").replace(".", ",")
            return f"≈ {s} {unit}"
    return str(x)


def _max_aircraft_at_k(cap: int, k: int) -> int:
    """Largest aircraft count n with k**n <= cap (business guidance, exact loop)."""
    n = 0
    space = 1
    while space * k <= cap:
        space *= k
        n += 1
    return n


def _business_reason_fr(
    method: dict[str, Any], *, n: int, k: int, search_space: int,
    exact_cap: int, resources: dict[str, Any] | None,
) -> str | None:
    """Plain-language (French) reason a method is not applicable — never an error.

    The technical `reason` string stays untouched (collapsible detail in the UI);
    this is the primary business explanation shown instead of it.
    """
    if method["applicable"]:
        return None
    name = method["method"]
    space_fr = f"{k}^{n} {_fr_approx(search_space)} d'états"
    if not method.get("dependency_available", True):
        extra = "[quantum]" if name == "qaoa" else "[dwave]"
        return (
            f"Méthode indisponible dans cette installation (module optionnel "
            f"{extra} non installé) — ce n'est pas une erreur, les autres "
            f"méthodes restent comparables."
        )
    if name == "qubo-exact":
        max_n = _max_aircraft_at_k(exact_cap, k)
        return (
            f"Recherche exacte non applicable : l'espace de recherche "
            f"({space_fr}) dépasse le budget mémoire/temps de la démo — "
            f"utilisable jusqu'à ~{max_n} avions à K={k}. "
            f"Réduire K ou utiliser une méthode heuristique."
        )
    if name == "qaoa":
        if resources is not None:
            return (
                f"QAOA simulé non applicable : le circuit demanderait "
                f"{resources['raw_qubits']} qubits (plus grand bloc exact "
                f"{resources['largest_component_qubits']} qubits), au-delà du "
                f"plafond mémoire du simulateur local "
                f"({resources['max_safe_qubits']} qubits) — réduire K ou "
                f"choisir une autre méthode."
            )
        return (
            "QAOA simulé non applicable : le problème dépasse la capacité de "
            "la démo — réduire K ou le nombre d'avions."
        )
    return (
        "Méthode non applicable à cette configuration — réduire K ou choisir "
        "une autre méthode."
    )



def _objective_degeneracy(grid: Any, w: float) -> dict[str, Any]:
    """Are `quadratic_control_cost_v1` and `maneuver_count_v1` proportional on THIS grid?

    They coincide (up to a factor) exactly when every non-NOOP option has the same
    quadratic amplitude, so the quadratic cost is `factor x (number of deviated
    aircraft)`. That holds on the demo grid K=3/n_q=1 and fails as soon as several
    amplitudes exist (K=5, K=7, or n_q>1). Computed, never inferred from K.
    """
    from ..objectives import option_cost_vector

    quad = option_cost_vector("quadratic_control_cost_v1", grid, w)
    cnt = option_cost_vector("maneuver_count_v1", grid, w)
    amplitudes = sorted({round(q, 12) for q, c in zip(quad, cnt) if c == 1.0})
    proportional = len(amplitudes) == 1
    return {
        "proportional": proportional,
        "distinct_non_noop_amplitudes": len(amplitudes),
        "factor": amplitudes[0] if proportional else None,
        "note_fr": (
            "Sur cette grille, les deux objectifs sont proportionnels : chaque deviation "
            "non nulle possede la meme amplitude ("
            + (f"{amplitudes[0]:.6f}" if proportional else "")
            + "). Ils peuvent donc classer les solutions de maniere identique. Leur "
              "difference apparait sur des grilles ou des familles de controles comportant "
              "plusieurs amplitudes (K=5, K=7, ou n_q>1)."
            if proportional else
            "Sur cette grille, les deux objectifs ne sont PAS proportionnels : "
            f"{len(amplitudes)} amplitudes de deviation distinctes coexistent, "
            "les deux objectifs peuvent donc classer les solutions differemment."
        ),
    }

def preflight_payload(
    loader: InstanceLoader, name: str, *, n_theta: int = 5, n_q: int = 1,
    subset=None, w=None,
) -> dict[str, Any]:
    """Single source of truth for which local methods apply to a configuration.

    Every applicability decision the dashboard makes — the auto-comparison
    skips, the manual-solver guard rails, the Explain-tab verdicts — is derived
    here, from the *same* constants the solvers enforce (``QuboExactSolver``'s
    enumeration cap, the reference solver's exact-search cap, the statevector
    RAM ceiling). The front-end therefore can never disagree with the backend
    about what is runnable, why a method is skipped, or which strategy is
    planned. Real quantum hardware is reported but always marked as excluded
    from any automatic execution.
    """
    from ..classical.reference import DiscreteReferenceSolver
    from ..quantum.explain import resource_analysis
    from ..quantum.qubo_solvers import QuboExactSolver

    _validate_grid_size(n_theta, n_q)
    inst, grid = _get_instance_grid(loader, name, subset, n_theta, n_q, w)
    n_vars = inst.n * grid.k
    search_space = grid.k**inst.n

    # The QUBO (and its statevector/decomposition resources) is only meaningful
    # within the dashboard variable cap; beyond it QAOA is reported unavailable.
    resources: dict[str, Any] | None = None
    if n_vars <= _MAX_QUBO_VARS:
        _, _, discrete = _get_discrete(loader, name, subset, n_theta, n_q, w)
        qubo = _get_qubo(inst, discrete, name, subset, n_theta, n_q, w, _MAX_QUBO_VARS)
        resources = resource_analysis(qubo)

    ref_cap = DiscreteReferenceSolver().max_search_space
    exact_cap = QuboExactSolver().max_states
    quantum_ok = have("qiskit") and have("qiskit_optimization")
    dwave_ok = have("dimod") and have("dwave")
    space_str = f"{grid.k}^{inst.n} = {search_space:,} states"

    ref_exact = search_space <= ref_cap
    methods: list[dict[str, Any]] = [
        {
            "method": "reference", "label": "Discrete reference",
            "family": "classical CPU", "applicable": True,
            "resolution": "exact" if ref_exact else "heuristic",
            "reason": (
                f"exact in-grid enumeration ({space_str} <= cap {ref_cap:,})"
                if ref_exact else
                f"documented greedy/local-search fallback ({space_str} > cap {ref_cap:,})"
            ),
            "search_space": search_space,
            "dependency_available": True, "real_hardware_required": False,
        },
        {
            "method": "qubo-exact", "label": "Exact QUBO",
            "family": "exact CPU", "applicable": search_space <= exact_cap,
            "resolution": "exact",
            "reason": (
                f"global minimum of H(x) over {space_str}"
                if search_space <= exact_cap else
                f"not applicable: {space_str} exceeds the {exact_cap:,} "
                f"one-hot enumeration cap — use annealing or QAOA"
            ),
            "search_space": search_space,
            "dependency_available": True, "real_hardware_required": False,
        },
        {
            "method": "qubo-annealing", "label": "QUBO simulated annealing",
            "family": "heuristic CPU", "applicable": True, "resolution": "heuristic",
            "reason": f"stochastic local search on the same H(x) ({n_vars} vars)",
            "search_space": search_space,
            "dependency_available": True, "real_hardware_required": False,
        },
        {
            "method": "dwave", "label": "Ocean SA (local CPU)",
            "family": "annealing model on CPU", "applicable": dwave_ok,
            "resolution": "heuristic",
            "reason": (
                "D-Wave binary model sampled by classical neal"
                if dwave_ok else "not available: install the [dwave] extra"
            ),
            "search_space": search_space,
            "dependency_available": dwave_ok, "real_hardware_required": False,
        },
    ]

    qaoa_fits = bool(
        resources
        and (
            resources["raw_fits_statevector"]
            or (
                resources["exact_decomposition_available"]
                and resources["decomposed_fits_statevector"]
            )
        )
    )
    decomposed = bool(qaoa_fits and resources and not resources["raw_fits_statevector"])
    if not quantum_ok:
        qaoa_reason = "not available: install the [quantum] extra"
    elif resources is None:
        qaoa_reason = f"not applicable: {n_vars} variables exceed the dashboard cap"
    elif not qaoa_fits:
        qaoa_reason = (
            f"not applicable: raw {resources['raw_qubits']}q and largest exact "
            f"component {resources['largest_component_qubits']}q exceed the "
            f"{resources['max_safe_qubits']}q statevector RAM ceiling"
        )
    elif decomposed and resources:
        qaoa_reason = (
            f"exact split into {resources['n_exact_components']} independent "
            f"circuits (largest {resources['largest_component_qubits']}q)"
        )
    else:
        qaoa_reason = (
            f"one {resources['raw_qubits']}q variational circuit"
            if resources else "applicable"
        )
    methods.append(
        {
            "method": "qaoa", "mode": "aer_sim", "label": "QAOA Aer",
            "family": "gate-model simulator",
            "applicable": bool(quantum_ok and qaoa_fits),
            "resolution": "approximate", "reason": qaoa_reason,
            "search_space": search_space,
            "raw_qubits": resources["raw_qubits"] if resources else n_vars,
            "largest_component_qubits": (
                resources["largest_component_qubits"] if resources else None
            ),
            "planned_strategy": "exact-component split" if decomposed else "monolithic",
            "dependency_available": quantum_ok, "real_hardware_required": False,
        }
    )

    # Additive business-language layer (French): honest method labels, plain
    # reasons for non-applicable methods, and what the instance optimises.
    for m in methods:
        m["label_fr"] = _METHOD_LABELS_FR.get(m["method"], m["label"])
        m["business_reason_fr"] = _business_reason_fr(
            m, n=inst.n, k=grid.k, search_space=search_space,
            exact_cap=exact_cap, resources=resources,
        )

    objective_fr = (
        f"{inst.name} : {inst.n} avions en route convergente. Objectif : "
        f"minimiser la déviation totale des trajectoires (changements de cap θ "
        f"et de vitesse q) par rapport au planning initial, tout en garantissant "
        f"une séparation minimale de {inst.d:g} entre chaque paire d'avions. "
        f"Chaque avion choisit 1 option de manœuvre parmi K={grid.k} "
        f"(incluant l'option « ne rien faire »)."
    )
    k_explanation_fr = (
        f"K = nombre d'options candidates par avion (n_theta × n_q, incluant "
        f"l'option no-op « ne rien faire »). Variables binaires = n·K = "
        f"{inst.n}·{grid.k} = {n_vars} = qubits. Espace de recherche = K^n = "
        f"{grid.k}^{inst.n} = {search_space:,} états. K=3 par défaut pour la "
        f"démo (grille de cap symétrique minimale, équilibre couverture/qubits). "
        f"Changer K change la discrétisation : l'optimum en grille ne peut que s'améliorer "
        f"par rapport à K=3 (grilles emboîtées) — entre K quelconques les grilles ne "
        f"sont pas emboîtées et l'optimum peut varier ; les comparaisons entre méthodes "
        f"ne sont valides qu'à K identique."
    )

    return {
        "instance": inst.name, "n_aircraft": inst.n, "grid_k": grid.k,
        "n_qubits": n_vars, "search_space": search_space,
        "exact_search_cap": exact_cap, "reference_exact_cap": ref_cap,
        "resources": resources, "methods": methods,
        # The scientific identity MUST travel with every payload: two versioned
        # objectives exist and the UI previously showed neither (MoE Expert G, BLOCKER-1).
        "objective_id": DEFAULT_OBJECTIVE_ID,
        # Proportionality is COMPUTED, never assumed from K: the two objectives rank
        # identically iff every non-NOOP option carries the same quadratic amplitude.
        # True for K=3/n_q=1; false at K=5, K=7 or as soon as n_q>1.
        "objective_degeneracy": _objective_degeneracy(grid, inst.w),
        # Published separately from the objective: a tie-break, never a cost component.
        "selection_policy": selection_policy_doc(),
        "objective_id_fr": _OBJECTIVE_ID_FR[DEFAULT_OBJECTIVE_ID],
        "objective_fr": objective_fr,
        "k_explanation_fr": k_explanation_fr,
        "excluded_real_hardware": [
            {
                "method": "ibm_real", "label": "IBM QPU (gate model)",
                "label_fr": _HARDWARE_LABEL_FR,
                "real_hardware_required": True,
                "reason": "real hardware: never auto-run; manual, confirmed launch only",
            },
            {
                "method": "dwave-qpu", "label": "D-Wave QPU (annealer)",
                "label_fr": _HARDWARE_LABEL_FR,
                "real_hardware_required": True,
                "reason": "real hardware: never auto-run; needs an account token",
            },
        ],
    }


_INSTANCE_SIZE_RE = re.compile(r"^(?:CP|FP|GP)_(\d+)$|^RCP_(\d+)_")


def instance_catalog_payload(
    loader: InstanceLoader, *, n_theta: int = 5, n_q: int = 1,
) -> dict[str, Any]:
    """Classify instances by whether every automatic local route applies.

    The exact one-hot cap gives a cheap proof that large instances cannot be in
    the all-methods group. Only candidates below that cap need the full QUBO
    preflight, avoiding hundreds of pointless model builds at page load.
    """
    from ..quantum.qubo_solvers import QuboExactSolver

    _validate_grid_size(n_theta, n_q)
    grid_k = n_theta * n_q
    exact_cap = QuboExactSolver().max_states
    all_applicable: list[dict[str, Any]] = []
    limited: list[dict[str, Any]] = []

    for name in loader.list_names():
        match = _INSTANCE_SIZE_RE.match(name)
        n_aircraft = int(next(g for g in match.groups() if g)) if match else None
        if n_aircraft is not None and grid_k**n_aircraft > exact_cap:
            limited.append(
                {
                    "name": name,
                    "n_aircraft": n_aircraft,
                    "n_qubits": n_aircraft * grid_k,
                    "unavailable": ["qubo-exact"],
                    "reason": (
                        f"exact QUBO: {grid_k}^{n_aircraft} states exceeds "
                        f"cap {exact_cap:,}"
                    ),
                }
            )
            continue

        pf = preflight_payload(loader, name, n_theta=n_theta, n_q=n_q)
        blocked = [m for m in pf["methods"] if not m["applicable"]]
        record = {
            "name": name,
            "n_aircraft": pf["n_aircraft"],
            "n_qubits": pf["n_qubits"],
            "unavailable": [m["method"] for m in blocked],
            "reason": "; ".join(m["reason"] for m in blocked) or None,
        }
        (limited if blocked else all_applicable).append(record)

    return {
        "grid_k": grid_k,
        "all_applicable": all_applicable,
        "limited": limited,
        "counts": {"all_applicable": len(all_applicable), "limited": len(limited)},
    }


_DEMO_QAOA_MAX_QUBITS = 24


def grid_limit_payload(loader: InstanceLoader, name: str, *, n_q: int = 1) -> dict[str, Any]:
    """Largest odd heading grid suitable for an interactive demonstration.

    ``max_n_theta_all_methods`` is the technical applicability ceiling derived
    from preflight (including RAM).  ``slider_max_n_theta`` is deliberately
    more conservative: a monolithic Aer QAOA circuit must also stay at or below
    the measured interactive-demo ceiling.  This prevents a configuration such
    as CP_3/K9 (27 qubits) from being advertised as demo-safe merely because its
    statevector fits in memory; its variational loop exceeds the UI time budget.
    """
    from ..quantum.qubo_solvers import QuboExactSolver

    _validate_grid_size(3, n_q)
    inst = loader.load(name)
    exact_cap = QuboExactSolver().max_states
    max_all: int | None = None
    max_demo: int | None = None
    blocked_at_min: list[str] = []
    # Global validation permits at most 25 options. Heading grids are kept odd
    # so zero is represented exactly and the range stays symmetric.
    for n_theta in range(25, 2, -2):
        grid_k = n_theta * n_q
        if grid_k**inst.n > exact_cap:
            continue
        pf = preflight_payload(loader, name, n_theta=n_theta, n_q=n_q)
        blocked = [m["method"] for m in pf["methods"] if not m["applicable"]]
        if n_theta == 3:
            blocked_at_min = blocked
        if not blocked:
            if max_all is None:
                max_all = n_theta
            qaoa = next(m for m in pf["methods"] if m["method"] == "qaoa")
            largest_circuit = qaoa.get("largest_component_qubits")
            if largest_circuit is not None and largest_circuit <= _DEMO_QAOA_MAX_QUBITS:
                max_demo = n_theta
                break
    return {
        "instance": name,
        "n_aircraft": inst.n,
        "recommended_n_theta": 3,
        "max_n_theta_all_methods": max_all,
        "max_n_theta_demo": max_demo,
        "slider_max_n_theta": max_all or 3,
        "technical_qaoa_max_qubits": (
            preflight_payload(loader, name, n_theta=max_all, n_q=n_q)["n_qubits"]
            if max_all is not None else None
        ),
        "demo_qaoa_max_qubits": _DEMO_QAOA_MAX_QUBITS,
        "all_methods_at_slider_max": max_all is not None,
        "blocked_at_k3": blocked_at_min,
    }


def applicability_block(loader: InstanceLoader, req: dict[str, Any]) -> str | None:
    """Reason a requested ``(solver, mode)`` is non-applicable, else ``None``.

    Same preflight the auto-comparison uses, so the manual path can never run a
    method the dashboard reports as impossible. Never blocks the always-applicable
    routes (reference, simulated annealing) or real/remote hardware — the local
    statevector ceiling does not apply to an IBM/D-Wave QPU.
    """
    solver = req["solver"]
    if solver in {"reference", "qubo-annealing", "dwave", "dwave-qpu"}:
        return None
    if solver == "qaoa" and req["mode"] != "aer_sim":
        return None  # real/remote IBM is not bound by the local statevector ceiling
    pf = preflight_payload(
        loader, req["instance"], n_theta=req["n_theta"], n_q=req["n_q"],
        subset=req["subset"], w=req["w"],
    )
    method = next((m for m in pf["methods"] if m["method"] == solver), None)
    if method is not None and not method["applicable"]:
        return method["reason"]
    return None


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
_SOLVE_DEFAULTS = {
    "solver": "reference",  # "reference" | "qaoa"
    "n_theta": 5,
    "n_q": 1,
    "mode": "aer_sim",
    "reps": 2,
    "maxiter": 100,
    "shots": 1024,
    "seed": 42,
    "subset": None,
    "w": None,
    "profile": "demo",
}


def _coerce_solve_request(body: dict) -> dict:
    """Validate/normalise a /api/solve JSON body without a Pydantic model.

    Using a plain dict keeps this optional module importable without pydantic
    and sidesteps FastAPI forward-ref resolution under
    ``from __future__ import annotations``.
    """
    if not isinstance(body, dict) or "instance" not in body:
        raise ValueError("body must be a JSON object with an 'instance' field")
    allowed = set(_SOLVE_DEFAULTS) | {"instance"}
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise ValueError(f"unknown request fields: {', '.join(unknown)}")
    req = dict(_SOLVE_DEFAULTS)
    req.update(body)
    return _validate_solve_values(req)


def _bounded_int(name: str, value: Any, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer between {lo} and {hi}")
    return value


def _validate_grid_size(n_theta: Any, n_q: Any) -> tuple[int, int]:
    n_theta = _bounded_int("n_theta", n_theta, 1, 25)
    n_q = _bounded_int("n_q", n_q, 1, 10)
    if n_theta * n_q > _MAX_GRID_OPTIONS:
        raise ValueError(f"n_theta * n_q must not exceed {_MAX_GRID_OPTIONS}")
    return n_theta, n_q


def _validate_solve_values(req: dict[str, Any]) -> dict[str, Any]:
    req = dict(req)
    if not isinstance(req["instance"], str) or not req["instance"].strip():
        raise ValueError("instance must be a non-empty string")
    if req["solver"] not in _ALLOWED_SOLVERS:
        raise ValueError(f"solver must be one of {sorted(_ALLOWED_SOLVERS)}")
    if req["mode"] not in _ALLOWED_MODES:
        raise ValueError(f"mode must be one of {sorted(_ALLOWED_MODES)}")
    req["profile"] = req.get("profile", "demo")
    if req["profile"] not in _PROFILES:
        raise ValueError(f"profile must be one of {sorted(_PROFILES)}")
    req["n_theta"], req["n_q"] = _validate_grid_size(req["n_theta"], req["n_q"])
    req["reps"] = _bounded_int("reps", req["reps"], 1, 20)
    req["maxiter"] = _bounded_int("maxiter", req["maxiter"], 1, 10_000)
    req["shots"] = _bounded_int("shots", req["shots"], 1, 100_000)
    req["seed"] = _bounded_int("seed", req["seed"], -(2**63), 2**63 - 1)
    subset = req["subset"]
    if subset is not None:
        if (
            not isinstance(subset, list)
            or not subset
            or any(isinstance(i, bool) or not isinstance(i, int) for i in subset)
        ):
            raise ValueError("subset must be a non-empty list of integer aircraft ids")
        if len(set(subset)) != len(subset):
            raise ValueError("subset aircraft ids must be unique")
    weight = req["w"]
    if weight is not None and (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or not 0.0 <= weight <= 1.0
    ):
        raise ValueError("w must be a finite number between 0 and 1")
    return req


def create_app(
    *, data_root: str | None = None, results_dir: str = "results",
    runs_dir: str | None = None,
):
    import os

    runs_dir = runs_dir or os.environ.get("ACRPQ_RUNS_DIR", "runs")
    require("fastapi")
    from fastapi import Body, FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    # ``from __future__ import annotations`` stringifies the endpoint signatures;
    # FastAPI resolves them against this module's globals, so the lazily imported
    # ``Request`` must be visible there or it is misread as a query parameter.
    globals().setdefault("Request", Request)

    app = FastAPI(title="ACRP Radar Dashboard", version="0.1.0")
    loader = InstanceLoader(data_root=Path(data_root) if data_root is not None else None)
    from .jobs import JobManager

    jobs = JobManager(max_workers=2, max_queue=8)

    def _client_detail(exc: Exception, *, with_type: bool = True) -> str:
        """Client-visible error text with server filesystem paths stripped."""
        msg = _sanitize_error_text(str(exc))
        return f"{type(exc).__name__}: {msg}" if with_type else msg

    def _host_of(header: str | None) -> str | None:
        if not header:
            return None
        h = header.strip().lower()
        if h.startswith("http://"):
            h = h[len("http://"):]
        elif h.startswith("https://"):
            h = h[len("https://"):]
        return (
            h.split("/")[0].rsplit(":", 1)[0]
            if not h.startswith("[") else h.split("]")[0] + "]"
        )

    def _local_mutation_guard(request: Request) -> None:
        """Loopback Host/Origin allowlist + body cap for non-QPU mutating POSTs.

        Mirrors the QPU routes' guard (DNS-rebinding CSRF defense). GET routes
        stay untouched; forwarded headers are never trusted.
        """
        if _host_of(request.headers.get("host")) not in _ALLOWED_LOCAL_HOSTS:
            raise HTTPException(status_code=403, detail="Host header not allowed")
        origin = request.headers.get("origin")
        if origin is not None and _host_of(origin) not in _ALLOWED_LOCAL_HOSTS:
            raise HTTPException(status_code=403, detail="cross-origin request refused")
        clen = request.headers.get("content-length")
        if clen is not None:
            try:
                too_big = int(clen) > _MAX_LOCAL_BODY_BYTES
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="invalid content-length") from None
            if too_big:
                raise HTTPException(status_code=413, detail="request body too large")

    @app.get("/api/health")
    def health():
        import os

        return {
            "status": "ok",
            "quantum_available": have("qiskit") and have("qiskit_optimization"),
            "ibm_available": have("qiskit_ibm_runtime"),
            "minlp_available": have("pyomo"),
            "dwave_available": have("dimod") and have("dwave"),
            "dwave_qpu_available": have("dwave") and bool(os.environ.get("DWAVE_API_TOKEN")),
        }

    @app.get("/api/instances")
    def instances(family: str | None = None):
        return {"instances": loader.list_names(family)}

    @app.get("/api/instances/catalog")
    def instance_catalog(n_theta: int = 5, n_q: int = 1):
        try:
            return instance_catalog_payload(loader, n_theta=n_theta, n_q=n_q)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/instances/{name}/grid-limit")
    def instance_grid_limit(name: str, n_q: int = 1):
        try:
            return grid_limit_payload(loader, name, n_q=n_q)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/scenarios")
    def scenarios():
        from ..benchmark.scenarios import list_scenarios

        return {
            "scenarios": [
                {
                    "id": s.id,
                    "instance_name": s.instance_name,
                    "description": s.description,
                    "n_theta": s.n_theta,
                    "n_q": s.n_q,
                    "subset": list(s.subset) if s.subset else None,
                    "expected_qubits": s.expected_qubits(),
                }
                for s in list_scenarios()
            ]
        }

    @app.get("/api/instance/{name}")
    def instance(name: str):
        try:
            inst = loader.load(name)
        except Exception as exc:  # surface a clean 404, without server paths
            raise HTTPException(
                status_code=404, detail=_client_detail(exc, with_type=False)) from exc
        return instance_payload(inst)

    @app.post("/api/preview")
    def preview(body: dict = Body(...)):
        """Unsolved radar payload for the EXACT configured problem (incl. subset).

        Keeps the radar in sync with the QUBO/solver view when a quick scenario
        sub-samples aircraft — the plain /api/instance endpoint always returns
        the full instance.
        """
        name = body.get("instance")
        if not name:
            raise HTTPException(status_code=422, detail="body needs an 'instance' field")
        subset = body.get("subset")
        try:
            inst = _resolve_instance(loader, name, subset, body.get("w"))
        except Exception as exc:
            raise HTTPException(
                status_code=404, detail=_client_detail(exc, with_type=False)) from exc
        return instance_payload(inst)

    @app.get("/api/qubo/{name}")
    def qubo(name: str, n_theta: int = 5, n_q: int = 1):
        try:
            return qubo_payload(loader, name, n_theta=n_theta, n_q=n_q)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=_client_detail(exc)) from exc

    @app.post("/api/qubo")
    def qubo_post(body: dict = Body(...)):
        """Subset-aware QUBO structure (keeps the QUBO tab in sync with a scenario)."""
        name = body.get("instance")
        if not name:
            raise HTTPException(status_code=422, detail="body needs an 'instance' field")
        try:
            return qubo_payload(
                loader, name,
                n_theta=int(body.get("n_theta", 5)), n_q=int(body.get("n_q", 1)),
                subset=body.get("subset"), w=body.get("w"),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=_client_detail(exc)) from exc

    @app.post("/api/preflight")
    def preflight(body: dict = Body(...)):
        """Per-method applicability for a configuration — the one source of truth."""
        name = body.get("instance")
        if not name:
            raise HTTPException(status_code=422, detail="body needs an 'instance' field")
        try:
            return preflight_payload(
                loader, name,
                n_theta=int(body.get("n_theta", 5)), n_q=int(body.get("n_q", 1)),
                subset=body.get("subset"), w=body.get("w"),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=_client_detail(exc)) from exc

    def _require_solver_deps(solver: str) -> None:
        """503 with an install hint when a solver's optional extra is missing."""
        if solver == "qaoa" and not (have("qiskit") and have("qiskit_optimization")):
            raise HTTPException(
                status_code=503,
                detail='Quantum solver unavailable. Install: pip install "acrp-quantum[quantum]"',
            )
        if solver in {"dwave", "dwave-qpu"} and not (have("dimod") and have("dwave")):
            raise HTTPException(
                status_code=503,
                detail='D-Wave solver unavailable. Install: pip install "acrp-quantum[dwave]"',
            )

    def _do_solve(body: dict):
        """Validate a solve request and run it (shared by /api/solve and jobs)."""
        try:
            req = _coerce_solve_request(body)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        _require_solver_deps(req["solver"])
        # Preflight guard: refuse a method the dashboard reports as non-applicable
        # with a structured 422 (the formula/reason), not a generic solver crash.
        # Runs INSIDE error mapping: an unknown instance must be a clean 404,
        # never an unhandled InstanceError -> 500.
        from ..exceptions import ACRPError, InstanceError

        try:
            block = applicability_block(loader, req)
        except InstanceError as exc:
            raise HTTPException(
                status_code=404, detail=_client_detail(exc, with_type=False)) from exc
        except (ACRPError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=_client_detail(exc)) from exc
        if block:
            raise HTTPException(status_code=422, detail=block)
        try:
            return solve_payload(
                loader, req["instance"], solver=req["solver"], n_theta=req["n_theta"],
                n_q=req["n_q"], mode=req["mode"], reps=req["reps"], maxiter=req["maxiter"],
                shots=req["shots"], seed=req["seed"], subset=req["subset"], w=req["w"],
                profile=req["profile"],
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=_client_detail(exc)) from exc

    @app.post("/api/solve")
    def solve(request: Request, body: dict = Body(...)):
        _local_mutation_guard(request)
        return _do_solve(body)

    def _do_scientific(body: dict):
        """Validate + run a seed-repeated SCIENTIFIC run, optionally persisting it."""
        seeds = body.get("seeds")
        if (
            not isinstance(seeds, list) or not seeds
            or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds)
        ):
            raise HTTPException(
                status_code=422, detail="body needs a non-empty 'seeds' list of integers"
            )
        save = body.get("save", False)
        if not isinstance(save, bool):
            raise HTTPException(
                status_code=422,
                detail="'save' must be a JSON boolean (true/false), not "
                f"{type(save).__name__}",
            )
        base = {k: v for k, v in body.items() if k not in {"seeds", "save"}}
        try:
            req = _coerce_solve_request(base)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        _require_solver_deps(req["solver"])
        # Same preflight guard as /api/solve, applied ONCE before sweeping seeds so
        # a non-applicable method fails fast with a structured 422 (not mid-sweep).
        # Inside error mapping: unknown instance -> 404, never a 500.
        from ..exceptions import ACRPError, InstanceError

        try:
            block = applicability_block(loader, req)
        except InstanceError as exc:
            raise HTTPException(
                status_code=404, detail=_client_detail(exc, with_type=False)) from exc
        except (ACRPError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=_client_detail(exc)) from exc
        if block:
            raise HTTPException(status_code=422, detail=block)
        try:
            out = scientific_run(
                loader, req["instance"], solver=req["solver"], seeds=seeds,
                n_theta=req["n_theta"], n_q=req["n_q"], mode=req["mode"],
                reps=req["reps"], maxiter=req["maxiter"], shots=req["shots"],
                subset=req["subset"], w=req["w"],
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=_client_detail(exc, with_type=False)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=_client_detail(exc)) from exc

        if save:
            from .persist import build_scientific_record, save_run

            try:
                methods = preflight_payload(
                    loader, req["instance"], n_theta=req["n_theta"], n_q=req["n_q"],
                    subset=req["subset"], w=req["w"],
                )["methods"]
            except Exception:  # applicability is best-effort context, never blocks a save
                methods = []
            record = build_scientific_record(
                out, instance=req["instance"], subset=req["subset"],
                config={
                    "solver": req["solver"], "mode": req["mode"],
                    "n_theta": req["n_theta"], "n_q": req["n_q"], "reps": req["reps"],
                    "maxiter": req["maxiter"], "shots": req["shots"], "seeds": seeds,
                },
                applicability=methods,
            )
            out["saved"] = save_run(record, root=runs_dir)
        return out

    @app.post("/api/scientific")
    def scientific(request: Request, body: dict = Body(...)):
        _local_mutation_guard(request)
        return _do_scientific(body)

    @app.get("/api/runs/{kind}")
    def runs_index(kind: str):
        from .persist import list_runs

        try:
            return {"kind": kind, "runs": list_runs(kind, root=runs_dir)}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{kind}/{run_id}")
    def run_record(kind: str, run_id: str):
        from .persist import read_run

        try:
            return read_run(kind, run_id, root=runs_dir)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{kind}/{run_id}/verify")
    def run_verify(kind: str, run_id: str):
        """Recompute a saved run's hashes and compare them to the manifest."""
        from .persist import verify_run

        try:
            return verify_run(kind, run_id, root=runs_dir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            # malformed id vs a genuine integrity failure: 422 either way is a
            # clear, structured client-visible failure (never a silent success)
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    # ---- bounded asynchronous jobs (long runs) ---- #
    @app.post("/api/jobs")
    def submit_job(request: Request, body: dict = Body(...)):
        _local_mutation_guard(request)
        from .jobs import QueueFullError

        try:
            job_id = jobs.submit("solve", lambda: _do_solve(body))
        except QueueFullError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return {"job_id": job_id, **(jobs.snapshot(job_id) or {})}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        snap = jobs.snapshot(job_id)
        if snap is None:
            raise HTTPException(status_code=404, detail="unknown or expired job")
        return snap

    @app.get("/api/jobs/{job_id}/result")
    def job_result(job_id: str):
        snap = jobs.snapshot(job_id)
        if snap is None:
            raise HTTPException(status_code=404, detail="unknown or expired job")
        if snap["state"] != "completed":
            raise HTTPException(status_code=409, detail=f"job is {snap['state']}, not completed")
        return jobs.result(job_id)

    @app.delete("/api/jobs/{job_id}")
    def cancel_job(job_id: str):
        snap = jobs.cancel(job_id)
        if snap is None:
            raise HTTPException(status_code=404, detail="unknown or expired job")
        return snap

    @app.get("/api/campaign")
    def campaign():
        return JSONResponse(campaign_payload(results_dir))

    @app.get("/api/baseline")
    def baseline():
        # Read-only view of CLI-produced original-AMPL baselines; no execution.
        return JSONResponse(baseline_payload(results_dir))

    @app.get("/api/ampl-gurobi")
    def ampl_gurobi():
        # Read-only view of the official AMPL/Gurobi v3 campaign + non-official recovery.
        # No AMPL/Gurobi execution, no client paths, no writes, no v1/v2 fallback.
        return JSONResponse(ampl_gurobi_payload(results_dir))

    # Guarded QPU workflow routes (Phase 7) — prepare/dry-run/confirm/status/…;
    # fail-closed, loopback-only mutations, never submits a real IBM job.
    from .qpu_api import register_qpu_routes

    register_qpu_routes(app, loader=loader, runs_dir=runs_dir)

    # Static front-end (mounted last so /api/* wins).
    if _STATIC.is_dir():
        @app.get("/")
        def index():
            return FileResponse(_STATIC / "index.html")

        app.mount("/", StaticFiles(directory=str(_STATIC)), name="static")

    return app


def run(host: str = "127.0.0.1", port: int = 8000, **kwargs) -> None:
    require("uvicorn")
    import uvicorn

    uvicorn.run(create_app(**kwargs), host=host, port=port)
