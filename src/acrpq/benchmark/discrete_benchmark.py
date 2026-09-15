"""Bounded discrete benchmark across the six comparable solver legs (Phase 3B/3.1).

For a small, affordable set of instances this runs — on the *identical* objective,
grid and QUBO — the legs the supervisor asked to compare:

  gurobi_milp     classical discrete compatibility MILP (Gurobi via AMPL)
  reference       exhaustive in-grid enumeration (certified optimum)
  qubo_exact      exact min of H(x) over the one-hot subspace
  qubo_annealing  internal simulated annealing on H(x)
  dwave_sa        Ocean/D-Wave reference SA (CPU, neal_sim)
  qaoa_aer        QAOA on the Aer statevector simulator

Not every leg is applicable to every instance: large instances have no exhaustive
reference and no exact QUBO (their search spaces exceed the caps), and QAOA is
bounded by an explicit statevector compute budget. Applicability is reported
honestly (dependency_available / computationally_applicable / resolution / reason /
authoritative_cap_source), and skipped legs are LOGGED, never dropped silently.

Comparison uses TWO clearly-separated certified baselines, never conflated:
  * the EXHAUSTIVE optimum (ground truth) — present only when enumeration fits;
  * the CERTIFIED Gurobi-MILP optimum — present only when Gurobi *proves*
    optimality.
A leg is compared to a baseline only for its OWN objective and only when it
produced a FEASIBLE solution. If neither certified baseline exists, only the
incumbent/bound is reported — never an "optimum". An incumbent is never labelled
an optimum.

Scientific guarantees enforced (never relaxed): objectives are never compared
across kinds; raw one-hot validity is kept separate from decoded feasibility (a
repaired bitstring is never counted as raw-feasible); a missing dependency yields
``status="unavailable"`` (never a fake success); the objective-aware tolerance
(maneuver_count exact, quadratic 1e-9) is never widened.

Resume is idempotent and fail-closed: every job file embeds the canonical
``config_hash`` of the whole protocol plus the instance/grid identity hashes,
source commit and a ``result_sha256``; a resume revalidates all of these and
refuses any old/incompatible/corrupt file rather than reusing it silently.

Timing is separated into wall_time_s (perf_counter), cpu_time_s (process_time —
THIS process, so it excludes subprocess solvers such as AMPL/Gurobi and can exceed
wall for the multi-threaded Aer simulator) and provider_time_s (the solver's own
reported time; gurobi _solve_time for the MILP leg, None for the Aer/CPU legs).
"""

from __future__ import annotations

import csv
import functools
import hashlib
import io
import json
import time
from dataclasses import dataclass, field

from ..io.loader import InstanceLoader
from ..model import ManeuverGrid
from ..objectives import ObjectiveId, coerce_objective_id

SCHEMA_JOB = "acrpq-discrete-benchmark-job/2"
SCHEMA_BENCH = "acrpq-discrete-benchmark/2"

# Volatile (measured) fields excluded from every content hash — the hash is an
# INTEGRITY hash, not a cross-run reproducibility claim (sampler legs are only
# seed-pinned).
VOLATILE_FIELDS = ("wall_time_s", "cpu_time_s", "provider_time_s", "solve_time_s")

# ---------------------------------------------------------------------------
# Method taxonomy
# ---------------------------------------------------------------------------
OPTIMUM_METHODS = ("gurobi_milp", "reference", "qubo_exact")
HEURISTIC_METHODS = ("qubo_annealing", "dwave_sa", "qaoa_aer")
ALL_METHODS = OPTIMUM_METHODS + HEURISTIC_METHODS

# build_qubo's default max_qubits=29 is a *statevector-simulation* safety limit
# (only QAOA needs it). The exact one-hot search (K**n, capped by exact_max_states),
# the internal SA (choice-space) and Ocean neal (binary sampling) do NOT enumerate
# 2**n_qubits, so building the QUBO coefficient dict past 29 qubits is safe for
# them (e.g. CP_5 at K=7 = 35 qubits). Far below anything that would explode.
_QUBO_BUILD_CEILING = 4096

# Objective-aware absolute tolerance — mirrors the Phase 2.1 equivalence proof:
# integer cardinality compared EXACTLY; the float quadratic gets a tiny
# grouping/solver tolerance. Never widened to make a result pass.
_ABS_TOL: dict[str, float] = {
    ObjectiveId.MANEUVER_COUNT_V1.value: 0.0,
    ObjectiveId.QUADRATIC_CONTROL_COST_V1.value: 1e-9,
}


def abs_tol_for(objective_id: ObjectiveId | str) -> float:
    return _ABS_TOL[coerce_objective_id(objective_id).value]


# ---------------------------------------------------------------------------
# CSV / hash helpers
# ---------------------------------------------------------------------------
# Every scientific scalar in a job record is mirrored to CSV and parity-checked
# (all columns, not a subset). Timing columns are included for completeness but
# excluded from the integrity hash.
CSV_COLS = [
    "instance", "family", "n_aircraft", "grid_k", "n_theta", "n_q", "n_qubits",
    "objective_id", "method", "seed", "status", "result_class",
    "objective_value", "feasible", "raw_onehot_valid", "repair_applied",
    "decoded_feasible", "n_conflicts",
    "exhaustive_optimum", "exhaustive_status", "matches_exhaustive_optimum",
    "gap_to_exhaustive_optimum",
    "certified_milp_optimum", "milp_certified", "gap_to_certified_milp",
    "comparison_tolerance", "certified", "qubo_energy",
    "config_hash", "instance_sha256", "grid_identity_sha256", "source_commit",
    "result_sha256", "wall_time_s", "cpu_time_s", "provider_time_s",
]


def _csv_repr(v) -> str:
    """How csv.DictWriter serialises a value (None -> empty cell)."""
    return "" if v is None else str(v)


def rows_to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLS, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c) for c in CSV_COLS})
    return buf.getvalue()


def assert_json_csv_parity(rows: list[dict], csv_text: str) -> None:
    """Fail-closed: the CSV must mirror EVERY scientific column of the JSON rows."""
    csv_rows = list(csv.DictReader(io.StringIO(csv_text)))
    if len(csv_rows) != len(rows):
        raise ValueError(f"JSON/CSV row-count mismatch: {len(rows)} vs {len(csv_rows)}")
    for jr, cr in zip(rows, csv_rows):
        for col in CSV_COLS:
            if _csv_repr(jr.get(col)) != cr[col]:
                raise ValueError(
                    f"JSON/CSV parity mismatch at {jr.get('job_key')} col={col}: "
                    f"{jr.get(col)!r} != {cr[col]!r}")


# Fields excluded from the aggregate hash: the volatile per-row timings; the
# volatile synthesis timing TOTALS (their "_total" suffix means the exact-key
# match above would otherwise miss them, leaking wall-clock jitter into the hash);
# transient environment state (repository_dirty flips as soon as the run writes
# its own output files). The meaningful scientific_code_dirty flag IS hashed.
_SYNTHESIS_TIMING = {"wall_time_s_total", "cpu_time_s_total", "provider_time_s_total"}
_HASH_EXCLUDED = set(VOLATILE_FIELDS) | _SYNTHESIS_TIMING | {
    "benchmark_sha256", "benchmark_sha256_note", "repository_dirty"}


def _canonical(obj, excluded: set[str]):
    if isinstance(obj, dict):
        return {k: _canonical(v, excluded) for k, v in obj.items() if k not in excluded}
    if isinstance(obj, list):
        return [_canonical(v, excluded) for v in obj]
    return obj


def canonical_benchmark_sha256(bench: dict) -> str:
    """Deterministic sha over aggregate content (volatile timings + sha fields out)."""
    blob = json.dumps(_canonical(bench, _HASH_EXCLUDED), sort_keys=True,
                      separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# result_sha256 fingerprints one job's SCIENTIFIC content: volatile timings and
# the hash field itself are excluded (deterministic legs reproduce it exactly;
# sampler legs reproduce it for a fixed seed on the same platform).
_RESULT_HASH_EXCLUDED = set(VOLATILE_FIELDS) | {"result_sha256"}


def result_sha256(record: dict) -> str:
    blob = json.dumps(_canonical(record, _RESULT_HASH_EXCLUDED), sort_keys=True,
                      separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Instance / grid identity
# ---------------------------------------------------------------------------
def instance_sha256(loader: InstanceLoader, name: str) -> str:
    """sha256 of the instance's raw .dat text — pins the exact problem data."""
    text, _source = loader.read_dat_text(name)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def grid_identity_sha256(loader: InstanceLoader, name: str, grid_k: int, n_q: int) -> str:
    """sha256 of the canonical grid identity (levels + options) for this instance."""
    inst = loader.load(name)
    grid = ManeuverGrid.build(n_theta=grid_k, n_q=n_q, instance=inst)
    canon = json.dumps({
        "q_levels": list(grid.q_levels), "theta_levels": list(grid.theta_levels),
        "noop_index": grid.noop_index(),
        "options": [[o.idx, o.q, o.theta, o.label] for o in grid.options],
    }, sort_keys=True, allow_nan=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# LegConfig
# ---------------------------------------------------------------------------
@dataclass
class LegConfig:
    """Bounded, recorded hyper-parameters for the heuristic/QAOA legs."""

    # internal SA
    sa_sweeps: int = 400
    sa_restarts: int = 8
    # Ocean SA
    dwave_num_reads: int = 200
    dwave_num_sweeps: int = 1000
    # QAOA (Aer)
    qaoa_reps: int = 1
    qaoa_maxiter: int = 100
    qaoa_shots: int = 1024
    # caps (recorded so the protocol hash covers them)
    exact_max_states: int = 5_000_000
    # QAOA statevector COMPUTE budget (2**n_qubits cost) — tighter than the
    # 29-qubit memory ceiling; jobs above it are skipped and LOGGED.
    qaoa_max_qubits: int = 16

    def as_dict(self) -> dict[str, int]:
        return {
            "sa_sweeps": self.sa_sweeps, "sa_restarts": self.sa_restarts,
            "dwave_num_reads": self.dwave_num_reads,
            "dwave_num_sweeps": self.dwave_num_sweeps,
            "qaoa_reps": self.qaoa_reps, "qaoa_maxiter": self.qaoa_maxiter,
            "qaoa_shots": self.qaoa_shots, "exact_max_states": self.exact_max_states,
            "qaoa_max_qubits": self.qaoa_max_qubits,
        }


# ---------------------------------------------------------------------------
# Protocol identity (config_hash)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Protocol:
    """The full, canonical description of a benchmark run — everything that could
    change a result. Its ``config_hash`` keys resume: a job computed under a
    different protocol is never silently reused."""

    instances: tuple[str, ...]
    k_primary: int
    k_sensitivity: tuple[int, ...]
    sensitivity_max_aircraft: int
    n_q: int
    objectives: tuple[str, ...]
    methods: tuple[str, ...]
    seeds: tuple[int, ...]
    leg_config: dict
    schema_job: str
    schema_bench: str
    harness_sha256: str

    def _payload(self) -> dict:
        return {
            "instances": list(self.instances),          # list AND order matter
            "k_primary": self.k_primary,
            "k_sensitivity": list(self.k_sensitivity),
            "sensitivity_max_aircraft": self.sensitivity_max_aircraft,
            "n_q": self.n_q,
            "objectives": list(self.objectives),
            "methods": list(self.methods),
            "seeds": list(self.seeds),
            "leg_config": self.leg_config,
            "schema_job": self.schema_job,
            "schema_bench": self.schema_bench,
            "harness_sha256": self.harness_sha256,
        }

    @property
    def config_hash(self) -> str:
        blob = json.dumps(self._payload(), sort_keys=True, separators=(",", ":"),
                          allow_nan=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @property
    def protocol_id(self) -> str:
        return f"{self.schema_bench}:{self.config_hash[:16]}"

    def descriptor(self) -> dict:
        d = self._payload()
        d["config_hash"] = self.config_hash
        d["protocol_id"] = self.protocol_id
        return d


def build_protocol(
    *, instances, objectives, methods, seeds, leg_config: LegConfig,
    harness_sha256: str, k_primary: int = 3, k_sensitivity=(), n_q: int = 1,
    sensitivity_max_aircraft: int = 5,
) -> Protocol:
    return Protocol(
        instances=tuple(instances), k_primary=k_primary,
        k_sensitivity=tuple(k_sensitivity),
        sensitivity_max_aircraft=sensitivity_max_aircraft, n_q=n_q,
        objectives=tuple(objectives), methods=tuple(methods), seeds=tuple(seeds),
        leg_config=leg_config.as_dict(), schema_job=SCHEMA_JOB,
        schema_bench=SCHEMA_BENCH, harness_sha256=harness_sha256)


# ---------------------------------------------------------------------------
# Job identity
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchmarkJob:
    """One (instance, grid, objective, method, seed) cell of the benchmark."""

    instance: str
    grid_k: int
    n_q: int
    objective_id: str
    method: str
    seed: int | None  # None for the deterministic optimum legs

    def key(self) -> str:
        s = "det" if self.seed is None else f"s{self.seed}"
        return (f"{self.instance}__k{self.grid_k}q{self.n_q}"
                f"__{self.objective_id}__{self.method}__{s}")


class JobValidationError(RuntimeError):
    """A persisted job file is corrupt, tampered, or from another protocol/instance."""


def finalize_job_record(
    record: dict, *, protocol: Protocol, instance_sha: str, grid_identity_sha: str,
    source_commit: str,
) -> dict:
    """Stamp a freshly-computed leg record with its protocol + identity fields and
    a ``result_sha256`` over its scientific content."""
    record["protocol_id"] = protocol.protocol_id
    record["config_hash"] = protocol.config_hash
    record["instance_sha256"] = instance_sha
    record["grid_identity_sha256"] = grid_identity_sha
    record["source_commit"] = source_commit
    record["result_sha256"] = result_sha256(record)
    return record


def validate_job_record(
    record: dict, *, expected_job: BenchmarkJob, protocol: Protocol,
    instance_sha: str, grid_identity_sha: str,
) -> None:
    """Fail-closed resume check. Raises JobValidationError on ANY mismatch — an
    old/incompatible/corrupt job is never silently reused."""
    stored = record.get("result_sha256")
    recomputed = result_sha256(record)
    if stored != recomputed:
        raise JobValidationError(
            f"result_sha256 mismatch (corrupt/tampered): stored {stored} != {recomputed}")
    checks = {
        "job_key": (record.get("job_key"), expected_job.key()),
        "config_hash": (record.get("config_hash"), protocol.config_hash),
        "instance": (record.get("instance"), expected_job.instance),
        "grid_k": (record.get("grid_k"), _grid_k_of(expected_job)),
        "n_q": (record.get("n_q"), expected_job.n_q),
        "objective_id": (record.get("objective_id"),
                         coerce_objective_id(expected_job.objective_id).value),
        "method": (record.get("method"), expected_job.method),
        "seed": (record.get("seed"), expected_job.seed),
        "instance_sha256": (record.get("instance_sha256"), instance_sha),
        "grid_identity_sha256": (record.get("grid_identity_sha256"), grid_identity_sha),
    }
    for name_, (got, want) in checks.items():
        if got != want:
            raise JobValidationError(
                f"{name_} mismatch (incompatible job): stored {got!r} != expected {want!r}")


def _grid_k_of(job: BenchmarkJob) -> int:
    # grid_k stored in the record is the built grid's k (== n_theta for n_q=1);
    # for validation we compare against the record's own grid_k which was derived
    # the same way at run time, so use the job's grid_k as the trusted expectation.
    return job.grid_k


# ---------------------------------------------------------------------------
# Per-leg execution helpers
# ---------------------------------------------------------------------------
def _timed(fn):
    """Run ``fn`` returning (value, wall_s, cpu_s). CPU is this-process only."""
    w0, c0 = time.perf_counter(), time.process_time()
    value = fn()
    return value, time.perf_counter() - w0, time.process_time() - c0


def _matches(value: float | None, ref: float | None, tol: float) -> bool | None:
    if value is None or ref is None:
        return None
    return abs(value - ref) <= tol


def _primary(inst, objective_id, q_seq, theta_seq) -> float:
    """Objective-aware value of a decoded solution on the JOB's own axis (never the
    always-quadratic decode().objective — that would conflate two objectives)."""
    from ..objectives import primary_objective

    return primary_objective(objective_id, q_seq, theta_seq, inst.w)


# ---------------------------------------------------------------------------
# Certified baselines (two, kept strictly separate)
# ---------------------------------------------------------------------------
def exhaustive_baseline(
    loader: InstanceLoader, instance: str, grid_k: int, n_q: int,
    objective_id: ObjectiveId | str,
) -> dict:
    """Exhaustive-enumeration optimum (ground truth) for one (instance, grid,
    objective). ``optimum`` is None when the search space exceeds the cap — then
    no gap-to-exhaustive can be computed (never a fabricated baseline)."""
    from ..classical.discrete_enum import exhaustive_discrete_optimum

    inst = loader.load(instance)
    res, wall, cpu = _timed(lambda: exhaustive_discrete_optimum(
        inst, objective_id=objective_id, n_theta=grid_k, n_q=n_q))
    optimum = res.optimum if res.status == "optimal" else None
    return {
        "kind": "exhaustive", "objective_id": coerce_objective_id(objective_id).value,
        "status": res.status, "optimum": optimum,
        "n_optima": res.n_optima if res.status == "optimal" else None,
        "tie_kind": res.tie_kind if res.status == "optimal" else None,
        "search_space": res.search_space, "wall_time_s": wall, "cpu_time_s": cpu,
    }


def certified_milp_baseline(
    loader: InstanceLoader, instance: str, grid_k: int, n_q: int,
    objective_id: ObjectiveId | str,
) -> dict:
    """Certified Gurobi-MILP optimum for one (instance, grid, objective).
    ``optimum`` is None unless Gurobi PROVED optimality (certified) — an
    uncertified incumbent is never used as an optimum baseline."""
    from ..classical.discrete_milp import solve_discrete_milp

    inst = loader.load(instance)
    res, wall, cpu = _timed(lambda: solve_discrete_milp(
        inst, objective_id=objective_id, n_theta=grid_k, n_q=n_q))
    optimum = res.optimum if res.certified else None
    return {
        "kind": "certified_milp", "objective_id": coerce_objective_id(objective_id).value,
        "status": res.status, "certified": res.certified, "optimum": optimum,
        "abs_gap": res.abs_gap, "rel_gap": res.rel_gap,
        "solver_version": res.solver_version, "wall_time_s": wall, "cpu_time_s": cpu,
    }


def build_baselines(
    loader: InstanceLoader, instance: str, grid_k: int, n_q: int,
    objective_id: ObjectiveId | str,
) -> dict:
    """Both certified baselines for one (instance, grid, objective)."""
    return {
        "exhaustive": exhaustive_baseline(loader, instance, grid_k, n_q, objective_id),
        "certified_milp": certified_milp_baseline(loader, instance, grid_k, n_q, objective_id),
    }


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------
def _base_record(loader: InstanceLoader, job: BenchmarkJob) -> dict:
    inst = loader.load(job.instance)
    grid = ManeuverGrid.build(n_theta=job.grid_k, n_q=job.n_q, instance=inst)
    return {
        "schema": SCHEMA_JOB,
        "job_key": job.key(),
        "instance": job.instance,
        "family": inst.family.value,
        "n_aircraft": inst.n,
        "grid_k": grid.k,
        "n_theta": job.grid_k,
        "n_q": job.n_q,
        "n_qubits": inst.n * grid.k,
        "objective_id": coerce_objective_id(job.objective_id).value,
        "method": job.method,
        "seed": job.seed,
    }


def _finish(rec: dict, *, status: str, result_class: str,
            objective_value: float | None, feasible: bool | None,
            wall_s: float | None, cpu_s: float | None, provider_s: float | None,
            baselines: dict, **extra) -> dict:
    """Attach the dual-baseline comparison + timing fields to a leg record.

    A leg is only ever scored against a CERTIFIED baseline for its OWN objective,
    and only when it produced a FEASIBLE solution. The two baselines
    (exhaustive ground truth, certified-MILP optimum) are reported separately and
    never conflated; if neither exists, no gap and no optimum is claimed.
    """
    tol = abs_tol_for(rec["objective_id"])
    comparable = objective_value if feasible else None
    ex = baselines.get("exhaustive", {}) or {}
    milp = baselines.get("certified_milp", {}) or {}
    ex_opt = ex.get("optimum")       # None unless exhaustive reached optimal
    milp_opt = milp.get("optimum")   # None unless Gurobi certified optimality
    rec.update({
        "status": status,
        "result_class": result_class,
        "objective_value": objective_value,
        "feasible": feasible,
        "exhaustive_optimum": ex_opt,
        "exhaustive_status": ex.get("status"),
        "matches_exhaustive_optimum": _matches(comparable, ex_opt, tol),
        "gap_to_exhaustive_optimum": (None if comparable is None or ex_opt is None
                                      else comparable - ex_opt),
        "certified_milp_optimum": milp_opt,
        "milp_certified": milp.get("certified"),
        "gap_to_certified_milp": (None if comparable is None or milp_opt is None
                                  else comparable - milp_opt),
        "comparison_tolerance": tol,
        "wall_time_s": wall_s,
        "cpu_time_s": cpu_s,
        "provider_time_s": provider_s,
    })
    rec.update(extra)
    return rec


def run_job(loader: InstanceLoader, job: BenchmarkJob, *,
            baselines: dict, cfg: LegConfig | None = None) -> dict:
    """Execute one leg and return a JSON-native record (without protocol/identity
    stamps — call finalize_job_record for those)."""
    cfg = cfg or LegConfig()
    rec = _base_record(loader, job)
    m = job.method
    if m == "gurobi_milp":
        return _run_milp(loader, job, rec, baselines)
    if m == "reference":
        return _run_reference(loader, job, rec, baselines)
    if m == "qubo_exact":
        return _run_qubo_exact(loader, job, rec, baselines, cfg)
    if m == "qubo_annealing":
        return _run_qubo_annealing(loader, job, rec, baselines, cfg)
    if m == "dwave_sa":
        return _run_dwave_sa(loader, job, rec, baselines, cfg)
    if m == "qaoa_aer":
        return _run_qaoa(loader, job, rec, baselines, cfg)
    raise ValueError(f"unknown method: {m}")


# -- optimum legs -----------------------------------------------------------
def _run_milp(loader, job, rec, baselines) -> dict:
    from ..classical.discrete_milp import solve_discrete_milp

    inst = loader.load(job.instance)
    res, wall, cpu = _timed(lambda: solve_discrete_milp(
        inst, objective_id=job.objective_id, n_theta=job.grid_k, n_q=job.n_q))
    if res.status == "unavailable":
        return _finish(rec, status="unavailable", result_class="unavailable",
                       objective_value=None, feasible=None, wall_s=wall, cpu_s=cpu,
                       provider_s=res.solve_time_s, baselines=baselines,
                       message=res.message)
    # An incumbent Gurobi could not certify is NOT an optimum.
    result_class = "certified_optimum" if res.certified else "incumbent"
    return _finish(
        rec, status=res.status,
        result_class=(result_class if res.feasible else "infeasible"),
        objective_value=res.optimum, feasible=res.feasible,
        wall_s=wall, cpu_s=cpu, provider_s=res.solve_time_s, baselines=baselines,
        certified=res.certified, solve_result=res.solve_result,
        best_bound=res.best_bound, abs_gap=res.abs_gap, rel_gap=res.rel_gap,
        solver_version=res.solver_version, n_conflicts=res.n_conflicts,
        raw_onehot_valid=True, repair_applied=False,  # choice vector by construction
        decoded_feasible=res.feasible)


def _run_reference(loader, job, rec, baselines) -> dict:
    from ..classical.discrete_enum import exhaustive_discrete_optimum

    inst = loader.load(job.instance)
    res, wall, cpu = _timed(lambda: exhaustive_discrete_optimum(
        inst, objective_id=job.objective_id, n_theta=job.grid_k, n_q=job.n_q))
    if res.status == "search_too_large":
        return _finish(rec, status="search_too_large", result_class="unavailable",
                       objective_value=None, feasible=None, wall_s=wall, cpu_s=cpu,
                       provider_s=None, baselines=baselines,
                       search_space=res.search_space)
    feasible = res.status == "optimal" and res.feasible
    return _finish(
        rec, status=res.status,
        result_class=("certified_optimum" if feasible else "infeasible"),
        objective_value=res.optimum, feasible=res.feasible,
        wall_s=wall, cpu_s=cpu, provider_s=None, baselines=baselines,
        n_optima=res.n_optima, tie_kind=res.tie_kind, search_space=res.search_space,
        n_conflicts=res.n_conflicts,
        raw_onehot_valid=True, repair_applied=False, decoded_feasible=res.feasible)


def _run_qubo_exact(loader, job, rec, baselines, cfg) -> dict:
    from ..quantum.qubo import build_qubo, verify_qubo_wellformed
    from ..quantum.qubo_solvers import QuboExactSolver

    inst = loader.load(job.instance)
    qubo = build_qubo(inst, n_theta=job.grid_k, n_q=job.n_q,
                      objective_id=job.objective_id, max_qubits=_QUBO_BUILD_CEILING)
    onehot_guaranteed = bool(qubo.meta.get("onehot_optimum_guaranteed"))
    states = qubo.discrete.k ** inst.n if inst.n else 1
    if states > cfg.exact_max_states:
        return _finish(rec, status="search_too_large", result_class="unavailable",
                       objective_value=None, feasible=None, wall_s=None, cpu_s=None,
                       provider_s=None, baselines=baselines,
                       states=float(states), max_states=float(cfg.exact_max_states))
    try:
        verify_qubo_wellformed(qubo)
    except ValueError as exc:  # fail-closed: never trust a malformed QUBO
        return _finish(rec, status="error", result_class="unavailable",
                       objective_value=None, feasible=None, wall_s=None, cpu_s=None,
                       provider_s=None, baselines=baselines, message=str(exc))
    solver = QuboExactSolver(max_states=cfg.exact_max_states)
    result, wall, cpu = _timed(lambda: solver.solve(qubo))
    # "one_hot_optimum" is unreachable-by-design defense-in-depth: verify above
    # already fails closed unless onehot_optimum_guaranteed is True. The
    # conservative label is kept so the class would degrade honestly if that gate
    # were ever loosened, rather than overclaim a global "qubo_optimum".
    rc = ("qubo_optimum" if onehot_guaranteed else "one_hot_optimum") if result.feasible \
        else "infeasible"
    obj_val = _primary(inst, job.objective_id, result.solution.q, result.solution.theta)
    return _finish(
        rec, status="optimal", result_class=rc,
        objective_value=obj_val, feasible=result.feasible,
        wall_s=wall, cpu_s=cpu, provider_s=None, baselines=baselines,
        qubo_energy=result.extra.get("qubo_energy"),
        onehot_optimum_guaranteed=onehot_guaranteed,
        n_conflicts=result.n_conflicts,
        raw_onehot_valid=True, repair_applied=False,  # one-hot block search
        decoded_feasible=result.feasible)


# -- heuristic legs ---------------------------------------------------------
def _run_qubo_annealing(loader, job, rec, baselines, cfg) -> dict:
    from ..quantum.qubo import build_qubo
    from ..quantum.qubo_solvers import QuboAnnealingSolver

    inst = loader.load(job.instance)
    qubo = build_qubo(inst, n_theta=job.grid_k, n_q=job.n_q,
                      objective_id=job.objective_id, max_qubits=_QUBO_BUILD_CEILING)
    solver = QuboAnnealingSolver(n_sweeps=cfg.sa_sweeps, restarts=cfg.sa_restarts,
                                 seed=job.seed if job.seed is not None else 1234)
    result, wall, cpu = _timed(lambda: solver.solve(qubo))
    obj_val = _primary(inst, job.objective_id, result.solution.q, result.solution.theta)
    return _finish(
        rec, status="incumbent", result_class="incumbent",
        objective_value=obj_val, feasible=result.feasible,
        wall_s=wall, cpu_s=cpu, provider_s=None, baselines=baselines,
        qubo_energy=result.extra.get("qubo_energy"), n_conflicts=result.n_conflicts,
        raw_onehot_valid=True, repair_applied=False,  # SA moves in choice space
        decoded_feasible=result.feasible)


def _run_dwave_sa(loader, job, rec, baselines, cfg) -> dict:
    from ..quantum.dwave_solver import DWaveAnnealingSolver, DWaveConfig, DWaveMode

    qubo = _build(loader, job)
    conf = DWaveConfig(mode=DWaveMode.NEAL_SIM, num_reads=cfg.dwave_num_reads,
                       num_sweeps=cfg.dwave_num_sweeps,
                       seed=job.seed if job.seed is not None else 1234)
    solver = DWaveAnnealingSolver(config=conf)
    try:
        raw, wall, cpu = _timed(lambda: solver.solve_raw(qubo))
    except Exception as exc:  # missing Ocean dep -> explicit unavailable, no fake win
        return _finish(rec, status="unavailable", result_class="unavailable",
                       objective_value=None, feasible=None, wall_s=None, cpu_s=None,
                       provider_s=None, baselines=baselines, message=str(exc))
    return _decode_sample(rec, baselines, qubo, raw.best_bitstring, wall, cpu,
                          provider_s=raw.qpu_access_time_s,  # None for neal_sim
                          qubo_energy=raw.best_energy, num_reads=raw.num_reads,
                          is_qpu=raw.is_qpu)


def _run_qaoa(loader, job, rec, baselines, cfg) -> dict:
    from ..quantum.backends import BackendConfig, BackendMode
    from ..quantum.qaoa import QAOASolver

    qubo = _build(loader, job)
    if qubo.n_qubits > cfg.qaoa_max_qubits:  # defense-in-depth (planner also filters)
        return _finish(rec, status="skipped_budget", result_class="unavailable",
                       objective_value=None, feasible=None, wall_s=None, cpu_s=None,
                       provider_s=None, baselines=baselines,
                       message=(f"{qubo.n_qubits} qubits > qaoa_max_qubits="
                                f"{cfg.qaoa_max_qubits}"))
    backend = BackendConfig(mode=BackendMode.AER_SIM, shots=cfg.qaoa_shots,
                            seed=job.seed if job.seed is not None else 1234)
    solver = QAOASolver(backend=backend, reps=cfg.qaoa_reps, maxiter=cfg.qaoa_maxiter)
    try:
        qres, wall, cpu = _timed(lambda: solver.solve(qubo))
    except Exception as exc:  # missing qiskit/aer -> explicit unavailable
        return _finish(rec, status="unavailable", result_class="unavailable",
                       objective_value=None, feasible=None, wall_s=None, cpu_s=None,
                       provider_s=None, baselines=baselines, message=str(exc))
    return _decode_sample(rec, baselines, qubo, qres.best_bitstring, wall, cpu,
                          provider_s=qres.qpu_time_s,  # None for Aer sim
                          qubo_energy=qres.best_energy, qaoa_reps=cfg.qaoa_reps,
                          shots=cfg.qaoa_shots, eval_count=qres.eval_count,
                          circuit_depth=qres.circuit_depth)


def _build(loader: InstanceLoader, job: BenchmarkJob):
    from ..quantum.qubo import build_qubo

    return build_qubo(loader.load(job.instance), n_theta=job.grid_k, n_q=job.n_q,
                      objective_id=job.objective_id, max_qubits=_QUBO_BUILD_CEILING)


def _decode_sample(rec, baselines, qubo, bitstring, wall, cpu, *, provider_s,
                   qubo_energy, **extra) -> dict:
    """Score a *raw* sampled bitstring, keeping raw one-hot validity separate from
    decoded (repaired) feasibility. The two sampling legs share this."""
    from ..quantum.decode import decode
    from ..quantum.explain import bitstring_explanation

    exp = bitstring_explanation(qubo, bitstring)   # raw_onehot_valid / repair_applied
    result = decode(qubo, bitstring)               # geometry-rescored, repaired
    sol = result.solution
    if sol is None:  # decode() always sets a Solution; guard fail-closed anyway
        raise RuntimeError("decode() returned no solution")
    obj_val = _primary(qubo.discrete.instance, rec["objective_id"], sol.q, sol.theta)
    return _finish(
        rec, status="incumbent", result_class="incumbent",
        objective_value=obj_val, feasible=result.feasible,
        wall_s=wall, cpu_s=cpu, provider_s=provider_s, baselines=baselines,
        qubo_energy=qubo_energy,
        raw_onehot_valid=exp["raw_onehot_valid"],
        repair_applied=exp["repair_applied"],
        decoded_feasible=exp["decoded_feasible"],
        decoded_conflicts=exp["decoded_conflicts"],
        n_conflicts=result.n_conflicts,
        **extra)


# ---------------------------------------------------------------------------
# Honest applicability
# ---------------------------------------------------------------------------
def _sanitize_solver_version(text: str) -> str | None:
    """Extract ONLY a ``Gurobi X.Y.Z`` version token. The raw AMPL banner contains
    the licensee's name/email and license path — never publish that. Returns None
    if no clean version token is found."""
    import re
    m = re.search(r"Gurobi\s+(\d+\.\d+\.\d+)", text)
    return f"Gurobi {m.group(1)}" if m else None


@functools.lru_cache(maxsize=1)
def probe_gurobi() -> dict:
    """Verify the Gurobi/AMPL driver AND license with a tiny LP (no scientific
    model). ``amplpy`` being importable is NOT sufficient — the driver must load
    and the license must let it solve. No fallback: on any failure
    ``solver_verified`` is False and the MILP leg is reported unavailable. Only a
    sanitized ``Gurobi X.Y.Z`` token is published (never the license banner)."""
    import importlib.util
    if importlib.util.find_spec("amplpy") is None:
        return {"dependency_available": False, "solver_verified": False,
                "version": None, "reason": "amplpy not importable"}
    try:
        import amplpy
        ampl = amplpy.AMPL()
        try:
            ampl.eval("var x >= 0 <= 1; maximize obj: x;")
            ampl.setOption("solver", "gurobi")
            banner = ampl.get_output("solve;")   # captured, not printed to stdout
            solve_result = str(ampl.getValue("solve_result"))
            xval = float(ampl.getValue("x"))
            ok = solve_result == "solved" and abs(xval - 1.0) < 1e-6
            return {"dependency_available": True, "solver_verified": bool(ok),
                    "version": _sanitize_solver_version(banner),
                    "reason": (f"probe LP solve_result={solve_result}"
                               if ok else f"probe LP not solved: {solve_result}")}
        finally:
            ampl.close()
    except Exception as exc:  # driver missing / license invalid / anything
        return {"dependency_available": True, "solver_verified": False,
                "version": None, "reason": f"gurobi probe failed: {type(exc).__name__}"}


def method_applicability(
    loader: InstanceLoader, name: str, grid_k: int, n_q: int, cfg: LegConfig,
    gurobi_probe: dict | None = None,
) -> dict[str, dict]:
    """Per-method honest applicability for one (instance, grid).

    Each entry publishes SEPARATELY: dependency_available, computationally_applicable,
    resolution (certified_exact | heuristic | unavailable), reason, and
    authoritative_cap_source. The five enum/anneal legs read applicability from
    ``preflight_payload`` (the solvers' own cap authority); the MILP leg from a
    verified Gurobi driver/license probe.
    """
    from ..dashboard.api import preflight_payload

    from .family_matrix import _PF_RENAME

    gp = gurobi_probe or probe_gurobi()
    pf = preflight_payload(loader, name, n_theta=grid_k, n_q=n_q)
    inst = loader.load(name)
    n_qubits = inst.n * ManeuverGrid.build(n_theta=grid_k, n_q=n_q, instance=inst).k

    out: dict[str, dict] = {}
    for m in pf["methods"]:
        key = _PF_RENAME.get(m["method"])
        if key is None:
            continue
        dep = bool(m.get("dependency_available", True))
        applicable = bool(m["applicable"])
        pf_res = m.get("resolution")
        resolution = ("certified_exact" if applicable and pf_res == "exact"
                      else "heuristic" if applicable else "unavailable")
        cap_source = ("preflight.exact_search_cap" if key == "qubo_exact"
                      else "preflight.memory_ceiling" if key == "qaoa_aer"
                      else "preflight.dependency_probe")
        out[key] = {
            "dependency_available": dep,
            "computationally_applicable": applicable,
            "resolution": resolution,
            "reason": m["reason"],
            "authoritative_cap_source": cap_source,
        }
    # The `reference` leg is the EXHAUSTIVE enumerator (certified optimum), NOT
    # preflight's reference.py greedy fallback — so gate it by the exhaustive
    # search cap. Above that cap it has no fallback (returns search_too_large),
    # so it is honestly unavailable, never a silent heuristic.
    from ..classical.discrete_enum import DEFAULT_MAX_SEARCH
    grid_k_val = ManeuverGrid.build(n_theta=grid_k, n_q=n_q, instance=inst).k
    states = grid_k_val ** inst.n if inst.n else 1
    ref_ok = states <= DEFAULT_MAX_SEARCH
    out["reference"] = {
        "dependency_available": True,
        "computationally_applicable": ref_ok,
        "resolution": "certified_exact" if ref_ok else "unavailable",
        "reason": (f"exhaustive enumeration of {grid_k_val}**{inst.n}={states} states "
                   + ("within" if ref_ok else "exceeds")
                   + f" the exhaustive cap {DEFAULT_MAX_SEARCH}"),
        "authoritative_cap_source": "exhaustive_enum.max_search",
    }
    # QAOA additionally bounded by the benchmark's statevector COMPUTE budget.
    if "qaoa_aer" in out and out["qaoa_aer"]["computationally_applicable"] \
            and n_qubits > cfg.qaoa_max_qubits:
        out["qaoa_aer"].update({
            "computationally_applicable": False, "resolution": "unavailable",
            "reason": (f"{n_qubits} qubits > benchmark qaoa_max_qubits="
                       f"{cfg.qaoa_max_qubits} (statevector compute budget)"),
            "authoritative_cap_source": "benchmark.qaoa_max_qubits",
        })
    # Gurobi MILP: driver+license-verified, no enumeration/RAM ceiling. amplpy
    # importable is not enough — solver_verified must be True.
    out["gurobi_milp"] = {
        "dependency_available": gp["dependency_available"],
        "computationally_applicable": bool(gp["solver_verified"]),
        "resolution": "certified_exact" if gp["solver_verified"] else "unavailable",
        "reason": gp["reason"],
        "authoritative_cap_source": "gurobi_driver_license_probe",
    }
    return out


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
DEFAULT_INSTANCES = ("CP_3", "CP_4", "CP_5", "CP_6", "CP_7", "CP_8")
DEFAULT_OBJECTIVES = (ObjectiveId.QUADRATIC_CONTROL_COST_V1.value,
                      ObjectiveId.MANEUVER_COUNT_V1.value)
DEFAULT_SEEDS = (1234, 2, 3)

# Multi-family selection (Phase 3.1B): a small, justified cross-family set at K=3.
# Not every leg is applicable everywhere — see method_applicability + the logged
# skips. FP_4/FP_5/RCP_10 keep an exhaustive reference; RCP_FL (n=50) has no
# exhaustive/exact-QUBO/QAOA (search space + qubit count too large) so it is
# characterised by the certified Gurobi MILP + the two CPU heuristics only.
MULTIFAMILY_INSTANCES = ("FP_4", "FP_5", "RCP_10_1", "RCP_10_2", "GP_4", "RCP_50_1")


@dataclass
class Plan:
    jobs: list[BenchmarkJob] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)  # (instance,grid,method,reason)


def plan_jobs(
    loader: InstanceLoader,
    *,
    instances: tuple[str, ...] = DEFAULT_INSTANCES,
    objectives: tuple[str, ...] = DEFAULT_OBJECTIVES,
    k_primary: int = 3,
    k_sensitivity: tuple[int, ...] = (5, 7),
    sensitivity_max_aircraft: int = 5,
    n_q: int = 1,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    cfg: LegConfig | None = None,
    gurobi_probe: dict | None = None,
) -> Plan:
    """Enumerate the benchmark jobs, running only genuinely-applicable legs.

    Applicability comes from ``method_applicability`` (preflight caps + the
    verified Gurobi probe). Non-applicable (instance, grid, method) triples are
    recorded in ``skipped`` with the reason and cap source — never silently
    dropped. Heuristic legs get one job per seed; the optimum legs are
    deterministic (single job, ``seed=None``).
    """
    cfg = cfg or LegConfig()
    gp = gurobi_probe or probe_gurobi()
    plan = Plan()
    for name in instances:
        n_aircraft = loader.load(name).n
        grids = [k_primary] + [k for k in k_sensitivity
                               if n_aircraft <= sensitivity_max_aircraft]
        for k in grids:
            applic = method_applicability(loader, name, k, n_q, cfg, gp)
            for objective_id in objectives:
                for method in ALL_METHODS:
                    info = applic.get(method, {})
                    if not info.get("computationally_applicable"):
                        plan.skipped.append({
                            "instance": name, "grid_k": k, "objective_id": objective_id,
                            "method": method, "reason": info.get("reason", "not applicable"),
                            "authoritative_cap_source": info.get("authoritative_cap_source")})
                        continue
                    job_seeds: tuple[int | None, ...] = (
                        seeds if method in HEURISTIC_METHODS else (None,))
                    for seed in job_seeds:
                        plan.jobs.append(BenchmarkJob(
                            instance=name, grid_k=k, n_q=n_q,
                            objective_id=objective_id, method=method, seed=seed))
    return plan


# ---------------------------------------------------------------------------
# Aggregation validation
# ---------------------------------------------------------------------------
def validate_aggregate(rows: list[dict], plan: Plan) -> None:
    """Fail-closed: the aggregated rows must be EXACTLY the planned jobs — no
    unexpected/extra files, no missing keys, no duplicates — and every leg in a
    (instance, grid, objective) group must cite the same certified baselines."""
    expected = [j.key() for j in plan.jobs]
    expected_set = set(expected)
    if len(expected) != len(expected_set):
        raise ValueError("plan contains duplicate job keys")
    present = [r["job_key"] for r in rows]
    present_set = set(present)
    if len(present) != len(present_set):
        dups = sorted({k for k in present if present.count(k) > 1})
        raise ValueError(f"duplicate job records: {dups}")
    extra = present_set - expected_set
    if extra:
        raise ValueError(f"unexpected job files not in the current plan: {sorted(extra)}")
    missing = expected_set - present_set
    if missing:
        raise ValueError(f"missing planned jobs: {sorted(missing)}")
    # each record's filename-key must match its internal job_key (already keyed on
    # job_key here) and its baseline must belong to the same instance/grid/objective
    by_group: dict[tuple, tuple] = {}
    for r in rows:
        g = (r["instance"], r["grid_k"], r["objective_id"])
        base = (r.get("exhaustive_optimum"), r.get("exhaustive_status"),
                r.get("certified_milp_optimum"), r.get("milp_certified"))
        if g in by_group and by_group[g] != base:
            raise ValueError(
                f"inconsistent certified baselines within {g}: {by_group[g]} != {base}")
        by_group[g] = base


# ---------------------------------------------------------------------------
# Per-family / objective / method synthesis
# ---------------------------------------------------------------------------
def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def summarize(rows: list[dict]) -> dict:
    """Per (family, objective, method) synthesis. Gaps are averaged/median-ed ONLY
    over FEASIBLE solutions against a CERTIFIED baseline (exhaustive preferred,
    else certified MILP). Deliberately NOT an IID aggregation across families, and
    the seed count is too small to be a strong statistical characterisation — both
    caveats are published in the artifact alongside these numbers."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["family"], r["objective_id"], r["method"]), []).append(r)

    summary = []
    for (family, objective_id, method), rs in sorted(groups.items()):
        executed = [r for r in rs if r["status"] not in ("unavailable", "skipped_budget")]
        feasible = [r for r in executed if r.get("feasible") is True]
        infeasible = [r for r in executed if r.get("feasible") is False]
        # gaps only on feasible rows that HAVE a certified baseline
        gaps_ex = [r["gap_to_exhaustive_optimum"] for r in feasible
                   if r.get("gap_to_exhaustive_optimum") is not None]
        gaps_milp = [r["gap_to_certified_milp"] for r in feasible
                     if r.get("gap_to_certified_milp") is not None]
        # "optimum reached among comparable runs" = matched a certified baseline
        matched_ex = [r for r in feasible if r.get("matches_exhaustive_optimum") is True]
        summary.append({
            "family": family, "objective_id": objective_id, "method": method,
            "n_runs": len(rs),
            "n_executed": len(executed),
            "n_unavailable": len(rs) - len(executed),
            "n_feasible": len(feasible),
            "n_infeasible": len(infeasible),
            "n_matched_exhaustive_optimum": len(matched_ex),
            "gap_to_exhaustive_mean": (sum(gaps_ex) / len(gaps_ex)) if gaps_ex else None,
            "gap_to_exhaustive_median": _median(gaps_ex),
            "n_gap_to_exhaustive": len(gaps_ex),
            "gap_to_certified_milp_mean": (sum(gaps_milp) / len(gaps_milp)) if gaps_milp else None,
            "gap_to_certified_milp_median": _median(gaps_milp),
            "n_gap_to_certified_milp": len(gaps_milp),
            "wall_time_s_total": sum(r.get("wall_time_s") or 0.0 for r in executed),
            "cpu_time_s_total": sum(r.get("cpu_time_s") or 0.0 for r in executed),
            "provider_time_s_total": sum(r.get("provider_time_s") or 0.0 for r in executed),
        })
    return {
        "by_family_objective_method": summary,
        "caveats": [
            "Per-family rows are NOT IID observations of one population; do not "
            "pool CP/FP/GP/RCP/RCP_FL into a single mean.",
            "Heuristic legs use only a few seeds — these counts characterise this "
            "bounded run, not a rigorous statistical distribution.",
            "Gaps are computed only on FEASIBLE solutions and only against a "
            "CERTIFIED baseline (exhaustive preferred, else certified MILP); an "
            "instance with neither certified baseline contributes no gap.",
        ],
    }
