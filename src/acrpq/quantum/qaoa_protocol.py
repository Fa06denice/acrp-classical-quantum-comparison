"""Rigorous, versioned QAOA execution protocol (Phase 4).

A QAOA result is only meaningful if you can say *exactly* how it was produced.
This module makes that explicit: :class:`QAOAProtocol` is a frozen, validated,
content-hashed description of every knob that defines a QAOA run — the ansatz
(``reps`` = p layers, mixer, parametrisation), the classical optimiser and its
budget, the **initial-angle strategy**, the number of shots, the backend and its
simulation method / transpilation ``optimization_level``, the **ISA/transpilation
target**, the single master ``seed`` (which pins all three independent RNG
consumers), and the deterministic decoding rule. Its ``protocol_hash`` /
``protocol_id`` identify the protocol the way ``benchmark.Protocol.config_hash``
identifies a benchmark run, so any QAOA result can be stamped with the exact
protocol that produced it and reproduced later.

Scope of Phase 4: Aer simulator ONLY. No IBM hardware is contacted. Real-hardware
ISA (a sparse coupling map + fixed basis gates + calibration) is a Phase-5
extension; the ``IsaSpec`` here honestly records the Aer target as a generic,
fully-connected 30-qubit target and marks it ``aer_generic`` so it is never
mistaken for a device.

Reproducibility contract (verified on Aer, same platform + library versions): a
protocol with a fixed ``seed`` reproduces ``best_bitstring``, ``best_energy``,
``optimal_params`` and the post-transpilation circuit metrics bit-for-bit. Library
versions (qiskit / qiskit-aer / qiskit-optimization) affect the RNG algorithm and
transpiler internals and therefore ride in a ReproInfo-style provenance block, not
in the protocol identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType

from .qaoa import validate_optimizer_options

# ---------------------------------------------------------------------------
# Versioned vocabularies (values are a CONTRACT — never rename, only add)
# ---------------------------------------------------------------------------
SCHEMA_QAOA_PROTOCOL = "acrpq-qaoa-protocol/1"

# The standard QAOA ansatz this codebase runs (qiskit_optimization QAOA with the
# default transverse-field X mixer and the standard alternating-operator form).
MIXER_STANDARD_X = "standard_x_transverse_field"
PARAM_STANDARD = "standard_alternating_operator"

# Decoding is a pure, deterministic function of the sampled bitstring (see
# decode.decode_assignment): min-energy selection by MinimumEigenOptimizer, then
# the deterministic one-hot repair (single->use, none->noop, multiple->cheapest).
DECODING_RULE = "min_eigen_optimizer_best+deterministic_one_hot_repair"


class InitStrategy(str, Enum):
    """How the initial QAOA angles are chosen."""

    SEEDED_RANDOM_UNIFORM = "seeded_random_uniform_2pi"  # qiskit default, seed-pinned
    ZEROS = "zeros"
    FIXED = "fixed"                                       # an explicit angle vector


class OptimizerName(str, Enum):
    COBYLA = "COBYLA"
    NELDER_MEAD = "NELDER_MEAD"


class DecompositionStrategy(str, Enum):
    """Whether QAOA solves the QUBO whole or per exact connected component."""

    NONE = "none"                                   # solver.solve() — matches Phase 3
    EXACT_CONNECTED_COMPONENTS = "exact_connected_components"  # solver.solve_decomposed()


# Aer simulation methods this protocol accepts (mirrors backends.AerMethod values;
# kept as a literal set so __post_init__ needs no qiskit import).
_VALID_SIM_METHODS = ("statevector", "matrix_product_state", "automatic")

# version_name: a short, safe, non-empty label (identity-bearing, see the class
# docstring) — bounded length + conservative charset so it cannot smuggle content.
_VERSION_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


# The three independent RNG consumers a single master seed pins (see the Phase-4
# investigation): Aer shot sampling, the transpiler's stochastic layout/routing,
# and — when init_strategy is seeded_random_uniform — the initial-angle draw.
SEED_PINS = ("aer_sampler_shots", "transpiler_layout_routing", "initial_angle_draw")


# ---------------------------------------------------------------------------
# ISA / transpilation target
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IsaSpec:
    """The transpilation target a protocol maps circuits onto, plus the knobs
    that make transpilation reproducible. For Aer this is a generic fully-connected
    target (no SWAP routing needed); a real-device sparse-coupling ISA is Phase 5."""

    isa_kind: str                # "aer_generic"
    target_n_qubits: int | None
    coupling: str                # "fully_connected" | "sparse"
    routing_required: bool
    basis_gates_sha256: str | None
    n_basis_gates: int | None
    optimization_level: int
    seed_transpiler: int
    note: str

    def to_json(self) -> dict:
        return {
            "isa_kind": self.isa_kind, "target_n_qubits": self.target_n_qubits,
            "coupling": self.coupling, "routing_required": self.routing_required,
            "basis_gates_sha256": self.basis_gates_sha256,
            "n_basis_gates": self.n_basis_gates,
            "optimization_level": self.optimization_level,
            "seed_transpiler": self.seed_transpiler, "note": self.note,
        }


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QAOAProtocol:
    """Immutable, content-hashed description of a QAOA execution protocol.

    The scientific identity is ``protocol_hash`` (and the short ``protocol_id``),
    which changes if ANY field below changes — INCLUDING ``version_name``. The name
    is therefore part of the identity, not free-form metadata: renaming a protocol
    deliberately mints a new versioned identity, and two protocols with identical
    execution knobs but different names are treated as distinct (the hash
    over-distinguishes rather than silently aliasing them).
    """

    version_name: str
    # -- ansatz --
    reps: int = 1
    mixer: str = MIXER_STANDARD_X
    parametrization: str = PARAM_STANDARD
    # -- classical optimiser --
    optimizer: str = OptimizerName.COBYLA.value
    maxiter: int = 100
    optimizer_options: tuple[tuple[str, float], ...] = ()   # future tol/rhobeg; kept hashable
    # -- initial angles (any Iterable of reals in — incl. a generator; stored as an
    #    immutable float tuple after __post_init__; required iff init_strategy=fixed) --
    init_strategy: str = InitStrategy.SEEDED_RANDOM_UNIFORM.value
    initial_point: Iterable[float] | None = None
    # -- sampling / backend --
    shots: int = 1024
    backend_mode: str = "aer_sim"
    sim_method: str = "statevector"
    optimization_level: int = 1
    max_memory_mb: int | None = None
    # -- decomposition strategy (part of the identity) --
    decomposition_strategy: str = DecompositionStrategy.NONE.value
    # -- master seed (pins the RNG consumers in SEED_PINS) --
    seed: int = 1234
    # -- decoding --
    decoding: str = DECODING_RULE
    schema: str = SCHEMA_QAOA_PROTOCOL

    def __post_init__(self) -> None:
        # -- fixed contract fields (fail-closed; no free-form content) --
        if self.schema != SCHEMA_QAOA_PROTOCOL:
            raise ValueError(f"schema must be {SCHEMA_QAOA_PROTOCOL!r}")
        if self.decoding != DECODING_RULE:
            raise ValueError(f"decoding must be {DECODING_RULE!r}")
        if not (isinstance(self.version_name, str) and _VERSION_NAME_RE.match(self.version_name)):
            raise ValueError("version_name must match [A-Za-z0-9._-]{1,64}")
        for name, value in (("mixer", self.mixer), ("parametrization", self.parametrization),
                            ("backend_mode", self.backend_mode), ("sim_method", self.sim_method),
                            ("optimizer", self.optimizer), ("init_strategy", self.init_strategy),
                            ("decomposition_strategy", self.decomposition_strategy)):
            if not (isinstance(value, str) and value == value.strip() and value):
                raise ValueError(f"{name} must be a normalised non-empty string")
        # -- numeric fields (bools refused everywhere an int/real is expected) --
        _int_ge(self.reps, 1, "reps")
        _int_ge(self.maxiter, 1, "maxiter")
        _int_ge(self.shots, 1, "shots")
        _int_ge(self.seed, 0, "seed")
        if not (isinstance(self.optimization_level, int)
                and not isinstance(self.optimization_level, bool)
                and 0 <= self.optimization_level <= 3):
            raise ValueError("optimization_level must be an int in [0, 3]")
        if self.max_memory_mb is not None:
            _int_ge(self.max_memory_mb, 1, "max_memory_mb")
        # -- enumerated vocabularies --
        if self.optimizer not in {o.value for o in OptimizerName}:
            raise ValueError(f"optimizer must be one of {[o.value for o in OptimizerName]}")
        if self.init_strategy not in {s.value for s in InitStrategy}:
            raise ValueError(f"init_strategy must be one of {[s.value for s in InitStrategy]}")
        if self.decomposition_strategy not in {s.value for s in DecompositionStrategy}:
            raise ValueError(
                f"decomposition_strategy must be one of {[s.value for s in DecompositionStrategy]}")
        if self.backend_mode != "aer_sim":
            raise ValueError("Phase 4 QAOAProtocol supports backend_mode='aer_sim' only")
        if self.sim_method not in _VALID_SIM_METHODS:
            raise ValueError(f"sim_method must be one of {list(_VALID_SIM_METHODS)}")
        if self.mixer != MIXER_STANDARD_X or self.parametrization != PARAM_STANDARD:
            raise ValueError("only the standard X mixer / alternating-operator form is supported")
        # -- initial angles: accept any Sequence of reals, but IMMEDIATELY
        #    materialise + defensively copy into an immutable tuple of floats. The
        #    caller's container is never retained, so mutating it afterwards can't
        #    change the protocol/hash/angles. str/bytes/bool/NaN/Inf and wrong
        #    length are refused. --
        if self.init_strategy == InitStrategy.FIXED.value:
            ip = self.initial_point
            if ip is None:
                raise ValueError("init_strategy='fixed' requires an initial_point")
            if isinstance(ip, (str, bytes, bytearray)):
                raise ValueError("initial_point must be a sequence of reals, not str/bytes")
            try:
                materialised = tuple(ip)  # consumes a generator; copies a list
            except TypeError:
                raise ValueError("initial_point must be an iterable of reals") from None
            if len(materialised) != 2 * self.reps:
                raise ValueError(f"initial_point must have length {2 * self.reps}")
            for v in materialised:
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    raise ValueError("initial_point must be finite real numbers (no bool/NaN/Inf)")
            # store a fresh immutable tuple of python floats — never the caller's object
            object.__setattr__(self, "initial_point", tuple(float(v) for v in materialised))
        elif self.initial_point is not None:
            raise ValueError("initial_point may only be set when init_strategy='fixed'")
        # -- optimizer_options: validate against the real per-optimizer allowlist,
        #    then re-store as a canonical sorted tuple of JSON-native pairs (never
        #    hash an unsupported/ignored option). Frozen dataclass -> setattr via
        #    object.__setattr__. --
        validated = validate_optimizer_options(self.optimizer, self.optimizer_options or ())
        object.__setattr__(self, "optimizer_options",
                           tuple(sorted((k, v) for k, v in validated.items())))

    # -- identity ----------------------------------------------------------
    def _payload(self) -> dict:
        return {
            "schema": self.schema, "version_name": self.version_name,
            "reps": self.reps, "mixer": self.mixer, "parametrization": self.parametrization,
            "optimizer": self.optimizer, "maxiter": self.maxiter,
            "optimizer_options": [list(p) for p in sorted(self.optimizer_options)],
            "init_strategy": self.init_strategy,
            "initial_point": list(self.initial_point) if self.initial_point is not None else None,
            "shots": self.shots, "backend_mode": self.backend_mode,
            "sim_method": self.sim_method, "optimization_level": self.optimization_level,
            "max_memory_mb": self.max_memory_mb,
            "decomposition_strategy": self.decomposition_strategy,
            "seed": self.seed, "seed_pins": list(SEED_PINS), "decoding": self.decoding,
        }

    @property
    def protocol_hash(self) -> str:
        blob = json.dumps(self._payload(), sort_keys=True, separators=(",", ":"),
                          allow_nan=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @property
    def protocol_id(self) -> str:
        return f"{self.schema}:{self.protocol_hash[:16]}"

    def descriptor(self) -> dict:
        d = self._payload()
        d["protocol_hash"] = self.protocol_hash
        d["protocol_id"] = self.protocol_id
        return d

    # -- build the concrete solver (the protocol is the single source of truth) --
    def resolved_initial_point(self) -> list[float] | None:
        if self.init_strategy == InitStrategy.FIXED.value:
            assert self.initial_point is not None
            return list(self.initial_point)
        if self.init_strategy == InitStrategy.ZEROS.value:
            return [0.0] * (2 * self.reps)
        return None  # SEEDED_RANDOM_UNIFORM -> qiskit draws it, seeded by self.seed

    def _require_aer(self) -> None:
        # Defense-in-depth for the no-IBM invariant: __post_init__ already refuses
        # a non-Aer backend_mode, but re-assert it at the point of USE so that even
        # a low-level object.__setattr__ bypass of the frozen dataclass can never
        # produce an IBM-mode backend/solver from a QAOAProtocol in this phase.
        if self.backend_mode != "aer_sim":
            raise ValueError(
                "QAOAProtocol is Aer-only in Phase 4; refusing to build a "
                f"'{self.backend_mode}' backend (no IBM path).")

    def build_backend(self):
        from .backends import AerMethod, BackendConfig, BackendMode

        self._require_aer()
        return BackendConfig(
            mode=BackendMode(self.backend_mode), shots=self.shots, seed=self.seed,
            optimization_level=self.optimization_level,
            sim_method=AerMethod(self.sim_method), max_memory_mb=self.max_memory_mb)

    def build_solver(self):
        from .qaoa import QAOASolver

        return QAOASolver(backend=self.build_backend(), reps=self.reps,
                          maxiter=self.maxiter, optimizer=self.optimizer,
                          initial_point=self.resolved_initial_point(),
                          optimizer_options=dict(self.optimizer_options))

    def aer_isa_spec(self) -> IsaSpec:
        """Introspect the Aer transpilation target this protocol maps onto."""
        from .backends import make_sampler

        _sampler, backend, _meta = make_sampler(self.build_backend())
        target = getattr(backend, "target", None)
        n_qubits = getattr(target, "num_qubits", None)
        cmap = None
        if target is not None:
            try:
                cmap = target.build_coupling_map()
            except Exception:
                cmap = None
        coupling = "fully_connected" if cmap is None else "sparse"
        ops = sorted(getattr(target, "operation_names", []) or [])
        basis_sha = hashlib.sha256(",".join(ops).encode("utf-8")).hexdigest() if ops else None
        return IsaSpec(
            isa_kind="aer_generic", target_n_qubits=n_qubits, coupling=coupling,
            routing_required=(coupling == "sparse"), basis_gates_sha256=basis_sha,
            n_basis_gates=len(ops) or None, optimization_level=self.optimization_level,
            seed_transpiler=self.seed,
            note=("generic Aer target (fully connected, no SWAP routing); a real "
                  "device ISA — sparse coupling + fixed basis + calibration — is a "
                  "Phase-5 extension via generate_preset_pass_manager(backend=<device>)"))


def _int_ge(value, lo: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < lo:
        raise ValueError(f"{name} must be an int >= {lo}")


def coerce_protocol(version_name: str) -> QAOAProtocol:
    """Look a registered protocol up by version_name (fail-closed)."""
    try:
        return PROTOCOL_REGISTRY[version_name]
    except KeyError:
        raise ValueError(
            f"unknown QAOA protocol '{version_name}'; "
            f"known: {sorted(PROTOCOL_REGISTRY)}") from None


# ---------------------------------------------------------------------------
# Registry of named, versioned protocols
# ---------------------------------------------------------------------------
# aer_default_p1_v1 is EXACTLY what the Phase-3 benchmark's qaoa_aer leg runs
# (reps=1, maxiter=100, COBYLA, seeded-random init, 1024 shots, Aer statevector,
# optimization_level=1, seed=1234) — so the benchmark's QAOA results are an
# instance of this versioned protocol.
_DEFAULT_P1 = QAOAProtocol(version_name="aer_default_p1_v1")   # decomposition_strategy=none
_DEFAULT_P2 = QAOAProtocol(version_name="aer_default_p2_v1", reps=2)
_P1_ZEROS = QAOAProtocol(version_name="aer_zeros_init_p1_v1",
                         init_strategy=InitStrategy.ZEROS.value)
# Same execution knobs as aer_default_p1_v1 but the exact-connected-components
# decomposition — a DISTINCT protocol identity (decomposition is part of the hash).
_P1_DECOMP = QAOAProtocol(
    version_name="aer_decomposed_p1_v1",
    decomposition_strategy=DecompositionStrategy.EXACT_CONNECTED_COMPONENTS.value)

# Private mutable dict; the public registry is a read-only view so a named
# protocol can never be accidentally replaced (e.g. during a run). NOTE: this
# protects the KEY BINDINGS; the QAOAProtocol values are frozen dataclasses, so
# their fields are immutable under normal use. As with any Python frozen
# dataclass, a deliberate object.__setattr__ bypass can still mutate a field in
# place — that is an explicit low-level override outside the immutability
# contract, not a supported operation.
_PROTOCOL_REGISTRY: dict[str, QAOAProtocol] = {
    p.version_name: p for p in (_DEFAULT_P1, _DEFAULT_P2, _P1_ZEROS, _P1_DECOMP)
}
PROTOCOL_REGISTRY = MappingProxyType(_PROTOCOL_REGISTRY)
DEFAULT_PROTOCOL = _DEFAULT_P1


# ---------------------------------------------------------------------------
# Protocol-stamped execution
# ---------------------------------------------------------------------------
# Volatile fields excluded from any content hash of a run record.
RUN_VOLATILE_FIELDS = ("wall_time_s", "qpu_time_s")

# Fields excluded from the artifact integrity hash: the volatile per-run timings,
# transient environment state, and the hash field itself.
_ARTIFACT_HASH_EXCLUDED = set(RUN_VOLATILE_FIELDS) | {
    "protocol_artifact_sha256", "protocol_artifact_sha256_note", "repository_dirty"}


def _canonical(obj, excluded: set[str]):
    if isinstance(obj, dict):
        return {k: _canonical(v, excluded) for k, v in obj.items() if k not in excluded}
    if isinstance(obj, list):
        return [_canonical(v, excluded) for v in obj]
    return obj


def canonical_artifact_sha256(artifact: dict) -> str:
    """Deterministic sha over the protocol artifact (volatile timings + transient
    env state + the sha field itself excluded)."""
    blob = json.dumps(_canonical(artifact, _ARTIFACT_HASH_EXCLUDED), sort_keys=True,
                      separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# The deterministic fields of a run record — the ones a fixed-seed protocol must
# reproduce bit-for-bit. Wall/qpu times and any per-component timing are excluded.
DETERMINISTIC_RUN_FIELDS = (
    "best_bitstring", "best_energy", "optimal_params", "objective_value",
    "decoded_feasible", "raw_onehot_valid", "repair_applied", "decoded_conflicts",
    "n_qubits", "circuit_depth", "circuit_size", "two_qubit_gates", "eval_count",
    "n_jobs", "decomposition",
)


def deterministic_subset(run: dict) -> dict:
    """The deterministic-only projection of a run record (timings dropped, incl.
    the volatile per-component timings inside ``decomposition``)."""
    out = {f: run.get(f) for f in DETERMINISTIC_RUN_FIELDS}
    return _canonical(out, set(RUN_VOLATILE_FIELDS))


def deterministic_run_sha256(run_or_subset: dict) -> str:
    """sha256 over the deterministic subset of a run — auditable: anyone can
    recompute it from the stored subset (full record or already-projected subset)
    without re-running QAOA."""
    subset = deterministic_subset(run_or_subset)
    blob = json.dumps(subset, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _decomposition_report(qubo, qres, strategy: str, decomposed: bool) -> dict:
    """Honest, auditable decomposition metadata for the run.

    ``applied`` is True ONLY when a genuine multi-component split was solved. Under
    the exact-connected-components strategy, a QUBO with a single connected
    component falls back to a whole solve (solve_decomposed does), and we report
    that truthfully (applied=False, n_components=1) instead of leaving the
    composition null. For the non-decomposed strategy everything is None.
    """
    if not decomposed:
        return {"strategy": strategy, "applied": False, "n_components": None,
                "largest_component_qubits": None, "component_qubit_sizes": None,
                "component_bits": None}
    meta = qres.backend_meta or {}
    meta_components = meta.get("components")
    if meta_components:  # a genuine (>1) component decomposition was solved
        sizes = [c.get("n_qubits") for c in meta_components]
        bits = [list(c.get("original_bits", [])) for c in meta_components]
        n_components = meta.get("n_components", len(meta_components))
        largest = meta.get("largest_component_qubits", max(sizes) if sizes else None)
        applied = True
    else:  # decomposed strategy requested, but the QUBO is a single component
        from .explain import split_qubo_components
        parts = split_qubo_components(qubo)
        bits = [list(component_bits) for component_bits, _sub in parts]
        sizes = [len(b) for b in bits]
        n_components = len(parts)
        largest = max(sizes) if sizes else None
        applied = n_components > 1
    return {"strategy": strategy, "applied": applied, "n_components": n_components,
            "largest_component_qubits": largest,
            "component_qubit_sizes": sizes, "component_bits": bits}


def run_protocol(qubo, protocol: QAOAProtocol) -> dict:
    """Run ``qubo`` under ``protocol`` on Aer and return a JSON-native record
    stamped with the protocol identity. The decomposition path is driven ENTIRELY
    by ``protocol.decomposition_strategy`` — there is no free override, so a run can
    never diverge from its recorded protocol. Raw one-hot validity is kept separate
    from decoded feasibility (a repaired bitstring is never counted raw-feasible),
    and the objective value is scored on the QUBO's OWN objective (never the
    always-quadratic decode().objective)."""
    from ..objectives import primary_objective
    from .decode import decode
    from .explain import bitstring_explanation

    solver = protocol.build_solver()
    decomposed = protocol.decomposition_strategy == DecompositionStrategy.EXACT_CONNECTED_COMPONENTS.value
    qres = solver.solve_decomposed(qubo) if decomposed else solver.solve(qubo)
    exp = bitstring_explanation(qubo, qres.best_bitstring)
    result = decode(qubo, qres.best_bitstring)
    sol = result.solution
    if sol is None:
        raise RuntimeError("decode() returned no solution")
    inst = qubo.discrete.instance
    obj_val = primary_objective(qubo.objective_id, sol.q, sol.theta, inst.w)
    decomposition = _decomposition_report(qubo, qres, protocol.decomposition_strategy, decomposed)
    return {
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.protocol_hash,
        "version_name": protocol.version_name,
        "objective_id": qubo.objective_id,
        "n_qubits": qres.n_qubits,
        "reps": protocol.reps,
        "best_bitstring": qres.best_bitstring,
        "best_energy": qres.best_energy,
        "optimal_params": [float(v) for v in (qres.optimal_params or [])],
        "eval_count": qres.eval_count,
        "n_jobs": qres.n_jobs,
        "circuit_depth": qres.circuit_depth,
        "circuit_size": qres.circuit_size,
        "two_qubit_gates": qres.two_qubit_gates,
        "sim_method": qres.sim_method,
        "raw_onehot_valid": exp["raw_onehot_valid"],
        "repair_applied": exp["repair_applied"],
        "decoded_feasible": exp["decoded_feasible"],
        "decoded_conflicts": exp["decoded_conflicts"],
        "objective_value": obj_val,
        "decomposition": decomposition,
        "wall_time_s": qres.wall_time_s,
        "qpu_time_s": qres.qpu_time_s,   # None on Aer
    }


def with_seed(protocol: QAOAProtocol, seed: int) -> QAOAProtocol:
    """A copy of ``protocol`` with a different master seed (new protocol identity)."""
    return replace(protocol, seed=seed)
