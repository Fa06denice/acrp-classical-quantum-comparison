"""Hardware-validation QAOA preparation, transpilation and decoding (Phase 5).

This module turns the *existing* scientific QUBO into a QAOA circuit with
explicitly-provided angles, transpiles it to a backend's ISA against an injectable
``fake_provider`` backend, produces an immutable, hash-verifiable prepared-run
artefact, and decodes/scores **synthetic** Sampler results. It PREPARES a job; it
never submits one.

Hard guarantees (enforced + tested):
  * imports no Qiskit at module load — every circuit/transpile/QPY call imports
    lazily inside the function that needs it;
  * never constructs an IBM runtime service, never reads a token, never calls the
    sampler primitive, the Phase-3 submit entry point, or a transition to
    ``submitting`` (asserted by a source-grep test);
  * a prepared artefact carries an explicit ``submittable`` flag: it is only
    submittable when it is bound to a *verified* Phase-4 decision whose selected,
    applicable, low-uncertainty backend matches the request; a synthetic/test
    artefact (fake backend, no decision) is explicitly marked NOT submittable;
  * cheap size budgets are checked before any expensive build/transpile; the depth
    / 2-qubit-gate budgets are checked immediately after transpilation.

It reuses the audited pure-Python stack (:mod:`acrpq.quantum.qubo`,
:mod:`acrpq.quantum.ising`, :mod:`acrpq.quantum.decode`, :mod:`acrpq.quantum.explain`)
and the Phase-2/3 hashing (:func:`acrpq.dashboard.persist.config_hash`) so a
prepared artefact lines up byte-for-byte with what Phase 3 submission expects. It
does NOT modify the classical solvers or regenerate any scientific result.

Conventions (documented, part of the contract):
  * QUBO energy ``H(x) = c + sum_b a_b x_b + sum_{b<c} q_{bc} x_b x_c`` (see
    :mod:`acrpq.quantum.qubo`). Ising via ``x = (1-z)/2`` (see
    :mod:`acrpq.quantum.ising`); the QAOA cost layer implements
    ``exp(-i*gamma*(sum_b h_b Z_b + sum_{b<c} J_bc Z_b Z_c))`` (offset is a global
    phase and is dropped), the mixer implements ``exp(-i*beta*sum_b X_b)``.
  * Bit order: qubit ``b`` is measured into classical bit ``b``. A raw Qiskit
    count string is little-endian (leftmost char = highest qubit), so it is
    reversed to the QUBO order (``bits[b]`` = value of bit ``b``) before decoding.
    ``decode_samples`` takes an explicit ``bit_order`` and normalises accordingly.
  * Angles are radians, finite, and bounded to ``|angle| <= 4*pi``. The angle
    strategy is recorded verbatim and is NEVER presented as "optimal" without
    proof.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..objectives import DEFAULT_OBJECTIVE, coerce_objective_id, primary_objective
from ..quantum.decode import decode as _decode_result
from ..quantum.decode import decode_assignment
from ..quantum.explain import energy_breakdown
from ..quantum.ising import qubo_to_ising
from ..quantum.qubo import ManeuverQUBO, assert_objective_consistent
from .backend_scoring import (
    _canonical,
    _clean_str,
    _json_native,
    _req_finite,
    _req_int,
    verify_decision_hash,
)
from .backend_scoring import workload_commitment as _workload_commitment
from .persist import config_hash

HW_VALIDATION_SCHEMA_VERSION = "acrpq-hw-validation/1"

# Explicit cost limits. Logical-qubit capacity is deliberately NOT hard-coded:
# the selected backend's ``num_qubits`` is the source of truth, followed by the
# measured post-transpilation depth/2Q budgets below.
MAX_QUBO_TERMS = 4_000
MAX_QAOA_REPS = 8
MAX_ANGLE = 4.0 * math.pi
MAX_DECODED_CANDIDATES = 4_096
# Hard cost caps for a hardware-VALIDATION run (a real full study is a later,
# separately-confirmed phase). These bound the two billing-relevant fields
# regardless of what Phase-4 evaluated, so a client can never request a huge budget.
MAX_HW_MAX_JOBS = 5
MAX_HW_MAX_QUANTUM_SECONDS = 600.0

_SHA_HEX = 64
_MODE = "hardware_validation"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class HardwareValidationError(ValueError):
    """Base error for hardware-validation preparation/decoding."""


class BudgetExceeded(HardwareValidationError):
    """A depth / gate / size budget was exceeded."""


class Phase4CoherenceError(HardwareValidationError):
    """The request is not coherently backed by a verified Phase-4 decision."""


class BackendMismatch(Phase4CoherenceError):
    """The requested backend does not match the Phase-4 decision's selection."""


class NonReproducibleTranspilation(HardwareValidationError):
    """The preset pass manager did not produce a reproducible ISA (fails closed)."""


def _sha256_hex(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _norm_sha(v: Any, name: str) -> str:
    s = _clean_str(v, name)
    body = s[len("sha256:"):] if s.startswith("sha256:") else s
    if len(body) != _SHA_HEX or any(c not in "0123456789abcdef" for c in body):
        raise HardwareValidationError(f"{name} must be a 'sha256:<64 hex>' digest")
    return "sha256:" + body


def _bounded_angle(v: Any, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise HardwareValidationError(f"{name} must be a finite, non-bool number")
    fv = float(v)
    if abs(fv) > MAX_ANGLE:
        raise HardwareValidationError(f"{name} out of bounds; |angle| must be <= 4*pi")
    return fv


# --------------------------------------------------------------------------- #
# Strict input contract
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HardwareValidationRequest:
    instance_id: str
    qubo_sha256: str
    n_binary_vars: int
    variable_meaning: tuple[str, ...]           # bit b -> business label
    qaoa_reps: int
    gammas: tuple[float, ...]
    betas: tuple[float, ...]
    shots: int
    transpiler_seed: int
    backend_requested: str
    max_jobs: int
    max_quantum_seconds: float
    n_theta: int
    objective_id: str = DEFAULT_OBJECTIVE.value
    n_q: int = 1
    optimization_level: int = 1
    angle_strategy: str = "provided"
    bit_order: str = "qiskit"                    # how RAW results will be interpreted
    max_depth: int | None = None
    max_two_qubit_gates: int | None = None
    decision_ts: float = 0.0
    mode: str = _MODE

    def __post_init__(self) -> None:
        if self.mode != _MODE:
            raise HardwareValidationError(f"mode must be {_MODE!r}")
        object.__setattr__(self, "instance_id", _clean_str(self.instance_id, "instance_id"))
        object.__setattr__(self, "backend_requested", _clean_str(self.backend_requested, "backend_requested"))
        object.__setattr__(self, "qubo_sha256", _norm_sha(self.qubo_sha256, "qubo_sha256"))
        n = _req_int(self.n_binary_vars, "n_binary_vars", lo=1)
        if not isinstance(self.variable_meaning, tuple):
            raise HardwareValidationError("variable_meaning must be a tuple")
        if len(self.variable_meaning) != n:
            raise HardwareValidationError("variable_meaning must have one label per binary var")
        object.__setattr__(self, "variable_meaning",
                           tuple(_clean_str(m, f"variable_meaning[{i}]")
                                 for i, m in enumerate(self.variable_meaning)))
        p = _req_int(self.qaoa_reps, "qaoa_reps", lo=1)
        if p > MAX_QAOA_REPS:
            raise HardwareValidationError(f"qaoa_reps {p} > MAX_QAOA_REPS {MAX_QAOA_REPS}")
        for angles, label in ((self.gammas, "gammas"), (self.betas, "betas")):
            if not isinstance(angles, tuple):
                raise HardwareValidationError(f"{label} must be a tuple")
            if len(angles) != p:
                raise HardwareValidationError(f"{label} must contain exactly qaoa_reps={p} values")
        object.__setattr__(self, "gammas",
                           tuple(_bounded_angle(g, f"gammas[{i}]") for i, g in enumerate(self.gammas)))
        object.__setattr__(self, "betas",
                           tuple(_bounded_angle(b, f"betas[{i}]") for i, b in enumerate(self.betas)))
        _req_int(self.shots, "shots", lo=1)
        _req_int(self.transpiler_seed, "transpiler_seed", lo=0)
        _req_int(self.max_jobs, "max_jobs", lo=1)
        if self.max_jobs > MAX_HW_MAX_JOBS:
            raise HardwareValidationError(
                f"max_jobs {self.max_jobs} exceeds the hardware-validation cap {MAX_HW_MAX_JOBS}")
        _req_int(self.n_theta, "n_theta", lo=1)
        _req_int(self.n_q, "n_q", lo=1)
        try:
            objective_id = coerce_objective_id(self.objective_id).value
        except (TypeError, ValueError) as exc:
            raise HardwareValidationError(str(exc)) from exc
        object.__setattr__(self, "objective_id", objective_id)
        lvl = _req_int(self.optimization_level, "optimization_level", lo=0)
        if lvl > 3:
            raise HardwareValidationError("optimization_level must be 0..3")
        mqs = _req_finite(self.max_quantum_seconds, "max_quantum_seconds")
        if mqs <= 0:
            raise HardwareValidationError("max_quantum_seconds must be > 0")
        if mqs > MAX_HW_MAX_QUANTUM_SECONDS:
            raise HardwareValidationError(
                f"max_quantum_seconds {mqs} exceeds the hardware-validation cap "
                f"{MAX_HW_MAX_QUANTUM_SECONDS}")
        _req_finite(self.decision_ts, "decision_ts", lo=0.0)
        _clean_str(self.angle_strategy, "angle_strategy")
        if self.bit_order not in ("qiskit", "qubo"):
            raise HardwareValidationError("bit_order must be 'qiskit' or 'qubo'")
        if self.max_depth is not None:
            _req_int(self.max_depth, "max_depth", lo=1)
        if self.max_two_qubit_gates is not None:
            _req_int(self.max_two_qubit_gates, "max_two_qubit_gates", lo=0)

    def to_submission_params(self) -> dict[str, Any]:
        """The exact Phase-3 hardware-mode params this request implies."""
        return {
            "mode": _MODE,
            "instance": self.instance_id,
            "n_theta": self.n_theta,
            "n_q": self.n_q,
            "objective_id": self.objective_id,
            "shots": self.shots,
            "transpiler_seed": self.transpiler_seed,
            "max_jobs": self.max_jobs,
            "max_quantum_seconds": self.max_quantum_seconds,
            "qubo_sha256": self.qubo_sha256,
            "backend_requested": self.backend_requested,
        }


# --------------------------------------------------------------------------- #
# Canonical QUBO hash (auditable, order-independent)
# --------------------------------------------------------------------------- #
def canonical_qubo_hash(qubo: ManeuverQUBO) -> str:
    """Order-independent SHA-256 of the QUBO's mathematical content + objective.

    ``objective_id`` is part of the hashed payload so two QUBOs that happen to
    share coefficients but optimise different objectives never collide — the
    objective identity travels with the canonical hash.

    This function is intentionally PURE: it hashes whatever ``objective_id`` the
    QUBO carries (so hash-binding can be exercised on a deliberately spoofed
    QUBO). It is NOT the validation boundary. The record/persistence boundary is
    ``_validate_qubo`` (below), which calls ``assert_objective_consistent`` and
    runs inside every prepare path before an artefact is produced. When
    ``objective_id`` is later threaded through the ``/api`` request layer
    (Phase 8), the prepare handlers that hash a QUBO before that validation must
    also assert consistency at the hash site.
    """
    payload = {
        "objective_id": qubo.objective_id,
        "n": qubo.n_qubits,
        "constant": qubo.constant,
        "linear": {str(b): qubo.linear[b] for b in sorted(qubo.linear)},
        "quadratic": {f"{a},{c}": v for (a, c), v in
                      sorted(qubo.quadratic.items(), key=lambda kv: kv[0])},
    }
    return _sha256_hex(_canonical(payload))


def _validate_qubo(qubo: ManeuverQUBO, n_expected: int) -> None:
    n = qubo.n_qubits
    if n != n_expected:
        raise HardwareValidationError(f"QUBO has {n} qubits but request declares {n_expected}")
    # reject a QUBO whose objective_id was swapped without rebuilding coefficients
    # (spoof) before it can be hashed/prepared for a scientific record
    try:
        assert_objective_consistent(qubo)
    except ValueError as exc:
        raise HardwareValidationError(str(exc)) from exc
    if not math.isfinite(qubo.constant):
        raise HardwareValidationError("QUBO constant must be finite")
    for b, a in qubo.linear.items():
        if not 0 <= b < n:
            raise HardwareValidationError(f"linear bit index {b} out of range 0..{n - 1}")
        if not math.isfinite(a):
            raise HardwareValidationError(f"linear coefficient for bit {b} is not finite")
    for (a, c), q in qubo.quadratic.items():
        if not (0 <= a < n and 0 <= c < n):
            raise HardwareValidationError(f"quadratic index ({a},{c}) out of range")
        if a >= c:
            raise HardwareValidationError(f"quadratic key ({a},{c}) must have a < c (no double counting)")
        if not math.isfinite(q):
            raise HardwareValidationError(f"quadratic coefficient ({a},{c}) is not finite")
    if len(qubo.linear) + len(qubo.quadratic) > MAX_QUBO_TERMS:
        raise BudgetExceeded(f"QUBO has too many terms (> {MAX_QUBO_TERMS})")


# --------------------------------------------------------------------------- #
# QAOA circuit construction (lazy qiskit, explicit conventions)
# --------------------------------------------------------------------------- #
def build_parameterized_qaoa(qubo: ManeuverQUBO, reps: int):
    """Build a parameterised QAOA circuit + its (gammas, betas) ParameterVectors.

    Cost layer: RZ(2*gamma*h_b) on each spin, RZZ(2*gamma*J_bc) on each coupled
    pair (offset dropped as a global phase). Mixer: RX(2*beta) on each qubit.
    Initial state: uniform superposition. Measurement: qubit b -> clbit b.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    ising = qubo_to_ising(qubo)
    n = qubo.n_qubits
    gammas = ParameterVector("g", reps)
    betas = ParameterVector("b", reps)
    # a STABLE circuit name — QuantumCircuit otherwise auto-names each instance
    # ("circuit-<global counter>"), which would make the QPY bytes (and thus the
    # artefact hash) non-deterministic across identical builds.
    qc = QuantumCircuit(n, n, name="acrpq_qaoa")
    qc.h(range(n))
    for layer in range(reps):
        g = gammas[layer]
        for b, h in ising.h.items():
            if h:
                qc.rz(2.0 * g * h, b)
        for (a, c), j in ising.J.items():
            if j:
                qc.rzz(2.0 * g * j, a, c)
        qc.barrier()
        beta = betas[layer]
        for b in range(n):
            qc.rx(2.0 * beta, b)
        qc.barrier()
    qc.measure(range(n), range(n))
    return qc, gammas, betas


def build_bound_qaoa(qubo: ManeuverQUBO, request: HardwareValidationRequest):
    """Return a fully parameter-bound, measured QAOA circuit (no free params)."""
    qc, gammas, betas = build_parameterized_qaoa(qubo, request.qaoa_reps)
    binding = {gammas[i]: request.gammas[i] for i in range(request.qaoa_reps)}
    binding.update({betas[i]: request.betas[i] for i in range(request.qaoa_reps)})
    bound = qc.assign_parameters(binding, inplace=False)
    if bound.num_parameters != 0:
        raise HardwareValidationError("bound circuit still has free parameters")
    return bound


# --------------------------------------------------------------------------- #
# Injectable transpilation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TranspileReport:
    backend_name: str
    backend_num_qubits: int
    backend_is_fake: bool
    optimization_level: int
    seed_transpiler: int
    depth_before: int
    depth_after: int
    size_after: int
    two_qubit_gates_after: int
    swap_count: int
    basis_gates: tuple[str, ...]
    non_basis_ops: tuple[str, ...]
    initial_layout: tuple[int, ...]
    transpile_seconds: float                 # wall-clock; NOT part of the artefact hash

    def deterministic_dict(self) -> dict[str, Any]:
        """Reproducible transpile fields (wall-clock deliberately excluded)."""
        d = asdict(self)
        d.pop("transpile_seconds")
        d["basis_gates"] = list(self.basis_gates)
        d["non_basis_ops"] = list(self.non_basis_ops)
        d["initial_layout"] = list(self.initial_layout)
        return d


@runtime_checkable
class TranspilerBackend(Protocol):
    """A minimal backend contract; a ``fake_provider`` backend satisfies it."""

    @property
    def num_qubits(self) -> int: ...

    def __getattr__(self, name: str) -> Any: ...


def _is_fake_backend(backend: Any) -> bool:
    """Robustly detect a fake/simulated backend.

    Primary signal is ``isinstance`` against qiskit's ``FakeBackendV2`` base, so a
    subclass of a ``fake_provider`` backend under an innocuous name is STILL
    detected as fake (a name-only heuristic could be dodged). The name/module
    heuristic is only a fallback when the base class cannot be imported.
    """
    try:
        from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2
        if isinstance(backend, FakeBackendV2):
            return True
    except Exception:  # noqa: BLE001 - fall back to the name heuristic
        pass
    mod = (type(backend).__module__ or "").lower()
    return "fake" in mod or "fake" in type(backend).__name__.lower()


def transpile_to_isa(circuit, backend, *, seed: int, optimization_level: int) -> tuple[Any, TranspileReport]:
    """Transpile ``circuit`` to ``backend``'s ISA (lazy qiskit)."""
    import time

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    backend_name = getattr(backend, "name", type(backend).__name__)
    if callable(backend_name):
        backend_name = backend_name()
    num_qubits = int(backend.num_qubits)
    if circuit.num_qubits > num_qubits:
        raise BudgetExceeded(
            f"circuit needs {circuit.num_qubits} qubits > backend {backend_name} has {num_qubits}")
    depth_before = int(circuit.depth())

    t0 = time.perf_counter()
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level, backend=backend, seed_transpiler=seed)
    isa = pm.run(circuit)
    dt = time.perf_counter() - t0

    ops = isa.count_ops()
    swap_count = int(ops.get("swap", 0))
    two_q = sum(1 for inst in isa.data if inst.operation.num_qubits == 2)
    try:
        target_ops = set(backend.target.operation_names)
    except Exception:  # noqa: BLE001 - target introspection is best-effort
        target_ops = set()
    non_basis = tuple(sorted(
        name for name in ops
        if target_ops and name not in target_ops and name not in ("barrier", "measure")))
    layout = _extract_layout(isa, circuit.num_qubits)
    report = TranspileReport(
        backend_name=str(backend_name), backend_num_qubits=num_qubits,
        backend_is_fake=_is_fake_backend(backend), optimization_level=int(optimization_level),
        seed_transpiler=int(seed), depth_before=depth_before, depth_after=int(isa.depth()),
        size_after=int(isa.size()), two_qubit_gates_after=int(two_q), swap_count=swap_count,
        basis_gates=tuple(sorted(target_ops)), non_basis_ops=non_basis,
        initial_layout=layout, transpile_seconds=dt,
    )
    return isa, report


def _extract_layout(isa, n_logical: int) -> tuple[int, ...]:
    layout = getattr(isa, "layout", None)
    if layout is None:
        return ()
    try:
        phys = layout.final_index_layout(filter_ancillas=True)
        return tuple(int(x) for x in phys[:n_logical])
    except Exception:  # noqa: BLE001 - layout shape varies; best-effort only
        return ()


def _circuit_fingerprint(circuit) -> tuple[str, str]:
    """(fingerprint_sha256, qiskit_version).

    A canonical, deterministic hash of the circuit's gate list — gate name, the
    logical qubit/clbit indices it acts on, and its rounded numeric parameters.
    QPY bytes are NOT byte-stable for transpiled circuits (auto-generated names,
    metadata), so a gate-list fingerprint is used instead. An ISA fingerprint is
    only comparable within the same qiskit version + backend (transpilation is
    version-dependent); the logical fingerprint is fully version-stable.
    """
    import qiskit

    def _params(op) -> list[float | str]:
        out: list[float | str] = []
        for p in op.params:
            try:
                out.append(round(float(p), 12))
            except (TypeError, ValueError):
                out.append(str(p))
        return out

    instrs = []
    for inst in circuit.data:
        qubits = [int(circuit.find_bit(q).index) for q in inst.qubits]
        clbits = [int(circuit.find_bit(c).index) for c in inst.clbits]
        instrs.append([inst.operation.name, qubits, clbits, _params(inst.operation)])
    payload = {"n_qubits": int(circuit.num_qubits), "n_clbits": int(circuit.num_clbits),
               "global_phase": round(float(circuit.global_phase), 12), "instructions": instrs}
    return _sha256_hex(_canonical(payload)), qiskit.__version__


# --------------------------------------------------------------------------- #
# Phase-4 coherence
# --------------------------------------------------------------------------- #
def _backend_name(backend: Any) -> str:
    name = getattr(backend, "name", type(backend).__name__)
    return name() if callable(name) else str(name)


def _selected_snapshot_synthetic(decision: dict[str, Any], selected: str) -> bool:
    snap = next((s for s in decision.get("snapshots", []) if s.get("name") == selected), None)
    return bool(snap and snap.get("synthetic"))


def _check_phase4(
    request: HardwareValidationRequest,
    decision: dict[str, Any] | None,
    workload: Any,
    *,
    allow_synthetic: bool,
    backend_is_fake: bool,
    decision_max_age_s: float,
) -> dict[str, Any]:
    """Bind a verified Phase-4 decision to this workload, or refuse.

    ``submittable`` is only ever True when a verified decision selects an
    applicable, low-uncertainty backend whose identity matches, the decision was
    made for exactly this workload, AND neither the backend nor the selected
    snapshot is synthetic/fake. A fake backend (or synthetic snapshot) forces
    ``submittable = False`` with no override.
    """
    if decision is None:
        if not allow_synthetic:
            raise Phase4CoherenceError(
                "no Phase-4 decision supplied and synthetic preparation was not allowed")
        if not backend_is_fake:
            raise Phase4CoherenceError("synthetic preparation requires a fake backend")
        return {"submittable": False, "source": "synthetic_test", "decision_sha256": None,
                "downgrade": "synthetic_no_decision"}

    if not verify_decision_hash(decision):
        raise Phase4CoherenceError("Phase-4 decision hash does not verify")
    selected = decision.get("selected")
    if selected is None:
        raise Phase4CoherenceError("Phase-4 decision selected no backend")
    if selected != request.backend_requested:
        raise BackendMismatch(
            f"requested backend {request.backend_requested!r} != decision selection {selected!r}")
    row = next((r for r in decision.get("ranking", []) if r.get("name") == selected), None)
    if row is None or not row.get("applicable"):
        raise Phase4CoherenceError("selected backend is not an applicable ranked candidate")
    block = decision.get("uncertainty_block")
    if not isinstance(block, (int, float)) or row.get("uncertainty", 1.0) > block:
        raise Phase4CoherenceError("selected backend uncertainty exceeds the decision block")

    # --- staleness policy (explicit) ---
    dts = decision.get("decision_ts")
    if isinstance(dts, (int, float)):
        age = request.decision_ts - dts
        if age < 0:
            raise Phase4CoherenceError("decision timestamp is in the future relative to the request")
        if age > decision_max_age_s:
            raise Phase4CoherenceError(
                f"Phase-4 decision is stale ({age}s > {decision_max_age_s}s)")

    # --- workload binding: the decision must have been made for THIS workload ---
    if workload is None:
        raise Phase4CoherenceError("a WorkloadRequirements must be supplied to bind a decision")
    if _workload_commitment(workload) != decision.get("workload_sha256"):
        raise Phase4CoherenceError(
            "workload does not match the workload the Phase-4 decision was made for")
    # the Phase-5 request must not be STRONGER than what Phase-4 evaluated
    if request.n_binary_vars != int(getattr(workload, "logical_qubits", -1)):
        raise Phase4CoherenceError(
            "request qubit count differs from the evaluated workload logical_qubits")
    if request.shots > int(getattr(workload, "shots", -1)):
        raise Phase4CoherenceError("request shots exceed the evaluated workload shots")
    # bind the billing-relevant time budget to what Phase-4 evaluated (when set)
    wl_qs = getattr(workload, "max_quantum_seconds", None)
    if wl_qs is not None and request.max_quantum_seconds > float(wl_qs):
        raise Phase4CoherenceError(
            "request max_quantum_seconds exceeds the evaluated workload budget")

    # --- submittable downgrade: fake/synthetic never yields a payable payload ---
    downgrade = None
    submittable = True
    if backend_is_fake:
        submittable, downgrade = False, "fake_backend_never_submittable"
    elif _selected_snapshot_synthetic(decision, selected):
        submittable, downgrade = False, "synthetic_snapshot_never_submittable"
    return {"submittable": submittable, "source": "phase4_decision",
            "decision_sha256": decision.get("decision_sha256"), "downgrade": downgrade}


# --------------------------------------------------------------------------- #
# Prepared artefact
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PreparedHardwareRun:
    schema: str
    request: dict[str, Any]
    qubo_sha256: str
    config_hash: str
    backend: str
    backend_synthetic: dict[str, Any]
    decision_sha256: str | None
    submittable: bool
    logical_circuit: dict[str, Any]
    isa_circuit: dict[str, Any]
    transpilation: dict[str, Any]
    variable_mapping: dict[str, Any]
    bit_order: str
    decision_ts: float
    provenance: dict[str, Any]
    warnings: tuple[str, ...]
    artefact_sha256: str = ""

    def to_manifest(self) -> dict[str, Any]:
        """The JSON-native manifest that the artefact hash is computed over."""
        d = asdict(self)
        d.pop("artefact_sha256")
        return _json_native(d)


def _hash_manifest(manifest: dict[str, Any]) -> str:
    return _sha256_hex(_canonical(manifest))


def verify_artefact_hash(prepared: PreparedHardwareRun) -> bool:
    """Recompute the artefact hash over everything except ``artefact_sha256``."""
    return prepared.artefact_sha256 == _hash_manifest(prepared.to_manifest())


DEFAULT_DECISION_MAX_AGE_S = 24 * 3600.0


def prepare_hardware_run(
    request: HardwareValidationRequest,
    qubo: ManeuverQUBO,
    backend: Any,
    *,
    decision: dict[str, Any] | None = None,
    workload: Any = None,
    allow_synthetic: bool = False,
    provenance: dict[str, Any] | None = None,
    verify_reproducible: bool = True,
    decision_max_age_s: float = DEFAULT_DECISION_MAX_AGE_S,
) -> PreparedHardwareRun:
    """Validate, build, transpile and freeze a hardware-validation artefact.

    See :func:`prepare_hardware_bundle` for the full contract; this returns only
    the JSON-native artefact (the circuits are dropped).
    """
    prepared, _logical, _isa = _prepare_impl(
        request, qubo, backend, decision=decision, workload=workload,
        allow_synthetic=allow_synthetic, provenance=provenance,
        verify_reproducible=verify_reproducible, decision_max_age_s=decision_max_age_s)
    return prepared


def _prepare_impl(
    request: HardwareValidationRequest,
    qubo: ManeuverQUBO,
    backend: Any,
    *,
    decision: dict[str, Any] | None,
    workload: Any,
    allow_synthetic: bool,
    provenance: dict[str, Any] | None,
    verify_reproducible: bool,
    decision_max_age_s: float,
) -> tuple[PreparedHardwareRun, Any, Any]:
    # ---- cheap validation first (before any expensive build/transpile) ----
    _validate_qubo(qubo, request.n_binary_vars)
    got = canonical_qubo_hash(qubo)
    if got != request.qubo_sha256:
        raise HardwareValidationError("QUBO hash does not match request.qubo_sha256")
    # backend identity: the real backend name must equal the requested name
    backend_name = _backend_name(backend)
    if backend_name != request.backend_requested:
        raise BackendMismatch(
            f"transpiler backend {backend_name!r} != request.backend_requested "
            f"{request.backend_requested!r}")
    backend_is_fake = _is_fake_backend(backend)
    phase4 = _check_phase4(request, decision, workload, allow_synthetic=allow_synthetic,
                           backend_is_fake=backend_is_fake, decision_max_age_s=decision_max_age_s)
    if int(backend.num_qubits) < request.n_binary_vars:
        raise BudgetExceeded("backend has fewer qubits than the workload needs")

    # ---- build + transpile ----
    logical = build_bound_qaoa(qubo, request)
    isa, report = transpile_to_isa(
        logical, backend, seed=request.transpiler_seed,
        optimization_level=request.optimization_level)

    # ---- reproducibility: refuse an artefact whose ISA is not reproducible ----
    isa_sha, qk_version = _circuit_fingerprint(isa)
    if verify_reproducible:
        isa_again, _ = transpile_to_isa(
            logical, backend, seed=request.transpiler_seed,
            optimization_level=request.optimization_level)
        if _circuit_fingerprint(isa_again)[0] != isa_sha:
            raise NonReproducibleTranspilation(
                f"transpilation is not reproducible for backend {report.backend_name!r} at "
                f"optimization_level {request.optimization_level}; pin a deterministic "
                f"optimization_level (0 or 3) or a deterministic backend")

    # ---- post-transpilation budgets ----
    warnings: list[str] = []
    if report.non_basis_ops:
        warnings.append("non_basis_ops_present:" + ",".join(report.non_basis_ops))
    if request.max_depth is not None and report.depth_after > request.max_depth:
        raise BudgetExceeded(f"ISA depth {report.depth_after} > budget {request.max_depth}")
    if (request.max_two_qubit_gates is not None
            and report.two_qubit_gates_after > request.max_two_qubit_gates):
        raise BudgetExceeded(
            f"ISA 2Q gates {report.two_qubit_gates_after} > budget {request.max_two_qubit_gates}")
    if phase4.get("downgrade"):
        warnings.append(phase4["downgrade"])
    if not phase4["submittable"]:
        warnings.append("artefact_not_submittable")

    # ---- fingerprint the logical circuit (ISA fingerprint already computed) ----
    logical_sha, _ = _circuit_fingerprint(logical)

    params = request.to_submission_params()
    chash = config_hash(params)

    manifest: dict[str, Any] = {
        "schema": HW_VALIDATION_SCHEMA_VERSION,
        "request": {
            "instance_id": request.instance_id, "n_binary_vars": request.n_binary_vars,
            "qaoa_reps": request.qaoa_reps, "gammas": list(request.gammas),
            "betas": list(request.betas), "shots": request.shots,
            "transpiler_seed": request.transpiler_seed, "backend_requested": request.backend_requested,
            "max_jobs": request.max_jobs, "max_quantum_seconds": request.max_quantum_seconds,
            "n_theta": request.n_theta, "n_q": request.n_q,
            "objective_id": request.objective_id,
            "optimization_level": request.optimization_level,
            "angle_strategy": request.angle_strategy, "bit_order": request.bit_order,
            "max_depth": request.max_depth, "max_two_qubit_gates": request.max_two_qubit_gates,
            "mode": request.mode,
        },
        "qubo_sha256": request.qubo_sha256,
        "config_hash": chash,
        "backend": report.backend_name,
        "backend_synthetic": {
            "is_fake": report.backend_is_fake, "num_qubits": report.backend_num_qubits},
        "decision_sha256": phase4["decision_sha256"],
        "submittable": phase4["submittable"],
        "logical_circuit": {"fingerprint_sha256": logical_sha, "qiskit_version": qk_version,
                            "n_qubits": logical.num_qubits, "depth": int(logical.depth())},
        "isa_circuit": {"fingerprint_sha256": isa_sha, "qiskit_version": qk_version},
        "transpilation": report.deterministic_dict(),
        "variable_mapping": {str(b): request.variable_meaning[b]
                             for b in range(request.n_binary_vars)},
        "bit_order": request.bit_order,
        "decision_ts": request.decision_ts,
        "provenance": _json_native(provenance) if provenance is not None else None,
        "warnings": list(warnings),
    }
    artefact_sha = _hash_manifest(manifest)
    prepared = PreparedHardwareRun(
        schema=manifest["schema"], request=manifest["request"], qubo_sha256=manifest["qubo_sha256"],
        config_hash=chash, backend=manifest["backend"],
        backend_synthetic=manifest["backend_synthetic"], decision_sha256=manifest["decision_sha256"],
        submittable=manifest["submittable"], logical_circuit=manifest["logical_circuit"],
        isa_circuit=manifest["isa_circuit"], transpilation=manifest["transpilation"],
        variable_mapping=manifest["variable_mapping"], bit_order=manifest["bit_order"],
        decision_ts=manifest["decision_ts"], provenance=manifest["provenance"],
        warnings=tuple(warnings), artefact_sha256=artefact_sha,
    )
    return prepared, logical, isa


@dataclass(frozen=True)
class PreparedCircuitBundle:
    """An artefact bound to its actual logical + ISA circuits (never serialised)."""

    prepared: PreparedHardwareRun
    logical_circuit: Any
    isa_circuit: Any


def prepare_hardware_bundle(
    request: HardwareValidationRequest,
    qubo: ManeuverQUBO,
    backend: Any,
    *,
    decision: dict[str, Any] | None = None,
    workload: Any = None,
    allow_synthetic: bool = False,
    provenance: dict[str, Any] | None = None,
    verify_reproducible: bool = True,
    decision_max_age_s: float = DEFAULT_DECISION_MAX_AGE_S,
) -> PreparedCircuitBundle:
    """Validate, build, transpile and freeze a bundle (artefact + real circuits).

    When ``decision`` is supplied it must be a *verified* Phase-4 record made for
    exactly this ``workload`` whose selected, applicable, low-uncertainty,
    non-stale backend matches ``request.backend_requested`` and the real
    ``backend``; a fake/synthetic backend or snapshot forces ``submittable=False``.
    When ``decision`` is ``None`` the bundle is only produced with
    ``allow_synthetic`` on a fake backend and is marked NOT submittable. Nothing
    is ever submitted.
    """
    prepared, logical, isa = _prepare_impl(
        request, qubo, backend, decision=decision, workload=workload,
        allow_synthetic=allow_synthetic, provenance=provenance,
        verify_reproducible=verify_reproducible, decision_max_age_s=decision_max_age_s)
    return PreparedCircuitBundle(prepared=prepared, logical_circuit=logical, isa_circuit=isa)


# --------------------------------------------------------------------------- #
# Phase-3 hand-off (pure — never submits)
# --------------------------------------------------------------------------- #
def verify_bundle_integrity(bundle: PreparedCircuitBundle) -> list[str]:
    """Structural checks binding an artefact to its ACTUAL circuits (empty == ok).

    Does NOT check ``submittable`` (that is a separate policy gate) — it verifies
    the artefact hash, that the supplied ISA/logical circuits re-fingerprint to the
    artefact, the Qiskit version, full binding, expected measurements and backend
    identity. This is what stops an arbitrary ``object()`` or a swapped circuit.
    """
    p = bundle.prepared
    problems: list[str] = []
    if not verify_artefact_hash(p):
        problems.append("artefact_hash_invalid")
    isa, logical = bundle.isa_circuit, bundle.logical_circuit
    for circ, label, meta in ((isa, "isa", p.isa_circuit), (logical, "logical", p.logical_circuit)):
        if circ is None or not hasattr(circ, "num_qubits"):
            problems.append(f"{label}_circuit_not_a_circuit")
            continue
        try:
            fp, ver = _circuit_fingerprint(circ)
        except Exception:  # noqa: BLE001 - a non-circuit object fails fingerprinting
            problems.append(f"{label}_circuit_not_fingerprintable")
            continue
        if fp != meta.get("fingerprint_sha256"):
            problems.append(f"{label}_fingerprint_mismatch")
        if ver != meta.get("qiskit_version"):
            problems.append(f"{label}_qiskit_version_mismatch")
        if getattr(circ, "num_parameters", 1) != 0:
            problems.append(f"{label}_circuit_not_fully_bound")
    n = p.request["n_binary_vars"]
    try:
        if isa is not None and isa.count_ops().get("measure", 0) != n:
            problems.append("isa_missing_measurements")
    except Exception:  # noqa: BLE001
        problems.append("isa_measurements_uninspectable")
    if getattr(isa, "num_qubits", -1) < n:
        problems.append("isa_too_few_qubits")
    return problems


def to_submission_payload(bundle: PreparedCircuitBundle, *, local_run_id: str) -> dict[str, Any]:
    """The arguments a *future* Phase-3 submission would consume — fail-closed.

    Refuses (before any hand-off) a not-submittable artefact, a tampered artefact,
    or an ISA circuit that does not re-fingerprint to the artefact. This does NOT
    call the Phase-3 submit entry point, does NOT build an IBM runtime service,
    does NOT read a token and does NOT transition any run to ``submitting``.
    """
    if not isinstance(bundle, PreparedCircuitBundle):
        raise HardwareValidationError("to_submission_payload requires a PreparedCircuitBundle")
    p = bundle.prepared
    if not p.submittable:
        raise Phase4CoherenceError("artefact is marked not submittable (synthetic/fake/test)")
    problems = verify_bundle_integrity(bundle)
    if problems:
        raise HardwareValidationError(f"bundle integrity check failed: {problems}")
    if not isinstance(local_run_id, str) or not local_run_id.strip():
        raise HardwareValidationError("local_run_id must be a non-empty string")
    return {
        "local_run_id": local_run_id,
        "isa_circuit": bundle.isa_circuit,
        "expected_config_hash": p.config_hash,
        "shots": p.request["shots"],
        "backend_requested": p.request["backend_requested"],
        "qubo_sha256": p.qubo_sha256,
        "max_jobs": p.request["max_jobs"],
        "max_quantum_seconds": p.request["max_quantum_seconds"],
    }


# The exact Phase-3 hardware param model (order fixed for a stable config hash).
_PHASE3_PARAM_KEYS = (
    "mode", "instance", "n_theta", "n_q", "objective_id", "shots", "transpiler_seed",
    "max_jobs", "max_quantum_seconds", "qubo_sha256", "backend_requested",
)


def _expected_phase3_params(prepared: PreparedHardwareRun) -> dict[str, Any]:
    req = prepared.request
    return {
        "mode": _MODE, "instance": req["instance_id"], "n_theta": req["n_theta"],
        "n_q": req["n_q"], "objective_id": req["objective_id"],
        "shots": req["shots"], "transpiler_seed": req["transpiler_seed"],
        "max_jobs": req["max_jobs"], "max_quantum_seconds": req["max_quantum_seconds"],
        "qubo_sha256": prepared.qubo_sha256, "backend_requested": req["backend_requested"],
    }


def check_submission_consistency(prepared: PreparedHardwareRun,
                                 params: dict[str, Any]) -> list[str]:
    """Return the list of Phase-3/Phase-5 divergences (empty == consistent).

    Covers the FULL hardware param model (mode/instance/n_theta/n_q/shots/
    transpiler_seed/max_jobs/max_quantum_seconds/qubo_sha256/backend_requested)
    plus the config hash.
    """
    expected = _expected_phase3_params(prepared)
    mismatches = [k for k in _PHASE3_PARAM_KEYS if params.get(k) != expected[k]]
    if config_hash(expected) != prepared.config_hash:
        mismatches.append("config_hash")
    return mismatches


def assert_submission_consistency(prepared: PreparedHardwareRun,
                                  params: dict[str, Any]) -> None:
    """Fail-closed variant: raise a typed error on any divergence."""
    mismatches = check_submission_consistency(prepared, params)
    if mismatches:
        raise HardwareValidationError(f"Phase-3/Phase-5 params diverge: {mismatches}")


# --------------------------------------------------------------------------- #
# Strict Sampler decoding + scientific scoring
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DecodedCandidate:
    bitstring: str          # QUBO order (bits[b] == value of bit b)
    probability: float
    energy: float
    feasible: bool
    onehot_valid: bool
    n_onehot_violations: int
    n_conflicts: int
    objective: float
    rank: int


@dataclass(frozen=True)
class DecodedSummary:
    n_shots: int
    n_unique: int                       # == n_unique_total (full validated distribution)
    n_unique_total: int
    n_candidates_returned: int
    probability_mass_returned: float
    best_raw: DecodedCandidate | None
    best_feasible: DecodedCandidate | None
    modal: DecodedCandidate | None
    feasibility_rate: float
    mean_energy: float
    energy_dispersion: float
    optimality_gap: float | None        # relative gap; None if reference is 0
    absolute_gap: float | None
    candidates: tuple[DecodedCandidate, ...]   # truncated DETAIL list only
    warnings: tuple[str, ...]
    top_explanations: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def _to_qubo_order(raw: str, n: int, bit_order: str) -> str:
    if len(raw) != n or set(raw) - {"0", "1"}:
        raise HardwareValidationError(f"bitstring {raw!r} is not {n} binary characters")
    return raw[::-1] if bit_order == "qiskit" else raw


def _normalise_distribution(
    raw: dict[str, float], n: int, *, bit_order: str, kind: str, shots: int | None,
) -> tuple[dict[str, float], int]:
    """Validate raw results and fold to a {qubo_bitstring: probability} map.

    counts: if ``shots`` is given it MUST equal the count sum (else a typed error);
    otherwise the sum is used. quasi: non-negative, sums to ~1. Bad keys, bools,
    zero totals and over-large inputs are refused before any expensive decode.
    """
    if len(raw) > MAX_DECODED_CANDIDATES:
        raise BudgetExceeded(f"too many raw result entries ({len(raw)} > {MAX_DECODED_CANDIDATES})")
    prob: dict[str, float] = {}
    if kind == "counts":
        total = 0
        for bs, c in raw.items():
            if isinstance(c, bool) or not isinstance(c, int) or c < 0:
                raise HardwareValidationError(f"count for {bs!r} must be a non-negative int")
            total += c
        if total <= 0:
            raise HardwareValidationError("counts sum to zero")
        if shots is not None and shots != total:
            raise HardwareValidationError(f"shots {shots} != count sum {total}")
        n_shots = total
        for bs, c in raw.items():
            key = _to_qubo_order(bs, n, bit_order)
            prob[key] = prob.get(key, 0.0) + c / total
    else:
        s = 0.0
        for bs, pv in raw.items():
            fv = _req_finite(pv, f"probability for {bs!r}")
            if fv < 0:
                raise HardwareValidationError("quasi-distribution has a negative probability")
            s += fv
            key = _to_qubo_order(bs, n, bit_order)
            prob[key] = prob.get(key, 0.0) + fv
        if not math.isclose(s, 1.0, abs_tol=1e-6):
            raise HardwareValidationError(f"quasi-distribution must sum to 1 (got {s})")
        n_shots = shots if shots is not None else 0
    return prob, n_shots


def decode_samples(
    qubo: ManeuverQUBO,
    raw: dict[str, float],
    *,
    bit_order: str = "qiskit",
    kind: str = "counts",
    shots: int | None = None,
    max_candidates: int = 256,
    reference_objective: float | None = None,
    n_explain: int = 3,
) -> DecodedSummary:
    """Decode a synthetic Sampler result into scored, ranked ACRP candidates.

    ``kind='counts'`` expects integer counts; ``kind='quasi'`` expects
    probabilities (non-negative, summing to ~1). Endianness is normalised via
    ``bit_order``. ``best_raw`` (lowest energy), ``best_feasible`` (lowest energy
    among conflict-free) and ``modal`` (most probable) are kept distinct.

    All scientific aggregates are computed over the **full validated
    distribution**; ``max_candidates`` truncates only the returned detail list
    (``candidates``/``top_explanations``) — a rare but lower-energy sample is
    never dropped from ``best_raw``. Truncation is surfaced via
    ``n_unique_total`` / ``n_candidates_returned`` / ``probability_mass_returned``
    and a warning.
    """
    if bit_order not in ("qiskit", "qubo"):
        raise HardwareValidationError("bit_order must be 'qiskit' or 'qubo'")
    if kind not in ("counts", "quasi"):
        raise HardwareValidationError("kind must be 'counts' or 'quasi'")
    if not isinstance(raw, dict):
        raise HardwareValidationError("raw results must be a dict")
    _req_int(max_candidates, "max_candidates", lo=1)
    _req_int(n_explain, "n_explain", lo=0)
    if reference_objective is not None:
        _req_finite(reference_objective, "reference_objective")

    warnings: list[str] = []
    n = qubo.n_qubits
    inst = qubo.discrete.instance

    if not raw:
        return DecodedSummary(
            n_shots=shots or 0, n_unique=0, n_unique_total=0, n_candidates_returned=0,
            probability_mass_returned=0.0, best_raw=None, best_feasible=None, modal=None,
            feasibility_rate=0.0, mean_energy=0.0, energy_dispersion=0.0, optimality_gap=None,
            absolute_gap=None, candidates=(), warnings=("empty_result",))

    prob, n_shots = _normalise_distribution(raw, n, bit_order=bit_order, kind=kind, shots=shots)

    # Decode + score EVERY unique bitstring (aggregates must see the full set).
    from .. import geometry
    full: list[DecodedCandidate] = []
    for bs, p in prob.items():
        onehot_violations = sum(
            sum(bs[qubo.discrete.var_index(i, o)] == "1" for o in range(qubo.discrete.k)) != 1
            for i in inst.aircraft()
        )
        onehot_valid = onehot_violations == 0
        assignment = decode_assignment(qubo, bs)
        choice = tuple(assignment[i] for i in inst.aircraft())
        q, theta = qubo.discrete.maneuvers(choice)
        n_conf = geometry.count_conflicts(inst, q, theta)
        full.append(DecodedCandidate(
            bitstring=bs, probability=p, energy=qubo.energy(bs),
            feasible=(onehot_valid and n_conf == 0), onehot_valid=onehot_valid,
            n_onehot_violations=onehot_violations,
            n_conflicts=n_conf,
            objective=primary_objective(qubo.objective_id, q, theta, inst.w), rank=0))
    full.sort(key=lambda c: (c.energy, c.bitstring))
    full = [DecodedCandidate(**{**asdict(c), "rank": i + 1}) for i, c in enumerate(full)]

    # aggregates over the FULL distribution
    best_raw = full[0]
    feas = [c for c in full if c.feasible]
    best_feasible = min(feas, key=lambda c: (c.energy, c.bitstring)) if feas else None
    modal = max(full, key=lambda c: (c.probability, -c.rank))
    feasibility_rate = sum(c.probability for c in full if c.feasible)
    mean_energy = sum(c.probability * c.energy for c in full)
    var = sum(c.probability * (c.energy - mean_energy) ** 2 for c in full)
    dispersion = math.sqrt(var) if var > 0 else 0.0

    optimality_gap: float | None = None
    absolute_gap: float | None = None
    if reference_objective is not None and best_feasible is not None:
        ref = reference_objective
        absolute_gap = best_feasible.objective - ref
        if ref == 0:
            optimality_gap = None       # relative gap undefined; never NaN/Inf
            warnings.append("relative_gap_undefined_zero_reference")
        else:
            # minimisation: gap = (sampled_best_feasible - reference) / |reference|
            optimality_gap = absolute_gap / abs(ref)
    elif reference_objective is not None:
        warnings.append("no_feasible_sample_for_gap")

    # truncate ONLY the returned detail list (by rank = energy asc)
    if len(full) > max_candidates:
        warnings.append(f"truncated_detail_list:{len(full)}->{max_candidates}")
    returned = full[:max_candidates]
    mass_returned = sum(c.probability for c in returned)

    explanations = tuple(
        {"bitstring": c.bitstring, "rank": c.rank, "probability": round(c.probability, 6),
         "feasible": c.feasible, "onehot_valid": c.onehot_valid,
         "n_onehot_violations": c.n_onehot_violations,
         "n_conflicts": c.n_conflicts, "objective": c.objective,
         "energy": c.energy, "energy_breakdown": energy_breakdown(qubo, c.bitstring)}
        for c in returned[:n_explain])

    return DecodedSummary(
        n_shots=n_shots, n_unique=len(full), n_unique_total=len(full),
        n_candidates_returned=len(returned), probability_mass_returned=mass_returned,
        best_raw=best_raw, best_feasible=best_feasible, modal=modal,
        feasibility_rate=feasibility_rate, mean_energy=mean_energy,
        energy_dispersion=dispersion, optimality_gap=optimality_gap, absolute_gap=absolute_gap,
        candidates=tuple(returned), warnings=tuple(warnings), top_explanations=explanations)


def decode_to_result(qubo: ManeuverQUBO, bitstring_qubo_order: str, *,
                     backend: str | None = None, shots: int | None = None):
    """Re-score a QUBO-order bitstring through the geometry kernel (read-only)."""
    return _decode_result(qubo, bitstring_qubo_order, backend=backend, shots=shots)
