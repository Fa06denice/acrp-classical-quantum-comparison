"""QAOA solver over the maneuver QUBO (OPTIONAL — requires ``[quantum]``).

Pipeline::

    ManeuverQUBO -> QuadraticProgram -> MinimumEigenOptimizer(QAOA(sampler))
                 -> best bitstring + sampled counts -> decode (caller)

Uses the self-contained solvers shipped with ``qiskit-optimization`` 0.6+
(``qiskit_optimization.minimum_eigensolvers.QAOA`` and
``qiskit_optimization.optimizers.COBYLA``), which work with a Qiskit 1.x/2.x
``SamplerV2`` and do not depend on the deprecated standalone
``qiskit-algorithms`` package.

Determinism: the backend seed is threaded into the Aer sampler and
``algorithm_globals.random_seed`` (when available); shot noise on real hardware
is inherently non-reproducible.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

from .._optional import require
from .backends import BackendConfig, make_sampler
from .qubo import ManeuverQUBO

# QAOA seeds a PROCESS-GLOBAL RNG (qiskit_optimization.utils.algorithm_globals),
# so two QAOA runs interleaving in one process would race on that seed and lose
# determinism. This lock serialises the RNG-dependent section of solve() within a
# process — a DELIBERATE serialisation for reproducibility, NOT parallelism. QAOA
# runs in different processes are unaffected.
_QAOA_RUN_LOCK = threading.Lock()

# Per-optimizer allowlist of tuning knobs, derived from the REAL constructor
# signatures of qiskit-optimization 0.7.0's COBYLA / NELDER_MEAD. Only genuine
# finite numeric knobs are versionable here; display/escape-hatch params (disp,
# adaptive, options, **kwargs) and the dedicated `maxiter` field are refused.
OPTIMIZER_OPTION_ALLOWLIST: dict[str, dict[str, str]] = {
    "COBYLA": {"rhobeg": "real", "tol": "real"},
    "NELDER_MEAD": {"xatol": "real", "tol": "real", "maxfev": "int"},
}

# Semantic bounds enforced UP FRONT (never wait for a late SciPy/Qiskit failure):
# step sizes and tolerances must be strictly positive; the evaluation budget >= 1.
_OPTIMIZER_OPTION_BOUNDS: dict[str, tuple[str, float]] = {
    "rhobeg": ("gt", 0.0), "tol": ("gt", 0.0), "xatol": ("gt", 0.0),
    "maxfev": ("ge", 1),
}


def validate_optimizer_options(optimizer: str, options) -> MappingProxyType:
    """Validate/normalise ``optimizer_options`` against the per-optimizer allowlist.

    Accepts a Mapping or an iterable of ``(key, value)`` pairs; returns an immutable
    :class:`MappingProxyType` of JSON-native values. Fail-closed: refuses unknown
    keys, duplicates, bools, non-numeric or non-finite (NaN/Inf) values, and
    type mismatches. ``None``/empty -> empty mapping."""
    allow = OPTIMIZER_OPTION_ALLOWLIST.get(optimizer)
    if allow is None:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    if options is None:
        return MappingProxyType({})
    try:
        items = list(options.items()) if isinstance(options, Mapping) else list(options)
    except TypeError:  # not a mapping and not iterable -> fail-closed with ValueError
        raise ValueError("optimizer_options must be a mapping or an iterable of "
                         "(key, value) pairs") from None
    out: dict[str, float] = {}
    for pair in items:
        if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
            raise ValueError("optimizer_options entries must be (key, value) pairs")
        key, value = pair
        if not isinstance(key, str) or not key.strip():
            raise ValueError("optimizer_options keys must be non-empty strings")
        if key in out:
            raise ValueError(f"duplicate optimizer option {key!r}")
        if key not in allow:
            raise ValueError(f"unsupported optimizer option {key!r} for {optimizer}; "
                             f"allowed: {sorted(allow)}")
        if isinstance(value, bool):
            raise ValueError(f"optimizer option {key!r} must not be a bool")
        if allow[key] == "int":
            if not isinstance(value, int):
                raise ValueError(f"optimizer option {key!r} must be an int")
        elif not isinstance(value, (int, float)):
            raise ValueError(f"optimizer option {key!r} must be a real number")
        if not math.isfinite(value):
            raise ValueError(f"optimizer option {key!r} must be finite (no NaN/Inf)")
        bound = _OPTIMIZER_OPTION_BOUNDS.get(key)
        if bound is not None:
            op, threshold = bound
            if op == "gt" and not value > threshold:
                raise ValueError(f"optimizer option {key!r} must be > {threshold}")
            if op == "ge" and not value >= threshold:
                raise ValueError(f"optimizer option {key!r} must be >= {threshold}")
        out[key] = value
    return MappingProxyType(out)


@dataclass
class QAOAResult:
    """Outcome of a QAOA solve over a QUBO (before ACRP decoding)."""

    best_bitstring: str  # value of var at bit position b == best_bitstring[b]
    best_energy: float
    counts: dict[str, int] = field(default_factory=dict)
    optimal_params: list[float] = field(default_factory=list)
    eval_count: int = 0
    n_qubits: int = 0
    circuit_depth: int = 0
    circuit_size: int = 0
    two_qubit_gates: int = 0
    backend_meta: dict = field(default_factory=dict)
    wall_time_s: float = 0.0  # total: connect + queue + execute + classical opt
    qpu_time_s: float | None = None  # real quantum-processor seconds (IBM only)
    n_jobs: int = 0  # number of backend jobs submitted
    sim_method: str | None = None  # Aer method (statevector / matrix_product_state)


@dataclass
class QAOASolver:
    """Solve a :class:`ManeuverQUBO` with QAOA on the configured backend."""

    backend: BackendConfig = field(default_factory=BackendConfig)
    reps: int = 1
    maxiter: int = 100
    optimizer: str = "COBYLA"
    initial_point: list[float] | None = None
    optimizer_options: object = None  # Mapping | iterable of pairs | None; frozen in __post_init__

    def __post_init__(self) -> None:
        require("qiskit_optimization")  # fail fast with a clear message
        if isinstance(self.reps, bool) or not isinstance(self.reps, int) or self.reps < 1:
            raise ValueError("reps must be an integer >= 1")
        if isinstance(self.maxiter, bool) or not isinstance(self.maxiter, int) or self.maxiter < 1:
            raise ValueError("maxiter must be an integer >= 1")
        self.optimizer = self.optimizer.upper()
        if self.optimizer not in {"COBYLA", "NELDER_MEAD"}:
            raise ValueError("optimizer must be COBYLA or NELDER_MEAD")
        if self.initial_point is not None and len(self.initial_point) != 2 * self.reps:
            raise ValueError(f"initial_point must contain {2 * self.reps} values")
        # Validate + defensively freeze the optimizer options against the real
        # per-optimizer allowlist; the frozen mapping is what solve() forwards.
        self.optimizer_options = validate_optimizer_options(self.optimizer, self.optimizer_options)

    def to_quadratic_program(self, qubo: ManeuverQUBO):
        """Build a ``QuadraticProgram`` whose variable order == QUBO bit order."""
        from qiskit_optimization import QuadraticProgram

        qp = QuadraticProgram(name="acrp_maneuver_qubo")
        for b in range(qubo.n_qubits):
            qp.binary_var(name=f"x{b}")

        linear = {f"x{b}": coeff for b, coeff in qubo.linear.items()}
        quadratic = {
            (f"x{a}", f"x{c}"): coeff for (a, c), coeff in qubo.quadratic.items()
        }
        qp.minimize(constant=qubo.constant, linear=linear, quadratic=quadratic)
        return qp

    def solve(self, qubo: ManeuverQUBO) -> QAOAResult:
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        from qiskit_optimization.minimum_eigensolvers import QAOA
        from qiskit_optimization.optimizers import COBYLA, NELDER_MEAD

        from .backends import AerMethod, BackendMode
        from .memguard import check_memory
        from ..exceptions import QubitBudgetError

        # Re-validate + re-freeze the optimizer options at the point of USE, so a
        # post-construction mutation (QAOASolver is a mutable dataclass) can never
        # forward an unsupported/unvalidated option to the optimiser — fail-closed.
        self.optimizer_options = validate_optimizer_options(self.optimizer, self.optimizer_options)

        # Refuse a statevector run that would exceed the RAM budget, before any
        # circuit allocation (the 2**N wall). MPS is never refused here.
        if (
            self.backend.mode == BackendMode.AER_SIM
            and self.backend.sim_method == AerMethod.STATEVECTOR
        ):
            verdict = check_memory(qubo.n_qubits, "statevector")
            if not verdict.ok:
                raise QubitBudgetError(verdict.message)

        # Serialise the RNG-dependent section: seeding the process-global RNG,
        # sampler/transpiler construction (both seeded), and the optimisation loop
        # all read/write shared RNG state. The lock keeps a concurrent QAOA run in
        # the same process from perturbing this one (deliberate serialisation, not
        # parallelism). Timing is measured inside so wall_time_s includes any wait.
        t0 = time.perf_counter()
        with _QAOA_RUN_LOCK:
            try:  # seed the global RNG when the helper is available
                from qiskit_optimization.utils import algorithm_globals

                algorithm_globals.random_seed = self.backend.seed
            except Exception:  # noqa: BLE001 - optional convenience only
                pass

            raw_sampler, backend, backend_meta = make_sampler(self.backend)
            from .backends import _UsageTrackingSampler

            sampler = _UsageTrackingSampler(raw_sampler)
            opt_cls = {"COBYLA": COBYLA, "NELDER_MEAD": NELDER_MEAD}[self.optimizer]
            opts = cast("Mapping[str, float]", self.optimizer_options)
            optimizer = opt_cls(maxiter=self.maxiter, **dict(opts))
            initial_point = None
            if self.initial_point is not None:
                import numpy as np

                initial_point = np.asarray(self.initial_point, dtype=float)

            # Pass manager so SamplerV2 circuits are transpiled to the backend's ISA.
            pass_manager = None
            if backend is not None:
                try:
                    from qiskit.transpiler.preset_passmanagers import (
                        generate_preset_pass_manager,
                    )

                    pass_manager = generate_preset_pass_manager(
                        optimization_level=self.backend.optimization_level,
                        backend=backend,
                        seed_transpiler=self.backend.seed,
                    )
                except Exception:  # noqa: BLE001 - transpilation aid, fall back to default
                    pass_manager = None

            qaoa = QAOA(
                sampler=sampler,
                optimizer=optimizer,
                reps=self.reps,
                initial_point=initial_point,
                pass_manager=pass_manager,
            )
            meo = MinimumEigenOptimizer(qaoa)
            qp = self.to_quadratic_program(qubo)
            result = meo.solve(qp)

        if result.x is None or result.fval is None:
            raise RuntimeError("QAOA returned no solution")
        best_bitstring = "".join(str(int(round(v))) for v in result.x)
        best_energy = float(result.fval)
        counts = _extract_counts(result, qubo.n_qubits)
        mes = getattr(result, "min_eigen_solver_result", None)
        params = _safe_list(getattr(mes, "optimal_point", None))
        eval_count = int(getattr(mes, "cost_function_evals", 0) or 0)
        depth, size, twoq = _circuit_metrics(mes)

        return QAOAResult(
            best_bitstring=best_bitstring,
            best_energy=best_energy,
            counts=counts,
            optimal_params=params,
            eval_count=eval_count,
            n_qubits=qubo.n_qubits,
            circuit_depth=depth,
            circuit_size=size,
            two_qubit_gates=twoq,
            backend_meta=backend_meta,
            wall_time_s=time.perf_counter() - t0,
            qpu_time_s=sampler.qpu_seconds if sampler.usage_available else None,
            n_jobs=sampler.n_jobs,
            sim_method=backend_meta.get("sim_method"),
        )

    def solve_decomposed(self, qubo: ManeuverQUBO) -> QAOAResult:
        """Solve disconnected QUBO components independently and recombine.

        This is exact at the model-decomposition level: there are no omitted
        coefficients between components.  Each component is still solved by
        the configured approximate QAOA algorithm.  The method is mainly useful
        when the full statevector exceeds RAM but every component fits.
        """
        from .explain import split_qubo_components

        parts = split_qubo_components(qubo)
        if len(parts) <= 1:
            return self.solve(qubo)

        t0 = time.perf_counter()
        combined = ["0"] * qubo.n_qubits
        results: list[QAOAResult] = []
        component_meta: list[dict[str, object]] = []
        for component, local_qubo in parts:
            local_result = self.solve(local_qubo)
            results.append(local_result)
            for local_bit, original_bit in enumerate(component):
                combined[original_bit] = local_result.best_bitstring[local_bit]
            component_meta.append(
                {
                    "n_qubits": len(component),
                    "original_bits": list(component),
                    "energy_without_global_constant": local_result.best_energy,
                    "wall_time_s": local_result.wall_time_s,
                }
            )

        bitstring = "".join(combined)
        backend_meta = dict(results[0].backend_meta)
        backend_meta.update(
            {
                "exact_component_decomposition": True,
                "n_components": len(parts),
                "largest_component_qubits": max(len(component) for component, _ in parts),
                "components": component_meta,
            }
        )
        return QAOAResult(
            best_bitstring=bitstring,
            best_energy=qubo.energy(bitstring),
            # A full joint distribution would require a potentially exponential
            # Cartesian product.  Do not fabricate one from marginal samples.
            counts={},
            optimal_params=[],
            eval_count=sum(result.eval_count for result in results),
            n_qubits=qubo.n_qubits,
            circuit_depth=max(result.circuit_depth for result in results),
            circuit_size=sum(result.circuit_size for result in results),
            two_qubit_gates=sum(result.two_qubit_gates for result in results),
            backend_meta=backend_meta,
            wall_time_s=time.perf_counter() - t0,
            qpu_time_s=(
                sum(result.qpu_time_s or 0.0 for result in results)
                if any(result.qpu_time_s is not None for result in results)
                else None
            ),
            n_jobs=sum(result.n_jobs for result in results),
            sim_method=results[0].sim_method,
        )


def _extract_counts(result, n_qubits: int) -> dict[str, int]:
    """Aggregate sampled solutions into a {bitstring: count} map."""
    counts: dict[str, int] = {}
    samples = getattr(result, "samples", None) or []
    for s in samples:
        x = getattr(s, "x", None)
        prob = getattr(s, "probability", 0.0)
        if x is None:
            continue
        bitstring = "".join(str(int(round(v))) for v in x)
        # probabilities -> pseudo-counts on a 10_000 grid (stable ordering for export)
        counts[bitstring] = counts.get(bitstring, 0) + max(1, int(round(prob * 10_000)))
    return counts


def _safe_list(obj) -> list[float]:
    if obj is None:
        return []
    try:
        return [float(v) for v in obj]
    except TypeError:
        return []


def _circuit_metrics(mes) -> tuple[int, int, int]:
    """Best-effort QAOA ansatz depth / size / two-qubit-gate count."""
    if mes is None:
        return 0, 0, 0
    circ = getattr(mes, "optimal_circuit", None)
    if circ is None:
        return 0, 0, 0
    try:
        depth = int(circ.depth())
        size = int(circ.size())
        twoq = sum(1 for inst in circ.data if inst.operation.num_qubits == 2)
        return depth, size, twoq
    except Exception:  # noqa: BLE001 - metrics are diagnostic, never fatal
        return 0, 0, 0
