"""Versioned, immutable, fully-hashed batch protocol (Phase 5.2.B).

A ``BatchProtocol`` captures *everything* that defines a (future, real) QPU batch so a
single canonical hash pins the whole scientific + operational configuration. It is a
forward-looking design object — building one contacts nothing and submits nothing. Its
hash is the identity that claim / run / confirmation / provider-tag / budget / export are
all bound to (see ``qpu_batch`` batch-identity companion, ``qpu_dedup`` tag,
``qpu_confirm`` nonce).

The hash is canonical (sorted keys), order-independent, refuses NaN/Inf and non-JSON
types, changes when any scientific/operational field changes, and contains no secret,
timestamp, or local path. ``backend_effective`` is recorded only once known (chosen at
the last moment) and, being part of the identity, changes the hash — so a plan hashed
before backend selection and the same plan after are distinct, on purpose.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

PROTOCOL_SCHEMA = "acrpq-qpu-batch-protocol/1"
_ENDIANNESS = frozenset({"qiskit", "qubo"})
_ANGLE_STRATEGIES = frozenset({"seeded_random_uniform", "zeros", "fixed"})
_OPTIMIZERS = frozenset({"COBYLA", "NELDER_MEAD"})


class ProtocolError(ValueError):
    """The protocol is structurally invalid; fail closed."""


def _reject_nonfinite(obj: Any, path: str = "") -> None:
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ProtocolError(f"non-finite value at {path or '<root>'}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ProtocolError(f"non-string key at {path}: {k!r}")
            _reject_nonfinite(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _reject_nonfinite(v, f"{path}[{i}]")
    elif not isinstance(obj, (str, int, float, bool, type(None))):
        raise ProtocolError(f"non-JSON-native type at {path or '<root>'}: {type(obj).__name__}")


@dataclass(frozen=True)
class JobSpec:
    """One job within a batch — identity bound to the exact QUBO/instance/grid."""

    instance: str
    objective_id: str
    n_theta: int
    n_q: int
    n_qubits: int
    qubo_sha256: str
    instance_sha256: str
    logical_circuit_sha256: str | None = None
    isa_sha256: str | None = None


@dataclass(frozen=True)
class BatchProtocol:
    """Immutable, fully-hashed configuration of a QPU batch."""

    qaoa_protocol_id: str
    qaoa_reps: int
    angle_strategy: str
    optimizer: str
    shots: int
    transpiler_seed: int
    backend_requested: str
    optimization_level: int
    max_jobs: int
    max_quantum_seconds: float
    total_shot_budget: int
    read_retry_policy: str
    reconciliation_policy: str
    max_wait_seconds: float
    endianness: str
    qiskit_version: str
    jobs: tuple[JobSpec, ...]
    schema_version: str = PROTOCOL_SCHEMA
    backend_effective: str | None = None       # recorded once known (last-moment choice)
    optimizer_options: dict[str, Any] = field(default_factory=dict)
    dependency_versions: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != PROTOCOL_SCHEMA:
            raise ProtocolError(f"unknown schema_version {self.schema_version!r}")
        if not self.jobs:
            raise ProtocolError("a batch protocol needs at least one job")
        for n, v in (("qaoa_reps", self.qaoa_reps), ("shots", self.shots),
                     ("max_jobs", self.max_jobs), ("total_shot_budget", self.total_shot_budget)):
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise ProtocolError(f"{n} must be an int >= 1; got {v!r}")
        if not isinstance(self.transpiler_seed, int) or isinstance(self.transpiler_seed, bool) \
                or self.transpiler_seed < 0:
            raise ProtocolError("transpiler_seed must be a non-negative int")
        if not (isinstance(self.optimization_level, int)
                and not isinstance(self.optimization_level, bool)
                and 0 <= self.optimization_level <= 3):
            raise ProtocolError("optimization_level must be 0..3")
        for fn, fv in (("max_quantum_seconds", self.max_quantum_seconds),
                       ("max_wait_seconds", self.max_wait_seconds)):
            if not isinstance(fv, (int, float)) or isinstance(fv, bool) \
                    or not math.isfinite(fv) or fv <= 0:
                raise ProtocolError(f"{fn} must be a finite number > 0; got {fv!r}")
        if self.angle_strategy not in _ANGLE_STRATEGIES:
            raise ProtocolError(f"angle_strategy must be one of {sorted(_ANGLE_STRATEGIES)}")
        if self.optimizer not in _OPTIMIZERS:
            raise ProtocolError(f"optimizer must be one of {sorted(_OPTIMIZERS)}")
        if self.endianness not in _ENDIANNESS:
            raise ProtocolError(f"endianness must be one of {sorted(_ENDIANNESS)}")
        if self.total_shot_budget < self.shots * len(self.jobs):
            raise ProtocolError("total_shot_budget is below shots * number of jobs")
        _reject_nonfinite(self.optimizer_options, "optimizer_options")

    def to_canonical(self) -> dict:
        """JSON-native, deterministic dict over which the hash is computed."""
        d = asdict(self)
        _reject_nonfinite(d)
        return d

    def protocol_hash(self) -> str:
        blob = json.dumps(self.to_canonical(), sort_keys=True, separators=(",", ":"),
                          allow_nan=False)
        return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def with_effective_backend(self, backend_effective: str) -> BatchProtocol:
        """Return a copy with the effective backend recorded (a distinct identity)."""
        if not isinstance(backend_effective, str) or not backend_effective.strip():
            raise ProtocolError("backend_effective must be a non-empty string")
        from dataclasses import replace
        return replace(self, backend_effective=backend_effective)
