"""Quantum execution backends (OPTIONAL — requires the ``[quantum]``/``[ibm]`` extras).

Three execution modes, matching the brief:

* ``AER_SIM``     — local ``qiskit-aer`` ``SamplerV2``; no IBM account needed.
* ``IBM_RUNTIME`` — IBM Quantum Runtime ``SamplerV2`` on a cloud *simulator*.
* ``IBM_REAL``    — IBM Quantum Runtime ``SamplerV2`` on a real QPU (least-busy
  by default, or a named backend).

All heavy imports happen lazily inside :func:`make_sampler`. IBM credentials are
read only from the environment / a saved account (``QISKIT_IBM_TOKEN``); they are
never embedded in code and never written to any result file.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .._optional import require


class BackendMode(str, Enum):
    AER_SIM = "aer_sim"
    IBM_RUNTIME = "ibm_runtime"
    IBM_REAL = "ibm_real"


class AerMethod(str, Enum):
    """Aer local-simulation method.

    ``STATEVECTOR`` is exact but memory grows as ``2**N`` (the qubit wall);
    ``MATRIX_PRODUCT_STATE`` (MPS) trades exactness for memory that scales with
    entanglement, reaching more qubits on low-entanglement circuits.
    """

    STATEVECTOR = "statevector"
    MATRIX_PRODUCT_STATE = "matrix_product_state"
    AUTOMATIC = "automatic"


class _UsageTrackingSampler:
    """Proxy around a SamplerV2 that accumulates real QPU usage seconds.

    QAOA submits many sampler jobs through ``MinimumEigenOptimizer``; we cannot
    reach them directly. Wrapping ``.run()`` lets us sum each IBM job's
    ``usage()`` (actual quantum-processor seconds) and the number of jobs, which
    is then reported separately from the wall-clock time (the latter is
    dominated by queue wait on the open plan).
    """

    def __init__(self, sampler):
        self._sampler = sampler
        self.qpu_seconds = 0.0
        self.n_jobs = 0
        self.usage_available = False

    def __getattr__(self, name):  # delegate everything else (e.g. .options)
        return getattr(self._sampler, name)

    def run(self, *args, **kwargs):
        job = self._sampler.run(*args, **kwargs)
        self.n_jobs += 1
        # IBM RuntimeJobV2 exposes usage(); Aer's job does not. Be defensive:
        # querying usage() may block until the job finishes, which is fine here
        # because QAOA consumes the result synchronously right after.
        usage = getattr(job, "usage", None)
        if callable(usage):
            try:
                seconds = usage()
                if seconds is not None:
                    self.qpu_seconds += float(seconds)
                    self.usage_available = True
            except Exception:  # noqa: BLE001 - usage is best-effort telemetry
                pass
        return job


@dataclass(frozen=True)
class BackendConfig:
    """Configuration for a quantum execution backend."""

    mode: BackendMode = BackendMode.AER_SIM
    shots: int = 1024
    seed: int = 1234
    backend_name: str | None = None  # explicit IBM backend; None -> sim / least_busy
    optimization_level: int = 1
    channel: str | None = None  # IBM Runtime channel, e.g. "ibm_quantum"
    instance: str | None = None  # IBM hub/group/project
    sim_method: AerMethod = AerMethod.STATEVECTOR  # AER_SIM only
    max_memory_mb: int | None = None  # AER_SIM only: abort (not swap) past this

    def __post_init__(self) -> None:
        if not isinstance(self.mode, BackendMode):
            raise TypeError("mode must be a BackendMode")
        if not isinstance(self.sim_method, AerMethod):
            raise TypeError("sim_method must be an AerMethod")
        if isinstance(self.shots, bool) or not isinstance(self.shots, int) or self.shots < 1:
            raise ValueError("shots must be an integer >= 1")
        if not 0 <= self.optimization_level <= 3:
            raise ValueError("optimization_level must be between 0 and 3")
        if self.max_memory_mb is not None and self.max_memory_mb < 1:
            raise ValueError("max_memory_mb must be >= 1")

    @property
    def label(self) -> str:
        if self.backend_name:
            return self.backend_name
        if self.mode == BackendMode.AER_SIM:
            return f"aer_simulator[{self.sim_method.value}]"
        return {
            BackendMode.IBM_RUNTIME: "ibm_runtime_sim",
            BackendMode.IBM_REAL: "ibm_least_busy",
        }[self.mode]


def make_sampler(cfg: BackendConfig) -> tuple[object, object | None, dict]:
    """Return ``(sampler, backend, backend_meta)`` for the configured mode.

    ``backend`` is the transpilation target (an Aer/IBM backend) used to build a
    pass manager so ``SamplerV2`` circuits are transpiled correctly; it may be
    ``None`` only if transpilation is unnecessary. ``backend_meta`` carries
    name/simulator/qubit-count info for metrics.
    """
    if cfg.mode == BackendMode.AER_SIM:
        return _aer_sampler(cfg)
    return _ibm_sampler(cfg)


def _aer_sampler(cfg: BackendConfig) -> tuple[object, object, dict]:
    aer = require("qiskit_aer")
    from qiskit_aer import AerSimulator  # local import after require
    from qiskit_aer.primitives import SamplerV2

    method = cfg.sim_method.value
    backend_options: dict = {"method": method}
    if cfg.max_memory_mb is not None:
        backend_options["max_memory_mb"] = cfg.max_memory_mb  # abort, don't swap
    # Injecting the method through options also sets the sampler's internal
    # transpile target (aer 0.17 verified); the AerSimulator(method=...) below
    # matches it as the pass-manager target used by QAOASolver.
    sampler = SamplerV2(
        default_shots=cfg.shots,
        seed=cfg.seed,
        options={"backend_options": backend_options},
    )
    backend = AerSimulator(method=method)
    meta = {
        "name": cfg.label,
        "is_simulator": True,
        "n_qubits": None,
        "sim_method": method,
        "provider": "qiskit-aer",
        "version": getattr(aer, "__version__", "unknown"),
    }
    return sampler, backend, meta


def _ibm_sampler(cfg: BackendConfig) -> tuple[object, object, dict]:
    require("qiskit_ibm_runtime")
    import os

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    from ..io.env import load_dotenv

    # Pull credentials from a .env file (if present) without overriding a real
    # environment variable. The token is passed explicitly and never logged.
    load_dotenv()
    token = os.environ.get("QISKIT_IBM_TOKEN")
    service_kwargs: dict = {"channel": cfg.channel, "instance": cfg.instance}
    if token:
        service_kwargs["token"] = token
    service = QiskitRuntimeService(**{k: v for k, v in service_kwargs.items() if v is not None})
    if cfg.backend_name:
        backend = service.backend(cfg.backend_name)
    elif cfg.mode == BackendMode.IBM_RUNTIME:
        # cloud simulator backend if available, else least-busy
        backend = service.least_busy(simulator=True)
    else:  # IBM_REAL
        backend = service.least_busy(operational=True, simulator=False)

    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = cfg.shots
    meta = {
        "name": backend.name,
        "is_simulator": bool(
            getattr(getattr(backend, "configuration", lambda: None)(), "simulator", False)
        ),
        "n_qubits": getattr(backend, "num_qubits", None),
        "provider": "qiskit-ibm-runtime",
        "version": "ibm",
    }
    return sampler, backend, meta
