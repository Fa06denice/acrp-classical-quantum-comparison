"""D-Wave third route over the maneuver QUBO (OPTIONAL — requires [dwave]).

Solves the IDENTICAL Hamiltonian H(x) that :class:`QuboExactSolver` and
:class:`QAOASolver` target, decoded through the IDENTICAL geometry kernel
(:func:`decode`), as a third quantum route beside IBM/QAOA. Two backends,
selected by :class:`DWaveMode`:

* ``NEAL_SIM`` — the ``dwave-samplers`` ``SimulatedAnnealingSampler`` (D-Wave's
  reference *classical* simulated annealer; no token; always works). NOTE: the
  legacy top-level ``neal`` package is absent in modern Ocean; the import is
  ``from dwave.samplers import SimulatedAnnealingSampler`` (``neal`` used to
  re-export this exact class). The enum name ``NEAL_SIM`` is kept for the
  familiar mental model.
* ``QPU_REAL`` — ``dwave-system`` ``DWaveSampler`` + ``EmbeddingComposite`` on a
  real Leap QPU; requires ``DWAVE_API_TOKEN``. Minor-embeds via ``minorminer``;
  captures chain-break fraction, QPU access time and embedding size.

All dimod/dwave imports live INSIDE methods (never module-top) so the package
core still imports with zero heavy dependencies.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum

from ..model import Result, SolverKind
from .decode import decode
from .qubo import ManeuverQUBO


class DWaveMode(str, Enum):
    NEAL_SIM = "neal_sim"  # dwave.samplers SA reference sampler (no token) — default
    QPU_REAL = "qpu_real"  # real quantum annealer (Leap token required)


@dataclass(frozen=True)
class DWaveConfig:
    mode: DWaveMode = DWaveMode.NEAL_SIM
    num_reads: int = 1000
    seed: int = 1234
    num_sweeps: int = 1000  # SA only
    solver_name: str | None = None  # QPU only: explicit Leap solver
    annealing_time_us: float | None = None  # QPU only: per-read anneal time
    chain_strength: float | None = None  # QPU only: None -> Ocean uniform-torque default
    token: str | None = None  # QPU only: else env DWAVE_API_TOKEN
    label: str = "acrpq-maneuver-qubo"

    def __post_init__(self) -> None:
        if not isinstance(self.mode, DWaveMode):
            raise TypeError("mode must be a DWaveMode")
        if isinstance(self.num_reads, bool) or not isinstance(self.num_reads, int) or self.num_reads < 1:
            raise ValueError("num_reads must be an integer >= 1")
        if isinstance(self.num_sweeps, bool) or not isinstance(self.num_sweeps, int) or self.num_sweeps < 1:
            raise ValueError("num_sweeps must be an integer >= 1")
        if self.annealing_time_us is not None and self.annealing_time_us <= 0:
            raise ValueError("annealing_time_us must be > 0")
        if self.chain_strength is not None and self.chain_strength <= 0:
            raise ValueError("chain_strength must be > 0")

    @property
    def kind(self) -> SolverKind:
        return (
            SolverKind.QUBO_DWAVE
            if self.mode is DWaveMode.QPU_REAL
            else SolverKind.QUBO_DWAVE_SA
        )

    @property
    def backend_label(self) -> str:
        if self.mode is DWaveMode.QPU_REAL:
            return self.solver_name or "dwave_qpu_default"
        return "dwave_neal_sa"


@dataclass
class DWaveResult:
    """Raw D-Wave outcome before ACRP decoding (parallels QAOAResult)."""

    best_bitstring: str  # bits[b] == value of qubit b, b = 0..n-1
    best_energy: float  # dimod energy == qubo.energy(best_bitstring)
    counts: dict[str, int] = field(default_factory=dict)
    num_reads: int = 0
    n_qubits: int = 0
    backend_name: str = ""
    is_qpu: bool = False
    wall_time_s: float = 0.0
    qpu_access_time_s: float | None = None
    chain_break_mean: float | None = None
    chain_break_max: float | None = None
    n_physical_qubits: int | None = None
    embedding_max_chain: int | None = None
    n_jobs: int = 1
    backend_meta: dict = field(default_factory=dict)


@dataclass
class DWaveAnnealingSolver:
    """Solve a :class:`ManeuverQUBO` on D-Wave (reference SA or a real QPU)."""

    config: DWaveConfig = field(default_factory=DWaveConfig)

    def to_bqm(self, qubo: ManeuverQUBO):
        """Build a dimod BQM whose variables ARE the QUBO's integer bit indices.

        Setting ``bqm.offset = qubo.constant`` makes ``bqm.energy(x)`` identical
        to ``qubo.energy(x)`` — no relabelling, so the sampled bitstring feeds
        straight into :func:`decode`.
        """
        import dimod

        bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
        for b in range(qubo.n_qubits):
            bqm.add_linear(b, float(qubo.linear.get(b, 0.0)))
        for (a, c), coeff in qubo.quadratic.items():
            bqm.add_quadratic(a, c, float(coeff))
        bqm.offset = float(qubo.constant)
        return bqm

    def solve_raw(self, qubo: ManeuverQUBO) -> DWaveResult:
        from .._optional import require

        require("dimod")
        t0 = time.perf_counter()
        bqm = self.to_bqm(qubo)
        if self.config.mode is DWaveMode.QPU_REAL:
            ss, meta, is_qpu = self._sample_qpu(bqm)
        else:
            ss, meta, is_qpu = self._sample_neal(bqm)
        wall = time.perf_counter() - t0
        raw = self._to_result(qubo, ss, meta, wall, is_qpu)
        # Load-bearing gate: bit order + offset reproduce H(x) exactly.
        if not math.isclose(
            raw.best_energy, qubo.energy(raw.best_bitstring), rel_tol=1e-6, abs_tol=1e-6
        ):
            raise RuntimeError("D-Wave energy != H(x) — bit-order/offset bug")
        return raw

    def solve(self, qubo: ManeuverQUBO) -> Result:
        raw = self.solve_raw(qubo)
        extra: dict[str, float] = {
            "qubo_energy": raw.best_energy,
            "num_reads": float(raw.num_reads),
            "is_qpu": 1.0 if raw.is_qpu else 0.0,
        }
        if raw.qpu_access_time_s is not None:
            extra["qpu_time_s"] = raw.qpu_access_time_s  # reuse IBM column
            extra["qpu_access_time_s"] = raw.qpu_access_time_s  # D-Wave-specific too
        for k, v in (
            ("chain_break_mean", raw.chain_break_mean),
            ("chain_break_max", raw.chain_break_max),
            ("n_physical_qubits", raw.n_physical_qubits),
            ("embedding_max_chain", raw.embedding_max_chain),
        ):
            if v is not None:
                extra[k] = float(v)
        return decode(
            qubo,
            raw.best_bitstring,
            kind=self.config.kind,
            wall_time_s=raw.wall_time_s,
            seed=self.config.seed,
            backend=raw.backend_name,
            shots=raw.num_reads,  # num_reads is the shot analogue
            extra=extra,
        )

    # ---- SA path: no token, always works --------------------------------- #
    def _sample_neal(self, bqm):
        from .._optional import require

        require("dwave")
        from dwave.samplers import SimulatedAnnealingSampler

        ss = SimulatedAnnealingSampler().sample(
            bqm,
            num_reads=self.config.num_reads,
            num_sweeps=self.config.num_sweeps,
            seed=self.config.seed,
        )
        meta = {
            "name": "dwave_neal_sa",
            "is_simulator": True,
            "provider": "dwave-samplers",
            "solver": "SimulatedAnnealingSampler",
        }
        return ss, meta, False

    # ---- Real QPU path: Leap token, minor-embedding, chain metrics ------- #
    def _sample_qpu(self, bqm):
        from .._optional import require

        require("dwave")
        require("minorminer")
        from dwave.system import DWaveSampler, EmbeddingComposite

        from ..io.env import load_dotenv

        load_dotenv()  # non-override: a real env var wins
        token = self.config.token or os.environ.get("DWAVE_API_TOKEN")
        base_kwargs: dict = {}
        if token:
            base_kwargs["token"] = token
        if self.config.solver_name:
            base_kwargs["solver"] = self.config.solver_name
        base = DWaveSampler(**base_kwargs)  # raises SolverAuthenticationError w/o token
        sampler = EmbeddingComposite(base)
        skw: dict = {
            "num_reads": self.config.num_reads,
            "return_embedding": True,
            "chain_break_fraction": True,
            "label": self.config.label,
        }
        if self.config.chain_strength is not None:
            skw["chain_strength"] = self.config.chain_strength
        if self.config.annealing_time_us is not None:
            skw["annealing_time"] = self.config.annealing_time_us
        ss = sampler.sample(bqm, **skw)
        ss.resolve()  # block until results before reading timing/embedding
        topo = base.properties.get("topology", {}).get("type", "unknown")
        meta = {
            "name": getattr(getattr(base, "solver", None), "id", self.config.backend_label),
            "is_simulator": False,
            "provider": "dwave-system",
            "topology": topo,
            "chip_n_qubits": len(base.nodelist),
        }
        return ss, meta, True

    # ---- SampleSet -> DWaveResult ---------------------------------------- #
    def _to_result(self, qubo, ss, meta, wall, is_qpu) -> DWaveResult:
        n = qubo.n_qubits

        def _bits(sample: dict) -> str:
            return "".join(str(int(sample.get(b, 0))) for b in range(n))

        first = ss.first
        best_bitstring = _bits(dict(first.sample))
        best_energy = float(first.energy)

        counts: dict[str, int] = {}
        for rec in ss.data(fields=["sample", "energy", "num_occurrences"]):
            counts[_bits(dict(rec.sample))] = counts.get(
                _bits(dict(rec.sample)), 0
            ) + int(rec.num_occurrences)

        res = DWaveResult(
            best_bitstring=best_bitstring,
            best_energy=best_energy,
            counts=counts,
            num_reads=int(self.config.num_reads),
            n_qubits=n,
            backend_name=meta.get("name", self.config.backend_label),
            is_qpu=is_qpu,
            wall_time_s=wall,
            backend_meta=meta,
        )
        if not is_qpu:
            return res

        # --- QPU-only telemetry ---
        info = getattr(ss, "info", {}) or {}
        timing = info.get("timing", {}) or {}
        qat = timing.get("qpu_access_time")  # microseconds
        if qat is not None:
            res.qpu_access_time_s = float(qat) * 1e-6

        # chain-break fraction: occurrence-weighted mean + max over reads
        try:
            cbf = ss.record.chain_break_fraction
            occ = ss.record.num_occurrences
            total = int(occ.sum())
            if total > 0:
                res.chain_break_mean = float((cbf * occ).sum() / total)
                res.chain_break_max = float(cbf.max())
        except (AttributeError, ValueError):
            pass

        # embedding size / longest chain
        emb = info.get("embedding_context", {}).get("embedding")
        if emb:
            chains = [len(v) for v in emb.values()]
            res.n_physical_qubits = int(sum(chains))
            res.embedding_max_chain = int(max(chains)) if chains else None
        return res
