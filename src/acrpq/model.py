"""Shared data model for the ACRP benchmark.

Every constant here mirrors ``Code/NC_ACRP.mod`` / ``Code/NC_ACRP.run`` exactly
so that the Python pipeline reproduces the original AMPL formulation:

* ``PI = 3.141592`` is the **literal** from ``NC_ACRP.mod`` line 11 (not
  ``math.pi``). The circle-default coordinates for instances that omit
  ``x0``/``y0`` are reconstructed with this literal, so parsed geometry matches
  what AMPL would compute. The discrepancy versus ``math.pi`` is ~1e-6 and
  intentional.
* Control bounds ``q in [0.94, 1.03]`` and ``theta in [-PI/6, +PI/6]`` come from
  lines 49-52.
* ``w = 0.5`` (objective weight) is set in ``NC_ACRP.run``, not in any ``.dat``.

All containers are frozen, slotted dataclasses. Aircraft use **1-based** ids in
the public API (matching the AMPL set ``A := 1..n``) while being stored
internally as parallel tuples indexed ``0..n-1``. Unordered pairs ``(i, j)``
with ``i < j`` correspond to the AMPL set ``P``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

# --------------------------------------------------------------------------- #
# Constants — mirror NC_ACRP.mod / NC_ACRP.run exactly
# --------------------------------------------------------------------------- #
PI: float = 3.141592  # NC_ACRP.mod line 11 (literal, NOT math.pi)
QMIN, QMAX = 0.94, 1.03  # speed-rate bounds (lines 49-50)
HMIN, HMAX = -PI / 6, +PI / 6  # heading-change bounds (lines 51-52)
DEFAULT_D = 0.05  # separation norm used by every library instance
DEFAULT_RADIUS = 2.0
DEFAULT_W = 0.5  # objective weight, from NC_ACRP.run


class Family(str, Enum):
    """Instance family. Mirrors the ``Data/`` archive layout."""

    CP = "CP"  # Circle Problem
    FP = "FP"  # Flow Problem
    GP = "GP"  # Grid Problem
    RCP = "RCP"  # Random Circle Problem (2D)
    RCP_FL = "RCP_FL"  # Random Circle Problem with flight levels


class SolverKind(str, Enum):
    """Which solver produced a :class:`Result`."""

    CLASSICAL_MINLP = "classical_minlp"  # Pyomo/AMPL faithful NC_ACRP (continuous)
    CLASSICAL_ORIGINAL_AMPL = "original_ampl"  # direct run of Code/NC_ACRP.mod via AMPL+Couenne
    CLASSICAL_DISCRETE = "classical_discrete"  # exact brute force on the grid — always runnable
    QUBO_EXACT = "qubo_exact"  # exact minimisation of the QUBO energy H(x)
    QUBO_ANNEALING = "qubo_annealing"  # classical simulated annealing on H(x)
    QUANTUM_QAOA = "quantum_qaoa"  # QAOA over the maneuver QUBO
    QUBO_DWAVE = "qubo_dwave"  # quantum annealing on H(x) via a real D-Wave QPU
    QUBO_DWAVE_SA = "qubo_dwave_sa"  # D-Wave reference simulated-annealing sampler on H(x)


class Status(str, Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    ERROR = "error"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"  # solver/binary not installed — honest, not a failure


# --------------------------------------------------------------------------- #
# Instance — the common INPUT for every solver
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Instance:
    """A parsed ACRP instance.

    Speeds/headings/coordinates are stored 0-based: ``v0[i - 1]`` is the AMPL
    ``v0[i]`` for aircraft id ``i in 1..n``. ``x0``/``y0`` are *always*
    materialised — when an instance file omits them (the RCP_FL family) the
    loader reconstructs the circle defaults from ``NC_ACRP.mod`` lines 17-18.
    """

    name: str
    family: Family
    n: int
    d: float
    radius: float
    v0: tuple[float, ...]
    cap: tuple[float, ...]  # raw initial heading (pre-normalisation)
    x0: tuple[float, ...]
    y0: tuple[float, ...]
    w: float = DEFAULT_W
    nf: int | None = None  # number of flight levels (RCP_FL only)
    l0: tuple[int, ...] = ()  # initial flight level per aircraft (RCP_FL only)
    level: float = 0.0
    source_path: str | None = None  # provenance, e.g. "RCP_Instances.zip::RCP_10_1.dat"
    qmin: float = QMIN
    qmax: float = QMAX
    hmin: float = HMIN
    hmax: float = HMAX

    @property
    def theta0(self) -> tuple[float, ...]:
        """Normalised initial heading (``NC_ACRP.mod`` line 16)."""
        return tuple(c - 2 * PI if c >= PI else c for c in self.cap)

    @property
    def has_flight_levels(self) -> bool:
        return self.nf is not None and len(self.l0) == self.n

    def aircraft(self) -> range:
        """1-based aircraft ids (AMPL set ``A``)."""
        return range(1, self.n + 1)

    def pairs(self) -> tuple[tuple[int, int], ...]:
        """Unordered aircraft pairs ``i < j`` (AMPL set ``P``)."""
        ids = list(self.aircraft())
        return tuple((i, j) for a, i in enumerate(ids) for j in ids[a + 1 :])

    def n_pairs(self) -> int:
        return self.n * (self.n - 1) // 2

    def subset(self, ids: Sequence[int]) -> "Instance":
        """Return a new instance containing only ``ids`` (1-based), renumbered.

        Used to carve quantum-tractable sub-scenarios out of large instances
        without mutating the original problem definition.
        """
        ids = list(ids)
        if not ids:
            raise ValueError("aircraft subset must not be empty")
        if len(set(ids)) != len(ids):
            raise ValueError("aircraft subset ids must be unique")
        for i in ids:
            if not 1 <= i <= self.n:
                raise ValueError(f"aircraft id {i} out of range 1..{self.n}")
        sel = [i - 1 for i in ids]  # 0-based
        return Instance(
            name=f"{self.name}[{','.join(map(str, ids))}]",
            family=self.family,
            n=len(ids),
            d=self.d,
            radius=self.radius,
            v0=tuple(self.v0[k] for k in sel),
            cap=tuple(self.cap[k] for k in sel),
            x0=tuple(self.x0[k] for k in sel),
            y0=tuple(self.y0[k] for k in sel),
            w=self.w,
            nf=self.nf,
            l0=tuple(self.l0[k] for k in sel) if self.l0 else (),
            level=self.level,
            source_path=self.source_path,
            qmin=self.qmin,
            qmax=self.qmax,
            hmin=self.hmin,
            hmax=self.hmax,
        )


# --------------------------------------------------------------------------- #
# Maneuver discretisation — shared by the discrete classical solver AND quantum
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Maneuver:
    """A chosen control for a single aircraft."""

    q: float  # speed rate in [QMIN, QMAX]
    theta: float  # heading change in [HMIN, HMAX]


@dataclass(frozen=True, slots=True)
class ManeuverOption:
    """One cell of the maneuver grid, with a stable index and label."""

    idx: int
    q: float
    theta: float
    label: str

    @property
    def is_noop(self) -> bool:
        return self.q == 1.0 and self.theta == 0.0

    @property
    def maneuver(self) -> Maneuver:
        return Maneuver(self.q, self.theta)


def _snap_to(values: list[float], target: float) -> list[float]:
    """Snap the entry of ``values`` nearest ``target`` to exactly ``target``.

    The maneuver grid is an approximation, so we always snap the closest level
    to ``target`` (guaranteeing e.g. an exact NOOP exists) rather than only when
    it is already within a tolerance.
    """
    if not values:
        return values
    k = min(range(len(values)), key=lambda i: abs(values[i] - target))
    values[k] = target
    return values


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    if n <= 1:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + step * i for i in range(n)]


@dataclass(frozen=True, slots=True)
class ManeuverGrid:
    """The discrete set of maneuver options offered to every aircraft.

    The grid is the cartesian product of speed-rate levels and heading-change
    levels. The deterministic option order (q outer, theta inner) is
    *load-bearing*: it defines the qubit indexing in the QUBO.
    """

    q_levels: tuple[float, ...]
    theta_levels: tuple[float, ...]
    options: tuple[ManeuverOption, ...]

    @property
    def k(self) -> int:
        return len(self.options)

    def cost(self, opt: ManeuverOption, w: float) -> float:
        """Per-aircraft objective term (``NC_ACRP.mod`` line 67)."""
        return w * opt.theta**2 + (1.0 - w) * (1.0 - opt.q) ** 2

    def noop_index(self) -> int:
        for opt in self.options:
            if opt.is_noop:
                return opt.idx
        raise ValueError("grid has no NOOP option")  # build() guarantees this

    @staticmethod
    def build(
        *, n_theta: int = 5, n_q: int = 1, instance: Instance | None = None
    ) -> "ManeuverGrid":
        """Construct a grid inside the control box, guaranteeing a NOOP option.

        Parameters
        ----------
        n_theta:
            Number of heading-change levels across ``[HMIN, HMAX]``.
        n_q:
            Number of speed-rate levels across ``[QMIN, QMAX]`` (``1`` keeps
            speed fixed, the cheapest meaningful, heading-only model).
        instance:
            If given, its control bounds are used instead of the global
            defaults.
        """
        if (
            isinstance(n_theta, bool)
            or not isinstance(n_theta, int)
            or isinstance(n_q, bool)
            or not isinstance(n_q, int)
            or n_theta < 1
            or n_q < 1
        ):
            raise ValueError("n_theta and n_q must be integers >= 1")
        hmin = instance.hmin if instance else HMIN
        hmax = instance.hmax if instance else HMAX
        qmin = instance.qmin if instance else QMIN
        qmax = instance.qmax if instance else QMAX
        bounds = (hmin, hmax, qmin, qmax)
        if not all(math.isfinite(value) for value in bounds):
            raise ValueError("maneuver bounds must be finite")
        if hmin > hmax or qmin > qmax:
            raise ValueError("maneuver lower bounds must not exceed upper bounds")
        if not hmin <= 0.0 <= hmax or not qmin <= 1.0 <= qmax:
            raise ValueError("maneuver bounds must contain the NOOP control (q=1, theta=0)")

        thetas = _snap_to(_linspace(hmin, hmax, n_theta), 0.0)
        qs = [1.0] if n_q == 1 else _snap_to(_linspace(qmin, qmax, n_q), 1.0)

        options: list[ManeuverOption] = []
        idx = 0
        for q in qs:  # outer
            for th in thetas:  # inner
                noop = q == 1.0 and th == 0.0
                label = "NOOP" if noop else f"q{q:.4f}_t{th:+.4f}"
                options.append(ManeuverOption(idx=idx, q=q, theta=th, label=label))
                idx += 1

        grid = ManeuverGrid(
            q_levels=tuple(qs), theta_levels=tuple(thetas), options=tuple(options)
        )
        # Invariant: NOOP must exist (it is the guaranteed-feasible fallback for
        # the reference solver and the one-hot repair target for decoding).
        grid.noop_index()
        return grid


# --------------------------------------------------------------------------- #
# Solution / Result — the common OUTPUT every solver returns
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Solution:
    """A resolution proposal: per-aircraft control plus scored quality.

    ``objective`` and ``n_conflicts`` are always computed by the single
    geometry kernel on the *continuous* maneuvers, so a discrete/quantum
    solution and a continuous MINLP solution are scored identically.
    """

    q: tuple[float, ...]
    theta: tuple[float, ...]
    choice: tuple[int, ...] = ()  # grid option idx per aircraft (discrete/quantum); () for MINLP
    objective: float = math.inf
    n_conflicts: int = -1
    feasible: bool = False


@dataclass(frozen=True, slots=True)
class Result:
    """One solve of one instance by one solver — the common interface output."""

    instance_name: str
    solver: SolverKind
    status: Status
    solution: Solution | None
    objective: float = math.inf
    n_conflicts: int = -1
    n_initial_conflicts: int = -1
    feasible: bool = False
    wall_time_s: float = 0.0
    seed: int | None = None
    versions: dict[str, str] = field(default_factory=dict)
    # quantum-only fields (None / absent for classical solvers)
    backend: str | None = None
    shots: int | None = None
    qaoa_reps: int | None = None
    n_qubits: int | None = None
    qubo_penalty: float | None = None
    message: str = ""
    extra: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """One classical-vs-quantum comparison row."""

    instance_name: str
    family: str
    n: int
    n_pairs: int
    grid_k: int
    classical: Result  # CLASSICAL_DISCRETE — the ground-truth anchor
    quantum: Result
    objective_gap: float = math.inf
    both_feasible: bool = False
    speedup: float = math.nan
