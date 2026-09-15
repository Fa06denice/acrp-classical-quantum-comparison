"""Classical solvers that operate directly on the QUBO energy ``H(x)``.

These put the **QUBO formulation itself** at the centre of the study, rather
than QAOA. They minimise the same Hamiltonian
``H(x) = const + Σ a_b x_b + Σ q_{bc} x_b x_c`` that QAOA targets, so all three
methods (exact, simulated annealing, QAOA) are compared on the *identical*
QUBO and decoded through the *identical* geometry kernel.

* :class:`QuboExactSolver` — exact minimisation of ``H(x)``. By default it
  exploits the one-hot block structure (one option per aircraft) and searches
  ``K**n`` assignments; with ``brute_force_bits=True`` it enumerates all
  ``2**n_qubits`` bitstrings (used to *prove* the QUBO optimum equals the
  one-hot optimum on tiny cases). Pure Python, no Qiskit.
* :class:`QuboAnnealingSolver` — classical **simulated annealing** on ``H(x)``,
  the classical analogue of quantum annealing (e.g. D-Wave). Pure Python,
  deterministic given a seed.

Both return the package's common :class:`~acrpq.model.Result`, scored on the
continuous geometry of the chosen maneuvers, so they slot into the benchmark
next to the classical and QAOA solvers.
"""

from __future__ import annotations

import itertools
import math
import random
import time
from dataclasses import dataclass

from ..model import Result, SolverKind
from .decode import decode
from .qubo import ManeuverQUBO


def _energy_from_choice(qubo: ManeuverQUBO, choice: tuple[int, ...]) -> float:
    """Energy of a one-hot assignment given a per-aircraft option choice."""
    d = qubo.discrete
    bits = {d.var_index(i, choice[i - 1]): 1 for i in d.instance.aircraft()}
    return qubo.energy(bits)


@dataclass
class QuboExactSolver:
    """Exact minimiser of ``H(x)`` over the explicitly enumerated domain.

    Parameters
    ----------
    brute_force_bits:
        If ``True``, enumerate all ``2**n_qubits`` bitstrings (only feasible for
        tiny problems; demonstrates the QUBO optimum is one-hot). If ``False``
        (default), enumerate the ``K**n`` one-hot assignments — far cheaper and
        the relevant optimum for the ACRP encoding.
    max_states:
        Safety cap on the number of states enumerated.
    """

    brute_force_bits: bool = False
    max_states: int = 5_000_000

    def __post_init__(self) -> None:
        if isinstance(self.max_states, bool) or not isinstance(self.max_states, int):
            raise ValueError("max_states must be an integer >= 1")
        if self.max_states < 1:
            raise ValueError("max_states must be an integer >= 1")

    def solve(self, qubo: ManeuverQUBO) -> Result:
        t0 = time.perf_counter()
        d = qubo.discrete
        n, k = d.n, d.k

        if self.brute_force_bits:
            states = 2**qubo.n_qubits
            if states > self.max_states:
                raise ValueError(
                    f"brute_force_bits would enumerate 2**{qubo.n_qubits} states; "
                    f"exceeds max_states={self.max_states}"
                )
            best_bits: tuple[int, ...] | None = None
            best_e = math.inf
            for combo in itertools.product((0, 1), repeat=qubo.n_qubits):
                e = qubo.energy(list(combo))
                if e < best_e:
                    best_e, best_bits = e, combo
            if best_bits is None:  # product() always yields for a non-negative repeat
                raise RuntimeError("exact bit search produced no candidate")
            return decode(
                qubo, list(best_bits), kind=SolverKind.QUBO_EXACT,
                wall_time_s=time.perf_counter() - t0, backend="qubo_exact_bits",
                extra={
                    "qubo_energy": best_e,
                    "states_enumerated": float(states),
                    "global_unconstrained_qubo": 1.0,
                },
            )

        # one-hot block search (K**n)
        states = k**n if n else 1
        if states > self.max_states:
            raise ValueError(
                f"exact one-hot search would enumerate {k}**{n} states; "
                f"exceeds max_states={self.max_states}. Use the annealing solver."
            )
        best_choice: tuple[int, ...] | None = None
        best_e = math.inf
        for choice in itertools.product(range(k), repeat=n):
            e = _energy_from_choice(qubo, choice)
            if e < best_e:
                best_e, best_choice = e, choice
        if best_choice is None:  # product() always yields for non-negative n/k
            raise RuntimeError("exact one-hot search produced no candidate")
        bits = {d.var_index(i, best_choice[i - 1]): 1 for i in d.instance.aircraft()}
        return decode(
            qubo, bits, kind=SolverKind.QUBO_EXACT,
            wall_time_s=time.perf_counter() - t0, backend="qubo_exact",
            extra={
                "qubo_energy": best_e,
                "states_enumerated": float(states),
                "global_unconstrained_qubo": 0.0,
            },
        )


@dataclass
class QuboAnnealingSolver:
    """Classical simulated annealing on the QUBO energy ``H(x)``.

    Works on one-hot states (each aircraft holds exactly one option) and
    proposes single-aircraft option changes, accepting worse moves with the
    Metropolis criterion under a geometric cooling schedule. This is the
    classical counterpart of quantum annealing; it never violates the one-hot
    constraint by construction, so the conflict/cost penalties drive the search.
    """

    n_sweeps: int = 400
    restarts: int = 8
    t_start: float = 5.0
    t_end: float = 0.05
    seed: int = 1234

    def __post_init__(self) -> None:
        for name, value in (("n_sweeps", self.n_sweeps), ("restarts", self.restarts)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if not math.isfinite(self.t_start) or self.t_start <= 0:
            raise ValueError("t_start must be finite and > 0")
        if not math.isfinite(self.t_end) or self.t_end <= 0:
            raise ValueError("t_end must be finite and > 0")

    def solve(self, qubo: ManeuverQUBO) -> Result:
        t0 = time.perf_counter()
        d = qubo.discrete
        n, k = d.n, d.k
        rng = random.Random(self.seed)
        noop = d.grid.noop_index()

        best_choice: tuple[int, ...] | None = None
        best_e = math.inf
        for r in range(self.restarts):
            # start from NOOP on the first restart, random thereafter
            if r == 0:
                choice = [noop] * n
            else:
                choice = [rng.randrange(k) for _ in range(n)]
            cur_e = _energy_from_choice(qubo, tuple(choice))

            for sweep in range(self.n_sweeps):
                frac = sweep / max(1, self.n_sweeps - 1)
                temp = self.t_start * (self.t_end / self.t_start) ** frac
                for i in range(1, n + 1):
                    old = choice[i - 1]
                    new = rng.randrange(k)
                    if new == old:
                        continue
                    choice[i - 1] = new
                    new_e = _energy_from_choice(qubo, tuple(choice))
                    delta = new_e - cur_e
                    if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
                        cur_e = new_e  # accept
                    else:
                        choice[i - 1] = old  # reject
            if cur_e < best_e:
                best_e, best_choice = cur_e, tuple(choice)

        if best_choice is None:  # validated restarts >= 1
            raise RuntimeError("annealing produced no candidate")
        bits = {d.var_index(i, best_choice[i - 1]): 1 for i in d.instance.aircraft()}
        return decode(
            qubo, bits, kind=SolverKind.QUBO_ANNEALING,
            wall_time_s=time.perf_counter() - t0, seed=self.seed,
            backend="simulated_annealing",
            extra={"qubo_energy": best_e, "restarts": float(self.restarts),
                   "sweeps": float(self.n_sweeps)},
        )
