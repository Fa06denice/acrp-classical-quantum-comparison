"""Post-process a QUBO bitstring back into an ACRP :class:`Result` (pure Python).

This closes the brief's loop: a QAOA bitstring is decoded into a per-aircraft
maneuver assignment and **re-scored with the same geometry kernel** used for the
classical solvers, so quantum and classical results are directly comparable.

Decoding repairs one-hot violations from valid binary samples. Structurally
malformed inputs (wrong length, non-binary values, out-of-range indices) raise
``ValueError`` instead of being silently reinterpreted.
"""

from __future__ import annotations

import math

from .. import geometry
from ..model import Result, Solution, SolverKind, Status
from ..objectives import option_cost_vector
from .qubo import ManeuverQUBO, _as_bit_dict
from ..selection_policy import choose_option


def decode_assignment(qubo: ManeuverQUBO, bits) -> dict[int, int]:
    """Map a bit assignment to ``{aircraft_id: option_idx}`` with one-hot repair."""
    x = _as_bit_dict(bits, qubo.n_qubits)
    discrete = qubo.discrete
    inst = discrete.instance
    k = discrete.k
    noop = discrete.grid.noop_index()
    # per-option cost under THIS qubo's objective (not the historical quadratic cost)
    obj_cost = option_cost_vector(qubo.objective_id, discrete.grid, inst.w)

    assignment: dict[int, int] = {}
    for i in inst.aircraft():
        set_opts = [o for o in range(k) if x.get(discrete.var_index(i, o), 0)]
        if len(set_opts) == 1:
            assignment[i] = set_opts[0]
        elif not set_opts:
            assignment[i] = noop  # no choice -> do nothing
        else:
            # Multiple bits set -> repair. Two distinct steps, deliberately not merged:
            #   1. OPTIMALITY, decided solely by the run's own objective: keep only the
            #      candidates of exactly minimal cost under `qubo.objective_id`. The old
            #      rule used the quadratic grid cost whatever the objective_id, so at
            #      w in {0, 1} (reachable via the API) it could return a non-NOOP option
            #      and inflate the deviated-aircraft count.
            #   2. TIE-BREAK, applied ONLY to that already-equal set, by the versioned
            #      selection policy. Consulting the policy on the full candidate set
            #      would let it override the objective — at w = 0 the quadratic cost
            #      ignores theta, so a zero-cost option can carry a large theta and the
            #      policy's smallest-variation criterion would pick a costlier option.
            #      Step 1 makes that impossible: the repair is never worse under the
            #      primary objective than the objective-minimal choice.
            best = min(obj_cost[o] for o in set_opts)
            tied = [o for o in set_opts if obj_cost[o] == best]
            assignment[i] = choose_option(discrete.grid, tied)
    return assignment


def decode(
    qubo: ManeuverQUBO,
    bits,
    *,
    kind: SolverKind = SolverKind.QUANTUM_QAOA,
    wall_time_s: float = 0.0,
    status: Status | None = None,
    seed: int | None = None,
    backend: str | None = None,
    shots: int | None = None,
    qaoa_reps: int | None = None,
    versions: dict[str, str] | None = None,
    extra: dict[str, float] | None = None,
    message: str = "",
) -> Result:
    """Decode ``bits`` into a scored :class:`Result` via the geometry kernel."""
    discrete = qubo.discrete
    inst = discrete.instance
    assignment = decode_assignment(qubo, bits)
    choice = tuple(assignment[i] for i in inst.aircraft())
    q, theta = discrete.maneuvers(choice)

    n_conf = geometry.count_conflicts(inst, q, theta)
    obj = geometry.objective(inst, q, theta)
    n_initial = geometry.count_initial_conflicts(inst)
    feasible = n_conf == 0

    sol = Solution(
        q=q, theta=theta, choice=choice,
        objective=obj, n_conflicts=n_conf, feasible=feasible,
    )
    if status is None:
        status = Status.FEASIBLE if feasible else Status.INFEASIBLE

    return Result(
        instance_name=inst.name,
        solver=kind,
        status=status,
        solution=sol,
        objective=obj,
        n_conflicts=n_conf,
        n_initial_conflicts=n_initial,
        feasible=feasible,
        wall_time_s=wall_time_s,
        seed=seed,
        versions=versions or {},
        backend=backend,
        shots=shots,
        qaoa_reps=qaoa_reps,
        n_qubits=qubo.n_qubits,
        qubo_penalty=qubo.penalty_conflict,
        message=message,
        extra=extra or {},
    )


def best_feasible_bitstring(qubo: ManeuverQUBO, counts: dict[str, int]) -> tuple[str | None, float]:
    """From sampled ``counts``, return the lowest-QUBO-energy bitstring that
    decodes to a feasible (conflict-free) ACRP solution, plus its probability.

    Returns ``(None, 0.0)`` if no sampled bitstring is feasible.
    """
    total = sum(counts.values()) or 1
    best: tuple[str | None, float, float] = (None, math.inf, 0.0)
    for bitstring, c in counts.items():
        assignment = decode_assignment(qubo, bitstring)
        inst = qubo.discrete.instance
        choice = tuple(assignment[i] for i in inst.aircraft())
        q, theta = qubo.discrete.maneuvers(choice)
        if geometry.count_conflicts(inst, q, theta) == 0:
            energy = qubo.energy(bitstring)
            if energy < best[1]:
                best = (bitstring, energy, c / total)
    return best[0], best[2]
