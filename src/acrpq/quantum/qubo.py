"""Build a QUBO from the discrete ACRP (pure Python, no Qiskit).

Binary variable ``x[i, o]`` = 1 iff aircraft ``i`` takes maneuver option ``o``;
bit index ``b = (i-1)*K + o`` (see :mod:`acrpq.discretize`).

Energy::

    H(x) =  sum_{i,o} cost_o * x[i,o]                                   # maneuver cost
          + lam_pen * sum_{(i,j)} sum_{oi,oj : conflict} x[i,oi] x[j,oj]  # conflict penalty
          + lam_oh  * sum_i ( sum_o x[i,o] - 1 )^2                       # one-hot constraint

One-hot expansion ``(sum_o x - 1)^2 = -sum_o x + 2 sum_{o<o'} x_o x_{o'} + 1``
contributes ``-lam_oh`` to each linear term, ``+2 lam_oh`` to each same-aircraft
pair, and ``+lam_oh`` to the constant per aircraft.

Penalty weights (recorded for reproducibility)::

    cost_max = max_o cost_o
    lam_pen  = pen_scale * (n * cost_max + 1)        # one avoided conflict beats all cost
    lam_oh   = oh_scale  * lam_pen,  oh_scale >= n   # one-hot dominates conflicts

The ``oh_scale >= n`` (more precisely ``lam_oh > lam_pen * max conflict degree``)
guarantees the QUBO global minimum is always a valid one-hot assignment — it
never pays to drop an aircraft to escape its conflicts — so QAOA bitstrings stay
interpretable even when no conflict-free assignment exists in the coarse grid.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from ..discretize import DiscreteACRP
from ..exceptions import QubitBudgetError
from ..model import Instance, ManeuverGrid
from ..objectives import DEFAULT_OBJECTIVE, ObjectiveId, coerce_objective_id, option_cost_vector


@dataclass(slots=True, frozen=True)
class ManeuverQUBO:
    """A QUBO in dict form, plus the discrete model it came from.

    Frozen: once built (and therefore hashable/persistable) its fields — above
    all its ``objective_id`` identity — can never be silently reassigned.
    """

    discrete: DiscreteACRP
    linear: Mapping[int, float]  # bit -> coeff (read-only after construction)
    quadratic: Mapping[tuple[int, int], float]  # (a, b), a < b -> coeff (read-only)
    constant: float
    n_qubits: int
    penalty_onehot: float
    penalty_conflict: float
    meta: Mapping[str, float] = field(default_factory=dict)
    # identity of the optimisation objective this QUBO encodes; travels with
    # every result/artifact/hash so objectives are never mixed.
    objective_id: str = DEFAULT_OBJECTIVE.value

    def __post_init__(self) -> None:
        # (identity) normalise + validate: an unknown objective_id is refused
        # here (fail-closed), and the stored value is always the canonical string.
        object.__setattr__(
            self, "objective_id", coerce_objective_id(self.objective_id).value
        )
        # (immutability) take defensive COPIES wrapped read-only, so a
        # caller-owned dict passed in — or later mutated — can never alter this
        # already-built, hash-bound QUBO. The dataclass is frozen (no rebinding)
        # and its coefficient/meta maps are now MappingProxyType (no in-place
        # mutation). Together this makes the hashed content truly immutable.
        object.__setattr__(self, "linear", MappingProxyType(dict(self.linear)))
        object.__setattr__(self, "quadratic", MappingProxyType(dict(self.quadratic)))
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))

    @property
    def n_conflict_terms(self) -> int:
        return int(self.meta.get("n_conflict_terms", 0))

    @property
    def n_onehot_terms(self) -> int:
        return int(self.meta.get("n_onehot_terms", 0))

    def energy(self, bits: Mapping[int, int] | Sequence[int] | str) -> float:
        """Evaluate ``H(x)`` for a bit assignment (dict, list, or bitstring)."""
        x = _as_bit_dict(bits, self.n_qubits)
        e = self.constant
        for b, coeff in self.linear.items():
            if x.get(b, 0):
                e += coeff
        for (a, c), coeff in self.quadratic.items():
            if x.get(a, 0) and x.get(c, 0):
                e += coeff
        return e


def _as_bit_dict(bits, n: int) -> dict[int, int]:
    """Normalise a binary assignment without silently accepting malformed data."""
    if n < 0:
        raise ValueError("n must be >= 0")
    if isinstance(bits, Mapping):
        out: dict[int, int] = {}
        for b, value in bits.items():
            if isinstance(b, bool) or not isinstance(b, int) or not 0 <= b < n:
                raise ValueError(f"bit index {b!r} out of range 0..{n - 1}")
            if value not in (0, 1, False, True):
                raise ValueError(f"bit {b} must be binary, got {value!r}")
            if value:
                out[b] = 1
        return out
    if isinstance(bits, str):
        # Convention: bits[b] is the value of qubit b (left-to-right == bit 0..).
        if len(bits) != n or set(bits) - {"0", "1"}:
            raise ValueError(f"bitstring must contain exactly {n} binary characters")
        return {b: 1 for b, ch in enumerate(bits) if ch == "1"}
    values = list(bits)
    if len(values) != n:
        raise ValueError(f"bit sequence must have length {n}, got {len(values)}")
    if any(value not in (0, 1, False, True) for value in values):
        raise ValueError("bit sequence values must all be binary")
    return {b: 1 for b, value in enumerate(values) if value}


def _max_conflict_degree(discrete: DiscreteACRP) -> int:
    """Largest number of other aircraft any single aircraft can conflict with."""
    deg: dict[int, int] = {}
    for i, j in discrete.instance.pairs():
        deg[i] = deg.get(i, 0) + 1
        deg[j] = deg.get(j, 0) + 1
    return max(deg.values(), default=0)


def build_qubo(
    inst: Instance,
    grid: ManeuverGrid | None = None,
    *,
    discrete: DiscreteACRP | None = None,
    n_theta: int = 5,
    n_q: int = 1,
    pen_scale: float = 10.0,
    oh_scale: float | None = None,
    max_qubits: int = 29,
    penalty_conflict: float | None = None,
    penalty_onehot: float | None = None,
    objective_id: ObjectiveId | str = DEFAULT_OBJECTIVE,
) -> ManeuverQUBO:
    """Construct the maneuver QUBO for ``inst``.

    Parameters
    ----------
    grid / discrete:
        Provide a pre-built grid or discrete model, else one is built from
        ``n_theta``/``n_q``.
    pen_scale:
        Multiplier for the conflict penalty relative to maneuver cost.
    oh_scale:
        Multiplier for the one-hot penalty relative to the conflict penalty.
        Defaults to ``max(n, max_conflict_degree + 1)`` so the QUBO optimum is
        guaranteed one-hot.
    max_qubits:
        Refuse to build a QUBO needing more than this many qubits.
    penalty_conflict / penalty_onehot:
        Override the computed penalties explicitly (for sensitivity studies).
    """
    if max_qubits < 0:
        raise ValueError("max_qubits must be >= 0")
    if grid is not None and discrete is not None:
        raise ValueError("provide either grid or discrete, not both")
    if discrete is None:
        if grid is None:
            grid = ManeuverGrid.build(n_theta=n_theta, n_q=n_q, instance=inst)
        discrete = DiscreteACRP.build(inst, grid)
    elif discrete.instance != inst:
        raise ValueError("discrete model was built for a different instance")

    n_qubits = discrete.n_vars
    if n_qubits > max_qubits:
        raise QubitBudgetError(
            f"QUBO for {inst.name} needs {n_qubits} qubits "
            f"(n={inst.n} x K={discrete.k}) > max_qubits={max_qubits}. "
            f"Reduce the grid (n_theta/n_q), sub-sample aircraft, or raise max_qubits."
        )

    oid = coerce_objective_id(objective_id)
    # Linear term = the SELECTED objective's per-option cost. For the historical
    # quadratic objective this equals discrete.cost exactly (see discretize.py);
    # for maneuver_count every non-NOOP option costs 1. Penalties are unchanged
    # and auto-scale to this objective's cost_max, preserving one-hot dominance.
    cost = option_cost_vector(oid, discrete.grid, inst.w)
    cost_max = max(cost) if cost else 0.0
    maxdeg = _max_conflict_degree(discrete)
    lam_pen = penalty_conflict if penalty_conflict is not None else pen_scale * (inst.n * cost_max + 1.0)
    if penalty_onehot is not None:
        lam_oh = penalty_onehot
    else:
        scale = oh_scale if oh_scale is not None else max(inst.n, maxdeg + 1)
        lam_oh = scale * lam_pen
    if not math.isfinite(lam_pen) or lam_pen <= 0:
        raise ValueError("conflict penalty must be finite and > 0")
    if not math.isfinite(lam_oh) or lam_oh <= 0:
        raise ValueError("one-hot penalty must be finite and > 0")
    # Whether the penalties still guarantee a valid one-hot global optimum:
    #   lam_pen > n*cost_max      -> avoiding one conflict beats all maneuver cost
    #   lam_oh  > lam_pen*maxdeg  -> one-hot never pays to drop an aircraft
    # Auto-computed penalties always satisfy this; an explicit override MAY not,
    # in which case the QUBO is a non-guaranteed sensitivity experiment (flagged,
    # never silently presented as guaranteed).
    onehot_guaranteed = lam_pen > inst.n * cost_max and lam_oh > lam_pen * maxdeg

    linear: dict[int, float] = {}
    quadratic: dict[tuple[int, int], float] = {}
    constant = 0.0
    k = discrete.k

    # 1) maneuver cost (linear) + one-hot linear/constant terms
    for i in inst.aircraft():
        constant += lam_oh  # +lam_oh per aircraft from (sum-1)^2
        for o in range(k):
            b = discrete.var_index(i, o)
            linear[b] = linear.get(b, 0.0) + cost[o] - lam_oh

    # 2) one-hot quadratic terms (same aircraft, distinct options): +2 lam_oh
    for i in inst.aircraft():
        for o1 in range(k):
            b1 = discrete.var_index(i, o1)
            for o2 in range(o1 + 1, k):
                b2 = discrete.var_index(i, o2)
                key = (b1, b2) if b1 < b2 else (b2, b1)
                quadratic[key] = quadratic.get(key, 0.0) + 2.0 * lam_oh

    # 3) conflict penalty (cross aircraft): +lam_pen for each conflicting option pair
    n_conflict_terms = 0
    for (i, j), mat in discrete.conflict.items():
        for oi in range(k):
            row = mat[oi]
            bi = discrete.var_index(i, oi)
            for oj in range(k):
                if row[oj]:
                    bj = discrete.var_index(j, oj)
                    key = (bi, bj) if bi < bj else (bj, bi)
                    quadratic[key] = quadratic.get(key, 0.0) + lam_pen
                    n_conflict_terms += 1

    n_onehot_terms = inst.n * (k * (k - 1) // 2)
    meta = {
        "n_conflict_terms": float(n_conflict_terms),
        "n_onehot_terms": float(n_onehot_terms),
        "cost_max": cost_max,
        "grid_k": float(k),
        "n_aircraft": float(inst.n),
        "n_pairs": float(inst.n_pairs()),
        "max_conflict_degree": float(maxdeg),
        "onehot_optimum_guaranteed": 1.0 if onehot_guaranteed else 0.0,
    }
    return ManeuverQUBO(
        discrete=discrete,
        linear=linear,
        quadratic=quadratic,
        constant=constant,
        n_qubits=n_qubits,
        penalty_onehot=lam_oh,
        penalty_conflict=lam_pen,
        meta=meta,
        objective_id=oid.value,
    )


def assert_objective_consistent(qubo: ManeuverQUBO) -> None:
    """Reject a QUBO whose ``objective_id`` does not match its coefficients.

    ``build_qubo`` sets ``linear[b=(i,o)] = option_cost(objective_id, o) - lam_oh``.
    A QUBO whose ``objective_id`` was swapped (e.g. via ``dataclasses.replace``)
    without rebuilding the coefficients is a SPOOF: it may be useful to exercise
    hash-binding in a test, but it must never cross the scientific-validation /
    persistence / equivalence boundary. This check recomputes the expected
    linear term with the *same* ``option_cost_vector`` used to build it, so the
    comparison is EXACT (no tolerance to relax).
    """
    d = qubo.discrete
    cost_vec = option_cost_vector(qubo.objective_id, d.grid, d.instance.w)
    lam_oh = qubo.penalty_onehot
    for i in d.instance.aircraft():
        for o in range(d.k):
            b = d.var_index(i, o)
            expected = cost_vec[o] - lam_oh
            actual = qubo.linear.get(b, 0.0)
            if actual != expected:
                raise ValueError(
                    f"QUBO objective_id={qubo.objective_id!r} is inconsistent with "
                    f"its linear coefficient at bit {b} (aircraft {i}, option {o}): "
                    f"expected {expected!r}, got {actual!r} — refusing a spoofed "
                    f"objective at the scientific boundary"
                )


def verify_qubo_wellformed(qubo: ManeuverQUBO) -> None:
    """Full fail-closed structural + dominance check for the equivalence proof.

    Unlike :func:`assert_objective_consistent` (which checks only the linear term),
    this rebuilds the QUBO from ``(discrete, objective_id, penalties)`` and requires
    EVERY component to match exactly — the constant, every linear coefficient, and
    every quadratic coefficient (one-hot AND conflict) — so a falsified
    constant/coefficient/meta cannot pass. It then REFUSES the QUBO unless the
    penalty-dominance conditions hold (``onehot_optimum_guaranteed``), i.e. unless
    the one-hot subspace is certified to contain the global minimum of ``H`` over
    the full ``2**(nK)`` binary space (see the reduction theorem in
    :mod:`acrpq.benchmark.equivalence`).
    """
    d = qubo.discrete
    n_vars = d.n_vars
    # n_qubits coherence is checked FIRST, against the trusted discrete model, so a
    # falsified value fails closed here (ValueError) and never reaches — or blows
    # up — the 2**(nK) enumeration (a too-large value) or build_qubo's budget (a
    # too-small value). Reject bool (a subclass of int) explicitly.
    if isinstance(qubo.n_qubits, bool) or not isinstance(qubo.n_qubits, int):
        raise ValueError(
            f"n_qubits must be a non-bool int, got {type(qubo.n_qubits).__name__} "
            f"{qubo.n_qubits!r}")
    if qubo.n_qubits != n_vars:
        raise ValueError(
            f"n_qubits {qubo.n_qubits} != discrete.n_vars {n_vars} (incoherent QUBO)")
    ref = build_qubo(
        d.instance, discrete=d, objective_id=qubo.objective_id,
        penalty_conflict=qubo.penalty_conflict, penalty_onehot=qubo.penalty_onehot,
        max_qubits=n_vars,  # trusted bound (never the possibly-falsified qubo.n_qubits)
    )
    problems: list[str] = []
    if qubo.n_qubits != ref.n_qubits:
        problems.append(f"n_qubits {qubo.n_qubits!r} != rebuilt {ref.n_qubits!r}")
    if qubo.constant != ref.constant:
        problems.append(f"constant {qubo.constant!r} != rebuilt {ref.constant!r}")
    if dict(qubo.linear) != dict(ref.linear):
        problems.append("linear term differs from the objective's canonical build")
    if dict(qubo.quadratic) != dict(ref.quadratic):
        problems.append("quadratic term (one-hot + conflict) differs from the build")
    if dict(qubo.meta) != dict(ref.meta):
        problems.append("meta differs from the objective's canonical build")
    if problems:
        raise ValueError(
            f"QUBO not well-formed for objective {qubo.objective_id!r}: "
            + "; ".join(problems))
    if ref.meta.get("onehot_optimum_guaranteed", 0.0) != 1.0:
        raise ValueError(
            "penalty dominance NOT guaranteed (onehot_optimum_guaranteed is false): "
            "the one-hot subspace is not certified to contain the global QUBO "
            "minimum, so no one-hot-reduction equivalence proof is admissible")
