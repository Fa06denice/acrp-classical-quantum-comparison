"""Prove the "reassignment" metric chain on tiny hand-checkable cases.

The business variant "minimise the number of aircraft reassigned (changed)
versus the initial plan" is the repo's ``maneuver_count_v1`` objective
(src/acrpq/objectives.py).  The exact metric is::

    reassignments = #{ i : option_i != NOOP },   NOOP = (q=1, theta=0)

These tests build the real QUBO for CP_3 with K=3 (n_theta=3, n_q=1), take the
certified exhaustive optimum bitstring via
``acrpq.dashboard.qpu_batch.optimum_onehot_bits``, decode it with the
production decoder, and assert the reassignment count against hand-computed
values.  Everything is read-only over the library; no network, no IBM.

Hand-computed facts for CP_3, K=3 (verified by exhaustive enumeration, 3^3=27
assignments, conflict table from the shared geometry kernel):

* CP_3 has 3 aircraft converging on a circle -> 3 initial pairwise conflicts.
* The grid is theta in {-PI/6, 0, +PI/6} (PI = 3.141592, the .mod literal),
  q fixed at 1.0; option 1 is the exact NOOP.
* No conflict-free assignment leaves all three, or two, aircraft on the NOOP:
  the minimum number of maneuvered aircraft is 2 = n - 1, with 6 exact optima;
  the lexicographically smallest representative is choice (0, 0, 1)
  (aircraft 1 and 2 turn -PI/6, aircraft 3 stays NOOP).
"""

from __future__ import annotations

import pytest

from acrpq.classical.discrete_enum import exhaustive_discrete_optimum
from acrpq.dashboard.qpu_batch import optimum_onehot_bits
from acrpq.io.loader import InstanceLoader
from acrpq.model import ManeuverGrid
from acrpq.objectives import ObjectiveId, primary_objective
from acrpq.quantum.decode import decode_assignment
from acrpq.quantum.qubo import build_qubo

N_THETA = 3
N_Q = 1  # K = N_THETA * N_Q = 3 options per aircraft


@pytest.fixture(scope="module")
def cp3():
    inst = InstanceLoader().load("CP_3")
    grid = ManeuverGrid.build(n_theta=N_THETA, n_q=N_Q, instance=inst)
    qubo = build_qubo(
        inst, grid, objective_id=ObjectiveId.MANEUVER_COUNT_V1
    )
    return inst, grid, qubo


def reassignments(assignment: dict[int, int], noop_idx: int) -> int:
    """THE metric under test: #aircraft whose decoded option != NOOP."""
    return sum(1 for o in assignment.values() if o != noop_idx)


def onehot_bitstring(qubo, choice: tuple[int, ...]) -> str:
    """Encode a per-aircraft choice as the canonical one-hot bitstring."""
    d = qubo.discrete
    bits = ["0"] * qubo.n_qubits
    for i in d.instance.aircraft():
        bits[d.var_index(i, choice[i - 1])] = "1"
    return "".join(bits)


# --------------------------------------------------------------------------- #
# structural sanity: grid, NOOP, qubit layout
# --------------------------------------------------------------------------- #
def test_grid_k3_has_exact_noop(cp3):
    inst, grid, qubo = cp3
    assert inst.n == 3
    assert grid.k == 3
    noop = grid.options[grid.noop_index()]
    assert (noop.q, noop.theta) == (1.0, 0.0)
    assert noop.label == "NOOP"
    # K=3 heading-only grid: {-PI/6, 0, +PI/6}, middle option is the NOOP
    assert grid.noop_index() == 1
    assert qubo.n_qubits == inst.n * grid.k == 9


# --------------------------------------------------------------------------- #
# case 1: certified optimum of CP_3/K=3 -> exactly n-1 = 2 reassignments
# --------------------------------------------------------------------------- #
def test_optimum_bitstring_decodes_to_2_reassignments(cp3):
    inst, grid, qubo = cp3
    opt = optimum_onehot_bits(
        inst, qubo, ObjectiveId.MANEUVER_COUNT_V1.value, N_THETA, N_Q
    )
    assert opt["status"] == "optimal"
    assert opt["feasible"] is True
    # hand-verified: minimum maneuver count is 2 (= n-1), with 6 exact optima
    assert opt["optimum"] == 2.0
    assert opt["n_optima"] == 6

    assignment = decode_assignment(qubo, opt["bitstring"])
    n_re = reassignments(assignment, grid.noop_index())
    assert n_re == inst.n - 1 == 2
    # the reassignment count IS the maneuver_count_v1 objective value
    assert n_re == opt["optimum"]
    # deterministic representative: lexicographically smallest optimum (0, 0, 1)
    choice = tuple(assignment[i] for i in inst.aircraft())
    assert choice == (0, 0, 1)
    assert opt["bitstring"] == "100100010"
    # for a valid one-hot conflict-free assignment, QUBO energy == objective
    assert qubo.energy(opt["bitstring"]) == pytest.approx(2.0)


def test_optimum_agrees_with_exhaustive_enum_and_secondary_metrics(cp3):
    inst, grid, qubo = cp3
    res = exhaustive_discrete_optimum(
        inst,
        objective_id=ObjectiveId.MANEUVER_COUNT_V1,
        n_theta=N_THETA,
        n_q=N_Q,
    )
    assert res.status == "optimal"
    assert res.optimum == 2.0
    assert res.n_optima == 6
    assert res.tie_kind == "exact"
    # secondary_metrics reports the same count under both names
    assert res.secondary["maneuver_count"] == 2.0
    assert res.secondary["n_maneuvered"] == 2.0
    assert res.secondary["n_unchanged"] == 1.0
    # no conflict-free all-NOOP or single-maneuver assignment exists:
    # every feasible choice has >= 2 non-NOOP options (hand check via enum)
    noop = grid.noop_index()
    d = qubo.discrete
    import itertools

    for choice in itertools.product(range(grid.k), repeat=inst.n):
        if d.assignment_conflicts(choice) == 0:
            assert sum(1 for o in choice if o != noop) >= 2


# --------------------------------------------------------------------------- #
# case 2: all-NOOP assignment -> 0 reassignments (both encodings)
# --------------------------------------------------------------------------- #
def test_all_noop_assignment_counts_zero(cp3):
    inst, grid, qubo = cp3
    noop = grid.noop_index()
    bits = onehot_bitstring(qubo, (noop,) * inst.n)
    assignment = decode_assignment(qubo, bits)
    assert reassignments(assignment, noop) == 0
    q, theta = qubo.discrete.maneuvers(tuple(assignment[i] for i in inst.aircraft()))
    assert q == (1.0, 1.0, 1.0)
    assert theta == (0.0, 0.0, 0.0)
    # maneuver_count_v1 objective of the all-NOOP plan is 0 by definition
    assert primary_objective(ObjectiveId.MANEUVER_COUNT_V1, q, theta, inst.w) == 0.0


def test_all_zeros_bitstring_repairs_to_noop_zero_reassignments(cp3):
    inst, grid, qubo = cp3
    # decoder's documented repair: an aircraft with NO bit set falls back to NOOP
    assignment = decode_assignment(qubo, "0" * qubo.n_qubits)
    assert reassignments(assignment, grid.noop_index()) == 0


# --------------------------------------------------------------------------- #
# case 3: mixed assignment counted by hand
# --------------------------------------------------------------------------- #
def test_mixed_assignment_hand_counted(cp3):
    inst, grid, qubo = cp3
    noop = grid.noop_index()  # == 1
    # aircraft 1 turns left (option 0), aircraft 2 stays (NOOP), aircraft 3
    # turns right (option 2): exactly 2 aircraft deviate from the initial plan.
    choice = (0, noop, 2)
    bits = onehot_bitstring(qubo, choice)
    assignment = decode_assignment(qubo, bits)
    assert tuple(assignment[i] for i in inst.aircraft()) == choice
    assert reassignments(assignment, noop) == 2
    q, theta = qubo.discrete.maneuvers(choice)
    assert primary_objective(ObjectiveId.MANEUVER_COUNT_V1, q, theta, inst.w) == 2.0

    # single deviation: only aircraft 2 maneuvers
    choice1 = (noop, 0, noop)
    assignment1 = decode_assignment(qubo, onehot_bitstring(qubo, choice1))
    assert reassignments(assignment1, noop) == 1
    q1, t1 = qubo.discrete.maneuvers(choice1)
    assert primary_objective(ObjectiveId.MANEUVER_COUNT_V1, q1, t1, inst.w) == 1.0


def test_decode_tiebreak_is_option_index_order_not_sample_dict_order():
    """MoE Expert H, surviving mutation 14: `decode_assignment` resolves a one-hot repair
    tie by LOWEST OPTION INDEX. Rebuilding the candidate list in the sample mapping's
    insertion order kept all 12 relevant test files green while making decoding
    non-deterministic w.r.t. sample ordering."""
    from acrpq.io.loader import InstanceLoader
    from acrpq.quantum.decode import decode_assignment
    from acrpq.quantum.qubo import build_qubo

    inst = InstanceLoader().load("CP_3")
    qubo = build_qubo(inst, n_theta=3, n_q=1, objective_id="maneuver_count_v1")
    d = qubo.discrete
    i = next(iter(inst.aircraft()))
    tied = [o for o in range(d.k) if d.cost[o] == d.cost[0]]
    assert len(tied) >= 2, "fixture must expose a genuine cost tie"
    a, b = d.var_index(i, tied[0]), d.var_index(i, tied[1])
    fwd = decode_assignment(qubo, {a: 1, b: 1})[i]
    rev = decode_assignment(qubo, {b: 1, a: 1})[i]
    assert fwd == rev == min(tied), "one-hot repair tie-break leaked dict ordering"
