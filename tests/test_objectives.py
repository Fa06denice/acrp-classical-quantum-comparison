"""Adversarial tests for the two versioned objectives (mission Phase 1).

Covers every adversarial requirement:
  1. total NOOP -> cardinality 0
  2. any non-NOOP command -> cost 1 (independent of type/magnitude)
  3. two different commands of the same aircraft -> same primary cost
  4. two maneuvered aircraft -> cost 2
  5. a one-hot violation is never the QUBO optimum
  6. QUBO energy == primary objective + penalties
  7. penalties dominate enough to preserve the constraints
  8. variable order and hashes are deterministic
plus objective identity, the byte-identical quadratic path, secondary metrics,
the epsilon-free deterministic tie-break, and objective-id propagation.
"""

from __future__ import annotations

import math

import pytest

from acrpq.io.loader import InstanceLoader
from acrpq.model import ManeuverGrid
from acrpq.objectives import (
    ALL_OBJECTIVES,
    DEFAULT_OBJECTIVE,
    ObjectiveId,
    coerce_objective_id,
    is_noop,
    option_cost,
    option_cost_vector,
    pick_optimum,
    primary_objective,
    secondary_metrics,
)
from acrpq.quantum.explain import split_qubo_components
from acrpq.quantum.qubo import build_qubo


@pytest.fixture(scope="module")
def cp3():
    return InstanceLoader().load("CP_3")


# --- objective identity -----------------------------------------------------
def test_objective_ids_are_stable_immutable_strings():
    assert ObjectiveId.QUADRATIC_CONTROL_COST_V1.value == "quadratic_control_cost_v1"
    assert ObjectiveId.MANEUVER_COUNT_V1.value == "maneuver_count_v1"
    assert DEFAULT_OBJECTIVE is ObjectiveId.QUADRATIC_CONTROL_COST_V1
    assert set(ALL_OBJECTIVES) == set(ObjectiveId)
    assert coerce_objective_id("maneuver_count_v1") is ObjectiveId.MANEUVER_COUNT_V1
    assert coerce_objective_id(ObjectiveId.MANEUVER_COUNT_V1) is ObjectiveId.MANEUVER_COUNT_V1
    with pytest.raises(ValueError):
        coerce_objective_id("count_v2")


# --- maneuver_count semantics (adversarial 1-4) -----------------------------
def test_all_noop_has_cardinality_zero():
    q = (1.0, 1.0, 1.0)
    th = (0.0, 0.0, 0.0)
    assert primary_objective(ObjectiveId.MANEUVER_COUNT_V1, q, th, 0.5) == 0.0
    assert all(is_noop(a, b) for a, b in zip(q, th))


def test_any_nonnoop_costs_one_regardless_of_type_or_magnitude():
    oid = "maneuver_count_v1"
    assert option_cost(oid, 1.0, 0.0, 0.5) == 0.0        # exact NOOP
    assert option_cost(oid, 1.0, 1e-9, 0.5) == 1.0       # tiny heading change
    assert option_cost(oid, 1.0, 0.5, 0.5) == 1.0        # large heading change
    assert option_cost(oid, 0.95, 0.0, 0.5) == 1.0       # speed only
    assert option_cost(oid, 0.94, -0.3, 0.5) == 1.0      # both, negative theta


def test_two_different_commands_same_aircraft_have_equal_primary_cost():
    a = option_cost("maneuver_count_v1", 1.0, 0.30, 0.5)   # heading-only
    b = option_cost("maneuver_count_v1", 0.94, -0.20, 0.5)  # speed+heading
    assert a == b == 1.0


def test_two_maneuvered_aircraft_cost_two():
    q = (1.0, 0.95, 1.0)
    th = (0.0, 0.0, 0.2)   # a/c 2 (speed) + a/c 3 (heading) maneuvered
    assert primary_objective("maneuver_count_v1", q, th, 0.5) == 2.0


# --- quadratic path is byte-identical to the historical cost ----------------
def test_quadratic_option_cost_matches_grid_cost():
    grid = ManeuverGrid.build(n_theta=5, n_q=3)
    w = 0.5
    vec = option_cost_vector(ObjectiveId.QUADRATIC_CONTROL_COST_V1, grid, w)
    assert vec == tuple(grid.cost(o, w) for o in grid.options)


def test_default_objective_qubo_uses_grid_cost(cp3):
    qubo = build_qubo(cp3, n_theta=3, n_q=1)  # default objective
    assert qubo.objective_id == ObjectiveId.QUADRATIC_CONTROL_COST_V1.value
    d = qubo.discrete
    assert option_cost_vector(DEFAULT_OBJECTIVE, d.grid, cp3.w) == d.cost


@pytest.mark.parametrize("w", [0.09, 0.1, 0.3, 0.5, 0.7, 0.9, 0.123456])
def test_quadratic_vector_is_byte_identical_to_grid_cost_for_nondyadic_w(w):
    # regression for the ULP grouping bug: option_cost_vector(QUADRATIC) must
    # equal grid.cost EXACTLY, including for non-dyadic w (not only w=0.5).
    grid = ManeuverGrid.build(n_theta=5, n_q=3)
    vec = option_cost_vector(ObjectiveId.QUADRATIC_CONTROL_COST_V1, grid, w)
    expected = tuple(grid.cost(o, w) for o in grid.options)
    assert vec == expected  # exact float equality, no tolerance


@pytest.mark.parametrize("w", [0.09, 0.3, 0.7, 0.9])
def test_primary_quadratic_matches_geometry_scorer_for_nondyadic_w(cp3, w):
    # primary_objective(QUADRATIC) must reproduce the canonical geometry scorer
    # exactly (its own grouping), independent of the QUBO's grid.cost grouping.
    from dataclasses import replace

    from acrpq import geometry

    inst = replace(cp3, w=w)
    q = (1.0, 0.97, 0.95)
    theta = (0.0, 0.11, -0.2)
    assert primary_objective(ObjectiveId.QUADRATIC_CONTROL_COST_V1, q, theta, w) \
        == geometry.objective(inst, q, theta)


def test_maneuver_count_linear_coefficients_are_exact(cp3):
    # direct coefficient check (not just an energy identity): under maneuver_count
    # a NOOP bit contributes 0-lam_oh and every other option bit contributes 1-lam_oh.
    qubo = build_qubo(cp3, n_theta=3, n_q=1, objective_id="maneuver_count_v1")
    d = qubo.discrete
    noop = d.grid.noop_index()
    lam_oh = qubo.penalty_onehot
    for i in cp3.aircraft():
        for o in range(d.k):
            b = d.var_index(i, o)
            expected = (0.0 if o == noop else 1.0) - lam_oh
            assert qubo.linear[b] == pytest.approx(expected)


# --- secondary metrics reported separately ----------------------------------
def test_secondary_metrics_expose_both_views():
    q = (1.0, 0.95)
    th = (0.0, 0.2)
    m = secondary_metrics(q, th, 0.5)
    assert m["n_maneuvered"] == 1.0 and m["n_unchanged"] == 1.0
    assert m["max_abs_theta"] == 0.2
    assert m["max_abs_dq"] == pytest.approx(0.05)
    assert m["quadratic_control_cost"] == pytest.approx(0.5 * 0.2**2 + 0.5 * 0.05**2)
    assert m["mean_abs_theta"] == pytest.approx(0.1)


# --- no epsilon tie-break: equal objectives are EXACTLY equal ---------------
def test_no_epsilon_tiebreak_equal_objectives_are_exactly_equal():
    a = primary_objective("maneuver_count_v1", (1.0, 0.95), (0.0, 0.2), 0.5)
    b = primary_objective("maneuver_count_v1", (0.95, 1.0), (0.2, 0.0), 0.5)
    assert a == b == 1.0  # exact equality, no perturbation added


def test_pick_optimum_is_deterministic_and_counts_ties():
    cands = [((1, 0), 2.0), ((0, 1), 2.0), ((0, 0), 3.0), ((1, 1), 2.0)]
    rep, n = pick_optimum(cands)
    assert rep == (0, 1) and n == 3  # lexicographically smallest of the three ties
    assert pick_optimum([]) == (None, 0)


# --- QUBO energy == primary + penalties (adversarial 6) ---------------------
def test_qubo_valid_onehot_energy_is_primary_plus_conflict_penalty(cp3):
    for oid in ObjectiveId:
        qubo = build_qubo(cp3, n_theta=3, n_q=1, objective_id=oid)
        d = qubo.discrete
        k = d.k
        cost = option_cost_vector(oid, d.grid, cp3.w)
        for choice in [tuple([0] * cp3.n), tuple([k - 1] * cp3.n), tuple(range(cp3.n))]:
            bits = {d.var_index(i, choice[i - 1]): 1 for i in cp3.aircraft()}
            energy = qubo.energy(bits)
            expected = sum(cost[choice[i - 1]] for i in cp3.aircraft()) \
                + qubo.penalty_conflict * d.assignment_conflicts(choice)
            assert energy == pytest.approx(expected)


# --- penalties force a one-hot global minimum (adversarial 5 + 7) -----------
def test_penalties_force_a_valid_onehot_global_minimum(cp3):
    for oid in ObjectiveId:
        qubo = build_qubo(cp3, n_theta=3, n_q=1, objective_id=oid)
        nq = qubo.n_qubits  # CP_3 K3 = 9 qubits = 512 states, exhaustive
        best_e, best_bits = math.inf, None
        for mask in range(1 << nq):
            bits = {b: 1 for b in range(nq) if (mask >> b) & 1}
            e = qubo.energy(bits)
            if e < best_e:
                best_e, best_bits = e, bits
        d = qubo.discrete
        for i in cp3.aircraft():
            active = sum(1 for o in range(d.k) if best_bits.get(d.var_index(i, o), 0))
            assert active == 1, f"{oid}: aircraft {i} not one-hot at the global min"


def test_a_onehot_violation_costs_more_than_the_best_onehot(cp3):
    qubo = build_qubo(cp3, n_theta=3, n_q=1, objective_id="maneuver_count_v1")
    d = qubo.discrete
    # best all-valid one-hot energy (all NOOP is a valid one-hot)
    noop = d.grid.noop_index()
    valid_bits = {d.var_index(i, noop): 1 for i in cp3.aircraft()}
    valid_e = qubo.energy(valid_bits)
    # a one-hot violation: aircraft 1 takes TWO options at once
    bad_bits = dict(valid_bits)
    bad_bits[d.var_index(1, (noop + 1) % d.k)] = 1
    assert qubo.energy(bad_bits) > valid_e
    # and dropping an aircraft entirely (zero active) also costs more
    empty_bits = {d.var_index(i, noop): 1 for i in cp3.aircraft() if i != 1}
    assert qubo.energy(empty_bits) > valid_e


# --- determinism of variable order + build (adversarial 8) ------------------
def test_qubo_build_is_deterministic(cp3):
    a = build_qubo(cp3, n_theta=3, n_q=1, objective_id="maneuver_count_v1")
    b = build_qubo(cp3, n_theta=3, n_q=1, objective_id="maneuver_count_v1")
    assert a.linear == b.linear
    assert a.quadratic == b.quadratic
    assert a.constant == b.constant
    assert a.objective_id == b.objective_id == "maneuver_count_v1"
    assert list(a.linear.keys()) == list(b.linear.keys())  # variable order stable


def test_objective_id_propagates_to_decomposed_components(cp3):
    qubo = build_qubo(cp3, n_theta=3, n_q=1, objective_id="maneuver_count_v1")
    for _component, local in split_qubo_components(qubo):
        assert local.objective_id == "maneuver_count_v1"


# ===================== Phase 1.1 hardening (Codex review) =================== #

# --- (1+6) explainability identity over ALL 512 states, both objectives -----
def test_energy_breakdown_identity_holds_over_all_states_both_objectives(cp3):
    from acrpq.quantum.explain import energy_breakdown

    for oid in ObjectiveId:
        qubo = build_qubo(cp3, n_theta=3, n_q=1, objective_id=oid)
        nq = qubo.n_qubits  # 9 -> 512 states, fully exhaustive
        for mask in range(1 << nq):
            bits = {b: 1 for b in range(nq) if (mask >> b) & 1}
            bd = energy_breakdown(qubo, bits)
            assert bd["objective_id"] == oid.value
            assert bd["total"] == pytest.approx(qubo.energy(bits), abs=1e-9)
            assert bd["identity_verified"] is True


def test_explainability_surfaces_publish_objective_id(cp3):
    from acrpq.quantum.explain import (
        bitstring_explanation,
        coefficient_explanation,
        energy_breakdown,
    )

    qubo = build_qubo(cp3, n_theta=3, n_q=1, objective_id="maneuver_count_v1")
    d = qubo.discrete
    noop = d.grid.noop_index()
    # a bitstring that moves aircraft 1 (non-NOOP) -> its per-row cost must be 1
    moved = (noop + 1) % d.k
    bits = {d.var_index(i, noop): 1 for i in cp3.aircraft()}
    bits[d.var_index(1, noop)] = 0
    bits[d.var_index(1, moved)] = 1
    be = bitstring_explanation(qubo, bits)
    assert be["objective_id"] == "maneuver_count_v1"
    row1 = next(r for r in be["aircraft"] if r["aircraft"] == 1)
    assert row1["cost"] == 1.0  # objective-aware (was quadratic before the fix)
    ce = coefficient_explanation(qubo)
    assert ce["objective_id"] == "maneuver_count_v1"
    for row in ce["linear"]:
        assert row["objective_cost"] in (0.0, 1.0)
    assert energy_breakdown(qubo, bits)["objective_id"] == "maneuver_count_v1"


# --- (6) conflict-free global minimum when one exists, both objectives ------
def test_global_minimum_is_conflict_free_when_a_conflict_free_onehot_exists(cp3):
    import itertools

    for oid in ObjectiveId:
        qubo = build_qubo(cp3, n_theta=3, n_q=1, objective_id=oid)
        d = qubo.discrete
        conflict_free_exists = any(
            d.assignment_conflicts(choice) == 0
            for choice in itertools.product(range(d.k), repeat=cp3.n)
        )
        if not conflict_free_exists:
            continue
        best_e, best_bits = math.inf, None
        for mask in range(1 << qubo.n_qubits):
            bits = {b: 1 for b in range(qubo.n_qubits) if (mask >> b) & 1}
            e = qubo.energy(bits)
            if e < best_e:
                best_e, best_bits = e, bits
        choice = tuple(
            next(o for o in range(d.k) if best_bits.get(d.var_index(i, o), 0))
            for i in cp3.aircraft()
        )
        assert d.assignment_conflicts(choice) == 0


# --- (6) insufficient penalty overrides are flagged, not silently trusted ---
def test_penalty_override_qualification_flag(cp3):
    auto = build_qubo(cp3, n_theta=3, n_q=1)
    assert auto.meta["onehot_optimum_guaranteed"] == 1.0
    weak = build_qubo(cp3, n_theta=3, n_q=1, penalty_onehot=1e-3, penalty_conflict=1e-3)
    assert weak.meta["onehot_optimum_guaranteed"] == 0.0


# --- (3) ManeuverQUBO identity is validated, normalised, and immutable ------
def test_qubo_rejects_unknown_objective_and_is_frozen(cp3):
    import dataclasses

    with pytest.raises(ValueError):
        build_qubo(cp3, n_theta=3, n_q=1, objective_id="nope_v9")
    qubo = build_qubo(cp3, n_theta=3, n_q=1, objective_id=ObjectiveId.MANEUVER_COUNT_V1)
    assert qubo.objective_id == "maneuver_count_v1"  # normalised to the string
    with pytest.raises(dataclasses.FrozenInstanceError):
        qubo.objective_id = "quadratic_control_cost_v1"  # no silent mutation


# --- (2) canonical_qubo_hash includes the objective identity ----------------
def test_canonical_hash_binds_the_objective_identity(cp3):
    import dataclasses

    from acrpq.dashboard.hardware_validation import canonical_qubo_hash

    qa = build_qubo(cp3, n_theta=3, n_q=1, objective_id="quadratic_control_cost_v1")
    qa2 = build_qubo(cp3, n_theta=3, n_q=1, objective_id="quadratic_control_cost_v1")
    qb = build_qubo(cp3, n_theta=3, n_q=1, objective_id="maneuver_count_v1")
    # same QUBO + same objective -> same hash
    assert canonical_qubo_hash(qa) == canonical_qubo_hash(qa2)
    # different objective -> different hash
    assert canonical_qubo_hash(qa) != canonical_qubo_hash(qb)
    # identical coefficients but different objective_id -> different hash
    spoof = dataclasses.replace(qa, objective_id="maneuver_count_v1")
    assert spoof.linear == qa.linear and spoof.quadratic == qa.quadratic
    assert canonical_qubo_hash(spoof) != canonical_qubo_hash(qa)
    # dictionary insertion order has no effect
    reordered = dataclasses.replace(
        qa,
        linear=dict(reversed(list(qa.linear.items()))),
        quadratic=dict(reversed(list(qa.quadratic.items()))),
    )
    assert canonical_qubo_hash(reordered) == canonical_qubo_hash(qa)


# --- (4) pick_optimum: exact ties, explicit tolerance, NaN/Inf refused ------
def test_pick_optimum_exact_by_default_and_tolerance_is_explicit():
    # exact (tol=0): a value 1e-12 away is NOT tied
    cands = [((0, 0), 2.0), ((0, 1), 2.0), ((1, 0), 2.0 + 1e-12)]
    rep, n = pick_optimum(cands)
    assert rep == (0, 0) and n == 2  # the 1e-12-away one is excluded
    # explicit finite tolerance includes it (deliberate, documented)
    rep_t, n_t = pick_optimum(cands, tol=1e-9)
    assert n_t == 3
    # integer maneuver-count cardinalities: exact equality
    ints = [((0,), 1.0), ((1,), 1.0), ((2,), 2.0)]
    assert pick_optimum(ints) == ((0,), 2)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_pick_optimum_refuses_non_finite_values_and_tolerances(bad):
    with pytest.raises(ValueError):
        pick_optimum([((0,), bad)])
    with pytest.raises(ValueError):
        pick_optimum([((0,), 1.0)], tol=bad)
    with pytest.raises(ValueError):
        pick_optimum([((0,), 1.0)], tol=-1.0)  # negative tolerance
    with pytest.raises(TypeError):
        pick_optimum([((0,), 1.0)], tol=True)  # bool is not a tolerance


# --- (5) fail-closed validation of q/theta/w -------------------------------
@pytest.mark.parametrize("w", [-0.01, 1.01, float("nan"), float("inf")])
def test_objective_refuses_bad_weight(w):
    with pytest.raises((ValueError, TypeError)):
        option_cost("quadratic_control_cost_v1", 1.0, 0.0, w)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_objective_refuses_non_finite_controls(bad):
    with pytest.raises(ValueError):
        option_cost("maneuver_count_v1", bad, 0.0, 0.5)
    with pytest.raises(ValueError):
        primary_objective("maneuver_count_v1", (1.0, bad), (0.0, 0.0), 0.5)
    with pytest.raises(ValueError):
        secondary_metrics((1.0, 0.9), (0.0, bad), 0.5)


def test_objective_refuses_bool_controls():
    with pytest.raises(TypeError):
        option_cost("maneuver_count_v1", True, 0.0, 0.5)


# ============ Phase 2 opening hardenings (Codex) ============================ #

# --- (H1) deep-frozen: caller-dict mutation cannot alter a built/hashed QUBO -
def test_caller_dict_mutation_cannot_alter_built_qubo(cp3):
    import dataclasses

    from acrpq.dashboard.hardware_validation import canonical_qubo_hash

    base = build_qubo(cp3, n_theta=3, n_q=1)
    caller_linear = dict(base.linear)
    caller_meta = dict(base.meta)
    q = dataclasses.replace(base, linear=caller_linear, meta=caller_meta)
    h0 = canonical_qubo_hash(q)
    # mutate the caller-owned dicts AFTER the QUBO was built and hashed
    caller_linear[0] = 999.0
    caller_meta["evil"] = 1.0
    # the QUBO took defensive copies: content + hash are unchanged
    assert q.linear.get(0) != 999.0
    assert "evil" not in q.meta
    assert canonical_qubo_hash(q) == h0
    # and its own coefficient/meta maps are read-only (no in-place mutation)
    with pytest.raises(TypeError):
        q.linear[0] = 5.0  # type: ignore[index]
    with pytest.raises(TypeError):
        q.meta["x"] = 1.0  # type: ignore[index]


# --- (H2) objective_id must be semantically consistent with the coefficients -
def test_assert_objective_consistent_passes_for_real_qubos(cp3):
    from acrpq.quantum.qubo import assert_objective_consistent

    for oid in ObjectiveId:
        assert_objective_consistent(build_qubo(cp3, n_theta=3, n_q=1, objective_id=oid))


def test_spoofed_objective_rejected_at_scientific_boundary(cp3):
    import dataclasses

    from acrpq.dashboard.hardware_validation import (
        HardwareValidationError,
        _validate_qubo,
        canonical_qubo_hash,
    )
    from acrpq.quantum.qubo import assert_objective_consistent

    real = build_qubo(cp3, n_theta=3, n_q=1, objective_id="quadratic_control_cost_v1")
    spoof = dataclasses.replace(real, objective_id="maneuver_count_v1")  # coeffs kept
    # hash-binding still separates them, but the spoof must never be accepted
    assert canonical_qubo_hash(spoof) != canonical_qubo_hash(real)
    with pytest.raises(ValueError):
        assert_objective_consistent(spoof)
    _validate_qubo(real, real.n_qubits)  # the real QUBO passes the boundary
    with pytest.raises(HardwareValidationError):
        _validate_qubo(spoof, spoof.n_qubits)  # the spoof is refused


# --- (2.1-1) full-coefficient wellformedness + fail-closed penalty dominance -
def test_verify_qubo_wellformed_passes_for_real_qubos(cp3):
    from acrpq.quantum.qubo import verify_qubo_wellformed

    for oid in ObjectiveId:
        verify_qubo_wellformed(build_qubo(cp3, n_theta=3, n_q=1, objective_id=oid))


@pytest.mark.parametrize(
    "falsify",
    ["constant", "linear", "onehot_quadratic", "conflict_quadratic", "meta", "objective_id"])
def test_verify_qubo_wellformed_rejects_falsification(cp3, falsify):
    import dataclasses

    from acrpq.quantum.qubo import verify_qubo_wellformed

    q = build_qubo(cp3, n_theta=3, n_q=1, objective_id="quadratic_control_cost_v1")
    d = q.discrete

    def _quad_key(conflict: bool):
        # one-hot terms are same-aircraft pairs; conflict terms are cross-aircraft
        for (a, b) in q.quadratic:
            same = d.inv_index(a)[0] == d.inv_index(b)[0]
            if same != conflict:
                return (a, b)
        raise AssertionError(f"no {'conflict' if conflict else 'one-hot'} quadratic term")

    if falsify == "constant":
        bad = dataclasses.replace(q, constant=q.constant + 1.0)
    elif falsify == "linear":
        lin = dict(q.linear)
        lin[next(iter(lin))] += 3.0
        bad = dataclasses.replace(q, linear=lin)
    elif falsify == "onehot_quadratic":
        quad = dict(q.quadratic)
        quad[_quad_key(conflict=False)] = 999.0
        bad = dataclasses.replace(q, quadratic=quad)
    elif falsify == "conflict_quadratic":
        quad = dict(q.quadratic)
        quad[_quad_key(conflict=True)] = 999.0
        bad = dataclasses.replace(q, quadratic=quad)
    elif falsify == "meta":
        m = dict(q.meta)
        m["cost_max"] = m["cost_max"] + 9.0
        bad = dataclasses.replace(q, meta=m)
    else:  # objective_id swapped, coefficients kept
        bad = dataclasses.replace(q, objective_id="maneuver_count_v1")
    with pytest.raises(ValueError):
        verify_qubo_wellformed(bad)


def test_verify_qubo_wellformed_refuses_non_dominant_penalties(cp3):
    from acrpq.quantum.qubo import verify_qubo_wellformed

    weak = build_qubo(cp3, n_theta=3, n_q=1, penalty_onehot=1e-3, penalty_conflict=1e-3)
    assert weak.meta["onehot_optimum_guaranteed"] == 0.0
    with pytest.raises(ValueError):
        verify_qubo_wellformed(weak)  # one-hot reduction not admissible


@pytest.mark.parametrize("bad_nq", ["too_small", "too_large", "bool", "incoherent"])
def test_verify_qubo_wellformed_rejects_falsified_n_qubits(cp3, bad_nq):
    # a falsified n_qubits must fail closed (ValueError) — never crash and never
    # explode the 2**(nK) enumeration on a too-large value.
    import dataclasses

    from acrpq.quantum.qubo import verify_qubo_wellformed

    q = build_qubo(cp3, n_theta=3, n_q=1)  # real n_qubits == discrete.n_vars
    value = {"too_small": 2, "too_large": 9999, "bool": True,
             "incoherent": q.n_qubits + 1}[bad_nq]
    bad = dataclasses.replace(q, n_qubits=value)
    with pytest.raises(ValueError):
        verify_qubo_wellformed(bad)
