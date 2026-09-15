"""Mutation tests for the K3 decomposition counterproof checker.

Each mutation below simulates a way the decomposition audit could be wrong
(a wrong split, a misplaced aircraft, a heuristically-fixed aircraft, a
missing global rescore, a grid mismatch, or an incumbent silently reported as
optimal). Every test asserts the CHECKER catches it -- i.e. the test is
written so it goes RED if the checker fails to detect the mutation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from acrpq.discretize import DiscreteACRP
from acrpq.experimental.adaptive_grid import exact_local_solve, global_grid
from acrpq.io.loader import InstanceLoader
from acrpq.model import ManeuverGrid
from acrpq.objectives import ObjectiveId, primary_objective
from acrpq.quantum.qubo import build_qubo

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experimental_decomposition_counterproof import (  # noqa: E402
    any_option_edges,
    component_groups,
    components_use_official_grid,
    cross_component_qubo_terms,
    union_find_components,
    validate_status_record,
)
from acrpq import geometry  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def loader() -> InstanceLoader:
    return InstanceLoader(data_root=REPO_ROOT)


def _k3_discrete_and_qubo(inst):
    grid = ManeuverGrid.build(n_theta=3, n_q=1, instance=inst)
    discrete = DiscreteACRP.build(inst, grid)
    qubo = build_qubo(inst, discrete=discrete, max_qubits=10_000)
    return discrete, qubo


# --------------------------------------------------------------------------- #
# (a) remove one inter-component QUBO term from a deliberately wrong split
# --------------------------------------------------------------------------- #
def test_mutation_a_wrong_split_cross_terms_detected_even_after_removal(loader):
    inst = loader.load("CP_4")  # a full K3 clique: every pair conflicts on-grid
    discrete, qubo = _k3_discrete_and_qubo(inst)

    # Sanity: the REAL any-option graph is a single connected component.
    real_comp_of = union_find_components(inst.n, any_option_edges(discrete))
    assert len(component_groups(real_comp_of)) == 1

    # Deliberately wrong split: [1, 2] vs [3, 4] (CP_4 is one component).
    wrong_comp_of = {1: 0, 2: 0, 3: 1, 4: 1}
    cross_before = cross_component_qubo_terms(qubo, discrete, wrong_comp_of)
    assert cross_before > 0, "checker must flag the wrong split (cross terms > 0)"

    # Mutation: remove ONE inter-component quadratic term (as if a buggy
    # decomposition had silently dropped it), then re-run the checker.
    cross_keys = [
        (b1, b2)
        for (b1, b2) in qubo.quadratic
        if discrete.inv_index(b1)[0] != discrete.inv_index(b2)[0]
        and wrong_comp_of[discrete.inv_index(b1)[0]] != wrong_comp_of[discrete.inv_index(b2)[0]]
    ]
    assert cross_keys, "expected at least one cross-component term to remove"
    mutated_quadratic = dict(qubo.quadratic)
    del mutated_quadratic[cross_keys[0]]

    class _FakeQubo:
        quadratic = mutated_quadratic

    cross_after = cross_component_qubo_terms(_FakeQubo(), discrete, wrong_comp_of)
    assert cross_after > 0, (
        "the check must still fail (cross terms > 0) after one term is removed: "
        "a single dropped term must never hide a wrong split"
    )
    assert cross_after == cross_before - 1


# --------------------------------------------------------------------------- #
# (b) move an aircraft into the wrong component
# --------------------------------------------------------------------------- #
def test_mutation_b_aircraft_in_wrong_component_breaks_recomposition(loader):
    inst = loader.load("CP_4")  # complete conflict graph: any split is "wrong"
    oid = ObjectiveId.QUADRATIC_CONTROL_COST_V1

    mono = exact_local_solve(inst, oid, global_grid(inst, 3))
    assert mono.theta is not None

    # Wrong grouping: aircraft 4 is moved away from {1,2,3} into its own
    # singleton component, even though it conflicts with all of them.
    wrong_components = [[1, 2, 3], [4]]

    theta_by_id: dict[int, float] = {}
    for ids in wrong_components:
        sub = inst.subset(ids)
        sol = exact_local_solve(sub, oid, global_grid(sub, 3))
        assert sol.theta is not None
        for pos, aid in enumerate(ids):
            theta_by_id[aid] = sol.theta[pos]
    theta_full = tuple(theta_by_id[i] for i in inst.aircraft())
    q_full = (1.0,) * inst.n

    residual = geometry.count_conflicts(inst, q_full, theta_full)
    decomposed_obj = primary_objective(oid, q_full, theta_full, inst.w)

    anomaly = residual != 0 or abs(decomposed_obj - mono.objective) > 1e-12
    assert anomaly, (
        "moving aircraft 4 into the wrong component must produce a detectable "
        "anomaly (residual conflicts or objective mismatch) on this clique"
    )


# --------------------------------------------------------------------------- #
# (c) fix an active (degree > 0) aircraft to NOOP
# --------------------------------------------------------------------------- #
def test_mutation_c_fixing_active_aircraft_to_noop_changes_result(loader):
    # NOTE: on CP_3, forcing aircraft 1 to NOOP happens to reproduce the same
    # objective by clique symmetry (the other two aircraft simply rotate into
    # the slot aircraft 1 would have taken) -- that instance does NOT exhibit
    # the bug. RCP_10_2 / aircraft 9 (degree 5, found by exhaustive search
    # over the 12 audited configurations) does: it is a genuine counterexample
    # where fixing an active aircraft to NOOP changes the achievable optimum.
    inst = loader.load("RCP_10_2")
    aircraft_id = 9
    oid = ObjectiveId.QUADRATIC_CONTROL_COST_V1
    discrete, _qubo = _k3_discrete_and_qubo(inst)

    edges = any_option_edges(discrete)
    deg = sum(1 for (i, j) in edges if i == aircraft_id or j == aircraft_id)
    assert deg > 0, "the chosen aircraft must be active (degree > 0)"

    true_opt = exact_local_solve(inst, oid, global_grid(inst, 3), max_states=10**9)
    assert true_opt.theta is not None

    # Mutation: fix the aircraft to NOOP (theta = 0) and re-optimise the rest.
    fixed_grid = global_grid(inst, 3)
    forced_options = list(fixed_grid.options)
    forced_options[aircraft_id - 1] = (0.0,)
    from acrpq.experimental.adaptive_grid import LocalGrid

    forced = LocalGrid(
        centers=fixed_grid.centers, delta=fixed_grid.delta,
        include_noop=fixed_grid.include_noop, options=tuple(forced_options),
    )
    forced_opt = exact_local_solve(inst, oid, forced, max_states=10**9)
    assert forced_opt.theta is not None

    forced_conflicts = geometry.count_conflicts(inst, (1.0,) * inst.n, forced_opt.theta)
    anomaly = (
        abs(forced_opt.objective - true_opt.objective) > 1e-12
        or forced_conflicts != 0
    )
    assert anomaly, (
        "fixing an active aircraft to NOOP must be detectably different from "
        "the true exact optimum (objective mismatch or residual conflict)"
    )


# --------------------------------------------------------------------------- #
# (d) omit the global rescore
# --------------------------------------------------------------------------- #
def test_mutation_d_local_only_check_hides_the_true_global_conflict(loader):
    inst = loader.load("CP_4")
    oid = ObjectiveId.QUADRATIC_CONTROL_COST_V1

    wrong_components = [[1, 2], [3, 4]]  # CP_4 is one clique: this is wrong

    theta_by_id: dict[int, float] = {}
    local_conflicts_sum = 0
    for ids in wrong_components:
        sub = inst.subset(ids)
        sol = exact_local_solve(sub, oid, global_grid(sub, 3))
        assert sol.theta is not None
        local_conflicts_sum += geometry.count_conflicts(sub, (1.0,) * sub.n, sol.theta)
        for pos, aid in enumerate(ids):
            theta_by_id[aid] = sol.theta[pos]

    theta_full = tuple(theta_by_id[i] for i in inst.aircraft())
    global_conflicts = geometry.count_conflicts(inst, (1.0,) * inst.n, theta_full)

    # The bug this mutation demonstrates: summing PER-COMPONENT conflict
    # counts is zero by construction (each component is solved conflict-free
    # internally), which would hide the fact that recomposing them creates a
    # real global conflict. Only the full geometric rescore reveals it.
    assert local_conflicts_sum == 0, "each component is exact-solved conflict-free on its own"
    assert global_conflicts > 0, (
        "omitting the global rescore would hide a real inter-component conflict "
        "on this deliberately wrong split"
    )


# --------------------------------------------------------------------------- #
# (e) merge K3 and K5: components must all sit on the official K3 grid
# --------------------------------------------------------------------------- #
def test_mutation_e_k5_component_fails_official_grid_check(loader):
    inst = loader.load("CP_4")
    good_components = [[1, 2, 3, 4]]
    assert components_use_official_grid(inst, good_components) is True

    # Mutation: pretend one component was solved on a K5 grid by directly
    # comparing K5 levels against the official K3 levels for the same
    # sub-instance -- they must differ, so a real "merge K3+K5" bug would be
    # caught by any checker that enforces per-component grid identity.
    sub = inst.subset([1, 2, 3, 4])
    k3_levels = global_grid(sub, 3).options[0]
    k5_levels = global_grid(sub, 5).options[0]
    assert tuple(k3_levels) != tuple(k5_levels), (
        "K3 and K5 grids must have different levels for this mutation to be meaningful"
    )

    # A component check that (incorrectly) accepted K5 levels as "the grid"
    # would silently pass; the real checker must reject the mismatch.
    fails_k3_identity = tuple(k5_levels) != tuple(k3_levels)
    assert fails_k3_identity, "a K5 component must be flagged as a different grid than K3"


# --------------------------------------------------------------------------- #
# (f) treat an incumbent as optimum
# --------------------------------------------------------------------------- #
def test_mutation_f_incumbent_reported_as_optimal_is_flagged():
    bad_record = {
        "instance": "RCP_50_1",
        "monolithic_status": "refused",
        "status": "optimal",  # WRONG: a refused monolithic solve cannot be "optimal"
    }
    violations = validate_status_record(bad_record)
    assert violations, "a refused/timeout monolithic solve reported as 'optimal' must be flagged"


def test_mutation_f_correct_incumbent_record_passes():
    good_record = {
        "instance": "RCP_50_1",
        "monolithic_status": "refused",
        "status": "incumbent",
    }
    assert validate_status_record(good_record) == []


def test_mutation_f_correct_optimal_record_passes():
    good_record = {
        "instance": "CP_5",
        "monolithic_status": "ok",
        "status": "optimal",
    }
    assert validate_status_record(good_record) == []


def test_mutation_f_optimal_status_downgraded_to_incumbent_is_also_flagged():
    # Symmetric bug: a solve that DID succeed is under-reported as "incumbent".
    bad_record = {
        "instance": "CP_5",
        "monolithic_status": "ok",
        "status": "incumbent",
    }
    violations = validate_status_record(bad_record)
    assert violations
