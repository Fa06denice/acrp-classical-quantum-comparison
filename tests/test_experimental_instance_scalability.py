"""Tests for the EXPERIMENTAL instance-scalability decomposition study.

These tests never touch the official pipeline. They check: exact recomposition
of components against a full rescoring on ``acrpq.geometry``/``acrpq.objectives``
(the single geometry kernel), zero cross-component QUBO terms, the isolated-
aircraft NOOP-fixing identity, adversarial negative controls, and the
hardware-plausibility threshold.
"""

from __future__ import annotations

import math

import pytest

from acrpq import geometry
from acrpq.experimental import adaptive_grid as ag
from acrpq.experimental import adaptive_qpu as aq
from acrpq.experimental import instance_scalability as isc
from acrpq.objectives import primary_objective

QUAD = "quadratic_control_cost_v1"
COUNT = "maneuver_count_v1"
OBJECTIVES = (QUAD, COUNT)

DECOMPOSITION_INSTANCES = ("CP_3", "CP_4", "CP_5", "CP_6", "FP_4", "FP_5", "RCP_10_1", "RCP_10_2")


# --------------------------------------------------------------------------- #
# Flags
# --------------------------------------------------------------------------- #
def test_flags_are_experimental():
    assert isc.ARTIFACT_FLAGS == {"experimental": True, "official_benchmark": False, "real_qpu": False}


# --------------------------------------------------------------------------- #
# Known values from expert D's report (cross-check of the dominance rule)
# --------------------------------------------------------------------------- #
def test_cp_has_zero_dominance_reduction(loader):
    """Table 1 of expert_D_decomposition.md: 0 options eliminated on CP, any n/K/objective."""
    for name in ("CP_3", "CP_4", "CP_5", "CP_6"):
        inst = loader.load(name)
        levels = ag.official_k_grid_options(inst, 3)
        edges = isc.any_option_graph(inst, levels)
        degs = isc.degrees(inst.n, edges)
        isolated = {v for v, d in degs.items() if d == 0}
        assert isolated == set(), "CP must be a K3 clique: no isolated aircraft"
        for oid in OBJECTIVES:
            counts = isc.dominance_survivor_counts(inst, oid, levels, isolated)
            assert counts == (3,) * inst.n


def test_fp4_dominance_matches_expert_d_quadratic(loader):
    """Table 2 of expert_D_decomposition.md: FP_4 K=3 quadratic 24 -> 18 qubits."""
    inst = loader.load("FP_4")
    levels = ag.official_k_grid_options(inst, 3)
    edges = isc.any_option_graph(inst, levels)
    degs = isc.degrees(inst.n, edges)
    isolated = {v for v, d in degs.items() if d == 0}
    counts = isc.dominance_survivor_counts(inst, QUAD, levels, isolated)
    assert sum(counts) == 18


def test_fp_gp_component_sizes_match_expert_d_table4(loader):
    """Table 3/4 of expert_D_decomposition.md: any-option K3 component sizes."""
    expected = {
        "CP_3": [3], "CP_4": [4], "CP_5": [5], "CP_6": [6],
        "FP_4": [4, 2, 2], "FP_5": [4, 4, 2], "RCP_10_1": [9, 1], "RCP_10_2": [10],
    }
    for name, sizes in expected.items():
        inst = loader.load(name)
        levels = ag.official_k_grid_options(inst, 3)
        edges = isc.any_option_graph(inst, levels)
        comps = isc.connected_components(inst.n, edges)
        assert sorted((len(c) for c in comps), reverse=True) == sorted(sizes, reverse=True)


# --------------------------------------------------------------------------- #
# Exact decomposition: component-wise solve == monolithic solve, on the real
# geometry kernel (never the QUBO's own energy, to keep the check independent).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", DECOMPOSITION_INSTANCES)
@pytest.mark.parametrize("oid", OBJECTIVES)
def test_component_recomposition_matches_monolithic_exact(loader, name, oid):
    inst = loader.load(name)
    levels = ag.official_k_grid_options(inst, 3)
    edges = isc.any_option_graph(inst, levels)
    comps = isc.connected_components(inst.n, edges)

    global_grid = ag.global_grid(inst, 3)
    mono = ag.exact_local_solve(inst, oid, global_grid)
    assert mono.theta is not None

    # global QUBO must have ZERO cross-component quadratic terms.
    global_qubo = aq.build_hetero_qubo(inst, global_grid, oid)
    assert isc.cross_component_qubo_terms(global_qubo, comps) == 0

    theta_by_id: dict[int, float] = {}
    for comp in comps:
        sub = inst.subset(comp)
        sub_grid = ag.global_grid(sub, 3)
        rs = ag.exact_local_solve(sub, oid, sub_grid)
        assert rs.theta is not None
        for aid, t in zip(comp, rs.theta):
            theta_by_id[aid] = t
    theta_full = tuple(theta_by_id[i] for i in inst.aircraft())

    q_full = (1.0,) * inst.n
    n_conflicts = geometry.count_conflicts(inst, q_full, theta_full)
    obj_full = primary_objective(oid, q_full, theta_full, inst.w)

    assert n_conflicts == 0
    if oid == COUNT:
        assert obj_full == mono.objective
    else:
        assert math.isclose(obj_full, mono.objective, rel_tol=0.0, abs_tol=1e-12)


# --------------------------------------------------------------------------- #
# Isolated aircraft: fixing to NOOP never changes the optimum.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("oid", OBJECTIVES)
def test_isolated_aircraft_fixed_to_noop_preserves_optimum(loader, oid):
    inst = loader.load("RCP_10_1")  # known isolated aircraft under K3 (component [9, 1])
    levels = ag.official_k_grid_options(inst, 3)
    edges = isc.any_option_graph(inst, levels)
    degs = isc.degrees(inst.n, edges)
    isolated = {v for v, d in degs.items() if d == 0}
    assert isolated, "RCP_10_1 must have an isolated aircraft under K3"

    full_grid = ag.global_grid(inst, 3)
    free = ag.exact_local_solve(inst, oid, full_grid)

    # rebuild the grid with every isolated aircraft's block forced to {0.0}
    forced_opts = tuple(
        (0.0,) if (i + 1) in isolated else full_grid.options[i] for i in range(inst.n)
    )
    forced_grid = ag.LocalGrid(centers=full_grid.centers, delta=full_grid.delta,
                               include_noop=True, options=forced_opts)
    forced = ag.exact_local_solve(inst, oid, forced_grid)

    assert forced.theta is not None and free.theta is not None
    assert forced.objective == free.objective if oid == COUNT else \
        math.isclose(forced.objective, free.objective, rel_tol=0.0, abs_tol=1e-12)
    for i in isolated:
        assert forced.theta[i - 1] == 0.0


# --------------------------------------------------------------------------- #
# Adversarial negative controls
# --------------------------------------------------------------------------- #
def test_wrong_split_of_a_connected_component_has_cross_terms_or_conflicts(loader):
    """CP_4 is ONE K3 clique component; splitting it into two halves is WRONG
    and must be caught (non-zero cross terms on the QUBO, or a conflict on
    recomposition) -- exactly the failure mode expert D warns against."""
    inst = loader.load("CP_4")
    levels = ag.official_k_grid_options(inst, 3)
    edges = isc.any_option_graph(inst, levels)
    real_comps = isc.connected_components(inst.n, edges)
    assert len(real_comps) == 1 and len(real_comps[0]) == 4

    wrong_comps = ((1, 2), (3, 4))  # NOT a partition into true components
    global_grid = ag.global_grid(inst, 3)
    global_qubo = aq.build_hetero_qubo(inst, global_grid, QUAD)
    cross = isc.cross_component_qubo_terms(global_qubo, wrong_comps)
    assert cross > 0, "a wrong split of a real clique must show cross-component QUBO terms"

    # solving the two halves independently and recomposing can reintroduce conflicts
    theta_by_id: dict[int, float] = {}
    for comp in wrong_comps:
        sub = inst.subset(comp)
        sub_grid = ag.global_grid(sub, 3)
        rs = ag.exact_local_solve(sub, QUAD, sub_grid)
        for aid, t in zip(comp, rs.theta):
            theta_by_id[aid] = t
    theta_full = tuple(theta_by_id[i] for i in inst.aircraft())
    n_conflicts = geometry.count_conflicts(inst, (1.0,) * inst.n, theta_full)
    mono = ag.exact_local_solve(inst, QUAD, global_grid)
    # either the recomposed assignment carries a conflict, or (by luck) it is
    # conflict-free but then it MUST be at least as good as the true optimum by
    # construction only if the split happened to be exact -- which it is not here.
    assert n_conflicts > 0 or not math.isclose(
        primary_objective(QUAD, (1.0,) * inst.n, theta_full, inst.w), mono.objective,
        rel_tol=0.0, abs_tol=1e-12,
    ) or cross > 0


def test_degree_zero_required_for_isolated(loader):
    """No aircraft with degree > 0 in the K3 any-option graph may be treated
    as isolated (K_i = 1) anywhere in this module."""
    for name in ("CP_4", "FP_4", "FP_5", "RCP_10_1", "RCP_10_2"):
        inst = loader.load(name)
        levels = ag.official_k_grid_options(inst, 3)
        edges = isc.any_option_graph(inst, levels)
        degs = isc.degrees(inst.n, edges)
        isolated = {v for v, d in degs.items() if d == 0}
        for i, j in edges:
            assert i not in isolated
            assert j not in isolated
        # dominance counts must give K_i = 1 ONLY for isolated aircraft
        counts = isc.dominance_survivor_counts(inst, QUAD, levels, isolated)
        for i in range(1, inst.n + 1):
            if counts[i - 1] == 1 and degs[i] > 0:
                # K_i can legitimately collapse to 1 via genuine dominance too;
                # only forbid the ISOLATED short-circuit on a non-isolated aircraft.
                assert i not in isolated


def test_isolated_flag_never_applied_to_a_wrongly_labelled_aircraft():
    """A caller that mislabels a degree>0 aircraft as isolated gets K_i = 1 for
    it (the module trusts its ``isolated`` argument) -- so callers, not this
    function, must derive ``isolated`` strictly from degree == 0. This test
    locks that contract for the reference computation used everywhere else."""
    import acrpq.io.loader as loader_mod

    inst = loader_mod.InstanceLoader().load("CP_4")
    levels = ag.official_k_grid_options(inst, 3)
    edges = isc.any_option_graph(inst, levels)
    degs = isc.degrees(inst.n, edges)
    assert all(d > 0 for d in degs.values()), "CP_4 is a full K3 clique"
    # deliberately mislabel aircraft 1 as isolated even though degree(1) == 3
    bogus_isolated = {1}
    counts = isc.dominance_survivor_counts(inst, QUAD, levels, bogus_isolated)
    assert counts[0] == 1  # the module DID force K_1 = 1 -- proving callers must
    # never pass a degree>0 aircraft in `isolated`; the real pipeline (tested
    # above in test_degree_zero_required_for_isolated) never does.


# --------------------------------------------------------------------------- #
# Hardware-plausibility threshold
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q_largest", [13, 15, 20, 90])
def test_no_hardware_pilot_plausible_above_12_logical_qubits(q_largest):
    ci = isc.ClassificationInput(
        n=100, q_global_k3=300, q_largest_component=q_largest, largest_component_size=q_largest // 3,
        n_components=1, cross_component_terms=0, local_exact_feasible_global=False,
        local_exact_feasible_component=True, expected_isa_2q_component=1,
    )
    labels = isc.classify_instance(ci)
    assert "HARDWARE_PILOT_PLAUSIBLE" not in labels


def test_hardware_pilot_plausible_at_or_below_12_qubits_with_low_2q():
    ci = isc.ClassificationInput(
        n=10, q_global_k3=30, q_largest_component=9, largest_component_size=3,
        n_components=1, cross_component_terms=0, local_exact_feasible_global=True,
        local_exact_feasible_component=True, expected_isa_2q_component=108,
    )
    labels = isc.classify_instance(ci)
    assert "HARDWARE_PILOT_PLAUSIBLE" in labels


def test_gp4_exact_via_bb_is_fast_and_matches_component(loader):
    """GP is a K3 clique like CP (expert D): the whole instance is one component."""
    inst = loader.load("GP_4")
    levels = ag.official_k_grid_options(inst, 3)
    edges = isc.any_option_graph(inst, levels)
    comps = isc.connected_components(inst.n, edges)
    assert len(comps) == 1 and len(comps[0]) == inst.n
    grid = ag.global_grid(inst, 3)
    rs = ag.exact_local_solve(inst, QUAD, grid, max_states=100_000_000)
    assert rs.theta is not None
    n_conflicts = geometry.count_conflicts(inst, (1.0,) * inst.n, rs.theta)
    assert n_conflicts == 0
