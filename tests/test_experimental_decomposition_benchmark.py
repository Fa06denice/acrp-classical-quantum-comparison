"""Tests for scripts/experimental_decomposition_benchmark.py (counter-review §9/§10)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from acrpq import geometry
from acrpq.experimental import adaptive_grid as ag
from acrpq.experimental import instance_scalability as isc

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "decomp_bench", ROOT / "scripts" / "experimental_decomposition_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_rescore_uses_full_instance_geometry_not_a_stub(cp4):
    mod = _mod()
    theta_noop = (0.0,) * cp4.n  # do-nothing assignment: the initial conflicts remain
    n_conf, obj = mod.rescore(cp4, "quadratic_control_cost_v1", theta_noop)
    assert n_conf == geometry.count_initial_conflicts(cp4) > 0
    assert obj == 0.0


def test_components_match_analysis_module(loader):
    mod = _mod()
    for name in ("FP_4", "FP_5", "FP_6", "RCP_10_1", "CP_5"):
        inst = loader.load(name)
        comps, edges = mod.components_k3(inst)
        levels = ag.official_k_grid_options(inst, 3)
        ref = isc.connected_components(inst.n, isc.any_option_graph(inst, levels))
        assert sorted(sorted(c) for c in comps) == sorted(sorted(c) for c in ref)


def test_recomposition_without_rescore_would_hide_cross_component_conflicts(loader):
    """Mutation guard: a WRONG split (breaking a connected component) recomposes to a
    conflicting assignment that ONLY a full-geometry rescore reveals."""
    mod = _mod()
    inst = loader.load("CP_4")  # single clique: any split is wrong
    wrong_split = [[1, 2], [3, 4]]
    theta = [0.0] * inst.n
    for ids in wrong_split:
        sub = inst.subset(ids)
        rs = ag.exact_local_solve(sub, "quadratic_control_cost_v1", ag.global_grid(sub, 3))
        assert rs.theta is not None
        for k, aid in enumerate(ids):
            theta[aid - 1] = rs.theta[k]
    n_conf, _ = mod.rescore(inst, "quadratic_control_cost_v1", theta)
    assert n_conf > 0  # the per-component solutions are each conflict-free, the union is not


def test_exact_decomposition_equals_monolithic_on_fp4_both_objectives(loader):
    mod = _mod()
    inst = loader.load("FP_4")
    comps, _ = mod.components_k3(inst)
    assert [len(c) for c in comps] == [4, 2, 2]
    for oid in ("quadratic_control_cost_v1", "maneuver_count_v1"):
        mono = ag.exact_local_solve(inst, oid, ag.global_grid(inst, 3))
        theta = [0.0] * inst.n
        for ids in comps:
            sub = inst.subset(ids)
            rs = ag.exact_local_solve(sub, oid, ag.global_grid(sub, 3))
            for k, aid in enumerate(ids):
                theta[aid - 1] = rs.theta[k]
        n_conf, obj = mod.rescore(inst, oid, theta)
        assert n_conf == 0 and math.isclose(obj, mono.objective, rel_tol=0, abs_tol=1e-12)


def test_adaptive_vs_k3_difference_is_flagged_not_comparable():
    src = (ROOT / "scripts" / "experimental_decomposition_benchmark.py").read_text()
    assert "domains_comparable_adaptive_vs_K3" in src and '"adaptive_gain_vs_decomp_K3"' not in src


@pytest.mark.parametrize("oid", ["quadratic_control_cost_v1", "maneuver_count_v1"])
def test_dominance_pruning_never_changes_exact_optimum_on_random_sample(loader, oid):
    import random
    names = [n for n in random.Random(20240908).sample(loader.list_names(), 40) if loader.load(n).n <= 12]
    assert names
    for name in names:
        inst = loader.load(name)
        levels = ag.official_k_grid_options(inst, 3)
        degs = isc.degrees(inst.n, isc.any_option_graph(inst, levels))
        isolated = {v for v, d in degs.items() if d == 0}
        full = ag.exact_local_solve(inst, oid, ag.global_grid(inst, 3), max_states=10**12)
        pruned = ag.exact_local_solve(inst, oid, isc.hetero_exact_grid(inst, oid, levels, isolated), max_states=10**12)
        assert math.isclose(pruned.objective, full.objective, rel_tol=0.0, abs_tol=1e-9)


def test_no_pilot_plausible_without_measured_isa():
    ci = isc.ClassificationInput(n=4, q_global_k3=12, q_largest_component=12, largest_component_size=4,
                                 n_components=1, cross_component_terms=0, local_exact_feasible_global=True,
                                 local_exact_feasible_component=True, expected_isa_2q_component=None,
                                 q_hetero_exact=12)
    assert "HARDWARE_PILOT_PLAUSIBLE" not in isc.classify_instance(ci)
    ci2 = isc.ClassificationInput(**{**ci.__dict__, "expected_isa_2q_component": 142})
    assert "HARDWARE_PILOT_PLAUSIBLE" in isc.classify_instance(ci2)
    ci3 = isc.ClassificationInput(**{**ci.__dict__, "expected_isa_2q_component": 400})
    assert "HARDWARE_PILOT_PLAUSIBLE" not in isc.classify_instance(ci3)
