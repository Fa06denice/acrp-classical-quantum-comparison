"""Tests for the EXPERIMENTAL adaptive iterative grid (isolated module).

These tests never touch the official pipeline; they check the experimental
module's invariants, its equivalence with the official grid at round 0, the
monotonicity theorem for exact round solvers, and adversarial inputs.
"""

from __future__ import annotations

import math
import re

import pytest

from acrpq.classical.discrete_enum import exhaustive_discrete_optimum
from acrpq.experimental import adaptive_grid as ag
from acrpq.model import ManeuverGrid

QUAD = "quadratic_control_cost_v1"
COUNT = "maneuver_count_v1"


def test_flags_are_experimental():
    assert ag.EXPERIMENTAL is True
    assert ag.OFFICIAL_BENCHMARK is False
    assert ag.REAL_QPU is False
    assert ag.ARTIFACT_FLAGS == {"experimental": True, "official_benchmark": False, "real_qpu": False}


def test_experimental_package_not_imported_by_official_pipeline():
    import pathlib

    import acrpq

    root = pathlib.Path(acrpq.__file__).parent
    pattern = re.compile(r"(acrpq\.experimental|from \.+experimental|import experimental)")
    for p in root.rglob("*.py"):
        if "experimental" in p.parts:
            continue
        assert not pattern.search(p.read_text(encoding="utf-8")), \
            f"{p} imports the experimental package"


@pytest.mark.parametrize("k", [3, 5, 7])
def test_official_grid_levels_match(cp4, k):
    g = ManeuverGrid.build(n_theta=k, n_q=1, instance=cp4)
    assert g.theta_levels == ag.official_k_grid_options(cp4, k)


def test_round0_noop_init_is_official_k3_grid(cp4):
    grid = ag.build_local_grid(cp4, (0.0,) * cp4.n, cp4.hmax)
    off = ManeuverGrid.build(n_theta=3, n_q=1, instance=cp4).theta_levels
    for opts in grid.options:
        assert opts == tuple(sorted(off))
    assert grid.n_qubits == 3 * cp4.n


@pytest.mark.parametrize("oid", [QUAD, COUNT])
@pytest.mark.parametrize("k", [3, 5])
def test_exact_bb_equals_official_exhaustive(cp4, oid, k):
    ref = exhaustive_discrete_optimum(cp4, objective_id=oid, n_theta=k)
    grid = ag.global_grid(cp4, k)
    bb = ag.exact_local_solve(cp4, oid, grid)
    oracle = ag.exhaustive_local_solve(cp4, oid, grid)
    assert bb.objective == ref.optimum == oracle.objective
    assert bb.theta == ref.theta == oracle.theta  # same lexicographic representative


def test_bb_equals_oracle_on_random_local_grids(cp4):
    import random
    rng = random.Random(7)
    for _ in range(20):
        centers = tuple(rng.uniform(cp4.hmin, cp4.hmax) for _ in range(cp4.n))
        delta = rng.uniform(0.01, 0.6)
        for oid in (QUAD, COUNT):
            grid = ag.build_local_grid(cp4, centers, delta, include_noop=(oid == COUNT))
            bb = ag.exact_local_solve(cp4, oid, grid)
            ox = ag.exhaustive_local_solve(cp4, oid, grid)
            assert bb.objective == ox.objective and bb.theta == ox.theta


@pytest.mark.parametrize("oid", [QUAD, COUNT])
def test_previous_solution_always_in_next_grid_and_round_monotone_exact(cp4, oid):
    cfg = ag.AdaptiveConfig(objective_id=oid, rho=0.5, max_rounds=6, include_noop=(oid == COUNT))
    res = ag.run_adaptive(cp4, cfg)
    assert all(rr.previous_solution_in_grid for rr in res.rounds)
    assert ag.round_solutions_monotone(res)
    assert ag.archive_is_monotone(res)
    assert res.best_feasible
    assert 3 * cp4.n <= res.max_qubits <= (4 if oid == COUNT else 3) * cp4.n
    if oid == COUNT:
        assert all(0.0 in o for rr in res.rounds for o in rr.options)
    # round 0 equals the official K3 optimum
    ref = exhaustive_discrete_optimum(cp4, objective_id=oid, n_theta=3)
    assert res.rounds[0].objective == ref.optimum


def test_quadratic_adaptive_beats_global_k3_on_cp4(cp4):
    ref = exhaustive_discrete_optimum(cp4, objective_id=QUAD, n_theta=3)
    res = ag.run_adaptive(cp4, ag.AdaptiveConfig(objective_id=QUAD, rho=0.5, max_rounds=6))
    assert res.best_objective is not None and res.best_objective < ref.optimum


def test_count_objective_without_noop_cannot_unmove(cp4):
    # centre every aircraft on a moved heading, pure local grid (no NOOP): the count
    # can never drop below n because no option is exactly theta=0
    centers = (0.3,) * cp4.n
    grid = ag.build_local_grid(cp4, centers, 0.05, include_noop=False)
    assert all(0.0 not in o for o in grid.options)
    rs = ag.exact_local_solve(cp4, COUNT, grid)
    if rs.objective is not None:
        assert rs.objective == float(cp4.n)
    grid2 = ag.build_local_grid(cp4, centers, 0.05, include_noop=True)
    assert all(0.0 in o for o in grid2.options)
    assert grid2.n_qubits == 4 * cp4.n


def test_two_objectives_give_different_rankings(cp4):
    # same local grid, different objective -> different optimal thetas is allowed;
    # what is REQUIRED is that the reported objective is the selected one
    grid = ag.global_grid(cp4, 5)
    rq = ag.exact_local_solve(cp4, QUAD, grid)
    rc = ag.exact_local_solve(cp4, COUNT, grid)
    assert rq.objective == ag.objective_value(cp4, QUAD, rq.theta)
    assert rc.objective == ag.objective_value(cp4, COUNT, rc.theta)
    assert float(rc.objective).is_integer()


# ------------------------------ adversarial -------------------------------- #
def test_center_on_bound_is_clipped_and_deduplicated(cp3):
    grid = ag.build_local_grid(cp3, (cp3.hmax,) * cp3.n, 0.1)
    for opts in grid.options:
        assert len(opts) == 2 and opts[-1] == cp3.hmax
    grid_big = ag.build_local_grid(cp3, (cp3.hmax,) * cp3.n, 10.0)
    for opts in grid_big.options:
        assert opts == (cp3.hmin, cp3.hmax)


def test_center_outside_bounds_rejected(cp3):
    with pytest.raises(ValueError):
        ag.build_local_grid(cp3, (cp3.hmax + 1e-3,) + (0.0,) * (cp3.n - 1), 0.1)


def test_duplicate_option_after_clipping_keeps_centre(cp3):
    c = cp3.hmax - 0.01
    grid = ag.build_local_grid(cp3, (c,) * cp3.n, 0.1)
    for opts in grid.options:
        assert c in opts and cp3.hmax in opts and len(opts) == 3


def test_previous_solution_removed_is_detected(cp3):
    grid = ag.build_local_grid(cp3, (0.0,) * cp3.n, 0.1)
    assert not grid.contains((0.3,) * cp3.n)
    assert grid.contains((0.0,) * cp3.n)
    assert not grid.contains((0.0,) * (cp3.n - 1))


def test_infeasible_round_without_incumbent_stops_honestly(cp3):
    # a tiny delta around NOOP on CP_3 (all aircraft converge) gives no feasible point
    cfg = ag.AdaptiveConfig(objective_id=QUAD, delta0=1e-4, rho=0.5, max_rounds=3)
    res = ag.run_adaptive(cp3, cfg)
    assert not res.best_feasible and res.best_objective is None
    assert res.stop_reason == "infeasible_no_incumbent"
    assert res.rounds[0].feasible is False and res.rounds[0].theta is None


def test_infeasible_round_with_expansion_recovers(cp3):
    cfg = ag.AdaptiveConfig(objective_id=QUAD, delta0=1e-4, rho=0.5, max_rounds=4,
                            expand_on_infeasible=True)
    res = ag.run_adaptive(cp3, cfg)
    assert res.rounds[0].note == "infeasible_expanded_delta"
    # expansion is guarded: at most once, no cycle
    assert sum(1 for rr in res.rounds if rr.note == "infeasible_expanded_delta") == 1


def test_no_cycles_states_are_unique(cp4):
    res = ag.run_adaptive(cp4, ag.AdaptiveConfig(objective_id=QUAD, rho=0.5, max_rounds=8))
    keys = [(rr.centers, rr.delta) for rr in res.rounds]
    assert len(keys) == len(set(keys))
    assert all(a > b for a, b in zip(res.delta_trajectory, res.delta_trajectory[1:]))


@pytest.mark.parametrize("rho", [0.0, 1.0, -0.5, 1.5, math.nan, math.inf])
def test_invalid_rho_rejected(cp3, rho):
    with pytest.raises(ValueError):
        ag.AdaptiveConfig(objective_id=QUAD, rho=rho).validate()


@pytest.mark.parametrize("delta", [0.0, -0.1, math.nan, math.inf])
def test_invalid_delta_rejected(cp3, delta):
    with pytest.raises((ValueError, TypeError)):
        ag.build_local_grid(cp3, (0.0,) * cp3.n, delta)
    with pytest.raises(ValueError):
        ag.AdaptiveConfig(objective_id=QUAD, delta0=delta).validate()


def test_unknown_objective_rejected(cp3):
    with pytest.raises(ValueError):
        ag.AdaptiveConfig(objective_id="bogus_v9").validate()
    with pytest.raises(ValueError):
        ag.objective_value(cp3, "bogus_v9", (0.0,) * cp3.n)


def test_invalid_choice_rejected(cp3):
    grid = ag.build_local_grid(cp3, (0.0,) * cp3.n, 0.1)
    with pytest.raises(ValueError):
        grid.thetas((0, 0))  # wrong length
    with pytest.raises(ValueError):
        grid.thetas((0, 0, 7))  # option out of range
    with pytest.raises(ValueError):
        grid.thetas((0, 0, True))


def test_unknown_init_mode_and_missing_init_theta(cp3):
    cfg = ag.AdaptiveConfig(objective_id=QUAD)
    with pytest.raises(ValueError):
        ag.run_adaptive(cp3, cfg, init_mode="magic")
    with pytest.raises(ValueError):
        ag.run_adaptive(cp3, cfg, init_mode="continuous")
    with pytest.raises(ValueError):
        ag.run_adaptive(cp3, cfg, init_mode="continuous", init_theta=(0.0, math.nan, 0.0))


def test_continuous_init_is_clipped_and_round0_contains_it(cp3):
    cfg = ag.AdaptiveConfig(objective_id=QUAD, delta0=0.05, max_rounds=2)
    res = ag.run_adaptive(cp3, cfg, init_mode="continuous", init_theta=(0.01, 0.02, 5.0))
    assert res.rounds[0].centers == (0.01, 0.02, cp3.hmax)


def test_sa_archive_monotone_even_if_rounds_not(cp4):
    cfg = ag.AdaptiveConfig(objective_id=QUAD, rho=0.5, max_rounds=6, solver="sa",
                            seed=3, sa_sweeps=20, sa_restarts=1)
    res = ag.run_adaptive(cp4, cfg)
    assert ag.archive_is_monotone(res)
    assert all(not rr.exact for rr in res.rounds)


def test_sa_deterministic_given_seed(cp4):
    cfg = ag.AdaptiveConfig(objective_id=QUAD, rho=0.5, max_rounds=4, solver="sa", seed=11)
    a = ag.run_adaptive(cp4, cfg)
    b = ag.run_adaptive(cp4, cfg)
    assert a.best_theta == b.best_theta and a.best_objective == b.best_objective


def test_snap_baseline_reports_conflicts(cp3):
    out = ag.snap_to_grid_baseline(cp3, QUAD, (0.01, 0.01, 0.01), 3)
    assert out["theta"] == (0.0, 0.0, 0.0)
    assert out["feasible"] is False and out["n_conflicts"] > 0


def test_result_carries_flags(cp3):
    res = ag.run_adaptive(cp3, ag.AdaptiveConfig(objective_id=COUNT, include_noop=True, max_rounds=2))
    d = res.to_dict()
    assert d["flags"] == {"experimental": True, "official_benchmark": False, "real_qpu": False}


# ------------------------ red-team additions -------------------------------- #
def test_cost_order_bb_same_optimum_value(cp4):
    for oid in (QUAD, COUNT):
        for k in (3, 5, 9):
            g = ag.global_grid(cp4, k)
            a = ag.exact_local_solve(cp4, oid, g)
            b = ag.exact_local_solve(cp4, oid, g, branch_order="cost")
            assert a.objective == b.objective
    with pytest.raises(ValueError):
        ag.exact_local_solve(cp4, QUAD, ag.global_grid(cp4, 3), branch_order="magic")


def test_reach_bound_theorem(cp3):
    # sum of steps Delta0 * (1 + rho + rho^2 + ...) = Delta0 / (1 - rho)
    assert math.isclose(ag.reach_bound(0.5, 0.5), 1.0)
    assert ag.reach_bound(cp3.hmax, 0.5) >= cp3.hmax - cp3.hmin  # NOOP init + rho>=1/2 covers the box
    assert ag.reach_bound(cp3.hmax, 0.35) < cp3.hmax - cp3.hmin  # rho=0.35 cannot travel from +HMAX to -HMAX
    with pytest.raises(ValueError):
        ag.reach_bound(0.1, 1.0)


def test_bounded_reach_counterexample(cp3):
    # centres at +HMAX with tiny delta0 and rho=0.5: the NOOP region is unreachable
    # (reach = 2*delta0), so the archive stays far from the fine-grid optimum.
    cfg = ag.AdaptiveConfig(objective_id=QUAD, delta0=0.01, rho=0.5, max_rounds=10, stagnation_patience=10)
    res = ag.run_adaptive(cp3, cfg, init_mode="continuous", init_theta=(cp3.hmax,) * cp3.n)
    fine = ag.exact_local_solve(cp3, QUAD, ag.global_grid(cp3, 65), branch_order="cost")
    if res.best_theta is not None:
        assert min(abs(t) for t in res.best_theta) >= cp3.hmax - 2 * 0.01 - 1e-12
        assert res.best_objective > fine.objective


def test_snap_and_greedy_repair_baseline(cp3):
    out = ag.snap_and_greedy_repair_baseline(cp3, QUAD, (0.01, 0.01, 0.01), 3)
    assert out["feasible"] is True and out["n_moves"] >= 1
    assert out["objective"] == ag.objective_value(cp3, QUAD, out["theta"])
    # exact global K3 optimum is a lower bound for any feasible K3 point
    ref = ag.exact_local_solve(cp3, QUAD, ag.global_grid(cp3, 3))
    assert out["objective"] >= ref.objective
