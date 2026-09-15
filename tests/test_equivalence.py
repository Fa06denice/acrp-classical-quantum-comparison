"""Phase 2D — three-way discrete equivalence proof tests.

exhaustive == Gurobi discrete MILP == Exact QUBO, for both objectives, on small
instances. Skipped when AMPL/Gurobi is unavailable (the MILP leg needs it).
Includes a NEGATIVE test proving the harness actually hard-stops on divergence.
"""

from __future__ import annotations

import dataclasses

import pytest

from acrpq.io.loader import InstanceLoader


def _ampl_available() -> bool:
    try:
        from acrpq.classical.discrete_milp import solve_discrete_milp
        r = solve_discrete_milp(
            InstanceLoader().load("CP_3"), objective_id="maneuver_count_v1", n_theta=3)
        return r.status in ("optimal", "infeasible")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _ampl_available(), reason="AMPL/Gurobi unavailable")


@pytest.fixture(scope="module")
def ld():
    return InstanceLoader()


@pytest.mark.parametrize("name,k", [("CP_3", 3), ("CP_4", 3), ("CP_5", 3), ("CP_3", 5)])
@pytest.mark.parametrize("oid", ["quadratic_control_cost_v1", "maneuver_count_v1"])
def test_three_way_equivalence_holds(ld, name, k, oid):
    from acrpq.benchmark.equivalence import prove_equivalence

    r = prove_equivalence(ld.load(name), objective_id=oid, n_theta=k, n_q=1)
    assert r.equivalent, r.mismatches
    assert set(r.legs) == {"exhaustive", "gurobi_milp", "qubo_global_min"}
    # every leg that solved agrees on the optimum within the declared tolerance
    for leg in r.legs.values():
        if leg.status == "optimal":
            assert abs(leg.optimum - r.optimum) <= r.abs_tol
            assert leg.feasible == r.feasible
    # the two enumerating legs agree on the optimal-set cardinality
    assert r.legs["exhaustive"].n_optima == r.legs["qubo_global_min"].n_optima
    # the Gurobi leg is CERTIFIED optimal (not a bare "solved")
    assert r.legs["gurobi_milp"].certified is True
    # small cases enumerate the full binary space; tie semantics are labelled
    assert r.legs["qubo_global_min"].qubo_search in ("full_2^nK", "one_hot_reduction")
    if oid == "maneuver_count_v1":
        assert r.tie_kind == "exact"
    else:
        assert r.tie_kind == "within_absolute_tolerance"


def test_small_config_proves_qubo_min_via_full_binary_enumeration(ld):
    from acrpq.benchmark.equivalence import prove_equivalence

    r = prove_equivalence(ld.load("CP_3"), objective_id="maneuver_count_v1", n_theta=3)
    ql = r.legs["qubo_global_min"]
    assert ql.qubo_search == "full_2^nK"  # 3^3 grid -> 2^9 bitstrings enumerated
    assert r.reduction_theorem["empirically_validated_full_binary"] is True
    assert r.reduction_theorem["onehot_optimum_guaranteed"] is True


def test_maneuver_count_is_exact_zero_tolerance(ld):
    from acrpq.benchmark.equivalence import prove_equivalence

    r = prove_equivalence(ld.load("CP_3"), objective_id="maneuver_count_v1", n_theta=3)
    assert r.abs_tol == 0.0  # integer objective => no tolerance
    # CP_n at K3 needs exactly n-1 maneuvers
    assert r.optimum == 2.0


def test_harness_is_fail_closed_when_reference_cannot_enumerate(ld):
    # if the exhaustive reference can't run, equivalence is NOT proven -> False
    # (regression for the fail-open gap: "nothing compared" must never be "agree")
    from acrpq.benchmark.equivalence import prove_equivalence

    r = prove_equivalence(ld.load("CP_3"), objective_id="maneuver_count_v1",
                          n_theta=3, max_search=1)  # K^n=27 > 1 -> search_too_large
    assert r.legs["exhaustive"].status == "search_too_large"
    assert r.equivalent is False
    assert any(m["field"] == "reference" for m in r.mismatches)


def test_harness_hard_stops_on_a_forced_divergence(ld, monkeypatch):
    import acrpq.benchmark.equivalence as E

    real = E.solve_discrete_milp

    def wrong(inst, **kw):
        r = real(inst, **kw)
        return dataclasses.replace(
            r, optimum=(r.optimum if r.optimum is not None else 0.0) + 1.0)

    monkeypatch.setattr(E, "solve_discrete_milp", wrong)
    r = E.prove_equivalence(ld.load("CP_3"), objective_id="maneuver_count_v1", n_theta=3)
    assert r.equivalent is False
    assert any(m["field"].startswith("optimum") for m in r.mismatches)


def test_nondominant_qubo_fails_closed_without_crashing(ld, monkeypatch):
    # a non-dominant QUBO (weak penalties) must be REFUSED with a diagnosis, never
    # crash the decoder — exercises the fail-closed path in _qubo_leg.
    import acrpq.benchmark.equivalence as E
    from acrpq.quantum.qubo import build_qubo as _real_build

    def weak_build(inst, **kw):
        keep = {k: kw[k] for k in ("discrete", "objective_id", "max_qubits") if k in kw}
        return _real_build(inst, penalty_conflict=1e-6, penalty_onehot=1e-6, **keep)

    monkeypatch.setattr(E, "build_qubo", weak_build)
    r = E.prove_equivalence(ld.load("CP_3"), objective_id="maneuver_count_v1", n_theta=3)
    assert r.equivalent is False
    assert r.legs["qubo_global_min"].status in ("not_wellformed", "reduction_violated")
    assert any(m["field"] in ("qubo_wellformed", "reduction_theorem") for m in r.mismatches)


@pytest.mark.parametrize("bad_nq", [2, 9999, True])
def test_harness_fails_closed_on_falsified_n_qubits(ld, monkeypatch, bad_nq):
    # inject a QUBO with a falsified n_qubits into the harness; it must produce
    # equivalent=False with a diagnosis, never crash and never try to enumerate
    # 2**9999 (verify_qubo_wellformed rejects it before enumeration).
    import dataclasses

    import acrpq.benchmark.equivalence as E
    from acrpq.quantum.qubo import build_qubo as _real_build

    def bad_build(inst, **kw):
        keep = {k: kw[k] for k in ("discrete", "objective_id", "max_qubits") if k in kw}
        return dataclasses.replace(_real_build(inst, **keep), n_qubits=bad_nq)

    monkeypatch.setattr(E, "build_qubo", bad_build)
    r = E.prove_equivalence(ld.load("CP_3"), objective_id="maneuver_count_v1", n_theta=3)
    assert r.equivalent is False
    assert r.legs["qubo_global_min"].status == "not_wellformed"


def _infeasible_enum():
    from acrpq.classical.discrete_enum import DiscreteEnumResult
    return DiscreteEnumResult(
        objective_id="maneuver_count_v1", status="infeasible", feasible=False,
        optimum=None, representative_choice=None, n_optima=0, tie_kind="exact",
        tie_tolerance=0.0, q=(), theta=(), n_conflicts=-1, grid_k=3, search_space=27)


def _milp(status, feasible, optimum, certified):
    from acrpq.classical.discrete_milp import DiscreteMILPResult
    return DiscreteMILPResult(
        objective_id="maneuver_count_v1", status=status, feasible=feasible,
        optimum=optimum, choice=None, q=(), theta=(), n_conflicts=-1, grid_k=3,
        solver="gurobi", certified=certified)


def test_infeasible_branch_all_agree_is_equivalent(ld, monkeypatch):
    import acrpq.benchmark.equivalence as E

    monkeypatch.setattr(E, "exhaustive_discrete_optimum", lambda *a, **k: _infeasible_enum())
    monkeypatch.setattr(E, "solve_discrete_milp",
                        lambda *a, **k: _milp("infeasible", False, None, False))
    monkeypatch.setattr(E, "_qubo_leg", lambda *a, **k: (
        E.LegResult(method="qubo_global_min", status="optimal", feasible=False,
                    optimum=None, representative_choice=None, n_optima=0,
                    tie_kind="exact", tie_tolerance=0.0, n_conflicts=1), {}, []))
    r = E.prove_equivalence(ld.load("CP_3"), objective_id="maneuver_count_v1", n_theta=3)
    assert r.equivalent is True  # all three agree there is no conflict-free assignment


def test_infeasible_branch_divergence_is_caught(ld, monkeypatch):
    import acrpq.benchmark.equivalence as E

    monkeypatch.setattr(E, "exhaustive_discrete_optimum", lambda *a, **k: _infeasible_enum())
    monkeypatch.setattr(E, "solve_discrete_milp",
                        lambda *a, **k: _milp("optimal", True, 2.0, True))  # disagrees!
    monkeypatch.setattr(E, "_qubo_leg", lambda *a, **k: (
        E.LegResult(method="qubo_global_min", status="optimal", feasible=False,
                    optimum=None, representative_choice=None, n_optima=0,
                    tie_kind="exact", tie_tolerance=0.0, n_conflicts=1), {}, []))
    r = E.prove_equivalence(ld.load("CP_3"), objective_id="maneuver_count_v1", n_theta=3)
    assert r.equivalent is False
    assert any(m["field"] == "feasibility" for m in r.mismatches)
