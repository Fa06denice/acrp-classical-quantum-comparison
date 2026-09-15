"""Phase 3 — versioned, deterministic tie-break policy (closure of MoE condition 3b).

Expert A found that the one-hot repair broke ties on the QUADRATIC grid cost whatever the
QUBO's objective_id. At w = 0.5 the NOOP option is uniquely cheapest so the choice was
accidentally right; at w in {0, 1} (reachable through the dashboard API) several options
tie at zero and the repair could return a non-NOOP option, inflating the deviated-aircraft
count. These tests pin the policy AND its contract: it never changes an objective value.
"""
from __future__ import annotations

import dataclasses

import pytest

from acrpq.io.loader import InstanceLoader
from acrpq.objectives import option_cost_vector
from acrpq.quantum.decode import decode_assignment
from acrpq.quantum.qubo import build_qubo
from acrpq.selection_policy import (
    SELECTION_POLICY_CRITERIA,
    SELECTION_POLICY_ID,
    choose_option,
    selection_policy_doc,
)

MANEUVER = "maneuver_count_v1"


def _qubo(w, *, n_theta=3, n_q=3, objective_id=MANEUVER):
    inst = dataclasses.replace(InstanceLoader().load("CP_3"), w=w)
    return inst, build_qubo(inst, n_theta=n_theta, n_q=n_q, objective_id=objective_id)


def _zero_quadratic_cost_options(d):
    return [o for o in range(d.k) if abs(d.cost[o]) < 1e-15]


@pytest.mark.parametrize("w", [0.0, 1.0])
def test_repair_prefers_noop_at_degenerate_weights(w):
    """w = 0 (or 1) makes several options tie at zero QUADRATIC cost. The repair must
    still return NOOP — the option that costs nothing under the maneuver-count objective."""
    _inst, q = _qubo(w)
    d = q.discrete
    noop = d.grid.noop_index()
    tied = _zero_quadratic_cost_options(d)
    assert len(tied) > 1, f"w={w} must expose a genuine quadratic tie"
    other = next(o for o in tied if o != noop)
    bits = {d.var_index(1, other): 1, d.var_index(1, noop): 1}
    assert decode_assignment(q, bits)[1] == noop


def test_several_exactly_tied_candidates_resolve_deterministically():
    """All options tied on the primary objective: the policy must pick one, always the
    same one, by fewest-deviation then smallest unweighted variation then lowest index."""
    _inst, q = _qubo(0.0)
    d = q.discrete
    tied = _zero_quadratic_cost_options(d)
    first = choose_option(d.grid, tied)
    assert choose_option(d.grid, list(reversed(tied))) == first
    assert choose_option(d.grid, set(tied)) == first
    assert first == d.grid.noop_index(), "among exact ties, NOOP must win (criterion 2)"


def test_order_independence_of_input_container():
    """The result must not depend on dict/set/list iteration order."""
    _inst, q = _qubo(0.0)
    d = q.discrete
    noop = d.grid.noop_index()
    tied = _zero_quadratic_cost_options(d)
    a, b = (o for o in tied if o != noop).__next__(), noop
    fwd = decode_assignment(q, {d.var_index(1, a): 1, d.var_index(1, b): 1})[1]
    rev = decode_assignment(q, {d.var_index(1, b): 1, d.var_index(1, a): 1})[1]
    assert fwd == rev == noop


@pytest.mark.parametrize("w", [0.0, 0.5, 1.0])
def test_policy_never_changes_the_primary_objective_value(w):
    """The tie-break only orders candidates ALREADY equal under the objective, so a
    repaired assignment can never cost more than the objective-minimal candidate.

    Note this test deliberately sets TWO bits per block: with a single bit set the repair
    short-circuits and never consults the policy, so a one-hot input could not detect a
    policy that overrides the objective (the earlier version of this test could not).
    """
    inst, q = _qubo(w, objective_id=MANEUVER)
    d = q.discrete
    cost = option_cost_vector(MANEUVER, d.grid, w)
    seen_multi = 0
    for a in range(d.k):
        for b in range(a + 1, d.k):
            got = decode_assignment(q, {d.var_index(1, a): 1, d.var_index(1, b): 1})[1]
            assert cost[got] == min(cost[a], cost[b]), (
                f"repair returned option {got} (cost {cost[got]}) over the "
                f"objective-minimal one among {{{a}, {b}}}")
            seen_multi += 1
    assert seen_multi > 0


@pytest.mark.parametrize("w", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("objective_id", [MANEUVER, "quadratic_control_cost_v1"])
def test_repair_is_never_worse_than_the_objective_minimum(w, objective_id):
    """Regression, reproduced by an independent reviewer: consulting the policy on the
    FULL candidate set let it override the objective. At w = 0 the quadratic cost ignores
    theta, so a zero-cost option can carry a large theta while a costlier option has a
    smaller total variation — and the policy picked the costlier one.

    Concretely (CP_3, n_theta = n_q = 3, w = 0, quadratic): options {1, 3} have costs
    0.0036 and 0.0; the policy alone returned option 1. This asserts the invariant over
    every candidate pair, for both objectives.
    """
    inst, q = _qubo(w, objective_id=objective_id)
    d = q.discrete
    cost = option_cost_vector(objective_id, d.grid, w)
    for a in range(d.k):
        for b in range(a + 1, d.k):
            got = decode_assignment(q, {d.var_index(1, a): 1, d.var_index(1, b): 1})[1]
            assert cost[got] <= min(cost[a], cost[b]) + 0.0, (
                f"{objective_id} w={w}: repair chose {got} (cost {cost[got]}) over "
                f"min({cost[a]}, {cost[b]})")


def test_policy_is_versioned_and_published():
    doc = selection_policy_doc()
    assert doc["selection_policy_id"] == SELECTION_POLICY_ID == "acrpq-selection-policy/1"
    assert tuple(doc["criteria"]) == SELECTION_POLICY_CRITERIA
    assert "never changes an objective value" in doc["note"]


def test_policy_is_exposed_by_the_api_separately_from_the_objective():
    """Published as selection_policy, never folded into the objective (contract rule 4)."""
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app

    d = TestClient(create_app()).post(
        "/api/preflight", json={"instance": "CP_3", "n_theta": 3, "n_q": 1}).json()
    assert d["selection_policy"]["selection_policy_id"] == SELECTION_POLICY_ID
    assert d["objective_id"] == "quadratic_control_cost_v1"     # distinct fields


def test_deterministic_reproduction_across_fresh_builds():
    """Same inputs, fresh objects: identical decode. No hidden state, no epsilon."""
    outs = []
    for _ in range(3):
        _inst, q = _qubo(0.0)
        d = q.discrete
        tied = _zero_quadratic_cost_options(d)
        other = next(o for o in tied if o != d.grid.noop_index())
        outs.append(decode_assignment(
            q, {d.var_index(1, other): 1, d.var_index(1, d.grid.noop_index()): 1})[1])
    assert len(set(outs)) == 1
