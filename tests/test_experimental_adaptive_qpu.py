"""Tests for the EXPERIMENTAL heterogeneous-grid QUBO, sampling decoder and
hash-chained adaptive rounds (gate study). No IBM contact anywhere.
"""

from __future__ import annotations

import itertools
import json
import math
import shutil

import pytest

from acrpq.experimental import adaptive_grid as ag
from acrpq.experimental import adaptive_qpu as aq
from acrpq.quantum.qubo import build_qubo

QUAD = "quadratic_control_cost_v1"
COUNT = "maneuver_count_v1"
qiskit = pytest.importorskip("qiskit")


# ------------------------------- formulation ------------------------------- #
@pytest.mark.parametrize("name", ["cp3", "cp4"])
def test_uniform_k3_hetero_qubo_equals_official(request, name):
    inst = request.getfixturevalue(name)
    off = build_qubo(inst, n_theta=3, n_q=1)
    het = aq.build_hetero_qubo(inst, ag.global_grid(inst, 3), QUAD)
    assert het.constant == off.constant
    assert dict(het.linear) == dict(off.linear)
    assert dict(het.quadratic) == dict(off.quadratic)
    assert het.penalty_conflict == off.penalty_conflict and het.penalty_onehot == off.penalty_onehot
    for combo in itertools.product((0, 1), repeat=het.n_qubits):
        assert het.energy(list(combo)) == off.energy(list(combo))


def test_offsets_and_inverse_for_heterogeneous_blocks(cp3):
    grid = ag.build_local_grid(cp3, (cp3.hmax, -0.1, 0.0), 0.1)
    het = aq.build_hetero_qubo(cp3, grid, QUAD)
    assert [len(o) for o in grid.options] == [2, 3, 3]
    assert het.offsets == (0, 2, 5) and het.n_qubits == 8
    seen = set()
    for i in cp3.aircraft():
        for o in range(het.block_size(i)):
            b = het.var_index(i, o)
            assert het.inv_index(b) == (i, o)
            seen.add(b)
    assert seen == set(range(8))
    with pytest.raises(ValueError):
        het.var_index(1, 2)
    with pytest.raises(ValueError):
        het.inv_index(8)


def test_energy_equals_cost_plus_penalties_on_onehot_states(cp4):
    grid = ag.build_local_grid(cp4, (0.3, -0.2, 0.0, cp4.hmin), 0.15)
    het = aq.build_hetero_qubo(cp4, grid, QUAD)
    for ch in itertools.product(*(range(len(o)) for o in grid.options)):
        bs = aq.encode_choice(het, ch)
        d = aq.decode_bitstring(het, bs)
        assert d.choice == ch and d.onehot_valid and not d.repaired
        assert math.isclose(d.energy, d.objective + het.penalty_conflict * d.n_conflicts, abs_tol=1e-9)
        assert d.objective == ag.objective_value(cp4, QUAD, d.theta)


def test_dominance_flag_and_block_size_one(cp3):
    grid = ag.build_local_grid(cp3, (cp3.hmax, 0.0, 0.0), 10.0)  # first block clipped to 2, others 2
    het = aq.build_hetero_qubo(cp3, grid, QUAD)
    assert het.meta["onehot_optimum_guaranteed"] == 1.0
    # brute-force global minimum is one-hot
    best = min(itertools.product((0, 1), repeat=het.n_qubits), key=lambda c: (het.energy(list(c)), c))
    d = aq.decode_bitstring(het, "".join(map(str, best)))
    assert d.onehot_valid


def test_duplicate_or_unsorted_options_refused(cp3):
    bad = ag.LocalGrid(centers=(0.0,) * 3, delta=0.1, include_noop=False,
                       options=((0.0, 0.0, 0.1), (0.0,), (0.0,)))
    with pytest.raises(ValueError):
        aq.build_hetero_qubo(cp3, bad, QUAD)
    bad2 = ag.LocalGrid(centers=(0.0,) * 3, delta=0.1, include_noop=False,
                        options=((0.1, 0.0), (0.0,), (0.0,)))
    with pytest.raises(ValueError):
        aq.build_hetero_qubo(cp3, bad2, QUAD)


def test_qubo_hash_depends_on_grid_order_and_values(cp3):
    g1 = ag.build_local_grid(cp3, (0.1, -0.1, 0.0), 0.05)
    g2 = ag.build_local_grid(cp3, (-0.1, 0.1, 0.0), 0.05)  # aircraft permuted
    g3 = ag.build_local_grid(cp3, (0.1, -0.1, 0.0), 0.04)
    hs = {aq.build_hetero_qubo(cp3, g, QUAD).canonical_hash() for g in (g1, g2, g3)}
    assert len(hs) == 3
    assert aq.build_hetero_qubo(cp3, g1, QUAD).canonical_hash() != aq.build_hetero_qubo(cp3, g1, COUNT).canonical_hash()


# --------------------------------- decoding -------------------------------- #
def _grid(inst, sizes_hint):
    if sizes_hint == [3, 3, 3]:
        return ag.build_local_grid(inst, (0.0,) * 3, 0.1)
    if sizes_hint == [2, 3, 3]:
        return ag.build_local_grid(inst, (inst.hmax, 0.0, 0.0), 0.1)
    if sizes_hint == [1, 2, 3]:
        g = ag.build_local_grid(inst, (inst.hmax, inst.hmin, 0.0), 0.1)
        return ag.LocalGrid(centers=g.centers, delta=g.delta, include_noop=False,
                            options=((inst.hmax,), g.options[1], g.options[2]))
    raise AssertionError


@pytest.mark.parametrize("sizes", [[3, 3, 3], [2, 3, 3], [1, 2, 3]])
def test_encode_decode_bijection_all_onehot_states(cp3, sizes):
    grid = _grid(cp3, sizes)
    het = aq.build_hetero_qubo(cp3, grid, QUAD)
    assert [len(o) for o in grid.options] == sizes
    seen = set()
    for ch in itertools.product(*(range(len(o)) for o in grid.options)):
        bs = aq.encode_choice(het, ch)
        assert bs not in seen
        seen.add(bs)
        d = aq.decode_bitstring(het, bs)
        assert d.choice == ch and d.theta == grid.thetas(ch)
    assert len(seen) == math.prod(sizes)


def test_qiskit_little_endian_roundtrip(cp3):
    het = aq.build_hetero_qubo(cp3, _grid(cp3, [2, 3, 3]), QUAD)
    bs = aq.encode_choice(het, (1, 0, 2))
    qiskit_key = bs[::-1]
    dec = aq.decode_counts(het, {qiskit_key: 10}, bit_order="qiskit")
    assert dec.samples[0].choice == (1, 0, 2)
    dec2 = aq.decode_counts(het, {bs: 10}, bit_order="qubo")
    assert dec2.samples[0].choice == (1, 0, 2)
    # wrong endianness assumption changes the decoded choice for an asymmetric string
    dec3 = aq.decode_counts(het, {bs: 10}, bit_order="qiskit")
    assert dec3.samples[0].choice != (1, 0, 2) or dec3.samples[0].repaired


def test_int_keys_and_spaces_and_quasi(cp3):
    het = aq.build_hetero_qubo(cp3, _grid(cp3, [3, 3, 3]), QUAD)
    bs = aq.encode_choice(het, (0, 1, 2))
    key_int = int(bs[::-1], 2)
    d1 = aq.decode_counts(het, {key_int: 5}, bit_order="qiskit")
    d2 = aq.decode_counts(het, {bs[::-1][:3] + " " + bs[::-1][3:]: 5}, bit_order="qiskit")
    d3 = aq.decode_counts(het, {bs[::-1]: 1.0}, bit_order="qiskit", kind="quasi")
    assert d1.samples[0].choice == d2.samples[0].choice == d3.samples[0].choice == (0, 1, 2)


@pytest.mark.parametrize("raw,kind", [
    ({"0" * 8: 1}, "counts"),          # too short
    ({"0" * 10: 1}, "counts"),         # too long
    ({"00000000x": 1}, "counts"),      # non-binary
    ({}, "counts"),                    # empty
    ({"0" * 9: 0}, "counts"),          # zero shots
    ({"0" * 9: -1}, "counts"),         # negative
    ({"0" * 9: True}, "counts"),       # bool
    ({"0" * 9: 1.5}, "counts"),        # non-int count
    ({"0" * 9: -0.1, "1" * 9: 1.1}, "quasi"),  # negative weight
    ({"0" * 9: math.nan}, "quasi"),    # non-finite
    ({"0" * 9: 0.5}, "quasi"),         # does not sum to 1
    ({512: 1}, "counts"),              # int key out of range for 9 bits
    ({(1, 2): 1}, "counts"),           # unsupported key type
])
def test_malformed_raw_refused(cp3, raw, kind):
    het = aq.build_hetero_qubo(cp3, _grid(cp3, [3, 3, 3]), QUAD)
    with pytest.raises(ValueError):
        aq.decode_counts(het, raw, bit_order="qiskit", kind=kind)


def test_shots_mismatch_and_bad_bit_order(cp3):
    het = aq.build_hetero_qubo(cp3, _grid(cp3, [3, 3, 3]), QUAD)
    with pytest.raises(ValueError):
        aq.decode_counts(het, {"0" * 9: 3}, shots=4)
    with pytest.raises(ValueError):
        aq.decode_counts(het, {"0" * 9: 3}, bit_order="big")


def test_repair_multiple_and_zero_bits(cp3):
    grid = _grid(cp3, [3, 3, 3])
    het = aq.build_hetero_qubo(cp3, grid, QUAD)
    bs = list("0" * 9)
    bs[het.var_index(1, 0)] = "1"
    bs[het.var_index(1, 2)] = "1"  # two bits in block 1, none in blocks 2 and 3
    d = aq.decode_bitstring(het, "".join(bs))
    assert not d.onehot_valid and d.repaired and d.n_onehot_violations == 3
    # blocks 2/3 repaired to the centre (0.0); block 1 to the objective-minimal among set
    assert d.theta[1] == 0.0 and d.theta[2] == 0.0
    assert d.choice[0] in (0, 2) and het.cost[0][d.choice[0]] == min(het.cost[0][0], het.cost[0][2])
    assert d.penalty > 0


def test_best_feasible_independent_of_counts_order(cp3):
    het = aq.build_hetero_qubo(cp3, ag.global_grid(cp3, 3), QUAD)
    items = {}
    for ch in itertools.product(range(3), repeat=3):
        items[aq.encode_choice(het, ch)[::-1]] = 1 + sum(ch)
    keys = list(items)
    d_a = aq.decode_counts(het, {k: items[k] for k in keys})
    d_b = aq.decode_counts(het, {k: items[k] for k in reversed(keys)})
    assert d_a.best_feasible.bitstring_qubo == d_b.best_feasible.bitstring_qubo
    assert d_a.best_feasible.objective == min(s.objective for s in d_a.samples if s.feasible)
    ref = ag.exact_local_solve(cp3, QUAD, ag.global_grid(cp3, 3))
    assert d_a.best_feasible.objective == ref.objective


def test_secondary_tiebreak_cannot_degrade_objective(cp4):
    het = aq.build_hetero_qubo(cp4, ag.global_grid(cp4, 3), QUAD)
    raw = {aq.encode_choice(het, ch)[::-1]: 1 for ch in itertools.product(range(3), repeat=4)}
    d = aq.decode_counts(het, raw)
    feas = [s for s in d.samples if s.feasible]
    assert d.best_feasible.objective == min(s.objective for s in feas)


# ------------------------------ circuit / ISA ------------------------------ #
def test_logical_circuit_matches_official_builder_for_k3(cp3):
    from acrpq.dashboard.hardware_validation import build_parameterized_qaoa

    off = build_qubo(cp3, n_theta=3, n_q=1)
    qc_off, g, b = build_parameterized_qaoa(off, 1)
    bound = qc_off.assign_parameters({g[0]: 0.4, b[0]: 0.3}, inplace=False)
    het = aq.build_hetero_qubo(cp3, ag.global_grid(cp3, 3), QUAD)
    qc_het = aq.build_fixed_angle_circuit(het, [0.4], [0.3])
    # same multiset of operations with same params (order within a layer may differ)
    def bag(c):
        return sorted((i.operation.name, tuple(c.find_bit(q).index for q in i.qubits),
                       tuple(round(float(p), 12) for p in i.operation.params)) for i in c.data)
    assert bag(bound) == bag(qc_het)


def test_isa_transpile_fake_marrakesh_reproducible(cp3):
    fp = pytest.importorskip("qiskit_ibm_runtime.fake_provider")
    be = fp.FakeMarrakesh()
    het = aq.build_hetero_qubo(cp3, _grid(cp3, [2, 3, 3]), QUAD)
    qc = aq.build_fixed_angle_circuit(het, [0.4], [0.3])
    isa, rep = aq.transpile_isa(qc, be, seed_transpiler=1, optimization_level=0)
    assert rep.basis_ok and rep.reproducible and rep.n_logical == 8 and rep.is_fake
    assert len(rep.layout_physical) == 8 and len(set(rep.layout_physical)) == 8
    assert rep.n_2q > 0 and rep.depth > 0
    assert aq.circuit_fingerprint(isa) == rep.isa_hash


def test_effective_angles_rule(cp3):
    p = aq.make_preregistration(cp3, protocol=aq.ADAPTIVE_PROTOCOL_QPU_PILOT, driver_kind="aer_sim",
                                objective_id=QUAD, backend="aer_simulator")
    g, b = aq.effective_angles(p, 14.0, 7.0)
    assert g == (0.8,) and b == (0.3,)
    p2 = aq.make_preregistration(cp3, protocol=aq.ADAPTIVE_PROTOCOL_QPU_PILOT, driver_kind="aer_sim",
                                 objective_id=QUAD, backend="aer_simulator")
    p2 = aq.Preregistration(**{**p2.to_dict(), "angle_rule": "fixed", "init_theta": None,
                               "gammas": (0.4,), "betas": (0.3,)})
    assert aq.effective_angles(p2, 14.0, 7.0) == ((0.4,), (0.3,))
    with pytest.raises(ValueError):
        aq.effective_angles(p, 14.0, 0.0)


# ------------------------------ chain identity ----------------------------- #
@pytest.fixture
def chain_dir(tmp_path):
    d = tmp_path / "chain"
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _exact_prereg(inst, **kw):
    base = dict(protocol=aq.ADAPTIVE_PROTOCOL_LOCAL_EXACT, driver_kind="exact_local", objective_id=QUAD,
                max_rounds=4, backend="none")
    base.update(kw)
    return aq.make_preregistration(inst, **base)


def test_exact_chain_runs_verifies_and_is_idempotent(cp3, chain_dir):
    p = _exact_prereg(cp3)
    out = aq.run_chain(chain_dir, p, transpile=False)
    assert out["n_rounds"] == 4 and out["problems"] == []
    store = aq.ChainStore(chain_dir)
    assert store.verify() == []
    objs = [store.load_round(r)["selected_objective"] for r in range(4)]
    assert objs == sorted(objs, reverse=True)  # exact rounds: non-increasing
    ref = ag.run_adaptive(cp3, ag.AdaptiveConfig(objective_id=QUAD, rho=0.5, max_rounds=4))
    assert out["archive_objective"] == ref.best_objective
    out2 = aq.run_chain(chain_dir, p, transpile=False)  # resume: no-op
    assert out2["n_rounds"] == 4
    man = json.loads((chain_dir / "chain_manifest.json").read_text())
    assert man["adaptive_driver"] == "exact_local" and man["real_qpu"] is False
    assert man["experimental"] is True and man["official_benchmark"] is False


def test_round_zero_qubo_hash_equals_uniform_k3(cp3, chain_dir):
    p = _exact_prereg(cp3, max_rounds=1)
    aq.run_chain(chain_dir, p, transpile=False)
    rec = aq.ChainStore(chain_dir).load_round(0)
    # the round-0 grid is the official K3 grid built as a LOCAL grid (centres 0, delta HMAX);
    # the hash includes grid provenance (centres/delta), so compare with the same construction
    same = aq.build_hetero_qubo(cp3, ag.build_local_grid(cp3, (0.0,) * 3, cp3.hmax), QUAD)
    assert rec["qubo_hash"] == same.canonical_hash()
    # coefficient-level identity with the uniform official grid (provenance differs, coefficients not)
    uni = aq.build_hetero_qubo(cp3, ag.global_grid(cp3, 3), QUAD)
    assert dict(uni.linear) == dict(same.linear) and dict(uni.quadratic) == dict(same.quadratic)
    assert rec["n_qubits"] == 9 and rec["block_sizes"] == [3, 3, 3]


def _mutate(chain_dir, r, fn):
    p = chain_dir / f"round_{r:03d}" / "round.json"
    rec = json.loads(p.read_text())
    fn(rec)
    p.write_text(json.dumps(rec, indent=1, sort_keys=True))


@pytest.mark.parametrize("mutation", [
    lambda rec: rec.__setitem__("center_before", [0.01, 0.0, 0.0]),
    lambda rec: rec.__setitem__("selected_objective", 0.0),
    lambda rec: rec.__setitem__("selected_theta", [0.0, 0.0, 0.0]),
    lambda rec: rec.__setitem__("archive_objective", 0.0),
    lambda rec: rec.__setitem__("grids", rec["grids"][::-1]),
    lambda rec: rec.__setitem__("driver_kind", "qpu_result"),
    lambda rec: rec.__setitem__("real_qpu", True),
    lambda rec: rec.__setitem__("effective_gammas", [0.1]),
    lambda rec: rec.__setitem__("qubo_hash", "0" * 64),
    lambda rec: rec.__setitem__("delta", rec["delta"] * 0.9),
])
def test_round_mutations_are_detected(cp3, chain_dir, mutation):
    aq.run_chain(chain_dir, _exact_prereg(cp3, max_rounds=3), transpile=False)
    _mutate(chain_dir, 1, mutation)
    store = aq.ChainStore(chain_dir)
    assert store.verify() != []
    with pytest.raises(aq.ChainIntegrityError):
        aq.run_chain(chain_dir, _exact_prereg(cp3, max_rounds=4), transpile=False)


def test_mutation_with_rehash_still_breaks_chain_link(cp3, chain_dir):
    # attacker recomputes round_hash after editing round 1: round 2's parent hash no longer matches
    aq.run_chain(chain_dir, _exact_prereg(cp3, max_rounds=3), transpile=False)

    def edit(rec):
        rec["selected_objective"] = 0.0
        rec["round_hash"] = aq.compute_round_hash(rec)
    _mutate(chain_dir, 1, edit)
    problems = aq.ChainStore(chain_dir).verify()
    assert any("round 2" in p for p in problems)


def test_deleted_parent_and_swapped_rounds(cp3, chain_dir):
    aq.run_chain(chain_dir, _exact_prereg(cp3, max_rounds=3), transpile=False)
    a, b = chain_dir / "round_001" / "round.json", chain_dir / "round_002" / "round.json"
    ta, tb = a.read_text(), b.read_text()
    a.write_text(tb)
    b.write_text(ta)
    assert aq.ChainStore(chain_dir).verify() != []
    a.write_text(ta)
    b.write_text(tb)
    assert aq.ChainStore(chain_dir).verify() == []
    shutil.rmtree(chain_dir / "round_001")
    assert aq.ChainStore(chain_dir).verify() != []


def test_different_preregistration_in_same_dir_refused(cp3, chain_dir):
    aq.run_chain(chain_dir, _exact_prereg(cp3, max_rounds=2), transpile=False)
    with pytest.raises(aq.ChainIntegrityError):
        aq.run_chain(chain_dir, _exact_prereg(cp3, max_rounds=2, rho=0.6), transpile=False)
    with pytest.raises(aq.ChainIntegrityError):
        aq.run_chain(chain_dir, _exact_prereg(cp3, max_rounds=2, objective_id=COUNT, include_noop=True),
                     transpile=False)


def test_instance_change_refused(cp3, cp4, chain_dir):
    p3 = _exact_prereg(cp3, max_rounds=1)
    aq.run_chain(chain_dir, p3, transpile=False)
    forged = aq.Preregistration(**{**p3.to_dict(), "instance_name": "CP_4", "init_theta": None,
                                   "gammas": (0.4,), "betas": (0.3,)})
    with pytest.raises(aq.ChainIntegrityError):
        aq.run_chain(chain_dir, forged, transpile=False)


def test_qpu_driver_refused_locally(cp3, chain_dir):
    p = aq.make_preregistration(cp3, protocol=aq.ADAPTIVE_PROTOCOL_QPU_PILOT, driver_kind="qpu_result",
                                objective_id=QUAD, backend="ibm_marrakesh")
    with pytest.raises(aq.ChainIntegrityError):
        aq.run_chain(chain_dir, p, transpile=False)


def test_preregistration_validation():
    with pytest.raises(ValueError):
        aq.Preregistration(adaptive_protocol_id=aq.ADAPTIVE_PROTOCOL_LOCAL_EXACT, instance_name="CP_3",
                           instance_sha256="x", objective_id=QUAD, initialization_kind="noop", init_theta=None,
                           rho=0.5, delta0=0.5, max_rounds=2, driver_kind="aer_sim", shots_per_round=1,
                           backend="none", gammas=(0.4,), betas=(0.3,), seed_transpiler=1, optimization_level=0,
                           include_noop=False, no_feasible_policy="stop", stagnation_patience=1,
                           reference_identity={}, source_commit="x").validate()
    with pytest.raises(ValueError):
        aq.Preregistration(adaptive_protocol_id=aq.ADAPTIVE_PROTOCOL_QPU_PILOT, instance_name="CP_3",
                           instance_sha256="x", objective_id=QUAD, initialization_kind="noop", init_theta=None,
                           rho=0.5, delta0=0.5, max_rounds=2, driver_kind="qpu_result", shots_per_round=512,
                           backend="ibm_torino", gammas=(0.4,), betas=(0.3,), seed_transpiler=1, optimization_level=0,
                           include_noop=False, no_feasible_policy="stop", stagnation_patience=1,
                           reference_identity={}, source_commit="x").validate()


def test_no_feasible_policies(cp3, chain_dir):
    # tiny delta0 around NOOP on CP_3: every local grid infeasible -> no incumbent -> stop
    p = _exact_prereg(cp3, max_rounds=3, delta0=1e-4, no_feasible_policy="keep_center_retry_once")
    out = aq.run_chain(chain_dir, p, transpile=False)
    rec = aq.ChainStore(chain_dir).load_round(0)
    assert rec["status"] == "closed_no_feasible" and rec["no_feasible_action"] == "stop_no_incumbent"
    assert out["n_rounds"] == 1 and out["archive_objective"] is None


def test_aer_chain_cp3_completes_with_normalised_angles(cp3, chain_dir):
    pytest.importorskip("qiskit_aer")
    p = aq.make_preregistration(cp3, protocol=aq.ADAPTIVE_PROTOCOL_QPU_PILOT, driver_kind="aer_sim",
                                objective_id=QUAD, max_rounds=3, backend="aer_simulator", shots_per_round=512)
    out = aq.run_chain(chain_dir, p, transpile=False)
    store = aq.ChainStore(chain_dir)
    assert store.verify() == [] and out["n_rounds"] == 3
    archive = [store.load_round(r)["archive_objective"] for r in range(3)]
    assert all(a is None or b is None or b <= a for a, b in zip(archive, archive[1:]))
    rec0 = store.load_round(0)
    assert rec0["effective_gammas"] == [0.4] and rec0["raw_result_hash"] is not None
    man = json.loads((chain_dir / "chain_manifest.json").read_text())
    assert man["adaptive_driver"] == "aer_sim" and man["real_qpu"] is False


# ------------------------- red-team reproducers (expert 6) ------------------- #
def _rechain(chain_dir, n_rounds):
    """Attacker helper: recompute every round_hash and parent links consistently."""
    prev_h, prev_s = aq.GENESIS_HASH, aq.GENESIS_HASH
    for r in range(n_rounds):
        p = chain_dir / f"round_{r:03d}" / "round.json"
        rec = json.loads(p.read_text())
        rec["parent_round_hash"], rec["parent_selected_candidate_hash"] = prev_h, prev_s
        rec["round_hash"] = aq.compute_round_hash(rec)
        p.write_text(json.dumps(rec, indent=1, sort_keys=True))
        prev_h, prev_s = rec["round_hash"], rec["selected_candidate_hash"]


def test_full_rechained_forgery_is_detected_by_deep_verify(cp3, chain_dir):
    aq.run_chain(chain_dir, _exact_prereg(cp3, max_rounds=4), transpile=False)
    p = chain_dir / "round_001" / "round.json"
    rec = json.loads(p.read_text())
    rec["selected_theta"] = [0.0, 0.0, 0.0]
    rec["selected_objective"] = 1e-9
    rec["archive_objective"] = 1e-9
    rec["archive_theta"] = [0.0, 0.0, 0.0]
    p.write_text(json.dumps(rec, indent=1, sort_keys=True))
    r2 = chain_dir / "round_002" / "round.json"
    rec2 = json.loads(r2.read_text())
    rec2["center_before"] = [0.0, 0.0, 0.0]
    r2.write_text(json.dumps(rec2, indent=1, sort_keys=True))
    _rechain(chain_dir, 4)
    store = aq.ChainStore(chain_dir)
    assert store.verify(deep=False) != [] or store.verify(deep=True) != []
    deep = store.verify(deep=True)
    assert any("round 1" in x and ("re-solve" in x or "best feasible" in x or "archive" in x) for x in deep)


def test_raw_counts_replacement_detected(cp3, chain_dir):
    aq.run_chain(chain_dir, _exact_prereg(cp3, max_rounds=2), transpile=False)
    (chain_dir / "round_001" / "raw_counts.json").write_text(json.dumps({"counts": {"111111111": 999}}))
    assert any("raw_counts.json" in x for x in aq.ChainStore(chain_dir).verify())
    (chain_dir / "round_001" / "decoded.json").write_text(json.dumps({"totally": "fabricated"}))
    assert any("decoded.json" in x for x in aq.ChainStore(chain_dir).verify())


@pytest.mark.parametrize("field,value", [
    ("experimental", False), ("official_benchmark", True), ("instance_sha256", "0" * 64),
    ("objective_id", COUNT), ("penalty_onehot", 1.0), ("angle_rule", "fixed"), ("block_sizes", [1, 1, 1]),
])
def test_previously_uncovered_fields_now_detected(cp3, chain_dir, field, value):
    aq.run_chain(chain_dir, _exact_prereg(cp3, max_rounds=2), transpile=False)
    _mutate(chain_dir, 1, lambda rec: rec.__setitem__(field, value))
    assert aq.ChainStore(chain_dir).verify() != []
    _rechain(chain_dir, 2)  # even with consistent re-hashing the semantic checks fire
    assert aq.ChainStore(chain_dir).verify() != []


def test_preregistration_edit_detected_and_flags_present(cp3, chain_dir):
    aq.run_chain(chain_dir, _exact_prereg(cp3, max_rounds=1), transpile=False)
    pre = json.loads((chain_dir / "preregistration.json").read_text())
    assert pre["experimental"] is True and pre["official_benchmark"] is False and pre["real_qpu"] is False
    pre["source_commit"] = "forged"
    (chain_dir / "preregistration.json").write_text(json.dumps(pre, indent=1, sort_keys=True))
    problems = aq.ChainStore(chain_dir).verify()
    assert any("preregistration_hash" in x or "chain_id" in x for x in problems)


def test_manifest_flag_flip_detected(cp3, chain_dir):
    aq.run_chain(chain_dir, _exact_prereg(cp3, max_rounds=1), transpile=False)
    m = json.loads((chain_dir / "chain_manifest.json").read_text())
    m["official_benchmark"] = True
    (chain_dir / "chain_manifest.json").write_text(json.dumps(m, indent=1, sort_keys=True))
    assert any("manifest flags" in x for x in aq.ChainStore(chain_dir).verify())


def test_min_delta_floor_stops_cleanly(cp3, chain_dir):
    p = aq.make_preregistration(cp3, protocol=aq.ADAPTIVE_PROTOCOL_LOCAL_EXACT, driver_kind="exact_local",
                                objective_id=QUAD, max_rounds=40, rho=0.5, backend="none", stagnation_patience=40)
    p = aq.Preregistration(**{**p.to_dict(), "min_delta": 0.01, "init_theta": None, "gammas": (0.4,), "betas": (0.3,)})
    out = aq.run_chain(chain_dir, p, transpile=False)
    assert out["n_rounds"] < 40
    last = aq.ChainStore(chain_dir).load_round(out["n_rounds"] - 1)
    assert last["next_delta"] < 0.01 <= last["delta"]
    with pytest.raises(ValueError):
        aq.Preregistration(**{**p.to_dict(), "min_delta": 0.0, "init_theta": None, "gammas": (0.4,), "betas": (0.3,)}).validate()


def test_writer_lock_refuses_second_writer(cp3, chain_dir):
    store = aq.ChainStore(chain_dir)
    store.init(_exact_prereg(cp3, max_rounds=1))
    with store.writer_lock():
        with pytest.raises(aq.ChainIntegrityError):
            aq.run_round(store, _exact_prereg(cp3, max_rounds=1), 0, transpile=False)
    # lock released: the round can now run
    aq.run_round(store, _exact_prereg(cp3, max_rounds=1), 0, transpile=False)
    assert store.verify() == []
