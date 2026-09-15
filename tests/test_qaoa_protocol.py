"""Phase 4 — adversarial tests for the versioned QAOA protocol.

Pins the protocol identity contract (content hash sensitive to every knob),
fail-closed validation, the initial-angle strategies, the ISA spec, and — the
scientific claim of the phase — that a fixed-seed protocol reproduces its QAOA
output bit-for-bit on Aer. No IBM.
"""

from __future__ import annotations

import pytest

from acrpq.io.loader import InstanceLoader
from acrpq.quantum.qaoa import validate_optimizer_options
from acrpq.quantum.qaoa_protocol import (
    DEFAULT_PROTOCOL,
    DETERMINISTIC_RUN_FIELDS,
    PROTOCOL_REGISTRY,
    DecompositionStrategy,
    InitStrategy,
    QAOAProtocol,
    canonical_artifact_sha256,
    coerce_protocol,
    deterministic_run_sha256,
    deterministic_subset,
    run_protocol,
    with_seed,
)
from acrpq.quantum.qubo import build_qubo


def _p(**kw) -> QAOAProtocol:
    base: dict = {"version_name": "t"}
    base.update(kw)
    return QAOAProtocol(**base)


# --- identity / hashing -----------------------------------------------------
def test_protocol_hash_sensitive_to_every_knob():
    ref = _p().protocol_hash
    assert _p(reps=2).protocol_hash != ref
    assert _p(optimizer="NELDER_MEAD").protocol_hash != ref
    assert _p(maxiter=99).protocol_hash != ref
    assert _p(init_strategy=InitStrategy.ZEROS.value).protocol_hash != ref
    assert _p(shots=2048).protocol_hash != ref
    assert _p(sim_method="matrix_product_state").protocol_hash != ref
    assert _p(optimization_level=2).protocol_hash != ref
    assert _p(seed=9999).protocol_hash != ref
    assert _p(max_memory_mb=4096).protocol_hash != ref
    assert _p(version_name="other").protocol_hash != ref
    assert _p().protocol_hash == ref                      # stable
    # a fixed initial point changes identity; and order of optimizer_options doesn't
    fixed = _p(init_strategy=InitStrategy.FIXED.value, initial_point=(0.1, 0.2))
    assert fixed.protocol_hash != ref
    assert (_p(optimizer_options=(("rhobeg", 1.0), ("tol", 2.0))).protocol_hash
            == _p(optimizer_options=(("tol", 2.0), ("rhobeg", 1.0))).protocol_hash)


def test_with_seed_changes_identity():
    a = DEFAULT_PROTOCOL
    b = with_seed(a, a.seed + 1)
    assert b.seed == a.seed + 1
    assert b.protocol_hash != a.protocol_hash


def test_registry_protocols_have_distinct_ids_and_default_matches_benchmark():
    ids = {p.protocol_id for p in PROTOCOL_REGISTRY.values()}
    assert len(ids) == len(PROTOCOL_REGISTRY)
    # aer_default_p1_v1 mirrors the Phase-3 benchmark qaoa_aer leg exactly
    d = DEFAULT_PROTOCOL
    assert (d.reps, d.maxiter, d.optimizer, d.shots, d.sim_method,
            d.optimization_level, d.seed, d.init_strategy) == (
        1, 100, "COBYLA", 1024, "statevector", 1, 1234,
        InitStrategy.SEEDED_RANDOM_UNIFORM.value)


def test_coerce_protocol_is_fail_closed():
    assert coerce_protocol("aer_default_p1_v1") is PROTOCOL_REGISTRY["aer_default_p1_v1"]
    with pytest.raises(ValueError, match="unknown QAOA protocol"):
        coerce_protocol("does_not_exist")


# --- fail-closed validation -------------------------------------------------
@pytest.mark.parametrize("kw", [
    {"reps": 0}, {"reps": True}, {"maxiter": 0}, {"shots": 0}, {"seed": -1},
    {"optimizer": "SPSA"}, {"init_strategy": "warm"}, {"optimization_level": 4},
    {"optimization_level": -1}, {"backend_mode": "ibm_real"},
    {"mixer": "custom_xy"}, {"max_memory_mb": 0},
    {"init_strategy": InitStrategy.FIXED.value},                       # missing point
    {"init_strategy": InitStrategy.FIXED.value, "initial_point": (0.1,)},  # wrong len
    {"initial_point": (0.1, 0.2)},                                    # set w/o fixed
])
def test_invalid_protocols_are_refused(kw):
    with pytest.raises(ValueError):
        _p(**kw)


def test_build_refuses_non_aer_even_after_setattr_bypass():
    # __post_init__ blocks a non-Aer mode at construction; the point-of-use guard
    # also blocks it even if a caller bypasses the frozen dataclass via
    # object.__setattr__ (defense-in-depth for the no-IBM invariant).
    p = _p()
    object.__setattr__(p, "backend_mode", "ibm_real")
    with pytest.raises(ValueError, match="Aer-only|no IBM"):
        p.build_backend()
    with pytest.raises(ValueError, match="Aer-only|no IBM"):
        p.build_solver()


def test_fixed_init_requires_matching_length():
    ok = _p(reps=2, init_strategy=InitStrategy.FIXED.value,
            initial_point=(0.1, 0.2, 0.3, 0.4))                       # len == 2*reps
    assert ok.resolved_initial_point() == [0.1, 0.2, 0.3, 0.4]


# --- initial-angle strategies map to concrete points ------------------------
def test_resolved_initial_point_per_strategy():
    assert _p(init_strategy=InitStrategy.SEEDED_RANDOM_UNIFORM.value
              ).resolved_initial_point() is None
    assert _p(reps=2, init_strategy=InitStrategy.ZEROS.value
              ).resolved_initial_point() == [0.0, 0.0, 0.0, 0.0]
    assert _p(init_strategy=InitStrategy.FIXED.value, initial_point=(1.0, 2.0)
              ).resolved_initial_point() == [1.0, 2.0]


# --- protocol is the single source of truth for the solver ------------------
def test_build_solver_and_backend_match_protocol():
    p = _p(reps=2, maxiter=42, optimizer="NELDER_MEAD", shots=512,
           optimization_level=2, seed=7,
           init_strategy=InitStrategy.ZEROS.value)
    solver = p.build_solver()
    assert solver.reps == 2 and solver.maxiter == 42
    assert solver.optimizer == "NELDER_MEAD"
    assert solver.initial_point == [0.0, 0.0, 0.0, 0.0]
    b = solver.backend
    assert b.shots == 512 and b.seed == 7 and b.optimization_level == 2
    assert b.mode.value == "aer_sim" and b.sim_method.value == "statevector"


# --- ISA spec ---------------------------------------------------------------
def test_aer_isa_spec_is_generic_fully_connected():
    pytest.importorskip("qiskit_aer")
    isa = DEFAULT_PROTOCOL.aer_isa_spec()
    assert isa.isa_kind == "aer_generic"
    assert isa.coupling == "fully_connected"
    assert isa.routing_required is False
    assert isa.optimization_level == DEFAULT_PROTOCOL.optimization_level
    assert isa.seed_transpiler == DEFAULT_PROTOCOL.seed
    assert isa.n_basis_gates and isa.basis_gates_sha256          # captured + stable
    assert isa.basis_gates_sha256 == DEFAULT_PROTOCOL.aer_isa_spec().basis_gates_sha256


# --- the scientific claim: a fixed-seed protocol reproduces on Aer ----------
def test_run_protocol_is_reproducible_and_stamped_and_objective_aware():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    ld = InstanceLoader()
    fast = _p(maxiter=25)                                   # small budget, still seeded
    qubo = build_qubo(ld.load("CP_3"), n_theta=3, n_q=1, objective_id="maneuver_count_v1")
    a = run_protocol(qubo, fast)
    b = run_protocol(qubo, fast)
    for f in ("best_bitstring", "best_energy", "optimal_params", "objective_value",
              "circuit_depth", "eval_count"):
        assert a[f] == b[f], f
    # stamped with the protocol identity
    assert a["protocol_id"] == fast.protocol_id
    assert a["protocol_hash"] == fast.protocol_hash
    # objective-aware value on the JOB's axis (maneuver count is an integer here),
    # not the always-quadratic decode().objective
    assert float(a["objective_value"]).is_integer()
    # raw one-hot validity kept separate from decoded feasibility
    assert set(a) >= {"raw_onehot_valid", "repair_applied", "decoded_feasible"}
    assert a["qpu_time_s"] is None                          # Aer: no provider time


def test_a_different_seed_can_change_the_run():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    ld = InstanceLoader()
    qubo = build_qubo(ld.load("CP_4"), n_theta=3, n_q=1, objective_id="quadratic_control_cost_v1")
    r1 = run_protocol(qubo, _p(maxiter=25, seed=1))
    r2 = run_protocol(qubo, _p(maxiter=25, seed=2))
    # different seeds are different protocols (identity differs); the run MAY differ
    assert r1["protocol_hash"] != r2["protocol_hash"]


# --- artifact hash honesty --------------------------------------------------
def test_canonical_artifact_hash_excludes_timing_and_repo_dirty():
    art: dict = {"schema": "x", "reproducibility_demo": {"runs": [
        {"best_bitstring": "010", "wall_time_s": 0.1, "qpu_time_s": None}]}}
    h1 = canonical_artifact_sha256(art)
    art["reproducibility_demo"]["runs"][0]["wall_time_s"] = 999.0
    assert canonical_artifact_sha256(art) == h1                    # timing excluded
    art["provenance"] = {"repository_dirty": False}
    h2 = canonical_artifact_sha256(art)
    art["provenance"]["repository_dirty"] = True
    assert canonical_artifact_sha256(art) == h2                    # repo_dirty excluded
    art["reproducibility_demo"]["runs"][0]["best_bitstring"] = "111"
    assert canonical_artifact_sha256(art) != h2                    # science IS hashed


# ===========================================================================
# Phase 4.1 — additional adversarial coverage
# ===========================================================================

# --- optimizer_options: real allowlist, refusals, wiring, hashing -----------
def test_validate_optimizer_options_allowlist_and_refusals():
    assert dict(validate_optimizer_options("COBYLA", {"rhobeg": 0.7, "tol": 1e-4})) == {
        "rhobeg": 0.7, "tol": 1e-4}
    assert dict(validate_optimizer_options("NELDER_MEAD", {"xatol": 1e-3, "maxfev": 50})) == {
        "xatol": 1e-3, "maxfev": 50}
    assert dict(validate_optimizer_options("COBYLA", None)) == {}
    for bad in ({"nope": 1.0}, {"tol": True}, {"rhobeg": float("nan")},
                {"rhobeg": float("inf")}, {"maxfev": 1.5}, {"disp": True}):
        with pytest.raises(ValueError):
            validate_optimizer_options("COBYLA", bad)
    # maxfev is COBYLA-unknown but NELDER_MEAD-known -> allowlist is per optimizer
    with pytest.raises(ValueError):
        validate_optimizer_options("COBYLA", {"maxfev": 10})
    # duplicate key via a pair-list
    with pytest.raises(ValueError, match="duplicate"):
        validate_optimizer_options("COBYLA", [("tol", 1e-3), ("tol", 1e-4)])


def test_protocol_optimizer_options_are_validated_hashed_and_forwarded():
    ref = _p().protocol_hash
    p = _p(optimizer_options=(("rhobeg", 0.5), ("tol", 1e-5)))
    assert p.protocol_hash != ref                                   # hashed
    assert p.optimizer_options == (("rhobeg", 0.5), ("tol", 1e-5))  # canonical sorted tuple
    # an unsupported option is refused (never hashed as an ignored knob)
    with pytest.raises(ValueError, match="unsupported optimizer option"):
        _p(optimizer_options=(("bogus", 1.0),))
    # forwarded to the concrete solver, immutably
    solver = p.build_solver()
    assert dict(solver.optimizer_options) == {"rhobeg": 0.5, "tol": 1e-5}
    with pytest.raises(TypeError):
        solver.optimizer_options["x"] = 1  # frozen MappingProxyType


# --- decomposition strategy is part of the identity -------------------------
def test_decomposition_strategy_is_in_identity_and_default_is_none():
    assert DEFAULT_PROTOCOL.decomposition_strategy == DecompositionStrategy.NONE.value
    none_id = PROTOCOL_REGISTRY["aer_default_p1_v1"].protocol_id
    dec_id = PROTOCOL_REGISTRY["aer_decomposed_p1_v1"].protocol_id
    assert none_id != dec_id
    # same knobs but different decomposition -> distinct identity
    a = _p(decomposition_strategy=DecompositionStrategy.NONE.value)
    b = _p(decomposition_strategy=DecompositionStrategy.EXACT_CONNECTED_COMPONENTS.value)
    assert a.protocol_hash != b.protocol_hash
    with pytest.raises(ValueError):
        _p(decomposition_strategy="magic")


def test_run_protocol_has_no_free_decompose_override():
    with pytest.raises(TypeError):
        run_protocol(object(), DEFAULT_PROTOCOL, decompose=True)  # type: ignore[call-arg]


def test_decomposed_run_publishes_component_composition():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    ld = InstanceLoader()
    qubo = build_qubo(ld.load("FP_4"), n_theta=3, n_q=1,
                      objective_id="maneuver_count_v1", max_qubits=4096)
    r = run_protocol(qubo, PROTOCOL_REGISTRY["aer_decomposed_p1_v1"])
    d = r["decomposition"]
    assert d["strategy"] == "exact_connected_components" and d["applied"] is True
    assert d["n_components"] == 3
    assert sorted(d["component_qubit_sizes"]) == [6, 6, 12]
    assert sum(len(bits) for bits in d["component_bits"]) == r["n_qubits"]


# --- closed __post_init__ (exact adversarial cases) -------------------------
@pytest.mark.parametrize("kw", [
    {"schema": "acrpq-qaoa-protocol/2"},              # wrong schema
    {"decoding": "argmax"},                            # wrong decoding rule
    {"version_name": ""},                              # empty name
    {"version_name": "a b"},                            # unsafe charset (space)
    {"version_name": "x" * 65},                         # too long
    {"sim_method": "tensor_network"},                  # not an AerMethod value
    {"mixer": "xy"}, {"parametrization": "warm"},
    {"init_strategy": InitStrategy.FIXED.value, "initial_point": (0.1, float("nan"))},
    {"init_strategy": InitStrategy.FIXED.value, "initial_point": (0.1, float("inf"))},
    {"init_strategy": InitStrategy.FIXED.value, "initial_point": (0.1, True)},  # bool angle
])
def test_closed_post_init_refuses(kw):
    with pytest.raises(ValueError):
        _p(**kw)


# --- registry immutability --------------------------------------------------
def test_registry_is_read_only():
    from types import MappingProxyType
    assert isinstance(PROTOCOL_REGISTRY, MappingProxyType)
    with pytest.raises(TypeError):
        PROTOCOL_REGISTRY["aer_default_p1_v1"] = DEFAULT_PROTOCOL  # type: ignore[index]
    with pytest.raises(TypeError):
        del PROTOCOL_REGISTRY["aer_default_p1_v1"]  # type: ignore[attr-defined]


# --- concurrency: the RNG lock preserves per-seed determinism ---------------
def test_concurrent_qaoa_runs_stay_deterministic_under_the_lock():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    import threading

    from acrpq.quantum.backends import BackendConfig, BackendMode
    from acrpq.quantum.qaoa import QAOASolver
    ld = InstanceLoader()
    qubo = build_qubo(ld.load("CP_3"), n_theta=3, n_q=1, objective_id="maneuver_count_v1")

    def run(seed):
        cfg = BackendConfig(mode=BackendMode.AER_SIM, shots=256, seed=seed)
        return QAOASolver(backend=cfg, reps=1, maxiter=15).solve(qubo).best_bitstring

    seeds = (11, 22, 33)
    ref = {s: run(s) for s in seeds}                    # serial references
    # Many workers racing across several seeds in the same process, repeated —
    # without the lock they would perturb each other's process-global RNG and
    # diverge from the references. 9 threads (3x oversubscribed) x 3 rounds.
    for _ in range(3):
        results: list[tuple[int, str]] = []
        lock = threading.Lock()

        def worker(s):
            r = run(s)
            with lock:
                results.append((s, r))

        threads = [threading.Thread(target=worker, args=(seeds[i % len(seeds)],))
                   for i in range(9)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for s, r in results:
            assert r == ref[s], f"seed {s} diverged under concurrency"


# --- auditable reproducibility ledger ---------------------------------------
def test_deterministic_subset_and_sha_are_auditable():
    run = {"best_bitstring": "010", "best_energy": 2.0, "optimal_params": [0.1, 0.2],
           "objective_value": 2.0, "n_qubits": 3, "circuit_depth": 5,
           "wall_time_s": 1.23, "qpu_time_s": None, "decomposition": {"applied": False}}
    sub = deterministic_subset(run)
    assert "wall_time_s" not in sub and "qpu_time_s" not in sub  # volatile dropped
    assert set(sub) == set(DETERMINISTIC_RUN_FIELDS)
    # a reader can recompute the sha from the stored subset alone
    assert deterministic_run_sha256(sub) == deterministic_run_sha256(run)
    # changing a deterministic field changes the sha; changing timing does not
    run2 = dict(run, wall_time_s=99.0)
    assert deterministic_run_sha256(run2) == deterministic_run_sha256(run)
    run3 = dict(run, best_bitstring="111")
    assert deterministic_run_sha256(run3) != deterministic_run_sha256(run)


# --- Phase 4.1 contre-review fixes ------------------------------------------
def test_solve_revalidates_optimizer_options_after_post_construction_mutation():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    from acrpq.quantum.backends import BackendConfig, BackendMode
    from acrpq.quantum.qaoa import QAOASolver
    ld = InstanceLoader()
    qubo = build_qubo(ld.load("CP_3"), n_theta=3, n_q=1, objective_id="maneuver_count_v1")
    solver = QAOASolver(backend=BackendConfig(mode=BackendMode.AER_SIM, shots=128, seed=1),
                        reps=1, maxiter=8)
    # ordinary (non-frozen) attribute mutation to an UNSUPPORTED option
    solver.optimizer_options = {"disp": True}
    # solve() must re-validate and refuse it fail-closed, not forward it silently
    with pytest.raises(ValueError, match="unsupported optimizer option|must not be a bool"):
        solver.solve(qubo)
    solver.optimizer_options = {"rhobeg": float("nan")}
    with pytest.raises(ValueError, match="finite"):
        solver.solve(qubo)


def test_decomposed_on_single_component_reports_applied_false():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    ld = InstanceLoader()
    # CP_3 is a single connected component -> the decomposed strategy falls back to
    # a whole solve; the report must be HONEST (applied=False, n_components=1), not
    # applied=True with null composition.
    qubo = build_qubo(ld.load("CP_3"), n_theta=3, n_q=1, objective_id="maneuver_count_v1")
    r = run_protocol(qubo, PROTOCOL_REGISTRY["aer_decomposed_p1_v1"])
    d = r["decomposition"]
    assert d["strategy"] == "exact_connected_components"
    assert d["applied"] is False
    assert d["n_components"] == 1
    assert d["component_qubit_sizes"] == [r["n_qubits"]]
    assert sum(len(b) for b in d["component_bits"]) == r["n_qubits"]


# ===========================================================================
# Phase 4.2 — closed initial_point + optimizer_options semantic bounds
# ===========================================================================
_FIXED = InitStrategy.FIXED.value


def test_initial_point_caller_list_mutation_is_a_noop():
    pts = [0.1, 0.2]
    p = _p(init_strategy=_FIXED, initial_point=pts)
    h0 = p.protocol_hash
    pts[0] = 999.0                                   # mutate the caller's container
    assert p.initial_point == (0.1, 0.2)             # protocol angles unchanged
    assert p.protocol_hash == h0                     # identity unchanged
    assert isinstance(p.initial_point, tuple)        # stored as an immutable tuple
    assert p.resolved_initial_point() == [0.1, 0.2]


def test_initial_point_tuple_input_unchanged_and_immutable():
    p = _p(init_strategy=_FIXED, initial_point=(0.3, 0.4))
    assert p.initial_point == (0.3, 0.4)
    assert isinstance(p.initial_point, tuple)
    with pytest.raises(TypeError):
        p.initial_point[0] = 1.0  # type: ignore[index]   # internal is immutable


def test_initial_point_generator_is_normalised():
    p = _p(init_strategy=_FIXED, initial_point=(x for x in [0.5, 0.6]))
    assert p.initial_point == (0.5, 0.6)             # documented normalisation
    assert isinstance(p.initial_point, tuple)


@pytest.mark.parametrize("bad", ["ab", b"ab", bytearray(b"ab")])
def test_initial_point_refuses_str_and_bytes(bad):
    with pytest.raises(ValueError, match="str/bytes|sequence of reals"):
        _p(init_strategy=_FIXED, initial_point=bad)


@pytest.mark.parametrize("bad", [
    (0.1, float("nan")), (0.1, float("inf")), (0.1, float("-inf")), (0.1, True),
    (0.1,), (0.1, 0.2, 0.3),                          # wrong length (2*reps == 2)
])
def test_initial_point_refuses_bad_values_and_length(bad):
    with pytest.raises(ValueError):
        _p(init_strategy=_FIXED, initial_point=bad)


def test_build_solver_receives_a_fresh_defensive_list():
    pytest.importorskip("qiskit_optimization")
    p = _p(init_strategy=_FIXED, initial_point=(0.1, 0.2))
    s1 = p.build_solver()
    s2 = p.build_solver()
    assert s1.initial_point == [0.1, 0.2] and isinstance(s1.initial_point, list)
    assert s1.initial_point is not s2.initial_point            # fresh each build
    assert s1.initial_point is not p.initial_point             # not the stored tuple
    s1.initial_point[0] = 999.0                                # mutating the solver's list
    assert p.initial_point == (0.1, 0.2)                       # cannot affect the protocol


def test_optimizer_options_semantic_bounds_are_enforced_up_front():
    # positive step sizes / tolerances; evaluation budget >= 1
    assert dict(validate_optimizer_options("COBYLA", {"rhobeg": 0.5, "tol": 1e-6})) == {
        "rhobeg": 0.5, "tol": 1e-6}
    assert dict(validate_optimizer_options("NELDER_MEAD", {"xatol": 1e-4, "maxfev": 1})) == {
        "xatol": 1e-4, "maxfev": 1}
    bad_opt: dict
    for bad_opt in ({"rhobeg": 0.0}, {"rhobeg": -1.0}, {"tol": 0.0}, {"tol": -1e-9}):
        with pytest.raises(ValueError, match="must be >"):
            validate_optimizer_options("COBYLA", bad_opt)
    for bad_opt in ({"xatol": 0.0}, {"xatol": -1.0}, {"maxfev": 0}, {"maxfev": -5}):
        with pytest.raises(ValueError, match="must be >"):
            validate_optimizer_options("NELDER_MEAD", bad_opt)
    # enforced at the protocol level too (not a late SciPy/Qiskit failure)
    with pytest.raises(ValueError, match="must be >"):
        _p(optimizer_options=(("rhobeg", 0.0),))
