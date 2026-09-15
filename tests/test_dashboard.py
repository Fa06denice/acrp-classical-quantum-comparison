"""Dashboard API tests — skipped automatically without the [dashboard] extra.

The pure payload builders are tested without a server; the HTTP layer is tested
with FastAPI's TestClient when FastAPI is installed.
"""

from __future__ import annotations

import json

import pytest

from acrpq._optional import have

# The payload builders are pure (no FastAPI) and always testable.
from acrpq.dashboard.api import instance_payload, solve_payload  # noqa: E402
from acrpq.io.loader import InstanceLoader  # noqa: E402


def test_instance_payload_initial():
    inst = InstanceLoader().load("CP_4")
    p = instance_payload(inst)
    assert p["name"] == "CP_4"
    assert p["n"] == 4
    assert len(p["aircraft"]) == 4
    # CP_4: all pairs conflict with no maneuver
    assert p["n_conflicts"] == inst.n_pairs()
    a0 = p["aircraft"][0]
    assert {"id", "x", "y", "vx", "vy", "heading_deg"} <= set(a0)


def test_solve_payload_reference_resolves():
    p = solve_payload(InstanceLoader(), "CP_4", solver="reference", n_theta=5)
    assert p["solution"]["solver"] == "classical_discrete"
    assert p["solution"]["feasible"] is True
    assert p["n_conflicts"] == 0  # radar now shows the resolved state
    assert p["solution"]["n_resolved"] == 6


def test_qubo_cache_reuses_same_config_and_isolates_others():
    from acrpq.dashboard.api import _get_qubo
    from acrpq.discretize import DiscreteACRP
    from acrpq.model import ManeuverGrid

    inst = InstanceLoader().load("CP_4")
    grid = ManeuverGrid.build(n_theta=3, n_q=1, instance=inst)
    discrete = DiscreteACRP.build(inst, grid)

    q1 = _get_qubo(inst, discrete, "CP_4", None, 3, 1, None, 1024)
    q2 = _get_qubo(inst, discrete, "CP_4", None, 3, 1, None, 1024)
    assert q1 is q2  # identical configuration -> a single shared build

    # A different grid is a different configuration: it must never reuse q1.
    grid5 = ManeuverGrid.build(n_theta=5, n_q=1, instance=inst)
    discrete5 = DiscreteACRP.build(inst, grid5)
    q3 = _get_qubo(inst, discrete5, "CP_4", None, 5, 1, None, 1024)
    assert q3 is not q1
    assert q3.n_qubits != q1.n_qubits


def test_model_cache_shares_discretisation_and_isolates_configs():
    # The expensive stage (DiscreteACRP.build) must be cached and shared across
    # the comparison routes, yet never reused across different configurations.
    from acrpq.dashboard.api import _MODEL_CACHE, _get_discrete

    loader = InstanceLoader()
    _MODEL_CACHE.clear()
    _, _, d1 = _get_discrete(loader, "CP_4", None, 3, 1, None)
    _, _, d2 = _get_discrete(loader, "CP_4", None, 3, 1, None)
    assert d1 is d2  # shared build for identical config
    _, _, d_w = _get_discrete(loader, "CP_4", None, 3, 1, 0.25)
    _, _, d_k = _get_discrete(loader, "CP_4", None, 5, 1, None)
    assert d_w is not d1 and d_k is not d1  # weight and grid each isolate
    # bounded (no unbounded growth / leak)
    for i in range(40):
        _get_discrete(loader, "CP_4", [(i % 3) + 1, (i % 3) + 2], 3, 1, None)
    assert len(_MODEL_CACHE) <= 16


def test_weighted_config_does_not_crash_and_isolates_cache():
    from acrpq.dashboard.api import _get_qubo, preflight_payload
    from acrpq.discretize import DiscreteACRP
    from acrpq.model import ManeuverGrid

    loader = InstanceLoader()
    # regression: a scalar weight used to raise "'float' object is not iterable"
    for w in (None, 0, 1, 0.25):
        preflight_payload(loader, "CP_3", n_theta=3, w=w)
        assert solve_payload(loader, "CP_3", solver="reference", n_theta=3, w=w)["solution"]

    def build(w):
        inst = loader.load("CP_3", w=w)
        grid = ManeuverGrid.build(n_theta=3, n_q=1, instance=inst)
        return inst, DiscreteACRP.build(inst, grid)

    inst_a, disc_a = build(0.25)
    q1 = _get_qubo(inst_a, disc_a, "CP_3", None, 3, 1, 0.25, 1024)
    q2 = _get_qubo(inst_a, disc_a, "CP_3", None, 3, 1, 0.25, 1024)
    assert q1 is q2  # identical weight -> shared build

    inst_b, disc_b = build(0.5)
    q3 = _get_qubo(inst_b, disc_b, "CP_3", None, 3, 1, 0.5, 1024)
    assert q3 is not q1  # w=0.25 must never reuse the w=0.5 QUBO


def test_solve_stamps_demo_profile_and_is_not_citable():
    p = solve_payload(InstanceLoader(), "CP_4", solver="reference", n_theta=3)
    prof = p["profile"]
    assert prof["name"] == "demo"
    assert prof["citable"] is False
    assert prof["params"]["seed"] == 42 and prof["params"]["n_theta"] == 3


def test_scientific_profile_is_citable_and_invalid_profile_rejected():
    p = solve_payload(InstanceLoader(), "CP_4", solver="reference", profile="scientific")
    assert p["profile"]["name"] == "scientific"
    assert p["profile"]["citable"] is True
    with pytest.raises(ValueError, match="unknown profile"):
        solve_payload(InstanceLoader(), "CP_4", solver="reference", profile="bogus")


def test_scientific_run_reports_seed_dispersion():
    from acrpq.dashboard.api import scientific_run

    out = scientific_run(
        InstanceLoader(), "CP_4", solver="reference", seeds=[1, 2, 3], n_theta=3
    )
    agg = out["aggregate"]
    assert out["profile"] == "scientific"
    assert agg["n_runs"] == 3 and agg["seeds"] == [1, 2, 3]
    assert agg["feasible_rate"] == 1.0
    # the reference solver is deterministic: no dispersion across seeds
    assert agg["objective_stdev"] == 0.0
    assert agg["deterministic"] is True
    assert all(r["profile"]["name"] == "scientific" for r in out["runs"])


def test_scientific_run_rejects_too_many_and_duplicate_seeds():
    from acrpq.dashboard.api import scientific_run

    with pytest.raises(ValueError, match="12 seeds"):
        scientific_run(InstanceLoader(), "CP_4", solver="reference", seeds=list(range(13)))
    with pytest.raises(ValueError, match="distinct"):
        scientific_run(InstanceLoader(), "CP_4", solver="reference", seeds=[1, 1, 2])


def test_reference_exact_is_provable_optimum_and_allows_gap():
    p = solve_payload(InstanceLoader(), "CP_4", solver="reference", n_theta=3)
    proof = p["solution"]["proof_status"]
    assert proof["status"] == "exact_optimum"
    assert proof["gap_allowed"] is True  # a proven optimum may anchor an optimality gap


def test_reference_heuristic_fallback_forbids_optimality_gap():
    # CP_10 at K=7 enumerates 7^10 = 282M states, far past the reference exact
    # cap (10M), so the solver falls back to its documented heuristic.
    p = solve_payload(InstanceLoader(), "CP_10", solver="reference", n_theta=7)
    proof = p["solution"]["proof_status"]
    assert proof["status"] == "feasible_heuristic"
    assert proof["gap_allowed"] is False  # heuristic anchor -> 'optimality gap' forbidden
    assert "optimality gap" in proof["caveat"]


def test_explanation_summary_names_dominant_contributors():
    p = solve_payload(InstanceLoader(), "CP_4", solver="reference", n_theta=3)
    summary = p["solution"]["explanation"]["summary"]
    assert summary["dominant_energy_term"] in {"maneuver_cost", "conflict_penalty", "onehot_penalty"}
    assert summary["top_cost_aircraft"] in set(range(1, 5))
    assert summary["active_conflict_pairs"] == 0  # a feasible resolution has no active conflicts


def test_config_hash_is_order_independent_and_distinct():
    from acrpq.dashboard.persist import config_hash

    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
    with pytest.raises((TypeError, ValueError)):
        config_hash({"not_json": object()})
    with pytest.raises(ValueError):
        config_hash({"not_finite": float("nan")})


def test_qubo_exact_proof_names_the_enumerated_domain():
    p = solve_payload(InstanceLoader(), "CP_4", solver="qubo-exact", n_theta=3)
    sol = p["solution"]
    assert sol["exact_domain"] == "one_hot_subspace"
    assert sol["global_unconstrained_qubo"] is False
    assert sol["proof_status"]["label"] == "exact one-hot QUBO minimum"
    assert "not a claim about every unconstrained bitstring" in sol["proof_status"]["caveat"]


def _sci_record(seeds=(1, 2)):
    from acrpq.dashboard.api import scientific_run
    from acrpq.dashboard.persist import build_scientific_record

    out = scientific_run(InstanceLoader(), "CP_4", solver="reference", seeds=list(seeds), n_theta=3)
    return build_scientific_record(
        out, instance="CP_4", subset=None,
        config={"solver": "reference", "n_theta": 3, "seeds": list(seeds)},
    )


def _mp_scientific_record(seed):
    from acrpq.dashboard.api import scientific_run
    from acrpq.dashboard.persist import build_scientific_record

    out = scientific_run(InstanceLoader(), "CP_4", solver="reference", seeds=[seed], n_theta=3)
    return build_scientific_record(
        out, instance="CP_4", subset=None,
        config={"solver": "reference", "n_theta": 3, "seeds": [seed]},
    )


def _mp_save_worker(args):
    """Top-level (picklable) worker: build + save a scientific run in a subprocess."""
    root, seed = args
    from acrpq.dashboard.persist import save_run

    return save_run(_mp_scientific_record(seed), root=root)["run_id"]


def _mp_save_fixed_worker(args):
    """Save a run under a FIXED run_id; report whether this process won the race."""
    root, fixed_id, seed = args
    from acrpq.dashboard.persist import save_run

    rec = _mp_scientific_record(seed)
    rec["run_id"] = fixed_id
    try:
        save_run(rec, root=root)
        return "ok"
    except FileExistsError:
        return "exists"


def test_same_second_saves_get_distinct_ids_and_never_overwrite(tmp_path):
    from acrpq.dashboard.persist import list_runs, save_run, verify_run

    # identical configuration built + saved twice within the same second
    r1 = save_run(_sci_record(), root=str(tmp_path))
    r2 = save_run(_sci_record(), root=str(tmp_path))
    assert r1["run_id"] != r2["run_id"]  # nonce guarantees uniqueness
    ids = {row["run_id"] for row in list_runs("scientific", root=str(tmp_path))}
    assert {r1["run_id"], r2["run_id"]} <= ids  # both indexed, neither clobbered
    assert verify_run("scientific", r1["run_id"], root=str(tmp_path))["json_ok"]
    assert verify_run("scientific", r2["run_id"], root=str(tmp_path))["json_ok"]


def test_save_receipt_and_manifest_carry_verifiable_hashes(tmp_path):
    from acrpq.dashboard.persist import list_runs, save_run, verify_run

    receipt = save_run(_sci_record(), root=str(tmp_path))
    assert receipt["json_sha256"].startswith("sha256:")
    assert receipt["csv_sha256"].startswith("sha256:")
    entry = list_runs("scientific", root=str(tmp_path))[0]
    assert entry["json_sha256"] == receipt["json_sha256"]
    assert verify_run("scientific", receipt["run_id"], root=str(tmp_path)) == {
        "run_id": receipt["run_id"], "json_ok": True, "csv_ok": True,
    }


def test_tampered_artifact_fails_verification(tmp_path):
    from pathlib import Path

    from acrpq.dashboard.persist import read_run, save_run, verify_run

    receipt = save_run(_sci_record(), root=str(tmp_path))
    art = Path(receipt["path"])
    art.write_text(art.read_text() + "\n")  # a single trailing byte is enough
    with pytest.raises(ValueError, match="integrity check FAILED"):
        verify_run("scientific", receipt["run_id"], root=str(tmp_path))
    # a verifying read also refuses the tampered artifact
    with pytest.raises(ValueError, match="integrity"):
        read_run("scientific", receipt["run_id"], root=str(tmp_path), verify=True)


def test_no_silent_overwrite_of_an_existing_run(tmp_path):
    from acrpq.dashboard.persist import save_run

    rec = _sci_record()
    fixed = "20260101T000000000000Z-deadbeef-cafe1234"
    rec["run_id"] = fixed
    save_run(rec, root=str(tmp_path))
    rec2 = _sci_record()
    rec2["run_id"] = fixed
    with pytest.raises(FileExistsError):
        save_run(rec2, root=str(tmp_path))


def test_concurrent_saves_are_all_distinct_and_intact(tmp_path):
    import threading

    from acrpq.dashboard.persist import list_runs, save_run, verify_run

    records = [_sci_record() for _ in range(8)]  # fresh (unique id) per save
    receipts, errors = [], []

    def worker(rec):
        try:
            receipts.append(save_run(rec, root=str(tmp_path)))
        except Exception as exc:  # pragma: no cover - surfaced via the assert
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(r,)) for r in records]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    ids = [r["run_id"] for r in receipts]
    assert len(set(ids)) == 8  # every concurrent save got a unique id
    assert len(list_runs("scientific", root=str(tmp_path))) == 8  # manifest intact
    for rid in ids:
        assert verify_run("scientific", rid, root=str(tmp_path))["json_ok"]


def test_malicious_run_id_is_rejected(tmp_path):
    from acrpq.dashboard.persist import read_run, verify_run

    for bad in ["../../etc/passwd", "a/b", "foo.json", "x" * 300, ""]:
        with pytest.raises(ValueError, match="invalid run_id"):
            read_run("scientific", bad, root=str(tmp_path))
        with pytest.raises(ValueError, match="invalid run_id"):
            verify_run("scientific", bad, root=str(tmp_path))


@pytest.mark.parametrize("fail_on", [1, 2, 3])
def test_save_failure_rolls_back_and_indexes_nothing(tmp_path, monkeypatch, fail_on):
    # Inject a failure at each publish step (CSV=1, JSON=2, meta=3). The save must
    # raise, index nothing, and leave no orphan artifact behind.
    from acrpq.dashboard import persist as P

    real = P._publish_atomic_exclusive
    calls = {"n": 0}

    def flaky(path, text):
        calls["n"] += 1
        if calls["n"] == fail_on:
            raise OSError(f"simulated crash at publish step {fail_on}")
        return real(path, text)

    monkeypatch.setattr(P, "_publish_atomic_exclusive", flaky)
    with pytest.raises(OSError, match="simulated crash"):
        P.save_run(_sci_record(), root=str(tmp_path))

    assert P.list_runs("scientific", root=str(tmp_path)) == []  # never falsely indexed
    leftovers = [
        p.name for p in (tmp_path / "scientific").glob("*")
        if not p.name.startswith(".tmp-")
    ]
    assert leftovers == []  # rollback removed the partially published artifacts


def test_orphan_without_meta_is_invisible_and_recoverable(tmp_path):
    from acrpq.dashboard.persist import list_runs, recover_orphans

    d = tmp_path / "scientific"
    d.mkdir(parents=True)
    rid = "20260101T000000000000Z-deadbeef-cafe1234"
    (d / f"{rid}.json").write_text("{}")  # a crash left artifacts but no meta
    (d / f"{rid}.csv").write_text("seed\n1\n")
    (d / ".tmp-stale.json").write_text("partial")

    assert list_runs("scientific", root=str(tmp_path)) == []  # commit point absent
    # fresh files are within the default lease window and must be left untouched
    assert recover_orphans("scientific", root=str(tmp_path)) == []
    assert (d / f"{rid}.json").exists()
    # past the lease (min_age_s=0 here) they are recovered
    removed = recover_orphans("scientific", root=str(tmp_path), min_age_s=0)
    assert set(removed) == {f"{rid}.json", f"{rid}.csv", ".tmp-stale.json"}
    assert list(d.glob("*")) == []  # store is clean afterwards


def test_rollback_leaves_a_concurrently_created_artifact_intact(tmp_path, monkeypatch):
    # Regression for the same-run_id race: a losing save must delete ONLY the
    # files it created, never a winner's. Here B publishes its CSV, then loses the
    # JSON to a "winner" that lands first — B's rollback must remove B's CSV but
    # leave the winner's JSON untouched.
    from acrpq.dashboard import persist as P

    real = P._publish_atomic_exclusive

    def racing(path, text):
        if path.name.endswith(".json") and not path.name.endswith(".meta.json"):
            real(path, '{"owner":"winner"}')  # a racing winner lands the JSON first
            raise FileExistsError("json already exists (raced)")
        return real(path, text)

    monkeypatch.setattr(P, "_publish_atomic_exclusive", racing)
    rec = _sci_record()
    rid = rec["run_id"]
    with pytest.raises(FileExistsError):
        P.save_run(rec, root=str(tmp_path))

    d = tmp_path / "scientific"
    assert not (d / f"{rid}.csv").exists()  # B's own artifact rolled back
    assert (d / f"{rid}.json").read_text() == '{"owner":"winner"}'  # winner intact
    assert not (d / f"{rid}.meta.json").exists()  # never committed


def test_same_run_id_across_processes_has_exactly_one_winner(tmp_path):
    import concurrent.futures as cf
    import multiprocessing as mp

    from acrpq.dashboard.persist import list_runs, verify_run

    fixed = "20260101T000000000000Z-deadbeef-cafe1234"
    args = [(str(tmp_path), fixed, seed) for seed in range(5)]
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=5, mp_context=ctx) as ex:
        results = list(ex.map(_mp_save_fixed_worker, args))

    assert results.count("ok") == 1  # exactly one winner
    assert results.count("exists") == 4  # the others refused (no overwrite)
    listed = list_runs("scientific", root=str(tmp_path))
    assert [r["run_id"] for r in listed] == [fixed]  # one committed run
    d = tmp_path / "scientific"
    assert (d / f"{fixed}.json").exists()
    assert (d / f"{fixed}.csv").exists()
    assert (d / f"{fixed}.meta.json").exists()
    # winner fully verifiable — no loser corrupted or deleted its files
    assert verify_run("scientific", fixed, root=str(tmp_path)) == {
        "run_id": fixed, "json_ok": True, "csv_ok": True,
    }


def test_recover_orphans_spares_an_active_interprocess_save(tmp_path):
    import concurrent.futures as cf
    import multiprocessing as mp

    from acrpq.dashboard.persist import list_runs, recover_orphans, verify_run

    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
        fut = ex.submit(_mp_save_worker, (str(tmp_path), 3))
        # hammer recovery (default lease) while the other process is saving
        while not fut.done():
            recover_orphans("scientific", root=str(tmp_path))
        rid = fut.result()
    # the active save was never touched: it committed and verifies
    assert rid in {r["run_id"] for r in list_runs("scientific", root=str(tmp_path))}
    assert verify_run("scientific", rid, root=str(tmp_path))["json_ok"]


def test_multi_process_saves_lose_no_entry(tmp_path):
    # The verdict-gating inter-process test: several processes saving at once must
    # each land a distinct, verifiable run — no lost entry (no shared index file).
    import concurrent.futures as cf
    import multiprocessing as mp

    from acrpq.dashboard.persist import list_runs, verify_run

    args = [(str(tmp_path), seed) for seed in range(6)]
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=4, mp_context=ctx) as ex:
        ids = list(ex.map(_mp_save_worker, args))

    assert len(set(ids)) == 6  # distinct ids across processes
    listed = {r["run_id"] for r in list_runs("scientific", root=str(tmp_path))}
    assert listed == set(ids)  # every process's entry survived
    for rid in ids:
        assert verify_run("scientific", rid, root=str(tmp_path))["json_ok"]


def test_scientific_record_round_trips_with_full_provenance(tmp_path):
    from acrpq.dashboard.api import scientific_run
    from acrpq.dashboard.persist import build_scientific_record, read_run, save_run

    out = scientific_run(InstanceLoader(), "CP_4", solver="reference", seeds=[1, 2], n_theta=3)
    record = build_scientific_record(
        out, instance="CP_4", subset=None,
        config={"solver": "reference", "n_theta": 3, "seeds": [1, 2]},
    )
    receipt = save_run(record, root=str(tmp_path))
    assert receipt["citable"] is True

    reloaded = read_run("scientific", receipt["run_id"], root=str(tmp_path))
    assert reloaded["schema"] == "acrpq-run/1"
    assert reloaded["citable"] is True
    assert reloaded["aggregate"]["n_runs"] == 2
    assert len(reloaded["runs"]) == 2
    prov = reloaded["provenance"]
    assert "git_commit" in prov and "source_hash" in prov and "git_dirty" in prov
    assert reloaded["config_hash"].startswith("sha256:")


def test_scientific_save_also_writes_per_seed_csv(tmp_path):
    from pathlib import Path

    from acrpq.dashboard.api import scientific_run
    from acrpq.dashboard.persist import build_scientific_record, save_run

    out = scientific_run(InstanceLoader(), "CP_4", solver="reference", seeds=[7, 8], n_theta=3)
    record = build_scientific_record(
        out, instance="CP_4", subset=None,
        config={"solver": "reference", "n_theta": 3, "seeds": [7, 8]},
    )
    receipt = save_run(record, root=str(tmp_path))
    csv_path = Path(receipt["path"]).with_suffix(".csv")
    assert csv_path.is_file()
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0].startswith("seed,feasible,objective")
    assert len(lines) == 3  # header + two seeds
    assert lines[1].startswith("7,") and lines[2].startswith("8,")


def test_save_run_refuses_non_json_native_or_non_finite_scientific_data(tmp_path):
    from acrpq.dashboard.persist import save_run

    record = _sci_record(seeds=(1,))
    record["aggregate"]["bad"] = object()
    with pytest.raises(TypeError):
        save_run(record, root=str(tmp_path))

    record = _sci_record(seeds=(1,))
    record["aggregate"]["bad"] = float("inf")
    with pytest.raises(ValueError, match="Out of range float"):
        save_run(record, root=str(tmp_path))


def test_interactive_record_is_not_citable_and_kind_is_separate(tmp_path):
    from acrpq.dashboard.persist import build_interactive_record, list_runs, save_run

    solve = solve_payload(InstanceLoader(), "CP_4", solver="reference", n_theta=3)
    record = build_interactive_record(
        solve, instance="CP_4", subset=None, config={"solver": "reference", "n_theta": 3}
    )
    save_run(record, root=str(tmp_path))
    assert record["citable"] is False
    assert record["kind"] == "interactive"
    # kinds live in separate directories / manifests
    assert len(list_runs("interactive", root=str(tmp_path))) == 1
    assert list_runs("scientific", root=str(tmp_path)) == []


def test_build_scientific_record_rejects_demo_output():
    from acrpq.dashboard.persist import build_scientific_record

    demo = {"profile": "demo", "aggregate": {}, "runs": []}
    with pytest.raises(ValueError, match="only SCIENTIFIC"):
        build_scientific_record(demo, instance="CP_4", subset=None, config={})


def test_preflight_reports_five_methods_and_excludes_real_hardware():
    from acrpq.dashboard.api import preflight_payload

    pf = preflight_payload(InstanceLoader(), "CP_4", n_theta=3)
    methods = {m["method"]: m for m in pf["methods"]}
    assert set(methods) == {"reference", "qubo-exact", "qubo-annealing", "dwave", "qaoa"}
    # small instance: exact routes apply, reference is exact, none need hardware
    assert methods["qubo-exact"]["applicable"] is True
    assert methods["reference"]["resolution"] == "exact"
    assert all(not m["real_hardware_required"] for m in pf["methods"])
    excluded = {m["method"] for m in pf["excluded_real_hardware"]}
    assert excluded == {"ibm_real", "dwave-qpu"}


def test_grid_limit_is_instance_specific_and_uses_backend_preflight():
    from acrpq.dashboard.api import grid_limit_payload

    loader = InstanceLoader()
    cp3 = grid_limit_payload(loader, "CP_3")
    assert cp3["max_n_theta_all_methods"] == 9
    assert cp3["max_n_theta_demo"] == 7
    assert cp3["slider_max_n_theta"] == 9
    assert cp3["demo_qaoa_max_qubits"] == 24
    assert grid_limit_payload(loader, "CP_4")["slider_max_n_theta"] == 7
    assert grid_limit_payload(loader, "CP_5")["slider_max_n_theta"] == 5
    cp10 = grid_limit_payload(loader, "CP_10")
    assert cp10["all_methods_at_slider_max"] is False
    assert cp10["slider_max_n_theta"] == 3
    assert "qaoa" in cp10["blocked_at_k3"]


def test_preflight_marks_qubo_exact_not_applicable_past_cap_with_formula():
    from acrpq.dashboard.api import preflight_payload
    from acrpq.quantum.qubo_solvers import QuboExactSolver

    # CP_10 at K=5 enumerates 5^10 = 9,765,625 one-hot states, past the cap.
    pf = preflight_payload(InstanceLoader(), "CP_10", n_theta=5, n_q=1)
    assert pf["search_space"] > QuboExactSolver().max_states
    exact = next(m for m in pf["methods"] if m["method"] == "qubo-exact")
    assert exact["applicable"] is False
    # the reason states the concrete formula and the shared backend cap
    assert f"{pf['grid_k']}^{pf['n_aircraft']}" in exact["reason"]
    assert str(QuboExactSolver().max_states) in exact["reason"].replace(",", "")


def test_instance_catalog_separates_all_methods_from_limited(monkeypatch):
    from acrpq.dashboard import api

    class CatalogLoader:
        @staticmethod
        def list_names():
            return ["CP_4", "CP_10"]

    calls = []

    def fake_preflight(loader, name, *, n_theta, n_q):
        calls.append(name)
        return {
            "n_aircraft": 4,
            "n_qubits": 20,
            "methods": [
                {"method": method, "applicable": True, "reason": None}
                for method in ("reference", "qubo-exact", "qubo-annealing", "dwave", "qaoa")
            ],
        }

    monkeypatch.setattr(api, "preflight_payload", fake_preflight)
    catalog = api.instance_catalog_payload(CatalogLoader(), n_theta=5, n_q=1)

    assert [row["name"] for row in catalog["all_applicable"]] == ["CP_4"]
    assert [row["name"] for row in catalog["limited"]] == ["CP_10"]
    assert catalog["limited"][0]["unavailable"] == ["qubo-exact"]
    assert "5^10" in catalog["limited"][0]["reason"]
    assert catalog["counts"] == {"all_applicable": 1, "limited": 1}
    # CP_10 is proven limited from the exact-state cap; no expensive model build.
    assert calls == ["CP_4"]


def test_reference_solve_includes_explanation_identity():
    # The reference route builds no QUBO to solve, yet the response still carries
    # an energy explanation whose terms sum to the encoded QUBO energy.
    p = solve_payload(InstanceLoader(), "CP_4", solver="reference", n_theta=3)
    energy = p["solution"]["explanation"]["energy"]
    assert energy["identity_verified"] is True
    assert energy["total"] == pytest.approx(energy["encoded_total"])


fastapi_only = pytest.mark.skipif(not have("fastapi"), reason="[dashboard] extra not installed")


@fastapi_only
def test_health_and_instances():
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app

    client = TestClient(create_app())
    h = client.get("/api/health").json()
    assert h["status"] == "ok"
    insts = client.get("/api/instances").json()["instances"]
    assert "CP_4" in insts


@fastapi_only
def test_baseline_endpoint_is_read_only_display(tmp_path):
    import json

    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app

    client = TestClient(create_app(results_dir=str(tmp_path)))
    # No artifact yet -> honest empty marker, never an error or a fake number.
    empty = client.get("/api/baseline").json()
    assert empty == {"available": False, "comparisons": []}

    (tmp_path / "baseline_all.json").write_text(
        json.dumps([{"scenario_id": "Q4", "delta_discretization": {"delta": None}}]),
        encoding="utf-8",
    )
    got = client.get("/api/baseline").json()
    assert got["available"] is True
    assert got["comparisons"][0]["scenario_id"] == "Q4"


# --- /api/ampl-gurobi read-only endpoint ----------------------------------- #
def _build_campaign(tmp_path, *, anchor_protocol=None, tamper_hash=False):
    """Write a minimal valid v3 campaign (CP_4 anchor) + recovery (CP_10) into tmp_path."""
    import hashlib

    from acrpq.classical.ampl_protocol import OFFICIAL_PROTOCOL

    pid = anchor_protocol or OFFICIAL_PROTOCOL.protocol_id
    camp = tmp_path / "ampl_gurobi_campaign_v3"
    rec = tmp_path / "ampl_gurobi_timeout_recovery"
    camp.mkdir(parents=True)
    rec.mkdir(parents=True)
    cp4 = {
        "protocol_id": pid, "instance": "CP_4", "n": 4, "status": "optimal",
        "is_official_anchor": True,
        "anchor_criteria": {"a": True, "b": True},
        "rescored_objective": 6.25e-4, "best_bound": 6.25e-4,
        "optimality_gap": 9.7e-7, "max_model_constraint_residual": 9e-9,
        "wall_time_s": 2.3, "feasible_common_numeric": True,
        "minimum_safety_margin": -2.9e-9,
        "q": [1.0, 1.0, 1.0, 1.0],
        "theta": [0.01, 0.01, 0.01, 0.01],
        "discrete_references": {
            "K3_q1": {"objective": 0.41, "exhaustive_optimality_proven": True,
                      "search_space_size": 81},
            "K5_q1": {"objective": 0.10, "exhaustive_optimality_proven": True,
                      "search_space_size": 625},
            "K7_q1": {"objective": 0.045, "exhaustive_optimality_proven": True,
                      "search_space_size": 2401},
        },
        "discretisation_costs": {
            "discretisation_cost_K3_q1": {"value": 0.4094, "official": True},
            "discretisation_cost_K5_q1": {"value": 0.0994, "official": True},
            "discretisation_cost_K7_q1": {"value": 0.0444, "official": True},
        },
    }
    (camp / "CP_4.json").write_text(json.dumps(cp4), encoding="utf-8")
    h = "sha256:" + hashlib.sha256((camp / "CP_4.json").read_bytes()).hexdigest()
    if tamper_hash:
        h = "sha256:deadbeef"
    (camp / "campaign_manifest.json").write_text(json.dumps({
        "protocol_id": pid, "counts": {"anchor": 1},
        "artifact_hashes": {"CP_4": h}}), encoding="utf-8")
    (camp / "campaign_summary.json").write_text(json.dumps({"rows": [
        {"instance": "CP_4", "n": 4, "status": "optimal", "rescored_objective": 6.25e-4,
         "ampl_version": "AMPL 20260809"},
        {"instance": "CP_10", "n": 10, "status": "timeout", "wall_time_s": 111.0},
    ]}), encoding="utf-8")
    (rec / "recovery_summary.json").write_text(json.dumps({"rows": [
        {"instance": "CP_10", "n": 10, "incumbent_available": True,
         "incumbent_kind": "provisional_feasible_incumbent", "objective": 5.55e-3,
         "best_bound": 2.18e-3, "optimality_gap": 0.607, "feasible_common_numeric": True,
         "minimum_safety_margin": -1e-9, "wall_time_s": 112.0, "is_official_anchor": False,
         "official_exclusion_reason": "timeout: gap 0.607 > 1e-6 not certified"},
    ]}), encoding="utf-8")
    rec_hashes = {}
    # The endpoint accepts recovery figures only when both the summary and the
    # matching per-instance artifact are pinned by the integrity manifest.
    cp10 = {"instance": "CP_10", "protocol_id": "ampl_gurobi_v3_timeout_recovery",
            "q": [1.0] * 10, "theta": [0.02] * 10}
    (rec / "CP_10.json").write_text(json.dumps(cp10), encoding="utf-8")
    for filename in ("CP_10.json", "recovery_summary.json"):
        rec_hashes[filename] = "sha256:" + hashlib.sha256(
            (rec / filename).read_bytes()).hexdigest()
    (rec / "integrity_manifest.json").write_text(json.dumps({
        "schema": "acrpq-recovery-integrity/1",
        "protocol_id": "ampl_gurobi_v3_timeout_recovery",
        "sha256": rec_hashes,
    }), encoding="utf-8")
    return camp, rec


@fastapi_only
def test_ampl_gurobi_absent_is_honest(tmp_path):
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app
    got = TestClient(create_app(results_dir=str(tmp_path))).get("/api/ampl-gurobi").json()
    assert got["available"] is False and got["instances"] == []


@fastapi_only
def test_ampl_gurobi_v3_certified_and_provisional(tmp_path):
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app
    _build_campaign(tmp_path)
    got = TestClient(create_app(results_dir=str(tmp_path))).get("/api/ampl-gurobi").json()
    assert got["available"] is True
    by = {e["instance"]: e for e in got["instances"]}
    assert by["CP_4"]["is_official_anchor"] is True
    assert by["CP_4"]["classification"] == "certified_numerical_anchor"
    assert "1e-6" in by["CP_4"]["anchor_qualification"] and "1e-8" in by["CP_4"]["anchor_qualification"]
    assert [d["label"] for d in by["CP_4"]["discrete"]] == ["Demo grid", "Sensitivity only",
                                                            "Sensitivity only"]
    assert all(d["discretisation_cost_official"] for d in by["CP_4"]["discrete"])
    assert by["CP_4"]["continuous_controls"]["theta"] == [0.01] * 4
    # CP_10 provisional (never official), continuous figure from recovery
    assert by["CP_10"]["is_official_anchor"] is False
    assert by["CP_10"]["classification"] == "provisional_feasible_incumbent"
    assert by["CP_10"]["original_ampl_gurobi_objective"] == 5.55e-3
    assert by["CP_10"]["continuous_controls"]["theta"] == [0.02] * 10
    assert "not certified" in by["CP_10"]["exclusion_reason"] or "timeout" in by["CP_10"]["exclusion_reason"]
    assert all(not d["discretisation_cost_official"] for d in by["CP_10"]["discrete"])


@fastapi_only
def test_ampl_gurobi_refuses_non_official_protocol(tmp_path):
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app
    _build_campaign(tmp_path, anchor_protocol="ampl_gurobi_supervised_v2")
    got = TestClient(create_app(results_dir=str(tmp_path))).get("/api/ampl-gurobi").json()
    assert got["available"] is False and "not the official" in got["error"]


@fastapi_only
def test_ampl_gurobi_hash_mismatch_never_official(tmp_path):
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app
    _build_campaign(tmp_path, tamper_hash=True)
    got = TestClient(create_app(results_dir=str(tmp_path))).get("/api/ampl-gurobi").json()
    by = {e["instance"]: e for e in got["instances"]}
    assert by["CP_4"]["is_official_anchor"] is False           # hash mismatch -> not official
    assert by["CP_4"]["artifact_error"] == "artifact hash mismatch (corrupt)"


@fastapi_only
def test_ampl_gurobi_missing_hash_never_official(tmp_path):
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app
    camp, _ = _build_campaign(tmp_path)
    manifest = json.loads((camp / "campaign_manifest.json").read_text())
    manifest["artifact_hashes"] = {}
    (camp / "campaign_manifest.json").write_text(json.dumps(manifest))
    got = TestClient(create_app(results_dir=str(tmp_path))).get("/api/ampl-gurobi").json()
    by = {e["instance"]: e for e in got["instances"]}
    assert by["CP_4"]["is_official_anchor"] is False
    assert "hash missing" in by["CP_4"]["artifact_error"]


@fastapi_only
def test_ampl_gurobi_ignores_unverified_recovery(tmp_path):
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app
    _, rec = _build_campaign(tmp_path)
    (rec / "integrity_manifest.json").unlink()
    got = TestClient(create_app(results_dir=str(tmp_path))).get("/api/ampl-gurobi").json()
    by = {e["instance"]: e for e in got["instances"]}
    assert by["CP_10"]["classification"] == "no_incumbent"
    assert by["CP_10"]["original_ampl_gurobi_objective"] is None


@fastapi_only
def test_ampl_gurobi_invalidated_protocol_is_refused(tmp_path):
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app
    camp, _ = _build_campaign(tmp_path)
    import hashlib
    h = "sha256:" + hashlib.sha256((camp / "CP_4.json").read_bytes()).hexdigest()
    (camp / "INVALIDATED.json").write_text(json.dumps({
        "valid_for_anchoring": False, "invalidation_reason": "test",
        "affected_artifact_hashes": {"CP_4.json": h}}), encoding="utf-8")
    got = TestClient(create_app(results_dir=str(tmp_path))).get("/api/ampl-gurobi").json()
    by = {e["instance"]: e for e in got["instances"]}
    assert by["CP_4"]["is_official_anchor"] is False  # invalidated -> refused by guard


@fastapi_only
def test_job_solve_round_trip_via_endpoints():
    # The auto-comparison now runs each route as a bounded job: submit -> poll ->
    # result. Prove that path end-to-end through the real HTTP endpoints.
    import time

    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app

    client = TestClient(create_app())
    sub = client.post("/api/jobs", json={"instance": "CP_4", "solver": "reference", "n_theta": 3})
    assert sub.status_code == 200
    job_id = sub.json()["job_id"]

    for _ in range(200):
        snap = client.get(f"/api/jobs/{job_id}").json()
        if snap["state"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.02)
    assert snap["state"] == "completed"
    result = client.get(f"/api/jobs/{job_id}/result").json()
    assert result["solution"]["feasible"] is True


@fastapi_only
def test_job_endpoints_reject_unknown_ids():
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app

    client = TestClient(create_app())
    assert client.get("/api/jobs/deadbeef").status_code == 404
    assert client.get("/api/jobs/deadbeef/result").status_code == 404
    assert client.delete("/api/jobs/deadbeef").status_code == 404


def test_applicability_block_gates_only_capped_methods():
    from acrpq.dashboard.api import applicability_block

    loader = InstanceLoader()

    def req(solver, name, n_theta, mode="aer_sim"):
        return {"solver": solver, "instance": name, "n_theta": n_theta, "n_q": 1,
                "mode": mode, "subset": None, "w": None}

    # qubo-exact past the enumeration cap is blocked with the formula
    block = applicability_block(loader, req("qubo-exact", "CP_10", 5))
    assert block and "5^10" in block
    # applicable exact, and the always-applicable routes, are never blocked
    assert applicability_block(loader, req("qubo-exact", "CP_4", 3)) is None
    assert applicability_block(loader, req("qubo-annealing", "CP_10", 5)) is None
    assert applicability_block(loader, req("reference", "CP_10", 5)) is None
    # real/remote IBM is not bound by the local statevector ceiling
    assert applicability_block(loader, req("qaoa", "CP_10", 5, mode="ibm_real")) is None


@fastapi_only
def test_manual_solve_endpoint_returns_422_for_non_applicable():
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app

    client = TestClient(create_app())
    r = client.post("/api/solve", json={"instance": "CP_10", "solver": "qubo-exact", "n_theta": 5})
    assert r.status_code == 422  # structured, not a generic 400 solver crash
    assert "5^10" in r.json()["detail"]
    # a compatible configuration still solves
    ok = client.post("/api/solve", json={"instance": "CP_4", "solver": "qubo-exact", "n_theta": 3})
    assert ok.status_code == 200 and ok.json()["solution"]["feasible"] is True


@fastapi_only
def test_solve_endpoint_reference():
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app

    client = TestClient(create_app())
    r = client.post("/api/solve", json={"instance": "CP_4", "solver": "reference", "n_theta": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["solution"]["feasible"] is True


@fastapi_only
def test_scientific_save_flag_must_be_strict_boolean(tmp_path):
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app

    client = TestClient(create_app(runs_dir=str(tmp_path)))
    base = {"instance": "CP_4", "solver": "reference", "n_theta": 3, "seeds": [1, 2]}

    # non-boolean save values are rejected, never coerced (e.g. "false" -> True)
    for bad in ["false", "true", 1, 0, [], {}]:
        r = client.post("/api/scientific", json={**base, "save": bad})
        assert r.status_code == 422, f"save={bad!r} should be 422"

    # absent -> not saved; explicit false -> not saved; explicit true -> saved
    assert "saved" not in client.post("/api/scientific", json=base).json()
    assert "saved" not in client.post("/api/scientific", json={**base, "save": False}).json()
    assert "saved" in client.post("/api/scientific", json={**base, "save": True}).json()


@fastapi_only
def test_scientific_endpoint_guards_non_applicable_before_seeds(tmp_path):
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app

    client = TestClient(create_app(runs_dir=str(tmp_path)))
    # qubo-exact on CP_10 K=5 is non-applicable: reject once, before any seed runs
    r = client.post(
        "/api/scientific",
        json={"instance": "CP_10", "solver": "qubo-exact", "n_theta": 5, "seeds": [1, 2, 3]},
    )
    assert r.status_code == 422
    assert "5^10" in r.json()["detail"]  # rejected with the formula, not mid-sweep


@fastapi_only
def test_scientific_endpoint_saves_and_relists(tmp_path):
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app

    client = TestClient(create_app(runs_dir=str(tmp_path)))
    r = client.post(
        "/api/scientific",
        json={
            "instance": "CP_4", "solver": "reference", "n_theta": 3,
            "seeds": [1, 2, 3], "save": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["aggregate"]["n_runs"] == 3
    run_id = body["saved"]["run_id"]

    # the saved run is indexed and re-readable (round-trips through disk)
    listed = client.get("/api/runs/scientific").json()["runs"]
    assert any(row["run_id"] == run_id and row["citable"] for row in listed)
    record = client.get(f"/api/runs/scientific/{run_id}").json()
    assert record["schema"] == "acrpq-run/1" and record["citable"] is True


@fastapi_only
def test_solve_endpoint_rejects_unbounded_or_malformed_work():
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app

    client = TestClient(create_app())
    too_large = client.post(
        "/api/solve", json={"instance": "CP_4", "n_theta": 25, "n_q": 10}
    )
    assert too_large.status_code == 422
    unknown = client.post("/api/solve", json={"instance": "CP_4", "typo": 1})
    assert unknown.status_code == 422
    # Large encodings remain inspectable for preflight/explainability even when
    # a local statevector solve would be refused.
    large_qubo = client.get("/api/qubo/CP_10?n_theta=5")
    assert large_qubo.status_code == 200
    assert large_qubo.json()["resources"]["raw_fits_statevector"] is False


@fastapi_only
def test_qubo_endpoint_and_solvers():
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app

    client = TestClient(create_app())
    q = client.get("/api/qubo/CP_3?n_theta=3").json()
    assert q["n_qubits"] == 9
    assert q["matrix_included"] is True
    assert len(q["matrix"]) == 9 and len(q["labels"]) == 9
    for solver in ("qubo-exact", "qubo-annealing"):
        r = client.post("/api/solve", json={"instance": "CP_4", "solver": solver, "n_theta": 5})
        assert r.status_code == 200
        s = r.json()["solution"]
        assert s["feasible"] is True
        assert "qubo_energy" in s
        assert s["explanation"]["energy"]["identity_verified"] is True
        assert s["explanation"]["encoding"]["raw_onehot_valid"] is True


def test_qubo_payload_pure():
    """The QUBO payload builder works without a server."""
    from acrpq.dashboard.api import qubo_payload
    from acrpq.io.loader import InstanceLoader

    q = qubo_payload(InstanceLoader(), "CP_3", n_theta=3)
    assert q["n_qubits"] == 9
    assert q["lambda_conflict"] > 0
    assert q["matrix_included"] is True
    assert q["resources"]["raw_qubits"] == 9
    assert len(q["coefficient_explanations"]["linear"]) == 9


@fastapi_only
def test_solve_quantum_unavailable_is_503_when_no_qiskit():
    if have("qiskit") and have("qiskit_optimization"):
        pytest.skip("qiskit installed; the 503 path only triggers without it")
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app

    client = TestClient(create_app())
    r = client.post("/api/solve", json={"instance": "CP_4", "solver": "qaoa"})
    assert r.status_code == 503


# --------------------------------------------------------------------------- #
# Business-language (French) preflight layer + reassignment metrics
# --------------------------------------------------------------------------- #
def test_preflight_business_layer_exact_french_labels():
    from acrpq.dashboard.api import preflight_payload

    pf = preflight_payload(InstanceLoader(), "CP_3", n_theta=3)
    by = {m["method"]: m for m in pf["methods"]}
    # EXACT honest phrasings — a simulation is never called "quantique réel"
    assert by["reference"]["label_fr"] == "référence classique"
    assert by["qubo-exact"]["label_fr"] == "recherche exacte"
    assert by["qubo-annealing"]["label_fr"] == "heuristique classique (recuit simulé)"
    assert by["dwave"]["label_fr"] == "heuristique classique (recuit simulé — Ocean, CPU local)"
    assert by["qaoa"]["label_fr"] == "QAOA simulé localement (Aer)"
    assert "quantique réel" not in json.dumps(pf, ensure_ascii=False)
    for hw in pf["excluded_real_hardware"]:
        assert hw["label_fr"] == "circuit préparé pour hardware, non soumis"
    # applicable methods carry no business excuse
    assert by["qubo-exact"]["applicable"] is True
    assert by["qubo-exact"]["business_reason_fr"] is None
    # objective + K explanations exist in plain business language
    assert "minimiser la déviation totale" in pf["objective_fr"]
    assert "ne rien faire" in pf["objective_fr"]
    assert pf["k_explanation_fr"].startswith("K = nombre d'options candidates par avion")
    assert "3·3 = 9 = qubits" in pf["k_explanation_fr"]
    assert "K identique" in pf["k_explanation_fr"]


def test_preflight_business_reason_for_oversized_exact_search():
    from acrpq.dashboard.api import preflight_payload

    # CP_10 at K=5: 5^10 = 9,765,625 one-hot states exceeds the exact cap.
    pf = preflight_payload(InstanceLoader(), "CP_10", n_theta=5)
    exact = next(m for m in pf["methods"] if m["method"] == "qubo-exact")
    assert exact["applicable"] is False
    business = exact["business_reason_fr"]
    assert business.startswith("Recherche exacte non applicable")
    assert "5^10" in business and "millions" in business  # ≈ 9,8 millions d'états
    assert "~9 avions à K=5" in business  # 5^9 still fits under the cap
    # the technical string is preserved verbatim for the collapsible detail
    assert "enumeration cap" in exact["reason"]


def test_preflight_business_reason_for_qaoa_over_ram_ceiling():
    from acrpq._optional import have as _have
    from acrpq.dashboard.api import preflight_payload

    if not (_have("qiskit") and _have("qiskit_optimization")):
        pytest.skip("[quantum] extra not installed; RAM-ceiling reason unreachable")
    # CP_10 at K=3 is one connected 30-qubit component: no local statevector.
    pf = preflight_payload(InstanceLoader(), "CP_10", n_theta=3)
    qaoa = next(m for m in pf["methods"] if m["method"] == "qaoa")
    assert qaoa["applicable"] is False
    assert qaoa["business_reason_fr"].startswith("QAOA simulé non applicable")
    assert "qubits" in qaoa["business_reason_fr"]
    assert "réduire K" in qaoa["business_reason_fr"]


def test_solution_reports_deviated_aircraft_reassignments():
    p = solve_payload(InstanceLoader(), "CP_4", solver="reference", n_theta=3)
    s = p["solution"]
    assert s["n_maneuvered"] + s["n_unchanged"] == p["n"]
    flags = [a["deviated"] for a in p["aircraft"]]
    assert sum(flags) == s["n_maneuvered"]
    # deviated == chosen option is not the exact NOOP
    for a in p["aircraft"]:
        assert a["deviated"] == (a["option_label"] != "NOOP")
    # CP_4 at K=3: the certified in-grid optimum deviates n-1 = 3 aircraft
    assert s["n_maneuvered"] == 3


def test_objective_degeneracy_is_computed_per_grid_never_assumed():
    """Condition 2 of the MoE verdict: at K=3/n_q=1 the two objectives are proportional
    (one single non-NOOP amplitude), so they can rank solutions identically. This must be
    COMPUTED per grid and must NOT be claimed at K=5/K=7/n_q>1, where several amplitudes
    coexist and the objectives genuinely differ."""
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app

    tc = TestClient(create_app())

    def deg(n_theta, n_q):
        r = tc.post("/api/preflight", json={"instance": "CP_3", "n_theta": n_theta, "n_q": n_q})
        assert r.status_code == 200, r.text
        return r.json()["objective_degeneracy"]

    d3 = deg(3, 1)
    assert d3["proportional"] is True
    assert d3["distinct_non_noop_amplitudes"] == 1
    assert d3["factor"] == pytest.approx(0.137077781868, rel=1e-9)
    assert "proportionnels" in d3["note_fr"]

    for n_theta, n_q in ((5, 1), (7, 1), (3, 3)):
        d = deg(n_theta, n_q)
        assert d["proportional"] is False, f"K{n_theta}/q{n_q} must NOT claim proportionality"
        assert d["distinct_non_noop_amplitudes"] > 1
        assert d["factor"] is None
        assert "PAS proportionnels" in d["note_fr"]
