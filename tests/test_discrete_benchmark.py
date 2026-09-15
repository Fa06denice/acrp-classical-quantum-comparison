"""Phase 3B/3.1 — adversarial tests for the bounded discrete benchmark.

Pins the scientific invariants the mission forbids breaking (objectives never
conflated, incumbent never an optimum, raw one-hot vs decoded feasibility kept
separate, infeasible never scored, tolerance never widened, no silent caps) and
the Phase-3.1 resume/idempotency guarantees (canonical config_hash; fail-closed
job validation; aggregation membership/dup/missing/baseline-consistency).
"""

from __future__ import annotations

import pytest

from acrpq.benchmark.discrete_benchmark import (
    DEFAULT_SEEDS,
    HEURISTIC_METHODS,
    OPTIMUM_METHODS,
    BenchmarkJob,
    JobValidationError,
    LegConfig,
    Plan,
    _decode_sample,
    _finish,
    _matches,
    abs_tol_for,
    assert_json_csv_parity,
    build_baselines,
    build_protocol,
    canonical_benchmark_sha256,
    exhaustive_baseline,
    finalize_job_record,
    method_applicability,
    plan_jobs,
    probe_gurobi,
    result_sha256,
    rows_to_csv,
    run_job,
    summarize,
    validate_aggregate,
    validate_job_record,
)
from acrpq.io.loader import InstanceLoader
from acrpq.quantum.qubo import build_qubo

MANEUVER = "maneuver_count_v1"
QUADRATIC = "quadratic_control_cost_v1"


@pytest.fixture(scope="module")
def ld():
    return InstanceLoader()


def _baselines(ld, instance, k, obj):
    return build_baselines(ld, instance, k, 1, obj)


def _run(ld, method, objective_id, *, instance="CP_3", k=3, seed=None, cfg=None):
    base = _baselines(ld, instance, k, objective_id)
    job = BenchmarkJob(instance, k, 1, objective_id, method, seed)
    return run_job(ld, job, baselines=base, cfg=cfg or LegConfig()), base


# --- 1. objectives are never conflated -------------------------------------
def test_objective_axis_is_never_conflated(ld):
    pytest.importorskip("amplpy")
    ex_man = exhaustive_baseline(ld, "CP_3", 3, 1, MANEUVER)["optimum"]
    ex_quad = exhaustive_baseline(ld, "CP_3", 3, 1, QUADRATIC)["optimum"]
    assert ex_man != ex_quad
    assert float(ex_man).is_integer()

    r_man, _ = _run(ld, "qubo_exact", MANEUVER)
    r_quad, _ = _run(ld, "qubo_exact", QUADRATIC)
    assert r_man["objective_value"] == pytest.approx(ex_man)
    assert r_quad["objective_value"] == pytest.approx(ex_quad)
    assert r_man["objective_value"] != r_quad["objective_value"]


# --- 2. an incumbent is never labelled an optimum --------------------------
def test_incumbent_never_labelled_optimum(ld):
    pytest.importorskip("amplpy")
    for objective_id in (MANEUVER, QUADRATIC):
        r, _ = _run(ld, "qubo_annealing", objective_id, seed=1234)
        assert r["result_class"] == "incumbent"


def test_optimum_legs_agree_and_carry_optimum_class(ld):
    pytest.importorskip("amplpy")
    for objective_id in (MANEUVER, QUADRATIC):
        vals = {}
        for method in OPTIMUM_METHODS:
            r, _ = _run(ld, method, objective_id)
            assert r["matches_exhaustive_optimum"] is True
            assert r["result_class"] in ("certified_optimum", "qubo_optimum",
                                         "one_hot_optimum")
            vals[method] = r["objective_value"]
        assert vals["reference"] == pytest.approx(vals["qubo_exact"])
        assert vals["reference"] == pytest.approx(vals["gurobi_milp"])


# --- 3. raw one-hot validity is kept separate from decoded feasibility ------
def test_raw_onehot_and_decoded_feasible_are_independent_axes(ld):
    qubo = build_qubo(ld.load("CP_3"), n_theta=3, n_q=1, objective_id=MANEUVER)
    d = qubo.discrete
    noop = d.grid.noop_index()
    base = {"exhaustive": {"optimum": 2.0, "status": "optimal"}, "certified_milp": {}}

    bits_allnoop = {d.var_index(i, noop): 1 for i in d.instance.aircraft()}
    rec_a = _decode_sample({"objective_id": MANEUVER}, base, qubo, bits_allnoop,
                           0.0, 0.0, provider_s=None, qubo_energy=0.0)
    assert rec_a["raw_onehot_valid"] is True
    assert rec_a["repair_applied"] is False
    assert rec_a["decoded_feasible"] is False           # conflicts remain
    assert rec_a["matches_exhaustive_optimum"] is None  # infeasible -> not scored
    assert rec_a["gap_to_exhaustive_optimum"] is None

    first = next(iter(d.instance.aircraft()))
    bits_bad = dict(bits_allnoop)
    bits_bad[d.var_index(first, (noop + 1) % d.k)] = 1  # second active option
    rec_b = _decode_sample({"objective_id": MANEUVER}, base, qubo, bits_bad,
                           0.0, 0.0, provider_s=None, qubo_energy=0.0)
    assert rec_b["raw_onehot_valid"] is False
    assert rec_b["repair_applied"] is True
    assert isinstance(rec_b["decoded_feasible"], bool)


# --- 4. infeasible never scored; tolerance never widened -------------------
def test_finish_never_scores_infeasible_and_respects_tolerance():
    base = {"exhaustive": {"optimum": 3.0, "status": "optimal"}, "certified_milp": {}}
    infeasible = _finish({"objective_id": MANEUVER}, status="incumbent",
                         result_class="incumbent", objective_value=3.0, feasible=False,
                         wall_s=0.0, cpu_s=0.0, provider_s=None, baselines=base)
    assert infeasible["matches_exhaustive_optimum"] is None
    assert infeasible["gap_to_exhaustive_optimum"] is None

    off = _finish({"objective_id": MANEUVER}, status="incumbent",
                  result_class="incumbent", objective_value=3.0 + 1e-6, feasible=True,
                  wall_s=0.0, cpu_s=0.0, provider_s=None, baselines=base)
    assert off["matches_exhaustive_optimum"] is False

    exact = _finish({"objective_id": MANEUVER}, status="incumbent",
                    result_class="incumbent", objective_value=3.0, feasible=True,
                    wall_s=0.0, cpu_s=0.0, provider_s=None, baselines=base)
    assert exact["matches_exhaustive_optimum"] is True


def test_abs_tol_policy_is_objective_specific():
    assert abs_tol_for(MANEUVER) == 0.0
    assert abs_tol_for(QUADRATIC) == 1e-9
    assert _matches(1.0 + 1e-12, 1.0, abs_tol_for(QUADRATIC)) is True
    assert _matches(1.0 + 1e-6, 1.0, abs_tol_for(MANEUVER)) is False
    assert _matches(None, 1.0, 0.0) is None


# --- 5. dual baselines: absent exhaustive => gap only vs certified MILP -----
def test_absent_exhaustive_uses_certified_milp_never_calls_incumbent_optimum(ld):
    pytest.importorskip("amplpy")
    # GP_4 (n=16, K=3): 3**16 exceeds both the exhaustive and exact-QUBO caps, so
    # there is NO exhaustive ground truth — only the certified Gurobi MILP.
    base = _baselines(ld, "GP_4", 3, MANEUVER)
    assert base["exhaustive"]["optimum"] is None            # no exhaustive
    assert base["certified_milp"]["optimum"] is not None     # certified MILP exists

    milp, _ = _run(ld, "gurobi_milp", MANEUVER, instance="GP_4")
    assert milp["result_class"] == "certified_optimum"
    assert milp["gap_to_exhaustive_optimum"] is None         # no exhaustive to gap against
    assert milp["gap_to_certified_milp"] == 0.0

    heur, _ = _run(ld, "qubo_annealing", MANEUVER, instance="GP_4", seed=1234)
    assert heur["result_class"] == "incumbent"               # never "optimum"
    assert heur["matches_exhaustive_optimum"] is None        # no exhaustive baseline
    if heur["feasible"]:
        assert heur["gap_to_certified_milp"] is not None     # scored vs certified MILP only


# --- 6. honest applicability -----------------------------------------------
def test_applicability_reports_separated_honest_fields(ld):
    pytest.importorskip("amplpy")
    cfg = LegConfig()
    gp = probe_gurobi()
    # Gurobi probe: driver+license actually verified, version sanitized (no PII).
    assert gp["solver_verified"] is True
    assert gp["version"] is None or ("@" not in gp["version"]
                                     and "Licensed" not in gp["version"])
    a = method_applicability(ld, "FP_4", 3, 1, cfg, gp)
    for info in a.values():
        assert set(info) >= {"dependency_available", "computationally_applicable",
                             "resolution", "reason", "authoritative_cap_source"}
        assert info["resolution"] in ("certified_exact", "heuristic", "unavailable")
    # FP_4 keeps an exhaustive reference; GP_4/RCP_50_1 have only Gurobi + heuristics.
    assert a["reference"]["computationally_applicable"] is True
    for big in ("GP_4", "RCP_50_1"):
        ab = method_applicability(ld, big, 3, 1, cfg, gp)
        assert ab["reference"]["computationally_applicable"] is False
        assert ab["qubo_exact"]["computationally_applicable"] is False
        assert ab["gurobi_milp"]["computationally_applicable"] is True
        assert ab["qubo_annealing"]["computationally_applicable"] is True


# --- 7. planning: seeds, applicability, logged skips -----------------------
def test_plan_jobs_seeds_and_logged_skips(ld):
    pytest.importorskip("amplpy")
    plan = plan_jobs(ld, instances=("CP_3",))
    for method in HEURISTIC_METHODS:
        seeds = {j.seed for j in plan.jobs if j.method == method}
        assert seeds == set(DEFAULT_SEEDS)
    for method in OPTIMUM_METHODS:
        assert all(j.seed is None for j in plan.jobs if j.method == method)
    budget_skips = [s for s in plan.skipped
                    if s["method"] == "qaoa_aer" and "qaoa_max_qubits" in s["reason"]]
    assert budget_skips
    assert all(s.get("authoritative_cap_source") for s in plan.skipped)


# --- 8. determinism of deterministic legs ----------------------------------
def test_deterministic_legs_are_reproducible(ld):
    pytest.importorskip("amplpy")
    for method in ("reference", "qubo_exact"):
        a, _ = _run(ld, method, QUADRATIC)
        b, _ = _run(ld, method, QUADRATIC)
        assert a["objective_value"] == b["objective_value"]
    a, _ = _run(ld, "qubo_annealing", QUADRATIC, seed=7)
    b, _ = _run(ld, "qubo_annealing", QUADRATIC, seed=7)
    assert a["objective_value"] == b["objective_value"]


def test_qubo_legs_build_beyond_statevector_ceiling(ld):
    pytest.importorskip("amplpy")
    r, base = _run(ld, "qubo_exact", QUADRATIC, instance="CP_5", k=7)
    assert r["status"] == "optimal"
    assert r["objective_value"] == pytest.approx(base["exhaustive"]["optimum"])
    assert r["matches_exhaustive_optimum"] is True


# --- 9. CSV full-column parity + tamper + hash -----------------------------
def _synthetic_rows():
    return [
        {"job_key": "a", "instance": "CP_3", "method": "reference",
         "objective_value": 2.0, "result_class": "certified_optimum",
         "matches_exhaustive_optimum": True, "gap_to_certified_milp": 0.0},
        {"job_key": "b", "instance": "CP_3", "method": "qaoa_aer",
         "objective_value": 3.5, "result_class": "incumbent",
         "matches_exhaustive_optimum": None, "gap_to_certified_milp": None},
    ]


def test_csv_parity_full_column_handles_none_and_detects_tampering():
    rows = _synthetic_rows()
    csv_text = rows_to_csv(rows)
    assert_json_csv_parity(rows, csv_text)               # None cells fine
    rows[0]["objective_value"] = 99.0
    with pytest.raises(ValueError, match="parity mismatch"):
        assert_json_csv_parity(rows, csv_text)


def test_canonical_hash_excludes_timing_and_repo_dirty():
    bench = {"schema": "x", "results": [
        {"method": "reference", "objective_value": 2.0,
         "wall_time_s": 0.1, "cpu_time_s": 0.1, "provider_time_s": 0.05}]}
    h1 = canonical_benchmark_sha256(bench)
    bench["results"][0]["wall_time_s"] = 999.0
    assert canonical_benchmark_sha256(bench) == h1
    bench["provenance"] = {"repository_dirty": False, "scientific_code_dirty": False}
    h2 = canonical_benchmark_sha256(bench)
    bench["provenance"]["repository_dirty"] = True
    assert canonical_benchmark_sha256(bench) == h2       # repo_dirty excluded
    bench["provenance"]["scientific_code_dirty"] = True
    assert canonical_benchmark_sha256(bench) != h2       # scientific flag hashed
    # synthesis timing TOTALS are volatile too — must not enter the hash
    bench["synthesis"] = {"by_family_objective_method": [
        {"family": "FP", "wall_time_s_total": 1.0, "cpu_time_s_total": 1.0,
         "provider_time_s_total": 1.0, "n_feasible": 2}]}
    h3 = canonical_benchmark_sha256(bench)
    bench["synthesis"]["by_family_objective_method"][0]["wall_time_s_total"] = 999.0
    bench["synthesis"]["by_family_objective_method"][0]["cpu_time_s_total"] = 999.0
    bench["synthesis"]["by_family_objective_method"][0]["provider_time_s_total"] = 999.0
    assert canonical_benchmark_sha256(bench) == h3       # timing totals excluded
    bench["synthesis"]["by_family_objective_method"][0]["n_feasible"] = 5
    assert canonical_benchmark_sha256(bench) != h3       # science IS hashed
    bench["results"][0]["objective_value"] = 5.0
    assert canonical_benchmark_sha256(bench) != h3


def test_job_key_is_stable_and_unique():
    j1 = BenchmarkJob("CP_3", 3, 1, MANEUVER, "qaoa_aer", 2)
    j2 = BenchmarkJob("CP_3", 3, 1, MANEUVER, "qaoa_aer", 2)
    j3 = BenchmarkJob("CP_3", 3, 1, MANEUVER, "qaoa_aer", 3)
    j4 = BenchmarkJob("CP_3", 3, 1, MANEUVER, "reference", None)
    assert j1.key() == j2.key()
    assert len({j1.key(), j3.key(), j4.key()}) == 3
    assert j4.key().endswith("det")


# --- 10. Phase 3.1: canonical config_hash sensitivity ----------------------
def _proto(**overrides):
    base = dict(instances=("CP_3",), objectives=(MANEUVER,),
                methods=("reference",), seeds=(1, 2, 3), leg_config=LegConfig(),
                harness_sha256="deadbeef")
    base.update(overrides)
    return build_protocol(**base)


def test_config_hash_covers_every_protocol_axis():
    ref = _proto().config_hash
    assert _proto(leg_config=LegConfig(qaoa_maxiter=999)).config_hash != ref
    assert _proto(leg_config=LegConfig(qaoa_shots=1)).config_hash != ref
    assert _proto(leg_config=LegConfig(dwave_num_reads=1)).config_hash != ref
    assert _proto(leg_config=LegConfig(dwave_num_sweeps=1)).config_hash != ref
    assert _proto(leg_config=LegConfig(exact_max_states=1)).config_hash != ref
    assert _proto(leg_config=LegConfig(qaoa_max_qubits=1)).config_hash != ref
    assert _proto(seeds=(1, 2)).config_hash != ref
    assert _proto(instances=("CP_4",)).config_hash != ref
    assert _proto(instances=("CP_4", "CP_3")).config_hash != _proto(
        instances=("CP_3", "CP_4")).config_hash          # ORDER matters
    assert _proto(objectives=(QUADRATIC,)).config_hash != ref
    assert _proto(harness_sha256="other").config_hash != ref
    assert _proto().config_hash == ref                   # stable


# --- 11. Phase 3.1: fail-closed job validation -----------------------------
def _finalized_record(job, protocol, ish="ish", gsh="gsh"):
    rec = {
        "schema": "acrpq-discrete-benchmark-job/2", "job_key": job.key(),
        "instance": job.instance, "grid_k": job.grid_k, "n_q": job.n_q,
        "objective_id": job.objective_id, "method": job.method, "seed": job.seed,
        "objective_value": 3.0, "feasible": True, "wall_time_s": 0.1,
    }
    return finalize_job_record(rec, protocol=protocol, instance_sha=ish,
                              grid_identity_sha=gsh, source_commit="abc")


def test_validate_job_record_clean_roundtrip_and_refusals():
    job = BenchmarkJob("CP_3", 3, 1, MANEUVER, "qubo_annealing", 1)
    proto = _proto(methods=("qubo_annealing",))
    rec = _finalized_record(job, proto)
    # clean roundtrip
    validate_job_record(rec, expected_job=job, protocol=proto,
                        instance_sha="ish", grid_identity_sha="gsh")

    # tampered scientific content -> result_sha256 mismatch
    bad = dict(rec)
    bad["objective_value"] = 99.0
    with pytest.raises(JobValidationError, match="result_sha256"):
        validate_job_record(bad, expected_job=job, protocol=proto,
                            instance_sha="ish", grid_identity_sha="gsh")

    # different protocol (e.g. changed qaoa_maxiter) -> config_hash mismatch
    other = _proto(methods=("qubo_annealing",), leg_config=LegConfig(qaoa_maxiter=7))
    with pytest.raises(JobValidationError, match="config_hash"):
        validate_job_record(rec, expected_job=job, protocol=other,
                            instance_sha="ish", grid_identity_sha="gsh")

    # old job from another instance -> instance_sha mismatch
    with pytest.raises(JobValidationError, match="instance_sha256"):
        validate_job_record(rec, expected_job=job, protocol=proto,
                            instance_sha="DIFFERENT", grid_identity_sha="gsh")

    # wrong grid identity -> mismatch
    with pytest.raises(JobValidationError, match="grid_identity_sha256"):
        validate_job_record(rec, expected_job=job, protocol=proto,
                            instance_sha="ish", grid_identity_sha="DIFFERENT")

    # wrong objective/grid expectation -> job_key/objective mismatch
    wrong_job = BenchmarkJob("CP_3", 3, 1, QUADRATIC, "qubo_annealing", 1)
    with pytest.raises(JobValidationError):
        validate_job_record(rec, expected_job=wrong_job, protocol=proto,
                            instance_sha="ish", grid_identity_sha="gsh")


# --- 12. Phase 3.1: aggregation validation ---------------------------------
def _row(job, **extra):
    r = {"job_key": job.key(), "instance": job.instance, "grid_k": job.grid_k,
         "n_q": job.n_q, "objective_id": job.objective_id, "method": job.method,
         "seed": job.seed, "exhaustive_optimum": 2.0, "exhaustive_status": "optimal",
         "certified_milp_optimum": 2.0, "milp_certified": True}
    r.update(extra)
    return r


def test_validate_aggregate_membership_dup_missing_and_baseline_consistency():
    j1 = BenchmarkJob("CP_3", 3, 1, MANEUVER, "reference", None)
    j2 = BenchmarkJob("CP_3", 3, 1, MANEUVER, "qubo_exact", None)
    plan = Plan(jobs=[j1, j2])

    validate_aggregate([_row(j1), _row(j2)], plan)              # clean

    with pytest.raises(ValueError, match="missing planned jobs"):
        validate_aggregate([_row(j1)], plan)

    extra_job = BenchmarkJob("CP_9", 3, 1, MANEUVER, "reference", None)
    with pytest.raises(ValueError, match="unexpected job files"):
        validate_aggregate([_row(j1), _row(j2), _row(extra_job)], plan)

    with pytest.raises(ValueError, match="duplicate job records"):
        validate_aggregate([_row(j1), _row(j1), _row(j2)], plan)

    # inconsistent certified baseline within the same (instance,grid,objective)
    bad = _row(j2, exhaustive_optimum=999.0)
    with pytest.raises(ValueError, match="inconsistent certified baselines"):
        validate_aggregate([_row(j1), bad], plan)


# --- 13. Phase 3.1: result_sha256 is deterministic over non-volatile content
def test_result_sha256_ignores_timing():
    rec = {"job_key": "k", "objective_value": 2.0, "wall_time_s": 1.0,
           "cpu_time_s": 2.0, "provider_time_s": 3.0}
    h = result_sha256(rec)
    rec2 = dict(rec, wall_time_s=99.0, cpu_time_s=88.0, provider_time_s=77.0)
    assert result_sha256(rec2) == h
    rec3 = dict(rec, objective_value=5.0)
    assert result_sha256(rec3) != h


# --- 14. Phase 3.1: clean idempotent resume + config-change refusal (integration)
def test_resume_is_idempotent_and_refuses_incompatible(tmp_path):
    pytest.importorskip("amplpy")
    import json
    import subprocess
    import sys

    out = tmp_path / "bench"
    common = [sys.executable, "scripts/run_discrete_benchmark.py", "--preset",
              "multifamily", "--instances", "FP_4", "--out", str(out),
              "--qaoa-maxiter", "20", "--dwave-reads", "30"]
    r1 = subprocess.run(common, capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    sha1 = json.loads((out / "benchmark.json").read_text())["benchmark_sha256"]

    # clean idempotent resume: nothing re-runs, identical hash
    r2 = subprocess.run(common, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    assert "0 ran" in r2.stdout
    sha2 = json.loads((out / "benchmark.json").read_text())["benchmark_sha256"]
    assert sha1 == sha2

    # a changed protocol (different qaoa_maxiter -> different config_hash) must be
    # REFUSED, never silently reused
    changed = [x if x != "20" else "40" for x in common]
    r3 = subprocess.run(changed, capture_output=True, text=True)
    assert r3.returncode == 4, (r3.returncode, r3.stdout, r3.stderr)
    assert "REFUSING to resume" in r3.stdout

    # Seeds are a first-class protocol axis: changing the repetition set cannot
    # reuse rows produced under the previous stochastic design.
    changed_seeds = [*common, "--seeds", "11", "12"]
    r_seed = subprocess.run(changed_seeds, capture_output=True, text=True)
    assert r_seed.returncode == 4, (r_seed.returncode, r_seed.stdout, r_seed.stderr)
    assert "REFUSING to resume" in r_seed.stdout

    # a rogue job file for an instance outside the current plan must be refused
    # CLEANLY at aggregation (dedicated exit code, not an uncaught traceback)
    rogue = out / "jobs" / "CP_3__k3q1__maneuver_count_v1__reference__det.json"
    rogue.write_text(json.dumps({"job_key": rogue.stem, "instance": "CP_3"}))
    r4 = subprocess.run(common, capture_output=True, text=True)
    assert r4.returncode == 6, (r4.returncode, r4.stdout, r4.stderr)
    assert "REFUSING to aggregate" in r4.stdout


# --- 15. Phase 3.1: synthesis is honest about gaps + caveats ---------------
def test_summarize_gaps_only_on_feasible_and_caveats_present(ld):
    j = BenchmarkJob("GP_4", 3, 1, MANEUVER, "qubo_annealing", 1)
    rows = [
        _row(j, family="GP", feasible=True, status="incumbent",
             gap_to_exhaustive_optimum=None, gap_to_certified_milp=1.0,
             matches_exhaustive_optimum=None),
        _row(j, family="GP", feasible=False, status="incumbent",
             gap_to_exhaustive_optimum=None, gap_to_certified_milp=None,
             matches_exhaustive_optimum=None),
    ]
    s = summarize(rows)
    row = s["by_family_objective_method"][0]
    assert row["n_feasible"] == 1 and row["n_infeasible"] == 1
    assert row["n_gap_to_certified_milp"] == 1          # only the feasible one
    assert row["gap_to_certified_milp_mean"] == 1.0
    assert row["gap_to_exhaustive_mean"] is None        # GP_4 has no exhaustive
    assert any("IID" in c for c in s["caveats"])
    assert any("seed" in c.lower() for c in s["caveats"])


def test_uncertified_gurobi_timeout_is_an_incumbent_never_a_certified_optimum(monkeypatch):
    """MoE Expert H, surviving mutation 6a: `result_class = "certified_optimum" if
    res.certified else "incumbent"` could be collapsed to the constant
    "certified_optimum" without a single test turning red — no test ever drove Gurobi
    terminating on a time limit with a feasible-but-unproven incumbent (exactly the
    CP_8..CP_20 regime the dashboard labels 'provisional')."""
    from acrpq.benchmark import discrete_benchmark as db
    from acrpq.classical.discrete_milp import DiscreteMILPResult

    res = DiscreteMILPResult(
        objective_id=MANEUVER, status="timeout", feasible=True, optimum=3.0,
        choice=(0, 0, 0), q=(1.0, 1.0, 1.0), theta=(0.0, 0.0, 0.0),
        n_conflicts=0, grid_k=3, solver="gurobi",
        certified=False,                      # Gurobi could NOT prove optimality
        solve_result="limit", solve_result_num=400, best_bound=1.0, abs_gap=2.0,
        rel_gap=0.66, solve_time_s=1.0, solver_version="stub", solver_options={},
        message="time limit", secondary={},
    )
    monkeypatch.setattr(db, "_timed", lambda fn: (res, 0.1, 0.1))
    job = db.BenchmarkJob(instance="CP_3", method="gurobi_milp",
                          objective_id=MANEUVER, seed=None, grid_k=3, n_q=1)
    out = db._run_milp(InstanceLoader(), job, {"objective_id": MANEUVER},
                       {"exhaustive": {}, "certified_milp": {}})
    assert out["result_class"] == "incumbent"
    assert out["result_class"] != "certified_optimum"
    assert out.get("certified") is False
