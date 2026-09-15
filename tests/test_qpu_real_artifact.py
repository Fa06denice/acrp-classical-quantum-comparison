"""Validate the QPU result adapter against REAL recorded IBM hardware artifacts.

These tests load — strictly read-only — the counts that were recorded from the
real ``ibm_marrakesh`` fixed-angle campaign (``results/ibm_fixed_angle_campaign_v1``
+ ``runs/qpu/<run_id>/artefacts/result_raw.json``) and run them through the SAME
adapter/decoder stack a live retrieval would use:

* ``qpu_store.load_result_raw``           (hash-verified read of the stored doc)
* ``qpu_result.normalize_ibm_sampler_result``  (endianness / probability model)
* ``qpu_result.decode_scientific``        (feasibility / one-hot / reference gap)
* ``qpu_batch.optimum_onehot_bits``       (certified exhaustive reference)

Scope — what IS and IS NOT validated (no overclaim):

* The stored payload is the POST-EXTRACTION envelope
  ``{"counts": {...}, "bit_order": "qiskit", "kind": "counts"}`` written by
  ``qpu_api._fetch_result_if_completed`` from ``RuntimeGateway.get_result``.
  It is NOT a raw SamplerV2 ``PrimitiveResult``; the BitArray→counts extraction
  step inside ``RuntimeGateway.get_result`` (ibm_runner.py) is therefore NOT
  re-validated here against provider bytes — only everything downstream of it.
* The QUBO is REBUILT from the manifest params and its canonical hash is
  asserted equal to the recorded ``qubo_sha256``, so the decode provably runs
  against the same mathematical object that was submitted.
* Nothing here contacts IBM, constructs a service, or reads a token. All
  artifacts are opened read-only and never modified.

Two tiers, kept explicit:

* MANDATORY tests read only the VERSIONED published artefacts
  (``results/ibm_fixed_angle_campaign_v1``) and must pass in a fresh clone;
* OPTIONAL tests read the gitignored private exports (``runs/qpu/<run_id>/artefacts``)
  and are skipped — with an explicit reason, never faked — when they are absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from acrpq.dashboard import qpu_runs, qpu_store
from acrpq.dashboard.hardware_validation import canonical_qubo_hash
from acrpq.dashboard.qpu_batch import optimum_onehot_bits
from acrpq.dashboard.qpu_result import (
    ResultAdapterError,
    decode_scientific,
    normalize_ibm_sampler_result,
)
from acrpq.io.loader import InstanceLoader
from acrpq.quantum.qubo import build_qubo

REPO = Path(__file__).resolve().parent.parent
CAMPAIGN_MANIFEST = REPO / "results" / "ibm_fixed_angle_campaign_v1" / "manifest.json"
RUNS_ROOT = str(REPO / "runs")

pytestmark = pytest.mark.skipif(
    not CAMPAIGN_MANIFEST.exists(),
    reason="real IBM campaign artifacts not present in this checkout",
)


def _campaign() -> dict:
    return json.loads(CAMPAIGN_MANIFEST.read_text())


def _jobs_with_recorded_counts() -> list[dict]:
    """Campaign jobs whose run directory holds a hash-verified result_raw doc."""
    jobs = []
    for job in _campaign()["jobs"]:
        run_id = job.get("run_id")
        if run_id and qpu_store.has(RUNS_ROOT, run_id, "result_raw"):
            jobs.append(job)
    return jobs


_JOBS = _jobs_with_recorded_counts() if CAMPAIGN_MANIFEST.exists() else []
_JOB_IDS = [j["job_key"] for j in _JOBS]

# QUBO / reference caches: (instance, objective, n_theta, n_q) is shared by
# repetitions, so build each once.
_QUBO_CACHE: dict[tuple, object] = {}
_REF_CACHE: dict[tuple, dict] = {}
_LOADER = InstanceLoader()


def _params_of(job: dict) -> dict:
    return job["export"]["artefact"]["request"]


def _qubo_for(job: dict):
    p = _params_of(job)
    key = (p["instance_id"], p["objective_id"], p["n_theta"], p["n_q"])
    if key not in _QUBO_CACHE:
        inst = _LOADER.load(p["instance_id"])
        _QUBO_CACHE[key] = (
            inst,
            build_qubo(inst, n_theta=p["n_theta"], n_q=p["n_q"],
                       objective_id=p["objective_id"]),
        )
    return _QUBO_CACHE[key]


def _reference_for(job: dict) -> dict:
    p = _params_of(job)
    key = (p["instance_id"], p["objective_id"], p["n_theta"], p["n_q"])
    if key not in _REF_CACHE:
        inst, qubo = _qubo_for(job)
        _REF_CACHE[key] = optimum_onehot_bits(
            inst, qubo, p["objective_id"], p["n_theta"], p["n_q"])
    return _REF_CACHE[key]


def test_campaign_artifacts_present_and_inventoried():
    """MANDATORY (fresh clone): the PUBLISHED campaign manifest is complete — every job
    completed, was really submitted, carries an IBM job id and a hardware (non-fake)
    artefact. Depends only on versioned files under results/ibm_fixed_angle_campaign_v1."""
    jobs = _campaign()["jobs"]
    assert len(jobs) == 30
    assert _campaign()["protocol"]["backend"] == "ibm_marrakesh"
    for job in jobs:
        assert job["state"] == "completed"
        assert job["submitted"] is True
        ids = job["ibm_job_ids"]
        assert isinstance(ids, list) and len(ids) == 1 and ids[0]
        art = job["export"]["artefact"]
        assert art["backend"] == "ibm_marrakesh"
        assert art["backend_synthetic"]["is_fake"] is False
        assert art["submittable"] is True
        assert job.get("run_id")  # every published job names its local run id


def test_published_campaign_results_match_manifest():
    """MANDATORY (fresh clone): the versioned campaign_results.json carries exactly the
    30 published jobs with the same IBM job ids as the manifest."""
    results = json.loads((CAMPAIGN_MANIFEST.parent / "campaign_results.json").read_text())
    rows = results["rows"] if isinstance(results, dict) else results
    assert len(rows) == 30
    manifest_ids = sorted(j["ibm_job_ids"][0] for j in _campaign()["jobs"])
    row_ids = sorted(str(r.get("ibm_job_id") or (r.get("ibm_job_ids") or [None])[0]) for r in rows)
    assert row_ids == manifest_ids


@pytest.mark.skipif(
    not Path(RUNS_ROOT).exists(),
    reason="OPTIONAL: local private IBM exports (runs/, gitignored) are absent in this checkout",
)
def test_local_private_exports_cover_every_run_dir():
    """OPTIONAL (local private exports only): every published job has a hash-verified
    result_raw document in the gitignored runs/ tree."""
    jobs = _campaign()["jobs"]
    assert len(_JOBS) == len([j for j in jobs if j.get("run_id")])


@pytest.mark.parametrize("job", _JOBS, ids=_JOB_IDS)
def test_run_event_chain_verifies(job):
    """The persisted run's hash chain (manifest-seeded, per-event) is intact and
    the run reached 'completed' with exactly the manifest's IBM job id."""
    run = qpu_runs.load(job["run_id"], root=RUNS_ROOT, verify_integrity=True)
    assert run["state"] == "completed"
    assert list(run["ibm_job_ids"]) == list(job["ibm_job_ids"])
    p = _params_of(job)
    assert run["params"]["instance"] == p["instance_id"]
    assert run["params"]["shots"] == p["shots"]
    assert run["params"]["backend_requested"] == "ibm_marrakesh"


@pytest.mark.parametrize("job", _JOBS, ids=_JOB_IDS)
def test_qubo_rebuilds_to_recorded_hash(job):
    """The QUBO rebuilt from the manifest params hashes to the recorded
    qubo_sha256 — the decode below therefore runs on the submitted object."""
    _inst, qubo = _qubo_for(job)
    art = job["export"]["artefact"]
    assert canonical_qubo_hash(qubo) == art["qubo_sha256"]
    p = _params_of(job)
    assert qubo.n_qubits == p["n_binary_vars"]


@pytest.mark.parametrize("job", _JOBS, ids=_JOB_IDS)
def test_normalize_real_recorded_counts(job):
    """normalize_ibm_sampler_result accepts the REAL recorded counts envelope:
    coherent bit length (n = n_aircraft * K), qiskit->qubo reversal applied,
    probabilities sum to ~1, shot total preserved."""
    raw = qpu_store.load_result_raw(job["run_id"], root=RUNS_ROOT)  # hash-verified
    p = _params_of(job)
    n = p["n_binary_vars"]
    assert n == p["n_theta"] * p["n_q"] * (n // (p["n_theta"] * p["n_q"]))  # n = K * n_aircraft
    assert raw["kind"] == "counts" and raw["bit_order"] == "qiskit"
    assert all(len(bs) == n and set(bs) <= {"0", "1"} for bs in raw["counts"])

    norm = normalize_ibm_sampler_result(raw, expected_n_bits=n)
    assert norm.kind == "counts"
    assert norm.endianness_applied == "reversed"          # qiskit -> qubo order
    assert norm.n_shots == p["shots"] == sum(raw["counts"].values())
    assert norm.n_unique == len(raw["counts"])
    total = sum(norm.prob_qubo_order.values())
    assert abs(total - 1.0) < 1e-9
    assert all(0.0 <= v <= 1.0 for v in norm.prob_qubo_order.values())
    assert all(len(bs) == n for bs in norm.prob_qubo_order)
    # the reversal is a bijection on these keys: reversing back recovers the raw set
    assert {bs[::-1] for bs in norm.prob_qubo_order} == set(raw["counts"])
    # wrong declared bit length fails CLOSED, never silently pads/truncates
    with pytest.raises(ResultAdapterError):
        normalize_ibm_sampler_result(raw, expected_n_bits=n + 1)


@pytest.mark.parametrize("job", _JOBS, ids=_JOB_IDS)
def test_decode_scientific_real_counts_against_certified_reference(job):
    """decode_scientific on the REAL counts with the certified exhaustive
    reference: feasibility_rate in [0,1], one-hot mass in [0,1], candidates
    coherent, and gap comparability declared honestly."""
    raw = qpu_store.load_result_raw(job["run_id"], root=RUNS_ROOT)
    p = _params_of(job)
    _inst, qubo = _qubo_for(job)
    ref = _reference_for(job)
    assert ref["status"] == "optimal" and ref["feasible"] is True

    dec = decode_scientific(
        qubo, raw, objective_id=p["objective_id"], reference_objective=ref["optimum"])

    assert 0.0 <= dec.feasibility_rate <= 1.0
    assert 0.0 <= dec.onehot_valid_mass <= 1.0
    assert dec.feasibility_rate <= dec.onehot_valid_mass + 1e-12  # feasible => one-hot
    assert dec.best_raw_bitstring is not None and len(dec.best_raw_bitstring) == qubo.n_qubits
    assert dec.modal_bitstring is not None
    if dec.best_feasible_bitstring is not None:
        assert dec.comparability == "comparable"
        # a feasible candidate can never beat the certified exhaustive optimum
        assert dec.delta_vs_reference is not None
        assert dec.delta_vs_reference >= -1e-9
        assert dec.best_feasible_primary_cost == pytest.approx(
            ref["optimum"] + dec.delta_vs_reference)
        # decoded (q, theta) is present for the feasible candidate
        assert dec.decoded_q_theta is not None and len(dec.decoded_q_theta) > 0
    else:
        assert dec.comparability == "not_comparable"
        assert dec.feasibility_rate == 0.0


@pytest.mark.parametrize("job", _JOBS, ids=_JOB_IDS)
def test_decode_matches_recorded_campaign_summary(job):
    """The decode pipeline TODAY reproduces the summary recorded at campaign time
    (regression pin: feasibility_rate, n_shots, best_feasible energy/bitstring)."""
    recorded = (job.get("result") or {}).get("summary")
    if not recorded:
        pytest.skip("campaign manifest carries no recorded summary for this job")
    raw = qpu_store.load_result_raw(job["run_id"], root=RUNS_ROOT)
    _inst, qubo = _qubo_for(job)
    from acrpq.dashboard.hardware_validation import decode_samples

    summary = decode_samples(qubo, raw["counts"], bit_order=raw["bit_order"],
                             kind=raw["kind"])
    assert summary.n_shots == recorded["n_shots"]
    assert summary.n_unique_total == recorded["n_unique_total"]
    assert summary.feasibility_rate == pytest.approx(recorded["feasibility_rate"])
    assert summary.mean_energy == pytest.approx(recorded["mean_energy"])
    rb, sb = recorded.get("best_feasible"), summary.best_feasible
    if rb is None:
        assert sb is None
    else:
        assert sb is not None
        assert sb.bitstring == rb["bitstring"]
        assert sb.energy == pytest.approx(rb["energy"])
        assert sb.objective == pytest.approx(rb["objective"])
