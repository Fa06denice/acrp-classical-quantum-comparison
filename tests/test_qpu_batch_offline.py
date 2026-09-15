"""Phase 5 — adversarial tests for the OFFLINE IBM batch (fake gateways only).

Pins the safety posture (fail-closed offline guard; only a dry_run_safe fake
gateway; a real RuntimeGateway is refused; no token/PII in the record) and the
scientific behaviour (the full prepare→submit→reconcile→decode pipeline reaches
`completed` and the decoded solution matches the certified classical optimum,
with exactly one submission). NO IBM, NO network, NO token.
"""

from __future__ import annotations

import json
import re
import tempfile
import warnings
from pathlib import Path

import pytest

from acrpq.io.loader import InstanceLoader

warnings.filterwarnings("ignore")

qpu_batch = pytest.importorskip("acrpq.dashboard.qpu_batch")
pytest.importorskip("qiskit_ibm_runtime")


@pytest.fixture(scope="module")
def ld():
    return InstanceLoader()


def _run(ld, instance="CP_3", objective_id="maneuver_count_v1", **kw):
    return qpu_batch.run_offline_qpu_job(
        ld, instance, objective_id=objective_id, n_theta=3,
        runs_dir=tempfile.mkdtemp(), **kw)


# --- 1. fail-closed offline guard ------------------------------------------
def test_assert_offline_passes_when_flags_are_safe_defaults():
    snap = qpu_batch.assert_offline(env=lambda _n: None)   # nothing set -> all safe
    assert snap["submission_allowed"] is False
    assert snap["runtime_factory_enabled"] is False


def test_assert_offline_refuses_a_real_submission_environment():
    real = {"ACRPQ_QPU_ENABLED": "true", "ACRPQ_IBM_SUBMISSION_ENABLED": "true",
            "ACRPQ_QPU_DRY_RUN": "false"}
    with pytest.raises(qpu_batch.OfflineSafetyError, match="submission_allowed"):
        qpu_batch.assert_offline(env=real.get)


def test_assert_offline_refuses_when_runtime_factory_enabled():
    with pytest.raises(qpu_batch.OfflineSafetyError, match="runtime factory"):
        qpu_batch.assert_offline(env={"ACRPQ_IBM_RUNTIME_FACTORY_ENABLED": "true"}.get)


# --- 2. the fake gateway is dry-run-safe; the real seam refuses others ------
def test_fake_batch_gateway_declares_dry_run_safe():
    gw = qpu_batch.FakeBatchGateway({"000": 8}, bit_order="qubo", n_bits=3)
    assert gw.dry_run_safe is True
    job = gw.run_sampler(isa_circuit=None, shots=8, tags=["t"], max_execution_time=None)
    assert job.status() == "DONE"
    assert gw.get_result(job)["counts"] == {"000": 8}


def test_orchestrator_refuses_a_gateway_without_dry_run_safe():
    # the simulate seam allow-lists dry_run_safe; a naive object is refused
    from acrpq.dashboard import qpu_orchestrator
    with pytest.raises(qpu_orchestrator.RealGatewayRefused):
        qpu_orchestrator._require_safe_gateway(object())


def test_orchestrator_refuses_a_real_runtime_gateway():
    from acrpq.dashboard import ibm_runner, qpu_orchestrator
    real = ibm_runner.RuntimeGateway()          # no service; construction is inert
    with pytest.raises(qpu_orchestrator.RealGatewayRefused):
        qpu_orchestrator._require_safe_gateway(real)


# --- 3. the full offline pipeline completes and decodes to the optimum -----
def test_offline_job_completes_and_decodes_to_the_certified_optimum(ld):
    r = _run(ld, "CP_3", "maneuver_count_v1")
    assert r["final_state"] == "completed"
    assert r["anomalies"] == []
    assert r["gateway_run_calls"] == 1                     # exactly one submit
    assert r["reference_status"] == "optimal"
    assert r["decoded_matches_reference"] is True
    assert r["decoded_best_energy"] == pytest.approx(r["reference_optimum"])
    assert r["no_ibm"] is True


def test_offline_job_objective_aware_and_bits_are_onehot(ld):
    r = _run(ld, "CP_4", "maneuver_count_v1")
    # maneuver_count optimum is an integer cardinality
    assert float(r["reference_optimum"]).is_integer()
    # one-hot: exactly one '1' per aircraft -> total ones == n_aircraft
    bits = r["decoded_best_bitstring"]
    assert bits is not None and bits.count("1") == r["n_aircraft"]


# --- 4. no token / PII / local path anywhere in a record -------------------
def test_record_has_no_token_pii_or_local_path(ld):
    blob = json.dumps(_run(ld, "CP_3", "maneuver_count_v1"))
    assert "/Users/" not in blob
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", blob)          # no email
    assert not re.search(r"(?i)bearer |api[_-]?key|secret|password|access_token", blob)
    # the ONLY 'token' mention is the reassuring safety note "no token read"
    assert all("no token read" in blob[max(0, m.start() - 15):m.start() + 12]
               for m in re.finditer("token", blob))


# --- 5. reproducibility (scientific content, excluding volatile run id) -----
def test_offline_job_scientific_content_is_reproducible(ld):
    a = _run(ld, "CP_3", "quadratic_control_cost_v1")
    b = _run(ld, "CP_3", "quadratic_control_cost_v1")
    for k in ("final_state", "decoded_best_bitstring", "decoded_best_energy",
              "plan_sha256", "artefact_sha256", "job_ids", "reference_optimum"):
        assert a[k] == b[k], k
    assert a["local_run_id"] != b["local_run_id"]           # volatile, differs


# --- 6. hash + CSV parity + volatile exclusion -----------------------------
def test_canonical_hash_excludes_volatile_run_id_and_timing():
    batch = {"schema": "x", "results": [
        {"instance": "CP_3", "decoded_best_energy": 2.0,
         "local_run_id": "20260101T000000Z-a-b", "wall_time_s": 0.1}]}
    h1 = qpu_batch.canonical_batch_sha256(batch)
    batch["results"][0]["local_run_id"] = "20260909T999999Z-c-d"
    batch["results"][0]["wall_time_s"] = 999.0
    assert qpu_batch.canonical_batch_sha256(batch) == h1            # volatile excluded
    batch["results"][0]["decoded_best_energy"] = 5.0
    assert qpu_batch.canonical_batch_sha256(batch) != h1           # science IS hashed


def test_backend_name_is_allow_listed_to_local_fakes(ld):
    # only a local Fake* provider backend may be named — no arbitrary getattr
    for bad in ("QiskitRuntimeService", "__class__", "os", "RuntimeGateway", "not_a_backend"):
        with pytest.raises(ValueError, match="Fake"):
            _run(ld, "CP_3", "maneuver_count_v1", backend_name=bad)


def test_json_csv_parity_and_tamper_detection(ld):
    rows = [_run(ld, "CP_3", "maneuver_count_v1")]
    csv_text = qpu_batch.rows_to_csv(rows)
    qpu_batch.assert_json_csv_parity(rows, csv_text)
    rows[0]["decoded_best_energy"] = 999.0
    with pytest.raises(ValueError, match="parity mismatch"):
        qpu_batch.assert_json_csv_parity(rows, csv_text)


# ===========================================================================
# Phase 5.1 — exact-type fence, shared normalizer, resumable store, crash recovery
# ===========================================================================
from acrpq.dashboard import hardware_validation as _H  # noqa: E402
from acrpq.quantum.qubo import build_qubo as _build_qubo  # noqa: E402


def _crash_bundle(ld, instance="CP_3", n_theta=2, objective_id="maneuver_count_v1",
                  gammas=(0.4,), betas=(0.3,)):
    """Build a (bundle, qubo, optimum) triple for the crash/resume tests."""
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2
    inst = ld.load(instance)
    q = _build_qubo(inst, n_theta=n_theta, n_q=1, objective_id=objective_id)
    n = q.n_qubits
    o = qpu_batch.optimum_onehot_bits(inst, q, objective_id, n_theta, 1)
    req = _H.HardwareValidationRequest(
        instance_id=instance, qubo_sha256=_H.canonical_qubo_hash(q), n_binary_vars=n,
        variable_meaning=tuple(f"x{b}" for b in range(n)), qaoa_reps=len(gammas), gammas=gammas,
        betas=betas, shots=128, transpiler_seed=1, backend_requested="fake_lagos", max_jobs=1,
        max_quantum_seconds=20.0, n_theta=n_theta, optimization_level=0, decision_ts=0.0)
    bundle = _H.prepare_hardware_bundle(req, q, FakeLagosV2(), allow_synthetic=True)
    return bundle, q, o


def _cfg(q, *, instance="CP_3", n_theta=2, n_q=1, objective_id="maneuver_count_v1",
         gammas=(0.4,), betas=(0.3,), shots=128, transpiler_seed=1, optimization_level=0,
         backend_name="FakeLagosV2", max_quantum_seconds=20.0):
    """Full batch protocol config_hash for the direct-driver tests (mirrors _crash_bundle)."""
    return qpu_batch.batch_config_hash(qpu_batch.batch_config_identity(
        instance=instance, objective_id=objective_id, n_theta=n_theta, n_q=n_q,
        gammas=gammas, betas=betas, shots=shots, transpiler_seed=transpiler_seed,
        optimization_level=optimization_level, backend_name=backend_name,
        qubo_sha256=_H.canonical_qubo_hash(q), max_quantum_seconds=max_quantum_seconds))


# --- exact-type fence vs the composition-wrapper attack --------------------
def test_composition_wrapper_defeats_marker_but_not_exact_type():
    from acrpq.dashboard import ibm_runner, qpu_orchestrator
    fake = qpu_batch.FakeBatchGateway({"000000": 8}, bit_order="qubo", n_bits=6)

    class Wrapper:  # forwards to a (would-be) real gateway but fakes the marker
        dry_run_safe = True

        def __init__(self, real):
            self._real = real

        def run_sampler(self, **k):
            return self._real.run_sampler(**k)

        def find_jobs(self, **k):
            return self._real.find_jobs(**k)

        def get_job(self, j):
            return self._real.get_job(j)

        def get_result(self, j):
            return self._real.get_result(j)

    w = Wrapper(fake)
    # the OLD self-declared marker fence is DEFEATED by the wrapper (accepts it):
    qpu_orchestrator._require_safe_gateway(w)            # does NOT raise -> marker defeated
    # the exact-type fence REFUSES the wrapper (its type is Wrapper, not FakeBatchGateway)
    with pytest.raises(qpu_batch.OfflineSafetyError, match="exact offline type"):
        qpu_batch.require_offline_exact_gateway(w)
    # a real RuntimeGateway is refused; the exact fake is accepted
    with pytest.raises(qpu_batch.OfflineSafetyError):
        qpu_batch.require_offline_exact_gateway(ibm_runner.RuntimeGateway())
    qpu_batch.require_offline_exact_gateway(fake)        # accepted


def test_exact_type_fence_refuses_a_subclass():
    class Sneaky(qpu_batch.FakeBatchGateway):
        pass
    s = Sneaky({"000": 1}, bit_order="qubo", n_bits=3)
    with pytest.raises(qpu_batch.OfflineSafetyError):   # exact type, not isinstance
        qpu_batch.require_offline_exact_gateway(s)


# --- shared normalization adapter ------------------------------------------
def test_normalize_result_counts_quasi_and_endianness():
    c = qpu_batch.normalize_result(
        {"counts": {"000": 6, "111": 2}, "bit_order": "qubo", "kind": "counts"}, expected_n_bits=3)
    assert c["prob_qubo_order"] == {"000": 0.75, "111": 0.25}
    assert c["n_shots"] == 8
    q = qpu_batch.normalize_result(
        {"counts": {"000": 0.75, "111": 0.25}, "bit_order": "qubo", "kind": "quasi"}, expected_n_bits=3)
    assert q["prob_qubo_order"] == {"000": 0.75, "111": 0.25}
    # qiskit bit order is reversed into qubo order
    e = qpu_batch.normalize_result(
        {"counts": {"001": 4}, "bit_order": "qiskit", "kind": "counts"}, expected_n_bits=3)
    assert list(e["prob_qubo_order"]) == ["100"]


@pytest.mark.parametrize("env", [
    {"counts": {}, "kind": "counts", "bit_order": "qubo"},                 # empty
    {"counts": {"000": -1}, "kind": "counts", "bit_order": "qubo"},         # negative count
    {"counts": {"000": 1.5}, "kind": "counts", "bit_order": "qubo"},        # non-int count
    {"counts": {"000": float("nan")}, "kind": "quasi", "bit_order": "qubo"},  # non-finite
    {"counts": {"000": float("inf")}, "kind": "quasi", "bit_order": "qubo"},  # non-finite
    {"kind": "counts", "bit_order": "qubo"},                                # missing counts
    {"counts": {"000": 1}, "kind": "counts", "bit_order": "weird"},          # bad endianness
    "not-a-dict",                                                            # malformed
])
def test_normalize_result_fails_closed_on_bad_input(env):
    with pytest.raises(_H.HardwareValidationError):
        qpu_batch.normalize_result(env, expected_n_bits=3)


# --- resumable batch store -------------------------------------------------
def test_batch_store_is_atomic_and_resumable(tmp_path):
    store = qpu_batch.BatchJobStore(str(tmp_path))
    assert store.load("j") is None and store.is_exported("j") is False
    store.record("j", {"local_run_id": "rid-1", "stage": "prepared"})
    assert store.load("j")["local_run_id"] == "rid-1"
    store.record("j", {"stage": "exported", "record": {"ok": True}})
    assert store.is_exported("j") is True
    assert store.load("j")["local_run_id"] == "rid-1"        # earlier field preserved (merge)


def test_run_offline_job_resume_returns_stored_record_without_rerun(ld, tmp_path):
    store = qpu_batch.BatchJobStore(str(tmp_path / "store"))
    runs = str(tmp_path / "runs")
    r1 = qpu_batch.run_offline_qpu_job(ld, "CP_3", objective_id="maneuver_count_v1", n_theta=3,
                                       runs_dir=runs, store=store, job_key="k")
    r2 = qpu_batch.run_offline_qpu_job(ld, "CP_3", objective_id="maneuver_count_v1", n_theta=3,
                                       runs_dir=runs, store=store, job_key="k")
    assert r2["local_run_id"] == r1["local_run_id"]          # same run, returned from the store
    assert r2["gateway_run_calls"] == 1                       # never re-submitted


# --- crash recovery at every boundary proves NO double submission ----------
@pytest.mark.parametrize("crash_point", [
    "before_submit", "after_job_id", "during_wait", "after_result_before_export"])
def test_crash_then_resume_never_submits_twice(ld, tmp_path, crash_point):
    bundle, q, o = _crash_bundle(ld)
    runs = str(tmp_path / "runs")
    store = qpu_batch.BatchJobStore(str(tmp_path / "store"))
    n = q.n_qubits
    # ONE shared gateway across crash + resume, so run_calls counts TOTAL submissions
    gw = qpu_batch.FakeBatchGateway({o["bitstring"]: 128}, bit_order="qubo", n_bits=n)
    base = {"objective_id": "maneuver_count_v1", "reference_optimum": o["optimum"]}

    class Boom(Exception):
        pass

    def hook(point):
        if point == crash_point:
            raise Boom(point)

    with pytest.raises(Boom):
        qpu_batch._run_offline_lifecycle(bundle, q, gw, root=runs, reference_objective=o["optimum"],
                                         store=store, job_key="j", base_record=base,
                                         config_hash=_cfg(q), crash_hook=hook)
    # resume with the SAME gateway, no crash
    rec = qpu_batch._run_offline_lifecycle(bundle, q, gw, root=runs, reference_objective=o["optimum"],
                                           store=store, job_key="j", base_record=base,
                                           config_hash=_cfg(q))
    assert rec["final_state"] == "completed"
    assert rec["resumed"] is True
    assert rec["decoded_matches_reference"] is True
    assert gw.run_calls == 1                                  # EXACTLY one submission ever


# --- durability: the run store, not the batch index, is the authority ------
def test_lost_batch_index_after_submit_never_resubmits(ld, tmp_path):
    """SCENARIO H: the batch-store index is destroyed AFTER a submission while the
    durable runs_dir is intact. A blank index must NOT look like a fresh job — the
    config_hash scan finds the existing run and reconciles it, never resubmitting."""
    from acrpq.dashboard import qpu_runs
    bundle, q, o = _crash_bundle(ld)
    runs = str(tmp_path / "runs")
    store = qpu_batch.BatchJobStore(str(tmp_path / "store"))
    gw = qpu_batch.FakeBatchGateway({o["bitstring"]: 128}, bit_order="qubo", n_bits=q.n_qubits)
    base = {"objective_id": "maneuver_count_v1", "reference_optimum": o["optimum"]}
    rec1 = qpu_batch._run_offline_lifecycle(bundle, q, gw, root=runs, reference_objective=o["optimum"],
                                            store=store, job_key="j", base_record=base,
                                           config_hash=_cfg(q))
    assert rec1["final_state"] == "completed" and gw.run_calls == 1
    # destroy the ENTIRE batch index (every per-job file) — runs_dir untouched
    for f in Path(str(tmp_path / "store")).glob("*.json"):
        f.unlink()
    blank = qpu_batch.BatchJobStore(str(tmp_path / "store"))
    assert blank.load("j") is None                            # index truly gone
    rec2 = qpu_batch._run_offline_lifecycle(bundle, q, gw, root=runs, reference_objective=o["optimum"],
                                            store=blank, job_key="j", base_record=base, config_hash=_cfg(q))
    assert rec2["resumed"] is True                            # recovered via config_hash scan
    assert rec2["final_state"] == "completed"
    assert gw.run_calls == 1                                  # NO second submission
    all_runs = qpu_runs.list_runs(root=runs)
    assert len(all_runs) == 1                                 # NO second prepare
    assert len(all_runs[0]["ibm_job_ids"]) == 1              # exactly one job attached


def test_orphan_run_before_first_index_write_is_recovered_not_duplicated(ld, tmp_path):
    """A crash between qpu_runs.prepare()/transition and the first batch-index write
    orphans a never-submitted run. Resume must adopt that orphan (config_hash scan)
    and submit it ONCE, not prepare a second run."""
    from acrpq.dashboard import qpu_runs
    bundle, q, o = _crash_bundle(ld)
    runs = str(tmp_path / "runs")
    # simulate the orphan: prepared + awaiting_confirmation, NO store entry, NO submit
    params = _H._expected_phase3_params(bundle.prepared)
    orphan = qpu_runs.prepare(params, root=runs)
    qpu_runs.transition(orphan, qpu_runs.QpuState.AWAITING_CONFIRMATION, root=runs)
    blank = qpu_batch.BatchJobStore(str(tmp_path / "store"))
    gw = qpu_batch.FakeBatchGateway({o["bitstring"]: 128}, bit_order="qubo", n_bits=q.n_qubits)
    base = {"objective_id": "maneuver_count_v1", "reference_optimum": o["optimum"]}
    rec = qpu_batch._run_offline_lifecycle(bundle, q, gw, root=runs, reference_objective=o["optimum"],
                                           store=blank, job_key="j", base_record=base, config_hash=_cfg(q))
    assert rec["local_run_id"] == orphan                      # adopted the orphan
    assert rec["resumed"] is True
    assert rec["final_state"] == "completed"
    assert gw.run_calls == 1                                  # submitted exactly once
    assert len(qpu_runs.list_runs(root=runs)) == 1           # no duplicate run


def test_find_run_by_config_hash_fails_closed_on_duplicates(ld, tmp_path):
    from acrpq.dashboard import qpu_runs
    bundle, q, o = _crash_bundle(ld)
    runs = str(tmp_path / "runs")
    params = _H._expected_phase3_params(bundle.prepared)
    r1 = qpu_runs.prepare(params, root=runs)
    r2 = qpu_runs.prepare(params, root=runs)                  # same params -> same config_hash
    assert r1 != r2
    with pytest.raises(qpu_batch.OfflineSafetyError, match="refusing to guess"):
        qpu_batch._find_run_by_config_hash(runs, bundle.prepared.config_hash)


def _corrupt_last_event(runs_root: str, local_run_id: str) -> None:
    """Tamper a run's tail event so integrity verification fails and list_runs
    surfaces it as state=='corrupted' — the manifest (config_hash) stays intact."""
    run_dir = Path(runs_root) / "qpu" / local_run_id
    events = sorted(run_dir.glob("*.event.json"), key=lambda p: int(p.name.split(".", 1)[0]))
    assert events, "run has no events to corrupt"
    events[-1].write_text('{"tampered": true}')          # breaks the hash chain


def test_corrupted_submitted_run_fails_closed_never_resubmits(ld, tmp_path):
    """The reviewer's residual: a run that was SUBMITTED then had its event log
    corrupted (runs_dir intact) must NOT be mistaken for 'never submitted'. With the
    batch index also lost, resume must FAIL CLOSED (no fresh prepare, no 2nd submit)."""
    from acrpq.dashboard import qpu_runs
    bundle, q, o = _crash_bundle(ld)
    runs = str(tmp_path / "runs")
    store = qpu_batch.BatchJobStore(str(tmp_path / "store"))
    gw = qpu_batch.FakeBatchGateway({o["bitstring"]: 128}, bit_order="qubo", n_bits=q.n_qubits)
    base = {"objective_id": "maneuver_count_v1", "reference_optimum": o["optimum"]}
    rec1 = qpu_batch._run_offline_lifecycle(bundle, q, gw, root=runs, reference_objective=o["optimum"],
                                            store=store, job_key="j", base_record=base,
                                           config_hash=_cfg(q))
    assert rec1["final_state"] == "completed" and gw.run_calls == 1
    rid = rec1["local_run_id"]
    _corrupt_last_event(runs, rid)
    # list_runs now surfaces it corrupted, but WITH the manifest config_hash preserved
    marker = next(r for r in qpu_runs.list_runs(root=runs) if r["local_run_id"] == rid)
    assert marker["state"] == "corrupted"
    assert marker["config_hash"] == bundle.prepared.config_hash
    # lose the batch index; resume must refuse rather than re-prepare/re-submit
    for f in Path(str(tmp_path / "store")).glob("*.json"):
        f.unlink()
    blank = qpu_batch.BatchJobStore(str(tmp_path / "store"))
    with pytest.raises(qpu_batch.OfflineSafetyError, match="corrupted run"):
        qpu_batch._run_offline_lifecycle(bundle, q, gw, root=runs, reference_objective=o["optimum"],
                                         store=blank, job_key="j", base_record=base, config_hash=_cfg(q))
    assert gw.run_calls == 1                                  # NO second submission
    assert len(qpu_runs.list_runs(root=runs)) == 1           # NO second prepare


def test_tampered_manifest_config_hash_fails_closed(ld, tmp_path):
    """Variant 2b: a SUBMITTED run whose manifest config_hash is itself tampered to a
    different valid-looking value must NOT be waved through as 'a different config'.
    The manifest seeds the hash chain, so the tamper breaks event0.prev_sha256 and the
    manifest hash is rejected (-> None -> fail closed). No re-prepare, no re-submit."""
    from acrpq.dashboard import qpu_runs
    bundle, q, o = _crash_bundle(ld)
    runs = str(tmp_path / "runs")
    store = qpu_batch.BatchJobStore(str(tmp_path / "store"))
    gw = qpu_batch.FakeBatchGateway({o["bitstring"]: 128}, bit_order="qubo", n_bits=q.n_qubits)
    base = {"objective_id": "maneuver_count_v1", "reference_optimum": o["optimum"]}
    rec1 = qpu_batch._run_offline_lifecycle(bundle, q, gw, root=runs, reference_objective=o["optimum"],
                                            store=store, job_key="j", base_record=base,
                                           config_hash=_cfg(q))
    assert rec1["final_state"] == "completed" and gw.run_calls == 1
    rid = rec1["local_run_id"]
    # tamper the manifest's config_hash to a DIFFERENT valid-looking sha (breaks the chain)
    mpath = Path(runs) / "qpu" / rid / "manifest.json"
    man = json.loads(mpath.read_text())
    man["config_hash"] = "sha256:" + "0" * 64
    mpath.write_text(json.dumps(man))
    # list_runs now surfaces it corrupted; the tampered hash must be REJECTED (None)
    marker = next(r for r in qpu_runs.list_runs(root=runs) if r["local_run_id"] == rid)
    assert marker["state"] == "corrupted"
    assert marker["config_hash"] is None                 # chain-verify rejected the tamper
    for f in Path(str(tmp_path / "store")).glob("*.json"):
        f.unlink()
    blank = qpu_batch.BatchJobStore(str(tmp_path / "store"))
    with pytest.raises(qpu_batch.OfflineSafetyError, match="corrupted run"):
        qpu_batch._run_offline_lifecycle(bundle, q, gw, root=runs, reference_objective=o["optimum"],
                                         store=blank, job_key="j", base_record=base, config_hash=_cfg(q))
    assert gw.run_calls == 1                              # NO second submission
    assert len(qpu_runs.list_runs(root=runs)) == 1       # NO second prepare


def test_corrupt_run_for_a_different_config_does_not_block(ld, tmp_path):
    """A corrupted run whose manifest is CHAIN-VERIFIABLE and provably belongs to a
    DIFFERENT config must be ignored (not fail-closed), so it does not block an
    unrelated fresh job sharing the runs_dir. (event0 + manifest intact; a LATER
    event rots — the realistic corruption case.)"""
    from acrpq.dashboard import qpu_runs
    runs = str(tmp_path / "runs")
    # fully run a DIFFERENT-objective job so it has event0 (intact) + later events
    other, oq, oo = _crash_bundle(ld, objective_id="quadratic_control_cost_v1")
    ogw = qpu_batch.FakeBatchGateway({oo["bitstring"]: 128}, bit_order="qubo", n_bits=oq.n_qubits)
    obase = {"objective_id": "quadratic_control_cost_v1", "reference_optimum": oo["optimum"]}
    orec = qpu_batch._run_offline_lifecycle(
        other, oq, ogw, root=runs, reference_objective=oo["optimum"],
        store=qpu_batch.BatchJobStore(str(tmp_path / "os")), job_key="o", base_record=obase,
        config_hash=_cfg(oq, objective_id="quadratic_control_cost_v1"))
    other_rid = orec["local_run_id"]
    _corrupt_last_event(runs, other_rid)                 # LATER event; event0+manifest intact
    marker = next(r for r in qpu_runs.list_runs(root=runs) if r["local_run_id"] == other_rid)
    assert marker["state"] == "corrupted"
    assert marker["config_hash"] == other.prepared.config_hash   # chain-verified real hash
    # our (unrelated) config must NOT be blocked by that corrupt-but-different run
    bundle, q, o = _crash_bundle(ld, objective_id="maneuver_count_v1")
    assert other.prepared.config_hash != bundle.prepared.config_hash
    assert qpu_batch._find_run_by_config_hash(runs, bundle.prepared.config_hash) is None


# --- Phase 5.1.1: full protocol-config identity guards store reuse ---------
@pytest.mark.parametrize("field,newval", [
    ("shots", 256),
    ("gammas", (0.5,)),
    ("betas", (0.2,)),
    ("transpiler_seed", 2),
    ("optimization_level", 1),
    ("backend_name", "FakeLagosV2"),
    ("objective_id", "quadratic_control_cost_v1"),
    ("n_theta", 5),
    ("n_q", 2),
    ("max_quantum_seconds", 40.0),
])
def test_each_config_axis_changes_the_batch_hash(field, newval):
    """Every protocol axis must move the batch config hash — so none can silently
    ride a stale job_key: shots, angles (gammas/betas), seed, transpile level,
    backend, objective, grid (n_theta/n_q), and the quantum-time budget."""
    base = dict(instance="CP_3", objective_id="maneuver_count_v1", n_theta=3, n_q=1,
                gammas=(0.4,), betas=(0.3,), shots=512, transpiler_seed=1,
                optimization_level=0, backend_name="FakeGuadalupeV2",
                qubo_sha256="sha256:" + "a" * 64, max_quantum_seconds=20.0)
    h0 = qpu_batch.batch_config_hash(qpu_batch.batch_config_identity(**base))
    h1 = qpu_batch.batch_config_hash(qpu_batch.batch_config_identity(**{**base, field: newval}))
    assert h0 != h1


def test_qubo_hash_and_protocol_version_change_the_batch_hash():
    ident = qpu_batch.batch_config_identity(
        instance="CP_3", objective_id="maneuver_count_v1", n_theta=3, n_q=1,
        gammas=(0.4,), betas=(0.3,), shots=512, transpiler_seed=1, optimization_level=0,
        backend_name="FakeGuadalupeV2", qubo_sha256="sha256:" + "a" * 64, max_quantum_seconds=20.0)
    h0 = qpu_batch.batch_config_hash(ident)
    # objective/grid/instance also flow through qubo_sha256 -> a different QUBO flips it
    assert h0 != qpu_batch.batch_config_hash({**ident, "qubo_sha256": "sha256:" + "b" * 64})
    # the protocol VERSION is part of the identity
    assert h0 != qpu_batch.batch_config_hash(
        {**ident, "protocol_version": "acrpq-qpu-batch-offline/999"})


@pytest.mark.parametrize("stage_point,expect_stage", [
    ("before_submit", "prepared"),
    ("after_job_id", "submitted"),
    (None, "exported"),
])
def test_config_mismatch_refused_at_every_stage(ld, tmp_path, stage_point, expect_stage):
    """A store entry created for config A must refuse a resume under config B at EVERY
    stage (prepared / submitted / exported) — fail closed, never reuse or overwrite,
    never a (further) submission."""
    bundle, q, o = _crash_bundle(ld)
    runs = str(tmp_path / "runs")
    store = qpu_batch.BatchJobStore(str(tmp_path / "store"))
    gw = qpu_batch.FakeBatchGateway({o["bitstring"]: 128}, bit_order="qubo", n_bits=q.n_qubits)
    base = {"objective_id": "maneuver_count_v1", "reference_optimum": o["optimum"]}
    cfg_a = _cfg(q, shots=128)
    if stage_point is not None:
        class Boom(Exception):
            pass

        def hook(pt):
            if pt == stage_point:
                raise Boom(pt)

        with pytest.raises(Boom):
            qpu_batch._run_offline_lifecycle(bundle, q, gw, root=runs, reference_objective=o["optimum"],
                store=store, job_key="j", base_record=base, config_hash=cfg_a, crash_hook=hook)
    else:
        qpu_batch._run_offline_lifecycle(bundle, q, gw, root=runs, reference_objective=o["optimum"],
            store=store, job_key="j", base_record=base, config_hash=cfg_a)
    entry = store.load("j")
    assert entry["stage"] == expect_stage and entry["config_hash"] == cfg_a
    calls = gw.run_calls
    cfg_b = _cfg(q, shots=256)                          # a single-axis change (shots)
    assert cfg_a != cfg_b
    with pytest.raises(qpu_batch.OfflineSafetyError, match="different protocol config"):
        qpu_batch._run_offline_lifecycle(bundle, q, gw, root=runs, reference_objective=o["optimum"],
            store=store, job_key="j", base_record=base, config_hash=cfg_b)
    assert gw.run_calls == calls                        # refused before any (further) submit
    assert store.load("j")["config_hash"] == cfg_a      # old entry NOT overwritten


def test_run_offline_job_refuses_reuse_across_config_change(ld, tmp_path):
    """Through the PUBLIC entry: a fixed job_key with a changed config (shots) must be
    refused, never returning the stale exported record."""
    store = qpu_batch.BatchJobStore(str(tmp_path / "store"))
    runs = str(tmp_path / "runs")
    r1 = qpu_batch.run_offline_qpu_job(ld, "CP_3", objective_id="maneuver_count_v1", n_theta=3,
        shots=512, backend_name="FakeGuadalupeV2", runs_dir=runs, store=store, job_key="fixed")
    assert r1["final_state"] == "completed"
    with pytest.raises(qpu_batch.OfflineSafetyError, match="different protocol config"):
        qpu_batch.run_offline_qpu_job(ld, "CP_3", objective_id="maneuver_count_v1", n_theta=3,
            shots=1024, backend_name="FakeGuadalupeV2", runs_dir=runs, store=store, job_key="fixed")


@pytest.mark.parametrize("override", [
    {"gammas": (0.5,)},             # weak axis: NOT in the run-store hash (Finding #2)
    {"betas": (0.2,)},              # weak axis
    {"optimization_level": 3},      # weak axis: NOT in the run-store hash (Finding #2)
    {"shots": 1024},                # strong axis: IS in the run-store hash (control)
    {"max_quantum_seconds": 40.0},  # Finding #1: in plan/artefact hashes, not run hash
])
def test_distinct_configs_get_distinct_runs_in_a_shared_store(ld, tmp_path, override):
    """Two configs sharing runs_dir+store must get DISTINCT runs even when they differ
    ONLY in an axis the run-store hash omits (gammas/betas/optimization_level) or that
    is absent from it (max_quantum_seconds). The recovery scan must not adopt the other
    config's run, and the exported-return must not reuse its record."""
    store = qpu_batch.BatchJobStore(str(tmp_path / "store"))
    runs = str(tmp_path / "runs")
    base_kwargs = dict(objective_id="maneuver_count_v1", n_theta=3, gammas=(0.4,), betas=(0.3,),
                       shots=512, optimization_level=0, max_quantum_seconds=20.0,
                       backend_name="FakeGuadalupeV2", runs_dir=runs, store=store)
    r1 = qpu_batch.run_offline_qpu_job(ld, "CP_3", **base_kwargs)
    r2 = qpu_batch.run_offline_qpu_job(ld, "CP_3", **{**base_kwargs, **override})
    assert r1["final_state"] == "completed" and r2["final_state"] == "completed"
    assert r1["local_run_id"] != r2["local_run_id"]     # no cross-config run adoption/reuse


def test_claimed_by_other_config_fails_closed_on_unidentified_claim(tmp_path):
    """A store entry that claims a run but records no config_hash (legacy/foreign) is
    ambiguous — the guard must fail closed rather than silently adopt or resubmit."""
    store = qpu_batch.BatchJobStore(str(tmp_path / "store"))
    store.record("legacy", {"local_run_id": "run-X"})        # no config_hash recorded
    with pytest.raises(qpu_batch.OfflineSafetyError, match="without a config_hash"):
        qpu_batch._claimed_by_other_config(store, "run-X", "sha256:" + "a" * 64)
    # an entry for a DIFFERENT run is not consulted (no false positive)
    assert qpu_batch._claimed_by_other_config(store, "run-Y", "sha256:" + "a" * 64) is False


def test_crash_before_runid_one_designated_orphan_never_submitted(ld, tmp_path):
    """Codex adversarial scenario: winner prepares candidate A then crashes before
    committing the runid; a recoverer prepares candidate B and designates it. Exactly
    ONE run is designated (B), only B reaches submit_run, and orphan A is never
    submitted. (Guarantee is 'one designated + submitted', not 'one candidate prepared'.)"""
    from acrpq.dashboard import ibm_runner, qpu_claim, qpu_runs
    from acrpq.dashboard.persist import _sha256
    bundle, q, o = _crash_bundle(ld)
    runs = str(tmp_path / "runs")
    cfg = _cfg(q)
    claims = qpu_batch._claims_dir(runs)
    Path(claims).mkdir(parents=True, exist_ok=True)
    # 1. winner claimed + prepared candidate A, then crashed before committing the runid
    claim_path, _runid = qpu_claim.claim_paths(claims, cfg)
    body = {"schema": qpu_claim.CLAIM_SCHEMA, "config_hash": cfg, "nonce": "winner"}
    body["content_sha256"] = _sha256(qpu_claim._canonical(body))
    claim_path.write_text(qpu_claim._canonical(body))
    orphan_a = qpu_batch._prepare_awaiting_run(bundle.prepared, runs)
    # 2. recoverer waits, times out, prepares B, designates it
    ticks = iter([0.0, 0.0, 100.0, 100.0, 100.0])
    res = qpu_claim.claim_or_adopt(
        cfg, claims_dir=claims,
        prepare_fn=lambda: qpu_batch._prepare_awaiting_run(bundle.prepared, runs),
        wait_timeout=1.0, sleeper=lambda _s: None, monotonic=lambda: next(ticks))
    designated_b = res.local_run_id
    assert res.role == "recovered" and designated_b != orphan_a
    assert qpu_claim.read_designated(cfg, claims) == designated_b     # exactly one designated
    # 3. only the designated run B is submitted; orphan A is never submitted
    gw = qpu_batch.FakeBatchGateway({o["bitstring"]: 128}, bit_order="qubo", n_bits=q.n_qubits)
    ibm_runner.submit_run(designated_b, gw, isa_circuit=bundle.isa_circuit,
        expected_config_hash=bundle.prepared.config_hash, root=runs,
        sleeper=lambda _s: None, jitter=lambda: 0.0)
    assert qpu_runs.load(designated_b, root=runs)["ibm_job_ids"]      # B submitted once
    a = qpu_runs.load(orphan_a, root=runs)
    assert not a.get("ibm_job_ids")                                  # A NEVER submitted
    assert a["state"] == qpu_runs.QpuState.AWAITING_CONFIRMATION.value
    assert gw.run_calls == 1
    # 4. orphan inventory flags A (not B); past the lease it is safe_to_gc
    inv = qpu_batch.list_orphan_runs(runs, cfg, now_epoch=9e18, lease_seconds=3600.0)
    ids = [x["local_run_id"] for x in inv["orphans"]]
    assert orphan_a in ids and designated_b not in ids
    assert all(x["classification"] == "safe_to_gc" for x in inv["orphans"])


def test_orphan_inventory_respects_the_lease(ld, tmp_path):
    """A freshly-prepared orphan (within the lease) is 'possibly_active' and must NOT be
    flagged safe to garbage-collect."""
    from acrpq.dashboard import qpu_claim
    bundle, q, o = _crash_bundle(ld)
    runs = str(tmp_path / "runs")
    cfg = _cfg(q)
    claims = qpu_batch._claims_dir(runs)
    designated = qpu_batch._prepare_awaiting_run(bundle.prepared, runs)
    qpu_claim._commit_runid(qpu_claim.claim_paths(claims, cfg)[1], cfg, designated)
    orphan = qpu_batch._prepare_awaiting_run(bundle.prepared, runs)   # a same-config candidate
    inv = qpu_batch.list_orphan_runs(runs, cfg, now_epoch=None, lease_seconds=3600.0)  # ~now
    match = [x for x in inv["orphans"] if x["local_run_id"] == orphan]
    assert match and match[0]["classification"] == "possibly_active"  # young -> leave alone


# --- Phase 5.2.B: full batch identity persisted in the run store -----------
def test_batch_identity_persisted_and_verified(ld, tmp_path):
    bundle, q, o = _crash_bundle(ld)
    runs = str(tmp_path / "runs")
    store = qpu_batch.BatchJobStore(str(tmp_path / "store"))
    gw = qpu_batch.FakeBatchGateway({o["bitstring"]: 128}, bit_order="qubo", n_bits=q.n_qubits)
    base = {"objective_id": "maneuver_count_v1", "reference_optimum": o["optimum"]}
    cfg = _cfg(q)
    rec = qpu_batch._run_offline_lifecycle(bundle, q, gw, root=runs, reference_objective=o["optimum"],
        store=store, job_key="j", base_record=base, config_hash=cfg)
    assert rec["batch_identity_status"] == "verified"
    assert qpu_batch.read_batch_identity(runs, rec["local_run_id"]) == cfg
    assert "batch_identity:legacy_unverified" not in rec["anomalies"]


def test_batch_identity_mismatch_is_refused(ld, tmp_path):
    """A run bound to batch config A must never be reused under config B."""
    bundle, _q, _o = _crash_bundle(ld)
    runs = str(tmp_path / "runs")
    rid = qpu_batch._prepare_awaiting_run(bundle.prepared, runs, "sha256:" + "a" * 64)
    assert qpu_batch.verify_batch_identity(runs, rid, "sha256:" + "a" * 64) == "verified"
    with pytest.raises(qpu_batch.OfflineSafetyError, match="different protocol identity"):
        qpu_batch.verify_batch_identity(runs, rid, "sha256:" + "b" * 64)


def test_legacy_run_without_identity_is_flagged_not_refused(ld, tmp_path):
    """A run prepared before 5.2.B (no identity companion) is honestly flagged
    legacy_unverified (offline) — allowed but never silently 'verified'."""
    from acrpq.dashboard import qpu_runs
    bundle, _q, _o = _crash_bundle(ld)
    runs = str(tmp_path / "runs")
    rid = qpu_runs.prepare(qpu_batch.H._expected_phase3_params(bundle.prepared), root=runs)
    assert qpu_batch.read_batch_identity(runs, rid) is None
    assert qpu_batch.verify_batch_identity(runs, rid, "sha256:" + "a" * 64) == "legacy_unverified"


def test_batch_identity_fails_closed_on_tamper(ld, tmp_path):
    """The identity lives in the append-only chain, so tampering it breaks the integrity
    chain (CorruptRunError), not a companion content hash."""
    from acrpq.dashboard import qpu_runs
    bundle, _q, _o = _crash_bundle(ld)
    runs = str(tmp_path / "runs")
    rid = qpu_batch._prepare_awaiting_run(bundle.prepared, runs, "sha256:" + "a" * 64)
    run_dir = Path(runs) / "qpu" / rid
    for ev in sorted(run_dir.glob("*.event.json")):
        obj = json.loads(ev.read_text())
        if obj.get("event_kind") == "batch_identity":
            obj["batch_config_hash"] = "sha256:" + "e" * 64
            ev.write_text(json.dumps(obj))
            break
    with pytest.raises(qpu_runs.CorruptRunError):
        qpu_batch.read_batch_identity(runs, rid)


def test_identity_is_in_the_chain_before_awaiting(ld, tmp_path):
    """Codex item 4: the identity is bound in the authoritative chain BEFORE
    AWAITING_CONFIRMATION."""
    from acrpq.dashboard import qpu_runs
    bundle, _q, _o = _crash_bundle(ld)
    runs = str(tmp_path / "runs")
    rid = qpu_batch._prepare_awaiting_run(bundle.prepared, runs, "sha256:" + "a" * 64)
    assert qpu_runs.batch_identity_of(rid, root=runs) == "sha256:" + "a" * 64
    # the identity event's seq precedes the awaiting_confirmation transition
    run_dir = Path(runs) / "qpu" / rid
    seqs = {}
    for ev in run_dir.glob("*.event.json"):
        o = json.loads(ev.read_text())
        if o.get("event_kind") == "batch_identity":
            seqs["identity"] = o["seq"]
        if o.get("event_kind") == "transition" and o.get("state") == "awaiting_confirmation":
            seqs["awaiting"] = o["seq"]
    assert seqs["identity"] < seqs["awaiting"]


def test_hardware_submittable_refuses_legacy_but_accepts_verified(ld, tmp_path):
    """Codex item 4: a legacy run (no chain identity, not claim-designated) is inspectable
    but NOT hardware-submittable; a bound run is."""
    from acrpq.dashboard import qpu_runs
    bundle, _q, _o = _crash_bundle(ld)
    runs = str(tmp_path / "runs")
    cfg = "sha256:" + "a" * 64
    legacy = qpu_runs.prepare(qpu_batch.H._expected_phase3_params(bundle.prepared), root=runs)
    with pytest.raises(qpu_batch.OfflineSafetyError, match="not submittable|no authoritative"):
        qpu_batch.assert_hardware_submittable(runs, legacy, cfg)
    bound = qpu_batch._prepare_awaiting_run(bundle.prepared, runs, cfg)
    assert qpu_batch.assert_hardware_submittable(runs, bound, cfg) == "verified"


def test_crash_before_identity_commit_completes_the_transaction(ld, tmp_path):
    """Codex item 5: prepare succeeded but the identity event was not committed (crash).
    On restart the claim DESIGNATES the run, so verify completes the transaction (binds the
    identity) rather than continuing as legacy — and it becomes hardware-submittable."""
    from acrpq.dashboard import qpu_claim, qpu_runs
    bundle, _q, _o = _crash_bundle(ld)
    runs = str(tmp_path / "runs")
    cfg = "sha256:" + "a" * 64
    # simulate the crash: prepared + AWAITING but NO batch_identity event
    rid = qpu_runs.prepare(qpu_batch.H._expected_phase3_params(bundle.prepared), root=runs)
    qpu_runs.transition(rid, qpu_runs.QpuState.AWAITING_CONFIRMATION, root=runs)
    assert qpu_runs.batch_identity_of(rid, root=runs) is None      # identity not committed
    # the single-winner claim designates this run for cfg (as it would after recovery)
    qpu_claim._commit_runid(qpu_claim.claim_paths(qpu_batch._claims_dir(runs), cfg)[1], cfg, rid)
    status = qpu_batch.verify_batch_identity(runs, rid, cfg)
    assert status == "identity_completed"                          # NOT legacy_unverified
    assert qpu_runs.batch_identity_of(rid, root=runs) == cfg        # transaction completed
    assert qpu_batch.assert_hardware_submittable(runs, rid, cfg) == "verified"


def test_configs_differing_only_by_weak_axes_get_distinct_identities(ld, tmp_path):
    """angles / shots / seed / transpile-level / protocol each yield a distinct persisted
    identity, so a cross-config reuse would be refused."""
    bundle, q, _o = _crash_bundle(ld, gammas=(0.4,))
    runs = str(tmp_path / "runs")
    cfg_a = _cfg(q, gammas=(0.4,))
    cfg_b = _cfg(q, gammas=(0.5,))                          # differs only in angle
    assert cfg_a != cfg_b
    rid = qpu_batch._prepare_awaiting_run(bundle.prepared, runs, cfg_a)
    assert qpu_batch.verify_batch_identity(runs, rid, cfg_a) == "verified"
    with pytest.raises(qpu_batch.OfflineSafetyError):
        qpu_batch.verify_batch_identity(runs, rid, cfg_b)   # angle-only change refused


def test_fresh_prepare_goes_through_the_single_winner_claim(ld, tmp_path):
    """The genuinely-fresh path materialises the run via the single-winner claim, so a
    committed runid file (keyed by the FULL batch config_hash) is published alongside
    the run and names it. This is the durable single-winner authority (5.2.C)."""
    from acrpq.dashboard import qpu_claim
    bundle, q, o = _crash_bundle(ld)
    runs = str(tmp_path / "runs")
    store = qpu_batch.BatchJobStore(str(tmp_path / "store"))
    gw = qpu_batch.FakeBatchGateway({o["bitstring"]: 128}, bit_order="qubo", n_bits=q.n_qubits)
    base = {"objective_id": "maneuver_count_v1", "reference_optimum": o["optimum"]}
    cfg = _cfg(q)
    rec = qpu_batch._run_offline_lifecycle(bundle, q, gw, root=runs, reference_objective=o["optimum"],
        store=store, job_key="j", base_record=base, config_hash=cfg)
    assert rec["final_state"] == "completed" and gw.run_calls == 1 and rec["resumed"] is False
    # the claim's committed runid names exactly the designated run
    _claim, runid = qpu_claim.claim_paths(qpu_batch._claims_dir(runs), cfg)
    assert runid.exists()
    assert qpu_claim._read_committed_runid(runid, cfg) == rec["local_run_id"]


def test_recovery_scan_does_not_adopt_a_different_configs_run(ld, tmp_path):
    """Driver-level: two bundles differing ONLY in gammas share bundle.prepared.config_hash
    (the run-store hash omits gammas). Config B must NOT adopt config A's run via the scan
    — it prepares and submits its OWN run."""
    runs = str(tmp_path / "runs")
    store = qpu_batch.BatchJobStore(str(tmp_path / "store"))
    ba, qa, oa = _crash_bundle(ld, gammas=(0.4,))
    bb, qb, ob = _crash_bundle(ld, gammas=(0.5,))
    assert ba.prepared.config_hash == bb.prepared.config_hash   # run-store hash can't tell them apart
    base = {"objective_id": "maneuver_count_v1", "reference_optimum": oa["optimum"]}
    gwa = qpu_batch.FakeBatchGateway({oa["bitstring"]: 128}, bit_order="qubo", n_bits=qa.n_qubits)
    gwb = qpu_batch.FakeBatchGateway({ob["bitstring"]: 128}, bit_order="qubo", n_bits=qb.n_qubits)
    ra = qpu_batch._run_offline_lifecycle(ba, qa, gwa, root=runs, reference_objective=oa["optimum"],
        store=store, job_key="A", base_record=base, config_hash=_cfg(qa, gammas=(0.4,)))
    rb = qpu_batch._run_offline_lifecycle(bb, qb, gwb, root=runs, reference_objective=ob["optimum"],
        store=store, job_key="B", base_record=base, config_hash=_cfg(qb, gammas=(0.5,)))
    assert ra["local_run_id"] != rb["local_run_id"]     # B did NOT adopt A's run
    assert rb["resumed"] is False                        # B prepared+submitted its own
    assert gwb.run_calls == 1                            # B genuinely submitted once


def test_batch_store_load_fails_closed_on_corruption(tmp_path):
    store = qpu_batch.BatchJobStore(str(tmp_path))
    store.record("j", {"local_run_id": "rid-1", "stage": "prepared"})
    store._path("j").write_text("{ this is not valid json")
    with pytest.raises(qpu_batch.OfflineSafetyError, match="corrupt batch-store entry"):
        store.load("j")
    # a JSON value that isn't an object is also refused
    store._path("j").write_text("[1, 2, 3]")
    with pytest.raises(qpu_batch.OfflineSafetyError, match="expected object"):
        store.load("j")
