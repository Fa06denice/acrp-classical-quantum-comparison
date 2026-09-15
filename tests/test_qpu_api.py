"""Phase 7 tests: fail-closed feature flags, replay-proof confirmation, and the
guarded QPU API (prepare/dry-run/confirm/status/cancel). No IBM, no submission.
"""

from __future__ import annotations

import tempfile
import warnings

import pytest

from acrpq.dashboard import qpu_confirm, qpu_flags

warnings.filterwarnings("ignore")

_SNAP = {
    "name": "fake_lagos", "region": "us-east", "family": "Falcon", "physical_qubits": 7,
    "programmable_qubits": 7, "operational": True, "maintenance": False, "pending_jobs": 3,
    "two_qubit_error": 0.006, "readout_error": 0.012, "clops": 2500.0,
    "avg_coupling_degree": 2.0, "data_ts": 0.0,
}
_BODY = {
    "instance": "CP_3", "n_theta": 2, "shots": 512, "transpiler_seed": 1,
    "backend_requested": "fake_lagos", "gammas": [0.4], "betas": [0.3], "max_jobs": 1,
    "max_quantum_seconds": 20.0, "optimization_level": 0, "snapshots": [_SNAP],
}


# --------------------------------------------------------------------------- #
# feature flags (fail closed)
# --------------------------------------------------------------------------- #
def test_flags_default_closed():
    env = {}.get
    assert qpu_flags.qpu_enabled(env=env) is False
    assert qpu_flags.ibm_submission_enabled(env=env) is False
    assert qpu_flags.dry_run_only(env=env) is True          # safe default: force dry-run
    assert qpu_flags.submission_allowed(env=env) is False


def test_submission_requires_all_three():
    env = {"ACRPQ_QPU_ENABLED": "true", "ACRPQ_IBM_SUBMISSION_ENABLED": "true",
           "ACRPQ_QPU_DRY_RUN": "false"}.get
    assert qpu_flags.submission_allowed(env=env) is True
    # flip any one -> not allowed
    for override in ({"ACRPQ_QPU_ENABLED": "false"}, {"ACRPQ_IBM_SUBMISSION_ENABLED": "false"},
                     {"ACRPQ_QPU_DRY_RUN": "true"}):
        e = {"ACRPQ_QPU_ENABLED": "true", "ACRPQ_IBM_SUBMISSION_ENABLED": "true",
             "ACRPQ_QPU_DRY_RUN": "false", **override}.get
        assert qpu_flags.submission_allowed(env=e) is False


def test_invalid_flag_values_are_safe():
    env = {"ACRPQ_QPU_ENABLED": "banana", "ACRPQ_QPU_DRY_RUN": "maybe"}.get
    assert qpu_flags.qpu_enabled(env=env) is False        # unrecognised -> default False
    assert qpu_flags.dry_run_only(env=env) is True        # unrecognised -> default True


def test_empty_flag_value_is_treated_as_unset():
    # an explicitly-empty ACRPQ_QPU_DRY_RUN (unresolved interpolation / empty secret)
    # must NOT fail open — it stays at the safe default (force dry-run).
    env = {"ACRPQ_QPU_ENABLED": "true", "ACRPQ_IBM_SUBMISSION_ENABLED": "true",
           "ACRPQ_QPU_DRY_RUN": ""}.get
    assert qpu_flags.dry_run_only(env=env) is True
    assert qpu_flags.submission_allowed(env=env) is False


# --------------------------------------------------------------------------- #
# confirmation store (single-use, expiry, replay, binding)
# --------------------------------------------------------------------------- #
def _store(now_ref):
    return qpu_confirm.ConfirmationStore(now=lambda: now_ref[0], ttl_s=100.0,
                                         nonce_factory=lambda: "nonce-1")


def _issue(store):
    return store.issue(run_id="r1", config_hash="sha256:" + "a" * 64,
                       artefact_sha256="sha256:" + "b" * 64, decision_sha256="sha256:" + "c" * 64,
                       backend="fake_lagos", total_shots=512, max_jobs=1, max_quantum_seconds=20.0)


def _good_kwargs(ch):
    return dict(nonce=ch.nonce, run_id=ch.run_id, config_hash=ch.config_hash,
                artefact_sha256=ch.artefact_sha256, decision_sha256=ch.decision_sha256,
                backend=ch.backend, total_shots=ch.total_shots, max_jobs=ch.max_jobs,
                max_quantum_seconds=ch.max_quantum_seconds, consent=qpu_confirm.CONSENT_PHRASE)


def test_confirm_happy_path_and_single_use():
    now = [1000.0]
    store = _store(now)
    ch = _issue(store)
    consumed = store.consume(**_good_kwargs(ch))
    assert consumed.run_id == "r1"
    # replay -> refused
    with pytest.raises(qpu_confirm.ConfirmationError):
        store.consume(**_good_kwargs(ch))


def test_confirm_expiry():
    now = [1000.0]
    store = _store(now)
    ch = _issue(store)
    now[0] = 1000.0 + 200.0  # past ttl
    with pytest.raises(qpu_confirm.ConfirmationError):
        store.consume(**_good_kwargs(ch))


@pytest.mark.parametrize("mutate", [
    {"consent": "nope"}, {"config_hash": "sha256:" + "0" * 64},
    {"artefact_sha256": "sha256:" + "0" * 64}, {"backend": "other"},
    {"total_shots": 999}, {"nonce": "wrong"},
])
def test_confirm_rejects_mismatch(mutate):
    now = [1000.0]
    store = _store(now)
    ch = _issue(store)
    kwargs = _good_kwargs(ch)
    kwargs.update(mutate)
    with pytest.raises(qpu_confirm.ConfirmationError):
        store.consume(**kwargs)


def test_confirm_rejects_changed_config():
    now = [1000.0]
    store = _store(now)
    ch = _issue(store)
    with pytest.raises(qpu_confirm.ConfirmationError):
        store.consume(**_good_kwargs(ch), current_config_hash="sha256:" + "9" * 64)


# --------------------------------------------------------------------------- #
# guarded API (TestClient)
# --------------------------------------------------------------------------- #
fastapi = pytest.importorskip("fastapi")


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("ACRPQ_QPU_ENABLED", "true")   # enable prepare/confirm for tests
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app
    rd = tempfile.mkdtemp()
    app = create_app(runs_dir=rd)
    tc = TestClient(app)
    tc.runs_dir = rd
    return tc


def test_flags_endpoint(client):
    r = client.get("/api/qpu/flags")
    assert r.status_code == 200 and r.json()["submission_allowed"] is False


def test_run_list_supports_restart_safe_ui_resume(client):
    assert client.get("/api/qpu/runs").json()["runs"] == []
    prepared = client.post("/api/qpu/prepare", json=_BODY).json()
    rows = client.get("/api/qpu/runs").json()["runs"]
    assert rows[0]["run_id"] == prepared["run_id"]
    assert rows[0]["state"] == "prepared"
    assert rows[0]["backend"] == "fake_lagos"


def test_assess_endpoint(client):
    r = client.post("/api/qpu/backends/assess",
                    json={"logical_qubits": 6, "interaction_density": 0.3, "shots": 512,
                          "snapshots": [_SNAP]})
    assert r.status_code == 200
    assert r.json()["ranking"][0]["name"] == "fake_lagos"


def test_live_backend_endpoint_uses_server_side_sdk_snapshot(client, monkeypatch):
    from acrpq.dashboard import qpu_api

    class Param:
        def __init__(self, name, value):
            self.name, self.value = name, value

    class Gate:
        gate, qubits = "cz", [0, 1]
        parameters = [Param("gate_error", 0.004)]

    class Props:
        gates, last_update_date = [Gate()], "2026-08-17T14:00:00Z"

        @staticmethod
        def readout_error(_qubit):
            return 0.01

    class Status:
        operational, pending_jobs = True, 0

    class Coupling:
        @staticmethod
        def get_edges():
            return [(0, 1), (1, 0)]

    class Backend:
        name, num_qubits = "ibm_marrakesh", 156
        processor_type = {"family": "Heron"}
        coupling_map = Coupling()

        @staticmethod
        def status():
            return Status()

        @staticmethod
        def properties():
            return Props()

    monkeypatch.setattr(qpu_api, "_resolve_real_backend", lambda *_a, **_k: Backend())
    r = client.get("/api/qpu/backends/live/ibm_marrakesh")
    assert r.status_code == 200
    body = r.json()
    assert body["read_only"] is True and body["submitted"] is False
    assert body["backend"]["name"] == "ibm_marrakesh"
    assert body["backend"]["synthetic"] is False
    assert body["backend"]["pending_jobs"] == 0


def test_dry_run_never_submittable(client):
    r = client.post("/api/qpu/dry-run", json=_BODY)
    assert r.status_code == 200
    body = r.json()
    assert body["submittable"] is False
    assert "no state transitioned to submitting" in body["notes"]


def test_prepare_threads_objective_identity_into_persisted_run(client):
    prepared = client.post(
        "/api/qpu/prepare", json={**_BODY, "objective_id": "maneuver_count_v1"})
    assert prepared.status_code == 200
    run = client.get(f"/api/qpu/{prepared.json()['run_id']}").json()
    assert run["params"]["objective_id"] == "maneuver_count_v1"


def test_prepare_rejects_unknown_objective_before_submission(client):
    response = client.post(
        "/api/qpu/prepare", json={**_BODY, "objective_id": "invented"})
    assert response.status_code == 422


def test_prepare_confirm_flow(client):
    p = client.post("/api/qpu/prepare", json=_BODY)
    assert p.status_code == 200
    pj = p.json()
    assert pj["submittable"] is False and pj["state"] == "prepared"
    conf = pj["confirmation"]
    good = {
        "nonce": conf["nonce"], "config_hash": pj["config_hash"],
        "artefact_sha256": pj["artefact_sha256"], "decision_sha256": pj["decision_sha256"],
        "backend": pj["backend"], "total_shots": pj["budget"]["total_shots"],
        "max_jobs": pj["budget"]["max_jobs"],
        "max_quantum_seconds": pj["budget"]["max_quantum_seconds"],
        "consent": conf["consent_phrase"],
    }
    ok = client.post(f"/api/qpu/{pj['run_id']}/confirm", json=good)
    assert ok.status_code == 200
    assert ok.json()["state"] == "awaiting_confirmation"
    assert ok.json()["submission_allowed"] is False   # flags still withhold submission
    # an IDENTICAL re-confirm (lost response / double click) is idempotent -> 200
    assert client.post(f"/api/qpu/{pj['run_id']}/confirm", json=good).status_code == 200
    # a DIVERGENT replay (same nonce, changed payload) is refused
    bad = {**good, "total_shots": 999999}
    assert client.post(f"/api/qpu/{pj['run_id']}/confirm", json=bad).status_code == 422


def test_confirm_rejects_bad_nonce_and_consent(client):
    pj = client.post("/api/qpu/prepare", json=_BODY).json()
    base = {
        "config_hash": pj["config_hash"], "artefact_sha256": pj["artefact_sha256"],
        "decision_sha256": pj["decision_sha256"], "backend": pj["backend"],
        "total_shots": pj["budget"]["total_shots"], "max_jobs": pj["budget"]["max_jobs"],
        "max_quantum_seconds": pj["budget"]["max_quantum_seconds"],
    }
    # wrong nonce
    r1 = client.post(f"/api/qpu/{pj['run_id']}/confirm",
                     json={**base, "nonce": "wrong", "consent": qpu_confirm.CONSENT_PHRASE})
    assert r1.status_code == 422
    # missing consent
    r2 = client.post(f"/api/qpu/{pj['run_id']}/confirm",
                     json={**base, "nonce": pj["confirmation"]["nonce"], "consent": "no"})
    assert r2.status_code == 422


def test_backend_mismatch_is_rejected(client):
    body = {**_BODY, "backend_requested": "fake_manila"}   # snapshot names fake_lagos
    r = client.post("/api/qpu/prepare", json=body)
    assert r.status_code == 422


def test_unbounded_budget_is_rejected(client):
    # a 10-year quantum-seconds budget must be refused by the hard cap, not echoed
    body = {**_BODY, "max_quantum_seconds": 315360000.0}
    assert client.post("/api/qpu/dry-run", json=body).status_code == 422
    assert client.post("/api/qpu/prepare", json=body).status_code == 422


def test_status_events_and_cancel(client):
    pj = client.post("/api/qpu/prepare", json=_BODY).json()
    rid = pj["run_id"]
    assert client.get(f"/api/qpu/{rid}").json()["state"] == "prepared"
    ev = client.get(f"/api/qpu/{rid}/events").json()
    # /events is EXHAUSTIVE: total == n_events, ordered, first is the prepared transition
    assert ev["n_events"] >= 1 and ev["total"] == ev["n_events"] and len(ev["events"]) == ev["total"]
    seqs = [e["seq"] for e in ev["events"]]
    assert seqs == sorted(seqs)
    assert ev["events"][0]["event_kind"] == "transition"
    assert ev["events"][0]["payload"]["state"] == "prepared"
    c = client.post(f"/api/qpu/{rid}/cancel")
    assert c.status_code == 200 and c.json()["state"] == "cancelled"


def test_prepare_persists_full_chain(client):
    from acrpq.dashboard import qpu_store
    pj = client.post("/api/qpu/prepare", json=_BODY).json()
    rid, root = pj["run_id"], client.runs_dir
    # the whole chain is durable and reconstructable without any in-memory state
    assert qpu_store.load_decision(rid, root=root)["decision_sha256"] == pj["decision_sha256"]
    assert qpu_store.load_artefact(rid, root=root)["artefact_sha256"] == pj["artefact_sha256"]
    assert qpu_store.load_plan(rid, root=root)["plan_sha256"] == pj["plan_sha256"]
    circ = qpu_store.load_isa_circuit(rid, root=root, as_circuit=True)   # QPY-reloads + verifies
    assert circ.num_qubits >= 1
    report = qpu_store.verify_full_run(rid, root=root)
    assert report["complete"] is True and report["coherent"] is True


def test_prepare_and_confirm_refused_when_qpu_disabled(monkeypatch):
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app
    monkeypatch.delenv("ACRPQ_QPU_ENABLED", raising=False)  # disabled (default)
    c = TestClient(create_app(runs_dir=tempfile.mkdtemp()))
    assert c.post("/api/qpu/prepare", json=_BODY).status_code == 403
    # dry-run stays available even when disabled (safe demo path)
    assert c.post("/api/qpu/dry-run", json=_BODY).status_code == 200


def test_unknown_run_is_404(client):
    assert client.get("/api/qpu/does-not-exist").status_code == 404


def test_mutation_requires_loopback(client, monkeypatch):
    import acrpq.dashboard.qpu_api as qa
    monkeypatch.setattr(qa, "_LOOPBACK_HOSTS", frozenset({"127.0.0.1"}))  # drop 'testclient'
    r = client.post("/api/qpu/dry-run", json=_BODY)
    assert r.status_code == 403


# --------------------------------------------------------------------------- #
# HTTP security (Phase 7.3)
# --------------------------------------------------------------------------- #
class _Job:
    def __init__(self, jid):
        self._id = jid

    def job_id(self):
        return self._id

    def status(self):
        return "DONE"

    def cancel(self):
        return None


class _FakeGateway:
    def __init__(self):
        self.run_calls = 0
        self.result_calls = 0

    def run_sampler(self, *, isa_circuit, shots, tags, max_execution_time):
        self.run_calls += 1
        return _Job("fake-job-1")

    def find_jobs(self, *, tags, created_after_iso):
        return []

    def get_job(self, job_id):
        return _Job(job_id)

    def get_result(self, _job):
        self.result_calls += 1
        return {"counts": {"0" * 6: 512}, "bit_order": "qiskit", "kind": "counts"}


def _qpu_app(monkeypatch, *, enabled=True, submission=False, gateway_factory=None, **reg):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from acrpq.dashboard.qpu_api import register_qpu_routes
    from acrpq.io.loader import InstanceLoader
    monkeypatch.setenv("ACRPQ_QPU_ENABLED", "true" if enabled else "false")
    monkeypatch.setenv("ACRPQ_IBM_SUBMISSION_ENABLED", "true" if submission else "false")
    monkeypatch.setenv("ACRPQ_QPU_DRY_RUN", "false" if submission else "true")
    app = FastAPI()
    rd = tempfile.mkdtemp()
    register_qpu_routes(app, loader=InstanceLoader(), runs_dir=rd,
                        gateway_factory=gateway_factory, **reg)
    tc = TestClient(app)
    tc.runs_dir = rd
    return tc


def test_cross_origin_refused(client):
    r = client.post("/api/qpu/dry-run", json=_BODY, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_bad_host_refused(client):
    r = client.post("/api/qpu/dry-run", json=_BODY, headers={"Host": "evil.example"})
    assert r.status_code == 403


def test_wrong_content_type_refused(client):
    import json as _json
    r = client.post("/api/qpu/dry-run", content=_json.dumps(_BODY),
                    headers={"Content-Type": "text/plain"})
    # refused either by the guard (415) or by FastAPI's JSON body parsing (422)
    assert r.status_code in (415, 422)


def test_body_too_large_refused(monkeypatch):
    c = _qpu_app(monkeypatch, max_body_bytes=10)
    assert c.post("/api/qpu/dry-run", json=_BODY).status_code == 413


def test_rate_limit(monkeypatch):
    c = _qpu_app(monkeypatch, rate_limit=(2, 60.0))
    codes = [c.post("/api/qpu/dry-run", json=_BODY).status_code for _ in range(4)]
    assert 429 in codes                    # burst beyond the window limit is refused


def test_streaming_body_limit_ignores_content_length(monkeypatch):
    # a real body far larger than the cap is refused by the ASGI middleware even
    # when streamed (the check does not trust a possibly-lying Content-Length).
    c = _qpu_app(monkeypatch, max_body_bytes=50)
    big = b'{"x":"' + b"A" * 5000 + b'"}'
    r = c.post("/api/qpu/dry-run", content=big, headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_rate_limit_config_validated(monkeypatch):
    from fastapi import FastAPI

    from acrpq.dashboard.qpu_api import register_qpu_routes
    from acrpq.io.loader import InstanceLoader
    with pytest.raises(ValueError):
        register_qpu_routes(FastAPI(), loader=InstanceLoader(), runs_dir=tempfile.mkdtemp(),
                            rate_limit=(0, 60.0))


# --------------------------------------------------------------------------- #
# submit / reconcile / result / export (Phase 7.2)
# --------------------------------------------------------------------------- #
def test_submit_refused_when_submission_disabled(monkeypatch):
    c = _qpu_app(monkeypatch, enabled=True, submission=False)   # QPU on, submission off
    pj = c.post("/api/qpu/prepare", json=_BODY).json()
    r = c.post(f"/api/qpu/{pj['run_id']}/submit")
    assert r.status_code == 403


def test_real_submit_refuses_fake_artefact_run_sampler_zero(monkeypatch):
    # BYPASS 1 regression: a fake artefact (submittable=false) with all flags on and
    # a real confirmation must be refused by /submit BEFORE any gateway -> run_sampler=0.
    gw = _FakeGateway()
    c = _qpu_app(monkeypatch, enabled=True, submission=True, gateway_factory=lambda b: gw)
    pj = c.post("/api/qpu/prepare", json=_BODY).json()
    assert pj["submittable"] is False
    _confirm(c, pj)
    r = c.post(f"/api/qpu/{pj['run_id']}/submit")
    assert r.status_code == 409 and gw.run_calls == 0


def test_real_submit_refuses_empty_marker_run_sampler_zero(monkeypatch):
    # BYPASS 2 regression: a {} confirmation_consumed marker must not authorise /submit.
    from acrpq.dashboard import qpu_runs, qpu_store
    gw = _FakeGateway()
    c = _qpu_app(monkeypatch, enabled=True, submission=True, gateway_factory=lambda b: gw)
    pj = c.post("/api/qpu/prepare", json=_BODY).json()
    rid = pj["run_id"]
    qpu_runs.transition(rid, qpu_runs.QpuState.AWAITING_CONFIRMATION, root=c.runs_dir)
    m = qpu_store._artefacts_dir(c.runs_dir, rid) / "confirmation_consumed.json"
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text("{}")
    r = c.post(f"/api/qpu/{rid}/submit")
    assert r.status_code == 409 and gw.run_calls == 0


def test_real_submit_refuses_dry_run_safe_gateway(monkeypatch):
    # even if a chain were real-submittable, /submit refuses a dry_run_safe gateway
    gw = _FakeGateway()
    gw.dry_run_safe = True
    c = _qpu_app(monkeypatch, enabled=True, submission=True, gateway_factory=lambda b: gw)
    pj = c.post("/api/qpu/prepare", json=_BODY).json()
    _confirm(c, pj)
    r = c.post(f"/api/qpu/{pj['run_id']}/submit")
    assert r.status_code == 409 and gw.run_calls == 0


def test_simulate_seam_runs_fake_lifecycle(monkeypatch):
    # the fake submit MECHANICS are exercised on the dedicated /simulate seam only
    gw = _FakeGateway()
    gw.dry_run_safe = True
    c = _qpu_app(monkeypatch, enabled=True, submission=True,
                 simulate_gateway_factory=lambda b: gw)
    pj = c.post("/api/qpu/prepare", json=_BODY).json()
    rid = pj["run_id"]
    # /simulate is refused before confirmation
    assert c.post(f"/api/qpu/{rid}/simulate/submit").status_code == 409
    _confirm(c, pj)
    r = c.post(f"/api/qpu/{rid}/simulate/submit")
    assert r.status_code == 200 and r.json()["verdict"] == "submitted" and gw.run_calls == 1


def test_real_reconcile_persists_completed_result(monkeypatch):
    """Regression: real /reconcile must fetch counts after observing DONE."""
    gw = _FakeGateway()
    gw.dry_run_safe = True
    c = _qpu_app(
        monkeypatch, enabled=True, submission=True,
        gateway_factory=lambda _backend: gw,
        simulate_gateway_factory=lambda _backend: gw,
    )
    pj = c.post("/api/qpu/prepare", json=_BODY).json()
    rid = pj["run_id"]
    _confirm(c, pj)
    assert c.post(f"/api/qpu/{rid}/simulate/submit").status_code == 200
    rec = c.post(f"/api/qpu/{rid}/reconcile")
    assert rec.status_code == 200 and rec.json()["state"] == "completed"
    assert gw.result_calls == 1
    result = c.get(f"/api/qpu/{rid}/result").json()
    assert result["available"] is True and result["summary"]["n_shots"] == 512


def test_simulate_seam_disabled_without_factory(monkeypatch):
    c = _qpu_app(monkeypatch, enabled=True, submission=True)   # no simulate factory
    pj = c.post("/api/qpu/prepare", json=_BODY).json()
    _confirm(c, pj)
    assert c.post(f"/api/qpu/{pj['run_id']}/simulate/submit").status_code == 404


def test_events_exhaustive_after_lifecycle(monkeypatch):
    gw = _FakeGateway()
    gw.dry_run_safe = True
    c = _qpu_app(monkeypatch, enabled=True, submission=True, simulate_gateway_factory=lambda b: gw)
    pj = c.post("/api/qpu/prepare", json=_BODY).json()
    rid = pj["run_id"]
    _confirm(c, pj)
    c.post(f"/api/qpu/{rid}/simulate/submit")          # more events: transitions + job_id
    ev = c.get(f"/api/qpu/{rid}/events").json()
    assert ev["total"] == ev["n_events"] and len(ev["events"]) == ev["total"]
    kinds = {e["event_kind"] for e in ev["events"]}
    assert "transition" in kinds and "job_id" in kinds     # not just prepared transitions
    # pagination: after_seq drops earlier events
    last = ev["events"][-1]["seq"]
    assert c.get(f"/api/qpu/{rid}/events?after_seq={last}").json()["events"] == []


@pytest.mark.parametrize("bad", [
    {"shots": True}, {"shots": "512"}, {"n_theta": 2.0}, {"max_quantum_seconds": "x"},
    {"transpiler_seed": True}, {"nonsense_key": 1},
])
def test_prepare_rejects_coercions_and_unknown_keys(client, bad):
    assert client.post("/api/qpu/dry-run", json={**_BODY, **bad}).status_code == 422


def test_full_qaoa_dry_run_multi_job(client):
    body = {"instance": "CP_3", "n_theta": 2, "shots": 256, "transpiler_seed": 1,
            "backend_requested": "fake_lagos", "snapshots": [_SNAP], "strategy": "deterministic_sweep",
            "reps": 1, "max_quantum_seconds": 60.0, "total_shots_cap": 10_000}
    r = client.post("/api/qpu/full-qaoa/dry-run", json=body)
    assert r.status_code == 200
    j = r.json()
    assert j["protocol"] == "full_qaoa_hardware" and len(j["jobs"]) >= 2   # multiple angle points
    assert j["submittable"] is False                                       # fake -> dry-run only
    assert j["full_qaoa_submission_enabled"] is False                      # own flag, default off
    tags = [job["local_tag"] for job in j["jobs"]]
    assert len(set(tags)) == len(tags)                                     # unique tags


def test_concurrent_simulate_submit_exactly_one(monkeypatch):
    # Item 11: two concurrent submits of the same run -> exactly ONE run_sampler call
    # (the append-only transition-to-submitting is the inter-process/thread mutex).
    import threading
    gw = _FakeGateway()
    gw.dry_run_safe = True
    c = _qpu_app(monkeypatch, enabled=True, submission=True, simulate_gateway_factory=lambda b: gw)
    pj = c.post("/api/qpu/prepare", json=_BODY).json()
    rid = pj["run_id"]
    _confirm(c, pj)
    codes = []
    errors = []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        try:
            code = c.post(f"/api/qpu/{rid}/simulate/submit").status_code
            with lock:
                codes.append(code)
        except BaseException as exc:  # test must surface every worker failure
            with lock:
                errors.append(exc)

    ts = [threading.Thread(target=worker) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert errors == []
    assert len(codes) == 8
    assert gw.run_calls == 1                 # exactly one submission attempt
    assert codes.count(200) == 1             # exactly one winner
    assert codes.count(409) == 7             # every loser receives a controlled response


def test_result_unavailable_then_decoded(client):
    from acrpq.dashboard import qpu_store
    pj = client.post("/api/qpu/prepare", json=_BODY).json()
    rid = pj["run_id"]
    assert client.get(f"/api/qpu/{rid}/result").json()["available"] is False
    # persist a synthetic raw result -> /result decodes it via the Phase-5 decoder
    from acrpq.quantum.qubo import build_qubo
    from acrpq.io.loader import InstanceLoader
    q = build_qubo(InstanceLoader().load("CP_3"), n_theta=2)
    n = q.n_qubits
    qpu_store.save_result_raw(rid, {"counts": {"0" * n: 512}, "bit_order": "qubo",
                                    "kind": "counts"}, root=client.runs_dir)
    res = client.get(f"/api/qpu/{rid}/result").json()
    assert res["available"] is True and res["summary"]["n_unique_total"] == 1


def test_export_auto_loads_chain(client):
    from acrpq.dashboard import qpu_export
    pj = client.post("/api/qpu/prepare", json=_BODY).json()
    rid = pj["run_id"]
    export = client.post(f"/api/qpu/{rid}/export").json()
    assert qpu_export.verify_export(export) is True
    assert export["decision"] is not None and export["artefact"] is not None
    assert export["plan"] is not None


def _confirm(c, pj):
    conf = pj["confirmation"]
    good = {"nonce": conf["nonce"], "config_hash": pj["config_hash"],
            "artefact_sha256": pj["artefact_sha256"], "decision_sha256": pj["decision_sha256"],
            "backend": pj["backend"], "total_shots": pj["budget"]["total_shots"],
            "max_jobs": pj["budget"]["max_jobs"],
            "max_quantum_seconds": pj["budget"]["max_quantum_seconds"],
            "consent": qpu_confirm.CONSENT_PHRASE}
    return c.post(f"/api/qpu/{pj['run_id']}/confirm", json=good)
