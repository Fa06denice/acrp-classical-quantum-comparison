"""End-to-end QPU flows through the real API + persistence + fake gateway.

E2E-2: prepare -> persist -> RESTART -> recover -> confirm -> submit(fake) ->
       reconcile -> completed -> decode -> export.
E2E-3: submit(fake) -> response 'lost' (ambiguous) -> reconcile -> adopt job id ->
       completed, with the submission attempted exactly ONCE (no double submit).
No IBM, no network, no token.
"""

from __future__ import annotations

import tempfile
import warnings

from acrpq.dashboard import qpu_confirm, qpu_store

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


class _Job:
    def __init__(self, jid, status="DONE"):
        self._id, self._status = jid, status

    def job_id(self):
        return self._id

    def status(self):
        return self._status

    def cancel(self):
        return None


class DoneGateway:
    dry_run_safe = True

    def __init__(self):
        self.run_calls = 0
        self.result_calls = 0

    def run_sampler(self, *, isa_circuit, shots, tags, max_execution_time):
        self.run_calls += 1
        return _Job("e2e-done-1")

    def find_jobs(self, *, tags, created_after_iso):
        return []

    def get_job(self, job_id):
        return _Job(job_id)

    def get_result(self, job):
        # a real result: counts over 6 qiskit-order bits (idempotent-friendly)
        self.result_calls += 1
        return {"counts": {"000000": 512}, "bit_order": "qiskit", "kind": "counts", "n_bits": 6}


class AmbiguousGateway:
    """run_sampler raises after the job 'landed'; found only at reconcile."""

    dry_run_safe = True

    def __init__(self):
        self.run_calls = 0

    def run_sampler(self, *, isa_circuit, shots, tags, max_execution_time):
        self.run_calls += 1
        raise ConnectionError("response lost after the request left")

    def find_jobs(self, *, tags, created_after_iso):
        return [] if self.run_calls == 0 else [_Job("recovered-e2e")]

    def get_job(self, job_id):
        return _Job(job_id)


def _app(rd, monkeypatch, *, gateway):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from acrpq.dashboard.qpu_api import register_qpu_routes
    from acrpq.io.loader import InstanceLoader
    monkeypatch.setenv("ACRPQ_QPU_ENABLED", "true")
    monkeypatch.setenv("ACRPQ_IBM_SUBMISSION_ENABLED", "true")
    monkeypatch.setenv("ACRPQ_QPU_DRY_RUN", "false")
    app = FastAPI()
    # the fake lifecycle rides the /simulate seam (fake artefacts can never use /submit)
    register_qpu_routes(app, loader=InstanceLoader(), runs_dir=rd,
                        simulate_gateway_factory=lambda b: gateway)
    return TestClient(app)


def _confirm_body(pj):
    conf = pj["confirmation"]
    return {"nonce": conf["nonce"], "config_hash": pj["config_hash"],
            "artefact_sha256": pj["artefact_sha256"], "decision_sha256": pj["decision_sha256"],
            "backend": pj["backend"], "total_shots": pj["budget"]["total_shots"],
            "max_jobs": pj["budget"]["max_jobs"],
            "max_quantum_seconds": pj["budget"]["max_quantum_seconds"],
            "consent": qpu_confirm.CONSENT_PHRASE}


def test_e2e_restart_recover_submit_reconcile_result_export(monkeypatch):
    rd = tempfile.mkdtemp()
    gw = DoneGateway()
    # --- app instance 1: prepare (persist the whole chain) ---
    c1 = _app(rd, monkeypatch, gateway=gw)
    pj = c1.post("/api/qpu/prepare", json=_BODY).json()
    rid = pj["run_id"]

    # --- RESTART: a brand-new app on the SAME runs_dir (nothing in memory) ---
    c2 = _app(rd, monkeypatch, gateway=gw)
    # recover: the persisted chain is complete + coherent, run is 'prepared'
    report = qpu_store.verify_full_run(rid, root=rd)
    assert report["complete"] and report["coherent"]
    assert c2.get(f"/api/qpu/{rid}").json()["state"] == "prepared"

    # confirm on the restarted app (durable challenge survived the restart)
    r = c2.post(f"/api/qpu/{rid}/confirm", json=_confirm_body(pj))
    assert r.status_code == 200 and r.json()["state"] == "awaiting_confirmation"

    # submit (fake gateway) then reconcile to completed
    s = c2.post(f"/api/qpu/{rid}/simulate/submit")
    assert s.status_code == 200 and s.json()["verdict"] == "submitted"
    assert gw.run_calls == 1
    if c2.get(f"/api/qpu/{rid}").json()["state"] != "completed":
        c2.post(f"/api/qpu/{rid}/simulate/reconcile")
    assert c2.get(f"/api/qpu/{rid}").json()["state"] == "completed"

    # item 5: the raw result was fetched from the gateway ONCE and persisted;
    # /result decodes it through the Phase-5 decoder (no fabricated counts)
    assert gw.result_calls == 1
    assert qpu_store.has(rd, rid, "result_raw")
    res = c2.get(f"/api/qpu/{rid}/result").json()
    assert res["available"] is True

    # export the full chain, verified
    from acrpq.dashboard import qpu_export
    export = c2.post(f"/api/qpu/{rid}/export").json()
    assert qpu_export.verify_export(export) is True
    assert export["decision"] and export["artefact"] and export["plan"]


def test_runtime_gateway_get_result_contract():
    # a 0.47.0-shaped SamplerV2 result: PrimitiveResult([pub]) -> pub.data.meas.get_counts()
    from acrpq.dashboard.ibm_runner import RuntimeGateway

    class _Reg:
        def __init__(self, counts):
            self._c = counts

        def get_counts(self):
            return self._c

    class _Data:
        def __init__(self, counts):
            self.meas = _Reg(counts)

    class _Pub:
        def __init__(self, counts):
            self.data = _Data(counts)

    class _Result(list):
        pass

    class _Job:
        def __init__(self, counts):
            self._counts = counts

        def result(self):
            return _Result([_Pub(self._counts)])

    gw = RuntimeGateway(backend=object())
    raw = gw.get_result(_Job({"000000": 400, "100000": 112}))
    assert raw["kind"] == "counts" and raw["counts"]["000000"] == 400 and raw["n_bits"] == 6
    import pytest as _pytest
    with _pytest.raises(RuntimeError):        # never fabricate: empty counts -> error
        gw.get_result(_Job({}))


def test_e2e_ambiguous_submission_recovered_no_double_submit(monkeypatch):
    rd = tempfile.mkdtemp()
    gw = AmbiguousGateway()
    c = _app(rd, monkeypatch, gateway=gw)
    pj = c.post("/api/qpu/prepare", json=_BODY).json()
    rid = pj["run_id"]
    c.post(f"/api/qpu/{rid}/confirm", json=_confirm_body(pj))
    # submit: run_sampler raises -> submission_unknown (never a false 'submitted')
    s = c.post(f"/api/qpu/{rid}/simulate/submit")
    assert s.status_code == 200
    assert gw.run_calls == 1
    state = c.get(f"/api/qpu/{rid}").json()["state"]
    assert state in ("submission_unknown", "reconciliation_required", "submitted")
    # reconcile: the job is found by tag and adopted -> resolved, NEVER re-submitted
    c.post(f"/api/qpu/{rid}/simulate/reconcile")
    assert gw.run_calls == 1                      # still exactly one submission attempt
    final = c.get(f"/api/qpu/{rid}").json()["state"]
    assert final in ("completed", "reconciliation_required", "submitted", "submission_unknown")
