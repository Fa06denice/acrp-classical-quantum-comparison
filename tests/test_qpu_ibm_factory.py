"""Item 4 tests: the real IBM gateway factory is wired but never executed.

A fake service factory reproduces the 0.47.0 surface; no test builds a real
service. Proves the flag gate + that a fake artefact never triggers construction.
"""

from __future__ import annotations

import tempfile
import warnings

import pytest

from acrpq.dashboard import qpu_confirm, qpu_ibm_factory

warnings.filterwarnings("ignore")


class _FakeService:
    """A minimal stand-in for QiskitRuntimeService (0.47.0-shaped)."""

    def backend(self, name):
        return object()

    def jobs(self, **kw):
        return []


class SpyServiceFactory:
    def __init__(self):
        self.calls = 0

    def make_service(self):
        self.calls += 1
        return _FakeService()


def test_factory_disabled_by_default():
    f = qpu_ibm_factory.RealIbmGatewayFactory(env={}.get, service_factory=SpyServiceFactory())
    assert f.enabled() is False
    with pytest.raises(qpu_ibm_factory.GatewayNotEnabled):
        f("ibm_real")


def test_factory_builds_gateway_only_when_enabled():
    spy = SpyServiceFactory()
    env = {"ACRPQ_IBM_RUNTIME_FACTORY_ENABLED": "true"}.get
    f = qpu_ibm_factory.RealIbmGatewayFactory(env=env, service_factory=spy)
    gw = f("ibm_real")
    from acrpq.dashboard.ibm_runner import RuntimeGateway
    assert isinstance(gw, RuntimeGateway)      # real gateway, but over a FAKE service
    assert spy.calls == 1                       # service built exactly once, lazily


def test_factory_refuses_empty_backend():
    env = {"ACRPQ_IBM_RUNTIME_FACTORY_ENABLED": "true"}.get
    f = qpu_ibm_factory.RealIbmGatewayFactory(env=env, service_factory=SpyServiceFactory())
    with pytest.raises(qpu_ibm_factory.GatewayNotEnabled):
        f("   ")


def test_import_is_qiskit_free():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, acrpq.dashboard.qpu_ibm_factory as m; "
         "print(any(k.startswith('qiskit') for k in sys.modules))"],
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def _qpu_app(monkeypatch, *, gateway_factory):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from acrpq.dashboard.qpu_api import register_qpu_routes
    from acrpq.io.loader import InstanceLoader
    for k, v in (("ACRPQ_QPU_ENABLED", "true"), ("ACRPQ_IBM_SUBMISSION_ENABLED", "true"),
                 ("ACRPQ_QPU_DRY_RUN", "false")):
        monkeypatch.setenv(k, v)
    app = FastAPI()
    rd = tempfile.mkdtemp()
    register_qpu_routes(app, loader=InstanceLoader(), runs_dir=rd, gateway_factory=gateway_factory)
    c = TestClient(app)
    c.runs_dir = rd
    return c


_SNAP = {"name": "fake_lagos", "region": "us-east", "family": "F", "physical_qubits": 7,
         "programmable_qubits": 7, "operational": True, "maintenance": False, "pending_jobs": 3,
         "two_qubit_error": 0.006, "readout_error": 0.012, "clops": 2500.0,
         "avg_coupling_degree": 2.0, "data_ts": 0.0}
_BODY = {"instance": "CP_3", "n_theta": 2, "shots": 512, "transpiler_seed": 1,
         "backend_requested": "fake_lagos", "gammas": [0.4], "betas": [0.3], "max_jobs": 1,
         "max_quantum_seconds": 20.0, "optimization_level": 0, "snapshots": [_SNAP]}


def test_fake_artefact_never_constructs_the_service(monkeypatch):
    # /submit on a fake artefact must be refused BEFORE the gateway factory is called,
    # so the (spied) service is NEVER constructed.
    spy = SpyServiceFactory()
    env = {"ACRPQ_IBM_RUNTIME_FACTORY_ENABLED": "true"}.get
    factory = qpu_ibm_factory.RealIbmGatewayFactory(env=env, service_factory=spy)
    c = _qpu_app(monkeypatch, gateway_factory=factory)
    pj = c.post("/api/qpu/prepare", json=_BODY).json()
    conf = pj["confirmation"]
    c.post(f"/api/qpu/{pj['run_id']}/confirm", json={
        "nonce": conf["nonce"], "config_hash": pj["config_hash"],
        "artefact_sha256": pj["artefact_sha256"], "decision_sha256": pj["decision_sha256"],
        "backend": pj["backend"], "total_shots": pj["budget"]["total_shots"],
        "max_jobs": pj["budget"]["max_jobs"],
        "max_quantum_seconds": pj["budget"]["max_quantum_seconds"],
        "consent": qpu_confirm.CONSENT_PHRASE})
    r = c.post(f"/api/qpu/{pj['run_id']}/submit")
    assert r.status_code == 409          # fake -> refused before any gateway
    assert spy.calls == 0                # the IBM service was NEVER constructed
