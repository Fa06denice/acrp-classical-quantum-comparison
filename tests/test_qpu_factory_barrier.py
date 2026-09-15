"""Phase 5.2.E — the real IBM factory is structurally inaccessible from the offline path.

Proves, by construction and by monkeypatch-to-raise, that no offline command can build a
real QiskitRuntimeService or call the real gateway factory: the offline modules never
import the real path; running the whole offline pipeline with the real factory AND the
service class rigged to raise still completes; and the seam stays closed by default and
needs several independent conditions to open.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from acrpq.dashboard import qpu_batch, qpu_ibm_factory
from acrpq.io.loader import InstanceLoader


@pytest.fixture(scope="module")
def ld():
    return InstanceLoader()


# --- structural: the offline modules never import the real path -----------
def test_offline_modules_do_not_import_real_service_or_api():
    """No offline module IMPORTS the real service/API or CONSTRUCTS a service (docstring
    mentions of the fenced-off names are fine; imports/constructions are not)."""
    import inspect
    for mod in ("qpu_batch", "qpu_claim", "qpu_dedup"):
        src = inspect.getsource(__import__(f"acrpq.dashboard.{mod}", fromlist=[mod]))
        assert "QiskitRuntimeService(" not in src, f"{mod} constructs a real service"
        assert "import QiskitRuntimeService" not in src
        assert "import qpu_api" not in src and "from .qpu_api" not in src
        assert "import backends" not in src and "quantum.backends" not in src


def test_import_of_qpu_batch_constructs_no_service():
    """Importing the offline driver in a fresh interpreter must not construct a
    QiskitRuntimeService (we rig the class to fail on construction)."""
    code = (
        "import qiskit_ibm_runtime as q\n"
        "class Boom:\n"
        "    def __init__(self,*a,**k): raise AssertionError('service constructed on import')\n"
        "q.QiskitRuntimeService = Boom\n"
        "import acrpq.dashboard.qpu_batch  # noqa\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


# --- monkeypatch-to-raise: the offline pipeline never calls the real factory
def test_offline_pipeline_never_calls_real_factory_or_service(ld, tmp_path, monkeypatch):
    import qiskit_ibm_runtime

    def _boom_service(*a, **k):
        raise AssertionError("real QiskitRuntimeService was constructed by an offline run")

    def _boom_factory(*a, **k):
        raise AssertionError("real gateway factory was called by an offline run")

    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", _boom_service)
    monkeypatch.setattr(qpu_ibm_factory._DefaultServiceFactory, "make_service", _boom_factory)
    monkeypatch.setattr(qpu_ibm_factory.RealIbmGatewayFactory, "__call__", _boom_factory)

    store = qpu_batch.BatchJobStore(str(tmp_path / "store"))
    rec = qpu_batch.run_offline_qpu_job(
        ld, "CP_3", objective_id="maneuver_count_v1", n_theta=3, shots=128,
        backend_name="FakeGuadalupeV2", runs_dir=str(tmp_path / "runs"), store=store)
    assert rec["final_state"] == "completed" and rec["no_ibm"] is True


# --- the seam is closed by default and needs independent conditions -------
def test_factory_disabled_by_default_and_fails_closed():
    assert qpu_ibm_factory.runtime_factory_enabled(env=lambda _k: None) is False
    factory = qpu_ibm_factory.RealIbmGatewayFactory(env=lambda _k: None)
    assert factory.enabled() is False
    with pytest.raises(qpu_ibm_factory.GatewayNotEnabled):
        factory("some_backend")                          # disabled -> refuse before any build


def test_factory_flag_alone_builds_nothing_without_a_call():
    """Even with the flag on, constructing the factory builds no service; only a call
    would — and a call uses the injected (fake) service factory in tests."""
    calls = []

    class FakeSvcFactory:
        def make_service(self):
            calls.append("made")
            return object()

    factory = qpu_ibm_factory.RealIbmGatewayFactory(
        env=lambda k: "true" if k == "ACRPQ_IBM_RUNTIME_FACTORY_ENABLED" else None,
        service_factory=FakeSvcFactory())
    assert factory.enabled() is True and calls == []     # constructing built nothing


def test_assert_offline_refuses_if_factory_enabled():
    """The offline CLI's first fence refuses to run at all if the real factory flag is on."""
    env = {"ACRPQ_IBM_RUNTIME_FACTORY_ENABLED": "true"}.get
    with pytest.raises(qpu_batch.OfflineSafetyError):
        qpu_batch.assert_offline(env=env)


def test_assert_offline_refuses_if_submission_allowed():
    env = {"ACRPQ_QPU_ENABLED": "true", "ACRPQ_IBM_SUBMISSION_ENABLED": "true",
           "ACRPQ_QPU_DRY_RUN": "false"}.get
    with pytest.raises(qpu_batch.OfflineSafetyError):
        qpu_batch.assert_offline(env=env)
