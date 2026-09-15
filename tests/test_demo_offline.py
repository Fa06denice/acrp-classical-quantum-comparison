"""Phase 7 — the offline demo mode must be deterministic and refuse an unsafe machine."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_P = Path(__file__).resolve().parent.parent / "scripts" / "demo_offline.py"
_spec = importlib.util.spec_from_file_location("demo_offline", _P)
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


def test_environment_check_passes_offline():
    assert demo.check_environment() == 0


def test_frozen_artifacts_verify():
    assert demo.check_artifacts() == 0


def test_api_smoke_passes_and_submission_stays_refused():
    assert demo.smoke() == 0


def test_the_smoke_refusal_is_the_flag_guard_not_a_missing_run():
    """A nonexistent run id answers 404 naturally, so accepting 404 would let the smoke
    test pass with the feature-flag guard deleted. The observed refusal must be the
    flags' 403, raised before the run is even looked up."""
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app

    r = TestClient(create_app()).post("/api/qpu/nonexistent-run/submit", json={})
    assert r.status_code == 403, r.text
    assert "flag" in r.text.lower() or "disabl" in r.text.lower()


def test_smoke_fails_if_submit_answers_anything_other_than_403(monkeypatch):
    """Neuter-check on the smoke test itself: make /submit answer 404 and smoke() must
    refuse with exit 3. Without the `!= 403` pin this test fails."""
    import fastapi.testclient as tcmod
    real = tcmod.TestClient

    class _Client:
        def __init__(self, app):
            self._c = real(app)

        def get(self, ep):
            return self._c.get(ep)

        def post(self, ep, **kw):
            if ep.endswith("/submit"):
                class _R:
                    status_code = 404
                return _R()
            return self._c.post(ep, **kw)

    monkeypatch.setattr(tcmod, "TestClient", _Client)
    assert demo.smoke() == 3


def test_refuses_to_start_when_submission_could_be_enabled(monkeypatch):
    """Fail-closed: a machine configured for real submission must be refused (exit 4)."""
    for var in demo.UNSAFE_ENV:
        monkeypatch.setenv(var, "true")
        assert demo.check_environment() == 4
        monkeypatch.delenv(var)


def test_the_four_provenance_classes_are_distinct():
    """Live / recorded-local / Aer / recorded-IBM must never be conflated."""
    keys = set(demo.PROVENANCE_CLASSES)
    assert keys == {"live_local", "recorded_local", "aer_simulation", "recorded_ibm_hardware"}
    assert len(set(demo.PROVENANCE_CLASSES.values())) == 4
    assert "aucune soumission" in demo.PROVENANCE_CLASSES["recorded_ibm_hardware"]
