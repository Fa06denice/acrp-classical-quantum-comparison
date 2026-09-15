"""Phase 9 tests: verified atomic exports and read-only startup recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from acrpq.dashboard import qpu_export, qpu_runs

_SHA = "sha256:" + "a" * 64


def _params(instance="CP_3"):
    return {
        "mode": "hardware_validation", "instance": instance, "n_theta": 2, "n_q": 1,
        "shots": 512, "transpiler_seed": 1, "max_jobs": 1, "max_quantum_seconds": 20.0,
        "qubo_sha256": _SHA, "backend_requested": "fake_lagos",
    }


def _prepared_run(root):
    return qpu_runs.prepare(_params(), root=root)


def test_export_roundtrip_and_tamper(tmp_path):
    root = str(tmp_path / "runs")
    rid = _prepared_run(root)
    run = qpu_runs.load(rid, root=root)
    export = qpu_export.build_export(run, decision={"selected": "fake_lagos"},
                                     provenance={"tool": "acrpq"})
    assert qpu_export.verify_export(export) is True
    assert export["run"]["config_hash"] == run["config_hash"]
    tampered = dict(export)
    tampered["run"] = {**export["run"], "state": "completed"}
    assert qpu_export.verify_export(tampered) is False


def test_write_export_is_atomic_and_verifiable(tmp_path):
    root = str(tmp_path / "runs")
    rid = _prepared_run(root)
    dest = str(tmp_path / "export.json")
    qpu_export.export_run(rid, root=root, dest_path=dest, decision={"selected": "fake_lagos"})
    on_disk = json.loads(Path(dest).read_text())
    assert qpu_export.verify_export(on_disk) is True
    # refuses to overwrite an existing export
    with pytest.raises(Exception):
        qpu_export.export_run(rid, root=root, dest_path=dest)


def test_export_refuses_secret_like_field(tmp_path):
    root = str(tmp_path / "runs")
    rid = _prepared_run(root)
    run = qpu_runs.load(rid, root=root)
    with pytest.raises(ValueError):
        qpu_export.build_export(run, provenance={"ibm_token": "shh"})


def test_export_refuses_non_native_value(tmp_path):
    root = str(tmp_path / "runs")
    rid = _prepared_run(root)
    run = qpu_runs.load(rid, root=root)
    with pytest.raises(ValueError):
        qpu_export.build_export(run, decision={"weights": {1, 2, 3}})  # a set is not native


def test_export_includes_run_lifecycle(tmp_path):
    # regression for the red-team finding: the export must carry the run's actual
    # transitions/result, not an always-empty "events" list.
    root = str(tmp_path / "runs")
    rid = _prepared_run(root)
    qpu_runs.transition(rid, qpu_runs.QpuState.AWAITING_CONFIRMATION, root=root)
    qpu_runs.transition(rid, qpu_runs.QpuState.SUBMITTING, root=root)
    qpu_runs.transition(rid, qpu_runs.QpuState.SUBMITTED, root=root)
    qpu_runs.transition(rid, qpu_runs.QpuState.COMPLETED, root=root)
    run = qpu_runs.load(rid, root=root)
    export = qpu_export.build_export(run)
    assert len(export["run"]["transitions"]) == len(run["transitions"]) >= 4
    assert export["run"]["state"] == "completed"
    assert qpu_export.verify_export(export) is True


def test_startup_recovery_classifies_without_submitting(tmp_path):
    root = str(tmp_path / "runs")
    rid = _prepared_run(root)
    report = qpu_export.startup_recovery_report(root=root)
    assert rid in report["non_terminal"]
    assert rid not in report["terminal"]
    assert "never auto-submits" in report["policy"]
    # a completed run moves to terminal
    qpu_runs.transition(rid, qpu_runs.QpuState.AWAITING_CONFIRMATION, root=root)
    qpu_runs.transition(rid, qpu_runs.QpuState.SUBMITTING, root=root)
    qpu_runs.transition(rid, qpu_runs.QpuState.SUBMITTED, root=root)
    qpu_runs.transition(rid, qpu_runs.QpuState.COMPLETED, root=root)
    report2 = qpu_export.startup_recovery_report(root=root)
    assert rid in report2["terminal"] and rid not in report2["non_terminal"]


def test_import_is_qiskit_free():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, acrpq.dashboard.qpu_export as m; "
         "print(any(k.startswith('qiskit') for k in sys.modules))"],
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
