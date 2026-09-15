"""Phase 5.5 — offline operator CLI: refusals, plan, verify, no-PII (no IBM)."""
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")

_CLI_PATH = Path(__file__).resolve().parent.parent / "scripts" / "qpu_batch_cli.py"
_spec = importlib.util.spec_from_file_location("qpu_batch_cli", _CLI_PATH)
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def _run(argv, capsys):
    rc = cli.main(argv)
    out = capsys.readouterr()
    return rc, out.out, out.err


# --- hard refusals --------------------------------------------------------
@pytest.mark.parametrize("flag", ["--real", "--token", "--submit", "--api-token",
                                  "--real=1", "--submit=1", "--api-token=x", "--ibm-token=x"])
def test_forbidden_flags_refused(flag, capsys):
    rc, _out, err = _run(["plan", flag], capsys)
    assert rc == 2 and "REFUSED" in err                  # incl. --flag=value forms (red-team NIT1)


def test_no_working_submit_subcommand(capsys):
    with pytest.raises(SystemExit):                      # argparse rejects an unknown subcommand
        _run(["submit"], capsys)


def test_non_fake_backend_refused(capsys):
    rc, _out, err = _run(["plan", "--backend", "ibm_marrakesh"], capsys)
    assert rc == 2 and "REFUSED" in err


def test_refuses_when_not_offline(capsys, monkeypatch):
    from acrpq.dashboard import qpu_batch

    def _boom(**_k):
        raise qpu_batch.OfflineSafetyError("submission could be enabled")
    monkeypatch.setattr(qpu_batch, "assert_offline", _boom)
    rc, _out, err = _run(["plan"], capsys)
    assert rc == 4 and "not offline" in err


# --- plan -----------------------------------------------------------------
def test_json_flag_works_before_or_after_subcommand(capsys):
    for argv in (["--json", "plan", "--instances", "CP_3"], ["plan", "--instances", "CP_3", "--json"]):
        rc, out, _err = _run(argv, capsys)
        assert rc == 0 and json.loads(out)["command"] == "plan"   # valid JSON either way


def test_plan_json_is_valid_and_offline(capsys):
    rc, out, _err = _run(["--json", "plan", "--instances", "CP_3", "--objectives",
                          "maneuver_count_v1", "--n-theta", "3", "--backend", "FakeGuadalupeV2"],
                         capsys)
    assert rc == 0
    payload = json.loads(out)
    assert payload["n_jobs"] == 1 and payload["qubits_per_job"] == [9]
    assert payload["status"] == "OFFLINE — NO SUBMISSION"
    assert any("FakeGuadalupe" in b for b in payload["candidate_local_backends"])
    assert "/Users/" not in out and "token" not in out.lower()


# --- verify ---------------------------------------------------------------
def test_verify_ok_and_tampered(tmp_path, capsys):
    from acrpq.dashboard import qpu_batch
    batch = {"schema": "x", "results": [], "n_jobs": 0}
    batch["batch_sha256"] = qpu_batch.canonical_batch_sha256(batch)
    good = tmp_path / "good.json"
    good.write_text(json.dumps(batch))
    rc, _out, _err = _run(["verify", str(good)], capsys)
    assert rc == 0
    batch["n_jobs"] = 999                                # tamper without updating the hash
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(batch))
    rc, _out, _err = _run(["verify", str(bad)], capsys)
    assert rc == 3                                       # verification failed
    rc, _out, _err = _run(["verify", str(tmp_path / "missing.json")], capsys)
    assert rc == 5


# --- inspect / resume-check ----------------------------------------------
def test_export_produces_validated_versioned_document(tmp_path, capsys):
    from acrpq.dashboard import qpu_export_schema as X
    src = "results/qpu_batch_offline_v1/batch.json"       # existing offline artefact
    out = tmp_path / "export.json"
    rc, out_s, _err = _run(["--json", "export", src, "--out", str(out)], capsys)
    assert rc == 0
    payload = json.loads(out_s)
    assert payload["schema"] == X.EXPORT_SCHEMA and payload["validated"] is True
    doc = json.loads(out.read_text())
    X.validate_export(doc, require_synthetic_offline=True, strict=True)   # end-to-end valid + strict
    assert (tmp_path / "export.csv").exists()
    # re-export identical content to the same --out is fine (same hash)
    rc2, _o, _e = _run(["--json", "export", src, "--out", str(out)], capsys)
    assert rc2 == 0


def test_export_refuses_incompatible_overwrite(tmp_path, capsys):
    src = "results/qpu_batch_offline_v1/batch.json"
    out = tmp_path / "export.json"
    out.write_text(json.dumps({"export_sha256": "sha256:" + "0" * 64}))   # a DIFFERENT export
    rc, _o, err = _run(["export", src, "--out", str(out)], capsys)
    assert rc == 3 and "REFUSING to overwrite" in err


def _write_fixture(tmp_path, obj):
    fx = tmp_path / "fx.json"
    fx.write_text(json.dumps(obj))
    return str(fx)


def test_reconcile_simulated_valid_fixture(tmp_path, capsys):
    fx = _write_fixture(tmp_path, {"counts": {"000": 6, "111": 2}, "bit_order": "qubo",
                                   "kind": "counts", "n_bits": 3})
    rc, out, _err = _run(["--json", "reconcile-simulated", "--fixture", fx], capsys)
    assert rc == 0
    payload = json.loads(out)
    assert payload["reconciled"] is True and payload["n_shots"] == 8
    assert "FakeBatchGateway" in payload["gateway"]


@pytest.mark.parametrize("bad", [
    {"counts": {"000": -1}, "bit_order": "qubo", "kind": "counts", "n_bits": 3},   # negative
    {"counts": {}, "bit_order": "qubo", "kind": "counts", "n_bits": 3},            # empty
    {"nope": 1},                                                                    # missing keys
    {"counts": {"00": 1}, "bit_order": "qubo", "kind": "counts", "n_bits": 3},     # wrong length
])
def test_reconcile_simulated_bad_fixture_fails_closed(tmp_path, bad, capsys):
    rc, _o, _err = _run(["reconcile-simulated", "--fixture", _write_fixture(tmp_path, bad)], capsys)
    assert rc == 3


@pytest.mark.parametrize("leak", [
    "ghp_ABCDEFGHIJKLMNOP0000", "gho_ABCDEFGHIJKLMNOP0000", "AKIAABCDEFGHIJKLMNOP",
    "-----BEGIN RSA PRIVATE KEY-----", "eyJhbGciOiJIUzI1NiIs.eyJzdWIiOiIxMjM0NTY",
    "/etc/passwd", "/private/tmp/secret",
])
def test_reconcile_simulated_secret_or_path_fixture_refused(tmp_path, leak, capsys):
    """Red-team #2: the reconcile fixture scan is now as strict as the canonical scanner."""
    fx = _write_fixture(tmp_path, {"counts": {"000": 1}, "bit_order": "qubo", "kind": "counts",
                                   "n_bits": 3, "note": f"leak {leak}"})
    rc, _o, err = _run(["reconcile-simulated", "--fixture", fx], capsys)
    assert rc == 3 and "REFUSED" in err


def test_export_refuses_overwriting_a_non_export_file(tmp_path, capsys):
    """Red-team NIT2: an existing file that is not a valid export (no export_sha256) is not
    silently overwritten."""
    src = "results/qpu_batch_offline_v1/batch.json"
    out = tmp_path / "export.json"
    out.write_text(json.dumps({"unrelated": "file"}))    # no export_sha256
    rc, _o, err = _run(["export", src, "--out", str(out)], capsys)
    assert rc == 3 and "REFUSING to overwrite" in err


def test_inspect_and_resume_check(tmp_path, capsys):
    from acrpq.dashboard import qpu_runs
    runs = str(tmp_path / "runs")
    params = {"mode": "hardware_validation", "instance": "CP_3", "n_theta": 3, "n_q": 1,
              "shots": 128, "transpiler_seed": 1, "max_jobs": 1, "max_quantum_seconds": 20.0,
              "qubo_sha256": "sha256:" + "c" * 64, "backend_requested": "fake"}
    rid = qpu_runs.prepare(params, root=runs)
    qpu_runs.transition(rid, qpu_runs.QpuState.AWAITING_CONFIRMATION, root=runs)
    rc, out, _err = _run(["--json", "inspect", "--runs-dir", runs], capsys)
    assert rc == 0 and json.loads(out)["n_runs"] == 1
    rc, out, _err = _run(["--json", "resume-check", "--runs-dir", runs], capsys)
    assert rc == 0
    payload = json.loads(out)
    assert len(payload["interrupted_non_terminal"]) == 1   # the awaiting run is interrupted
