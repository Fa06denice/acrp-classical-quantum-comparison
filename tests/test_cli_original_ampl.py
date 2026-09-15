"""CLI surface for the original-AMPL leg — honest unavailable + subset refusal.

These never require a real AMPL. When AMPL cannot be resolved (binary absent) the
command must report ``unavailable`` (exit 4) rather than crash or fake a number;
when a real AMPL IS resolvable (e.g. amplpy-managed) that path is skipped — the run
is then genuine and covered by the gated real test.
"""

from __future__ import annotations

import json

from acrpq.cli.main import main


def test_classical_original_ampl_unavailable(monkeypatch, capsys) -> None:
    # Force AMPL absent (deterministic, no real solve) to exercise the honest path.
    monkeypatch.setenv("ACRPQ_AMPL_BINARY", "/nonexistent/ampl_zzz")
    rc = main(["classical", "CP_4", "--solver", "original-ampl"])
    out = capsys.readouterr().out
    assert rc == 4  # honest unavailable exit code
    assert "solver:     original_ampl" in out
    assert "status:     unavailable" in out
    assert "CP_Instances" in out  # verbatim .dat source is reported
    assert "ampl solver: gurobi" in out  # supervised default is visible
    # the note is machine-readable JSON provenance, not a fabricated result
    note_line = next(line for line in out.splitlines() if line.startswith("note:") and "{" in line)
    payload = json.loads(note_line.split("note:", 1)[1].strip())
    assert payload["status_kind"] == "unavailable"


def test_classical_original_ampl_subset_refused(capsys) -> None:
    rc = main(["classical", "Q2", "--solver", "original-ampl"])  # Q2 is a subset scenario
    err = capsys.readouterr().err
    assert rc == 2
    assert "subset" in err
    assert "verbatim" in err


def test_bench_writes_baseline_artifact_with_honest_none(tmp_path, monkeypatch, capsys) -> None:
    # Force AMPL unavailable so the suite never incidentally runs a real solver;
    # the single real execution is done explicitly out-of-band, not in tests.
    monkeypatch.setenv("ACRPQ_AMPL_BINARY", "/nonexistent/ampl_zzz")
    rc = main(["bench", "Q4", "--no-quantum", "--original-ampl", "--out", str(tmp_path)])
    assert rc == 0
    baseline = json.loads((tmp_path / "baseline_Q4.json").read_text())[0]
    leg = baseline["legs"]["original_ampl"]
    assert leg["status"] == "unavailable"          # AMPL forced absent
    assert leg["is_optimal_anchor"] is False
    assert baseline["delta_discretization"]["delta"] is None  # never fabricated


def test_check_ampl_solver_reports_structured_status(monkeypatch, capsys) -> None:
    # Force AMPL unavailable: the CLI plumbing is tested without any real solve.
    monkeypatch.setenv("ACRPQ_AMPL_BINARY", "/nonexistent/ampl_zzz")
    rc = main(["check-ampl-solver", "gurobi"])
    out = capsys.readouterr().out
    report = json.loads(out.split("\nNOTE:", 1)[0])
    assert report["solver_requested"] == "gurobi"
    assert report["is_supervised_solver"] is True
    assert report["ampl_available"] is False
    assert report["ok"] is False
    assert rc == 5
