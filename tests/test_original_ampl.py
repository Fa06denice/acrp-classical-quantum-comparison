"""Original-AMPL runner tests — driven by a controllable FAKE ``ampl`` binary.

No test requires a real AMPL licence. A tiny Python script stands in for ``ampl``:
it reads the temp ``.run`` + ``.dat`` the runner produced and emits the structured
marker blocks (or a chosen failure), so the subprocess / parser / timeout /
rescoring / integrity paths are all exercised end-to-end.
"""

from __future__ import annotations

import json
import math
import os
import stat
import sys

import pytest

from acrpq.classical.original_ampl import OriginalAMPLError, OriginalAMPLSolver
from acrpq.io.loader import InstanceLoader
from acrpq.model import SolverKind, Status

_LOADER = InstanceLoader()


def _inst_and_dat(name="CP_4"):
    inst = _LOADER.load(name)
    dat, _src = _LOADER.read_dat_text(name)
    return inst, dat


def _write_fake_ampl(tmp_path, body: str, *, version_line: str | None = None) -> str:
    """Write an executable fake `ampl` that runs `body` (a Python snippet).

    `body` may use: RUN (the .run path, argv[1]), N (aircraft count parsed from the
    .dat in cwd), and must print marker blocks / set exit code. Like a real AMPL,
    ``ampl --version`` short-circuits (prints ``version_line`` if given, then exits)
    and never runs the solve body.
    """
    script = f'''#!{sys.executable}
import sys, glob, re, time, math
RUN = sys.argv[1]
if RUN == "--version":
    _v = {version_line!r}
    if _v:
        print(_v)
    sys.exit(0)
_dat = glob.glob("*.dat")
N = 0
if _dat:
    m = re.search(r"param\\s+n\\s*:=\\s*(\\d+)", open(_dat[0]).read())
    N = int(m.group(1)) if m else 0
{body}
'''
    p = tmp_path / "fake_ampl"
    p.write_text(script)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IRWXU)
    return str(p)


def _emit(status="solved", obj="0.0", q="1.0", theta="0.0", srnum=None, solvetime=None):
    """Body emitting a full valid marker set: q=<q> theta=<theta> for all i.

    ``srnum``/``solvetime`` markers are emitted only when provided (they are
    optional sections; their absence must not fail an otherwise-valid parse).
    """
    extra = ""
    if srnum is not None:
        extra += f'print("ACRPQ_SRNUM_BEGIN"); print("{srnum}"); print("ACRPQ_SRNUM_END")\n'
    if solvetime is not None:
        extra += (f'print("ACRPQ_SOLVETIME_BEGIN"); print("{solvetime}"); '
                  f'print("ACRPQ_SOLVETIME_END")\n')
    return f'''
print("ACRPQ_STATUS_BEGIN"); print("{status}"); print("ACRPQ_STATUS_END")
{extra}print("ACRPQ_OBJECTIVE_BEGIN"); print("{obj}"); print("ACRPQ_OBJECTIVE_END")
print("ACRPQ_Q_BEGIN")
for i in range(1, N+1): print(f"{{i}},{q}")
print("ACRPQ_Q_END")
print("ACRPQ_THETA_BEGIN")
for i in range(1, N+1): print(f"{{i}},{theta}")
print("ACRPQ_THETA_END")
'''


def _run(tmp_path, body, *, name="CP_4", timeout_s=30.0, tol=1e-6, max_out=256 * 1024,
         version_line=None, solver="gurobi", solver_options=""):
    inst, dat = _inst_and_dat(name)
    binary = _write_fake_ampl(tmp_path, body, version_line=version_line)
    runner = OriginalAMPLSolver(ampl_binary=binary, solver=solver,
                                solver_options=solver_options, timeout_s=timeout_s,
                                objective_tolerance=tol, max_output_bytes=max_out)
    return runner.solve(inst, dat_text=dat), inst


def _prov(result):
    return json.loads(result.message)


# --- 1: AMPL absent -> UNAVAILABLE ----------------------------------------- #
def test_ampl_absent_is_unavailable():
    inst, dat = _inst_and_dat()
    r = OriginalAMPLSolver(ampl_binary="/nonexistent/ampl_xyz").solve(inst, dat_text=dat)
    assert r.status is Status.UNAVAILABLE and r.solver is SolverKind.CLASSICAL_ORIGINAL_AMPL
    assert _prov(r)["status_kind"] == "unavailable"


# --- 3/19: valid optimal parse (incl. scientific notation + tabs handled) -- #
def test_valid_optimal_parse(tmp_path):
    r, _ = _run(tmp_path, _emit(status="solved", obj="0", q="1.0", theta="0.0"))
    assert r.solution is not None and len(r.solution.q) == 4 and len(r.solution.theta) == 4
    assert r.extra["provider_objective"] == 0.0 and r.extra["integrity_ok"] == 1.0


def test_scientific_notation_values(tmp_path):
    r, _ = _run(tmp_path, _emit(obj="1.0e-9", q="1.0", theta="1e-12"))
    assert r.solution is not None and r.extra["integrity_ok"] in (0.0, 1.0)


# --- 4: feasible-not-optimal not labelled OPTIMAL -------------------------- #
def test_feasible_not_optimal(tmp_path):
    r, _ = _run(tmp_path, _emit(status="solved?"))
    assert r.status is not Status.OPTIMAL          # 'solved?' -> FEASIBLE (or degraded)


# --- 5: infeasible --------------------------------------------------------- #
def test_infeasible(tmp_path):
    r, _ = _run(tmp_path, _emit(status="infeasible"))
    assert r.status is Status.INFEASIBLE


# --- 6: timeout kills the process (and children) --------------------------- #
def test_timeout(tmp_path):
    r, _ = _run(tmp_path, "import time\ntime.sleep(10)\n", timeout_s=1.0)
    assert r.status is Status.TIMEOUT and r.solution is None


def test_none_timeout_runs_without_wrapper_deadline(tmp_path):
    r, _ = _run(tmp_path, _emit(), timeout_s=None)
    assert r.status in {Status.OPTIMAL, Status.UNKNOWN}
    assert _prov(r)["timeout_s"] is None


# --- 7: non-zero exit with no stdout --------------------------------------- #
def test_nonzero_exit(tmp_path):
    r, _ = _run(tmp_path, "import sys\nsys.exit(3)\n")
    assert r.status is Status.ERROR


# --- 8/11/12: incomplete stdout / missing q / missing theta ---------------- #
def test_missing_q_section(tmp_path):
    body = _emit().replace('print("ACRPQ_Q_BEGIN")', "pass").replace('print("ACRPQ_Q_END")', "pass")
    r, _ = _run(tmp_path, body)
    assert r.status is Status.ERROR and _prov(r)["status_kind"] == "parse_error"


def test_missing_theta_section(tmp_path):
    body = _emit().replace('print("ACRPQ_THETA_BEGIN")', "pass").replace('print("ACRPQ_THETA_END")', "pass")
    r, _ = _run(tmp_path, body)
    assert r.status is Status.ERROR


# --- 9: duplicated marker -------------------------------------------------- #
def test_duplicated_marker(tmp_path):
    body = _emit() + '\nprint("ACRPQ_OBJECTIVE_BEGIN"); print("9"); print("ACRPQ_OBJECTIVE_END")\n'
    r, _ = _run(tmp_path, body)
    assert r.status is Status.ERROR


# --- 10: NaN/Inf objective ------------------------------------------------- #
def test_nan_objective(tmp_path):
    r, _ = _run(tmp_path, _emit(obj="nan"))
    assert r.status is Status.ERROR


# --- 13: duplicate index --------------------------------------------------- #
def test_duplicate_index(tmp_path):
    body = '''
print("ACRPQ_STATUS_BEGIN"); print("solved"); print("ACRPQ_STATUS_END")
print("ACRPQ_OBJECTIVE_BEGIN"); print("0"); print("ACRPQ_OBJECTIVE_END")
print("ACRPQ_Q_BEGIN")
print("1,1.0"); print("1,1.0")
for i in range(3, N+1): print(f"{i},1.0")
print("ACRPQ_Q_END")
print("ACRPQ_THETA_BEGIN")
for i in range(1, N+1): print(f"{i},0.0")
print("ACRPQ_THETA_END")
'''
    r, _ = _run(tmp_path, body)
    assert r.status is Status.ERROR


# --- 14: wrong cardinality ------------------------------------------------- #
def test_wrong_cardinality(tmp_path):
    body = _emit().replace("for i in range(1, N+1): print(f\"{i},1.0\")",
                           "for i in range(1, N): print(f\"{i},1.0\")", 1)  # one short in q
    r, _ = _run(tmp_path, body)
    assert r.status is Status.ERROR


# --- 15/16: output bounded ------------------------------------------------- #
def test_output_bounded_no_crash(tmp_path):
    body = 'print("X" * 5000)\n' + _emit()
    r, _ = _run(tmp_path, body, max_out=1000)
    assert len(r.message) <= 256 * 1024      # stored provenance bounded; no crash
    assert r.status in (Status.ERROR, Status.OPTIMAL, Status.FEASIBLE, Status.UNKNOWN)


# --- 17/18: path traversal / unknown instance ------------------------------ #
def test_path_traversal_refused():
    with pytest.raises(Exception):
        _LOADER.read_dat_text("../../etc/passwd")


def test_unknown_instance_refused():
    with pytest.raises(Exception):
        _LOADER.read_dat_text("NOPE_NOPE_123")


def test_unsafe_instance_name_refused(tmp_path):
    from acrpq.model import Family, Instance
    bad = Instance(name="../evil", family=Family.CP, n=2, d=0.05, radius=2.0,
                   v0=(5.0, 5.0), cap=(0.0, 0.0), x0=(0.0, 1.0), y0=(0.0, 0.0))
    binary = _write_fake_ampl(tmp_path, _emit())
    with pytest.raises(OriginalAMPLError):
        OriginalAMPLSolver(ampl_binary=binary).solve(bad, dat_text="param n := 2;\n")


def test_unsafe_solver_name_refused(tmp_path):
    # a solver name carrying AMPL syntax must be refused before any run script is built
    binary = _write_fake_ampl(tmp_path, _emit())
    inst, dat = _inst_and_dat()
    with pytest.raises(OriginalAMPLError):
        OriginalAMPLSolver(ampl_binary=binary, solver="couenne;\nshell 'x'").solve(
            inst, dat_text=dat
        )


def test_markers_at_tail_survive_verbose_solver_log(tmp_path):
    # A verbose solver log BEFORE the markers must not defeat parsing: we keep the
    # tail, so a valid trailing marker block is still read (bound large enough).
    body = 'print("noise line " * 50 + "\\n")\nfor _ in range(200): print("log")\n' + _emit(
        obj="0.0", q="1.0", theta="0.0"
    )
    r, _ = _run(tmp_path, body, max_out=256 * 1024)
    # parsing succeeded despite the leading log (markers read from the tail): the
    # provider objective was recovered and re-scored. Status is UNKNOWN here only
    # because q=1/theta=0 is conflictual for CP_4, not because parsing failed.
    assert r.status is not Status.ERROR
    assert r.extra["rescored_objective"] == 0.0
    assert r.extra["provider_objective"] == 0.0


# --- 20/21: rescoring integrity -------------------------------------------- #
def test_objective_matches_rescore_integrity_ok(tmp_path):
    # q=1, theta=0 -> geometry.objective == 0; provider says 0 -> integrity OK
    r, _ = _run(tmp_path, _emit(obj="0.0", q="1.0", theta="0.0"))
    assert r.extra["integrity_ok"] == 1.0
    assert r.extra["rescored_objective"] == 0.0


def test_objective_divergence_flagged(tmp_path):
    # provider claims a wildly different objective than the common re-score
    r, _ = _run(tmp_path, _emit(obj="123.456", q="1.0", theta="0.0"))
    assert r.extra["integrity_ok"] == 0.0
    assert r.status is Status.UNKNOWN          # degraded: not trustworthy as reference


# --- 22: AMPL feasible but common scorer conflictual ----------------------- #
def test_ampl_feasible_but_scorer_conflictual_not_silent(tmp_path):
    # all-noop (q=1,theta=0) leaves CP_4's initial conflicts; AMPL says 'solved'
    r, inst = _run(tmp_path, _emit(status="solved", obj="0.0", q="1.0", theta="0.0"))
    if r.n_conflicts and r.n_conflicts > 0:
        assert r.extra["ampl_feasible"] == 1.0 and r.extra["rescored_feasible"] == 0.0
        assert r.status is Status.UNKNOWN      # never silent ground truth


# --- 25/26: verbatim .dat + stable provenance hashes ----------------------- #
def test_data_hash_matches_original_dat(tmp_path):
    import hashlib
    inst, dat = _inst_and_dat("CP_4")
    r, _ = _run(tmp_path, _emit(), name="CP_4")
    expected = "sha256:" + hashlib.sha256(dat.encode("utf-8")).hexdigest()
    assert _prov(r)["data_sha256"] == expected


def test_provenance_has_model_and_preprocessing_hashes(tmp_path):
    r, _ = _run(tmp_path, _emit())
    p = _prov(r)
    assert p["model_sha256"].startswith("sha256:") and p["preprocessing_sha256"].startswith("sha256:")
    assert "ampl" not in json.dumps(p).lower() or "ampl_binary" in p   # binary path allowed, no token


# --- solver unavailable vs binary absent (generic, any authorised solver) --- #
def test_solver_unavailable_is_distinct_from_binary_absent(tmp_path):
    # AMPL runs, but the requested solver (gurobi) can't be loaded -> explicit
    # ERROR/solver_unavailable, NOT UNAVAILABLE (ampl missing), NEVER a success.
    body = 'print(\'Error: cannot find "gurobi"\')\n' + _emit()
    r, _ = _run(tmp_path, body, solver="gurobi")
    assert r.status is Status.ERROR
    assert r.feasible is False
    assert _prov(r)["status_kind"] == "solver_unavailable"


def test_binary_absent_is_unavailable_not_solver_unavailable():
    inst, dat = _inst_and_dat()
    r = OriginalAMPLSolver(ampl_binary="/nonexistent/ampl_zzz").solve(inst, dat_text=dat)
    assert r.status is Status.UNAVAILABLE
    assert _prov(r)["status_kind"] == "unavailable"


def test_gurobi_requested_but_only_couenne_available_never_falls_back(tmp_path):
    # Gurobi requested; only a couenne banner/driver present + gurobi-missing error.
    # Must NOT silently switch to couenne: report solver_unavailable for gurobi.
    body = ('print(\'Error: cannot find "gurobi"\')\n'
            'print("Couenne 0.5.8")\n') + _emit()
    r, _ = _run(tmp_path, body, solver="gurobi")
    assert r.status is Status.ERROR
    assert _prov(r)["status_kind"] == "solver_unavailable"
    assert _prov(r)["solver_requested"] == "gurobi"
    assert _prov(r)["solver_effective"] == "gurobi"   # never rewritten to couenne
    assert "couenne_version" not in _prov(r)          # no couenne substitution


def test_no_couenne_specific_message_when_gurobi_requested(tmp_path):
    # A couenne-not-found line must not trigger when gurobi is the requested solver.
    body = 'print(\'Error: cannot find "couenne"\')\n' + _emit()
    r, _ = _run(tmp_path, body, solver="gurobi")
    assert _prov(r)["status_kind"] != "solver_unavailable"


def test_unauthorised_solver_refused(tmp_path):
    inst, dat = _inst_and_dat()
    binary = _write_fake_ampl(tmp_path, _emit())
    with pytest.raises(OriginalAMPLError):
        OriginalAMPLSolver(ampl_binary=binary, solver="evilsolver").solve(inst, dat_text=dat)


def test_unsafe_solver_options_refused(tmp_path):
    inst, dat = _inst_and_dat()
    binary = _write_fake_ampl(tmp_path, _emit())
    with pytest.raises(OriginalAMPLError):
        OriginalAMPLSolver(ampl_binary=binary, solver="gurobi",
                           solver_options="x'; shell 'rm -rf /").solve(inst, dat_text=dat)


def test_solver_options_injected_into_temp_run_only(tmp_path):
    # the fake reads its .run and echoes an option line iff our option was injected
    body = ('_run = open(RUN).read()\n'
            'if "option gurobi_options \'pre:funcnonlinear=1\'" in _run:\n'
            '    print("  pre:funcnonlinear = 1")\n') + _emit(srnum="0")
    r, _ = _run(tmp_path, body, solver="gurobi", solver_options="pre:funcnonlinear=1")
    p = _prov(r)
    assert p["solver_options"] == "pre:funcnonlinear=1"
    assert p["solver_options_echo"].get("pre:funcnonlinear") == "1"


def test_no_options_line_when_options_empty(tmp_path):
    body = ('_run = open(RUN).read()\n'
            'assert "_options" not in _run, "no options line expected"\n') + _emit(srnum="0")
    r, _ = _run(tmp_path, body, solver="gurobi", solver_options="")
    assert _prov(r)["solver_options"] == ""
    assert _prov(r)["solver_options_echo"] == {}


# --- failed solve => NO fabricated primal (q/theta null, never zeros) -------- #
def test_solver_failure_yields_no_primal(tmp_path):
    # srnum 500 (failure); the fake still prints all-zero q/theta (stale values).
    # The runner must NOT serialise them as a solution.
    r, _ = _run(tmp_path, _emit(status="failure", srnum="500", obj="2.0",
                                q="0.0", theta="0.0"), solver="gurobi")
    assert r.status is Status.ERROR
    assert r.solution is None                 # never the fabricated zeros
    assert "provider_objective" not in r.extra  # no primal-derived numbers trusted
    assert "no usable primal" in _prov(r)["message"]


def test_infeasible_yields_no_primal(tmp_path):
    r, _ = _run(tmp_path, _emit(status="infeasible", srnum="200"), solver="gurobi")
    assert r.status is Status.INFEASIBLE
    assert r.solution is None


def test_mp_check_failure_rejects_primal_and_records_residual(tmp_path):
    # 'Solution check failed - reporting as fatal' + a 9e-9 quadratic residual, srnum
    # 150 with stale zeros. The runner must record the residual and NOT serialise the
    # rejected primal.
    body = ('print("Gurobi 13.0.2: Solution check failed - reporting as fatal:")\n'
            'print("Type                         MaxAbs [Name]   MaxRel [Name]")\n'
            'print("* quadratic con(s)             9E-09           -")\n'
            ) + _emit(status="solved?", srnum="150", obj="2.0", q="0.0", theta="0.0")
    r, _ = _run(tmp_path, body, solver="gurobi")
    assert r.solution is None                       # rejected primal not serialised
    assert _prov(r)["status_kind"] == "mp_check_failed"
    assert r.extra["max_model_constraint_residual"] == 9e-09
    assert _prov(r)["mp_solution_check"]["failed"] is True


# --- MP solution-check residual table is parsed ----------------------------- #
def test_mp_constraint_residual_captured(tmp_path):
    table = ('print("------------ WARNINGS ------------")\n'
             'print(\'WARNING:  "Tolerance violations"\')\n'
             'print("  Type                         MaxAbs [Name]   MaxRel [Name]")\n'
             'print("* quadratic con(s)             7E-06           -")\n'
             'print("*: Using aux variable values.")\n')
    r, _ = _run(tmp_path, table + _emit(srnum="0", q="1.0", theta="0.0"), solver="gurobi")
    assert r.extra["max_model_constraint_residual"] == 7e-06
    mp = _prov(r)["mp_solution_check"]
    assert mp["max_abs_residual"] == 7e-06
    assert mp["violations"][0]["type"] == "quadratic con(s)"


def test_license_error_is_explicit_never_success(tmp_path):
    body = 'print("Gurobi: license expired")\n' + _emit()
    r, _ = _run(tmp_path, body, solver="gurobi")
    assert r.status is Status.ERROR
    assert r.feasible is False
    assert _prov(r)["status_kind"] == "license_error"


def test_normal_run_is_not_misflagged(tmp_path):
    r, _ = _run(tmp_path, _emit(status="solved", srnum="0"), solver="gurobi")
    assert _prov(r)["status_kind"] not in ("solver_unavailable", "license_error")


# --- status from solve_result_num, never the bare word 'solved' ------------- #
def test_status_from_srnum_optimal(tmp_path):
    r, _ = _run(tmp_path, _emit(status="weirdtext", srnum="0", q="1.0", theta="0.0"),
                solver="gurobi")
    # srnum 0 => OPTIMAL band regardless of the status text
    assert r.extra["solve_result_num"] == 0.0
    assert r.extra["optimality_proven"] == 1.0


def test_time_limit_srnum_is_timeout_not_optimum(tmp_path):
    # Text says 'solved' but the numeric code is a LIMIT band -> TIMEOUT, never optimum.
    r, _ = _run(tmp_path, _emit(status="solved", srnum="410"), solver="gurobi")
    assert r.status is Status.TIMEOUT
    assert r.extra["optimality_proven"] == 0.0


def test_feasible_srnum_band(tmp_path):
    r, _ = _run(tmp_path, _emit(status="solved?", srnum="150", q="1.0", theta="0.0"),
                solver="gurobi")
    assert r.status in (Status.FEASIBLE, Status.UNKNOWN)  # UNKNOWN only if rescore degrades
    assert r.extra["optimality_proven"] == 0.0


# --- provenance: solver + AMPL versions (best-effort, no licence) ------------ #
def test_solver_version_and_driver_scanned_from_banner(tmp_path):
    banner = ('print("AMPL/Gurobi Optimizer [13.0.2] (Darwin x86_64), '
              'driver(20260624), MP(20260624)")\n')
    r, _ = _run(tmp_path, banner + _emit(srnum="0", solvetime="0.07"), solver="gurobi")
    p = _prov(r)
    assert p["solver_version"] == "13.0.2"
    assert p["solver_driver_version"] == "20260624"
    assert p["solver_requested"] == "gurobi" and p["solver_effective"] == "gurobi"
    assert r.extra["solver_time_s"] == 0.07


def test_couenne_legacy_version_field_only_for_couenne(tmp_path):
    body = 'print("Couenne 0.5.8")\n' + _emit(srnum="0")
    r, _ = _run(tmp_path, body, solver="couenne")
    p = _prov(r)
    assert p["solver_version"] == "0.5.8"
    assert p["couenne_version"] == "0.5.8"   # legacy field populated only for couenne


def test_truncated_or_unrecognised_banner_gives_null_version(tmp_path):
    # a partial/unrecognised banner must yield None, never a fabricated version
    body = 'print("Gurobi optimizing...")\n' + _emit(srnum="0")
    r, _ = _run(tmp_path, body, solver="gurobi")
    assert _prov(r)["solver_version"] is None


def test_mip_gap_scanned_when_present(tmp_path):
    body = 'print("absmipgap=5.05e-16, relmipgap=0")\n' + _emit(srnum="0")
    r, _ = _run(tmp_path, body, solver="gurobi")
    assert r.extra["mip_abs_gap"] == 5.05e-16
    assert r.extra["mip_rel_gap"] == 0.0


def test_ampl_version_recorded_best_effort(tmp_path):
    r, _ = _run(tmp_path, _emit(obj="0.0", q="1.0", theta="0.0"),
                version_line="AMPL Version 20240115 (fake build)")
    ver = _prov(r)["ampl_version"]
    assert ver and "20240115" in ver


def test_versions_absent_are_null_not_faked(tmp_path):
    # a fake that supports neither --version nor a solver banner -> honest None
    r, _ = _run(tmp_path, _emit(obj="0.0", q="1.0", theta="0.0"))
    p = _prov(r)
    assert p["solver_version"] is None
    assert p["solver_driver_version"] is None
    assert p["ampl_version"] is None
    assert "couenne_version" not in p   # legacy field absent for non-couenne solvers


# --- 27: Code/ never modified during a run --------------------------------- #
def test_code_dir_unchanged_by_run(tmp_path):
    import hashlib
    from acrpq.classical.original_ampl import _repo_code_dir
    code = _repo_code_dir()
    before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
              for f in code.iterdir() if f.is_file()}
    _run(tmp_path, _emit())
    after = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
             for f in code.iterdir() if f.is_file()}
    assert before == after


# --- real AMPL integration (opt-in, never in CI) --------------------------- #
@pytest.mark.ampl
@pytest.mark.skipif(os.environ.get("ACRPQ_RUN_REAL_AMPL") != "1",
                    reason="set ACRPQ_RUN_REAL_AMPL=1 with a real AMPL+solver to run")
def test_real_ampl_cp4():
    # Supervised solver (Gurobi) on the whole CP_4 instance; resolves ampl via
    # amplpy if not on PATH.
    solver = OriginalAMPLSolver(solver="gurobi", timeout_s=120.0)
    if solver._resolve_binary() is None:
        pytest.skip("no resolvable ampl binary")
    inst, dat = _inst_and_dat("CP_4")
    r = solver.solve(inst, dat_text=dat)
    assert r.status in (Status.OPTIMAL, Status.FEASIBLE, Status.TIMEOUT, Status.UNKNOWN)
    if r.solution is not None:
        assert len(r.solution.q) == inst.n and len(r.solution.theta) == inst.n
        assert all(math.isfinite(v) for v in r.solution.q + r.solution.theta)
