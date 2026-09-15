"""Direct execution of the historical AMPL model ``Code/NC_ACRP.mod`` (OPTIONAL).

This is the *authoritative continuous baseline* — the thesis director's own model,
run verbatim through AMPL and an **authorised solver**. The supervised protocol
solver is **Gurobi** (the director's explicit choice); Couenne and the other AMPL
solvers are permitted but NEVER substituted automatically. It is deliberately kept
separate from the Pyomo rebuild (:mod:`acrpq.classical.pyomo_minlp`):
``pyomo_minlp`` is a Python reconstruction, ``original_ampl`` is the original. The
word "AMPL" here never means Pyomo.

Safety / honesty contract:

* ``Code/`` is never modified. The runner copies the *verbatim* original ``.dat``
  into a per-run temp directory and generates its own temp ``.run`` that
  ``model``/``include`` the original files by absolute, validated path. The solver
  is selected only in that temporary ``.run`` (``option solver <name>``).
* the ``ampl`` binary is invoked with ``shell=False``, list arguments, an explicit
  cwd, a minimal environment (amplpy solver dirs prepended so the chosen solver
  resolves), a real timeout, and the whole process group is killed on timeout so no
  solver child survives; stdout/stderr are bounded.
* the historical constant ``PI := 3.141592`` is left untouched — the model computes
  its own default positions; we never inject ``math.pi``-derived coordinates.
* the solver name is charset-validated and whitelist-checked (anti AMPL-injection);
  an unauthorised solver is refused.
* AMPL binary absent → ``Status.UNAVAILABLE``; AMPL present but the requested solver
  unloadable → explicit ``solver_unavailable``; solver licence problem → explicit
  ``license_error``. Never a crash, a fabricated number, or a silent solver swap.
* status comes from the canonical ``solve_result_num`` (not the bare word
  'solved'); a time/iteration limit is TIMEOUT, never an optimum.
* the reported objective is re-scored by the repository's common geometry; a
  divergence beyond tolerance raises an integrity flag and degrades the status.
* no licence/token/secret is read, stored, logged or exported.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from .. import geometry
from ..model import Instance, Result, Solution, SolverKind, Status

# Divergence beyond this (absolute OR relative) between the AMPL-reported objective
# and the common-geometry re-score raises an integrity flag.
DEFAULT_OBJECTIVE_TOLERANCE = 1e-6
_MAX_OUTPUT_BYTES = 256 * 1024

# The supervised protocol solver (the thesis director asked for Gurobi). Couenne is
# an OPTIONAL open-source cross-check, never an automatic fallback.
SUPERVISED_SOLVER = "gurobi"
# AMPL solvers this runner is allowed to drive. An unauthorised name is refused
# (in addition to the AMPL-injection charset guard). No solver is EVER auto-swapped
# for another — a missing requested solver is reported, not silently replaced.
ALLOWED_SOLVERS = frozenset({"gurobi", "couenne", "cplex", "xpress", "highs"})

_MARKER = {
    "status": ("ACRPQ_STATUS_BEGIN", "ACRPQ_STATUS_END"),
    "srnum": ("ACRPQ_SRNUM_BEGIN", "ACRPQ_SRNUM_END"),
    "objective": ("ACRPQ_OBJECTIVE_BEGIN", "ACRPQ_OBJECTIVE_END"),
    "solvetime": ("ACRPQ_SOLVETIME_BEGIN", "ACRPQ_SOLVETIME_END"),
    "q": ("ACRPQ_Q_BEGIN", "ACRPQ_Q_END"),
    "theta": ("ACRPQ_THETA_BEGIN", "ACRPQ_THETA_END"),
}
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")
# The solver name is interpolated into AMPL's ``option solver <name>;`` — restrict
# it to a safe charset so no AMPL statement can ever be injected through it.
_SOLVER_RE = re.compile(r"^[A-Za-z0-9_.+-]{1,32}$")
# The solver-options string is interpolated into ``option <solver>_options '<opts>'``.
# Allow only option-syntax characters (no quotes/semicolons) so no AMPL statement
# can be injected. e.g. "pre:funcnonlinear=1 alg:feastol=1e-9 sol:chk:fail=1".
_SOLVER_OPTS_RE = re.compile(r"^[A-Za-z0-9_:=.+\- eE]{0,512}$")
# Driver option echoes ("pre:funcnonlinear = 1") — proof of the mode actually used.
# Searched anywhere in the line (the driver may prefix it with "Gurobi 13.0.2:").
_OPT_ECHO_RE = re.compile(
    r"((?:pre|alg|mip|sol|tech|cvt|lim|bar|nlbar):[\w:]+)\s*=\s*(\S+)")
# A plausible version token (rejects marker/log lines when reading `ampl --version`).
_VERSION_RE = re.compile(r"\b\d+\.\d+(?:\.\d+)?\b|\bVersion\s+\d", re.IGNORECASE)
# Version/gap/termination scanned best-effort from the solver's own stdout banner.
_BRACKET_VER_RE = re.compile(r"\[(\d+\.\d+(?:\.\d+)?)\]")  # e.g. AMPL/Gurobi ... [13.0.2]
_DRIVER_VER_RE = re.compile(r"driver\((\d+)\)")            # e.g. driver(20260624)
_ABS_GAP_RE = re.compile(r"absmipgap\s*=\s*([0-9.eE+\-]+)")
_REL_GAP_RE = re.compile(r"relmipgap\s*=\s*([0-9.eE+\-]+)")
# MP driver "Tolerance violations" rows: "* quadratic con(s)   7E-06   -"
# (a violation row starts "* " with whitespace; the "*:" footnote line does not).
_MP_VIOL_RE = re.compile(r"^\*\s+(.+?)\s{2,}([0-9][0-9.eE+\-]*)\s+(\S+)\s*$")
# A licence PROBLEM reported by the solver (message only — the licence FILE is never
# read, copied or logged). Any match => explicit error, never a false success. NB:
# a benign "size-limited license" community-edition note is deliberately NOT matched.
_LICENSE_ERROR_RE = re.compile(
    r"licen[cs]e (?:expired|not found|is invalid|is required|error|has expired)|"
    r"no valid licen[cs]e|no licen[cs]e (?:found|available)|expired licen[cs]e|"
    r"invalid licen[cs]e|failed to (?:find|read) [^\n]*licen[cs]e",
    re.IGNORECASE,
)


class OriginalAMPLError(RuntimeError):
    """A structured failure of the original-AMPL runner (never masks a problem)."""


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _repo_code_dir() -> Path:
    """Locate the repository ``Code/`` directory (holding the historical model)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "Code" / "NC_ACRP.mod"
        if cand.is_file():
            return parent / "Code"
    raise OriginalAMPLError("could not locate Code/NC_ACRP.mod from the package layout")


@dataclass
class OriginalAMPLSolver:
    """Run the original ``NC_ACRP.mod`` via AMPL + an authorised solver.

    ``solver`` defaults to the supervised protocol solver (Gurobi). Couenne and the
    other AMPL solvers are allowed but never substituted automatically. If the AMPL
    binary is absent the result is ``Status.UNAVAILABLE``; if AMPL runs but the
    requested solver cannot be loaded the result is an explicit ``solver_unavailable``
    error — never a fabricated optimum, never a silent fallback.
    """

    solver: str = SUPERVISED_SOLVER
    solver_options: str = ""                  # e.g. "pre:funcnonlinear=1 alg:feastol=1e-9"
    timeout_s: float | None = 300.0
    ampl_binary: str | None = None            # default: $ACRPQ_AMPL_BINARY, amplpy, PATH
    code_dir: Path | None = None
    objective_tolerance: float = DEFAULT_OBJECTIVE_TOLERANCE
    max_output_bytes: int = _MAX_OUTPUT_BYTES

    # ------------------------------------------------------------------ #
    def _resolve_binary(self) -> str | None:
        cand = self.ampl_binary or os.environ.get("ACRPQ_AMPL_BINARY")
        if cand:
            if os.path.sep in cand:
                return cand if (Path(cand).is_file() and os.access(cand, os.X_OK)) else None
            return shutil.which(cand)
        # No explicit binary: prefer an amplpy-managed AMPL, else PATH.
        found = _amplpy_find("ampl")
        return found or shutil.which("ampl")

    def _code_paths(self) -> tuple[Path, Path]:
        code = Path(self.code_dir) if self.code_dir else _repo_code_dir()
        mod = (code / "NC_ACRP.mod").resolve()
        pre = (code / "Preprocessing.run").resolve()
        if not mod.is_file() or not pre.is_file():
            raise OriginalAMPLError(f"missing Code/ files under {code}")
        return mod, pre

    def solve(self, inst: Instance, *, dat_text: str, time_limit_s: float | None = None) -> Result:
        """Run the original model on ``inst`` using the verbatim original ``.dat``.

        ``dat_text`` MUST be the original instance file (e.g. from
        :meth:`InstanceLoader.read_dat_text`) — it is written verbatim, never
        reconstructed.
        """
        t0 = time.perf_counter()
        n_initial = geometry.count_initial_conflicts(inst)
        timeout = (float(time_limit_s) if time_limit_s is not None
                   else (float(self.timeout_s) if self.timeout_s is not None else None))
        if not _RUN_NAME_RE.match(inst.name):
            raise OriginalAMPLError(f"unsafe instance name {inst.name!r}")
        if not _SOLVER_RE.match(self.solver):
            raise OriginalAMPLError(f"unsafe solver name {self.solver!r}")
        if self.solver not in ALLOWED_SOLVERS:
            raise OriginalAMPLError(
                f"unauthorised solver {self.solver!r}; allowed: {sorted(ALLOWED_SOLVERS)}")
        if not _SOLVER_OPTS_RE.match(self.solver_options):
            raise OriginalAMPLError(f"unsafe solver options {self.solver_options!r}")
        if not math.isfinite(inst.w):
            raise OriginalAMPLError(f"non-finite objective weight w={inst.w!r}")

        binary = self._resolve_binary()
        if binary is None:
            return self._unavailable(inst, n_initial, t0, timeout,
                                     reason=f"AMPL binary not found (tried {self.ampl_binary or 'ampl'})")
        mod, pre = self._code_paths()
        mod_bytes, pre_bytes = mod.read_bytes(), pre.read_bytes()
        dat_bytes = dat_text.encode("utf-8")

        tmp = Path(tempfile.mkdtemp(prefix="acrpq_ampl_"))
        try:
            dat_path = tmp / f"{inst.name}.dat"
            dat_path.write_text(dat_text, encoding="utf-8")
            run_path = tmp / "acrpq_run.run"
            run_path.write_text(self._run_script(mod, pre, dat_path, inst), encoding="utf-8")
            proc = self._invoke(binary, run_path, tmp, timeout)
            prov = self._provenance(inst, binary, mod_bytes, pre_bytes, dat_bytes, timeout)
            if proc["timed_out"]:
                return self._status_result(
                    inst, n_initial, t0, Status.TIMEOUT, prov,
                    message="ampl timed out; process group killed", solver_time_s=None)
            # AMPL ran but the requested solver (e.g. Couenne) is missing/unloadable:
            # an explicit diagnostic, distinct from "ampl absent", NEVER a success.
            license_problem = _detect_license_error(proc["stdout"], proc["stderr"])
            if license_problem:
                return self._status_result(
                    inst, n_initial, t0, Status.ERROR, prov,
                    message=f"solver licence problem (licence file never read): {license_problem}",
                    solver_time_s=None, status_kind="license_error")
            solver_missing = _detect_solver_unavailable(
                proc["stdout"], proc["stderr"], self.solver)
            if solver_missing:
                return self._status_result(
                    inst, n_initial, t0, Status.ERROR, prov,
                    message=(f"ampl ran but solver {self.solver!r} could not be "
                             f"loaded: {solver_missing}"),
                    solver_time_s=None, status_kind="solver_unavailable")
            if proc["returncode"] != 0 and not proc["stdout"]:
                return self._status_result(
                    inst, n_initial, t0, Status.ERROR, prov,
                    message=f"ampl exited {proc['returncode']}: {proc['stderr'][:400]}",
                    solver_time_s=None)
            return self._parse_and_score(inst, n_initial, t0, proc, prov)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------------------------ #
    def check_solver(self) -> dict[str, Any]:
        """Verify the AMPL toolchain WITHOUT the scientific model (cheap 1-var QP).

        Confirms the AMPL binary, the requested solver + its version/driver, and
        that the licence is *usable* — by a trivial ``min (x-2)^2`` solve, never by
        reading the licence file. It does NOT validate the ACRP model. Returns a
        structured dict; ``ok`` is true only when AMPL ran the solver to a proven
        ``solved`` result with no licence/solver error.
        """
        report: dict[str, Any] = {
            "solver_requested": self.solver,
            "supervised_solver": SUPERVISED_SOLVER,
            "is_supervised_solver": self.solver == SUPERVISED_SOLVER,
            "ampl_available": False, "solver_available": False,
            "license_ok": False, "ok": False,
            "ampl_version": None, "solver_version": None, "solver_driver_version": None,
            "solve_result": None, "status_kind": None, "note": "",
        }
        if not _SOLVER_RE.match(self.solver) or self.solver not in ALLOWED_SOLVERS:
            report["note"] = f"unauthorised solver {self.solver!r}"
            return report
        binary = self._resolve_binary()
        if binary is None:
            report["note"] = "AMPL binary not found"
            return report
        report["ampl_available"] = True
        report["ampl_binary"] = binary
        report["ampl_version"] = _ampl_version(binary)

        tmp = Path(tempfile.mkdtemp(prefix="acrpq_amplcheck_"))
        try:
            run_path = tmp / "check.run"
            run_path.write_text(
                "reset;\nvar x >= 0;\nminimize O: (x-2)^2;\n"
                f"option solver {self.solver};\nsolve;\n"
                'printf "ACRPQ_CHECK SR=%s SRNUM=%d\\n", solve_result, solve_result_num;\n',
                encoding="utf-8",
            )
            # Scientific solves may deliberately be unlimited; the cheap
            # toolchain probe itself must nevertheless remain bounded.
            check_timeout = min(self.timeout_s, 60.0) if self.timeout_s is not None else 60.0
            proc = self._invoke(binary, run_path, tmp, check_timeout)
            out, err = proc["stdout"], proc["stderr"]
            lic = _detect_license_error(out, err)
            missing = _detect_solver_unavailable(out, err, self.solver)
            ver, drv = _scan_solver_banner(out, self.solver)
            report["solver_version"], report["solver_driver_version"] = ver, drv
            m = re.search(r"ACRPQ_CHECK SR=(\S+) SRNUM=(-?\d+)", out)
            report["solve_result"] = m.group(1) if m else None
            srnum = int(m.group(2)) if m else None
            if lic:
                report["status_kind"] = "license_error"
                report["note"] = f"licence problem (file never read): {lic}"
            elif missing:
                report["status_kind"] = "solver_unavailable"
                report["note"] = f"solver {self.solver!r} not loadable: {missing}"
            elif proc["timed_out"]:
                report["status_kind"] = "timeout"
                report["note"] = "toolchain check timed out"
            elif srnum is not None and 0 <= srnum < 100:
                report["solver_available"] = True
                report["license_ok"] = True
                report["ok"] = True
                report["status_kind"] = "ok"
            else:
                report["status_kind"] = "unknown"
                report["note"] = f"unexpected result (rc={proc['returncode']}, srnum={srnum})"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return report

    # ------------------------------------------------------------------ #
    def _options_line(self) -> str:
        """The validated ``option <solver>_options '<opts>';`` line, or empty.

        ``solver_options`` is charset-restricted (no quotes/semicolons) so nothing
        can escape the single-quoted AMPL string. Set only in this TEMP .run.
        """
        if not self.solver_options:
            return ""
        return f"option {self.solver}_options '{self.solver_options}';\n"

    def _run_script(self, mod: Path, pre: Path, dat: Path, inst: Instance) -> str:
        """A temp .run that runs the ORIGINAL model and prints machine markers.

        Uses absolute, validated paths (never a user string interpolated into AMPL
        code). Sets ``A``/``wd`` exactly like the historical ``NC_ACRP.run``.
        """
        def amplstr(p: Path) -> str:
            return str(p).replace("\\", "\\\\").replace('"', '\\"')

        return f"""reset;
model "{amplstr(mod)}";
problem NC_ACRP: Obj,q,theta,vrx,vry,z,
    bin11,bin12,cvrx,cvry,
    bin311,bin312,bin321,bin322,
    bin331,bin332,bin341,bin342;
option solver {self.solver};
{self._options_line()}data "{amplstr(dat)}";
let A := 1..n;
let wd := {inst.w!r};
include "{amplstr(pre)}";
problem NC_ACRP;
solve NC_ACRP;
printf "{_MARKER['status'][0]}\\n";
printf "%s\\n", solve_result;
printf "{_MARKER['status'][1]}\\n";
printf "{_MARKER['srnum'][0]}\\n";
printf "%d\\n", solve_result_num;
printf "{_MARKER['srnum'][1]}\\n";
printf "{_MARKER['objective'][0]}\\n";
printf "%.12g\\n", Obj;
printf "{_MARKER['objective'][1]}\\n";
printf "{_MARKER['solvetime'][0]}\\n";
printf "%.6f\\n", _solve_time;
printf "{_MARKER['solvetime'][1]}\\n";
printf "{_MARKER['q'][0]}\\n";
for {{i in A}} printf "%d,%.12g\\n", i, q[i];
printf "{_MARKER['q'][1]}\\n";
printf "{_MARKER['theta'][0]}\\n";
for {{i in A}} printf "%d,%.12g\\n", i, theta[i];
printf "{_MARKER['theta'][1]}\\n";
"""

    def _invoke(self, binary: str, run_path: Path, cwd: Path,
                timeout: float | None) -> dict[str, Any]:
        """subprocess with shell=False, list args, minimal env, process-group kill."""
        # Prepend the amplpy module bin dirs so an amplpy-managed solver (gurobi,
        # cplex, ...) resolves when we drive an amplpy-managed ampl, without
        # inheriting the user's full environment.
        base_path = os.environ.get("PATH", "/usr/bin:/bin")
        mod_path = _amplpy_module_path()
        path = f"{mod_path}{os.pathsep}{base_path}" if mod_path else base_path
        env = {"PATH": path, "HOME": os.environ.get("HOME", str(cwd)), "TMPDIR": str(cwd)}
        # keep a licence dir on PATH-adjacent vars if the operator set them, but
        # never read/copy the licence content ourselves
        for k in ("AMPL_LICFILE", "ampl_lic"):
            if k in os.environ:
                env[k] = os.environ[k]
        popen = subprocess.Popen(  # noqa: S603 - shell=False, list args, validated paths
            [binary, str(run_path)], cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            out, err = popen.communicate(timeout=timeout)
        except KeyboardInterrupt:
            # An operator abort must not leave the AMPL/solver process group
            # consuming CPU in the background.  Timeout already has this
            # guarantee; interactive cancellation needs the same cleanup.
            try:
                os.killpg(os.getpgid(popen.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                popen.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(popen.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                popen.communicate()
            raise
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(popen.pid), signal.SIGKILL)  # kill Couenne too
            except (ProcessLookupError, PermissionError):
                pass
            try:
                out, err = popen.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                out, err = "", ""
        # Keep the TAIL of stdout: the machine markers are printed *after* the
        # (potentially verbose) solver log, so trailing bytes must survive
        # truncation. A cut that clips a marker fails the strict parser -> ERROR,
        # never a false success.
        return {
            "returncode": popen.returncode, "timed_out": timed_out,
            "stdout": (out or "")[-self.max_output_bytes:],
            "stderr": (err or "")[: self.max_output_bytes],
        }

    # ------------------------------------------------------------------ #
    def _section(self, text: str, name: str) -> str:
        begin, end = _MARKER[name]
        n_begin, n_end = text.count(begin), text.count(end)
        if n_begin != 1 or n_end != 1:
            raise OriginalAMPLError(f"section {name!r}: expected exactly one block, "
                                    f"got begin={n_begin} end={n_end}")
        body = text.split(begin, 1)[1].split(end, 1)[0]
        return body.strip("\n")

    def _optional_section(self, text: str, name: str) -> str | None:
        """Like :meth:`_section` but returns ``None`` if the block is absent.

        Used for solver-provided extras (status code, solve time) that older
        artifacts / minimal solvers may not emit — their absence must never fail
        the parse of an otherwise-valid solution.
        """
        begin, end = _MARKER[name]
        if text.count(begin) != 1 or text.count(end) != 1:
            return None
        return text.split(begin, 1)[1].split(end, 1)[0].strip("\n")

    def _map_status(self, srnum: int | None, raw: str) -> Status:
        """Status from the canonical AMPL ``solve_result_num`` (never text alone).

        AMPL bands: 0-99 solved, 100-199 solved?, 200-299 infeasible, 300-399
        unbounded, 400-499 limit (e.g. time), 500+ failure. A time/iteration LIMIT
        is TIMEOUT, never an optimum. Falls back to the text only when the numeric
        code is unavailable.
        """
        if srnum is not None:
            if 0 <= srnum < 100:
                return Status.OPTIMAL
            if 100 <= srnum < 200:
                return Status.FEASIBLE
            if 200 <= srnum < 300:
                return Status.INFEASIBLE
            if 300 <= srnum < 400:
                return Status.UNKNOWN  # unbounded — not a usable optimum here
            if 400 <= srnum < 500:
                return Status.TIMEOUT
            return Status.ERROR
        return self._map_ampl_status(raw)

    def _parse_index_value(self, body: str, name: str, n: int) -> tuple[float, ...]:
        seen: dict[int, float] = {}
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                raise OriginalAMPLError(f"{name}: bad line {line!r}")
            try:
                idx = int(parts[0])
                val = float(parts[1])
            except ValueError as exc:
                raise OriginalAMPLError(f"{name}: non-numeric {line!r}") from exc
            if not math.isfinite(val):
                raise OriginalAMPLError(f"{name}: non-finite value at index {idx}")
            if idx in seen:
                raise OriginalAMPLError(f"{name}: duplicate index {idx}")
            if not 1 <= idx <= n:
                raise OriginalAMPLError(f"{name}: index {idx} out of range 1..{n}")
            seen[idx] = val
        if len(seen) != n:
            raise OriginalAMPLError(f"{name}: expected {n} values, got {len(seen)}")
        return tuple(seen[i] for i in range(1, n + 1))

    def _map_ampl_status(self, raw: str) -> Status:
        s = raw.strip().lower()
        if s.startswith("solved"):          # includes 'solved?' (couenne local opt)
            return Status.OPTIMAL if s == "solved" else Status.FEASIBLE
        if "infeasible" in s:
            return Status.INFEASIBLE
        if s.startswith("limit") or "stop" in s:
            return Status.TIMEOUT
        if s.startswith("failure") or "error" in s:
            return Status.ERROR
        return Status.UNKNOWN

    def _parse_and_score(self, inst, n_initial, t0, proc, prov) -> Result:
        text = proc["stdout"]
        try:
            raw_status = self._section(text, "status")
            obj_body = self._section(text, "objective")
            q = self._parse_index_value(self._section(text, "q"), "q", inst.n)
            theta = self._parse_index_value(self._section(text, "theta"), "theta", inst.n)
            provider_obj = float(obj_body.strip())
            if not math.isfinite(provider_obj):
                raise OriginalAMPLError("objective is not finite")
            srnum = _parse_int_or_none(self._optional_section(text, "srnum"))
            solver_time_s = _parse_float_or_none(self._optional_section(text, "solvetime"))
        except OriginalAMPLError as exc:
            return self._status_result(inst, n_initial, t0, Status.ERROR, prov,
                                       message=f"parse error: {exc}", solver_time_s=None,
                                       status_kind="parse_error")
        # Status from the canonical numeric code, NOT the bare word 'solved'.
        ampl_status = self._map_status(srnum, raw_status)
        abs_gap, rel_gap = _scan_gaps(text)
        solver_version, driver_version = _scan_solver_banner(text, self.solver)
        mp_check = _scan_mp_check(text)

        # Strict MP solution check FAILED (sol:chk:fail fatal): the solver's own check
        # rejected the returned point, whose variables are now stale/uninitialised.
        # Record the real residual and diagnosis; NEVER serialise the rejected primal.
        if mp_check.get("failed"):
            resid = mp_check.get("max_abs_residual")
            prov_mp = {
                **prov, "ampl_solve_result": raw_status.strip()[:120],
                "solve_result_num": srnum, "solver_effective": self.solver,
                "solver_options": self.solver_options,
                "solver_options_echo": _scan_option_echo(text),
                "mp_solution_check": mp_check,
            }
            extra_mp = {"timeout_s": prov["timeout_s"]}
            if resid is not None:
                extra_mp["max_model_constraint_residual"] = resid
            return Result(
                instance_name=inst.name, solver=SolverKind.CLASSICAL_ORIGINAL_AMPL,
                status=Status.UNKNOWN, solution=None, n_initial_conflicts=n_initial,
                wall_time_s=time.perf_counter() - t0, backend=f"ampl+{self.solver}",
                message=self._bounded_json({
                    **prov_mp, "status_kind": "mp_check_failed",
                    "message": (f"MP solution check failed (max residual "
                                f"{resid}); rejected primal not serialised")}),
                extra=extra_mp)

        # No usable primal: on solver ERROR / INFEASIBLE the reported variable values
        # are stale/uninitialised (e.g. all-zero) — NEVER serialise them as a
        # solution. Record the diagnosis with solution=None instead.
        if ampl_status in (Status.ERROR, Status.INFEASIBLE):
            prov_err = {
                **prov, "ampl_solve_result": raw_status.strip()[:120],
                "solve_result_num": srnum, "solver_effective": self.solver,
                "solver_options": self.solver_options,
                "solver_options_echo": _scan_option_echo(text),
                "mp_solution_check": mp_check or None,
            }
            return self._status_result(
                inst, n_initial, t0, ampl_status, prov_err,
                message=f"solver reported {raw_status.strip()[:80]!r} (srnum={srnum}); "
                        "no usable primal — variables not serialised",
                solver_time_s=solver_time_s, status_kind=ampl_status.value)

        # A proven optimum requires the 'solved' band AND a (reported) zero-ish gap.
        optimality_proven = ampl_status is Status.OPTIMAL and (
            rel_gap is None or rel_gap <= max(self.objective_tolerance, 1e-9))

        # --- common-geometry re-score (never trust the reported objective) ---
        rescored_obj = geometry.objective(inst, q, theta)
        n_conf = geometry.count_conflicts(inst, q, theta)
        rescored_feasible = n_conf == 0
        abs_delta = abs(provider_obj - rescored_obj)
        rel_delta = abs_delta / abs(provider_obj) if provider_obj != 0 else math.inf
        integrity_ok = abs_delta <= self.objective_tolerance or (
            math.isfinite(rel_delta) and rel_delta <= self.objective_tolerance)
        ampl_feasible = ampl_status in (Status.OPTIMAL, Status.FEASIBLE)

        status = ampl_status
        notes = []
        if not integrity_ok:
            notes.append(f"objective_divergence abs={abs_delta:.3g} rel={rel_delta:.3g}")
            status = Status.UNKNOWN          # degrade: do not trust as reference
        if ampl_feasible and not rescored_feasible:
            notes.append(f"ampl feasible but common scorer finds {n_conf} conflict(s)")
            status = Status.UNKNOWN          # cannot be silent ground truth

        sol = Solution(q=q, theta=theta, choice=(), objective=rescored_obj,
                       n_conflicts=n_conf, feasible=rescored_feasible)
        extra = {
            "provider_objective": provider_obj,
            "rescored_objective": rescored_obj,
            "objective_abs_delta": abs_delta,
            "objective_rel_delta": rel_delta if math.isfinite(rel_delta) else -1.0,
            "ampl_feasible": 1.0 if ampl_feasible else 0.0,
            "rescored_feasible": 1.0 if rescored_feasible else 0.0,
            "integrity_ok": 1.0 if integrity_ok else 0.0,
            "optimality_proven": 1.0 if optimality_proven else 0.0,
            "timeout_s": prov["timeout_s"],
        }
        if srnum is not None:
            extra["solve_result_num"] = float(srnum)
        if solver_time_s is not None:
            extra["solver_time_s"] = solver_time_s
        if abs_gap is not None:
            extra["mip_abs_gap"] = abs_gap
        if rel_gap is not None:
            extra["mip_rel_gap"] = rel_gap
        # The solver's own max constraint residual, when it reported a violation
        # table (in the model constraints' units — NOT a geometric distance).
        if mp_check.get("max_abs_residual") is not None:
            extra["max_model_constraint_residual"] = mp_check["max_abs_residual"]
        prov = {
            **prov,
            "ampl_solve_result": raw_status.strip()[:120],
            "solve_result_num": srnum,
            "solver_effective": self.solver,
            "solver_options": self.solver_options,
            "solver_options_echo": _scan_option_echo(text),
            "mp_solution_check": mp_check or None,
            "solver_version": solver_version or prov.get("solver_version"),
            "solver_driver_version": driver_version or prov.get("solver_driver_version"),
            "optimality_proven": bool(optimality_proven),
            "termination_reason": raw_status.strip()[:120],
            "status_kind": status.value,
            "notes": "; ".join(notes),
        }
        # Legacy field kept only for artifact compatibility (documented as deprecated).
        if self.solver == "couenne":
            prov["couenne_version"] = solver_version or prov.get("couenne_version")
        return Result(
            instance_name=inst.name, solver=SolverKind.CLASSICAL_ORIGINAL_AMPL, status=status,
            solution=sol, objective=rescored_obj, n_conflicts=n_conf,
            n_initial_conflicts=n_initial, feasible=rescored_feasible,
            wall_time_s=time.perf_counter() - t0, backend=f"ampl+{self.solver}",
            message=self._bounded_json(prov), extra=extra,
        )

    # ------------------------------------------------------------------ #
    def _provenance(self, inst, binary, mod_bytes, pre_bytes, dat_bytes, timeout) -> dict[str, Any]:
        return {
            "instance": inst.name, "source_path": inst.source_path,
            "ampl_binary": binary,
            # Neutral solver-identity fields (no solver-specific assumptions):
            "solver_requested": self.solver,
            "solver_effective": self.solver,   # confirmed after the run
            "solver_version": None,            # filled from the solve banner
            "solver_driver_version": None,     # filled from the solve banner
            "supervised_solver": SUPERVISED_SOLVER,
            "is_supervised_solver": self.solver == SUPERVISED_SOLVER,
            "ampl_version": _ampl_version(binary),   # best-effort; None if unknown
            "model_sha256": _sha256_bytes(mod_bytes),
            "preprocessing_sha256": _sha256_bytes(pre_bytes),
            "data_sha256": _sha256_bytes(dat_bytes),
            "timeout_s": float(timeout) if timeout is not None else None,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "platform": _platform_tag(),
        }

    def _bounded_json(self, obj: dict[str, Any]) -> str:
        return json.dumps(obj, sort_keys=True)[: self.max_output_bytes]

    def _status_result(self, inst, n_initial, t0, status, prov, *, message,
                        solver_time_s, status_kind=None) -> Result:
        prov = {**prov, "status_kind": status_kind or status.value, "message": message[:400]}
        return Result(
            instance_name=inst.name, solver=SolverKind.CLASSICAL_ORIGINAL_AMPL, status=status,
            solution=None, n_initial_conflicts=n_initial,
            wall_time_s=time.perf_counter() - t0, backend=f"ampl+{self.solver}",
            message=self._bounded_json(prov),
            extra={"timeout_s": prov.get("timeout_s", 0.0)},
        )

    def _unavailable(self, inst, n_initial, t0, timeout, *, reason) -> Result:
        prov = {"instance": inst.name, "solver": self.solver,
                "timeout_s": float(timeout) if timeout is not None else None,
                "status_kind": "unavailable", "message": reason[:400],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "platform": _platform_tag()}
        return Result(
            instance_name=inst.name, solver=SolverKind.CLASSICAL_ORIGINAL_AMPL,
            status=Status.UNAVAILABLE, solution=None, n_initial_conflicts=n_initial,
            wall_time_s=time.perf_counter() - t0, backend=f"ampl+{self.solver}",
            message=self._bounded_json(prov),
            extra={"timeout_s": float(timeout) if timeout is not None else None},
        )


def _platform_tag() -> str:
    import platform
    return f"{platform.system()}-{platform.machine()}-py{platform.python_version()}"


# Signatures AMPL emits when it CAN find its driver but the requested solver binary
# is missing or cannot be loaded — distinct from "ampl itself is absent".
_SOLVER_MISSING_RE = re.compile(
    r"can't find|cannot find|can't invoke|cannot invoke|no such file|not found|"
    r"cannot load|failed to (?:load|run)|command not found|Bad option",
    re.IGNORECASE,
)


def _detect_solver_unavailable(stdout: str, stderr: str, solver: str) -> str | None:
    """Return a short diagnostic if AMPL ran but the solver could not be loaded.

    Scans the (already-bounded) AMPL output for a solver-not-found signature near
    the solver name. Never a false success: on any match the run is reported as an
    explicit ``solver_unavailable`` error, not an optimum.
    """
    hay = f"{stdout}\n{stderr}"
    for line in hay.splitlines():
        if _SOLVER_MISSING_RE.search(line) and (
            solver.lower() in line.lower() or "solver" in line.lower()
        ):
            return line.strip()[:200]
    return None


def _detect_license_error(stdout: str, stderr: str) -> str | None:
    """Return a short diagnostic if the solver reports a LICENCE problem.

    Matches only the solver's own error MESSAGE; the licence file itself is never
    read, copied or logged. Any match => explicit ``license_error``, never success.
    """
    for line in f"{stdout}\n{stderr}".splitlines():
        if _LICENSE_ERROR_RE.search(line):
            return line.strip()[:200]
    return None


def _scan_solver_banner(stdout: str, solver: str) -> tuple[str | None, str | None]:
    """Best-effort (solver_version, driver_version) from the solve banner.

    Zero extra cost — scans stdout the solver already produced. Recognises both the
    ``-v`` banner form ``AMPL/Gurobi Optimizer [13.0.2] ... driver(20260624)`` and
    the solve-log form ``Gurobi 13.0.2: optimal solution``. Returns ``None`` for a
    field that is missing/unrecognised — never a fabricated version.
    """
    ver: str | None = None
    m = _BRACKET_VER_RE.search(stdout)
    if m:
        ver = m.group(1)
    else:
        name_re = re.compile(rf"{re.escape(solver)}\s+(\d+\.\d+(?:\.\d+)?)", re.IGNORECASE)
        m2 = name_re.search(stdout)
        if m2:
            ver = m2.group(1)
    dm = _DRIVER_VER_RE.search(stdout)
    return ver, (dm.group(1) if dm else None)


def _scan_option_echo(stdout: str) -> dict[str, str]:
    """Capture the driver's echoed effective options (proof of the mode used).

    e.g. lines like ``  pre:funcnonlinear = 1`` / ``  alg:feastol = 1e-09``.
    Best-effort; empty dict when the solver echoes nothing.
    """
    echo: dict[str, str] = {}
    for line in stdout.splitlines():
        m = _OPT_ECHO_RE.search(line)
        if m:
            echo[m.group(1)] = m.group(2)
    return echo


def _mp_check_failed(stdout: str) -> bool:
    """True when the MP driver's strict solution check FAILED (sol:chk:fail fatal)."""
    return "Solution check failed" in stdout


def _scan_mp_check(stdout: str) -> dict[str, Any]:
    """Parse the MP driver's constraint-residual table (model constraint residuals).

    Triggered by a "Tolerance violations" warning (non-fatal) OR a "Solution check
    failed" fatal report. Returns ``{"violations": [...], "max_abs_residual": float,
    "failed": bool}`` when a table was reported, else an empty dict. These are the
    solver's OWN residuals on the model constraints (e.g. the bilinear velocity
    definitions), reported in the constraints' units — not a geometric distance.
    """
    failed = _mp_check_failed(stdout)
    if "Tolerance violations" not in stdout and not failed:
        return {}
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        m = _MP_VIOL_RE.match(line)
        if not m:
            continue
        max_abs = _parse_float_or_none(m.group(2))
        rows.append({
            "type": m.group(1).strip(),
            "max_abs": max_abs,
            "max_rel": _parse_float_or_none(m.group(3)),
        })
    abs_vals = [r["max_abs"] for r in rows if r["max_abs"] is not None]
    if not abs_vals:
        # A table was printed but no residual could be parsed: report it explicitly so
        # downstream anchoring fails closed (never a silent pass).
        return {"violations": rows, "max_abs_residual": None, "unparsed": True, "failed": failed}
    return {"violations": rows, "max_abs_residual": max(abs_vals), "failed": failed}


def _scan_gaps(stdout: str) -> tuple[float | None, float | None]:
    """Best-effort (absolute, relative) MIP gap from the solver log."""
    a = _ABS_GAP_RE.search(stdout)
    r = _REL_GAP_RE.search(stdout)
    return _parse_float_or_none(a.group(1) if a else None), \
        _parse_float_or_none(r.group(1) if r else None)


def _parse_int_or_none(s: str | None) -> int | None:
    if s is None:
        return None
    try:
        return int(s.strip())
    except (ValueError, AttributeError):
        return None


def _parse_float_or_none(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        v = float(s.strip())
    except (ValueError, AttributeError):
        return None
    return v if math.isfinite(v) else None


def _amplpy_find(name: str) -> str | None:
    """Locate an amplpy-managed binary (``ampl``/solver), or ``None`` if amplpy is
    absent or the module is not installed. Never raises into the caller."""
    try:
        from amplpy import modules  # type: ignore
        found = modules.find(name)
    except Exception:
        return None
    return found if (found and Path(found).is_file()) else None


@lru_cache(maxsize=1)
def _amplpy_module_path() -> str | None:
    """The amplpy module bin PATH (all installed solver dirs), or ``None``."""
    try:
        from amplpy import modules  # type: ignore
        p = modules.path()
    except Exception:
        return None
    return p or None


@lru_cache(maxsize=8)
def _ampl_version(binary: str) -> str | None:
    """Best-effort ``ampl --version`` (cheap, cached, minimal env, no licence read).

    Returns the first version-like line, or ``None`` if the binary does not
    support ``--version`` or produces nothing recognisable. Failures are swallowed
    — a missing version must never break or fake a solve.
    """
    try:
        proc = subprocess.run(  # noqa: S603 - shell=False, list args, validated binary
            [binary, "--version"],
            capture_output=True, text=True, timeout=10,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        low = line.lower()
        # A real `ampl --version` banner names AMPL/Version and carries digits;
        # require that so stray solver/marker output is never mistaken for a version.
        if line and ("ampl" in low or "version" in low) and _VERSION_RE.search(line):
            return line[:120]
    return None
