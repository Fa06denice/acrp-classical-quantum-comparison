"""Historical AMPL geometry with the alternative moved-aircraft objective.

This is deliberately a separate solver leg.  It loads ``Code/NC_ACRP.mod``
verbatim, then adds :mod:`classical/models/NC_ACRP_maneuver_count.mod`.  The
historical objective is excluded from the named AMPL problem; all historical
variables and separation constraints are retained.

The primary objective is the integer number of aircraft whose continuous
control differs from ``(q=1, theta=0)``.  The historical quadratic control cost
is reported only as a secondary metric.  As for the original runner, every
returned control is rechecked by the common geometry and no solver status is
trusted without a complete finite primal.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .. import geometry
from ..model import Instance, Result, Solution, Status
from .original_ampl import OriginalAMPLError, OriginalAMPLSolver, _sha256_bytes

OBJECTIVE_ID = "maneuver_count_v1"
MODEL_VARIANT = "original_geometry_maneuver_count_v1"
SUPERVISED_OPTIONS = (
    "pre:funcnonlinear=1 alg:feastol=1e-9 mip:gap=1e-6 mip:gapabs=1e-6 "
    "return_mipgap=7 sol:chk:feastol=1e-8 sol:chk:fail"
)
_Y_BEGIN = "ACRPQ_MOVED_BEGIN"
_Y_END = "ACRPQ_MOVED_END"


def _addon_path() -> Path:
    path = Path(__file__).resolve().parent / "models" / "NC_ACRP_maneuver_count.mod"
    if not path.is_file():
        raise OriginalAMPLError(f"missing packaged maneuver-count model: {path}")
    return path.resolve()


@dataclass
class OriginalAMPLManeuverCountSolver(OriginalAMPLSolver):
    """Run Gurobi on the original continuous geometry with ``min sum moved``."""

    solver_options: str = SUPERVISED_OPTIONS

    def _run_script(self, mod: Path, pre: Path, dat: Path, inst: Instance) -> str:
        def amplstr(path: Path) -> str:
            return str(path).replace("\\", "\\\\").replace('"', '\\"')

        addon = _addon_path()
        return f'''reset;
model "{amplstr(mod)}";
model "{amplstr(addon)}";
problem NC_ACRP_COUNT: ManeuverCount,q,theta,vrx,vry,z,moved,
    bin11,bin12,cvrx,cvry,
    bin311,bin312,bin321,bin322,
    bin331,bin332,bin341,bin342,
    LinkThetaUpper,LinkThetaLower,LinkSpeedUpper,LinkSpeedLower;
option solver {self.solver};
{self._options_line()}data "{amplstr(dat)}";
let A := 1..n;
let wd := {inst.w!r};
include "{amplstr(pre)}";
problem NC_ACRP_COUNT;
solve NC_ACRP_COUNT;
printf "absmipgap=%.12g, relmipgap=%.12g\\n", ManeuverCount.absmipgap, ManeuverCount.relmipgap;
printf "ACRPQ_STATUS_BEGIN\\n";
printf "%s\\n", solve_result;
printf "ACRPQ_STATUS_END\\n";
printf "ACRPQ_SRNUM_BEGIN\\n";
printf "%d\\n", solve_result_num;
printf "ACRPQ_SRNUM_END\\n";
printf "ACRPQ_OBJECTIVE_BEGIN\\n";
printf "%.12g\\n", ManeuverCount;
printf "ACRPQ_OBJECTIVE_END\\n";
printf "ACRPQ_SOLVETIME_BEGIN\\n";
printf "%.6f\\n", _solve_time;
printf "ACRPQ_SOLVETIME_END\\n";
printf "ACRPQ_Q_BEGIN\\n";
for {{i in A}} printf "%d,%.12g\\n", i, q[i];
printf "ACRPQ_Q_END\\n";
printf "ACRPQ_THETA_BEGIN\\n";
for {{i in A}} printf "%d,%.12g\\n", i, theta[i];
printf "ACRPQ_THETA_END\\n";
printf "{_Y_BEGIN}\\n";
for {{i in A}} printf "%d,%.12g\\n", i, moved[i];
printf "{_Y_END}\\n";
'''

    def _provenance(self, inst, binary, mod_bytes, pre_bytes, dat_bytes, timeout):
        prov = super()._provenance(
            inst, binary, mod_bytes, pre_bytes, dat_bytes, timeout
        )
        addon = _addon_path()
        return {
            **prov,
            "objective_id": OBJECTIVE_ID,
            "model_variant": MODEL_VARIANT,
            "historical_model_unchanged": True,
            "objective_addon_sha256": _sha256_bytes(addon.read_bytes()),
        }

    @staticmethod
    def _parse_moved(text: str, n: int) -> tuple[int, ...]:
        if text.count(_Y_BEGIN) != 1 or text.count(_Y_END) != 1:
            raise OriginalAMPLError("moved: expected exactly one complete marker block")
        body = text.split(_Y_BEGIN, 1)[1].split(_Y_END, 1)[0]
        seen: dict[int, int] = {}
        for raw in body.splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                raise OriginalAMPLError(f"moved: malformed row {line!r}")
            idx, value = int(parts[0]), float(parts[1])
            if idx in seen or not 1 <= idx <= n or not math.isfinite(value):
                raise OriginalAMPLError(f"moved: invalid index/value {line!r}")
            rounded = round(value)
            if rounded not in (0, 1) or abs(value - rounded) > 1e-6:
                raise OriginalAMPLError(f"moved: non-binary value {value!r}")
            seen[idx] = int(rounded)
        if set(seen) != set(range(1, n + 1)):
            raise OriginalAMPLError(f"moved: expected {n} unique values, got {len(seen)}")
        return tuple(seen[i] for i in range(1, n + 1))

    def _parse_and_score(self, inst, n_initial, t0, proc, prov) -> Result:
        base = super()._parse_and_score(inst, n_initial, t0, proc, prov)
        if base.solution is None:
            return base
        try:
            moved = self._parse_moved(proc["stdout"], inst.n)
        except (OriginalAMPLError, ValueError) as exc:
            return replace(
                base,
                status=Status.ERROR,
                solution=None,
                objective=None,
                feasible=False,
                message=self._bounded_json({**prov, "objective_id": OBJECTIVE_ID,
                                            "status_kind": "parse_error",
                                            "message": str(exc)[:400]}),
            )

        q, theta = base.solution.q, base.solution.theta
        link_tol = max(self.objective_tolerance, 1e-9)
        link_ok = all(
            flag == 1 or (abs(qi - 1.0) <= link_tol and abs(ti) <= link_tol)
            for flag, qi, ti in zip(moved, q, theta, strict=True)
        )
        provider = base.extra.get("provider_objective")
        objective = float(sum(moved))
        integrity_ok = (
            isinstance(provider, (int, float))
            and math.isfinite(float(provider))
            and abs(float(provider) - objective) <= self.objective_tolerance
            and link_ok
        )
        conflicts = geometry.count_conflicts(inst, q, theta)
        feasible = conflicts == 0

        try:
            meta: dict[str, Any] = json.loads(base.message)
        except (TypeError, json.JSONDecodeError):
            meta = {}
        raw_status = str(meta.get("ampl_solve_result", ""))
        srnum = meta.get("solve_result_num")
        status = self._map_status(srnum if isinstance(srnum, int) else None, raw_status)
        if not integrity_ok or not feasible:
            status = Status.UNKNOWN

        abs_gap = base.extra.get("mip_abs_gap")
        integer_certified = (
            status is Status.OPTIMAL
            and isinstance(abs_gap, (int, float))
            and math.isfinite(float(abs_gap))
            and float(abs_gap) < 0.5
        )
        strict_mp_check_passed = (
            "sol:chk:fail" in self.solver_options
            and "sol:chk:feastol=1e-8" in self.solver_options
        )
        integer_certified = integer_certified and strict_mp_check_passed
        historical = geometry.objective(inst, q, theta)
        sol = Solution(
            q=q,
            theta=theta,
            choice=(),
            objective=objective,
            n_conflicts=conflicts,
            feasible=feasible,
        )
        objective_abs_delta = abs(float(provider) - objective) if isinstance(
            provider, (int, float)
        ) else None
        objective_rel_delta = (
            objective_abs_delta / max(abs(float(provider)), abs(objective), 1.0)
            if objective_abs_delta is not None else None
        )
        extra = {
            **base.extra,
            "provider_objective": float(provider) if isinstance(provider, (int, float)) else None,
            "rescored_objective": objective,
            "objective_abs_delta": objective_abs_delta,
            "objective_rel_delta": objective_rel_delta,
            "maneuver_count": objective,
            "historical_quadratic_objective": historical,
            "integrity_ok": 1.0 if integrity_ok else 0.0,
            "optimality_proven": 1.0 if integer_certified else 0.0,
            "objective_integrity_ok": 1.0 if integrity_ok else 0.0,
            "indicator_link_ok": 1.0 if link_ok else 0.0,
            "strict_mp_check_passed": 1.0 if strict_mp_check_passed else 0.0,
            "integer_optimality_certified": 1.0 if integer_certified else 0.0,
        }
        meta.update({
            "objective_id": OBJECTIVE_ID,
            "model_variant": MODEL_VARIANT,
            "historical_model_unchanged": True,
            "moved": list(moved),
            "indicator_link_ok": link_ok,
            "strict_mp_check_passed": strict_mp_check_passed,
            "objective_integrity_ok": integrity_ok,
            "integer_optimality_certified": integer_certified,
            "status_kind": status.value,
        })
        return replace(
            base,
            status=status,
            solution=sol,
            objective=objective,
            n_conflicts=conflicts,
            feasible=feasible,
            message=self._bounded_json(meta),
            extra=extra,
        )
