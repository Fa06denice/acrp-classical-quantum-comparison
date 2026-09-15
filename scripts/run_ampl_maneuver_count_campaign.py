#!/usr/bin/env python3
"""Sequential AMPL/Gurobi campaign for the moved-aircraft objective.

This never edits ``Code/``.  It loads the historical model verbatim and layers
the packaged indicator objective on top.  One JSON per instance is written
atomically; an existing file is verified before resume.  Timeouts are honest
non-anchors.  No IBM path is imported or reachable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from acrpq.classical.discrete_enum import exhaustive_discrete_optimum
from acrpq.classical.original_ampl_maneuver_count import (
    MODEL_VARIANT,
    OBJECTIVE_ID,
    SUPERVISED_OPTIONS,
    OriginalAMPLManeuverCountSolver,
    _addon_path,
)
from acrpq.io.loader import InstanceLoader

SCHEMA = "acrpq-ampl-maneuver-count-campaign/1"
PROTOCOL_ID = "ampl_gurobi_maneuver_count_v1"
DEFAULT_INSTANCES = tuple(f"CP_{n}" for n in range(3, 21))


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(value, fh, indent=2, sort_keys=True, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _record_hash(record: dict) -> str:
    return _sha(_canonical({k: v for k, v in record.items() if k != "record_sha256"}))


def _validate(record: dict, *, instance: str, expected_source_commit: str) -> None:
    if record.get("schema") != SCHEMA or record.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"{instance}: wrong schema/protocol")
    if record.get("instance") != instance or record.get("record_sha256") != _record_hash(record):
        raise ValueError(f"{instance}: corrupt or mismatched artifact")
    if record.get("source_provenance", {}).get("source_commit") != expected_source_commit:
        raise ValueError(f"{instance}: artifact belongs to a different source commit")


def _finite_or_none(value: Any) -> float | None:
    """Persist failed/timeout numeric sentinels as JSON null, never NaN/Inf."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _result_record(name: str, result, discrete, *, source: str, source_provenance: dict) -> dict:
    message = json.loads(result.message) if result.message else {}
    # Local executable paths are useful interactively but are neither scientific
    # provenance nor portable/PII-safe artifact content.  Versions and hashes stay.
    message.pop("ampl_binary", None)
    message.pop("source_path", None)
    certified = bool(
        result.status.value == "optimal"
        and result.feasible
        and result.extra.get("integer_optimality_certified") == 1.0
        and result.extra.get("objective_integrity_ok") == 1.0
        and result.extra.get("indicator_link_ok") == 1.0
        and result.extra.get("strict_mp_check_passed") == 1.0
    )
    discrete_certified = discrete.status == "optimal" and discrete.feasible
    result_objective = _finite_or_none(result.objective)
    discrete_objective = _finite_or_none(discrete.optimum)
    comparison = None
    if certified and discrete_certified:
        assert result_objective is not None and discrete_objective is not None
        comparison = {
            "continuous_maneuver_count": result_objective,
            "discrete_K3_maneuver_count": discrete_objective,
            "discretization_difference_count": discrete_objective - result_objective,
            "same_primary_optimum": discrete_objective == result_objective,
            "interpretation": "difference in number of moved aircraft; not an amplitude gap",
        }
    record = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "model_variant": MODEL_VARIANT,
        "objective_id": OBJECTIVE_ID,
        "instance": name,
        "source": source,
        "source_provenance": source_provenance,
        "solver": "gurobi",
        "solver_options": SUPERVISED_OPTIONS,
        "status": result.status.value,
        "certified_integer_optimum": certified,
        "objective": result_objective,
        "best_bound": (
            result_objective - result.extra["mip_abs_gap"]
            if result_objective is not None
            and _finite_or_none(result.extra.get("mip_abs_gap")) is not None
            else None
        ),
        "abs_gap": _finite_or_none(result.extra.get("mip_abs_gap")),
        "rel_gap": _finite_or_none(result.extra.get("mip_rel_gap")),
        "feasible_common_geometry": result.feasible,
        "n_initial_conflicts": result.n_initial_conflicts,
        "n_residual_conflicts": result.n_conflicts,
        "q": list(result.solution.q) if result.solution else None,
        "theta": list(result.solution.theta) if result.solution else None,
        "moved": message.get("moved"),
        "historical_quadratic_objective": _finite_or_none(
            result.extra.get("historical_quadratic_objective")
        ),
        "indicator_link_ok": bool(result.extra.get("indicator_link_ok", 0.0)),
        "objective_integrity_ok": bool(result.extra.get("objective_integrity_ok", 0.0)),
        "strict_mp_check_passed": bool(result.extra.get("strict_mp_check_passed", 0.0)),
        "solver_time_s": _finite_or_none(result.extra.get("solver_time_s")),
        "wall_time_s": _finite_or_none(result.wall_time_s),
        "provenance": message,
        "discrete_K3": {
            "status": discrete.status,
            "certified": discrete_certified,
            "objective": discrete_objective,
            "search_space": discrete.search_space,
            "n_optima": discrete.n_optima,
            "historical_quadratic_objective": _finite_or_none(
                discrete.secondary.get("quadratic_control_cost")
            ),
        },
        "comparison": comparison,
    }
    record["record_sha256"] = _record_hash(record)
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    ap.add_argument("--out", default="results/ampl_gurobi_maneuver_count_v1")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SystemExit("--timeout must be finite and > 0")
    out = Path(args.out)
    loader = InstanceLoader()
    if args.verify:
        stored_manifest = json.loads((out / "manifest.json").read_text())
        source_provenance = stored_manifest["source_provenance"]
    else:
        from acrpq.classical.ampl_campaign import git_info

        git = git_info()
        source_provenance = {
            "source_commit": git["git_commit"],
            "scientific_code_dirty": git["scientific_code_dirty"],
            "scientific_code_paths": git["scientific_code_paths"],
            "repository_dirty": git["repository_dirty"],
        }
        if source_provenance["scientific_code_dirty"]:
            raise SystemExit("refusing a citable campaign from dirty scientific code")
    rows = []
    if not args.verify:
        solver = OriginalAMPLManeuverCountSolver(timeout_s=args.timeout)
        probe = solver.check_solver()
        if not probe.get("ok"):
            raise SystemExit(f"AMPL/Gurobi unavailable: {probe.get('status_kind')}")
    for name in args.instances:
        path = out / f"{name}.json"
        if path.exists():
            record = json.loads(path.read_text())
            _validate(
                record,
                instance=name,
                expected_source_commit=source_provenance["source_commit"],
            )
        elif args.verify:
            raise SystemExit(f"missing {path}")
        else:
            inst = loader.load(name)
            dat, source = loader.read_dat_text(name)
            result = solver.solve(inst, dat_text=dat)
            discrete = exhaustive_discrete_optimum(inst, objective_id=OBJECTIVE_ID, n_theta=3)
            record = _result_record(
                name, result, discrete, source=source, source_provenance=source_provenance
            )
            _atomic_json(path, record)
            _validate(
                record,
                instance=name,
                expected_source_commit=source_provenance["source_commit"],
            )
            print(name, record["status"], record["objective"], "anchor=" + str(record["certified_integer_optimum"]))
        rows.append(record)

    summary_rows = [{
        "instance": r["instance"], "status": r["status"],
        "certified_integer_optimum": r["certified_integer_optimum"],
        "continuous_maneuver_count": r["objective"],
        "discrete_K3_maneuver_count": r["discrete_K3"]["objective"],
        "difference_count": r["comparison"]["discretization_difference_count"] if r["comparison"] else None,
        "historical_quadratic_objective": r["historical_quadratic_objective"],
        "solver_time_s": r["solver_time_s"], "wall_time_s": r["wall_time_s"],
    } for r in rows]
    manifest = {
        "schema": SCHEMA, "protocol_id": PROTOCOL_ID, "objective_id": OBJECTIVE_ID,
        "solver": "gurobi", "solver_options": SUPERVISED_OPTIONS,
        "historical_model_sha256": _sha(Path("Code/NC_ACRP.mod").read_bytes()),
        "objective_addon_sha256": _sha(_addon_path().read_bytes()),
        "source_provenance": source_provenance,
        "instances": list(args.instances),
        "n_certified": sum(r["certified_integer_optimum"] for r in rows),
        "files": {r["instance"]: r["record_sha256"] for r in rows},
    }
    manifest["manifest_sha256"] = _sha(_canonical(manifest))
    if not args.verify:
        _atomic_json(out / "summary.json", {"rows": summary_rows})
        _atomic_json(out / "manifest.json", manifest)
        with (out / "summary.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        stored = stored_manifest
        if stored != manifest:
            raise SystemExit("manifest mismatch")
        print("OK", stored["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
