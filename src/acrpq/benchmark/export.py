"""Result export and reproducibility metadata.

Every benchmark run is captured as a :class:`RunRecord` containing the common
metrics, optional quantum metrics, and a :class:`ReproInfo` block with
everything needed to reproduce it: seed, package/python versions, dependency
versions, backend, shots, git commit, timestamp and platform.

Records serialise to JSON (full structure) and CSV (a flat, stable column set
for spreadsheets/plots).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from .. import __version__
from .._optional import versions as dep_versions
from .metrics import CommonMetrics, QuantumMetrics


@dataclass(frozen=True)
class ReproInfo:
    """Everything needed to reproduce a single benchmark run."""

    seed: int | None
    package_version: str
    python_version: str
    backend: str | None
    shots: int | None
    deps: dict[str, str]
    timestamp_utc: str
    git_commit: str | None
    git_dirty: bool | None
    source_hash: str | None
    platform: str

    @staticmethod
    def capture(
        *, seed: int | None = None, backend: str | None = None, shots: int | None = None
    ) -> "ReproInfo":
        git_commit, git_dirty = _git_state()
        return ReproInfo(
            seed=seed,
            package_version=__version__,
            python_version=platform.python_version(),
            backend=backend,
            shots=shots,
            deps=dep_versions(),
            timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            git_commit=git_commit,
            git_dirty=git_dirty,
            source_hash=_source_hash(),
            platform=platform.platform(),
        )


_REPRO_COLUMNS = [
    "seed",
    "package_version",
    "python_version",
    "git_commit",
    "git_dirty",
    "source_hash",
    "timestamp_utc",
]


def _repro_row(repro: ReproInfo) -> dict[str, Any]:
    """Return the stable, scalar provenance fields shared by benchmark exports."""
    return {
        "seed": repro.seed,
        "package_version": repro.package_version,
        "python_version": repro.python_version,
        "git_commit": repro.git_commit,
        "git_dirty": repro.git_dirty,
        "source_hash": repro.source_hash,
        "timestamp_utc": repro.timestamp_utc,
    }


@lru_cache(maxsize=1)
def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        if commit.returncode != 0:
            return None, None
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        return commit.stdout.strip() or None, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


@lru_cache(maxsize=1)
def _source_hash() -> str | None:
    """Hash the package sources so dirty/untracked experiment code is identifiable."""
    try:
        package_root = Path(__file__).resolve().parents[1]
        files = sorted(package_root.rglob("*.py")) + [package_root / "py.typed"]
        digest = hashlib.sha256()
        for path in files:
            if not path.is_file():
                continue
            digest.update(path.relative_to(package_root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()
    except OSError:
        return None


@dataclass
class RunRecord:
    """One solver run on one scenario, with metrics and reproducibility info."""

    scenario_id: str
    instance_name: str
    family: str
    n: int
    n_pairs: int
    grid_k: int
    solver_kind: str
    status: str
    common: CommonMetrics
    repro: ReproInfo
    quantum: QuantumMetrics | None = None
    extra: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Serialisation helpers
# --------------------------------------------------------------------------- #
def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    return obj


def write_json(records: list[RunRecord], path: str | Path) -> Path:
    """Write the full structured records to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([_jsonable(r) for r in records], indent=2), encoding="utf-8")
    return p


_CSV_COLUMNS = [
    "scenario_id",
    "instance_name",
    "family",
    "n",
    "n_pairs",
    "grid_k",
    "solver_kind",
    "status",
    "objective",
    "feasible",
    "n_initial_conflicts",
    "n_resolved_conflicts",
    "n_residual_conflicts",
    "resolution_rate",
    "max_violation_depth",
    "total_heading_change",
    "total_speed_change",
    "solve_time_s",
    "backend",
    "shots",
    "qaoa_reps",
    "n_qubits",
    "lambda_pen",
    "lambda_oh",
    "circuit_depth",
    "two_qubit_gates",
    "optimizer_evals",
    "approx_ratio",
    "best_prob",
    "seed",
    "package_version",
    "python_version",
    "qiskit_version",
    "pyomo_version",
    "git_commit",
    "git_dirty",
    "source_hash",
    "timestamp_utc",
]


def _row(rec: RunRecord) -> dict[str, Any]:
    c = rec.common
    q = rec.quantum
    repro = rec.repro
    row: dict[str, Any] = {
        "scenario_id": rec.scenario_id,
        "instance_name": rec.instance_name,
        "family": rec.family,
        "n": rec.n,
        "n_pairs": rec.n_pairs,
        "grid_k": rec.grid_k,
        "solver_kind": rec.solver_kind,
        "status": rec.status,
        "objective": _num(c.objective),
        "feasible": c.feasible,
        "n_initial_conflicts": c.n_initial_conflicts,
        "n_resolved_conflicts": c.n_resolved_conflicts,
        "n_residual_conflicts": c.n_residual_conflicts,
        "resolution_rate": _num(c.resolution_rate),
        "max_violation_depth": _num(c.max_violation_depth),
        "total_heading_change": _num(c.total_heading_change),
        "total_speed_change": _num(c.total_speed_change),
        "solve_time_s": _num(c.solve_time_s),
        "backend": repro.backend,
        "shots": repro.shots,
        "qaoa_reps": q.qaoa_reps if q else "",
        "n_qubits": q.n_qubits if q else "",
        "lambda_pen": _num(q.lambda_pen) if q else "",
        "lambda_oh": _num(q.lambda_oh) if q else "",
        "circuit_depth": q.circuit_depth if q else "",
        "two_qubit_gates": q.two_qubit_gates if q else "",
        "optimizer_evals": q.optimizer_evals if q else "",
        "approx_ratio": _num(q.approx_ratio) if q and q.approx_ratio is not None else "",
        "best_prob": _num(q.best_prob) if q and q.best_prob is not None else "",
        "seed": repro.seed,
        "package_version": repro.package_version,
        "python_version": repro.python_version,
        "qiskit_version": repro.deps.get("qiskit", "absent"),
        "pyomo_version": repro.deps.get("pyomo", "absent"),
        "git_commit": repro.git_commit or "",
        "git_dirty": repro.git_dirty if repro.git_dirty is not None else "",
        "source_hash": repro.source_hash or "",
        "timestamp_utc": repro.timestamp_utc,
    }
    return row


def _num(v: float | None) -> Any:
    if v is None:
        return ""
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return ""
    return v


def write_csv(records: list[RunRecord], path: str | Path) -> Path:
    """Write a flat CSV with a stable column order."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for rec in records:
            writer.writerow(_row(rec))
    return p
