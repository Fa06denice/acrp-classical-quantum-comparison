"""EXPERIMENTAL — honest provenance block for experimental artefacts (schema v2).

An artefact is generated FROM a scientific commit, in a (preferably clean) worktree, and is
committed AFTER its generation. The commit that contains the artefact therefore cannot be
known at generation time and must never be written into the artefact itself: it lives in a
separate sidecar (``PROVENANCE_SIDECAR.json``) written after the artefact commit.

Fields:

* ``scientific_source_commit``: HEAD of the worktree the code was executed from;
* ``artifact_generation_commit``: identical to the above (kept distinct on purpose: the
  scientific code and the generation run may differ in future protocols);
* ``artifact_packaging_commit`` / ``review_commit``: always ``None`` at generation; filled only
  by the sidecar;
* ``repository_dirty_at_generation``: ``git status --porcelain`` non-empty (any path);
* ``scientific_code_dirty_at_generation``: dirty paths under ``src/``, ``scripts/`` or ``Code/``;
* ``scientific_hash_exclusions``: fields that are legitimately non-deterministic and must be
  excluded from any scientific hash comparison between two clean reproductions.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROVENANCE_SCHEMA = "acrpq-experimental-provenance/2"
FLAGS = {"experimental": True, "official_benchmark": False, "real_qpu": False}
SCIENTIFIC_HASH_EXCLUSIONS = (
    "wall_time_s", "total_wall_time_s", "timing_s", "compile_seconds", "generated_utc", "recorded_utc",
    "python", "platform", "provenance", "sha256", "bytes", "files", "outputs", "runs",
    "local_exact_refusal_reason",
    # timing fields of the counter-proof and execution-identity fields of the hash chains
    "elapsed_s", "elapsed_s_part1", "elapsed_s_part2", "preregistered_utc", "preregistration_hash",
    "last_round_hash", "parent_round_hash", "source_commit",
)


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def provenance_block(root: Path) -> dict[str, Any]:
    root = Path(root)
    head = _git(root, "rev-parse", "HEAD") or "unknown"
    status = _git(root, "status", "--porcelain")
    dirty_paths = [line[3:] for line in status.splitlines() if line.strip()]
    sci_dirty = [p for p in dirty_paths if p.startswith(("src/", "scripts/", "Code/"))]
    return {
        "schema": PROVENANCE_SCHEMA,
        "scientific_source_commit": head,
        "artifact_generation_commit": head,
        "artifact_packaging_commit": None,
        "review_commit": None,
        "repository_dirty_at_generation": bool(dirty_paths),
        "scientific_code_dirty_at_generation": bool(sci_dirty),
        "dirty_paths_at_generation": dirty_paths[:50],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "scientific_hash_exclusions": list(SCIENTIFIC_HASH_EXCLUSIONS),
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strip_nondeterministic(obj: Any, exclusions: tuple[str, ...] = SCIENTIFIC_HASH_EXCLUSIONS) -> Any:
    """Recursively drop excluded keys so two clean reproductions can be hash-compared."""
    if isinstance(obj, dict):
        return {k: strip_nondeterministic(v, exclusions) for k, v in obj.items() if k not in exclusions}
    if isinstance(obj, list):
        return [strip_nondeterministic(v, exclusions) for v in obj]
    return obj


def scientific_hash(doc: Any) -> str:
    """sha256 of the canonical JSON of ``doc`` without the non-deterministic fields."""
    text = json.dumps(strip_nondeterministic(doc), sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def write_compact_json(path: Path, doc: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str) + "\n")
