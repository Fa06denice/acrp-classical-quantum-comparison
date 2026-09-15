"""Integrity validator for the versioned benchmark artifacts in ``results/``.

Deterministic, standard-library only. Fails if any committed artifact is
internally inconsistent, so a broken or fabricated result cannot silently enter
the repository or the CI. Checks:

* every simulator JSON is valid and its CSV has the same number of data rows;
* simulator provenance is present and internally consistent (a single
  ``git_commit`` and ``source_hash`` per file, ``git_dirty`` is ``False``, no
  empty provenance field);
* no numeric field is non-finite (NaN / Infinity);
* the IBM hardware artifacts match the integrity hashes recorded in
  ``results/hardware_manifest.json`` (legacy, byte-for-byte preserved — their
  provenance is documented by hash, not fabricated onto the rows).

The checks assert *internal* consistency, not equality with the current ``HEAD``
(later commits add data/docs, so the artifacts legitimately reference the source
commit at which they were generated).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"

# simulator artifacts: (json, csv). These MUST carry provenance.
SIM_PAIRS = [
    ("campaign_runs.json", "campaign_runs.csv"),
    ("campaign_summary.json", "campaign_summary.csv"),
    ("three_way.json", "three_way.csv"),
    ("scaling.json", "scaling.csv"),
]

_PROVENANCE_KEYS = ("git_commit", "source_hash", "git_dirty")

pytestmark = pytest.mark.skipif(
    not RESULTS.is_dir() or not (RESULTS / "campaign_runs.json").is_file(),
    reason="results/ artifacts not present (nothing to validate)",
)


def _csv_rows(path: Path) -> int:
    with path.open(newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1  # minus header


def _has_non_finite(obj) -> bool:
    if isinstance(obj, float):
        return math.isnan(obj) or math.isinf(obj)
    if isinstance(obj, dict):
        return any(_has_non_finite(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_non_finite(v) for v in obj)
    return False


@pytest.mark.parametrize(("json_name", "csv_name"), SIM_PAIRS)
def test_simulator_artifact_valid_and_consistent(json_name, csv_name):
    jpath, cpath = RESULTS / json_name, RESULTS / csv_name
    assert jpath.is_file() and cpath.is_file(), f"missing {json_name}/{csv_name}"

    rows = json.loads(jpath.read_text())  # invalid JSON -> raises here
    assert isinstance(rows, list) and rows, f"{json_name} is empty/not a list"

    # CSV and JSON must agree on the number of data rows
    assert _csv_rows(cpath) == len(rows), f"{json_name} row count != {csv_name}"

    # no NaN / Infinity anywhere
    assert not _has_non_finite(rows), f"{json_name} contains a non-finite number"

    # provenance present, non-empty, internally consistent
    commits, hashes, dirtys = set(), set(), set()
    for r in rows:
        for k in _PROVENANCE_KEYS:
            assert k in r, f"{json_name} row missing provenance field {k}"
        assert r["git_commit"], f"{json_name} has an empty git_commit"
        assert r["source_hash"], f"{json_name} has an empty source_hash"
        commits.add(r["git_commit"])
        hashes.add(r["source_hash"])
        dirtys.add(r["git_dirty"])
    assert len(commits) == 1, f"{json_name} mixes several git_commits: {commits}"
    assert len(hashes) == 1, f"{json_name} mixes several source_hashes: {hashes}"
    assert dirtys == {False}, f"{json_name} was generated from a dirty tree: {dirtys}"


def test_ibm_hardware_matches_manifest():
    manifest_path = RESULTS / "hardware_manifest.json"
    assert manifest_path.is_file(), "results/hardware_manifest.json missing"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["provenance_status"] == "legacy_preserved"
    for rel, meta in manifest["files"].items():
        path = RESULTS.parent / rel
        assert path.is_file(), f"hardware artifact {rel} missing"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == meta["sha256"], f"{rel} hash drift: {actual} != {meta['sha256']}"


def test_ibm_artifacts_not_replaced_by_simulator():
    """The IBM CSV must still describe a real QPU backend, not a sim-only run."""
    text = (RESULTS / "sim_vs_hardware.csv").read_text()
    assert "ibm" in text.lower(), "sim_vs_hardware.csv no longer references an IBM backend"
