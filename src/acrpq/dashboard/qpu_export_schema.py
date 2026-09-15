"""Versioned scientific QPU-benchmark export schema + parity + validator (Phase 5.6).

Defines one versioned row schema for the (future) QPU benchmark export, a full JSON⇄CSV
parity check, and an integrity validator. Wall-clock timings are published but excluded
from the reproducible scientific hash. The offline export MUST carry the honest status
``synthetic / plumbing-validation / not quantum evidence`` — never presented as hardware.
No IBM; pure serialisation/validation.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from typing import Any

EXPORT_SCHEMA = "acrpq-qpu-benchmark-export/1"
OFFLINE_STATUS = "synthetic / plumbing-validation / not quantum evidence"

# ordered columns — identity, parameters, states/times, results, provenance
EXPORT_COLUMNS: tuple[str, ...] = (
    # identity
    "batch_id", "batch_config_hash", "job_config_hash", "local_run_id", "provider_job_id_hash",
    "protocol_id", "instance", "family", "objective_id", "n_theta", "n_q", "n_qubits",
    "qubo_sha256", "logical_circuit_sha256", "isa_sha256", "backend_requested",
    "backend_effective",
    # parameters
    "qaoa_reps", "angle_strategy", "optimizer", "shots", "transpiler_seed",
    "optimization_level",
    # states / times (wall times published, excluded from the scientific hash)
    "final_state", "prepared_utc", "submitted_utc", "provider_start_utc", "provider_end_utc",
    "queue_seconds", "provider_seconds", "qpu_seconds", "wall_seconds",
    # results (distinct candidates, never conflated)
    "best_raw_bitstring", "modal_bitstring", "best_onehot_valid_bitstring",
    "best_feasible_bitstring", "onehot_valid_mass", "feasibility_rate", "primary_cost",
    "qubo_energy", "residual_conflicts", "delta_vs_reference", "comparability", "warnings",
    # provenance / honesty
    "data_source", "official",
)

_HASH_EXCLUDED_COLUMNS = frozenset({
    "local_run_id", "provider_job_id_hash", "prepared_utc", "submitted_utc",
    "provider_start_utc", "provider_end_utc", "queue_seconds", "provider_seconds",
    "wall_seconds",
})

_REQUIRED_PROVENANCE = ("source_commit", "scientific_code_dirty", "repository_dirty",
                        "versions", "data_source")
_ALLOWED_TOP_LEVEL = frozenset({"schema", "rows", "provenance", "export_sha256"})
_SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ExportError(ValueError):
    """The export is structurally invalid or fails integrity; fail closed."""


def _reject_nonfinite(v: Any, where: str) -> None:
    if isinstance(v, float) and not math.isfinite(v):
        raise ExportError(f"non-finite value at {where}")


def assert_row_schema(row: dict) -> None:
    """Every export column present, JSON-native, and finite."""
    missing = [c for c in EXPORT_COLUMNS if c not in row]
    if missing:
        raise ExportError(f"row missing columns: {missing}")
    extra = [c for c in row if c not in EXPORT_COLUMNS]
    if extra:
        raise ExportError(f"row has unknown columns: {extra}")
    for c in EXPORT_COLUMNS:
        v = row[c]
        _reject_nonfinite(v, c)
        if not isinstance(v, (str, int, float, bool, type(None))):
            raise ExportError(f"column {c} must be a JSON scalar; got {type(v).__name__}")


def empty_row() -> dict:
    """A row with every export column present and None — fill the known fields, leave the
    rest None (the schema permits None)."""
    return {c: None for c in EXPORT_COLUMNS}


def _csv_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def rows_to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(EXPORT_COLUMNS)
    for r in rows:
        assert_row_schema(r)
        w.writerow([_csv_cell(r[c]) for c in EXPORT_COLUMNS])
    return buf.getvalue()


def assert_json_csv_parity(rows: list[dict], csv_text: str) -> None:
    """Every JSON row's every column equals the CSV cell (string-projected). Fail closed on
    any divergence — no column may exist only in one representation."""
    reader = list(csv.reader(io.StringIO(csv_text)))
    if not reader or tuple(reader[0]) != EXPORT_COLUMNS:
        raise ExportError("CSV header does not match the export schema columns")
    if len(reader) - 1 != len(rows):
        raise ExportError(f"CSV has {len(reader) - 1} rows, JSON has {len(rows)}")
    for i, (r, cells) in enumerate(zip(rows, reader[1:])):
        for c, cell in zip(EXPORT_COLUMNS, cells):
            if _csv_cell(r[c]) != cell:
                raise ExportError(f"row {i} column {c}: JSON {r[c]!r} != CSV {cell!r}")


def canonical_export_sha256(doc: dict) -> str:
    """Hash over the scientific content only — volatile ids and wall-times are excluded, so
    the same science reproduces the same hash regardless of timing/run-ids."""
    rows = [{k: v for k, v in r.items() if k not in _HASH_EXCLUDED_COLUMNS}
            for r in doc.get("rows", [])]
    # repository_dirty IS bound into the hash (red-team #1): validate_export refuses an
    # official export whose provenance is dirty, so leaving it unhashed would let a
    # dirty->clean flip launder a dirty official export while keeping a valid stored hash.
    prov = {k: v for k, v in doc.get("provenance", {}).items()
            if k not in ("generated_artifact_commit",)}
    payload = {"schema": doc.get("schema"), "rows": rows, "provenance": prov}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


# local absolute path (unix single- or multi-segment "/etc", "/a/b", or windows "C:\a"),
# only when preceded by a JSON delimiter so "n/a"-style substrings don't false-positive;
# and common secret formats.
_PATH_RE = re.compile(
    r'(?:^|["\s,:=\[(])(?:/[A-Za-z0-9._~-]{2,}(?:/[^"\s,\])]*)?|[A-Za-z]:\\\\[^"\s]*)')
_SECRET_RE = re.compile(
    r'QISKIT_IBM_TOKEN|api[_-]?token|Bearer\s|ghp_[A-Za-z0-9]{16,}|gho_[A-Za-z0-9]{16,}|'
    r'AKIA[0-9A-Z]{16}|-----BEGIN|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}')


def validate_export(doc: dict, *, require_synthetic_offline: bool = False,
                    strict: bool = False) -> None:
    """Integrity validator. Always: schema version, no unknown top-level fields, row schema,
    provenance completeness + types, honest data-source label, official-export sanity
    (non-empty, clean provenance), a content-hash check when present, and a best-effort scan
    for local absolute paths / secret formats. ``strict=True`` (scientific/official exports)
    ADDITIONALLY makes ``export_sha256`` MANDATORY and well-formed. ``require_synthetic_offline``
    forces an OFFLINE export to declare ``data_source=='synthetic_offline'``.

    Note: offline provenance is not cryptographically provable; the honesty checks enforce a
    caller's declaration, and the path/secret scan is best-effort, not a proof of absence."""
    if not isinstance(doc, dict):
        raise ExportError("export must be an object")
    if doc.get("schema") != EXPORT_SCHEMA:
        raise ExportError(f"unknown export schema {doc.get('schema')!r}")
    unknown_top = set(doc) - _ALLOWED_TOP_LEVEL
    if unknown_top:
        raise ExportError(f"unknown top-level fields: {sorted(unknown_top)}")
    rows = doc.get("rows")
    if not isinstance(rows, list):
        raise ExportError("export.rows must be a list")
    for r in rows:
        assert_row_schema(r)
    prov = doc.get("provenance")
    if not isinstance(prov, dict):
        raise ExportError("export.provenance must be an object")
    for key in _REQUIRED_PROVENANCE:
        if key not in prov:
            raise ExportError(f"provenance missing {key!r}")
    # provenance TYPE validation
    for flag in ("scientific_code_dirty", "repository_dirty"):
        if not isinstance(prov.get(flag), bool):
            raise ExportError(f"provenance.{flag} must be a bool")
    if not isinstance(prov.get("versions"), dict):
        raise ExportError("provenance.versions must be an object")
    if not isinstance(prov.get("source_commit"), str) or not prov["source_commit"]:
        raise ExportError("provenance.source_commit must be a non-empty string")
    if require_synthetic_offline and prov.get("data_source") != "synthetic_offline":
        raise ExportError("an offline export must declare data_source='synthetic_offline'")
    # honest labelling: a synthetic export must be marked non-hardware, non-official
    if prov.get("data_source") == "synthetic_offline":
        if prov.get("status") != OFFLINE_STATUS:
            raise ExportError("offline export must carry the honest synthetic status")
        if prov.get("official") is True:
            raise ExportError("a synthetic offline export must not be marked official")
    # official-export sanity: non-empty + clean provenance
    if prov.get("official") is True:
        if not rows:
            raise ExportError("an official export must not be empty")
        if prov.get("scientific_code_dirty") or prov.get("repository_dirty"):
            raise ExportError("an official export must not have dirty provenance")
    stored = doc.get("export_sha256")
    if strict:
        if not isinstance(stored, str) or not _SHA_RE.match(stored):
            raise ExportError("strict: export_sha256 is missing or malformed")
    if stored is not None:
        if not isinstance(stored, str) or not _SHA_RE.match(stored):
            raise ExportError("export_sha256 is malformed")
        if stored != canonical_export_sha256(doc):
            raise ExportError("export_sha256 does not match the content (tampered)")
    blob = json.dumps(doc)
    if _PATH_RE.search(blob):
        raise ExportError("export contains a local absolute path")
    if _SECRET_RE.search(blob):
        raise ExportError("export contains a secret-like token")
