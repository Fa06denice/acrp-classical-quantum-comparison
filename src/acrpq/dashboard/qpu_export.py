"""Verified, atomic QPU exports and startup recovery (Phase 9).

Exports bundle a run's manifest, append-only events, and (optionally) its Phase-4
decision, Phase-5 artefact, Phase-6 plan and decoded result into a single
JSON-native document with a self-hash, written atomically. Everything is scanned
for secret-like fields first; no token / Authorization header / credential can be
serialised.

Startup recovery is read-only: it classifies runs (corrupted / awaiting-remote /
recoverable / terminal) by reusing the existing run-store queries and NEVER
submits or auto-resubmits anything.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from . import qpu_runs
from .persist import _publish_atomic_exclusive

QPU_EXPORT_SCHEMA_VERSION = "acrpq-qpu-export/1"


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _to_native(obj: Any) -> Any:
    """Convert dataclasses/tuples to JSON-native structures (no coercion of scalars)."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_native(asdict(obj))
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


def _canonical(obj: Any) -> str:
    import json
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def build_export(
    run: dict[str, Any],
    *,
    decision: dict[str, Any] | None = None,
    artefact: dict[str, Any] | None = None,
    plan: Any = None,
    decoded: Any = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a verifiable export document (adds ``export_sha256``).

    Refuses to serialise any secret-like field and any non-JSON-native/non-finite
    value (via the run store's strict scanners).
    """
    body: dict[str, Any] = {
        "schema": QPU_EXPORT_SCHEMA_VERSION,
        "run": {
            "local_run_id": run.get("local_run_id"),
            "state": run.get("state"),
            "terminal": run.get("terminal"),
            "config_hash": run.get("config_hash"),
            "created_utc": run.get("created_utc"),
            "params": run.get("params"),
            "ibm_job_ids": list(run.get("ibm_job_ids", [])),
            "late_ibm_job_ids": list(run.get("late_ibm_job_ids", [])),
            # the run's actual append-only lifecycle (qpu_runs.load exposes these
            # keys, NOT a single "events" list)
            "transitions": run.get("transitions", []),
            "result": run.get("result"),
            "error": run.get("error"),
            "provider_observations": run.get("provider_observations", []),
            "n_events": run.get("n_events"),
        },
        "decision": _to_native(decision) if decision is not None else None,
        "artefact": _to_native(artefact) if artefact is not None else None,
        "plan": _to_native(plan) if plan is not None else None,
        "decoded": _to_native(decoded) if decoded is not None else None,
        "provenance": _to_native(provenance) if provenance is not None else None,
    }
    # strict: no secrets, JSON-native, finite (reuse the run-store scanners)
    qpu_runs._scan_secrets(body)
    qpu_runs._assert_json_native_finite(body)
    body["export_sha256"] = _sha(_canonical(body))
    return body


def verify_export(export: dict[str, Any]) -> bool:
    """Recompute the export hash over everything except ``export_sha256``."""
    claimed = export.get("export_sha256")
    if not isinstance(claimed, str):
        return False
    payload = {k: v for k, v in export.items() if k != "export_sha256"}
    try:
        return claimed == _sha(_canonical(payload))
    except ValueError:
        return False


def write_export(export: dict[str, Any], dest_path: str) -> str:
    """Atomically write an export document; refuses to overwrite an existing file."""
    if not verify_export(export):
        raise ValueError("refusing to write an export whose hash does not verify")
    _publish_atomic_exclusive(Path(dest_path), _canonical(export))
    return dest_path


def export_run(
    local_run_id: str,
    *,
    root: str,
    dest_path: str,
    decision: dict[str, Any] | None = None,
    artefact: dict[str, Any] | None = None,
    plan: Any = None,
    decoded: Any = None,
    provenance: dict[str, Any] | None = None,
    auto_load: bool = True,
) -> str:
    """Load a run and write a verified export atomically. Read-only w.r.t. the run.

    With ``auto_load`` (Phase 9.1), the persisted decision/artefact/plan/decoded
    result are loaded from :mod:`qpu_store` when not supplied — the caller does not
    have to thread them through. Missing pieces are simply omitted.
    """
    from . import qpu_store

    run = qpu_runs.load(local_run_id, root=root, verify_integrity=True)
    if auto_load:
        if decision is None and qpu_store.has(root, local_run_id, "decision"):
            decision = qpu_store.load_decision(local_run_id, root=root)
        if artefact is None and qpu_store.has(root, local_run_id, "artefact"):
            artefact = qpu_store.load_artefact(local_run_id, root=root)
        if plan is None and qpu_store.has(root, local_run_id, "plan"):
            plan = qpu_store.load_plan(local_run_id, root=root)
        if decoded is None and qpu_store.has(root, local_run_id, "result_raw"):
            decoded = qpu_store.load_result_raw(local_run_id, root=root)
    export = build_export(run, decision=decision, artefact=artefact, plan=plan,
                          decoded=decoded, provenance=provenance)
    return write_export(export, dest_path)


def build_export_auto(local_run_id: str, *, root: str,
                      provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble (but do not write) a verified export, auto-loading the persisted
    chain from :mod:`qpu_store`. Used by the API /export endpoint."""
    from . import qpu_store

    run = qpu_runs.load(local_run_id, root=root, verify_integrity=True)
    kw: dict[str, Any] = {}
    if qpu_store.has(root, local_run_id, "decision"):
        kw["decision"] = qpu_store.load_decision(local_run_id, root=root)
    if qpu_store.has(root, local_run_id, "artefact"):
        kw["artefact"] = qpu_store.load_artefact(local_run_id, root=root)
    if qpu_store.has(root, local_run_id, "plan"):
        kw["plan"] = qpu_store.load_plan(local_run_id, root=root)
    if qpu_store.has(root, local_run_id, "result_raw"):
        kw["decoded"] = qpu_store.load_result_raw(local_run_id, root=root)
    return build_export(run, provenance=provenance, **kw)


# --------------------------------------------------------------------------- #
# Startup recovery (read-only; never submits)
# --------------------------------------------------------------------------- #
def startup_recovery_report(*, root: str) -> dict[str, Any]:
    """Classify persisted runs at startup without touching IBM or resubmitting.

    * ``corrupted`` — integrity chain/hashes do not verify (surfaced, not dropped);
    * ``reconciliation_candidates`` — carry a remote job id / ambiguous state and
      must be reconciled (never auto-resubmitted);
    * ``non_terminal`` — not yet in a terminal state;
    * ``terminal`` — completed/failed/cancelled.
    """
    runs = qpu_runs.list_runs(root=root)
    corrupted = [r for r in runs if r.get("state") == "corrupted"]
    terminal = [r for r in runs if r.get("terminal")]
    return {
        "root": root,
        "total": len(runs),
        "corrupted": [r.get("local_run_id") for r in corrupted],
        "reconciliation_candidates": [
            r.get("local_run_id") for r in qpu_runs.reconciliation_candidates(root=root)],
        "non_terminal": [r.get("local_run_id") for r in qpu_runs.non_terminal_runs(root=root)],
        "terminal": [r.get("local_run_id") for r in terminal],
        "policy": "read-only classification; never auto-submits or resubmits",
    }
