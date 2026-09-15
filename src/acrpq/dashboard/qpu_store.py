"""Durable per-run artefact store (Phase 6.1).

Persists the whole QPU workflow chain for a run so it survives a process restart
and can be fully reconstructed and re-verified without any in-memory or browser
state: the Phase-4 decision, the Phase-5 artefact, the Phase-6 plan, and the ISA
circuit (QPY bytes + fingerprint + qiskit version). Each document is JSON-native,
secret-scanned, self-hashed (SHA-256), and written **atomically with no
overwrite** (reusing the run store's `os.link` commit point). Reads verify the
hash and the schema version before returning; an incompatible schema is refused,
never silently migrated.

Documents live under the run directory (``root/qpu/<run_id>/artefacts/``), so they
share the run's lifecycle and path-traversal protection (`_validate_run_id`). A
crash mid-write can only leave an unpublished temp file (swept by the run store's
orphan recovery), never a partial document — and because the chain is written in
order, :func:`verify_full_run` reports exactly which stage a run reached, so a
partial run is never presented as complete.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from . import qpu_runs
from .backend_scoring import _canonical, _json_native
from .persist import _publish_atomic_exclusive, _sha256

QPU_STORE_SCHEMA_VERSION = "acrpq-qpu-store/1"

# The persisted chain, in the order it is produced.
CHAIN_KINDS = ("decision", "artefact", "plan", "isa_circuit")
_ALL_KINDS = frozenset(CHAIN_KINDS + ("confirmation", "result_raw"))


class StoreError(ValueError):
    """A persisted document is missing, corrupt, tampered, or schema-incompatible."""


def _artefacts_dir(root: str, run_id: str) -> Path:
    qpu_runs._validate_run_id(run_id)               # anti path-traversal
    return qpu_runs._run_dir(root, run_id) / "artefacts"


def _doc_path(root: str, run_id: str, kind: str) -> Path:
    if kind not in _ALL_KINDS:
        raise StoreError(f"unknown artefact kind {kind!r}")
    return _artefacts_dir(root, run_id) / f"{kind}.json"


def _wrap(kind: str, payload: Any) -> dict[str, Any]:
    # normalise to JSON-native FIRST (tuples -> lists explicitly; set/bytes/NaN/Inf
    # are refused by _json_native), so a decision embedding asdict(WorkloadRequirements)
    # tuple fields persists cleanly and deterministically.
    body: dict[str, Any] = {
        "schema": QPU_STORE_SCHEMA_VERSION, "kind": kind, "payload": _json_native(payload)}
    qpu_runs._scan_secrets(body)                    # no token/credential at rest
    qpu_runs._assert_json_native_finite(body)       # JSON-native, finite, no tuples
    body["doc_sha256"] = _sha256(_canonical(body))
    return body


def _verify_doc(doc: dict[str, Any], *, kind: str) -> Any:
    if not isinstance(doc, dict):
        raise StoreError("document is not an object")
    if doc.get("schema") != QPU_STORE_SCHEMA_VERSION:
        raise StoreError(f"incompatible store schema {doc.get('schema')!r} "
                         f"(expected {QPU_STORE_SCHEMA_VERSION}); refusing")
    if doc.get("kind") != kind:
        raise StoreError(f"document kind {doc.get('kind')!r} != expected {kind!r}")
    claimed = doc.get("doc_sha256")
    payload_only = {k: v for k, v in doc.items() if k != "doc_sha256"}
    if not isinstance(claimed, str) or claimed != _sha256(_canonical(payload_only)):
        raise StoreError(f"{kind} document hash does not verify (tampered?)")
    return doc["payload"]


def _require_committed_run(root: str, run_id: str) -> None:
    run_dir = qpu_runs._run_dir(root, run_id)
    if not (run_dir / "manifest.json").is_file():
        raise StoreError(f"no committed run {run_id!r} under {root}")


def _save(root: str, run_id: str, kind: str, payload: Any) -> str:
    _require_committed_run(root, run_id)
    doc = _wrap(kind, payload)
    path = _doc_path(root, run_id, kind)
    # atomic + exclusive: refuses to overwrite an existing document
    _publish_atomic_exclusive(path, json.dumps(doc, indent=2, sort_keys=True, allow_nan=False))
    return doc["doc_sha256"]


def _load(root: str, run_id: str, kind: str) -> Any:
    path = _doc_path(root, run_id, kind)
    if not path.is_file():
        raise StoreError(f"{kind} not persisted for run {run_id!r}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StoreError(f"{kind} document unreadable: {exc}") from exc
    return _verify_doc(doc, kind=kind)


def has(root: str, run_id: str, kind: str) -> bool:
    return _doc_path(root, run_id, kind).is_file()


def doc_sha256(run_id: str, kind: str, *, root: str) -> str:
    """Return the persisted document's self-hash, verifying it first."""
    path = _doc_path(root, run_id, kind)
    if not path.is_file():
        raise StoreError(f"{kind} not persisted for run {run_id!r}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    _verify_doc(doc, kind=kind)          # raises on tamper/schema mismatch
    return doc["doc_sha256"]


# --------------------------------------------------------------------------- #
# Typed save/load for each stage of the chain
# --------------------------------------------------------------------------- #
def save_decision(run_id: str, decision: dict[str, Any], *, root: str) -> str:
    return _save(root, run_id, "decision", decision)


def load_decision(run_id: str, *, root: str) -> dict[str, Any]:
    return _load(root, run_id, "decision")


def save_artefact(run_id: str, artefact_manifest: dict[str, Any], *, root: str) -> str:
    return _save(root, run_id, "artefact", artefact_manifest)


def load_artefact(run_id: str, *, root: str) -> dict[str, Any]:
    return _load(root, run_id, "artefact")


def save_plan(run_id: str, plan_manifest: dict[str, Any], *, root: str) -> str:
    return _save(root, run_id, "plan", plan_manifest)


def load_plan(run_id: str, *, root: str) -> dict[str, Any]:
    return _load(root, run_id, "plan")


def dump_isa_circuit(isa_circuit: Any) -> dict[str, Any]:
    """Serialise an ISA circuit to a persistable payload (QPY b64 + fingerprint)."""
    import io

    from qiskit import qpy

    from .hardware_validation import _circuit_fingerprint

    fingerprint, version = _circuit_fingerprint(isa_circuit)
    buf = io.BytesIO()
    qpy.dump(isa_circuit, buf)
    return {
        "qpy_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "fingerprint_sha256": fingerprint,
        "qiskit_version": version,
    }


def save_isa_circuit(run_id: str, isa_payload: dict[str, Any], *, root: str) -> str:
    for key in ("qpy_base64", "fingerprint_sha256", "qiskit_version"):
        if key not in isa_payload:
            raise StoreError(f"isa payload missing {key!r}")
    return _save(root, run_id, "isa_circuit", isa_payload)


def load_isa_circuit(run_id: str, *, root: str, as_circuit: bool = False) -> Any:
    """Return the persisted ISA payload, or (as_circuit=True) the QPY-reloaded
    circuit re-verified against the stored fingerprint (lazy qiskit)."""
    payload = _load(root, run_id, "isa_circuit")
    if not as_circuit:
        return payload
    import io

    from qiskit import qpy

    from .hardware_validation import _circuit_fingerprint

    raw = base64.b64decode(payload["qpy_base64"])
    circuits = qpy.load(io.BytesIO(raw))
    if not circuits:
        raise StoreError("persisted QPY contained no circuit")
    circ = circuits[0]
    fp, _ = _circuit_fingerprint(circ)
    if fp != payload["fingerprint_sha256"]:
        raise StoreError("reloaded ISA circuit fingerprint does not match the stored one")
    return circ


# --------------------------------------------------------------------------- #
# Confirmation challenge (durable — Phase 6.2 stores the challenge here)
# --------------------------------------------------------------------------- #
def save_confirmation(run_id: str, challenge: dict[str, Any], *, root: str) -> str:
    return _save(root, run_id, "confirmation", challenge)


def load_confirmation(run_id: str, *, root: str) -> dict[str, Any]:
    return _load(root, run_id, "confirmation")


def save_result_raw(run_id: str, raw: dict[str, Any], *, root: str) -> str:
    """Persist a raw sampler result (e.g. {'counts': {...}, 'bit_order': 'qiskit'})."""
    return _save(root, run_id, "result_raw", raw)


def load_result_raw(run_id: str, *, root: str) -> dict[str, Any]:
    return _load(root, run_id, "result_raw")


# --------------------------------------------------------------------------- #
# Whole-chain verification
# --------------------------------------------------------------------------- #
def verify_full_run(run_id: str, *, root: str) -> dict[str, Any]:
    """Verify every persisted document and their cross-hash coherence.

    Returns a report: which stages are present, whether each verifies, the stage
    the chain reached, and whether the cross-references are coherent. Never raises
    for a *missing* later stage (a partial run is legal); raises only for a present
    document that is corrupt/tampered/schema-incompatible.
    """
    present: dict[str, bool] = {k: has(root, run_id, k) for k in CHAIN_KINDS}
    report: dict[str, Any] = {"run_id": run_id, "present": present, "problems": []}
    problems: list[str] = report["problems"]

    decision = artefact = plan = isa = None
    if present["decision"]:
        decision = _load(root, run_id, "decision")
    if present["artefact"]:
        artefact = _load(root, run_id, "artefact")
    if present["plan"]:
        plan = _load(root, run_id, "plan")
    if present["isa_circuit"]:
        isa = _load(root, run_id, "isa_circuit")

    # cross-hash coherence (only between stages that are BOTH present)
    if decision is not None and artefact is not None:
        if artefact.get("decision_sha256") != decision.get("decision_sha256"):
            problems.append("artefact.decision_sha256 != decision.decision_sha256")
    if artefact is not None and plan is not None:
        if plan.get("artefact_sha256") != artefact.get("artefact_sha256"):
            problems.append("plan.artefact_sha256 != artefact.artefact_sha256")
    if artefact is not None and isa is not None:
        if isa.get("fingerprint_sha256") != artefact.get("isa_circuit", {}).get("fingerprint_sha256"):
            problems.append("isa fingerprint != artefact.isa_circuit.fingerprint_sha256")

    # the furthest contiguous stage reached (chain must be prefix-complete)
    reached = None
    for k in CHAIN_KINDS:
        if present[k]:
            reached = k
        else:
            break
    report["stage_reached"] = reached

    # ----- SEMANTIC (business-hash) verification of each present document -----
    verified: dict[str, Any] = {}
    report["verified_hashes"] = verified
    report.setdefault("warnings", [])
    if decision is not None:
        from .backend_scoring import verify_decision_hash
        if not verify_decision_hash(decision):
            problems.append("decision self-hash does not verify")
        else:
            verified["decision_sha256"] = decision.get("decision_sha256")
    if artefact is not None:
        # recompute the artefact manifest hash over everything but artefact_sha256
        from .hardware_validation import _hash_manifest
        from .backend_scoring import _json_native
        manifest = {k: v for k, v in artefact.items() if k != "artefact_sha256"}
        if _hash_manifest(_json_native(manifest)) != artefact.get("artefact_sha256"):
            problems.append("artefact self-hash does not verify")
        else:
            verified["artefact_sha256"] = artefact.get("artefact_sha256")
    if plan is not None:
        from .qpu_orchestrator import _hash_plan
        from .backend_scoring import _json_native
        pmanifest = {k: v for k, v in plan.items() if k != "plan_sha256"}
        if _hash_plan(_json_native(pmanifest)) != plan.get("plan_sha256"):
            problems.append("plan self-hash does not verify")
        else:
            verified["plan_sha256"] = plan.get("plan_sha256")
    if isa is not None and artefact is not None:
        # the persisted ISA circuit must QPY-reload and re-fingerprint to the stored value
        try:
            load_isa_circuit(run_id, root=root, as_circuit=True)
            verified["isa_fingerprint_sha256"] = isa.get("fingerprint_sha256")
        except StoreError as exc:
            problems.append(f"ISA circuit does not verify: {exc}")

    # run manifest coherence (config hash derives from the stored params)
    try:
        run = qpu_runs.load(run_id, root=root, verify_integrity=False)
        from .persist import config_hash as _cfg
        if _cfg(dict(run["params"])) != run["config_hash"]:
            problems.append("run config_hash does not derive from run.params")
        if artefact is not None:
            if artefact.get("config_hash") != run["config_hash"]:
                problems.append("artefact.config_hash != run.config_hash")
            if artefact.get("backend") != run["params"].get("backend_requested"):
                problems.append("artefact.backend != run.params.backend_requested")
    except Exception as exc:  # noqa: BLE001 - a load failure is itself a problem
        problems.append(f"run manifest unverifiable: {exc}")

    report["complete"] = all(present[k] for k in CHAIN_KINDS) and not problems
    report["coherent"] = not problems

    # real_submittable requires a complete+coherent chain whose artefact is
    # genuinely submittable on a NON-fake backend from a NON-synthetic snapshot.
    real = report["complete"] and report["coherent"]
    if real and artefact is not None:
        real = real and bool(artefact.get("submittable"))
        if artefact.get("backend_synthetic", {}).get("is_fake"):
            real = False
    if real and plan is not None:
        real = real and bool(plan.get("submittable"))
    if real and decision is not None:
        sel = decision.get("selected")
        snap = next((s for s in decision.get("snapshots", []) if s.get("name") == sel), None)
        if sel is None or (snap and snap.get("synthetic")):
            real = False
    report["real_submittable"] = bool(real)
    return report
