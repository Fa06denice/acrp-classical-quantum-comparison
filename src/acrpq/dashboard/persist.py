"""Persist dashboard runs as reproducible, versioned records.

Two record kinds are kept strictly separate on disk and in the schema:

* ``interactive`` — a single DEMO solve. Throwaway, explicitly **not citable**;
  saved (if at all) only so a session can be reopened.
* ``scientific`` — a seed-repeated SCIENTIFIC run carrying full provenance
  (git commit/dirty, source hash, dependency versions, profile, every
  parameter, every seed, per-seed results and computed explanations, method
  applicability, durations). This is the only kind a defense may cite.

Every field is copied from what the backend actually computed — nothing here
fabricates provenance. Each artifact is published atomically and *exclusively*
(temp file + ``fsync`` + ``os.link``, POSIX), and a per-run ``<run_id>.meta.json``
sidecar written **last** is the single commit point: a run is indexed iff its
meta exists. There is no shared, mutable manifest, so concurrent processes never
lose each other's entries; a crash before the meta leaves only invisible orphans
that :func:`recover_orphans` sweeps after a lease window. The parent directory is
fsynced after each publish/rollback for best-effort durability; the guarantee is
atomic *visibility* of the commit point, not a cross-file database transaction.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "acrpq-run/1"
_KINDS = ("interactive", "scientific")
# run_id is only ever produced by _run_id below; this pins exactly that shape and
# rejects anything else (in particular path separators / traversal) before a read.
_RUN_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6,}Z-[0-9a-f]{8}-[0-9a-f]{8}$")
# Serialises concurrent saves *within this process*. (Unique ids + exclusive
# os.link creation already prevent cross-process clobber; this just avoids two
# same-process threads interleaving their publishes.)
_SAVE_LOCK = threading.Lock()


def config_hash(config: dict[str, Any]) -> str:
    """Stable ``sha256`` of a configuration (identical configs → identical hash)."""
    canonical = json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_utc() -> str:
    # microsecond precision so the run_id is unique even for same-second saves
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _run_id(created_utc: str, chash: str) -> str:
    """Unique, sortable id: <UTC µs compacted>-<config prefix>-<random nonce>.

    Two saves of the same configuration within the same second (even the same
    microsecond) get distinct ids thanks to the uuid4 nonce, so neither can
    silently overwrite the other.
    """
    stamp = (
        created_utc.replace(":", "").replace("-", "").replace(".", "").replace("+0000", "Z")
    )
    nonce = uuid.uuid4().hex[:8]
    return f"{stamp}-{chash.split(':', 1)[1][:8]}-{nonce}"


def _validate_run_id(run_id: str) -> str:
    """Reject anything not produced by :func:`_run_id` (blocks path traversal)."""
    if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run_id {run_id!r}")
    return run_id


def _provenance() -> dict[str, Any]:
    """Non-fabricated provenance captured at save time."""
    from ..benchmark.export import ReproInfo

    repro = asdict(ReproInfo.capture())
    # drop per-run fields that are meaningless for a whole-record capture
    for key in ("seed", "backend", "shots"):
        repro.pop(key, None)
    return repro


def build_scientific_record(
    scientific_out: dict[str, Any],
    *,
    instance: str,
    subset: Any,
    config: dict[str, Any],
    applicability: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble a fully-provenanced scientific-run record from live results.

    ``scientific_out`` is exactly the dict returned by ``api.scientific_run``:
    ``{"profile", "aggregate", "runs"}``. ``applicability`` is the optional
    ``preflight_payload()['methods']`` list, so the record also states which
    methods applied or were skipped and why.
    """
    if scientific_out.get("profile") != "scientific":
        raise ValueError("only SCIENTIFIC runs may be saved as scientific records")
    runs = scientific_out["runs"]
    created = _now_utc()
    full_config = {"instance": instance, "subset": subset, **config}
    chash = config_hash(full_config)
    profile = runs[0].get("profile") if runs else scientific_out.get("profile")
    duration = sum(
        (r["solution"].get("wall_time_s") or 0.0)
        for r in runs
        if isinstance(r.get("solution"), dict)
    )
    return {
        "schema": SCHEMA_VERSION,
        "kind": "scientific",
        "run_id": _run_id(created, chash),
        "created_utc": created,
        "citable": True,
        "profile": profile,
        "instance": {"name": instance, "subset": subset},
        "config": full_config,
        "config_hash": chash,
        "provenance": _provenance(),
        "applicability": applicability or [],
        "aggregate": scientific_out["aggregate"],
        "runs": runs,
        "duration_s": duration,
    }


def build_interactive_record(
    solve_out: dict[str, Any], *, instance: str, subset: Any, config: dict[str, Any]
) -> dict[str, Any]:
    """Wrap a single interactive solve — explicitly not citable."""
    created = _now_utc()
    full_config = {"instance": instance, "subset": subset, **config}
    chash = config_hash(full_config)
    return {
        "schema": SCHEMA_VERSION,
        "kind": "interactive",
        "run_id": _run_id(created, chash),
        "created_utc": created,
        "citable": False,
        "profile": solve_out.get("profile"),
        "instance": {"name": instance, "subset": subset},
        "config": full_config,
        "config_hash": chash,
        "provenance": solve_out.get("provenance"),
        "result": solve_out.get("solution"),
    }


def scientific_csv(record: dict[str, Any]) -> str:
    """One row per seed: the per-seed outcome table for a scientific record."""
    seeds = record.get("config", {}).get("seeds") or []
    buf = io.StringIO()
    # "\n" (not the csv default "\r\n") so the hashed string round-trips through
    # text-mode write + universal-newline read without changing the digest.
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(
        ["seed", "feasible", "objective", "residual_conflicts", "wall_time_s", "energy_total"]
    )
    for idx, run in enumerate(record.get("runs", [])):
        sol = run.get("solution", {}) if isinstance(run, dict) else {}
        energy = (sol.get("explanation") or {}).get("energy") or {}
        writer.writerow(
            [
                seeds[idx] if idx < len(seeds) else "",
                sol.get("feasible"),
                sol.get("objective"),
                sol.get("n_residual_conflicts"),
                sol.get("wall_time_s"),
                energy.get("total"),
            ]
        )
    return buf.getvalue()


_TMP_PREFIX = ".tmp-"
# Lease window: an artifact younger than this may belong to an in-flight save in
# another process, so recover_orphans() never touches it. Recovery of a crashed
# save is therefore only guaranteed once this window has elapsed.
DEFAULT_ORPHAN_LEASE_S = 900.0


def _fsync_dir(directory: Path) -> None:
    """Best-effort fsync of a directory so a link/unlink survives power loss."""
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass  # some filesystems disallow directory fsync; visibility still holds
    finally:
        os.close(fd)


def _publish_atomic_exclusive(path: Path, text: str) -> None:
    """Create ``path`` atomically with ``text``; raise ``FileExistsError`` if taken.

    Content is written to a temp file in the same directory, fsynced, then
    **hard-linked** into place: ``os.link`` is atomic on POSIX and raises
    ``FileExistsError`` if the target exists, giving atomic content *and*
    exclusive (no-overwrite) creation that also holds across processes. Returning
    normally is the proof that *this* call created ``path`` (the caller may then
    own it for rollback); a raise means another writer owns it. The parent
    directory is fsynced so the new entry is durable. A crash can only leave an
    unpublished ``.tmp-*`` file, which ``recover_orphans`` sweeps.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=_TMP_PREFIX, suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(tmp, path)  # atomic + exclusive (FileExistsError if path exists)
        _fsync_dir(path.parent)  # make the new directory entry durable
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _meta_path(out_dir: Path, run_id: str) -> Path:
    return out_dir / f"{run_id}.meta.json"


def save_run(record: dict[str, Any], *, root: str = "runs") -> dict[str, Any]:
    """Persist a record with a single per-run commit point; return an integrity receipt.

    Coherence model (documented honestly — this is **not** a cross-file database
    transaction): the JSON (and, for scientific runs, the per-seed CSV) are each
    published with an atomic, exclusive ``os.link`` (temp + fsync + hard-link); the
    *last* write is a per-run ``<run_id>.meta.json`` sidecar carrying both SHA-256
    hashes, and it is the single commit point. A run is
    considered saved **iff** its meta sidecar exists, so ``list_runs`` never sees
    a run whose artifacts are missing, and a crash between artifacts and the meta
    leaves only *invisible* orphans that :func:`recover_orphans` cleans up. There
    is no shared, mutable index file, so concurrent saves from different processes
    cannot lose each other's entries. Unique ids + exclusive creation keep the
    no-overwrite guarantee.
    """
    kind = record.get("kind")
    if kind not in _KINDS:
        raise ValueError(f"record kind must be one of {_KINDS}; got {kind!r}")
    run_id = _validate_run_id(record["run_id"])
    out_dir = Path(root) / kind
    json_path = out_dir / f"{run_id}.json"
    csv_path = out_dir / f"{run_id}.csv"
    meta_path = _meta_path(out_dir, run_id)

    # 1) Materialise exact contents + integrity hashes before any disk write.
    # Scientific records must remain JSON-native and finite.  Silently coercing
    # an unexpected object with ``default=str`` would corrupt reproducibility.
    json_text = json.dumps(record, indent=2, allow_nan=False)
    json_sha = _sha256(json_text)
    csv_text = scientific_csv(record) if kind == "scientific" else None
    csv_sha = _sha256(csv_text) if csv_text is not None else None

    entry = {
        "run_id": run_id, "created_utc": record["created_utc"], "kind": kind,
        "citable": record["citable"], "instance": record["instance"]["name"],
        "config_hash": record["config_hash"], "file": json_path.name,
        "json_sha256": json_sha, "csv_sha256": csv_sha,
    }

    with _SAVE_LOCK:  # serialise concurrent in-process saves
        if meta_path.exists() or json_path.exists() or (csv_text and csv_path.exists()):
            raise FileExistsError(f"run {run_id} already exists under {out_dir}")
        # Track exactly the paths THIS call created (each publish returns normally
        # only if its os.link succeeded). Rollback removes only those, so a loser
        # racing the same run_id can never delete the winner's artifacts.
        created: list[Path] = []
        try:
            # publish artifacts first, then the meta sidecar as the commit point
            if csv_text is not None:
                _publish_atomic_exclusive(csv_path, csv_text)
                created.append(csv_path)
            _publish_atomic_exclusive(json_path, json_text)
            created.append(json_path)
            _publish_atomic_exclusive(meta_path, json.dumps(entry, indent=2, allow_nan=False))
            created.append(meta_path)
        except BaseException:
            for p in reversed(created):  # only what this call actually created
                try:
                    p.unlink()
                except OSError:
                    pass
            if created:
                _fsync_dir(out_dir)  # make the rollback durable too
            raise

    return {
        "run_id": run_id, "path": str(json_path),
        "csv_path": str(csv_path) if csv_text is not None else None,
        "kind": kind, "citable": record["citable"],
        "json_sha256": json_sha, "csv_sha256": csv_sha,
    }


def list_runs(kind: str, *, root: str = "runs") -> list[dict[str, Any]]:
    """List committed runs (most recent first) by reading the per-run meta sidecars.

    Only runs whose ``<run_id>.meta.json`` exists are returned, so a partially
    written run is never listed. No shared index file is read, so this is safe
    while other processes are saving.
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}; got {kind!r}")
    out_dir = Path(root) / kind
    if not out_dir.is_dir():
        return []
    rows = []
    for meta in out_dir.glob("*.meta.json"):
        try:
            rows.append(json.loads(meta.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue  # ignore an unreadable/partial sidecar
    rows.sort(key=lambda r: r.get("run_id", ""), reverse=True)
    return rows


def verify_run(kind: str, run_id: str, *, root: str = "runs") -> dict[str, Any]:
    """Recompute artifact hashes and compare to the committed meta. Raise on mismatch.

    A tampered or truncated JSON/CSV fails loudly here rather than being trusted.
    (Tamper-*evident* against the recorded hashes, not tamper-proof: rewriting an
    artifact *and* its meta consistently cannot be detected without a signature.)
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}; got {kind!r}")
    _validate_run_id(run_id)
    out_dir = Path(root) / kind
    meta_path = _meta_path(out_dir, run_id)
    if not meta_path.is_file():
        raise FileNotFoundError(f"no committed meta for {kind} run {run_id!r}")
    entry = json.loads(meta_path.read_text(encoding="utf-8"))
    json_path = out_dir / f"{run_id}.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"missing artifact {json_path}")
    actual = _sha256(json_path.read_text(encoding="utf-8"))
    if actual != entry.get("json_sha256"):
        raise ValueError(
            f"integrity check FAILED for {run_id}.json: "
            f"meta {entry.get('json_sha256')} != actual {actual}"
        )
    result: dict[str, Any] = {"run_id": run_id, "json_ok": True, "csv_ok": None}
    if entry.get("csv_sha256"):
        csv_path = out_dir / f"{run_id}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"missing artifact {csv_path}")
        actual_csv = _sha256(csv_path.read_text(encoding="utf-8"))
        if actual_csv != entry["csv_sha256"]:
            raise ValueError(f"integrity check FAILED for {run_id}.csv")
        result["csv_ok"] = True
    return result


def recover_orphans(
    kind: str, *, root: str = "runs", min_age_s: float = DEFAULT_ORPHAN_LEASE_S
) -> list[str]:
    """Delete *stale* artifacts/temp files that never reached their commit point.

    Safe to run while other processes are saving: an artifact younger than
    ``min_age_s`` (the lease window) is assumed to belong to an in-flight save and
    is never touched, and the absence of ``<run_id>.meta.json`` is re-checked
    immediately before each unlink. Recovery of a crashed save is therefore only
    guaranteed once ``min_age_s`` has elapsed since its last write. Returns the
    removed file names.
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}; got {kind!r}")
    out_dir = Path(root) / kind
    if not out_dir.is_dir():
        return []
    now = time.time()
    removed: list[str] = []
    for p in out_dir.iterdir():
        if p.name.endswith(".meta.json"):
            continue  # committed index entries are never orphans
        is_tmp = p.name.startswith(_TMP_PREFIX)
        if not is_tmp and p.suffix not in {".json", ".csv"}:
            continue
        try:
            if now - p.stat().st_mtime < min_age_s:
                continue  # within the lease window: may be an active save
        except OSError:
            continue
        # a real artifact is an orphan only if its commit meta is (still) absent
        if not is_tmp and _meta_path(out_dir, p.stem).exists():
            continue
        try:
            p.unlink()
        except OSError:
            continue
        removed.append(p.name)
    return removed


def read_run(
    kind: str, run_id: str, *, root: str = "runs", verify: bool = False
) -> dict[str, Any]:
    """Load a saved record back; ``verify=True`` checks integrity hashes first."""
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}; got {kind!r}")
    _validate_run_id(run_id)  # reject path traversal / malformed ids
    if verify:
        verify_run(kind, run_id, root=root)
    path = Path(root) / kind / f"{run_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no {kind} run {run_id!r} under {root}")
    return json.loads(path.read_text(encoding="utf-8"))
