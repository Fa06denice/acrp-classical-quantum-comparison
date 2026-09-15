"""Inter-process single-winner claim for a QPU config (Phase 5.2.C).

Closes the concurrency race identified in the 5.2.A audit: two processes that start
with the *same* configuration each scan an empty ``runs_dir``, find nothing, and each
call ``qpu_runs.prepare`` — which mints a *distinct* ``local_run_id`` (the id embeds a
``uuid4`` nonce), so both ``os.link`` manifest publishes succeed and both proceed to
submit. Offline that is two harmless synthetic runs; for a future real provider it is a
genuine double submission (the per-run job tag differs, so the pre-submit search cannot
see the sibling).

**Safety kernel (the exact guarantee).** Exactly one ``local_run_id`` ever wins the
atomic, exclusive publish of the *runid file* (``persist._publish_atomic_exclusive`` →
``os.link``, atomic + cross-process, ``FileExistsError`` if the target exists). Every
participant — winner, loser, crash-recoverer — returns that same committed
``local_run_id``. Precisely:

- In the **normal concurrent path** (no crash) exactly ONE candidate is prepared: the
  claim winner prepares; losers wait and adopt without preparing.
- **After a crash/timeout** several *candidates* may be prepared (a recoverer prepares
  its own), but exactly ONE ``local_run_id`` is *designated* atomically.
- Only the DESIGNATED run is ever handed to ``submit_run`` by the driver; orphan
  candidates are never designated and therefore can never be submitted.

So single-winner is a property of the atomic runid publish, not of lock liveness or any
timeout. The tolerated cost after a crash is orphan candidate runs (prepared, never
designated, never submitted); :func:`read_designated` and the driver's orphan inventory
identify them for a lease-guarded, non-destructive deferred cleanup that never touches
an active run.

**Protocol** (keyed by the FULL config hash, so distinct configs never contend):

1. Fast path — a committed ``runid`` file already exists ⇒ adopt it (verify first).
2. Contend for an exclusive ``claim`` (lock) file. The winner prepares the one run,
   then commits its ``runid`` atomically; if it loses that publish (a recovery race)
   it adopts whoever won.
3. A loser (claim already held) verifies the claim (a corrupt/foreign claim fails
   CLOSED, never triggering a fresh preparation) then waits for the winner's ``runid``.
4. If no ``runid`` appears within ``wait_timeout`` (winner crashed or is pathologically
   slow), recover by preparing a candidate and racing to commit the ``runid`` — the
   atomic publish still admits exactly one winner; the rest adopt.

Files live under ``<claims_dir>/`` as ``<sanitised-hash>.claim`` and
``<sanitised-hash>.runid``. Both are published with the same atomic-exclusive
``os.link`` primitive (temp + fsync + hard-link + parent fsync) reused from
``persist``; there is no partial file (a crash leaves only a swept ``.tmp-*``). This
module performs local-filesystem I/O only — it never contacts IBM, reads a token, or
builds a service.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .persist import _publish_atomic_exclusive, _sha256

CLAIM_SCHEMA = "acrpq-qpu-claim/1"
_HASH_RE = ("sha256:", 64)   # config_hash must look like sha256:<64 hex>


class ClaimError(RuntimeError):
    """A claim/runid file is corrupt, foreign, or ambiguous; fail closed."""


@dataclass(frozen=True)
class ClaimResult:
    """Outcome of :func:`claim_or_adopt`.

    ``role`` is ``"winner"`` (this process prepared the designated run), ``"adopter"``
    (adopted a runid another process committed), or ``"recovered"`` (committed a runid
    after the original winner failed to). ``local_run_id`` is the single designated id
    shared by every participant for this config."""

    local_run_id: str
    role: str


def _canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_config_hash(config_hash: str) -> str:
    prefix, hexlen = _HASH_RE
    body = config_hash[len(prefix):] if isinstance(config_hash, str) \
        and config_hash.startswith(prefix) else None
    if body is None or len(body) != hexlen or any(c not in "0123456789abcdef" for c in body):
        raise ClaimError(f"config_hash must be 'sha256:<64 hex>'; got {config_hash!r}")
    return config_hash


def _sanitise(config_hash: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in config_hash)


def claim_paths(claims_dir: str | os.PathLike, config_hash: str) -> tuple[Path, Path]:
    """Return the (claim, runid) file paths for a config hash. Public for tests."""
    _validate_config_hash(config_hash)
    base = Path(claims_dir) / _sanitise(config_hash)
    return base.with_suffix(".claim"), base.with_suffix(".runid")


def read_designated(config_hash: str, claims_dir: str | os.PathLike) -> str | None:
    """Return the DESIGNATED local_run_id committed for this config, or None if none is
    committed yet. Fail-closed (ClaimError) on a corrupt/tampered/foreign runid file.
    This is the single authority on 'which run may be submitted for this config'."""
    _validate_config_hash(config_hash)
    _claim_path, runid_path = claim_paths(claims_dir, config_hash)
    return _read_committed_runid(runid_path, config_hash)


def _runid_payload(config_hash: str, local_run_id: str) -> str:
    body = {"schema": CLAIM_SCHEMA, "config_hash": config_hash, "local_run_id": local_run_id}
    body["content_sha256"] = _sha256(_canonical(body))
    return _canonical(body)


def _read_committed_runid(runid_path: Path, config_hash: str) -> str | None:
    """Return the committed local_run_id, or None if not yet committed. Fail CLOSED
    (ClaimError) on a corrupt/tampered/foreign runid file — never treat it as absent
    (which would look like 'never claimed' and could drive a duplicate preparation)."""
    if not runid_path.exists():
        return None
    try:
        obj = json.loads(runid_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClaimError(f"corrupt runid file {runid_path.name}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ClaimError(f"corrupt runid file {runid_path.name}: not an object")
    stored_sha = obj.get("content_sha256")
    recomputed = _sha256(_canonical({k: obj[k] for k in obj if k != "content_sha256"}))
    if stored_sha != recomputed:
        raise ClaimError(f"runid file {runid_path.name} failed its content hash (tampered)")
    if obj.get("config_hash") != config_hash:
        raise ClaimError(
            f"runid file {runid_path.name} is for config {obj.get('config_hash')!r}, "
            f"not {config_hash!r}")
    rid = obj.get("local_run_id")
    if not isinstance(rid, str) or not rid:
        raise ClaimError(f"runid file {runid_path.name} has no local_run_id")
    return rid


def _claim_payload(config_hash: str, nonce: str) -> str:
    body = {"schema": CLAIM_SCHEMA, "config_hash": config_hash, "nonce": nonce}
    body["content_sha256"] = _sha256(_canonical(body))
    return _canonical(body)


def _verify_existing_claim(claim_path: Path, config_hash: str) -> None:
    """A claim we did not create must be a well-formed lock for THIS config, else fail
    CLOSED. A corrupt/foreign claim never triggers a fresh preparation."""
    try:
        obj = json.loads(claim_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClaimError(f"corrupt claim file {claim_path.name}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ClaimError(f"corrupt claim file {claim_path.name}: not an object")
    stored = obj.get("content_sha256")
    recomputed = _sha256(_canonical({k: obj[k] for k in obj if k != "content_sha256"}))
    if stored != recomputed:
        raise ClaimError(f"claim file {claim_path.name} failed its content hash (tampered)")
    if obj.get("config_hash") != config_hash:
        raise ClaimError(
            f"claim file {claim_path.name} is for a different config "
            f"({obj.get('config_hash')!r})")


def _commit_runid(runid_path: Path, config_hash: str, rid: str) -> str:
    """Atomically publish rid as THE committed runid; return the effective winner
    (rid if we won the publish, else the already-committed value)."""
    try:
        _publish_atomic_exclusive(runid_path, _runid_payload(config_hash, rid))
        return rid
    except FileExistsError:
        other = _read_committed_runid(runid_path, config_hash)
        if other is None:  # exists but unreadable-as-committed -> corrupt; fail closed
            raise ClaimError(
                f"runid {runid_path.name} exists but is not a committed runid") from None
        return other


def claim_or_adopt(
    config_hash: str,
    *,
    claims_dir: str | os.PathLike,
    prepare_fn: Callable[[], str],
    wait_timeout: float = 60.0,
    poll_interval: float = 0.05,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    nonce_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
) -> ClaimResult:
    """Acquire the single-winner claim for ``config_hash`` or adopt the existing one.

    ``prepare_fn`` is called AT MOST once by this process, only when it must materialise
    a candidate run (winner or recoverer); it must return a fresh ``local_run_id``. The
    designated run for a config is whichever ``local_run_id`` won the atomic runid
    publish; every caller returns that same id."""
    _validate_config_hash(config_hash)
    claim_path, runid_path = claim_paths(claims_dir, config_hash)
    Path(claims_dir).mkdir(parents=True, exist_ok=True)

    # 1. Fast path: already committed for everyone.
    rid = _read_committed_runid(runid_path, config_hash)
    if rid is not None:
        return ClaimResult(rid, "adopter")

    # 2. Contend for the exclusive claim.
    try:
        _publish_atomic_exclusive(claim_path, _claim_payload(config_hash, nonce_factory()))
        won_claim = True
    except FileExistsError:
        won_claim = False

    if won_claim:
        rid = prepare_fn()                      # the ONE preparation in the common path
        eff = _commit_runid(runid_path, config_hash, rid)
        return ClaimResult(eff, "winner" if eff == rid else "adopter")

    # 3. Lost the claim: it must be a valid lock for this config (else fail closed),
    #    then wait for the winner's runid.
    _verify_existing_claim(claim_path, config_hash)
    deadline = monotonic() + max(0.0, float(wait_timeout))
    while monotonic() < deadline:
        rid = _read_committed_runid(runid_path, config_hash)
        if rid is not None:
            return ClaimResult(rid, "adopter")
        sleeper(poll_interval)

    # 4. Winner never committed within the window: recover. The atomic publish still
    #    admits exactly one winner, so this cannot cause a second DESIGNATED run.
    rid = _read_committed_runid(runid_path, config_hash)
    if rid is not None:
        return ClaimResult(rid, "adopter")
    candidate = prepare_fn()
    eff = _commit_runid(runid_path, config_hash, candidate)
    return ClaimResult(eff, "recovered" if eff == candidate else "adopter")
