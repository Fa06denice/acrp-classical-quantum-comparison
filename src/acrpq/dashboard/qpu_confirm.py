"""Explicit, replay-proof QPU confirmation challenges (Phase 7).

A confirmation ``nonce`` is issued at ``prepare`` time, bound to the exact run and
its hashes/backend/budget. To confirm, the caller must echo *all* of those values,
supply the exact consent phrase, and present the single-use nonce before it
expires. Any mismatch, a reused nonce (replay / double-click), an expired nonce,
or a run whose config hash changed since prepare is refused.

The store is in-memory and process-local (a single local dev server); consumption
is guarded by a lock so two concurrent confirms cannot both win. Clocks are
injected so expiry is deterministic in tests. No secret is ever stored here.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

CONSENT_PHRASE = "I understand this may submit a real, billable IBM Quantum job"
DEFAULT_TTL_S = 300.0


class ConfirmationError(ValueError):
    """A confirmation attempt is invalid (mismatch, expired, replayed, changed)."""


@dataclass(frozen=True)
class Challenge:
    nonce: str
    run_id: str
    config_hash: str
    artefact_sha256: str
    decision_sha256: str | None
    backend: str
    total_shots: int
    max_jobs: int
    max_quantum_seconds: float
    issued_at: float
    expires_at: float

    def public(self) -> dict[str, Any]:
        """Fields safe to return to the client (the nonce is the bearer secret)."""
        return {
            "nonce": self.nonce, "run_id": self.run_id, "expires_at": self.expires_at,
            "consent_phrase": CONSENT_PHRASE,
        }


def _new_nonce() -> str:
    return uuid.uuid4().hex


class ConfirmationStore:
    """Process-local, lock-guarded, single-use confirmation challenges."""

    def __init__(self, *, now: Callable[[], float] = time.time, ttl_s: float = DEFAULT_TTL_S,
                 nonce_factory: Callable[[], str] = _new_nonce) -> None:
        self._now = now
        self._ttl = ttl_s
        self._nonce_factory = nonce_factory
        self._lock = threading.Lock()
        self._by_nonce: dict[str, Challenge] = {}
        self._consumed: set[str] = set()

    def issue(self, *, run_id: str, config_hash: str, artefact_sha256: str,
              decision_sha256: str | None, backend: str, total_shots: int, max_jobs: int,
              max_quantum_seconds: float) -> Challenge:
        now = self._now()
        ch = Challenge(
            nonce=self._nonce_factory(), run_id=run_id, config_hash=config_hash,
            artefact_sha256=artefact_sha256, decision_sha256=decision_sha256, backend=backend,
            total_shots=int(total_shots), max_jobs=int(max_jobs),
            max_quantum_seconds=float(max_quantum_seconds), issued_at=now,
            expires_at=now + self._ttl)
        with self._lock:
            self._by_nonce[ch.nonce] = ch
        return ch

    def consume(self, *, nonce: Any, run_id: Any, config_hash: Any, artefact_sha256: Any,
                decision_sha256: Any, backend: Any, total_shots: Any, max_jobs: Any,
                max_quantum_seconds: Any, consent: Any,
                current_config_hash: str | None = None) -> Challenge:
        """Atomically validate + consume a challenge, or raise ``ConfirmationError``.

        ``current_config_hash`` (the run's config hash *now*) must still equal the
        challenge's — proving nothing changed between prepare and confirm.
        """
        if consent != CONSENT_PHRASE:
            raise ConfirmationError("missing or incorrect consent phrase")
        if not isinstance(nonce, str) or not nonce:
            raise ConfirmationError("nonce must be a non-empty string")
        with self._lock:
            if nonce in self._consumed:
                raise ConfirmationError("nonce already used (replay/double-confirm refused)")
            ch = self._by_nonce.get(nonce)
            if ch is None:
                raise ConfirmationError("unknown nonce")
            if self._now() > ch.expires_at:
                # expired: drop it so it cannot be probed further
                self._by_nonce.pop(nonce, None)
                raise ConfirmationError("confirmation nonce expired")
            # every bound field must match exactly (use compare_digest for hashes)
            def _eq(a: str, b: Any) -> bool:
                return isinstance(b, str) and hmac.compare_digest(a, b)
            if ch.run_id != run_id:
                raise ConfirmationError("run_id mismatch")
            if not _eq(ch.config_hash, config_hash):
                raise ConfirmationError("config_hash mismatch")
            if not _eq(ch.artefact_sha256, artefact_sha256):
                raise ConfirmationError("artefact_sha256 mismatch")
            if (ch.decision_sha256 or "") != (decision_sha256 or ""):
                raise ConfirmationError("decision_sha256 mismatch")
            if ch.backend != backend:
                raise ConfirmationError("backend mismatch")
            if ch.total_shots != total_shots or ch.max_jobs != max_jobs:
                raise ConfirmationError("budget mismatch")
            if not _finite_eq(ch.max_quantum_seconds, max_quantum_seconds):
                raise ConfirmationError("budget mismatch")
            if current_config_hash is not None and not _eq(ch.config_hash, current_config_hash):
                raise ConfirmationError("run configuration changed since prepare")
            # consume (single-use) — mark BEFORE returning so no concurrent win
            self._consumed.add(nonce)
            self._by_nonce.pop(nonce, None)
            return ch

    def digest(self) -> str:
        """A stable digest of live nonces (for debugging; contains no secrets)."""
        with self._lock:
            joined = ",".join(sorted(self._by_nonce))
        return "sha256:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _finite_eq(a: float, b: Any) -> bool:
    return isinstance(b, (int, float)) and not isinstance(b, bool) and float(a) == float(b)


# --------------------------------------------------------------------------- #
# Durable, atomic confirmation (Phase 6.2) — survives a restart
# --------------------------------------------------------------------------- #
DURABLE_SCHEMA_VERSION = "acrpq-qpu-confirm/1"
MARKER_SCHEMA_VERSION = "acrpq-qpu-confirm-marker/1"
CONSENT_VERSION = "acrpq-consent/1"
_MARKER_FIELDS = (
    "schema", "run_id", "challenge_doc_sha256", "nonce_sha256", "config_hash",
    "decision_sha256", "artefact_sha256", "plan_sha256", "isa_fingerprint_sha256",
    "backend", "shots", "max_jobs", "max_quantum_seconds", "issued_at", "expires_at",
    "consumed_at", "consent_version",
)


def _hash_nonce(nonce: str) -> str:
    return "sha256:" + hashlib.sha256(nonce.encode("utf-8")).hexdigest()


class ConsumedConfirmation:
    """A verified consumed-confirmation marker (typed, immutable view)."""

    __slots__ = ("data",)

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def __getitem__(self, k: str) -> Any:
        return self.data[k]


class DurableConfirmationStore:
    """Persistent, restart-surviving, single-winner, CRYPTOGRAPHICALLY-VERIFIED
    confirmation.

    The challenge (nonce stored only as a hash) is persisted at issue time; it
    binds the WHOLE persisted chain (config/decision/artefact/plan/ISA/backend/
    budget). Consumption atomically writes a rich, self-hashed *consumed marker*
    that re-states every binding + the challenge document's own hash. A later
    ``/submit`` calls :meth:`verify_consumed_confirmation` with the expected chain
    hashes: an empty ``{}`` marker, a truncated marker, a marker whose
    ``marker_sha256`` does not recompute, or one whose bindings differ from the
    live chain is refused. The nonce plaintext is returned once and never stored.
    """

    def __init__(self, *, root: str, now: Callable[[], float] = time.time,
                 ttl_s: float = DEFAULT_TTL_S, nonce_factory: Callable[[], str] = _new_nonce) -> None:
        self._root = root
        self._now = now
        self._ttl = ttl_s
        self._nonce_factory = nonce_factory

    def _marker_path(self, run_id: str):
        from . import qpu_store
        return qpu_store._artefacts_dir(self._root, run_id) / "confirmation_consumed.json"

    @staticmethod
    def _marker_self_hash(marker: dict[str, Any]) -> str:
        from .backend_scoring import _canonical
        body = {k: marker[k] for k in _MARKER_FIELDS}
        return "sha256:" + hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()

    def issue(self, *, run_id: str, config_hash: str, artefact_sha256: str,
              decision_sha256: str | None, backend: str, total_shots: int, max_jobs: int,
              max_quantum_seconds: float, plan_sha256: str, isa_fingerprint_sha256: str,
              ) -> tuple[str, dict[str, Any]]:
        from . import qpu_store

        nonce = self._nonce_factory()
        now = self._now()
        challenge = {
            "schema": DURABLE_SCHEMA_VERSION, "run_id": run_id, "nonce_sha256": _hash_nonce(nonce),
            "config_hash": config_hash, "artefact_sha256": artefact_sha256,
            "decision_sha256": decision_sha256, "plan_sha256": plan_sha256,
            "isa_fingerprint_sha256": isa_fingerprint_sha256, "backend": backend,
            "total_shots": int(total_shots), "max_jobs": int(max_jobs),
            "max_quantum_seconds": float(max_quantum_seconds), "consent_version": CONSENT_VERSION,
            "issued_at": now, "expires_at": now + self._ttl,
        }
        qpu_store.save_confirmation(run_id, challenge, root=self._root)   # durable, no overwrite
        public = {"nonce": nonce, "run_id": run_id, "expires_at": challenge["expires_at"],
                  "consent_phrase": CONSENT_PHRASE, "consent_version": CONSENT_VERSION}
        return nonce, public

    def consume(self, *, run_id: Any, nonce: Any, config_hash: Any, artefact_sha256: Any,
                decision_sha256: Any, backend: Any, total_shots: Any, max_jobs: Any,
                max_quantum_seconds: Any, consent: Any,
                current_config_hash: str | None = None) -> dict[str, Any]:
        """Atomically validate + consume the durable challenge. Idempotent for the
        winner; raises :class:`ConfirmationError` on any refusal."""
        from . import qpu_store
        from .backend_scoring import _canonical
        from .persist import _publish_atomic_exclusive

        if consent != CONSENT_PHRASE:
            raise ConfirmationError("missing or incorrect consent phrase")
        if not isinstance(nonce, str) or not nonce:
            raise ConfirmationError("nonce must be a non-empty string")
        try:
            ch = qpu_store.load_confirmation(run_id, root=self._root)
            challenge_doc_sha256 = qpu_store.doc_sha256(run_id, "confirmation", root=self._root)
        except qpu_store.StoreError as exc:
            raise ConfirmationError(f"no valid confirmation challenge: {exc}") from exc

        supplied = {"run_id": run_id, "config_hash": config_hash, "artefact_sha256": artefact_sha256,
                    "decision_sha256": decision_sha256, "backend": backend,
                    "total_shots": total_shots, "max_jobs": max_jobs,
                    "max_quantum_seconds": max_quantum_seconds}
        if not hmac.compare_digest(ch["nonce_sha256"], _hash_nonce(nonce)):
            raise ConfirmationError("nonce does not match the issued challenge")
        for key in ("run_id", "config_hash", "artefact_sha256", "backend"):
            a, b = ch[key], supplied[key]
            ok = hmac.compare_digest(a, b) if isinstance(b, str) and isinstance(a, str) else (a == b)
            if not ok:
                raise ConfirmationError(f"{key} mismatch")
        if (ch["decision_sha256"] or "") != (decision_sha256 or ""):
            raise ConfirmationError("decision_sha256 mismatch")
        if ch["total_shots"] != total_shots or ch["max_jobs"] != max_jobs:
            raise ConfirmationError("budget mismatch")
        if not _finite_eq(ch["max_quantum_seconds"], max_quantum_seconds):
            raise ConfirmationError("budget mismatch")
        if current_config_hash is not None and not hmac.compare_digest(
                ch["config_hash"], current_config_hash):
            raise ConfirmationError("run configuration changed since prepare")

        now = self._now()
        path = self._marker_path(run_id)
        marker = {
            "schema": MARKER_SCHEMA_VERSION, "run_id": run_id,
            "challenge_doc_sha256": challenge_doc_sha256, "nonce_sha256": _hash_nonce(nonce),
            "config_hash": ch["config_hash"], "decision_sha256": ch["decision_sha256"],
            "artefact_sha256": ch["artefact_sha256"], "plan_sha256": ch["plan_sha256"],
            "isa_fingerprint_sha256": ch["isa_fingerprint_sha256"], "backend": ch["backend"],
            "shots": ch["total_shots"], "max_jobs": ch["max_jobs"],
            "max_quantum_seconds": ch["max_quantum_seconds"], "issued_at": ch["issued_at"],
            "expires_at": ch["expires_at"], "consumed_at": now, "consent_version": CONSENT_VERSION,
        }
        marker["marker_sha256"] = self._marker_self_hash(marker)

        def _same(existing: dict[str, Any]) -> bool:
            return (isinstance(existing, dict)
                    and existing.get("marker_sha256") == marker["marker_sha256"])

        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            # a genuine idempotent retry re-derives the same marker EXCEPT consumed_at;
            # compare on the bound fields (everything but consumed_at + its self hash)
            if isinstance(existing, dict) and existing.get("challenge_doc_sha256") == challenge_doc_sha256 \
                    and hmac.compare_digest(existing.get("nonce_sha256", ""), _hash_nonce(nonce)) \
                    and all(existing.get(k) == marker[k] for k in _MARKER_FIELDS if k != "consumed_at"):
                return {"confirmed": True, "idempotent": True}
            raise ConfirmationError("challenge already consumed with a different payload (replay)")

        if now > ch["expires_at"]:
            raise ConfirmationError("confirmation challenge expired")
        try:
            _publish_atomic_exclusive(path, _canonical(marker))   # atomic single-winner
        except FileExistsError:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and existing.get("challenge_doc_sha256") == challenge_doc_sha256 \
                    and all(existing.get(k) == marker[k] for k in _MARKER_FIELDS if k != "consumed_at"):
                return {"confirmed": True, "idempotent": True}
            raise ConfirmationError("challenge already consumed with a different payload (replay)")
        return {"confirmed": True, "idempotent": False}

    def verify_consumed_confirmation(
        self, run_id: str, *, expected_config_hash: str, expected_decision_sha256: str | None,
        expected_artefact_sha256: str, expected_plan_sha256: str, expected_isa_fingerprint: str,
        expected_backend: str, expected_shots: int, expected_max_jobs: int,
        expected_max_quantum_seconds: float,
    ) -> ConsumedConfirmation:
        """Cryptographically verify the consumed marker binds to the whole chain.

        Refuses an empty/truncated marker, a marker whose ``marker_sha256`` does not
        recompute, a wrong schema/consent version, a challenge-doc-hash that no
        longer matches the (possibly tampered) persisted challenge, or any binding
        that differs from the expected chain hashes/budget. Raises on any failure.
        """
        from . import qpu_store

        path = self._marker_path(run_id)
        if not path.is_file():
            raise ConfirmationError("run is not durably confirmed (no consumed marker)")
        try:
            marker = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfirmationError(f"consumed marker unreadable: {exc}") from exc
        if not isinstance(marker, dict):
            raise ConfirmationError("consumed marker is not an object")
        missing = [k for k in (*_MARKER_FIELDS, "marker_sha256") if k not in marker]
        if missing:
            raise ConfirmationError(f"consumed marker missing field(s): {missing}")
        if marker["schema"] != MARKER_SCHEMA_VERSION:
            raise ConfirmationError("consumed marker has an incompatible schema")
        if marker["consent_version"] != CONSENT_VERSION:
            raise ConfirmationError("consumed marker has an incompatible consent version")
        if not hmac.compare_digest(marker["marker_sha256"], self._marker_self_hash(marker)):
            raise ConfirmationError("consumed marker self-hash does not verify (tampered)")
        # the marker must still bind to the LIVE challenge document (tamper-after-consume)
        try:
            live_challenge_hash = qpu_store.doc_sha256(run_id, "confirmation", root=self._root)
        except qpu_store.StoreError as exc:
            raise ConfirmationError(f"challenge document invalid: {exc}") from exc
        if not hmac.compare_digest(marker["challenge_doc_sha256"], live_challenge_hash):
            raise ConfirmationError("challenge document changed since confirmation")
        # every binding must equal the expected chain hashes/budget
        checks = {
            "run_id": (marker["run_id"], run_id), "config_hash": (marker["config_hash"], expected_config_hash),
            "artefact_sha256": (marker["artefact_sha256"], expected_artefact_sha256),
            "plan_sha256": (marker["plan_sha256"], expected_plan_sha256),
            "isa_fingerprint_sha256": (marker["isa_fingerprint_sha256"], expected_isa_fingerprint),
            "backend": (marker["backend"], expected_backend),
        }
        for name, (a, b) in checks.items():
            ok = hmac.compare_digest(a, b) if isinstance(a, str) and isinstance(b, str) else (a == b)
            if not ok:
                raise ConfirmationError(f"consumed marker {name} does not match the chain")
        if (marker["decision_sha256"] or "") != (expected_decision_sha256 or ""):
            raise ConfirmationError("consumed marker decision_sha256 does not match the chain")
        if marker["shots"] != expected_shots or marker["max_jobs"] != expected_max_jobs:
            raise ConfirmationError("consumed marker budget does not match the chain")
        if not _finite_eq(marker["max_quantum_seconds"], expected_max_quantum_seconds):
            raise ConfirmationError("consumed marker budget does not match the chain")
        return ConsumedConfirmation(marker)

    def is_consumed(self, run_id: str) -> bool:
        """Cheap existence check ONLY — NOT a security gate. Use
        :meth:`verify_consumed_confirmation` before any real submission."""
        return self._marker_path(run_id).is_file()
