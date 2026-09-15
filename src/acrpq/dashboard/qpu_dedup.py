"""Remote-job deduplication decision (Phase 5.2.D) — fake service only, never IBM.

The 5.2.A audit found the remote-dedup gap: the only tag attached to a submitted job is
``[local_run_id]``, which dedups only the *same* run, not two independently-prepared
runs of the same config. This module defines a **deterministic correlation tag** derived
from the batch and job identity, plus a pure decision function that says whether a future
submission is permitted — adopting an already-existing remote job instead of resubmitting,
and failing CLOSED on any ambiguity.

**This module never contacts IBM.** It takes an injected ``RemoteJobSearch`` (a fake in
tests; the real provider's ``service.jobs(job_tags=...)`` wrapper later) and never builds a
service, reads a token, or opens a socket. It only *decides*; it never submits.

Decision rule (``decide_submission``):

- a ``local_job_id`` is already recorded ⇒ ``already_submitted`` (adopt, never resubmit);
- else search remote jobs by the deterministic tag:
  - the search raised / timed out / returned an ambiguous result ⇒ ``refuse`` (NEVER submit);
  - exactly zero jobs ⇒ ``submit_allowed`` (the ONLY outcome that permits a submission);
  - exactly one job whose recorded config matches ⇒ ``adopt_remote`` (adopt, never submit);
  - one job whose config does NOT match the tag's identity ⇒ ``refuse`` (tag collision/tamper);
  - more than one job with the tag ⇒ ``refuse`` (fail closed).

Only ``submit_allowed`` may lead a caller to submit, and only after a *successful* search
returning zero jobs. A failed/ambiguous search never authorises a submission, so a network
error can never trigger an automatic (double) submission.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

TAG_SCHEMA = "acrpq"
_SHORT = 12                                   # hex chars kept from each hash in the tag
_TAG_RE = re.compile(r"^acrpq:[0-9a-f]{%d}:[0-9a-f]{%d}$" % (_SHORT, _SHORT))
_MAX_TAG_LEN = 63                             # conservative Runtime job-tag length budget


class RemoteSearchError(RuntimeError):
    """Base: the remote search could not return an authoritative answer."""


class RemoteTimeout(RemoteSearchError):
    """The remote search timed out (no authoritative answer)."""


class RemoteAuthError(RemoteSearchError):
    """Authentication/authorization failed."""


class RemoteQuotaError(RemoteSearchError):
    """Quota/rate limit exhausted."""


class RemoteBackendUnavailable(RemoteSearchError):
    """The backend/service is unavailable."""


class RemoteMalformedResult(RemoteSearchError):
    """The search returned a structurally invalid/ambiguous result."""


def _hex_body(h: str, name: str) -> str:
    body = h[len("sha256:"):] if isinstance(h, str) and h.startswith("sha256:") else None
    if body is None or len(body) < _SHORT or any(c not in "0123456789abcdef" for c in body):
        raise ValueError(f"{name} must be 'sha256:<>=12 hex>'; got {h!r}")
    return body[:_SHORT]


def job_tag(batch_config_hash: str, job_config_hash: str) -> str:
    """Deterministic, provider-safe correlation tag ``acrpq:<batch12>:<job12>``.

    Stable across processes/restarts (pure function of the two identity hashes), contains
    no secret, path, or timestamp, and fits the Runtime tag character/length limits."""
    tag = f"{TAG_SCHEMA}:{_hex_body(batch_config_hash, 'batch_config_hash')}:" \
          f"{_hex_body(job_config_hash, 'job_config_hash')}"
    if len(tag) > _MAX_TAG_LEN or not _TAG_RE.match(tag):
        raise ValueError(f"constructed tag {tag!r} is invalid")
    return tag


@dataclass(frozen=True)
class RemoteJobRef:
    """A remote job as seen by the search (anonymised; no PII/token)."""

    job_id: str
    tags: tuple[str, ...] = ()
    job_config_hash: str | None = None
    status: str | None = None


@runtime_checkable
class RemoteJobSearch(Protocol):
    """Injected search seam. The real impl wraps ``service.jobs(job_tags=[tag])``; tests
    inject a fake. Implementations MUST raise a :class:`RemoteSearchError` subclass on any
    non-authoritative outcome (timeout/auth/quota/unavailable/malformed) — never return a
    partial list as if authoritative."""

    def find_by_tag(self, tag: str) -> list[RemoteJobRef]: ...


@dataclass(frozen=True)
class SubmissionDecision:
    action: str                                # submit_allowed|adopt_remote|already_submitted|refuse
    reason: str
    job_id: str | None = None
    tag: str | None = None

    @property
    def may_submit(self) -> bool:
        return self.action == "submit_allowed"


def decide_submission(
    *,
    batch_config_hash: str,
    job_config_hash: str,
    local_job_id: str | None,
    search: RemoteJobSearch,
) -> SubmissionDecision:
    """Decide whether a future submission is permitted. NEVER submits; only decides.
    Fails CLOSED (``refuse``) on any error/timeout/ambiguity — a failed search never
    authorises a submission."""
    tag = job_tag(batch_config_hash, job_config_hash)

    if local_job_id is not None:
        return SubmissionDecision("already_submitted", "a job id is already recorded locally",
                                  job_id=local_job_id, tag=tag)

    try:
        jobs = search.find_by_tag(tag)
    except RemoteSearchError as exc:
        return SubmissionDecision("refuse", f"remote search non-authoritative: "
                                  f"{type(exc).__name__}: {exc}", tag=tag)
    except Exception as exc:  # noqa: BLE001 — any unexpected failure is non-authoritative
        return SubmissionDecision("refuse", f"remote search raised {type(exc).__name__}: {exc}",
                                  tag=tag)

    if not isinstance(jobs, list) or any(not isinstance(j, RemoteJobRef) for j in jobs):
        return SubmissionDecision("refuse", "remote search returned a malformed result", tag=tag)

    # only jobs actually carrying this tag count (defensive against a loose fake/provider)
    matching = [j for j in jobs if tag in j.tags]
    if len(matching) == 0:
        return SubmissionDecision("submit_allowed", "no remote job carries this tag", tag=tag)
    if len(matching) > 1:
        return SubmissionDecision("refuse",
                                  f"{len(matching)} remote jobs share tag {tag} — fail closed",
                                  tag=tag)
    job = matching[0]
    if job.job_config_hash is not None and job.job_config_hash != job_config_hash:
        return SubmissionDecision("refuse",
                                  "remote job carries this tag but a different config hash "
                                  "(collision/tamper) — fail closed", job_id=job.job_id, tag=tag)
    return SubmissionDecision("adopt_remote", "adopt the existing remote job; do not submit",
                              job_id=job.job_id, tag=tag)


# --- fake service for tests (never reads credentials, never networks) -----------------
@dataclass
class FakeRemoteJobService:
    """Deterministic in-memory search for tests. Configure ``jobs`` to return, or set
    ``raise_error`` to a :class:`RemoteSearchError` instance to simulate timeout/auth/quota/
    unavailable/malformed. Records the tags it was queried with. No IBM, no token, no net."""

    jobs: list[RemoteJobRef] = field(default_factory=list)
    raise_error: RemoteSearchError | None = None
    malformed: bool = False
    queried_tags: list[str] = field(default_factory=list)

    def find_by_tag(self, tag: str) -> list[RemoteJobRef]:
        self.queried_tags.append(tag)
        if self.raise_error is not None:
            raise self.raise_error
        if self.malformed:
            return ["not-a-job-ref"]  # type: ignore[list-item]
        return [j for j in self.jobs if tag in j.tags]
