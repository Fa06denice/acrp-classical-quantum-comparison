"""Point-of-use hardware submission gate (Codex coordinated integration) — no IBM.

Codex requires that the ONLY real ``run_sampler`` call be preceded, at the point of use, by
ALL of these invariants holding simultaneously:

1. an AUTHORITATIVE batch identity in the run chain (``assert_hardware_submittable`` — a
   ``legacy_unverified`` run is refused);
2. a CONSUMED single-use confirmation nonce bound to the exact identity;
3. a COMMITTED budget reservation;
4. an AUTHORITATIVE remote-dedup decision of ``submit_allowed`` (a successful search that
   returned zero jobs — never a failed/ambiguous search).

:func:`authorize_hardware_submission` is a VERIFY-ONLY gate (no side effects): it raises
:class:`SubmissionForbidden` if ANY invariant is absent, listing every missing one. It is
the single check to place immediately before ``run_sampler``.

**This module never contacts IBM, builds a service, reads a token, opens a socket, or calls
``run_sampler`` itself.** The real submit path (``ibm_runner.submit_run``) invokes the exact
``HardwareAuthorizer`` immediately before ``gateway.run_sampler(...)``. That integration
is covered by point-of-use tests, including a composed non-fake gateway that previously
could have bypassed a concrete-class-only check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from . import qpu_batch
from .qpu_batch import OfflineSafetyError
from .qpu_dedup import SubmissionDecision, decide_submission, job_tag


class SubmissionForbidden(RuntimeError):
    """One or more hardware-submission invariants are absent; run_sampler must not fire."""


def authorize_hardware_submission(
    *,
    root: str,
    local_run_id: str,
    batch_config_hash: str,
    job_config_hash: str,
    confirmation: Any,
    confirmation_store: Any,
    ledger: Any,
    reservation: Any,
    dedup_decision: Any,
) -> None:
    """Verify EVERY invariant holds AND is cross-bound to the SAME (batch, job) identity;
    raise :class:`SubmissionForbidden` (listing all missing ones) if not. No side effects,
    no IBM. Call immediately before run_sampler.

    Cross-binding matters: a confirmation/reservation/dedup minted for a cheap job must not
    authorise a different (expensive) run — each is required to name exactly this
    ``batch_config_hash``/``job_config_hash`` (red-team finding #1)."""
    missing: list[str] = []

    # 1. authoritative batch identity in the run chain (refuses legacy_unverified)
    try:
        qpu_batch.assert_hardware_submittable(root, local_run_id, batch_config_hash)
    except OfflineSafetyError as exc:
        missing.append(f"identity_not_authoritative:{exc}")
    except Exception as exc:  # noqa: BLE001 — any identity error blocks submission
        missing.append(f"identity_check_failed:{type(exc).__name__}")

    # 2. consumed single-use confirmation, BOUND to this exact identity
    try:
        if not confirmation_store.is_consumed(confirmation):
            missing.append("confirmation_not_consumed")
    except Exception as exc:  # noqa: BLE001
        missing.append(f"confirmation_check_failed:{type(exc).__name__}")
    if getattr(confirmation, "batch_config_hash", None) != batch_config_hash:
        missing.append("confirmation_batch_mismatch")
    if getattr(confirmation, "job_config_hash", None) != job_config_hash:
        missing.append("confirmation_job_mismatch")

    # 3. committed budget reservation, BOUND to this exact identity
    try:
        if not ledger.is_committed(reservation):
            missing.append("budget_not_committed")
    except Exception as exc:  # noqa: BLE001
        missing.append(f"budget_check_failed:{type(exc).__name__}")
    if getattr(reservation, "batch_config_hash", None) != batch_config_hash:
        missing.append("budget_batch_mismatch")
    if getattr(reservation, "job_config_hash", None) != job_config_hash:
        missing.append("budget_job_mismatch")

    # 4. authoritative remote-dedup decision == submit_allowed, FOR this identity's tag
    if not (isinstance(dedup_decision, SubmissionDecision)
            and dedup_decision.action == "submit_allowed"):
        missing.append(f"dedup_not_submit_allowed:{getattr(dedup_decision, 'action', None)}")
    elif dedup_decision.tag != job_tag(batch_config_hash, job_config_hash):
        missing.append("dedup_tag_mismatch")

    if missing:
        raise SubmissionForbidden(
            "hardware submission refused — missing/mismatched invariants: " + "; ".join(missing))


@dataclass
class HardwareAuthorizer:
    """The object a real submit path invokes AT THE POINT OF USE (Codex items 1-2).

    ``authorize_and_tag`` (a) executes remote dedup FRESH against the injected search — it
    never trusts a caller-supplied decision — refusing unless the search authoritatively
    returned zero jobs; (b) runs the full :func:`authorize_hardware_submission` gate with
    that fresh decision; and (c) returns EXACTLY the deterministic tag to submit under. Any
    failure raises (``SubmissionForbidden``/search error), so ``run_sampler`` never fires.
    It builds no service, reads no token, and does not itself call ``run_sampler``."""

    root: str
    local_run_id: str
    batch_config_hash: str
    job_config_hash: str
    confirmation: Any
    confirmation_store: Any
    ledger: Any
    reservation: Any
    search: Any                       # a qpu_dedup.RemoteJobSearch (fake in tests)

    def authorize_and_tag(self) -> list[str]:
        decision = decide_submission(
            batch_config_hash=self.batch_config_hash, job_config_hash=self.job_config_hash,
            local_job_id=None, search=self.search)          # FRESH, in-line
        if decision.action != "submit_allowed":
            raise SubmissionForbidden(
                f"point-of-use dedup is {decision.action}, not submit_allowed ({decision.reason})")
        authorize_hardware_submission(
            root=self.root, local_run_id=self.local_run_id,
            batch_config_hash=self.batch_config_hash, job_config_hash=self.job_config_hash,
            confirmation=self.confirmation, confirmation_store=self.confirmation_store,
            ledger=self.ledger, reservation=self.reservation, dedup_decision=decision)
        tag = decision.tag              # submit_allowed always carries the tag
        if tag is None:
            raise SubmissionForbidden("submit_allowed decision has no tag")
        return [tag]                    # submit under EXACTLY the fresh dedup tag


@dataclass
class ApiHardwareAuthorizer:
    """Point-of-use authorizer for the dashboard's persisted single-run chain.

    The batch authorizer above binds a batch ledger.  Dashboard runs instead bind
    their per-run job/shot/time budget into the durable consumed confirmation.
    This authorizer re-verifies that complete chain and performs a fresh remote
    search immediately before submission.  It is deliberately a distinct exact
    type so neither trust model can be spoofed by a duck-typed wrapper.
    """

    local_run_id: str
    tag: str
    created_after_iso: str
    verify_chain: Callable[[], None]
    search: Any

    def authorize_and_tag(self) -> list[str]:
        # the tag IS the run identity for this trust model; a mismatch means the
        # authorizer was minted for a different run (MoE Expert E, IMPORTANT-2)
        if self.tag != self.local_run_id:
            raise SubmissionForbidden(
                "authorizer tag does not match its local_run_id")
        self.verify_chain()
        jobs = self.search.find_jobs(tags=[self.tag], created_after_iso=self.created_after_iso)
        if jobs:
            raise SubmissionForbidden("point-of-use dedup found an existing remote job")
        return [self.tag]
