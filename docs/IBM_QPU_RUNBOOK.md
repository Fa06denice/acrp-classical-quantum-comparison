# IBM QPU runbook — dry-run today, first real job later (operator guide)

This is the operating procedure for the local QPU pipeline. **As shipped, a bare
installation cannot submit a real IBM job**: every feature flag defaults to the
safe value and no IBM account/token is configured. The real-submission endpoint
and lazy runtime factory now exist, but both fail closed behind independent
guards. A first real job is possible **only** after a human completes the gated
checklist below on their own machine, with their own account.

> **Never** put an IBM token in this document, in a command you record, in a
> commit, or in any exported file. The pipeline never reads a token; keep it that
> way.

## What is safe today (no account, no cost)

- **Dry-run** end to end: choose a small instance → local classical comparison →
  explainable backend scoring (Phase 4) → QAOA preparation + fake-backend
  transpilation (Phase 5) → immutable execution plan (Phase 6) → a report of
  *what would be submitted* — with **no gateway call, no token, no spend**.
- **Fake-gateway lifecycle**: `qpu_orchestrator.simulate_lifecycle` drives
  prepare → confirm → submit → reconcile → completed → decode through a **fake**
  gateway; a real `RuntimeGateway` is refused.
- Artefacts prepared against a fake backend are **always** `submittable = false`.

## Real vs. fake: two separate paths (safety boundary)

`POST /api/qpu/{id}/submit` is the **real** path. It refuses, before building any
gateway (so `run_sampler` is never reached), anything that is not a
`real_submittable` chain: a fake/simulated backend, a synthetic snapshot/decision,
a non-submittable/dry-run artefact, a plan marked not-submittable, a business-hash
that does not verify, or a confirmation whose consumed marker is missing/`{}`/
forged/not-bound-to-the-chain. It also refuses a `dry_run_safe` gateway. Because a
bare local server only ever has **fake** backends, `/submit` always refuses locally
(this is the point).

The **fake lifecycle** (submit mechanics, reconciliation, decode) runs only on the
`POST /api/qpu/{id}/simulate/submit` + `/simulate/reconcile` **seam**, which is
enabled only when a test/dev fake gateway is injected and requires a `dry_run_safe`
gateway. It never touches `/submit`.

Two previously-reproduced bypasses are closed and regression-tested: a fake artefact
riding `/submit`, and a `{}` `confirmation_consumed.json` authorising submission —
both now yield `409` with `run_sampler == 0`.

## Feature flags (fail-closed)

| Env var | Default | Meaning |
|---|---|---|
| `ACRPQ_QPU_ENABLED` | `false` | QPU workflow switched on at all (prepare/confirm) |
| `ACRPQ_QPU_DRY_RUN` | `true`  | force dry-run only (safe default) |
| `ACRPQ_IBM_SUBMISSION_ENABLED` | `false` | allow real (billable) submission |
| `ACRPQ_IBM_RUNTIME_FACTORY_ENABLED` | `false` | allow the real gateway factory to build a `QiskitRuntimeService` (still last-moment, token via IBM only) |
| `ACRPQ_FULL_QAOA_SUBMISSION_ENABLED` | `false` | allow real full-QAOA submission (kept off; full-QAOA is dry-run only) |

A real submission is permitted **only** when all three align
(`qpu_flags.submission_allowed()` → `qpu_enabled AND ibm_submission_enabled AND
NOT dry_run_only`). An absent or unrecognised value always resolves to the safe
setting. Even a fully UI-confirmed run reports `submission_allowed = false` until
these are set **and** a real gateway/account is wired in.

## Pre-flight checklist — REQUIRED before any first real job

Do not skip a step. Each is a gate; if it is not green, stop.

1. **IBM account configured by the operator** — the user sets up their own
   `qiskit-ibm-runtime` account/instance in their environment. The repo ships no
   token and no account.
2. **Backend visible** — confirm the target backend is real, operational and
   visible to the account (outside this repo's fake providers).
3. **Flags set explicitly** — `ACRPQ_QPU_ENABLED=true`,
   `ACRPQ_IBM_SUBMISSION_ENABLED=true`, `ACRPQ_QPU_DRY_RUN=false`, and verify
   `GET /api/qpu/flags` shows `submission_allowed=true`.
4. **Dry-run green** — run the exact configuration through `/api/qpu/dry-run`
   (or the orchestrator `dry_run`) and read the `would_submit` report. It must be
   coherent (backend, shots, budgets, ISA depth/2Q).
5. **Hashes verified** — check `verify_decision_hash`, `verify_artefact_hash`,
   `verify_plan_hash`, and `check_submission_consistency` (or the fail-closed
   `assert_submission_consistency`) all pass for the prepared run.
6. **Budget confirmed** — confirm `max_jobs`, total shots and
   `max_quantum_seconds` are the intended small values, and complete the explicit
   `/api/qpu/{run_id}/confirm` with the single-use nonce and the consent phrase.
7. **One small instance only** — pick a single small instance (e.g. `CP_3` /
   `CP_4`, ≤ ~9 logical qubits) whose QUBO fits the backend after routing. A high
   qubit count does not make a dense QUBO runnable.
8. **Prudent shots** — start with conservative shots (e.g. 512–1024); do not
   raise them for a first run.
9. **Watch the job** — monitor the job (status/events) actively; never fire-and-
   forget. Submissions are never auto-retried.
10. **Export + reconcile** — after completion, reconcile the run to a true state
    and write a verified export (`qpu_export.export_run`) — decision, artefact,
    plan, events, decoded result, hashes. On any ambiguity, reconcile; never
    resubmit.
11. **Disable the flag afterwards** — set `ACRPQ_IBM_SUBMISSION_ENABLED=false`
    (and ideally `ACRPQ_QPU_DRY_RUN=true`) immediately after the trial so the
    system returns to its safe default.

## What is built, and what remains unvalidated

The trust chain, budgets, confirmation, plan, decode, guarded `/submit` endpoint
and lazy `RealIbmGatewayFactory` are present and tested against fakes. The
default factory constructs `QiskitRuntimeService` only at the last possible
moment, after every consistency and permission guard, and lets IBM's standard
account mechanism provide credentials; this project does not accept or export a
token. Reaching that path requires: (a) the operator's account, (b) all four
real-execution flags, (c) a real-submittable hardware chain, and (d) a fresh
single-use human confirmation for the run. No test constructs a real service and
no real job was launched during development. Until a job has actually run and
been reconciled, the system is **dry-run / demo ready and technically prepared
for a first small job under human confirmation — but NOT "QPU-validated"**.

## Threat model & residual risks (from the red-team review)

This is a **local, single-user** tool. The following are known, deliberate
boundaries — documented rather than over-engineered away:

- **Hashes are integrity, not authentication.** `verify_decision_hash`,
  `verify_artefact_hash`, `verify_plan_hash` and `verify_export` prove a record
  has not been *mutated* since it was produced; they are unkeyed SHA-256, so they
  do **not** prove *provenance*. The trust chain relies on the decision/artefact/
  plan being produced **in-process by the trusted pipeline**. The HTTP API
  enforces this: it always self-generates the decision inside the same request and
  never accepts a decision/artefact/bundle blob from a client. Do not build a
  path that ingests an externally-supplied decision/bundle and treats a passing
  `verify_*` as authentication; if you ever need that, add an HMAC/signature with
  a server-held key.
- **Loopback check vs. reverse proxy.** `_require_local` inspects the TCP peer
  (`request.client.host`). Behind a reverse proxy on localhost every request
  appears to come from `127.0.0.1`, so the check cannot see the true origin.
  Run the dashboard **directly on loopback with no proxy** (the shipped default
  binds `127.0.0.1`); do not expose it via a proxy without adding real auth.
- **Confirmation nonce store is in-memory / process-local.** With multiple ASGI
  workers a `prepare` and `confirm` on different workers will fail (fail-closed),
  and a restart orphans a prepared run's nonce (re-prepare to recover). Run a
  single worker, or wire a shared store, before scaling out.
- **Confirm → transition is not one atomic step.** If the state transition after
  a successful nonce-consume fails (e.g. a concurrent cancel), the one-time nonce
  is spent with no compensation (fail-closed: no illegitimate transition occurs).
  Re-prepare to retry.

## Recovery after a crash

On startup, `qpu_export.startup_recovery_report` classifies runs
(corrupted / reconciliation-candidates / non-terminal / terminal) **read-only**;
it never auto-submits. Reconcile ambiguous runs explicitly; a run that crashed
after a remote job id was recorded is recoverable and must be reconciled, never
resubmitted.
