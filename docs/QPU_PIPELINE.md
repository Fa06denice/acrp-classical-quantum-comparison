# QPU pipeline architecture (Phases 2–9)

A local, explainable, crash-safe pipeline that goes from the existing scientific
QUBO to a guarded IBM QAOA lifecycle. A real submission seam now exists, but
every payable action fails closed behind trust-chain, feature-flag, budget and
single-use human-confirmation guards. No development test or recent hardening run
contacted IBM. Qiskit is imported lazily everywhere.

## Modules

| Phase | Module | Responsibility |
|---|---|---|
| 2 | `qpu_runs.py` | event-sourced, crash-safe run store + state machine |
| 3 | `ibm_runner.py` | idempotent submission / reconciliation / cancel over an injectable gateway (fakes in tests) |
| 4 | `backend_scoring.py` | explainable, deterministic backend ranking + `workload_commitment` |
| 5 | `hardware_validation.py` | QAOA build + fake transpile + immutable artefact + decode |
| 5.1 | (same) | trust chain: bind decision → workload → backend → circuit → payload |
| 6 | `qpu_orchestrator.py` | immutable execution plan, dry-run, fake-gateway lifecycle |
| 7 | `qpu_flags.py`, `qpu_confirm.py`, `qpu_api.py` | fail-closed flags, replay-proof confirm, guarded API |
| 9 | `qpu_export.py` | verified atomic exports + read-only startup recovery |

## State machine (`QpuState`)

`prepared → awaiting_confirmation → submitting → submitted → queued → running →
completed`, with `failed` / `cancel_requested` / `cancelled`, and the ambiguity
lanes `submission_unknown` and `reconciliation_required`. Hardware modes MUST pass
through `awaiting_confirmation` before `submitting`; `submission_unknown` cannot go
directly to `failed` (it must be reconciled); terminal states are immutable except
append-only probative evidence. Transitions are an append-only, tamper-evident,
inter-process-safe event log.

## Trust chain (Phase 5.1) — what makes a payload trustworthy

```
WorkloadRequirements ──workload_commitment()──► decision.workload_sha256
        │                                              │
        ▼                                              ▼
Phase-4 rank_backends ──► decision (verify_decision_hash, selected, applicable,
        │                            uncertainty ≤ block, not stale)
        ▼
HardwareValidationRequest ─┬─ backend_requested == real backend .name == decision.selected
                           │                      == ranking row name == artefact.backend
                           ▼
prepare_hardware_bundle ──► PreparedHardwareRun (artefact_sha256 over whole manifest)
        │                     + real logical/ISA circuits (PreparedCircuitBundle)
        │   submittable = true  ⟺  verified decision AND real (non-fake) backend
        │                          AND non-synthetic snapshot  (else false, always)
        ▼
verify_bundle_integrity ──► artefact hash + ISA re-fingerprint + version + fully-bound
        │                    + measurements + qubit count
        ▼
build_execution_plan ──► QpuExecutionPlan (plan_sha256, budgets, coherence)
        ▼
to_submission_payload ──► fails closed unless submittable AND bundle intact
```

Decoding computes all scientific aggregates (best-raw / best-feasible / modal /
feasibility-rate / mean / dispersion / optimality-gap) over the **full** validated
distribution; truncation only shortens the returned detail list.

## API (mounted under `/api/qpu`)

`GET /flags` · `POST /backends/assess` · `POST /dry-run` · `POST /prepare` ·
`POST /{run_id}/confirm` · `GET /{run_id}` · `GET /{run_id}/events` ·
`POST /{run_id}/cancel` · guarded `POST /{run_id}/submit` ·
`POST /{run_id}/reconcile` · `GET /{run_id}/result` ·
`POST /{run_id}/export`. A separate `/simulate/*` seam exercises the lifecycle
only with an injected dry-run-safe fake gateway. Mutations require a loopback
client, allowed Host/Origin, bounded request body and local rate limit.
Confirmation needs a single-use, expiring nonce bound to the run, hashes,
backend, budget and consent phrase. `/submit` refuses fake/synthetic or
non-submittable chains before constructing the real gateway.

## Feature flags (fail-closed)

`ACRPQ_QPU_ENABLED` (default false), `ACRPQ_QPU_DRY_RUN` (default true),
`ACRPQ_IBM_SUBMISSION_ENABLED` (default false) and
`ACRPQ_IBM_RUNTIME_FACTORY_ENABLED` (default false). A real submission needs the
submission trio aligned and the independently guarded runtime factory enabled;
absent/invalid values fail closed. Full-QAOA exposes a separate intent flag, but
its current endpoint remains dry-run-only even if that flag is enabled. See
`docs/IBM_QPU_RUNBOOK.md`.

## Scientific limits

Heuristic backend ranking (not a success probability); hardware-validation is for
small instances only; logical qubits ≠ physical qubits after routing; a hardware
result is noisy sampling with no feasibility/optimality guarantee. See
`docs/BACKEND_SCORING.md` and `docs/HARDWARE_VALIDATION.md`.
