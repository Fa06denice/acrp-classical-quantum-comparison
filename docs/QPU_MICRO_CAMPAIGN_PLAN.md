# IBM micro-campaign plan (Phase 5.7) — NOT executable

A conservative plan for a possible future micro-campaign against ~10 minutes of QPU
credit. **This is a plan, not a command.** No job is submitted by anything in this
repository without a separate, explicit Fabio authorisation and the fail-closed flags
being deliberately flipped. Numbers are estimates to size the decision, not guarantees.

## Ground rules
- One objective and one QAOA configuration per job; a backend chosen at the last moment
  from availability; shots kept small.
- Every job is gated by: a durable `local_run_id` + single-winner claim, a deterministic
  dedup tag (`acrpq:<batch>:<job>`) searched remotely first (submit only on a *successful*
  zero-result search), a budget reservation, and a single-use confirmation nonce.
- **Queue depth shown by the provider does NOT guarantee wall-clock delay** — never plan
  as if a low queue means fast turnaround, or a high queue means safe idleness.

## Scenario A — minimal validation (recommended first)
- 1 instance (CP_3, 9 qubits at K=3), 1 objective (`maneuver_count_v1`), 1 QAOA protocol
  (`aer_default_p1_v1` transferred to hardware: reps=1, COBYLA, seeded angles), shots ≈ 512,
  **1 job**.
- Purpose: validate the end-to-end real path (submit → job_id → reconcile → result →
  normalise → decode) and the endianness/normalisation on a *real* result, against the
  known certified optimum — **plumbing + a single sanity point, not a quantum result**.
- Estimate: 1 job, ~512 shots, 9 qubits, likely a few seconds of QPU time (queue excluded);
  scientific value: low-but-necessary (de-risks the pipeline); risk: low; **stop criterion:
  after the single job completes, STOP and inspect — no auto-follow-up.**

## Scenario B — small comparative (only if A succeeded and budget remains)
- CP_3, CP_4, CP_5 (9/12/15 qubits) — begin with the smallest; add the next only if time
  and the previous result justify it. One objective first; the second objective only if a
  discriminating question needs it. No duplication.
- Estimate: up to 3 jobs, ~512 shots each, 9–15 qubits; a handful of QPU-seconds each
  (queue excluded); value: modest (feasibility-rate / decode behaviour vs the certified
  optimum across sizes on real hardware); risk: medium (qubit count, noise); **stop
  criterion: stop at the first infeasible/degenerate result or when the reserved budget /
  the 10-minute envelope is reached, whichever first.**

## Scenario C — after a failed pilot
- **No automatic resubmission.** Diagnose (a bad result, a network error, an ambiguous
  reconciliation) locally from the persisted run store and the recorded result. A network
  error or ambiguous search NEVER authorises a retry-submit. Preserve the remaining credit;
  a human decides the next step.
- Estimate: 0 new jobs; value: diagnostic; risk: none (no submission); stop: human decision.

## Pre-flight checklist (must all hold before any real job)
1. Explicit Fabio authorisation for a real submission, recorded.
2. Fail-closed flags deliberately set (`qpu_enabled`, `ibm_submission_enabled`,
   `dry_run` off) AND the runtime factory flag on — several independent conditions.
3. The result adapter validated against at least one **recorded** (not live) Runtime result.
4. A budget policy + reservation in place; a fresh single-use confirmation nonce bound to
   the exact identity.
5. The run-store batch-identity decision (open issue D1) resolved or explicitly accepted.
6. A named operator watching, ready to STOP.

## Explicitly out of scope of this plan
Building or enabling a working `submit` command; reading an IBM account/token; contacting
IBM; consuming any credit. All of that remains disabled.
