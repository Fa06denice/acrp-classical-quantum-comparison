# IBM QPU — candidate first run (PREPARED, NOT SUBMITTED)

Status: **the pipeline is verified and fail-closed; nothing was submitted.** This
document is the STOP point before any real (payable) IBM job. A real run requires
the explicit human confirmation described at the bottom.

## Pipeline verification (this session)

- 53 QPU/IBM/hardware tests pass (`pytest -k "qpu or hardware or ibm or submission or flags"`).
- Fail-closed feature flags (`src/acrpq/dashboard/qpu_flags.py`), every default safe:
  - `ACRPQ_QPU_ENABLED` → default **False**
  - `ACRPQ_QPU_DRY_RUN` → default **True** (forces dry-run)
  - `ACRPQ_IBM_SUBMISSION_ENABLED` → default **False**
  - `submission_allowed()` = `qpu_enabled AND ibm_submission_enabled AND NOT dry_run_only` → **False** unless all three are deliberately aligned.
- Credentials are read from the environment by qiskit-ibm-runtime only; the token
  never appears in the UI, logs, or any saved JSON.
- No IBM call happens at import or page load; dry-run / prepare / confirm never
  submit; the UI SUBMIT button is disabled by the server flags.

## Candidate configuration (smallest scientifically-meaningful case)

| Field | Value | Rationale |
|-------|-------|-----------|
| Instance | `CP_3` | smallest whole instance (3 aircraft) |
| Grid | `K=3`, `n_q=1` | smallest demonstrable maneuver grid |
| Qubits | **9** (`n × K`) | far below any hardware/simulator ceiling |
| QAOA depth | `reps = 1` | lowest depth → shallowest circuit |
| Shots | `2048` (bounded) | enough statistics, bounded cost |
| Optimizer | `maxiter ≈ 50` | bounded classical outer loop |
| Backend | least-busy **real** backend, chosen at the *assess* step | explainable ranking, not hard-coded |
| Budget cap | 1 job · 2048 shots · ≤ 300 s wall | hard ceiling |

Logical circuit (Aer, reps=1, pre-ISA): **depth 15, 18 two-qubit gates, 9 qubits,
~30 optimizer evaluations**. ISA-transpiled depth / 2-qubit-gate counts depend on
the selected backend's coupling map and are produced by the **dry-run** once a
real backend is assessed (not available without live credentials, so not invented
here).

## Procedure to run it (only when you decide to)

In the UI, under **MORE DETAILS → IBM QPU** (all steps are already wired):

1. **Assess backends** — rank candidate real backends (explainable).
2. **Dry run** — builds the plan and ISA circuit; *never* calls the gateway.
3. **Prepare** — freezes an artefact + budget; shows the run id and `sha256`.
4. **Confirm** — type the consent phrase exactly.
5. **Submit** — enabled *only* if the three flags below are set.

To actually enable submission (deliberate, out of band):

```bash
export ACRPQ_QPU_ENABLED=true
export ACRPQ_IBM_SUBMISSION_ENABLED=true
export ACRPQ_QPU_DRY_RUN=false
# IBM credentials via the environment (qiskit-ibm-runtime); never in the UI/logs
```

## Risks

- Real backends are noisy: at 9 qubits, depth 15, expect the sampled feasibility
  and objective to be **worse** than the Aer simulator; this is a hardware-noise
  measurement, not a result about the algorithm, and **never** a quantum speedup.
- Queue time and cost are account-dependent; the budget cap bounds exposure.
- ISA transpilation can raise depth / 2-qubit gates materially — re-read the
  dry-run numbers before confirming.

## STOP — exact confirmation required before a real job

I will not submit anything. To proceed with a real IBM run, reply explicitly, e.g.:

> "Submit the CP_3 K3 candidate to IBM backend `<name>`, reps 1, shots 2048,
>  budget 1 job / ≤300 s — I accept the queue time and cost."

I will then walk the assess → dry-run → prepare → confirm steps and stop again to
show you the ISA circuit, budget and consent phrase before the final submit.
