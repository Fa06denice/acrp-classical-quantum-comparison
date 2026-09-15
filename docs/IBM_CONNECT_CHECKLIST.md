# IBM Quantum connect + micro-campaign operator checklist

Audience: the single human operator (Fabio) connecting a real IBM Quantum account
to this repo and running a small, bounded fixed-angle sampling campaign.
Every flag, endpoint and guard cited below is the actual code in the working tree
(branch `acrpq-scientific-ui`); citations are `file:line`.

This is the exact sequence that produced the recorded 2026-09-03
`ibm_marrakesh` campaign (`results/ibm_fixed_angle_campaign_v1/manifest.json`,
30 completed jobs, 512 shots each), formalised.

---

## 0. Hard rules (read before anything else)

1. **The token never enters this repo.** No env var, no CLI argument, no request
   body, no file under the repo. The service is built only by IBM's own account
   store (`src/acrpq/dashboard/qpu_ibm_factory.py:42-48`,
   `_DefaultServiceFactory.make_service()` → `QiskitRuntimeService()` reads the
   saved account itself). Nothing in the API accepts a token field.
2. **Never resubmit on ambiguity.** Any of these verdicts means STOP and
   reconcile — never call submit again for the same run:
   `submission_unknown`, `pre_submit_search_failed`, `existing_job_found`,
   `duplicate_remote_jobs` (`src/acrpq/dashboard/ibm_runner.py:343-360, 399-403`).
   A crash mid-submit leaves the run reconcilable; `reconcile_run` never
   resubmits (`ibm_runner.py:452-464`).
3. **Queue depth is not a wall-clock guarantee.** `pending_jobs` from the live
   snapshot (`src/acrpq/dashboard/qpu_api.py:196-209`) is advisory only. The
   only per-job execution cap the provider honours is `max_execution_time`
   (from `max_quantum_seconds`, ceiled to whole seconds,
   `ibm_runner.py:212-216, 678-680`); queue wait is unbounded.
4. **Stop criteria.** Stop the campaign immediately if: (a) any run enters
   `submission_unknown` or `reconciliation_required` and a follow-up
   `/reconcile` does not resolve it; (b) `duplicate_remote_jobs` is ever
   reported; (c) a canary result is not retrievable; (d) the confirmation
   summary shows shots/seconds you did not intend. Cancellation via
   `/api/qpu/{run_id}/cancel` only works BEFORE submission
   (`qpu_api.py:695-708`); a submitted job must be cancelled through
   `ibm_runner.request_cancel` semantics — and "a running computation may
   finish server-side" (`ibm_runner.py:642-643`) — i.e. cancellation is
   best-effort, not a spend guarantee.
5. **Never edit anything under `runs/qpu/` or `results/ibm_*`.** Run
   directories are hash-chained (`qpu_runs`), and every stored document is
   self-hashed (`qpu_store`); manual edits turn runs into `corrupted` and
   poison the dedup/resume logic (`src/acrpq/dashboard/qpu_batch.py:525-559`).

---

## 1. Account setup (outside the repo)

Done once, in a throwaway Python session (NOT saved in any repo file):

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
                                  token="<paste, then clear shell history>",
                                  set_as_default=True)
```

Verify read-only, still outside the repo: `QiskitRuntimeService().backends()`.
The repo will only ever call `QiskitRuntimeService()` with no arguments
(`qpu_ibm_factory.py:46-48`, `qpu_api.py:161-163`).

## 2. Feature flags (all fail closed; defaults refuse submission)

A real submission requires ALL of these simultaneously
(`src/acrpq/dashboard/qpu_flags.py:53-57` and
`src/acrpq/dashboard/qpu_ibm_factory.py:30-32`):

| Env var | Value for a real run | Default (unset) | Source |
|---|---|---|---|
| `ACRPQ_QPU_ENABLED` | `true` | `false` (workflow off) | `qpu_flags.py:38-40` |
| `ACRPQ_QPU_DRY_RUN` | `false` | `true` (forced dry-run) | `qpu_flags.py:43-45` |
| `ACRPQ_IBM_SUBMISSION_ENABLED` | `true` | `false` | `qpu_flags.py:48-50` |
| `ACRPQ_IBM_RUNTIME_FACTORY_ENABLED` | `true` | disabled (503 at the gateway seam) | `qpu_ibm_factory.py:30-32`, `qpu_api.py:736-751` |
| `ACRPQ_RUNS_DIR` | a PERSISTENT path (default `runs`) | `runs` | `api.py:1336` |

Unrecognised/empty values fall back to the SAFE default (`qpu_flags.py:24-35`).
`ACRPQ_FULL_QAOA_SUBMISSION_ENABLED` is a separate flag for the (still dry-run
only) full-QAOA plan route (`qpu_api.py:730-732`); leave it unset.

Set the flags ONLY in the shell that launches the local server, e.g.:

```bash
export ACRPQ_QPU_ENABLED=true
export ACRPQ_QPU_DRY_RUN=false
export ACRPQ_IBM_SUBMISSION_ENABLED=true
export ACRPQ_IBM_RUNTIME_FACTORY_ENABLED=true
.venv/bin/python -c "import uvicorn; from acrpq.dashboard.api import create_app; \
uvicorn.run(create_app(), host='127.0.0.1', port=8055)"
```

The server binds loopback; every QPU mutation additionally requires a loopback
client + Host/Origin check + JSON content type + body cap + rate limit
(`qpu_api.py:451-498`).

Verify: `GET /api/qpu/flags` must return
`{"qpu_enabled": true, "dry_run_only": false, "ibm_submission_enabled": true,
"submission_allowed": true, "runtime_factory_enabled": true}`
(`qpu_api.py:513-520`). If `submission_allowed` is false, stop — nothing will
submit and nothing should.

## 3. Pre-flight (per campaign, before any submission)

1. **Dry-run first, always.** With the same body you will use for prepare,
   `POST /api/qpu/dry-run` (`qpu_api.py:582-598`). Check `would_submit`
   (backend, shots, config hash, ISA fingerprint, `max_quantum_seconds`) and
   `budget` — no gateway, no persistence.
2. **Backend choice.** `GET /api/qpu/backends/live/ibm_marrakesh`
   (`qpu_api.py:540-556`) — read-only snapshot: `operational: true`, low
   `two_qubit_error`, note `pending_jobs` (advisory). The prepare body must
   embed exactly this snapshot; the scorer must select the requested backend or
   prepare is refused (`qpu_api.py:324-327`).
3. **Size/cost sanity per job** (recorded campaign values in brackets):
   - qubits: `n_binary_vars = n_aircraft × K` [9–21 for CP_3–CP_7 at K=3],
     must fit the backend (`hardware_validation.py:669-670`);
   - ISA depth / 2Q gates: from the prepare response `isa` block
     (`qpu_api.py:635-636`) [CP_3: depth 211, 108 CZ at opt level 0]; set
     `max_depth` / `max_two_qubit_gates` budgets if you want a hard refusal
     (`hardware_validation.py:694-699`);
   - shots [512], `max_jobs` (hard cap 5 per hardware-validation run,
     `hardware_validation.py:79`), `max_quantum_seconds` [20.0; hard cap 600,
     `hardware_validation.py:80`];
   - transpiler seed [1] and `optimization_level` [0] — prepare refuses a
     non-reproducible transpilation (double-transpile check,
     `hardware_validation.py:678-688`).
4. **Hashes.** The prepare response returns the whole chain:
   `qubo_sha256` (canonical QUBO+objective, `hardware_validation.py:236-260`),
   `config_hash` (Phase-3 params, `hardware_validation.py:869-883`),
   `decision_sha256`, `artefact_sha256`, `plan_sha256`, ISA fingerprint. All
   are persisted per run and re-verified before submit by
   `qpu_store.verify_full_run` (`qpu_store.py:218+`) — `real_submittable`
   requires a complete, coherent, non-fake, non-synthetic chain
   (`qpu_api.py:753-791`). Record `artefact_sha256` in your campaign notes.
5. **Budget arithmetic (manual, no ledger on the API path).** The API binds the
   budget into the human confirmation but keeps NO monetary ledger: total spend
   = jobs × shots × per-shot cost + queue-side overhead, bounded per job by
   `max_quantum_seconds`. For the recorded campaign: 30 jobs × 512 shots,
   ≤ 20 quantum-seconds per job. Decide the total job count BEFORE starting and
   pass it to the campaign script (`--max-total-jobs`).

## 4. The per-run gate sequence (what you actually do)

For each job the flow is prepare → confirm → submit → reconcile → result →
export, all on loopback:

1. `POST /api/qpu/prepare` (`qpu_api.py:600-641`) — builds the QUBO, ranks the
   LIVE snapshot, transpiles, persists decision/artefact/plan/ISA, issues a
   single-use confirmation nonce (TTL 300 s, `qpu_confirm.py:26`). State:
   `prepared`.
2. `POST /api/qpu/{run_id}/confirm` (`qpu_api.py:643-675`) — echo back nonce +
   `config_hash` + `artefact_sha256` + `decision_sha256` + backend +
   `total_shots` + `max_jobs` + `max_quantum_seconds` + the exact consent
   phrase `"I understand this may submit a real, billable IBM Quantum job"`
   (`qpu_confirm.py:25`). **This is the human-confirmation point: read the
   budget you are echoing.** Single-use, atomic, durable; replays are refused.
   State: `awaiting_confirmation`. Confirmation NEVER submits.
3. `POST /api/qpu/{run_id}/submit` (`qpu_api.py:793-833`) — in order:
   flags gate (403), full-chain `real_submittable` verification (409),
   consumed-confirmation cryptographic verification bound to every hash (409),
   gateway built only now (503 if factory disabled), fake gateways refused
   (`qpu_api.py:806-807`), then `ibm_runner.submit_run` runs the skew-adjusted
   pre-submit tag search, the `max_jobs` budget check, and the point-of-use
   `ApiHardwareAuthorizer` (fresh remote dedup + chain re-verify,
   `ibm_runner.py:375-391`, `qpu_submit_gate.py:139-160`) before the one
   `run_sampler` call. Job id is persisted immediately.
4. `POST /api/qpu/{run_id}/reconcile` (`qpu_api.py:920-934`) — poll until
   `completed`; counts are fetched ONCE and persisted as `result_raw`
   (idempotent, never fabricated, `qpu_api.py:835-855`).
5. `GET /api/qpu/{run_id}/result` then `POST /api/qpu/{run_id}/export`
   (`qpu_api.py:936-988`) — decode + write `export.json` next to the run.

### Micro-campaign driver

`scripts/run_ibm_micro_campaign.py` automates exactly that sequence with:
loopback-only base URL (line 27), explicit double-ack
(`--execute --ack I_AUTHORIZE_REAL_IBM_QPU_SUBMISSIONS`, lines 20, 138-139),
flags re-check against the server (142-144), resume-before-create (147-168),
2 canary jobs that must return retrieved results before the rest proceeds
(193-197, 218-227), atomic manifest writes, and a `--max-total-jobs` cap.

```bash
# 1. DRY RUN of the plumbing first (server flags still safe): expect a clean refusal
.venv/bin/python scripts/run_ibm_micro_campaign.py            # refuses: no --execute
# 2. Real run, only after sections 2-3 all check out:
.venv/bin/python scripts/run_ibm_micro_campaign.py \
  --execute --ack I_AUTHORIZE_REAL_IBM_QPU_SUBMISSIONS \
  --backend ibm_marrakesh --instances CP_3 CP_4 CP_5 CP_6 CP_7 \
  --repetitions 3 --shots 512 --max-quantum-seconds 20 --max-total-jobs 30 \
  --out results/<new_campaign_dir>/manifest.json
```

Use a NEW `--out` path for a new protocol; the script refuses to extend a
manifest whose protocol differs (lines 80-89).

## 5. Verify after each step

| After | Verify |
|---|---|
| flags | `GET /api/qpu/flags` shows the exact expected 5 booleans |
| dry-run | `would_submit` backend/shots/seconds match intent; `submittable` true |
| prepare | response hashes recorded; `state == "prepared"`; warnings empty |
| confirm | `state == "awaiting_confirmation"`; the echoed budget is what you meant |
| submit | verdict `submitted` + exactly ONE job id; anything else → rule 0.2 |
| reconcile | run reaches `completed`; `result_raw` exists (result route `available: true`) |
| result | `n_shots` == requested shots; `feasibility_rate` ∈ [0,1]; counts bit length == n_binary_vars |
| export | `export.json` present in `runs/<ACRPQ_RUNS_DIR>/qpu/<run_id>/artefacts/` |
| campaign end | `pytest tests/test_qpu_real_artifact.py` — chain-verifies every recorded run, re-derives the QUBO hash, re-normalises and re-decodes the real counts |

## 6. After the campaign

1. Unset all four flags (or close the shell) — the server reverts to fail-closed.
2. Snapshot `runs/qpu/` and the campaign manifest into results storage; never
   rewrite them.
3. Cross-check spend in the IBM dashboard against `jobs × shots` and the
   recorded `ibm_job_ids` (each manifest job carries its provider job id —
   that id is the externally verifiable link to IBM's records).
