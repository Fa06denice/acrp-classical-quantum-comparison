# QPU Submission Pipeline — Architecture Audit (Phase 5.2.A)

Read-only cartography of the QPU preparation/submission/reconciliation pipeline,
produced before any Phase 5.2 hardening. Point-in-time against `HEAD = 690554f`
(branch `acrpq-scientific-ui`). Line numbers drift as later phases edit files; treat
them as anchors at that commit.

**Scope note.** This maps *what exists*. It proposes no fixes. Phases 5.2.C–5.2.E act
on the gaps identified in §5–§6.

---

## 0. Three drivers, one engine

One low-level submit/reconcile engine (`ibm_runner`, `qpu_runs`) is driven from three
places:

| Driver | Entry | Gateway | Fence | Credit-capable? |
|---|---|---|---|---|
| **Offline CLI** (audit focus) | `scripts/run_qpu_batch_offline.py` → `qpu_batch.run_offline_qpu_job` → `_run_offline_lifecycle` | `FakeBatchGateway` | **exact type** | No |
| Orchestrator simulate seam | `qpu_orchestrator.simulate_lifecycle` | any `dry_run_safe=True` | marker (weaker) | No (fake only), NOT on CLI path |
| Real API | `qpu_api.py` `/submit` | real `RuntimeGateway` | flags + confirmation chain | **Yes**, gated, NOT reachable from CLI |

**Import-boundary fact:** `qpu_batch.py` imports only `ibm_runner, qpu_flags,
qpu_ibm_factory, qpu_orchestrator, qpu_runs, hardware_validation`. It never imports
`qpu_api` or `quantum.backends`. Therefore no credit-capable code is reachable from the
offline CLI.

---

## 1. Call diagram — offline CLI

```
main()  [run_qpu_batch_offline.py]
├─ assert_offline()                                   ← FENCE #1 (refuse if submission could be enabled)
├─ InstanceLoader().load(...)
└─ for instance × objective:
   run_offline_qpu_job(...)
   ├─ import qiskit_ibm_runtime.fake_provider          (LOCAL fakes only)
   ├─ backend allow-list "Fake*"                        ← FENCE (backend)
   ├─ build_qubo(...) / canonical_qubo_hash(...)        (pure)
   ├─ batch_config_hash(batch_config_identity(...))     (pure, full identity)
   ├─ _assert_matching_config(store, job_key, cfg)      ← FENCE (config drift)
   ├─ store.load → early-return if stage==exported
   ├─ optimum_onehot_bits → exhaustive_discrete_optimum (pure compute)
   ├─ prepare_hardware_bundle(req, qubo, FAKE backend)  (CPU transpile, NO network)
   ├─ build_execution_plan / dry_run                    (pure, budgets)
   ├─ gateway = FakeBatchGateway(synthetic)
   └─ _run_offline_lifecycle(...)
      ├─ require_offline_exact_gateway(gateway)          ← FENCE #2 (EXACT TYPE)
      ├─ _assert_matching_config(...)                    ← FENCE (config drift)
      ├─ prior = store.load(job_key)                      (resume via index)
      ├─ else existing = _find_run_by_config_hash(...)    (scan runs_dir authority)
      │        _claimed_by_other_config(...)
      ├─ else FRESH: qpu_runs.prepare → transition → store.record("prepared")   ← RACE SITE (§6)
      ├─ if AWAITING and no job: ibm_runner.submit_run(...)   ← run_sampler ONCE (fake here)
      │        ├─ _require_confirmation(cfg_hash)          ← FENCE (confirmation)
      │        ├─ find_jobs(tags=[local_run_id])            (idempotent tag search)
      │        ├─ max_jobs budget check                     ← FENCE (budget)
      │        └─ run_sampler(tags=[local_run_id])          ← credit IFF real gateway
      ├─ if not terminal: reconcile_run(...)
      ├─ if COMPLETED: get_result → normalize_result → decode_samples
      └─ store.record("exported")
└─ canonical_batch_sha256 + rows_to_csv + parity + atomic write
```

## 1b. Call diagram — real path (gated, never reached by CLI)

```
POST /api/qpu/{run_id}/submit
├─ qpu_flags.submission_allowed(env)          ← FENCE (3-flag AND)
├─ _validate_real_submission_chain(run_id)    ← FENCE (chain + consumed confirmation)
├─ _gateway_for(backend) → RealIbmGatewayFactory(env)(backend)
│     ├─ runtime_factory_enabled(env)         ← FENCE (factory flag)
│     └─ QiskitRuntimeService()               ← SERVICE CONSTRUCTION
├─ reject if gateway.dry_run_safe
└─ ibm_runner.submit_run → RuntimeGateway.run_sampler → SamplerV2(...).run()   ← QPU CREDIT
```

---

## 2. Purity / offline / credit classification

**CREDIT-CAPABLE (none reachable from CLI):** `RuntimeGateway.run_sampler`
(`ibm_runner.py` — the single actual spend at `SamplerV2.run`), `.find_jobs/.get_job/
.get_result` (network reads), `ibm_runner.submit_run` (credit-capable *only* via
`run_sampler`; with `FakeBatchGateway` spends nothing), `reconcile_run`,
`request_cancel`, `qpu_ibm_factory._DefaultServiceFactory.make_service` /
`RealIbmGatewayFactory.__call__` (build the service), `qpu_api._resolve_real_backend` /
`_gateway_for` / `qpu_submit` / `qpu_reconcile`, `quantum.backends._ibm_sampler`
(reads `QISKIT_IBM_TOKEN` — separate QAOA benchmark path, not in the dashboard
pipeline).

**OFFLINE-IO (local fs / env only):** `assert_offline`, `run_offline_qpu_job`,
`_run_offline_lifecycle`, `BatchJobStore.*`, `_find_run_by_config_hash`,
`_claimed_by_other_config`, all `qpu_runs.*`, all `persist.*`
(`_publish_atomic_exclusive`, `save_run`, …), `prepare_hardware_bundle` /
`transpile_to_isa` (CPU-heavy, no network), `qpu_confirm.*`, `qpu_flags.*`,
`runtime_factory_enabled`.

**PURE (deterministic, no I/O — except `_run_id` uses uuid4):** `persist.config_hash`,
`_sha256`, `_run_id` (nondeterministic nonce), `batch_config_identity`,
`batch_config_hash`, `normalize_result`, `require_offline_exact_gateway`,
`rows_to_csv`, `canonical_batch_sha256`, `FakeBatchGateway.*` (in-memory),
`canonical_qubo_hash`, `_normalise_distribution`, `decode_samples`,
`to_submission_params`, `_expected_phase3_params`, `build_execution_plan`, `dry_run`,
all of `quantum/decode.py` and `quantum/explain.py`.

---

## 3. Security fences

| # | Fence | Refuses |
|---|---|---|
| 1 | `assert_offline` | Runs the whole CLI only if `submission_allowed()` AND `runtime_factory_enabled()` are both false. |
| 2 | `require_offline_exact_gateway` | Any real `RuntimeGateway`, any non-exactly-`FakeBatchGateway` type (subclass or composition wrapper faking `dry_run_safe`). |
| 3 | `qpu_orchestrator._require_safe_gateway` | Marker fence (weaker); guards `simulate_*`; NOT on CLI path. |
| 4 | `qpu_flags.submission_allowed` | AND of `qpu_enabled`(F) ∧ `ibm_submission_enabled`(F) ∧ `not dry_run_only`(dry_run_only default **True**). |
| 5 | `runtime_factory_enabled` | Gates `RealIbmGatewayFactory` and `_resolve_real_backend`; default False. |
| 6 | `ibm_runner._require_confirmation` | Constant-time `config_hash` match before any gateway call in `submit_run`. |
| 7 | `submit_run` max_jobs | `known+1 > max_jobs` before `run_sampler`. |
| 8 | `build_execution_plan` budgets | jobs>max, duplicate tags, shots<1. |
| 9 | HW-validation caps | `MAX_HW_MAX_JOBS=5`, `MAX_HW_MAX_QUANTUM_SECONDS=600`. |
| 10 | Backend allow-list | `run_offline_qpu_job` accepts only local `Fake*` names. |
| 11 | `to_submission_payload` | Not-submittable / tampered / non-re-fingerprinting artefact. |
| 12 | `_validate_real_submission_chain` | Fake/synthetic/incoherent artefact or unverified confirmation. |
| 13 | Terminal-state immutability | A terminal run accepts only probative evidence, never a state/result rewrite. |

---

## 4. `QiskitRuntimeService` construction sites (exhaustive)

Three code sites, **none reachable from the offline CLI**:

1. `qpu_ibm_factory._DefaultServiceFactory.make_service` — via `RealIbmGatewayFactory` ← `qpu_api._gateway_for` ← `/submit`; lazy import.
2. `qpu_api._resolve_real_backend` — gated by `runtime_factory_enabled`; API-only.
3. `quantum.backends._ibm_sampler` — token from `QISKIT_IBM_TOKEN`; QAOA benchmark path; callers `quantum/qaoa.py`, `quantum/qaoa_protocol.py`, never `qpu_batch`/CLI.

Non-code: `static/index.html` has a `save_account(...)` documentation snippet (text
only). No `api_token` anywhere; `QISKIT_IBM_TOKEN` read only on the benchmark path.

---

## 5. Persisted data & remote-dedup fields

**Run store** (`qpu_runs`, append-only event log per `local_run_id`):
- *Before submit:* `manifest.json` (`schema, local_run_id, created_utc, config_hash,
  params`) — the atomic-exclusive commit point; events `prepared`, `awaiting_confirmation`.
- *After submit:* `submitting`, a **`job_id` event** (append-only), `submitted`, then
  `queued/running/completed`, plus provider-observation/metrics/error evidence.

**Batch index** (`BatchJobStore`, one JSON per `job_key`): `job_key, local_run_id,
stage, config_hash` (full batch hash), later `final_state, record`.

**Remote-dedup gap.** The *only* value attached to a remote job at submission is
`job_tags = [local_run_id]` (set on `sampler.options.environment.job_tags`; the
pre-submit search uses the same tag). `qpu_runs` sets **no** config/batch tag
(grep for `tags/job_tags/metadata` is empty). Consequence: the tag dedups only the
**same `local_run_id`** (crash recovery of one run re-adopts its own job) — it does
**not** dedup two independently-prepared runs of the same config, whose tags differ.
This is the gap Phase 5.2.D fills with a deterministic `acrpq:<batch>:<job>` tag.

---

## 6. The remaining race window (drives 5.2.C)

Sequence in `_run_offline_lifecycle` for two concurrent processes A, B with an
identical config (⇒ identical `p.config_hash` and identical `job_key`), empty
`runs_dir` + empty index:

1. Both `store.load(job_key)` → `None`.
2. Both `_find_run_by_config_hash(root, p.config_hash)` **before either publishes a
   manifest** → `None`.
3. Both enter the FRESH branch → both `qpu_runs.prepare(...)`.

`qpu_runs.prepare` publishes its manifest via `_publish_atomic_exclusive` (`os.link`,
atomic + `FileExistsError` on collision) — but that protects the *specific path*
`runs_dir/qpu/<local_run_id>/manifest.json`, **not "a run for this config."** Because
`local_run_id = <ts>-<config[:8]>-<uuid4[:8]>`, the uuid4 nonce makes A and B target
**different** run directories, so both `os.link`s succeed. Both proceed to `submit_run`
→ each `run_sampler` once.

**Consequences:** offline → two synthetic runs (harmless). Real provider → genuine
double submission (per-run tags differ, so the pre-submit tag search can't see the
sibling). The batch index is last-writer-wins on the shared `job_key`, so the sibling
becomes an orphan run with the same `config_hash`. Detection is **deferred**: a later
invocation's `_find_run_by_config_hash` finds two matches and raises
`OfflineSafetyError` (fail-closed) — caught after the fact, not prevented at the race.
`_claimed_by_other_config` does not help (it only blocks adopting a *different*
config's run; same-config concurrency is the uncovered case).

**5.2.C requirement:** an atomic exclusive *single-winner claim keyed by the full
config hash* so exactly one run is prepared/submitted per config, all losers adopt
that one `local_run_id`, and a winner crash recovers safely.
