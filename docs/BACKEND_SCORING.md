# Explainable IBM-backend scoring (Phase 4)

`src/acrpq/dashboard/backend_scoring.py` ranks candidate IBM backends for a QAOA
workload. It is a **recommendation engine only**: it never submits a job, never
holds a token, never touches the network, and imports no Qiskit at load time. The
selected backend must still be shown and explicitly confirmed later (Phase 7).

## Pipeline

1. **Hard applicability filters** — a rejected backend can never be selected,
   whatever its partial score. Each rejection carries a stable `code`
   (`offline`, `maintenance`, `insufficient_qubits`, `no_dynamic_circuits`,
   `no_fractional_gates`, `region_not_allowed`, `not_in_allowlist`, `blocked`,
   `two_qubit_error_too_high`, `readout_error_too_high`,
   `missing_required_metrics`) and a human message.
2. **Bounded multi-criteria scoring**, split into a *scientific* (physical
   quality) and an *operational* (availability) score.
3. **Structured explanation** per backend (top favourable factors, top
   penalties, missing metrics, metric age, uncertainty, weights, heuristic
   warning).
4. **Deterministic ranking** with a stable name tie-break, a canonical JSON
   export and a SHA-256 decision hash.

## Scoring formula (exact)

Each dimension `d` yields a subscore `s_d ∈ [0,1]`; weights `w_d ≥ 0` sum to 1.

```
scientific_score  = Σ_{d∈SCI} w_d · s_d       SCI = {two_qubit_quality, readout_quality,
                                                     topological_fit, qubit_margin}
operational_score = Σ_{d∈OP}  w_d · s_d        OP  = {queue_pressure, throughput,
                                                     data_freshness, metric_completeness,
                                                     region_preference}
total_score       = clamp01(scientific_score + operational_score)
```

Subscores (`clamp01(x)` clips to `[0,1]`; reference scales are tunable constants,
**not** IBM ground truth):

| dimension | formula | reference |
|---|---|---|
| two_qubit_quality | `1 − err_2q / REF_2Q` | `REF_2Q = 0.02` |
| readout_quality | `1 − err_readout / REF_RO` | `REF_RO = 0.05` |
| topological_fit | `avg_coupling_degree / max(1, density·(n−1))` | coarse, pre-transpilation |
| qubit_margin | `(programmable − logical) / max(1, logical)` | — |
| queue_pressure | `1 − pending_jobs / REF_QUEUE` | `REF_QUEUE = 200` |
| throughput | `clops / REF_CLOPS` | `REF_CLOPS = 5000` |
| data_freshness | `1 − age / REF_FRESH` (age = `decision_ts − data_ts`) | `REF_FRESH = 86400 s` |
| metric_completeness | present optional metrics / 5 | — |
| region_preference | `1` if preferred (or no preference) else `0.5` | — |

**A missing metric scores 0 and is flagged** — it never scores as if the metric
were good, so it can never beat a known-excellent metric "for free". `qubit_margin`,
`data_freshness`, `metric_completeness` and `region_preference` are always
computable (from mandatory fields + the injected clock).

Default weights: `two_qubit_quality 0.30`, `readout_quality 0.15`,
`topological_fit 0.15`, `qubit_margin 0.05` (scientific = 0.65);
`queue_pressure 0.15`, `throughput 0.05`, `data_freshness 0.05`,
`metric_completeness 0.05`, `region_preference 0.05` (operational = 0.35).
`ScoringWeights` rejects negative weights and any set that does not sum to 1.

## Uncertainty

```
uncertainty = clamp01(0.6 · missing_fraction + 0.4 · (1 − data_freshness))
```
where `missing_fraction` is over the optional physical/operational metrics. If the
top-ranked applicable backend's `uncertainty` exceeds `uncertainty_block`
(default `0.6`), **selection is withheld** (`selected = None`) rather than made
silently — the ranking is still returned and fully explained.

## Explanations (contribution-based)

`top_favourable` and `top_penalties` rank dimensions by their **real weighted
contribution** `weight × subscore` (not the raw subscore), so a subscore of `1.0`
on a `0.05`-weight dimension can never leapfrog a `0.30` contribution, and a
**zero-weight dimension never appears as a favourable factor**. Each entry
exposes `dimension`, `subscore`, `weight`, and `contribution`/`penalty`. The
explanation also carries `contribution_sum`, which equals `total_score` to the
documented rounding.

## Determinism, hashing & provenance

Ranking sorts by `(−total_score, backend_name)` — a stable name tie-break — and
the embedded snapshots are sorted by name, so the canonical JSON and the
`decision_sha256` are **byte-for-byte independent of the caller's input order**.
A backend's **identity within a decision is its name** (case-sensitive, stripped
of surrounding whitespace); `rank_backends` **refuses two snapshots with the same
name** rather than silently merging them.

The decision hash covers the **entire record** — schema, workload, all snapshots
(with timestamps), weights, uncertainty block, ranking, rejections, explanations
**and any supplied provenance**; only `decision_sha256` itself is excluded from
the hashed content. `verify_decision_hash(record)` recomputes and compares, and
**refuses** any record whose content is not JSON-native / finite. Serialisation
is strict: `set`/`bytes`/`datetime`/arbitrary objects/non-string keys/`NaN`/`Inf`
are rejected — there is no `default=str` coercion anywhere. `decision_ts` is
injected (no hidden clock); a **future** data timestamp is not silently treated
as perfectly fresh (freshness `0` + a `future_data_timestamp` warning). There is
no network access. Provenance is deep-copied defensively, so mutating the caller's
dict after the call cannot change the record; snapshot `provenance` is likewise
copied at construction. Git provenance (commit / dirty / source hash) is attached
only when the caller passes it, and is never fabricated as clean on a dirty tree.

## Topology & transpilation

`topological_fit` is a **coarse pre-transpilation estimate** from the device's
average coupling degree versus the workload's interaction density — it does not
claim to know the true routing cost. `refine_with_transpilation()` is an optional
hook that records real post-transpilation metrics (ISA depth, 2-qubit gate count,
SWAPs, layout) **alongside** the estimate; it never silently rewrites the score.

## Scientific limits (read before citing)

- The score is a **heuristic ranking of hardware suitability**, explicitly **not
  a probability of quantum-computational success**.
- Reference scales (`REF_*`) define "good vs bad" for a study; they are tunable
  and are **not** durable IBM truth. Fixture metrics are marked `synthetic`.
- Topology fit without transpilation is approximate; a high qubit count does not
  make a dense QUBO runnable.
- The engine produces a **recommendation**, never an authorisation to submit:
  no state ever moves to `submitting`, no `run_sampler`, no token, no side
  effects. The choice is confirmed explicitly in Phase 7.

## Known modeling limitations

- `WorkloadRequirements.max_quantum_seconds` is carried and validated but **not
  yet enforced as a hard filter**: no `BackendSnapshot` field exposes a
  per-shot/per-circuit runtime estimate, and inventing one would fabricate
  precision the provider did not give. Budget/duration is instead enforced at
  the real spend gate in Phase 3 submission (`expected_config_hash`, `max_jobs`),
  which fails closed. A duration hard-filter here is deferred until a genuine
  runtime estimate (e.g. a post-transpilation timing model) is available.
- Region/allow-list/block-list and error caps are enforced; feature flags
  (`dynamic`, `fractional`) are trusted as reported by the snapshot.
