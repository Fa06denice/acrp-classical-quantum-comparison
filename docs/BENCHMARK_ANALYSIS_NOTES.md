# Local benchmark — analysis notes (Phase 6)

Phase 6 asked to strengthen the supervisor's local benchmark (discrete↔QUBO on identical
controls, two objectives, more families). The requirements were already met by the
versioned artifacts from Phases 2–3; per the mission ("reuse the versioned results; do not
re-run a costly campaign") this note **reuses and re-verifies** them rather than
regenerating. No new campaign was run.

## Artefacts re-verified (all pass their content hash)
- `results/equivalence_proof/proof.json` — `proof_sha256 222315a3…` OK.
- `results/discrete_benchmark/benchmark.json` — `benchmark_sha256 0fb68f8e…` OK.
- `results/discrete_benchmark_multifamily_v1/benchmark.json` — `benchmark_sha256 ecf869be…` OK.
- `results/family_matrix/matrix.json` — `matrix_sha256 a6dfe60d…` OK.

## 6.A — same-grid comparison (already enforced)
Every method (Gurobi discrete, exhaustive reference, QUBO global-min, internal SA, Ocean SA,
QAOA-Aer) is scored on the **identical** QUBO / grid / variable order / objective / rescore.
`benchmark.equivalence`/`discrete_benchmark` refuse a comparison across differing grids.

## 6.B — two objectives kept separate (verified)
`benchmark.json` rows never aggregate the two objectives. Counts by (method, objective,
result_class) on the 240-row benchmark:
- `reference`, `gurobi_milp`, `qubo_exact`: 12 + 12 per objective, all
  `certified_optimum` / `qubo_optimum` (three-way equivalence re-confirmed).
- `qubo_annealing`, `dwave_sa`: 36 + 36 per objective, all `incumbent` (multi-seed).
- `qaoa_aer`: 12 + 12 per objective, all `incumbent`.
An **incumbent is never labelled an optimum**; infeasible incumbents are scored `None`.
For `maneuver_count_v1` the multifamily artefact additionally reports deviated/undeviated
aircraft and the tie set (diagnostic only, never folded into the primary objective).

## 6.C — instance families / applicability (already built)
`family_matrix` covers CP/FP/GP/RCP/RCP_FL with a per-cell, per-method
`{applicable, resolution, reason}`; every non-applicable cell carries a machine-readable
reason (memory / qubit count / exhaustive space / timeout / dependency / license /
computational budget). A skip is never counted as a success. The multifamily benchmark uses
dual honest baselines (exhaustive ground-truth vs certified Gurobi-MILP) and only reports a
gap where a feasible result is compared to a valid same-objective baseline.

## 6.D — honest caveats (must accompany any reading)
- **Few seeds** for the stochastic heuristics (SA/Ocean/QAOA) — dispersion is indicative,
  not a distributional claim.
- **Instances are not IID**; per-family aggregates are not pooled across families.
- **No quantum speed-up** is claimed or measured anywhere.
- **Simulator ≠ hardware**: QAOA is Aer (noiseless / synthetic in the offline batch),
  labelled `not quantum evidence`.
- **Timings are not naively comparable**: Gurobi (subprocess, C++), QAOA-Aer (Python +
  threads) and the CPU heuristics run in different regimes; wall/CPU/provider durations are
  reported separately and excluded from the reproducible scientific hash.

## Conclusion
The local benchmark meets the supervisor's four asks and is reproducible from its versioned
artefacts. No costly re-run was warranted; a wider instance sweep remains optional future
work and should start from the applicability matrix, not a blind campaign.
