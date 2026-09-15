# The original-AMPL continuous baseline

This document defines the *scientific reference truth* against which every other
ACRP result in this repository is measured, and the exact protocol by which that
truth is produced, checked and compared. It exists so a thesis defence can state,
without ambiguity, **what was run, what was not, and what each number means.**

## 1. Why this baseline exists

The benchmark previously compared a discretised reference optimum, an exact QUBO
minimisation, simulated annealing and QAOA. None of those is the *historical*
ACRP objective. The authoritative formulation is the thesis director's nonconvex
MINLP in [`Code/NC_ACRP.mod`](../Code/NC_ACRP.mod), run via AMPL and the
**supervised solver Gurobi** (the director's explicit choice) with the bounds from
[`Code/Preprocessing.run`](../Code/Preprocessing.run). Comparing quantum results to
anything else silently changes the reference and inflates or deflates the apparent
gap. This baseline restores the correct anchor.

**Solver policy.** Gurobi is the *supervised principal baseline*. Couenne (and the
other AMPL solvers) are *optional open-source cross-checks*, never an automatic
fallback and never the official anchor. Runs by different AMPL solvers are kept as
**separate, explicitly-named legs** (`original_ampl` = Gurobi; a Couenne run is
recorded as `original_ampl_couenne`) — never merged under one `original_ampl`
value. NB: the historical trigonometric constraints (`cos`/`sin`) are handled by
the AMPL/Gurobi MP driver via a nonlinear reformulation; whether the returned point
is a *proven* optimum of the exact model is decided by the numeric status + the
common-geometry rescoring, never assumed.

## 2. The four objective levels (never conflate them)

| Identifier | Meaning | Solver |
|---|---|---|
| `original_ampl` | continuous solution of the historical model — a *proven optimum* only when the solver's numeric `solve_result_num` is in the `solved` band (`Status.OPTIMAL`) AND rescoring agrees; a `solved?`/feasible result is a feasible **incumbent**, not the optimum | AMPL + Gurobi (supervised) on `Code/NC_ACRP.mod` |
| `pyomo_minlp` | Python rebuild of the same continuous model | `acrpq.classical.pyomo_minlp` (Pyomo + solver) |
| `discrete_reference` | grid optimum — a *proven optimum* only in its exact mode; its heuristic mode returns a feasible incumbent | `acrpq.classical.reference` (always runnable) |
| approximate (`qubo_exact`, `qubo_annealing`, `quantum_qaoa`, D-Wave) | approximate solutions of the grid QUBO | `acrpq.quantum.*` |

**The bare word "AMPL" always means `original_ampl` — a direct run of `Code/`.**
`pyomo_minlp` is a *rebuild*, never a substitute; it is never labelled "AMPL
original", and it is never used to fill in a missing `original_ampl` number.

## 3. The three deltas and three ratios

All computed on the **rescored** (common-geometry) objective of each leg — the
value the solver reports is kept only for auditing, never for comparison.

```
delta_adaptation     = pyomo_minlp        − original_ampl      (cost of the Python rebuild)
delta_discretization = discrete_reference − original_ampl      (cost of gridding the maneuvers)
delta_algorithm      = approximate        − discrete_reference (cost of the approximate method)

adaptation_ratio     = pyomo_minlp        / original_ampl
discretization_ratio = discrete_reference / original_ampl
algorithm_ratio      = approximate        / discrete_reference
```

An **official** delta anchors ONLY on a **proven optimum**. The base leg must be:

* `Status.OPTIMAL` (a merely `feasible` incumbent never qualifies — that includes a
  Couenne `solved?` local optimum and the discrete reference in its heuristic mode);
* feasible under the shared geometry, with a finite objective;
* `integrity_ok` — for a leg that self-reports an objective (`original_ampl`,
  `pyomo_minlp`), provider and rescored objectives must agree.

So `delta_adaptation` / `delta_discretization` anchor on `original_ampl` only when
it is a proven continuous optimum, and `delta_algorithm` anchors on
`discrete_reference` only when it is a proven grid optimum. When the anchor is
merely feasible, the official delta is `null` and a clearly-labelled entry is added
to **`provisional_vs_feasible_incumbent`** (base `*_feasible_incumbent`) — a
non-official, explicitly-provisional figure that is never presented as the gap.
`unavailable`, `unknown` (integrity divergence), `timeout`, `infeasible` and
`error` contribute nothing. Ratios are additionally dropped (delta kept) when the
denominator is zero or non-finite.

A merely FEASIBLE `original_ampl` is therefore **never** called the continuous
optimum, and a heuristic FEASIBLE `discrete_reference` is **never** called an exact
optimum.

## 4. How the original model is executed safely

`acrpq.classical.original_ampl.OriginalAMPLSolver`:

* writes the **verbatim** original instance `.dat` (via
  `InstanceLoader.read_dat_text`, resolved through a fixed archive index — no path
  interpolation, so path traversal / unknown / unsafe names are refused) into a
  per-run temp directory, preserving the historical `PI := 3.141592`;
* selects the solver ONLY in that temporary `.run` (`option solver <name>`), never
  in `Code/`; the solver name is charset-validated and whitelist-checked
  (`gurobi`/`couenne`/`cplex`/`xpress`/`highs`) so no AMPL statement can be injected;
* generates its own temporary `.run` that model/includes the original files by
  **absolute, validated** path and never modifies anything under `Code/`;
* invokes `ampl` with `shell=False`, list arguments, an explicit `cwd`, a minimal
  environment (the amplpy solver dirs prepended so the chosen solver resolves), a
  real timeout, and kills the whole process group (`SIGKILL`) on timeout so no
  solver child is orphaned; stdout/stderr are size-bounded;
* takes the status from the canonical `solve_result_num` band (`0–99` solved,
  `100–199` solved?, `200–299` infeasible, `400–499` limit) — **not** the bare word
  'solved'; a time/iteration limit is `TIMEOUT`, never an optimum;
* parses **machine-readable marker blocks** (not the human `display`) with a strict
  parser that refuses missing/duplicate sections, `NaN`/`Inf`, duplicate /
  out-of-range / missing indices, wrong cardinality and non-numeric objectives.

It distinguishes the failure modes honestly, never a false success and never a
silent solver swap:

* **AMPL binary absent** → `Status.UNAVAILABLE` (`status_kind: unavailable`);
* **AMPL present but the requested solver missing/unloadable** → `Status.ERROR`
  with `status_kind: solver_unavailable` and the AMPL diagnostic line (generic —
  works for any authorised solver; no Couenne-specific message when Gurobi is asked);
* **solver licence problem** → `Status.ERROR` with `status_kind: license_error`,
  matched from the solver's *message* only. A benign "size-limited license"
  community-edition note is deliberately not treated as an error.

No AMPL/solver licence, token or environment secret is read, stored or logged.

## 5. Mandatory rescoring and the integrity flag

Every returned `q`/`theta` is re-scored by the shared `acrpq.geometry` objective.
The record carries both `provider_objective` (the solver's self-report) and
`rescored_objective`, plus their absolute and relative deltas, the reported
`solve_result_num`, the MIP gap (`mip_abs_gap`/`mip_rel_gap` when the solver prints
it), the solver time and an `optimality_proven` flag. If provider and rescored
objectives diverge beyond tolerance, or the solver reports feasible while the shared
scorer finds residual conflicts, the status is **degraded to `unknown`** and the leg
no longer anchors any delta. A solver-feasible-but-conflictual point is never silent
ground truth. (For the historical trig model this matters concretely: the MP driver
approximates `cos`/`sin`, so a Gurobi "optimal" can carry a small constraint
tolerance violation — rescoring is what decides whether it may anchor.)

## 6. Whole-instance only

The historical model runs the verbatim whole-instance `.dat`. Subset scenarios
(`Q2`, `Q2h`, `QR3`) would require synthesising a reduced `.dat` (reconstruction),
which we never do. For those the `original_ampl` leg is recorded as `unavailable`
with a `not_representable` note, and the CLI refuses `--solver original-ampl` on a
subset target (exit 2). Whole-instance scenarios (`Q3`, `Q3f`, `Q4`, `Q4f`) are the
comparable set.

## 7. Provenance recorded with every run

SHA-256 of `NC_ACRP.mod`, `Preprocessing.run` and the instance `.dat`; the neutral
solver-identity fields `solver_requested` / `solver_effective` /
`is_supervised_solver`; the AMPL `solve_result` and numeric `solve_result_num`; the
**AMPL version** (best-effort `ampl --version`, cached, minimal env) and the
**solver version / driver version** (read from the solve banner — zero extra cost),
each `null` when not obtainable and never faked; package/Python versions; the
timeout used; a UTC timestamp; the git commit and dirty flag; and the platform. A
legacy `couenne_version` field is written **only** for a Couenne run (deprecated;
prefer `solver_version`). This is stored as bounded JSON in the result message
(numeric audit fields live in `result.extra`). No licence file content is ever read
or recorded.

## 8. Reproducing the baseline

```bash
# Verify the toolchain + licence usability first (cheap 1-var QP, no model):
acrpq check-ampl-solver gurobi

# One instance, direct historical run with the supervised solver (Gurobi default):
acrpq classical CP_4 --solver original-ampl --ampl-solver gurobi

# Full comparison for one scenario, writing benchmark_* + baseline_* artifacts:
acrpq bench Q4 --original-ampl --pyomo-minlp --ampl-solver gurobi

# Reproducible campaign over all whole-instance scenarios, versioned output that
# never touches the QPU/IBM results:
python scripts/run_original_ampl_baseline.py --label run1
# -> results/original_ampl_baseline/run1/{baseline_all.json,campaign_manifest.json,...}
```

When AMPL is **absent** (as in CI), every command above records `original_ampl` as
`unavailable`, the continuous deltas stay `null`, and the manifest reports zero
proven optima. Nothing is fabricated.

## 9. Current execution status

AMPL Community Edition + Gurobi are now installed locally (AMPL `20260809`,
AMPL/Gurobi `13.0.2`, MP driver `20260624`). The full infrastructure is tested with
a controllable fake `ampl` (no licence needed in CI). A single real run of **CP_4
with Gurobi** at default options is recorded under `results/original_ampl_real/`
(status `unknown`), and a **CP_4-only four-variant diagnostic** is under
`results/original_ampl_diag/`. A full multi-instance campaign has **not** been run.
The opt-in real regression test is gated behind `ACRPQ_RUN_REAL_AMPL=1`.

### CP_4 diagnostic verdict (source of the borderline separation)

The default-options CP_4 run leaves 3 of 6 pairs short of `d=0.05` by `5.7e-7…8e-8`.
A conflict *count* alone is misleading; the six-variant diagnostic reports per-pair
violation *depths*, a **signed** `minimum_safety_margin = min(min_sep − d)`, the
solver's own model-constraint residual, the optimality gap, a PI check and the
driver's echoed options. Terminology is kept sign-correct: a positive shortfall —
even `6.7e-12` — is **not** exact-theoretical feasibility.

| variant | options (added to control) | status | min_safety_margin | max model residual | opt. gap | common-numeric | exact-theoretical |
|---|---|---|---|---|---|---|---|
| A control | (none) | unknown | −5.75e-7 | **7e-6** | 0 | ✗ | ✗ |
| B | `pre:funcnonlinear=1` | unknown | −5.75e-7 | 7e-6 | 0 | ✗ | ✗ |
| C | `pre:funcnonlinear=1 alg:feastol=1e-9` | optimal | −6.75e-12 | < check tol | 2.4e-5 | ✓ | ✗ |
| D | C + `sol:chk:feastol=1e-9 sol:chk:fail` (flag) | optimal | −6.75e-12 | < check tol | 2.4e-5 | ✓ | ✗ |
| E | C + `mip:gap=1e-6` | optimal | −2.88e-9 | < check tol | **9.7e-7** | ✓ | ✗ |
| F | C + `mip:gap=1e-8` | timeout | — (no primal) | — | — | — | — |

* **Trigonometric mode — RULED OUT.** B (`pre:funcnonlinear=1`, true nonlinear, no
  piecewise; confirmed by the driver echo) is byte-identical to A. The earlier
  "trig approximation" hypothesis is **refuted**.
* **Solver feasibility tolerance — the operative factor.** The default-`FeasibilityTol`
  run (A/B) leaves a model-constraint residual of `7e-6` on the bilinear velocity
  constraints and a separation shortfall of `5.7e-7`; tightening `alg:feastol` to
  `1e-9` (C) removes the reported residual (the MP check passes) and shrinks the
  worst shortfall to `6.7e-12`. We state only that these magnitudes are **compatible
  with the effect of the declared feasibility tolerance** — we do NOT equate a
  solver tolerance with a distance tolerance (the residual is in the constraints'
  units; the separation depth is a derived geometric quantity).
* **Variant D — corrected.** The earlier "D failed on an auxiliary constraint" reading
  was **wrong**: `sol:chk:fail` is a *flag*, and the mistaken `sol:chk:fail=1` was
  rejected by the driver, failing the solve. With the correct flag, **D is `optimal`
  and passes the MP driver's own strict `1e-9` solution check** — C's point has **no
  material model-constraint residual**.
* **Optimality gap.** C/D are "solved" only to Gurobi's default relative MIP gap
  (`2.4e-5`). Adding `mip:gap=1e-6` (E) reaches `9.7e-7 ≤ 1e-6` and meets every
  proposed official-anchor criterion; `mip:gap=1e-8` (F) does **not** converge in the
  120 s budget (wall timeout, null primal).
* **PI (3.141592 vs math.pi) — NOT the cause** (established previously; the scorer
  correctly uses the historical literal).
* **No exact-theoretical feasibility anywhere.** Every variant's `minimum_safety_margin`
  is **negative** (A/B −5.75e-7, C/D −6.75e-12, E −2.88e-9): a tolerance-based solver
  optimum lands marginally on the *violating* side of the exact boundary `min_sep=d`.
  None is "feasible with a positive margin"; C/D/E are accepted only under the common
  geometry's **documented numeric tolerance**, which is a distinct, weaker claim.

**Candidate Gurobi protocol config: C** — `pre:funcnonlinear=1 alg:feastol=1e-9`
(common-numeric-accepted, MP-check-validated, provider≈rescored). **E** adds
`mip:gap=1e-6` and is the config that additionally satisfies the proposed relative-gap
threshold — meeting all proposed official-anchor criteria — while still being
`common_numeric`-accepted but **not** exact-theoretical (`min_margin = −2.88e-9`).

## 9a. The official protocol: `ampl_gurobi_supervised_v3` (ADOPTED)

Frozen in one place (`acrpq.classical.ampl_protocol.AMPL_GUROBI_SUPERVISED_V3`),
never a free string scattered across scripts. The reviewer-reconciled config keeps
the gap ≤ 1e-6 and `alg:feastol=1e-9` and adopts an **a-posteriori MP check at 1e-8**
(the measured residual scale is 7–9e-9):

| field | value |
|---|---|
| `protocol_id` | `ampl_gurobi_supervised_v3` |
| `solver` | `gurobi` (supervised; Couenne stays an optional open-source cross-check) |
| `solver_options` | `pre:funcnonlinear=1 alg:feastol=1e-9 mip:gap=1e-6 sol:chk:feastol=1e-8 sol:chk:fail` |
| `timeout_s` | `120` (mini-sweep) |
| historical `PI` | `3.141592` (the `NC_ACRP.mod` literal, never `math.pi`) |
| scorer tolerance | `1e-9` on squared separation |
| objective-concordance | `abs_delta ≤ abs_tol + rel_tol·max(\|prov\|,\|resc\|)`, `abs_tol=rel_tol=1e-6` |
| **`max_rel_optimality_gap`** | `1e-6` (relative optimality gap — a distinct field) |
| **`max_abs_mp_residual`** | `1e-8` (absolute MP residual — a *separate* field; the gap threshold is never used as the residual gate) |
| `anchor_kind` | `numerical_optimum_gap_1e-6_mp_1e-8` |

`sol:chk:fail` is a **flag** (no `=1`). The fatal MP check at 1e-8 *certifies*
residual ≤ 1e-8 but passes silently, so the sweep additionally runs a warn-mode
**measurement** (`mp_measure_options`, `sol:chk:feastol=1e-30`) on the identical
deterministic solution to obtain the actual residual number — no `null` is ever
accepted. Experiments may pass an EXPLICIT override; there is never a silent fallback.

**Qualification (use exactly this; never say "exact optimum", "exact feasibility" or
"positive operational margin"):** a leg is a *numerical optimum certified to a relative
gap ≤ 1e-6, with maximum absolute MP residual ≤ 1e-8, and accepted by the common
geometry under its documented tolerance* (fr: *optimum numérique certifié à un gap
relatif ≤ 10⁻⁶, avec résidu MP absolu maximal ≤ 10⁻⁸, et accepté par la géométrie
commune selon sa tolérance documentée*). The historical model optimises **to the
constraint boundary** with **no positive margin** — a certified leg sits a few
nano-units on the *violating* side of `min_sep = d` (`minimum_safety_margin` slightly
negative), published not hidden. A `d + ε` robust study would be a **different problem**.

### Anchor semantics — FAIL-CLOSED (no `null` accepted)

A leg anchors OFFICIAL deltas (`evaluate_anchor`) only when EVERY criterion is
positively satisfied by a present, finite measurement — an absent value is a
**failure**, never a pass: solver status `optimal` (the fatal `sol:chk:fail` at 1e-8
ran and passed); finite relative gap ≤ `max_rel_optimality_gap` (1e-6); complete +
finite primal; documented objective concordance; `common_numeric_acceptance`; a
**present, finite, parsable MP residual ≤ `max_abs_mp_residual` (1e-8)**; complete
provenance (model/preprocessing/data SHA-256, ampl/solver versions, `solver_requested
= solver_effective = gurobi`, effective options == protocol, `protocol_id`); and a
clean code tree. `feasible_exact_theoretical`, `feasible_common_numeric`,
`minimum_safety_margin`, `optimality_gap` and `max_model_constraint_residual` are kept
separately, plus a published `mp_residual_detail` (constraint type, indicative scale,
relative residual if provided, echoed MP tolerance).

## 9b. v3 mini-sweep — CP_3 / CP_4 / CP_5 (certified anchors)

`results/ampl_gurobi_supervised_v3/` (one at a time, 120 s, hard stop conditions). All
three are certified numerical anchors:

| instance | status | opt. gap | measured MP residual (≤ 1e-8) | min_safety_margin | `original_ampl_gurobi` obj |
|---|---|---|---|---|---|
| CP_3 | optimal · **anchor** | 7.1e-7 | 7e-9 | −3.7e-9 | 3.124e-4 |
| CP_4 | optimal · **anchor** | 9.7e-7 | 9e-9 | −2.9e-9 | 6.250e-4 |
| CP_5 | optimal · **anchor** | 8.2e-7 | 7e-9 | −2.3e-9 | 1.137e-3 |

The residual is on the **quadratic velocity-definition constraints** (`|q·v0| ~ O(10)`,
so relative ~1e-9); the driver echoes `sol:chk:feastol = 1e-8`. Each leg is
`common_numeric`-accepted but **not** exact-theoretical (`minimum_safety_margin < 0`).

### Superseded protocols (kept, never re-used as anchors)

* **v1** ran no MP check — the old `evaluate_anchor` failed *open* (P1), so its
  `is_official_anchor=true` was unfounded. Marked
  `results/ampl_gurobi_supervised_v1/INVALIDATED.md`; JSONs unmodified.
* **v2** enabled a fatal MP check at **1e-9**; the gap-1e-6 incumbent's residual
  (7–9e-9) exceeds it, so the check fails and **no v2 leg anchors** (recorded honestly
  in `results/ampl_gurobi_supervised_v2/`, kept unmodified). This is why v3 adopts the
  1e-8 residual threshold, coherent with the measured 7–9e-9 scale.

**Terminology note:** the run *without* `mip:gap` is **not** a "full solve" — it is a
configuration without explicit gap tightening that stops on the solver's **default**
gap criterion (reaching a relative gap of ~2.4e-5). It is not "more complete"; it
simply optimises to a looser gap where the constraint residual happens to fall below
1e-9.

### Official discretisation costs (K3 demo, K5/K7 sensitivity) — why they are large

Each instance records three grid-aligned, **exhaustively-proven** discrete optima
(`grid_key`, `n_theta`, `n_q`, exact grid values, `search_space = K^n ≤ 16807 ≪ 10⁷`
cap, `mode=exact`, `exhaustive_optimality_proven=true`). With a v3 anchor present, the
named per-grid deltas `discretisation_cost_K{3,5,7}_q1` are **official**:

| instance | anchor obj | ΔK3 (demo) | ΔK5 (sens.) | ΔK7 (sens.) |
|---|---|---|---|---|
| CP_3 | 3.124e-4 | **0.2738** | 0.0682 | 0.0301 |
| CP_4 | 6.250e-4 | **0.4106** | 0.1022 | 0.0451 |
| CP_5 | 1.137e-3 | **0.5472** | 0.1359 | 0.0598 |

**K3 is the main demonstration protocol; K5/K7 are sensitivity analyses.** The high
costs come largely from **angular granularity** — the smallest non-zero heading step —
not from a fundamental modelling loss:

* K3 → minimum non-zero step **±30°** (`{−π/6, 0, +π/6}`);
* K5 → **±15°**; K7 → **±10°**;
* whereas the certified continuous optimum uses only ≈ **0.8–1°** (θ ≈ 0.0176 rad).

A grid that cannot represent a ~1° maneuver must instead pay a ±10–30° turn, so the
discrete optimum is orders larger than the continuum and falls monotonically as K
grows. Any future approximate/QUBO/QAOA comparison MUST use the **same** `grid_key` as
its discrete reference (`assert_same_grid`) — a K5 reference can never score a K3
result. **Adaptation cost** (`pyomo_minlp − original`) is unavailable (no
Pyomo-compatible MINLP solver). **Algorithmic gap** (`approximate − discrete`) is
unavailable — no comparable artifact; quantum not re-run, existing results untouched.

## 9c. Full CP-family campaign (`results/ampl_gurobi_campaign_v3/`)

A resume-safe sequential campaign (`scripts/run_ampl_campaign.py`) over the 18 CP
instances (n = 3…20), one at a time, 120 s each, atomic writes, licence-expiry
pre-check, hard stop only on licence error / corruption (a scientific timeout does
**not** stop the campaign):

* **5 certified anchors** — CP_3…CP_7 (gap `7.1e-7…1.0e-6`, measured MP residual
  `7–9e-9 ≤ 1e-8`, measurement reproduced the primal `Δq=Δθ=0`);
* **13 honest timeouts** — CP_8…CP_20. The nonconvex trig MINLP does not reach gap
  `1e-6` within 120 s, so `status=timeout` with **null** gap/residual and **no
  anchor** (never fabricated); each is listed in `excluded_results` with its failing
  criteria.
* 0 error, 0 unavailable; no early stop; no solver fallback.

The manifest records per-file SHA-256, the counts, the official-anchor and
excluded-with-reasons lists, and gap/time/residual/margin distributions;
`campaign_summary.{json,csv}` (parity-checked) carry the per-K discretisation costs.
`scientific_code_dirty` (False at run time) and `repository_dirty` are published
separately (C) — only the former gates anchoring. Every stored anchor is re-checked
through the fail-closed `accept_official_anchor` guard, which refuses v1/v2. The
anchor boundary (~n ≤ 7 at a 120 s budget) is a property of the solver + time budget,
not of the model; larger instances would need a longer budget to certify.

## 10. Recommended wording for the thesis

> The continuous reference is computed with AMPL and Gurobi under the frozen
> `ampl_gurobi_supervised_v3` protocol. A leg is admitted as the continuous anchor
> only as a **numerical optimum certified to a relative optimality gap ≤ 1e-6, with a
> maximum absolute model-constraint (MP) residual ≤ 1e-8 verified by the solver's
> strict solution check, and accepted by the common geometric scorer under the
> scorer's documented numerical tolerance**. It is **not** an exact mathematical
> optimum, and the original model
> optimises to the separation boundary without a positive safety margin (a `d + ε`
> robust study would be a different problem). Every solution — the discretised
> reference optima (reported per maneuver grid K) and the quantum/QUBO approximations
> — is re-scored by the same shared geometric objective; the discretisation gap (grid
> vs certified continuum) and the algorithmic gap (approximate vs discrete optimum,
> on the identical grid) are reported separately from any modelling cost. Gurobi is
> the supervisor's requested solver; Couenne remains an optional open-source
> cross-check. Where a leg is not a valid anchor, its gap is reported as undefined
> rather than approximated.

### Non-negotiable criterion

No quantum result is presented as scientifically compared to the historical model
until the `original_ampl` leg has been **actually executed** or **explicitly marked
`unavailable`**. `pyomo_minlp` is never a silent substitute for the original AMPL.
