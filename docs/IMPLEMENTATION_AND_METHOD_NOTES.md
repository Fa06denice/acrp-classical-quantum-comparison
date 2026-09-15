# Implementation & method notes

Factual technical notes to seed the manuscript. No marketing. This documents what is
implemented and verified, and — explicitly — what is not. It is not the manuscript.

## 1. Problem

Aircraft conflict resolution: given a set of aircraft with an initial trajectory, choose a
maneuver per aircraft so that pairwise conflicts are removed at minimum operational cost.
The original model (`Code/`, read-only) is a continuous MINLP over heading/speed changes.

## 2. Continuous → discrete → binary

- **Discretisation** (`discretize.py`, `model.py`): each aircraft's maneuver space is a
  finite grid of `(q, θ)` options (`q` = speed regime, `θ` = heading change), including a
  no-op. A grid of `K` options per aircraft on `n` aircraft gives `K^n` assignments.
- **Binary encoding** (`quantum/qubo.py`): one binary `x_{i,o}` per (aircraft `i`, option
  `o`). One-hot per aircraft (exactly one option). `H(x) = Σ cost·x + λ_pen·Σ conflict·x·x +
  λ_oh·Σ(Σ_o x_{i,o} − 1)²`. Penalty weights are derived from the cost scale so the one-hot
  and conflict constraints dominate.

## 3. Two versioned objectives

- `quadratic_control_cost_v1`: Σ_i [w·θ_i² + (1−w)(1−q_i)²] — the continuous control cost on
  the grid.
- `maneuver_count_v1`: number of deviated aircraft (an option ≠ no-op). QUBO linear term
  `Σ_i Σ_{o≠noop} x_{i,o}`.

They are kept strictly separate everywhere (`objectives.py`); the energy identity
`objective_cost + penalties == qubo.energy` holds for **both** (verified over all 512
CP_3/K3 states). Objective identity is hashed into the QUBO hash; a spoofed objective is
refused (`assert_objective_consistent`).

## 4. Methods compared (all on the identical QUBO / identical controls)

- **Exhaustive** (`classical/reference.py`, read-only; objective-aware enumerator
  `classical/discrete_enum.py`) — the certified discrete optimum, capped by problem size.
- **Gurobi discrete MILP** (`classical/discrete_milp.py`, via amplpy; `Code/` untouched) —
  certified with published gap/bound.
- **QUBO global minimum** — full binary enumeration for small sizes, else a recorded
  one-hot reduction gated fail-closed.
- **Internal SA**, **Ocean SA** (D-Wave `dwave.samplers`, CPU) — heuristics, multi-seed.
- **QAOA on Aer** (`quantum/qaoa.py`, `quantum/qaoa_protocol.py`) — statevector/shots
  simulator only; a versioned, reproducible protocol (angles, optimizer, shots, seed, ISA).

Three-way equivalence (exhaustive == Gurobi-MILP == QUBO-global-min, per objective,
objective-aware tolerance) is proven on 20 cases and fails closed on any divergence.

## 5. QAOA and the IBM pipeline (offline)

- QAOA circuits are built and transpiled locally; Aer is the only executor in the
  benchmark. The IBM submission pipeline is fully implemented but **offline**: a fake
  gateway fenced by exact type, a resumable atomic run/index store, a single-winner
  inter-process claim, a deterministic remote-dedup tag decision, a budget/reservation/
  confirmation layer, and a robust result adapter — all tested with fakes/fixtures.
- Result decoding publishes **distinct** quantities: best-raw (energy min), modal, best
  one-hot-valid (raw), best geometrically-feasible, one-hot mass, feasibility rate, QUBO
  energy, primary cost by objective, one-hot/conflict penalties, residual conflicts,
  decoded `(q, θ)`, and Δ vs the discrete reference with an explicit comparability status.

## 6. Limits (honest)

- **Qubit count**: `n·K` qubits; Aer statevector is memory-bounded (a 16-qubit compute cap
  in the benchmark, tighter than the 29-qubit memory ceiling); real-hardware bundles cap at
  20 qubits.
- **Simulator ≠ hardware**: Aer results are noiseless (or synthetic in the offline batch);
  they are labelled `synthetic / plumbing-validation / not quantum evidence` and are **not**
  quantum results.
- **No quantum advantage** is claimed or measured. Gurobi/QAOA wall-clocks are not compared
  naively (different machines/regimes); timings are published but excluded from the
  reproducible scientific hash.

## 7. Difficulties encountered & scientific bugs found (with fixes)

- **Objective conflation risk** (the "jury trap"): QUBO legs would report the quadratic
  value against a maneuver-count baseline. Fixed by scoring every leg on the job's own
  objective axis and comparing only to the same-objective reference; regression-tested.
- **ULP byte-identity** of the quadratic term (`w·θ²` vs `grid.cost`): the QUBO now delegates
  to `grid.cost` for byte-identical coefficients (non-dyadic `w` tested).
- **Fail-open in the equivalence proof**: `equivalent=True` was returned when the exhaustive
  leg didn't reach optimality. Fixed fail-closed (reference must complete).
- **Config-reuse / double-submit hardening** (offline IBM): the batch store could reuse a
  result across an incompatible config; the run-store hash omitted angles/shots/opt-level;
  two concurrent processes could prepare two runs. Closed by a full protocol identity, a
  persisted batch-identity companion, and an atomic single-winner claim (see the QPU audit
  and open-issues docs). Each fix has a regression test that fails if the guard is removed.
- **PII leak** in an early Gurobi `$version` dump (licensee email/path) — sanitised to the
  version string only; exports are PII/path/secret free.

## 8. Results obtained

- Two objectives implemented and separated; three-way discrete↔QUBO equivalence proven
  (20 cases); multi-family benchmark with dual honest baselines; versioned reproducible
  QAOA-Aer protocols; a fully offline, safety-fenced IBM batch pipeline reproducible
  byte-for-byte across independent worktrees.

## 9. Results still missing

- Any **real hardware** result (requires explicit authorisation + a recorded-result
  adapter validation).
- Wider instance sweeps beyond the current benchmark sets (optional; reuse artefacts first).

## 10. Threats to validity

- Few seeds for the stochastic heuristics; instances are not IID; the exhaustive/QUBO
  legs are exact only within the enumerable size caps; the offline "hardware" path is
  synthetic; endianness/normalisation correctness for a real Runtime result is validated
  only against fixtures until a recorded result is available.
