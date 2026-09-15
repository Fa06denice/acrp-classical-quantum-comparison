# Architecture

This document describes how the `acrpq` Python framework is structured and how
it integrates the original AMPL ACRP library with the quantum benchmarking
layer. It is written for a reader (e.g. a thesis examiner) who wants to verify
that the classical baseline is faithfully preserved and the quantum layer is a
real, honest experiment rather than a wrapper around the classical solver.

## 1. Relationship to the original repository

The repository started as an **AMPL benchmark library** for the 2D nonconvex
ACRP (Rey & Hijazi, EJOR 2021):

* `Code/NC_ACRP.mod`, `Code/NC_ACRP.run`, `Code/Preprocessing.run` — the model.
* `Data/*.zip` — 742 instances in AMPL `.dat` format (CP/FP/GP/RCP families,
  plus flight-level variants).

These files are **preserved verbatim**. The `acrpq` package sits alongside them
in `src/acrpq/` and treats them as the authoritative reference: it parses the
same `.dat` instances, reproduces the same objective and separation semantics,
and (optionally) rebuilds the same MINLP in Pyomo.

## 2. One common interface

Every solver — classical-discrete, classical-MINLP, quantum-QAOA — consumes the
same `Instance` and returns the same `Result`. Crucially, **all three are scored
by the single geometry kernel** (`acrpq.geometry`), so the objective and
conflict counts on a quantum solution are computed exactly as on a classical
one. This is what makes the comparison fair.

```
                         ┌───────────────────────────┐
  Data/*.zip ──load──►   │        Instance           │
                         └───────────────────────────┘
                              │                │
                  ┌───────────┘                └────────────┐
                  ▼                                          ▼
        ClassicalSolver                              discretize() → DiscreteACRP
        (reference / Pyomo MINLP)                          │
                  │                                  build_qubo() → ManeuverQUBO
                  │                                         │
                  │                                   QAOASolver (Aer / IBM)
                  │                                         │
                  │                                   decode() (one-hot repair)
                  ▼                                          ▼
              Result  ◄────────  acrpq.geometry scorer  ────►  Result
                  └──────────────► BenchmarkRunner ◄──────────┘
                                        │
                              metrics + CSV/JSON + plots
```

## 3. Module map

| Module | Responsibility | Heavy deps? |
|---|---|---|
| `acrpq.model` | constants (mirroring `NC_ACRP.mod`) + dataclasses (`Instance`, `Maneuver`, `ManeuverGrid`, `Solution`, `Result`, `BenchmarkRecord`) | none |
| `acrpq.geometry` | the only conflict-detection + objective implementation | none |
| `acrpq.io.ampl_dat` | AMPL `.dat` parse/serialize (CRLF, tabs, comment styles) | none |
| `acrpq.io.loader` | zip-aware instance discovery/loading, circle-default coords | none |
| `acrpq.discretize` | maneuver grid + precomputed `K×K` pairwise conflict table | none |
| `acrpq.classical.reference` | branch-and-bound inside its search cap; heuristic fallback above it | none |
| `acrpq.classical.pyomo_minlp` | optional Python translation of `NC_ACRP.mod` | Pyomo + MINLP solver |
| `acrpq.quantum.qubo` | `DiscreteACRP` → QUBO (one-hot + conflict penalties) | none |
| `acrpq.quantum.ising` | QUBO → Ising (for metrics / auditing) | none |
| `acrpq.quantum.decode` | bitstring → ACRP decisions, total one-hot repair | none |
| `acrpq.quantum.backends` | Aer / IBM Runtime / real-QPU samplers | Qiskit (+ IBM) |
| `acrpq.quantum.qaoa` | QAOA via `qiskit-optimization` | Qiskit |
| `acrpq.benchmark.*` | scenarios, metrics, export, classical-vs-quantum runner | none (lazy for quantum) |
| `acrpq.viz.plots` | matplotlib figures | matplotlib |
| `acrpq.cli.main` | `acrpq` command | none (lazy per subcommand) |

**Optional-dependency isolation.** `import acrpq` and every module in the
"none" rows import with zero third-party packages — enforced by
`tests/test_import_safety.py`. Heavy imports happen only inside method bodies,
immediately after an `acrpq._optional.require(...)` guard that raises a clear
`OptionalDependencyError` naming the exact `pip install` extra if missing.

## 4. Faithfulness of the geometry kernel

Aircraft `i` at `(x0_i, y0_i)` under control `(q_i, θ_i)` moves with velocity
`V_i = q_i·v0_i`, heading `α_i = θ_i + theta0_i` where
`theta0_i = cap_i − 2π if cap_i ≥ π else cap_i` (`NC_ACRP.mod` line 16, with the
literal `PI = 3.141592` from line 11). For a pair, the relative velocity is
`vr = vel_i − vel_j` (the `.mod`'s `cvrx`/`cvry`) and the relative initial
position is `r0 = (x0_i − x0_j, y0_i − y0_j)`.

The minimum separation over future time `t ≥ 0` of the straight-line
trajectories `r(t) = r0 + t·vr` is:

```
vv = |vr|²
if vv ≈ 0:           min_sep² = |r0|²
elif r0·vr ≥ 0:      min_sep² = |r0|²                         (diverging)
else: t* = −(r0·vr)/vv ;  min_sep² = |r0 + t*·vr|²            (closest approach)
```

and the pair is in conflict iff `min_sep² < d²`.

This closed form was verified (1M+ random samples) to be **equivalent to the
`bin11/bin12 + bin3xx` big-M tangent-cone disjunction** of `NC_ACRP.mod`
everywhere except the degenerate region `|r0| < d` (two aircraft already inside
each other's protection zone at `t = 0`), which never occurs in the library
instances (`radius = 2.0` ≫ `d = 0.05`). The objective is line 67 verbatim:
`Σ_i ( wd·θ_i² + (1−wd)·(1−q_i)² )` with `wd = 0.5` (`NC_ACRP.run`).

*Sanity check:* on `CP_4`, all four aircraft head to the circle centre, so with
no maneuver every pair has `min_sep = 0` (all six conflicts). A +30° turn on a
head-on pair raises `min_sep` to ≈ 1.04 (`min_sep² ≈ 1.072`), comfortably above
`d = 0.05` — the conflict is resolved.

## 5. The discrete reformulation (quantum bridge)

Continuous controls cannot be placed on qubits, so each aircraft is discretised
into `K` maneuver options (a grid over `[hmin,hmax] × [qmin,qmax]`, always
including the NOOP `(q=1, θ=0)`). With one-hot binary `x[i,o]` (bit index
`(i−1)·K + o`), the ACRP becomes a QUBO:

```
H(x) =  Σ_{i,o} cost_o · x[i,o]
      + λ_pen · Σ_{(i,j)} Σ_{oi,oj : conflict} x[i,oi]·x[j,oj]
      + λ_oh  · Σ_i ( Σ_o x[i,o] − 1 )²
```

with `cost_o = wd·θ_o² + (1−wd)·(1−q_o)²` (the per-aircraft objective term).
Because the `.mod`'s separation constraints are genuinely **pairwise** (one
disjunction per pair, depending only on the two aircraft's controls) and the
objective is **separable**, this pairwise penalty captures the constraint
coupling *exactly* — the only information lost is the resolution of the
continuous control box, not any many-body interaction.

**Penalty weights** (recorded for reproducibility):

```
λ_pen = pen_scale · (n · max_o cost_o + 1)
λ_oh  = oh_scale · λ_pen,   oh_scale ≥ max(n, max_conflict_degree + 1)
```

The `λ_oh > λ_pen·(n−1)` condition guarantees that the QUBO global minimum is
always a valid one-hot assignment — it never pays to drop an aircraft to escape
its conflict penalties — so QAOA bitstrings remain interpretable even when no
conflict-free assignment exists in the coarse grid. This is verified by an
exhaustive brute-force test (`tests/test_qubo.py::test_qubo_global_min_is_onehot_and_optimal`).

## 6. Quantum solving and decoding

`QAOASolver` builds a `QuadraticProgram` from the QUBO (variable order = bit
order), wraps `qiskit_optimization.minimum_eigensolvers.QAOA` (self-contained in
qiskit-optimization ≥ 0.6, so no dependency on the deprecated standalone
`qiskit-algorithms`) in a `MinimumEigenOptimizer`, and runs it on a `SamplerV2`
backend with a generated pass manager. Three modes: local Aer simulator, IBM
Runtime cloud simulator, and a real IBM QPU.

`decode()` maps the best bitstring back to a per-aircraft maneuver assignment
(with one-hot repair for valid binary samples; structurally invalid inputs are
rejected) and **re-scores it through the geometry kernel**, producing
a `Result` directly comparable to the classical ones.

## 7. Benchmark and reproducibility

`BenchmarkRunner.compare()` always runs the exact `DiscreteReferenceSolver` (the
ground-truth anchor), runs QAOA, and optionally the Pyomo MINLP. It computes
`approx_ratio = quantum.objective / reference.objective` (when both feasible) —
i.e. how close QAOA gets to the *in-grid* optimum. Every run is exported to CSV
and JSON with a full `ReproInfo` block (seed, versions, dependency versions,
backend, shots, QAOA depth, penalty weights, git commit and dirty state, source
hash, timestamp, platform).
