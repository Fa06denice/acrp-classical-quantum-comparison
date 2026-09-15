# Simulation limits and real executable sizes (measured)

Empirical characterization of what each solver route can actually execute in this
repository, measured on the target machine (Apple Silicon, 24 GiB RAM, Python 3.14,
qiskit 2.4.1, qiskit-aer 0.17.2, branch `acrpq-scientific-ui`, HEAD `ca6f09c`).

Every measurement below: one process per run (peak RSS via
`resource.getrusage(RUSAGE_SELF).ru_maxrss`), grid `n_theta=3, n_q=1` (K=3, so
qubits = 3·n), seed 1234, single seed. QAOA runs used `reps=1, maxiter=10`
(the shipped benchmark default is `maxiter=100` — scale QAOA wall times ~10x
for benchmark conditions). The Python-import floor of the qiskit stack is
~190 MiB RSS; the non-Qiskit solvers idle at ~22 MiB.

## 1. Measured table

### QAOA on Aer, `method=statevector` (CP family, monolithic circuit)

| qubits | instance | wall time (s) | peak RSS (MiB) | theoretical statevector (16 B · 2^q) |
|-------:|----------|--------------:|---------------:|-------------------------------------:|
|  9 | CP_3  |   1.37 |   189 | 8 KiB |
| 12 | CP_4  |   1.53 |   195 | 64 KiB |
| 15 | CP_5  |   1.49 |   196 | 512 KiB |
| 18 | CP_6  |   1.65 |   200 | 4 MiB |
| 21 | CP_7  |   2.97 |   217 | 32 MiB |
| 24 | CP_8  |  13.04 |   448 | 256 MiB |
| 27 | CP_9  | 109.54 | 2,117 | 2 GiB |
| 30 | CP_10 | **refused** | — | 16 GiB > 13.92 GiB budget (`memguard`) |

Peak RSS tracks the 16 B·2^q prediction once the statevector dominates the
~190 MiB import floor (27q: 2,117 MiB ≈ 2 GiB sv + floor — the documented
~1.05x factor in `memguard.py` holds). Wall time grows ~4x per +2 qubits past
~22q; extrapolating, 29q ≈ 7–8 min at `maxiter=10` and ~8.5 GiB RSS (not run —
above this characterization's 2-minute/8 GB budget), and >1 h at the benchmark's
`maxiter=100`. That compute growth — not memory — is why the benchmark caps QAOA
at 16 qubits (see §2).

### QAOA on Aer, `method=matrix_product_state` (one characterization run)

qiskit-aer 0.17.2 as installed **does** accept `method=matrix_product_state`
(`AerMethod.MATRIX_PRODUCT_STATE`, `src/acrpq/quantum/backends.py:29-35`).

| qubits | instance | wall (s) | peak RSS (MiB) | best energy | correctness vs statevector |
|-------:|----------|---------:|---------------:|------------:|----------------------------|
| 9 | CP_3 | 1.27 | 189 | 0.274155563735107 | identical optimum energy (bitstring differs — degenerate optimum) |

That is ONE 9-qubit run. It shows the MPS path works and reproduced the exact
optimum energy on CP_3; it says nothing about MPS fidelity on deeper/wider QAOA
circuits, where truncation error is entanglement-dependent. No reliability claim
beyond this measurement.

### Exact one-hot search, K^n (`QuboExactSolver`, default path)

| states K^n | instance (vars) | wall (s) | peak RSS (MiB) |
|-----------:|-----------------|---------:|---------------:|
| 27      | CP_3 (9)   | 0.00017 | 22 |
| 729     | CP_6 (18)  | 0.0050  | 22 |
| 6,561   | CP_8 (24)  | 0.067   | 22 |
| 59,049  | CP_10 (30) | 0.86    | 22 |
| 531,441 | CP_12 (36) | 9.46    | 22 |

~18 µs/state (energy re-evaluated per state, pure Python). At the coded cap of
5,000,000 states (≈ n=14 at K=3) an exact run would take ≈ 90 s. Memory is flat —
the limit is time and the explicit cap, never RAM.

### Exhaustive discrete enumeration (`classical/discrete_enum.py`, maneuver_count_v1)

| states | instance | wall (s) | status |
|-------:|----------|---------:|--------|
| 27      | CP_3  | 0.00013 | optimal |
| 729     | CP_6  | 0.00085 | optimal |
| 6,561   | CP_8  | 0.0103  | optimal |
| 59,049  | CP_10 | 0.144   | optimal |
| 531,441 | CP_12 | 1.82    | optimal |

~3.4 µs/state (conflict check short-circuits before objective evaluation). Its
cap of 2,000,000 states (≈ n=13 at K=3) ≈ 7 s of compute — the cap is a
proof-hygiene bound, far below the machine's actual capability (~2 min would
allow ~3.5·10^7 states, n=15–16).

### Brute-force bitstring enumeration, 2^(n·K) (`QuboExactSolver(brute_force_bits=True)`)

| bits | states | instance | wall (s) |
|-----:|-------:|----------|---------:|
|  9 |     512 | CP_3 | 0.0018 |
| 12 |   4,096 | CP_4 | 0.018 |
| 18 | 262,144 | CP_6 | 1.89 |

~7 µs/state, 8x per additional aircraft (2^3). The default cap of 5,000,000
states limits this proof-only path to ≤22 bits (n≤7 at K=3); it exists to prove
the QUBO optimum is one-hot on tiny cases, not as a solver.

### Simulated annealing (defaults: internal 400 sweeps × 8 restarts; Ocean 1000 reads)

| vars (qubits) | instance | internal SA wall (s) | dwave.samplers SA wall (s) | RSS (MiB) |
|--------------:|----------|---------------------:|---------------------------:|----------:|
|   9 | CP_3     | 0.024 | 0.42 | 22 / 60 |
|  15 | CP_5     | 0.068 | 0.41 | 22 / 61 |
|  21 | CP_7     | 0.132 | 0.41 | 22 / 60 |
|  30 | CP_10    | 0.320 | 0.50 | 22 / 61 |
|  60 | CP_20    | 1.82  | —    | 22 |
| 150 | RCP_50_1 | 4.05  | 1.40 | 23 / 63 |

Both scale polynomially (internal SA ≈ O(n²) per sweep-set: n proposals × O(n)
energy; Ocean's C sampler is nearly flat here). **Neither has any size ceiling in
this repo** — at 150 variables (the largest shipped instance, RCP_FL n=50) both
finish in seconds at flat memory. Caveat observed at 150 vars with these
defaults and one seed: internal one-hot SA found energy 2.47 while Ocean's
free-bit SA returned 318.95 — solution *quality* at scale is parameter-dependent
even though runtime is trivial.

### Exact component decomposition (already implemented) — measured on FP_4

FP_4 (n=8, 24 qubits) splits into 3 independent QUBO components: 12+6+6 qubits
(`split_qubo_components`, wall 0.00004 s).

| run | wall (s) | peak RSS (MiB) | best energy (maxiter=10) |
|-----|---------:|---------------:|-------------------------:|
| monolithic 24q `solve()`         | 7.92 | 448.6 | 21.65 |
| `solve_decomposed()` (12q max)   | 1.59 | 194.9 | 0.548 |

5.0x faster, statevector peak 2^12 vs 2^24 (4,096x smaller; RSS drops to the
import floor), and a *better* energy at equal optimizer budget because each small
circuit is easier to optimize. The recombination is exact at the model level —
see §4.

## 2. The caps as coded (file:line)

| cap | value | where | consequence when exceeded |
|-----|-------|-------|---------------------------|
| Exhaustive discrete enum cap | `DEFAULT_MAX_SEARCH = 2_000_000` (K^n) | `src/acrpq/classical/discrete_enum.py:40` (refusal at :95-101) | status `"search_too_large"`, no result |
| Reference solver exact cap | `_DEFAULT_MAX_SEARCH = 10_000_000` (K^n) | `src/acrpq/classical/reference.py:41` | documented greedy/local-search **heuristic fallback** (:219) |
| Exact QUBO one-hot cap | `QuboExactSolver.max_states = 5_000_000` | `src/acrpq/quantum/qubo_solvers.py:59` (K^n check :98-102; 2^(nK) check :73-78) | `ValueError`, method inapplicable |
| QUBO build ceiling (default) | `build_qubo(max_qubits=29)` | `src/acrpq/quantum/qubo.py:142` (raise :177-181) | `QubitBudgetError` — statevector-simulation safety default |
| QUBO build ceiling (benchmark) | `_QUBO_BUILD_CEILING = 4096` | `src/acrpq/benchmark/discrete_benchmark.py:79` | coefficient dict alone is safe past 29q for non-statevector solvers (comment :74-78) |
| Statevector RAM ceiling | `TOTAL_RAM_GIB = 24.0`, `SAFE_FRACTION = 0.58` → 13.92 GiB budget → `max_safe_qubits() == 29` | `src/acrpq/quantum/memguard.py:27-28,76-82` (verified on this host: detected 24 GiB, budget 13.92 GiB, max 29q) | `QubitBudgetError` raised pre-allocation in `qaoa.py:183-189`; MPS is never refused (:102-111) |
| Benchmark QAOA compute cap | `qaoa_max_qubits: int = 16` | `src/acrpq/benchmark/discrete_benchmark.py:220` (guard :677-682, planner filter :836-841) | job skipped and logged — a *time* budget (2^q compute), deliberately tighter than the 29q memory ceiling |
| Dashboard variable cap | `_MAX_QUBO_VARS = 1_024` | `src/acrpq/dashboard/api.py:45` | QUBO views/QAOA reported unavailable in preflight |

`preflight_payload` (`src/acrpq/dashboard/api.py:955`) derives all applicability
verdicts from these same constants, and `results/family_matrix/matrix.json` is
generated from preflight — the published matrix cannot disagree with the solvers.

## 3. Instance classification (K=3 grid; qubits = 3·n)

From the published matrix (546 rows, 532 at K=3) and the caps above. Shipped
families: CP n=3–20, FP n=8–30, GP n=16–60, RCP n=10–40, RCP_FL n=50.

| band (aircraft) | qubits | reference | exact QUBO / enum | QAOA Aer statevector | SA (both) / Gurobi MILP | binding limit |
|---|---|---|---|---|---|---|
| n ≤ 5 (CP_3–5, GP_4–5, FP_4*…) | 9–15 | exact | yes (even 2^(nK) proof path for n≤7) | yes, ≤2 s measured | yes | none |
| n = 6–9 (CP_6–9, FP_5*–…) | 18–27 | exact | yes (3^9 = 19,683) | **memory-OK** (27q = 2 GiB, 110 s at maxiter=10) but **excluded by the benchmark's 16q compute cap** — only manual/dashboard runs | yes | time: 2^q circuit simulation |
| n = 10–14 (CP_10–14, RCP_10_*) | 30–42 | exact (cap 10M → n≤14) | exact QUBO to n≤14 (cap 5M); enum to n≤13 (cap 2M) | **refused**: 30q = 16 GiB > 13.92 GiB budget; only rescued if decomposition gives components ≤29q (CP: never — single component; measured CP_10: one 30q component) | yes | memory: 16 B·2^q |
| n = 15–40 (CP_15–20, FP, GP, RCP_20/30/40) | 45–120 | **heuristic fallback** (3^15 = 14,348,907 > 10M) | no (> all enumeration caps) | no (raw and largest components ≫ 29q) | yes — the only remaining exact route is the Gurobi discrete MILP | space K^n and memory 2^q |
| n = 50 (RCP_FL) | 150 | heuristic | no | no (measured RCP_50_1: components 63+42+42+3 — largest 63q ≫ 29q) | yes (internal SA 4.0 s, Ocean 1.4 s) | same |

*FP_x names index flows, not aircraft: FP_4 has n=8 aircraft.

Matrix ground truth at K=3 (532 rows): reference / SA / Ocean / Gurobi applicable
on **532/532**; exact QUBO on **116** (search space ≤ 5M); QAOA Aer on **52**
(CP 7, FP 2, RCP 43 — every one via raw ≤29q or components ≤29q; GP and RCP_FL: 0).
The K=5/7 sensitivity rows shrink these further (K=7: exact QUBO cap at n≤7,
QAOA raw fit at n≤4).

Transpilation is never the binding limit locally: measured QAOA ansatz depth
grows linearly (15 at 9q → 39 at 27q) and transpile time is <1 s within the
simulable range.

## 4. Optimization opportunities that preserve the science

1. **Independent-component decomposition — implemented, measured, exact.**
   `split_qubo_components` (`src/acrpq/quantum/explain.py:72`) +
   `QAOASolver.solve_decomposed` (`src/acrpq/quantum/qaoa.py:272`).
   Measured on FP_4: 5.0x wall-time, 4,096x smaller statevector, better energy at
   equal maxiter. Equivalence is *proven by shipped tests*:
   `tests/test_quantum_explain.py:105`
   (`test_split_components_preserve_energy_identity`: `original.energy ==
   constant + Σ local.energy`) and `tests/test_quantum_explain.py:119`
   (`test_qaoa_decomposition_recombines_original_bit_order`). Risk: none at the
   model level (components are disconnected in the actual interaction graph);
   the only caveat is honest already — no joint counts distribution is
   fabricated (`qaoa.py:317-319`). Limitation, measured: CP instances are a
   single component (CP_4, CP_10), and RCP_50_1's largest component is 63q —
   decomposition rescues mid-density instances (FP_4-like), not the dense or
   the huge ones.

2. **Fixed variables via preprocessing (free aircraft → NOOP).** An aircraft
   whose conflict rows are all zero forms an isolated K-qubit component; its
   optimum is `argmin` of its option costs — no circuit needed. Today
   `solve_decomposed` still runs a K-qubit QAOA for it (RCP_50_1 has one such
   3q component). Expected gain: removes only trivial circuits — seconds, not
   feasibility; it never shrinks the *largest* component, which is what sets the
   memory wall. Energy equivalence exact (special case of the proven component
   identity). Risk: low, but it adds a second decode path; not currently
   implemented.

3. **Candidate/option pruning (dominated options).** Removing option `o` when
   another option has ≤ cost and a subset of conflict activations preserves the
   optimum analytically, and *could* shrink a >29q component below the ceiling.
   Risk: **medium** — it changes `n_qubits`, term counts and therefore the
   protocol/config hash and the published matrix identity; it needs its own
   equivalence proof per instance (no shipped test covers it). Do not do this
   silently; if done, report both encodings.

4. **Seed batching / parallel independent experiments.** QAOA is deliberately
   serialized *within a process* by `_QAOA_RUN_LOCK`
   (`src/acrpq/quantum/qaoa.py:33-38`) because qiskit-optimization seeds a
   process-global RNG — in-process threading would break determinism, so don't.
   Process-level parallelism of independent (instance, seed) jobs is safe and
   preserves per-job determinism exactly. Expected gain: ~linear in cores for
   the ≤16q benchmark sweep (each job ≈ 200 MiB). Constraint: `memguard`'s
   budget is per-process — two concurrent 29q jobs would want 2×8 GiB of a
   13.92 GiB budget; an aggregate admission gate does not exist, so cap
   concurrent statevector jobs by Σ 16 B·2^q ≤ budget.

5. **Symmetry reduction.** The QUBO has no exploitable global symmetry to
   quotient: aircraft differ by geometry (costs and conflict tables are
   instance-specific), and the one-hot blocks are already the minimal encoding
   given the fixed grid. Any grid coarsening (smaller K, fewer θ) changes the
   discrete model itself — that is changing the science, not an optimization.

6. **MPS instead of statevector.** Supported and correct on the one case
   measured (§1). Potential gain: escapes the 2^q wall on low-entanglement
   circuits; `memguard` already never refuses it (`memguard.py:102-111`). Risk:
   accuracy is entanglement-dependent and unvalidated here beyond 9q — using it
   for published numbers would require a per-size validation study first.

### What cannot be improved without changing the problem

- **The 2^q statevector wall.** 16 bytes × 2^q is a property of exact
  simulation, not of this code. 29q ≈ 8 GiB, 30q = 16 GiB, 40q = 16 TiB,
  90q (one IBM Heron) ≈ 2·10^13 TiB. No refactoring touches this.
- **K^n search-space growth** for every exact enumeration route (only the
  Gurobi MILP escapes it, and it is already in the benchmark).
- **CP-family density**: single QUBO component (measured), so decomposition can
  never help CP; CP_10+ is permanently outside local statevector QAOA.
- **In-process QAOA serialization** — required for reproducibility as designed.

## 5. "Two machines = twice the qubits" — refuted

**No.** Statevector memory is 16 B · 2^q: each *single* additional qubit doubles
the memory. Two 24 GiB machines hold 48 GiB, which buys **at most one extra
qubit** (30q = 16 GiB fits in 48 GiB, 31q = 32 GiB barely, 32q = 64 GiB already
does not). Doubling the qubit count from 29 to 58 would require
16 B · 2^58 = 4 EiB ≈ **190 million** such machines. Moreover, nothing in this
repo configures distributed simulation, and the stock qiskit-aer 0.17.2 pip
wheel used here does not run MPI-distributed statevectors — so in practice two
machines give **+0 qubits**. What a second machine *does* double is throughput
of independent experiments (seeds, instances, grid settings) — see §4.4.

---
*Measurement provenance: single-run, one seed (1234), `maxiter=10`, K=3 grid,
fresh process per point, script kept outside the repo (session scratchpad,
`measure_one.py`). Wall times on an otherwise idle machine; treat ±20% as noise.*
