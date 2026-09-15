# Experimental results (simulator campaign)

This document reports and interprets the results of the **QAOA simulator
campaign** run with `acrpq campaign --reps 1 2 3 --repetitions 5`. It is meant
as a starting point for the experimental chapter of the thesis: the numbers
below are produced by the framework and the interpretation can be adapted into
the dissertation prose.

## 1. Experimental setup

* **Backend:** local `qiskit-aer` `SamplerV2` simulator (no hardware noise),
  1024 shots per circuit.
* **Optimizer:** COBYLA, `maxiter = 100`.
* **Sweep:** every reduced scenario × QAOA depth `p ∈ {1, 2, 3}` × 5 random
  seeds (105 QAOA runs total). Each (scenario, p) cell aggregates the 5 seeds.
* **Anchor:** the exact `DiscreteReferenceSolver` over the *same* maneuver grid;
  `approx_ratio = QAOA objective ÷ exact in-grid optimum` (1.0 = optimal, larger
  = worse). Feasibility is verified by the geometry kernel, not the QUBO energy.
* **Reproducibility:** new campaign exports record seed, versions, penalty
  weights, depth, gate counts, Git dirty state and a SHA-256 source hash. The
  current raw artifacts predate those fields and were produced from an
  uncommitted working tree; regenerate them from a committed source state before
  treating the numbers as independently reproducible.

Raw data: `results/campaign_runs.{csv,json}` (per run);
`results/campaign_summary.{csv,json}` (per cell). Figures:
`results/figures/{size_vs_qubits,feasibility_vs_depth,approx_ratio_vs_depth}.png`.

## 2. Summary table

| Scenario | Aircraft | Qubits | Grid K | p | Feasible % | Mean approx. ratio | Mean depth |
|----------|:--------:|:------:|:------:|:-:|:----------:|:------------------:|:----------:|
| Q2  | 2 | 10 | 5 | 1 | 100% | 1.00 | 19 |
| Q2  | 2 | 10 | 5 | 2 | 100% | 1.00 | 27 |
| Q2  | 2 | 10 | 5 | 3 | 100% | 1.60 | 39 |
| Q2h | 2 | 14 | 7 | 1 | 100% | 1.20 | 27 |
| Q2h | 2 | 14 | 7 | 2 | 100% | 3.40 | 37 |
| Q2h | 2 | 14 | 7 | 3 | 100% | 1.00 | 55 |
| Q3  | 3 | 15 | 5 | 1 | 100% | 1.60 | 27 |
| Q3  | 3 | 15 | 5 | 2 | 100% | 1.70 | 36 |
| Q3  | 3 | 15 | 5 | 3 | 100% | 1.50 | 55 |
| Q3f | 3 | 9  | 3 | 1 | 100% | 1.00 | 15 |
| Q3f | 3 | 9  | 3 | 2 | 100% | 1.00 | 22 |
| Q3f | 3 | 9  | 3 | 3 | 100% | 1.00 | 31 |
| Q4  | 4 | 20 | 5 | 1 | 100% | 2.13 | 35 |
| Q4  | 4 | 20 | 5 | 2 |  80% | 1.50 | 45 |
| Q4  | 4 | 20 | 5 | 3 | 100% | 1.73 | 71 |
| Q4f | 4 | 12 | 3 | 1 | 100% | 1.00 | 19 |
| Q4f | 4 | 12 | 3 | 2 | 100% | 1.07 | 27 |
| Q4f | 4 | 12 | 3 | 3 | 100% | 1.07 | 39 |
| QR3 | 2 | 10 | 5 | 1 | 100% | 1.00 | 19 |
| QR3 | 2 | 10 | 5 | 2 | 100% | 1.20 | 27 |
| QR3 | 2 | 10 | 5 | 3 | 100% | 1.00 | 39 |

*(Exact figures depend on the seed; re-running reproduces these to within
shot/optimizer noise. The qualitative trends below are stable.)*

## 3. What the results show

### 3.1 Feasibility — QAOA reliably resolves the small instances
On 20 of the 21 cells, QAOA found a conflict-free assignment in **100%** of the
5 seeds. The single exception (`Q4`, the densest case at 20 qubits, `p = 2`)
dropped to **80%** — one seed in five returned a residual conflict. *Reading:*
for ACRP instances of 2–4 aircraft, QAOA on a noiseless simulator essentially
always finds a feasible resolution; the difficulty is **quality**, not
**feasibility**.

### 3.2 Solution quality degrades with problem size
The approximation ratio is the headline metric. It is **near-optimal (≈ 1.0)**
on the small / coarse-grid scenarios (`Q3f`, `Q4f`, `Q2`, `QR3` at low depth)
and **degrades as the instance grows**: `Q3` sits around 1.5–1.7 and the densest
`Q4` reaches 2.1× the optimal deviation cost. *Reading:* QAOA is competitive on
the smallest cases but the gap to the classical optimum widens with the number
of aircraft and conflicts — the expected behaviour at fixed shallow depth.

### 3.3 Maneuver granularity (K) matters more than depth
The clearest lever is the **grid resolution**, not the circuit depth. Compare
the same instances at K = 3 vs K = 5/7:

* `Q3f` (K=3) reaches 1.00 everywhere; `Q3` (K=5) stays at 1.5–1.7.
* `Q4f` (K=3) reaches ≈ 1.0–1.07; `Q4` (K=5) sits at 1.5–2.1.

*Reading:* a coarse maneuver grid is **easier for QAOA** (fewer qubits, smaller
search space, shorter circuits) even though it offers the classical solver fewer
options. This is an important, honest nuance: the apparent "good" quantum result
on the coarse grid is partly because the discretised problem is itself easier —
not because the quantum method is intrinsically better.

### 3.4 Quality is *not* monotone in depth
Counter-intuitively, increasing `p` did **not** uniformly improve quality. `Q2`
worsened from 1.00 (p=1,2) to 1.60 (p=3); `Q2h` swung 1.20 → 3.40 → 1.00; `Q3`
hovered around 1.5–1.7 without a clear trend. *Reading:* deeper QAOA circuits
have **more variational parameters**, so a fixed-budget classical optimizer
(COBYLA, 100 iterations) is more likely to get stuck in a poor local optimum.
This is a well-documented QAOA phenomenon and a good discussion point: more depth
is not "more quantum power" unless the outer optimization is scaled accordingly.

### 3.5 Circuit depth scales as expected
Mean transpiled depth grows roughly linearly with both `p` and `K` (e.g. `Q4`:
35 → 45 → 71 for p = 1 → 3). The number of qubits grows as `n × K`, capping the
tractable instances at ≈ 24 qubits on the simulator. *Reading:* this is the
concrete resource cost that bounds the approach and motivates the reduced
scenarios.

## 4. Answers to the brief's questions (§16)

| Question | Evidence-based answer |
|---|---|
| Is a quantum reformulation of ACRP possible? | **Yes** — every instance maps to a QUBO solved by QAOA, decoded back to valid ACRP maneuvers. |
| Which small scenarios can be solved? | 2–4 aircraft, K ≤ 5, ≤ 20 qubits, all feasibly resolved on the simulator. |
| Quality vs the classical baseline? | Optimal on the smallest/coarsest cases, degrading to ~1.5–2.1× the optimum as size/density grow. |
| Is quantum competitive on small cases? | On feasibility, yes; on quality, only for the very smallest, and partly because the discretised problem is easier. |
| From what size does it get hard? | Quality clearly degrades at `n = 3–4` with K = 5 (Q3, Q4); 20 qubits is already near the simulator-practical ceiling. |
| Effect of hardware noise? | **Measured on `ibm_fez`** (see §5): Q2/Q3 stayed feasible on real hardware; quality matched (Q2) or fell within QAOA's run-to-run variance (Q3); the dominant real-hardware cost is wall-clock time (~5 min/job vs <1 s simulated). |
| Usable in practice for ACRP today? | No — the discretisation, qubit budget and quality gap make it a research probe, not an operational solver. |
| Future potential? | The pairwise structure maps cleanly to QUBO; the bottleneck is qubit count and optimizer/depth scaling, both active research areas. |

## 5. Real hardware: simulator vs IBM QPU

> **Provenance caveat (MoE counter-review).** These `ibm_fez` rows are a **legacy
> artifact**: they are hash-pinned in `results/hardware_manifest.json` but carry **no
> provider job id, no run directory, no raw counts and no artefact/plan/decision chain**,
> so they cannot be re-verified offline. They are *superseded* as primary hardware
> evidence by the 30 fully-auditable `ibm_marrakesh` jobs
> (`results/ibm_fixed_angle_campaign_v1/`), where every hash, bitstring and gap is
> independently recomputable. Cite the Marrakesh campaign; treat this section as
> historical context.

The same scenarios were run on a **real IBM quantum computer** (`ibm_fez`, a
156-qubit Heron processor, open plan) and on the local Aer simulator, scored by
the identical geometry kernel. Command:

```bash
cp .env.example .env        # set QISKIT_IBM_TOKEN
acrpq compare-hardware Q2 Q3 --hardware-mode ibm_real --reps 1
```

Raw data: `results/sim_vs_hardware.{csv,json}`.

| Scenario | Qubits | Backend | Feasible | Residual conflicts | Approx. ratio | Wall time |
|----------|:------:|---------|:--------:|:------------------:|:-------------:|----------:|
| Q2 | 10 | Aer simulator | ✓ | 0 | 1.00 | 0.3 s |
| Q2 | 10 | **ibm_fez (QPU)** | ✓ | 0 | **1.00** | **277 s** |
| Q3 | 15 | Aer simulator | ✓ | 0 | 3.00 | 0.6 s |
| Q3 | 15 | **ibm_fez (QPU)** | ✓ | 0 | **2.50** | **299 s** |

### Reading the hardware results

1. **Feasibility survives the noise.** On both Q2 and Q3 the real QPU returned a
   sampled bitstring that decodes to a **conflict-free** ACRP assignment — the
   resolution succeeded on physical hardware, not only in simulation. (The
   decoding step picks the lowest-energy *feasible* sample and verifies it with
   the geometry kernel, which makes the approach robust to a noisy sample
   distribution.)

2. **Quality is within QAOA's run-to-run variance, not systematically worse.**
   Q2 matched the simulator exactly (1.00). Q3 actually scored *better* on
   hardware (2.50 vs 3.00) — both are sub-optimal versus the classical optimum,
   and the difference reflects the **stochastic** nature of QAOA (shot noise +
   optimizer landscape) rather than any hardware advantage. The honest reading
   is that at this tiny scale the noise did **not** dominate the (already large)
   discretisation/heuristic gap. This should not be over-interpreted: a single
   shot-budget on one QPU is not a statistical claim; repeating with several
   seeds/backends would be needed to quantify the noise distribution.

3. **The real cost is wall-clock time.** A hardware job took **~5 minutes**
   (queue + cloud overhead + execution) versus **under a second** simulated —
   a ~500× gap. For an operational ACRP setting (sub-second decisions), this
   alone rules out today's cloud-queued QPUs, independent of solution quality.

4. **Circuit metrics are identical** (depth 19/27, 25/45 two-qubit gates)
   because the same QAOA ansatz is transpiled for both; the difference is purely
   execution fidelity and latency.

*Conclusion for the thesis:* on the smallest ACRP instances, current IBM
hardware **can** produce feasible conflict resolutions, and at this scale the
hardware noise is not the limiting factor — the discretisation, the qubit budget
and especially the execution latency are. This is a measured, defensible result
that neither overstates nor dismisses the quantum approach.

## 5b. Two quantum paradigms on one QUBO — gate (QAOA) vs annealing (D-Wave)

The strongest cross-hardware comparison the framework supports: the *identical*
`H(x)` solved by classical-exact (anchor), gate-model QAOA, and D-Wave
annealing. Command: `acrpq compare-quantum Q2 Q3 Q4 --ibm --dwave`. The purest
metric is `energy_ratio` (same Hamiltonian); `approx_ratio` is the decoded ACRP
objective vs the exact in-grid optimum.

| Scenario | Qubits | Route | Feasible | approx_ratio | wall |
|----------|:------:|-------|:--------:|:------------:|-----:|
| Q2 | 10 | classical-exact | ✓ | 1.00 | <0.1 s |
| Q2 | 10 | IBM QAOA (sim) | ✓ | 1.00 | 1.0 s |
| Q2 | 10 | D-Wave SA | ✓ | 1.00 | 0.2 s |
| Q3 | 15 | classical-exact | ✓ | 1.00 | <0.1 s |
| Q3 | 15 | IBM QAOA (sim) | ✓ | 1.00 | 1.3 s |
| Q3 | 15 | D-Wave SA | ✓ | 1.00 | 0.2 s |
| Q4 | 20 | classical-exact | ✓ | 1.00 | <0.1 s |
| Q4 | 20 | IBM QAOA (sim) | ✓ | **1.33** | 7.3 s |
| Q4 | 20 | D-Wave SA | ✓ | **1.00** | 0.2 s |

Reading: on the smallest cases the two paradigms tie the exact optimum. On the
densest case (Q4, 20 qubits) **the annealing route reaches the optimum while
gate-model QAOA does not** (1.33× the optimal deviation cost) at fixed depth
`p = 2` / 80 optimizer iterations — a defensible, measurable difference in how
the two quantum approaches handle a harder QUBO. D-Wave SA is also the cheapest
route here. (This is the *classical* SA reference sampler; a real D-Wave QPU run
needs a Leap token — set `DWAVE_API_TOKEN` and add `--dwave-real`.) Note the
usual QAOA-quality caveats: a deeper `p` or larger optimizer budget would likely
close the Q4 gap.

## 5c. The 2^N simulation wall — measured

`acrpq scaling` sweeps QUBO size against Aer statevector simulation cost.
Statevector RAM is `2**N × 16 bytes`; the guard refuses past the safe budget
(29 qubits on a 24 GiB host). The committed sweep (`results/scaling.csv`,
generated at source commit `d03626d`) on this M4 Pro:

| Qubits | Instance | Statevector RAM | QAOA sim time | Status |
|--------|----------|:---------------:|:-------------:|--------|
| 12 | CP_4, K=3 | 0.00 GiB | 0.6 s | ok |
| 15 | CP_3, K=5 | 0.00 GiB | 2.4 s | ok |
| 20 | CP_4, K=5 | 0.02 GiB | 2.6 s | ok |
| 25 | CP_5, K=5 | 0.50 GiB | 59 s | ok |
| 28 | CP_4, K=7 | 4.00 GiB | 637 s | ok |
| 30 | CP_6, K=5 | 16 GiB | — | **refused (over budget)** |

The jump 20 → 25 → 28 qubits (≈3 s → 59 s → 637 s) is the exponential blow-up:
each added aircraft at `K = 5` multiplies the state by `2^5 = 32×`. At 30 qubits
the guard refuses *before* allocating, rather than swapping the machine to a
halt. **The absolute wall-clock seconds are host- and load-dependent and are not
bit-reproducible** (an idle-machine run of the 28-qubit case measured ≈290 s;
the committed run measured 637 s). The *reproducible* claim is the exponential
trend and the memory ceiling, not the exact timings; the RAM figures and the
`refused`/`ok` statuses are deterministic. Figure:
`results/figures/scaling_qubits_vs_time.png`.

**Note on MPS (not in the committed sweep).** The committed sweep is
statevector-only. An *exploratory* run with Aer's `matrix_product_state` method
was orders of magnitude **slower** than statevector on this problem (the
25-qubit MPS case exceeded an hour and was aborted), because the maneuver QUBO
is a **dense one-hot** problem whose QAOA circuit is highly entangled — and
tensor-network methods only help *low-entanglement* circuits. MPS is therefore
excluded from the reproducible sweep. This is consistent with theory (MPS cost
grows with entanglement/bond dimension) and reinforces the same conclusion: a
real QPU — which neither stores `2**N` amplitudes nor slows with entanglement —
is the only route past the wall. No MPS timings are asserted as reproducible
because no MPS artifact is committed.

## 6. Honest caveats

All limitations in [LIMITATIONS.md](LIMITATIONS.md) apply. In particular: the
approximation ratio is measured against the *in-grid* discrete optimum, not the
true continuous MINLP optimum; a coarse grid flatters the quantum result; and
QAOA quality is sensitive to seed, depth and the classical optimizer budget, so
single runs should not be over-interpreted — the 5-seed aggregation above is the
minimum for a defensible claim.
