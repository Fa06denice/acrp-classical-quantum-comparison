# ACRP — Classical Benchmark Library + Quantum Benchmarking Layer

This repository contains benchmarking instances and models for the **Aircraft
Conflict Resolution Problem (ACRP)** in air traffic control, **extended with a
Python benchmark framework that runs both a classical baseline and a
quantum/hybrid (QAOA) approach on the same instances and compares them with
shared metrics.**

The classical formulation is preserved unchanged as the reference; the quantum
layer is an *experimental* benchmarking extension built on IBM Quantum / Qiskit.
It does not claim quantum advantage — it measures, on small reduced instances,
whether the ACRP can be reformulated and solved with quantum optimization and
how the results compare to the classical baseline.

---

## 1. The classical ACRP (preserved baseline)

The ACRP consists of finding minimal-deviation conflict-free trajectories for a
set of aircraft by adjusting their velocities (speed and heading) within
specified control bounds. The 2D ACRP under uniform-motion assumptions is a
**nonconvex mixed-integer nonlinear problem (MINLP)**; its main decision
variables are the speed-control rate and the heading-control deviation per
aircraft.

![Data and variables 2D ACRP](https://github.com/acrp-lib/acrp-lib/blob/master/datavariables.PNG)

![Nonconvex 2D ACRP](https://github.com/acrp-lib/acrp-lib/blob/master/nonconvex.PNG)

The original, authoritative model lives in [Code/](Code/) (AMPL format, solved
with Couenne) and is **never modified**:

* [Code/NC_ACRP.mod](Code/NC_ACRP.mod) — the nonconvex 2D ACRP model
* [Code/NC_ACRP.run](Code/NC_ACRP.run) — main run script
* [Code/Preprocessing.run](Code/Preprocessing.run) — bound/coefficient preprocessing

The benchmark instances live in [Data/](Data/) (CP/FP/GP/RCP families, AMPL
`.dat` format, inside zip archives): 742 distinct instances, plus a 5-flight-level
variant of each random-circle instance (addressable via the `@FL5` alias /
`flight_levels=5`), for 1042 addressable names in total. Details of the formulation are in the paper
<https://doi.org/10.1016/j.ejor.2021.03.059>. Questions about the original
library: david.rey@skema.edu.

---

## 2. The `acrpq` Python framework (new)

`acrpq` is an installable Python package (`src/acrpq/`) that wraps the classical
formulation and adds the quantum benchmarking layer through **one common
interface**: every solver takes the same `Instance` and returns the same
`Result`, scored by a single geometry kernel, so classical and quantum outputs
are directly comparable.

```
Instance  ──►  ClassicalSolver  ──►  Result  ┐
   │                                          ├──►  BenchmarkRecord (CSV/JSON/plots)
   └──► discretize ──► QUBO ──► QAOA ──► Result┘
```

### Pipeline overview

| Stage | Module | Notes |
|---|---|---|
| Problem model | `acrpq.model` | constants + dataclasses mirroring `NC_ACRP.mod` |
| Instance loading | `acrpq.io` | zip-aware AMPL `.dat` parser/loader (742 instances) |
| Geometry kernel | `acrpq.geometry` | the single conflict/objective scorer |
| Discrete reformulation | `acrpq.discretize` | maneuver grid + pairwise conflict table |
| Classical reference | `acrpq.classical.reference` | exact inside its search cap; heuristic fallback above it |
| Classical MINLP | `acrpq.classical.pyomo_minlp` | optional Python translation of `NC_ACRP.mod` |
| Quantum mapping | `acrpq.quantum.qubo` / `.ising` | ACRP → QUBO/Ising (pure Python) |
| QUBO solving (classical) | `acrpq.quantum.qubo_solvers` | exact + simulated annealing on H(x) (pure Python) |
| Quantum solving — gate | `acrpq.quantum.qaoa` / `.backends` | QAOA on Aer (statevector/MPS) / IBM Runtime / real QPU |
| Quantum solving — annealing | `acrpq.quantum.dwave_solver` | D-Wave reference SA / real D-Wave QPU (optional) |
| Memory guard | `acrpq.quantum.memguard` | statevector 2^N RAM predictor + admission control |
| Post-processing | `acrpq.quantum.decode` | bitstring → ACRP decisions, re-scored by the kernel |
| Benchmark | `acrpq.benchmark` | scenarios, metrics, CSV/JSON export, runner |
| Visualisation | `acrpq.viz` | matplotlib plots (optional) |
| Dashboard | `acrpq.dashboard` | FastAPI radar web app (optional) |
| CLI | `acrpq.cli` | `acrpq` command |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design and
[docs/LIMITATIONS.md](docs/LIMITATIONS.md) for an honest account of the quantum
mapping's assumptions and limits.

---

## 3. Installation

The **core** has no third-party dependencies — instance loading, the geometry
kernel, the discrete reformulation, the QUBO build, decoding, the exact
reference solver, metrics and export all run on a bare Python ≥ 3.10. Heavier
features are optional extras.

```bash
# core only (classical reference solver + QUBO build, no heavy deps)
pip install -e .

# quantum layer (QAOA on the local Aer simulator)
pip install -e ".[quantum]"

# D-Wave quantum-annealing route (reference SA token-free; real QPU needs Leap token)
pip install -e ".[dwave]"

# the radar dashboard (FastAPI web app)
pip install -e ".[quantum,dashboard]"

# IBM Quantum cloud / real hardware
pip install -e ".[quantum,ibm]"

# continuous MINLP translation (also needs a solver e.g. Couenne on PATH)
pip install -e ".[classical]"

# everything + plots
pip install -e ".[all]"
```

---

## 4. Running the classical solver

```bash
# reference solver over the discretised model (exact inside its search cap)
acrpq classical CP_4 --solver reference --n-theta 5

# continuous MINLP — Python rebuild via Pyomo (requires Couenne/Bonmin/Ipopt on PATH)
acrpq classical CP_4 --solver couenne

# continuous MINLP — the HISTORICAL model, run directly (AMPL + Gurobi on Code/NC_ACRP.mod)
acrpq classical CP_4 --solver original-ampl --ampl-solver gurobi   # 'unavailable' if AMPL absent
acrpq check-ampl-solver gurobi                 # verify toolchain + licence (no scientific solve)
```

> The bare word **AMPL** always means `original_ampl`, a direct run of
> [`Code/NC_ACRP.mod`](Code/NC_ACRP.mod). The Pyomo path is a *rebuild*
> (`pyomo_minlp`), never a substitute. See
> [docs/ORIGINAL_AMPL_BASELINE.md](docs/ORIGINAL_AMPL_BASELINE.md) for the four
> objective levels, the adaptation/discretisation/algorithm deltas, and the
> reproduction protocol.

```python
from acrpq import InstanceLoader, DiscreteReferenceSolver

inst = InstanceLoader().load("CP_4")
result = DiscreteReferenceSolver(n_theta=5).solve(inst)
print(result.status, result.feasible, result.objective)
```

### Solving the QUBO classically (without QAOA)

The QUBO is the centre of the quantum formulation, and it can be solved by
several methods on the **same** Hamiltonian `H(x)`, all pure-Python:

```bash
acrpq classical CP_4 --solver qubo-exact       # exact minimisation of H(x)
acrpq classical CP_4 --solver qubo-annealing   # classical simulated annealing
acrpq classical CP_4 --solver dwave-sa         # D-Wave reference SA on the same QUBO
acrpq quantum   Q4   --mode aer_sim            # QAOA (gate model) on the same QUBO
acrpq quantum   Q4   --dwave --dwave-mode neal # D-Wave quantum-annealing route (SA)
acrpq quantum   Q4   --dwave --dwave-mode qpu  # real D-Wave QPU (needs DWAVE_API_TOKEN)
```

Two quantum paradigms on the identical QUBO: **QAOA** (gate model, IBM) and
**quantum annealing** (D-Wave). Compare all routes at once, and study how far
the classical simulator scales before the `2^N` memory wall:

```bash
# classical-exact vs IBM-QAOA vs D-Wave on the same QUBO, per scenario
acrpq compare-quantum Q2 Q3 --ibm --dwave            # add --ibm-real / --dwave-real for hardware

# qubit-scaling study (statevector vs MPS, up to the 24 GiB / 29-qubit wall)
acrpq scaling --figures                              # or --dry-run to just build QUBOs

# pick the Aer simulation method (statevector = exact 2^N; MPS reaches more qubits)
acrpq quantum Q4 --sim-method matrix_product_state
```

The objective minimised is exactly the paper's
`Σ_i (1−w)(1−q_i)² + w·θ_i²` (with `w = 0.5` by default); there is no explicit
"minimise the number of maneuvered aircraft" term — few/small maneuvers are an
emergent consequence of the squared-deviation cost (NOOP costs 0).

The original AMPL model can be run directly with AMPL + Gurobi (the supervised
solver) via the guarded `original-ampl` runner; [Code/NC_ACRP.run](Code/NC_ACRP.run)
is untouched.

---

## 5. Running the quantum benchmark

```bash
# list quantum-tractable reduced scenarios
acrpq scenarios

# run QAOA on the local Aer simulator for scenario Q3 (3 aircraft, 15 qubits)
acrpq quantum Q3 --mode aer_sim --reps 2 --shots 1024 --seed 42

# full classical-vs-quantum comparison, exported to results/
acrpq bench Q2 Q3 Q4 --mode aer_sim --reps 2 --out results

# every reduced scenario in one CSV/JSON
acrpq bench --all --mode aer_sim --reps 2 --out results

# full EXPERIMENTAL CAMPAIGN: sweep all scenarios x QAOA depth x 5 seeds,
# aggregate statistics, and render the thesis figures (~5 min on a laptop)
acrpq campaign --reps 1 2 3 --repetitions 5 --out results --figures
```

The campaign writes `results/campaign_summary.csv` (one aggregated row per
scenario × depth) and `results/campaign_runs.csv` (every individual run), plus
PNG figures under `results/figures/`. A worked analysis of a real campaign — with
the answers to the brief's research questions — is in
[docs/RESULTS.md](docs/RESULTS.md).

```python
from acrpq.benchmark.runner import BenchmarkRunner

records = BenchmarkRunner(out_dir="results", seed=42).compare("Q3", reps=2)
for r in records:
    print(r.solver_kind, r.common.feasible, r.common.objective)
```

Running on IBM Quantum requires credentials. Copy `.env.example` to `.env` and
paste your free token from <https://quantum.ibm.com> (the `.env` file is
git-ignored and read automatically; the token is never written to any output):

```bash
cp .env.example .env        # then edit .env and set QISKIT_IBM_TOKEN=...
acrpq quantum Q3 --mode ibm_runtime          # IBM cloud simulator
acrpq quantum Q3 --mode ibm_real             # least-busy real QPU

# simulator-vs-real-hardware comparison table (the noise study)
acrpq compare-hardware Q2 Q3 --hardware-mode ibm_real
```

You may also export `QISKIT_IBM_TOKEN` directly instead of using `.env`.
`compare-hardware` writes `results/sim_vs_hardware.{csv,json}` and prints a
side-by-side table; a worked example run on `ibm_fez` is analysed in
[docs/RESULTS.md](docs/RESULTS.md) §5.

---

## 5b. The radar dashboard

An air-traffic-control style web dashboard visualises instances, conflicts and
resolutions, and the benchmark results. It reuses the same `acrpq` package
through a FastAPI backend (no logic duplicated).

```bash
pip install -e ".[quantum,dashboard]"
acrpq dashboard            # -> http://127.0.0.1:8000
```

* **Radar tab** — a top-down radar of any instance: aircraft blips, velocity
  vectors, separation circles (radius `d`) and conflicting pairs highlighted in
  red. Pick a solver (classical or QAOA), press *Resolve*, and the radar
  redraws the conflict-free trajectories with a live metrics panel (conflicts
  resolved, objective, compute time, and — for QAOA — qubits / circuit depth).
* **Benchmark tab** — interactive charts of the campaign results (feasibility
  and approximation ratio vs QAOA depth) plus the classical-vs-quantum table.
  Run `acrpq campaign ...` first to populate it.

* **Explain tab** — a pre-run RAM/statevector estimate and exact QUBO
  components, an automatic five-route local comparison driven by one backend
  **preflight** (each route's applicability + reason from the solvers' own caps,
  with a Cancel-all control), and, after a solve, a **proof status**
  (exact optimum / feasible heuristic / approximate / infeasible), the energy
  decomposition and the dominant cost contributors.

The radar is a **time-stepped scientific simulator** (timeline, CPA markers,
before/after, multi-solver comparison, bounded async jobs, provenance panel,
exports, a fullscreen presentation mode, and a fully offline build). It also
carries an explicit **DEMO vs SCIENTIFIC** execution profile: SCIENTIFIC runs
sweep multiple seeds, report objective dispersion, and can be saved as versioned,
fully-provenanced records under `runs/scientific/`. Each artifact is published
with an atomic, exclusive `os.link`, and a per-run `<run_id>.meta.json` sidecar
written last is the single commit point (no shared manifest → concurrent processes
lose no entry); collision-safe unique ids, SHA-256 integrity hashes checked by
`verify_run`, and deterministic orphan recovery after a lease window. Full guide,
keyboard shortcuts and the 3-minute demo script:
[docs/DASHBOARD.md](docs/DASHBOARD.md).

> The dashboard animates the model's **linear kinematics** only. It is not an
> operational ATC simulator, not a high-fidelity physics/aircraft-dynamics
> simulation, and claims no quantum advantage; simulator, recorded IBM-hardware
> and D-Wave-reference results are always labelled distinctly.

---

## 6. Benchmark structure & scenarios

The quantum layer is bounded by qubit count (`n × K`, where `K` is the number of
maneuver options per aircraft). The built-in reduced scenarios stay within a
simulator-friendly budget and form a complexity ladder:

| ID | Instance | Aircraft | ~Qubits | Description |
|----|----------|----------|---------|-------------|
| `Q2`  | CP_3 (subset) | 2 | 10 | simplest head-on case |
| `Q2h` | CP_4 (subset) | 2 | 14 | finer maneuver grid |
| `Q3`  | CP_3 | 3 | 15 | three crossing conflicts |
| `Q3f` | CP_3 | 3 | 9  | fast smoke (coarse grid) |
| `Q4`  | CP_4 | 4 | 20 | six dense conflicts |
| `Q4f` | CP_4 | 4 | 12 | coarse grid |
| `QR3` | RCP_10_1 (subset) | 2 | 10 | sparse random instance |

You can also build your own scenario from any instance, optionally sub-sampling
aircraft to fit the qubit budget.

---

## 7. Reading the results

Each benchmark run writes a CSV (flat, for spreadsheets/plots) and a JSON (full
structure) to the output directory. Every record carries the metrics **and** a
reproducibility block: seed, package/Python versions, dependency versions,
backend, shots, QAOA depth, penalty weights, git commit and timestamp.

Key columns: `solver_kind`, `objective`, `feasible`, `n_initial_conflicts`,
`n_resolved_conflicts`, `n_residual_conflicts`, `resolution_rate`,
`solve_time_s`, `n_qubits`, `circuit_depth`, `two_qubit_gates`, `approx_ratio`
(quantum objective ÷ in-grid reference optimum), `lambda_pen`, `lambda_oh`,
`seed`, `qiskit_version`, `git_commit`, `timestamp_utc`.

---

## 8. Tests

```bash
pytest                 # 44 tests; quantum tests auto-skip if Qiskit is absent
```

The core test suite runs with **zero third-party dependencies**;
`tests/test_import_safety.py` enforces that `import acrpq` and every pure module
stay dependency-free.

---

## 9. Academic positioning

> The goal of the implementation is not to replace the classical ACRP solver,
> but to extend it with a quantum benchmarking layer based on IBM Quantum
> libraries. This benchmark evaluates whether small-scale ACRP instances can be
> reformulated and solved using quantum or hybrid quantum-classical optimization
> methods, and compares the obtained results against the classical baseline
> using common performance and quality metrics.

The known limitations of the quantum mapping (discretisation, qubit budget,
QAOA heuristics, penalty method, conflict-model fidelity) are documented
honestly in [docs/LIMITATIONS.md](docs/LIMITATIONS.md).
