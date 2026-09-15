# IBM Marrakech fixed-angle campaign — thesis analysis

## Protocol

Thirty real-hardware executions were completed on `ibm_marrakesh`:

\[
5\;\text{instances (CP3--CP7)} \times
2\;\text{objectives} \times 3\;\text{independent jobs}.
\]

Every run used K3 (`n_theta=3`, `n_q=1`), QAOA depth `p=1`, 512 shots,
`gamma=0.4`, `beta=0.3`, transpiler seed 1 and optimization level 0. These are
**fixed-angle sampling runs**: the angles were not optimized on the QPU. Each IBM
job is one experimental repetition; its 512 shots are correlated samples within
that run and are not counted as 512 independent benchmark rows.

The two objectives are never aggregated:

- `quadratic_control_cost_v1`: historical quadratic control cost;
- `maneuver_count_v1`: number of aircraft receiving a non-noop command.

The committed raw exports retain the counts and IBM job IDs. The canonical
summary is `results/ibm_fixed_angle_campaign_v1/campaign_results.{json,csv}`;
the merged local/Aer/hardware dataset is `results/thesis_benchmark_v3/`.

## Results

### Historical quadratic objective

| Instance | Logical qubits | ISA depth | ISA 2Q gates | Mean feasible-shot rate | Runs with ≥1 feasible sample | Runs matching K3 optimum |
|---|---:|---:|---:|---:|---:|---:|
| CP_3 | 9 | 211 | 108 | 9.51% | 3/3 | 3/3 |
| CP_4 | 12 | 259 | 207 | 1.04% | 3/3 | 3/3 |
| CP_5 | 15 | 376 | 348 | 0% | 0/3 | 0/3 |
| CP_6 | 18 | 469 | 479 | 0% | 0/3 | 0/3 |
| CP_7 | 21 | 640 | 700 | 0% | 0/3 | 0/3 |

### Minimum number of maneuvered aircraft

| Instance | Logical qubits | ISA depth | ISA 2Q gates | Mean feasible-shot rate | Runs with ≥1 feasible sample | Runs matching K3 optimum |
|---|---:|---:|---:|---:|---:|---:|
| CP_3 | 9 | 211 | 108 | 1.63% | 3/3 | 3/3 |
| CP_4 | 12 | 259 | 207 | 0.65% | 3/3 | 3/3 |
| CP_5 | 15 | 376 | 348 | 0.33% | 3/3 | 3/3 |
| CP_6 | 18 | 469 | 479 | 0.065% | 1/3 | 1/3 |
| CP_7 | 21 | 640 | 700 | 0% | 0/3 | 0/3 |

Overall, 16/30 runs contained at least one feasible encoded solution; whenever a
feasible solution was found, its primary objective matched the certified K3
reference (16/16). This does **not** mean the QPU solved every instance reliably:
the dominant observation is the collapse of feasible sampling probability as
circuit size/routing cost rises.

## Interpretation

1. The alternative moved-aircraft objective genuinely ran on hardware. For CP3,
   the first hardware result returned objective 2, equal to the certified
   discrete optimum, while the historical leg returned 0.2741556. Distinct QUBO,
   plan and configuration hashes prevent objective relabelling.
2. Logical qubit count alone is a poor hardware-readiness measure. From CP3 to
   CP7, the routed circuit grows from depth 211/108 two-qubit gates to depth
   640/700 two-qubit gates, while feasible sampling falls to zero.
3. The maneuver-count Hamiltonian produced feasible samples on CP5 and once on
   CP6 where the quadratic formulation produced none. With only three jobs per
   cell, this is an observed protocol-specific difference, not proof that the
   alternative objective is universally easier.
4. The median submit-to-observed-completion duration was about 27,650 seconds
   (7.68 hours). This includes queue and polling delay and is not QPU execution
   time. It cannot be compared directly with local CPU wall time.

## Claims that remain forbidden

- no quantum speedup or quantum advantage claim;
- no treatment of shots as independent repetitions;
- no claim that a run with zero feasible samples proves infeasibility;
- no aggregation of the two objective values;
- no attribution of all degradation to gate noise alone: topology, routing,
  calibration, fixed angles and the penalty landscape are confounded;
- no extrapolation beyond this backend, date and protocol.

## Reproduction and verification

```bash
source .venv/bin/activate
python scripts/build_ibm_campaign_dataset.py --verify
python scripts/build_thesis_benchmark_v3.py --verify
```

New submissions are no longer enabled on the running dashboard: it was returned
to forced dry-run mode immediately after all 30 results were recovered.
