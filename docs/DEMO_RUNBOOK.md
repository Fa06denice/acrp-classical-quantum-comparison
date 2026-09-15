# Demo runbook (supervisor demo, fully offline)

Branch `acrpq-scientific-ui`, HEAD `d67ec83`. This demo runs entirely on the
local machine. **No IBM account, token or network access is needed, and no real
QPU submission is possible in the safe configuration below.**

## 1. Prerequisites

- The repo venv: `<repo>/.venv` (Python 3.14,
  qiskit 2.4.1, qiskit-aer 0.17.2, fastapi/uvicorn installed).
- No IBM account needed. No environment variables needed.
- A browser on the same machine (the server binds loopback only).

## 2. Start command

From the repo root:

```bash
PYTHONPATH=src .venv/bin/python -c "from acrpq.dashboard.api import create_app; import uvicorn; uvicorn.run(create_app(), host='127.0.0.1', port=8000)"
```

Then open http://127.0.0.1:8000/.

## 3. Safe configuration = the default

Do **not** set any `ACRPQ_*` environment variable. All QPU flags fail closed
(`src/acrpq/dashboard/qpu_flags.py`): workflow off, forced dry-run, submission
disabled, IBM runtime factory disabled (503 at the gateway seam). The UI QPU
badge shows **"QPU DISABLED · OFFLINE — NO SUBMISSION"** (`app.js:2449`) and the
QPU tab states "QPU workflow is disabled on this server
(ACRPQ_QPU_ENABLED=false)". The header banner reads "Local demo — no real QPU
submission".

Instances: primary **CP_3 at K=3** (9 qubits), backup **CP_4 at K=3**
(12 qubits). K=3 is the UI default.

## 4. Five-minute script

1. **Select the instance** CP_3 in the instance dropdown (K slider stays at 3).
   The dashboard **auto-runs the comparison** — every applicable local method on
   the identical grid/QUBO/objective; nothing to click.
2. **Read the business objective**: the headline "Avions déviés / réaffectations
   vs planning initial : X / n" plus the K info. Expected for CP_3 at K=3,
   objective maneuver-count: **2 deviated aircraft of 3** — a certified optimum
   (exhaustive enumeration of all 27 assignments; 6 equivalent optimal
   solutions).
3. **Comparison table**: point out the `= réf ✓` markers (methods matching the
   certified reference), the "Δ vs ref" column, and the "Δ time vs ref (s)"
   column. Say explicitly that wall times are host-dependent and never evidence
   of advantage.
4. **Explanation panel**: the per-aircraft table (initial heading, new heading,
   Δθ, q, option, cost) shows exactly which aircraft maneuver and which keeps
   its initial plan (NOOP).
5. **Export**: use the comparison export ("Export comparison — instantané figé
   de la configuration affichée"). The export is a frozen snapshot of exactly
   what is displayed (tested: `test_export_matches_displayed_snapshot_exactly`).

## 5. Fifteen-minute script

Steps 1-5 above, then:

6. **Not-applicable is business language, not an error**: select **CP_10**
   (30 aircraft × 3 = 30 qubits). QAOA statevector is refused (16 GiB > the
   13.92 GiB memory budget) and exact enumeration caps apply; the UI shows a
   French business reason per method ("non applicable — …") plus a suggestion
   of what remains feasible, and still runs the applicable methods (reference,
   SA, Ocean, Gurobi). Nothing is faked (tested:
   `test_not_applicable_is_business_language_never_an_error`,
   `test_too_large_instance_partial_applicability_still_runs`).
7. **DÉTAILS SCIENTIFIQUES view**: click the "DÉTAILS SCIENTIFIQUES ▾" toggle.
   It reveals the Explain tab (proof status: exact optimum / feasible heuristic /
   approximate sampled), QUBO views, and the IBM QPU tab. Note the honesty rule:
   "optimality gap" is only used against a proven optimum.
8. **QPU readiness panel, offline**: open the IBM QPU tab. Show the disabled
   state and the badge "QPU DISABLED · OFFLINE — NO SUBMISSION". Explain the
   fail-closed flags: four environment variables must ALL be set for a real
   submission, plus a single-use human confirmation with an explicit consent
   phrase — none of which is the case tonight.
9. **The real marrakesh artifact story** (evidence that the pipeline has run for
   real, without touching IBM tonight): `results/ibm_fixed_angle_campaign_v1/`
   contains the manifest of **30 completed jobs on ibm_marrakesh** (CP_3–CP_7 at
   K=3, both objectives, 512 shots each, fixed-angle QAOA p=1, provider job ids
   recorded per job). The result route `GET /api/qpu/<run_id>/result` decodes
   the real counts and, when the instance is enumeration-bounded, reports
   `reference_source: "exhaustive_certified"` — the hardware samples are scored
   against a certified classical optimum, not against themselves. 151 tests
   (`tests/test_qpu_real_artifact.py`) chain-verify these recorded runs.

## 6. Expected results at each step

| Step | Expectation |
|---|---|
| CP_3 select | auto-comparison completes in seconds; all local families run |
| Objective | maneuver-count optimum **2 of 3** aircraft deviated, K=3, certified |
| Table | exact routes show `= réf ✓`; heuristics may match or show a Δ; QAOA-Aer is an incumbent, never labeled optimum |
| Explanation | 1 aircraft on NOOP, 2 with a heading change of ±30° (π/6); tie set has 6 equivalent optima, one deterministic representative shown |
| CP_4 (backup) | same flow at 12 qubits; same certified-reference behavior |
| CP_10 | partial applicability with per-method business reasons; no errors |
| QPU tab | disabled state, OFFLINE badge, no submit path |
| Export | JSON snapshot identical to the displayed configuration |

## 7. What to say honestly about each method label

- **référence classique** (exhaustive): certified discrete optimum within the
  enumeration cap — the ground truth at this K.
- **Gurobi MILP**: certified discrete optimum with a published bound/gap; the
  only exact route past the enumeration caps.
- **QUBO exact**: classical enumeration of the QUBO — proves the encoding, not a
  quantum method.
- **SA / Ocean SA**: classical heuristics; their result is an *incumbent*, shown
  matching or not matching the reference; never called an optimum.
- **QAOA (Aer)**: quantum *algorithm* on a classical *simulator* — not quantum
  hardware evidence; incumbent only.
- **IBM QPU**: real hardware exists in the recorded campaign artifacts only;
  tonight's demo submits nothing.
- Wall-clock differences are host-dependent and are never a quantum-advantage
  claim; the dataset's claims block records `quantum_speedup: false`.

## 8. Diagnostics

```bash
# End-to-end demo-flow check (7 Playwright E2E tests):
PYTHONPATH=src .venv/bin/python -m pytest tests/ui/test_demo_flow.py -q

# Server capability check while it runs:
curl -s http://127.0.0.1:8000/api/health
```

## 9. If something goes wrong

- **A method seems slow**: every route is bounded (enumeration caps, memguard,
  benchmark compute caps) — it will finish or be reported not applicable. Wait,
  or switch to the backup instance CP_4. Do not restart mid-comparison; the UI
  is stale-safe (results from a previous instance never render or export).
- **IBM unavailable / no network**: nothing to do — the demo is fully offline
  by design; no view used in this script contacts IBM.
- **Port already in use**: change `port=8000` in the start command.

## 10. ABSOLUTE rule

**No real submission during the demo.** All QPU flags stay unset (fail-closed
defaults). A real IBM run is a separate, individually authorized procedure
([IBM_CONNECT_CHECKLIST.md](IBM_CONNECT_CHECKLIST.md)) requiring four explicit
environment flags, a live human confirmation with a consent phrase, and a
double-acked campaign script — none of which is part of this demo.
