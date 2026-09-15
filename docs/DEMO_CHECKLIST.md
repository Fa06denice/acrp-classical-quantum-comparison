# Demo checklist — before, during and after supervisor review

## Before the demo

| Priority | Task | Done when |
|---|---|---|
| P0 | Run `./scripts/demo_preflight.sh` | Every required item says `OK`. |
| P0 | Run the complete local test/lint/type/build gate | Commands and SHA are recorded; no hidden failure. |
| P0 | Rehearse `docs/SUPERVISOR_DEMO.md` twice | Route fits in 10 minutes without enabling a real provider. |
| P0 | Verify the safe launch profile | `/api/qpu/flags` reports `submission_allowed=false`. |
| P0 | Test one ALL-ALGORITHMS and one LIMITED instance | Outcomes and refusal reasons are understandable. |
| P1 | Check screen/projector readability | Demo mode is readable at the target resolution and zoom. |
| P1 | Keep TRACEABILITY and LIMITATIONS open offline | The demo survives a UI or network failure. |
| P1 | Record current commit and clean/dirty state | Exact code shown can be identified later. |
| P2 | Prepare a blank notes page with the six supervisor questions | Decisions are captured verbatim. |

## During the demo

| Rule | Why |
|---|---|
| Start from the scientific question, then show the interface | The UI is evidence of the method, not the contribution by itself. |
| Say “local simulator”, “fake backend” or “real QPU” every time | Prevents an accidental hardware claim. |
| Show one refusal/limit deliberately | Demonstrates honest applicability rather than hiding scale limits. |
| Never change K or resource caps to rescue a slow run | Keeps the demonstrated configuration interpretable. |
| Capture objections and requested experiments, not just UI comments | Those determine the post-review research work. |

## After the demo

| Order | Task | Depends on supervisor feedback? |
|---:|---|:---:|
| 1 | Convert notes into accepted decisions, open questions and requested changes | Yes |
| 2 | Freeze a reviewed experimental protocol (instances, K, seeds, metrics, exclusions) | Yes |
| 3 | Implement only approved code/protocol changes; add regression tests | Yes |
| 4 | Re-run scientific campaigns from a clean commit and archive provenance | Yes |
| 5 | Optionally execute a smallest-case IBM QPU pilot using `IBM_QPU_RUNBOOK.md` | Yes; also needs account and explicit operator approval |
| 6 | Reconcile/export any real job and update wording from “prepared” to the exact observed status | Only if a real job ran |
| 7 | Create the final code-to-thesis handoff document | After code/protocol/results are frozen |
| 8 | Final defense rehearsal, screenshots and offline backup | After the narrative is frozen |

## Explicitly deferred

The thesis text is not rewritten before supervisor review. The later handoff
should describe the final method, architecture, mathematical conventions,
experimental protocol, implementation difficulties, fixes, provenance,
limitations and exact admissible claims. Another writing session can then use
that frozen technical source without guessing what changed.
