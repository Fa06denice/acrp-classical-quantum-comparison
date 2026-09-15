# Variant mapping: "minimiser le nombre de vols/avions réaffectés" → `maneuver_count_v1`

**Status: PROVEN** (by code reading with file:line citations below, plus executable
proof in `tests/test_reassignment_metric.py` — 6 tests, all passing with
`.venv/bin/python -m pytest tests/test_reassignment_metric.py -q`).

## 1. Verdict in one paragraph

The business variant *"minimiser le nombre de vols/avions réaffectés (changés) par
rapport au planning initial"* is **not an instance family** (not CP, FP, GP, RCP or
RCP_FL — those are geometric scenario families, see §3). It is the **objective**
`maneuver_count_v1` defined in `src/acrpq/objectives.py:16-22,118-119`:

```
reassignments = #{ i : (q_i, theta_i) != (1, 0) }  =  #{ i : option_i != NOOP }
```

i.e. the number of **aircraft** whose chosen control deviates from the initial plan
(the initial plan for every aircraft is "keep initial speed and heading": speed
rate `q = 1`, heading change `theta = 0`). This is exactly "nombre d'avions déviés
par rapport au planning initial".

**Important honesty note:** the repo models *tactical conflict-resolution
maneuvers* (Aircraft Conflict Resolution Problem, ACRP), **not** fleet/tail
assignment. The notion *"flights reassigned to another aircraft"* (aircraft swaps
/ tail reassignment) **does not exist anywhere in this repository** — grep for
"reassign/réaffect/swap" only hits unrelated code (SWAP gates, solver-swap
guards). The closest real metric — and the one every layer of this repo agrees on
— is **"number of aircraft ordered to maneuver"** (`maneuver_count` /
`n_maneuvered`). If the business wording means tail swaps, this repo cannot
compute it; if it means "how many aircraft had their plan changed", it is exactly
`maneuver_count_v1`.

## 2. The proven chain, layer by layer

### 2.1 Data — the initial plan

An instance (`Instance`, `src/acrpq/model.py:94-102`) contains, per aircraft:
initial speed `v0`, initial heading `cap`, initial position `(x0, y0)`, plus the
separation norm `d` and (RCP_FL only) flight levels `nf`/`l0`
(`src/acrpq/io/loader.py:97-152`, `Data/README.md`). The *initial plan* is
"every aircraft continues at its current speed and heading". A resolution is
expressed as per-aircraft controls relative to that plan: speed rate
`q ∈ [0.94, 1.03]` and heading change `theta ∈ [-PI/6, +PI/6]`
(`Code/NC_ACRP.mod:49-52,60-61`; mirrored in `src/acrpq/model.py:32-33`).
Therefore **`(q, theta) = (1, 0)` IS the initial plan** — the NOOP
(`src/acrpq/objectives.py:74-78`, `src/acrpq/model.py:180-182`
`ManeuverOption.is_noop`).

### 2.2 Discretisation — the NOOP option

`ManeuverGrid.build` (`src/acrpq/model.py:239-295`) builds `K = n_theta × n_q`
options per aircraft and **guarantees an exact NOOP option exists** by snapping
the nearest theta level to `0.0` and q level to `1.0` (`_snap_to`,
`src/acrpq/model.py:277-278,292-294`; `grid.noop_index()`,
`src/acrpq/model.py:233-237`). For the demo grid `n_theta=3, n_q=1`:
options `{(1, -PI/6), (1, 0)=NOOP idx 1, (1, +PI/6)}` (verified at runtime in
`tests/test_reassignment_metric.py::test_grid_k3_has_exact_noop`).

### 2.3 Decision variables — one-hot per aircraft

`x[i, o] ∈ {0,1}`, aircraft `i` (1-based) takes option `o`; bit index
`b = (i-1)·K + o`; total `n·K` binary variables
(`src/acrpq/discretize.py:13-17,52-54`). One-hot (`Σ_o x[i,o] = 1`) and pairwise
conflict-freedom are **constraints** (penalties in the QUBO), never objective
terms (`src/acrpq/objectives.py:26-28`, `src/acrpq/quantum/qubo.py:6-14`).

### 2.4 Classical objective (continuous leg) — `moved[i]`

The new AMPL leg (`src/acrpq/classical/original_ampl_maneuver_count.py`, addon
model `src/acrpq/classical/models/NC_ACRP_maneuver_count.mod`) keeps the
historical geometry/constraints of `Code/NC_ACRP.mod` verbatim and adds one
binary indicator per aircraft:

```
var moved{i in A} binary;                               (.mod:10)
minimize ManeuverCount: sum{i in A} moved[i];           (.mod:12)
theta[i] <= hmax*moved[i];  theta[i] >= hmin*moved[i];  (.mod:14-15)
q[i]-1 <= (qmax-1)*moved[i];  1-q[i] <= (1-qmin)*moved[i];  (.mod:16-17)
```

`moved[i] = 0 ⇒ (q_i, theta_i) = (1, 0)` — exactly the NOOP. The Python wrapper
re-derives the objective as `sum(moved)` and cross-checks the indicator link
within `1e-9` (`original_ampl_maneuver_count.py:149-155`), declares
`OBJECTIVE_ID = "maneuver_count_v1"` (line 26). **Consistent with
`maneuver_count_v1`** — same NOOP, same counting — with one domain caveat (§5, D1).

### 2.5 Discrete classical reference — exhaustive enumeration

`exhaustive_discrete_optimum` (`src/acrpq/classical/discrete_enum.py:65-140`)
enumerates all `K^n` assignments, keeps only **conflict-free** ones (hard
constraint, discrete_enum.py:105: `assignment_conflicts(choice) == 0`), and minimises
`primary_objective(objective_id, …)` — for `maneuver_count_v1` that is
`Σ_i [ (q_i, theta_i) != (1,0) ]` (`src/acrpq/objectives.py:145-155,118-119`).
Ties are resolved exactly (`tol=0.0`) with a deterministic lexicographic
representative (`pick_optimum`, `src/acrpq/objectives.py:190-223`).

### 2.6 QUBO objective — 0 for NOOP, 1 otherwise

`option_cost_vector(maneuver_count_v1, grid, w)` returns per option:
`0.0` for the NOOP, `1.0` otherwise (`src/acrpq/objectives.py:123-142`).
`build_qubo` puts exactly this vector into the linear term
(`src/acrpq/quantum/qubo.py:184-189,215-220`), so on the one-hot subspace

```
C(x) = Σ_i Σ_{o != noop} x[i, o]
```

(`src/acrpq/objectives.py:20-22`). Penalty dominance (`lam_pen > n·cost_max`,
`lam_oh > lam_pen·maxdeg`) guarantees the QUBO global optimum is a valid one-hot
assignment (`src/acrpq/quantum/qubo.py:16-25,202-208`); spoof/rebuild guards:
`assert_objective_consistent` (qubo.py:268) and `verify_qubo_wellformed`
(qubo.py:296). For a valid, conflict-free one-hot bitstring the QUBO energy
**equals** the reassignment count (verified: energy of the CP_3 optimum
bitstring is exactly `2.0`).

### 2.7 Decoding — bitstring → per-aircraft option → who changed

`decode_assignment` (`src/acrpq/quantum/decode.py:21-40`) maps bits back to
`{aircraft: option_idx}` with documented repairs: no bit set → **NOOP fallback**
(line 36), multiple bits → cheapest active option (line 39). `decode`
(decode.py:43-96) rescores with the common geometry kernel.
`bitstring_explanation` (`src/acrpq/quantum/explain.py:212-259`) exposes, per
aircraft, `decoded_option`, `decoded_label` (== `"NOOP"` for unchanged aircraft),
`q`, `theta`, `cost`, and any repair applied — everything needed to display who
was "réaffecté". The certified-optimum bitstring helper is
`optimum_onehot_bits` (`src/acrpq/dashboard/qpu_batch.py:291-307`).

### 2.8 UI metric — computable but not yet exposed as a number

`src/acrpq/dashboard/api.py` attaches per aircraft `option_idx`,
`option_label`, `maneuver_cost` (api.py:519-525) and in `payload["solution"]`
the full `choice` list and `grid_k` (api.py:543-544), plus the explanation block
(encoding rows with `decoded_label`). Therefore

```
reassignments = count(aircraft where option_label != "NOOP")
              = count(choice[i] != noop_index)
```

is **fully computable from the existing payload** — but **no field named
`n_maneuvered`/`reassignments` was, at audit time, in the API payload or shown in
`app.js`** (grep of `api.py`/`app.js` finds no such field; `app.js` only shows
per-aircraft maneuver rows and costs, e.g. app.js:834,1524). The library-level
metric exists in `secondary_metrics` (`src/acrpq/objectives.py:175-182`:
`maneuver_count`, `n_maneuvered`, `n_unchanged`) and in the QPU export path, but
the dashboard solve payload does not surface it as a headline number.

## 3. What CP / FP / GP / RCP / RCP_FL actually are

Per `Data/README.md` and `src/acrpq/io/loader.py:39-46,139-152`, these are
**scenario-geometry families**, not objective variants:

| Family | Meaning | Extra data |
|---|---|---|
| CP | Circle Problem (aircraft on a circle, converging) | — |
| FP | Flow Problem | — |
| GP | Grid Problem | — |
| RCP | Random Circle Problem (2D) | — |
| RCP_FL | Random Circle with Flight Levels (3 or 5 levels) | `nf`, `l0` (loader.py:139-152) |

All families share the same single historical objective in `Code/NC_ACRP.mod:67`
(quadratic control cost `Σ wd·theta² + (1-wd)(1-q)²`). **No family encodes a
"minimum reassignments" objective** — the variant is selected by
`objective_id = maneuver_count_v1`, orthogonal to the family. The hypothesis
"the variant is called RCP-FL" is **false**: RCP_FL only adds flight levels.

## 4. K (`n_theta`) — exact meaning and impact

- **Definition:** `K = n_theta × n_q` = number of candidate maneuver options per
  aircraft (`src/acrpq/model.py:225-227`, grid = cartesian product of `n_q`
  speed levels and `n_theta` heading levels, model.py:239-295). The demo uses
  `n_q = 1` (speed fixed), so `K = n_theta` heading options spanning
  `[-PI/6, +PI/6]`, always including the exact NOOP.
- **Qubits:** `n_qubits = n_aircraft × K` (`src/acrpq/discretize.py:17`,
  qubo.py:176-182 enforces the budget). CP_3 at K=3 → 9 qubits.
- **Search space:** `K^n` one-hot assignments
  (`src/acrpq/classical/discrete_enum.py:92` `space = k**n`; exhaustive proof
  capped at 2,000,000, discrete_enum.py:40).
- **Why K=3 for the demo:** smallest grid with a NOOP plus one maneuver each
  side (`{-PI/6, 0, +PI/6}`); minimises qubits (n×3) and keeps `K^n`
  exhaustively certifiable. The dashboard slider defaults to 3
  (`src/acrpq/dashboard/static/index.html:56`, `min=3 max=7 step=2`); the JS
  state literal is 5 (app.js:408) but is overwritten from the DOM at init
  (app.js:2041), so the effective default is K=3. The API function default is
  `n_theta=5` (api.py:351) — callers always pass the UI value.
- **Comparability across K:** for `maneuver_count_v1` the objective *value* is
  comparable across K (it is an integer count with identical meaning), but the
  *feasible set grows with K* (finer headings can dodge conflicts), so the
  optimal count is **non-increasing in K on nested grids** and results must be
  reported "at K": a count of 2 at K=3 may drop at K=5/7 and can be lower still
  in the continuous AMPL leg (see D1). Odd K keeps the grid symmetric and the
  NOOP exact by construction (snap, model.py:277).

## 5. Discrepancies found (severity-ranked)

- **D1 (medium, inherent — must be stated when comparing):** the continuous
  AMPL `ManeuverCount` leg (§2.4) minimises moved-aircraft over the *continuous*
  control box, while `maneuver_count_v1` in the discrete/QUBO legs minimises
  over the K-option grid. Same semantics, different feasible domains: the
  continuous optimum is a lower bound on any grid optimum. Nothing in the code
  conflates them (the AMPL leg is "deliberately a separate solver leg",
  original_ampl_maneuver_count.py:3-5), but any UI/report that shows both side
  by side must label the domain. Proposal: always annotate counts with
  `@K=<k>` vs `@continuous`.
- **D2 (low):** NOOP equality is exact (`==`) in the discrete chain
  (`objectives.py:105-107`, `model.py:180-182` — safe because `_snap_to`
  guarantees exact `1.0`/`0.0` grid values), but tolerance-based (`1e-9`) in the
  AMPL leg's indicator-link integrity check
  (original_ampl_maneuver_count.py:149-153). Correct for floating solver
  output; a solution with `theta = 1e-10` counts as moved=0 there while
  `is_noop` would say False. Impact: at most off-by-noise on the continuous leg
  only; the reported objective is `sum(moved)` so it is self-consistent.
- **D3 (low, UI gap):** the dashboard payload does not expose
  `n_maneuvered`/`n_unchanged` as a headline field even though
  `secondary_metrics` computes them (objectives.py:175-182) and the payload
  already carries everything needed (`choice`, per-aircraft `option_label`).
  Proposal: add `"n_maneuvered"`/`"n_unchanged"` to `payload["solution"]` in
  api.py (one-line change from `sol.choice` + `grid.noop_index()`), and a
  headline "aircraft rerouted: X of N" in app.js.
- **D4 (cosmetic):** dead initial value `n_theta: 5` in the JS state literal
  (app.js:408) vs the real DOM-driven default 3 (index.html:56, app.js:2041);
  and the API function-signature default `n_theta=5` (api.py:351). No runtime
  effect observed, but three different "defaults" invite drift.

## 6. Executable proof (CP_3, K=3)

`tests/test_reassignment_metric.py` (all passing):

1. Grid/NOOP/layout sanity: K=3, NOOP at index 1, 9 qubits.
2. Certified optimum: `optimum_onehot_bits` returns bitstring `100100010`,
   optimum `2.0`, `n_optima = 6`, feasible; decoding gives choice `(0, 0, 1)` →
   **reassignments = 2 = n−1**; QUBO energy of that bitstring is exactly 2.0.
3. Exhaustive cross-check: enumeration of all 27 assignments confirms optimum
   2.0 with 6 exact ties and proves **no** conflict-free assignment maneuvers
   fewer than 2 aircraft; `secondary_metrics` agrees
   (`maneuver_count = n_maneuvered = 2`, `n_unchanged = 1`).
4. All-NOOP assignment → 0 reassignments; all-zeros bitstring repairs to NOOP →
   0 reassignments.
5. Mixed hand-counted cases: `(0, NOOP, 2)` → 2; `(NOOP, 0, NOOP)` → 1; both
   equal `primary_objective(maneuver_count_v1, …)`.


> **UPDATE (post-integration, same night):** finding D3 is resolved — the dashboard payload now exposes `solution.n_maneuvered` / `n_unchanged` and a per-aircraft `deviated` flag, and the UI renders 'avions déviés X/n' with the per-aircraft table (commit e599553). D4 (K default) unified at K=3 in the UI.
