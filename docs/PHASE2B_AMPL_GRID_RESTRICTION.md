# Phase 2B — grid-restricted original AMPL model: investigation & conclusion

**Question (supervisor):** add discrete grid controls to the *original*
`Code/NC_ACRP.mod` through a temporary supplemental model/run — `Code/` read-only,
no copied or silently altered historical model, common-geometry rescore mandatory
— as a second discrete validation leg. *If technically redundant or unreliable,
prove and document that conclusion.*

## What the original model is (evidence: `Code/NC_ACRP.mod`)

Continuous MINLP:

- variables `q[i] ∈ [0.94,1.03]`, `theta[i] ∈ [-π/6,π/6]`, auxiliaries
  `vrx[i,j]`, `vry[i,j]`, and separation-side binaries `z[i,j]` (line 60-64);
- objective `Σ_i wd·theta[i]² + (1-wd)·(1-q[i])²` (line 67) — exactly our
  `quadratic_control_cost_v1`;
- the coupling constraints are **trigonometric and non-convex** (lines 70-71):
  `vrx = q[i]·v0[i]·cos(theta[i]+theta0[i]) − q[j]·v0[j]·cos(theta[j]+theta0[j])`
  (and `sin` for `vry`);
- conflict avoidance is a **big-M disjunction** on `vrx/vry` driven by `z[i,j]`
  (lines 72-81), with `Mbin*`, `gamma*`, `phi*` params supplied by
  `Preprocessing.run`.

## Is a grid-restricted version implementable?

**Yes, but only by linearising the trigonometry.** Restricting `theta[i]` (and
`q[i]`) to the maneuver grid means `cos(theta[i]+theta0[i])` and
`sin(theta[i]+theta0[i])` take one of `K` *precomputable constants* per aircraft.
Introducing `sel[i,o] ∈ {0,1}`, `Σ_o sel[i,o]=1`, and

```
vrx[i,j] = Σ_o sel[i,o]·(q_o·v0[i]·cos(θ_o+θ0[i])) − Σ_p sel[j,p]·(q_p·v0[j]·cos(θ_p+θ0[j]))
```

turns lines 70-71 into **linear** constraints in `sel`, leaving the big-M
disjunction (linear in `vrx/vry/z`) intact. The result is a MILP in
`sel[i,o] + z[i,j]` that Gurobi can solve. So B is *technically implementable*
via a supplemental `.run` that `model`s `Code/NC_ACRP.mod` read-only, `include`s
`Preprocessing.run`, and adds only the `sel` layer + trig coefficients — never
editing `Code/`.

## Why it is redundant AND higher-risk than Phase 2A — documented conclusion

1. **The scientific claim is already proven without it.** Phase 2A/C/D show, on
   20 cases (both objectives), `exhaustive == Gurobi discrete MILP == Exact QUBO`
   with a hard-stop harness. Phase 2A's MILP forbids exactly the option pairs the
   **common geometry** marks as conflicting (`DiscreteACRP.conflict`), so it
   *is* the compatibility formulation of the discrete problem.

2. **B's only distinct content is geometry↔model fidelity**, i.e. "does
   `geometry`'s conflict predicate equal `NC_ACRP`'s big-M separation on the
   grid?". That is a property of the shared geometry kernel versus the model —
   orthogonal to the discrete-equivalence claim — and is already exercised by the
   existing continuous `original_ampl` leg, which runs the *unmodified*
   `NC_ACRP.mod` and is re-scored by the same `geometry` kernel (status degraded
   to `UNKNOWN` on any divergence). Routing that fidelity check through a second
   MILP does not strengthen it; it only adds a heavier path that can diverge for
   reasons unrelated to the claim.

3. **Reliability / new confounds.** B re-introduces the big-M constants
   (`Mbin*`), which are computed by `Preprocessing.run` for the *continuous*
   ranges; with the `z[i,j]` disjunction they add solver-tolerance and modelling
   surface (a mis-scaled big-M silently loosens a separation constraint) that
   Phase 2A deliberately avoids by using the exact per-option-pair conflict truth
   table. A divergence in B could therefore reflect big-M scaling or the
   disjunction encoding rather than any real discrete disagreement — the opposite
   of a clean cross-check.

**Conclusion:** the grid-restricted original model is implementable (via trig
linearisation) but is **redundant for the discrete-equivalence proof and strictly
higher-risk** than Phase 2A. We therefore do **not** add it as an equivalence
leg. The geometry↔original-model fidelity it would probe is (a) the design
contract of the `geometry` kernel and (b) already covered by the continuous
`original_ampl` leg run on the untouched `Code/NC_ACRP.mod`. `Code/` was only
read, never modified, during this investigation.
