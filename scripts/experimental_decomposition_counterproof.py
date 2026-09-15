"""Independent mathematical audit of the K3 conflict-graph decomposition.

This script is a *counterproof*, not a reimplementation of
``src/acrpq/experimental/instance_scalability.py`` (which it never imports,
and whose outputs it never reads as inputs). It rebuilds, from the OFFICIAL
modules only (``acrpq.io.loader``, ``acrpq.model``, ``acrpq.discretize``,
``acrpq.quantum.qubo``, ``acrpq.geometry``, ``acrpq.objectives``), the two
central claims of ``docs/INSTANCE_SCALABILITY_STUDY.md``:

1. For every one of the 1042 shipped instances, the official K3 QUBO
   (``build_qubo(inst, n_theta=3, n_q=1)``) has ZERO quadratic coefficients
   linking two different connected components of the "any-option" conflict
   graph (Part 1 -> ``counterproof.json["instances"]``).
2. For the 12 small configurations where a monolithic exact K3 solve is
   tractable, decomposing into components, solving each exactly, and
   recomposing reproduces the monolithic optimum bit-for-bit (1e-12) with
   zero residual conflicts after a full geometric rescoring on the ORIGINAL
   (undecomposed) instance (Part 2 -> ``counterproof.json["recompositions"]``).

Reusable checker functions (also imported by
``tests/test_experimental_decomposition_counterproof.py``):

    any_option_edges, union_find_components, component_groups,
    cross_component_qubo_terms, recompose_and_check,
    components_use_official_grid, validate_status_record

Run:
    PYTHONPATH=.../acrp-lib-quantum-adaptive/src \
        .../acrp-lib-quantum/.venv/bin/python \
        scripts/experimental_decomposition_counterproof.py
"""

from __future__ import annotations

import csv
import json
import pathlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from acrpq.experimental import provenance as prov
from acrpq import geometry
from acrpq.discretize import DiscreteACRP
from acrpq.experimental.adaptive_grid import exact_local_solve, global_grid
from acrpq.io.loader import InstanceLoader
from acrpq.model import Instance, ManeuverGrid
from acrpq.objectives import ObjectiveId, primary_objective
from acrpq.quantum.qubo import ManeuverQUBO, build_qubo

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "experimental_instance_scalability_v1" / "counterproof"
INSTANCES_JSON = REPO_ROOT / "results" / "experimental_instance_scalability_v1" / "instances.json"

K3 = 3  # n_theta for the official K3 grid, q fixed (n_q=1)

# The 12 small configurations audited exactly in Part 2 (§8.1 announcement +
# CP_3/CP_4 added per the audit brief).
SMALL_CONFIGS: tuple[str, ...] = (
    "CP_3",
    "CP_4",
    "CP_5",
    "CP_8",
    "GP_4",
    "FP_4",
    "FP_5",
    "FP_6",
    "FP_7",
    "RCP_10_1",
    "RCP_10_2",
    "RCP_20_1",
)

OBJECTIVES: tuple[ObjectiveId, ...] = (
    ObjectiveId.QUADRATIC_CONTROL_COST_V1,
    ObjectiveId.MANEUVER_COUNT_V1,
)


# --------------------------------------------------------------------------- #
# Union-find over the any-option K3 conflict graph
# --------------------------------------------------------------------------- #
def any_option_edges(discrete: DiscreteACRP) -> set[tuple[int, int]]:
    """Edge (i, j) iff at least one (oi, oj) pair conflicts in the K3 table."""
    edges: set[tuple[int, int]] = set()
    for (i, j), mat in discrete.conflict.items():
        if any(any(row) for row in mat):
            edges.add((i, j))
    return edges


def union_find_components(n: int, edges: set[tuple[int, int]]) -> dict[int, int]:
    """aircraft id (1..n) -> component representative id."""
    parent = list(range(n + 1))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in edges:
        union(i, j)
    return {i: find(i) for i in range(1, n + 1)}


def component_groups(comp_of: dict[int, int]) -> list[list[int]]:
    groups: dict[int, list[int]] = {}
    for aid, rep in sorted(comp_of.items()):
        groups.setdefault(rep, []).append(aid)
    return sorted(groups.values(), key=lambda g: (-len(g), g))


def cross_component_qubo_terms(
    qubo: ManeuverQUBO, discrete: DiscreteACRP, comp_of: dict[int, int]
) -> int:
    """Count quadratic QUBO coefficients whose two bits belong to different
    aircraft that sit in different components. Same-aircraft one-hot terms
    are never cross-component (an aircraft is always in exactly one
    component), so this isolates conflict-penalty terms only."""
    count = 0
    for b1, b2 in qubo.quadratic:
        i1, _ = discrete.inv_index(b1)
        i2, _ = discrete.inv_index(b2)
        if i1 != i2 and comp_of[i1] != comp_of[i2]:
            count += 1
    return count


# --------------------------------------------------------------------------- #
# Part 2: exact per-component decomposition + recomposition + rescore
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Recomposition:
    instance: str
    objective_id: str
    components: list[list[int]]
    monolithic_objective: float
    decomposed_objective: float
    objective_equal: bool
    residual_conflicts: int
    max_states: int


def components_use_official_grid(inst: Instance, components: list[list[int]]) -> bool:
    """Every component's theta options (built at K3) must equal the official
    K3 levels exactly. A component solved on a different grid (e.g. K5) must
    fail this check -- mutation (e)."""
    official = global_grid(inst, K3).options[0]
    for ids in components:
        sub = inst.subset(ids)
        comp_grid = global_grid(sub, K3)
        for opts in comp_grid.options:
            if tuple(opts) != tuple(official):
                return False
    return True


def recompose_and_check(
    inst: Instance,
    objective_id: ObjectiveId,
    components: list[list[int]],
    *,
    max_states: int = 10**9,
) -> Recomposition:
    """Solve each component exactly (official K3 grid), recompose the full
    theta vector, rescore it globally, and compare to the monolithic exact
    K3 optimum on the UNDECOMPOSED instance."""
    mono = exact_local_solve(
        inst, objective_id, global_grid(inst, K3), max_states=max_states
    )
    assert mono.theta is not None, f"monolithic solve infeasible for {inst.name}"

    theta_by_id: dict[int, float] = {}
    for ids in components:
        sub = inst.subset(ids)
        sol = exact_local_solve(
            sub, objective_id, global_grid(sub, K3), max_states=max_states
        )
        assert sol.theta is not None, f"component solve infeasible for {ids}"
        for local_pos, aid in enumerate(ids):
            theta_by_id[aid] = sol.theta[local_pos]

    theta_full = tuple(theta_by_id[i] for i in inst.aircraft())
    q_full = (1.0,) * inst.n
    decomposed_obj = primary_objective(objective_id, q_full, theta_full, inst.w)
    residual = geometry.count_conflicts(inst, q_full, theta_full)

    return Recomposition(
        instance=inst.name,
        objective_id=objective_id.value,
        components=components,
        monolithic_objective=mono.objective,
        decomposed_objective=decomposed_obj,
        objective_equal=math.isclose(
            mono.objective, decomposed_obj, rel_tol=0.0, abs_tol=1e-12
        ),
        residual_conflicts=residual,
        max_states=max_states,
    )


# --------------------------------------------------------------------------- #
# Mutation (f): incumbent-vs-optimal status validation
# --------------------------------------------------------------------------- #
def validate_status_record(record: dict[str, Any]) -> list[str]:
    """A record for an instance whose monolithic solve was refused/timeout
    must carry status == 'incumbent', never 'optimal'. Returns a list of
    violation messages (empty == valid)."""
    violations: list[str] = []
    refused = record.get("monolithic_status") in ("refused", "timeout")
    status = record.get("status")
    if refused and status == "optimal":
        violations.append(
            f"{record.get('instance', '?')}: monolithic solve was "
            f"{record.get('monolithic_status')!r} but status is 'optimal' "
            "(must be 'incumbent')"
        )
    if not refused and status == "incumbent":
        violations.append(
            f"{record.get('instance', '?')}: monolithic solve succeeded but "
            "status is 'incumbent' (should be 'optimal')"
        )
    return violations


# --------------------------------------------------------------------------- #
# Part 1: full 1042-instance sweep
# --------------------------------------------------------------------------- #
def audit_instance(loader: InstanceLoader, name: str) -> dict[str, Any]:
    inst = loader.load(name)
    grid = ManeuverGrid.build(n_theta=K3, n_q=1, instance=inst)
    discrete = DiscreteACRP.build(inst, grid)
    qubo = build_qubo(inst, discrete=discrete, max_qubits=10_000)

    edges = any_option_edges(discrete)
    comp_of = union_find_components(inst.n, edges)
    groups = component_groups(comp_of)
    sizes = sorted((len(g) for g in groups), reverse=True)
    isolated = sum(1 for s in sizes if s == 1)
    cross = cross_component_qubo_terms(qubo, discrete, comp_of)

    return {
        "id": name,
        "family": inst.family.value,
        "n": inst.n,
        "n_edges_K3": len(edges),
        "n_components_K3": len(groups),
        "component_sizes_K3": sizes,
        "largest_component_K3": sizes[0] if sizes else 0,
        "n_isolated_K3": isolated,
        "n_quadratic_K3": len(qubo.quadratic),
        "cross_component_qubo_terms": cross,
    }


def compare_with_instances_json(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not INSTANCES_JSON.exists():
        return [{"severity": "MAJOR", "detail": f"{INSTANCES_JSON} not found; comparison skipped"}]
    ref = {r["id"]: r for r in json.loads(INSTANCES_JSON.read_text())["rows"]}
    fields = [
        "n",
        "n_edges_K3",
        "n_components_K3",
        "component_sizes_K3",
        "largest_component_K3",
        "n_isolated_K3",
        "n_quadratic_K3",
        "cross_component_qubo_terms",
    ]
    discrepancies: list[dict[str, Any]] = []
    for row in rows:
        other = ref.get(row["id"])
        if other is None:
            discrepancies.append(
                {"id": row["id"], "severity": "MAJOR", "detail": "not found in instances.json"}
            )
            continue
        for f in fields:
            ours = row[f]
            theirs = other.get(f)
            if isinstance(ours, list):
                ours_cmp, theirs_cmp = sorted(ours, reverse=True), sorted(theirs or [], reverse=True)
            else:
                ours_cmp, theirs_cmp = ours, theirs
            if ours_cmp != theirs_cmp:
                sev = "BLOCKER" if f == "cross_component_qubo_terms" and ours != 0 else "MINOR"
                discrepancies.append(
                    {
                        "id": row["id"],
                        "field": f,
                        "ours": ours,
                        "instances_json": theirs,
                        "severity": sev,
                    }
                )
    return discrepancies


def run_part1(loader: InstanceLoader) -> dict[str, Any]:
    names = loader.list_names()
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for name in names:
        rows.append(audit_instance(loader, name))
    elapsed = time.perf_counter() - t0

    by_family: dict[str, int] = {}
    max_cross = 0
    for r in rows:
        by_family[r["family"]] = by_family.get(r["family"], 0) + 1
        max_cross = max(max_cross, r["cross_component_qubo_terms"])

    discrepancies = compare_with_instances_json(rows)

    return {
        "n_instances": len(rows),
        "elapsed_s": elapsed,
        "family_counts": by_family,
        "max_cross_component_qubo_terms": max_cross,
        "all_cross_component_terms_zero": max_cross == 0,
        "discrepancies_vs_instances_json": discrepancies,
        "rows": rows,
    }


def run_part2(loader: InstanceLoader) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in SMALL_CONFIGS:
        inst = loader.load(name)
        grid = ManeuverGrid.build(n_theta=K3, n_q=1, instance=inst)
        discrete = DiscreteACRP.build(inst, grid)
        comp_of = union_find_components(inst.n, any_option_edges(discrete))
        groups = component_groups(comp_of)
        for oid in OBJECTIVES:
            rec = recompose_and_check(inst, oid, groups, max_states=3 ** max(inst.n, 1) + 10)
            grid_ok = components_use_official_grid(inst, groups)
            out.append(
                {
                    "instance": name,
                    "objective_id": oid.value,
                    "components": groups,
                    "monolithic_objective": rec.monolithic_objective,
                    "decomposed_objective": rec.decomposed_objective,
                    "objective_equal_1e12": rec.objective_equal,
                    "residual_conflicts": rec.residual_conflicts,
                    "official_grid_check": grid_ok,
                }
            )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loader = InstanceLoader(data_root=REPO_ROOT)

    print("[part1] auditing all instances (union-find + QUBO cross-term count)...")
    part1 = run_part1(loader)
    print(
        f"[part1] {part1['n_instances']} instances in {part1['elapsed_s']:.2f}s; "
        f"max cross-component QUBO terms = {part1['max_cross_component_qubo_terms']}"
    )
    print(f"[part1] family counts: {part1['family_counts']}")
    if part1["discrepancies_vs_instances_json"]:
        print(f"[part1] {len(part1['discrepancies_vs_instances_json'])} discrepancies vs instances.json")

    print("[part2] recomposition audit on 12 small configurations x 2 objectives...")
    t0 = time.perf_counter()
    part2 = run_part2(loader)
    elapsed2 = time.perf_counter() - t0
    print(f"[part2] {len(part2)} recompositions in {elapsed2:.2f}s")
    n_fail = sum(1 for r in part2 if not r["objective_equal_1e12"] or r["residual_conflicts"] != 0)
    print(f"[part2] failures: {n_fail}/{len(part2)}")

    payload = {
        "theorem": (
            "If no quadratic QUBO coefficient links two components, then "
            "H(x) = sum_c H_c(x_c) and argmin over one-hot states factorises "
            "componentwise; feasibility of the union follows because every "
            "conflict pair (i, j) with i, j in different components has an "
            "all-False K3 conflict table. Exactness holds for the K3 grid "
            "only: conflict tables are grid-dependent, so a finer grid can "
            "reconnect components that were disjoint at K3."
        ),
        "part1_summary": {
            k: v for k, v in part1.items() if k != "rows"
        },
        "instances": part1["rows"],
        "recompositions": part2,
    }
    (OUT_DIR / "counterproof.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    with (OUT_DIR / "counterproof.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "id", "family", "n", "n_edges_K3", "n_components_K3",
                "largest_component_K3", "n_isolated_K3", "n_quadratic_K3",
                "cross_component_qubo_terms",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for r in part1["rows"]:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    manifest = {
        "schema": "acrpq-experimental-decomposition-counterproof-manifest/2",
        "experimental": True,
        "official_benchmark": False,
        "real_qpu": False,
        "provenance": prov.provenance_block(pathlib.Path(__file__).resolve().parents[1]),
        "script": "scripts/experimental_decomposition_counterproof.py",
        "n_instances_audited": part1["n_instances"],
        "n_recompositions": len(part2),
        "elapsed_s_part1": part1["elapsed_s"],
        "elapsed_s_part2": elapsed2,
        "all_cross_component_terms_zero": part1["all_cross_component_terms_zero"],
        "n_recomposition_failures": n_fail,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[done] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
