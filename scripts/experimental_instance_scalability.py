#!/usr/bin/env python
"""EXPERIMENTAL — instance-scalability audit over ALL library instances.

Produces (under ``results/experimental_instance_scalability_v1/``):

* ``instances.json`` / ``instances.csv`` — one row per instance.
* ``family_summary.json`` — class counts per family, "newly accessible"
  counts (decomposition/active-reduction vs the monolithic K3 global width).
* ``manifest.json`` — git commit, python version, timings, sha256 of outputs.

``experimental = true``, ``official_benchmark = false``, ``real_qpu = false``
on every artefact. Reuses ``acrpq.experimental.adaptive_grid`` /
``adaptive_qpu`` / ``instance_scalability`` ONLY — no official QUBO builder.
"""

from __future__ import annotations

import argparse
from acrpq.experimental import provenance as prov
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acrpq.experimental import adaptive_grid as ag  # noqa: E402
from acrpq.experimental import adaptive_qpu as aq  # noqa: E402
from acrpq.experimental import instance_scalability as isc  # noqa: E402
from acrpq.io.loader import InstanceLoader  # noqa: E402
from acrpq.model import Instance  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "experimental_instance_scalability_v1"

FIELDS = [
    "id", "family", "n", "nf", "n_initial_conflicts", "n_aircraft_in_initial_conflict",
    "n_isolated_K3", "n_edges_K3", "density_K3", "deg_min", "deg_mean", "deg_max",
    "n_components_K3", "component_sizes_K3", "largest_component_K3", "n_articulation_points",
    "n_bridges", "structure_label", "Q_global_K3", "Q_active_K3", "Q_hetero_exact_K3_quadratic",
    "Q_hetero_exact_K3_count", "Q_largest_component_K3", "cross_component_qubo_terms",
    "components_equal_under_dense_sampling", "dense_sampling_k", "n_linear_K3", "n_quadratic_K3",
    "n_quadratic_largest_component", "max_degree_interaction", "statevector_bytes_global",
    "statevector_bytes_largest_component", "exact_search_space_largest_component",
    "local_exact_feasible_largest_component", "local_exact_refusal_reason", "local_exact_feasible_global", "monolithic_representable_156_logical", "isa_2q_largest_component_measured", "isa_depth_largest_component_measured", "classes",
    "refusal_reasons",
]


def _dense_sampling_k(n: int) -> int | None:
    if n <= 20:
        return 65
    if n <= 60:
        return 9
    return None


def analyse_instance(inst: Instance) -> dict:
    refusals: list[str] = []
    levels_k3 = ag.official_k_grid_options(inst, 3)

    init_edges = isc.initial_conflict_graph(inst)
    n_initial_conflicts = len(init_edges)
    init_degs = isc.degrees(inst.n, init_edges)
    n_aircraft_in_initial_conflict = sum(1 for d in init_degs.values() if d > 0)

    edges_k3 = isc.any_option_graph(inst, levels_k3)
    degs = isc.degrees(inst.n, edges_k3)
    isolated = {v for v, d in degs.items() if d == 0}
    dens = isc.density(inst.n, len(edges_k3))
    comps = isc.connected_components(inst.n, edges_k3)
    comp_sizes = [len(c) for c in comps]
    largest = comps[0] if comps else ()
    ap, br = isc.articulation_points_and_bridges(inst.n, edges_k3)
    label = isc.structure_label(inst.n, len(edges_k3), len(comps), dens, degs)
    deg_vals = list(degs.values())
    deg_min, deg_max = min(deg_vals, default=0), max(deg_vals, default=0)
    deg_mean = sum(deg_vals) / len(deg_vals) if deg_vals else 0.0

    q_global = 3 * inst.n
    q_active = 3 * (inst.n - len(isolated))
    counts_quad = isc.dominance_survivor_counts(inst, "quadratic_control_cost_v1", levels_k3, isolated)
    counts_cnt = isc.dominance_survivor_counts(inst, "maneuver_count_v1", levels_k3, isolated)
    q_hetero_quad = sum(counts_quad)
    q_hetero_cnt = sum(counts_cnt)
    q_largest = 3 * len(largest)

    global_grid = ag.global_grid(inst, 3)
    global_qubo = aq.build_hetero_qubo(inst, global_grid, "quadratic_control_cost_v1")
    cross_terms = isc.cross_component_qubo_terms(global_qubo, comps)
    n_linear = global_qubo.n_qubits
    n_quadratic = len(global_qubo.quadratic)
    max_deg_interaction = isc.max_interaction_degree(global_qubo)

    if largest:
        sub_inst = inst.subset(largest)
        sub_grid = ag.global_grid(sub_inst, 3)
        sub_qubo = aq.build_hetero_qubo(sub_inst, sub_grid, "quadratic_control_cost_v1")
        n_quadratic_largest = len(sub_qubo.quadratic)
        search_space_largest = sub_grid.search_space()
        local_exact, local_reason = isc.local_exact_feasible(
            sub_inst, "quadratic_control_cost_v1", sub_grid, timeout_s=60, max_states=2_000_000
        )
    else:
        n_quadratic_largest = 0
        search_space_largest = 1
        local_exact, local_reason = True, "empty_component"

    sv_bytes_global = isc.statevector_bytes(q_global)
    sv_bytes_largest = isc.statevector_bytes(q_largest)

    dense_k = _dense_sampling_k(inst.n)
    components_equal = None
    if dense_k is not None:
        dense_levels = ag.official_k_grid_options(inst, dense_k)
        dense_edges = isc.any_option_graph(inst, dense_levels)
        dense_comps = isc.connected_components(inst.n, dense_edges)
        components_equal = (
            tuple(sorted(tuple(sorted(c)) for c in dense_comps))
            == tuple(sorted(tuple(sorted(c)) for c in comps))
        )
    else:
        refusals.append(f"dense_sampling_not_attempted_n_{inst.n}_over_60")

    # global (monolithic) exact feasibility under the SAME budget as the component
    global_exact, global_reason = isc.local_exact_feasible(
        inst, "quadratic_control_cost_v1", global_grid, timeout_s=60, max_states=2_000_000)
    # measured ISA 2Q count of the largest component, ONLY for pilot candidates (<= 12 logical
    # qubits); the classifier refuses HARDWARE_PILOT_PLAUSIBLE without a measurement.
    isa_2q_largest = None
    isa_depth_largest = None
    if largest and q_largest <= isc.HARDWARE_PILOT_MAX_LOGICAL:
        try:
            from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

            circ = aq.build_fixed_angle_circuit(sub_qubo, [0.4], [0.3])
            _, rep = aq.transpile_isa(circ, FakeMarrakesh(), seed_transpiler=1, optimization_level=0)
            isa_2q_largest, isa_depth_largest = rep.n_2q, rep.depth
        except Exception as exc:  # noqa: BLE001 - recorded as refusal, never as plausibility
            refusals.append(f"isa_measurement_failed: {type(exc).__name__}")
    ci = isc.ClassificationInput(
        n=inst.n, q_global_k3=q_global, q_largest_component=q_largest,
        largest_component_size=len(largest), n_components=len(comps),
        cross_component_terms=cross_terms, local_exact_feasible_global=global_exact,
        local_exact_feasible_component=local_exact, expected_isa_2q_component=isa_2q_largest,
        q_hetero_exact=min(q_hetero_quad, q_hetero_cnt),
    )
    classes = isc.classify_instance(ci)
    if not local_exact and local_reason.startswith("search_space"):
        refusals.append(local_reason)

    row = {
        "id": inst.name, "family": inst.family.value, "n": inst.n, "nf": inst.nf,
        "n_initial_conflicts": n_initial_conflicts,
        "n_aircraft_in_initial_conflict": n_aircraft_in_initial_conflict,
        "n_isolated_K3": len(isolated), "n_edges_K3": len(edges_k3), "density_K3": dens,
        "deg_min": deg_min, "deg_mean": deg_mean, "deg_max": deg_max,
        "n_components_K3": len(comps), "component_sizes_K3": comp_sizes,
        "largest_component_K3": len(largest), "n_articulation_points": len(ap),
        "n_bridges": len(br), "structure_label": label, "Q_global_K3": q_global,
        "Q_active_K3": q_active, "Q_hetero_exact_K3_quadratic": q_hetero_quad,
        "Q_hetero_exact_K3_count": q_hetero_cnt, "Q_largest_component_K3": q_largest,
        "cross_component_qubo_terms": cross_terms,
        "components_equal_under_dense_sampling": components_equal, "dense_sampling_k": dense_k,
        "n_linear_K3": n_linear, "n_quadratic_K3": n_quadratic,
        "n_quadratic_largest_component": n_quadratic_largest,
        "max_degree_interaction": max_deg_interaction,
        "statevector_bytes_global": sv_bytes_global,
        "statevector_bytes_largest_component": sv_bytes_largest,
        "exact_search_space_largest_component": search_space_largest,
        "local_exact_feasible_largest_component": local_exact,
        "local_exact_refusal_reason": local_reason,
        "local_exact_feasible_global": global_exact,
        "monolithic_representable_156_logical": q_global <= isc.ISA_COMPILABLE_MAX_LOGICAL,
        "isa_2q_largest_component_measured": isa_2q_largest,
        "isa_depth_largest_component_measured": isa_depth_largest,
        "classes": classes, "refusal_reasons": refusals,
    }
    return row


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--families", type=str, default=None, help="comma-separated family filter")
    ap_.add_argument("--max-n", type=int, default=None)
    ap_.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap_.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True) if args.out == OUT_DIR else args.out.mkdir(parents=True, exist_ok=True)
    out_dir = args.out
    loader = InstanceLoader()
    names = loader.list_names()
    if args.families:
        fams = {f.strip().upper() for f in args.families.split(",")}
        names = [n for n in names if n.split("_", 1)[0].upper() in fams
                 or (n.split("_", 1)[0].upper() == "RCP" and "FL" in fams and "@FL" in n)]

    t_start = time.time()
    rows: list[dict] = []
    by_family: dict[str, int] = {}
    n_done = 0
    t_family_start = time.time()
    cur_family = None
    for name in names:
        inst = loader.load(name)
        if args.max_n is not None and inst.n > args.max_n:
            continue
        if inst.family.value != cur_family:
            if cur_family is not None:
                print(f"  [{cur_family}] done in {time.time() - t_family_start:.1f}s "
                      f"({by_family.get(cur_family, 0)} instances)", flush=True)
            cur_family = inst.family.value
            t_family_start = time.time()
            print(f"family {cur_family}: starting...", flush=True)
        row = analyse_instance(inst)
        rows.append(row)
        by_family[cur_family] = by_family.get(cur_family, 0) + 1
        n_done += 1
        if n_done % 100 == 0:
            print(f"  ... {n_done}/{len(names)} instances processed", flush=True)
    if cur_family is not None:
        print(f"  [{cur_family}] done in {time.time() - t_family_start:.1f}s "
              f"({by_family.get(cur_family, 0)} instances)", flush=True)

    instances_json = out_dir / "instances.json"
    instances_csv = out_dir / "instances.csv"
    with instances_json.open("w") as fh:
        json.dump({"schema": "acrpq-experimental-instance-scalability/2", **isc.ARTIFACT_FLAGS,
                   "rows": rows}, fh, sort_keys=True, separators=(",", ":"))
        fh.write("\n")
    with instances_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator='\n')
        w.writeheader()
        for r in rows:
            rr = dict(r)
            for k in ("component_sizes_K3", "classes", "refusal_reasons"):
                rr[k] = json.dumps(rr[k])
            w.writerow(rr)

    # family_summary.json (schema 2: nominative lists, no 'accessible' for merely decomposable)
    fam_summary: dict[str, dict] = {}
    for r in rows:
        fam = r["family"]
        fs = fam_summary.setdefault(fam, {
            "n_instances": 0, "class_counts": {c: 0 for c in isc.CLASSES},
            "largest_component_size_histogram": {},
            "decomposable_formally": [], "width_reduced": [], "newly_local_exact_accessible": [],
            "newly_statevector_accessible": [], "hardware_pilot_plausible": [],
            "not_representable_156_logical": [],
        })
        fs["n_instances"] += 1
        for c in r["classes"]:
            fs["class_counts"][c] += 1
        lc = r["largest_component_K3"]
        fs["largest_component_size_histogram"][str(lc)] = fs["largest_component_size_histogram"].get(str(lc), 0) + 1
        for label, key in (("DECOMPOSABLE_FORMALLY", "decomposable_formally"), ("WIDTH_REDUCED", "width_reduced"),
                           ("NEWLY_LOCAL_EXACT_ACCESSIBLE", "newly_local_exact_accessible"),
                           ("NEWLY_STATEVECTOR_ACCESSIBLE", "newly_statevector_accessible"),
                           ("HARDWARE_PILOT_PLAUSIBLE", "hardware_pilot_plausible")):
            if label in r["classes"]:
                fs[key].append(r["id"])
        if not r["monolithic_representable_156_logical"]:
            fs["not_representable_156_logical"].append(r["id"])
    for fs in fam_summary.values():
        for key in ("decomposable_formally", "width_reduced", "newly_local_exact_accessible",
                    "newly_statevector_accessible", "hardware_pilot_plausible", "not_representable_156_logical"):
            fs[f"n_{key}"] = len(fs[key])
            if len(fs[key]) > 60:  # keep the file readable: full lists live in instances.json
                fs[key] = fs[key][:60] + [f"... {len(fs[key]) - 60} more (see instances.json)"]
    prov.write_compact_json(out_dir / "family_summary.json",
                            {"schema": "acrpq-experimental-instance-scalability-family-summary/2",
                             **isc.ARTIFACT_FLAGS, "families": fam_summary})
    # rewrite instances.json compact (schema 2)
    prov.write_compact_json(instances_json, {"schema": "acrpq-experimental-instance-scalability/2",
                                             **isc.ARTIFACT_FLAGS, "rows": rows})
    root = Path(__file__).resolve().parents[1]
    manifest = {
        "schema": "acrpq-experimental-instance-scalability-manifest/2", **isc.ARTIFACT_FLAGS,
        "provenance": prov.provenance_block(root), "n_instances": len(rows),
        "wall_time_s": time.time() - t_start,
        "scientific_hash_instances": prov.scientific_hash({"rows": rows}),
        "outputs": {
            p.name: {"sha256": _sha256_file(p), "bytes": p.stat().st_size}
            for p in (instances_json, instances_csv, out_dir / "family_summary.json")
        },
    }
    prov.write_compact_json(out_dir / "manifest.json", manifest)

    print(f"done: {len(rows)} instances in {time.time() - t_start:.1f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
