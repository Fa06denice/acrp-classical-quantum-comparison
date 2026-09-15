#!/usr/bin/env python
"""EXPERIMENTAL — Phase 4: transpile the K3/heterogeneous/decomposed circuits
of a small "frontier" of instances per family to FakeMarrakesh.

Only runs when invoked with ``--transpile`` (real transpilation, seconds to
minutes per circuit). Without the flag it only prints the selected frontier
(no qiskit import, no transpilation) so the selection logic can be smoke-tested
cheaply.

Selection per family (from ``results/experimental_instance_scalability_v1/instances.json``,
which ``scripts/experimental_instance_scalability.py`` must have produced first):

* the LARGEST instance with ``Q_global_K3 <= T`` and the FIRST instance ABOVE
  ``T``, for ``T`` in ``{15, 21, 30}`` (i.e. n in {5,6}, {7,8}, {10,11});
* the instance with the LARGEST exactly-decomposable K3 component
  (``n_components_K3 >= 2`` and ``cross_component_qubo_terms == 0``).

For each selected case three p=1 fixed-angle circuits are built (gamma=0.4,
beta=0.3, matching the historical campaign angles):

* ``global_K3`` — the official K3 grid, whole instance (``3n`` qubits);
* ``hetero_exact`` — the expert-D dominance-reduced per-aircraft grid on K3,
  quadratic objective (``Q_hetero_exact_K3_quadratic`` qubits);
* ``largest_component`` — the K3 grid restricted to the instance's largest
  exactly-decomposable component only (``3 * largest_component_K3`` qubits).

A HARD LIMIT refuses any circuit above 60 logical qubits (never attempts to
build or transpile it) and a 120s SIGALRM timeout guards each transpile call
(recorded as ``"timeout"`` rather than hanging the run).

Output: ``results/experimental_instance_scalability_v1/transpile_frontier.json``.
"""

from __future__ import annotations

import argparse
import json
from acrpq.experimental import provenance as prov
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acrpq.experimental import adaptive_grid as ag  # noqa: E402
from acrpq.experimental import adaptive_qpu as aq  # noqa: E402
from acrpq.experimental import instance_scalability as isc  # noqa: E402
from acrpq.io.loader import InstanceLoader  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "experimental_instance_scalability_v1"
INSTANCES_JSON = RESULTS_DIR / "instances.json"
OUT_PATH = RESULTS_DIR / "transpile_frontier.json"

THRESHOLDS = (15, 21, 30)
MAX_LOGICAL_QUBITS = 60
TRANSPILE_TIMEOUT_S = 120
GAMMAS = (0.4,)
BETAS = (0.3,)


class _Timeout(RuntimeError):
    pass


def _alarm(signum, frame):  # noqa: ARG001
    raise _Timeout()


def select_frontier(rows: list[dict]) -> dict[str, list[dict]]:
    by_family: dict[str, list[dict]] = {}
    for r in rows:
        by_family.setdefault(r["family"], []).append(r)
    selection: dict[str, list[dict]] = {}
    for fam, frows in by_family.items():
        frows = sorted(frows, key=lambda r: r["n"])
        picked: dict[str, dict] = {}
        for t in THRESHOLDS:
            le = [r for r in frows if r["Q_global_K3"] <= t]
            above = [r for r in frows if r["Q_global_K3"] > t]
            if le:
                picked[f"largest_le_{t}"] = le[-1]
            if above:
                picked[f"first_above_{t}"] = above[0]
        decomposable = [r for r in frows if r["n_components_K3"] >= 2 and r["cross_component_qubo_terms"] == 0]
        if decomposable:
            best = max(decomposable, key=lambda r: r["largest_component_K3"])
            picked["largest_exact_decomposition"] = best
        # dedupe by instance id, keep the reason(s)
        by_id: dict[str, dict] = {}
        for reason, row in picked.items():
            entry = by_id.setdefault(row["id"], {"row": row, "reasons": []})
            entry["reasons"].append(reason)
        selection[fam] = list(by_id.values())
    return selection


def build_and_transpile(name: str, grid: ag.LocalGrid, inst, backend) -> dict:
    n_qubits = grid.n_qubits
    if n_qubits > MAX_LOGICAL_QUBITS:
        return {"case": name, "n_logical": n_qubits, "status": "refused_over_60_qubits"}
    qubo = aq.build_hetero_qubo(inst, grid, "quadratic_control_cost_v1")
    circuit = aq.build_fixed_angle_circuit(qubo, GAMMAS, BETAS)
    depth_before = int(circuit.depth())
    have_alarm = hasattr(signal, "SIGALRM")
    old = None
    t0 = time.perf_counter()
    try:
        if have_alarm:
            old = signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(TRANSPILE_TIMEOUT_S)
        isa, report = aq.transpile_isa(circuit, backend, seed_transpiler=1, optimization_level=0)
    except _Timeout:
        return {"case": name, "n_logical": n_qubits, "status": "timeout",
                "timeout_s": TRANSPILE_TIMEOUT_S}
    finally:
        if have_alarm:
            signal.alarm(0)
            if old is not None:
                signal.signal(signal.SIGALRM, old)
    return {
        "case": name, "status": "ok", "n_logical": report.n_logical,
        "layout_physical": list(report.layout_physical), "depth_before": depth_before,
        "depth_after": report.depth, "n_2q": report.n_2q, "n_1q": report.n_1q,
        "ops": report.ops, "compile_seconds": time.perf_counter() - t0,
        "reproducible": report.reproducible, "isa_hash": report.isa_hash,
        "backend_name": report.backend_name, "basis_ok": report.basis_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transpile", action="store_true", help="actually build/transpile circuits")
    args = parser.parse_args()

    if not INSTANCES_JSON.exists():
        print(f"missing {INSTANCES_JSON}; run scripts/experimental_instance_scalability.py first",
              file=sys.stderr)
        return 1
    doc = json.loads(INSTANCES_JSON.read_text())
    rows = doc["rows"]
    selection = select_frontier(rows)

    print("Frontier selection (Q_global_K3 = 3n; thresholds 15/21/30):")
    for fam, entries in selection.items():
        for e in entries:
            r = e["row"]
            print(f"  {fam:10s} {r['id']:16s} n={r['n']:4d} Q_global_K3={r['Q_global_K3']:4d} "
                  f"largest_component_K3={r['largest_component_K3']:4d} reasons={e['reasons']}")

    if not args.transpile:
        print("\n(--transpile not passed: selection only, no qiskit import, no circuits built)")
        return 0

    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    backend = FakeMarrakesh()
    loader = InstanceLoader()
    results: dict[str, list[dict]] = {}
    for fam, entries in selection.items():
        results[fam] = []
        for e in entries:
            r = e["row"]
            inst = loader.load(r["id"])
            levels3 = ag.official_k_grid_options(inst, 3)
            edges = isc.any_option_graph(inst, levels3)
            degs = isc.degrees(inst.n, edges)
            isolated = {v for v, d in degs.items() if d == 0}
            comps = isc.connected_components(inst.n, edges)

            cases = {}
            global_grid = ag.global_grid(inst, 3)
            cases["global_K3"] = build_and_transpile(f"{r['id']}/global_K3", global_grid, inst, backend)

            hetero_grid = isc.hetero_exact_grid(inst, "quadratic_control_cost_v1", levels3, isolated)
            cases["hetero_exact"] = build_and_transpile(f"{r['id']}/hetero_exact", hetero_grid, inst, backend)

            if "largest_exact_decomposition" in e["reasons"] and comps:
                largest = comps[0]
                sub = inst.subset(largest)
                sub_grid = ag.global_grid(sub, 3)
                cases["largest_component"] = build_and_transpile(
                    f"{r['id']}/largest_component[{largest}]", sub_grid, sub, backend
                )
            results[fam].append({"instance": r["id"], "reasons": e["reasons"],
                                 "Q_global_K3": r["Q_global_K3"],
                                 "largest_component_K3": r["largest_component_K3"], "cases": cases})
            print(f"  transpiled {r['id']}: "
                  + ", ".join(f"{k}={v.get('status')}" for k, v in cases.items()))

    out = {
        "schema": "acrpq-experimental-transpile-frontier/2", **isc.ARTIFACT_FLAGS,
        "provenance": prov.provenance_block(Path(__file__).resolve().parents[1]),
        "backend": "FakeMarrakesh", "seed_transpiler": 1, "optimization_level": 0,
        "gammas": list(GAMMAS), "betas": list(BETAS),
        "max_logical_qubits_hard_limit": MAX_LOGICAL_QUBITS,
        "transpile_timeout_s": TRANSPILE_TIMEOUT_S, "families": results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
