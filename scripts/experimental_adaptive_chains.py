#!/usr/bin/env python
"""EXPERIMENTAL — regenerate the compact adaptive chains (exact and Aer-driven) deterministically.

Reproduces results/experimental_adaptive_grid_v1/chains/{CP_3,CP_4}__{exact_local,aer_sim}
with the same preregistration used for the QPU gate (quadratic objective, NOOP init, rho=0.5,
512 shots, gamma=0.4/beta=0.3 penalty-normalised, ISA on FakeMarrakesh seed 1 opt-level 0),
runs the deep integrity verification and writes integrity_report.json (schema 2, provenance).

The read-only live-backend check (live_backend_readonly_check.json) is NOT regenerated: it
required network access and is kept as a dated historical artefact.

Flags: experimental=true, official_benchmark=false, real_qpu=false. No QPU, no network.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from acrpq.experimental import adaptive_qpu as aq
from acrpq.experimental import provenance as prov
from acrpq.io.loader import InstanceLoader

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = {
    "global": "results/experimental_adaptive_grid_v1/local_benchmark_final/references.json",
    "theta_only_q1": "results/experimental_adaptive_grid_v1/references/theta_only_q1_reference.json",
    "ibm_fixed_angle_campaign": "results/ibm_fixed_angle_campaign_v1/campaign_results.json",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "experimental_adaptive_grid_v1" / "chains"))
    ap.add_argument("--no-transpile", action="store_true")
    args = ap.parse_args()
    base = Path(args.out)
    backend = None
    if not args.no_transpile:
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

        backend = FakeMarrakesh()
    loader = InstanceLoader()
    block = prov.provenance_block(ROOT)
    report = {"schema": "acrpq-experimental-chain-integrity/2", **prov.FLAGS, "provenance": block, "chains": {}}
    for name in ("CP_3", "CP_4"):
        inst = loader.load(name)
        for driver, proto, rounds in (("exact_local", aq.ADAPTIVE_PROTOCOL_LOCAL_EXACT, 8),
                                      ("aer_sim", aq.ADAPTIVE_PROTOCOL_QPU_PILOT, 4)):
            root = base / f"{name}__{driver}"
            if root.exists():
                shutil.rmtree(root)
            prereg = aq.make_preregistration(
                inst, protocol=proto, driver_kind=driver, objective_id="quadratic_control_cost_v1",
                max_rounds=rounds, backend="aer_simulator" if driver == "aer_sim" else "none",
                shots_per_round=512, source_commit=block["scientific_source_commit"], stagnation_patience=rounds,
                reference_identity=dict(REFERENCES))
            out = aq.run_chain(root, prereg, backend=backend, transpile=backend is not None)
            store = aq.ChainStore(root)
            problems = store.verify(deep=True)
            rows = []
            for r in range(out["n_rounds"]):
                rec = store.load_round(r)
                assert rec is not None
                isa = rec.get("isa_report") or {}
                ds = rec.get("decoded_summary") or {}
                rows.append({"round": r, "n_qubits": rec["n_qubits"], "block_sizes": rec["block_sizes"],
                             "delta": rec["delta"], "effective_gammas": rec["effective_gammas"],
                             "isa_depth": isa.get("depth"), "isa_2q": isa.get("n_2q"), "isa_hash": rec.get("isa_hash"),
                             "feasibility_rate": ds.get("feasibility_rate"),
                             "selected_objective": rec["selected_objective"], "archive_objective": rec["archive_objective"],
                             "status": rec["status"], "qubo_hash": rec["qubo_hash"], "round_hash": rec["round_hash"]})
            report["chains"][f"{name}__{driver}"] = {
                "chain_id": out["chain_id"], "n_rounds": out["n_rounds"], "archive_objective": out["archive_objective"],
                "deep_verify_problems": problems, "rounds": rows}
            print(name, driver, out["n_rounds"], "rounds, archive", out["archive_objective"], "deep verify:", problems or "OK")
            # the chain.lock is a runtime artefact, never part of the pack
            lock = root / "chain.lock"
            if lock.exists():
                lock.unlink()
    # chain_id and round_hash embed the preregistration timestamp (execution identity, by design):
    # they are excluded from the SCIENTIFIC hash, which covers physics only (grids, QUBO hashes,
    # selections, archives, ISA hashes, feasibility rates).
    report["scientific_hash_exclusions_extra"] = ["chain_id", "round_hash"]
    report["scientific_hash_chains"] = prov.scientific_hash(
        prov.strip_nondeterministic(report["chains"], prov.SCIENTIFIC_HASH_EXCLUSIONS + ("chain_id", "round_hash")))
    prov.write_compact_json(base / "integrity_report.json", report)
    print("scientific hash of chains:", report["scientific_hash_chains"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
