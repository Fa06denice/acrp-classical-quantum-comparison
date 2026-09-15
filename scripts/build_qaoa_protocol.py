#!/usr/bin/env python3
"""Emit the versioned QAOA protocol artifact (Phase 4).

Documents the registered, content-hashed QAOA protocols, their Aer ISA /
transpilation targets, and a reproducibility DEMONSTRATION: each protocol is run
twice on small instances and the deterministic outputs (best_bitstring,
best_energy, optimal_params, circuit metrics) must be bit-identical — proving the
protocol's seed pins the run. Aer simulator ONLY; no IBM hardware is contacted.

The artifact carries full provenance (git/source commit, scientific_code_dirty,
harness sha, and the qiskit/qiskit-aer/qiskit-optimization versions that govern
reproducibility) and a protocol_artifact_sha256; --verify recomputes it.

Run:  python scripts/build_qaoa_protocol.py [--out results/qaoa_protocol_v1]
      python scripts/build_qaoa_protocol.py --verify results/qaoa_protocol_v1/protocol.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from acrpq.io.loader import InstanceLoader
from acrpq.quantum.qaoa_protocol import (
    DETERMINISTIC_RUN_FIELDS,
    PROTOCOL_REGISTRY,
    SCHEMA_QAOA_PROTOCOL,
    canonical_artifact_sha256,
    deterministic_run_sha256,
    deterministic_subset,
    run_protocol,
)
from acrpq.quantum.qubo import build_qubo

SCHEMA_ARTIFACT = "acrpq-qaoa-protocol-artifact/2"

DEMO_OBJECTIVES = ("maneuver_count_v1", "quadratic_control_cost_v1")

# Explicit (protocol, instance) pairs, each chosen so every SOLVED circuit is
# small (<= ~16 qubits). CP_3/CP_4 (9/12 qubits) exercise the non-decomposed
# protocols; FP_4 (24 qubits total) is run ONLY by the exact-connected-components
# protocol, which decomposes it into 3 components of <= 12 qubits each — running a
# non-decomposed protocol on the full 24-qubit circuit would be a 2**24 statevector.
DEMO_PLAN = (
    ("aer_default_p1_v1", "CP_3"), ("aer_default_p1_v1", "CP_4"),
    ("aer_default_p2_v1", "CP_3"), ("aer_default_p2_v1", "CP_4"),
    ("aer_zeros_init_p1_v1", "CP_3"), ("aer_zeros_init_p1_v1", "CP_4"),
    ("aer_decomposed_p1_v1", "FP_4"),
)

_HARNESS_FILES = [
    "src/acrpq/quantum/qaoa_protocol.py",
    "src/acrpq/quantum/qaoa.py",
    "src/acrpq/quantum/backends.py",
    "src/acrpq/quantum/memguard.py",
    "src/acrpq/quantum/decode.py",
    "src/acrpq/quantum/explain.py",
    "src/acrpq/quantum/qubo.py",
    "src/acrpq/objectives.py",
    "src/acrpq/geometry.py",
    "src/acrpq/model.py",
    "src/acrpq/discretize.py",
    "src/acrpq/io/loader.py",
    "scripts/build_qaoa_protocol.py",
]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _harness_sha256() -> str:
    h = hashlib.sha256()
    for f in _HARNESS_FILES:
        h.update(Path(f).read_bytes())
    return h.hexdigest()


def _provenance() -> dict:
    from dataclasses import asdict as _asdict

    from acrpq.benchmark.export import ReproInfo
    from acrpq.classical.ampl_campaign import git_info
    repro = _asdict(ReproInfo.capture())
    git = git_info()
    return {
        "git_commit": git["git_commit"], "source_commit": git["git_commit"],
        "generated_artifact_commit": None,
        "scientific_code_dirty": git["scientific_code_dirty"],
        "scientific_code_paths": git["scientific_code_paths"],
        "repository_dirty": git["repository_dirty"],
        "python_version": repro["python_version"], "package_version": repro["package_version"],
        "platform": repro["platform"],
        "versions": {k: repro["deps"].get(k) for k in
                     ("qiskit", "qiskit_aer", "qiskit_optimization", "numpy")},
        "harness_files": list(_HARNESS_FILES), "harness_sha256": _harness_sha256(),
        "reproducibility_note": (
            "A fixed-seed protocol reproduces its deterministic outputs bit-for-bit "
            "on Aer for the SAME platform + library versions (pinned above). The "
            "qiskit/qiskit-aer/qiskit-optimization versions govern the RNG algorithm "
            "and transpiler internals, so they ride in provenance, not in the "
            "protocol identity."),
        "no_ibm": True,
    }


def _verify(path: str) -> int:
    """Recompute the artifact hash AND independently re-derive all_reproducible
    from the stored deterministic subsets — so a reader trusts arithmetic on the
    committed bytes, not the stored boolean."""
    art = json.loads(Path(path).read_text())
    hash_ok = art.get("protocol_artifact_sha256") == canonical_artifact_sha256(art)
    ledger_ok = True
    recomputed_all = True
    for entry in art.get("reproducibility_demo", {}).get("runs", []):
        a_sha = deterministic_run_sha256(entry["deterministic_a"])
        b_sha = deterministic_run_sha256(entry["deterministic_b"])
        stored_a = entry.get("deterministic_run_a_sha256")
        stored_b = entry.get("deterministic_run_b_sha256")
        if a_sha != stored_a or b_sha != stored_b:
            ledger_ok = False  # stored sha does not match the stored subset -> tampered
        eq = (a_sha == b_sha)
        if eq != bool(entry.get("equality_verified")):
            ledger_ok = False  # stored equality_verified inconsistent with the shas
        recomputed_all = recomputed_all and eq
    stored_all = art.get("reproducibility_demo", {}).get("all_reproducible")
    if recomputed_all != bool(stored_all):
        ledger_ok = False
    ok = hash_ok and ledger_ok
    print(f"{'OK protocol_artifact_sha256 verified' if hash_ok else 'HASH TAMPERED'}: "
          f"{art.get('protocol_artifact_sha256')}")
    print(f"{'OK reproducibility ledger re-derived' if ledger_ok else 'LEDGER TAMPERED'} "
          f"(all_reproducible recomputed={recomputed_all}, stored={stored_all})")
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/qaoa_protocol_v1")
    ap.add_argument("--verify", metavar="PROTOCOL_JSON")
    args = ap.parse_args()
    if args.verify:
        return _verify(args.verify)

    provenance = _provenance()  # before any output
    ld = InstanceLoader()

    protocols = []
    for name in sorted(PROTOCOL_REGISTRY):
        p = PROTOCOL_REGISTRY[name]
        d = p.descriptor()
        d["isa"] = p.aer_isa_spec().to_json()
        protocols.append(d)

    # Reproducibility demonstration + AUDITABLE ledger: each protocol run TWICE
    # per (instance, objective). For every pair we serialise the deterministic
    # subset of BOTH runs and their sha256, plus equality_fields and
    # equality_verified — so all_reproducible can be re-derived from the committed
    # bytes (see --verify) without trusting the stored boolean.
    demo = []
    all_reproducible = True
    for version_name, inst in DEMO_PLAN:
        p = PROTOCOL_REGISTRY[version_name]
        for objective_id in DEMO_OBJECTIVES:
            qubo = build_qubo(ld.load(inst), n_theta=3, n_q=1, objective_id=objective_id,
                              max_qubits=4096)
            a = run_protocol(qubo, p)
            b = run_protocol(qubo, p)
            det_a = deterministic_subset(a)
            det_b = deterministic_subset(b)
            a_sha = deterministic_run_sha256(det_a)
            b_sha = deterministic_run_sha256(det_b)
            equality_verified = (a_sha == b_sha)
            all_reproducible = all_reproducible and equality_verified
            demo.append({
                "protocol_id": p.protocol_id, "version_name": p.version_name,
                "instance": inst, "objective_id": objective_id,
                "n_qubits": a["n_qubits"],
                "decomposition": a["decomposition"],
                "deterministic_a": det_a, "deterministic_b": det_b,
                "deterministic_run_a_sha256": a_sha,
                "deterministic_run_b_sha256": b_sha,
                "equality_fields": list(DETERMINISTIC_RUN_FIELDS),
                "equality_verified": equality_verified,
            })
            print(f"  {version_name:22} {inst} {objective_id:26} "
                  f"bits={a['best_bitstring']} obj={a['objective_value']} "
                  f"ncomp={a['decomposition']['n_components']} eq={equality_verified}")

    artifact = {
        "schema": SCHEMA_ARTIFACT,
        "qaoa_protocol_schema": SCHEMA_QAOA_PROTOCOL,
        "no_ibm": True,
        "n_protocols": len(protocols),
        "protocols": protocols,
        "reproducibility_demo": {
            "plan": [{"protocol": vn, "instance": inst} for vn, inst in DEMO_PLAN],
            "objectives": list(DEMO_OBJECTIVES),
            "equality_fields": list(DETERMINISTIC_RUN_FIELDS),
            "ledger_note": ("all_reproducible is re-derivable from each entry's "
                            "deterministic_a/deterministic_b via deterministic_run_sha256 "
                            "(see --verify); the stored boolean is not trusted on its own."),
            "all_reproducible": all_reproducible,
            "runs": demo,
        },
        "provenance": provenance,
    }
    artifact["protocol_artifact_sha256"] = canonical_artifact_sha256(artifact)
    artifact["protocol_artifact_sha256_note"] = (
        "sha over protocol + demo scientific content; wall/qpu times + repo_dirty excluded")

    out = Path(args.out)
    (out).mkdir(parents=True, exist_ok=True)
    bench_path = out / "protocol.json"
    _atomic_write(bench_path, json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False))

    print(f"\n{len(protocols)} protocols, {len(demo)} demo runs, "
          f"all_reproducible={all_reproducible}")
    print(f"-> {bench_path}  sha256={artifact['protocol_artifact_sha256'][:16]}…")
    return 0 if all_reproducible else 1


if __name__ == "__main__":
    raise SystemExit(main())
