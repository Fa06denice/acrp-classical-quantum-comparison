#!/usr/bin/env python3
"""Run the OFFLINE IBM batch pipeline (Phase 5) — fake gateways only, no real IBM.

Drives the full safe QPU pipeline (prepare → confirm → submit → reconcile →
completed → decode) for a small batch of instances, through the resumable,
EXACT-TYPE-fenced ``qpu_batch._run_offline_lifecycle`` driver with a FAKE gateway
and a LOCAL fake backend. It NEVER constructs a real gateway, NEVER reads a token,
and NEVER contacts IBM. It refuses to run at all if the environment is configured
for real submission (``assert_offline``). The batch is persistent and resumable
(BatchJobStore); a crash never causes a second submission of the same job.

Emits results/qpu_batch_offline_v1/batch.json (+ parity-checked .csv) with
provenance and a content hash; ``--verify`` recomputes it.

Run:  python scripts/run_qpu_batch_offline.py [--out results/qpu_batch_offline_v1]
      python scripts/run_qpu_batch_offline.py --verify DIR/batch.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from acrpq.dashboard.qpu_batch import (
    BatchJobStore,
    BATCH_SCHEMA,
    DEFAULT_FAKE_BACKEND,
    assert_json_csv_parity,
    assert_offline,
    canonical_batch_sha256,
    rows_to_csv,
    run_offline_qpu_job,
)
from acrpq.io.loader import InstanceLoader

# Small batch: every job fits the 16-qubit local FakeGuadalupeV2 and has a
# certified exhaustive optimum to check the decoded result against.
DEFAULT_INSTANCES = ("CP_3", "CP_4", "CP_5")
DEFAULT_OBJECTIVES = ("maneuver_count_v1", "quadratic_control_cost_v1")

_HARNESS_FILES = [
    "src/acrpq/dashboard/qpu_batch.py",
    "src/acrpq/dashboard/qpu_orchestrator.py",
    "src/acrpq/dashboard/hardware_validation.py",
    "src/acrpq/dashboard/ibm_runner.py",
    "src/acrpq/dashboard/qpu_runs.py",
    "src/acrpq/dashboard/qpu_flags.py",
    "src/acrpq/dashboard/qpu_ibm_factory.py",
    "src/acrpq/classical/discrete_enum.py",
    "src/acrpq/quantum/qubo.py",
    "src/acrpq/objectives.py",
    "src/acrpq/geometry.py",
    "src/acrpq/model.py",
    "src/acrpq/discretize.py",
    "src/acrpq/io/loader.py",
    "scripts/run_qpu_batch_offline.py",
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


def _provenance(safety: dict) -> dict:
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
        "versions": {k: repro["deps"].get(k)
                     for k in ("qiskit", "qiskit_aer", "qiskit_ibm_runtime", "numpy")},
        "harness_files": list(_HARNESS_FILES), "harness_sha256": _harness_sha256(),
        "safety": {
            **safety,
            "no_ibm": True,
            "gateway": "acrpq.dashboard.qpu_batch.FakeBatchGateway (dry_run_safe=True)",
            "backend": "local qiskit_ibm_runtime.fake_provider (no service/token)",
            "seam": "qpu_batch._run_offline_lifecycle (exact-type-fenced, resumable)",
            "gateway_fence": "exact type (require_offline_exact_gateway), not a self-declared marker",
            "result_normalizer": "hardware_validation._normalise_distribution (same as real provider)",
            "resumable": "BatchJobStore persists local_run_id before submit; no double submission",
            "config_identity": "each job keyed by the FULL protocol config hash (protocol version, "
                               "instance, objective, grid, angles, shots, seed, transpile level, "
                               "backend, qubo hash, max_quantum_seconds); computed before any store "
                               "return; a store entry for a different config is refused fail-closed "
                               "(never reused/overwritten), and the recovery scan never adopts "
                               "another config's run when the index is intact.",
            "durability": "run store (runs_dir) is authority via config_hash scan; a lost batch "
                          "index cannot resubmit and an orphaned run is adopted once. A corrupted "
                          "run is matched by its CHAIN-VERIFIED manifest config_hash (accidental "
                          "or naive single-field tamper breaks event0.prev_sha256 and is rejected) "
                          "and fails CLOSED (never prepared past as a duplicate) when it matches or "
                          "cannot be chain-verified. Tamper-evident not tamper-proof (qpu_runs "
                          "honest limit): an adversary who also re-anchors event0 is the "
                          "signature-required class, out of scope offline and harmless (synthetic "
                          "job). Residual durability case: losing runs_dir itself forfeits resume "
                          "(offline: new synthetic job, no real duplicate; real provider would "
                          "need job-tag dedup, which also covers a concurrent-prepare race).",
            "synthetic_counts_source": "classical_optimum_delta (labelled synthetic; "
                                       "no quantum sampling occurred)",
            "note": "OFFLINE ONLY. No real IBM submission, no token read/exported. A real "
                    "submission would require flipping the fail-closed qpu_flags AND a "
                    "separate explicit human authorization; this CLI does neither.",
        },
    }


def _verify(path: str) -> int:
    batch = json.loads(Path(path).read_text())
    ok = batch.get("batch_sha256") == canonical_batch_sha256(batch)
    print(f"{'OK batch_sha256 verified' if ok else 'TAMPERED'}: {batch.get('batch_sha256')}")
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/qpu_batch_offline_v1")
    ap.add_argument("--verify", metavar="BATCH_JSON")
    ap.add_argument("--instances", nargs="*", default=list(DEFAULT_INSTANCES))
    ap.add_argument("--backend", default=DEFAULT_FAKE_BACKEND)
    ap.add_argument("--shots", type=int, default=512)
    ap.add_argument("--runs-dir", default=None,
                    help="Durable QPU run store. Default: ephemeral tempdir (no cross-restart "
                         "resume). Pass a persistent path so a crashed batch can resume across "
                         "process restarts without ever resubmitting a job.")
    ap.add_argument("--store-dir", default=None,
                    help="Durable batch index. Default: ephemeral tempdir. Pair with a persistent "
                         "--runs-dir for cross-restart resume.")
    args = ap.parse_args()
    if args.verify:
        return _verify(args.verify)

    # Fail-closed FIRST: refuse to run if the environment could permit real submission.
    try:
        safety = assert_offline()
    except Exception as exc:  # noqa: BLE001
        print(f"REFUSING (not offline): {exc}")
        return 4

    provenance = _provenance(safety)
    ld = InstanceLoader()
    # Durable run store + index if given (cross-restart resume); else ephemeral. With
    # a persistent --runs-dir, a crashed batch resumes without ever resubmitting a job
    # (the run store is the authority via config_hash; a lost index cannot re-submit).
    runs_dir = args.runs_dir or tempfile.mkdtemp(prefix="acrpq-qpu-offline-")
    store = BatchJobStore(args.store_dir or tempfile.mkdtemp(prefix="acrpq-qpu-batchstore-"))
    if not args.runs_dir:
        print("[note] ephemeral runs-dir: no cross-restart resume. Pass --runs-dir "
              "(and --store-dir) for durable resume.")

    rows = []
    for instance in args.instances:
        for objective_id in DEFAULT_OBJECTIVES:
            # job_key omitted on purpose: run_offline_qpu_job derives it from the FULL
            # protocol config hash, so a config change can never reuse a stale entry.
            rec = run_offline_qpu_job(ld, instance, objective_id=objective_id, n_theta=3,
                                      shots=args.shots, backend_name=args.backend,
                                      runs_dir=runs_dir, store=store)
            rows.append(rec)
            print(f"  {instance} {objective_id:26} n_q={rec['n_qubits']:2} "
                  f"state={rec['final_state']:9} opt={rec['reference_optimum']} "
                  f"decoded={rec['decoded_best_energy']} match={rec['decoded_matches_reference']} "
                  f"runs={rec['gateway_run_calls']}")

    n_completed = sum(1 for r in rows if r["final_state"] == "completed")
    n_matched = sum(1 for r in rows if r["decoded_matches_reference"] is True)
    batch = {
        "schema": BATCH_SCHEMA,
        "no_ibm": True,
        "backend": args.backend,
        "n_jobs": len(rows),
        "n_completed": n_completed,
        "n_decoded_matches_reference": n_matched,
        "provenance": provenance,
        "results": rows,
    }
    batch["batch_sha256"] = canonical_batch_sha256(batch)
    batch["batch_sha256_note"] = ("sha over scientific content; volatile run ids/timings "
                                  "+ repo_dirty excluded")

    out = Path(args.out)
    bench_path = out / "batch.json"
    if bench_path.exists():
        existing = json.loads(bench_path.read_text())
        if existing.get("batch_sha256") != batch["batch_sha256"]:
            print(f"REFUSING to overwrite differing historical {bench_path} "
                  f"(existing {existing.get('batch_sha256', '')[:16]}… != "
                  f"new {batch['batch_sha256'][:16]}…). Use a fresh --out.")
            return 3

    csv_text = rows_to_csv(rows)
    assert_json_csv_parity(rows, csv_text)
    _atomic_write(bench_path, json.dumps(batch, indent=2, sort_keys=True, allow_nan=False))
    _atomic_write(out / "batch.csv", csv_text)

    print(f"\n[offline] {len(rows)} jobs, {n_completed} completed, "
          f"{n_matched} decoded==reference-optimum; NO IBM, no token.")
    print(f"-> {bench_path} (+ .csv, parity OK)  sha256={batch['batch_sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
