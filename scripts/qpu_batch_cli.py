#!/usr/bin/env python3
"""Offline QPU-batch operator CLI (Phase 5.5) — strictly offline, NO real submission.

Subcommands (all read-only or fixture-driven; none submits):

  plan                estimate a batch: #jobs, qubits/job, candidate LOCAL backends,
                      max budget, and risks — no account, no network.
  inspect             show run states / batch identities / anomalies from a run store.
  verify              recompute an artefact's content hash (tamper check).
  resume-check        diagnose interrupted runs + orphan candidates (lease-classified).
  export              rebuild the CSV from an artefact JSON (parity-checked).
  reconcile-simulated simulate reconciliation with a FAKE gateway + a recorded fixture.

Hard refusals: ``assert_offline`` runs first (exit 4 if the environment could permit real
submission); ``--real``, ``--token``, ``--submit`` and any non-``Fake*`` backend are
refused (exit 2). There is NO working submit command. Output is human-readable by default
and strict JSON with ``--json``; it carries no local absolute path, PII, or secret. Exit
codes: 0 ok, 2 forbidden/usage, 3 verification failed, 4 not offline, 5 not found/corrupt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from acrpq.dashboard import qpu_batch, qpu_budget

_FORBIDDEN = ("--real", "--token", "--submit", "--api-token", "--ibm-token")


def _sanitise_out(obj):
    """Strip anything that looks like a local absolute path from CLI output."""
    if isinstance(obj, str):
        return "<path>" if obj.startswith("/") or obj.startswith("\\") else obj
    if isinstance(obj, dict):
        return {k: _sanitise_out(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise_out(v) for v in obj]
    return obj


def _emit(payload: dict, as_json: bool) -> None:
    payload = _sanitise_out(payload)
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        for k, v in payload.items():
            print(f"{k}: {v}")


def _local_fake_backends() -> list[str]:
    from qiskit_ibm_runtime import fake_provider as fp
    return sorted(n for n in dir(fp) if n.startswith("Fake"))


def cmd_plan(args) -> int:
    from acrpq.io.loader import InstanceLoader
    from acrpq.quantum.qubo import build_qubo
    if not (isinstance(args.backend, str) and args.backend.startswith("Fake")):
        print("REFUSED: only local Fake* backends are allowed offline", file=sys.stderr)
        return 2
    ld = InstanceLoader()
    jobs = []
    for inst_id in args.instances:
        for obj in args.objectives:
            q = build_qubo(ld.load(inst_id), n_theta=args.n_theta, n_q=1, objective_id=obj)
            jobs.append({"instance": inst_id, "objective_id": obj, "n_qubits": q.n_qubits})
    policy = qpu_budget.BudgetPolicy(
        max_jobs=max(1, len(jobs)), shots_per_job=args.shots,
        total_shot_budget=max(args.shots, args.shots * len(jobs)),
        qpu_seconds_per_job=args.qpu_seconds_per_job,
        max_total_qpu_seconds=max(args.qpu_seconds_per_job, args.qpu_seconds_per_job * len(jobs)),
        max_wait_seconds=600.0, max_jobs_per_backend=max(1, len(jobs)),
        max_jobs_per_instance=max(1, len(jobs)), stop_after_consecutive_errors=2)
    _emit({
        "command": "plan", "n_jobs": len(jobs),
        "qubits_per_job": [j["n_qubits"] for j in jobs], "jobs": jobs,
        "candidate_local_backends": _local_fake_backends(),
        "backend_selected": args.backend, "shots_per_job": args.shots,
        "max_total_shots": policy.total_shot_budget,
        "max_total_qpu_seconds": policy.max_total_qpu_seconds,
        "effective_max_slots": policy.effective_max_slots(),
        "risks": ["synthetic/offline — NOT quantum evidence",
                  "qubit count limits statevector simulation",
                  "provider queue depth does not guarantee wall-clock",
                  "no real backend availability is known offline"],
        "status": "OFFLINE — NO SUBMISSION",
    }, args.json)
    return 0


def cmd_inspect(args) -> int:
    from acrpq.dashboard import qpu_runs
    runs = qpu_runs.list_runs(root=args.runs_dir)
    rows = [{"local_run_id_tail": r["local_run_id"][-8:], "state": r.get("state"),
             "batch_config_hash_present": r.get("batch_config_hash") is not None,
             "n_jobs": len(r.get("ibm_job_ids", [])),
             "corrupt": r.get("state") == "corrupted"} for r in runs]
    _emit({"command": "inspect", "n_runs": len(rows), "runs": rows,
           "status": "OFFLINE"}, args.json)
    return 0


def cmd_verify(args) -> int:
    p = Path(args.artifact)
    if not p.exists():
        print(f"NOT FOUND: {p.name}", file=sys.stderr)
        return 5
    try:
        batch = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        print(f"CORRUPT: {exc}", file=sys.stderr)
        return 5
    ok = batch.get("batch_sha256") == qpu_batch.canonical_batch_sha256(batch)
    _emit({"command": "verify", "verified": ok,
           "batch_sha256": batch.get("batch_sha256")}, args.json)
    return 0 if ok else 3


def cmd_resume_check(args) -> int:
    from acrpq.dashboard import qpu_runs
    runs = qpu_runs.list_runs(root=args.runs_dir)
    interrupted = [r["local_run_id"][-8:] for r in runs
                   if not r.get("terminal") and r.get("state") not in ("corrupted", None)]
    corrupt = [r["local_run_id"][-8:] for r in runs if r.get("state") == "corrupted"]
    orphans = {}
    if args.batch_config_hash:
        inv = qpu_batch.list_orphan_runs(args.runs_dir, args.batch_config_hash)
        orphans = {"designated_present": inv.get("designated_local_run_id") is not None,
                   "orphans": [{"tail": o["local_run_id"][-8:],
                                "classification": o["classification"]} for o in inv["orphans"]]}
    _emit({"command": "resume-check", "n_runs": len(runs),
           "interrupted_non_terminal": interrupted, "corrupt_runs": corrupt,
           "orphan_inventory": orphans, "status": "OFFLINE"}, args.json)
    return 0


def _atomic_write(path: Path, text: str) -> None:
    import os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _batch_record_to_export_row(rec: dict, batch_id: str):
    from acrpq.dashboard import qpu_export_schema as X
    row = X.empty_row()
    row.update(
        batch_id=batch_id, local_run_id=rec.get("local_run_id"), instance=rec.get("instance"),
        family=rec.get("family"), objective_id=rec.get("objective_id"),
        n_theta=rec.get("n_theta"), n_q=rec.get("n_q"), n_qubits=rec.get("n_qubits"),
        backend_requested=rec.get("backend"), backend_effective=rec.get("backend"),
        shots=rec.get("shots"), optimization_level=rec.get("optimization_level"),
        final_state=rec.get("final_state"),
        best_feasible_bitstring=rec.get("decoded_best_bitstring"),
        best_raw_bitstring=rec.get("decoded_best_bitstring"),
        primary_cost=rec.get("decoded_best_energy"), qubo_energy=rec.get("decoded_best_energy"),
        comparability=("comparable" if rec.get("decoded_matches_reference") is not None
                       else "not_comparable"),
        warnings=";".join(rec.get("anomalies", [])) or "",
        data_source="synthetic_offline", official=False)
    return row


def cmd_export(args) -> int:
    from acrpq.dashboard import qpu_export_schema as X
    p = Path(args.artifact)
    if not p.exists():
        print(f"NOT FOUND: {p.name}", file=sys.stderr)
        return 5
    try:
        batch = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        print(f"CORRUPT: {exc}", file=sys.stderr)
        return 5
    batch_id = str(batch.get("batch_sha256", ""))[:16] or "offline-batch"
    rows = [_batch_record_to_export_row(r, batch_id) for r in batch.get("results", [])]
    bprov = batch.get("provenance", {})
    doc = {"schema": X.EXPORT_SCHEMA, "rows": rows, "provenance": {
        "source_commit": bprov.get("source_commit") or bprov.get("git_commit") or "unknown",
        "scientific_code_dirty": bool(bprov.get("scientific_code_dirty", False)),
        "repository_dirty": bool(bprov.get("repository_dirty", False)),
        "versions": bprov.get("versions", {}), "data_source": "synthetic_offline",
        "status": X.OFFLINE_STATUS, "official": False}}
    doc["export_sha256"] = X.canonical_export_sha256(doc)
    try:
        X.validate_export(doc, require_synthetic_offline=True, strict=True)
        csv_text = X.rows_to_csv(rows)
        X.assert_json_csv_parity(rows, csv_text)
    except X.ExportError as exc:
        print(f"EXPORT INVALID: {exc}", file=sys.stderr)
        return 3
    if args.out:
        out_json = Path(args.out)
        if out_json.exists():                            # refuse an incompatible overwrite
            try:
                prev = json.loads(out_json.read_text())
            except (OSError, ValueError):
                prev = {}
            if prev.get("export_sha256") != doc["export_sha256"]:
                print("REFUSING to overwrite a differing/non-export file (use a fresh --out)",
                      file=sys.stderr)
                return 3
        _atomic_write(out_json, json.dumps(doc, indent=2, sort_keys=True, allow_nan=False))
        _atomic_write(out_json.with_suffix(".csv"), csv_text)
    _emit({"command": "export", "schema": X.EXPORT_SCHEMA, "n_rows": len(rows),
           "export_sha256": doc["export_sha256"], "parity_ok": True, "validated": True,
           "wrote": bool(args.out), "status": "OFFLINE — synthetic, not quantum evidence"},
          args.json)
    return 0


def cmd_reconcile_simulated(args) -> int:
    """Real fixture-driven reconciliation through a FakeBatchGateway + the SAME result
    normaliser the pipeline uses — never a real gateway. Invalid/malformed/secret-bearing
    fixtures fail closed (exit 3)."""
    fp = Path(args.fixture)
    if not fp.exists():
        print(f"NOT FOUND: {fp.name}", file=sys.stderr)
        return 5
    try:
        fx = json.loads(fp.read_text())
    except (OSError, ValueError) as exc:
        print(f"FIXTURE CORRUPT: {exc}", file=sys.stderr)
        return 3
    # fail closed on a secret/path-bearing fixture — reuse the canonical scanners so the CLI
    # is exactly as strict as qpu_export_schema (red-team #2: AKIA/PEM/JWT/gho_/single-segment
    # paths were previously missed).
    from acrpq.dashboard import qpu_export_schema as _X
    blob = json.dumps(fx)
    if _X._PATH_RE.search(blob) or _X._SECRET_RE.search(blob):
        print("FIXTURE REFUSED: contains a local path or secret", file=sys.stderr)
        return 3
    if not isinstance(fx, dict) or "counts" not in fx or "n_bits" not in fx:
        print("FIXTURE INVALID: expected {counts, bit_order, kind, n_bits}", file=sys.stderr)
        return 3
    n_bits = fx["n_bits"]
    try:
        # build a FAKE gateway from the fixture and reconcile through get_result -> normalise
        gw = qpu_batch.FakeBatchGateway(fx["counts"], bit_order=fx.get("bit_order", "qubo"),
                                        n_bits=int(n_bits))
        job = gw.get_job("offline-fake-job-0")
        envelope = gw.get_result(job)
        norm = qpu_batch.normalize_result(envelope, expected_n_bits=int(n_bits))
    except Exception as exc:  # noqa: BLE001 — any adapter/gateway error fails closed
        print(f"FIXTURE INVALID: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    run_state = None
    if args.runs_dir and args.local_run_id:
        from acrpq.dashboard import qpu_runs
        try:
            run_state = qpu_runs.load(args.local_run_id, root=args.runs_dir).get("state")
        except Exception as exc:  # noqa: BLE001
            print(f"RUN NOT LOADABLE: {type(exc).__name__}", file=sys.stderr)
            return 5
    _emit({"command": "reconcile-simulated", "reconciled": True,
           "gateway": "FakeBatchGateway (no IBM, no network)",
           "n_shots": norm.get("n_shots"), "kind": norm.get("kind"),
           "persisted_run_state": run_state,
           "note": "reconciled from a recorded fixture through the real normaliser",
           "status": "OFFLINE"}, args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="qpu-batch", description="Offline QPU-batch operator CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--instances", nargs="+", default=["CP_3", "CP_4", "CP_5"])
    p.add_argument("--objectives", nargs="+", default=["maneuver_count_v1"])
    p.add_argument("--n-theta", dest="n_theta", type=int, default=3)
    p.add_argument("--shots", type=int, default=512)
    p.add_argument("--qpu-seconds-per-job", dest="qpu_seconds_per_job", type=float, default=10.0)
    p.add_argument("--backend", default="FakeGuadalupeV2")
    p.set_defaults(func=cmd_plan)

    for name, fn in (("inspect", cmd_inspect), ("resume-check", cmd_resume_check)):
        q = sub.add_parser(name)
        q.add_argument("--runs-dir", required=True)
        if name == "resume-check":
            q.add_argument("--batch-config-hash", dest="batch_config_hash", default=None)
        q.set_defaults(func=fn)

    v = sub.add_parser("verify")
    v.add_argument("artifact")
    v.set_defaults(func=cmd_verify)

    e = sub.add_parser("export")
    e.add_argument("artifact")
    e.add_argument("--out", default=None)
    e.set_defaults(func=cmd_export)

    r = sub.add_parser("reconcile-simulated")
    r.add_argument("--fixture", required=True)
    r.add_argument("--runs-dir", dest="runs_dir", default=None)
    r.add_argument("--local-run-id", dest="local_run_id", default=None)
    r.set_defaults(func=cmd_reconcile_simulated)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Hard refusal of any real-submission flag, BEFORE argparse (incl. --flag=value forms).
    for tok in argv:
        if tok.split("=", 1)[0] in _FORBIDDEN:
            print(f"REFUSED: {tok} is not available (offline CLI, no real submission)",
                  file=sys.stderr)
            return 2
    # Fail-closed: refuse to run at all if the environment could permit real submission.
    try:
        qpu_batch.assert_offline()
    except Exception as exc:  # noqa: BLE001
        print(f"REFUSING (not offline): {exc}", file=sys.stderr)
        return 4
    # --json is accepted anywhere (before OR after the subcommand); handle it manually so
    # a parent/subparser default can't clobber it.
    as_json = "--json" in argv
    argv = [t for t in argv if t != "--json"]
    args = build_parser().parse_args(argv)
    args.json = as_json
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
