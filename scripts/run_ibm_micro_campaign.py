#!/usr/bin/env python3
"""Run/resume a bounded real-IBM fixed-angle sampling campaign via the local API.

This script never accepts an IBM token.  Credentials remain in Qiskit's private
account store and every submission still crosses the dashboard's prepare,
single-use confirmation and point-of-use hardware authorization gates.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SCHEMA = "acrpq-ibm-fixed-angle-campaign/1"
ACK = "I_AUTHORIZE_REAL_IBM_QPU_SUBMISSIONS"
OBJECTIVES = ("quadratic_control_cost_v1", "maneuver_count_v1")


def _request(base_url: str, method: str, path: str, body: dict | None = None) -> dict:
    url = base_url.rstrip("/") + path
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("campaign API must be loopback")
    data = None if body is None else json.dumps(body, allow_nan=False).encode()
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json", "Origin": base_url.rstrip("/")},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read(1000).decode(errors="replace")
        raise RuntimeError(f"local API {method} {path} failed ({exc.code}): {detail}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"local API {path} returned a non-object")
    return result


def _atomic_write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temp.open("w") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temp, path)
    try:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _new_manifest(args: argparse.Namespace) -> dict:
    return {
        "schema": SCHEMA,
        "protocol": {
            "backend": args.backend, "instances": args.instances,
            "objectives": list(OBJECTIVES), "repetitions": args.repetitions,
            "n_theta": 3, "n_q": 1, "shots": args.shots,
            "gammas": [args.gamma], "betas": [args.beta],
            "transpiler_seed": args.transpiler_seed, "optimization_level": 0,
            "max_quantum_seconds_per_job": args.max_quantum_seconds,
            "interpretation": "fixed-angle QAOA sampling; no hardware-side optimization",
        },
        "jobs": [],
    }


def _load_or_create(path: Path, args: argparse.Namespace) -> dict:
    expected = _new_manifest(args)
    if not path.exists():
        return expected
    document = json.loads(path.read_text())
    if document.get("schema") != SCHEMA or document.get("protocol") != expected["protocol"]:
        raise ValueError("existing campaign manifest uses a different protocol")
    if not isinstance(document.get("jobs"), list):
        raise ValueError("campaign jobs must be a list")
    return document


def _job_key(instance: str, objective: str, repetition: int) -> str:
    return f"{instance}__K3__{objective}__fixed_p1__r{repetition}"


def _body(args: argparse.Namespace, instance: str, objective: str, snapshot: dict) -> dict:
    return {
        "instance": instance, "n_theta": 3, "n_q": 1,
        "objective_id": objective, "shots": args.shots,
        "transpiler_seed": args.transpiler_seed,
        "backend_requested": args.backend,
        "gammas": [args.gamma], "betas": [args.beta],
        "max_jobs": 1, "max_quantum_seconds": args.max_quantum_seconds,
        "optimization_level": 0, "snapshots": [snapshot],
    }


def _confirm_body(prepared: dict) -> dict:
    challenge = prepared["confirmation"]
    return {
        "nonce": challenge["nonce"], "config_hash": prepared["config_hash"],
        "artefact_sha256": prepared["artefact_sha256"],
        "decision_sha256": prepared["decision_sha256"], "backend": prepared["backend"],
        "total_shots": prepared["budget"]["total_shots"],
        "max_jobs": prepared["budget"]["max_jobs"],
        "max_quantum_seconds": prepared["budget"]["max_quantum_seconds"],
        "consent": challenge["consent_phrase"],
    }


def _refresh(base_url: str, job: dict) -> None:
    run_id = job["run_id"]
    run = _request(base_url, "GET", f"/api/qpu/{run_id}")
    state = run["state"]
    if state not in {"completed", "failed", "cancelled"} and run.get("ibm_job_ids"):
        reconciled = _request(base_url, "POST", f"/api/qpu/{run_id}/reconcile", {})
        state = reconciled["state"]
    job["state"] = state
    job["ibm_job_ids"] = run.get("ibm_job_ids", job.get("ibm_job_ids", []))
    if state == "completed":
        result = _request(base_url, "GET", f"/api/qpu/{run_id}/result")
        job["result"] = result if result.get("available") else None
        if result.get("available"):
            job["export"] = _request(base_url, "POST", f"/api/qpu/{run_id}/export", {})


def run(args: argparse.Namespace) -> dict:
    if not args.execute or args.ack != ACK:
        raise ValueError(f"real execution requires --execute --ack {ACK}")
    out = Path(args.out)
    document = _load_or_create(out, args)
    flags = _request(args.base_url, "GET", "/api/qpu/flags")
    if not (flags.get("submission_allowed") and flags.get("runtime_factory_enabled")):
        raise RuntimeError("local server does not currently allow real IBM submission")
    existing = {job["job_key"]: job for job in document["jobs"]}

    # Reconcile persisted runs before creating anything new.
    for job in document["jobs"]:
        needs_refresh = (
            job.get("state") not in {"completed", "failed", "cancelled"}
            or (job.get("state") == "completed" and not job.get("result"))
        )
        if job.get("run_id") and needs_refresh:
            try:
                _refresh(args.base_url, job)
                job.pop("last_reconciliation_error", None)
            except Exception as exc:  # remote read failure: preserve and continue, never resubmit
                job["last_reconciliation_error"] = type(exc).__name__
                _atomic_write(out, document)
                continue
            _atomic_write(out, document)
            if job["state"] == "awaiting_confirmation" and not job.get("submitted"):
                submitted = _request(
                    args.base_url, "POST", f"/api/qpu/{job['run_id']}/submit", {})
                job["submitted"] = True
                job["state"] = submitted["state"]
                job["ibm_job_ids"] = submitted.get("job_ids", [])
                _atomic_write(out, document)

    planned = [
        (instance, objective, repetition)
        for repetition in range(1, args.repetitions + 1)
        for instance in args.instances
        for objective in OBJECTIVES
    ]
    needs_new_jobs = (
        len(document["jobs"]) < args.max_total_jobs
        and any(_job_key(*item) not in existing for item in planned)
    )
    live = None
    if needs_new_jobs:
        live = _request(
            args.base_url, "GET", f"/api/qpu/backends/live/{args.backend}")["backend"]
    for index, (instance, objective, repetition) in enumerate(planned):
        key = _job_key(instance, objective, repetition)
        job = existing.get(key)
        if job is not None:
            continue
        if len(document["jobs"]) >= args.max_total_jobs:
            break
        if live is None:
            raise RuntimeError("live backend snapshot unavailable for a new job")
        # Canaries are the first two objective-specific CP_3 runs.  Do not move
        # to the rest of the campaign until both have a retrieved result.
        if index >= 2:
            canaries = [existing.get(_job_key("CP_3", oid, 1)) for oid in OBJECTIVES]
            if any(not c or not c.get("result", {}).get("available") for c in canaries):
                break
        prepared = _request(args.base_url, "POST", "/api/qpu/prepare",
                            _body(args, instance, objective, live))
        job = {
            "job_key": key, "instance": instance, "objective_id": objective,
            "repetition": repetition, "run_id": prepared["run_id"],
            "state": prepared["state"], "submitted": False,
            "artefact_sha256": prepared["artefact_sha256"],
        }
        document["jobs"].append(job)
        existing[key] = job
        _atomic_write(out, document)
        _request(args.base_url, "POST", f"/api/qpu/{prepared['run_id']}/confirm",
                 _confirm_body(prepared))
        submitted = _request(
            args.base_url, "POST", f"/api/qpu/{prepared['run_id']}/submit", {})
        job["submitted"] = True
        job["state"] = submitted["state"]
        job["ibm_job_ids"] = submitted.get("job_ids", [])
        _atomic_write(out, document)
        if index < 2:
            deadline = time.monotonic() + args.canary_wait_seconds
            while time.monotonic() < deadline:
                _refresh(args.base_url, job)
                _atomic_write(out, document)
                if job["state"] in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(args.poll_seconds)
            if not job.get("result", {}).get("available"):
                break
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8055")
    parser.add_argument("--out", default="results/ibm_fixed_angle_campaign_v1/manifest.json")
    parser.add_argument("--backend", default="ibm_marrakesh")
    parser.add_argument("--instances", nargs="+", default=[f"CP_{n}" for n in range(3, 8)])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--transpiler-seed", type=int, default=1)
    parser.add_argument("--max-quantum-seconds", type=float, default=20.0)
    parser.add_argument("--max-total-jobs", type=int, default=30)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--canary-wait-seconds", type=float, default=3600.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ack", default="")
    args = parser.parse_args()
    if args.repetitions < 1 or args.max_total_jobs < 1 or args.shots < 1:
        raise ValueError("repetitions, max-total-jobs and shots must be positive")
    document = run(args)
    print(json.dumps({"jobs": len(document["jobs"]),
                      "states": [j["state"] for j in document["jobs"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
