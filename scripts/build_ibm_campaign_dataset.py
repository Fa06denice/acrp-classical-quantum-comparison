#!/usr/bin/env python3
"""Build and verify the canonical CSV/JSON summary of the real IBM campaign."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = "acrpq-ibm-fixed-angle-dataset/1"
SOURCE = Path("results/ibm_fixed_angle_campaign_v1/manifest.json")
REFERENCE = Path("results/thesis_benchmark_v2/benchmark_results.json")
OUT = Path("results/ibm_fixed_angle_campaign_v1")


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def _references() -> dict[tuple[str, str], float]:
    rows = json.loads(REFERENCE.read_text())["rows"]
    refs = {}
    for row in rows:
        if row["grid_k"] == 3 and row["method"] == "reference" \
                and row["result_class"] == "certified_optimum":
            refs[(row["instance"], row["objective_id"])] = row["objective_value"]
    return refs


def build() -> list[dict[str, Any]]:
    campaign = json.loads(SOURCE.read_text())
    refs = _references()
    rows = []
    for job in campaign["jobs"]:
        if job.get("state") != "completed" or not job.get("result", {}).get("available"):
            raise ValueError(f"incomplete campaign row: {job.get('job_key')}")
        result = job["result"]
        summary = result["summary"]
        export = job["export"]
        request = export["artefact"]["request"]
        transpilation = export["artefact"]["transpilation"]
        transitions = {x["state"]: x["stamp_utc"] for x in export["run"]["transitions"]}
        best = summary.get("best_feasible")
        modal = summary.get("modal")
        reference = refs[(job["instance"], job["objective_id"])]
        objective = None if best is None else best["objective"]
        gap = None if objective is None else objective - reference
        rows.append({
            "schema": SCHEMA, "data_source": "ibm_hardware",
            "protocol_id": "ibm_marrakesh_fixed_angles_p1_v1",
            "config_hash": export["run"]["config_hash"],
            "qubo_sha256": export["plan"]["qubo_sha256"],
            "job_key": job["job_key"], "local_run_id": job["run_id"],
            "ibm_job_id": job["ibm_job_ids"][0], "backend": export["run"]["params"]["backend_requested"],
            "instance": job["instance"], "n_aircraft": int(job["instance"].split("_")[1]),
            "grid_k": 3, "n_qubits": request["n_binary_vars"],
            "objective_id": job["objective_id"], "repetition": job["repetition"],
            "shots": summary["n_shots"], "gamma": request["gammas"][0],
            "beta": request["betas"][0], "transpiler_seed": request["transpiler_seed"],
            "optimization_level": request["optimization_level"],
            "isa_depth": transpilation["depth_after"],
            "isa_two_qubit_gates": transpilation["two_qubit_gates_after"],
            "n_unique_bitstrings": summary["n_unique_total"],
            "feasibility_rate": summary["feasibility_rate"],
            "has_feasible_sample": best is not None,
            "best_feasible_bitstring": None if best is None else best["bitstring"],
            "best_feasible_probability": None if best is None else best["probability"],
            "objective_value": objective, "certified_discrete_reference": reference,
            "gap_to_discrete_reference": gap,
            "matches_discrete_reference": None if gap is None else math.isclose(gap, 0.0, abs_tol=1e-9),
            "modal_bitstring": None if modal is None else modal["bitstring"],
            "modal_feasible": None if modal is None else modal["feasible"],
            "mean_qubo_energy": summary["mean_energy"],
            "energy_dispersion": summary["energy_dispersion"],
            "submit_to_complete_s": _seconds(
                transitions.get("submitted"), transitions.get("completed")),
            "artefact_sha256": job["artefact_sha256"],
            "export_sha256": export["export_sha256"],
        })
    rows.sort(key=lambda row: row["job_key"])
    if len(rows) != 30 or len({row["job_key"] for row in rows}) != len(rows):
        raise ValueError("expected exactly 30 unique campaign rows")
    json.dumps(rows, allow_nan=False)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0])
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, allow_nan=False) for key, value in row.items()})


def write(out: Path = OUT) -> dict[str, Any]:
    rows = build()
    document = {"schema": SCHEMA, "rows": rows}
    payload = json.dumps(document, indent=2, allow_nan=False).encode() + b"\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "campaign_results.json").write_bytes(payload)
    _write_csv(out / "campaign_results.csv", rows)
    manifest = {
        "schema": SCHEMA, "n_rows": len(rows),
        "source_sha256": _sha(SOURCE.read_bytes()),
        "reference_sha256": _sha(REFERENCE.read_bytes()),
        "dataset_sha256": _sha(payload), "hardware_rows": len(rows),
        "claim": "fixed-angle QAOA sampling on real IBM hardware; no quantum speedup claim",
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify(out)
    return manifest


def verify(out: Path = OUT) -> None:
    manifest = json.loads((out / "dataset_manifest.json").read_text())
    payload = (out / "campaign_results.json").read_bytes()
    document = json.loads(payload)
    if manifest["dataset_sha256"] != _sha(payload) or document["rows"] != build():
        raise ValueError("IBM dataset JSON/source mismatch")
    with (out / "campaign_results.csv").open(newline="") as fh:
        actual = [{key: json.loads(value) for key, value in row.items()}
                  for row in csv.DictReader(fh)]
    if actual != document["rows"]:
        raise ValueError("IBM dataset CSV/JSON mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(Path(args.out))
        print("OK")
    else:
        print(json.dumps(write(Path(args.out))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
