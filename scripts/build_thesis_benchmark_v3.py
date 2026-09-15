#!/usr/bin/env python3
"""Merge the frozen local v2 benchmark and real-IBM campaign into thesis v3."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "acrpq-thesis-benchmark/3"
LOCAL = Path("results/thesis_benchmark_v2/benchmark_results.json")
HARDWARE = Path("results/ibm_fixed_angle_campaign_v1/campaign_results.json")
OUT = Path("results/thesis_benchmark_v3")


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _hardware_row(row: dict[str, Any]) -> dict[str, Any]:
    feasible = row["has_feasible_sample"]
    best = row["objective_value"]
    return {
        "schema": row["schema"], "protocol_id": row["protocol_id"],
        "config_hash": row["config_hash"], "source_commit": None,
        "job_key": row["job_key"], "result_sha256": row["export_sha256"],
        "instance": row["instance"], "family": "CP", "n_aircraft": row["n_aircraft"],
        "grid_k": 3, "n_theta": 3, "n_q": 1, "n_qubits": row["n_qubits"],
        "objective_id": row["objective_id"], "method": "qaoa_ibm_fixed_angles",
        "seed": row["repetition"], "status": "completed", "result_class": "incumbent",
        "objective_value": best, "feasible": feasible,
        "n_conflicts": None, "decoded_feasible": feasible, "decoded_conflicts": None,
        "raw_onehot_valid": None, "repair_applied": False, "qubo_energy": None,
        "exhaustive_optimum": row["certified_discrete_reference"],
        "exhaustive_status": "optimal", "certified_milp_optimum": None,
        "milp_certified": None,
        "gap_to_exhaustive_optimum": row["gap_to_discrete_reference"],
        "gap_to_certified_milp": None,
        "matches_exhaustive_optimum": row["matches_discrete_reference"],
        "comparison_tolerance": 1e-9, "wall_time_s": row["submit_to_complete_s"],
        "cpu_time_s": None, "provider_time_s": None, "num_reads": row["shots"],
        "instance_sha256": None, "grid_identity_sha256": None, "is_qpu": True,
        "data_source": "ibm_hardware", "domain": "grid_K3",
        "official_comparison": feasible,
        "backend": row["backend"], "ibm_job_id": row["ibm_job_id"],
        "repetition": row["repetition"], "shots": row["shots"],
        "gamma": row["gamma"], "beta": row["beta"],
        "isa_depth": row["isa_depth"],
        "isa_two_qubit_gates": row["isa_two_qubit_gates"],
        "feasibility_rate": row["feasibility_rate"],
        "best_feasible_probability": row["best_feasible_probability"],
        "n_unique_bitstrings": row["n_unique_bitstrings"],
    }


def build_rows() -> list[dict[str, Any]]:
    local = json.loads(LOCAL.read_text())["rows"]
    hardware = json.loads(HARDWARE.read_text())["rows"]
    rows = [dict(row) for row in local] + [_hardware_row(row) for row in hardware]
    rows.sort(key=lambda row: (row["data_source"], row["job_key"]))
    if len(rows) != 1736 or len({(row["data_source"], row["job_key"]) for row in rows}) != len(rows):
        raise ValueError("expected 1,736 unique source/job rows")
    for row in rows:
        if not row["feasible"] and (row.get("gap_to_exhaustive_optimum") is not None
                                    or row.get("gap_to_certified_milp") is not None):
            raise ValueError(f"infeasible row carries a gap: {row['job_key']}")
    json.dumps(rows, allow_nan=False)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), allow_nan=False) for key in columns})


def write(out: Path = OUT) -> dict[str, Any]:
    rows = build_rows()
    document = {"schema": SCHEMA, "rows": rows}
    payload = json.dumps(document, indent=2, allow_nan=False).encode() + b"\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_results.json").write_bytes(payload)
    _write_csv(out / "benchmark_results.csv", rows)
    manifest = {
        "schema": SCHEMA, "n_rows": len(rows), "n_hardware_rows": 30,
        "n_local_and_aer_rows": len(rows) - 30,
        "local_source_sha256": _sha(LOCAL.read_bytes()),
        "hardware_source_sha256": _sha(HARDWARE.read_bytes()),
        "dataset_sha256": _sha(payload),
        "claims": {"quantum_speedup": False, "objectives_aggregated_together": False,
                   "shots_are_independent_runs": False},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify(out)
    return manifest


def verify(out: Path = OUT) -> None:
    manifest = json.loads((out / "manifest.json").read_text())
    payload = (out / "benchmark_results.json").read_bytes()
    document = json.loads(payload)
    if manifest["dataset_sha256"] != _sha(payload) or document["rows"] != build_rows():
        raise ValueError("v3 JSON/source mismatch")
    with (out / "benchmark_results.csv").open(newline="") as fh:
        actual = [{key: json.loads(value) for key, value in row.items()}
                  for row in csv.DictReader(fh)]
    columns = list(actual[0])
    expected = [{key: row.get(key) for key in columns} for row in document["rows"]]
    if actual != expected:
        raise ValueError("v3 CSV/JSON mismatch")


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
