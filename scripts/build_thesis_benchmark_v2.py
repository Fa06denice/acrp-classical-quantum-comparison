#!/usr/bin/env python3
"""Build/verify the canonical thesis-v2 dataset from immutable source artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

SCHEMA = "acrpq-thesis-benchmark/2"
SOURCES = {
    "cp": Path("results/discrete_benchmark_cp_v2/benchmark.json"),
    "multifamily": Path("results/discrete_benchmark_multifamily_v2/benchmark.json"),
    "continuous_count": Path("results/ampl_gurobi_maneuver_count_v1/manifest.json"),
}
OUT = Path("results/thesis_benchmark_v2")


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _family(instance: str) -> str:
    return "RCP_FL" if instance.startswith("RCP_FL") else instance.split("_", 1)[0]


def _cp_size(instance: str) -> int:
    prefix, size = instance.split("_", 1)
    if prefix != "CP" or not size.isdigit():
        raise ValueError(f"unexpected continuous-count instance: {instance}")
    return int(size)


def _load() -> tuple[list[dict], dict[str, str]]:
    rows: list[dict] = []
    hashes = {name: _sha(path.read_bytes()) for name, path in SOURCES.items()}
    for key in ("cp", "multifamily"):
        document = json.loads(SOURCES[key].read_text())
        for source in document["results"]:
            row = dict(source)
            row.update({
                "data_source": "aer_simulator" if source["method"] == "qaoa_aer"
                else "local_classical",
                "domain": f"grid_K{source['grid_k']}",
                "official_comparison": bool(
                    source["feasible"]
                    and (source["exhaustive_optimum"] is not None
                         or source["milp_certified"])
                ),
            })
            rows.append(row)

    count_dir = SOURCES["continuous_count"].parent
    manifest = json.loads(SOURCES["continuous_count"].read_text())
    for instance in manifest["instances"]:
        source = json.loads((count_dir / f"{instance}.json").read_text())
        rows.append({
            "schema": source["schema"], "protocol_id": source["protocol_id"],
            "config_hash": None, "source_commit": source["source_provenance"]["source_commit"],
            "job_key": f"{instance}__continuous__maneuver_count_v1__original_ampl_count__det",
            "result_sha256": source["record_sha256"], "instance": instance,
            "family": _family(instance), "n_aircraft": _cp_size(instance),
            "grid_k": None, "n_theta": None, "n_q": None, "n_qubits": None,
            "objective_id": "maneuver_count_v1", "method": "original_ampl_count",
            "seed": None, "status": source["status"],
            "result_class": "certified_optimum" if source["certified_integer_optimum"]
            else "noncertified",
            "objective_value": source["objective"],
            "feasible": source["feasible_common_geometry"],
            "n_conflicts": source["n_residual_conflicts"],
            "decoded_feasible": source["feasible_common_geometry"],
            "decoded_conflicts": source["n_residual_conflicts"],
            "raw_onehot_valid": None, "repair_applied": None, "qubo_energy": None,
            "exhaustive_optimum": None, "exhaustive_status": None,
            "certified_milp_optimum": None, "milp_certified": None,
            "gap_to_exhaustive_optimum": None, "gap_to_certified_milp": None,
            "matches_exhaustive_optimum": None, "comparison_tolerance": 0.0,
            "wall_time_s": source["wall_time_s"], "cpu_time_s": None,
            "provider_time_s": source["solver_time_s"], "num_reads": None,
            "instance_sha256": source["provenance"].get("data_sha256"),
            "grid_identity_sha256": None, "is_qpu": False,
            "data_source": "original_ampl_gurobi",
            "domain": "continuous_controls_with_binary_moved_indicators",
            "official_comparison": source["certified_integer_optimum"],
            "historical_quadratic_objective": source["historical_quadratic_objective"],
            "discrete_K3_objective": source["discrete_K3"]["objective"],
            "count_discretization_difference": (
                source["comparison"]["discretization_difference_count"]
                if source["comparison"] else None
            ),
        })
    return rows, hashes


def _validate(rows: list[dict]) -> None:
    keys = [r["job_key"] for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate canonical job_key")
    for row in rows:
        json.dumps(row, allow_nan=False)
        if row["objective_id"] not in {"quadratic_control_cost_v1", "maneuver_count_v1"}:
            raise ValueError("unknown objective")
        if not row["feasible"] and (
            row.get("gap_to_exhaustive_optimum") is not None
            or row.get("gap_to_certified_milp") is not None
        ):
            raise ValueError(f"infeasible row carries a gap: {row['job_key']}")
        for value in row.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite value: {row['job_key']}")


def _synthesis(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["objective_id"], row["method"])].append(row)
    out = []
    for (family, objective, method), group in sorted(groups.items()):
        feasible = [r for r in group if r["feasible"]]
        gaps = [r["gap_to_exhaustive_optimum"] for r in feasible
                if r.get("gap_to_exhaustive_optimum") is not None]
        matches = [r["matches_exhaustive_optimum"] for r in feasible
                   if r.get("matches_exhaustive_optimum") is not None]
        objectives = [r["objective_value"] for r in feasible if r["objective_value"] is not None]
        out.append({
            "family": family, "objective_id": objective, "method": method,
            "n_rows": len(group), "n_feasible": len(feasible),
            "feasibility_rate": len(feasible) / len(group),
            "n_reference_comparisons": len(matches),
            "reference_match_rate": sum(bool(v) for v in matches) / len(matches) if matches else None,
            "gap_mean": mean(gaps) if gaps else None,
            "gap_median": median(gaps) if gaps else None,
            "objective_mean_feasible": mean(objectives) if objectives else None,
        })
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), sort_keys=True, allow_nan=False)
                             for key in columns})


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        columns = reader.fieldnames
        return columns, [
            {key: json.loads(value) for key, value in row.items()}
            for row in reader
        ]


def _assert_csv_parity(path: Path, rows: list[dict]) -> None:
    columns, actual = _read_csv(path)
    expected_columns = sorted({key for row in rows for key in row})
    if columns != expected_columns:
        raise ValueError(f"CSV columns differ from JSON: {path}")
    expected = [{key: row.get(key) for key in columns} for row in rows]
    if actual != expected:
        raise ValueError(f"CSV rows differ from JSON: {path}")


def build(out: Path = OUT) -> dict:
    rows, source_hashes = _load()
    rows.sort(key=lambda r: r["job_key"])
    _validate(rows)
    synthesis = _synthesis(rows)
    document = {"schema": SCHEMA, "rows": rows}
    document_hash = _sha(_bytes(document))
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_results.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    _write_csv(out / "benchmark_results.csv", rows)
    _write_csv(out / "synthesis.csv", synthesis)
    manifest = {
        "schema": SCHEMA, "n_rows": len(rows), "n_discrete_rows": len(rows) - 6,
        "n_continuous_count_rows": 6, "source_hashes": source_hashes,
        "dataset_sha256": document_hash,
        "claims": {
            "hardware_rows": 0, "ibm_contacted": False,
            "objectives_aggregated_together": False,
            "timings_establish_speedup": False,
        },
    }
    manifest["manifest_sha256"] = _sha(_bytes(manifest))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(out: Path = OUT) -> None:
    manifest = json.loads((out / "manifest.json").read_text())
    document = json.loads((out / "benchmark_results.json").read_text())
    _validate(document["rows"])
    if manifest["dataset_sha256"] != _sha(_bytes(document)):
        raise ValueError("dataset hash mismatch")
    own = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest["manifest_sha256"] != _sha(_bytes(own)):
        raise ValueError("manifest hash mismatch")
    current = {name: _sha(path.read_bytes()) for name, path in SOURCES.items()}
    if current != manifest["source_hashes"]:
        raise ValueError("source artifact hash mismatch")
    _assert_csv_parity(out / "benchmark_results.csv", document["rows"])
    expected_synthesis = _synthesis(document["rows"])
    _assert_csv_parity(out / "synthesis.csv", expected_synthesis)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify(Path(args.out))
        print("OK")
    else:
        print(build(Path(args.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
