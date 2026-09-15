#!/usr/bin/env python3
"""Consolide la campagne IBM finale en un dataset canonique, une ligne par job.

Sources, jamais réécrites : ``results/ibm_qaoa_final_campaign_v1/campaign_ledger.json``
(brut) et ``sidecar/*.json`` (métadonnées de référence corrigées, liées par
``raw_sha256``). Ce script ne fait que LIRE et PROJETER — toute correction scientifique a
déjà eu lieu en amont (scripts/build_campaign_sidecar.py).

Séparations strictes, jamais de pooling silencieux :

* K3 et K5 restent des colonnes (``grid_k``), jamais agrégées ensemble ;
* les deux ``objective_id`` restent des colonnes, jamais moyennées ;
* le bras (``A0_P0_replication`` / ``A1_P1_feas_angles`` / ``A2_P0_transpile_L2`` /
  ``CE_endianness_canary``) reste une colonne ;
* un job canari (``official_hypothesis_test=false``) n'entre dans aucune statistique
  d'hypothèse — il reste visible mais marqué.

Écrit ``results/ibm_qaoa_final_campaign_v1/campaign_results.json`` (provenance complète)
et ``campaign_results.csv`` (plat), plus ``campaign_manifest.json`` et
``integrity_report.json``. Aucun écart n'est calculé pour une solution infaisable.

Usage : PYTHONPATH=src python scripts/build_final_campaign_dataset.py [--verify]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path("results/ibm_qaoa_final_campaign_v1")
LEDGER = ROOT / "campaign_ledger.json"
SIDECAR_DIR = ROOT / "sidecar"
PREREG = ROOT / "preregistration.json"

COLUMNS = (
    "experiment_id", "protocol_id", "protocol_sha256", "configuration_hash",
    "instance", "family", "objective_id", "grid_k", "n_theta", "n_q",
    "logical_qubits", "physical_qubits", "qubo_sha256",
    "certified_reference_objective", "certified_reference_optima_count",
    "angle_source", "gamma", "beta", "qaoa_reps",
    "backend_requested", "backend_effective", "job_id", "repetition_id",
    "arm", "block_id", "execution_order", "is_canary", "official_hypothesis_test",
    "shots_requested", "shots_received",
    "transpiler_seed", "optimization_level",
    "isa_depth_before", "isa_depth_after", "isa_two_qubit_gates", "isa_size_after",
    "isa_swap_count",
    "raw_onehot_valid_count", "geometry_feasible_count", "n_unique_bitstrings",
    "feasible_shot_rate_strict",
    "best_raw_bitstring", "best_raw_energy",
    "best_feasible_bitstring", "best_feasible_objective",
    "modal_bitstring", "modal_probability",
    "probability_mass_on_best_feasible",
    "gap_to_certified_reference", "gap_suppressed_reason", "certified_optimum_hit",
    "billed_qpu_seconds", "billed_qpu_seconds_incremental_estimate_do_not_trust", "prepared_utc", "submitted_utc",
    "source_commit", "scientific_code_dirty", "repository_dirty",
    "raw_sha256", "artefact_sha256", "state",
)


def _sha(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _family(instance: str) -> str:
    return instance.split("_")[0]


def _row(job_key: str, record: dict[str, Any], sidecar: dict[str, Any],
        protocol: dict[str, Any]) -> dict[str, Any]:
    art = ((record.get("export") or {}).get("artefact") or {})
    req = art.get("request") or {}
    tr = art.get("transpilation") or {}
    summary = ((record.get("result") or {}).get("summary") or {})
    best_raw = summary.get("best_raw") or {}
    best_feasible = summary.get("best_feasible") or {}
    modal = summary.get("modal") or {}
    res = record.get("resolved") or {}
    ref = sidecar.get("corrected_reference") or {}
    optimum = sidecar.get("corrected_optimum") or {}
    counts = ((record.get("export") or {}).get("decoded") or {}).get("counts") or {}
    onehot_valid = None                    # dérivable seulement si décodage réexposé ;
    geometry_feasible = None               # laissé null plutôt qu'estimé pour ne pas inventer

    return {
        "experiment_id": job_key,
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": protocol.get("protocol_sha256"),
        "configuration_hash": record.get("config_hash"),
        "instance": record.get("instance"), "family": _family(record.get("instance", "")),
        "objective_id": record.get("objective_id"), "grid_k": record.get("grid_k"),
        "n_theta": req.get("n_theta"), "n_q": req.get("n_q"),
        "logical_qubits": (art.get("logical_circuit") or {}).get("n_qubits")
                          or res.get("expected_n_qubits"),
        "physical_qubits": (art.get("backend_synthetic") or {}).get("num_qubits"),
        "qubo_sha256": art.get("qubo_sha256") or res.get("expected_qubo_sha256"),
        "certified_reference_objective": ref.get("reference_objective"),
        "certified_reference_optima_count": ref.get("reference_optima_count"),
        "angle_source": res.get("angle_set"),
        "gamma": (res.get("gammas") or [None])[0], "beta": (res.get("betas") or [None])[0],
        "qaoa_reps": 1,
        "backend_requested": req.get("backend_requested"), "backend_effective": art.get("backend"),
        "job_id": (record.get("ibm_job_ids") or [None])[0], "repetition_id": record.get("repetition_id"),
        "arm": record.get("arm"), "block_id": record.get("block_id"),
        "execution_order": record.get("execution_order"),
        "is_canary": sidecar.get("classification", {}).get("technical_canary"),
        "official_hypothesis_test": sidecar.get("classification", {}).get("official_hypothesis_test"),
        "shots_requested": req.get("shots"), "shots_received": summary.get("n_shots"),
        "transpiler_seed": req.get("transpiler_seed"),
        "optimization_level": tr.get("optimization_level") or res.get("optimization_level"),
        "isa_depth_before": tr.get("depth_before"), "isa_depth_after": tr.get("depth_after"),
        "isa_two_qubit_gates": tr.get("two_qubit_gates_after"), "isa_size_after": tr.get("size_after"),
        "isa_swap_count": tr.get("swap_count"),
        "raw_onehot_valid_count": onehot_valid, "geometry_feasible_count": geometry_feasible,
        "n_unique_bitstrings": summary.get("n_unique_total") or len(counts),
        "feasible_shot_rate_strict": summary.get("feasibility_rate"),
        "best_raw_bitstring": best_raw.get("bitstring"), "best_raw_energy": best_raw.get("energy"),
        "best_feasible_bitstring": best_feasible.get("bitstring"),
        "best_feasible_objective": optimum.get("best_feasible_objective"),
        "modal_bitstring": modal.get("bitstring"), "modal_probability": modal.get("probability"),
        "probability_mass_on_best_feasible": optimum.get("probability_mass_on_best_feasible"),
        "gap_to_certified_reference": optimum.get("gap_to_certified_reference"),
        "gap_suppressed_reason": optimum.get("gap_suppressed_reason"),
        "certified_optimum_hit": optimum.get("hits_certified_optimum"),
        "billed_qpu_seconds": sidecar.get("measured", {}).get(
            "billed_seconds_authoritative_provider_metrics"),
        "billed_qpu_seconds_incremental_estimate_do_not_trust":
            sidecar.get("measured", {}).get("billed_seconds_incremental_runner_estimate"),
        "prepared_utc": record.get("prepared_utc"), "submitted_utc": record.get("submitted_utc"),
        "source_commit": art.get("provenance", {}).get("source_commit")
                         if isinstance(art.get("provenance"), dict) else None,
        "scientific_code_dirty": None, "repository_dirty": None,
        "raw_sha256": sidecar.get("raw_sha256"), "artefact_sha256": record.get("artefact_sha256"),
        "state": record.get("state"),
    }


def build() -> dict[str, Any]:
    ledger = json.loads(LEDGER.read_text())
    protocol = json.loads(PREREG.read_text())["protocol"]
    rows: list[dict[str, Any]] = []
    missing_sidecar: list[str] = []

    for job_key in sorted(ledger["jobs"]):
        side_path = SIDECAR_DIR / f"{job_key}.json"
        if not side_path.exists():
            missing_sidecar.append(job_key)
            continue
        sidecar = json.loads(side_path.read_text())
        rows.append(_row(job_key, ledger["jobs"][job_key], sidecar, protocol))

    counts_by = {
        "grid_k": sorted({(r["grid_k"], sum(1 for x in rows if x["grid_k"] == r["grid_k"]))
                          for r in rows}),
        "objective_id": sorted({r["objective_id"] for r in rows}),
        "arm": sorted({r["arm"] for r in rows}),
        "state": {s: sum(1 for r in rows if r["state"] == s)
                 for s in sorted({r["state"] for r in rows})},
    }
    manifest = {
        "schema": "acrpq-ibm-final-campaign-dataset/1",
        "protocol_sha256": protocol["protocol_sha256"],
        "n_rows": len(rows), "missing_sidecar": missing_sidecar,
        "separations": {
            "grids_pooled": False, "objectives_pooled": False, "arms_pooled": False,
            "canaries_excluded_from_hypothesis_stats": True,
        },
        "counts_by": counts_by,
        "columns": list(COLUMNS),
    }
    manifest["manifest_sha256"] = _sha({k: v for k, v in manifest.items()
                                        if k != "manifest_sha256"})
    return {"rows": rows, "manifest": manifest}


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLUMNS})


def _assert_json_csv_parity(rows: list[dict[str, Any]], csv_path: Path) -> None:
    with csv_path.open() as fh:
        csv_rows = list(csv.DictReader(fh))
    if len(csv_rows) != len(rows):
        raise SystemExit(f"parité JSON/CSV rompue : {len(rows)} vs {len(csv_rows)} lignes")
    for j, c in zip(rows, csv_rows):
        for col in COLUMNS:
            jv = j.get(col)
            cv = c.get(col) or None
            if jv is None:
                if cv not in (None, ""):
                    raise SystemExit(f"parité rompue {j['experiment_id']}.{col}: json=None csv={cv!r}")
            elif str(jv) != cv:
                raise SystemExit(f"parité rompue {j['experiment_id']}.{col}: json={jv!r} csv={cv!r}")


def _verify() -> int:
    p = ROOT / "campaign_manifest.json"
    if not p.exists():
        print("MISSING campaign_manifest.json")
        return 2
    manifest = json.loads(p.read_text())
    recon = _sha({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    if recon != manifest["manifest_sha256"]:
        print("MANIFEST HASH MISMATCH")
        return 2
    rows = json.loads((ROOT / "campaign_results.json").read_text())["rows"]
    _assert_json_csv_parity(rows, ROOT / "campaign_results.csv")
    print(f"OK campaign dataset verified ({manifest['n_rows']} rows, parité JSON/CSV ok)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Construire le dataset canonique de campagne")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        return _verify()
    doc = build()
    (ROOT / "campaign_results.json").write_text(
        json.dumps({"rows": doc["rows"]}, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _write_csv(doc["rows"], ROOT / "campaign_results.csv")
    _assert_json_csv_parity(doc["rows"], ROOT / "campaign_results.csv")
    (ROOT / "campaign_manifest.json").write_text(
        json.dumps(doc["manifest"], indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"[dataset] {doc['manifest']['n_rows']} lignes "
          f"({len(doc['manifest']['missing_sidecar'])} sidecars manquants)")
    print(json.dumps(doc["manifest"]["counts_by"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
