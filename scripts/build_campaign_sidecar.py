#!/usr/bin/env python3
"""Fige les résultats matériels bruts et publie les corrections dans un SIDECAR.

Règle absolue : les counts, les job_id et les artefacts bruts ne sont JAMAIS réécrits.
Ce script

  * copie chaque enregistrement du registre vers ``raw/<job_key>.json``, une seule fois,
    et refuse d'écraser une copie existante dont le contenu diffère ;
  * publie dans ``sidecar/<job_key>.json`` la classification du job et les seules
    métadonnées corrigées (référence certifiée, nombre d'optima, masse sur l'optimum,
    écart), chaque sidecar étant lié à son brut par ``raw_sha256``.

La correction porte uniquement sur le SCORING des références : le paysage hors ligne
évaluait la référence avec ``geometry.objective()``, qui renvoie le coût quadratique quel
que soit l'``objective_id``. Aucun circuit soumis n'est affecté.

Usage : PYTHONPATH=src python scripts/build_campaign_sidecar.py [--verify]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

ROOT = Path("results/ibm_qaoa_final_campaign_v1")
LEDGER = ROOT / "campaign_ledger.json"
RAW = ROOT / "raw"
SIDE = ROOT / "sidecar"
CANARY_ARM = "CE_endianness_canary"


def _sha_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonical(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


_METRICS_CACHE: dict[str, float | None] = {}


def authoritative_billed_seconds(job_id: str) -> float | None:
    """Secondes QPU réellement facturées, lues chez le fournisseur (lecture seule).

    Le champ ``billed_seconds`` calculé en direct par le runner (différence de
    ``usage()`` entre deux appels HTTP) souffre d'un décalage de latence dans l'API et
    peut mal répartir le coût entre jobs consécutifs (vu : -2.0 s sur un job, 0.0 s sur un
    autre, alors que le total réel était correct). ``job.metrics()`` interroge directement
    l'enregistrement du job et est la source qui a donné exactement 2.000 s sur les 30
    jobs de la cohorte P0 — beaucoup plus fiable qu'une différence entre deux lectures.
    """
    if job_id in _METRICS_CACHE:
        return _METRICS_CACHE[job_id]
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        svc = QiskitRuntimeService()
        m = svc.job(job_id).metrics() or {}
        usage = m.get("usage")
        val = float(usage["quantum_seconds"]) if isinstance(usage, dict) and "quantum_seconds" in usage else None
    except Exception:
        val = None
    _METRICS_CACHE[job_id] = val
    return val


def certified_reference(instance: str, k: int, objective_id: str) -> dict[str, Any]:
    """Référence certifiée par ÉNUMÉRATION COMPLÈTE, dans les unités de l'objective_id."""
    from acrpq import geometry
    from acrpq.io.loader import InstanceLoader
    from acrpq.objectives import primary_objective
    from acrpq.quantum.qubo import build_qubo

    inst = InstanceLoader().load(instance)
    qubo = build_qubo(inst, n_theta=k, n_q=1, objective_id=objective_id, max_qubits=32)
    d = qubo.discrete
    aircraft = list(inst.aircraft())
    values: list[float] = []
    for choice in itertools.product(range(d.k), repeat=len(aircraft)):
        q, theta = d.maneuvers(choice)
        if geometry.count_conflicts(inst, q, theta) == 0:
            values.append(float(primary_objective(objective_id, q, theta, inst.w)))
    if not values:
        return {"reference_objective": None, "reference_optima_count": 0,
                "n_feasible_assignments": 0, "method": "exhaustive_enumeration"}
    best = min(values)
    return {"reference_objective": best,
            "reference_optima_count": sum(1 for v in values if v == best),
            "n_feasible_assignments": len(values),
            "n_assignments": d.k ** len(aircraft),
            "objective_units": objective_id,
            "method": "exhaustive_enumeration"}


def build(verify_only: bool = False) -> int:
    ledger = json.loads(LEDGER.read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    SIDE.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    written = 0

    for key, record in sorted(ledger["jobs"].items()):
        raw_path = RAW / f"{key}.json"
        payload = _canonical(record)
        if raw_path.exists():
            existing = raw_path.read_bytes()
            if existing != payload:
                # le brut est immuable : on ne l'écrase JAMAIS, on signale
                problems.append(f"{key}: le brut figé diffère du registre courant "
                                f"(brut conservé, registre ignoré)")
            payload = existing
        elif not verify_only:
            raw_path.write_bytes(payload)
            written += 1
        raw_sha = _sha_bytes(payload)

        res = record.get("resolved") or {}
        is_canary = record.get("arm") == CANARY_ARM or record.get("kind") == "canary"
        ref = certified_reference(record["instance"], record["grid_k"],
                                  record["objective_id"])
        summary = ((record.get("result") or {}).get("summary") or {})
        best = summary.get("best_feasible") or {}
        feasible = bool(best) and bool(best.get("feasible"))
        gap = None
        gap_suppressed = None
        if not feasible:
            gap_suppressed = "aucun échantillon faisable : aucun écart n'est calculé"
        elif ref["reference_objective"] is None:
            gap_suppressed = "aucune référence certifiée pour cette cellule"
        else:
            gap = float(best["objective"]) - float(ref["reference_objective"])

        sidecar = {
            "schema": "acrpq-campaign-sidecar/1",
            "job_key": key,
            "raw_path": str(raw_path.as_posix()),
            "raw_sha256": raw_sha,
            "ibm_job_ids": list(record.get("ibm_job_ids") or []),
            "classification": {
                "technical_canary": is_canary,
                "protocol_superseded_for_reference_metadata": True,
                "raw_hardware_result_valid": record.get("state") == "completed",
                "official_hypothesis_test": (not is_canary
                                             and record.get("kind") == "campaign"),
            },
            "corrected_reference": ref,
            "corrected_optimum": {
                "best_feasible_objective": (float(best["objective"]) if feasible else None),
                "hits_certified_optimum": (
                    feasible and ref["reference_objective"] is not None
                    and float(best["objective"]) == float(ref["reference_objective"])),
                "probability_mass_on_best_feasible": (
                    float(best.get("probability")) if feasible else None),
                "gap_to_certified_reference": gap,
                "gap_suppressed_reason": gap_suppressed,
            },
            "ideal_offline_reference": {
                "ideal_feasible_mass": res.get("ideal_feasible_mass"),
                "ideal_onehot_valid_mass": res.get("ideal_onehot_valid_mass"),
                "ideal_certified_optimum_mass": res.get("ideal_certified_optimum_mass"),
            },
            "measured": {
                "feasibility_rate_strict": summary.get("feasibility_rate"),
                "n_shots": summary.get("n_shots"),
                "n_unique_total": summary.get("n_unique_total"),
                "billed_seconds_incremental_runner_estimate": record.get("billed_seconds"),
                "billed_seconds_authoritative_provider_metrics": (
                    authoritative_billed_seconds(sidecar_job_ids[0])
                    if (sidecar_job_ids := list(record.get("ibm_job_ids") or [])) else None),
            },
            "never_rewritten": ["counts", "ibm_job_ids", "export.artefact", "run_id"],
        }
        side_path = SIDE / f"{key}.json"
        if not verify_only:
            side_path.write_bytes(_canonical(sidecar))
        elif side_path.exists():
            stored = json.loads(side_path.read_text())
            if stored.get("raw_sha256") != raw_sha:
                problems.append(f"{key}: le sidecar ne pointe plus vers son brut")

        flags = sidecar["classification"]
        print(f"  {key}")
        print(f"    canari={flags['technical_canary']} "
              f"test_officiel={flags['official_hypothesis_test']} "
              f"brut_valide={flags['raw_hardware_result_valid']}")
        print(f"    référence corrigée = {ref['reference_objective']} "
              f"({ref['reference_optima_count']} optima, unités "
              f"{ref.get('objective_units')})")
        print(f"    meilleur faisable = {sidecar['corrected_optimum']['best_feasible_objective']} "
              f"· optimum atteint = {sidecar['corrected_optimum']['hits_certified_optimum']} "
              f"· écart = {gap if gap is not None else gap_suppressed}")

    if problems:
        print("\nPROBLÈMES :")
        for pr in problems:
            print(f"  {pr}")
        return 3
    print(f"\n{written} brut(s) figé(s), {len(ledger['jobs'])} sidecar(s) "
          f"{'vérifié(s)' if verify_only else 'écrit(s)'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Figer les bruts et publier les sidecars")
    ap.add_argument("--verify", action="store_true")
    return build(verify_only=ap.parse_args().verify)


if __name__ == "__main__":
    raise SystemExit(main())
