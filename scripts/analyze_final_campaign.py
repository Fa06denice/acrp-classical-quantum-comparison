#!/usr/bin/env python3
"""Analyse statistique de la campagne IBM finale — le JOB est l'unité expérimentale.

**Correction Codex (contre-review) : l'analyse précédente poolait les 512 shots de
chacun des 3 jobs d'un bras en un dénominateur unique de 1536, puis calculait un
intervalle de Newcombe dessus.** Cela traite implicitement les shots inter-job comme
échangeables/indépendants, alors que chaque job est un tirage distinct sur un état
matériel possiblement différent (dérive de calibration entre jobs). L'intervalle qui en
résultait était artificiellement étroit et ne mesurait PAS l'incertitude inter-run.

Ce module calcule maintenant, pour chaque cellule (instance × grille × objectif) et
chaque `repetition_id` ∈ {1,2,3}, la différence appariée
``d_r = feasible_rate(A1, r) − feasible_rate(A0, r)``, et ne publie que des statistiques
DESCRIPTIVES sur ces 3 valeurs (moyenne, médiane, min/max, écart-type, décompte de
signes). Aucune inférence forte n'est produite : un test des signes exact bilatéral ne
peut jamais atteindre α=0,05 avec 3 paires (p minimal = 2×0,5³ = 0,25), ce qui est
calculé et affiché explicitement plutôt que caché.

L'intervalle binomial sur les shots N'A PAS disparu : il est conservé, renommé
``conditional_shot_sampling_interval``, calculé **par job individuel** (jamais poolé
entre jobs), et documenté comme décrivant uniquement le bruit de tir CONDITIONNEL à
l'état matériel de CE job précis — jamais comme une preuve d'effet inter-run.

``statistically_significant`` n'est JAMAIS émis à ``true`` par ce module : à n=3, aucune
inférence de ce module n'atteint ce statut, et le champ n'existe simplement pas dans la
sortie (voir test dédié).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("results/ibm_qaoa_final_campaign_v1")
RESULTS = ROOT / "campaign_results.json"
SCHEMA = "acrpq-ibm-final-campaign-statistics/2"
SUPERSEDES_CONTENT_SHA256 = None  # renseigné dynamiquement si un fichier v1 existe


def _sha(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - margin) / denom, (centre + margin) / denom)


def clopper_pearson_upper(successes: int, n: int, alpha: float = 0.05) -> float:
    if n == 0:
        return 1.0
    if successes == 0:
        return 1.0 - alpha ** (1.0 / n)
    from math import comb

    def cdf(k: int, p: float) -> float:
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if cdf(successes, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def exact_two_sided_sign_test_pvalue(n_pos: int, n_neg: int) -> float:
    """p bilatéral du test des signes exact (binomial, p=0.5), ties exclues.

    Documenté explicitement : avec n<=3 paires non nulles, le p minimal atteignable est
    2*0.5**3 = 0.25, très loin de 0.05 — cette fonction sert à AFFICHER cette limite,
    jamais à revendiquer une signification.
    """
    from math import comb
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = min(n_pos, n_neg)
    p_one_sided = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * p_one_sided)


def _cell_key(r: dict[str, Any]) -> tuple:
    return (r["instance"], r["grid_k"], r["objective_id"])


def _descriptive(xs: list[float]) -> dict[str, Any]:
    n = len(xs)
    mean = sum(xs) / n if n else None
    sd = (math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else None)
    return {
        "n": n, "mean": mean, "median": sorted(xs)[n // 2] if xs else None,
        "min": min(xs) if xs else None, "max": max(xs) if xs else None,
        "sd_descriptive_do_not_treat_as_inferential": sd,
        "individual_values": xs,
    }


def job_level_paired_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Cœur de la correction : une paire (A0_r, A1_r) par cellule × répétition."""
    official = [r for r in rows if r.get("official_hypothesis_test")]
    by_cell_rep: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in official:
        if r["arm"] not in ("A0_P0_replication", "A1_P1_feas_angles"):
            continue
        # appariement STRICT : cellule (instance, grid_k, objective_id) + repetition_id
        by_cell_rep[(*_cell_key(r), r["repetition_id"])][r["arm"]] = r

    per_cell: dict[str, dict[str, Any]] = {}
    n_cells_mean_positive = 0
    n_cells_all_pairs_positive = 0
    for cell in sorted({_cell_key(r) for r in official}):
        reps = sorted({rep for (*c, rep) in by_cell_rep if tuple(c) == cell})
        a0_rates, a1_rates, diffs = [], [], []
        for rep in reps:
            pair = by_cell_rep.get((*cell, rep), {})
            a0, a1 = pair.get("A0_P0_replication"), pair.get("A1_P1_feas_angles")
            if a0 is None or a1 is None:
                continue  # cellule sans les deux bras (ex. CP_5/K3, CP_3/K5 : A0/A1 seuls, ok)
            a0r = float(a0["feasible_shot_rate_strict"] or 0.0)
            a1r = float(a1["feasible_shot_rate_strict"] or 0.0)
            a0_rates.append(a0r)
            a1_rates.append(a1r)
            diffs.append(a1r - a0r)
        if not diffs:
            continue
        n_pos = sum(1 for d in diffs if d > 0)
        n_neg = sum(1 for d in diffs if d < 0)
        n_zero = sum(1 for d in diffs if d == 0)
        desc = _descriptive(diffs)
        ratios_secondary = [
            {"repetition_id": rep, "ratio_A1_over_A0": (a1 / a0 if a0 > 0 else None)}
            for rep, a0, a1 in zip(reps, a0_rates, a1_rates)
        ]
        label = f"{cell[0]}/K{cell[1]}/{cell[2]}"
        per_cell[label] = {
            "repetition_ids": reps,
            "A0_rates": a0_rates, "A1_rates": a1_rates,
            "paired_differences_d_r": diffs,
            "descriptive": desc,
            "n_pairs_positive": n_pos, "n_pairs_negative": n_neg, "n_pairs_zero": n_zero,
            "ratio_A1_over_A0_secondary_only_when_A0_nonzero": ratios_secondary,
            "exact_two_sided_sign_test_pvalue": exact_two_sided_sign_test_pvalue(n_pos, n_neg),
            "sign_test_note": (
                "Avec au plus 3 paires non nulles, le p bilatéral minimal atteignable "
                "est 2*0.5^3=0.25 : ce test ne peut PAS démontrer un effet à alpha=0.05 "
                "à cette taille d'échantillon, quel que soit le résultat observé."),
        }
        if desc["mean"] is not None and desc["mean"] > 0:
            n_cells_mean_positive += 1
        if n_pos == len(diffs) and len(diffs) > 0:
            n_cells_all_pairs_positive += 1

    return {
        "per_cell": per_cell,
        "n_cells_total": len(per_cell),
        "n_cells_with_mean_diff_positive": n_cells_mean_positive,
        "n_cells_with_all_pairs_positive": n_cells_all_pairs_positive,
        "authorized_phrasing_fr": (
            f"Dans {n_cells_mean_positive} cellules sur {len(per_cell)}, la différence "
            "moyenne observée entre A1 et A0 est positive. Avec trois jobs par bras, ces "
            "résultats constituent un signal préliminaire cohérent avec les prédictions "
            "idéales, sans puissance suffisante pour une conclusion statistique forte."),
    }


def exploratory_bootstrap_over_job_pairs(per_cell: dict[str, dict]) -> dict[str, Any]:
    """Rééchantillonnage des PAIRES DE JOBS (jamais des shots), n=3 par cellule.

    Explicitement EXPLORATOIRE ET INSTABLE à cette taille : avec 3 valeurs, il n'existe
    que 3^3=27 tirages possibles avec remise, donc la distribution rééchantillonnée est
    elle-même grossière. Jamais utilisé pour un vocabulaire de significativité. Les
    observations individuelles restent toujours affichées à côté (voir per_cell).
    """
    import random
    rng = random.Random(20260904)  # figé, déterministe et documenté
    out: dict[str, Any] = {}
    for label, cell in per_cell.items():
        diffs = cell["paired_differences_d_r"]
        if len(diffs) < 2:
            continue
        means = [sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(2000)]
        means.sort()
        out[label] = {
            "n_original_pairs": len(diffs),
            "n_bootstrap_resamples": 2000,
            "resampled_mean_2_5_pct": means[int(0.025 * len(means))],
            "resampled_mean_97_5_pct": means[int(0.975 * len(means))],
            "warning": "EXPLORATOIRE, INSTABLE À N=3 — ne jamais citer comme un IC valide",
        }
    return out


def conditional_shot_sampling_intervals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Wilson par JOB INDIVIDUEL sur ses propres shots — jamais poolé entre jobs.

    Décrit uniquement le bruit de tir binomial conditionnel à l'état matériel mesuré
    CE job précis. Ne sert JAMAIS de preuve d'effet inter-run ni de comparaison entre
    bras : deux jobs ont pu être exécutés à des instants de calibration différents.
    """
    out: dict[str, Any] = {}
    for r in rows:
        shots = r.get("shots_received")
        rate = r.get("feasible_shot_rate_strict")
        if not shots or rate is None:
            continue
        successes = round(rate * shots)
        lo, hi = wilson_interval(successes, shots)
        out[r["experiment_id"]] = {
            "n_shots_this_job_only": shots,
            "successes_this_job_only": successes,
            "conditional_shot_sampling_interval_95": [lo, hi],
            "note": ("Bruit de tir intra-job uniquement, conditionnel à l'état matériel "
                    "de CE job. Pas un intervalle inter-run, pas une preuve d'effet."),
        }
    return out


def a2_transpilation_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A2 (transpilation seule) — descriptif uniquement, séparé d'A0/A1."""
    official = [r for r in rows if r.get("official_hypothesis_test")]
    out = []
    for cell in sorted({_cell_key(r) for r in official}):
        a0 = [r for r in official if _cell_key(r) == cell and r["arm"] == "A0_P0_replication"]
        a2 = [r for r in official if _cell_key(r) == cell and r["arm"] == "A2_P0_transpile_L2"]
        if not a0 or not a2:
            continue
        out.append({
            "cell": f"{cell[0]}/K{cell[1]}/{cell[2]}",
            "note": "A2 change SEULEMENT le niveau de transpilation (angles = P0), descriptif",
            "A0_level0_rates": [r["feasible_shot_rate_strict"] for r in a0],
            "A2_level2_rates": [r["feasible_shot_rate_strict"] for r in a2],
            "isa_depth_level0": [r["isa_depth_after"] for r in a0],
            "isa_depth_level2": [r["isa_depth_after"] for r in a2],
        })
    return out


def canary_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"experiment_id": r["experiment_id"], "arm": r["arm"],
            "instance": r["instance"], "objective_id": r["objective_id"],
            "feasible_shot_rate_strict": r["feasible_shot_rate_strict"],
            "note": "EXCLU de toute statistique d'hypothèse (official_hypothesis_test=false)"}
           for r in rows if not r.get("official_hypothesis_test")]


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    official = [r for r in rows if r.get("official_hypothesis_test")]
    canaries = [r for r in rows if not r.get("official_hypothesis_test")]
    job_level = job_level_paired_analysis(rows)
    return {
        "schema": SCHEMA,
        "experimental_unit": "job (JAMAIS le shot — un shot n'est pas un tirage indépendant "
                             "inter-job ; voir job_level_paired_analysis)",
        "n_official_jobs": len(official), "n_canary_jobs": len(canaries),
        "job_level_paired_analysis": job_level,
        "exploratory_bootstrap_over_job_pairs_UNSTABLE_AT_N3":
            exploratory_bootstrap_over_job_pairs(job_level["per_cell"]),
        "conditional_shot_sampling_interval_per_job":
            conditional_shot_sampling_intervals(rows),
        "a2_transpilation_level_analysis_SEPARATE_FROM_A0_A1": a2_transpilation_analysis(rows),
        "canaries_excluded_from_hypothesis_stats": canary_summary(rows),
        "forbidden": [
            "traiter les shots comme l'unité expérimentale ou pooler des shots entre jobs",
            "toute inférence forte à n=3 (aucun champ statistically_significant n'existe ici)",
            "employer « significatif » au sens statistique à cette taille d'échantillon",
            "citer conditional_shot_sampling_interval comme preuve d'un effet entre bras",
            "citer exploratory_bootstrap_over_job_pairs comme un intervalle de confiance valide",
            "agrégation entre K, entre objectifs, entre bras, entre matériel et simulation",
        ],
        "language_note": job_level["authorized_phrasing_fr"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--invalidates", default=None,
                    help="chemin d'un statistics.json v1 à marquer explicitement invalidé")
    args = ap.parse_args()
    if not RESULTS.exists():
        print("MISSING campaign_results.json — lancer build_final_campaign_dataset.py")
        return 2
    rows = json.loads(RESULTS.read_text())["rows"]
    doc = analyze(rows)
    if args.invalidates and Path(args.invalidates).exists():
        old = json.loads(Path(args.invalidates).read_text())
        old_sha = _sha(old)
        doc["invalidates_previous_version"] = {
            "previous_schema": old.get("schema"),
            "previous_content_sha256": old_sha,
            "reason": ("La version précédente calculait un intervalle de Newcombe sur des "
                      "shots poolés entre les 3 jobs d'un bras (n=1536), traitant "
                      "implicitement les shots inter-job comme indépendants. Corrigé : le "
                      "job est l'unité expérimentale (n=3/bras), analyse appariée "
                      "descriptive, aucune inférence forte."),
        }
    doc["content_sha256"] = _sha({k: v for k, v in doc.items() if k != "content_sha256"})
    (ROOT / "statistics.json").write_text(
        json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n")
    jl = doc["job_level_paired_analysis"]
    print(f"[stats v2] {doc['n_official_jobs']} jobs officiels, {doc['n_canary_jobs']} canaris")
    print(f"  {jl['n_cells_with_mean_diff_positive']}/{jl['n_cells_total']} cellules : "
          f"moyenne(d_r) > 0")
    print(f"  {jl['n_cells_with_all_pairs_positive']}/{jl['n_cells_total']} cellules : "
          f"les 3 paires sont individuellement positives")
    for label, c in jl["per_cell"].items():
        d = c["descriptive"]
        print(f"    {label:26s} d_r={[round(x,4) for x in c['paired_differences_d_r']]} "
              f"moyenne={d['mean']:+.5f} mediane={d['median']:+.5f} "
              f"+/-/0={c['n_pairs_positive']}/{c['n_pairs_negative']}/{c['n_pairs_zero']} "
              f"p_signe={c['exact_two_sided_sign_test_pvalue']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
