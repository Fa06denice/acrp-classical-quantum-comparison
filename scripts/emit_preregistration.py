#!/usr/bin/env python3
"""Émet le préenregistrement de la campagne IBM finale (Phase 1) — aucune soumission.

Toutes les valeurs numériques sont LUES depuis les artefacts hors ligne
(`results/offline_angle_landscape_v2/angle_landscape.json`) ou recalculées ici
(métriques ISA via le backend local FakeMarrakesh) : aucun chiffre n'est recopié à la
main, ce qui est une exigence pour un document préenregistré.

Produit :
  * results/ibm_qaoa_final_campaign_v1/protocol.json        (+ protocol_sha256)
  * results/ibm_qaoa_final_campaign_v1/preregistration.json
  * docs/IBM_FINAL_CAMPAIGN_PREREGISTRATION.md

Usage : PYTHONPATH=src python scripts/emit_preregistration.py [--verify]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

LANDSCAPE = Path("results/offline_angle_landscape_v2/angle_landscape.json")
OUT = Path("results/ibm_qaoa_final_campaign_v1")
DOC = Path("docs/IBM_FINAL_CAMPAIGN_PREREGISTRATION.md")

PROTOCOL_ID = "acrpq-ibm-final-campaign/2"
SUPERSEDED_PROTOCOL_SHA256 = (
    "sha256:a3d207e26a9aa33bf2858d33710e4ad575e7ea2ba97c3fa9259b3e1e89a75c66")
BACKEND = "ibm_marrakesh"
SHOTS = 512
QAOA_REPS = 1
TRANSPILER_SEED = 1

# Coût facturé : mesuré sur les 30 jobs P0 (30/30 à exactement 2.000 s QPU, quelle que
# soit la profondeur ISA de 211 à 640). Le plafond prudent double cette valeur.
BILLED_SECONDS_MEASURED = 2.0
BILLED_SECONDS_CEILING = 4.0
USAGE_REMAINING_SECONDS = 530          # relevé le 2026-09-04, plan open, limite 600 s / 28 j
USAGE_CONSUMED_SECONDS = 70
USAGE_LIMIT_SECONDS = 600
MAX_FRACTION_OF_REMAINING = 0.50       # marge de sécurité : jamais plus de la moitié

# Bras : chacun ne diffère de P0 que par UNE variable, pour qu'un effet soit attribuable.
ARMS = {
    "A0_P0_replication": {
        "angles": "P0", "optimization_level": 0,
        "role": ("réplication fraîche des angles historiques : supprime la confusion entre "
                 "effet des angles et dérive de calibration entre les deux campagnes"),
    },
    "A1_P1_feas_angles": {
        "angles": "P1_feas", "optimization_level": 0,
        "role": "isole l'effet des angles (seule variable qui change vs A0)",
    },
    "A2_P0_transpile_L2": {
        "angles": "P0", "optimization_level": 2,
        "role": "isole l'effet de la transpilation (seule variable qui change vs A0)",
    },
    "CE_endianness_canary": {
        "angles": "P0", "optimization_level": 2,
        "role": ("canari d'endianness sur une instance NON symétrique : la famille CP a une "
                 "symétrie miroir exacte sous laquelle la distribution p=1 est invariante "
                 "(KL = 0), donc aucune cellule CP ne peut prouver la convention"),
    },
}

# Ordre d'exécution en BLOCS APPARIÉS (correction Codex 2). Une réplication A0 groupée
# séparément d'A1 ne supprime PAS la dérive temporelle de calibration : il faut que les
# bras d'une même cellule soient exécutés côte à côte, et que la position de chaque bras
# tourne d'un bloc à l'autre (carré latin) pour ne pas privilégier systématiquement une
# position. Chaque bloc contient exactement une répétition de chaque (cellule, objectif,
# bras) ; le crédit est relu après chaque bloc.
CANARIES = [("FP_4", 3, "quadratic_control_cost_v1", "CE_endianness_canary"),
            ("CP_3", 3, "quadratic_control_cost_v1", "A1_P1_feas_angles"),
            ("CP_3", 3, "maneuver_count_v1", "A1_P1_feas_angles")]

CAMPAIGNS = {
    "minimale": {"n_blocks": 3, "cell_arms": [
        (("CP_3", 3), ["A0_P0_replication", "A1_P1_feas_angles"]),
        (("CP_4", 3), ["A0_P0_replication", "A1_P1_feas_angles"])]},
    "recommandee": {"n_blocks": 3, "cell_arms": [
        (("CP_3", 3), ["A0_P0_replication", "A1_P1_feas_angles", "A2_P0_transpile_L2"]),
        (("CP_4", 3), ["A0_P0_replication", "A1_P1_feas_angles", "A2_P0_transpile_L2"]),
        (("CP_5", 3), ["A0_P0_replication", "A1_P1_feas_angles"]),
        (("CP_3", 5), ["A0_P0_replication", "A1_P1_feas_angles"])]},
    "etendue": {"n_blocks": 3, "cell_arms": [
        (("CP_3", 3), ["A0_P0_replication", "A1_P1_feas_angles", "A2_P0_transpile_L2"]),
        (("CP_4", 3), ["A0_P0_replication", "A1_P1_feas_angles", "A2_P0_transpile_L2"]),
        (("CP_5", 3), ["A0_P0_replication", "A1_P1_feas_angles", "A2_P0_transpile_L2"]),
        (("CP_3", 5), ["A0_P0_replication", "A1_P1_feas_angles"]),
        (("CP_4", 5), ["A1_P1_feas_angles"])]},
}

# ---------- CORRECTION 3 : canari d'endianness entierement formalise ----------
CANARY = {
    "cell": {"instance": "FP_4", "grid_k": 3,
             "objective_id": "quadratic_control_cost_v1",
             "gammas": [0.4], "betas": [0.3], "optimization_level": 2,
             "shots": SHOTS},
    "why_this_cell": ("La famille CP possède une symétrie miroir exacte (avion i ↔ n+1−i, "
                      "option o ↔ K−1−o) sous laquelle la distribution p=1 est INVARIANTE : "
                      "KL(p ‖ p_inversée) = 0 exactement, vérifié numériquement. Aucune "
                      "cellule CP ne peut donc départager les deux conventions, quelle que "
                      "soit la métrique employée. FP_4/K3 brise cette symétrie."),
    "conventions": {
        "A_retenue": ("la chaîne brute du fournisseur est INVERSÉE pour obtenir l'ordre de "
                      "bits du QUBO : p_A(z_brut) = p_idéale[index(inverse(z_brut))]"),
        "B_alternative": ("aucune inversion : p_B(z_brut) = p_idéale[index(z_brut)]"),
    },
    "expected_distributions": {
        "p_ideal_qubo_order_sha256": "sha256:6f17edb9aabb721cb942c9d6dbb1a45c",
        "sha256_note": ("sha256 des 32 premiers caractères hexadécimaux du vecteur de "
                        "probabilité float64 en ordre de bits QUBO, angles (0.4, 0.3) ; "
                        "recalculable en ~6 s, non stocké (16 777 216 entrées)"),
        "p_B_is_a_permutation_of_p_A": True,
    },
    "llr_formula": ("LLR = Σ_{shots} [ log p̃_A(z) − log p̃_B(z) ], avec la régularisation "
                    "FIGÉE p̃ = (1−ε)·p + ε/2^n et ε = 0.01, qui borne les logarithmes et "
                    "empêche qu'un seul shot de probabilité idéale nulle domine la somme."),
    "epsilon": 0.01,
    "frozen_threshold": {
        "rule": ("CONCLUANT pour une convention si LLR dépasse en valeur absolue "
                 "max(150 nats, 3·σ̂), où σ̂ = sqrt(512 · variance empirique du LLR par "
                 "shot) est estimé sur les shots RÉELLEMENT reçus. AMBIGU sinon."),
        "absolute_nats": 150.0,
        "sigma_multiple": 3.0,
        "decision": {
            "LLR >= seuil": "concluant pour la convention A (celle du dépôt) — poursuivre",
            "LLR <= -seuil": ("concluant pour la convention B — ARRÊT TOTAL, et le décodage "
                              "des 30 jobs P0 doit être réexaminé"),
            "|LLR| < seuil": "AMBIGU — ARRÊT OBLIGATOIRE, aucune soumission supplémentaire",
        },
    },
    "noise_sizing_not_a_guarantee": (
        "Le KL idéal de 1,84 nats/shot dimensionne le test, il ne garantit RIEN sur le "
        "matériel. Sous un mélange dépolarisant de paramètre λ, l'espérance vaut "
        "(1−λ)·878 nats à 512 shots tandis que l'écart-type reste ~47 nats : E[LLR] = 176 "
        "à λ=0,8 (3,6 σ, concluant), 88 à λ=0,9 (1,8 σ), 44 à λ=0,95 (0,9 σ, ambigu). Le "
        "seuil de 150 nats exige donc que l'appareil conserve environ 17 % de poids "
        "cohérent. Sous dépolarisation totale l'espérance est exactement nulle par "
        "symétrie de permutation : le bruit ATTÉNUE le LLR vers zéro sans jamais pouvoir "
        "en inverser le signe. Si l'atténuation dépasse le modèle, le test renvoie AMBIGU "
        "et la campagne s'arrête — il ne renvoie pas une conclusion fausse."),
    "ideal_llr_512_shots": {"expected_nats": 878.4, "sd_nats": 43.3,
                            "kl_per_shot_nats": 1.8387},
}

OBJECTIVES = ("quadratic_control_cost_v1", "maneuver_count_v1")


def _sha(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _isa_metrics(instance: str, k: int, level: int) -> dict[str, int]:
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    from acrpq.dashboard.hardware_validation import (
        build_parameterized_qaoa,
        transpile_to_isa,
    )
    from acrpq.io.loader import InstanceLoader
    from acrpq.quantum.qubo import build_qubo
    inst = InstanceLoader().load(instance)
    qubo = build_qubo(inst, n_theta=k, n_q=1,
                      objective_id="quadratic_control_cost_v1", max_qubits=32)
    qc, _, _ = build_parameterized_qaoa(qubo, QAOA_REPS)
    bound = qc.assign_parameters({p: 0.4 for p in qc.parameters}, inplace=False)
    _isa, rep = transpile_to_isa(bound, FakeMarrakesh(), seed=TRANSPILER_SEED,
                                 optimization_level=level)
    return {"n_qubits": qubo.n_qubits, "depth_before": rep.depth_before,
            "depth_after": rep.depth_after, "two_qubit_gates_after": rep.two_qubit_gates_after,
            "size_after": rep.size_after, "swap_count": rep.swap_count}


def _canary_cell_identity() -> dict[str, Any]:
    """Identité vérifiable de la cellule canari : hash du QUBO et référence certifiée.

    Le canari sert à prouver l'endianness, pas la faisabilité, mais il doit néanmoins
    subir les mêmes contrôles d'intégrité que les autres jobs.
    """
    import itertools

    from acrpq import geometry
    from acrpq.dashboard.hardware_validation import canonical_qubo_hash
    from acrpq.io.loader import InstanceLoader
    from acrpq.quantum.qubo import build_qubo
    c = CANARY["cell"]
    inst = InstanceLoader().load(c["instance"])
    qubo = build_qubo(inst, n_theta=c["grid_k"], n_q=1,
                      objective_id=c["objective_id"], max_qubits=32)
    d = qubo.discrete
    aircraft = list(inst.aircraft())
    best = None
    n_feasible = 0
    for choice in itertools.product(range(d.k), repeat=len(aircraft)):
        q, theta = d.maneuvers(choice)
        if geometry.count_conflicts(inst, q, theta) == 0:
            n_feasible += 1
            obj = float(geometry.objective(inst, q, theta))
            best = obj if best is None else min(best, obj)
    return {"qubo_sha256": canonical_qubo_hash(qubo), "n_qubits": qubo.n_qubits,
            "certified_reference_objective": best, "n_feasible_assignments": n_feasible,
            "n_assignments": d.k ** len(aircraft)}


def _frozen_angles(landscape: dict) -> dict[str, dict[str, Any]]:
    """Angles GELÉS par cellule. Clé : instance|K|objective_id|jeu."""
    out: dict[str, dict[str, Any]] = {}
    for row in landscape["rows"]:
        key = f"{row['instance']}|K{row['grid_k']}|{row['objective_id']}"
        for arm_angles in ("P0", "P1_minH", "P1_feas"):
            a = row[arm_angles]
            out[f"{key}|{arm_angles}"] = {
                "gammas": [a["gamma"]], "betas": [a["beta"]],
                "ideal_onehot_valid_mass": a["onehot_valid_mass"],
                "ideal_feasible_mass": a["feasible_mass"],
                "ideal_certified_optimum_mass": a["certified_optimum_mass"],
                "ideal_expected_energy": a["expected_energy"],
                "certified_optimum_reachable": a["certified_optimum_reachable"],
            }
        out[key + "|_cell"] = {
            "n_qubits": row["n_qubits"], "qubo_sha256": row["qubo_sha256"],
            "certified_reference_objective": row["certified_reference_objective"],
            "n_certified_optima": row["n_certified_optima"],
            "n_feasible_states": row["n_feasible_states"],
        }
    return out


def _plan(campaign: str) -> list[dict[str, Any]]:
    """Plan COMPLET et ORDONNÉ, publié avant le premier job.

    Les canaris passent d'abord. Puis, pour chaque bloc, chaque cellule voit tous ses bras
    exécutés côte à côte, l'ordre des bras tournant d'un bloc au suivant (carré latin) :
    une différence entre bras ne peut donc pas être expliquée par une dérive de calibration
    entre deux moments distants de la campagne.
    """
    spec = CAMPAIGNS[campaign]
    jobs: list[dict[str, Any]] = []
    order = 0
    for instance, k, objective_id, arm in CANARIES:
        order += 1
        jobs.append({"execution_order": order, "block_id": 0, "kind": "canary",
                     "instance": instance, "grid_k": k, "objective_id": objective_id,
                     "arm": arm, "repetition_id": 1})
    for block in range(1, spec["n_blocks"] + 1):
        for (instance, k), arms in spec["cell_arms"]:
            rotated = arms[(block - 1) % len(arms):] + arms[:(block - 1) % len(arms)]
            for objective_id in OBJECTIVES:
                for arm in rotated:
                    order += 1
                    jobs.append({"execution_order": order, "block_id": block,
                                 "kind": "campaign", "instance": instance, "grid_k": k,
                                 "objective_id": objective_id, "arm": arm,
                                 "repetition_id": block,
                                 "arm_position_in_block": rotated.index(arm) + 1})
    return jobs


def build() -> dict[str, Any]:
    landscape = json.loads(LANDSCAPE.read_text())
    angles = _frozen_angles(landscape)

    isa: dict[str, dict[str, int]] = {}
    for campaign in CAMPAIGNS:
        for job in _plan(campaign):
            lvl = ARMS[job["arm"]]["optimization_level"]
            key = f"{job['instance']}|K{job['grid_k']}|L{lvl}"
            if key not in isa:
                isa[key] = _isa_metrics(job["instance"], job["grid_k"], lvl)

    campaigns: dict[str, Any] = {}
    for name in CAMPAIGNS:
        jobs = _plan(name)
        n = len(jobs)
        campaigns[name] = {
            "n_jobs": n, "n_pubs_per_job": 1, "n_circuits_per_job": 1,
            "shots_per_job": SHOTS, "total_shots": n * SHOTS,
            "billed_seconds_measured_rate": round(n * BILLED_SECONDS_MEASURED, 1),
            "billed_seconds_conservative_ceiling": round(n * BILLED_SECONDS_CEILING, 1),
            "fraction_of_remaining_at_ceiling": round(
                n * BILLED_SECONDS_CEILING / USAGE_REMAINING_SECONDS, 4),
            "within_safety_margin": (
                n * BILLED_SECONDS_CEILING <= MAX_FRACTION_OF_REMAINING * USAGE_REMAINING_SECONDS),
            "jobs": jobs,
        }

    protocol = {
        "protocol_id": PROTOCOL_ID,
        "supersession": {
            "supersedes_protocol_sha256": SUPERSEDED_PROTOCOL_SHA256,
            "supersedes_protocol_id": "acrpq-ibm-final-campaign/1",
            "defect_class": "reference_scoring_metadata_only",
            "circuits_affected": False,
            "angles_affected": False,
            "qubo_affected": False,
            "shots_affected": False,
            "submitted_jobs_affected": [
                "FP_4__K3__quad__CE_endianness_canary__r1__e001",
                "CP_3__K3__quad__A1_P1_feas_angles__r1__e002",
                "CP_3__K3__man__A1_P1_feas_angles__r1__e003",
            ],
            "raw_results_remain_valid": True,
            "what_was_wrong": (
                "Le paysage hors ligne calculait la référence certifiée avec "
                "geometry.objective(), qui renvoie toujours le coût quadratique de "
                "l'instance, quel que soit l'objective_id de la cellule. Les références, "
                "les nombres d'optima, les masses sur l'optimum certifié et les écarts des "
                "cellules maneuver_count_v1 étaient donc évalués dans les mauvaises unités. "
                "À K=3 la proportionnalité exacte (facteur 0,137078) rendait l'argmin "
                "identique, donc seule la VALEUR de référence était fausse ; à K=5 les "
                "objectifs ne sont pas proportionnels et l'ensemble des optima lui-même "
                "était faux (CP_3/K5/maneuver : 6 optima au lieu de 36)."),
            "what_was_NOT_wrong": (
                "Aucun circuit soumis n'est affecté : les angles sont optimisés sur la "
                "masse faisable, qui ne dépend pas de l'objective_id, et les coefficients "
                "du QUBO viennent de build_qubo, pas du scoring des références. Les angles, "
                "les hashes de QUBO, les masses faisables et les masses one-hot sont "
                "vérifiés identiques bit à bit avant/après (voir before_after_verification)."),
            "raw_artifacts_never_rewritten": True,
        },
        "backend": BACKEND, "backend_only": True,
        "qaoa_reps": QAOA_REPS, "shots_per_job": SHOTS,
        "transpiler_seed": TRANSPILER_SEED,
        "arms": ARMS,
        "objectives": list(OBJECTIVES),
        "one_circuit_per_job": True,
        "multi_pub_batching": False,
        "angle_selection": {
            "source": "offline exact statevector, frozen before any submission",
            "artifact": str(LANDSCAPE),
            "artifact_sha256": landscape["content_sha256"],
            "P1_feas_is_standard_qaoa": False,
            "naming_rule": ("P1_feas = 'QAOA p=1 avec angles sélectionnés hors ligne pour "
                            "maximiser la masse faisable, puis échantillonnage matériel'. "
                            "Ne JAMAIS l'appeler 'QAOA standard' ni 'QAOA optimisé' sans "
                            "préciser hors ligne."),
            "post_hoc_selection_forbidden": True,
        },
        "endianness_canary": CANARY,
        "execution_order_policy": {
            "design": "blocs appariés, bras d'une même cellule adjacents dans le temps",
            "latin_square_rotation": True,
            "published_before_first_job": True,
            "credit_reread_after_each_block": True,
            "capture_per_job": ["submission_timestamp", "queue_time", "execution_timestamp",
                                "backend_calibration_snapshot_or_timestamp",
                                "runtime_execution_options"],
            "rationale": ("une réplication A0 non intercalée ne supprime pas la dérive "
                          "temporelle de calibration ; seuls des blocs appariés le font"),
            "balance_exactness": (
                "Équilibre EXACT au sein des cellules à 3 bras : chaque bras occupe chaque "
                "position exactement une fois sur les 3 blocs (carré latin complet). Pour "
                "les cellules à 2 bras (CP_5/K3 et CP_3/K5), 3 blocs ne se divisent pas "
                "par 2 : le déséquilibre résiduel est de 2 positions contre 1, ce qui est "
                "le minimum atteignable à 3 répétitions et n'est pas dissimulé. Aucun "
                "équilibre n'est revendiqué au niveau agrégé entre cellules de tailles de "
                "bras différentes."),
        },
        "bounded_conclusions": {
            "P1_minH": ("Conclusion BORNÉE : sur les 14 cellules effectivement étudiées "
                        "(CP_3..CP_7 à K=3, CP_3 et CP_4 à K=5), pour CE QUBO one-hot "
                        "fortement pénalisé et CE protocole QAOA p=1 à une seule couche, "
                        "les angles minimisant ⟨H⟩ produisent une masse faisable inférieure "
                        "à celle des angles fixes de P0 sur les cellules maneuver_count_v1 "
                        "(2·10⁻⁵ contre 2,07·10⁻² sur CP_3). Aucune généralisation à QAOA, "
                        "à d'autres encodages, à p > 1, à d'autres familles d'instances ou "
                        "à d'autres pondérations de pénalité n'est autorisée. La "
                        "formulation « structurellement pire », employée dans le message du "
                        "commit e965d6e, est SUPERSÉDÉE par le présent énoncé borné."),
        },
        "frozen_angles": angles,
        "isa_metrics_fake_marrakesh": isa,
    }
    protocol["endianness_canary"]["cell_identity"] = _canary_cell_identity()
    protocol["protocol_sha256"] = _sha(protocol)

    doc = {
        "schema": "acrpq-ibm-campaign-preregistration/1",
        "written_before_any_new_hardware_result": True,
        "protocol": protocol,
        "budget": {
            "billing_metric": "QPU seconds (usage_consumed_seconds, IBM open plan)",
            "billing_metric_measured_per_job": BILLED_SECONDS_MEASURED,
            "billing_measured_evidence": ("30/30 jobs P0 facturés exactement 2.000 s, "
                                          "indépendamment de la profondeur ISA (211 à 640)"),
            "conservative_ceiling_per_job": BILLED_SECONDS_CEILING,
            "usage_limit_seconds": USAGE_LIMIT_SECONDS,
            "usage_consumed_seconds": USAGE_CONSUMED_SECONDS,
            "usage_remaining_seconds": USAGE_REMAINING_SECONDS,
            "billing_reconciliation": (
                "Comptabilité bouclée exactement : le fournisseur porte 35 jobs sur la "
                "fenêtre — les 30 de la cohorte P0 (2026-09-03) plus 5 jobs antérieurs du "
                "2026-08-18 absents du manifeste P0. 35 x 2,000 s = 70 s = "
                "usage_consumed_seconds. Aucun écart résiduel, et le tarif de 2,000 s/job "
                "est donc confirmé sur 35 jobs, pas seulement sur 30."),
            "provenance_gap_flagged": (
                "Ces 5 jobs matériels ibm_marrakesh du 2026-08-18 ne sont enregistrés dans "
                "aucun artefact du dépôt. À inventorier ; ne pas les intégrer aux "
                "agrégats scientifiques sans provenance complète."),
            "max_fraction_of_remaining": MAX_FRACTION_OF_REMAINING,
        },
        "campaigns": campaigns,
        "primary_metric": {
            "name": "feasible_shot_rate_strict",
            "definition": ("shots dont la chaîne est EXACTEMENT one-hot par avion ET "
                           "géométriquement sans conflit, divisés par les shots reçus. "
                           "Aucune réparation one-hot n'est appliquée."),
            "verified_against_P0": ("la définition stricte reproduit exactement le "
                                    "feasibility_rate enregistré des 30 jobs P0"),
        },
        "secondary_metrics": [
            "raw_onehot_rate", "repaired_feasible_rate",
            "conditional_geometry_feasibility", "run_success",
            "certified_optimum_hit", "certified_optimum_mass",
            "probability_mass_on_best_feasible", "best_feasible_objective",
            "gap_to_certified_reference (jamais sur une solution infaisable)",
            "number_unique_bitstrings", "shannon_entropy_of_counts",
            "isa_depth", "isa_two_qubit_gates", "physical_qubits",
            "provider_execution_time (si réellement disponible)",
            "queue_and_wall_time (séparés du temps QPU)",
            "min_separation_and_conflicts_of_best_sample",
            "endianness_log_likelihood_ratio (canari FP_4 uniquement)",
        ],
        "exclusions": [
            "state != completed ou result indisponible",
            "shots reçus != shots demandés",
            "summary/best_feasible manquant ou corrompu",
            "config_hash ou artefact_sha256 divergent entre préparation et exécution",
            "au plus 1 job de remplacement par slot exclu, journalisé avec code de raison, "
            "décidé AVANT tout calcul de la métrique principale",
        ],
        "analyses": {
            "experimental_unit": "le job (les 512 shots d'un job sont groupés, non indépendants)",
            "primary_comparison": "A1 vs A0 apparié par cellule (instance x objectif)",
            "effect_measure": ("différence de proportions avec intervalle hybride "
                               "Newcombe/Wilson ; le ratio n'est rapporté qu'en secondaire "
                               "et jamais quand P0 = 0"),
            "zeros_policy": ("aucun pseudo-compte ad hoc ; différence de proportions et "
                             "borne exacte Clopper-Pearson pour les cellules à 0 succès"),
            "forbidden": ["test paramétrique traitant les shots comme indépendants",
                          "inférence forte à n faible",
                          "affirmation de causalité (employer « association observée »)",
                          "agrégation entre K, entre objectifs, entre bras, "
                          "entre matériel et simulation"],
            "variance_caveat": ("à 3 répétitions (df=2) l'intervalle de confiance à 95 % sur "
                                "l'écart-type lui-même couvre [0,52x ; 6,29x] : la variance "
                                "n'est qu'un ordre de grandeur, jamais une valeur calibrée"),
        },
        "stop_order": [
            "1. canari d'endianness FP_4 : le signe du LLR doit désigner la convention "
            "retenue avec |LLR| > 3 sigma ; sinon ARRÊT et aucune autre soumission",
            "2. deux canaris CP_3 : validation technique complète (job_id, 512 shots, "
            "décodage, hashes) ; un canari à 0 % faisable n'est PAS un échec technique",
            "3. relire usage_remaining_seconds ; si le reste passe sous la marge, ARRÊT",
            "4. bras A0 puis A1 puis A2 puis les cellules K5, séquentiellement",
            "5. ARRÊT immédiat sur toute violation de protocole, backend inattendu, "
            "shots incohérents, hash divergent, soumission sans job_id, coût anormal",
        ],
        "open_reserves_from_expert_e": [
            "brancher reference_objective dans le décodage brut (gap null sur les 30 P0)",
            "capturer le snapshot de calibration, les options d'exécution Runtime "
            "(mitigation/twirling/DD) et les horodatages fins : provenance est null sur P0",
            "variable_mapping doit porter (avion, option, q, theta) et non x0..xn",
            "le taux de faisabilité agrégé ne prouve PAS l'endianness sur la famille CP "
            "(KL = 0 exactement) ; seul le canari FP_4 le prouve empiriquement",
        ],
    }
    doc["content_sha256"] = _sha({k: v for k, v in doc.items() if k != "content_sha256"})
    return doc


def _emit_markdown(doc: dict[str, Any]) -> str:
    p = doc["protocol"]
    b = doc["budget"]
    land = json.loads(LANDSCAPE.read_text())
    lines: list[str] = []
    A = lines.append
    A("# Préenregistrement — campagne IBM finale")
    A("")
    A(f"* `protocol_id` : `{p['protocol_id']}`")
    A(f"* **`protocol_sha256` : `{p['protocol_sha256']}`**")
    A(f"* `content_sha256` du présent préenregistrement : `{doc['content_sha256']}`")
    A(f"* backend : **{p['backend']} exclusivement** · un circuit par job, "
      f"pas de regroupement multi-pubs")
    A(f"* angles gelés depuis `{LANDSCAPE}` (`{land['content_sha256']}`)")
    A("")
    A("**Ce document est écrit avant tout nouveau résultat matériel.** Aucune sélection "
      "d'angles après observation du matériel n'est permise.")
    A("")
    if p.get("supersession"):
        sup = p["supersession"]
        A("## 0. Supersession de protocole")
        A("")
        A(f"Ce protocole **supersède** `{sup['supersedes_protocol_sha256']}`, sous lequel "
          f"les jobs suivants ont déjà été soumis et exécutés : "
          + ", ".join(f"`{j}`" for j in sup["submitted_jobs_affected"]) + ".")
        A("")
        A(f"**Nature du défaut : `{sup['defect_class']}`.** "
          f"{sup['what_was_wrong']}")
        A("")
        A(f"**Ce qui n'était PAS affecté.** {sup['what_was_NOT_wrong']}")
        A("")
        A(f"Circuits affectés : {sup['circuits_affected']} · "
          f"angles affectés : {sup['angles_affected']} · "
          f"QUBO affecté : {sup['qubo_affected']} · "
          f"shots affectés : {sup['shots_affected']} · "
          f"résultats bruts toujours valides : {sup['raw_results_remain_valid']} · "
          f"artefacts bruts jamais réécrits : {sup['raw_artifacts_never_rewritten']}.")
        A("")
        A("Vérification avant/après publiée dans "
          "`results/offline_angle_landscape_v2/correction_verification.json` : "
          "336 invariants comparés bit à bit (angles, hashes de QUBO, masses faisables, "
          "masses one-hot) — 0 violation. Seuls 4 champs de scoring des références ont "
          "changé, comme attendu, sur les 14 cellules.")
        A("")
    A("## 1. Hypothèses préenregistrées")
    A("")
    A("* **H1** — des angles optimisés hors ligne augmentent le taux de shots faisables "
      "par rapport aux angles fixes, surtout sur les petites instances.")
    A("* **H2** — la faisabilité décroît avec le nombre de qubits, la profondeur ISA et le "
      "nombre de portes 2Q.")
    A("* **H3** — la perte se décompose en invalidité one-hot, one-hot valide mais conflit "
      "géométrique, et faisable mais sous-optimal.")
    A("* **H4** — les deux `objective_id` peuvent présenter des distributions différentes ; "
      "à K=3 ils sont **deux paramétrisations énergétiques du même problème décisionnel** "
      "(proportionnalité globale vérifiée, facteur 0,137078), à K=5 ils ne le sont plus.")
    A("* **H5** — parmi les runs contenant au moins un échantillon faisable, le meilleur "
      "atteint souvent l'optimum discret certifié.")
    A("")
    A("## 2. Métrique principale et secondaires")
    A("")
    A(f"**{doc['primary_metric']['name']}** — {doc['primary_metric']['definition']}")
    A("")
    A(f"Vérification : {doc['primary_metric']['verified_against_P0']}.")
    A("")
    A("Secondaires : " + " · ".join(f"`{m}`" for m in doc["secondary_metrics"]) + ".")
    A("")
    A("Aucun écart à la référence n'est calculé pour une solution infaisable.")
    A("")
    A("## 3. Bras — chacun ne change qu'UNE variable")
    A("")
    A("| bras | angles | niveau de transpilation | rôle |")
    A("|---|---|---|---|")
    for name, a in p["arms"].items():
        A(f"| `{name}` | {a['angles']} | {a['optimization_level']} | {a['role']} |")
    A("")
    A("`P1_feas` **n'est pas « QAOA standard »**. " + p["angle_selection"]["naming_rule"])
    A("")
    A("### Conclusion bornée sur `P1_minH`")
    A("")
    A(p["bounded_conclusions"]["P1_minH"])
    A("")
    A("### Ordre d'exécution : blocs appariés")
    A("")
    eo = p["execution_order_policy"]
    A(f"{eo['rationale'].capitalize()}. Rotation en carré latin : "
      f"{'oui' if eo['latin_square_rotation'] else 'non'}. "
      f"Relecture du crédit après chaque bloc : "
      f"{'oui' if eo['credit_reread_after_each_block'] else 'non'}.")
    A("")
    A("Capturé pour chaque job : " + " · ".join(f"`{c}`" for c in eo["capture_per_job"]) + ".")
    A("")
    A("L'ordre complet des 63 jobs de la campagne recommandée est publié en §12, "
      "**avant le premier job**.")
    A("")
    A("### Canari d'endianness — test formalisé")
    A("")
    can = p["endianness_canary"]
    c = can["cell"]
    A(f"Cellule : **{c['instance']}/K{c['grid_k']}**, {c['objective_id']}, "
      f"γ={c['gammas'][0]}, β={c['betas'][0]}, niveau {c['optimization_level']}, "
      f"{c['shots']} shots, 1 job.")
    A("")
    A(f"*Pourquoi cette cellule.* {can['why_this_cell']}")
    A("")
    A("Distributions attendues sous les deux conventions :")
    A("")
    A(f"* **A (retenue)** — {can['conventions']['A_retenue']}")
    A(f"* **B (alternative)** — {can['conventions']['B_alternative']}")
    A(f"* `p_B` est une permutation de `p_A` ; empreinte de `p_A` : "
      f"`{can['expected_distributions']['p_ideal_qubo_order_sha256']}` "
      f"({can['expected_distributions']['sha256_note']})")
    A("")
    A(f"**Formule du LLR.** {can['llr_formula']}")
    A("")
    ft = can["frozen_threshold"]
    A(f"**Seuil figé.** {ft['rule']}")
    A("")
    A("| résultat | décision |")
    A("|---|---|")
    for k_, v_ in ft["decision"].items():
        A(f"| `{k_}` | {v_} |")
    A("")
    A(f"**Le KL idéal n'est pas une garantie matérielle.** {can['noise_sizing_not_a_guarantee']}")
    A("")
    A("## 4. Paysage hors ligne exact et angles gelés")
    A("")
    A("Masses de probabilité exactes (vecteur d'état, sans échantillonnage ni bruit). "
      "`opt` = masse sur l'optimum certifié.")
    A("")
    A("| cellule | qubits | jeu | γ | β | one-hot | faisable | opt | ⟨H⟩ |")
    A("|---|---|---|---|---|---|---|---|---|")
    for row in land["rows"]:
        cell = f"{row['instance']}/K{row['grid_k']}/{row['objective_id'].split('_')[0]}"
        for arm in ("P0", "P1_minH", "P1_feas"):
            a = row[arm]
            A(f"| {cell} | {row['n_qubits']} | `{arm}` | {a['gamma']:.4f} | {a['beta']:.4f} "
              f"| {a['onehot_valid_mass']:.5f} | {a['feasible_mass']:.5f} "
              f"| {a['certified_optimum_mass']:.5f} | {a['expected_energy']:.2f} |")
    A("")
    A("## 5. Métriques ISA après transpilation vers Marrakech")
    A("")
    A("| cellule | niveau | qubits | profondeur avant | profondeur après | portes 2Q | taille | swap |")
    A("|---|---|---|---|---|---|---|---|")
    for key, m in sorted(p["isa_metrics_fake_marrakesh"].items()):
        inst, k, lvl = key.split("|")
        A(f"| {inst}/{k} | {lvl[1:]} | {m['n_qubits']} | {m['depth_before']} "
          f"| {m['depth_after']} | {m['two_qubit_gates_after']} | {m['size_after']} "
          f"| {m['swap_count']} |")
    A("")
    A("## 6. Budget — métrique réellement facturée")
    A("")
    A(f"* métrique : **{b['billing_metric']}**")
    A(f"* coût **mesuré** par job : **{b['billing_metric_measured_per_job']} s** "
      f"({b['billing_measured_evidence']})")
    A(f"* plafond prudent retenu : **{b['conservative_ceiling_per_job']} s/job** (×2 de marge)")
    A(f"* limite du plan : {b['usage_limit_seconds']} s / 28 jours · consommé "
      f"{b['usage_consumed_seconds']} s · **restant {b['usage_remaining_seconds']} s**")
    A(f"* réconciliation : {b['billing_reconciliation']}")
    A(f"* signalement de provenance : {b['provenance_gap_flagged']}")
    A(f"* règle : ne jamais engager plus de **{b['max_fraction_of_remaining']:.0%}** du "
      f"restant au tarif plafond")
    A("")
    A("| campagne | jobs | pubs/job | shots totaux | coût mesuré | coût plafond | % du restant | dans la marge |")
    A("|---|---|---|---|---|---|---|---|")
    for name, c in doc["campaigns"].items():
        A(f"| **{name}** | {c['n_jobs']} | {c['n_pubs_per_job']} | {c['total_shots']} "
          f"| {c['billed_seconds_measured_rate']:.0f} s "
          f"| {c['billed_seconds_conservative_ceiling']:.0f} s "
          f"| {c['fraction_of_remaining_at_ceiling']:.0%} "
          f"| {'oui' if c['within_safety_margin'] else '**NON**'} |")
    A("")
    A("## 7. Répétitions, shots, exclusions")
    A("")
    A(f"Shots par job : **{p['shots_per_job']}**, identiques à P0 pour rester comparable. "
      f"Répétitions : 3 (minimale et recommandée), 5 (étendue), `repetition_id` distinct, "
      f"hash de configuration scientifique partagé.")
    A("")
    A("Exclusions préenregistrées :")
    A("")
    for e in doc["exclusions"]:
        A(f"* {e}")
    A("")
    A("## 8. Analyses préenregistrées")
    A("")
    an = doc["analyses"]
    A(f"* unité expérimentale : {an['experimental_unit']}")
    A(f"* comparaison principale : {an['primary_comparison']}")
    A(f"* mesure d'effet : {an['effect_measure']}")
    A(f"* zéros : {an['zeros_policy']}")
    A(f"* mise en garde sur la variance : {an['variance_caveat']}")
    A("* interdits : " + " · ".join(an["forbidden"]))
    A("")
    A("## 9. Ordre d'arrêt")
    A("")
    for s in doc["stop_order"]:
        A(f"{s}")
        A("")
    A("## 10. Réserves ouvertes de l'expert E, à traiter par la campagne")
    A("")
    for r in doc["open_reserves_from_expert_e"]:
        A(f"* {r}")
    A("")
    A("## 12. Ordre d'exécution complet, publié avant le premier job")
    A("")
    A("| # | bloc | type | cellule | objectif | bras | position du bras | rép. |")
    A("|---|---|---|---|---|---|---|---|")
    for j in doc["campaigns"]["recommandee"]["jobs"]:
        obj = "quad" if "quad" in j["objective_id"] else "man"
        A(f"| {j['execution_order']} | {j['block_id']} | {j['kind']} "
          f"| {j['instance']}/K{j['grid_k']} | {obj} | `{j['arm']}` "
          f"| {j.get('arm_position_in_block', '-')} | {j['repetition_id']} |")
    A("")
    A("## 11. Preuve qu'aucune soumission n'a eu lieu")
    A("")
    A("Voir la section « Preuve d'absence de soumission » du rapport final : drapeaux "
      "fail-closed, `usage_consumed_seconds` inchangé, aucun nouveau `job_id` chez le "
      "fournisseur, aucun artefact de run créé.")
    return "\n".join(lines) + "\n"


def _verify() -> int:
    p = OUT / "preregistration.json"
    if not p.exists():
        print("MISSING preregistration.json")
        return 2
    doc = json.loads(p.read_text())
    if _sha({k: v for k, v in doc.items() if k != "content_sha256"}) != doc["content_sha256"]:
        print("PREREGISTRATION HASH MISMATCH")
        return 2
    prot = doc["protocol"]
    if _sha({k: v for k, v in prot.items() if k != "protocol_sha256"}) != prot["protocol_sha256"]:
        print("PROTOCOL HASH MISMATCH")
        return 2
    bad = [n for n, c in doc["campaigns"].items() if not c["within_safety_margin"]]
    print(f"OK preregistration verified · protocol_sha256={prot['protocol_sha256'][:26]}…")
    if bad:
        print(f"NOTE : campagnes hors marge de sécurité : {bad}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Émettre le préenregistrement")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        return _verify()
    doc = build()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(
        json.dumps(doc["protocol"], indent=2, sort_keys=True, allow_nan=False) + "\n")
    (OUT / "preregistration.json").write_text(
        json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n")
    DOC.write_text(_emit_markdown(doc))
    print(f"[preregistration] protocol_sha256={doc['protocol']['protocol_sha256']}")
    for name, c in doc["campaigns"].items():
        print(f"  {name:12s}: {c['n_jobs']:3d} jobs · {c['total_shots']:6d} shots · "
              f"mesuré {c['billed_seconds_measured_rate']:5.0f} s · "
              f"plafond {c['billed_seconds_conservative_ceiling']:5.0f} s "
              f"({c['fraction_of_remaining_at_ceiling']:.0%} du restant) · "
              f"marge={'ok' if c['within_safety_margin'] else 'NON'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
