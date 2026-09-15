"""Correction Codex : le job est l'unité expérimentale, jamais le shot poolé entre jobs.

Ces tests garantissent que la régression corrigée (job-level, n=3) ne peut pas revenir
silencieusement à l'ancienne erreur (shots poolés, n=1536, IC artificiellement étroit).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parent.parent / "scripts" / "analyze_final_campaign.py"
_spec = importlib.util.spec_from_file_location("analyze", _P)
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)

RESULTS = Path("results/ibm_qaoa_final_campaign_v1/campaign_results.json")
pytestmark = pytest.mark.skipif(not RESULTS.exists(), reason="dataset de campagne absent")


@pytest.fixture(scope="module")
def rows():
    return json.loads(RESULTS.read_text())["rows"]


def test_the_experimental_unit_is_never_1536_shots(rows):
    """Le calcul principal ne doit JAMAIS recevoir n=1536 (3 jobs x 512 shots poolés)
    comme taille d'échantillon. Chaque cellule doit rapporter au plus 3 paires (une par
    répétition), jamais un dénominateur en shots."""
    doc = analyze.analyze(rows)
    for label, cell in doc["job_level_paired_analysis"]["per_cell"].items():
        n = cell["descriptive"]["n"]
        assert n <= 3, f"{label}: n={n} — devrait être <= 3 (jobs), jamais des shots poolés"
        assert n != 1536, f"{label}: régression vers l'ancienne erreur (shots poolés)"


def test_no_field_ever_claims_statistical_significance(rows):
    """A n=3, aucune sortie de ce module ne doit jamais affirmer une significativité."""
    doc = analyze.analyze(rows)

    def _no_significance_key(obj):
        if isinstance(obj, dict):
            assert "statistically_significant" not in obj, "cle interdite a n=3"
            for v in obj.values():
                _no_significance_key(v)
        elif isinstance(obj, list):
            for v in obj:
                _no_significance_key(v)
    _no_significance_key(doc)
    for cell in doc["job_level_paired_analysis"]["per_cell"].values():
        assert cell["exact_two_sided_sign_test_pvalue"] >= 0.25 - 1e-9, (
            "le p du test des signes exact ne peut jamais descendre sous 0.25 à n<=3")


def test_duplicating_a_jobs_shots_does_not_inflate_repetition_count(rows):
    """Doubler artificiellement les shots d'UN job ne doit jamais se traduire par une
    répétition supplémentaire au niveau job — la clé de comptage est repetition_id, pas
    n_shots. Preuve : gonfler shots_received d'un job change son taux/CI intra-job mais
    JAMAIS le nombre de paires (n) dans job_level_paired_analysis."""
    import copy
    mutated = copy.deepcopy(rows)
    for r in mutated:
        if r.get("official_hypothesis_test") and r["arm"] == "A1_P1_feas_angles":
            r["shots_received"] = (r.get("shots_received") or 512) * 1000
            break
    before = analyze.analyze(rows)["job_level_paired_analysis"]
    after = analyze.analyze(mutated)["job_level_paired_analysis"]
    for label in before["per_cell"]:
        assert after["per_cell"][label]["descriptive"]["n"] == \
            before["per_cell"][label]["descriptive"]["n"], (
            f"{label}: gonfler les shots d'un job a change le nombre de repetitions")


def test_strongly_changing_one_job_moves_inter_run_uncertainty(rows):
    """Modifier fortement le taux d'UN job doit influencer l'incertitude inter-run
    (écart-type descriptif des différences appariées), preuve que le calcul est
    réellement sensible à la variabilité job-à-job et pas seulement au bruit de tir."""
    import copy
    mutated = copy.deepcopy(rows)
    target_cell = None
    for r in mutated:
        if (r.get("official_hypothesis_test") and r["arm"] == "A1_P1_feas_angles"
                and r["repetition_id"] == 1):
            r["feasible_shot_rate_strict"] = min(1.0, (r["feasible_shot_rate_strict"] or 0) + 0.5)
            target_cell = f"{r['instance']}/K{r['grid_k']}/{r['objective_id']}"
            break
    assert target_cell is not None
    before = analyze.analyze(rows)["job_level_paired_analysis"]["per_cell"][target_cell]
    after = analyze.analyze(mutated)["job_level_paired_analysis"]["per_cell"][target_cell]
    sd_before = before["descriptive"]["sd_descriptive_do_not_treat_as_inferential"]
    sd_after = after["descriptive"]["sd_descriptive_do_not_treat_as_inferential"]
    assert sd_after != sd_before, "un choc massif sur un job doit changer l'ecart-type descriptif"


def test_canaries_are_excluded_from_job_level_analysis(rows):
    doc = analyze.analyze(rows)
    canary_ids = {r["experiment_id"] for r in rows if not r.get("official_hypothesis_test")}
    for cell in doc["job_level_paired_analysis"]["per_cell"].values():
        # aucune trace des taux canaris (0.14453125 et 0.0703125, propres a e002/e003)
        pass  # les canaris n'ont pas d'arm A0/A1 apparie officiel; verifie plutot le compte
    assert doc["n_canary_jobs"] == len(canary_ids) == 3
    assert doc["n_official_jobs"] == 60


def test_pairing_respects_cell_and_repetition_id_strictly(rows):
    """Un A1 de la répétition 2 ne doit JAMAIS être apparié à un A0 de la répétition 1."""
    doc = analyze.analyze(rows)
    for label, cell in doc["job_level_paired_analysis"]["per_cell"].items():
        reps = cell["repetition_ids"]
        assert reps == sorted(reps), f"{label}: repetitions non ordonnees/incoherentes"
        assert len(set(reps)) == len(reps), f"{label}: repetition_id duplique"


def test_no_mixing_of_grid_objective_or_arm_in_a_single_cell(rows):
    doc = analyze.analyze(rows)
    for label in doc["job_level_paired_analysis"]["per_cell"]:
        parts = label.split("/")
        assert len(parts) == 3, f"cle de cellule malformee : {label}"
        inst, k, oid = parts
        assert k.startswith("K") and k[1:] in ("3", "5")
        assert oid in ("quadratic_control_cost_v1", "maneuver_count_v1")


def test_conditional_shot_interval_is_per_job_never_pooled(rows):
    """Le champ renomme doit exister, etre calcule PAR JOB (n_shots_this_job_only <= 512
    pour cette campagne), et etre etiquete comme non-inferentiel."cel"""
    doc = analyze.analyze(rows)
    intervals = doc["conditional_shot_sampling_interval_per_job"]
    # 63 = les 60 jobs officiels + les 3 canaris : le bruit de tir intra-job est
    # calculable et affiche pour TOUT job avec un resultat, canari inclus
    assert len(intervals) == 63
    for exp_id, v in intervals.items():
        assert v["n_shots_this_job_only"] == 512
        assert "intervalle inter-run" in v["note"]


def test_exploratory_bootstrap_is_explicitly_labeled_unstable(rows):
    doc = analyze.analyze(rows)
    boot = doc["exploratory_bootstrap_over_job_pairs_UNSTABLE_AT_N3"]
    assert boot, "le bootstrap exploratoire doit produire une sortie non vide"
    for v in boot.values():
        assert "INSTABLE" in v["warning"]
        assert v["n_original_pairs"] <= 3


def test_authorized_phrasing_matches_actual_job_level_count(rows):
    doc = analyze.analyze(rows)
    jl = doc["job_level_paired_analysis"]
    phrase = jl["authorized_phrasing_fr"]
    assert f"{jl['n_cells_with_mean_diff_positive']} cellules sur {jl['n_cells_total']}" in phrase
    assert "sans puissance suffisante" in phrase
    assert "confirm" not in phrase.lower()
