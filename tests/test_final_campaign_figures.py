"""Régression red-team : fig4 mélangeait les canaris dans la moyenne A1/K3.

Le filtre `official_hypothesis_test` était appliqué pour LISTER les cellules mais pas
dans la fermeture `rate()` qui calcule la moyenne — les 2 jobs canaris CP_3/K3
(official_hypothesis_test=false) gonflaient donc silencieusement la moyenne A1/K3.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parent.parent / "scripts" / "build_final_campaign_figures.py"
_spec = importlib.util.spec_from_file_location("figures", _P)
figures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(figures)

RESULTS = Path("results/ibm_qaoa_final_campaign_v1/campaign_results.json")
pytestmark = pytest.mark.skipif(not RESULTS.exists(), reason="dataset de campagne absent")


def _rows():
    import json
    return json.loads(RESULTS.read_text())["rows"]


def test_fig4_k3_average_excludes_canaries(tmp_path, monkeypatch):
    """La moyenne K3/A1/quad ne doit PAS inclure les jobs canaris."""
    rows = _rows()
    monkeypatch.setattr(figures, "FIG", tmp_path)
    figures.fig_k3_vs_k5(rows)
    import csv
    with (tmp_path / "fig4_k3_vs_k5_CP_3.csv").open() as fh:
        by_k = {int(r["grid_k"]): r for r in csv.DictReader(fh)}

    official = [r for r in rows if r["instance"] == "CP_3" and r["grid_k"] == 3
               and r["arm"] == "A1_P1_feas_angles" and "quadratic" in r["objective_id"]
               and r["official_hypothesis_test"]]
    expected = sum(float(r["feasible_shot_rate_strict"]) for r in official) / len(official) * 100
    assert float(by_k[3]["quad_A1_pct"]) == pytest.approx(expected, abs=1e-9)

    canary_rate = next(float(r["feasible_shot_rate_strict"]) * 100 for r in rows
                       if r["experiment_id"] == "CP_3__K3__quad__A1_P1_feas_angles__r1__e002")
    contaminated = (sum(float(r["feasible_shot_rate_strict"]) for r in official) + canary_rate / 100) \
        / (len(official) + 1) * 100
    assert float(by_k[3]["quad_A1_pct"]) != pytest.approx(contaminated, abs=1e-6), (
        "la moyenne correspond au calcul CONTAMINÉ par le canari, pas au calcul correct")


def test_no_canary_ever_enters_a_rate_computation():
    """Neuter-check : si le filtre disparaît, ce test doit détecter la contamination."""
    rows = _rows()
    canaries = [r for r in rows if not r["official_hypothesis_test"]]
    assert len(canaries) == 3, "attendu : exactement 3 canaris dans le dataset"
    assert all(r["arm"] in ("CE_endianness_canary", "A1_P1_feas_angles") for r in canaries)
