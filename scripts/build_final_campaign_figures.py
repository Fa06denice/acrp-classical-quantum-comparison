#!/usr/bin/env python3
"""Figures de la campagne IBM finale (Phase 8), sobres, sans décoration pseudo-quantique.

Sources : results/ibm_qaoa_final_campaign_v1/{campaign_results.json, statistics.json}.
Chaque figure exporte SVG + PNG + les données sources (CSV) dans
results/ibm_qaoa_final_campaign_v1/figures/.

Conventions : légendes en français, n affiché, K3/K5 jamais mélangés dans un même point,
les deux objectifs distingués par marqueur, zéro visible, aucun axe tronqué trompeur.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/ibm_qaoa_final_campaign_v1")
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

COLOR_QUAD = "#1f5aa8"
COLOR_MAN = "#c65a11"
COLOR_A0 = "#6b6b6b"
COLOR_A1 = "#1f8a4c"


def _save(fig, name: str, rows: list[dict]) -> None:
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    if rows:
        with (FIG / f"{name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    plt.close(fig)


def fig_a0_a1_paired(stats: dict) -> None:
    """Correction Codex : ne montre plus d'intervalle sur shots poolés (n=1536, faux).

    Affiche les 3 paires (A0_r, A1_r) individuelles par cellule, reliées par une ligne, et
    la différence appariée d_r à droite avec les 3 points individuels (jamais un
    intervalle de confiance calculé dessus à n=3 — seulement moyenne/médiane comme repères
    descriptifs)."""
    per_cell = stats["job_level_paired_analysis"]["per_cell"]
    labels = list(per_cell)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    rows_csv = []
    for i, label in enumerate(labels):
        c = per_cell[label]
        a0s, a1s, diffs = c["A0_rates"], c["A1_rates"], c["paired_differences_d_r"]
        for rep, a0, a1, d in zip(c["repetition_ids"], a0s, a1s, diffs):
            ax1.plot([i - 0.12, i + 0.12], [a0 * 100, a1 * 100], color="#bbbbbb",
                    linewidth=1, zorder=1)
            rows_csv.append({"cell": label, "repetition_id": rep, "A0_pct": a0 * 100,
                            "A1_pct": a1 * 100, "diff_pct": d * 100})
        ax1.scatter([i - 0.12] * len(a0s), [v * 100 for v in a0s], color=COLOR_A0,
                   s=45, zorder=2, label="A0 (n=3 jobs)" if i == 0 else None)
        ax1.scatter([i + 0.12] * len(a1s), [v * 100 for v in a1s], color=COLOR_A1,
                   s=45, zorder=2, label="A1 (n=3 jobs)" if i == 0 else None)

        d = per_cell[label]["descriptive"]
        ax2.scatter(diffs, [i] * len(diffs), color="#333333", s=35, zorder=2,
                   label="différences individuelles d_r" if i == 0 else None)
        ax2.scatter([d["mean"]], [i], color="#c62828", marker="D", s=55, zorder=3,
                   label="moyenne (repère descriptif, PAS un IC)" if i == 0 else None)

    def _short(c: str) -> str:
        inst, k, oid = c.split("/")
        obj = "quad" if "quadratic" in oid else "man"
        return f"{inst}\n{k} {obj}"
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels([_short(c) for c in labels], fontsize=7.5)
    ax1.set_ylabel("Taux de faisabilité stricte, par job (%)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.axhline(0, color="black", linewidth=0.6)

    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels([])
    ax2.set_xlabel("d_r = A1 − A0 par répétition (points de %)\n"
                  "3 points par cellule — AUCUN intervalle de confiance à n=3")
    ax2.legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Campagne IBM finale (ibm_marrakesh) — A0 vs A1, JOB comme unité "
                "expérimentale (n=3/bras)\ngauche : taux par job individuel — droite : "
                "différences appariées d_r, descriptif seulement",
                fontsize=9.5, y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    _save(fig, "fig1_a0_vs_a1_paired", rows_csv)


def fig_feasibility_vs_qubits(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    by_q: dict[int, list[float]] = {}
    for r in rows:
        if not r["official_hypothesis_test"] or r["arm"] not in ("A0_P0_replication", "A1_P1_feas_angles"):
            continue
        q = int(r["logical_qubits"])
        rate = float(r["feasible_shot_rate_strict"] or 0) * 100
        marker = "o" if "quadratic" in r["objective_id"] else "^"
        color = COLOR_A0 if r["arm"] == "A0_P0_replication" else COLOR_A1
        jitter = -0.15 if r["arm"] == "A0_P0_replication" else 0.15
        ax.scatter(q + jitter, rate, marker=marker, color=color, alpha=0.75, s=45,
                  edgecolors="black", linewidths=0.4)
        by_q.setdefault((q, r["arm"]), []).append(rate)
    for (q, arm), vals in by_q.items():
        jitter = -0.15 if arm == "A0_P0_replication" else 0.15
        med = sorted(vals)[len(vals) // 2]
        ax.plot([q + jitter - 0.25, q + jitter + 0.25], [med, med],
               color=(COLOR_A0 if arm == "A0_P0_replication" else COLOR_A1), linewidth=2)
    ax.set_xlabel("Qubits logiques (K=3 uniquement — K=5 exclu, grilles non comparables)")
    ax.set_ylabel("Taux de faisabilité stricte (%)")
    ax.set_title("Faisabilité vs taille, K=3 — points individuels + médiane par bras\n"
                "cercle=quadratic_control_cost_v1, triangle=maneuver_count_v1")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_A0,
                      markeredgecolor="black", markersize=8, label="A0 (angles fixes)"),
              Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_A1,
                     markeredgecolor="black", markersize=8, label="A1 (angles optimisés)"),
              Line2D([0], [0], marker="^", color="w", markerfacecolor="grey",
                     markeredgecolor="black", markersize=8, label="maneuver_count_v1"),
              Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
                     markeredgecolor="black", markersize=8, label="quadratic_control_cost_v1")]
    ax.legend(handles=handles, fontsize=7.5, loc="upper right")
    ax.set_ylim(bottom=-0.5)
    csv_rows = [{"logical_qubits": r["logical_qubits"], "arm": r["arm"],
                "objective_id": r["objective_id"],
                "feasible_shot_rate_pct": float(r["feasible_shot_rate_strict"] or 0) * 100,
                "experiment_id": r["experiment_id"]}
               for r in rows if r["official_hypothesis_test"]
               and r["arm"] in ("A0_P0_replication", "A1_P1_feas_angles") and r["grid_k"] == 3]
    _save(fig, "fig2_feasibility_vs_qubits_K3", csv_rows)


def fig_a2_transpilation(stats: dict) -> None:
    comps = stats["a2_transpilation_level_analysis_SEPARATE_FROM_A0_A1"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(comps))
    for i, c in enumerate(comps):
        l0 = [v * 100 for v in c["A0_level0_rates"]]
        l2 = [v * 100 for v in c["A2_level2_rates"]]
        ax.scatter([i - 0.1] * len(l0), l0, color=COLOR_A0, marker="s", s=40,
                  edgecolors="black", linewidths=0.4,
                  label="niveau 0 (angles P0)" if i == 0 else None)
        ax.scatter([i + 0.1] * len(l2), l2, color="#8e24aa", marker="D", s=40,
                  edgecolors="black", linewidths=0.4,
                  label="niveau 2 (angles P0)" if i == 0 else None)
    ax.set_xticks(list(x))
    ax.set_xticklabels([c["cell"].replace("/K", "\nK") for c in comps], fontsize=7.5)
    ax.set_ylabel("Taux de faisabilité stricte (%)")
    ax.set_title("A2 — effet du niveau de transpilation SEUL (angles P0 inchangés)\n"
                "profondeur réduite de 42% à 28% selon la cellule, sans gain systématique")
    ax.legend(fontsize=8)
    csv_rows = []
    for c in comps:
        for v, lvl in [(v, 0) for v in c["A0_level0_rates"]] + [(v, 2) for v in c["A2_level2_rates"]]:
            csv_rows.append({"cell": c["cell"], "optimization_level": lvl,
                            "feasible_shot_rate_pct": v * 100})
    _save(fig, "fig3_a2_transpilation_level", csv_rows)


def fig_k3_vs_k5(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cells = sorted({(r["instance"], r["grid_k"]) for r in rows
                   if r["official_hypothesis_test"] and r["instance"] == "CP_3"})
    labels, quad_a0, quad_a1, man_a0, man_a1 = [], [], [], [], []
    for inst, k in cells:
        def rate(arm, oid):
            # Bug trouve par le red-team (verifie et reproduit independamment) : ce filtre
            # excluait bien K/arm/objectif mais PAS les canaris, gonflant la moyenne K3/A1
            # avec les 2 jobs canaris CP_3/K3 (e002, e003, official_hypothesis_test=false).
            vals = [float(r["feasible_shot_rate_strict"] or 0) for r in rows
                    if r["instance"] == inst and r["grid_k"] == k and r["arm"] == arm
                    and r["objective_id"] == oid and r["official_hypothesis_test"]]
            return (sum(vals) / len(vals) * 100) if vals else 0.0
        labels.append(f"K={k}")
        quad_a0.append(rate("A0_P0_replication", "quadratic_control_cost_v1"))
        quad_a1.append(rate("A1_P1_feas_angles", "quadratic_control_cost_v1"))
        man_a0.append(rate("A0_P0_replication", "maneuver_count_v1"))
        man_a1.append(rate("A1_P1_feas_angles", "maneuver_count_v1"))
    x = range(len(labels))
    w = 0.2
    ax.bar([i - 1.5 * w for i in x], quad_a0, w, color=COLOR_QUAD, alpha=0.5, label="quad, A0")
    ax.bar([i - 0.5 * w for i in x], quad_a1, w, color=COLOR_QUAD, label="quad, A1")
    ax.bar([i + 0.5 * w for i in x], man_a0, w, color=COLOR_MAN, alpha=0.5, label="maneuver, A0")
    ax.bar([i + 1.5 * w for i in x], man_a1, w, color=COLOR_MAN, label="maneuver, A1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Taux de faisabilité stricte, moyenne (%)")
    ax.set_title("CP_3 — K3 vs K5, jamais comparés comme si c'était la même grille\n"
                "(K5 : espace de recherche 5³ vs 3³, coûts par option différents)")
    ax.legend(fontsize=8)
    csv_rows = [{"grid_k": k, "quad_A0_pct": a, "quad_A1_pct": b, "man_A0_pct": c_, "man_A1_pct": d}
               for (_, k), a, b, c_, d in zip(cells, quad_a0, quad_a1, man_a0, man_a1)]
    _save(fig, "fig4_k3_vs_k5_CP_3", csv_rows)


def main() -> int:
    rows = json.loads((ROOT / "campaign_results.json").read_text())["rows"]
    stats = json.loads((ROOT / "statistics.json").read_text())
    fig_a0_a1_paired(stats)
    fig_feasibility_vs_qubits(rows)
    fig_a2_transpilation(stats)
    fig_k3_vs_k5(rows)
    print(f"4 figures écrites dans {FIG}/ (SVG+PNG+CSV chacune)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
