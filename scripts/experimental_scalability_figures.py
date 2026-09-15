#!/usr/bin/env python
"""EXPERIMENTAL — the four minimal thesis figures of the instance-scalability study.

Each figure: source CSV, SVG, PNG (300 dpi), French axes with units, legend, no pooling
across families, caption ready for the thesis, sha256 of the source CSV in captions.json.

1. fig1_largest_component_vs_n: largest any-option K3 component vs n, per family.
2. fig2_width_monolithic_vs_decomposed: 3n vs 3·|largest component| (logical qubits).
3. fig3_accessibility_classes: class counts per family (schema-2 labels).
4. fig4_method_schematic: continuous -> global one-hot -> adaptive precision -> width decomposition.

Flags: experimental=true, official_benchmark=false, real_qpu=false.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from acrpq.experimental import instance_scalability as isc  # noqa: E402
from acrpq.experimental import provenance as prov  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "experimental_instance_scalability_v1"
OUT = SRC / "figures"
FAMILY_ORDER = ["CP", "FP", "GP", "RCP", "RCP_FL"]
FAMILY_LABEL = {"CP": "CP (cercle)", "FP": "FP (flux)", "GP": "GP (grille)", "RCP": "RCP (aléatoire)",
                "RCP_FL": "RCP_FL (niveaux de vol)"}
MARK = {"CP": "o", "FP": "s", "GP": "^", "RCP": ".", "RCP_FL": "x"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _save(fig, stem: str) -> dict:
    svg, png = OUT / f"{stem}.svg", OUT / f"{stem}.png"
    fig.savefig(svg, format="svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(png, format="png", dpi=300, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return {"svg": svg.name, "png": png.name}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads((SRC / "instances.json").read_text())["rows"]
    captions: dict[str, dict] = {"schema": "acrpq-experimental-figures/1", **prov.FLAGS,
                                 "provenance": prov.provenance_block(ROOT), "figures": {}}

    # ---- fig 1 & 2 share a CSV
    csv1 = OUT / "fig1_fig2_source.csv"
    with csv1.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "family", "n", "largest_component_K3", "Q_global_K3", "Q_largest_component_K3", "classes"])
        for r in rows:
            w.writerow([r["id"], r["family"], r["n"], r["largest_component_K3"], r["Q_global_K3"],
                        r["Q_largest_component_K3"], json.dumps(r["classes"])])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for fam in FAMILY_ORDER:
        xs = [r["n"] for r in rows if r["family"] == fam]
        ys = [r["largest_component_K3"] for r in rows if r["family"] == fam]
        ax.scatter(xs, ys, s=14 if fam.startswith("RCP") else 34, marker=MARK[fam], label=FAMILY_LABEL[fam], alpha=0.75)
    ax.plot([0, 155], [0, 155], color="0.6", linewidth=0.8, linestyle="--", label="composante = instance (aucune décomposition)")
    ax.set_xlabel("Nombre total d'avions n (–)")
    ax.set_ylabel("Plus grande composante K3 (avions)")
    ax.set_title("Plus grande composante du graphe any-option K3, par famille")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    f1 = _save(fig, "fig1_largest_component_vs_n")
    captions["figures"]["fig1"] = {**f1, "source_csv": csv1.name, "source_sha256": _sha(csv1),
                                   "caption": "Figure 1 — Taille de la plus grande composante connexe du graphe de conflit "
                                              "any-option K3 en fonction du nombre d'avions, pour les 1042 instances du dépôt, "
                                              "par famille. Les points sur la diagonale ne sont pas décomposables (CP, GP presque "
                                              "toujours). Aucune agrégation entre familles."}

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for fam in FAMILY_ORDER:
        xs = [r["Q_global_K3"] for r in rows if r["family"] == fam]
        ys = [r["Q_largest_component_K3"] for r in rows if r["family"] == fam]
        ax.scatter(xs, ys, s=14 if fam.startswith("RCP") else 34, marker=MARK[fam], label=FAMILY_LABEL[fam], alpha=0.75)
    ax.plot([0, 460], [0, 460], color="0.6", linewidth=0.8, linestyle="--", label="largeur inchangée")
    for y, txt in ((30, "30 qubits : limite statevector locale"), (12, "12 qubits : régime pilote observé")):
        ax.axhline(y, color="0.3", linewidth=0.7, linestyle=":")
        ax.text(462, y, txt, fontsize=6.5, va="bottom", ha="right")
    ax.set_xlabel("Largeur monolithique one-hot K3 = 3n (qubits logiques)")
    ax.set_ylabel("Largeur de la plus grande composante = 3·|C_max| (qubits logiques)")
    ax.set_title("Largeur monolithique vs largeur décomposée (qubits logiques, jamais physiques)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    f2 = _save(fig, "fig2_width_monolithic_vs_decomposed")
    captions["figures"]["fig2"] = {**f2, "source_csv": csv1.name, "source_sha256": _sha(csv1),
                                   "caption": "Figure 2 — Largeur logique du QUBO one-hot K3 monolithique (3n) contre largeur de la "
                                              "plus grande composante exacte (3·|C_max|). Les lignes horizontales marquent la limite "
                                              "pratique du vecteur d'état local (30 qubits) et le régime où la campagne réelle a "
                                              "produit des échantillons faisables (12 qubits). Une largeur réduite n'implique pas une "
                                              "résolution exacte globale."}

    # ---- fig 3
    csv3 = OUT / "fig3_source.csv"
    counts = {fam: {c: 0 for c in isc.CLASSES} for fam in FAMILY_ORDER}
    totals = {fam: 0 for fam in FAMILY_ORDER}
    for r in rows:
        totals[r["family"]] += 1
        for c in r["classes"]:
            counts[r["family"]][c] += 1
    with csv3.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["family", "n_instances", *isc.CLASSES])
        for fam in FAMILY_ORDER:
            w.writerow([fam, totals[fam], *[counts[fam][c] for c in isc.CLASSES]])
    fig, axes = plt.subplots(1, len(FAMILY_ORDER), figsize=(11, 3.6), sharey=False)
    short = {"DECOMPOSABLE_FORMALLY": "décomposable\n(formel)", "WIDTH_REDUCED": "largeur\nréduite",
             "NEWLY_LOCAL_EXACT_ACCESSIBLE": "nouv. exact\nlocal", "NEWLY_STATEVECTOR_ACCESSIBLE": "nouv.\nstatevector",
             "LOCAL_HEURISTIC_ONLY": "heuristique\nseul", "ISA_COMPILABLE_ONLY": "ISA\ncompilable\nseul",
             "HARDWARE_PILOT_PLAUSIBLE": "pilote\nplausible", "HARDWARE_SCIENTIFICALLY_IMPLAUSIBLE": "matériel\nnon crédible"}
    for ax, fam in zip(axes, FAMILY_ORDER):
        vals = [100.0 * counts[fam][c] / totals[fam] for c in isc.CLASSES]
        ax.barh(range(len(isc.CLASSES)), vals, color="0.45")
        ax.set_yticks(range(len(isc.CLASSES)))
        ax.set_yticklabels([short[c] for c in isc.CLASSES] if fam == "CP" else [""] * len(isc.CLASSES), fontsize=6.5)
        ax.set_xlim(0, 100)
        ax.set_title(f"{FAMILY_LABEL[fam]}\n({totals[fam]} instances)", fontsize=8)
        ax.set_xlabel("% des instances de la famille", fontsize=7)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Classes d'accessibilité par famille (une instance peut porter plusieurs classes)", fontsize=9)
    f3 = _save(fig, "fig3_accessibility_classes")
    captions["figures"]["fig3"] = {**f3, "source_csv": csv3.name, "source_sha256": _sha(csv3),
                                   "caption": "Figure 3 — Part des instances de chaque famille portant chaque classe "
                                              "d'accessibilité (schéma 2). « Décomposable formellement » n'implique pas "
                                              "« accessible » ; « ISA compilable seul » n'implique pas « exploitable ». Aucun "
                                              "cumul entre familles."}

    # ---- fig 4 schematic
    csv4 = OUT / "fig4_source.csv"
    steps = [("Modèle continu\n(AMPL, q et θ continus)", "référence historique"),
             ("Discrétisation globale one-hot\nK niveaux, 3n qubits à K3", "validée : énumération = MILP = QUBO"),
             ("Raffinement adaptatif\ngrilles locales K3, largeur 3n", "gain : PRÉCISION angulaire\n(quadratique seulement)"),
             ("Décomposition en composantes\nany-option K3, largeur 3·|C_max|", "gain : LARGEUR\n(exact sur K3 seulement)")]
    with csv4.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["step", "title", "role"])
        for i, (t, r) in enumerate(steps):
            w.writerow([i + 1, t.replace("\n", " "), r.replace("\n", " ")])
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    for i, (t, r) in enumerate(steps):
        x = 0.3 + i * 2.45
        ax.add_patch(plt.Rectangle((x, 1.0), 2.1, 1.3, fill=False, linewidth=1.0))
        ax.text(x + 1.05, 1.65, t, ha="center", va="center", fontsize=7.5)
        ax.text(x + 1.05, 0.45, r, ha="center", va="center", fontsize=7, color="0.25")
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 2.45, 1.65), xytext=(x + 2.1, 1.65), arrowprops={"arrowstyle": "->", "linewidth": 0.9})
    ax.text(5.0, 2.5, "Deux bénéfices distincts : précision (adaptatif) et largeur (décomposition) ne se fusionnent pas",
            ha="center", va="center", fontsize=7.5, style="italic")
    f4 = _save(fig, "fig4_method_schematic")
    captions["figures"]["fig4"] = {**f4, "source_csv": csv4.name, "source_sha256": _sha(csv4),
                                   "caption": "Figure 4 — Chaîne méthodologique : modèle continu, discrétisation globale one-hot "
                                              "validée, raffinement adaptatif (gain de précision à largeur 3n constante, objectif "
                                              "quadratique) et décomposition en composantes (gain de largeur, exact pour le QUBO K3 "
                                              "seulement). Les deux derniers bénéfices restent distincts."}

    prov.write_compact_json(OUT / "captions.json", captions)
    print("figures written to", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
