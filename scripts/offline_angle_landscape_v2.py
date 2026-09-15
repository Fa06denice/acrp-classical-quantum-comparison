#!/usr/bin/env python3
"""Paysage QAOA p=1 exact, hors ligne — K=3 (CP_3..CP_7) et K=5 (CP_3, CP_4).

Version 2, script SÉPARÉ de scripts/offline_angle_optimization.py (gelé, cf.
docs/WIP_INVENTORY_OFFLINE_ANGLES.md) afin de ne pas mélanger les travaux.

Ce que v2 ajoute :

* CP_6 (18 qubits), CP_7 (21 qubits) — hors de portée de v1 (plafond 15 qubits) ;
* les cellules K=5, où les deux objectifs ne sont PLUS proportionnels et où la
  comparaison devient une vraie comparaison de fonctions objectif ;
* la **masse sur l'optimum certifié** (et pas seulement la masse faisable) ;
* les quatre quantités séparées demandées : masse one-hot valide, masse
  géométriquement faisable, masse sur l'optimum certifié, espérance de H.

Méthode. Un simulateur p=1 dédié remplace le simulateur générique : la couche de coût
est diagonale (phase e^{-iγ E_Ising(z)}) et le mélangeur est un produit d'opérateurs à un
qubit, donc l'état s'obtient en O(n·2^n) au lieu d'une simulation de circuit générale
(6,7 s par évaluation à 21 qubits, soit 2 h 15 pour une grille — inacceptable, et
dégrader la grille aurait rendu CP_7 non comparable aux autres cellules). Le simulateur
est VALIDÉ contre qiskit ``Statevector`` sur toutes les cellules où c'est abordable, avec
une tolérance stricte ; le script échoue si l'écart dépasse VALIDATION_TOL.

La faisabilité géométrique n'est évaluée que sur les états one-hot (K^n états, ≤ 2187),
jamais sur les 2^n : hors du sous-espace one-hot la notion n'est pas définie.

Aucun réseau, aucun compte IBM, aucune soumission. Ce n'est PAS une boucle QAOA sur QPU.

Usage :
    PYTHONPATH=src python scripts/offline_angle_landscape_v2.py
    PYTHONPATH=src python scripts/offline_angle_landscape_v2.py --cells CP_3:3,CP_7:3
    PYTHONPATH=src python scripts/offline_angle_landscape_v2.py --verify
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "acrpq-offline-angle-landscape/2"
OUT = Path("results/offline_angle_landscape_v2")

P0_GAMMA, P0_BETA = 0.4, 0.3
GAMMA_GRID_N = 49
BETA_GRID_N = 25
INIT_POINTS = ((0.4, 0.3), (0.8, 0.8), (1.6, 0.4), (2.2, 1.0))
OPTIMIZER = "Nelder-Mead"
MAXITER = 200
XATOL = 1e-4
FATOL = 1e-8
ANGLE_DECIMALS = 6
VALIDATION_TOL = 1e-10
VALIDATE_UP_TO_QUBITS = 21          # toutes les cellules sont validees, aucune exception

OBJECTIVES = ("quadratic_control_cost_v1", "maneuver_count_v1")
# (instance, K) ; K est réalisé comme n_theta=K, n_q=1
CELLS = (("CP_3", 3), ("CP_4", 3), ("CP_5", 3), ("CP_6", 3), ("CP_7", 3),
         ("CP_3", 5), ("CP_4", 5))


def _sha(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


class Landscape:
    """Énergies, sous-espace faisable et distribution p=1 exacte d'une cellule."""

    def __init__(self, instance: str, k: int, objective_id: str) -> None:
        from acrpq import geometry
        from acrpq.io.loader import InstanceLoader
        from acrpq.objectives import primary_objective
        from acrpq.quantum.ising import qubo_to_ising
        from acrpq.quantum.qubo import build_qubo

        inst = InstanceLoader().load(instance)
        qubo = build_qubo(inst, n_theta=k, n_q=1, objective_id=objective_id)
        n = qubo.n_qubits
        size = 1 << n
        d = qubo.discrete
        aircraft = list(inst.aircraft())

        # --- énergie Ising, vectorisée. z_b = +1 si bit_b = 0, -1 sinon, exactement la
        # convention des portes rz(2*gamma*h)/rzz(2*gamma*J) du circuit matériel.
        ising = qubo_to_ising(qubo)
        idx = np.arange(size, dtype=np.int64)
        bit = np.empty((n, size), dtype=np.int8)
        for b in range(n):
            bit[b] = (idx >> (n - 1 - b)) & 1        # bit b du QUBO
        z = (1 - 2 * bit).astype(np.float64)
        e_ising = np.zeros(size)
        for b, h in ising.h.items():
            if h:
                e_ising += float(h) * z[b]
        for (a, c), j in ising.J.items():
            if j:
                e_ising += float(j) * z[a] * z[c]

        # --- énergie QUBO (pour l'espérance de H), vectorisée
        e_qubo = np.full(size, float(qubo.constant))
        xb = bit.astype(np.float64)
        for b, coeff in qubo.linear.items():
            if coeff:
                e_qubo += float(coeff) * xb[b]
        for (a, c), coeff in qubo.quadratic.items():
            if coeff:
                e_qubo += float(coeff) * xb[a] * xb[c]

        # --- sous-espace one-hot : K^n états seulement, énumérés directement
        onehot_idx: list[int] = []
        feasible_idx: list[int] = []
        feasible_obj: list[float] = []
        for choice in itertools.product(range(d.k), repeat=len(aircraft)):
            s = 0
            for i, o in zip(aircraft, choice):
                s |= 1 << (n - 1 - d.var_index(i, o))
            onehot_idx.append(s)
            q, theta = d.maneuvers(choice)
            if geometry.count_conflicts(inst, q, theta) == 0:
                feasible_idx.append(s)
                # La reference DOIT etre dans les unites de l'objective_id de la cellule.
                # geometry.objective() renvoie toujours le cout quadratique de l'instance :
                # a K=3 la proportionnalite masquait l'erreur (meme argmin), mais a K=5 les
                # objectifs ne sont pas proportionnels et l'argmin differe, donc la masse
                # sur l'optimum certifie des cellules K5/maneuver etait fausse.
                feasible_obj.append(float(primary_objective(objective_id, q, theta, inst.w)))

        self.instance, self.k, self.objective_id = instance, d.k, objective_id
        self.n_qubits, self.size = n, size
        self.n_aircraft = len(aircraft)
        self.e_ising, self.e_qubo = e_ising, e_qubo
        self.onehot_idx = np.array(onehot_idx, dtype=np.int64)
        self.feasible_idx = np.array(feasible_idx, dtype=np.int64)
        self.feasible_obj = np.array(feasible_obj, dtype=np.float64)
        self.qubo_sha256 = self._qubo_hash(qubo)
        self.penalty_conflict = float(qubo.penalty_conflict)
        if self.feasible_obj.size:
            self.reference = float(self.feasible_obj.min())
            same = np.isclose(self.feasible_obj, self.reference, rtol=0.0, atol=1e-12)
            self.optimum_idx = self.feasible_idx[same]
            self.n_optima = int(same.sum())
        else:                                        # aucune solution faisable sur la grille
            self.reference, self.n_optima = None, 0
            self.optimum_idx = np.array([], dtype=np.int64)
        self._qubo = qubo

    @staticmethod
    def _qubo_hash(qubo: Any) -> str:
        from acrpq.dashboard.hardware_validation import canonical_qubo_hash
        return canonical_qubo_hash(qubo)

    # ---- simulateur p=1 dédié ------------------------------------------------ #
    def probs(self, gamma: float, beta: float) -> np.ndarray:
        n, size = self.n_qubits, self.size
        psi = (np.exp(-1j * float(gamma) * self.e_ising) / np.sqrt(size)).reshape((2,) * n)
        cb, sb = np.cos(float(beta)), np.sin(float(beta))
        # e^{-i beta X_b} qubit par qubit. Avec l'index s = sum_b bit_b 2^(n-1-b), le bit b
        # du QUBO est l'axe b du tenseur : retourner cet axe EST l'application de X_b.
        # Le retournement d'axe evite l'indexation fantaisie sur un tableau de 2^21
        # entiers, seul goulot a 21 qubits (6,7 s -> quelques dizaines de ms).
        for b in range(n):
            psi = cb * psi - 1j * sb * np.flip(psi, axis=b)
        return (np.abs(psi) ** 2).reshape(-1)

    def validate_against_qiskit(self, gamma: float, beta: float) -> float:
        """Écart max entre le simulateur dédié et qiskit. Le script échoue si trop grand."""
        from qiskit.quantum_info import Statevector

        from acrpq.dashboard.hardware_validation import build_parameterized_qaoa
        qc, g, b = build_parameterized_qaoa(self._qubo, 1)
        qc = qc.remove_final_measurements(inplace=False)
        sv = Statevector(qc.assign_parameters({g[0]: gamma, b[0]: beta}, inplace=False))
        p_qiskit = np.abs(sv.data) ** 2
        n = self.n_qubits
        # index qiskit (little-endian) -> index en ordre de bits QUBO
        ii = np.arange(self.size)
        bits = ((ii[:, None] >> np.arange(n)[None, :]) & 1)[:, ::-1]
        perm = (bits * (2 ** np.arange(n)[::-1])).sum(1)
        ref = np.zeros_like(p_qiskit)
        ref[perm] = p_qiskit
        return float(np.abs(ref - self.probs(gamma, beta)).max())

    # ---- métriques ----------------------------------------------------------- #
    def metrics(self, gamma: float, beta: float) -> dict[str, Any]:
        p = self.probs(gamma, beta)
        onehot_mass = float(p[self.onehot_idx].sum())
        feasible_mass = float(p[self.feasible_idx].sum()) if self.feasible_idx.size else 0.0
        optimum_mass = float(p[self.optimum_idx].sum()) if self.optimum_idx.size else 0.0
        best_obj = None
        if self.feasible_idx.size:
            reach = p[self.feasible_idx] > 0.0
            if reach.any():
                best_obj = float(self.feasible_obj[reach].min())
        return {
            "expected_energy": float(p @ self.e_qubo),
            "onehot_valid_mass": onehot_mass,
            "feasible_mass": feasible_mass,
            "certified_optimum_mass": optimum_mass,
            "conditional_geometry_feasibility": (
                feasible_mass / onehot_mass if onehot_mass > 0 else 0.0),
            "best_reachable_objective": best_obj,
            "certified_optimum_reachable": (
                best_obj is not None and self.reference is not None
                and best_obj == self.reference),
        }


def _optimise(land: Landscape, key: str, maximise: bool) -> dict[str, Any]:
    from scipy.optimize import minimize
    sign = -1.0 if maximise else 1.0

    def loss(x: Any) -> float:
        return sign * float(land.metrics(float(x[0]), float(x[1]))[key])

    best = None
    for g in np.linspace(0.0, np.pi, GAMMA_GRID_N):
        for b in np.linspace(0.0, np.pi / 2, BETA_GRID_N):
            v = loss((g, b))
            if best is None or v < best[0]:
                best = (v, float(g), float(b))
    assert best is not None
    for x0 in [np.array([best[1], best[2]])] + [np.array(p) for p in INIT_POINTS]:
        r = minimize(loss, x0, method=OPTIMIZER,
                     options={"maxiter": MAXITER, "xatol": XATOL, "fatol": FATOL})
        if float(r.fun) < best[0]:
            best = (float(r.fun), float(r.x[0]), float(r.x[1]))
    gamma = round(best[1], ANGLE_DECIMALS)
    beta = round(best[2], ANGLE_DECIMALS)
    out: dict[str, Any] = {"criterion": key, "maximise": maximise,
                           "gamma": gamma, "beta": beta}
    out.update(land.metrics(gamma, beta))
    return out


def build(cells: list[tuple[str, int]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for instance, k in cells:
        for objective_id in OBJECTIVES:
            t0 = time.time()
            land = Landscape(instance, k, objective_id)
            if land.n_qubits <= VALIDATE_UP_TO_QUBITS:
                dev = land.validate_against_qiskit(P0_GAMMA, P0_BETA)
                validations.append({"instance": instance, "grid_k": k,
                                    "objective_id": objective_id,
                                    "n_qubits": land.n_qubits, "max_abs_deviation": dev})
                if not dev < VALIDATION_TOL:
                    raise SystemExit(
                        f"REFUS : simulateur p=1 dévie de qiskit de {dev:.3e} "
                        f"(> {VALIDATION_TOL:.0e}) sur {instance}/K{k}/{objective_id}")
            row: dict[str, Any] = {
                "instance": instance, "grid_k": land.k, "objective_id": objective_id,
                "n_aircraft": land.n_aircraft, "n_qubits": land.n_qubits,
                "n_states": land.size, "n_onehot_states": int(land.onehot_idx.size),
                "n_feasible_states": int(land.feasible_idx.size),
                "qubo_sha256": land.qubo_sha256,
                "penalty_conflict": land.penalty_conflict,
                "certified_reference_objective": land.reference,
                "n_certified_optima": land.n_optima,
                "P0": {"criterion": "fixed_historical", "maximise": False,
                       "gamma": P0_GAMMA, "beta": P0_BETA,
                       **land.metrics(P0_GAMMA, P0_BETA)},
                "P1_minH": _optimise(land, "expected_energy", maximise=False),
                "P1_feas": _optimise(land, "feasible_mass", maximise=True),
            }
            p0 = row["P0"]["feasible_mass"]
            for arm in ("P1_minH", "P1_feas"):
                row[arm]["feasible_mass_diff_vs_P0"] = row[arm]["feasible_mass"] - p0
                row[arm]["feasible_mass_ratio_vs_P0"] = (
                    row[arm]["feasible_mass"] / p0 if p0 > 0 else None)
            row["wall_time_s"] = round(time.time() - t0, 2)
            rows.append(row)
            print(f"[{instance}/K{k}/{objective_id[:18]:18s}] n={land.n_qubits:2d} "
                  f"onehot={row['P0']['onehot_valid_mass']:.4f} "
                  f"P0={p0:.5f} minH={row['P1_minH']['feasible_mass']:.5f} "
                  f"feas={row['P1_feas']['feasible_mass']:.5f} "
                  f"({row['wall_time_s']}s)", flush=True)

    from acrpq.classical.ampl_campaign import git_info
    git = git_info()
    doc = {
        "schema": SCHEMA,
        "note": ("Paysage QAOA p=1 exact hors ligne. Aucune soumission, aucun réseau, "
                 "aucun compte IBM. P1_feas n'est PAS 'QAOA standard' : ses angles sont "
                 "choisis hors ligne pour maximiser la masse faisable. P1_minH est le "
                 "critère standard (minimisation de l'espérance d'énergie)."),
        "procedure": {
            "optimizer": OPTIMIZER, "maxiter": MAXITER, "xatol": XATOL, "fatol": FATOL,
            "gamma_grid_n": GAMMA_GRID_N, "beta_grid_n": BETA_GRID_N,
            "init_points": [list(p) for p in INIT_POINTS],
            "angle_decimals": ANGLE_DECIMALS, "deterministic": True,
            "random_seed_used": None,
            "simulator": ("dedicated exact p=1 (diagonal cost layer + per-qubit mixer), "
                          "O(n*2^n), validated against qiskit Statevector"),
            "validation_tolerance": VALIDATION_TOL,
            "validated_up_to_qubits": VALIDATE_UP_TO_QUBITS,
            "feasibility_evaluated_on": "one-hot subspace only (K^n states)",
        },
        "p0_reference_angles": {"gammas": [P0_GAMMA], "betas": [P0_BETA]},
        "simulator_validation": validations,
        "source_commit": git["git_commit"],
        "scientific_code_dirty": git["scientific_code_dirty"],
        "repository_dirty": git["repository_dirty"],
        "rows": rows,
    }
    doc["content_sha256"] = _sha({k: v for k, v in doc.items() if k != "content_sha256"})
    return doc


def _verify() -> int:
    p = OUT / "angle_landscape.json"
    if not p.exists():
        print("MISSING angle_landscape.json")
        return 2
    doc = json.loads(p.read_text())
    if _sha({k: v for k, v in doc.items() if k != "content_sha256"}) != doc["content_sha256"]:
        print("CONTENT HASH MISMATCH")
        return 2
    bad = [v for v in doc["simulator_validation"]
           if not v["max_abs_deviation"] < doc["procedure"]["validation_tolerance"]]
    if bad:
        print(f"SIMULATOR VALIDATION FAILED on {len(bad)} cells")
        return 2
    print(f"OK angle landscape verified ({len(doc['rows'])} rows, "
          f"{len(doc['simulator_validation'])} simulator validations)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Paysage QAOA p=1 exact hors ligne")
    ap.add_argument("--cells", default="", help="ex. CP_3:3,CP_7:3,CP_3:5")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        return _verify()
    cells = list(CELLS)
    if args.cells:
        cells = [(c.split(":")[0], int(c.split(":")[1]))
                 for c in args.cells.split(",") if c.strip()]
    doc = build(cells)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "angle_landscape.json").write_text(
        json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"\n[landscape] {len(doc['rows'])} lignes -> {OUT}/angle_landscape.json")
    print(f"  content_sha256={doc['content_sha256'][:24]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
