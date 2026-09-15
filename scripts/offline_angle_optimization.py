#!/usr/bin/env python3
"""Optimisation d'angles QAOA p=1 HORS LIGNE, déterministe et versionnée (Phase 3).

Rôle scientifique : décider, AVANT toute dépense QPU, si des angles optimisés hors ligne
améliorent la masse de probabilité faisable par rapport aux angles fixes de la cohorte P0
(gammas=[0.4], betas=[0.3]). Aucune connexion réseau, aucun compte IBM, aucun job soumis.

Trois jeux d'angles sont calculés par (instance, objectif) :

* ``P0``      — angles fixes de la cohorte historique, pour référence ;
* ``P1_minH`` — angles minimisant l'espérance d'énergie ⟨H⟩. C'est le critère QAOA
  **standard** de la littérature ;
* ``P1_feas`` — angles maximisant la masse de probabilité sur le sous-espace faisable
  (one-hot valide ET géométriquement sans conflit).

Les deux critères ne coïncident pas, et l'écart est le résultat central de cette phase :
sur une formulation one-hot fortement pénalisée, le minimum de ⟨H⟩ peut se situer là où la
masse faisable est quasi nulle. Le critère retenu pour le matériel doit donc être
préenregistré explicitement — voir docs/IBM_FINAL_CAMPAIGN_PREREGISTRATION.md.

Le calcul est EXACT : l'espérance et les masses sont obtenues par vecteur d'état, sans
échantillonnage, donc sans variance de tir. Le circuit utilisé est exactement celui du
chemin matériel (``build_parameterized_qaoa``), ce qui supprime toute ambiguïté sur la
convention (γ, β) : les angles figés ici sont directement ceux que le QPU exécutera.

Procédure déterministe (aucun aléa) :
  1. balayage sur une grille fixe GAMMA_GRID × BETA_GRID (résolution enregistrée) ;
  2. raffinement local Nelder-Mead depuis les INIT_POINTS enregistrés + le meilleur point
     de grille, ``maxiter`` et tolérances enregistrés ;
  3. le meilleur point est retenu, arrondi à ANGLE_DECIMALS pour être reproductible.

Usage :
    PYTHONPATH=src python scripts/offline_angle_optimization.py [--instances CP_3,CP_4]
    PYTHONPATH=src python scripts/offline_angle_optimization.py --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "acrpq-offline-angles/1"
OUT = Path("results/offline_angle_optimization_v1")

# --- protocole d'optimisation, entièrement enregistré ---------------------- #
P0_GAMMAS = (0.4,)
P0_BETAS = (0.3,)
GAMMA_GRID_N = 49          # [0, pi]
BETA_GRID_N = 25           # [0, pi/2]
INIT_POINTS = ((0.4, 0.3), (0.8, 0.8), (1.6, 0.4), (2.2, 1.0))
OPTIMIZER = "Nelder-Mead"
MAXITER = 200
XATOL = 1e-4
FATOL = 1e-8
ANGLE_DECIMALS = 6
# Au-delà, l'énumération exacte du sous-espace faisable et le vecteur d'état deviennent
# le facteur limitant ; on ne simule pas ce qu'on ne peut pas vérifier exactement.
MAX_EXACT_QUBITS = 15

OBJECTIVES = ("quadratic_control_cost_v1", "maneuver_count_v1")


def _sha(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


class Landscape:
    """Énergies exactes, masques de faisabilité et probabilités p=1 d'une instance."""

    def __init__(self, instance: str, objective_id: str) -> None:
        from acrpq import geometry
        from acrpq.dashboard.hardware_validation import build_parameterized_qaoa
        from acrpq.io.loader import InstanceLoader
        from acrpq.quantum.decode import decode_assignment
        from acrpq.quantum.qubo import build_qubo

        inst = InstanceLoader().load(instance)
        qubo = build_qubo(inst, n_theta=3, n_q=1, objective_id=objective_id)
        n = qubo.n_qubits
        if n > MAX_EXACT_QUBITS:
            raise ValueError(f"{instance}: {n} qubits > {MAX_EXACT_QUBITS} (exact only)")
        d = qubo.discrete
        size = 1 << n

        energy = np.empty(size)
        onehot = np.zeros(size, bool)
        feasible = np.zeros(size, bool)
        objective = np.full(size, np.nan)
        for s in range(size):
            bs = format(s, f"0{n}b")
            energy[s] = qubo.energy(bs)
            if all(sum(int(bs[d.var_index(i, o)]) for o in range(d.k)) == 1
                   for i in inst.aircraft()):
                onehot[s] = True
                choice = tuple(decode_assignment(qubo, bs)[i] for i in inst.aircraft())
                q, theta = d.maneuvers(choice)
                if geometry.count_conflicts(inst, q, theta) == 0:
                    feasible[s] = True
                    objective[s] = geometry.objective(inst, q, theta)

        circuit, gammas, betas = build_parameterized_qaoa(qubo, 1)
        self._circuit = circuit.remove_final_measurements(inplace=False)
        self._g, self._b = gammas[0], betas[0]
        # index du vecteur d'état (little-endian Qiskit) -> index en ordre de bits QUBO
        idx = np.arange(size)
        bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1)[:, ::-1]
        self._perm = (bits * (2 ** np.arange(n)[::-1])).sum(1)

        self.instance, self.objective_id, self.n_qubits = instance, objective_id, n
        self.energy, self.onehot, self.feasible, self.objective = (
            energy, onehot, feasible, objective)
        self.qubo_sha256 = self._qubo_hash(qubo)
        self.certified_reference = (float(np.nanmin(objective))
                                    if feasible.any() else None)
        self.n_optima = (int((objective == self.certified_reference).sum())
                         if self.certified_reference is not None else 0)

    @staticmethod
    def _qubo_hash(qubo: Any) -> str:
        from acrpq.dashboard.hardware_validation import canonical_qubo_hash
        return canonical_qubo_hash(qubo)

    def probs(self, gamma: float, beta: float) -> np.ndarray:
        from qiskit.quantum_info import Statevector
        bound = self._circuit.assign_parameters(
            {self._g: float(gamma), self._b: float(beta)}, inplace=False)
        p = np.abs(Statevector(bound).data) ** 2
        out = np.zeros_like(p)
        out[self._perm] = p
        return out

    def metrics(self, gamma: float, beta: float) -> dict[str, float]:
        p = self.probs(gamma, beta)
        feas_mass = float(p[self.feasible].sum())
        onehot_mass = float(p[self.onehot].sum())
        best_obj = None
        hit = False
        if feas_mass > 0.0:
            # meilleur échantillon faisable atteignable = celui d'objectif minimal ayant
            # une probabilité non nulle (exact, pas échantillonné)
            reachable = self.feasible & (p > 0.0)
            if reachable.any():
                best_obj = float(np.nanmin(self.objective[reachable]))
                hit = (self.certified_reference is not None
                       and best_obj == self.certified_reference)
        return {
            "expected_energy": float(p @ self.energy),
            "feasible_mass": feas_mass,
            "onehot_mass": onehot_mass,
            "conditional_feasibility": (feas_mass / onehot_mass) if onehot_mass else 0.0,
            "best_reachable_objective": best_obj,
            "certified_optimum_reachable": hit,
        }


def _optimise(land: Landscape, key: str) -> dict[str, Any]:
    """Grille déterministe puis raffinement Nelder-Mead. ``key`` = critère optimisé."""
    from scipy.optimize import minimize

    sign = 1.0 if key == "expected_energy" else -1.0

    def loss(x: np.ndarray) -> float:
        return sign * land.metrics(float(x[0]), float(x[1]))[key]

    gg = np.linspace(0.0, np.pi, GAMMA_GRID_N)
    bb = np.linspace(0.0, np.pi / 2, BETA_GRID_N)
    best = None
    for g in gg:
        for b in bb:
            v = loss(np.array([g, b]))
            if best is None or v < best[0]:
                best = (v, g, b)
    assert best is not None
    starts = [np.array([best[1], best[2]])] + [np.array(p) for p in INIT_POINTS]
    for x0 in starts:
        r = minimize(loss, x0, method=OPTIMIZER,
                     options={"maxiter": MAXITER, "xatol": XATOL, "fatol": FATOL})
        if float(r.fun) < best[0]:
            best = (float(r.fun), float(r.x[0]), float(r.x[1]))
    gamma = round(float(best[1]), ANGLE_DECIMALS)
    beta = round(float(best[2]), ANGLE_DECIMALS)
    out = {"criterion": key, "gamma": gamma, "beta": beta}
    out.update(land.metrics(gamma, beta))          # métriques AUX ANGLES ARRONDIS
    return out


def build(instances: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for instance in instances:
        for objective_id in OBJECTIVES:
            land = Landscape(instance, objective_id)
            row: dict[str, Any] = {
                "instance": instance,
                "objective_id": objective_id,
                "n_qubits": land.n_qubits,
                "grid_k": 3,
                "qubo_sha256": land.qubo_sha256,
                "n_onehot_states": int(land.onehot.sum()),
                "n_feasible_states": int(land.feasible.sum()),
                "certified_reference_objective": land.certified_reference,
                "n_certified_optima": land.n_optima,
                "P0": {"criterion": "fixed_historical",
                       "gamma": P0_GAMMAS[0], "beta": P0_BETAS[0],
                       **land.metrics(P0_GAMMAS[0], P0_BETAS[0])},
                "P1_minH": _optimise(land, "expected_energy"),
                "P1_feas": _optimise(land, "feasible_mass"),
            }
            for arm in ("P1_minH", "P1_feas"):
                p0 = row["P0"]["feasible_mass"]
                row[arm]["feasible_mass_ratio_vs_P0"] = (
                    row[arm]["feasible_mass"] / p0 if p0 > 0 else None)
                row[arm]["feasible_mass_diff_vs_P0"] = row[arm]["feasible_mass"] - p0
            rows.append(row)
            print(f"[{instance}/{objective_id[:18]:18s}] n={land.n_qubits} "
                  f"P0={row['P0']['feasible_mass']:.4f} "
                  f"minH={row['P1_minH']['feasible_mass']:.4f} "
                  f"feas={row['P1_feas']['feasible_mass']:.4f}", flush=True)

    from acrpq.classical.ampl_campaign import git_info
    git = git_info()
    doc = {
        "schema": SCHEMA,
        "note": ("Optimisation d'angles p=1 hors ligne, exacte (vecteur d'état, aucun "
                 "échantillonnage), sur le circuit du chemin matériel. Aucune soumission, "
                 "aucun réseau, aucun compte IBM. Ce n'est PAS une boucle QAOA sur QPU."),
        "procedure": {
            "optimizer": OPTIMIZER, "maxiter": MAXITER, "xatol": XATOL, "fatol": FATOL,
            "gamma_grid_n": GAMMA_GRID_N, "beta_grid_n": BETA_GRID_N,
            "init_points": [list(p) for p in INIT_POINTS],
            "angle_decimals": ANGLE_DECIMALS, "max_exact_qubits": MAX_EXACT_QUBITS,
            "deterministic": True, "random_seed_used": None,
            "exactness": "statevector expectation, no sampling noise",
        },
        "p0_reference_angles": {"gammas": list(P0_GAMMAS), "betas": list(P0_BETAS)},
        "source_commit": git["git_commit"],
        "scientific_code_dirty": git["scientific_code_dirty"],
        "repository_dirty": git["repository_dirty"],
        "rows": rows,
    }
    doc["content_sha256"] = _sha({k: v for k, v in doc.items() if k != "content_sha256"})
    return doc


def _verify() -> int:
    p = OUT / "offline_angles.json"
    if not p.exists():
        print("MISSING offline_angles.json")
        return 2
    doc = json.loads(p.read_text())
    recon = _sha({k: v for k, v in doc.items() if k != "content_sha256"})
    if recon != doc["content_sha256"]:
        print("CONTENT HASH MISMATCH")
        return 2
    print(f"OK offline angles verified ({len(doc['rows'])} rows)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Optimisation d'angles hors ligne")
    ap.add_argument("--instances", default="CP_3,CP_4,CP_5")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        return _verify()
    doc = build([s.strip() for s in args.instances.split(",") if s.strip()])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "offline_angles.json").write_text(
        json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"\n[offline-angles] {len(doc['rows'])} lignes -> {OUT}/offline_angles.json")
    print(f"  content_sha256={doc['content_sha256'][:24]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
