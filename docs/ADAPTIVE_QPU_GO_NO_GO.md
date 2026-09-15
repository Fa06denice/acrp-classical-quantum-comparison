# Porte GO / NO-GO — pilote QPU du raffinement adaptatif (ibm_marrakesh)

Date de la porte : vendredi 4 septembre 2026, après-midi. Branche expérimentale
`exp/adaptive-qpu-gate` (worktree séparé, base `4d2f75e`). Aucun job IBM n'a été soumis.
Aucun fichier de `Code/`, du solveur du superviseur, des résultats officiels, de l'UI ou des
WIP du worktree principal n'a été modifié (preuve §10).

## VERDICT = LOCAL_ONLY

`READY_FOR_BOUNDED_QPU_PILOT` n'est pas démontré ce soir. Trois conditions obligatoires
échouent ou ne sont que partielles (tableau §8). Conformément à la règle de temps : aucun job
IBM, gel du verdict `LOCAL_PROOF_OF_CONCEPT_VALIDATED` complété par les composants validés
localement ci-dessous, passage à la rédaction.

Ce qui EST validé localement (et peut être rédigé) :

* la formulation QUBO à grilles hétérogènes (A) ;
* le décodage exact des counts, endianness incluse, contre-vérifié indépendamment (B) ;
* le coût réel en qubits, circuits, shots et appels séquentiels (C) ;
* la compilation ISA vers le fake backend ET, en lecture seule, vers la cible réelle
  `ibm_marrakesh`, avec hashes identiques et reproductibles (D) ;
* l'identité scientifique hachée inter-rounds, vérification profonde par recalcul (E) ;
* une chaîne pilotée par les résultats d'un échantillonneur (Aer) sans sélection post hoc (F,
  simulateur uniquement).

Ce qui N'EST PAS validé : un pilote de soumission IBM (persistance du `job_id` avant toute
autre action, réconciliation, aucun retry) n'existe pas dans le module expérimental ; la
latence de file (médiane historique 7,68 h par job) rend une chaîne séquentielle de 4 rounds
incompatible avec le gel du samedi ; la référence continue q=1 est un incumbent provisoire.

---

## 1. Question A — QUBO à grilles hétérogènes

Formulation (expert 1, module `src/acrpq/experimental/adaptive_qpu.py`) : pour l'avion `i`,
grille `G_i = dedup(clip{c_i−Δ, c_i, c_i+Δ})` de taille `K_i ∈ {1,2,3}` (4 avec NOOP forcé),
offsets `off_i = Σ_{j<i} K_j`, bit `(i,o) ↦ off_i + o`, coût calculé sur le θ physique
(`w·θ²`, q=1), conflits recalculés pour chaque paire physique `G_i × G_j`, pénalités
officielles `λ_pen = 10(n·cost_max+1)`, `λ_oh = max(n, maxdeg+1)·λ_pen` avec `cost_max` pris
sur l'union des options du round.

Vérifié :

* bijection des offsets pour `K_i` quelconques (test, expert 1) ;
* dominance one-hot étendue aux blocs hétérogènes, y compris `K_i = 1` (bit forcé) et `K_i = 2` ;
* identité **coefficient par coefficient** avec `build_qubo(n_theta=3)` pour la grille K3
  uniforme sur CP_3, CP_4, CP_5 (constante, linéaire, quadratique, pénalités ; écart 0,0 sur
  tous les états one-hot) ;
* énergie = coût + λ_pen·conflits sur tout état one-hot d'une grille hétérogène (test) ;
* le hash du QUBO dépend des grilles ordonnées par identité d'avion, des valeurs θ, de
  l'objectif et de l'instance (test : permutation d'avions, changement de Δ, changement
  d'objectif donnent trois hashes distincts) ;
* options dupliquées ou non triées dans un bloc refusées à la construction.

Hypothèses cachées du code officiel qui interdisent sa réutilisation : `discrete.k` scalaire,
`var_index`, `decode_assignment`, `noop_index()`, `option_cost_vector`, `_validate_qubo`. Le
module expérimental les réimplémente et les teste contre l'officiel.

Les énergies ne sont pas comparables entre rounds (pénalités et grilles changent) ; seul
l'objectif physique l'est.

## 2. Question B — décodage sous échantillonnage

Décodeur : normalisation des clés (str avec espaces, int), `bit_order` qiskit → inversion,
validation stricte (longueur, binaire, bool, négatif, non fini, somme nulle, quasi-distribution
≠ 1), one-hot par bloc, réparation documentée (aucun bit → centre, toujours présent ; plusieurs
bits → option de coût minimal puis |θ| minimal puis indice), géométrie, objectif, énergie,
pénalité, poids. `best_feasible` = objectif physique minimal parmi les échantillons one-hot
valides et sans conflit, départage lexicographique sur la bitstring (l'énergie n'intervient pas,
correction suite à l'expert 2).

Vérifié :

* bijection encode→decode sur tous les états one-hot pour blocs [3,3,3], [2,3,3], [1,2,3] et
  CP_4 K3 (81 états) ;
* 13 formes malformées refusées (tests paramétrés) ;
* meilleur faisable indépendant de l'ordre des counts ; départage incapable de dégrader
  l'objectif ;
* endianness : décodeur indépendant de l'expert 2 (65/67 contrôles PASS, 2 écarts expliqués et
  corrigés), vérité terrain AerSimulator (X sur le qubit 0 → clé `0001`, caractère de droite =
  qubit 0), mesure `qubit b → clbit b` inspectée ;
* les 4 rounds de la chaîne Aer CP_3 recalculés à l'identique par le décodeur indépendant.

## 3. Question C — coût réel

Chaînes finales (`results/experimental_adaptive_grid_v1/chains/`), objectif quadratique,
init NOOP, ρ=0,5, 512 shots/round, p=1, angles `penalty_normalised`, ISA opt-level 0, seed 1.

| chaîne | rounds | largeur logique max | largeur physique (layout) | ancillas | circuits | shots totaux | Σ profondeur ISA | Σ portes 2Q | jobs IBM séquentiels |
|---|---|---|---|---|---|---|---|---|---|
| CP_3 aer_sim | 4 | 9 | 9 qubits physiques | 0 | 4 | 2048 | 508 | 238 | 4 |
| CP_4 aer_sim | 4 | 12 | 12 | 0 | 4 | 2048 | 646 | 414 | 4 |
| CP_3 exact (référence méthode) | 8 | 9 | 9 | 0 | 8 (construits, non échantillonnés) | 0 | 1262 | 557 | 0 |
| K3 global (campagne réelle) | 1 | 9 | 9 | 0 | 1 | 512 | 211 | 108 | 1 |
| K7 global (jamais exécuté) | 1 | 21 | — | 0 | 1 | 512 | 640 (CP_7 K3 comme ordre de grandeur) | 700 | 1 |

Par round CP_3 : profondeur ISA 211/85/106/106, portes 2Q 108/26/52/52 ; CP_4 : 259/121/133/133 et
207/45/81/81. Profondeur logique avant transpilation : 10 (CP_3), 12 (CP_4).

Temps : construction et recentrage classiques < 0,1 s par round ; optimisation d'angles 0 s
(règle fixe préenregistrée) ; résolution locale exacte < 0,05 s ; simulation Aer < 0,1 s par
round ; temps fournisseur inconnu (aucun champ de secondes QPU dans la campagne précédente) ;
temps de file : médiane historique 7,68 h par job (`submit_to_complete_s`), donc ≈ 31 h pour
4 rounds séquentiels, ≈ 61 h pour CP_3 + CP_4. À 12 h 18 UTC ce jour la file affichait
0 job en attente (lecture seule), ce qui ne garantit rien pour la durée d'une chaîne.
Décisions adaptatives : 4 par chaîne (une par round). Le coût ne se résume jamais à la largeur.

## 4. Question D — compilation ISA

* FakeMarrakesh : transpilation reproductible (double passage, hashes identiques) à opt-levels
  0 à 3 (expert 3, 32/32), basis gates conformes (`cz, rz, sx, x, measure`), aucun paramètre
  libre, `num_clbits == num_qubits`, registre `c`, aucun circuit synthétique.
* Bug de parité trouvé par l'expert 3 et corrigé : l'émission triée des termes RZZ changeait le
  routage Sabre (253/120 au lieu de 211/108). Après retour à l'ordre d'insertion officiel, le
  round 0 est bit pour bit le circuit ISA du builder officiel et reproduit les chiffres de la
  campagne réelle (CP_3 211/108, CP_4 259/207).
* Cible réelle en lecture seule (`QiskitRuntimeService().backend("ibm_marrakesh")`, aucun job) :
  pour les 8 rounds, hashes ISA **identiques** à ceux du fake backend, layouts identiques,
  basis OK, reproductibles (`chains/live_backend_readonly_check.json`). Cela vaut pour
  opt-level 0 (routage indépendant de la calibration) ; cela ne dit rien des taux de
  faisabilité sur matériel.

## 5. Question E — identité inter-itérations

Préinscription immuable (`chain_id` = sha256 du document) : protocole, instance (sha256 du
contenu), objectif, initialisation, ρ, Δ₀, `min_delta`, rounds max, driver
(`exact_local | sa | aer_sim | qpu_result`), shots, backend, angles et règle d'angles, seed,
politique sans échantillon faisable, identité des références, commit source. Par round :
centres, Δ, grilles ordonnées, hash de grille, hash QUBO, hash du circuit logique, hash ISA,
hash de configuration de soumission, `job_id` (null), hash des counts bruts, hash du décodage,
hash du candidat sélectionné, hash d'archive, statut, angles effectifs, référence de pénalité.
`round_hash_r = sha256(payload_r ‖ round_hash_{r−1} ‖ selected_candidate_hash_{r−1})`.

Refus fail-closed testés : parent absent, parent corrompu, hashes divergents, objectif ou
instance différents, centre ≠ candidat parent, round déjà existant, préinscription différente
dans le même répertoire, driver `qpu_result` sans job, second écrivain (verrou `fcntl`),
`delta` sous le plancher, cycle d'états.

Red-team (expert 6) : 1 BLOCKER, 2 CRITICAL, 3 MAJOR, tous corrigés le même jour :

| constat | correction |
|---|---|
| BLOCKER : `verify()` ne recalculait pas la physique ; un round entier pouvait être réécrit et re-chaîné | `verify(deep=True)` recharge l'instance, reconstruit grille et QUBO, relit et re-hache `raw_counts.json` / `decoded.json`, re-décode les counts et exige que la sélection soit le meilleur faisable, re-résout les rounds exact/SA ; test de forgeage complet re-chaîné |
| CRITICAL : drapeaux et champs sémantiques hors hash | `ROUND_PAYLOAD_KEYS` étendu (flags, instance, objectif, protocole, pénalités, règle d'angles, tailles de blocs, shots…) + invariant codé en dur `experimental=true / official_benchmark=false` sur manifeste, préinscription et rounds |
| CRITICAL : `preregistration_hash` jamais relu | vérifié contre le contenu de `preregistration.json` |
| MAJOR : préinscription sans drapeaux | drapeaux ajoutés |
| MAJOR : pas de `min_delta` ni garde anti-cycle | plancher préenregistré, garde `(centres, Δ)` |
| MAJOR : concurrence non gérée | verrou exclusif d'écrivain unique ; le multi-processus reste hors périmètre et documenté |

Ce que la chaîne ne protège pas : un auteur disposant de l'accès complet peut ré-exécuter
toute la chaîne et republier ; seule la publication préalable du `chain_id` (ou du hash de
préinscription) dans un canal externe, et pour un round réel la revalidation du `job_id`
auprès du service IBM, ancrent le résultat. Aucun des deux n'est fait. La re-vérification
indépendante des correctifs est rapportée en §8 (condition 12).

## 6. Question F — chaîne pilotée par les résultats

Protocoles préenregistrés (expert 5, module) :

* `LOCAL_EXACT_ADAPTIVE_V1` : chaque grille résolue exactement ; valeurs de round non
  croissantes tant que le parent est admissible (prouvé, vérifié sur 8 rounds CP_3 et CP_4).
* `QPU_DRIVEN_ADAPTIVE_PILOT_V1` : un circuit QAOA p=1 par round ; le meilleur échantillon
  faisable met à jour le centre ; archive élitiste ; sans échantillon faisable, politique
  préenregistrée `keep_center_retry_once` (un seul rejeu à l'identique, puis arrêt), jamais de
  centre inventé, aucune réparation ne pilote la chaîne ; seule l'archive est non croissante.

Règle d'angles : `γ_r = γ₀·λ_pen(0)/λ_pen(r)`, β fixe, avec γ₀=0,4 et β=0,3 (round 0 ≡ angles
historiques de la campagne). Motif : à angles fixes, l'angle RZZ one-hot (16,9 rad au round 0)
se replie modulo 2π dans un régime différent dès le round 2 ; en simulation sans bruit le taux
faisable tombe à 0,05 % au round 2 (0 % one-hot valide) et la chaîne meurt. Avec la règle
normalisée, les phases de pénalité sont invariantes ; taux faisable 10,5 % / 53 % / 37 % / 38 %
sur CP_3 et 2,3 % / 16 % / 12 % / 11 % sur CP_4, et chaque round sélectionne l'optimum exact de
sa grille. Limite documentée (expert 6) : la phase liée au coût pur dérive de +15 à +22 % aux
rounds 2–3 ; c'est une assistance classique dosée, préenregistrée sous ce nom, pas une propriété
quantique préservée. L'expert 5 recommandait au contraire de garder les angles fixes pour la
comparabilité avec la campagne ; les deux positions sont conservées dans les rapports.

Une chaîne pilotée par les résultats a été exécutée **sur Aer uniquement** : le driver `aer_sim`
échantillonne le circuit du round, décode, sélectionne, recentre, persiste ; `verify(deep=True)`
passe sur les quatre chaînes finales. Aucun code de soumission IBM n'existe dans le module ;
un round `qpu_result` est refusé.

## 7. Références équitables

* Continu q+θ (Gurobi, lecture seule) : autre domaine de contrôle, comparaison toujours
  qualifiée.
* Continu θ-seul q=1 (expert 5) : incumbent **provisoire** (B&B exact sur K=2049/2049/1025/1025
  puis polissage par section dorée), jamais « certifié » : CP_3 0,000322 ; CP_4 0,000641 ;
  CP_5 0,001169 (marge de sécurité à la limite de tolérance, −1e-8, à publier avec réserve) ;
  CP_6 0,001886. Fichier `results/experimental_adaptive_grid_v1/references/theta_only_q1_reference.json`.
* Discret global K3/K7/K33/K65 : certificat par ligne dans
  `local_benchmark_final/references.json` (algorithme, bornes, gap 0, nœuds, statut, temps,
  hash, finitude, faisabilité) ; K65 absent pour CP_6 et K33/K65 absents pour n ≥ 7 (non
  calculés, pas « incumbent » : simplement absents).
* Adaptatif local : `local_benchmark_final/adaptive_rounds.{json,csv}` (1104 lignes).

## 8. Conditions GO (toutes obligatoires)

| # | condition | état | preuve |
|---|---|---|---|
| 1 | formulation hétérogène validée | **OK** | §1, tests, expert 1 |
| 2 | bijection encode/decode | **OK** | §2, tests |
| 3 | endianness validée indépendamment | **OK** | expert 2 |
| 4 | identité inter-rounds fail-closed | **OK après correctifs** | §5, 65 tests |
| 5 | chaîne resume-safe et idempotente | **OK (écrivain unique)** | tests resume, verrou ; multi-processus hors périmètre |
| 6 | absence de double soumission | **ÉCHEC** | aucun driver de soumission ; la séquence submit → persist job_id → reconcile n'existe pas et ne peut être démontrée |
| 7 | circuits ISA reproductibles | **OK** | fake et cible réelle, hashes identiques |
| 8 | budget borné | **ÉCHEC** | shots bornés (2048/chaîne) mais secondes QPU inconnues et latence de file médiane 7,68 h/round → ≈ 31 h pour 4 rounds, incompatible avec le gel du samedi |
| 9 | angles et recentrage préenregistrés | **OK** | préinscription, §6 |
| 10 | référence q=1 disponible | **PARTIEL** | incumbent provisoire, non certifié |
| 11 | résultats locaux exacts reproductibles | **OK** | chaînes exact = `run_adaptive` bit pour bit ; campagne rejouée identique |
| 12 | red-team sans BLOCKER/CRITICAL | **OK (limites documentées)** | corrigés le jour même ; re-vérification indépendante : FIXED/DÉTECTÉ, aucun nouveau BLOCKER (§12) |
| 13 | tous les gates verts | **PARTIEL** | voir §9 : deux erreurs ruff préexistantes hors périmètre, un test dépendant d'artefacts locaux non versionnés |

Une seule condition en échec suffit : **LOCAL_ONLY**.

## 9. Gates

| gate | résultat |
|---|---|
| `ruff check src tests scripts` | vert |
| `ruff check .` | 2 erreurs préexistantes dans `docs/presentation/` (fichiers WIP du worktree principal, non touchés) |
| `mypy src/acrpq` | vert (67 fichiers) |
| tests expérimentaux | 45 + 65 verts |
| suite non-UI complète | tout vert sauf `test_qpu_real_artifact.py::test_campaign_artifacts_present_and_inventoried`, qui lit des exports IBM locaux non versionnés absents du worktree neuf ; passe dans le worktree principal |
| multiprocess/restart | resume idempotent testé ; verrou d'écrivain testé ; course multi-processus non testée (documenté) |
| `build --no-isolation` | vert (sdist + wheel) |
| `pip check` | vert |
| `pip-audit` | 1 vulnérabilité connue `diskcache 5.6.3` (PYSEC-2026-2447), dépendance de l'environnement, non neutralisée ; 6 paquets AMPL non auditables (hors PyPI) |
| `git diff --check` | vert |
| scans secrets/PII/chemins | aucun token, e-mail ou chemin absolu dans les nouveaux fichiers |
| manifests/hashes | manifests avec sha256 par fichier ; `integrity_report.json` deep-verify OK sur 4 chaînes |
| parité JSON/CSV | 294/294 lignes, 0 divergence (campagne v1) ; 1104 lignes benchmark final |
| chemins protégés | hashes identiques avant/après (§10) |
| WIP préservés | aucune écriture dans le worktree principal par cette étude (§10) |
| review indépendante | 6 experts + re-vérificateur |

## 10. Isolation et preuves

* Inventaire initial (commit `4d2f75e`, 09:53 UTC) : hashes agrégés de `Code/`, des modules
  scientifiques officiels, des résultats officiels et du dashboard. Recalcul en fin de journée
  dans le worktree principal : **identiques**.
* Le worktree principal a évolué pendant la journée par une autre session (4 commits, HEAD
  `91d97ec`, nouveaux WIP sur la campagne IBM finale). Cette étude n'y a rien écrit après la
  création du worktree ; ses fichiers y restent non suivis.
* Aucun token : la vérification live utilise le compte enregistré par Qiskit, rien n'est écrit
  dans le dépôt, les logs ne contiennent que des métadonnées de backend.

## 11. Plan matériel minimal (non exécuté, pour mémoire)

Si un jour GO : CP_3, objectif quadratique, K local 3, init NOOP, ρ=0,5, 4 rounds max, un run
par round, 512 shots, p=1, angles `penalty_normalised` (γ₀=0,4, β=0,3), `ibm_marrakesh`
uniquement ; CP_4 seulement si CP_3 termine techniquement (3 rounds). Séquence obligatoire :
submit → persist `job_id` → reconcile → decode → select → persist → build round suivant.
Pré-requis manquants aujourd'hui : driver `qpu_result` avec persistance atomique du `job_id`
avant toute autre action, réconciliation sans retry, publication externe du `chain_id`, et une
fenêtre de temps compatible avec la latence de file.

## 12. Annexe — re-vérification indépendante des correctifs red-team

Rapport `results/experimental_adaptive_grid_v1/moe_reports/phase2_qpu_gate/expert6b_reverify.md`
(re-vérificateur indépendant, rejeu des scripts d'attaque du red-team contre le module corrigé) :

| constat | statut |
|---|---|
| 1 BLOCKER — pas de recalcul physique | FIXED |
| 2 CRITICAL — champs hors hash | PARTIAL au moment du rejeu (`timing_s`, `decoded_summary`, `isa_report` restaient non hachés, sans impact sur la sélection ni les drapeaux) ; ces trois clés ont été ajoutées au payload haché après le rejeu |
| 3 MAJOR — préinscription sans drapeaux | FIXED |
| 4 CRITICAL — `preregistration_hash` jamais relu | FIXED |
| 5 MAJOR — `min_delta` / anti-cycle | FIXED (lecture de code, pas de rejeu extrême) |
| 6 MAJOR — verrou d'écrivain | FIXED (lecture de code ; pas de course multi-processus réelle) |
| deux nouvelles forgeries complètes re-chaînées (exact, Aer) | DÉTECTÉES par `verify(deep=True)` |
| nouveau BLOCKER/CRITICAL | aucun |

Limites résiduelles reconnues : une préinscription falsifiée avant tout lancement, suivie
d'une régénération cohérente de toute la chaîne, n'est détectable que par comparaison à un
`chain_id` publié hors du répertoire ; un futur round `qpu_result` n'est validé qu'au niveau
du fichier local tant qu'aucune revalidation contre l'API IBM n'existe. La condition 12 passe
donc à « OK, avec limites documentées » ; le verdict reste **LOCAL_ONLY** (conditions 6 et 8).
