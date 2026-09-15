# Étude de scalabilité structurelle par instance — blocs hétérogènes, avions actifs, composantes

**Statut : expérimental. `experimental = true`, `official_benchmark = false`, `real_qpu = false`.**
Branche `exp/adaptive-qpu-gate` (worktree séparé). Aucun job IBM, aucun contact réseau, aucun
fichier de `Code/`, du solveur original, des résultats officiels ou des WIP du worktree principal
modifié. Artefacts : `results/experimental_instance_scalability_v1/`. Analyse principale déléguée
à un agent indépendant, calculs recoupés par l'orchestrateur (composantes K3 sur 20 instances,
CP_30 synthétique, ISA des composantes FP), contre-review indépendante (§11).

## 1. Question de recherche

Le raffinement adaptatif améliore la précision angulaire à largeur instantanée ≈ 3n, sans
réduire la largeur sous celle du one-hot K3. La question posée ici est distincte : parmi les
1042 instances du dépôt (CP 18, FP 12, GP 12, RCP 400, RCP_FL 600 dont 200 alias `@FL5`),
existe-t-il davantage d'instances réellement accessibles grâce à (a) des blocs hétérogènes
exacts, (b) la réduction aux avions actifs, (c) la décomposition en composantes du graphe de
conflit ? Et que signifie « accessible » ?

## 2. Définitions

* **Graphe any-option K3** : sommets = avions ; arête (i, j) ssi au moins une paire d'options
  de la grille K3 officielle (`ManeuverGrid.build(n_theta=3, n_q=1)`, q = 1) est en conflit
  selon `geometry.in_conflict`. C'est exactement le support des termes de conflit du QUBO K3.
  Le graphe NOOP–NOOP (conflits initiaux) n'est jamais utilisé pour décomposer (il
  sous-estime le couplage).
* **Avion isolé** : degré 0 dans le graphe any-option K3. Le fixer à NOOP est exact : coût nul,
  aucun terme de conflit possible. `Q_active = 3·(n − n_isolés)`.
* **Bloc hétérogène exact** : `K_i = 1` uniquement pour un avion isolé ; sinon réduction par la
  règle de dominance certifiée (option a domine a′ si `cost_a ≤ cost_a′` et sa ligne de
  conflit est incluse dans celle de a′ pour tout voisin ; preuve par échange, rapport expert D).
  Aucune fixation heuristique, aucune hypothèse.
* **Décomposition exacte** : composantes connexes du graphe any-option K3 ; exactitude
  vérifiée **sur le QUBO** (`cross_component_qubo_terms = 0`), pas seulement sur le graphe.
  Le QUBO K3 se factorise alors exactement ; les objectifs sont séparables par avion ; les
  domaines de contrôle sont indépendants ; la réunion des solutions est rescorée sur la
  géométrie complète.
* **Décomposition heuristique** : voisinage de B avions (largeur 3B) sans garantie globale,
  dépendant de l'ordre, exigeant une archive globale et exposé à la réintroduction de conflits.
* **Six niveaux d'accessibilité** : (1) monolithique représentable (`3n ≤ 156` logiques) ;
  (2) compilable (ISA sur FakeMarrakesh, ≤ 60 qubits logiques dans cette étude) ; (3) simulable
  localement (statevector `16·2^Q` octets, `Q ≤ 30`) ; (4) susceptible d'échantillons utiles
  (`≤ 12` qubits logiques et ≈ 210 portes 2Q au plus, régime où la campagne réelle a produit
  des échantillons faisables : 9,5 % à 9 qubits, 1,0 % à 12, 0 % à partir de 15) ;
  (5) décomposable exactement ; (6) décomposable seulement heuristiquement.
* **Classes (schéma 2)** : `DECOMPOSABLE_FORMALLY`, `WIDTH_REDUCED`, `NEWLY_LOCAL_EXACT_ACCESSIBLE`,
  `NEWLY_STATEVECTOR_ACCESSIBLE`, `LOCAL_HEURISTIC_ONLY`, `ISA_COMPILABLE_ONLY`,
  `HARDWARE_PILOT_PLAUSIBLE`, `HARDWARE_SCIENTIFICALLY_IMPLAUSIBLE` (définitions en §6).
  `ISA_COMPILABLE_ONLY` ne signifie jamais qu'une solution utile sera produite. Les qubits
  logiques ne sont jamais confondus avec les qubits physiques : ces derniers ne proviennent que
  des layouts de transpilation effectivement calculés.

## 3. Analyse de toutes les familles (1042 instances, `instances.csv`)

| famille | instances | tailles n | cliques K3 | décomposables exactement | isolés (instances avec ≥ 1) | composante max (min / médiane / max) |
|---|---|---|---|---|---|---|
| CP | 18 | 3–20 | 18/18 | 0 | 0 | 3 / 11,5 / 20 (= n) |
| FP | 12 | 8–30 | 0 | 6 | 0 | 4 / 19 / 24 |
| GP | 12 | 16–60 | 0 (une seule composante 11/12) | 1 (GP_15 : [59, 1]) | 1 | 16 / 38 / 59 |
| RCP | 400 | 10, 20, 30, 40 | 0 | 44 | 37 | 5 / 25 / 40 |
| RCP_FL | 600 | 50, 100, 150 | 0 | 600 | 138 | 10 / 32 / 67 |

Observations vérifiées indépendamment (orchestrateur, tables de conflit officielles) : CP_5 et
CP_10 cliques ; GP_4 et GP_8 une composante ; FP_4 [4, 2, 2] ; FP_5 [4, 4, 2] ; FP_6 [10, 2] ;
FP_15 quatre composantes, la plus grande 24 ; RCP_10_1 [9, 1] ; RCP_50_1 quatre composantes,
la plus grande 21 ; RCP_50_1@FL5 sept composantes, la plus grande 16 ; RCP_100_1 la plus grande
37 ; RCP_100_1@FL5 la plus grande 23. Les niveaux de vol rendent RCP_FL toujours décomposable,
mais la composante restante dépasse presque toujours 12 avions.

Structure : CP est une clique complète (densité 1) ; GP est une composante unique de faible
densité (0,15 à 0,18) ; FP est clairsemé (0,10 à 0,21) et se scinde en flux ; RCP est
intermédiaire (0,31 à 0,33) et rarement scindé à n ≥ 20.

## 4. Méthode de réduction

Pour chaque instance : `Q_global_K3 = 3n` ; `Q_adaptive_homogeneous = 3n` ;
`Q_active = 3·(n − n_isolés)` ; `Q_heterogeneous = Σ K_i` (isolés + dominance, pour chaque
objectif) ; `Q_largest_component = 3·|composante max|` ; voisinages heuristiques `3B`,
B ∈ {3, 5, 7}. La dominance exacte ne retire aucune option sur CP (18/18 et CP_30 synthétique),
conformément à l'expert D ; elle retire 6 qubits sur FP_4 (24 → 18) et jusqu'à 26 sur RCP_50_1
(150 → 124). Un bug de la première version de la règle de dominance (comparaison de tuples au
lieu d'ensembles par voisin) aurait annoncé « 1 qubit par avion » sur CP ; il a été détecté par
recoupement avec les valeurs publiées de l'expert D et corrigé avant l'exécution du corpus.

## 5. Conditions d'exactitude

1. Décomposition : `cross_component_qubo_terms = 0` sur 1042/1042 instances (vérifié sur le
   QUBO construit) ; objectifs séparables ; aucune contrainte globale ; rescoring complet après
   recomposition (§8).
2. **Limite essentielle** : l'exactitude vaut pour le QUBO **K3**. Sous un échantillonnage
   plus dense (K = 65 pour n ≤ 20, K = 9 pour n ≤ 60), les composantes de FP_4, FP_5, FP_6,
   RCP_10_1, RCP_50_1 **fusionnent** (`components_equal_under_dense_sampling = false`). Un
   raffinement adaptatif par composante peut donc recréer un conflit inter-composantes : il
   n'est pas exact et doit être rescoré globalement (mesuré en §8). Cet échantillonnage est un
   proxy fini, pas une preuve sur le continuum.
3. Avions actifs : seul le degré 0 dans le graphe any-option autorise la fixation à NOOP. Un
   avion sans conflit initial mais de degré > 0 n'est jamais retiré.
4. Blocs `K_i = 1` : uniquement isolés ; toute autre réduction passe par la dominance certifiée.

## 6. Matrice d'applicabilité (schéma 2)

Vocabulaire (`family_summary.json`, `instances.csv`) : `DECOMPOSABLE_FORMALLY` (≥ 2 composantes,
0 terme croisé, K3 seulement ; ne signifie jamais « accessible ») ; `WIDTH_REDUCED` (plus grande
composante ou blocs hétérogènes exacts strictement sous 3n) ; `NEWLY_LOCAL_EXACT_ACCESSIBLE` (la
plus grande composante entre dans le budget exact de l'étude alors que le monolithique n'y entre
pas) ; `NEWLY_STATEVECTOR_ACCESSIBLE` (composante ≤ 30 qubits logiques alors que 3n > 30) ;
`LOCAL_HEURISTIC_ONLY` ; `ISA_COMPILABLE_ONLY` (30 < largeur ≤ 156, jamais « exploitable ») ;
`HARDWARE_PILOT_PLAUSIBLE` (≤ 12 qubits logiques et 2Q ISA **mesuré** ≤ 210) ;
`HARDWARE_SCIENTIFICALLY_IMPLAUSIBLE`.

| famille | n | DECOMP._FORMALLY | WIDTH_REDUCED | NEWLY_LOCAL_EXACT | NEWLY_STATEVECTOR | LOCAL_HEUR._ONLY | ISA_COMP._ONLY | PILOT_PLAUSIBLE | IMPLAUSIBLE | > 156 logiques |
|---|---|---|---|---|---|---|---|---|---|---|
| CP | 18 | 0 | 0 | 0 | 0 | 7 | 10 | 2 (CP_3, CP_4) | 16 | 0 |
| FP | 12 | 6 | 12 | 0 | 1 (FP_6) | 9 | 9 | 2 (FP_4, FP_5) | 10 | 0 |
| GP | 12 | 1 (GP_15) | 2 | 0 | 0 | 12 | 10 | 0 | 12 | 2 |
| RCP | 400 | 44 | 375 | 0 | 0 | 300 | 300 | 0 | 400 | 0 |
| RCP_FL | 600 | 600 | 600 | 54 | 2 (RCP_50_37@FL5, RCP_50_67@FL5) | 546 | 517 | 0 | 600 | 400 |

Instances réellement nouvellement accessibles et propriété qui le permet :

* accès statevector local (≤ 30 qubits) uniquement grâce à la composante : FP_6 (36 → 30),
  RCP_50_37@FL5 et RCP_50_67@FL5 (150 → 30) ;
* accès exact local uniquement grâce à la composante (budget 3^13 et 60 s) : 54 instances
  RCP_FL à 50 avions, toutes en alias `@FL5` (cinq niveaux de vol), liste nominative dans
  `family_summary.json` ;
* régime pilote matériel : CP_3 (108 portes 2Q), CP_4 (207), et par décomposition FP_4 (142) et
  FP_5 (142), mesures ISA hors ligne sur FakeMarrakesh ; aucune instance RCP ou RCP_FL.

`WIDTH_REDUCED` est fréquent (375/400 RCP par dominance ou isolement, 600/600 RCP_FL) mais la
composante restante dépasse presque toujours 12 qubits : une largeur réduite ne signifie pas une
résolution exacte globale. `DECOMPOSABLE_FORMALLY` : FP 6/12, GP 1/12 (gain pratiquement nul),
RCP 44/400, RCP_FL 600/600.

Frontière de transpilation (FakeMarrakesh, p = 1, γ = 0,4, β = 0,3, seed 1, opt-level 0,
limite dure 60 qubits logiques, `transpile_frontier.json`, reproductible en double passage) :

| cas | logiques | profondeur ISA | portes 2Q | statut |
|---|---|---|---|---|
| CP_5 global | 15 | 376 | 348 | identique à la campagne réelle |
| CP_7 global | 21 | 640 | 700 | identique à la campagne réelle |
| CP_10 global | 30 | 1162 | 1570 | compilable, non crédible |
| FP_4 global / hétérogène | 24 / 18 | 361 / 166 | 441 / 185 | compilable |
| FP_4 composante [1,3,5,7] | 12 | 220 | 142 | régime pilote (≈ CP_4 : 12 / 259 / 207) |
| FP_5 composantes [1,3,6,8], [2,5,7,10] | 12, 12 | 220, 148 | 142, 114 | régime pilote |
| FP_6 composante principale | 30 | 343 | 587 | compilable, non crédible |
| RCP_10_1 global / hétérogène / composante | 30 / 19 / 27 | 331 / 109 / 304 | 425 / 108 / 413 | compilable, non crédible |
| RCP_20_1 global / hétérogène | 60 / 56 | 769 / 547 | 1708 / 1141 | compilable, non crédible |
| FP_13, GP_15, RCP_50_1, RCP_150_32 | 72–450 | — | — | refusés (> 60 qubits) |

Mémoire statevector : `16·2^Q` octets ; 30 qubits = 17 Go ; 36 qubits = 1,1 To ; CP_30
synthétique (90 qubits) = 2·10^28 octets.

## 7. Réponse CP30

1. **CP_30 n'existe pas dans le dépôt** (CP s'arrête à CP_20). Une instance **synthétique**
   a été construite selon les conventions de CP_20 (v0 = 5, d = 0,05, rayon 2, positions
   uniformes, cap vers le centre) et marquée `synthetic = true` partout.
2. Avions actifs : 30 (aucun isolé) ; 435 conflits initiaux sur 435 paires.
3. Densité any-option K3 : 1,0 (clique complète).
4. Plus grande composante : 30.
5. Qubits K3 global : 90 logiques.
6. Après réduction exacte : 90 (aucun isolé).
7. Après blocs hétérogènes exacts : 90 pour les deux objectifs (dominance sans effet sur CP).
8. Profondeur et portes 2Q : non transpilé (90 qubits) ; extrapolation à partir des cinq points
   réels CP_3..CP_7 : ajustement linéaire ≈ 3 060 de profondeur ; ajustement quadratique
   (justifié par la croissance en n² des termes) ≈ 12 900 de profondeur et ≈ 14 400 portes 2Q
   (leave-one-out : 9 900 à 15 800 et 8 700 à 16 700).
   Ordre de grandeur, pas prédiction. Termes quadratiques : 1250 (coordonnées exactes) ou 1259
   (coordonnées arrondies façon `.dat`) ; la table de conflit est sensible à l'arrondi.
9. Simulable localement : non par statevector (2^90) ; SA classique trouve une affectation sans
   conflit en 1,5 s, sans référence d'optimalité.
10. Compilable sur Marrakech : en largeur oui (90 ≤ 156 logiques), non tenté ; ne prouve rien.
11. Scientifiquement crédible : non. CP_5 (15 qubits, 348 portes 2Q) donnait déjà 0 %
    d'échantillons faisables sur le matériel ; CP_30 est un à deux ordres de grandeur au-delà.
12. Décomposable exactement : non (clique).
13. Seulement heuristiquement : oui, par voisinages, sans garantie.
14. Coût d'une chaîne adaptative : 4 rounds × 512 shots par sous-problème, séquentiels, sans
    réduction de largeur (90 par round).

**Verdict retenu par l'orchestrateur : `CP30_ISA_COMPILABLE_BUT_NOT_CREDIBLE`.** L'analyste
proposait `CP30_ACCESSIBLE_HEURISTICALLY` au motif qu'un recuit classique trouve une
affectation faisable ; cette formulation est écartée car la question porte sur l'accessibilité
par la méthode quantique étudiée, et une faisabilité classique sans référence d'optimalité
n'établit pas une accessibilité. Le fait classique est conservé comme remarque.

## 8. Benchmark local (décomposition)

`results/experimental_instance_scalability_v1/decomposition_benchmark/` (script
`scripts/experimental_decomposition_benchmark.py`, composantes calculées par le chemin officiel
`DiscreteACRP`, indépendant du module d'analyse ; B&B exact avec délai de 120 s par
sous-problème ; adaptatif ρ = 0,5, 8 rounds, exact par round ; recomposition toujours rescorée
sur la géométrie complète ; durée totale 1222 s).

### 8.1 Objectif quadratique (q = 1)

| instance | composantes | largeur max mono → décomp. | K3 monolithique | K3 décomposé (rescoré) | égal | adaptatif par composante (grilles raffinées, rescoré) | conflits après recomposition |
|---|---|---|---|---|---|---|---|
| CP_5 | [5] | 15 → 15 | 0,548311 | 0,548311 | oui | 0,001322 | 0 |
| CP_8 | [8] | 24 → 24 | 0,959544 | 0,959544 | oui | 0,005422 | 0 |
| GP_4 | [16] | 48 → 48 | 1,096622 | 1,096622 | oui | 0,109610 | 0 |
| FP_4 | [4, 2, 2] | 24 → 12 | 0,548311 | 0,548311 | oui | 0,012541 | 0 |
| FP_5 | [4, 4, 2] | 30 → 12 | 0,685389 | 0,685389 | oui | 0,017377 | 0 |
| FP_6 | [10, 2] | 36 → 30 | 0,822467 | 0,822467 | oui | 0,055947 | 0 |
| FP_7 | [14] | 42 → 42 | 0,959544 | 0,959544 | oui | 0,254720 | 0 |
| FP_15 | [24, 2, 2, 2] | 90 → 72 | délai dépassé | incomplet (composante 24 : délai) | — | délai | — |
| RCP_10_1 | [9, 1] | 30 → 27 | 0,274156 | 0,274156 | oui | 0,011646 | 0 |
| RCP_10_2 | [10] | 30 → 30 | 0,274156 | 0,274156 | oui | 0,002460 | 0 |
| RCP_20_1 | [20] | 60 → 60 | 0,685389 | 0,685389 | oui | 0,121800 | 0 |
| RCP_50_1 | [21, 14, 14, 1] | 150 → 63 | refusé (3^50) | 2,330322 (incumbent, sans optimalité globale) | n.d. | délai sur la composante 21 | — |
| RCP_50_1@FL5 | [16, 10, 9, 7, 6, 1, 1] | 150 → 48 | refusé | 1,782011 (incumbent) | n.d. | 0,120353 | 0 |
| RCP_100_1@FL5 | [23, 22, 21, 17, 17] | 300 → 69 | refusé | 5,346033 (incumbent) | n.d. | délai (3 composantes) | — |

Lecture : partout où le monolithique est calculable, la recomposition exacte reproduit son
optimum à 1e-12 près avec zéro conflit (12/12 cas). Pour n ≥ 50 le monolithique est refusé et
la décomposition fournit un incumbent faisable, sans revendication d'optimalité. La colonne
adaptative est sur des **grilles différentes** (`domains_comparable_adaptive_vs_K3 = false`) :
elle mesure un gain de résolution angulaire, pas un gain de décomposition. Aucun conflit
inter-composantes n'est apparu après recomposition des solutions adaptatives sur ces instances,
mais la garantie n'existe pas (§5.2, conflit hors grille construit par la contre-review sur FP_4).

### 8.2 Objectif `maneuver_count_v1` (contrôle négatif)

| instance | K3 monolithique | K3 décomposé | égal | adaptatif par composante (NOOP forcé) |
|---|---|---|---|---|
| CP_5 / CP_8 / GP_4 | 4 / 7 / 8 | 4 / 7 / 8 | oui | 4 / 7 / 8 |
| FP_4 / FP_5 / FP_6 / FP_7 | 4 / 5 / 6 / 7 | idem | oui | idem |
| RCP_10_1 / RCP_10_2 / RCP_20_1 | 2 / 2 / 5 | idem | oui | idem |
| RCP_50_1 | refusé | 17 (incumbent) | n.d. | 17 |
| RCP_50_1@FL5 | refusé | 13 (incumbent) | n.d. | 13 |
| RCP_100_1@FL5 | refusé | 39 (incumbent) | n.d. | **38** (0 conflit après rescoring) |

Le résultat négatif est conservé : sur toutes les instances où l'optimum K3 est certifié,
l'adaptatif ne réduit pas le nombre d'avions déviés. La seule exception, RCP_100_1@FL5, oppose
un incumbent K3 (non certifié optimal) à une solution sur grilles raffinées ; elle montre qu'une
grille plus fine peut permettre une manœuvre de moins, pas que l'adaptatif bat l'optimum K3. La
décomposition, elle, réduit la largeur indépendamment de l'objectif (150 → 48, 300 → 69).

### 8.3 Ressources

Largeur cumulée de l'adaptatif (qubits × rounds) : 116 (CP_5) à 2454 (RCP_100_1@FL5) contre
3n pour un passage K3 ; rounds : 8 ; évaluations exactes et temps dans `benchmark.json`.
Mémoire statevector de la plus large composante : 48 qubits → 4,5 Po (RCP_50_1@FL5), 69
qubits → 9·10^21 octets : la décomposition ne rend aucune de ces instances simulable par
vecteur d'état.

## 9. Comparaison de ressources

| stratégie | largeur max | largeur cumulée | sous-problèmes | rounds | exactitude |
|---|---|---|---|---|---|
| one-hot K3 monolithique | 3n | 3n | 1 | 1 | exacte sur K3 |
| réduction active | 3(n − isolés) | idem | 1 | 1 | exacte |
| blocs hétérogènes exacts | Σ K_i | idem | 1 | 1 | exacte (dominance) |
| décomposition exacte | 3·|C_max| | 3n | #composantes | 1 | exacte sur K3, rescoring obligatoire |
| adaptatif par composante | 3·|C_max| | 3n × rounds | #composantes | R | locale par round ; recomposition à rescorer |
| voisinage heuristique B | 3B | ≥ 3n | ≥ n/B | itératif | aucune garantie |

Le gain de la décomposition porte sur la largeur maximale et la mémoire de simulation, jamais
sur le nombre total de qubits-rounds ni sur le temps total.

## 10. Résultats négatifs

* CP : clique pour tout n mesuré et pour CP_30 synthétique ; aucune réduction (active,
  dominance, composantes).
* GP : une seule composante jusqu'à n = 56 ; GP_15 isole un avion sur 60 (sans intérêt).
* RCP et RCP_FL : la décomposition, même exacte, ne rend aucune instance plausible pour un
  pilote matériel (0/1000) ; elle aide la simulation, pas le matériel.
* La dominance exacte ne réduit rien sur CP et GP.
* `maneuver_count_v1` : aucun gain de l'adaptatif (résultat négatif conservé) ; la
  décomposition réduit la largeur indépendamment de l'objectif (§8).
* La simulation Aer monolithique est déjà hors de portée (> 30 qubits) pour CP ≥ 11, FP ≥ 6,
  tout GP, RCP ≥ 20 et tout RCP_FL.

## 11. Menaces à la validité

* Les seuils de plausibilité matérielle reposent sur trois points réels (CP_3, CP_4, CP_5).
* L'exactitude de la décomposition vaut pour le QUBO K3 ; l'échantillonnage dense montre des
  fusions de composantes dès que la grille s'affine ; le raffinement adaptatif par composante
  n'hérite pas de l'exactitude.
* Les références fines certifiées n'existent que pour n ≤ 6 ; au-delà, seules des valeurs K3
  exactes et des incumbents sont publiés.
* CP_30 est synthétique ; l'extrapolation ISA repose sur cinq points.
* Budget d'exactitude locale : 60 s et `max_states` finis ; un refus n'est pas une preuve
  d'infaisabilité.
* Contre-review indépendante : §11.1.

### 11.1 Contre-review indépendante (rapport `moe_reports/scal_counter_review.md`)

Vérifié sans écart : comptes de qubits (30 instances aléatoires), termes QUBO (CP_5 = 45,
FP_4 = 36, CP_30 = 1250), indépendance des composantes (0 terme croisé sur 10 RCP/RCP_FL
aléatoires), NOOP jamais retiré (1042/1042), dominance sans perte d'optimum (40/40 RCP_10,
deux objectifs), formule mémoire. Constats corrigés avant gel :

| constat | sévérité | correction |
|---|---|---|
| le script de benchmark de décomposition n'avait aucun test ; un mutant « sans rescoring global » passait | CRITICAL | `tests/test_experimental_decomposition_benchmark.py` : rescoring sur géométrie complète, split erroné détecté, égalité décomposition = monolithique sur FP_4 pour les deux objectifs |
| `HARDWARE_PILOT_PLAUSIBLE` était attribué sans mesure ISA (`expected_isa_2q = None` rendait la condition vide) ; correct par coïncidence pour FP_4/FP_5 (142 portes 2Q mesurées) | MAJOR | le classificateur refuse le label sans mesure ; le balayage transpile la plus grande composante des candidats ≤ 12 qubits et enregistre profondeur et portes 2Q |
| conflit inter-composantes construit sur FP_4 à des angles hors grille K3 (avions 1 et 6) | MAJOR (confirme la limite §5.2) | rescoring obligatoire de la recomposition adaptative ; exactitude revendiquée pour K3 seulement |
| `adaptive_gain_vs_decomp_K3` comparait deux domaines (K3 fixe vs grilles raffinées) | MAJOR | champ renommé `..._DIFFERENT_GRIDS` avec `domains_comparable_adaptive_vs_K3 = false` |
| verdict CP30 « accessible heuristiquement » fondé sur un recuit classique | MAJOR | verdict `CP30_ISA_COMPILABLE_BUT_NOT_CREDIBLE` ; fait classique relégué en note |
| écart 1250 / 1259 termes sur CP_30 synthétique | MINOR | arrondi des coordonnées (exactes vs deux décimales du `.dat`), 0,8 %, sans effet qualitatif |
| test de dominance sur échantillon aléatoire absent | MINOR | ajouté (40 instances, deux objectifs) |

## 12. Rôle dans le mémoire

Le raffinement adaptatif améliore la précision à largeur K3 constante. La réduction
structurelle et la décomposition déterminent, séparément, si cette largeur peut être appliquée
à des instances comportant davantage d'avions. Ces deux bénéfices ne se fusionnent pas : sur
CP la décomposition n'apporte rien ; sur FP elle abaisse la largeur de 24–30 à 12 pour FP_4 et
FP_5 seulement ; sur RCP_FL elle abaisse la largeur de 150–450 à 30–200, ce qui reste hors du
régime matériel crédible.

## 13. Affirmations autorisées et interdites

Autorisées : « la décomposition exacte est vérifiée sur le QUBO K3 de 1042 instances » ;
« FP_4 et FP_5 se décomposent en composantes de 12 qubits logiques dont le circuit ISA
(220 de profondeur, 142 portes 2Q) est dans le régime où la campagne réelle a produit des
échantillons faisables » ; « aucune instance RCP ou RCP_FL n'atteint ce régime » ; « CP_30
n'existe pas ; sa version synthétique est une clique de 90 qubits, non crédible sur matériel ».

Interdites : « un QPU de 156 qubits peut résoudre un QUBO de 156 variables » ; « compilable
donc exploitable » ; « décomposable donc résolu » ; toute assimilation qubits logiques /
physiques ; « CP_30 accessible » ; toute exactitude du raffinement adaptatif par composante ;
tout gap entre domaines différents (q = 1 discret vs continu q + θ) sans qualification.

## 14. Gates de cette étude

| gate | résultat |
|---|---|
| `ruff check` sur les fichiers de l'étude (`src/acrpq/experimental`, `scripts/experimental_*`, `tests/test_experimental_*`) | vert |
| `mypy src/acrpq` | vert (68 fichiers) |
| tests expérimentaux (grille adaptative, QUBO/chaîne, scalabilité, benchmark de décomposition) | verts (voir nombre dans le rapport de clôture) |
| tests voisins (discretize, equivalence, objectives, import safety) | verts |
| tests de décomposition (recomposition exacte, split erroné détecté, rescoring obligatoire, dominance aléatoire) | verts |
| `build --no-isolation` | vert (sdist + wheel) |
| `pip check` | vert |
| `pip-audit` | 1 vulnérabilité préexistante de l'environnement (`diskcache 5.6.3`, PYSEC-2026-2447), non neutralisée, hors périmètre de l'étude |
| `git diff --check` | vert |
| parité JSON/CSV | sémantique exacte (0 divergence ; les listes CSV sont encodées JSON) ; benchmark de décomposition idem |
| manifests/hashes | sha256 par fichier dans chaque `manifest.json` ; commit source enregistré |
| reproduction depuis un worktree propre | le balayage complet a été rejoué deux fois avec les mêmes classes ; commandes dans `moe_reports/scal_analyst_report.md` §9 |
| scan secrets/PII/chemins | vide sur les nouveaux fichiers |
| `Code/` et solveur original | hash agrégé identique à la phase 0 dans les deux worktrees ; aucun diff sur les chemins protégés de la branche |
| WIP préservés | hashes identiques à la phase 0, sauf trois fichiers committés entre-temps par l'autre session (non touchés par cette étude) |
| inspection indépendante | contre-review (§11.1) et re-vérification des correctifs par tests |

Exceptions de la branche expérimentale, corrigées au commit d'intégration `6faaaa8` : les deux
erreurs ruff de `docs/presentation/` et la dépendance de
`test_qpu_real_artifact.py::test_campaign_artifacts_present_and_inventoried` à des exports IBM
privés non versionnés (désormais scindé en tests obligatoires sur artefacts publiés et tests
optionnels ignorés avec raison). En clone-fresh du commit d'intégration : `ruff check .` vert,
suite complète 1458 verts, 10 ignorés, 0 échec.

## 15. Section autonome pour le mémoire

1. **Question.** La largeur instantanée 3n du raffinement adaptatif peut-elle être appliquée
   à des instances comportant davantage d'avions grâce à la structure du problème ?
2. **Graphe.** Sommets = avions ; arête (i, j) ssi une paire d'options de la grille K3
   officielle (q = 1) est en conflit selon le noyau géométrique commun. C'est le support exact
   des termes de conflit du QUBO K3.
3. **Théorème.** Si aucun coefficient quadratique du QUBO K3 ne relie deux composantes
   connexes de ce graphe, alors H(x) = Σ_c H_c(x_c), l'argmin one-hot se factorise, et la
   réunion des solutions par composante est un optimum global du QUBO K3, sans conflit
   résiduel.
4. **Preuve.** Les termes one-hot sont internes à chaque bloc ; les termes de conflit relient
   uniquement des paires dont la table K3 contient un vrai ; par définition des composantes,
   toute paire inter-composantes a une table entièrement fausse, donc aucun terme et aucun
   conflit possible dans la géométrie K3 pour toute combinaison d'options. L'objectif est
   séparable par avion. L'exactitude est limitée à la grille K3 : les tables de conflit dépendent
   de la grille et des composantes fusionnent sous un échantillonnage plus fin.
5. **Algorithme.** Construire les tables K3, le graphe, les composantes (union-find) ; vérifier
   `cross_component_qubo_terms = 0` sur le QUBO construit ; résoudre chaque sous-instance ;
   recomposer par identifiant d'avion ; rescorer sur la géométrie complète ; ne publier
   « optimum » que si le monolithique est certifié, sinon « incumbent ».
6. **Familles.** CP : clique pour tout n (et CP_30 synthétique) ; GP : une composante jusqu'à
   n = 56 ; FP : scindé 6/12 ; RCP : 44/400 ; RCP_FL : 600/600 grâce aux niveaux de vol.
7. **Résultats.** 1042/1042 instances sans terme inter-composantes (contre-preuve indépendante
   par les seuls modules officiels) ; 24/24 recompositions (12 configurations × 2 objectifs)
   égales au monolithique à 1e-12 avec zéro conflit ; six mutations obligatoires détectées par
   des tests. Nouveaux accès : FP_6 et deux RCP_FL en statevector ; 54 RCP_FL@FL5 en exact
   local ; FP_4 et FP_5 rejoignent CP_3 et CP_4 dans le régime pilote (12 qubits, 142 portes 2Q).
8. **Ressources.** La décomposition réduit la largeur maximale (150 → 48, 300 → 69) mais ni la
   largeur cumulée ni la mémoire au-delà de 30 qubits (48 qubits ≈ 4,5 Po de vecteur d'état).
9. **CP30.** Absent du corpus ; version synthétique : 30 avions, K3, 90 qubits logiques,
   clique, aucun isolé, aucune composante, aucune réduction exacte, 2·10²⁸ octets de vecteur
   d'état ; profondeur ISA **estimée** (non transpilée) par ajustements linéaire (≈ 3 060,
   leave-one-out 2 600 à 3 460) et quadratique (≈ 12 900, leave-one-out 9 900 à 15 800) sur
   cinq points réels ; conclusion `CP30_ISA_COMPILABLE_BUT_NOT_CREDIBLE`.
10. **Limitations.** Exactitude K3 seulement ; seuils matériels calibrés sur trois points
    réels ; budget exact fini ; références fines certifiées pour n ≤ 6 seulement.
11. **Conséquences pour l'adaptatif.** Le raffinement adaptatif par composante n'hérite pas
    de l'exactitude (composantes fusionnant sous grille fine) et exige un rescoring global ;
    sur les instances testées aucun conflit inter-composantes n'est apparu, sans garantie.
12. **Affirmations permises / interdites.** Voir §13 ; en particulier « décomposable » ≠
    « accessible », « compilable » ≠ « exploitable », CP30 jamais « accessible ».
13. **Paragraphe prêt à intégrer.** « L'analyse structurelle des 1042 instances du dépôt
    montre que la décomposition en composantes du graphe de conflit any-option K3 est exacte
    pour le QUBO K3 : aucun terme quadratique ne relie deux composantes, et la recomposition
    reproduit l'optimum monolithique sur toutes les configurations vérifiables. Cette
    décomposition est formellement possible sur 6 des 12 instances FP, 44 des 400 RCP et la
    totalité des 600 RCP_FL, mais elle n'ouvre un accès réellement nouveau que dans un petit
    nombre de cas : trois instances deviennent simulables par vecteur d'état (FP_6 et deux
    RCP_FL), 54 RCP_FL entrent dans le budget de résolution exacte locale, et FP_4 et FP_5
    rejoignent CP_3 et CP_4 dans le régime où la campagne réelle avait produit des échantillons
    faisables. CP reste une clique complète pour tout n, et CP_30, absent du corpus, serait une
    clique de 90 qubits logiques compilable en largeur mais non crédible. Le raffinement
    adaptatif apporte la précision à largeur constante ; la décomposition apporte la largeur ;
    ces deux bénéfices restent distincts et l'exactitude de la seconde ne s'étend pas aux grilles
    raffinées de la première. »
