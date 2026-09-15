# Étude d'un encodage ACRP plus compact — grille adaptative itérative vs one-hot

**Statut : étude indépendante, expérimentale. `experimental = true`, `official_benchmark = false`, `real_qpu = false`.**
Aucun résultat officiel, aucun encodage officiel, aucun job IBM n'a été modifié ou soumis.
Le module de preuve de concept vit dans `src/acrpq/experimental/adaptive_grid.py` et n'est
importé par rien d'autre (vérifié par test). Les artefacts sont sous
`results/experimental_adaptive_grid_v1/`, les cinq rapports d'experts sous
`results/experimental_adaptive_grid_v1/moe_reports/`.

Organisation : cinq experts indépendants (A encodage binaire, B domain-wall, C grille
adaptative, D décomposition / candidate-set / LNS, E red-team) ont travaillé en lecture seule
sur le dépôt et sur les instances réelles `CP_n`. L'orchestrateur a implémenté la preuve de
concept isolée, ajouté les contrôles exigés par le red-team, puis rédigé ce document.

**Verdict : `LOCAL_PROOF_OF_CONCEPT_VALIDATED` pour l'objectif `quadratic_control_cost_v1`.**
Aucun gain pour `maneuver_count_v1` (constaté, pas masqué). Pas `READY_FOR_FUTURE_QPU_STUDY` :
le QUBO par avion hétérogène, son décodage sous échantillonnage bruité et la comptabilité
réelle des shots ne sont pas validés (§11).

---

## 1. Le problème de scalabilité

L'encodage officiel est one-hot : `x[i,o] ∈ {0,1}`, `N_qubits = n·K`. La précision angulaire
d'une grille globale uniforme sur `[-π/6, +π/6]` est `δ = (π/3)/(K-1)`. Les optima continus
Gurobi des instances CP (`results/ampl_gurobi_campaign_v3/CP_n.json`, lecture seule) ont des
`θ*` de l'ordre de 0,014 à 0,023 rad. Pour qu'une grille globale contienne un niveau à
0,005 rad d'un tel optimum il faut un pas de 0,01 rad, donc `K ≈ 106`.

| résolution δ (rad) | K | n=3 | n=5 | n=8 | n=10 | n=20 |
|---|---|---|---|---|---|---|
| π/6 = 0,524 | 3 | 9 | 15 | 24 | 30 | 60 |
| π/12 = 0,262 | 5 | 15 | 25 | 40 | 50 | 100 |
| π/24 = 0,131 | 9 | 27 | 45 | 72 | 90 | 180 |
| 0,02 | 54 | 162 | 270 | 432 | 540 | 1080 |
| 0,01 | 106 | 318 | 530 | 848 | 1060 | 2120 |

Le budget pratique est `max_qubits = 29` en simulation et 21 à 27 qubits sur IBM. La
motivation est donc réelle : la précision angulaire utile est hors de portée de toute méthode
dont le coût croît linéairement (ou plus) en `K`. C'est la contrainte qui structure toute
l'étude : un candidat n'est intéressant que s'il découple le coût de `K`.

## 2. L'encodage one-hot actuel (référence)

```
H(x) = Σ_{i,o} cost_o x[i,o]
     + λ_pen Σ_{(i,j)} Σ_{(oi,oj) ∈ conflict_ij} x[i,oi] x[j,oj]
     + λ_oh  Σ_i (Σ_o x[i,o] − 1)²
λ_pen = 10 (n·cost_max + 1),   λ_oh = max(n, maxdeg+1) · λ_pen
```

Propriétés : `n·K` qubits, 0 ancilla, degré 2 natif, `n·C(K,2)` termes one-hot,
`n_conflict_terms` termes de conflit (mesuré : ≈ `K` entrées non nulles par paire sur CP, la
matrice de conflit réelle est quasi une permutation anti-diagonale), fraction de bitstrings
valides `K/2^K`, décodage par argmax de bloc, théorème de dominance prouvé et vérifié en code
(`verify_qubo_wellformed`). C'est le seul candidat avec une preuve d'équivalence vérifiée.
Tout candidat est comparé à ce socle, pas à une version idéalisée.

## 3. Candidats étudiés

| # | candidat | expert |
|---|---|---|
| 1 | one-hot global (référence) | tous |
| 2 | binaire / logarithmique de l'indice, `m = ⌈log₂K⌉` bits par avion | A, E |
| 3 | domain-wall, `K−1` bits par avion | B, E |
| 4 | sparse / candidate-set par dominance exacte | D |
| 5 | grille adaptative itérative K=3 (proposition principale) | C, E, PoC |
| 6 | décomposition par composantes du graphe de conflit | D |
| 7 | hybride warm-start classique + voisinage (LNS) | D |

### 3.1 Tableau de cartographie (n avions, K options, m = ⌈log₂K⌉, R rounds)

| critère | 1 one-hot | 2 binaire (QUBO natif) | 3 domain-wall | 4 candidate-set | 5 adaptatif K3 | 6 composantes | 7 LNS |
|---|---|---|---|---|---|---|---|
| qubits logiques | `nK` | `nm` | `n(K−1)` | `Σ|S_i| ≤ nK` | `3n` par round (`4n` avec NOOP) | `max_c |V_c|·K` par run | `|D|·K` par run |
| ancillas | 0 | `n·K·(m−1)` (conflits quadratisés, schéma E) ; 4/26/120 par paire dans le schéma dense de A | 0 | 0 | 0 | 0 | 0 |
| degré avant quadratisation | 2 | `2m` (conflit), `m` (coût par indice) | 2 | 2 | 2 | 2 | 2 |
| termes quadratiques | `nC(K,2) + conflits` | après ancillas : plus que one-hot dès n≥5 | intra `n(K−2)` mais conflits ×1,3 à ×2,5 (mesuré) | ≤ one-hot | `Σ_i C(K_i,2) + conflits locaux` par round | identiques, regroupés | sous-problème seul |
| fraction de bitstrings valides | `K/2^K` | `K/2^m` + cohérence ancillas | `K/2^{K−1}` (×2) | ≥ one-hot | `Π_i K_i/2^{K_i}` | idem one-hot | idem one-hot |
| expressivité vs grille K | référence | identique | identique | identique (dominance exacte) | résolution implicite `Δ_R` avec portée bornée | identique | restreinte au voisinage |
| optimalité | exacte sur la grille | exacte si dominance triple prouvée (non faite) | exacte sous `λ_dw > λ_pen·r_max·maxdeg` | exacte | round 0 exact ; ensuite optimum local | exacte si graphe any-option | monotone seulement |
| appels QPU | 1 | 1 | 1 | 1 | `R` (5 à 12) | `#composantes` | itérations LNS |
| gain sur CP | — | négatif | `n` qubits, densité pire | **0 %** (mesuré) | largeur `3n` constante | **0 %** (clique) | dépend de `|D|` |

Le nombre brut de qubits ne suffit jamais à conclure : les candidats 2 et 3 réduisent le
registre mais augmentent la densité ou les ancillas ; les candidats 4 et 6 sont exacts et
gratuits mais inopérants sur la famille CP ; seul le candidat 5 découple le coût de `K`.

## 4. Dérivations mathématiques

### 4.1 Encodage binaire (expert A, contre-vérifié par E)

Indicatrice d'option : `1[idx(b_i)=a] = Π_k (b_ik si bit_k(a)=1 sinon 1−b_ik)`, degré `m`,
jusqu'à `2^m` monômes (développement de Möbius). Le conflit `1[idx(b_i)=a]·1[idx(b_j)=b]` est
de degré `2m`. Deux comptages indépendants d'ancillas donnent le même verdict :

* schéma « chaîne à préfixes partagés » par paire (A) : 4 / 26 / 120 ancillas par paire pour
  m = 2 / 3 / 4 dans le cas dense ; `2m−2` par paire dans le cas structuré CP (conflit =
  produit de XNOR) ;
* schéma « un bit matérialisé par option impliquée » (E) : `n·K·(m−1)` ancillas au total, soit
  `n(m + K(m−1))` qubits : 5n (K=3), 13n (K=5), 17n (K=7), 31n (K=9) contre `nK`.

Le terme de conflit se compte par paire (`O(n²)`) alors que le gain de registre est `O(n)` :
le binaire perd d'un facteur 1,5 à 3,4 pour tout K ∈ {3..9}, et le seul cas favorable trouvé
(n=3, K=8, schéma optimiste) disparaît dès n ≥ 5. Il ajoute un troisième niveau de pénalité
(cohérence des ancillas) qui dégrade la dynamique des coefficients. Point positif isolé : la
variante arithmétique `θ_i = Δ(Σ_k 2^k b_ik − offset)` rend `w·θ_i²` quadratique sans ancilla,
mais uniquement pour `quadratic_control_cost_v1` ; l'indicatrice « ≠ NOOP » de
`maneuver_count_v1` redevient de degré `m`. Conclusion : le gain `K → log₂K` ne survit pas.

### 4.2 Domain-wall (expert B, contre-vérifié par E)

`x_ia = z_ia − z_{i,a+1}` avec `z_i0 = 1`, `z_iK = 0`. Coût linéaire pour les deux objectifs
(`Σ_a (cost_a − cost_{a−1}) z_ia + cost_0` ; pour `maneuver_count_v1` seuls deux coefficients
sont non nuls). Conflit quadratique natif : chaque entrée `(a,b)` produit jusqu'à quatre
monômes. Mesuré sur CP réel : le conflit a **1,3 à 2,5 fois plus** de couplages non nuls
qu'en one-hot, car les matrices de conflit sont des permutations (pas de télescopage). Bilan
total intra + conflit : domain-wall gagne à n=3 (ratio 0,64 à 0,87) et perd dès n≥5 (jusqu'à
×1,8 à n=20, K=7).

Faille de signe : sur un état invalide, `x_ia` peut valoir −1 et la pénalité de conflit devient
négative. Borne dérivée : `λ_dw > λ_pen · r_max · maxdeg` où `r_max` est le degré maximal
ligne/colonne de la table de conflit (E obtient la borne plus conservative
`λ_pen · maxdeg · (K−1)`). Sur CP `r_max = 1`, mais la borne dépend de la géométrie, ce que
le one-hot n'a pas. Gain net : exactement `n` qubits et un doublement de la fraction valide,
au prix d'une densité plus grande et d'un décodage par comptage de murs.

### 4.3 Candidate-set et composantes (expert D)

Règle de dominance exacte : `a` domine `a'` si `cost_a ≤ cost_a'` et, pour tout `j`,
`conflict_ij[a] ⊆ conflict_ij[a']`. Preuve par échange : remplacer `a'` par `a` dans un
optimum ne peut ni augmenter le coût ni créer un conflit, donc il existe un optimum sans option
dominée. Mesuré : **0 option éliminée sur CP_3..CP_20 pour K ∈ {3,5,7,9} et les deux
objectifs** (symétrie de rotation) ; −9 % à −45 % de qubits sur FP et RCP. Le graphe de conflit
« any-option » de CP est une clique complète pour tout n et K : aucune décomposition. Il ne faut
jamais décomposer sur le graphe NOOP seul (GP se scinde en NOOP mais reste une clique en
any-option, la garantie serait perdue).

### 4.4 Grille adaptative (expert C, PoC, red-team E)

```
Θ_i^(r) = dedup(clip{ c_i^(r) − Δ_r, c_i^(r), c_i^(r) + Δ_r })   (∪ {0} si NOOP forcé)
c_i^(r+1) = θ_i^*(r)           Δ_(r+1) = ρ Δ_r,  0 < ρ < 1
```

* Identité round 0 : avec `c = 0` et `Δ_0 = HMAX = π/6`, `Θ^(0) = {−π/6, 0, π/6}` est exactement
  la grille K3 officielle (testé : `official_k_grid_options == ManeuverGrid.theta_levels`).
* Lemme du centre : `c_i^(r)` est toujours dans `[HMIN, HMAX]` (0, ou une option d'une grille
  déjà clippée), donc `clip` est l'identité sur le centre et `dedup` ne le retire jamais. La
  solution précédente appartient toujours à la grille suivante (enregistré par round dans
  `previous_solution_in_grid`, vrai sur toutes les exécutions).
* Théorème de monotonie (solveur de round exact) : `f_best^(r+1) ≤ f_best^(r)`. Preuve : la
  solution archivée est réalisable dans la grille `r+1` (lemme du centre), un solveur exact
  renvoie une valeur ≤ à la sienne, et l'archive prend le min. Valable pour les deux objectifs
  car la preuve n'utilise que la réalisabilité et l'exactitude.
* Borne de portée (red-team) : `|θ_final − c^(0)| ≤ Σ_r Δ_r = Δ_0/(1−ρ)`. Avec `c=0`,
  `Δ_0 = π/6`, la boîte entière est couverte ssi `ρ ≥ 1/2`. C'est vérifié empiriquement :
  `ρ = 0,35` reste bloqué loin de l'optimum (§7). La méthode n'est pas équivalente à une grille
  globale fine et peut converger vers un optimum local.
* `maneuver_count_v1` : sans l'option NOOP dans la grille locale, un avion déplacé ne peut
  jamais revenir à `θ = 0` et le compte ne peut pas décroître. La variante « NOOP forcé »
  (`4n` qubits au plus, `3n` quand le centre est déjà 0) est obligatoire pour cet objectif.
* Deux warm-starts distincts : warm-start de la grille / des contrôles (centres `c^(0)`, ce que
  fait le PoC) et warm-start des angles ou de l'état QAOA (non traité, orthogonal).

## 5. Comparaison de ressources (mesures réelles, CP_5, `quadratic_control_cost_v1`)

| méthode | largeur max | largeur × rounds | termes quadratiques (total) | appels solveur | évaluations exactes | objectif |
|---|---|---|---|---|---|---|
| global K3 | 15 | 15 | 45 | 1 | 11 | 0,548311 |
| global K5 | 25 | 25 | 100 | 1 | — | 0,137078 |
| global K7 | 35 | 35 | 175 | 1 | — | 0,060923 |
| global K33 | 165 | 165 | 3290 | 1 | — | 0,002677 |
| global K65 | 325 | 325 | 12960 | 1 | — | 0,001874 |
| adaptatif ρ=0,5, 8 rounds | **15** | 116 | 186 | 8 | 177 | **0,001322** |
| adaptatif ρ=0,7, 12 rounds | 15 | 176 | 335 | 12 | 174 | 0,001510 |
| ancre continue Gurobi (q et θ) | — | — | — | — | — | 0,001137 |

Lecture : à largeur 15, l'adaptatif descend sous la grille globale K65 (325 qubits), avec un
total de termes quadratiques cumulés (186) comparable à un seul round K7 (175). Le prix est
`R = 8` appels séquentiels : le gain est en **largeur**, pas en temps total ni en shots
(`R × shots_par_round`). L'ancre continue reste plus basse car elle optimise aussi la vitesse.

## 6. Proposition retenue

La grille adaptative itérative K=3 (variante grille pure pour le coût quadratique, NOOP forcé
pour le compte de manœuvres), initialisée sur la trajectoire NOOP avec `Δ_0 = π/6` et
`ρ ≥ 1/2`, avec archive élitiste, politique déterministe de recentrage sur l'archive quand un
round est infaisable ou moins bon, arrêt sur stagnation, garde anti-cycle par état
`(centres, Δ)` et provenance complète par round.

Le binaire est écarté (le gain ne survit pas aux ancillas). Le domain-wall n'est pas retenu
comme remplacement : il économise `n` qubits mais densifie le conflit et fragilise la
pénalité ; il reste le candidat le plus sûr si la seule contrainte est un plafond dur de qubits
à K petit, et le red-team le classe devant l'adaptatif sur la robustesse des garanties. Les
candidats 4 et 6 sont exacts mais nuls sur CP ; 7 est composable avec 5 (`|D|·K'` qubits).

## 7. Preuve de concept locale

Module `src/acrpq/experimental/adaptive_grid.py` (grille par avion, table de conflit via le
noyau `geometry`, solveur exact branch-and-bound vérifié contre l'énumération officielle
`exhaustive_discrete_optimum` et contre un oracle produit, recuit simulé déterministe par
seed, archive, baselines classiques). Script `scripts/experimental_adaptive_grid_campaign.py`.
45 tests dans `tests/test_experimental_adaptive_grid.py` (invariants, équivalence round 0,
monotonie, cas adversariaux, contre-exemple de portée). Durée totale de la campagne : 93 s,
CPU local uniquement.

### 7.1 Objectif `quadratic_control_cost_v1`, initialisation NOOP, solveur exact

| instance | K3 (nK) | K5 | K7 | K33 | K65 | adaptatif ρ=0,5 (3n, 8 rounds) | ρ=0,35 (6 r.) | ρ=0,7 | ancre continue |
|---|---|---|---|---|---|---|---|---|---|
| CP_3 | 0,274156 (9) | 0,068539 | 0,030462 | 0,001071 | 0,000402 | **0,000402** (9) | 0,059118 | 0,000483 | 0,000312 |
| CP_4 | 0,411233 (12) | 0,102808 | 0,045693 | 0,002142 | 0,001339 | **0,000686** (12) | 0,088677 | 0,001029 | 0,000625 |
| CP_5 | 0,548311 (15) | 0,137078 | 0,060923 | 0,002677 | 0,001874 | **0,001322** (15) | 0,118236 | 0,001510 | 0,001137 |
| CP_6 | 0,685389 (18) | 0,171347 | 0,076154 | 0,003213 | n.c. | **0,002217** (18) | 0,147795 | 0,002343 | 0,001810 |
| CP_7 | 0,822467 (21) | 0,205617 | 0,091385 | n.c. | n.c. | **0,003121** (21) | 0,177354 | 0,006103 | 0,002374 |
| CP_8 | 0,959544 (24) | 0,239886 | 0,106616 | n.c. | n.c. | **0,005422** (24) | 0,206913 | 0,007002 | timeout |

Toutes les solutions sont réalisables (0 conflit sur le noyau géométrique commun). Nombre
d'avions déviés : `n−1` pour K3/K5/K7, `n` pour les grilles fines et l'adaptatif. Les
références K33/K65 ne sont pas calculables en temps raisonnable pour n ≥ 7 (n.c.).

Cas d'échec : `ρ = 0,35` est un échec systématique, prédit par la borne de portée
(`ρΔ_0/(1−ρ) = 0,28 rad < 0,5 rad` nécessaires pour revenir de `±π/6` vers ~0,02 rad) ;
`ρ = 0,7` s'arrête sur stagnation avant d'avoir convergé sur CP_7 et CP_8 (patience 3).
Aucun round infaisable n'a été rencontré avec `Δ_0 = π/6`.

### 7.2 Recuit simulé (3 seeds par politique)

Les trois seeds reproduisent la valeur exacte sur CP_3 à CP_6. Sur CP_7 (ρ=0,5) deux seeds
sur trois s'arrêtent à 0,003388 au lieu de 0,003121, sur CP_8 deux sur trois à 0,005488 au
lieu de 0,005422. L'archive est monotone sur toutes les exécutions ; la propriété « chaque
round est meilleur que le précédent » n'est pas revendiquée pour le recuit.

### 7.3 Initialisation sur l'ancre continue et baselines classiques (exigées par le red-team)

| instance | adaptatif NOOP | adaptatif centre continu, Δ_0=π/6 | adaptatif centre continu, Δ_0=0,05 | snap K7 | snap+réparation K7 | snap+réparation K33 | snap+réparation K65 |
|---|---|---|---|---|---|---|---|
| CP_3 | 0,000402 | 0,000435 | 0,000314 | infaisable | 0,030462 | 0,001071 | 0,000402 |
| CP_4 | 0,000686 | 0,000733 | 0,000639 | infaisable | 0,045693 | 0,002142 | 0,001740 |
| CP_5 | 0,001322 | 0,001277 | 0,001179 | infaisable | 0,060923 | 0,002677 | 0,002276 |
| CP_6 | 0,002217 | 0,002157 | 0,001897 | infaisable | 0,076154 | 0,006426 | n.c. |
| CP_7 | 0,003121 | 0,003224 | 0,002953 | infaisable | 0,091385 | n.c. | n.c. |

Le snap pur de la solution continue est infaisable sur toute grille K ≤ 7 (et souvent K33/K65) :
le conflit est un phénomène de seuil, pas une fonction lisse. La réparation gloutonne
classique donne des valeurs 1,5 à 3 fois plus élevées que l'adaptatif à centre continu et
Δ_0 = 0,05. Le pas de résolution locale contribue donc réellement au-delà du snapping, mais ce
pas est ici un solveur **classique exact** : rien de quantique n'a été mesuré.

### 7.4 Objectif `maneuver_count_v1`

Toutes les méthodes (K3, K5, K7, K33, K65, adaptatif avec NOOP forcé, ancre Gurobi entière)
donnent exactement `n − 1` sur CP_3 à CP_8. L'adaptatif s'arrête après 4 rounds sur stagnation.
Aucun gain : l'objectif ne dépend pas de la précision angulaire et la structure CP impose que
tous les avions sauf un manœuvrent. Le résultat est cohérent, pas amélioré.

## 8. Garanties

1. Round 0 avec centres NOOP et `Δ_0 = π/6` est identique à la grille K3 officielle (test).
2. La solution précédente est toujours dans la grille suivante (lemme du centre, test, journal).
3. Monotonie des solutions de round pour un solveur exact, pour les deux objectifs (preuve
   §4.4, vérifiée sur toutes les exécutions exactes).
4. Monotonie de l'archive élitiste pour tout solveur (min cumulatif ; vérifiée pour le recuit).
5. Déterminisme : seeds explicites, départage lexicographique sans epsilon, états
   `(centres, Δ)` uniques (garde anti-cycle), provenance complète par round.
6. Conservation de la faisabilité : seule une solution à 0 conflit (rescorée par le noyau
   géométrique) entre dans l'archive ; un round infaisable ne dégrade jamais l'archive.
7. Aucune ancilla, degré 2, mêmes pénalités qu'en one-hot par round.

## 9. Limites

* Non exhaustif sur la grille fine implicite ; raffinement local glouton ; optimum local
  possible. Portée bornée `Δ_0/(1−ρ)` : avec un centre continu et `Δ_0` petit, une manœuvre
  franche nécessaire pour lever un conflit est inatteignable (test `bounded_reach`).
* Le centre continu introduit une forte assistance classique ; les deux initialisations sont
  rapportées séparément.
* Gain en largeur de registre uniquement : `R` appels QPU séquentiels, shots totaux
  `R × shots`, latence classique-quantique par round non modélisée.
* Non comparable à une grille globale K5/K7 sans protocole explicite : ce document publie à la
  fois la largeur max, largeur × rounds et le total de termes quadratiques (§5).
* `maneuver_count_v1` requiert le NOOP forcé (jusqu'à `4n` qubits) et n'a montré aucun gain.
* `q` fixé à 1 : la comparaison à l'ancre continue (qui optimise `q`) n'est pas à modèle égal.
* Le PoC résout chaque round exactement ou par recuit classique ; aucun QUBO hétérogène par
  avion n'a été construit, aucun circuit QAOA, aucun échantillonnage bruité.
* Références fines K33/K65 indisponibles pour n ≥ 7.

## 10. Implications expérimentales

Pour les instances CP et le coût quadratique, la discrétisation grossière K3/K5/K7 explique
l'essentiel de l'écart entre les résultats discrets officiels et l'optimum continu (facteur
50 à 500). Un protocole à largeur constante `3n` réduit cet écart à un facteur 1,1 à 2,3 sans
augmenter le registre. Le benchmark officiel n'est pas modifié ; si le protocole était
intégré un jour, il devrait être présenté comme un solveur hybride multi-rounds, avec les trois
métriques de ressources et la baseline snap + réparation, jamais comme « K fin à 3n qubits ».

## 11. Perspectives QPU (non exécutées)

Étapes nécessaires avant toute étude sur QPU réel, aucune n'est faite :

1. construire un QUBO à blocs hétérogènes `K_i` (offsets cumulés) avec les pénalités
   officielles, prouver la dominance one-hot pour ce QUBO et l'aligner sur
   `verify_qubo_wellformed` ;
2. définir le décodage et la réparation d'un round sous échantillonnage (repli sur le centre,
   toujours présent, plutôt que sur NOOP) et le protocole d'archive côté classique ;
3. fixer le protocole de comparaison (largeur égale, shots totaux égaux, portes 2-qubits
   totales égales) et mesurer sur simulateur avant tout matériel ;
4. traiter la latence des `R` allers-retours et la compilation de `R` circuits.

## 12. Affirmations autorisées et interdites

Autorisées :

* « La grille adaptative K=3 atteint, à largeur `3n`, une valeur inférieure à celle d'une grille
  globale uniforme K65 (`65n` qubits) sur CP_3 à CP_5, avec un solveur de round exact, en
  8 rounds. »
* « Le round 0 est exactement la grille K3 officielle ; la solution précédente est toujours
  dans la grille suivante ; avec un solveur exact la valeur ne peut pas empirer ; avec un
  solveur heuristique seule l'archive est monotone. »
* « Le gain est en largeur de registre par appel, au prix de `R` appels. »
* « Le binaire réduit les qubits logiques mais, après quadratisation du conflit, consomme plus
  de qubits que le one-hot pour K ∈ {3..9}. »
* « Le domain-wall économise exactement `n` qubits et double la fraction valide, mais densifie
  le terme de conflit (×1,3 à ×2,5 mesuré) et requiert `λ_dw > λ_pen·r_max·maxdeg`. »
* « La dominance exacte et la décomposition en composantes ne retirent rien sur CP. »

Interdites :

* « Équivalent à une grille globale fine » ou « précision K fin à 3n qubits ».
* « Monotone » sans préciser solveur exact ou archive élitiste.
* « Compatible avec `maneuver_count_v1` » sans le NOOP forcé, ou « améliore
  `maneuver_count_v1` » (aucun gain constaté).
* Tout gain en temps total, en shots ou en portes cumulées.
* Toute mention d'un résultat quantique : aucun circuit, aucun QPU, aucun échantillonnage n'a
  été exécuté dans cette étude.
* Comparer l'adaptatif à `q = 1` avec l'ancre continue comme s'il s'agissait du même modèle.
* Citer `results/experimental_adaptive_grid_v1/` comme benchmark officiel.

---

## Verdict final

`LOCAL_PROOF_OF_CONCEPT_VALIDATED` pour `quadratic_control_cost_v1` ; aucun gain pour
`maneuver_count_v1`. Le binaire est réfuté, le domain-wall n'est pas retenu comme remplacement,
les candidats structurels sont exacts mais nuls sur CP. `READY_FOR_FUTURE_QPU_STUDY` n'est pas
sélectionné : décodage sous échantillonnage, QUBO hétérogène et coût réel en shots ne sont pas
validés. Rien n'a été intégré au benchmark officiel, à l'UI ou à IBM, rien n'a été poussé ; ce
document est livré pour contre-review.

---

## 13. Étape 3 — raffinement adaptatif hybride à largeur bornée : validation locale (4 septembre 2026)

Cette section est rédigée pour être reprise dans le mémoire. Elle complète le verdict
`LOCAL_PROOF_OF_CONCEPT_VALIDATED` par les composants nécessaires à une exécution matérielle,
validés localement mais **non exécutés sur QPU** (porte `docs/ADAPTIVE_QPU_GO_NO_GO.md`,
verdict `LOCAL_ONLY`).

### 13.1 Position méthodologique

Le pipeline comporte trois étapes : (1) le modèle continu historique (AMPL, q et θ continus) ;
(2) la discrétisation globale one-hot validée (K niveaux communs, `n·K` qubits, équivalence
prouvée entre énumération, MILP et QUBO) ; (3) un raffinement adaptatif hybride : chaque round
résout, sur une grille locale de trois caps par avion centrée sur la solution précédente, le
même type de QUBO one-hot, puis recentre et divise le pas par deux. La largeur reste `3n`, la
résolution angulaire implicite décroît géométriquement. L'étape 3 est un solveur hybride
séquentiel, pas une grille fine globale.

### 13.2 QUBO à blocs hétérogènes

Après clipping et déduplication, les blocs peuvent avoir des tailles différentes
(`K_i ∈ {1,2,3}`). Les variables sont indexées par offsets cumulés ; le coût est calculé sur
la valeur physique du cap ; les conflits sont recalculés pour chaque paire d'options
physiques ; les pénalités suivent la règle officielle avec `cost_max` pris sur l'union des
options du round. Pour la grille K3 uniforme du round 0, les coefficients sont identiques à
ceux du QUBO officiel, terme par terme, et le circuit ISA compilé est identique à celui de la
campagne matérielle (profondeur 211, 108 portes à deux qubits sur CP_3). Le théorème de
dominance one-hot s'étend aux blocs hétérogènes. Les énergies ne sont pas comparables entre
rounds ; seul l'objectif physique l'est.

### 13.3 Décodage et endianness

Le décodeur normalise les clés Qiskit (petit-boutiste), valide strictement les counts,
détecte les violations one-hot par bloc, répare de façon documentée (bloc vide → centre,
toujours présent ; plusieurs bits → coût minimal), rescoure par le noyau géométrique commun
et retient, comme meilleur candidat, l'objectif physique minimal parmi les échantillons
valides et sans conflit. La bijection encodage/décodage est vérifiée sur tous les états
one-hot de blocs [3,3,3], [2,3,3], [1,2,3] et sur CP_4. Un décodeur indépendant, écrit
séparément, reproduit ces résultats et la vérité terrain d'endianness d'AerSimulator.

### 13.4 Règle d'angles

À angles fixes (γ=0,4, β=0,3), l'angle RZZ du terme one-hot (≈ 17 rad au round 0) se replie
modulo 2π dans un régime différent dès que les pénalités changent avec le pas : en simulation
sans bruit, le round 2 ne produit plus aucun échantillon one-hot valide. La règle
préenregistrée `γ_r = γ₀·λ_pen(0)/λ_pen(r)` maintient les phases de pénalité invariantes,
coïncide avec les angles historiques au round 0 et rétablit 37 à 53 % d'échantillons
faisables aux rounds 1 à 3 sur CP_3 (11 à 16 % sur CP_4), chaque round retrouvant l'optimum
exact de sa grille. La phase liée au coût pur dérive de 15 à 22 % : cette règle est une
assistance classique dosée et doit être nommée ainsi.

### 13.5 Chaîne pilotée par les résultats (simulateur)

Sur Aer, 512 tirs par round, quatre rounds : CP_3 atteint 0,00428 (contre 0,274 pour K3
global, 0,0305 pour K7 global) à largeur 9 ; CP_4 atteint 0,00643 (contre 0,411 et 0,0457) à
largeur 12. Le coût réel est de 4 circuits, 2048 tirs, 508 (CP_3) et 646 (CP_4) de profondeur
ISA cumulée, 238 et 414 portes à deux qubits cumulées, et 4 décisions séquentielles. Sur
matériel, ce coût inclurait 4 jobs séquentiels dont la latence de file médiane observée est de
7,68 h chacun.

### 13.6 Identité et provenance

Chaque chaîne porte une préinscription hachée (`chain_id`) et chaque round un hash chaîné au
parent et au candidat sélectionné. La vérification profonde recalcule la grille, le QUBO et le
décodage à partir de l'instance et refuse toute divergence, toute sélection qui ne serait pas
le meilleur échantillon faisable, tout drapeau `official_benchmark` ou `real_qpu` indu. Le
driver (`exact_local`, `sa`, `aer_sim`, `qpu_result`) est inscrit dans l'identité ; une
trajectoire classique ne peut pas être présentée comme pilotée par le QPU. Le red-team a
identifié un BLOCKER (absence de recalcul physique) et deux CRITICAL, corrigés et retestés le
même jour ; la chaîne ne protège pas contre un auteur qui ré-exécuterait tout et republierait,
seule une publication préalable du `chain_id` l'ancrerait.

### 13.7 Références et comparabilité

Les grilles globales K3/K7/K33/K65 sont certifiées exactes sur leur grille (certificat par
ligne) ; la référence continue θ-seul q=1 est un incumbent provisoire ; la référence continue
q+θ appartient à un autre domaine de contrôle et n'est comparée qu'avec qualification. Sur
CP_3 à CP_6, l'adaptatif exact à ρ=0,5 (init NOOP) termine à 8e-5 à 3,3e-4 de l'incumbent q=1
et sous la grille K65 ; initialisé sur cet incumbent, il le retrouve exactement, ce qui
mesure une assistance classique totale.

### 13.8 Affirmations autorisées et interdites (complément)

Autorisées : « composants matériels validés localement : formulation hétérogène, décodage,
compilation ISA reproductible sur la cible réelle en lecture seule, identité hachée » ;
« chaîne pilotée par un échantillonneur validée sur simulateur » ; « la règle d'angles
normalisée par la pénalité est nécessaire en simulation pour que la chaîne survive au
round 2 ».

Interdites : « pilote QPU exécuté » (aucun job) ; « prêt pour le QPU » (verdict `LOCAL_ONLY`,
conditions 6, 8, 10, 12, 13) ; « angles physiquement cohérents entre rounds » sans la réserve
sur la dérive du terme de coût ; toute durée de file ou de calcul QPU (inconnues).

## 14. Renvoi — scalabilité structurelle par instance

La question « la largeur 3n de l'adaptatif peut-elle s'appliquer à davantage d'avions ? » est
traitée séparément dans `docs/INSTANCE_SCALABILITY_STUDY.md` (blocs hétérogènes exacts,
réduction aux avions actifs, décomposition en composantes, matrice d'applicabilité sur les
1042 instances, réponse CP30). Les deux bénéfices, précision à largeur constante et largeur
permise par la structure, restent distincts.
