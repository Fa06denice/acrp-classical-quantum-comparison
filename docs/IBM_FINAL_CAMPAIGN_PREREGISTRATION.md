# Préenregistrement — campagne IBM finale

* `protocol_id` : `acrpq-ibm-final-campaign/2`
* **`protocol_sha256` : `sha256:29903c28fa2802457df9519bc62ff2b2b926b8293316d5e8a688a149490ab468`**
* `content_sha256` du présent préenregistrement : `sha256:4dce545d0814b7988606011236c56b0cad1d6f9eeb36832e4a1abeaf9ad91aa3`
* backend : **ibm_marrakesh exclusivement** · un circuit par job, pas de regroupement multi-pubs
* angles gelés depuis `results/offline_angle_landscape_v2/angle_landscape.json` (`sha256:0599b281f159849777e219b699ae5fb0a7325963008daa7022165ca7a6351e60`)

**Ce document est écrit avant tout nouveau résultat matériel.** Aucune sélection d'angles après observation du matériel n'est permise.

## 0. Supersession de protocole

Ce protocole **supersède** `sha256:a3d207e26a9aa33bf2858d33710e4ad575e7ea2ba97c3fa9259b3e1e89a75c66`, sous lequel les jobs suivants ont déjà été soumis et exécutés : `FP_4__K3__quad__CE_endianness_canary__r1__e001`, `CP_3__K3__quad__A1_P1_feas_angles__r1__e002`, `CP_3__K3__man__A1_P1_feas_angles__r1__e003`.

**Nature du défaut : `reference_scoring_metadata_only`.** Le paysage hors ligne calculait la référence certifiée avec geometry.objective(), qui renvoie toujours le coût quadratique de l'instance, quel que soit l'objective_id de la cellule. Les références, les nombres d'optima, les masses sur l'optimum certifié et les écarts des cellules maneuver_count_v1 étaient donc évalués dans les mauvaises unités. À K=3 la proportionnalité exacte (facteur 0,137078) rendait l'argmin identique, donc seule la VALEUR de référence était fausse ; à K=5 les objectifs ne sont pas proportionnels et l'ensemble des optima lui-même était faux (CP_3/K5/maneuver : 6 optima au lieu de 36).

**Ce qui n'était PAS affecté.** Aucun circuit soumis n'est affecté : les angles sont optimisés sur la masse faisable, qui ne dépend pas de l'objective_id, et les coefficients du QUBO viennent de build_qubo, pas du scoring des références. Les angles, les hashes de QUBO, les masses faisables et les masses one-hot sont vérifiés identiques bit à bit avant/après (voir before_after_verification).

Circuits affectés : False · angles affectés : False · QUBO affecté : False · shots affectés : False · résultats bruts toujours valides : True · artefacts bruts jamais réécrits : True.

Vérification avant/après publiée dans `results/offline_angle_landscape_v2/correction_verification.json` : 336 invariants comparés bit à bit (angles, hashes de QUBO, masses faisables, masses one-hot) — 0 violation. Seuls 4 champs de scoring des références ont changé, comme attendu, sur les 14 cellules.

## 1. Hypothèses préenregistrées

* **H1** — des angles optimisés hors ligne augmentent le taux de shots faisables par rapport aux angles fixes, surtout sur les petites instances.
* **H2** — la faisabilité décroît avec le nombre de qubits, la profondeur ISA et le nombre de portes 2Q.
* **H3** — la perte se décompose en invalidité one-hot, one-hot valide mais conflit géométrique, et faisable mais sous-optimal.
* **H4** — les deux `objective_id` peuvent présenter des distributions différentes ; à K=3 ils sont **deux paramétrisations énergétiques du même problème décisionnel** (proportionnalité globale vérifiée, facteur 0,137078), à K=5 ils ne le sont plus.
* **H5** — parmi les runs contenant au moins un échantillon faisable, le meilleur atteint souvent l'optimum discret certifié.

## 2. Métrique principale et secondaires

**feasible_shot_rate_strict** — shots dont la chaîne est EXACTEMENT one-hot par avion ET géométriquement sans conflit, divisés par les shots reçus. Aucune réparation one-hot n'est appliquée.

Vérification : la définition stricte reproduit exactement le feasibility_rate enregistré des 30 jobs P0.

Secondaires : `raw_onehot_rate` · `repaired_feasible_rate` · `conditional_geometry_feasibility` · `run_success` · `certified_optimum_hit` · `certified_optimum_mass` · `probability_mass_on_best_feasible` · `best_feasible_objective` · `gap_to_certified_reference (jamais sur une solution infaisable)` · `number_unique_bitstrings` · `shannon_entropy_of_counts` · `isa_depth` · `isa_two_qubit_gates` · `physical_qubits` · `provider_execution_time (si réellement disponible)` · `queue_and_wall_time (séparés du temps QPU)` · `min_separation_and_conflicts_of_best_sample` · `endianness_log_likelihood_ratio (canari FP_4 uniquement)`.

Aucun écart à la référence n'est calculé pour une solution infaisable.

## 3. Bras — chacun ne change qu'UNE variable

| bras | angles | niveau de transpilation | rôle |
|---|---|---|---|
| `A0_P0_replication` | P0 | 0 | réplication fraîche des angles historiques : supprime la confusion entre effet des angles et dérive de calibration entre les deux campagnes |
| `A1_P1_feas_angles` | P1_feas | 0 | isole l'effet des angles (seule variable qui change vs A0) |
| `A2_P0_transpile_L2` | P0 | 2 | isole l'effet de la transpilation (seule variable qui change vs A0) |
| `CE_endianness_canary` | P0 | 2 | canari d'endianness sur une instance NON symétrique : la famille CP a une symétrie miroir exacte sous laquelle la distribution p=1 est invariante (KL = 0), donc aucune cellule CP ne peut prouver la convention |

`P1_feas` **n'est pas « QAOA standard »**. P1_feas = 'QAOA p=1 avec angles sélectionnés hors ligne pour maximiser la masse faisable, puis échantillonnage matériel'. Ne JAMAIS l'appeler 'QAOA standard' ni 'QAOA optimisé' sans préciser hors ligne.

### Conclusion bornée sur `P1_minH`

Conclusion BORNÉE : sur les 14 cellules effectivement étudiées (CP_3..CP_7 à K=3, CP_3 et CP_4 à K=5), pour CE QUBO one-hot fortement pénalisé et CE protocole QAOA p=1 à une seule couche, les angles minimisant ⟨H⟩ produisent une masse faisable inférieure à celle des angles fixes de P0 sur les cellules maneuver_count_v1 (2·10⁻⁵ contre 2,07·10⁻² sur CP_3). Aucune généralisation à QAOA, à d'autres encodages, à p > 1, à d'autres familles d'instances ou à d'autres pondérations de pénalité n'est autorisée. La formulation « structurellement pire », employée dans le message du commit e965d6e, est SUPERSÉDÉE par le présent énoncé borné.

### Ordre d'exécution : blocs appariés

Une réplication a0 non intercalée ne supprime pas la dérive temporelle de calibration ; seuls des blocs appariés le font. Rotation en carré latin : oui. Relecture du crédit après chaque bloc : oui.

Capturé pour chaque job : `submission_timestamp` · `queue_time` · `execution_timestamp` · `backend_calibration_snapshot_or_timestamp` · `runtime_execution_options`.

L'ordre complet des 63 jobs de la campagne recommandée est publié en §12, **avant le premier job**.

### Canari d'endianness — test formalisé

Cellule : **FP_4/K3**, quadratic_control_cost_v1, γ=0.4, β=0.3, niveau 2, 512 shots, 1 job.

*Pourquoi cette cellule.* La famille CP possède une symétrie miroir exacte (avion i ↔ n+1−i, option o ↔ K−1−o) sous laquelle la distribution p=1 est INVARIANTE : KL(p ‖ p_inversée) = 0 exactement, vérifié numériquement. Aucune cellule CP ne peut donc départager les deux conventions, quelle que soit la métrique employée. FP_4/K3 brise cette symétrie.

Distributions attendues sous les deux conventions :

* **A (retenue)** — la chaîne brute du fournisseur est INVERSÉE pour obtenir l'ordre de bits du QUBO : p_A(z_brut) = p_idéale[index(inverse(z_brut))]
* **B (alternative)** — aucune inversion : p_B(z_brut) = p_idéale[index(z_brut)]
* `p_B` est une permutation de `p_A` ; empreinte de `p_A` : `sha256:6f17edb9aabb721cb942c9d6dbb1a45c` (sha256 des 32 premiers caractères hexadécimaux du vecteur de probabilité float64 en ordre de bits QUBO, angles (0.4, 0.3) ; recalculable en ~6 s, non stocké (16 777 216 entrées))

**Formule du LLR.** LLR = Σ_{shots} [ log p̃_A(z) − log p̃_B(z) ], avec la régularisation FIGÉE p̃ = (1−ε)·p + ε/2^n et ε = 0.01, qui borne les logarithmes et empêche qu'un seul shot de probabilité idéale nulle domine la somme.

**Seuil figé.** CONCLUANT pour une convention si LLR dépasse en valeur absolue max(150 nats, 3·σ̂), où σ̂ = sqrt(512 · variance empirique du LLR par shot) est estimé sur les shots RÉELLEMENT reçus. AMBIGU sinon.

| résultat | décision |
|---|---|
| `LLR >= seuil` | concluant pour la convention A (celle du dépôt) — poursuivre |
| `LLR <= -seuil` | concluant pour la convention B — ARRÊT TOTAL, et le décodage des 30 jobs P0 doit être réexaminé |
| `|LLR| < seuil` | AMBIGU — ARRÊT OBLIGATOIRE, aucune soumission supplémentaire |

**Le KL idéal n'est pas une garantie matérielle.** Le KL idéal de 1,84 nats/shot dimensionne le test, il ne garantit RIEN sur le matériel. Sous un mélange dépolarisant de paramètre λ, l'espérance vaut (1−λ)·878 nats à 512 shots tandis que l'écart-type reste ~47 nats : E[LLR] = 176 à λ=0,8 (3,6 σ, concluant), 88 à λ=0,9 (1,8 σ), 44 à λ=0,95 (0,9 σ, ambigu). Le seuil de 150 nats exige donc que l'appareil conserve environ 17 % de poids cohérent. Sous dépolarisation totale l'espérance est exactement nulle par symétrie de permutation : le bruit ATTÉNUE le LLR vers zéro sans jamais pouvoir en inverser le signe. Si l'atténuation dépasse le modèle, le test renvoie AMBIGU et la campagne s'arrête — il ne renvoie pas une conclusion fausse.

## 4. Paysage hors ligne exact et angles gelés

Masses de probabilité exactes (vecteur d'état, sans échantillonnage ni bruit). `opt` = masse sur l'optimum certifié.

| cellule | qubits | jeu | γ | β | one-hot | faisable | opt | ⟨H⟩ |
|---|---|---|---|---|---|---|---|---|
| CP_3/K3/quadratic | 9 | `P0` | 0.4000 | 0.3000 | 0.39028 | 0.10959 | 0.08309 | 140.88 |
| CP_3/K3/quadratic | 9 | `P1_minH` | 0.4338 | 0.3905 | 0.37347 | 0.13226 | 0.09826 | 54.51 |
| CP_3/K3/quadratic | 9 | `P1_feas` | 2.3159 | 1.1427 | 0.28236 | 0.18895 | 0.14731 | 85.51 |
| CP_3/K3/maneuver | 9 | `P0` | 0.4000 | 0.3000 | 0.06039 | 0.02066 | 0.00436 | 456.85 |
| CP_3/K3/maneuver | 9 | `P1_minH` | 2.4882 | 0.7703 | 0.42188 | 0.00002 | 0.00002 | 231.80 |
| CP_3/K3/maneuver | 9 | `P1_feas` | 2.2258 | 1.0636 | 0.14090 | 0.13646 | 0.00255 | 343.53 |
| CP_4/K3/quadratic | 12 | `P0` | 0.4000 | 0.3000 | 0.14698 | 0.02252 | 0.01790 | 137.35 |
| CP_4/K3/quadratic | 12 | `P1_minH` | 0.3981 | 0.3938 | 0.26711 | 0.04334 | 0.03450 | 109.61 |
| CP_4/K3/quadratic | 12 | `P1_feas` | 2.4223 | 0.3701 | 0.40816 | 0.06918 | 0.05363 | 153.11 |
| CP_4/K3/maneuver | 12 | `P0` | 0.4000 | 0.3000 | 0.02489 | 0.00909 | 0.00705 | 1214.99 |
| CP_4/K3/maneuver | 12 | `P1_minH` | 2.7492 | 0.7884 | 0.43722 | 0.00001 | 0.00001 | 505.02 |
| CP_4/K3/maneuver | 12 | `P1_feas` | 2.3551 | 0.5498 | 0.11533 | 0.09549 | 0.00398 | 864.93 |
| CP_5/K3/quadratic | 15 | `P0` | 0.4000 | 0.3000 | 0.00021 | 0.00004 | 0.00004 | 870.78 |
| CP_5/K3/quadratic | 15 | `P1_minH` | 1.1125 | 0.3920 | 0.19974 | 0.01519 | 0.01244 | 193.15 |
| CP_5/K3/quadratic | 15 | `P1_feas` | 1.3753 | 0.7012 | 0.05816 | 0.02783 | 0.00505 | 499.54 |
| CP_5/K3/maneuver | 15 | `P0` | 0.4000 | 0.3000 | 0.01427 | 0.00075 | 0.00066 | 1594.86 |
| CP_5/K3/maneuver | 15 | `P1_minH` | 0.3230 | 1.1571 | 0.16666 | 0.00051 | 0.00046 | 859.99 |
| CP_5/K3/maneuver | 15 | `P1_feas` | 0.8497 | 1.0402 | 0.10373 | 0.06966 | 0.04336 | 882.58 |
| CP_6/K3/quadratic | 18 | `P0` | 0.4000 | 0.3000 | 0.00097 | 0.00030 | 0.00003 | 1270.43 |
| CP_6/K3/quadratic | 18 | `P1_minH` | 1.3744 | 0.3906 | 0.14671 | 0.00493 | 0.00415 | 303.00 |
| CP_6/K3/quadratic | 18 | `P1_feas` | 1.6358 | 0.6798 | 0.02166 | 0.01235 | 0.00212 | 739.67 |
| CP_6/K3/maneuver | 18 | `P0` | 0.4000 | 0.3000 | 0.00676 | 0.00007 | 0.00002 | 3278.50 |
| CP_6/K3/maneuver | 18 | `P1_minH` | 0.4551 | 1.1644 | 0.12039 | 0.00012 | 0.00011 | 1417.16 |
| CP_6/K3/maneuver | 18 | `P1_feas` | 2.9465 | 0.5525 | 0.03679 | 0.03526 | 0.00258 | 3255.28 |
| CP_7/K3/quadratic | 21 | `P0` | 0.4000 | 0.3000 | 0.00328 | 0.00020 | 0.00016 | 1225.33 |
| CP_7/K3/quadratic | 21 | `P1_minH` | 1.5995 | 0.3882 | 0.10735 | 0.00155 | 0.00133 | 441.96 |
| CP_7/K3/quadratic | 21 | `P1_feas` | 2.8153 | 0.6607 | 0.01996 | 0.01602 | 0.00280 | 1058.62 |
| CP_7/K3/maneuver | 21 | `P0` | 0.4000 | 0.3000 | 0.00058 | 0.00005 | 0.00005 | 7332.96 |
| CP_7/K3/maneuver | 21 | `P1_minH` | 0.4702 | 0.3782 | 0.10097 | 0.00178 | 0.00148 | 1964.52 |
| CP_7/K3/maneuver | 21 | `P1_feas` | 2.4470 | 1.0334 | 0.03026 | 0.02032 | 0.01104 | 3581.59 |
| CP_3/K5/quadratic | 15 | `P0` | 0.4000 | 0.3000 | 0.07433 | 0.03819 | 0.00343 | 636.26 |
| CP_3/K5/quadratic | 15 | `P1_minH` | 2.3012 | 0.6490 | 0.08004 | 0.01375 | 0.00291 | 99.45 |
| CP_3/K5/quadratic | 15 | `P1_feas` | 1.7714 | 0.4388 | 0.21743 | 0.12173 | 0.00922 | 135.47 |
| CP_3/K5/maneuver | 15 | `P0` | 0.4000 | 0.3000 | 0.01053 | 0.00447 | 0.00173 | 1412.07 |
| CP_3/K5/maneuver | 15 | `P1_minH` | 0.6544 | 0.7607 | 0.04394 | 0.00140 | 0.00136 | 289.10 |
| CP_3/K5/maneuver | 15 | `P1_feas` | 1.4400 | 0.9692 | 0.05606 | 0.04713 | 0.00153 | 439.60 |
| CP_4/K5/quadratic | 20 | `P0` | 0.4000 | 0.3000 | 0.06384 | 0.01933 | 0.00085 | 295.52 |
| CP_4/K5/quadratic | 20 | `P1_minH` | 0.4014 | 0.4685 | 0.10208 | 0.03186 | 0.00136 | 164.89 |
| CP_4/K5/quadratic | 20 | `P1_feas` | 2.4891 | 1.0161 | 0.05876 | 0.04455 | 0.00088 | 209.59 |
| CP_4/K5/maneuver | 20 | `P0` | 0.4000 | 0.3000 | 0.00255 | 0.00106 | 0.00073 | 3589.98 |
| CP_4/K5/maneuver | 20 | `P1_minH` | 0.3927 | 0.6562 | 0.02866 | 0.00091 | 0.00055 | 610.20 |
| CP_4/K5/maneuver | 20 | `P1_feas` | 0.1237 | 0.4390 | 0.13473 | 0.04289 | 0.02707 | 823.59 |

## 5. Métriques ISA après transpilation vers Marrakech

| cellule | niveau | qubits | profondeur avant | profondeur après | portes 2Q | taille | swap |
|---|---|---|---|---|---|---|---|
| CP_3/K3 | 0 | 9 | 10 | 211 | 108 | 576 | 0 |
| CP_3/K3 | 2 | 9 | 10 | 120 | 73 | 368 | 0 |
| CP_3/K5 | 0 | 15 | 14 | 394 | 357 | 1626 | 0 |
| CP_4/K3 | 0 | 12 | 12 | 259 | 207 | 1011 | 0 |
| CP_4/K3 | 2 | 12 | 12 | 183 | 135 | 639 | 0 |
| CP_4/K5 | 0 | 20 | 16 | 604 | 659 | 2807 | 0 |
| CP_5/K3 | 0 | 15 | 14 | 376 | 348 | 1599 | 0 |
| CP_5/K3 | 2 | 15 | 14 | 197 | 202 | 952 | 0 |
| FP_4/K3 | 2 | 24 | 10 | 102 | 133 | 763 | 0 |

## 6. Budget — métrique réellement facturée

* métrique : **QPU seconds (usage_consumed_seconds, IBM open plan)**
* coût **mesuré** par job : **2.0 s** (30/30 jobs P0 facturés exactement 2.000 s, indépendamment de la profondeur ISA (211 à 640))
* plafond prudent retenu : **4.0 s/job** (×2 de marge)
* limite du plan : 600 s / 28 jours · consommé 70 s · **restant 530 s**
* réconciliation : Comptabilité bouclée exactement : le fournisseur porte 35 jobs sur la fenêtre — les 30 de la cohorte P0 (2026-09-03) plus 5 jobs antérieurs du 2026-08-18 absents du manifeste P0. 35 x 2,000 s = 70 s = usage_consumed_seconds. Aucun écart résiduel, et le tarif de 2,000 s/job est donc confirmé sur 35 jobs, pas seulement sur 30.
* signalement de provenance : Ces 5 jobs matériels ibm_marrakesh du 2026-08-18 ne sont enregistrés dans aucun artefact du dépôt. À inventorier ; ne pas les intégrer aux agrégats scientifiques sans provenance complète.
* règle : ne jamais engager plus de **50%** du restant au tarif plafond

| campagne | jobs | pubs/job | shots totaux | coût mesuré | coût plafond | % du restant | dans la marge |
|---|---|---|---|---|---|---|---|
| **minimale** | 27 | 1 | 13824 | 54 s | 108 s | 20% | oui |
| **recommandee** | 63 | 1 | 32256 | 126 s | 252 s | 48% | oui |
| **etendue** | 75 | 1 | 38400 | 150 s | 300 s | 57% | **NON** |

## 7. Répétitions, shots, exclusions

Shots par job : **512**, identiques à P0 pour rester comparable. Répétitions : 3 (minimale et recommandée), 5 (étendue), `repetition_id` distinct, hash de configuration scientifique partagé.

Exclusions préenregistrées :

* state != completed ou result indisponible
* shots reçus != shots demandés
* summary/best_feasible manquant ou corrompu
* config_hash ou artefact_sha256 divergent entre préparation et exécution
* au plus 1 job de remplacement par slot exclu, journalisé avec code de raison, décidé AVANT tout calcul de la métrique principale

## 8. Analyses préenregistrées

* unité expérimentale : le job (les 512 shots d'un job sont groupés, non indépendants)
* comparaison principale : A1 vs A0 apparié par cellule (instance x objectif)
* mesure d'effet : différence de proportions avec intervalle hybride Newcombe/Wilson ; le ratio n'est rapporté qu'en secondaire et jamais quand P0 = 0
* zéros : aucun pseudo-compte ad hoc ; différence de proportions et borne exacte Clopper-Pearson pour les cellules à 0 succès
* mise en garde sur la variance : à 3 répétitions (df=2) l'intervalle de confiance à 95 % sur l'écart-type lui-même couvre [0,52x ; 6,29x] : la variance n'est qu'un ordre de grandeur, jamais une valeur calibrée
* interdits : test paramétrique traitant les shots comme indépendants · inférence forte à n faible · affirmation de causalité (employer « association observée ») · agrégation entre K, entre objectifs, entre bras, entre matériel et simulation

## 9. Ordre d'arrêt

1. canari d'endianness FP_4 : le signe du LLR doit désigner la convention retenue avec |LLR| > 3 sigma ; sinon ARRÊT et aucune autre soumission

2. deux canaris CP_3 : validation technique complète (job_id, 512 shots, décodage, hashes) ; un canari à 0 % faisable n'est PAS un échec technique

3. relire usage_remaining_seconds ; si le reste passe sous la marge, ARRÊT

4. bras A0 puis A1 puis A2 puis les cellules K5, séquentiellement

5. ARRÊT immédiat sur toute violation de protocole, backend inattendu, shots incohérents, hash divergent, soumission sans job_id, coût anormal

## 10. Réserves ouvertes de l'expert E, à traiter par la campagne

* brancher reference_objective dans le décodage brut (gap null sur les 30 P0)
* capturer le snapshot de calibration, les options d'exécution Runtime (mitigation/twirling/DD) et les horodatages fins : provenance est null sur P0
* variable_mapping doit porter (avion, option, q, theta) et non x0..xn
* le taux de faisabilité agrégé ne prouve PAS l'endianness sur la famille CP (KL = 0 exactement) ; seul le canari FP_4 le prouve empiriquement

## 12. Ordre d'exécution complet, publié avant le premier job

| # | bloc | type | cellule | objectif | bras | position du bras | rép. |
|---|---|---|---|---|---|---|---|
| 1 | 0 | canary | FP_4/K3 | quad | `CE_endianness_canary` | - | 1 |
| 2 | 0 | canary | CP_3/K3 | quad | `A1_P1_feas_angles` | - | 1 |
| 3 | 0 | canary | CP_3/K3 | man | `A1_P1_feas_angles` | - | 1 |
| 4 | 1 | campaign | CP_3/K3 | quad | `A0_P0_replication` | 1 | 1 |
| 5 | 1 | campaign | CP_3/K3 | quad | `A1_P1_feas_angles` | 2 | 1 |
| 6 | 1 | campaign | CP_3/K3 | quad | `A2_P0_transpile_L2` | 3 | 1 |
| 7 | 1 | campaign | CP_3/K3 | man | `A0_P0_replication` | 1 | 1 |
| 8 | 1 | campaign | CP_3/K3 | man | `A1_P1_feas_angles` | 2 | 1 |
| 9 | 1 | campaign | CP_3/K3 | man | `A2_P0_transpile_L2` | 3 | 1 |
| 10 | 1 | campaign | CP_4/K3 | quad | `A0_P0_replication` | 1 | 1 |
| 11 | 1 | campaign | CP_4/K3 | quad | `A1_P1_feas_angles` | 2 | 1 |
| 12 | 1 | campaign | CP_4/K3 | quad | `A2_P0_transpile_L2` | 3 | 1 |
| 13 | 1 | campaign | CP_4/K3 | man | `A0_P0_replication` | 1 | 1 |
| 14 | 1 | campaign | CP_4/K3 | man | `A1_P1_feas_angles` | 2 | 1 |
| 15 | 1 | campaign | CP_4/K3 | man | `A2_P0_transpile_L2` | 3 | 1 |
| 16 | 1 | campaign | CP_5/K3 | quad | `A0_P0_replication` | 1 | 1 |
| 17 | 1 | campaign | CP_5/K3 | quad | `A1_P1_feas_angles` | 2 | 1 |
| 18 | 1 | campaign | CP_5/K3 | man | `A0_P0_replication` | 1 | 1 |
| 19 | 1 | campaign | CP_5/K3 | man | `A1_P1_feas_angles` | 2 | 1 |
| 20 | 1 | campaign | CP_3/K5 | quad | `A0_P0_replication` | 1 | 1 |
| 21 | 1 | campaign | CP_3/K5 | quad | `A1_P1_feas_angles` | 2 | 1 |
| 22 | 1 | campaign | CP_3/K5 | man | `A0_P0_replication` | 1 | 1 |
| 23 | 1 | campaign | CP_3/K5 | man | `A1_P1_feas_angles` | 2 | 1 |
| 24 | 2 | campaign | CP_3/K3 | quad | `A1_P1_feas_angles` | 1 | 2 |
| 25 | 2 | campaign | CP_3/K3 | quad | `A2_P0_transpile_L2` | 2 | 2 |
| 26 | 2 | campaign | CP_3/K3 | quad | `A0_P0_replication` | 3 | 2 |
| 27 | 2 | campaign | CP_3/K3 | man | `A1_P1_feas_angles` | 1 | 2 |
| 28 | 2 | campaign | CP_3/K3 | man | `A2_P0_transpile_L2` | 2 | 2 |
| 29 | 2 | campaign | CP_3/K3 | man | `A0_P0_replication` | 3 | 2 |
| 30 | 2 | campaign | CP_4/K3 | quad | `A1_P1_feas_angles` | 1 | 2 |
| 31 | 2 | campaign | CP_4/K3 | quad | `A2_P0_transpile_L2` | 2 | 2 |
| 32 | 2 | campaign | CP_4/K3 | quad | `A0_P0_replication` | 3 | 2 |
| 33 | 2 | campaign | CP_4/K3 | man | `A1_P1_feas_angles` | 1 | 2 |
| 34 | 2 | campaign | CP_4/K3 | man | `A2_P0_transpile_L2` | 2 | 2 |
| 35 | 2 | campaign | CP_4/K3 | man | `A0_P0_replication` | 3 | 2 |
| 36 | 2 | campaign | CP_5/K3 | quad | `A1_P1_feas_angles` | 1 | 2 |
| 37 | 2 | campaign | CP_5/K3 | quad | `A0_P0_replication` | 2 | 2 |
| 38 | 2 | campaign | CP_5/K3 | man | `A1_P1_feas_angles` | 1 | 2 |
| 39 | 2 | campaign | CP_5/K3 | man | `A0_P0_replication` | 2 | 2 |
| 40 | 2 | campaign | CP_3/K5 | quad | `A1_P1_feas_angles` | 1 | 2 |
| 41 | 2 | campaign | CP_3/K5 | quad | `A0_P0_replication` | 2 | 2 |
| 42 | 2 | campaign | CP_3/K5 | man | `A1_P1_feas_angles` | 1 | 2 |
| 43 | 2 | campaign | CP_3/K5 | man | `A0_P0_replication` | 2 | 2 |
| 44 | 3 | campaign | CP_3/K3 | quad | `A2_P0_transpile_L2` | 1 | 3 |
| 45 | 3 | campaign | CP_3/K3 | quad | `A0_P0_replication` | 2 | 3 |
| 46 | 3 | campaign | CP_3/K3 | quad | `A1_P1_feas_angles` | 3 | 3 |
| 47 | 3 | campaign | CP_3/K3 | man | `A2_P0_transpile_L2` | 1 | 3 |
| 48 | 3 | campaign | CP_3/K3 | man | `A0_P0_replication` | 2 | 3 |
| 49 | 3 | campaign | CP_3/K3 | man | `A1_P1_feas_angles` | 3 | 3 |
| 50 | 3 | campaign | CP_4/K3 | quad | `A2_P0_transpile_L2` | 1 | 3 |
| 51 | 3 | campaign | CP_4/K3 | quad | `A0_P0_replication` | 2 | 3 |
| 52 | 3 | campaign | CP_4/K3 | quad | `A1_P1_feas_angles` | 3 | 3 |
| 53 | 3 | campaign | CP_4/K3 | man | `A2_P0_transpile_L2` | 1 | 3 |
| 54 | 3 | campaign | CP_4/K3 | man | `A0_P0_replication` | 2 | 3 |
| 55 | 3 | campaign | CP_4/K3 | man | `A1_P1_feas_angles` | 3 | 3 |
| 56 | 3 | campaign | CP_5/K3 | quad | `A0_P0_replication` | 1 | 3 |
| 57 | 3 | campaign | CP_5/K3 | quad | `A1_P1_feas_angles` | 2 | 3 |
| 58 | 3 | campaign | CP_5/K3 | man | `A0_P0_replication` | 1 | 3 |
| 59 | 3 | campaign | CP_5/K3 | man | `A1_P1_feas_angles` | 2 | 3 |
| 60 | 3 | campaign | CP_3/K5 | quad | `A0_P0_replication` | 1 | 3 |
| 61 | 3 | campaign | CP_3/K5 | quad | `A1_P1_feas_angles` | 2 | 3 |
| 62 | 3 | campaign | CP_3/K5 | man | `A0_P0_replication` | 1 | 3 |
| 63 | 3 | campaign | CP_3/K5 | man | `A1_P1_feas_angles` | 2 | 3 |

## 11. Preuve qu'aucune soumission n'a eu lieu

Voir la section « Preuve d'absence de soumission » du rapport final : drapeaux fail-closed, `usage_consumed_seconds` inchangé, aucun nouveau `job_id` chez le fournisseur, aucun artefact de run créé.
