# Rapport final — campagne IBM QAOA finale (angles optimisés hors ligne)

`protocol_id = acrpq-ibm-final-campaign/2` · `protocol_sha256 =
sha256:29903c28fa2802457df9519bc62ff2b2b926b8293316d5e8a688a149490ab468`

Backend : **ibm_marrakesh exclusivement**. 63 jobs, 512 shots chacun, un circuit par job.
Aucune boucle QAOA hybride sur le QPU : les angles sont soit historiques (P0), soit
optimisés hors ligne par calcul exact (P1_feas), jamais réoptimisés à partir d'un résultat
matériel.

---

## 1. Hypothèses préenregistrées

Écrites dans `docs/IBM_FINAL_CAMPAIGN_PREREGISTRATION.md` avant tout job de cette campagne.

* **H1** — des angles optimisés hors ligne augmentent le taux de shots faisables par
  rapport aux angles fixes historiques, surtout sur les petites instances.
* **H2** — la faisabilité décroît avec le nombre de qubits, la profondeur ISA et le
  nombre de portes 2Q.
* **H3** — la perte se décompose en invalidité one-hot, one-hot valide mais conflit
  géométrique, et faisable mais sous-optimal.
* **H4** — à K=3 les deux `objective_id` sont deux paramétrisations énergétiques du même
  problème décisionnel (proportionnalité exacte, facteur 0,137078) ; à K=5 ils ne le sont
  plus (vérifié : ratios 0,034–0,137, argmin différent).
* **H5** — parmi les runs avec au moins un échantillon faisable, le meilleur atteint
  souvent l'optimum discret certifié.

## 2. Budget prévu et consommé

| | |
|---|---|
| métrique facturée | secondes QPU (`usage_consumed_seconds`, plan open) |
| limite de la fenêtre (28 j) | 600 s |
| consommé avant cette campagne (30 P0 + 5 jobs non provenancés d'août + 3 canaris) | 76 s |
| **consommé par cette campagne (63 jobs)** | **126,0 s** (réconcilié via `job.metrics()`, cohérent avec 196 s − 70 s = 126 s sur la fenêtre totale) |
| seuil d'arrêt opérationnel | 240 s (jamais approché — le plus haut niveau atteint fut 120 s) |
| plafond dur | 252 s |
| **consommé total sur la fenêtre, fin de campagne** | **196 s** |
| **restant, fin de campagne** | **404 s** |

Coût mesuré par job : **exactement 2,000 s**, indépendamment de la profondeur ISA (102 à
259 selon la cellule) — confirmé par `job.metrics()['usage']['quantum_seconds']` sur les
63 jobs, la même source qui donnait 2,000 s sur les 30 jobs P0.

Aucun job n'a dépassé 4 s/job. Aucun coût anormal.

## 3. Protocoles exacts

**P0 (angles fixes, historique)** : γ=0,4, β=0,3, `optimization_level=0`.
**P1_feas (angles optimisés hors ligne)** : angles par cellule (instance×K×objectif),
choisis pour **maximiser la masse de probabilité faisable** sur le vecteur d'état exact
(sans bruit, sans échantillonnage), gelés dans
`results/offline_angle_landscape_v2/angle_landscape.json`
(`sha256:0599b281f159849777e219b699ae5fb0a7325963008daa7022165ca7a6351e60`) **avant** le
premier job de cette campagne. **`P1_feas` n'est jamais appelé « QAOA standard » ni
« QAOA optimisé » sans préciser « hors ligne »** — le critère standard (minimiser
l'espérance d'énergie, `P1_minH`) reste **entièrement hors ligne** et n'a jamais été
soumis au matériel : il annule la masse faisable de `maneuver_count_v1` (masse 2·10⁻⁵ à
5·10⁻⁵), un résultat négatif publié gratuitement.

**Bras** (chacun ne change qu'une seule variable par rapport à A0) :

| bras | angles | niveau | rôle |
|---|---|---|---|
| A0_P0_replication | P0 | 0 | réplication contemporaine, supprime la confusion avec la dérive de calibration entre campagnes |
| A1_P1_feas_angles | P1_feas | 0 | isole l'effet des angles |
| A2_P0_transpile_L2 | P0 | 2 | isole l'effet de la transpilation |
| CE_endianness_canary | P0 | 2 | canari, cellule FP_4 non symétrique |

**Ordre d'exécution** : blocs appariés publiés avant le premier job. Au sein de chaque
bloc, les bras d'une cellule sont exécutés consécutivement ; leur position tourne en
carré latin d'un bloc au suivant (parfait pour les cellules à 3 bras CP_3/K3 et CP_4/K3 ;
déséquilibre résiduel 2 contre 1, minimum atteignable, pour les cellules à 2 bras CP_5/K3
et CP_3/K5).

## 4. Jobs et job_ids

63 jobs, tous `state=completed`, tous sur `ibm_marrakesh`, tous à 512 shots reçus. Liste
complète des `(execution_order, job_key, ibm_job_id)` dans
`results/ibm_qaoa_final_campaign_v1/campaign_ledger.json`. Extrait :

| # | job_key | job_id |
|---|---|---|
| 1 | `FP_4__K3__quad__CE_endianness_canary__r1__e001` | `dad9p9642tqs73at5bd0` |
| 2 | `CP_3__K3__quad__A1_P1_feas_angles__r1__e002` | `dad9q5t1ierc738kj6qg` |
| 3 | `CP_3__K3__man__A1_P1_feas_angles__r1__e003` | `dad9qhtnj4cs73adjnv0` |
| … | (60 jobs supplémentaires, ordres 4–63) | voir `campaign_ledger.json` |
| 63 | `CP_3__K5__man__A1_P1_feas_angles__r3__e063` | `dadcu7tnj4cs73adnkp0` |

## 5. Anomalies trouvées, corrigées et publiées

**Anomalie 1 (scientifique, importante) — références certifiées en mauvaises unités.**
`Landscape.__init__` (v2 du paysage hors ligne) calculait la référence avec
`geometry.objective()`, qui renvoie toujours le coût quadratique quel que soit
l'`objective_id`. À K=3 la proportionnalité exacte masquait l'erreur (même argmin) ; à
K=5 elle ne le fait pas : `CP_3/K5/maneuver_count_v1` passe de 6 à **36 optima
certifiés** après correction. Corrigé (`primary_objective(objective_id, …)`), vérifié
bit à bit avant/après (336 invariants, 0 violation d'angle/hash/masse), reproduit par un
agent indépendant. `protocol_sha256` a changé en conséquence
(`sha256:a3d207e2…` → `sha256:29903c28…`), avec une supersession lisible par machine
déclarant explicitement que les circuits, angles, QUBO et shots ne sont PAS affectés.
Trois jobs déjà soumis sous l'ancien hash (les canaris) ont été préservés et classés
`protocol_superseded_for_reference_metadata=true`, jamais réexécutés.

**Anomalie 2 (comptable, mineure) — `billed_seconds` mal réparti par job.** L'estimation
incrémentale du runner (différence entre deux appels `usage()`) souffre d'un décalage de
latence côté fournisseur : un job affichait 0,0 s, un autre −2,0 s, alors que chacun
coûte réellement 2,000 s. La somme totale restait juste ; seule la répartition par job
était faussée. Corrigé en interrogeant `job.metrics()` directement (même source fiable
que pour les 30 jobs P0), publié comme champ supplémentaire dans le sidecar sans réécrire
le champ d'origine.

**Anomalie 3 (opérationnelle, signalée, non résolue) — 5 jobs matériels sans provenance.**
5 jobs `ibm_marrakesh` du 2026-08-18 apparaissent chez le fournisseur sans figurer dans
aucun artefact du dépôt. Ils n'ont pas été intégrés à cette campagne ni à aucun agrégat
scientifique. Signalé, à inventorier séparément.

**Serveur tiers.** Un serveur (port 8055, PID 37368) tournait déjà au lancement de la
campagne, non démarré par cette mission, avec `runtime_factory_enabled=true` mais
`submission_allowed=false`. Jamais utilisé pour cette campagne (chemin exclusif : port
8056, arrêté proprement en fin de campagne). Signalé pour identification par son
propriétaire.

## 6. Dataset et hashes

* `results/ibm_qaoa_final_campaign_v1/campaign_ledger.json` — brut, jamais réécrit.
* `results/ibm_qaoa_final_campaign_v1/raw/*.json` — copie figée par job, un refus explicite si un brut existant diffère.
* `results/ibm_qaoa_final_campaign_v1/sidecar/*.json` — métadonnées corrigées, liées au brut par `raw_sha256`.
* `results/ibm_qaoa_final_campaign_v1/campaign_results.{json,csv}` — dataset canonique, 63 lignes, parité JSON/CSV vérifiée valeur par valeur.
* `results/ibm_qaoa_final_campaign_v1/campaign_manifest.json` — séparations déclarées et vérifiées : `grids_pooled=false`, `objectives_pooled=false`, `arms_pooled=false`, `canaries_excluded_from_hypothesis_stats=true`.
* `protocol_sha256 = sha256:29903c28fa2802457df9519bc62ff2b2b926b8293316d5e8a688a149490ab468`
* `preregistration content_sha256 = sha256:4dce545d0814b7988606011236c56b0cad1d6f9eeb36832e4a1abeaf9ad91aa3`
* `offline_angle_landscape_v2 content_sha256 = sha256:0599b281f159849777e219b699ae5fb0a7325963008daa7022165ca7a6351e60`

## 7. Statistiques (corrigées — voir §15bis, contre-review Codex)

**Correction méthodologique majeure.** La première version de cette section poolait les
512 shots des 3 jobs d'un bras en un dénominateur unique de 1536 et calculait un
intervalle de Newcombe dessus — traitant implicitement les shots inter-job comme
échangeables, alors que chaque job est un tirage sur un état matériel possiblement
différent (dérive de calibration). Cet intervalle était artificiellement étroit et ne
mesurait pas l'incertitude inter-run. **L'analyse correcte prend le job comme unité
expérimentale : 3 paires (A0, A1) par cellule, une par `repetition_id`.**

| cellule | A0 (3 jobs, %) | A1 (3 jobs, %) | d_r = A1−A0 (pt) | moyenne | médiane | +/−/0 | p signe |
|---|---|---|---|---|---|---|---|
| CP_3 K3/maneuver | 0,98 · 1,56 · 2,54 | 5,47 · 4,88 · 4,88 | +4,49 · +3,32 · +2,34 | +3,385 | +3,320 | 3/0/0 | 0,250 |
| CP_3 K3/quadratic | 9,77 · 8,40 · 9,38 | 11,13 · 6,64 · 9,57 | +1,37 · −1,76 · +0,20 | −0,065 | +0,195 | 2/1/0 | 1,000 |
| CP_3 K5/maneuver | 0,39 · 0,20 · 0,20 | 0,98 · 1,37 · 0,59 | +0,59 · +1,17 · +0,39 | +0,716 | +0,586 | 3/0/0 | 0,250 |
| CP_3 K5/quadratic | 1,37 · 0,78 · 1,37 | 3,52 · 2,73 · 2,54 | +2,15 · +1,95 · +1,17 | +1,758 | +1,953 | 3/0/0 | 0,250 |
| CP_4 K3/maneuver | 0,59 · 0,59 · 0,39 | 3,12 · 3,12 · 2,15 | +2,54 · +2,54 · +1,76 | +2,279 | +2,539 | 3/0/0 | 0,250 |
| CP_4 K3/quadratic | 0,78 · 0,59 · 0,59 | 2,34 · 1,95 · 3,12 | +1,56 · +1,37 · +2,54 | +1,823 | +1,562 | 3/0/0 | 0,250 |
| CP_5 K3/maneuver | 0,39 · 0,00 · 0,59 | 0,78 · 1,76 · 1,17 | +0,39 · +1,76 · +0,59 | +0,911 | +0,586 | 3/0/0 | 0,250 |
| CP_5 K3/quadratic | 0,00 · 0,00 · 0,00 | 0,20 · 0,20 · 0,20 | +0,20 · +0,20 · +0,20 | +0,195 | +0,195 | 3/0/0 | 0,250 |

**7 cellules sur 8** ont une moyenne(d_r) positive ; dans **7 de ces cellules, les 3
différences sont individuellement positives** (le signe ne s'inverse jamais entre
répétitions). `CP_3/K3/quadratic` est la seule cellule à signe mixte (2 positives,
1 négative), avec une moyenne quasi nulle — cohérente avec le plus faible gain idéal
prédit hors ligne pour cette cellule précise (×1,62 contre ×3,1 à ×627 pour les autres).

**Aucune inférence forte n'est possible à cette taille.** Le test des signes exact
bilatéral donne **p = 0,250 pour les 7 cellules à 3/3**, la valeur la plus basse
atteignable avec 3 paires (2×0,5³) — mathématiquement, ce test ne peut jamais démontrer
un effet à α=0,05 à n=3, quel que soit le résultat observé. Aucun intervalle de
confiance inter-run n'est publié.

Les anciens intervalles binomiaux sur shots sont conservés mais renommés et redéfinis :
`conditional_shot_sampling_interval_per_job`, calculés **par job individuel** (jamais
poolés), décrivant uniquement le bruit de tir conditionnel à l'état matériel de CE job
précis — jamais une preuve d'effet entre bras ni un intervalle inter-run.

**A2 (transpilation seule, angles P0 inchangés)** — résultat honnête, pas lissé : la
profondeur baisse de 28 % à 42 %, mais la faisabilité **ne s'améliore pas
systématiquement** — elle se dégrade même sur `CP_3/K3/quadratic` (9,18 %→5,99 %). Une
profondeur réduite n'implique donc pas mécaniquement une meilleure faisabilité dans ce
protocole (interaction probable avec le nouveau layout physique, non isolée ici).

**Formulation autorisée (Codex) pour citer ce résultat** : « Dans 7 cellules sur 8, la
différence moyenne observée entre A1 et A0 est positive. Avec trois jobs par bras, ces
résultats constituent un signal préliminaire cohérent avec les prédictions idéales, sans
puissance suffisante pour une conclusion statistique forte. »

## 8. Figures

4 figures (SVG+PNG+données CSV sources) dans `results/ibm_qaoa_final_campaign_v1/figures/` :
`fig1_a0_vs_a1_paired`, `fig2_feasibility_vs_qubits_K3`, `fig3_a2_transpilation_level`,
`fig4_k3_vs_k5_CP_3`. K3 et K5 jamais superposés sur un même axe de comparaison directe ;
les deux objectifs toujours distingués visuellement.

## 9. Conclusions autorisées

* Le QUBO reste validé contre les références discrètes recalculées (voir §5, référence
  corrigée et reproduite indépendamment).
* Dans 7 cellules sur 8, la différence moyenne observée entre A1 et A0 est positive ;
  dans 7 de ces cellules les 3 répétitions vont individuellement dans le même sens. Avec
  trois jobs par bras, ce résultat constitue un **signal préliminaire** cohérent avec les
  prédictions idéales hors ligne, **sans puissance suffisante pour une conclusion
  statistique forte** (le test des signes exact ne peut atteindre α=0,05 à cette taille).
* Minimiser l'espérance d'énergie (critère QAOA standard) donne, sur les 14 cellules
  étudiées à ce protocole p=1, une masse faisable idéale inférieure aux angles fixes
  historiques pour `maneuver_count_v1` — un résultat négatif obtenu sans dépense.
* Réduire la profondeur ISA seule (transpilation niveau 2, mêmes angles) n'améliore pas
  systématiquement la faisabilité observée dans ce protocole.
* Le canari d'endianness confirme empiriquement, sur matériel, la convention utilisée
  pour décoder les 30 jobs P0 et les 60 jobs de cette campagne.

## 10. Conclusions interdites

* Aucun avantage quantique.
* Aucune généralisation à QAOA au-delà de p=1 et de ce QUBO one-hot fortement pénalisé.
* Aucune conclusion causale (« les angles causent la faisabilité ») — associations
  observées seulement, sur n=3 répétitions par bras.
* Aucune comparaison directe K3 vs K5 comme si c'était la même grille.
* Aucun pooling des deux `objective_id`.
* Aucune inclusion des 3 canaris dans les statistiques d'hypothèse officielles.
* Aucune attribution de la faisabilité résiduelle au bruit matériel seul sans mentionner
  la part mesurée des angles et de la transpilation.

## 11. Limites

* n=3 répétitions par bras : variance non calibrée, signal préliminaire uniquement.
* A0 et A1 partagent le même niveau de transpilation (0) par construction ; A2 isole le
  niveau mais ne recoupe pas avec les angles optimisés (pas de cellule A1+niveau 2 dans
  cette campagne — extension possible, non financée ici).
* Le canari FP_4 ne couvre que la cellule quadratique ; aucun canari d'endianness dédié
  n'a été soumis pour `maneuver_count_v1` (l'argument de convention reste physique et
  documenté, pas testé empiriquement sur cet objectif précis).
* 5 jobs matériels historiques restent sans provenance retrouvée (voir §5, anomalie 3).

## 12. Fichiers et commits

Scripts : `scripts/offline_angle_landscape_v2.py`, `scripts/verify_landscape_correction.py`,
`scripts/emit_preregistration.py`, `scripts/run_ibm_final_campaign.py`,
`scripts/build_campaign_sidecar.py`, `scripts/build_final_campaign_dataset.py`,
`scripts/analyze_final_campaign.py`, `scripts/build_final_campaign_figures.py`.

Commits (branche `acrpq-scientific-ui`, rien poussé) :
`f18b408` (fix référence), `f176639` (protocole v2 + supersession),
`67bf00f` (canaris + sidecar), `628972d` (60 jobs réels), `1740f04` (billing autoritaire),
`afe6625` (dataset), `52f8e6d` (statistiques), `a8e14e9` (figures).

## 13. Commandes et gates

```
ruff check .                                        -> All checks passed!
mypy src/acrpq --exclude 'experimental/'             -> Success: no issues found in 64 source files
pytest tests --ignore=tests/ui                       -> 1438 passed, 3 skipped
git diff --check                                     -> clean
python scripts/build_final_campaign_dataset.py --verify   -> OK (63 rows, parité JSON/CSV)
python scripts/offline_angle_landscape_v2.py --verify      -> OK (14 rows, 14 validations)
python scripts/verify_landscape_correction.py <avant> <après>  -> SAFE_METADATA_ONLY
```

`src/acrpq/experimental/adaptive_grid.py` (chantier tiers parallèle, non créé par cette
mission) exclu explicitement de mypy — signalé, non touché.

## 14. Risques résiduels

1. 5 jobs matériels historiques (2026-08-18) sans provenance retrouvée.
2. Un serveur tiers (port 8055) tournait pendant la campagne, propriétaire non identifié.
3. `src/acrpq/experimental/adaptive_grid.py` et `docs/COMPACT_ENCODING_STUDY.md`,
   chantiers parallèles non touchés, présents dans l'arbre de travail.
4. Pas de cellule A1+niveau2 : l'interaction angles optimisés × transpilation réduite
   reste non mesurée.
5. Aucun canari d'endianness dédié pour `maneuver_count_v1`.

## 15. Red-team indépendant

Un agent indépendant, n'ayant lu aucun rapport narratif préalable, a tenté de falsifier
15 points sur le dataset : authenticité matérielle, total des shots, endianness,
objectif utilisé, référence utilisée, comparabilité de grille, gaps, pooling,
duplication, sélection post hoc, provenance, parité CSV/JSON, intervalles, figures,
budget réel.

**14/15 points confirmés**, souvent avec une précision remarquable :

* authenticité matérielle : 12 job_id tirés au sort interrogés en direct chez le
  fournisseur, statut `DONE`, comptages **bit à bit identiques** au brut pour l'un
  d'eux (275 bitstrings uniques) ;
* endianness : réimplémentation totale du canari FP_4 sans aucun import du dépôt
  (QUBO, simulateur QAOA maison, formule du LLR recopiée du protocole) —
  **LLR = 584,2673 nats contre 584,2670 déclaré**, concordance à 5–6 chiffres
  significatifs ;
* référence certifiée : énumération exhaustive indépendante confirmant **36 optima**
  pour CP_3/K5/maneuver_count_v1 et 112 pour CP_4/K5/maneuver_count_v1 — le bug corrigé
  §5 est bien corrigé correctement, pas seulement changé ;
* sélection post hoc : angles et `qubo_sha256` **identiques bit à bit** entre les
  canaris pré-correction et leurs répliques officielles post-correction ; le
  préenregistrement est chargé une seule fois en tête d'exécution, aucun code ne
  pouvait modifier les angles pendant les 60 jobs ;
* budget : `usage_consumed_seconds` interrogé en direct (196 s / 404 s restants),
  cohérent au dixième de seconde avec les 126,0 s revendiquées ;
* parité CSV/JSON, endianness et absence de pooling illégitime, absence de
  gap sur solution infaisable, absence de duplication de job, absence de secret/chemin
  privé : tous confirmés par recalcul indépendant.

**1 anomalie réelle trouvée (IMPORTANT, localisée)** : `fig4_k3_vs_k5_CP_3` mélangeait
silencieusement les 2 jobs canaris dans la moyenne K3/A1, gonflant les taux affichés
(10,45 %/5,57 % au lieu de 9,11 %/5,08 %). N'affectait ni `statistics.json`, ni le test
d'hypothèse officiel A0/A1, ni les 3 autres figures. **Corrigé dans le commit `5ab7ce5`**,
neuter-vérifié (le bug réintroduit fait échouer exactement le nouveau test de
régression), figure régénérée avec les valeurs correctes.

*Ce premier red-team portait sur l'authenticité et l'intégrité du dataset — l'analyse
statistique elle-même a depuis été corrigée et re-vérifiée séparément, voir §15bis.*

## 15bis. Deuxième contre-review (Codex) — correction de l'analyse statistique

Après le premier red-team (§15), une contre-review Codex indépendante a identifié un
**défaut de méthode important/critique** dans l'analyse statistique elle-même (pas dans
le dataset brut) : `analyze_final_campaign.py` poolait les 512 shots des 3 jobs d'un
bras en un dénominateur unique de 1536 et calculait un intervalle de Newcombe dessus —
traitant implicitement les shots inter-job comme des répétitions indépendantes, ce qui
produit des intervalles artificiellement étroits et ne mesure pas l'incertitude
inter-run. **Le dataset brut (63 résultats matériels) restait accepté sans réserve** ;
seule l'analyse dérivée était en cause. Conséquence directe : l'ancien décompte
« 6 cellules sur 8 » (fondé sur l'IC poolé) était lui-même **faux** — le bon décompte,
au niveau job, est 7/8 (voir §7).

**Correction appliquée** : le job redevient l'unité expérimentale de fait, pas seulement
de discours. Pour chaque cellule, 3 paires (A0, A1) appariées STRICTEMENT par
`repetition_id` (jamais par ordre d'apparition dans le fichier — un job canari intercalé
aurait décalé un appariement naïf), statistiques purement descriptives (moyenne,
médiane, min/max, écart-type descriptif, décompte de signes), test des signes exact
affiché avec sa limite théorique (p≥0,25 à n=3, ne peut jamais atteindre α=0,05),
intervalles binomiaux renommés `conditional_shot_sampling_interval_per_job` et calculés
**par job** (jamais poolés), bootstrap exploratoire sur les paires de jobs explicitement
marqué instable à n=3. `statistics.json` (schéma `acrpq-ibm-final-campaign-statistics/2`)
porte un champ `invalidates_previous_version` référençant le hash de l'ancien fichier et
expliquant le défaut. `fig1_a0_vs_a1_paired` a été redessinée pour montrer les 3 paires
individuelles par cellule plutôt qu'un intervalle poolé.

9 tests de régression neuter-vérifiés ajoutés (`tests/test_analyze_final_campaign.py`),
garantissant notamment qu'aucune sortie ne peut jamais porter un champ
`statistically_significant`, que l'unité expérimentale ne reçoit jamais n=1536, que
gonfler les shots d'un job ne crée pas de répétition fantôme, et que l'appariement
respecte strictement cellule + `repetition_id`.

**Deuxième red-team indépendant, sur l'analyse corrigée** (n'important aucun code du
dépôt, recalcul intégral depuis `campaign_results.json`) : **11/11 points confirmés,
zéro écart**. Il a lui-même identifié et déjoué le piège de l'appariement par position
(un canari intercalé dans l'ordre d'apparition aurait décalé un appariement naïf) en
indexant explicitement par `repetition_id`, confirmant que la méthode du dépôt est
robuste à ce piège. Il a recalculé à la main les 8×3 taux A0, 8×3 taux A1, 8×3
différences, les agrégats (7/8, 7/8), les 8 p-values du test des signes (toutes ≥0,25,
identiques), l'exclusion effective des 3 canaris (recherche directe de leurs taux dans
les listes de la statistique, aucune fuite trouvée), et la figure `fig1` — sans le
moindre écart.

## 16. Verdict

**THESIS_ANALYSIS_READY**

Justification, selon les critères de la mission : les données matérielles sont
**authentiques** (12/12 job_id vérifiés en direct chez le fournisseur, un job vérifié
bit à bit contre son brut), **complètes** (63/63 jobs `completed`, 0 échec, 32 256/32 256
shots recomptés indépendamment), **non dupliquées** (63 `job_key` et 63 `job_id`
uniques), **décodées avec le bon objectif et la bonne référence** (6/6 `qubo_sha256`
reconstruits indépendamment concordent, référence certifiée reproduite par énumération
exhaustive y compris sur la cellule où un bug avait été trouvé et corrigé), et **les
analyses sont désormais méthodologiquement correctes et reproductibles** (job comme
unité expérimentale, appariement strict, aucune inférence forte revendiquée,
11/11 points confirmés par un second red-team indépendant sans le moindre écart).

Deux défauts réels ont été trouvés dans ce cycle et tous deux corrigés avant ce
verdict, chacun avec un test de non-régression neuter-vérifié : une figure dérivée
contaminée par des canaris (§15), et un défaut de méthode statistique important/critique
qui aurait fait dire au dépôt « 6/8 cellules significatives » alors que le bon résultat,
honnêtement formulé, est « 7/8 cellules avec un signal préliminaire positif, sans
puissance statistique suffisante » (§15bis).

Ce verdict porte sur la **fiabilité du dataset et la validité méthodologique des
analyses**, pas sur la force des conclusions scientifiques elles-mêmes : celles-ci
restent, comme indiqué au §9–§11, des signaux préliminaires à n=3 répétitions par bras.
**Hypothèse H1** : soutenue préliminairement par 7/8 cellules à direction cohérente avec
les prédictions idéales — **non confirmée** au sens statistique, et ne le sera jamais à
cette taille d'échantillon (le test des signes exact ne peut pas atteindre α=0,05 à
n=3, quel que soit le résultat).
