# Rapport final — clôture du chantier hybride / scalabilité (ACRP-LIB-QUANTUM)

Date : vendredi 4 septembre 2026, soir. Branche `exp/adaptive-qpu-gate`, worktree
`acrp-lib-quantum-adaptive`. Aucun job IBM, aucun accès réseau, aucun push, merge, amend ou
rebase. `Code/`, `src/acrpq/classical/reference.py`, `src/acrpq/classical/pyomo_minlp.py`, les
résultats historiques, les résultats IBM bruts et les WIP du worktree principal sont inchangés
(hashes de la phase 0 égaux ; aucun diff sur ces chemins dans la branche).

## 1. Verdict

**`THESIS_EXPERIMENTAL_SECTION_READY_WITH_LIMITATIONS`.**

* La section expérimentale est exacte, compacte, reproduite deux fois depuis des worktrees
  propres, contre-vérifiée par quatre audits indépendants et intégrable dans un worktree propre
  basé sur `f16a0c5` sans toucher au worktree principal.
* Limitations : exactitude de la décomposition restreinte au QUBO K3 ; seuils de plausibilité
  matérielle calibrés sur trois points réels ; références fines certifiées pour n ≤ 6 ;
  CP_30 synthétique et profondeur extrapolée ; deux défauts de dépôt corrigés au commit
  d'intégration (dépendance d'un test à des exports IBM privés, deux erreurs ruff dans
  `docs/presentation/`) ; l'intégration dans la branche principale exige d'abord la suppression par
  Fabio des copies non suivies de la phase 1 présentes dans le worktree principal (§16).
* Verdicts acquis inchangés : `LOCAL_PROOF_OF_CONCEPT_VALIDATED` (adaptatif quadratique),
  aucun gain `maneuver_count_v1`, `LOCAL_ONLY` (QPU adaptatif),
  `ADDITIONAL_EXACT_DECOMPOSITION_ACCESS` (décomposition K3),
  `CP30_ISA_COMPILABLE_BUT_NOT_CREDIBLE`. Aucun verdict matériel.

## 2. Commits source (code scientifique)

| commit | contenu |
|---|---|
| `1d9009d` | encodage compact, QUBO hétérogène, chaîne hachée, porte QPU `LOCAL_ONLY` |
| `0c2263a` | annexe re-vérification red-team, clés hachées complétées |
| `a2accf3` | pack compact phase 1 (decoded.json reconstructibles retirés) |
| `7908a19`, `ddb0f78` | étude de scalabilité et pack compact |
| `8e62701` | terminologie d'accessibilité v2, provenance schéma 2, sensibilité CP30, scripts de régénération, figures |
| `1547655` | contre-preuve indépendante (agent A), hash scientifique des chaînes hors identité d'exécution |
| `1618148` | CP30 : temps de calcul sorti du texte de justification |
| `282b03a`, `6c10c60`, `9641e2f`, `b68538e` | pack compact généralisé, fins de ligne LF, correction de syntaxe |
| `442eb44` | exclusions de hash scientifique complétées (temps, identité d'exécution), `runs/` déclarés non versionnés et reconstructibles (audit B) |

## 3. Commits artefacts

| commit | contenu |
|---|---|
| `b005c7c` | artefacts régénérés dans des worktrees propres, schéma 2, pack compact |
| `33543fa` | sidecar v1 : `artifact_commit = b005c7c` |
| `fa934ba` | documentation (matrice schéma 2, section autonome, patch du pack) |
| `004759f` | campagne adaptative et chaînes régénérées dans le worktree propre de `442eb44` (corrections de l'audit B) |
| `57492a1` | sidecar v2 : `artifact_commit = 004759f`, couvre aussi `chain_manifest.json` et `preregistration.json` |
| `87ed11e` | rapport final et corrections documentaires de l'audit D |
| commit final | décompte final de la suite dans le worktree d'intégration |

## 4. Commits d'intégration

Aucun commit d'intégration dans la branche principale. Un worktree propre
`/private/tmp/acrpq-integration-f16a0c5` (détaché sur `f16a0c5`, HEAD de `acrpq-scientific-ui`)
contient la sélection décrite en §16 ; les gates y ont été exécutés (§17). Le commit
d'intégration est laissé à Fabio ou Codex après suppression des copies non suivies (§16).

## 5. Provenance corrigée

Les manifestes schéma 1 portaient un champ `git_commit` unique désignant le HEAD d'un worktree
sale au moment de la génération ou le commit d'un ré-empaquetage ultérieur. Ils sont invalidés
de façon machine-readable (`SUPERSEDED_PROVENANCE_v1.json`, hashes à `ddb0f78`). Les manifestes
schéma 2 portent : `scientific_source_commit`, `artifact_generation_commit`,
`artifact_packaging_commit` (`null` à la génération), `review_commit` (`null`),
`repository_dirty_at_generation`, `scientific_code_dirty_at_generation`, `generated_utc`,
`python`, `platform`, `scientific_hash_exclusions`. Protocole appliqué : commit du code →
génération dans un worktree propre détaché de ce commit → import → commit séparé des artefacts
→ sidecar `artifact_commit` dans un troisième commit. Aucun artefact ne nomme le commit qui le
contient.

| groupe d'artefacts | commit source scientifique | worktree propre |
|---|---|---|
| matrice des instances, résumés famille, frontière de transpilation, benchmark de décomposition, figures | `8e62701` | repro-1 (repro-2 pour comparaison) |
| chaînes adaptatives, rapport d'intégrité | `1547655` | repro-3 (deux exécutions, hash identique) |
| CP30 | `1618148` | repro-4 (deux exécutions, hash identique) |
| benchmark local adaptatif | `6c10c60` | repro-5 |
| campagne adaptative (summary), chaînes et rapport d'intégrité (version finale) | `442eb44` | repro-8 (`9641e2f`/repro-6 et `1547655`/repro-3 pour les versions intermédiaires) |
| contre-preuve | `b68538e` | repro-7 |

## 6. Inventaire des 1042 instances

CP 18 (n = 3 à 20), FP 12 (8 à 30), GP 12 (16 à 60), RCP 400 (10, 20, 30, 40), RCP_FL 600
(50, 100, 150 ; 200 alias `@FL5`). Totaux vérifiés indépendamment par l'agent A.

## 7. Théorème et preuves

Si aucun coefficient quadratique du QUBO K3 ne relie deux composantes connexes du graphe
any-option K3, alors H(x) = Σ_c H_c(x_c), l'argmin one-hot se factorise et la réunion des
solutions par composante est un optimum global du QUBO K3 sans conflit résiduel (toute paire
inter-composantes a une table de conflit K3 entièrement fausse). Exactitude limitée à K3 : les
tables dépendent de la grille et des composantes fusionnent sous échantillonnage plus fin.
Preuves : 1042/1042 instances avec 0 terme croisé (module d'analyse et contre-preuve par les
seuls modules officiels, 0 écart sur 8336 comparaisons) ; 24/24 recompositions (12
configurations × 2 objectifs) égales au monolithique à 1e-12 avec zéro conflit ; six mutations
obligatoires (terme inter-composante retiré, avion dans la mauvaise composante, avion actif fixé
à NOOP, rescoring omis, K3 et K5 mélangés, incumbent pris pour optimum) rendues rouges par des
tests.

## 8. Instances nouvellement accessibles (nominatives)

* Statevector local (≤ 30 qubits) grâce à la composante seulement : FP_6, RCP_50_37@FL5,
  RCP_50_67@FL5.
* Exact local (budget 3^13, 60 s) grâce à la composante seulement : 54 RCP_FL à 50 avions,
  alias `@FL5` (liste dans `family_summary.json`).
* Régime pilote matériel (≤ 12 qubits logiques, 2Q mesuré ≤ 210) : CP_3 (108), CP_4 (207),
  FP_4 (142), FP_5 (142). Aucune RCP ni RCP_FL.

## 9. Décomposition et accessibilité

`DECOMPOSABLE_FORMALLY` : FP 6/12, GP 1/12 (gain pratiquement nul), RCP 44/400, RCP_FL
600/600. `WIDTH_REDUCED` : FP 12, GP 2, RCP 375, RCP_FL 600. Une largeur réduite ne signifie
pas une résolution exacte globale quand la composante restante reste trop grande : 400 RCP_FL
dépassent encore 156 qubits logiques en monolithique et 517 restent `ISA_COMPILABLE_ONLY`
après décomposition. « Décomposable » n'est jamais employé pour « accessible ».

## 10. Ressources

Largeur maximale réduite (150 → 48, 300 → 69 qubits logiques) mais largeur cumulée inchangée
(3n) ; mémoire statevector 16·2^Q (48 qubits ≈ 4,5 Po) ; frontière de transpilation sur
FakeMarrakesh reproductible et identique à la campagne réelle pour CP_5 (376/348) et CP_7
(640/700) ; refus au-delà de 60 qubits logiques. Composantes FP_4/FP_5 : 12 qubits, profondeur
ISA 220, 142 portes 2Q.

## 11. CP30

Absent du corpus ; expérience synthétique : 30 avions, K3, 90 qubits logiques, clique
(densité 1), zéro isolé, aucune composante, aucune réduction exacte, 2·10²⁸ octets de vecteur
d'état ; profondeur ISA **estimée** (non transpilée) : linéaire ≈ 3 060 (leave-one-out 2 600 à
3 460), quadratique ≈ 12 900 (9 900 à 15 800) ; portes 2Q ≈ 4 000 à 14 400 selon le modèle ;
extrapolation ×4 en n depuis cinq points réels. Conclusion
`CP30_ISA_COMPILABLE_BUT_NOT_CREDIBLE`. Un recuit classique trouve une affectation faisable
(note séparée, sans référence d'optimalité, jamais une preuve matérielle).

## 12. Adaptatif

Précision à largeur 3n constante pour l'objectif quadratique (CP_3 : 0,000402 à 9 qubits
contre 0,274 pour K3 et 0,0305 pour K7) ; aucun gain `maneuver_count_v1` là où l'optimum K3 est
certifié ; chaînes Aer pilotées par les échantillons validées localement avec la règle d'angles
normalisée par la pénalité ; pilote QPU `LOCAL_ONLY`. Par composante, l'adaptatif n'hérite pas
de l'exactitude de la décomposition et exige un rescoring global (aucun conflit inter-composantes
constaté sur les instances testées).

## 13. Résultats négatifs

CP clique pour tout n ; GP une composante jusqu'à n = 56 ; dominance sans effet sur CP et GP ;
aucune instance RCP/RCP_FL plausible pour un pilote ; simulation Aer monolithique hors de portée
pour la majorité du corpus ; encodage binaire réfuté ; domain-wall non retenu ; angles fixes
historiques inopérants dès le round 2 en simulation.

## 14. Compactage

Phase 1 : 192 226 → 6 552 lignes de texte (decoded.json reconstructibles bit à bit supprimés,
locks retirés, JSON canonique compact, parité vérifiée). Freeze : artefacts régénérés puis
recompactés ; répertoires adaptatif 3,2 Mo (95 fichiers) et scalabilité 3,7 Mo (34 fichiers,
dont 8 figures SVG/PNG). Audit C : aucune suppression supplémentaire justifiée ; tous les
rapports d'experts sont cités et trois prouvent des défauts corrigés.

## 15. Reproduction

Deux worktrees propres de `8e62701` ont rejoué le pipeline complet ; tous les hashes
scientifiques (champs non déterministes exclus et documentés) sont égaux, sauf deux
divergences expliquées et corrigées : le champ `scientific_hash_chains` incluait `round_hash`
(porteur d'horodatage) à `8e62701`, corrigé à `1547655` (deux exécutions identiques) ; le texte
`verdict_rationale` de CP30 contenait le temps de calcul, corrigé à `1618148` (deux exécutions
identiques). `REPRODUCTION_COMPARISON.json` documente le tout.

## 16. Intégration sélective

Portés dans le worktree propre `f16a0c5` : `src/acrpq/experimental/` (6 modules),
5 fichiers de tests expérimentaux, 11 scripts `experimental_*`, 5 documents, les deux
répertoires de résultats compacts. Non portés : locks, caches, expansions reconstructibles,
sorties temporaires, l'addendum supersédé, tout WIP du worktree principal. Dépendances :
`geometry`, `model`, `objectives`, `io.loader`, `discretize`, `quantum.qubo`,
`dashboard.hardware_validation` (tests) — inchangées entre `4d2f75e` et `f16a0c5` ; aucun import
officiel vers l'expérimental. **Collision connue** : le worktree principal contient des copies
non suivies de la phase 1 (`src/acrpq/experimental/`, `docs/COMPACT_ENCODING_STUDY.md`,
`tests/test_experimental_adaptive_grid.py`, `scripts/experimental_adaptive_grid_campaign.py`,
`results/experimental_adaptive_grid_v1/`) plus anciennes que la branche : elles doivent être
supprimées par Fabio avant tout checkout du commit d'intégration ; aucun fichier non suivi n'a
été écrasé.

## 17. Tests et gates

### 17.1 Audits indépendants (quatre agents)

| agent | résultat | suites |
|---|---|---|
| A — audit mathématique | 1042/1042 sans terme croisé (modules officiels seulement, 0 écart sur 8336 comparaisons), 24/24 recompositions exactes, 6 mutations détectées par 9 tests | aucun |
| B — provenance / reproductibilité | 5 manifestes schéma 2 conformes, sidecar exact, invalidation schéma 1 exacte, CP30 et matrice reproduits bit à bit ; trois défauts d'outillage : `runs/` référencés mais non versionnés (haute), champs de temps et d'identité hors liste d'exclusion (basse), `chain_manifest.json` hors sidecar (moyenne) | corrigés : `442eb44` (manifeste déclare `runs_versioned=false` et la commande de reconstruction ; exclusions complétées), `004759f` (régénération propre), `57492a1` (sidecar v2) |
| C — minimalité | 118 fichiers audités, aucune suppression supplémentaire justifiée, aucun import officiel vers l'expérimental, dépendances déclarées, liens valides | aucune |
| D — intégration et affirmations | 145 chemins stagés sans écraser aucun fichier suivi de `f16a0c5` ; nombres et listes nominatives exacts ; aucun glissement de vocabulaire ; un résidu schéma 1 en §2 de l'étude (corrigé) ; collision avec cinq copies non suivies du worktree principal (documentée) ; verdict `THESIS_EXPERIMENTAL_SECTION_READY_WITH_LIMITATIONS` | corrigé dans le commit final |

### 17.2 Gates dans le worktree d'intégration propre (`f16a0c5` + sélection)

| gate | résultat |
|---|---|
| `ruff check` sur les fichiers intégrés | vert |
| `mypy src/acrpq` | vert (69 fichiers) |
| tests expérimentaux + décomposition + contre-preuve + voisins (discretize, equivalence, objectives, import safety, qubo_solvers) | 251 verts |
| suite non-UI complète | dans la branche expérimentale : un échec, `test_qpu_real_artifact.py::test_campaign_artifacts_present_and_inventoried`, dépendant d'exports IBM privés non versionnés ; corrigé au commit d'intégration `6faaaa8` (tests obligatoires sur artefacts publiés, tests optionnels sur exports privés ignorés avec raison) ; clone-fresh du commit d'intégration : 1458 verts, 10 ignorés, 0 échec |
| `build --no-isolation` | vert |
| `pip check` | vert |
| `pip-audit` | une vulnérabilité préexistante de l'environnement, `diskcache 5.6.3` (PYSEC-2026-2447), non neutralisée |
| `git diff --check` | vert (SVG générés exclus du contrôle d'espaces par `.gitattributes`) |
| scan secrets / PII / chemins absolus | vide sur les fichiers intégrés |
| parité JSON/CSV | sémantique exacte (listes CSV encodées JSON) |
| manifests / hashes | schéma 2, sidecar v2, invalidation schéma 1, vérifiés par l'audit B |
| reproduction propre ×2 | égalité des hashes scientifiques ; deux divergences expliquées et corrigées (`REPRODUCTION_COMPARISON.json`) ; chaînes, CP30 et contre-preuve rejoués deux fois à l'identique |
| liens documentaires | valides (audit D) |
| chemins protégés | hashes identiques à la phase 0 dans les deux worktrees ; aucun diff dans la branche |
| WIP préservés | worktree principal jamais écrit ; statut identique hors commits de l'autre session |
| review indépendante du diff | audits A, B, C, D |

Les deux exceptions de la branche expérimentale (échec `test_qpu_real_artifact` par dépendance
à des exports privés ; `ruff check .` sur `docs/presentation/`) sont corrigées au commit
d'intégration `6faaaa8` : `ruff check .` et la suite complète y sont verts en clone-fresh.

## 18. Risques

Exactitude K3 seulement ; extrapolation CP30 ; seuils matériels sur trois points ; références
fines certifiées n ≤ 6 ; `pip-audit` signale `diskcache 5.6.3` (environnement, non
neutralisé) ; la comparaison adaptatif / K3 porte sur des grilles différentes (drapeau
`domains_comparable_adaptive_vs_K3 = false`).

## 19. Texte prêt pour le mémoire

Voir `docs/INSTANCE_SCALABILITY_STUDY.md` §15.13 (paragraphe autonome) et
`docs/COMPACT_ENCODING_STUDY.md` §13 (raffinement adaptatif).

## 20. Affirmations autorisées et interdites

Autorisées : décomposition exacte pour le QUBO K3 vérifiée sur 1042 instances et
contre-prouvée ; trois nouveaux accès statevector et 54 nouveaux accès exacts locaux par
décomposition ; FP_4 et FP_5 rejoignent CP_3 et CP_4 dans le régime pilote observé ; CP_30
absent, synthétique, non crédible ; l'adaptatif apporte la précision, la décomposition la
largeur, séparément.

Interdites : « décomposable » = « accessible » ; « compilable » = « exploitable » ; « 156 qubits
suffisent » ; « Marrakech peut résoudre CP30 » ; « CP30 est supporté » ; un recuit classique
comme preuve matérielle ; qubits logiques = physiques ; exactitude de l'adaptatif par
composante ; tout gap entre domaines différents sans qualification ; tout nouveau pilote
matériel avant la remise.
