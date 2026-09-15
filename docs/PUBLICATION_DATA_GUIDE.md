# Guide des données publiables

Ce document dit, pour chaque jeu de données du dépôt : ce qu'il contient, comment le
vérifier, ce qu'on peut en conclure, et ce qu'il ne faut **pas** en conclure.

Règle transversale : **aucun artefact historique n'est réécrit sur place.** Toute
régénération produit un nouvel artefact explicitement versionné. Les sources gardent leurs
hachages publiés valides.

---

## 1. `results/thesis_benchmark_v1_1/` — jeu de données scientifique consolidé

* schéma `acrpq-thesis-benchmark/1.1`, `manifest_sha256 = sha256:1675238f…`
* **346 lignes canoniques** — une par (instance × grille × objectif × méthode × graine)
* fichiers : `benchmark_results.json`, `benchmark_results.csv`,
  `applicability_matrix.csv`, `synthesis_by_method.csv`, `synthesis_by_family.csv`,
  `synthesis_by_objective.csv`
* vérification : `python scripts/build_thesis_benchmark.py --verify`

Répartition par provenance, **strictement séparée** (aucun mélange dans un agrégat) :

| `data_source` | lignes | ce que c'est |
|---|---|---|
| `local_classical` | 316 | solveurs classiques locaux, CPU |
| `aer_simulator` | 24 | simulation d'état, **pas de QPU** |
| `synthetic_offline` | 6 | plomberie synthétique — **jamais** un résultat scientifique |
| `ibm_hardware` | 0 | *voir §2 : la campagne matérielle est un artefact séparé* |

Les 6 lignes `synthetic_offline` existent pour prouver que le pipeline transporte une
ligne de bout en bout ; elles portent leur classe dans la donnée et sont exclues de toute
conclusion. `ibm_hardware = 0` dans ce paquet **n'est pas une omission** : les résultats
matériels vivent dans leur propre artefact, avec leurs propres identifiants de job.

Garanties inscrites dans le manifeste (`claims`) :

* `grids_aggregated_together = false` — aucune synthèse ne mélange deux valeurs de `K`
  (toutes les tables de synthèse portent une colonne `grid_k`) ;
* `objectives_aggregated_together = false` — `quadratic_control_cost_v1` et
  `maneuver_count_v1` ne sont jamais moyennés ensemble.

Les lignes non applicables, en dépassement de temps, infaisables ou indisponibles sont
**conservées avec leur motif explicite** plutôt que supprimées : une absence silencieuse
biaiserait les taux. Un écart à la référence n'existe que face à une référence certifiée
de la même grille et du même objectif, et n'est jamais calculé sur une solution
infaisable.

### Dépendance à la grille (résultat à ne pas lisser)

Le paquet `v1_1` existe précisément parce que `v1` agrégeait des grilles différentes, ce
qui masquait une dépendance réelle à `K` pour `maneuver_count_v1` :

| `K` | taux de faisabilité | correspondance à la référence |
|---|---|---|
| 3 | 0,9381 | 0,531 |
| 5 | 0,9333 | 0,9333 |
| 7 | 1,000 | 1,000 |

`v1` n'a pas été réécrit : `v1_1` le remplace en le citant.

## 2. `results/ibm_fixed_angle_campaign_v1/` — matériel quantique réel (lecture seule)

30 jobs réellement exécutés sur `ibm_marrakesh` (Heron, 156 qubits), 512 shots chacun
(15 360 shots), QAOA `p = 1` à **angles fixes** (`betas = [0.3]`, `gammas = [0.4]`),
`optimization_level = 0`, `transpiler_seed = 1`, 5 instances × 2 objectifs × 3
répétitions. Chaque job porte son `ibm_job_ids`, son `config_hash`, l'empreinte du
circuit logique et du circuit ISA, et le détail de transpilation.

Résultat principal — taux de faisabilité mesuré :

| instance | qubits | médiane | max | jobs non nuls |
|---|---|---|---|---|
| CP_3 | 9 | 5,37 % | 9,96 % | 6/6 |
| CP_4 | 12 | 0,78 % | 2,15 % | 6/6 |
| CP_5 | 15 | 0,10 % | 0,39 % | 3/6 |
| CP_6 | 18 | 0,00 % | 0,20 % | 1/6 |
| CP_7 | 21 | 0,00 % | 0,00 % | 0/6 |

Médiane globale : **0,20 %**. Exemple documenté (`CP_3`,
`quadratic_control_cost_v1`) : profondeur 10 → 211 après transpilation, 108 portes à deux
qubits, 235 chaînes uniques / 512 shots, meilleure faisable `010001001`, objectif
`0.2741555637`.

### Cette dégradation n'est PAS attribuable au bruit matériel

Correction importante, établie par le calcul exact hors ligne
(`results/offline_angle_landscape_v2/`, schéma `acrpq-offline-angle-landscape/2`) : aux
angles fixes de P0 (`γ=0,4`, `β=0,3`), la masse de probabilité faisable **d'un simulateur
parfait, sans aucun bruit**, s'effondre déjà :

| instance | qubits | masse faisable idéale (quadratic) | shots faisables attendus sur 512 | mesuré sur matériel |
|---|---|---|---|---|
| CP_3 | 9 | 0,1096 | 56 | 9,51 % |
| CP_4 | 12 | 0,0225 | 11,5 | 1,04 % |
| CP_5 | 15 | **0,00004** | **0,02** | 0,00 % |
| CP_6 | 18 | 0,00030 | 0,15 | 0,00 % |
| CP_7 | 21 | 0,00020 | 0,10 | 0,00 % |

Les zéros observés sur CP_5, CP_6 et CP_7 en `quadratic_control_cost_v1` sont donc le
**résultat attendu d'un simulateur idéal** à ces angles : moins d'un shot faisable est
espéré sur 512. Aucun bruit n'est nécessaire pour les expliquer. En particulier,
**CP_5/quadratic possède une masse faisable idéale quasi nulle (4·10⁻⁵)** aux angles P0 —
ce n'est pas exactement zéro, et il ne faut pas l'écrire « nul », mais l'espérance est de
0,02 shot.

**Formulation autorisée** pour toute mention de cette dégradation :
« dégradation observée résultant conjointement du choix des angles, de la structure de
l'encodage, de la transpilation et du bruit matériel ». Toute formulation attribuant la
chute directement ou principalement au bruit est **interdite** : la part attribuable aux
angles est mesurée et dominante sur CP_5–CP_7.

**Interdictions de lecture.** Ne pas en conclure un avantage quantique. Ne pas en conclure
quoi que ce soit sur QAOA en général : les angles ne sont **pas** optimisés — c'est une
cohorte à angles fixes, et le calcul hors ligne montre que ce choix d'angles est mauvais.
Ne pas extrapoler au-delà de 21 qubits. Ne pas comparer ces valeurs à une ligne
`aer_simulator` comme si c'était le même dispositif. Ne pas présenter le taux de
faisabilité agrégé comme une preuve de la convention d'endianness : la famille CP à K=3
possède une symétrie miroir exacte (avion i ↔ n+1−i, option o ↔ K−1−o) sous laquelle ce
taux est **identique dans les deux conventions** — seule la comparaison bit-à-bit du
meilleur échantillon faisable tranche (30/30 dans le sens retenu, 18/30 dans l'autre).

Cet artefact est **immuable**. Aucun script du dépôt ne le réécrit ; le mode démonstration
ne fait que le relire.

## 3. `results/publication_bundle_v1/` — copies publiables AMPL/Gurobi

56 fichiers, 54 substitutions de chemins, `manifest_sha256 = sha256:0bb1f1b2…`.
Vérification : `python scripts/build_publication_bundle.py --verify`.

47 artefacts historiques contiennent des chemins absolus de la machine de développement
(`code_dir`, `ampl_binary`, …) **à l'intérieur de contenu haché** : les nettoyer sur place
invaliderait tous les hachages publiés. Le paquet résout cela sans mentir :

* les sources restent **octet pour octet identiques** (elles sont seulement lues) ;
* les copies remplacent les chemins privés par le marqueur `<portable>` ;
* `transformation_manifest.json` lie chaque sortie à sa source par SHA-256, avec les
  règles de nettoyage, les champs modifiés et le nombre de substitutions ;
* **deux** contrôles programmatiques, et non un seul : (1) les deux documents sont
  comparés **champs de chemin retirés** — cela attrape un nettoyage qui abîmerait autre
  chose que la provenance ; (2) la suite ordonnée de **toutes les valeurs numériques** doit
  être identique. Le contrôle (2) existe parce que le (1) retire la même liste de clés des
  deux côtés : élargir cette liste par erreur à une clé scientifique masquerait les dégâts,
  alors qu'une mesure est toujours un nombre. Le constructeur **refuse** si l'un des deux
  échoue (`scientific_values_modified = 0`) ;
* la sortie est scannée : chemins absolus, adresses e-mail, jetons, `Authorization`/
  `Bearer`, clés d'API, fichiers de licence, chemins de venv.

**Formulation exacte à employer** : ce paquet est *scientifiquement équivalent* aux
sources, il n'est **pas** identique octet pour octet. Citer les sources pour vérifier un
hachage, le paquet pour publier.

## 4. Ce qui n'est pas publiable en l'état

* la ligne continue AMPL/Gurobi n'apparaît pas dans les exports de l'interface
  (limitation connue, documentée, non corrigée) ;
* `feasibility_rate` n'a pas le même dénominateur entre les constructeurs `v2` et `v3` :
  ne pas comparer ces deux versions directement ;
* le champ `result_raw` n'est pas couvert par la chaîne de hachage anti-altération :
  il est inspectable mais pas certifié ;
* les artefacts `ibm_fez` sont hérités d'un réglage antérieur et portent leur mise en
  garde : ne pas les fusionner avec la campagne `ibm_marrakesh` ;
* trois anomalies connues des références classiques restent **non corrigées faute
  d'autorisation** du directeur ; elles sont signalées là où elles apparaissent et ne
  doivent pas être présentées comme des résultats.

## 5. Vérifier l'ensemble en trois commandes

```bash
python scripts/build_thesis_benchmark.py --verify       # jeu scientifique
python scripts/build_publication_bundle.py --verify     # copies publiables
PYTHONPATH=src python scripts/demo_offline.py --check    # les trois artefacts + fail-closed
```

Un hachage qui ne correspond plus fait **échouer** la commande. Aucun de ces scripts ne
contacte le réseau, ne lit de compte IBM et ne peut soumettre un job.
