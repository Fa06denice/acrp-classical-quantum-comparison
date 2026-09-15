# Inventaire WIP — scripts/offline_angle_optimization.py

Date d'inventaire : 2026-09-04T09:21:12Z
Statut git : ?? (non suivi)
Jamais commité : 0 commit(s) trouvé(s)
Aucun stash : 0 entrée(s)

## Hash de l'état courant (gelé à cet instant)
```
4decad68ce714c71564c308ee4e6e93fcf0cce68541c7df2a65ccb36f8c36bf5  scripts/offline_angle_optimization.py
37fa9d593d356fb2053ac5b411d42a2e3b9efdc52e8aac40a25aee6ed0305edf  results/offline_angle_optimization_v1/offline_angles.json
```

Taille : 12421 octets, 280 lignes

## Ambiguïté déclarée

Le `git status` du début de session listait ce chemin comme WIP non suivi. Il porte
aujourd'hui du contenu écrit par l'orchestrateur. Git n'a aucune trace d'un contenu
antérieur (jamais commité, aucun stash), donc si un contenu antérieur a existé, il
n'est pas récupérable depuis git. L'écriture a réussi là où l'outil aurait refusé
d'écraser un fichier existant non lu, ce qui indique un chemin vide au moment de
l'écriture — sans que ce soit une preuve. À vérifier par le propriétaire.

## Décision (Codex point 3)

Ce fichier est **gelé** : plus aucune écriture dessus. Toute extension (K5, CP_6/CP_7,
masse sur l'optimum certifié) va dans un script versionné distinct,
`scripts/offline_angle_landscape_v2.py`, afin de ne pas mélanger les travaux.
