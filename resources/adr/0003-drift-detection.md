# ADR-003 — Détection de drift écrite à la main, Evidently en visualisation

**Statut :** accepté · **Date :** 20/08/2026

## Contexte

Le pipeline doit comparer la distribution du lot de données le plus récent à celle
des lots antérieurs, produire un verdict exploitable en automatisation, et un
rapport lisible par un humain.

Evidently est la bibliothèque de référence pour cette tâche : présets de drift,
tests statistiques par type de colonne, rapports HTML interactifs.

## Décision

La logique de détection — index de stabilité de population (PSI), statistique de
Kolmogorov-Smirnov, seuils, verdict au niveau du jeu de données — est implémentée
en numpy dans `src/simplex_warmstart/drift.py` et couverte par des tests unitaires.

Evidently est utilisé uniquement pour produire `reports/drift.html`, et déclaré en
dépendance **optionnelle** (`--extra monitoring`). Son absence dégrade le rapport
sans faire échouer le pipeline.

## Justification

**Testabilité du verdict.** Le seuil de drift influe sur ce qui est signalé dans
une pull request. On veut pouvoir écrire des tests qui affirment « PSI croît avec
l'ampleur du décalage », « une colonne constante ne fait pas planter le calcul »,
« la drift du jeu de données se déclenche au-delà de la part configurée ». Ces
tests sont directs sur du code interne, indirects et fragiles sur une bibliothèque
tierce.

**Coût réel faible.** PSI et KS représentent une quarantaine de lignes de numpy.
Le rapport coût/bénéfice de la dépendance ne se justifie que pour la
visualisation, qui n'engage aucune décision.

**Poids de l'image de service.** Evidently tire plotly, pydantic et une longue
traîne transitive. En dépendance principale, elle alourdirait de plusieurs
centaines de Mo un conteneur d'API où ce code ne s'exécute jamais.

## Conséquences

- On perd les tests statistiques avancés d'Evidently (Wasserstein, Jensen-Shannon,
  drift sur embeddings) et sa sélection automatique de méthode selon le type de
  colonne et la taille d'échantillon. Non nécessaires ici : toutes les colonnes
  surveillées sont numériques et de taille comparable.
- Les seuils PSI retenus (0,1 / 0,25) sont des conventions issues du secteur
  bancaire, pas des valeurs dérivées de ce jeu de données. Elles sont exposées dans
  `params.yaml` et devraient être calibrées sur des données réelles.
- La statistique KS est calculée sans p-value : seul l'écart maximal entre CDF est
  reporté, à titre indicatif. Le verdict repose exclusivement sur le PSI. Ajouter
  une p-value demanderait scipy, donc une dépendance de plus dans l'image.
- Deux mécanismes de drift coexistent et ne doivent pas être confondus : ce module
  couvre le drift de covariables ; le drift de concept est couverte par
  `scripts/monitor_champion.py` (voir ADR-002).

## Révision

À reconsidérer si le projet doit surveiller des types de données que ce module ne
couvre pas (texte, embeddings, séries temporelles), ou si une plateforme de
monitoring hébergée est adoptée — auquel cas la question devient celle du format
d'export, pas celle de la bibliothèque de calcul.
