# ADR-001 — `params.yaml` plutôt que Hydra pour la configuration

**Statut :** accepté · **Date :** 20/08/2026

## Contexte

Le pipeline a besoin d'un mécanisme de configuration pour les hyperparamètres, le
manifeste de données, les fractions de découpage et les seuils du contrôle qualité.

Hydra est le standard de fait dans les projets de deep learning : composition de
configurations, surcharge en ligne de commande, mode `multirun` pour les balayages
d'hyperparamètres. DVC, de son côté, dispose de son propre mécanisme : un fichier
`params.yaml` dont chaque étape déclare les blocs qui la concernent.

## Décision

La configuration vit dans un unique `params.yaml`, lu par un chargeur trivial
(`src/simplex_warmstart/params.py`). Hydra n'est pas utilisé.

## Justification

DVC invalide une étape **au niveau du bloc de paramètres déclaré**. Une étape qui
déclare `params: [train]` est rejouée si et seulement si ce bloc a changé ;
modifier `gates.max_test_rmse` ne relance pas l'entraînement. Cette granularité
est ce qui rend `dvc repro` utilisable au quotidien plutôt que pénible.

Hydra ne s'y intègre pas : il compose sa configuration à l'exécution, souvent à
partir de plusieurs fichiers et de surcharges CLI. DVC ne peut alors ni hacher la
configuration effective, ni savoir quelle étape invalider. Faire cohabiter les
deux impose soit de dupliquer les valeurs (deux sources de vérité, qui divergent),
soit de renoncer à l'invalidation fine (le pipeline se rejoue entièrement à chaque
modification).

Entre les deux, la reproductibilité prime sur l'ergonomie de configuration : c'est
l'objet même du projet.

Le besoin qui aurait justifié Hydra — le balayage d'hyperparamètres — n'existe pas
ici. Le modèle est une baseline volontairement simple, et `dvc exp run --set-param`
couvre le cas ponctuel.

## Conséquences

- Pas de composition de configurations ni de `multirun` natif. Un balayage optimal
  demanderait soit `dvc exp run` en boucle, soit l'introduction d'Hydra avec une
  révision de cette décision.
- La configuration est flat et lisible d'un coup d'œil, ce qui rend les diffs de
  pull request de retrain immédiatement lisibles (voir ADR-004 sur
  le manifeste de données).
- Le loader maison ne valide rien. Si le fichier grossit, `pydantic-settings`
  apporterait la validation de schéma sans rompre l'intégration DVC — c'est la
  première évolution à envisager, avant Hydra.

## Révision

À reconsidérer si le projet passe à des entraînements coûteux nécessitant des
balayages structurés, ou si plusieurs variantes d'architecture doivent coexister.
