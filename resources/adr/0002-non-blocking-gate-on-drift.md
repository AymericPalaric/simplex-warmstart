# ADR-002 — Le seuil de qualité ne bloque pas sur le drift

**Statut :** accepté · **Date :** 20/08/2026

## Contexte

Le pipeline détecte le drift de distribution entre le lot le plus récent et les
lots antérieurs (PSI, KS). La question s'est posée d'en faire un critère bloquant
du seuil de qualité, au même titre que la RMSE.

## Décision

Le drift n'entre pas dans le seuil de qualité. Elle est signalée de façon
visible dans le rapport de pull request (bannière, tableau par colonne, nouvelles
modalités), mais ne refuse aucun modèle.

## Justification

Une dérive de distribution est attendue dès qu'un système apprend de données
nouvelles : c'est le fonctionnement normal, pas une anomalie. Bloquer dessus
reviendrait à interdire l'intégration de toute donnée inédite — précisément ce
que le système est censé rendre possible.

Ce qui doit bloquer est la **performance mesurée** : RMSE, R², et écart entre la
pire famille et la moyenne. Formule retenue : *le drift alerte, la métrique
tranche*.

## Conséquences

- Un lot fortement drifté peut être intégré si le modèle retrained tient les
  seuils — c'est le comportement voulu.
- Une dérive de concept n'est pas visible par ce mécanisme ; elle est couverte
  séparément par `scripts/monitor_champion.py`, qui mesure le modèle **déployé**
  sur les données récentes.
- Si l'on voulait un blocage sur le drift, il faudrait un critère de nouveauté
  distinct (part de points hors domaine d'entraînement) plutôt qu'un seuil de PSI.
