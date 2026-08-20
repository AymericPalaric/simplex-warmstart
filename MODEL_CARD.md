# Model card — simplex-warmstart

## Usage prévu

Prédire la réponse d'un mélange ternaire à partir de sa composition et des
descripteurs de ses trois composants, afin de proposer une zone prometteuse du
simplexe **avant** la première campagne expérimentale.

**Hors périmètre :** toute décision de formulation prise sans validation
expérimentale. La sortie est une aide au cadrage, pas une mesure.

## Données

Entièrement synthétiques (`src/simplex_warmstart/simulate.py`). Vérité terrain :
polynôme de Scheffé quadratique dont les coefficients dépendent des descripteurs.
Bruit de mesure gaussien, σ = 0,15. Manifeste des lots dans `params.yaml`.

## Modèle

MLP (2 × 64, SiLU), 9 features, sélection sur la meilleure époque de validation.
Augmentation par les 6 permutations des composants ; l'inférence moyenne sur ces
permutations, rendant l'invariance **exacte**.

## Métriques

Découpage **par étude** — jamais par point. Seuils courants dans `params.yaml`,
section `gates`. Chiffres à jour dans `metrics/eval.json`.

| Périmètre | RMSE | R² |
| --- | --- | --- |
| Test global | … | … |
| Pire famille | … | … |

## Invariants garantis

- Invariance par permutation des composants (testée sur l'artefact **et** à
  travers l'API)
- Déterminisme des prédictions
- Refus des compositions ne sommant pas à 1 ou négatives, au bord du service

## Limites connues

- Extrapolation médiocre hors du domaine de descripteurs vu à l'entraînement
  (démontré dans l'acte 1 de `make demo`)
- Aucune robustesse à un changement de protocole de mesure : c'est un drift de
  concept, seul un retrain la corrige (acte 2)
- Pas d'estimation d'incertitude : la zone proposée n'a pas d'intervalle de confiance

## Conditions de re-validation

Retrain et repasser le seuil de qualité si : une nouvelle famille chimique
apparaît, le protocole de mesure change, ou l'écart de RMSE entre la pire famille
et la moyenne dépasse le seuil `max_family_rmse_gap`.
