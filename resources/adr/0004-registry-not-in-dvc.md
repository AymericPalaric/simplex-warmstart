# ADR-004 — L'enregistrement au registry n'est pas une étape du pipeline DVC

**Statut :** accepté · **Date :** 20/08/2026

## Contexte

Le pipeline DVC compte six étapes, de la génération des données au contrôle
qualité. Trois actions supplémentaires existent : enregistrer une version du
modèle dans le registry MLflow, comparer le challenger au champion et déplacer
l'alias, et revenir à une version antérieure.

La question s'est posée de les intégrer comme étapes du `dvc.yaml`, ce qui aurait
donné un pipeline unique allant de la donnée brute au modèle en service.

## Décision

Ces trois actions vivent dans `scripts/` (`register_model.py`, `promote.py`,
`rollback.py`) et sont invoquées explicitement. Le `dvc.yaml` s'arrête au contrôle
qualité.

## Justification

**Une étape DVC doit être une fonction pure de ses dépendances.** Mêmes entrées,
mêmes sorties, autant de fois qu'on la rejoue. C'est ce qui permet à `dvc repro`
de restaurer une sortie depuis le cache plutôt que de la recalculer, et c'est ce
qui donne un sens au hachage dans `dvc.lock`.

L'enregistrement viole cette propriété par construction : il modifie un état
externe et incrémente un compteur de versions. Rejouer l'étape ne redonne pas le
même résultat, il en crée un nouveau. Un `dvc repro` un peu trop enthousiaste
produirait une dizaine de versions fantômes dans le registry.

**Les deux systèmes répondent à des questions différentes.** DVC répond à « ce
modèle vient de quelles données et de quel code, et comment le refaire ». Le
registry répond à « quelle version sert en production, qui l'a décidé, et quand ».
Les mélanger dans un même graphe brouille cette frontière au lieu de la rendre
lisible.

**La promotion est une décision, pas un calcul.** L'intégrer au pipeline reviendrait
à promouvoir un modèle à chaque `dvc repro` local, y compris lors d'un essai
exploratoire. La séparation rend l'action délibérée.

Corollaire cohérent : le manifeste de données vit dans `params.yaml`, donc dans
git, et non dans un état externe. L'historique du jeu de données est visible dans
`git log`, et chaque pull request de ré-entraînement se lit comme un diff de trois
lignes.

## Conséquences

- Il n'existe pas de commande unique menant de la donnée brute au modèle promu.
  C'est assumé : `make demo` enchaîne les étapes pour la démonstration, mais
  l'enchaînement reste explicite.
- `scripts/register_model.py` refuse d'enregistrer si `metrics/gate.json` indique
  un échec. Le contrôle qualité reste donc bloquant, bien qu'il soit dans le
  pipeline et l'enregistrement à l'extérieur.
- Les scripts dépendent d'artefacts produits par le pipeline (`models/model.pt`,
  `models/metadata.json`, `metrics/gate.json`) sans que DVC connaisse ce lien.
  Lancer `promote.py` sur des artefacts périmés est possible ; la parade est le
  `run_id` MLflow inscrit dans `metadata.json`, qui permet de remonter à la source.
- La comparaison champion/challenger recharge les deux modèles et les réévalue sur
  le golden set, plutôt que de comparer des métriques déjà enregistrées. Plus lent,
  mais c'est la seule façon de comparer deux artefacts sur exactement la même base
  quand le pipeline a évolué entre-temps.

## Révision

À reconsidérer si un orchestrateur (Airflow, Prefect, Dagster) est introduit :
ces outils, contrairement à DVC, admettent nativement des tâches à effet de bord
et fourniraient un graphe unique sans sacrifier la reproductibilité des étapes de
calcul.
