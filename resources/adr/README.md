# Décisions d'architecture

Ce dossier consigne les décisions structurantes du projet et les
raisons qui les ont motivées.

## Index

| N° | Décision | Statut |
| --- | --- | --- |
| [001](0001-params-yaml-over-hydra.md) | `params.yaml` plutôt que Hydra pour la configuration | accepté |
| [002](0002-non-blocking-gate-on-drift.md) | Le seuil de qualité ne bloque pas sur le drift | accepté |
| [003](0003-drift-detection.md) | Détection de drift maison, Evidently en visualisation | accepté |
| [004](0004-registry-not-in-dvc.md) | L'enregistrement au registry n'est pas une étape DVC | accepté |

## Format

Chaque ADR tient sur une page et suit la même structure :

- **Contexte** — la situation et les options en présence, sans conclusion
- **Décision** — ce qui a été retenu, en une ou deux phrases
- **Justification** — pourquoi, y compris les alternatives écartées
- **Conséquences** — ce que la décision coûte, pas seulement ce qu'elle apporte
- **Révision** — les conditions dans lesquelles il faudrait la rouvrir
