UV ?= uv

.PHONY: help setup pipeline test demo serve clean

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Installe l'environnement
	$(UV) sync --locked --extra monitoring
	$(UV) run pre-commit install

pipeline:  ## Rejoue le pipeline complet
	$(UV) run dvc repro

test:  ## Lance toute la suite de tests
	$(UV) run pytest -m "not artifact"
	$(UV) run pytest -m artifact

demo:  ## Rejoue l'incident de dérive de bout en bout
	$(UV) run python scripts/demo.py

serve:  ## Démarre l'API sur les artefacts locaux
	MODEL_URI=local $(UV) run uvicorn simplex_warmstart.serving.app:app --reload

clean:  ## Supprime les artefacts locaux
	rm -rf data models metrics reports demo .params.yaml.demo-backup
