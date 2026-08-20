"""API d'inférence"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from .. import exploration
from ..chemistry import N_COMPONENTS, N_DESCRIPTORS
from ..features import COMPOSITION_COLS, FEATURE_COLS
from .loader import LoadedModel, load_model

COMPOSITION_TOL = 1e-6
MIN_SAMPLES = 100
MAX_SAMPLES = 50_000
state: dict[str, LoadedModel] = {}


def check_descriptors(rows: list[list[float]]):
    if any(len(row) != N_DESCRIPTORS for row in rows):
        raise ValueError(f"Each component has exactly {N_DESCRIPTORS} descriptors")


class Formulation(BaseModel):
    composition: list[float] = Field(min_length=N_COMPONENTS, max_length=N_COMPONENTS)
    descriptors: list[list[float]] = Field(min_length=N_COMPONENTS, max_length=N_COMPONENTS)

    @model_validator(mode="after")
    def check_invariants(self) -> Formulation:
        if any(value < 0 for value in self.composition):
            raise ValueError("All fractions must be >= 0")
        if abs(sum(self.composition) - 1.0) > COMPOSITION_TOL:
            raise ValueError(f"Fractions must sum to 1. Got {sum(self.composition)}")
        check_descriptors(self.descriptors)
        return self

    def to_row(self) -> list[float]:
        return list(self.composition) + [v for row in self.descriptors for v in row]


class PredictRequest(BaseModel):
    formulations: list[Formulation] = Field(min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    predictions: list[float]
    model_version: str | None
    model_source: str


class RegionRequest(BaseModel):
    """Descripteurs des 3 composants d'une étude : on cherche les bonnes compositions"""

    descriptors: list[list[float]] = Field(min_length=N_COMPONENTS, max_length=N_COMPONENTS)
    n_samples: int = Field(default=4000, ge=MIN_SAMPLES, le=MAX_SAMPLES)
    top_fraction: float = Field(default=0.05, gt=0.0, le=0.5)
    objective: Literal["maximize", "minimize"] = "maximize"
    seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def check_invariants(self) -> RegionRequest:
        check_descriptors(self.descriptors)
        return self


class Candidate(BaseModel):
    composition: list[float]
    prediction: float


class Bounds(BaseModel):
    min: float
    max: float


class Summary(BaseModel):
    min: float
    mean: float
    max: float


class RegionResponse(BaseModel):
    """Enveloppe convexe de la région retenue, plus de quoi la situer"""

    best: Candidate
    centroid: list[float]
    envelope: list[list[float]]
    bounds: dict[str, Bounds]
    threshold: float
    region_size: int
    n_samples: int
    objective: str
    seed: int
    region_summary: Summary
    model_version: str | None
    model_source: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Un load par session
    state["model"] = load_model()
    yield
    state.clear()


app = FastAPI(title="simplex-warmstart", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, object]:
    model = state.get("model")
    return {
        "status": "ok" if model is not None else "degraded",
        "model_version": model.version if model else None,
        "model_source": model.source if model else None,
    }


@app.get("/model")
def model_info() -> dict[str, object]:
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="No loaded model")
    return {
        "version": model.version,
        "source": model.source,
        "features": FEATURE_COLS,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="No loaded model")

    df = pd.DataFrame([item.to_row() for item in request.formulations], columns=FEATURE_COLS)
    values = model.predict(df)
    if not np.isfinite(values).all():
        raise HTTPException(status_code=500, detail="Non-finite prediction")
    return PredictResponse(
        predictions=[float(v) for v in values],
        model_version=model.version,
        model_source=model.source,
    )


@app.post("/suggest-region", response_model=RegionResponse)
def suggest_region(request: RegionRequest) -> RegionResponse:
    """Échantillonne le simplexe et renvoie l'enveloppe des meilleures compositions"""
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="No loaded model")

    suggestion = exploration.suggest_region(
        model.predict,
        np.asarray(request.descriptors, dtype=float),
        n_samples=request.n_samples,
        top_fraction=request.top_fraction,
        objective=request.objective,
        seed=request.seed,
    )
    if not np.isfinite(suggestion.predictions).all():
        raise HTTPException(status_code=500, detail="Non-finite prediction")

    region = suggestion.region_predictions
    return RegionResponse(
        best=Candidate(
            composition=suggestion.best_composition.tolist(),
            prediction=suggestion.best_prediction,
        ),
        centroid=suggestion.centroid.tolist(),
        envelope=suggestion.envelope.tolist(),
        bounds={
            column: Bounds(min=float(low), max=float(high))
            for column, (low, high) in zip(COMPOSITION_COLS, suggestion.bounds, strict=True)
        },
        threshold=suggestion.threshold,
        region_size=len(suggestion.top_index),
        n_samples=request.n_samples,
        objective=request.objective,
        seed=request.seed,
        region_summary=Summary(
            min=float(region.min()), mean=float(region.mean()), max=float(region.max())
        ),
        model_version=model.version,
        model_source=model.source,
    )
