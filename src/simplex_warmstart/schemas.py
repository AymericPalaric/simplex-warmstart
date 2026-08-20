from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from .chemistry import N_COMPONENTS, PROTOCOLS
from .simulate import FAMILIES

COMPOSITION_COLS = [f"x{i + 1}" for i in range(N_COMPONENTS)]
COMPOSITION_TOL = 1e-9


class ExperimentSchema(pa.DataFrameModel):
    """Une ligne = un essai dans une étude"""

    study_id: Series[str] = pa.Field(nullable=False)
    batch_id: Series[int] = pa.Field(ge=0)
    family: Series[str] = pa.Field(isin=list(FAMILIES))
    protocol: Series[str] = pa.Field(isin=list(PROTOCOLS))

    # Compositions
    x1: Series[float] = pa.Field(ge=0.0, le=1.0)
    x2: Series[float] = pa.Field(ge=0.0, le=1.0)
    x3: Series[float] = pa.Field(ge=0.0, le=1.0)

    # Descripteurs
    descriptors: Series[float] = pa.Field(alias=r"c\d+_d\d+", regex=True)

    y: Series[float] = pa.Field(nullable=False)
    y_true: Series[float] = pa.Field(nullable=False)

    class Config:
        strict = True
        coerce = True

    @pa.dataframe_check
    def compositions_sum_to_one(cls, df: pd.DataFrame) -> Series[bool]:
        total = df[COMPOSITION_COLS].sum(axis=1)
        return pd.Series(np.isclose(total, 1.0, atol=COMPOSITION_TOL), index=df.index)

    @pa.check
    def is_finite(cls, series: Series[float]) -> Series[bool]:
        return pd.Series(np.isfinite(series), index=series.index)


def validate_experiments(df: pd.DataFrame) -> pd.DataFrame:
    """Valide un DataFrame d'expériences"""
    return ExperimentSchema.validate(df, lazy=True)
