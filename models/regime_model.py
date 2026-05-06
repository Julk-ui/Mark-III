from __future__ import annotations

"""Aprendizaje no supervisado simple para detectar regímenes de mercado."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class RegimeResult:
    features: pd.DataFrame
    regime_column: str


class MarketRegimeClusterer:
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("kmeans", KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)),
            ]
        )
        self._is_fitted = False
        self.regime_col = "market_regime"

    def _candidate_columns(self, df: pd.DataFrame) -> list[str]:
        preferred = [
            "Return_1", "ReturnFwd_1", "Return_5", "Return_10",
            "ATR_14", "RSI_14", "MACD", "MACD_Hist", "BB_Width",
            "Close_lag1", "Volume_lag1",
        ]
        cols = [c for c in preferred if c in df.columns]
        if len(cols) >= 3:
            return cols
        numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in {"Open", "High", "Low", "Close", "Volume"}]
        return numeric[:8]

    def fit_transform(self, df: pd.DataFrame, columns: Iterable[str] | None = None) -> RegimeResult:
        work = df.copy()
        cols = list(columns) if columns is not None else self._candidate_columns(work)
        if len(cols) < 2:
            work[self.regime_col] = 0
            return RegimeResult(features=work, regime_column=self.regime_col)

        labels = self.pipeline.fit_predict(work[cols])
        work[self.regime_col] = labels.astype(int)
        for regime_value in sorted(pd.Series(labels).unique()):
            work[f"regime_{regime_value}"] = (work[self.regime_col] == regime_value).astype(int)

        self._is_fitted = True
        return RegimeResult(features=work, regime_column=self.regime_col)

    def transform(self, df: pd.DataFrame, columns: Iterable[str] | None = None) -> RegimeResult:
        if not self._is_fitted:
            raise RuntimeError("El clusterer no ha sido entrenado.")
        work = df.copy()
        cols = list(columns) if columns is not None else self._candidate_columns(work)
        labels = self.pipeline.predict(work[cols])
        work[self.regime_col] = labels.astype(int)
        for regime_value in sorted(pd.Series(labels).unique()):
            work[f"regime_{regime_value}"] = (work[self.regime_col] == regime_value).astype(int)
        return RegimeResult(features=work, regime_column=self.regime_col)
