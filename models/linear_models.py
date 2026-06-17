from __future__ import annotations

"""Modelos lineales ligeros para trading.

Actualmente expone un regresor CPU-friendly:
- RidgeRegressorModel
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from .base_model import BaseModel


def _coerce_linear_params(params: dict[str, Any] | None) -> dict[str, Any]:
    params = dict(params or {})
    float_keys = {"alpha", "tol"}
    int_keys = {"max_iter", "random_state"}

    for key in float_keys:
        if key in params and params[key] is not None:
            try:
                params[key] = float(params[key])
            except Exception:
                pass

    for key in int_keys:
        if key in params and params[key] is not None:
            try:
                params[key] = int(float(params[key]))
            except Exception:
                pass

    return params


class RidgeRegressorModel(BaseModel):
    default_params: dict[str, Any] = {
        "alpha": 1.0,
        "random_state": 42,
    }

    def __init__(self, params: dict, logger):
        super().__init__(params=params, logger=logger)
        self.feature_names: list[str] = []

    def _build_pipeline(self):
        params = _coerce_linear_params({**self.default_params, **(self.params or {})})
        estimator = Ridge(**params)
        self.model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )
        return self.model

    def _prepare_X(self, X: pd.DataFrame | None, fit_mode: bool = False) -> pd.DataFrame:
        if X is None:
            return pd.DataFrame()

        X_df = pd.DataFrame(X).copy()
        if fit_mode:
            self.feature_names = list(X_df.columns)
            return X_df

        if self.feature_names:
            for col in self.feature_names:
                if col not in X_df.columns:
                    X_df[col] = np.nan
            X_df = X_df[self.feature_names]
        return X_df

    def train_and_predict(
        self,
        y_train: pd.Series,
        X_train: pd.DataFrame | None = None,
        X_test: pd.DataFrame | None = None,
    ) -> list[float]:
        if X_train is None or X_test is None:
            n = len(X_test) if X_test is not None else 1
            return [0.0] * n

        try:
            X_train_df = self._prepare_X(X_train, fit_mode=True)
            X_test_df = self._prepare_X(X_test, fit_mode=False)
            y_train_sr = pd.Series(y_train).astype(float)

            model = self._build_pipeline()
            model.fit(X_train_df, y_train_sr)
            self._is_fitted = True

            preds = model.predict(X_test_df)
            return [float(x) for x in np.asarray(preds).reshape(-1)]
        except Exception as exc:
            self.logger.error(f"{self.__class__.__name__} error: {exc}")
            n = len(X_test) if X_test is not None else 1
            return [0.0] * n

    def train_and_save(
        self,
        y_train: pd.Series,
        X_train: pd.DataFrame | None,
        model_name: str,
        models_dir: str | Path | None = None,
    ):
        if X_train is None:
            raise ValueError(f"{self.__class__.__name__}: X_train es obligatorio para train_and_save.")

        models_dir = Path(models_dir or Path("outputs") / "models")
        models_dir.mkdir(parents=True, exist_ok=True)

        X_train_df = self._prepare_X(X_train, fit_mode=True)
        y_train_sr = pd.Series(y_train).astype(float)

        model = self._build_pipeline()
        model.fit(X_train_df, y_train_sr)
        self._is_fitted = True

        model_path = models_dir / f"{model_name}.pkl"
        artifact = {
            "model": model,
            "params": _coerce_linear_params(self.params),
            "feature_names": self.feature_names,
            "model_class": self.__class__.__name__,
        }
        joblib.dump(artifact, model_path)
        self.logger.info(f"[{self.__class__.__name__}] Modelo guardado en: {model_path}")
        return model_path

    def save_model(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError(f"{self.__class__.__name__}: no hay modelo entrenado en memoria.")

        artifact = {
            "model": self.model,
            "params": _coerce_linear_params(self.params),
            "feature_names": self.feature_names,
            "model_class": self.__class__.__name__,
        }
        joblib.dump(artifact, Path(path))

    def load_model(self, path: str | Path) -> None:
        artifact = joblib.load(Path(path))

        if isinstance(artifact, dict):
            self.model = artifact.get("model")
            self.params = artifact.get("params", self.params)
            self.feature_names = list(artifact.get("feature_names", []))
        else:
            self.model = artifact
            self.feature_names = []

        self.params = _coerce_linear_params(self.params)
        self._is_fitted = True

    def predict_loaded(self, X_all: pd.DataFrame) -> list[float]:
        if not self._is_fitted or self.model is None:
            raise RuntimeError(f"{self.__class__.__name__}: el modelo no está cargado.")

        if X_all is None or len(X_all) == 0:
            return []

        X_last = self._prepare_X(pd.DataFrame(X_all).tail(1), fit_mode=False)
        pred = self.model.predict(X_last)
        return [float(pred[0])]
