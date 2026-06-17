from __future__ import annotations

"""Modelos clasificadores para targets de barrera/evento.

Conservan compatibilidad con el pipeline actual devolviendo un retorno esperado
como salida principal, pero además exponen probabilidades auditables:
- prob_up
- prob_hold
- prob_down
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .base_model import BaseModel


def _coerce_classifier_params(params: dict[str, Any] | None) -> dict[str, Any]:
    params = dict(params or {})
    int_keys = {
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "min_samples_split",
        "max_leaf_nodes",
        "random_state",
        "max_iter",
    }
    for key in int_keys:
        if key in params and params[key] is not None:
            try:
                params[key] = int(float(params[key]))
            except Exception:
                pass
    return params


class _SklearnClassifierBase(BaseModel):
    estimator_cls = None
    default_params: dict[str, Any] = {}

    def __init__(self, params: dict, logger):
        super().__init__(params=params, logger=logger)
        self.feature_names: list[str] = []
        self.class_values: list[float] = []

    def _build_pipeline(self):
        if self.estimator_cls is None:
            raise NotImplementedError("estimator_cls no está definido.")

        params = _coerce_classifier_params({**self.default_params, **(self.params or {})})
        estimator = self.estimator_cls(**params)
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

    def _prepare_target(self, y: pd.Series) -> tuple[pd.Series, dict[int, float]]:
        """
        Convierte un target continuo de barrera a clases discretas {-1, 0, +1}
        y guarda un valor esperado medio por clase para reconstruir retornos.
        """
        y_sr = pd.Series(y).astype(float)
        classes = pd.Series(
            np.where(
                np.isclose(y_sr, 0.0),
                0,
                np.where(y_sr > 0.0, 1, -1),
            ),
            index=y_sr.index,
            dtype=int,
        )

        class_values: dict[int, float] = {}
        for label in (-1, 0, 1):
            mask = classes == label
            if mask.any():
                class_values[label] = float(y_sr.loc[mask].mean())
            else:
                class_values[label] = float(label)
        return classes, class_values

    def _build_prediction_payload(self, proba: np.ndarray, class_values: np.ndarray) -> dict[str, list[float]]:
        class_values = np.asarray(class_values, dtype=float).reshape(-1)
        proba = np.asarray(proba, dtype=float)
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)

        expected_returns = (proba * class_values.reshape(1, -1)).sum(axis=1)

        up_mask = class_values > 0
        down_mask = class_values < 0
        hold_mask = np.isclose(class_values, 0.0)

        prob_up = proba[:, up_mask].sum(axis=1) if np.any(up_mask) else np.zeros(len(proba))
        prob_down = proba[:, down_mask].sum(axis=1) if np.any(down_mask) else np.zeros(len(proba))
        prob_hold = proba[:, hold_mask].sum(axis=1) if np.any(hold_mask) else np.zeros(len(proba))

        return {
            "predictions": [float(x) for x in expected_returns.tolist()],
            "prob_up": [float(x) for x in prob_up.tolist()],
            "prob_hold": [float(x) for x in prob_hold.tolist()],
            "prob_down": [float(x) for x in prob_down.tolist()],
        }

    def train_and_predict_details(
        self,
        y_train: pd.Series,
        X_train: pd.DataFrame | None = None,
        X_test: pd.DataFrame | None = None,
    ) -> dict[str, list[float]]:
        if X_train is None or X_test is None:
            n = len(X_test) if X_test is not None else 1
            return {
                "predictions": [0.0] * n,
                "prob_up": [0.0] * n,
                "prob_hold": [1.0] * n,
                "prob_down": [0.0] * n,
            }

        X_train_df = self._prepare_X(X_train, fit_mode=True)
        X_test_df = self._prepare_X(X_test, fit_mode=False)
        y_train_sr = pd.Series(y_train).astype(float)
        y_train_cls, class_values_map = self._prepare_target(y_train_sr)

        model = self._build_pipeline()
        model.fit(X_train_df, y_train_cls)
        self._is_fitted = True

        estimator = model.named_steps["model"]
        class_labels = np.asarray(estimator.classes_, dtype=int)
        class_values = np.asarray(
            [class_values_map.get(int(label), float(label)) for label in class_labels],
            dtype=float,
        )
        self.class_values = [float(x) for x in class_values.tolist()]
        proba = model.predict_proba(X_test_df)
        return self._build_prediction_payload(proba, class_values)

    def train_and_predict(
        self,
        y_train: pd.Series,
        X_train: pd.DataFrame | None = None,
        X_test: pd.DataFrame | None = None,
    ) -> list[float]:
        payload = self.train_and_predict_details(y_train, X_train, X_test)
        return payload.get("predictions", [])

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
        y_train_cls, class_values_map = self._prepare_target(y_train_sr)

        model = self._build_pipeline()
        model.fit(X_train_df, y_train_cls)
        self._is_fitted = True

        estimator = model.named_steps["model"]
        class_labels = np.asarray(estimator.classes_, dtype=int)
        self.class_values = [
            float(class_values_map.get(int(label), float(label)))
            for label in class_labels.tolist()
        ]

        model_path = models_dir / f"{model_name}.pkl"
        artifact = {
            "model": model,
            "params": _coerce_classifier_params(self.params),
            "feature_names": self.feature_names,
            "class_values": self.class_values,
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
            "params": _coerce_classifier_params(self.params),
            "feature_names": self.feature_names,
            "class_values": self.class_values,
            "model_class": self.__class__.__name__,
        }
        joblib.dump(artifact, Path(path))

    def load_model(self, path: str | Path) -> None:
        artifact = joblib.load(Path(path))
        if isinstance(artifact, dict):
            self.model = artifact.get("model")
            self.params = artifact.get("params", self.params)
            self.feature_names = list(artifact.get("feature_names", []))
            self.class_values = [float(x) for x in artifact.get("class_values", [])]
        else:
            self.model = artifact
            self.feature_names = []
            self.class_values = []

        self.params = _coerce_classifier_params(self.params)
        self._is_fitted = True

    def predict_loaded_details(self, X_all: pd.DataFrame) -> dict[str, list[float]]:
        if not self._is_fitted or self.model is None:
            raise RuntimeError(f"{self.__class__.__name__}: el modelo no está cargado.")

        if X_all is None or len(X_all) == 0:
            return {
                "predictions": [],
                "prob_up": [],
                "prob_hold": [],
                "prob_down": [],
            }

        X_last = self._prepare_X(pd.DataFrame(X_all).tail(1), fit_mode=False)
        estimator = self.model.named_steps["model"]
        class_values = np.asarray(
            self.class_values if self.class_values else getattr(estimator, "classes_", []),
            dtype=float,
        )
        proba = self.model.predict_proba(X_last)
        return self._build_prediction_payload(proba, class_values)

    def predict_loaded(self, X_all: pd.DataFrame) -> list[float]:
        payload = self.predict_loaded_details(X_all)
        return payload.get("predictions", [])


class LogisticRegressionClassifierModel(_SklearnClassifierBase):
    estimator_cls = LogisticRegression
    default_params = {
        "max_iter": 500,
        "solver": "lbfgs",
        "multi_class": "auto",
        "class_weight": "balanced",
        "random_state": 42,
    }


class RandomForestClassifierModel(_SklearnClassifierBase):
    estimator_cls = RandomForestClassifier
    default_params = {
        "n_estimators": 200,
        "max_depth": 6,
        "min_samples_leaf": 10,
        "random_state": 42,
        "n_jobs": 1,
        "class_weight": "balanced_subsample",
    }


class ExtraTreesClassifierModel(_SklearnClassifierBase):
    estimator_cls = ExtraTreesClassifier
    default_params = {
        "n_estimators": 200,
        "max_depth": 6,
        "min_samples_leaf": 10,
        "random_state": 42,
        "n_jobs": 1,
        "class_weight": "balanced_subsample",
    }


class HistGradientBoostingClassifierModel(_SklearnClassifierBase):
    estimator_cls = HistGradientBoostingClassifier
    default_params = {
        "learning_rate": 0.05,
        "max_depth": 6,
        "max_iter": 200,
        "min_samples_leaf": 20,
        "random_state": 42,
    }
