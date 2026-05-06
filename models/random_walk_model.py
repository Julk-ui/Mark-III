from __future__ import annotations

from typing import Optional
from pathlib import Path
import joblib

import pandas as pd

from models.base_model import BaseModel


class RandomWalkModel(BaseModel):
    """Baseline simple: predice retorno 0.0 para todos los puntos.

    Útil como benchmark "no signal" (equivalente a asumir retorno esperado 0).
    """

    def train_and_predict(
        self,
        y_train: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None,
    ) -> list[float]:
        n = int(len(X_test)) if X_test is not None else 1
        self.logger.info("RandomWalkModel: Prediciendo 0.0 (baseline).")
        return [0.0] * n

    def train_and_save(
        self,
        y_train: pd.Series,
        X_train: Optional[pd.DataFrame],
        model_name: str,
        models_dir: Path,
    ) -> Path:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model_class": self.__class__.__name__,
            "params": dict(self.params or {}),
        }
        model_path = models_dir / f"{model_name}.pkl"
        joblib.dump(artifact, model_path)
        self._is_fitted = True
        return model_path

    def save_model(self, path: str | Path) -> None:
        artifact = {
            "model_class": self.__class__.__name__,
            "params": dict(self.params or {}),
        }
        joblib.dump(artifact, Path(path))

    def load_model(self, path: str | Path) -> None:
        artifact = joblib.load(Path(path))
        if isinstance(artifact, dict):
            self.params = artifact.get("params", self.params)
        self._is_fitted = True

    def predict_loaded(self, X_all: pd.DataFrame) -> list[float]:
        return [0.0]


class MomentumModel(BaseModel):
    """Baseline momentum: predice que el próximo retorno será igual al último retorno observado.

    Esto sí genera predicción distinta de cero cuando el último retorno no es 0.
    Funciona bien como baseline para ReturnFwd_1.
    """

    def train_and_predict(
        self,
        y_train: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None,
    ) -> list[float]:
        lookback = int(self.params.get("lookback", 1) or 1)
        if lookback <= 0:
            lookback = 1

        if y_train is None or len(y_train) == 0:
            pred = 0.0
        else:
            pred = float(pd.Series(y_train).tail(lookback).mean())

        if self.logger:
            self.logger.info(
                f"MomentumModel: Prediciendo retorno momentum={pred:.6f} (lookback={lookback})."
            )

        n = int(len(X_test)) if X_test is not None else 1
        return [pred] * n

        """
        Momentum baseline:
        - Predice el promedio de los últimos N retornos (lookback).
        - Si no hay datos, predice 0.0
        """
        lookback = int(self.params.get("lookback", 1))
        if lookback <= 0:
            lookback = 1

        if y_train is None or len(y_train) == 0:
            pred = 0.0
        else:
            # promedio de los últimos N retornos
            pred = float(y_train.tail(lookback).mean())

        if self.logger:
            self.logger.info(f"MomentumModel: Prediciendo retorno momentum={pred:.6f} (lookback={lookback}).")

        return [pred] * len(X_test)

    def train_and_save(
        self,
        y_train: pd.Series,
        X_train: Optional[pd.DataFrame],
        model_name: str,
        models_dir: Path,
    ) -> Path:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model_class": self.__class__.__name__,
            "params": dict(self.params or {}),
        }
        model_path = models_dir / f"{model_name}.pkl"
        joblib.dump(artifact, model_path)
        self._is_fitted = True
        return model_path

    def save_model(self, path: str | Path) -> None:
        artifact = {
            "model_class": self.__class__.__name__,
            "params": dict(self.params or {}),
        }
        joblib.dump(artifact, Path(path))

    def load_model(self, path: str | Path) -> None:
        artifact = joblib.load(Path(path))
        if isinstance(artifact, dict):
            self.params = artifact.get("params", self.params)
        self._is_fitted = True

    def predict_loaded(self, X_all: pd.DataFrame) -> list[float]:
        if X_all is None or X_all.empty:
            return [0.0]

        lookback = int(self.params.get("lookback", 1) or 1)
        if lookback <= 0:
            lookback = 1

        if "Return_1" in X_all.columns:
            source = pd.to_numeric(X_all["Return_1"], errors="coerce")
        elif "Return1" in X_all.columns:
            source = pd.to_numeric(X_all["Return1"], errors="coerce")
        else:
            return [0.0]

        source = source.dropna()
        if source.empty:
            return [0.0]

        pred = float(source.tail(lookback).mean())
        return [pred]
