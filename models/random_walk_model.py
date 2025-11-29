#!/usr/bin/env python3
# models/random_walk_model.py
"""
Modelos baseline:

- RandomWalkModel:
    Modelo ingenuo tipo "random walk": asume que el próximo valor del target
    será igual al último valor observado (P_{t+1} = P_t o R_{t+1} = R_t).

- MomentumModel:
    Segundo baseline muy sencillo, actualmente igual que RandomWalk pero
    lo puedes adaptar si quieres otro tipo de benchmark.
"""

from __future__ import annotations
import pandas as pd
from .base_model import BaseModel


class RandomWalkModel(BaseModel):
    """Modelo ingenuo: predice el último valor conocido de la serie objetivo."""

    def train_and_predict(
        self,
        y_train: pd.Series,
        X_train: pd.DataFrame | None = None,
        X_test: pd.DataFrame | None = None,
    ) -> list:
        # Si no hay datos de test, no hay predicciones
        if X_test is None or len(X_test) == 0:
            return []

        if y_train is None or y_train.empty:
            self.logger.warning(
                "RandomWalkModel: y_train vacío. Prediciendo 0 para toda la ventana de test."
            )
            return [0.0] * len(X_test)

        # Baseline: usar el último valor observado del target
        last_value = y_train.iloc[-1]
        self.logger.info(
            "RandomWalkModel: usando el último valor conocido como baseline (valor = %s).",
            last_value,
        )
        return [last_value] * len(X_test)


class MomentumModel(BaseModel):
    """
    Modelo de referencia basado en 'momentum' muy simple.
    Por ahora hace lo mismo que RandomWalk (último valor), pero
    puedes modificarlo si quieres otro comportamiento.
    """

    def train_and_predict(
        self,
        y_train: pd.Series,
        X_train: pd.DataFrame | None = None,
        X_test: pd.DataFrame | None = None,
    ) -> list:
        if X_test is None or len(X_test) == 0:
            return []

        if y_train is None or y_train.empty:
            self.logger.warning(
                "MomentumModel: y_train vacío. Prediciendo 0 para toda la ventana de test."
            )
            return [0.0] * len(X_test)

        last_value = y_train.iloc[-1]
        self.logger.info(
            "MomentumModel: usando el último valor de la serie como proxy de momentum (valor = %s).",
            last_value,
        )
        return [last_value] * len(X_test)
