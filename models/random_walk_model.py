#!/usr/bin/env python3
# models/random_walk_model.py
"""
Implementación del modelo Random Walk (línea base).
Implementación de un modelo de Momentum de 1 día (línea base).
Este modelo asume que el retorno de mañana será igual al de hoy.
"""
from __future__ import annotations
import pandas as pd
from .base_model import BaseModel

class RandomWalkModel(BaseModel):
    """La predicción es el último valor conocido."""
class MomentumModel(BaseModel):
    """
    Modelo de referencia basado en Momentum.
    La predicción para el siguiente paso es el último valor conocido de la serie (y_train).
    Para retornos, esto significa predecir que el retorno de mañana será el mismo que el de hoy.
    """

    def train_and_predict(self, y_train: pd.Series, X_train: pd.DataFrame | None = None, X_test: pd.DataFrame | None = None) -> list:
        """La predicción para el siguiente paso es el último valor de y_train."""
        # --- MEJORA ---
        # Un modelo de referencia más robusto para retornos asume que el retorno esperado es 0.
        # Esto establece una línea base de hit_rate del 50% (azar).
        # El modelo anterior (usar el último retorno) mide la autocorrelación del mercado,
        # no la habilidad predictiva nula.
        self.logger.info("RandomWalkModel: Prediciendo un retorno de 0 como línea base.")
        return [0.0] * len(X_test)
        if y_train.empty:
            self.logger.warning("MomentumModel: y_train está vacío. Prediciendo 0.")
            return [0.0] * len(X_test)
        
        last_known_value = y_train.iloc[-1]
        return [last_known_value] * len(X_test)
