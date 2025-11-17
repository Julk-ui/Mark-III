#!/usr/bin/env python3
# models/base_model.py
"""
Define la interfaz base para todos los modelos de predicción.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

class BaseModel(ABC):
    """Clase abstracta para los modelos."""

    def __init__(self, params: dict, logger):
        self.params = params
        self.logger = logger
        self.model = None  # objeto interno del modelo (ARIMA, Prophet, LSTM, etc.)
        self._is_fitted = False
    
    @abstractmethod
    def train_and_predict(
        self,
        y_train: pd.Series,
        X_train: pd.DataFrame | None = None,
        X_test: pd.DataFrame | None = None
    ) -> list:
        """Entrena el modelo y devuelve predicciones para X_test."""
        raise NotImplementedError

    # === Interface de persistencia ===
    def save_model(self, path: str | Path) -> None:
        """
        Guarda el modelo entrenado en disco.
        Cada subclase debe sobreescribir este método.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} no implementa save_model()."
        )

    def load_model(self, path: str | Path) -> None:
        """
        Carga el modelo desde disco.
        Cada subclase debe sobreescribir este método.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} no implementa load_model()."
        )

    def train_and_save(
        self,
        y_train: pd.Series,
        X_train: pd.DataFrame | None,
        model_name: str,
        models_dir: str | Path | None = None
    ) -> None:
        """
        Entrena el modelo con todos los datos y lo guarda en disco.
        Cada subclase debe sobreescribir este método.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} no implementa train_and_save()."
        )

    def predict_loaded(self, X_all: pd.DataFrame) -> list[float]:
        """
        Usa un modelo ya cargado (load_model) para generar predicciones.
        Producción llamará siempre a este método.
        """
        raise NotImplementedError