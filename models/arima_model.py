#!/usr/bin/env python3
# models/arima_model.py
"""
Implementación del modelo ARIMA.
"""
from __future__ import annotations
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Ridge
import pickle
import joblib
from pathlib import Path
from .base_model import BaseModel


class ArimaModel(BaseModel):
    """Modelo ARIMA para predicción de series temporales."""

    def train_and_predict(self, y_train: pd.Series, X_train: pd.DataFrame | None = None, X_test: pd.DataFrame | None = None) -> list:
        """Entrena un modelo ARIMA y predice."""
        if X_test is None:
            self.logger.warning(
                "ARIMA.train_and_predict fue llamado con X_test=None. "
                "Se entrena internamente pero NO se generan predicciones."
            )
            return []
        
        try:
            order = (
                self.params.get("p", 1),
                self.params.get("d", 0),
                self.params.get("q", 0)
            )
            
            # --- PASO 1: Entrenar ARIMA sobre la serie objetivo (y_train) ---
            model = ARIMA(y_train, order=order)
            model_fit = model.fit()
            
            # Obtener los residuos del entrenamiento
            residuals = model_fit.resid
            
            # --- PASO 2: Entrenar un modelo de regresión para predecir los residuos ---
            # Usamos Ridge para evitar overfitting con las features.
            # El regresor aprenderá de las features que ARIMA ignora (X_train).
            residual_model = Ridge(alpha=1.0)
            residual_model.fit(X_train, residuals)
            
            # --- PASO 3: Realizar predicciones y combinarlas ---
            
            # Predicción base del modelo ARIMA
            arima_prediction = model_fit.forecast(steps=len(X_test))
            
            # Predicción de los residuos futuros usando las features de test (X_test)
            residual_prediction = residual_model.predict(X_test)
            
            # La predicción final es la suma de ambas
            final_prediction = arima_prediction + residual_prediction
            
            return final_prediction.tolist()
        except Exception as e:
            self.logger.error(f"ARIMA Error: {e}")
            return [0] * len(X_test) # Fallback a 0 si hay error
        # === Persistencia del modelo ARIMA ===

    def train_and_save(
        self,
        y_train: pd.Series,
        X_train: pd.DataFrame | None,
        model_name: str,
        models_dir: str | Path | None = None,
    ):
        """
        Entrena el ARIMA con TODOS los datos disponibles y guarda
        el modelo entrenado en disco como un .pkl.

        - y_train: serie objetivo (por ejemplo Return_1)
        - X_train: features (no se usan por ARIMA clásico, se deja por contrato)
        - model_name: nombre base del archivo (sin extensión)
        - models_dir: carpeta donde guardar el modelo
        """
        if models_dir is None:
            models_dir = Path("outputs") / "models"
        else:
            models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        order = (
            int(self.params.get("p", 1)),
            int(self.params.get("d", 0)),
            int(self.params.get("q", 0)),
        )

        self.logger.info(
            f"    -> Entrenando ARIMA final con params={self.params} "
            f"sobre {len(y_train)} observaciones..."
        )

        # Entrenamos un ARIMA "limpio" sólo para producción (sin regresor de residuos)
        model = ARIMA(y_train, order=order)
        model_fit = model.fit()

        # Guardamos también en el objeto
        self.model = model_fit

        # Artefacto que se va a guardar (por si mañana queremos añadir más cosas)
        artifact = {
            "model": model_fit,
        }

        model_path = models_dir / f"{model_name}.pkl"
        joblib.dump(artifact, model_path)

        self.logger.info(f"    -> Modelo ARIMA guardado en: {model_path}")

        return model_path

    def save_model(self, model_path: str | Path) -> None:
        """Guarda el modelo ya entrenado."""
        if not hasattr(self, "model_") or self.model_ is None:
            raise RuntimeError("ARIMA no tiene un modelo entrenado en memoria.")
        model_path = Path(model_path)
        joblib.dump(self.model_, model_path)
        self.logger.info(f"[ARIMA] Modelo guardado en: {model_path}")

    def load_model(self, model_path: str | Path) -> None:
        """
        Carga el modelo ARIMA desde disco.

        Soporta varios formatos posibles de .pkl:
        - El modelo guardado directamente (objeto statsmodels con .forecast)
        - Un dict con claves como 'model', 'arima_model'
        - Un dict legado donde alguna de sus values tiene .forecast
        """
        model_path = Path(model_path)
        artifact = joblib.load(model_path)

        # Caso 1: se guardó directamente el objeto ARIMAResults / SARIMAXResults
        if not isinstance(artifact, dict):
            self.model = artifact
            self.model_ = self.model
            self._is_fitted = True
            self.logger.info(f"[ARIMA] Modelo cargado (objeto directo) desde: {model_path}")
            return

        # Caso 2: es un dict -> intentamos varias estrategias
        keys = list(artifact.keys())
        self.logger.info(f"[ARIMA] Cargado dict desde {model_path} con claves: {keys}")

        # 2a) Claves estándar que podríamos haber usado
        if "model" in artifact:
            self.model = artifact["model"]
            self.logger.info("[ARIMA] Usando artifact['model'] como modelo.")
        elif "arima_model" in artifact:
            self.model = artifact["arima_model"]
            self.logger.info("[ARIMA] Usando artifact['arima_model'] como modelo.")
        else:
            # 2b) Fallback: buscar la primera value que tenga método 'forecast'
            candidate = None
            for k, v in artifact.items():
                if hasattr(v, "forecast"):
                    candidate = v
                    self.logger.info(f"[ARIMA] Usando artifact['{k}'] como modelo (tiene método .forecast).")
                    break

            if candidate is None:
                # 2c) Último recurso: tomar el primer value del dict
                candidate = next(iter(artifact.values()))
                self.logger.warning(
                    "[ARIMA] El dict no tiene 'model' ni 'arima_model' ni objeto con .forecast; "
                    "usando el primer valor del dict como modelo. Revisa este formato más adelante."
                )

            self.model = candidate

        # Compatibilidad con código viejo
        self.model_ = self.model
        self._is_fitted = True

        self.logger.info(f"[ARIMA] Modelo cargado desde: {model_path}")


    def predict_loaded(self, X_all: pd.DataFrame | None = None) -> list[float]:
        """
        Usa el modelo ARIMA ya cargado para predecir el próximo retorno.

        En la versión de producción actual, el ARIMA se entrenó sin exógenas,
        así que X_all se ignora (se deja sólo por compatibilidad de interfaz).
        """
        # Comprobamos que el modelo está cargado
        if getattr(self, "model", None) is None:
            raise RuntimeError(
                "ARIMA predict_loaded llamado pero el modelo no está cargado. "
                "Asegúrate de llamar antes a load_model()."
            )

        # Si quieres, dejamos un log por si X_all viene vacío
        if X_all is None or (hasattr(X_all, "empty") and X_all.empty):
            self.logger.info("[ARIMA] X_all vacío o None en producción; se usará ARIMA puro sin exógenas.")
        else:
            self.logger.debug(f"[ARIMA] Producción: X_all recibido con shape={X_all.shape}, "
                            "pero se ignora para ARIMA puro.")

        # Predicción a 1 paso adelante
        try:
            pred = self.model.forecast(steps=1)
        except Exception as e:
            self.logger.error(f"Error en ARIMA.predict_loaded: {e}")
            raise

        # Devolvemos una lista de floats
        return [float(v) for v in pred]

