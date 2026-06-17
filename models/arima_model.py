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

    def _get_order(self) -> tuple[int, int, int]:
        return (
            int(self.params.get("p", 1)),
            int(self.params.get("d", 0)),
            int(self.params.get("q", 0)),
        )

    def _prepare_feature_frame(
        self,
        X: pd.DataFrame | None,
        *,
        fit_mode: bool = False,
    ) -> pd.DataFrame:
        """Alinea columnas de features para el componente de residuos."""
        if X is None:
            return pd.DataFrame()

        X_df = pd.DataFrame(X).copy()

        if fit_mode:
            self.feature_names = list(X_df.columns)
            return X_df

        feature_names = list(getattr(self, "feature_names", []))
        if not feature_names:
            return X_df

        for col in feature_names:
            if col not in X_df.columns:
                X_df[col] = 0.0

        return X_df[feature_names]

    def _fit_hybrid_components(
        self,
        y_train: pd.Series,
        X_train: pd.DataFrame | None = None,
    ):
        """Entrena ARIMA y, si hay features, un Ridge sobre residuos."""
        model = ARIMA(y_train, order=self._get_order())
        model_fit = model.fit()

        residual_model = None
        X_train_df = self._prepare_feature_frame(X_train, fit_mode=True)
        if not X_train_df.empty:
            residuals = model_fit.resid
            residual_model = Ridge(alpha=1.0)
            residual_model.fit(X_train_df, residuals)
        else:
            self.feature_names = []

        return model_fit, residual_model

    def train_and_predict(self, y_train: pd.Series, X_train: pd.DataFrame | None = None, X_test: pd.DataFrame | None = None) -> list:
        """Entrena un modelo ARIMA y predice."""
        if X_test is None:
            self.logger.warning(
                "ARIMA.train_and_predict fue llamado con X_test=None. "
                "Se entrena internamente pero NO se generan predicciones."
            )
            return []
        
        try:
            model_fit, residual_model = self._fit_hybrid_components(y_train, X_train)
            arima_prediction = model_fit.forecast(steps=len(X_test))

            final_prediction = arima_prediction
            X_test_df = self._prepare_feature_frame(X_test, fit_mode=False)
            if residual_model is not None and not X_test_df.empty:
                residual_prediction = residual_model.predict(X_test_df)
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

        self.logger.info(
            f"    -> Entrenando ARIMA final con params={self.params} "
            f"sobre {len(y_train)} observaciones..."
        )

        model_fit, residual_model = self._fit_hybrid_components(y_train, X_train)

        # Guardamos también en el objeto
        self.model = model_fit
        self.residual_model = residual_model

        artifact = {
            "model": model_fit,
            "residual_model": residual_model,
            "feature_names": list(getattr(self, "feature_names", [])),
            "params": dict(self.params or {}),
            "target_name": getattr(y_train, "name", None),
            "model_class": self.__class__.__name__,
        }

        model_path = models_dir / f"{model_name}.pkl"
        joblib.dump(artifact, model_path)

        self.logger.info(f"    -> Modelo ARIMA guardado en: {model_path}")

        return model_path

    def save_model(self, model_path: str | Path) -> None:
        """Guarda el modelo ya entrenado."""
        current_model = getattr(self, "model_", None) or getattr(self, "model", None)
        if current_model is None:
            raise RuntimeError("ARIMA no tiene un modelo entrenado en memoria.")
        model_path = Path(model_path)
        artifact = {
            "model": current_model,
            "residual_model": getattr(self, "residual_model", None),
            "feature_names": list(getattr(self, "feature_names", [])),
            "params": dict(self.params or {}),
            "model_class": self.__class__.__name__,
        }
        joblib.dump(artifact, model_path)
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

        self.residual_model = artifact.get("residual_model")
        self.feature_names = list(artifact.get("feature_names", []))
        loaded_params = artifact.get("params")
        if loaded_params:
            self.params = loaded_params

        # Compatibilidad con código viejo
        self.model_ = self.model
        self._is_fitted = True

        self.logger.info(f"[ARIMA] Modelo cargado desde: {model_path}")


    def predict_loaded(self, X_all: pd.DataFrame | None = None) -> list[float]:
        """
        Usa el artefacto ARIMA ya cargado para predecir el próximo retorno.

        Si el artefacto incluye el regresor de residuos, también incorpora la
        última fila de features para reconstruir la predicción híbrida.
        """
        # Comprobamos que el modelo está cargado
        if getattr(self, "model", None) is None:
            raise RuntimeError(
                "ARIMA predict_loaded llamado pero el modelo no está cargado. "
                "Asegúrate de llamar antes a load_model()."
            )

        # Si quieres, dejamos un log por si X_all viene vacío
        if X_all is None or (hasattr(X_all, "empty") and X_all.empty):
            self.logger.info("[ARIMA] X_all vacío o None en producción; se usará ARIMA puro sin componente de residuos.")
        else:
            self.logger.debug(f"[ARIMA] Producción: X_all recibido con shape={X_all.shape}.")

        # Predicción a 1 paso adelante
        try:
            pred = self.model.forecast(steps=1)
            pred_value = float(pred.iloc[0] if hasattr(pred, "iloc") else pred[0])
            residual_model = getattr(self, "residual_model", None)
            if residual_model is not None and X_all is not None and len(X_all) > 0:
                X_last = self._prepare_feature_frame(pd.DataFrame(X_all).tail(1), fit_mode=False)
                if not X_last.empty:
                    pred_value += float(residual_model.predict(X_last)[0])
        except Exception as e:
            self.logger.error(f"Error en ARIMA.predict_loaded: {e}")
            raise

        # Devolvemos una lista de floats
        return [pred_value]

    def predict_loaded_with_context(
        self,
        X_all: pd.DataFrame | None = None,
        y_all: pd.Series | None = None,
        X_live: pd.DataFrame | None = None,
    ) -> list[float]:
        """
        Reajusta el híbrido ARIMA + Ridge con la serie más reciente disponible.

        Esto alinea producción con la lógica del backtest: ARIMA modela la dinámica
        temporal y el Ridge sobre residuos incorpora las features actuales.
        """
        if y_all is None or len(y_all) == 0:
            return self.predict_loaded(X_all)

        try:
            model_fit, residual_model = self._fit_hybrid_components(y_all, X_all)
            pred = model_fit.forecast(steps=1)
            pred_value = float(pred.iloc[0] if hasattr(pred, "iloc") else pred[0])

            live_frame = X_live if X_live is not None else X_all
            X_last = self._prepare_feature_frame(pd.DataFrame(live_frame).tail(1), fit_mode=False)
            if residual_model is not None and not X_last.empty:
                pred_value += float(residual_model.predict(X_last)[0])

            self.model = model_fit
            self.model_ = model_fit
            self.residual_model = residual_model
            self._is_fitted = True
            return [pred_value]
        except Exception as exc:
            self.logger.warning(
                f"[ARIMA] No se pudo reajustar con contexto live; se usa fallback del artefacto guardado. Error: {exc}"
            )
            return self.predict_loaded(X_all)

