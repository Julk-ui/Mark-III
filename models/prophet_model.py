#!/usr/bin/env python3
# models/prophet_model.py
"""
Implementación del modelo Prophet para predicción de series temporales.
Incluye manejo de regresores externos con rezago para evitar data leakage.
"""
from __future__ import annotations
import pandas as pd
from prophet import Prophet
import pickle
import joblib
from pathlib import Path
from .base_model import BaseModel

class ProphetModel(BaseModel):
    """Modelo Prophet para predicción de series temporales."""

    def train_and_predict(self, y_train: pd.Series, X_train: pd.DataFrame | None = None, X_test: pd.DataFrame | None = None) -> list:
        """
        Entrena un modelo Prophet y predice.

        Maneja los regresores de dos maneras para evitar data leakage:
        1. Si `use_lagged_regressors` es True: Usa los valores de los regresores del último día conocido (T-1)
           para predecir el día T. Esto simula un entorno real.
        2. Si es False (comportamiento por defecto y erróneo): Usa los regresores de X_test, causando data leakage.
        """
        try:
            # 1. Preparar el DataFrame para Prophet
            df_train = pd.DataFrame({'ds': y_train.index, 'y': y_train.values})

            # Parámetros del modelo
            prophet_params = {k: v for k, v in self.params.items() if k != 'use_lagged_regressors'}
            use_lagged_regressors = self.params.get('use_lagged_regressors', False)

            self.model = Prophet(**prophet_params)

            # 2. Añadir regresores (features)
            regressors = []
            if X_train is not None and not X_train.empty:
                regressors = list(X_train.columns)
                # --- CORRECCIÓN ---
                # Se une X_train a df_train usando la columna 'ds' de df_train
                # y el índice (que son fechas) de X_train. Esto alinea
                # correctamente los datos.
                df_train = df_train.set_index('ds').join(X_train).reset_index()
                for regressor in regressors:
                    self.model.add_regressor(regressor)

            #print(df_train.head())  # Debug: Verificar el DataFrame de entrenamiento
            # 3. Entrenar el modelo
            self.model.fit(df_train)

            # 4. Crear el DataFrame futuro para la predicción
            horizon = len(X_test) if X_test is not None else 1
            future = self.model.make_future_dataframe(periods=horizon)

            # 5. Llenar el DataFrame futuro con los valores de los regresores
            if regressors:
                if use_lagged_regressors:
                    # --- LÓGICA CORRECTA: Evita Data Leakage ---
                    # Usamos el último valor conocido de los regresores (de X_train)
                    # para predecir el siguiente paso.
                    if not X_train.empty:
                        last_known_regressors = X_train.iloc[-1:]
                        for regressor in regressors:
                            # Asigna el último valor conocido a todas las filas del dataframe 'future'
                            future[regressor] = last_known_regressors[regressor].values[0]
                    else:
                        self.logger.warning("Prophet: X_train está vacío, no se pueden usar regresores rezagados.")

                else:
                    # --- LÓGICA INCORRECTA: Causa Data Leakage ---
                    # Esto usa los valores futuros de los regresores, lo que lleva a resultados perfectos.
                    self.logger.warning("Prophet: 'use_lagged_regressors' es False. ¡ALERTA DE DATA LEAKAGE!")
                    if X_test is not None:
                        # Se asegura que el índice coincida para la unión
                        future.set_index('ds', inplace=True)
                        future = future.join(X_test, how='left')
                        future.reset_index(inplace=True)
                        future.ffill(inplace=True) # Rellenar por si acaso

            # 6. Predecir
            forecast = self.model.predict(future)

            # Devolver solo la predicción para el horizonte deseado
            prediction = forecast['yhat'].iloc[-horizon:].tolist()
            return prediction

        except Exception as e:
            self.logger.error(f"Prophet Error: {e}")
            return [0] * (len(X_test) if X_test is not None else 1)
        
        # === Persistencia del modelo Prophet ===

    def save_model(self, model_path: str | Path) -> None:
        if not hasattr(self, "model_") or self.model_ is None:
            raise RuntimeError("Prophet no tiene un modelo entrenado en memoria.")
        model_path = Path(model_path)
        joblib.dump(
            {
                "model": self.model_,
                "regressor_cols": getattr(self, "regressor_cols", []),
            },
            model_path
        )
        self.logger.info(f"[PROPHET] Modelo guardado en: {model_path}")

    def load_model(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        data = joblib.load(model_path)
        self.model_ = data["model"]
        self.regressor_cols = data.get("regressor_cols", [])
        self._is_fitted = True
        self.logger.info(f"[PROPHET] Modelo cargado desde: {model_path}")

    def predict_loaded(self, X_all: pd.DataFrame | None = None) -> list[float]:
        """
        Usa el modelo Prophet ya cargado para predecir el siguiente retorno.

        - X_all: dataframe de features con índice datetime.
        - Si el modelo se entrenó con regresores extra, tratamos de usarlos;
          si faltan algunos en X_all, se loguea un warning y se rellena con 0.0.
        """
        # 1) Verificar que el modelo está cargado
        if not getattr(self, "_is_fitted", False) or self.model_ is None:
            raise RuntimeError("[PROPHET] predict_loaded llamado sin que el modelo esté cargado.")

        if X_all is None or X_all.empty:
            self.logger.error("[PROPHET] X_all vacío al predecir en producción.")
            return []

        # 2) Determinar qué regresores espera Prophet
        if hasattr(self.model_, "extra_regressors"):
            expected_regs = list(self.model_.extra_regressors.keys())
        else:
            expected_regs = []

        # Intersección: solo usamos los regresores que existen en X_all
        available_regs = [c for c in expected_regs if c in X_all.columns]
        missing_regs = [c for c in expected_regs if c not in X_all.columns]

        if missing_regs:
            self.logger.warning(
                f"[PROPHET] Regressors faltantes en producción (se rellenan con 0.0): {missing_regs}"
            )

        # 3) Construir dataframe de predicción con la última fila
        last_timestamp = X_all.index[-1]
        df_fcst = pd.DataFrame({"ds": [last_timestamp]})

        # Asignar los valores de los regresores disponibles
        for reg in available_regs:
            df_fcst[reg] = X_all[reg].iloc[-1]

        # Para los regresores que faltan, crear la columna con 0.0
        for reg in missing_regs:
            df_fcst[reg] = 0.0

        # 4) Ejecutar la predicción con Prophet
        try:
            forecast = self.model_.predict(df_fcst)
        except Exception as e:
            self.logger.error(f"[PROPHET] Error en predict_loaded: {e}")
            return []

        # 5) Usar el último yhat como predicción de retorno
        yhat = float(forecast["yhat"].iloc[-1])
        return [yhat]

        """
        Usa el modelo Prophet ya cargado para predecir el siguiente retorno.

        - X_all: dataframe de features (incluye 'Open', 'Close', etc.) con índice datetime.
        """

        if self.model is None:
            raise RuntimeError("[PROPHET] predict_loaded llamado sin que el modelo esté cargado.")

        if X_all is None or X_all.empty:
            self.logger.error("[PROPHET] X_all vacío al predecir en producción.")
            return []

        # 1) Recuperar la lista de regresores que Prophet espera
        expected_regs = list(self.model.extra_regressors.keys())
        self.logger.debug(f"[PROPHET] Regressors esperados: {expected_regs}")

        # 2) Construir el dataframe future con índice de fechas + columnas de regresores
        #    Tomamos la ÚLTIMA fila (la más reciente) para predecir 1 paso adelante.
        X_last = X_all.tail(1).copy()

        # Asegurarnos de que haya una columna 'ds' con las fechas
        if "ds" in X_last.columns:
            future = X_last.copy()
        else:
            # El índice debe ser datetime
            future = X_last.reset_index().rename(columns={X_last.index.name or "index": "ds"})

        # 3) Verificar que todos los regresores existan en future
        missing = [r for r in expected_regs if r not in future.columns]
        if missing:
            self.logger.error(
                f"[PROPHET] Faltan estos regresores en future_df: {missing}. "
                "Revisa que _generate_features conserve esas columnas."
            )
            return []

        # 4) Dejar sólo ds + regresores, en el orden correcto
        cols = ["ds"] + expected_regs
        future = future[cols]

        # Rellenar posibles NaN
        future = future.ffill().bfill()

        # 5) Predecir con el modelo Prophet cargado
        try:
            forecast = self.model.predict(future)
        except Exception as e:
            self.logger.error(f"[PROPHET] Error en predict_loaded: {e}")
            return []

        # Usamos el último yhat como predicción de retorno
        yhat = float(forecast["yhat"].iloc[-1])
        return [yhat]

        """
        Usa el modelo Prophet ya cargado para predecir el próximo retorno.
        Tomamos la última fecha del índice y los últimos valores de los regresores.
        """
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError("Prophet no está cargado/entrenado. Llama antes a load_model().")

        if X_all is None or X_all.empty:
            self.logger.error("[PROPHET] X_all vacío al predecir en producción.")
            return []

        last_timestamp = X_all.index[-1]
        future = pd.DataFrame({"ds": [last_timestamp]})

        for col in getattr(self, "regressor_cols", []):
            future[col] = X_all[col].iloc[-1]

        forecast = self.model_.predict(future)
        # Asumimos que la variable objetivo es el retorno a 1 paso (yhat)
        yhat = forecast["yhat"].iloc[-1]
        return [float(yhat)]
    
    def train_and_save(
        self,
        y_train: pd.Series,
        X_train: pd.DataFrame | None,
        model_name: str,
        models_dir: str | Path | None = None,
    ):
        """
        Entrena Prophet con todos los datos y guarda:
        - el modelo Prophet pickled (.pkl)
        con sus parámetros y columnas.
        """
        # 1) Carpeta de salida
        if models_dir is None:
            models_dir = Path("outputs") / "models"
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        # 2) Entrenar usando toda la serie
        #    train_and_predict debe crear self.model (instancia de Prophet)
        self.train_and_predict(
            y_train=y_train,
            X_train=X_train,
            X_test=None,
        )

        # 3) Guardar modelo + metadatos
        model_path = models_dir / f"{model_name}.pkl"
        artifact = {
            "model_class": self.__class__.__name__,
            "params": self.params,
            "model": self.model,   # instancia de Prophet
            "target_name": getattr(y_train, "name", "y"),
            "feature_names": list(X_train.columns) if X_train is not None else None,
        }
        joblib.dump(artifact, model_path)

        if getattr(self, "logger", None) is not None:
            self.logger.info(f"    ✅ Modelo Prophet guardado en: {model_path}")