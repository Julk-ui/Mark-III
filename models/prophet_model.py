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
        """
        Carga un modelo Prophet previamente guardado.

        Soporta artefactos de dos tipos:

        1) Instancia directa de Prophet pickeada:
           joblib.dump(model, "prophet_best.pkl")

        2) Diccionario con metadatos, por ejemplo:
           {
               "model_class": "...",
               "params": {...},
               "model": <instancia Prophet>,
               "target_name": "Return_1",
               "feature_names": [...]
           }
        """
        from prophet import Prophet  # import local para asegurar la clase correcta

        model_path = Path(model_path)
        artifact = joblib.load(model_path)

        if getattr(self, "logger", None) is not None:
            self.logger.info(f"[PROPHET] Cargando artefacto desde: {model_path}")
            self.logger.info(f"[PROPHET] Tipo de artefacto: {type(artifact)}")
            if isinstance(artifact, dict):
                self.logger.info(f"[PROPHET] Claves del artefacto: {list(artifact.keys())}")

        # Caso 1: el archivo contiene directamente una instancia de Prophet
        if isinstance(artifact, Prophet):
            self.model_ = artifact
            self.model = artifact

            # En este formato antiguo normalmente no hay params ni regressor_cols
            if not hasattr(self, "params"):
                self.params = {}
            if not hasattr(self, "regressor_cols"):
                self.regressor_cols = []

            self._is_fitted = True
            if getattr(self, "logger", None) is not None:
                self.logger.info(f"[PROPHET] Modelo (instancia directa) cargado desde: {model_path}")
            return

        # Caso 2: el archivo contiene un diccionario con el modelo y metadatos
        if isinstance(artifact, dict):
            model = artifact.get("model") or artifact.get("model_", None)

            if model is None:
                raise ValueError(
                    f"[PROPHET] El artefacto cargado desde {model_path} no contiene "
                    "una clave 'model' ni 'model_'."
                )

            # No hacemos isinstance(model, Prophet) para evitar problemas de versiones;
            # solo exigimos que tenga un método predict (como Prophet).
            if not hasattr(model, "predict"):
                raise ValueError(
                    f"[PROPHET] La clave 'model' del artefacto cargado desde {model_path} "
                    "no parece un modelo Prophet válido (no tiene método 'predict')."
                )

            self.model_ = model
            self.model = model

            # Restaurar params si existen
            loaded_params = artifact.get("params")
            if loaded_params is not None:
                self.params = loaded_params

            # Restaurar lista de regresores / features
            reg_cols = artifact.get("regressor_cols")
            if reg_cols is None:
                # En tu artefacto actual aparece 'feature_names'
                reg_cols = artifact.get("feature_names")

            self.regressor_cols = list(reg_cols) if reg_cols is not None else []

            self._is_fitted = True

            if getattr(self, "logger", None) is not None:
                self.logger.info(f"[PROPHET] Modelo cargado desde: {model_path}")
                self.logger.info(f"[PROPHET] Regresores restaurados: {self.regressor_cols}")

            return

        # Si llegamos aquí, el formato es algo inesperado
        raise ValueError(
            f"[PROPHET] Formato de artefacto no soportado en {model_path}: {type(artifact)}"
        )

    def predict_loaded(self, X_all: pd.DataFrame | None = None) -> list[float]:
        """
        Usa el modelo Prophet ya cargado para predecir el siguiente retorno.

        - X_all: dataframe de features con índice datetime.
        - Si el modelo se entrenó con regresores extra, se intentan usar.
        Si faltan columnas en X_all, se loguea un warning y se rellenan con 0.0.
        """
        # 1) Verificar que el modelo está cargado
        if not getattr(self, "_is_fitted", False) or self.model_ is None:
            msg = "[PROPHET] predict_loaded llamado sin que el modelo esté cargado."
            self.logger.error(msg)
            raise RuntimeError(msg)

        if X_all is None or X_all.empty:
            self.logger.error("[PROPHET] X_all está vacío o es None.")
            return []

        # 2) Determinar timestamp objetivo: último índice de X_all
        if not isinstance(X_all.index, (pd.DatetimeIndex,)):
            self.logger.warning(
                "[PROPHET] X_all no tiene DatetimeIndex; se usará un índice artificial."
            )
            last_timestamp = pd.Timestamp.utcnow()
        else:
            last_timestamp = X_all.index[-1]

        # 3) Construir dataframe para forecast de 1 paso
        df_for_fcst = pd.DataFrame({"ds": [last_timestamp]})

        # Si no tenemos lista de regresores guardada, asumimos todas las columnas excepto el target
        target_col = "Return_1"
        if not getattr(self, "regressor_cols", []):
            self.regressor_cols = [c for c in X_all.columns if c != target_col]

        available_regs = [c for c in self.regressor_cols if c in X_all.columns]
        missing_regs = [c for c in self.regressor_cols if c not in X_all.columns]

        # Rellenar regresores disponibles con el último valor
        last_row = X_all.iloc[-1]
        for reg in available_regs:
            df_for_fcst[reg] = [last_row[reg]]

        # Rellenar regresores faltantes con 0.0 (y loguear)
        if missing_regs:
            self.logger.warning(
                f"[PROPHET] Faltan regresores en X_all: {missing_regs}. "
                "Se rellenan con 0.0 para el forecast."
            )
            for reg in missing_regs:
                df_for_fcst[reg] = [0.0]

        # 4) Hacer predicción
        try:
            forecast = self.model_.predict(df_for_fcst)
        except Exception as e:
            self.logger.error(f"[PROPHET] Error en model_.predict: {e}")
            return []

        if "yhat" not in forecast.columns:
            self.logger.error("[PROPHET] La salida de predict no contiene columna 'yhat'.")
            return []

        yhat = float(forecast["yhat"].iloc[-1])
        self.logger.info(
            f"[PROPHET] Predicción (yhat) para {last_timestamp}: {yhat:.6f}"
        )
        # Para ser consistente con el resto, devolvemos una lista
        return [yhat]
    
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