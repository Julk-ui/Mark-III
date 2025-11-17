#!/usr/bin/env python3
# models/lstm_model.py
"""
Implementación de un modelo LSTM para predicción.
"""
# --- imports (reemplaza solo este bloque) ---
from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# TensorFlow / Keras (robusto para distintas instalaciones)
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:  # fallback si alguien tiene keras standalone
    import keras     # type: ignore
    tf = None        # opcional

# Alias para mantener el código limpio
Sequential = keras.models.Sequential
LSTM = keras.layers.LSTM
Bidirectional = keras.layers.Bidirectional
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Adam = keras.optimizers.Adam
EarlyStopping = keras.callbacks.EarlyStopping

from .base_model import BaseModel

class LSTMModel(BaseModel):
    """Modelo LSTM para predicción de series temporales."""

    def _create_dataset(self, X_data: np.ndarray, y_data: np.ndarray, look_back: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Crea secuencias para el LSTM."""
        look_back = int(look_back)
        dataX, dataY = [], []
        # Empezamos desde 'look_back' para tener suficientes datos pasados
        for i in range(look_back, len(X_data)):
            # La secuencia de features es desde i-look_back hasta i-1
            a = X_data[i-look_back:i, :]
            dataX.append(a)
            # El objetivo es el valor en el momento i
            dataY.append(y_data[i, 0])
        return np.array(dataX), np.array(dataY)

    def train_and_predict(
        self,
        y_train: pd.Series,
        X_train: pd.DataFrame | None = None,
        X_test: pd.DataFrame | None = None
    ) -> list:
        """Entrena un modelo LSTM y predice."""
        try:
            # Parámetros (forzamos a int los que van a range / shapes)
            window = int(self.params.get("window", 30))
            units = int(self.params.get("units", 64))
            dropout = float(self.params.get("dropout", 0.2))
            epochs = int(self.params.get("epochs", 50))
            batch_size = int(self.params.get("batch_size", 32))
            lr = float(self.params.get("learning_rate", 0.001))
            architecture = self.params.get("architecture", "stacked_lstm")
            n_layers = int(self.params.get("n_layers", 2))
            patience = self.params.get("early_stopping_patience", None)
            validation_split = 0.1  # Usar 10% de los datos de entrenamiento para validación
            self.window = window 

            if X_train is None:
                self.logger.error("LSTM requiere features (X_train). No se puede entrenar.")
                # Si no hay X_test, devolvemos lista vacía para evitar len(None)
                if X_test is not None:
                    return [0] * len(X_test)
                return []

            # 1. Escalar features y target por separado
            feature_scaler = MinMaxScaler(feature_range=(0, 1))
            target_scaler = MinMaxScaler(feature_range=(0, 1))

            scaled_X = feature_scaler.fit_transform(X_train)
            scaled_y = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
            
            self.feature_scaler = feature_scaler
            self.target_scaler = target_scaler

            # 2. Crear secuencias
            X, y = self._create_dataset(scaled_X, scaled_y, window)
            if X.shape[0] == 0:
                self.logger.warning(
                    f"LSTM: No se pudieron crear secuencias con window={window}. Datos insuficientes."
                )
                if X_test is not None:
                    return [0] * len(X_test)
                return []

            # 3. Construir el modelo
            n_features = X.shape[2]
            self.model = Sequential()

            # --- Lógica de construcción dinámica ---
            if architecture == "bidirectional_lstm":
                # Añadir capas bidireccionales
                for i in range(n_layers):
                    return_sequences = (i < n_layers - 1)  # True para todas menos la última
                    if i == 0:
                        self.model.add(
                            Bidirectional(
                                LSTM(units, return_sequences=return_sequences),
                                input_shape=(window, n_features),
                            )
                        )
                    else:
                        self.model.add(
                            Bidirectional(LSTM(units, return_sequences=return_sequences))
                        )
                    self.model.add(Dropout(dropout))
            else:  # "stacked_lstm" por defecto
                # Añadir capas LSTM apiladas
                for i in range(n_layers):
                    return_sequences = (i < n_layers - 1)  # True para todas menos la última
                    if i == 0:
                        self.model.add(
                            LSTM(
                                units,
                                return_sequences=return_sequences,
                                input_shape=(window, n_features),
                            )
                        )
                    else:
                        self.model.add(LSTM(units, return_sequences=return_sequences))
                    self.model.add(Dropout(dropout))

            self.model.add(Dense(units=1))  # Capa de salida final

            optimizer = Adam(learning_rate=lr)
            self.model.compile(optimizer=optimizer, loss="mean_squared_error")
            self.model.summary(print_fn=self.logger.info)  # Loguear la arquitectura del modelo

            # 4. Entrenar
            callbacks = []
            if patience:
                # Añadir Early Stopping si está configurado
                early_stop = EarlyStopping(
                    monitor="val_loss", patience=int(patience), restore_best_weights=True
                )
                callbacks.append(early_stop)

            self.model.fit(
                X,
                y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1,
                shuffle=False,
            )

            # 5. Predecir
            # Tomar las últimas `window` filas de features del set de entrenamiento para predecir
            last_sequence_features = scaled_X[-window:]
            input_for_pred = last_sequence_features.reshape((1, window, n_features))

            prediction_scaled = self.model.predict(input_for_pred, verbose=0)
            # Revertir la escala de la predicción usando el 'target_scaler'
            prediction = target_scaler.inverse_transform(prediction_scaled)

            # Replicar la predicción para el tamaño de X_test (normalmente 1 en backtest)
            pred_list = prediction.flatten().tolist()
            if X_test is not None:
                return pred_list * len(X_test)
            else:
                return pred_list

        except Exception as e:
            self.logger.error(f"LSTM Error: {e}")
            if X_test is not None:
                return [0] * len(X_test)
            return []

    
    def save_model(self, model_path: str | Path) -> None:
        if not hasattr(self, "model") or self.model is None:
            raise RuntimeError("LSTM no tiene modelo entrenado en memoria.")
        model_path = Path(model_path)
        self.model.save(model_path)
        scaler_path = model_path.with_suffix(".scalers.pkl")
        joblib.dump(
            {
                "feature_scaler": getattr(self, "feature_scaler", None),
                "target_scaler": getattr(self, "target_scaler", None),
                "window": getattr(self, "window", None),
            },
            scaler_path,
        )
        self.logger.info(f"[LSTM] Modelo guardado en: {model_path}")
        self.logger.info(f"[LSTM] Scalers guardados en: {scaler_path}")

    def load_model(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        self.model = keras.models.load_model(model_path)
        scaler_path = model_path.with_suffix(".scalers.pkl")
        scaler_data = joblib.load(scaler_path)
        self.feature_scaler = scaler_data["feature_scaler"]
        self.target_scaler = scaler_data["target_scaler"]
        self.window = scaler_data["window"]
        self._is_fitted = True
        self.logger.info(f"[LSTM] Modelo cargado desde: {model_path}")
        self.logger.info(f"[LSTM] Scalers cargados desde: {scaler_path}")

    def predict_loaded(self, X_all: pd.DataFrame) -> list[float]:
        """
        Usa el LSTM ya cargado para predecir el siguiente retorno.
        Toma la última ventana de tamaño `window` sobre X_all.
        """
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError("LSTM no está cargado/entrenado. Llama antes a load_model().")

        if X_all is None or len(X_all) < self.window:
            self.logger.error("[LSTM] No hay suficientes datos en X_all para construir la ventana.")
            return []

        # Escalar features
        scaled_X_all = self.feature_scaler.transform(X_all)
        last_seq = scaled_X_all[-self.window:]
        input_seq = last_seq.reshape((1, self.window, scaled_X_all.shape[1]))

        pred_scaled = self.model.predict(input_seq, verbose=0)
        pred = self.target_scaler.inverse_transform(pred_scaled)
        return pred.flatten().tolist()
    def train_and_save(
        self,
        y_train: pd.Series,
        X_train: pd.DataFrame | None,
        model_name: str,
        models_dir: str | Path | None = None,
    ) -> None:
        """
        Entrena el LSTM con TODOS los datos disponibles y guarda
        el modelo entrenado en disco como un .keras.
        """
        if models_dir is None:
            models_dir = Path("outputs") / "models"
        else:
            models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f"    -> Entrenando LSTM final con params={self.params} "
            f"sobre {len(y_train)} observaciones..."
        )

        # Reutilizamos la lógica de entrenamiento; no necesitamos X_test aquí
        self.train_and_predict(
            y_train=y_train,
            X_train=X_train,
            X_test=None,
        )

        model_path = models_dir / f"{model_name}.keras"
        self.save_model(model_path) 
        self.logger.info(f"    -> Modelo LSTM final guardado en: {model_path}")

        return model_path
