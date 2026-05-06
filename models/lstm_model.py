#!/usr/bin/env python3
# models/lstm_model.py
"""
Implementación de un modelo LSTM para predicción.
"""
# --- imports (reemplaza solo este bloque) ---
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import gc
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
# TensorFlow / Keras (robusto para distintas instalaciones)
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import backend as K
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
    def _cleanup_tf(self) -> None:
        """Libera memoria de TensorFlow/Keras entre entrenamientos (especialmente en backtests)."""
        try:
            if getattr(self, "model", None) is not None:
                del self.model
        except Exception:
            pass

        self.model = None

        try:
            K.clear_session()
        except Exception:
            pass

        gc.collect()
    
    def build_model(self, input_shape, params):
        """
        Construye y compila el modelo según params.
        input_shape: (window, n_features)
        """
        architecture = params.get("architecture", "stacked_lstm")
        units = int(params.get("units", 32))
        n_layers = int(params.get("n_layers", 1))
        dropout = float(params.get("dropout", 0.2))
        learning_rate = float(params.get("learning_rate", 0.001))

        # Cache para no recompilar innecesariamente en cada ventana (acelera MUCHO)
        signature = (architecture, units, n_layers, dropout, learning_rate, tuple(input_shape))

        if not hasattr(self, "_cached_signature"):
            self._cached_signature = None
            self._cached_initial_weights = None

        # Si cambian hiperparámetros o input_shape => rebuild completo
        if self.model is None or self._cached_signature != signature:
            self.model = Sequential()

            if architecture == "bidirectional_lstm":
                # Capa(s) LSTM bidireccional(es)
                if n_layers <= 1:
                    self.model.add(Bidirectional(LSTM(units, return_sequences=False), input_shape=input_shape))
                    if dropout > 0:
                        self.model.add(Dropout(dropout))
                else:
                    self.model.add(Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape))
                    if dropout > 0:
                        self.model.add(Dropout(dropout))
                    for _ in range(n_layers - 2):
                        self.model.add(Bidirectional(LSTM(units, return_sequences=True)))
                        if dropout > 0:
                            self.model.add(Dropout(dropout))
                    self.model.add(Bidirectional(LSTM(units, return_sequences=False)))
                    if dropout > 0:
                        self.model.add(Dropout(dropout))

            else:
                # "stacked_lstm" (default)
                if n_layers <= 1:
                    self.model.add(LSTM(units, input_shape=input_shape, return_sequences=False))
                    if dropout > 0:
                        self.model.add(Dropout(dropout))
                else:
                    self.model.add(LSTM(units, input_shape=input_shape, return_sequences=True))
                    if dropout > 0:
                        self.model.add(Dropout(dropout))
                    for _ in range(n_layers - 2):
                        self.model.add(LSTM(units, return_sequences=True))
                        if dropout > 0:
                            self.model.add(Dropout(dropout))
                    self.model.add(LSTM(units, return_sequences=False))
                    if dropout > 0:
                        self.model.add(Dropout(dropout))

            self.model.add(Dense(1))

            optimizer = Adam(learning_rate=learning_rate)
            self.model.compile(optimizer=optimizer, loss="mse")

            # Guardar pesos iniciales para poder “resetear” rápido por ventana (sin reconstruir)
            self._cached_signature = signature
            self._cached_initial_weights = self.model.get_weights()

        else:
            # Mismos hiperparámetros/shape => reset rápido de pesos + recompilar optimizer (evita arrastre)
            if self._cached_initial_weights is not None:
                self.model.set_weights(self._cached_initial_weights)
            self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

        return self.model

    
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


    def train_and_predict(self, y_train, X_train, X_test=None, cleanup: bool = True):
        """
        Entrena un LSTM y (opcionalmente) predice sobre X_test.

        cleanup=True  -> recomendado en BACKTEST (libera memoria al final).
        cleanup=False -> usado por train_and_save (mantiene self.model vivo para guardar).
        """
        try:
            # Import local para NO tocar tus imports globales
            from sklearn.preprocessing import StandardScaler

            # --- params (desde self.params) ---
            window = int(self.params.get("window", 20))
            units = int(self.params.get("units", 32))
            n_layers = int(self.params.get("n_layers", 1))
            dropout = float(self.params.get("dropout", 0.2))
            batch_size = int(self.params.get("batch_size", 64))
            epochs = int(self.params.get("epochs", 10))
            learning_rate = float(self.params.get("learning_rate", 0.001))
            early_stopping_patience = int(self.params.get("early_stopping_patience", 2))
            architecture = self.params.get("architecture", "stacked_lstm")
            print_summary = bool(self.params.get("print_summary", False))

            # Limpieza antes de construir (rolling windows)
            K.clear_session()
            gc.collect()

            # --- asegurar arrays ---
            X_train = np.asarray(X_train)
            y_train = np.asarray(y_train).reshape(-1, 1)

            if X_train.ndim != 2:
                raise ValueError(f"X_train debe ser 2D (n_samples, n_features). Recibido: {X_train.shape}")
            if len(X_train) != len(y_train):
                raise ValueError(f"X_train y y_train deben tener misma longitud. {len(X_train)} != {len(y_train)}")
            if len(X_train) <= window:
                raise ValueError(f"No hay suficientes observaciones ({len(X_train)}) para window={window}.")

            # Cast a float32
            X_train = X_train.astype(np.float32, copy=False)
            y_train = y_train.astype(np.float32, copy=False)

            # Protecciones (evita NaNs/inf que rompen scaler/fit)
            if not np.isfinite(X_train).all():
                self.logger.warning("LSTM: X_train tenía NaN/Inf. Se reemplazan por 0.")
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            if not np.isfinite(y_train).all():
                self.logger.warning("LSTM: y_train tenía NaN/Inf. Se reemplazan por 0.")
                y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)

            # --- escalar ---
            feature_scaler = StandardScaler()
            target_scaler = StandardScaler()

            X_scaled = feature_scaler.fit_transform(X_train)
            y_scaled = target_scaler.fit_transform(y_train)

            # Guardar scalers en el objeto (para save_model)
            self.feature_scaler = feature_scaler
            self.target_scaler = target_scaler

            # --- crear secuencias ---
            X_seq, y_seq = self._create_dataset(X_scaled, y_scaled, window)

            if X_seq is None or len(X_seq) == 0:
                raise ValueError(f"No se pudieron crear secuencias con window={window}. Revisa longitud de datos.")

            X_seq = X_seq.astype(np.float32, copy=False)
            y_seq = np.asarray(y_seq).reshape(-1, 1).astype(np.float32, copy=False)

            # --- construir modelo (SIN self.build_model) ---
            model = tf.keras.Sequential()
            n_features = X_seq.shape[2]

            def add_core_lstm_layer(is_first: bool, return_sequences: bool):
                if architecture == "bidirectional_lstm":
                    layer = tf.keras.layers.Bidirectional(
                        tf.keras.layers.LSTM(
                            units,
                            return_sequences=return_sequences,
                            recurrent_dropout=0.0
                        ),
                        input_shape=(window, n_features) if is_first else None
                    )
                    model.add(layer)
                else:
                    if is_first:
                        model.add(
                            tf.keras.layers.LSTM(
                                units,
                                return_sequences=return_sequences,
                                input_shape=(window, n_features),
                                recurrent_dropout=0.0
                            )
                        )
                    else:
                        model.add(
                            tf.keras.layers.LSTM(
                                units,
                                return_sequences=return_sequences,
                                recurrent_dropout=0.0
                            )
                        )

            # Capas LSTM
            for i in range(max(1, n_layers)):
                return_seq = (i < max(1, n_layers) - 1)
                add_core_lstm_layer(is_first=(i == 0), return_sequences=return_seq)
                if dropout and dropout > 0:
                    model.add(tf.keras.layers.Dropout(dropout))

            # Salida
            model.add(tf.keras.layers.Dense(1))

            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=opt, loss="mse")

            self.model = model

            if print_summary:
                self.model.summary()

            # Callbacks (si hay validation_split)
            callbacks = []
            val_split = 0.1 if len(X_seq) >= 50 else 0.0
            monitor_metric = "val_loss" if val_split > 0 else "loss"

            if early_stopping_patience > 0:
                callbacks.append(
                    tf.keras.callbacks.EarlyStopping(
                        monitor=monitor_metric,
                        patience=early_stopping_patience,
                        restore_best_weights=True
                    )
                )

            self.model.fit(
                X_seq, y_seq,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=val_split,
                shuffle=False,     # importante en series de tiempo
                verbose=0,
                callbacks=callbacks
            )

            # --- predicción opcional ---
            if X_test is None:
                return np.array([])

            X_test = np.asarray(X_test)
            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)
            if X_test.ndim != 2:
                raise ValueError(f"X_test debe ser 2D. Recibido: {X_test.shape}")

            X_test = X_test.astype(np.float32, copy=False)
            if not np.isfinite(X_test).all():
                self.logger.warning("LSTM: X_test tenía NaN/Inf. Se reemplazan por 0.")
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

            X_test_scaled = feature_scaler.transform(X_test)

            # Predecir TODOS los puntos del test en 1 sola llamada (evita retracing excesivo)
            X_concat = np.vstack([X_scaled, X_test_scaled])  # (train + test) en features
            n_test = X_test_scaled.shape[0]

            blocks = []
            base = len(X_scaled)
            for t in range(n_test):
                start = base + t - window
                if start < 0:
                    # si por alguna razón falta historial, rellena con ceros
                    pad = np.zeros((abs(start), X_concat.shape[1]), dtype=np.float32)
                    block = np.vstack([pad, X_concat[0: base + t]])
                    block = block[-window:]
                else:
                    block = X_concat[start:start + window]
                blocks.append(block)

            X_blocks = np.stack(blocks, axis=0).astype(np.float32, copy=False)  # (n_test, window, n_features)

            yhat_scaled = self.model.predict(X_blocks, verbose=0, batch_size=min(batch_size, n_test))
            yhat = target_scaler.inverse_transform(yhat_scaled.reshape(-1, 1)).ravel()

            # Garantía anti-NaN
            if not np.isfinite(yhat).all():
                self.logger.warning("LSTM: predicción produjo NaN/Inf. Usando fallback con último y_train.")
                fallback = float(y_train[-1, 0]) if len(y_train) else 0.0
                yhat = np.full(shape=(n_test,), fill_value=fallback, dtype=np.float32)

            return yhat

        except Exception as e:
            self.logger.error(f"Error entrenando/prediciendo LSTM: {e}")

            # En BACKTEST: devuelve algo numérico para que NO exploten métricas por NaN
            if X_test is not None:
                try:
                    n_test = len(X_test)
                except Exception:
                    n_test = 1
                fallback = float(np.asarray(y_train).reshape(-1)[-1]) if y_train is not None and len(y_train) else 0.0
                return np.full(shape=(n_test,), fill_value=fallback, dtype=np.float32)

            # En TRAIN final (train_and_save) prefiero no "silenciar" el fallo si no hay test
            if cleanup is False:
                raise

            return np.array([])

        finally:
            # 🔥 CLAVE: en backtest cleanup=True, pero en train_and_save cleanup=False
            if cleanup:
                self._cleanup_tf()

    
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
    
    def train_and_save(self, y_train, X_train, model_name: str = "lstm_best", models_dir: str = "outputs/models"):
        """
        Entrena el LSTM final y lo guarda. Aquí NO se hace cleanup hasta después de guardar.
        """
        try:
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f"{model_name}.keras")

            # Entrena pero NO limpia (para que self.model exista cuando se guarde)
            _ = self.train_and_predict(y_train=y_train, X_train=X_train, X_test=None, cleanup=False)

            # Guardar
            self.save_model(model_path)

            self.logger.info(f"✅ Modelo LSTM guardado en: {model_path}")
            return model_path

        except Exception as e:
            self.logger.error(f"Error en train_and_save LSTM: {e}")
            raise

        finally:
            # Ahora sí: liberar memoria
            self._cleanup_tf()
