#!/usr/bin/env python3
# main_pipeline.py
"""
Pipeline principal del proyecto de Trading Algorítmico.
Integra todos los módulos: Conexión, Limpieza, EDA y Modelos.
"""


from __future__ import annotations
import debugpy
debugpy.listen(("localhost", 5680))
print("Esperando debugger… Conéctate desde VS Code.")
debugpy.wait_for_client()
import sys, os
from typing import Any

# --- Supresión de Warnings de librerías ---
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from itertools import product
from copy import deepcopy
from sklearn.model_selection import ParameterGrid

# Imports de módulos propios
from data.data_loader import DataLoader, DataValidator
from data.data_cleaner import DataCleaner, FeatureEngineer
from utils.metrics import calculate_all_metrics
from models.arima_model import ArimaModel
from models.prophet_model import ProphetModel
from models.lstm_model import LSTMModel # Asegúrate que este archivo exista
from models.random_walk_model import MomentumModel
# Agrega aquí otros modelos que crees

from eda.exploratory_analysis import ExploratoryAnalysis


class TradingPipeline:
    """
    Orquestador principal del pipeline de trading
    """
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        self.config, self.config_path = self._load_config(config_path)
        self._setup_logging()
        self._setup_directories()
        self._df_features_last_backtest = None
        
        # Componentes
        self.data_loader: DataLoader | None = None
        self.data_cleaner: DataCleaner | None = None
        self.feature_engineer: FeatureEngineer | None = None
        self.eda: ExploratoryAnalysis | None = None
    
    def _save_backtest_detail(self, model_name: str, df_bt: pd.DataFrame) -> None:
        """
        Guarda el detalle del mejor backtest para cada modelo.
        Crea CSV y, opcionalmente, Excel con señales, precios y pips.
        """
        if df_bt is None or df_bt.empty:
            return

        output_dir = Path(self.config.get("output", {}).get("dir", "outputs")) / "backtest"
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / f"{model_name}_best_backtest_detail.csv"
        df_bt.to_csv(csv_path)
        self.logger.info(f"    💾 Detalle de backtest guardado en: {csv_path}")

        # Si quieres también Excel
        if "excel" in self.config.get("output", {}).get("formats", []):
            xlsx_path = output_dir / f"{model_name}_best_backtest_detail.xlsx"
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                df_bt.to_excel(writer, sheet_name="backtest_detail")
            self.logger.info(f"    💾 Detalle de backtest (Excel) guardado en: {xlsx_path}")

    
    def _load_config(self, config_path: str) -> tuple[Dict[str, Any], str]:
        """Carga configuración desde YAML"""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"El archivo de configuración no se encontró en: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuración cargada desde: {config_path}")
        return config, config_path
    
    def _setup_logging(self) -> None:
        """Configura el sistema de logging"""
        import logging
        
        log_config = self.config.get("logging", {})
        if not log_config.get("enabled", True):
            return
        
        level = getattr(logging, log_config.get("level", "INFO"))
        
        # Formato
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Handlers
        handlers = []
        
        if log_config.get("to_console", True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(fmt)
            handlers.append(console_handler)
        
        if log_config.get("to_file", True):
            log_file = Path(log_config.get("file_path", "logs/trading.log"))
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(fmt)
            handlers.append(file_handler)
        
        # Configurar logger
        logging.basicConfig(level=level, handlers=handlers)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("="*60)
        self.logger.info("TRADING PIPELINE INICIADO")
        self.logger.info("="*60)
    
    def _setup_directories(self) -> None:
        """Crea estructura de directorios necesaria"""
        dirs = [
            "data/cache",
            "outputs/eda",
            "outputs/models",
            "outputs/backtest",
            "outputs/predictions",
            "logs"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("📁 Directorios de trabajo configurados")
    
    def run(self, mode: str = None) -> None:
        """
        Ejecuta el pipeline según el modo especificado
        
        Args:
            mode: "eda", "train", "backtest", "production"
                 Si es None, usa el modo del config
        """
        mode = mode or self.config.get("execution", {}).get("mode", "eda")
        
        self.logger.info(f"🚀 Ejecutando modo: {mode.upper()}")
        
        if mode == "eda":
            self._run_eda_mode()
        elif mode == "train":
            self._run_train_mode()
        elif mode == "backtest":
            self._run_backtest_mode()
        elif mode == "production":
            self._run_production_mode()
        elif mode == "test":
            self._run_test_mode()  # NUEVO
        elif mode == "clear_cache":
            self._run_clear_cache_mode()
        else:
            raise ValueError(f"Modo no soportado: {mode}")
    
    def _run_eda_mode(self) -> None:
        """
        Modo EDA: Carga → Limpia → Analiza
        Genera reportes estadísticos y gráficos
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("MODO: ANÁLISIS EXPLORATORIO (EDA)")
        self.logger.info("="*60 + "\n")
        
        # 1. Cargar datos
        df_raw = self._load_data()
        
        # 2. Limpiar datos
        df_clean = self._clean_data(df_raw)
        
        # 3. Generar features (opcional para EDA)
        df_features = self._generate_features(df_clean)
        
        # 4. Análisis exploratorio
        self._perform_eda(df_features)
        
        # 5. Guardar datos en diferentes formatos
        self._save_processed_data(df_features)
        self._save_dataframes_to_excel({
            "Raw Data": df_raw,
            "Cleaned Data": df_clean,
            "Features Data": df_features
        })
        
        self.logger.info("\n✅ MODO EDA COMPLETADO")
    
    def _run_train_mode(self) -> None:
        """
        Modo Train: Entrena modelos y guarda para producción
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("MODO: ENTRENAMIENTO DE MODELOS")
        self.logger.info("="*60 + "\n")
        
        # --- PASO 1: Carga, Limpieza y Generación de Features ---
        df = self._load_data()
        df_clean = self._clean_data(df)
        df_features = self._generate_features(df_clean)

        # --- PASO 2: División en Train y Test ---
        self.logger.info("PASO 2: DIVIDIENDO DATOS EN TRAIN Y TEST")
        self.logger.info("-" * 60)
        val_config = self.config.get("validation", {})
        test_size = val_config.get("test_size", 0.2)
        
        # Asegurarse de que no haya NaNs en el target antes de dividir
        target_col = self.config.get("backtest", {}).get("target", "Return_1")
        df_features = df_features.dropna(subset=[target_col])

        split_index = int(len(df_features) * (1 - test_size))
        df_train = df_features.iloc[:split_index]
        df_test = df_features.iloc[split_index:]
        self.logger.info(f"✓ Datos de entrenamiento: {len(df_train)} filas")
        self.logger.info(f"✓ Datos de prueba (hold-out): {len(df_test)} filas")

        # --- PASO 3: Búsqueda de Hiperparámetros (usando el set de TRAIN) ---
        self.logger.info("\n" + "="*60)
        self.logger.info("MODO: BÚSQUEDA DE HIPERPARÁMETROS (SOBRE TRAIN SET)")
        self.logger.info("="*60 + "\n")
        self._run_hyperparameter_tuning(df_train)

        # --- PASO 4: Validación Final (usando el set de TEST) ---
        self.logger.info("\n" + "="*60)
        self.logger.info("MODO: VALIDACIÓN FINAL (SOBRE TEST SET)")
        self.logger.info("="*60 + "\n")
        
        # Cargar la configuración recién optimizada
        optimized_config_path = Path(self.config_path).parent / "config_optimizado.yaml"
        if not optimized_config_path.exists():
            self.logger.error("No se encontró 'config_optimizado.yaml'. Ejecute el backtest primero.")
            return
        
        # Crear un nuevo pipeline temporal para la validación
        validation_pipeline = TradingPipeline(config_path=str(optimized_config_path))
        
        # Preparar datos de test
        y_test = df_test[target_col]
        X_test = df_test.drop(columns=[target_col])

        # Evaluar cada modelo habilitado en la config optimizada
        for model_config in validation_pipeline.config.get("models", []):
            if not model_config.get("enabled", False):
                continue
            
            model_name = model_config["name"]
            self.logger.info(f"Validando modelo final: {model_name}")
            # Aquí iría la lógica para cargar el modelo guardado (.h5, .joblib)
            # y predecir sobre df_test, luego calcular métricas.
            # Por simplicidad, re-entrenamos y predecimos en un solo paso.
            self._validate_model_on_test(model_name, model_config.get("params", {}), df_train, y_test, X_test)
        
        self.logger.info("\n✅ MODO TRAIN COMPLETADO")

    def _run_backtest_mode(self) -> None:
        """
        Modo Backtest: Evalúa modelos en histórico
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("MODO: BACKTESTING")
        self.logger.info("="*60 + "\n")
        
        # 1-3. Cargar, limpiar y features
        df = self._load_data()
        df_clean = self._clean_data(df)
        df_features = self._generate_features(df_clean)
        # 💾 Guardar features para que _find_and_save_best_params pueda reentrenar y guardar modelos
        self._df_features_last_backtest = df_features.copy()
        
        # 4. Preparar y ejecutar backtest
        self._run_hyperparameter_tuning(df_features)
        
        self.logger.info("\n✅ MODO BACKTEST COMPLETADO")

    def _run_hyperparameter_tuning(self, df_features: pd.DataFrame) -> None:
        """Realiza la búsqueda de hiperparámetros para cada modelo configurado."""
        self.logger.info("------------------------------------------------------------")
        self.logger.info("튜 PASO 4: INICIANDO BÚSQUEDA DE HIPERPARÁMETROS")
        self.logger.info("------------------------------------------------------------")
        # Guardamos el dataset de features usado en el backtest
        # para reentrenar luego los mejores modelos.
        self._df_features_last_backtest = df_features

        all_results = []
        models_config = self.config.get("models", [])

        # Recorremos los modelos definidos en config.yaml
        for model_conf in self.config.get("models", []):
            model_name = model_conf.get("name")
            enabled = model_conf.get("enabled", False)
            param_grid = model_conf.get("param_grid", {})

            if not enabled:
                self.logger.info(f"🔕 Modelo {model_name} deshabilitado, se omite.")
                continue

            self.logger.info(f"\n🔥 Procesando modelo: {model_name}")

            # Caso 1: modelo sin grid (solo parámetros fijos)
            if not param_grid:
                fixed_params = model_conf.get("params", {})
                self.logger.info(f"  -> Modelo sin grid, usando params fijos: {fixed_params}")

                predictions, true_values = self._run_walk_forward_for_params(
                    df_features, model_name, fixed_params
                )

                if not predictions:
                    self.logger.warning("    No se generaron predicciones, saltando métricas.")
                    continue

                metrics = self._calculate_metrics(true_values, predictions)
                self.logger.info(f"    - Métricas: {metrics}")

                result_row = {"model": model_name, **fixed_params, **metrics}
                all_results.append(result_row)

                # Guardamos también un 'report' por consistencia
                self._save_model_report(model_name, [result_row])
                continue

            # Caso 2: modelo con GridSearch (ParameterGrid)
            grid = ParameterGrid(param_grid)
            model_results: list[dict] = []

            for i, params in enumerate(grid):
                self.logger.info(f"  -> Probando combinación {i+1}/{len(grid)}: {params}")

                # Ejecutar walk-forward para esta combinación
                predictions, true_values = self._run_walk_forward_for_params(
                    df_features, model_name, params
                )

                # Calcular métricas
                if not predictions:
                    self.logger.warning("    No se generaron predicciones, saltando métricas.")
                    continue

                metrics = self._calculate_metrics(true_values, predictions)
                self.logger.info(f"    - Métricas: {metrics}")

                # Guardar resultado
                result_row = {"model": model_name, **params, **metrics}
                model_results.append(result_row)
                all_results.append(result_row)

            # Guardar reporte detallado para este modelo
            self._save_model_report(model_name, model_results)

        # Si no hay resultados, abortamos
        if not all_results:
            self.logger.error("No se generaron resultados de backtest para ningún modelo.")
            return

        # Consolidar y guardar la mejor configuración
        self._find_and_save_best_params(all_results)

    
    def _run_hyperparameter_tuning(self, df_features: pd.DataFrame) -> None:
        """Orquesta el backtesting con búsqueda de hiperparámetros."""
        self.logger.info("튜 PASO 4: INICIANDO BÚSQUEDA DE HIPERPARÁMETROS")
        self.logger.info("-" * 60)

        all_results = []
        models_config = self.config.get("models", [])

        for model_config in models_config:
            if not model_config.get("enabled", False):
                continue

            model_name = model_config["name"]
            self.logger.info(f"\n🔥 Procesando modelo: {model_name}")

            # Si se usan params fijos, crear una rejilla de 1
            if "params" in model_config:
                param_grid = model_config["params"]
            else:
                param_grid = model_config.get("param_grid", {})

            grid = ParameterGrid(param_grid)
            model_results = []

            for i, params in enumerate(grid):
                self.logger.info(f"  -> Probando combinación {i+1}/{len(grid)}: {params}")

                # Ejecutar walk-forward para esta combinación
                predictions, true_values = self._run_walk_forward_for_params(
                    df_features, model_name, params
                )

                # Calcular métricas
                if not predictions:
                    self.logger.warning("    No se generaron predicciones, saltando métricas.")
                    continue
                
                metrics = self._calculate_metrics(true_values, predictions)
                self.logger.info(f"    - Métricas: {metrics}")

                # Guardar resultado
                result_row = {"model": model_name, **params, **metrics}
                model_results.append(result_row)
                all_results.append(result_row)

            # Guardar reporte detallado para este modelo
            if model_results:
                self._save_model_report(model_name, model_results)

        # Guardar reporte consolidado
        if all_results:
            self._save_consolidated_summary(all_results)

            # Encontrar los mejores parámetros y guardar config optimizada
            self._find_and_save_best_params(all_results)
    
    def _run_test_mode(self) -> None:
        """
        Modo TEST / VALIDACIÓN:
        Usa los mejores parámetros (config_optimizado) y evalúa en un hold-out final.
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("MODO: TEST / VALIDACIÓN")
        self.logger.info("="*60 + "\n")

        # 1-3. Cargar, limpiar y features
        df = self._load_data()
        df_clean = self._clean_data(df)
        df_features = self._generate_features(df_clean)

        # 4. Determinar segmento de validación
        val_cfg = self.config.get("validation", {})
        mode = val_cfg.get("mode", "last_n")
        n = int(val_cfg.get("n", 500))

        target_col = self.config.get("backtest", {}).get("target", "Return_1")

        df_processed = df_features.dropna(subset=[target_col]).bfill().ffill()
        if len(df_processed) <= n + 10:
            self.logger.error("No hay suficientes datos para una validación con last_n=%s", n)
            return

        df_train = df_processed.iloc[:-n]
        df_test = df_processed.iloc[-n:]

        features_cols = [c for c in df_processed.columns if c != target_col]
        X_train_full = df_train[features_cols]
        y_train_full = df_train[target_col]
        X_test_full = df_test[features_cols]
        y_test_full = df_test[target_col]

        # 5. Mejor modelo desde config_optimizado.yaml (o config actual)
        best_model_config = self._get_best_model_from_config()
        if not best_model_config:
            self.logger.error("No se encontró un modelo con 'params' en la configuración. "
                            "Ejecuta primero el modo backtest para generar config_optimizado.")
            return

        model_name = best_model_config["name"]
        params = best_model_config.get("params", {})
        self.logger.info(f"Usando mejor modelo '{model_name}' para validación, params={params}")

        # 6. Validación tipo walk-forward sobre df_test
        all_pred = []
        all_true = []
        bt_rows = []
        close_prices = df_processed["Close"] if "Close" in df_processed.columns else None

        # Entrenamos una vez con df_train completo y vamos moviendo la ventana sobre df_test
        model_class_map = {
            "RandomWalk": MomentumModel,
            "ARIMA": ArimaModel,
            "PROPHET": ProphetModel,
            "LSTM": LSTMModel,
        }
        model_class = model_class_map.get(model_name)
        if model_class is None:
            self.logger.error(f"Modelo '{model_name}' no soportado en modo test.")
            return

        # Entrenar modelo una vez con todo df_train
        model_instance = model_class(params=params, logger=self.logger)
        # Truco: usamos train_and_predict iterativamente con X_test de tamaño 1
        for ts in X_test_full.index:
            # Ventana de entrenamiento = todo hasta ts-1
            mask_train = df_processed.index < ts
            X_tr = df_processed.loc[mask_train, features_cols]
            y_tr = df_processed.loc[mask_train, target_col]
            X_te = df_processed.loc[[ts], features_cols]
            y_te = df_processed.loc[[ts], target_col]

            pred_list = model_instance.train_and_predict(y_tr, X_tr, X_te)
            if pred_list is None or len(pred_list) == 0:
                continue

            pred = float(pred_list[0])
            true_val = float(y_te.iloc[0])

            all_pred.append(pred)
            all_true.append(true_val)

            true_sign = np.sign(true_val)
            pred_sign = np.sign(pred)
            if pred_sign > 0:
                signal = "BUY"
            elif pred_sign < 0:
                signal = "SELL"
            else:
                signal = "HOLD"

            price_prev = price_true = price_pred = delta_price = np.nan
            if close_prices is not None:
                pos = df_processed.index.get_loc(ts)
                if pos > 0:
                    price_prev = close_prices.iloc[pos - 1]
                    price_true = close_prices.iloc[pos]
                    price_pred = float(price_prev * (1.0 + pred))
                    delta_price = price_pred - price_prev

            bt_rows.append({
                "timestamp": ts,
                "y_true": true_val,
                "y_pred": pred,
                "direction_true": int(true_sign),
                "direction_pred": int(pred_sign),
                "signal": signal,
                "price_prev": price_prev,
                "price_true": price_true,
                "price_pred": price_pred,
                "delta_price": delta_price,
            })

        if not all_pred:
            self.logger.error("No se generaron predicciones en validación.")
            return

        # 7. Métricas de validación
        metrics = self._calculate_metrics(all_true, all_pred)
        self.logger.info(f"📊 Métricas de VALIDACIÓN para {model_name}: {metrics}")

        # 8. Guardar Excel consolidado (detalle + métricas)
        output_dir = Path(self.config.get("output", {}).get("dir", "outputs")) / "validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        xlsx_path = output_dir / "validation_consolidated.xlsx"

        df_bt = pd.DataFrame(bt_rows).set_index("timestamp")
        df_metrics = pd.DataFrame([metrics])

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df_bt.to_excel(writer, sheet_name="detail")
            df_metrics.to_excel(writer, sheet_name="metrics", index=False)

        self.logger.info(f"💾 Archivo de validación guardado en: {xlsx_path}")
        self.logger.info("\n✅ MODO TEST / VALIDACIÓN COMPLETADO")
        
    def _find_and_save_best_params(self, all_results: list[dict[str, Any]]) -> None:
        """
        A partir de todas las combinaciones evaluadas en el backtest:
        - Identifica la mejor por modelo usando las métricas de model_selection.
        - Construye un config_optimizado.yaml con esos mejores modelos.
        - (Opcional) Reentrena y guarda los modelos finales en outputs/models.
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🏆 ENCONTRANDO MEJORES HIPERPARÁMETROS")
        self.logger.info("=" * 60)

        if not all_results:
            self.logger.warning("No hay resultados en all_results; nada que optimizar.")
            return

        # 1. Pasar resultados a DataFrame
        df = pd.DataFrame(all_results)

        # Columnas de métricas que NO son hiperparámetros
        metric_cols = [
            "rmse",
            "mae",
            "hit_rate",
            "accuracy",
            "dm_stat",
            "dm_pvalue",
            "sharpe",
            "sortino",
            "max_drawdown",
            "profit_factor",
            "win_rate",
            "payoff_ratio",
        ]

        best_models: list[dict[str, Any]] = []

        # Funciones auxiliares
        def is_nan(v: Any) -> bool:
            try:
                return bool(np.isnan(v))
            except TypeError:
                return False

        def to_native(v: Any) -> Any:
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.bool_,)):
                return bool(v)
            return v

        # 2. Configuración de cómo se escoge el "mejor" modelo
        selection_cfg = self.config.get("model_selection", {})
        primary_metric = selection_cfg.get("primary_metric", "rmse")
        primary_greater_is_better = selection_cfg.get("primary_greater_is_better", False)
        secondary_metric = selection_cfg.get("secondary_metric", None)
        secondary_greater_is_better = selection_cfg.get("secondary_greater_is_better", True)

        # 3. Por cada modelo (ARIMA, PROPHET, LSTM, RandomWalk, etc.) encontrar la mejor fila
        for model_name in df["model"].unique():
            model_df = df[df["model"] == model_name].copy()

            if model_df.empty:
                continue

            # Construir criterios de ordenamiento dinámicos
            sort_by: list[str] = []
            ascending: list[bool] = []

            if primary_metric in model_df.columns:
                sort_by.append(primary_metric)
                ascending.append(not primary_greater_is_better)

            if secondary_metric and secondary_metric in model_df.columns:
                sort_by.append(secondary_metric)
                ascending.append(not secondary_greater_is_better)

            # Fallback si no se encuentra nada: usar rmse si existe
            if not sort_by:
                if "rmse" in model_df.columns:
                    sort_by = ["rmse"]
                    ascending = [True]  # menor rmse mejor
                else:
                    # último fallback: no sabemos qué métrica usar,
                    # nos quedamos con la primera fila tal cual
                    self.logger.warning(
                        f"  -> Modelo {model_name} sin métricas reconocidas para ordenar; "
                        "se toma la primera fila."
                    )
                    best_run = model_df.iloc[0]
                    # Hiperparámetros = todas las columnas excepto métricas + 'model'
                    param_cols = [c for c in model_df.columns if c not in metric_cols + ["model"]]
                    raw_params = {k: best_run[k] for k in param_cols}
                    clean_params = {
                        k: to_native(v)
                        for k, v in raw_params.items()
                        if not is_nan(v)
                    }
                    best_models.append(
                        {"name": model_name, "enabled": True, "params": clean_params}
                    )
                    continue

            # Ordenar según las métricas seleccionadas
            model_df = model_df.sort_values(by=sort_by, ascending=ascending)
            best_run = model_df.iloc[0]

            # Hiperparámetros = todas las columnas excepto métricas + 'model'
            param_cols = [c for c in model_df.columns if c not in metric_cols + ["model"]]
            raw_params = {k: best_run[k] for k in param_cols}

            clean_params = {
                k: to_native(v)
                for k, v in raw_params.items()
                if not is_nan(v)
            }

            # Para el log, si existe rmse lo mostramos
            best_rmse = best_run["rmse"] if "rmse" in best_run.index else None
            if best_rmse is not None:
                self.logger.info(
                    f"  -> Mejor para {model_name}: RMSE={float(best_rmse):.6f} "
                    f"con params={clean_params}"
                )
            else:
                self.logger.info(
                    f"  -> Mejor para {model_name} (sin RMSE) con params={clean_params}"
                )

            best_models.append(
                {
                    "name": model_name,
                    "enabled": True,
                    "params": clean_params,
                }
            )

        if not best_models:
            self.logger.warning("No se encontró ningún mejor modelo para guardar en config_optimizado.")
            return

        # 4. Construir config optimizado: copiamos config actual y reemplazamos sólo la sección de modelos
        optimized_config = dict(self.config)
        optimized_config["models"] = best_models

        base_config_path = Path(self.config_path)
        optimized_config_path = base_config_path.parent / "config_optimizado.yaml"

        with open(optimized_config_path, "w", encoding="utf-8") as f:
            yaml.dump(optimized_config, f, default_flow_style=False, sort_keys=False)

        self.logger.info(f"\n💾 Configuración optimizada guardada en: {optimized_config_path}")

        # 5. Reentrenar y guardar modelos finales (si tenemos features del último backtest)
        if self._df_features_last_backtest is None:
            self.logger.warning(
                "    -> self._df_features_last_backtest es None. "
                "No se reentrenan ni se guardan modelos en disco."
            )
            return

        target_col = self.config.get("backtest", {}).get("target", "Return_1")

        df_proc = (
            self._df_features_last_backtest
            .dropna(subset=[target_col])
            .bfill()
            .ffill()
        )

        if df_proc.empty:
            self.logger.warning(
                "    -> self._df_features_last_backtest quedó vacío tras limpiar NaNs. "
                "No se reentrenan ni se guardan modelos."
            )
            return

        X_full = df_proc.drop(columns=[target_col])
        y_full = df_proc[target_col]

        models_dir = Path("outputs") / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        model_class_map = {
            "RandomWalk": MomentumModel,
            "ARIMA": ArimaModel,
            "PROPHET": ProphetModel,
            "LSTM": LSTMModel,
        }

        self.logger.info("\n🧠 Reentrenando y guardando modelos óptimos...")

        for m in best_models:
            name = m["name"]
            params = m.get("params", {})

            model_class = model_class_map.get(name)
            if model_class is None:
                self.logger.warning(f"    -> Modelo '{name}' no está soportado para guardado. Se omite.")
                continue

            model = model_class(params=params, logger=self.logger)
            model_name = f"{name.lower()}_best"

            try:
                model.train_and_save(
                    y_train=y_full,
                    X_train=X_full,
                    model_name=model_name,
                    models_dir=models_dir,
                )
                self.logger.info(
                    f"    ✅ Modelo {name} entrenado y guardado en carpeta: {models_dir} "
                    f"(nombre base: {model_name})"
                )
            except NotImplementedError:
                self.logger.warning(
                    f"    ⚠️ El modelo {name} no implementa train_and_save(...). "
                    "Se omite el guardado en disco."
                )

        self.logger.info("\n✅ Proceso de optimización y guardado de modelos completado.")

    
    def _save_model_report(self, model_name: str, model_results: list[dict]) -> None:
        """Guarda el reporte detallado de un modelo en un archivo CSV."""
        if not model_results:
            return

        output_dir = Path(self.config.get("output", {}).get("dir", "outputs")) / "backtest"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / f"report_{model_name}.csv"
        df_report = pd.DataFrame(model_results)
        
        # Ordenar por la métrica principal (RMSE)
        if "rmse" in df_report.columns:
            df_report = df_report.sort_values(by="rmse", ascending=True)
            
        df_report.to_csv(report_path, index=False)
        self.logger.info(f"    💾 Reporte para {model_name} guardado en: {report_path}")

    def _save_consolidated_summary(self, all_results: list[dict]) -> None:
        """Guarda un resumen consolidado de todos los modelos."""
        if not all_results:
            return

        output_dir = Path(self.config.get("output", {}).get("dir", "outputs")) / "backtest"
        summary_path = output_dir / "summary_best_runs.csv"
        
        df_summary = pd.DataFrame(all_results)
        # Agrupar por modelo y obtener la mejor ejecución para cada uno (menor RMSE)
        best_runs = df_summary.loc[df_summary.groupby('model')['rmse'].idxmin()]
        best_runs.to_csv(summary_path, index=False)
        self.logger.info(f"\n📄 Resumen consolidado de mejores ejecuciones guardado en: {summary_path}")
        
        # OPCIONAL: Guardar también en Excel
        output_formats = self.config.get("output", {}).get("formats", [])
        if "excel" in output_formats:
            excel_path = output_dir / "summary_best_runs.xlsx"
            best_runs.to_excel(excel_path, index=False)
            self.logger.info(f"📊 Resumen consolidado guardado también en Excel: {excel_path}")

    def _run_walk_forward_for_params(
        self,
        df_features: pd.DataFrame,
        model_name: str,
        params: dict
    ) -> tuple[list, list]:
        """Ejecuta un backtest Walk-Forward para una configuración de modelo específica."""
        backtest_config = self.config.get("backtest", {})
        initial_train_size = backtest_config.get("initial_train", 800)
        step = backtest_config.get("step", 20)
        target_col = backtest_config.get("target", "Return_1")

        # 1) Eliminar filas donde el target es NaN
        df_processed = df_features.dropna(subset=[target_col])

        # 2) Rellenar NaNs restantes en las FEATURES
        df_processed = df_processed.bfill().ffill()

        features_cols = [col for col in df_processed.columns if col != target_col]
        y = df_processed[target_col]
        X = df_processed[features_cols]

        # Validación de datos suficientes
        if initial_train_size >= len(X):
            self.logger.warning(
                f"    -> No hay suficientes datos para el backtest con initial_train_size={initial_train_size}. "
                f"Datos disponibles después de limpiar NaNs: {len(X)}. Saltando combinación."
            )
            return [], []

        all_predictions: list = []
        all_true_values: list = []

        for i in range(initial_train_size, len(X), step):
            train_end = i
            test_end = i + 1  # predecimos un paso a la vez

            X_train, X_test = X.iloc[:train_end], X.iloc[train_end:test_end]
            y_train, y_test = y.iloc[:train_end], y.iloc[train_end:test_end]

            if len(X_test) == 0:
                continue

            # Log para depurar NaNs
            if self.logger.isEnabledFor(20):  # INFO
                nan_in_train = X_train.isnull().sum().sum()
                self.logger.info(
                    f"    -> Ventana {(i - initial_train_size) // step}: "
                    f"X_train shape={X_train.shape}, y_train len={len(y_train)}, NaNs en X_train={nan_in_train}"
                )

            prediction = self._train_and_predict(model_name, params, X_train, y_train, X_test)

            if prediction is not None:
                all_predictions.extend(prediction)
                all_true_values.extend(list(y_test.values))

        return all_predictions, all_true_values

    def _train_and_predict(self, model_name: str, params: dict, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> list | None:
        """Punto central para entrenar y predecir con un modelo específico."""
        
        model_map = {
            "RandomWalk": MomentumModel, # Lo mantenemos con el nombre original en config.yaml
            "ARIMA": ArimaModel,
            "PROPHET": ProphetModel, # Ahora apunta a la nueva clase
            "LSTM": LSTMModel,
            # "RandomForest": RandomForestModel # Podrías crear este archivo también
        }

        model_class = model_map.get(model_name)
        
        if not model_class:
            self.logger.warning(f"Modelo '{model_name}' no reconocido. Saltando.")
            return None
        
        try:
            self.logger.debug(f"Instanciando modelo {model_name} con params: {params}")
            model_instance = model_class(params=params, logger=self.logger)
            
            return model_instance.train_and_predict(y_train, X_train, X_test)

        except Exception as e:
            self.logger.error(f"Error al ejecutar {model_name}: {e}")
            return None

    def _calculate_metrics(self, y_true: list, y_pred: list) -> dict:
        """
        Calcula un conjunto de métricas de evaluación.

        - Calcula todas las métricas disponibles en utils.metrics.calculate_all_metrics.
        - Aplica un umbral de pips (backtest.threshold_pips) para las métricas de TRADING.
        - Filtra las métricas a las listadas en config['backtest']['metrics'].
        """
        if not y_true or not y_pred:
            self.logger.warning("Listas de valores vacías para calcular métricas.")
            return {"rmse": np.nan, "mae": np.nan, "hit_rate": np.nan}

        bt_cfg = self.config.get("backtest", {})

        # Parámetros opcionales para métricas de trading
        pip_size = float(bt_cfg.get("pip_size", 0.0001))
        threshold_pips = float(bt_cfg.get("threshold_pips", 0.0))
        risk_free = float(bt_cfg.get("risk_free", 0.0))  # anual
        periods_per_year = int(bt_cfg.get("periods_per_year", 252))

        all_metrics = calculate_all_metrics(
            y_true,
            y_pred,
            benchmark_values=None,  # benchmark ingenuo (pred = 0) por defecto
            risk_free=risk_free,
            periods_per_year=periods_per_year,
            pip_size=pip_size,
            threshold_pips=threshold_pips,
        )

        # Lista de métricas a usar según la configuración
        metrics_cfg = self.config.get("backtest", {}).get("metrics", [])
        if metrics_cfg:
            metrics = {k: all_metrics.get(k) for k in metrics_cfg if k in all_metrics}
        else:
            metrics = all_metrics

        # Redondear para guardar en CSV
        return {
            k: (round(v, 6) if isinstance(v, (int, float)) and not np.isnan(v) else v)
            for k, v in metrics.items()
        }

        """
        Calcula un conjunto de métricas de evaluación.

        - Calcula todas las métricas disponibles en utils.metrics.
        - Filtra las que estén listadas en config['backtest']['metrics'].
        """
        if not y_true or not y_pred:
            self.logger.warning("Listas de valores vacías para calcular métricas.")
            return {"rmse": np.nan, "mae": np.nan, "hit_rate": np.nan}

        all_metrics = calculate_all_metrics(y_true, y_pred)

        # Lista de métricas a usar según la configuración
        metrics_cfg = self.config.get("backtest", {}).get("metrics", [])
        if metrics_cfg:
            metrics = {k: all_metrics[k] for k in metrics_cfg if k in all_metrics}
        else:
            metrics = all_metrics

        # Redondear para guardar en CSV
        return {k: (round(v, 6) if isinstance(v, (int, float)) and not np.isnan(v) else v)
                for k, v in metrics.items()}

    def _validate_model_on_test(self, model_name: str, params: dict, df_train: pd.DataFrame, y_test: pd.Series, X_test: pd.DataFrame):
        """Entrena un modelo con datos de train y lo valida contra test."""
        target_col = self.config.get("backtest", {}).get("target", "Return_1")
        
        # Preparar datos de entrenamiento completos
        y_train = df_train[target_col]
        X_train = df_train.drop(columns=[target_col])

        # Entrenar y predecir en el conjunto de test
        # Para una validación real, se cargaría el modelo guardado.
        # Aquí, re-entrenamos y predecimos para demostrar el flujo.
        predictions = self._train_and_predict(model_name, params, X_train, y_train, X_test)

        if predictions is None or len(predictions) != len(y_test):
            self.logger.error(f"No se pudieron generar predicciones para {model_name} en el set de validación.")
            return

        # Calcular y mostrar métricas finales
        final_metrics = self._calculate_metrics(y_test.tolist(), predictions)
        self.logger.info(f"  -> Métricas finales para {model_name} en Test Set:")
        for metric, value in final_metrics.items():
            self.logger.info(f"    - {metric.upper()}: {value}")


    def _run_production_mode(self) -> None:
        """
        Modo Producción:
        - Carga datos recientes desde MT5
        - Genera features
        - Carga desde disco los modelos ganadores según la config (config_optimizado.yaml)
        - Genera una predicción de retorno por modelo
        - Traduce cada predicción a señal BUY/SELL/HOLD (aplicando un umbral en pips)
        - Calcula precio objetivo, delta de precio y pips
        - Guarda todo en outputs/production/production_signals.csv
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("MODO: PRODUCCIÓN")
        self.logger.info("="*60 + "\n")

        # 1) Cargar / limpiar / generar features
        self.logger.info("📥 Cargando datos para producción...")
        df_raw = self._load_data()
        df_clean = self._clean_data(df_raw)
        df_features = self._generate_features(df_clean)

        target_col = self.config.get("backtest", {}).get("target", "Return_1")

        # Quitamos filas sin target ni features
        feature_cols = [c for c in df_features.columns if c != target_col]
        df_processed = df_features.dropna(subset=[target_col] + feature_cols)

        if df_processed.empty:
            self.logger.error("No hay datos suficientes después del procesamiento para producción.")
            return

        X_all = df_processed[feature_cols]

        # 2) Leemos TODOS los modelos habilitados en la config
        models_cfg = self.config.get("models", [])
        enabled_models_cfg = [m for m in models_cfg if m.get("enabled", True)]

        if not enabled_models_cfg:
            self.logger.error(
                "No hay modelos habilitados en la configuración. Revisa la sección 'models' del YAML."
            )
            return

        # Determinar el modelo campeón global (usa la lógica existente)
        best_model_config = self._get_best_model_from_config()
        best_model_name = None
        if best_model_config:
            best_model_name = str(best_model_config.get("name", "")).upper()
            self.logger.info(
                f"🏆 Modelo campeón global según backtest / RMSE: {best_model_name}"
            )
        else:
            self.logger.warning(
                "No se pudo determinar un modelo campeón global con _get_best_model_from_config()."
            )

        # 3) Intentamos cargar las métricas del backtest (summary_best_runs.csv)
        metrics_by_model: dict[str, dict[str, float]] = {}
        backtest_dir = Path(self.config.get("output", {}).get("dir", "outputs")) / "backtest"
        summary_path = backtest_dir / "summary_best_runs.csv"

        if summary_path.exists():
            try:
                df_best = pd.read_csv(summary_path)
                metric_cols = ["rmse", "mae", "hit_rate", "accuracy", "dm_stat", "dm_pvalue"]
                for model_name in df_best["model"].unique():
                    sub = df_best[df_best["model"] == model_name]
                    # Tomamos la fila con menor RMSE
                    idx_min = sub["rmse"].idxmin()
                    row = sub.loc[idx_min]
                    metrics_by_model[str(model_name).upper()] = {
                        col: float(row[col]) if col in row and pd.notna(row[col]) else None
                        for col in metric_cols
                        if col in row
                    }
            except Exception as e:
                self.logger.error(
                    f"No se pudieron cargar métricas desde {summary_path}: {e}"
                )
        else:
            self.logger.warning(
                f"No se encontró {summary_path}; no se agregarán métricas de backtest al CSV de producción."
            )

        # 4) Mapa nombre -> clase de modelo
        model_map = {
            "ARIMA": ArimaModel,
            "PROPHET": ProphetModel,
            "LSTM": LSTMModel,
            "RANDOMWALK": MomentumModel,
        }

        # Directorio donde están los modelos guardados
        models_dir = Path(self.config.get("output", {}).get("dir", "outputs")) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Datos comunes para todas las filas de salida
        last_row = df_processed.iloc[-1]
        price_now = last_row["Close"] if "Close" in last_row else np.nan

        # tamaño de pip (puedes definirlo en config.data.pip_size)
        pip_size = self.config.get("data", {}).get("pip_size", None)
        if pip_size is None:
            symbol = self.config.get("data", {}).get("symbol", "")
            if len(symbol) == 6 and symbol.isalpha():
                pip_size = 0.0001  # típico FX
            else:
                pip_size = 0.01    # por ejemplo índices/ETFs
        pip_size = float(pip_size)

        # --- NUEVO: umbral de pips para generar señal en producción ---
        trading_cfg = self.config.get("trading", {})
        min_pips_signal = float(
            trading_cfg.get(
                "min_pips_signal",
                self.config.get("backtest", {}).get("threshold_pips", 0.0)
            )
        )

        rows = []

        self.logger.info("🔎 Generando señales de producción para TODOS los modelos habilitados...\n")

        for m_cfg in enabled_models_cfg:
            model_name = str(m_cfg.get("name", "UNKNOWN"))
            params = m_cfg.get("params", {})
            model_name_upper = model_name.upper()

            self.logger.info(f"➡ Procesando modelo: {model_name} | params={params}")

            model_class = model_map.get(model_name_upper)
            if model_class is None:
                self.logger.error(f"  ✗ No hay clase asociada al modelo '{model_name}'. Se omite.")
                continue

            model_instance = model_class(params=params, logger=self.logger)

            # Convención: LSTM -> .keras, resto -> .pkl
            file_prefix = f"{model_name.lower()}_best"
            if model_name_upper == "LSTM":
                model_path = models_dir / f"{file_prefix}.keras"
            else:
                model_path = models_dir / f"{file_prefix}.pkl"

            self.logger.info(f"  💾 Intentando cargar el modelo desde: {model_path}")

            if not hasattr(model_instance, "load_model") or not hasattr(model_instance, "predict_loaded"):
                self.logger.error(
                    f"  ✗ El modelo {model_name} no implementa 'load_model' o 'predict_loaded'. Se omite."
                )
                continue

            if not model_path.exists():
                self.logger.error(
                    f"  ✗ El archivo de modelo {model_path} no existe. Se omite."
                )
                continue

            # Cargar modelo
            try:
                model_instance.load_model(model_path)
            except Exception as e:
                self.logger.error(
                    f"  ✗ No se pudo cargar el modelo {model_name} desde disco: {e}"
                )
                continue

            # Predecir
            try:
                prediction = model_instance.predict_loaded(X_all)
            except Exception as e:
                self.logger.error(
                    f"  ✗ Error al predecir con el modelo cargado {model_name}: {e}"
                )
                continue

            if prediction is None or len(prediction) == 0:
                self.logger.error(
                    f"  ✗ El modelo {model_name} no devolvió ninguna predicción. Se omite."
                )
                continue

            # Tomamos la última predicción como "próximo" retorno
            pred_return = float(prediction[-1])

            # Precio objetivo y delta
            if not np.isnan(price_now):
                price_target = price_now * (1.0 + pred_return)
                delta_price = price_target - price_now
                pips = delta_price / pip_size
            else:
                price_target = np.nan
                delta_price = np.nan
                pips = np.nan

            # --- NUEVO: Señal BUY / SELL / HOLD usando umbral de pips ---
            if np.isnan(pips) or abs(pips) < min_pips_signal:
                signal = "HOLD"
            else:
                signal = "BUY" if pips > 0 else "SELL"

            # Métricas de backtest (si existen)
            m_metrics = metrics_by_model.get(model_name_upper, {})
            rmse = m_metrics.get("rmse")
            mae = m_metrics.get("mae")
            hit_rate = m_metrics.get("hit_rate")
            accuracy = m_metrics.get("accuracy")
            dm_stat = m_metrics.get("dm_stat")
            dm_pvalue = m_metrics.get("dm_pvalue")

            is_best = (model_name_upper == best_model_name)

            self.logger.info(
                f"  📈 Modelo {model_name} -> retorno={pred_return:.6f}, "
                f"pips={pips:.2f}, signal={signal}, price_now={price_now}, "
                f"price_target={price_target}, delta_price={delta_price}, "
                f"is_best_model={is_best}"
            )

            row = {
                "timestamp": df_processed.index[-1],
                "symbol": self.config.get("data", {}).get("symbol", "UNKNOWN"),
                "timeframe": self.config.get("data", {}).get("timeframe", "UNKNOWN"),
                "model": model_name,
                "pred_return": pred_return,
                "signal": signal,
                "entry_price": price_now,
                "price_target": price_target,
                "delta_price": delta_price,
                "pips": pips,
                "is_best_model": is_best,
                # Métricas de backtest (pueden ser None si no hay summary_best_runs)
                "rmse_backtest": rmse,
                "mae_backtest": mae,
                "hit_rate_backtest": hit_rate,
                "accuracy_backtest": accuracy,
                "dm_stat_backtest": dm_stat,
                "dm_pvalue_backtest": dm_pvalue,
            }

            rows.append(row)

        if not rows:
            self.logger.error("No se generó ninguna señal de producción (todas fallaron).")
            return

        df_rows = pd.DataFrame(rows)

        # 7) Guardar las señales en CSV
        output_dir = Path(self.config.get("output", {}).get("dir", "outputs")) / "production"
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "production_signals.csv"

        if csv_path.exists():
            existing = pd.read_csv(csv_path)
            # Construir el conjunto completo de columnas (viejas + nuevas)
            all_cols = list(existing.columns)
            for c in df_rows.columns:
                if c not in all_cols:
                    all_cols.append(c)
            # Reindexar ambos dataframes al mismo orden de columnas
            existing = existing.reindex(columns=all_cols)
            df_rows = df_rows.reindex(columns=all_cols)

            # Unir y reescribir el archivo completo (con header actualizado)
            combined = pd.concat([existing, df_rows], ignore_index=True)
            combined.to_csv(csv_path, index=False)
            
        else:
            df_rows.to_csv(csv_path, index=False)

        self.logger.info(f"\n💾 Señales de producción guardadas en: {csv_path}")
        self.logger.info("✅ MODO PRODUCCIÓN COMPLETADO\n")

        """
        Modo Producción:
        - Carga datos recientes desde MT5
        - Genera features
        - Carga desde disco los modelos ganadores según la config (config_optimizado.yaml)
        - Genera una predicción de retorno por modelo
        - Traduce cada predicción a señal BUY/SELL/HOLD
        - Calcula precio objetivo, delta de precio y pips
        - Guarda todo en outputs/production/production_signals.csv
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("MODO: PRODUCCIÓN")
        self.logger.info("="*60 + "\n")

        # 1) Cargar / limpiar / generar features
        self.logger.info("📥 Cargando datos para producción...")
        df_raw = self._load_data()
        df_clean = self._clean_data(df_raw)
        df_features = self._generate_features(df_clean)

        target_col = self.config.get("backtest", {}).get("target", "Return_1")

        # Quitamos filas sin target ni features
        feature_cols = [c for c in df_features.columns if c != target_col]
        df_processed = df_features.dropna(subset=[target_col] + feature_cols)

        if df_processed.empty:
            self.logger.error("No hay datos suficientes después del procesamiento para producción.")
            return

        X_all = df_processed[feature_cols]

        # 2) Leemos TODOS los modelos habilitados en la config
        models_cfg = self.config.get("models", [])
        enabled_models_cfg = [m for m in models_cfg if m.get("enabled", True)]

        if not enabled_models_cfg:
            self.logger.error(
                "No hay modelos habilitados en la configuración. Revisa la sección 'models' del YAML."
            )
            return

        # Determinar el modelo campeón global (usa la lógica existente)
        best_model_config = self._get_best_model_from_config()
        best_model_name = None
        if best_model_config:
            best_model_name = str(best_model_config.get("name", "")).upper()
            self.logger.info(
                f"🏆 Modelo campeón global según backtest / RMSE: {best_model_name}"
            )
        else:
            self.logger.warning(
                "No se pudo determinar un modelo campeón global con _get_best_model_from_config()."
            )

        # 3) Intentamos cargar las métricas del backtest (summary_best_runs.csv)
        metrics_by_model: dict[str, dict[str, float]] = {}
        backtest_dir = Path(self.config.get("output", {}).get("dir", "outputs")) / "backtest"
        summary_path = backtest_dir / "summary_best_runs.csv"

        if summary_path.exists():
            try:
                df_best = pd.read_csv(summary_path)
                metric_cols = ["rmse", "mae", "hit_rate", "accuracy", "dm_stat", "dm_pvalue"]
                for model_name in df_best["model"].unique():
                    sub = df_best[df_best["model"] == model_name]
                    # Tomamos la fila con menor RMSE
                    idx_min = sub["rmse"].idxmin()
                    row = sub.loc[idx_min]
                    metrics_by_model[str(model_name).upper()] = {
                        col: float(row[col]) if col in row and pd.notna(row[col]) else None
                        for col in metric_cols
                        if col in row
                    }
            except Exception as e:
                self.logger.error(
                    f"No se pudieron cargar métricas desde {summary_path}: {e}"
                )
        else:
            self.logger.warning(
                f"No se encontró {summary_path}; no se agregarán métricas de backtest al CSV de producción."
            )

        # 4) Mapa nombre -> clase de modelo
        model_map = {
            "ARIMA": ArimaModel,
            "PROPHET": ProphetModel,
            "LSTM": LSTMModel,
            "RANDOMWALK": MomentumModel,
        }

        # Directorio donde están los modelos guardados
        models_dir = Path(self.config.get("output", {}).get("dir", "outputs")) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Datos comunes para todas las filas de salida
        last_row = df_processed.iloc[-1]
        price_now = last_row["Close"] if "Close" in last_row else np.nan

        # tamaño de pip (puedes definirlo en config.data.pip_size)
        pip_size = self.config.get("data", {}).get("pip_size", None)
        if pip_size is None:
            symbol = self.config.get("data", {}).get("symbol", "")
            if len(symbol) == 6 and symbol.isalpha():
                pip_size = 0.0001  # típico FX
            else:
                pip_size = 0.01    # por ejemplo índices/ETFs
        pip_size = float(pip_size)

        rows = []

        self.logger.info("🔎 Generando señales de producción para TODOS los modelos habilitados...\n")

        for m_cfg in enabled_models_cfg:
            model_name = str(m_cfg.get("name", "UNKNOWN"))
            params = m_cfg.get("params", {})
            model_name_upper = model_name.upper()

            self.logger.info(f"➡ Procesando modelo: {model_name} | params={params}")

            model_class = model_map.get(model_name_upper)
            if model_class is None:
                self.logger.error(f"  ✗ No hay clase asociada al modelo '{model_name}'. Se omite.")
                continue

            model_instance = model_class(params=params, logger=self.logger)

            # Convención: LSTM -> .keras, resto -> .pkl
            file_prefix = f"{model_name.lower()}_best"
            if model_name_upper == "LSTM":
                model_path = models_dir / f"{file_prefix}.keras"
            else:
                model_path = models_dir / f"{file_prefix}.pkl"

            self.logger.info(f"  💾 Intentando cargar el modelo desde: {model_path}")

            if not hasattr(model_instance, "load_model") or not hasattr(model_instance, "predict_loaded"):
                self.logger.error(
                    f"  ✗ El modelo {model_name} no implementa 'load_model' o 'predict_loaded'. Se omite."
                )
                continue

            if not model_path.exists():
                self.logger.error(
                    f"  ✗ El archivo de modelo {model_path} no existe. Se omite."
                )
                continue

            # Cargar modelo
            try:
                model_instance.load_model(model_path)
            except Exception as e:
                self.logger.error(
                    f"  ✗ No se pudo cargar el modelo {model_name} desde disco: {e}"
                )
                continue

            # Predecir
            try:
                prediction = model_instance.predict_loaded(X_all)
            except Exception as e:
                self.logger.error(
                    f"  ✗ Error al predecir con el modelo cargado {model_name}: {e}"
                )
                continue

            if prediction is None or len(prediction) == 0:
                self.logger.error(
                    f"  ✗ El modelo {model_name} no devolvió ninguna predicción. Se omite."
                )
                continue

            # Tomamos la última predicción como "próximo" retorno
            pred_return = float(prediction[-1])

            # Señal BUY / SELL / HOLD
            if pred_return > 0:
                signal = "BUY"
            elif pred_return < 0:
                signal = "SELL"
            else:
                signal = "HOLD"

            # Precio objetivo y delta
            if not np.isnan(price_now):
                price_target = price_now * (1.0 + pred_return)
                delta_price = price_target - price_now
                pips = delta_price / pip_size
            else:
                price_target = np.nan
                delta_price = np.nan
                pips = np.nan

            # Métricas de backtest (si existen)
            m_metrics = metrics_by_model.get(model_name_upper, {})
            rmse = m_metrics.get("rmse")
            mae = m_metrics.get("mae")
            hit_rate = m_metrics.get("hit_rate")
            accuracy = m_metrics.get("accuracy")
            dm_stat = m_metrics.get("dm_stat")
            dm_pvalue = m_metrics.get("dm_pvalue")

            is_best = (model_name_upper == best_model_name)

            self.logger.info(
                f"  📈 Modelo {model_name} -> retorno={pred_return:.6f}, "
                f"signal={signal}, price_now={price_now}, "
                f"price_target={price_target}, delta_price={delta_price}, pips={pips}, "
                f"is_best_model={is_best}"
            )

            row = {
                "timestamp": df_processed.index[-1],
                "symbol": self.config.get("data", {}).get("symbol", "UNKNOWN"),
                "timeframe": self.config.get("data", {}).get("timeframe", "UNKNOWN"),
                "model": model_name,
                "pred_return": pred_return,
                "signal": signal,
                "entry_price": price_now,
                "price_target": price_target,
                "delta_price": delta_price,
                "pips": pips,
                "is_best_model": is_best,
                # Métricas de backtest (pueden ser None si no hay summary_best_runs)
                "rmse_backtest": rmse,
                "mae_backtest": mae,
                "hit_rate_backtest": hit_rate,
                "accuracy_backtest": accuracy,
                "dm_stat_backtest": dm_stat,
                "dm_pvalue_backtest": dm_pvalue,
            }

            rows.append(row)

        if not rows:
            self.logger.error("No se generó ninguna señal de producción (todas fallaron).")
            return

        df_rows = pd.DataFrame(rows)

        # 7) Guardar las señales en CSV
        output_dir = Path(self.config.get("output", {}).get("dir", "outputs")) / "production"
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "production_signals.csv"

        if csv_path.exists():
            existing = pd.read_csv(csv_path)
            # Construir el conjunto completo de columnas (viejas + nuevas)
            all_cols = list(existing.columns)
            for c in df_rows.columns:
                if c not in all_cols:
                    all_cols.append(c)
            # Reindexar ambos dataframes al mismo orden de columnas
            existing = existing.reindex(columns=all_cols)
            df_rows = df_rows.reindex(columns=all_cols)

            # Unir y reescribir el archivo completo (con header actualizado)
            combined = pd.concat([existing, df_rows], ignore_index=True)
            combined.to_csv(csv_path, index=False)
            
        else:
            df_rows.to_csv(csv_path, index=False)

        self.logger.info(f"\n💾 Señales de producción guardadas en: {csv_path}")
        self.logger.info("✅ MODO PRODUCCIÓN COMPLETADO\n")

    def _get_best_model_from_config(self) -> dict | None:
        """
        Identifica el mejor modelo según la config.
        Prioridad:
        1) Modelo con is_best: true y enabled.
        2) Primer modelo enabled que tenga 'params'.
        """
        models = self.config.get("models", [])

        # 1) Buscar marcado como is_best
        for m in models:
            if m.get("enabled", True) and m.get("is_best", False):
                return m

        # 2) Fallback: primer modelo enabled con params
        for m in models:
            if m.get("enabled", True) and "params" in m:
                return m

        return None

    def _run_clear_cache_mode(self) -> None:
        """
        Modo para limpiar los archivos de caché de datos.
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("MODO: LIMPIEZA DE CACHÉ")
        self.logger.info("="*60 + "\n")

        data_config = self.config.get("data", {})
        mt5_config = self.config.get("mt5", {})
        
        # No es necesario conectar a MT5, solo instanciar el loader
        # para acceder a su método de limpieza.
        data_loader = DataLoader(mt5_config=mt5_config)
        
        symbol_to_clear = data_config.get("symbol")
        self.logger.info(f"Limpiando caché para el símbolo: {symbol_to_clear}...")
        data_loader.clear_cache(symbol=symbol_to_clear)
        self.logger.info("\n✅ MODO LIMPIEZA DE CACHÉ COMPLETADO")

    # --- MÉTODOS AUXILIARES DEL PIPELINE ---

    def _load_data(self) -> pd.DataFrame:
        """Paso 1: Cargar datos usando DataLoader."""
        self.logger.info("PASO 1: CARGANDO DATOS")
        self.logger.info("-" * 60)
        
        data_config = self.config.get("data", {})
        mt5_config = self.config.get("mt5", {})
        
        self.data_loader = DataLoader(mt5_config=mt5_config)
        df = self.data_loader.load_data(
            symbol=data_config.get("symbol", "EURUSD"),
            timeframe=data_config.get("timeframe", "D1"),
            n_bars=data_config.get("n_bars", 1000),
            use_cache=data_config.get("use_cache", True),
            cache_expiry_hours=data_config.get("cache_expiry_hours", 24)
        )
        
        # Mensajes para indicar de dónde vienen los parámetros
        if "symbol" in data_config:
            self.logger.info(f"  -> Símbolo '{data_config['symbol']}' cargado desde config/config.yaml")
        else:
            self.logger.info(f"  -> Símbolo '{df.attrs['symbol']}' (por defecto) usado, no especificado en config/config.yaml")
        if "timeframe" in data_config:
            self.logger.info(f"  -> Timeframe '{data_config['timeframe']}' cargado desde config/config.yaml")
        else:
            self.logger.info(f"  -> Timeframe '{df.attrs['timeframe']}' (por defecto) usado, no especificado en config/config.yaml")
        self.logger.info(f"✓ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas.")
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Paso 2: Limpiar datos usando DataCleaner."""
        self.logger.info("PASO 2: LIMPIANDO DATOS")
        self.logger.info("-" * 60)
        self.data_cleaner = DataCleaner(self.config.get("data_cleaning", {}))
        df_clean = self.data_cleaner.clean(df)
        self.logger.info(f"✓ Datos limpios: {df_clean.shape[0]} filas restantes.")
        return df_clean

    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Paso 3: Generar features usando FeatureEngineer."""
        self.logger.info("PASO 3: GENERANDO FEATURES")
        self.logger.info("-" * 60)
        features_config = self.config.get("features", {})
        df_features = df.copy()

        # 1. Generar retornos
        if features_config.get("returns", {}).get("enabled", False):
            periods = features_config["returns"].get("periods", [1])
            df_features = FeatureEngineer.add_returns(df_features, periods=periods)
            self.logger.info(f"  -> Retornos agregados para períodos: {periods}")

        # 2. Generar indicadores técnicos
        if features_config.get("technical_indicators", {}).get("enabled", False):
            df_features = FeatureEngineer.add_technical_indicators(df_features)
            self.logger.info("  -> Indicadores técnicos agregados.")

        # 3. Generar features rezagados (lags)
        if features_config.get("lag_features", {}).get("enabled", False):
            lag_config = features_config["lag_features"]
            for col in lag_config.get("columns", []):
                if col in df_features.columns:
                    df_features = FeatureEngineer.add_lag_features(df_features, col=col, lags=lag_config.get("lags", []))
                    self.logger.info(f"  -> Lags agregados para la columna: '{col}'")

        # --- NUEVO: Log para inspeccionar NaNs después de la generación ---
        nan_counts = df_features.isnull().sum()
        nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
        if not nan_counts.empty:
            self.logger.info("  -> Conteos de valores NaN generados por las features:")
            # Usamos print para asegurar que se muestre completo sin truncar
            print(nan_counts.to_string())
        else:
            self.logger.info("  -> No se generaron valores NaN en este paso.")
        # --- FIN NUEVO ---
        self.logger.info(f"✓ Features generadas. Total columnas: {df_features.shape[1]}.")
        return df_features

    def _perform_eda(self, df: pd.DataFrame) -> None:
        """Ejecuta el análisis exploratorio."""
        if not self.config.get("eda", {}).get("enabled", False):
            self.logger.info("-> Análisis Exploratorio (EDA) deshabilitado en config. Saltando.")
            return
            
        self.logger.info("PASO 4: REALIZANDO ANÁLISIS EXPLORATORIO (EDA)")
        self.logger.info("-" * 60)
        self.eda = ExploratoryAnalysis(self.config)
        self.eda.run_analysis(df)
        self.logger.info("✓ Análisis exploratorio completado.")

    def _save_processed_data(self, df: pd.DataFrame) -> None:
        """Guarda el dataframe procesado en los formatos especificados."""
        output_config = self.config.get("output", {})
        if not output_config.get("save_predictions", False): return

        output_dir = Path(output_config.get("dir", "outputs"))
        formats = output_config.get("formats", ["csv"])
        
        if "csv" in formats:
            df.to_csv(output_dir / "processed_data.csv")
            self.logger.info(f"💾 Datos procesados guardados en: {output_dir / 'processed_data.csv'}")

    def _save_dataframes_to_excel(self, dataframes: dict[str, pd.DataFrame]):
        """Guarda múltiples dataframes en un solo archivo Excel."""
        output_config = self.config.get("output", {})
        if "excel" not in output_config.get("formats", []): return

        output_dir = Path(output_config.get("dir", "outputs"))
        excel_path = output_dir / "trading_data_analysis.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for sheet_name, df in dataframes.items():
                df.to_excel(writer, sheet_name=sheet_name, index=True)
        self.logger.info(f"💾 Reporte de datos guardado en: {excel_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Trading Algorítmico.")
    parser.add_argument("--mode", type=str, default="eda", 
                        choices=["eda", "train", "backtest","production", "test", "clear_cache"],
                        help="Modo de ejecución del pipeline.")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Ruta al archivo de configuración YAML.")
    args = parser.parse_args()
    
    pipeline = TradingPipeline(config_path=args.config)
    pipeline.run(mode=args.mode)
