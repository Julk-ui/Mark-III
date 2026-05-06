#!/usr/bin/env python3
# main_pipeline.py
"""
Pipeline principal del proyecto de Trading Algorítmico.
Integra todos los módulos: Conexión, Limpieza, EDA y Modelos.
"""


from __future__ import annotations
import debugpy
import sys, os
import matplotlib
matplotlib.use("Agg")   # <- importante en Windows para evitar Tkinter
import matplotlib.pyplot as plt
from typing import Any
if os.getenv("DEBUGPY", "0") == "1":
    debugpy.listen(("localhost", 5680))
    print("Esperando debugger… Conéctate desde VS Code.")
    debugpy.wait_for_client()

# --- Supresión de Warnings de librerías ---
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import json
import yaml
import argparse
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pandas.errors import EmptyDataError
from itertools import product
from copy import deepcopy
from sklearn.model_selection import ParameterGrid

# Imports de módulos propios
from data.data_loader import DataLoader, DataValidator
from data.data_cleaner import DataCleaner, FeatureEngineer
from utils.metrics_v2 import calculate_all_metrics
from models.arima_model import ArimaModel
from models.prophet_model import ProphetModel
from models.lstm_model import LSTMModel # Asegúrate que este archivo exista
from models.random_walk_model import MomentumModel, RandomWalkModel
from models.tree_models import RandomForestRegressorModel, HistGradientBoostingRegressorModel
from models.regime_model import MarketRegimeClusterer
from utils.decision_utils import build_signal_from_prediction

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
        self._global_champion = None
        self._backtest_run_label: str | None = None
        self._latest_backtest_summary_paths: dict[str, Path | None] = {
            "csv": None,
            "xlsx": None,
        }
        self._active_mode: str | None = None
        
        # Componentes
        self.data_loader: DataLoader | None = None
        self.data_cleaner: DataCleaner | None = None
        self.feature_engineer: FeatureEngineer | None = None
        self.eda: ExploratoryAnalysis | None = None
        self.regime_clusterer: MarketRegimeClusterer | None = None

    def _get_model_selection_settings(self) -> dict[str, Any]:
        """Normaliza la configuración usada para elegir los mejores runs."""
        selection_cfg = self.config.get("model_selection", {}) or {}
        return {
            "primary_metric": selection_cfg.get("primary_metric", "hit_rate"),
            "primary_greater_is_better": bool(selection_cfg.get("primary_greater_is_better", True)),
            "secondary_metric": selection_cfg.get("secondary_metric", "rmse"),
            "secondary_greater_is_better": bool(selection_cfg.get("secondary_greater_is_better", False)),
            "min_trades": int(selection_cfg.get("min_trades", 0) or 0),
            "min_test_points": int(selection_cfg.get("min_test_points", 0) or 0),
        }

    def _select_best_run(
        self,
        df_runs: pd.DataFrame,
        model_name: str | None = None,
        log_prefix: str = "",
    ) -> pd.Series | None:
        """Selecciona el mejor run con la misma lógica usada en todo el pipeline."""
        if df_runs is None or df_runs.empty:
            return None

        selection = self._get_model_selection_settings()
        primary = selection["primary_metric"]
        primary_greater = selection["primary_greater_is_better"]
        secondary = selection["secondary_metric"]
        secondary_greater = selection["secondary_greater_is_better"]
        min_trades = selection["min_trades"]
        min_test_points = selection["min_test_points"]

        ranked = df_runs.copy()

        for col in {primary, secondary, "rmse", "hit_rate", "n_trades", "n_test_points"}:
            if col in ranked.columns:
                ranked[col] = pd.to_numeric(ranked[col], errors="coerce")

        filtered = ranked
        applied_filters: list[str] = []

        if min_trades > 0 and "n_trades" in filtered.columns:
            filtered = filtered[filtered["n_trades"].fillna(0) >= min_trades]
            applied_filters.append(f"n_trades >= {min_trades}")

        if min_test_points > 0 and "n_test_points" in filtered.columns:
            filtered = filtered[filtered["n_test_points"].fillna(0) >= min_test_points]
            applied_filters.append(f"n_test_points >= {min_test_points}")

        if not filtered.empty:
            ranked = filtered
        elif applied_filters:
            scope = model_name or "corridas"
            self.logger.warning(
                f"{log_prefix}{scope}: no hay runs que cumplan {' y '.join(applied_filters)}. "
                "Se seleccionará sin esos filtros."
            )

        sort_cols: list[str] = []
        ascending: list[bool] = []

        if primary in ranked.columns:
            sort_cols.append(primary)
            ascending.append(not primary_greater)

        if secondary and secondary in ranked.columns and secondary != primary:
            sort_cols.append(secondary)
            ascending.append(not secondary_greater)

        if "n_trades" in ranked.columns and "n_trades" not in sort_cols:
            sort_cols.append("n_trades")
            ascending.append(False)

        if not sort_cols:
            if "rmse" in ranked.columns:
                sort_cols = ["rmse"]
                ascending = [True]
            else:
                return ranked.iloc[0]

        ranked = ranked.sort_values(by=sort_cols, ascending=ascending, na_position="last")
        return ranked.iloc[0]

    def _start_backtest_run(self) -> str:
        """Inicializa un identificador uniforme para los artefactos del run."""
        self._backtest_run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"Backtest run_id: {self._backtest_run_label}")
        return self._backtest_run_label

    def _ensure_backtest_run_label(self) -> str:
        """Devuelve el run_id actual o crea uno si todavía no existe."""
        if not self._backtest_run_label:
            return self._start_backtest_run()
        return self._backtest_run_label

    def _get_backtest_output_dir(self) -> Path:
        """Directorio estándar de salidas de backtest."""
        output_dir = Path(self.config.get("output", {}).get("dir", "outputs")) / "backtest"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _build_backtest_archive_path(self, path: Path) -> Path:
        """Construye una ruta versionada con timestamp para el mismo artefacto."""
        run_label = self._ensure_backtest_run_label()
        return path.with_name(f"{path.stem}_{run_label}{path.suffix}")

    def _archive_backtest_artifact(self, path: Path) -> Path | None:
        """Crea una copia archivada del artefacto sin romper la ruta estable."""
        if not path.exists():
            return None

        archive_path = self._build_backtest_archive_path(path)
        shutil.copy2(path, archive_path)
        self.logger.info(f"    Copia archivada con fecha: {archive_path}")
        return archive_path

    def _get_config_dir(self) -> Path:
        """Directorio donde viven los YAML del pipeline."""
        return Path(self.config_path).resolve().parent

    def _get_strategy_profile_name(self) -> str | None:
        """Nombre logico del perfil de estrategia actual, si existe."""
        profile_cfg = self.config.get("strategy_profile", {}) or {}
        raw_name = profile_cfg.get("name") or profile_cfg.get("profile_name")
        if raw_name is None:
            return None
        name = str(raw_name).strip()
        return name or None

    def _normalize_profile_label(self, profile_name: str | None) -> str | None:
        """Normaliza un label de perfil para usarlo en nombres de archivo."""
        if not profile_name:
            return None
        cleaned = "".join(
            ch.lower() if ch.isalnum() else "_"
            for ch in str(profile_name).strip()
        )
        cleaned = "_".join(part for part in cleaned.split("_") if part)
        return cleaned or None

    def _get_models_output_dir(self) -> Path:
        """Directorio raíz de modelos persistidos."""
        models_dir = Path(self.config.get("output", {}).get("dir", "outputs")) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir

    def _get_release_models_dir(self, release_id: str) -> Path:
        """Directorio versionado para una release de modelos."""
        release_dir = self._get_models_output_dir() / "releases" / release_id
        release_dir.mkdir(parents=True, exist_ok=True)
        return release_dir

    def _get_active_release_manifest_path(self, profile_name: str | None = None) -> Path:
        """Ruta del puntero a la release activa usada por producción."""
        profile_label = self._normalize_profile_label(profile_name)
        suffix = f"_{profile_label}" if profile_label else ""
        return self._get_config_dir() / f"active_release{suffix}.json"

    def _get_stable_optimized_config_path(self, profile_name: str | None = None) -> Path:
        """Alias estable de la configuración optimizada para un perfil."""
        profile_label = self._normalize_profile_label(profile_name)
        suffix = f"_{profile_label}" if profile_label else ""
        return self._get_config_dir() / f"config_optimizado{suffix}.yaml"

    def _write_yaml_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        """Escribe YAML de forma atómica para evitar lecturas parciales."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                yaml.dump(payload, fh, default_flow_style=False, sort_keys=False)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        """Escribe JSON de forma atómica para publicar la release activa."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _copy_file_atomic(self, source: Path, destination: Path) -> None:
        """Copia un archivo a su alias estable usando replace atómico."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = destination.with_name(f"{destination.name}.tmp")
        try:
            shutil.copy2(source, tmp_path)
            os.replace(tmp_path, destination)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _load_active_release_manifest(self, profile_name: str | None = None) -> dict[str, Any] | None:
        """Carga el manifiesto de la release activa si existe."""
        manifest_path = self._get_active_release_manifest_path(profile_name=profile_name)
        if not manifest_path.exists():
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as e:
            self.logger.warning(f"No se pudo leer la release activa desde {manifest_path}: {e}")
            return None

    def _resolve_manifest_path(self, raw_path: Any) -> Path | None:
        """Normaliza rutas leídas del manifiesto."""
        if not raw_path:
            return None
        try:
            candidate = Path(str(raw_path))
        except Exception:
            return None
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        return candidate

    def _resolve_active_release_assets(self, profile_name: str | None = None) -> dict[str, Any]:
        """Resuelve config, modelos y resumen de la release activa."""
        output_root = Path(self.config.get("output", {}).get("dir", "outputs")).resolve()
        config_dir = self._get_config_dir()
        profile_label = self._normalize_profile_label(profile_name or self._get_strategy_profile_name())
        profile_config_path = self._get_stable_optimized_config_path(profile_label)
        default_config_path = config_dir / "config_optimizado.yaml"

        assets: dict[str, Any] = {
            "release_id": None,
            "activated_at": None,
            "strategy_profile": profile_label,
            "config_path": profile_config_path if profile_config_path.exists() else default_config_path,
            "models_dir": output_root / "models",
            "summary_csv": output_root / "backtest" / "summary_best_runs.csv",
            "summary_xlsx": output_root / "backtest" / "summary_best_runs.xlsx",
        }

        manifest = self._load_active_release_manifest(profile_name=profile_label)
        if not manifest:
            return assets

        assets["release_id"] = manifest.get("release_id")
        assets["activated_at"] = manifest.get("activated_at")
        assets["strategy_profile"] = manifest.get("strategy_profile") or assets["strategy_profile"]

        for field, key in [
            ("config_path", "config_path"),
            ("models_dir", "models_dir"),
            ("summary_csv", "summary_csv_path"),
            ("summary_xlsx", "summary_xlsx_path"),
        ]:
            resolved = self._resolve_manifest_path(manifest.get(key))
            if resolved and resolved.exists():
                assets[field] = resolved

        return assets

    def _resolve_active_release_config_path(self, profile_name: str | None = None) -> Path:
        """Config optimizada activa, con fallback al alias estable."""
        assets = self._resolve_active_release_assets(profile_name=profile_name)
        return Path(assets["config_path"])

    def _build_dated_log_path(self, log_file: Path) -> Path:
        """Genera un nombre de log con fecha para evitar crecer un archivo unico."""
        date_label = datetime.now().strftime("%Y-%m-%d")
        if log_file.suffix:
            return log_file.with_name(f"{log_file.stem}_{date_label}{log_file.suffix}")
        return log_file.with_name(f"{log_file.name}_{date_label}.log")

    def _publish_active_release(
        self,
        *,
        release_id: str,
        optimized_config: dict[str, Any],
        versioned_config_path: Path,
        models_dir: Path,
        champion_name: str | None,
        profile_name: str | None = None,
    ) -> None:
        """Activa una release completa sin exponer artefactos parciales."""
        summary_csv = self._latest_backtest_summary_paths.get("csv")
        summary_xlsx = self._latest_backtest_summary_paths.get("xlsx")
        profile_label = self._normalize_profile_label(profile_name or self._get_strategy_profile_name())
        stable_config_path = self._get_stable_optimized_config_path(profile_label)
        manifest_path = self._get_active_release_manifest_path(profile_label)

        manifest_payload = {
            "release_id": release_id,
            "activated_at": datetime.now().isoformat(timespec="seconds"),
            "champion_model": champion_name,
            "strategy_profile": profile_label,
            "config_path": str(versioned_config_path.resolve()),
            "models_dir": str(models_dir.resolve()),
            "summary_csv_path": str(summary_csv.resolve()) if isinstance(summary_csv, Path) and summary_csv.exists() else None,
            "summary_xlsx_path": str(summary_xlsx.resolve()) if isinstance(summary_xlsx, Path) and summary_xlsx.exists() else None,
        }

        self._write_json_atomic(manifest_path, manifest_payload)
        self._write_yaml_atomic(stable_config_path, optimized_config)

        self.logger.info(
            "Release activa publicada%s: %s | config=%s | models=%s",
            f" [{profile_label}]" if profile_label else "",
            release_id,
            versioned_config_path,
            models_dir,
        )

    def _ensure_mt5_client(self):
        """Asegura una conexión MT5 reusable para producción/sync."""
        mt5_config = self.config.get("mt5", {}) or {}
        if self.data_loader is None:
            self.data_loader = DataLoader(mt5_config=mt5_config)
        if not self.data_loader.is_connected():
            self.data_loader.connect()
        return self.data_loader.mt5_client

    def _get_live_trading_settings(self) -> dict[str, Any]:
        trading_cfg = self.config.get("trading", {}) or {}
        return {
            "auto_execute_orders": bool(trading_cfg.get("auto_execute_orders", False)),
            "execute_best_model_only": bool(trading_cfg.get("execute_best_model_only", True)),
            "allow_multiple_positions": bool(trading_cfg.get("allow_multiple_positions", False)),
            "magic_number": int(trading_cfg.get("magic_number", 202204) or 202204),
            "order_comment_prefix": str(trading_cfg.get("order_comment_prefix", "MarkIII")),
            "order_deviation_points": int(trading_cfg.get("order_deviation_points", 20) or 20),
            "report_lookback_days": int(trading_cfg.get("report_lookback_days", 30) or 30),
        }

    def _is_future_leakage_column(self, column_name: str, target_col: str | None = None) -> bool:
        """Identifica columnas que contienen información futura y no deben ser features."""
        col = str(column_name or "")
        if target_col and col == str(target_col):
            return True
        return col.startswith("ReturnFwd_") or col.startswith("ReturnFwd")

    def _get_model_feature_columns(self, df: pd.DataFrame, target_col: str) -> list[str]:
        """Columnas válidas para modelado, excluyendo target y variables futuras."""
        feature_cols: list[str] = []
        for col in df.columns:
            if col == target_col:
                continue
            if self._is_future_leakage_column(col, target_col=target_col):
                continue
            if str(col).lower() == "date":
                continue
            try:
                if df[col].isna().all():
                    continue
            except Exception:
                pass
            feature_cols.append(col)
        return feature_cols

    def _get_signal_confirmation_settings(self) -> dict[str, Any]:
        """Configuración opcional de filtros híbridos para autorizar una señal."""
        trading_cfg = self.config.get("trading", {}) or {}
        raw_cfg = trading_cfg.get("signal_confirmation", {}) or {}
        return {
            "enabled": bool(raw_cfg.get("enabled", False)),
            "require_momentum_alignment": bool(raw_cfg.get("require_momentum_alignment", True)),
            "momentum_column": str(raw_cfg.get("momentum_column", "ROC_6")),
            "momentum_buy_min": float(raw_cfg.get("momentum_buy_min", 0.0) or 0.0),
            "momentum_sell_max": float(raw_cfg.get("momentum_sell_max", 0.0) or 0.0),
            "require_volume_confirmation": bool(raw_cfg.get("require_volume_confirmation", False)),
            "volume_column": str(raw_cfg.get("volume_column", "TickVolume_ZScore_20")),
            "volume_min_strength": float(raw_cfg.get("volume_min_strength", 0.0) or 0.0),
            "require_regime_confirmation": bool(raw_cfg.get("require_regime_confirmation", False)),
            "regime_column": str(raw_cfg.get("regime_column", "ADX_14")),
            "regime_min_strength": float(raw_cfg.get("regime_min_strength", 20.0) or 0.0),
        }

    def _coerce_feature_value(self, feature_row: pd.Series | dict[str, Any] | None, column_name: str) -> float | None:
        if feature_row is None or not column_name:
            return None
        if isinstance(feature_row, pd.Series):
            value = feature_row.get(column_name)
        else:
            value = dict(feature_row).get(column_name)
        try:
            if value is None or pd.isna(value):
                return None
            return float(value)
        except Exception:
            return None

    def _evaluate_signal_confirmation(
        self,
        *,
        signal: str,
        feature_row: pd.Series | dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Aplica filtros opcionales de momentum/volumen/régimen a una señal ya propuesta."""
        settings = self._get_signal_confirmation_settings()
        signal_upper = str(signal or "HOLD").upper()
        details = {
            "enabled": bool(settings["enabled"]),
            "passed": True,
            "reason": "confirmation_disabled",
            "momentum_column": settings["momentum_column"],
            "momentum_value": self._coerce_feature_value(feature_row, settings["momentum_column"]),
            "volume_column": settings["volume_column"],
            "volume_value": self._coerce_feature_value(feature_row, settings["volume_column"]),
            "regime_column": settings["regime_column"],
            "regime_value": self._coerce_feature_value(feature_row, settings["regime_column"]),
        }

        if signal_upper not in {"BUY", "SELL"}:
            details["reason"] = "signal_hold"
            return details

        if not settings["enabled"]:
            return details

        details["reason"] = "confirmation_passed"

        if settings["require_momentum_alignment"]:
            momentum_value = details["momentum_value"]
            if momentum_value is None:
                details["passed"] = False
                details["reason"] = f"missing_{settings['momentum_column']}"
                return details
            if signal_upper == "BUY" and momentum_value < settings["momentum_buy_min"]:
                details["passed"] = False
                details["reason"] = f"{settings['momentum_column']}_below_buy_threshold"
                return details
            if signal_upper == "SELL" and momentum_value > settings["momentum_sell_max"]:
                details["passed"] = False
                details["reason"] = f"{settings['momentum_column']}_above_sell_threshold"
                return details

        if settings["require_volume_confirmation"]:
            volume_value = details["volume_value"]
            if volume_value is None:
                details["passed"] = False
                details["reason"] = f"missing_{settings['volume_column']}"
                return details
            if volume_value < settings["volume_min_strength"]:
                details["passed"] = False
                details["reason"] = f"{settings['volume_column']}_below_strength"
                return details

        if settings["require_regime_confirmation"]:
            regime_value = details["regime_value"]
            if regime_value is None:
                details["passed"] = False
                details["reason"] = f"missing_{settings['regime_column']}"
                return details
            if regime_value < settings["regime_min_strength"]:
                details["passed"] = False
                details["reason"] = f"{settings['regime_column']}_below_strength"
                return details

        return details

    def _get_risk_budget_settings(self) -> dict[str, Any]:
        """Normaliza la configuracion de riesgo usada en produccion."""
        trading_cfg = self.config.get("trading", {}) or {}
        risk_cfg = self.config.get("risk", {}) or {}
        risk_per_trade_pct = float(risk_cfg.get("risk_per_trade_pct", 0.01) or 0.0)
        max_total_open_risk_pct = risk_cfg.get("max_total_open_risk_pct", risk_per_trade_pct)
        try:
            max_total_open_risk_pct = float(max_total_open_risk_pct)
        except Exception:
            max_total_open_risk_pct = risk_per_trade_pct
        max_total_open_risk_pct = max(max_total_open_risk_pct, risk_per_trade_pct, 0.0)
        return {
            "allow_multiple_positions": bool(trading_cfg.get("allow_multiple_positions", False)),
            "risk_per_trade_pct": risk_per_trade_pct,
            "max_total_open_risk_pct": max_total_open_risk_pct,
            "block_new_entries_without_sl": bool(risk_cfg.get("block_new_entries_without_sl", True)),
        }

    def _get_daily_loss_guard_settings(self) -> dict[str, Any]:
        """Configuracion del kill switch diario para produccion."""
        risk_cfg = self.config.get("risk", {}) or {}
        loss_limit_pct = float(risk_cfg.get("daily_loss_limit_pct", 0.03) or 0.0)
        return {
            "enabled": bool(risk_cfg.get("halt_on_daily_loss", True)) and loss_limit_pct > 0.0,
            "daily_loss_limit_pct": max(loss_limit_pct, 0.0),
            "daily_loss_measure": str(risk_cfg.get("daily_loss_measure", "equity")).lower(),
            "pre_trade_risk_validation": bool(risk_cfg.get("pre_trade_risk_validation", True)),
            "risk_validation_tolerance_pct": float(risk_cfg.get("risk_validation_tolerance_pct", 0.05) or 0.0),
        }

    def _estimate_open_positions_risk(
        self,
        mt5_client,
        open_positions: pd.DataFrame | None,
    ) -> dict[str, Any]:
        """Estima el riesgo monetario ya comprometido en posiciones abiertas."""
        from utils.risk_utils import estimate_position_risk_amount

        if open_positions is None or open_positions.empty:
            return {
                "open_risk_amount": 0.0,
                "open_positions_count": 0,
                "positions_without_sl": 0,
            }

        total_risk_amount = 0.0
        positions_without_sl = 0
        symbol_specs: dict[str, dict[str, Any]] = {}

        for _, position in open_positions.iterrows():
            symbol = str(position.get("symbol", ""))
            volume = pd.to_numeric(pd.Series([position.get("volume")]), errors="coerce").iloc[0]
            entry_price = pd.to_numeric(pd.Series([position.get("price_open")]), errors="coerce").iloc[0]
            sl_price = pd.to_numeric(pd.Series([position.get("sl")]), errors="coerce").iloc[0]

            if pd.isna(volume) or float(volume) <= 0 or pd.isna(entry_price):
                continue
            if pd.isna(sl_price) or float(sl_price) <= 0:
                positions_without_sl += 1
                continue

            if symbol not in symbol_specs:
                try:
                    symbol_specs[symbol] = mt5_client.get_symbol_spec(symbol) or {}
                except Exception:
                    symbol_specs[symbol] = {}

            spec = symbol_specs[symbol]
            point = float(spec.get("point") or 0.0)
            contract_size = float(spec.get("trade_contract_size") or 0.0)
            if point <= 0:
                point = 0.0001
            if contract_size <= 0:
                contract_size = 100000.0

            total_risk_amount += estimate_position_risk_amount(
                entry_price=float(entry_price),
                sl_price=float(sl_price),
                point=point,
                contract_size=contract_size,
                volume_lots=float(volume),
            )

        return {
            "open_risk_amount": float(total_risk_amount),
            "open_positions_count": int(len(open_positions)),
            "positions_without_sl": int(positions_without_sl),
        }

    def _get_production_output_paths(self) -> dict[str, Path]:
        output_dir = Path(self.config.get("output", {}).get("dir", "outputs")) / "production"
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "dir": output_dir,
            "signals": output_dir / "production_signals.csv",
            "lifecycle": output_dir / "trade_lifecycle_report.csv",
            "closed": output_dir / "closed_trades_report.csv",
            "daily": output_dir / "daily_trade_report.csv",
            "daily_risk_state": output_dir / "daily_risk_state.json",
            "automation_halt": output_dir / "automation_halt_state.json",
        }

    def _coerce_csv_scalar(self, value: Any) -> Any:
        """Normaliza valores leidos desde CSV a tipos simples."""
        if value is None:
            return pd.NA

        text = str(value).strip()
        if not text or text.lower() == "nan":
            return pd.NA
        if text.lower() == "true":
            return True
        if text.lower() == "false":
            return False

        numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
        if pd.notna(numeric):
            return numeric.item() if hasattr(numeric, "item") else numeric
        return text

    def _convert_legacy_signal_row(self, legacy_row: dict[str, Any]) -> dict[str, Any]:
        """Convierte una fila legacy ';'-separada al esquema actual de production."""
        field_mapping = {
            "timestamp": "timestamp",
            "symbol": "symbol",
            "timeframe": "timeframe",
            "model": "model",
            "pred_return": "pred_return",
            "signal": "signal",
            "confidence": "confidence",
            "entry_price": "entry_price",
            "planned_entry_price": "planned_entry_price",
            "price_now": "price_now",
            "price_target": "price_target",
            "delta_price": "delta_price",
            "pips": "pips",
            "sl_price": "sl_price",
            "tp_price": "tp_price",
            "sl_pips": "sl_pips",
            "tp_pips": "tp_pips",
            "market_reference_price": "market_reference_price",
            "live_entry_price": "live_entry_price",
            "live_sl_price": "live_sl_price",
            "live_tp_price": "live_tp_price",
            "live_sl_pips": "live_sl_pips",
            "live_tp_pips": "live_tp_pips",
            "symbol_digits": "symbol_digits",
            "stops_level_points": "stops_level_points",
            "freeze_level_points": "freeze_level_points",
            "volume_lots": "volume_lots",
            "account_balance": "account_balance",
            "risk_per_trade_pct": "risk_per_trade_pct",
            "risk_amount": "risk_amount",
            "is_best_model": "is_best_model",
            "rmse_backtest": "rmse_backtest",
            "mae_backtest": "mae_backtest",
            "hit_rate_backtest": "hit_rate_backtest",
            "accuracy_backtest": "accuracy_backtest",
            "f1_score_backtest": "f1_score_backtest",
            "precision_backtest": "precision_backtest",
            "recall_backtest": "recall_backtest",
            "dm_stat_backtest": "dm_stat_backtest",
            "dm_pvalue_backtest": "dm_pvalue_backtest",
            "sharpe_backtest": "sharpe_backtest",
            "sortino_backtest": "sortino_backtest",
            "calmar_backtest": "calmar_backtest",
            "max_drawdown_backtest": "max_drawdown_backtest",
            "profit_factor_backtest": "profit_factor_backtest",
            "win_rate_backtest": "win_rate_backtest",
            "payoff_ratio_backtest": "payoff_ratio_backtest",
            "consistency_ratio_backtest": "consistency_ratio_backtest",
            "avg_trade_return_backtest": "avg_trade_return_backtest",
        }

        normalized: dict[str, Any] = {}
        for legacy_key, current_key in field_mapping.items():
            if legacy_key not in legacy_row:
                continue
            normalized[current_key] = self._coerce_csv_scalar(legacy_row.get(legacy_key))
        return normalized

    def _normalize_signal_history_dataframe(self, df: pd.DataFrame, path: Path) -> pd.DataFrame:
        """Limpia historicos mezclados de production_signals.csv."""
        if df is None or df.empty or path.name != "production_signals.csv":
            return df

        legacy_columns = [
            column
            for column in df.columns
            if isinstance(column, str) and ";" in column and "timestamp" in column and "signal" in column
        ]
        if not legacy_columns:
            return df

        migrated_rows: list[dict[str, Any]] = []
        for legacy_column in legacy_columns:
            header_fields = [field.strip() for field in str(legacy_column).split(";") if field.strip()]
            if len(header_fields) < 5:
                continue

            for raw_value in df[legacy_column].dropna().tolist():
                raw_text = str(raw_value).strip()
                if not raw_text or raw_text.lower() == "nan" or raw_text.startswith("timestamp;"):
                    continue

                values = raw_text.split(";")
                if len(values) < len(header_fields):
                    values.extend([""] * (len(header_fields) - len(values)))
                legacy_payload = dict(zip(header_fields, values[: len(header_fields)]))
                migrated_rows.append(self._convert_legacy_signal_row(legacy_payload))

        normalized = df.drop(columns=legacy_columns, errors="ignore").copy()
        unnamed_columns = [
            column
            for column in normalized.columns
            if isinstance(column, str) and column.startswith("Unnamed:")
        ]
        if unnamed_columns:
            normalized = normalized.drop(columns=unnamed_columns, errors="ignore")

        if "timestamp" in normalized.columns:
            timestamp_as_text = normalized["timestamp"].astype(str).str.strip()
            normalized = normalized[normalized["timestamp"].notna() & (timestamp_as_text != "")]

        if migrated_rows:
            migrated_df = pd.DataFrame(migrated_rows)
            all_columns = list(normalized.columns)
            for column in migrated_df.columns:
                if column not in all_columns:
                    all_columns.append(column)
            normalized = normalized.reindex(columns=all_columns)
            migrated_df = migrated_df.reindex(columns=all_columns)
            normalized = pd.concat([migrated_df, normalized], ignore_index=True)

        self.logger.info(
            f"Se normalizo el historico legado de senales en {path.name}: "
            f"columnas_legacy={len(legacy_columns)}, filas_migradas={len(migrated_rows)}"
        )
        return normalized.reset_index(drop=True)

    def _load_json_safe(self, path: Path) -> dict[str, Any] | None:
        """Lee un JSON local si existe y es valido."""
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as e:
            self.logger.warning(f"No se pudo leer JSON desde {path}: {e}")
            return None

    def _select_daily_loss_reference_value(
        self,
        *,
        balance: float,
        equity: float | None,
        measure: str,
    ) -> float:
        """Selecciona la magnitud usada para medir la perdida diaria."""
        balance_value = max(float(balance or 0.0), 0.0)
        equity_value = balance_value if equity is None else max(float(equity or 0.0), 0.0)
        measure = str(measure or "equity").lower()
        if measure == "balance":
            return balance_value
        if measure == "min_balance_equity":
            return min(balance_value, equity_value)
        return equity_value

    def _update_daily_loss_guard_state(self, mt5_client=None) -> dict[str, Any]:
        """Actualiza el estado diario de perdida maxima y activa un halt si corresponde."""
        settings = self._get_daily_loss_guard_settings()
        paths = self._get_production_output_paths()
        today_label = datetime.now().date().isoformat()

        default_state = {
            "enabled": settings["enabled"],
            "date": today_label,
            "halt_active": False,
            "daily_loss_limit_pct": settings["daily_loss_limit_pct"],
            "daily_loss_measure": settings["daily_loss_measure"],
        }
        if not settings["enabled"]:
            return default_state

        try:
            if mt5_client is None:
                mt5_client = self._ensure_mt5_client()
            account_info = mt5_client.get_account_info() or {}
        except Exception as e:
            self.logger.warning(f"No se pudo actualizar el guard diario de perdida: {e}")
            return {**default_state, "error": str(e)}

        balance = float(account_info.get("balance", 0.0) or 0.0)
        equity = account_info.get("equity")
        equity = None if equity is None else float(equity or 0.0)
        current_value = self._select_daily_loss_reference_value(
            balance=balance,
            equity=equity,
            measure=settings["daily_loss_measure"],
        )

        existing_state = self._load_json_safe(paths["daily_risk_state"]) or {}
        start_value = existing_state.get("start_value")
        start_value = None if start_value is None else float(start_value)
        if existing_state.get("date") != today_label or not start_value or start_value <= 0:
            start_value = current_value
            existing_state = {
                "date": today_label,
                "start_balance": balance,
                "start_equity": equity,
                "start_value": current_value,
            }

        daily_loss_amount = max(start_value - current_value, 0.0)
        daily_loss_pct = (daily_loss_amount / start_value) if start_value > 0 else 0.0
        halt_active = daily_loss_pct >= settings["daily_loss_limit_pct"] > 0.0

        state_payload = {
            **existing_state,
            "enabled": settings["enabled"],
            "daily_loss_limit_pct": settings["daily_loss_limit_pct"],
            "daily_loss_measure": settings["daily_loss_measure"],
            "last_balance": balance,
            "last_equity": equity,
            "current_value": current_value,
            "daily_loss_amount": daily_loss_amount,
            "daily_loss_pct": daily_loss_pct,
            "halt_active": halt_active,
            "updated_at": datetime.now().isoformat(),
        }
        self._write_json_atomic(paths["daily_risk_state"], state_payload)

        halt_payload = {
            "active": halt_active,
            "date": today_label,
            "reason": "daily_loss_limit" if halt_active else None,
            "daily_loss_limit_pct": settings["daily_loss_limit_pct"],
            "daily_loss_measure": settings["daily_loss_measure"],
            "start_value": start_value,
            "current_value": current_value,
            "daily_loss_amount": daily_loss_amount,
            "daily_loss_pct": daily_loss_pct,
            "updated_at": datetime.now().isoformat(),
        }
        self._write_json_atomic(paths["automation_halt"], halt_payload)

        if halt_active:
            self.logger.warning(
                "Kill switch diario activo: perdida %.2f%% (limite %.2f%%) medida sobre %s.",
                daily_loss_pct * 100.0,
                settings["daily_loss_limit_pct"] * 100.0,
                settings["daily_loss_measure"],
            )

        return {
            **state_payload,
            "start_value": start_value,
        }

    def _validate_pre_trade_execution(
        self,
        *,
        row: pd.Series | dict[str, Any],
        balance: float,
    ) -> tuple[bool, str]:
        """Valida sizing y riesgo antes de enviar una orden real."""
        settings = self._get_daily_loss_guard_settings()
        if not settings["pre_trade_risk_validation"]:
            return True, "Validacion previa de riesgo desactivada."

        data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
        volume_lots = float(pd.to_numeric(pd.Series([data.get("volume_lots")]), errors="coerce").iloc[0] or 0.0)
        allocated_risk_budget = float(
            pd.to_numeric(pd.Series([data.get("allocated_risk_budget")]), errors="coerce").iloc[0] or 0.0
        )
        risk_amount = float(pd.to_numeric(pd.Series([data.get("risk_amount")]), errors="coerce").iloc[0] or 0.0)
        projected_total_open_risk = float(
            pd.to_numeric(pd.Series([data.get("projected_total_open_risk_after_trade")]), errors="coerce").iloc[0] or 0.0
        )
        live_entry_price = pd.to_numeric(pd.Series([data.get("live_entry_price")]), errors="coerce").iloc[0]
        live_sl_price = pd.to_numeric(pd.Series([data.get("live_sl_price")]), errors="coerce").iloc[0]
        live_sl_pips = pd.to_numeric(pd.Series([data.get("live_sl_pips")]), errors="coerce").iloc[0]
        max_total_open_risk_pct = float(
            pd.to_numeric(pd.Series([data.get("max_total_open_risk_pct")]), errors="coerce").iloc[0] or 0.0
        )
        risk_validation_tolerance_pct = max(float(settings["risk_validation_tolerance_pct"]), 0.0)

        if volume_lots <= 0:
            return False, "Lote calculado <= 0."
        if pd.isna(live_entry_price) or pd.isna(live_sl_price):
            return False, "Faltan entry/sl live para validar riesgo."
        if abs(float(live_entry_price) - float(live_sl_price)) <= 0:
            return False, "La distancia entry-SL es cero."
        if pd.notna(live_sl_pips) and float(live_sl_pips) <= 0:
            return False, "SL en pips invalido."
        if allocated_risk_budget <= 0:
            return False, "No hay presupuesto de riesgo asignado."

        tolerance_amount = max(1.0, allocated_risk_budget * risk_validation_tolerance_pct)
        if risk_amount - allocated_risk_budget > tolerance_amount:
            return False, (
                f"Riesgo estimado {risk_amount:.2f} excede el presupuesto "
                f"{allocated_risk_budget:.2f} por mas de la tolerancia {tolerance_amount:.2f}."
            )

        total_risk_limit = max(float(balance or 0.0) * max_total_open_risk_pct, 0.0)
        total_tolerance = max(1.0, total_risk_limit * risk_validation_tolerance_pct)
        if total_risk_limit > 0 and projected_total_open_risk - total_risk_limit > total_tolerance:
            return False, (
                f"Riesgo abierto proyectado {projected_total_open_risk:.2f} excede el limite total "
                f"{total_risk_limit:.2f} por mas de la tolerancia {total_tolerance:.2f}."
            )

        return True, "Validacion previa de riesgo OK."

    def _append_rows_to_csv(self, path: Path, df_rows: pd.DataFrame) -> Path | None:
        """Agrega filas a un CSV unificando columnas con el histórico."""
        if df_rows is None or df_rows.empty:
            return None

        if path.exists():
            try:
                existing = pd.read_csv(path)
            except EmptyDataError:
                existing = pd.DataFrame()
            existing = self._normalize_signal_history_dataframe(existing, path)
            all_cols = list(existing.columns)
            for c in df_rows.columns:
                if c not in all_cols:
                    all_cols.append(c)
            existing = existing.reindex(columns=all_cols)
            df_rows = df_rows.reindex(columns=all_cols)
            df_to_save = pd.concat([existing, df_rows], ignore_index=True)
        else:
            df_to_save = df_rows

        tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
        for attempt in range(3):
            try:
                df_to_save.to_csv(tmp_path, index=False)
                os.replace(tmp_path, path)
                return path
            except PermissionError:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                if attempt < 2:
                    self.logger.warning(
                        f"No se pudo escribir {path} porque está en uso. Reintentando ({attempt + 1}/3)..."
                    )
                    time.sleep(1.0)
                    continue

                fallback_path = path.with_name(
                    f"{path.stem}_locked_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}"
                )
                df_to_save.to_csv(fallback_path, index=False)
                self.logger.error(
                    f"No se pudo escribir {path}. Probablemente está abierto en Excel o bloqueado por otro proceso. "
                    f"Se guardó una copia alternativa en: {fallback_path}"
                )
                return fallback_path

        return path

    def _build_signal_id(self, row: pd.Series | dict[str, Any]) -> str:
        data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
        raw_ts = data.get("timestamp", data.get("signal_time"))
        ts = pd.to_datetime(raw_ts, errors="coerce")
        ts_str = ts.isoformat() if pd.notna(ts) else str(raw_ts)
        return "|".join(
            [
                str(data.get("symbol", "")),
                str(data.get("timeframe", "")),
                str(data.get("model", "")),
                str(data.get("signal", "")),
                ts_str,
            ]
        )

    def _build_order_comment(self, model_name: str) -> str:
        live_cfg = self._get_live_trading_settings()
        prefix = "".join(ch for ch in live_cfg["order_comment_prefix"] if ch.isalnum()) or "MarkIII"
        model_tag = "".join(ch for ch in str(model_name) if ch.isalnum())[:12] or "Model"
        return f"{prefix}_{model_tag}"[:31]

    def _save_daily_trade_report(self, lifecycle: pd.DataFrame) -> pd.DataFrame:
        """Construye un agregado diario por fecha/símbolo/modelo a partir del lifecycle."""
        paths = self._get_production_output_paths()

        if lifecycle is None or lifecycle.empty:
            pd.DataFrame().to_csv(paths["daily"], index=False)
            return pd.DataFrame()

        df = lifecycle.copy()
        df["signal_time"] = pd.to_datetime(df.get("signal_time"), errors="coerce")
        df["close_time"] = pd.to_datetime(df.get("close_time"), errors="coerce")
        df["close_profit_net"] = pd.to_numeric(df.get("close_profit_net"), errors="coerce")
        df["entry_date"] = df["signal_time"].dt.date
        df["status_upper"] = df.get("status", "").astype(str).str.upper()

        closed_mask = df["status_upper"] == "CLOSED"
        df["is_closed"] = closed_mask.astype(int)
        df["is_open"] = df["status_upper"].isin(["OPEN", "PENDING_CONFIRMATION"]).astype(int)
        df["is_failed"] = df["status_upper"].eq("FAILED").astype(int)
        df["is_skipped"] = df["status_upper"].str.startswith("SKIPPED").astype(int)
        df["is_win"] = (closed_mask & (df["close_profit_net"].fillna(0.0) > 0)).astype(int)
        df["is_loss"] = (closed_mask & (df["close_profit_net"].fillna(0.0) < 0)).astype(int)

        holding_hours = (
            (df["close_time"] - df["signal_time"]).dt.total_seconds() / 3600.0
        )
        df["holding_hours"] = holding_hours.where(closed_mask, np.nan)

        group_cols = [
            "entry_date",
            "strategy_profile",
            "release_id",
            "magic_number",
            "symbol",
            "timeframe",
            "model",
        ]
        for col in group_cols:
            if col not in df.columns:
                df[col] = pd.NA
        daily = (
            df.groupby(group_cols, dropna=False)
            .agg(
                signals=("signal_id", "count"),
                closed_trades=("is_closed", "sum"),
                open_trades=("is_open", "sum"),
                failed_trades=("is_failed", "sum"),
                skipped_trades=("is_skipped", "sum"),
                wins=("is_win", "sum"),
                losses=("is_loss", "sum"),
                net_profit=("close_profit_net", "sum"),
                avg_profit=("close_profit_net", "mean"),
                avg_holding_hours=("holding_hours", "mean"),
            )
            .reset_index()
        )

        daily["win_rate_closed"] = np.where(
            daily["closed_trades"] > 0,
            daily["wins"] / daily["closed_trades"] * 100.0,
            np.nan,
        )
        daily.to_csv(paths["daily"], index=False)
        return daily

    def _sync_live_trade_report(self) -> pd.DataFrame:
        """Actualiza el estado de trades ejecutados usando posiciones y deals de MT5."""
        paths = self._get_production_output_paths()
        lifecycle_path = paths["lifecycle"]

        if not lifecycle_path.exists():
            return pd.DataFrame()

        try:
            lifecycle = pd.read_csv(lifecycle_path)
        except EmptyDataError:
            return pd.DataFrame()
        if lifecycle.empty:
            paths["closed"].write_text("", encoding="utf-8")
            paths["daily"].write_text("", encoding="utf-8")
            return lifecycle

        try:
            mt5_client = self._ensure_mt5_client()
        except Exception as e:
            self.logger.warning(f"No se pudo conectar a MT5 para sincronizar trades: {e}")
            return lifecycle

        self._update_daily_loss_guard_state(mt5_client=mt5_client)

        live_cfg = self._get_live_trading_settings()
        open_positions = mt5_client.get_all_positions()
        if open_positions is None or open_positions.empty:
            open_position_ids: set[int] = set()
        else:
            open_position_ids = set()
            for col in ["ticket", "identifier"]:
                if col in open_positions.columns:
                    open_position_ids.update(
                        pd.to_numeric(open_positions[col], errors="coerce").dropna().astype(int).tolist()
                    )

        # MT5 puede devolver tiempos del broker varias horas por delante del reloj local.
        # Extendemos la ventana hacia el futuro para no perder cierres manuales recientes.
        deals_date_from = datetime.now() - timedelta(days=live_cfg["report_lookback_days"])
        deals_date_to = datetime.now() + timedelta(days=1)
        deals = mt5_client.get_history_deals(
            date_from=deals_date_from,
            date_to=deals_date_to,
            magic=live_cfg["magic_number"],
        )
        all_deals = mt5_client.get_history_deals(
            date_from=deals_date_from,
            date_to=deals_date_to,
        )

        lifecycle["status"] = lifecycle["status"].astype(str)
        now_iso = datetime.now().isoformat()
        changed = False

        for idx, row in lifecycle.iterrows():
            status = str(row.get("status", "")).upper()
            if status not in {"OPEN", "PENDING_CONFIRMATION"}:
                continue

            position_value = pd.to_numeric(pd.Series([row.get("mt5_position_id")]), errors="coerce").iloc[0]
            if pd.isna(position_value):
                continue
            position_id = int(position_value)

            lifecycle.at[idx, "last_sync_time"] = now_iso
            changed = True

            if position_id in open_position_ids:
                if open_positions is not None and not open_positions.empty and "ticket" in open_positions.columns:
                    pos_rows = open_positions[
                        pd.to_numeric(open_positions["ticket"], errors="coerce").fillna(-1).astype(int) == position_id
                    ].copy()
                    if pos_rows.empty and "identifier" in open_positions.columns:
                        pos_rows = open_positions[
                            pd.to_numeric(open_positions["identifier"], errors="coerce").fillna(-1).astype(int) == position_id
                        ].copy()

                    if not pos_rows.empty:
                        pos_row = pos_rows.iloc[0]
                        current_sl = pd.to_numeric(pd.Series([pos_row.get("sl")]), errors="coerce").iloc[0]
                        current_tp = pd.to_numeric(pd.Series([pos_row.get("tp")]), errors="coerce").iloc[0]
                        lifecycle.at[idx, "applied_sl_price"] = current_sl
                        lifecycle.at[idx, "applied_tp_price"] = current_tp

                        requested_sl = pd.to_numeric(pd.Series([row.get("requested_sl_price")]), errors="coerce").iloc[0]
                        requested_tp = pd.to_numeric(pd.Series([row.get("requested_tp_price")]), errors="coerce").iloc[0]
                        needs_sl = pd.notna(requested_sl) and (pd.isna(current_sl) or abs(float(current_sl)) <= 0.0)
                        needs_tp = pd.notna(requested_tp) and (pd.isna(current_tp) or abs(float(current_tp)) <= 0.0)

                        if needs_sl or needs_tp:
                            protection = mt5_client.ensure_position_protection(
                                symbol=str(row.get("symbol", "")),
                                position_ticket=position_id,
                                side=str(row.get("signal", "")).upper(),
                                sl=None if pd.isna(requested_sl) else float(requested_sl),
                                tp=None if pd.isna(requested_tp) else float(requested_tp),
                            )
                            lifecycle.at[idx, "protection_status"] = (
                                "PROTECTED" if protection.get("success") else "UNPROTECTED"
                            )
                            lifecycle.at[idx, "protection_comment"] = protection.get("comment")
                            lifecycle.at[idx, "applied_sl_price"] = protection.get("applied_sl")
                            lifecycle.at[idx, "applied_tp_price"] = protection.get("applied_tp")
                            changed = True
                continue

            if deals is None or deals.empty or "position_id" not in deals.columns:
                deals_pos = pd.DataFrame()
            else:
                deals_pos = deals[
                    pd.to_numeric(deals["position_id"], errors="coerce").fillna(-1).astype(int) == position_id
                ].copy()

            used_unfiltered_history = False
            if deals_pos.empty:
                if all_deals is None or all_deals.empty or "position_id" not in all_deals.columns:
                    continue
                deals_pos = all_deals[
                    pd.to_numeric(all_deals["position_id"], errors="coerce").fillna(-1).astype(int) == position_id
                ].copy()
                used_unfiltered_history = not deals_pos.empty

            if deals_pos.empty:
                continue

            if "entry" in deals_pos.columns:
                exit_mask = pd.to_numeric(deals_pos["entry"], errors="coerce").isin([1, 3])
                exit_deals = deals_pos[exit_mask].copy()
            else:
                exit_deals = deals_pos.iloc[1:].copy()

            if exit_deals.empty:
                if not used_unfiltered_history and all_deals is not None and not all_deals.empty and "position_id" in all_deals.columns:
                    deals_pos = all_deals[
                        pd.to_numeric(all_deals["position_id"], errors="coerce").fillna(-1).astype(int) == position_id
                    ].copy()
                    used_unfiltered_history = not deals_pos.empty
                    if "entry" in deals_pos.columns:
                        exit_mask = pd.to_numeric(deals_pos["entry"], errors="coerce").isin([1, 3])
                        exit_deals = deals_pos[exit_mask].copy()
                    else:
                        exit_deals = deals_pos.iloc[1:].copy()

                if exit_deals.empty:
                    continue

            exit_deals = exit_deals.sort_values("time")
            exit_deal = exit_deals.iloc[-1]

            net_profit = 0.0
            for col in ["profit", "commission", "swap", "fee"]:
                if col in deals_pos.columns:
                    net_profit += pd.to_numeric(deals_pos[col], errors="coerce").fillna(0.0).sum()

            lifecycle.at[idx, "status"] = "CLOSED"
            lifecycle.at[idx, "close_time"] = (
                pd.to_datetime(exit_deal.get("time"), errors="coerce").isoformat()
                if pd.notna(pd.to_datetime(exit_deal.get("time"), errors="coerce"))
                else exit_deal.get("time")
            )
            lifecycle.at[idx, "close_price"] = pd.to_numeric(
                pd.Series([exit_deal.get("price")]), errors="coerce"
            ).iloc[0]
            lifecycle.at[idx, "close_profit_net"] = float(net_profit)
            lifecycle.at[idx, "close_reason"] = exit_deal.get("reason_label", exit_deal.get("comment"))
            lifecycle.at[idx, "close_deal_ticket"] = exit_deal.get("ticket")
            if used_unfiltered_history:
                close_reason = str(lifecycle.at[idx, "close_reason"] or "").upper()
                if close_reason == "CLIENT":
                    lifecycle.at[idx, "status_detail"] = "Cierre manual detectado en MT5."
                else:
                    lifecycle.at[idx, "status_detail"] = "Cierre detectado via historial MT5 sin filtro de magic."
            changed = True

        if changed:
            lifecycle.to_csv(lifecycle_path, index=False)

        closed = lifecycle[lifecycle["status"].astype(str).str.upper() == "CLOSED"].copy()
        closed.to_csv(paths["closed"], index=False)
        self._save_daily_trade_report(lifecycle)
        return lifecycle

    def _execute_live_orders(self, df_rows: pd.DataFrame) -> None:
        """Ejecuta órdenes reales en MT5 para las señales elegibles."""
        if df_rows is None or df_rows.empty:
            return

        live_cfg = self._get_live_trading_settings()
        if not live_cfg["auto_execute_orders"]:
            return

        try:
            mt5_client = self._ensure_mt5_client()
        except Exception as e:
            self.logger.error(f"No se pudo conectar a MT5 para ejecutar órdenes: {e}")
            return

        daily_guard_state = self._update_daily_loss_guard_state(mt5_client=mt5_client)

        self._sync_live_trade_report()

        paths = self._get_production_output_paths()
        if paths["lifecycle"].exists():
            try:
                lifecycle = pd.read_csv(paths["lifecycle"])
            except EmptyDataError:
                lifecycle = pd.DataFrame()
        else:
            lifecycle = pd.DataFrame()

        existing_signal_ids = set()
        if not lifecycle.empty and "signal_id" in lifecycle.columns:
            existing_signal_ids = set(lifecycle["signal_id"].dropna().astype(str).tolist())

        open_positions = mt5_client.get_all_positions()
        execution_rows: list[dict[str, Any]] = []

        for _, row in df_rows.iterrows():
            signal = str(row.get("signal", "HOLD")).upper()
            if signal not in {"BUY", "SELL"}:
                continue

            signal_id = self._build_signal_id(row)
            if signal_id in existing_signal_ids:
                self.logger.info(f"⏭ Señal ya ejecutada anteriormente, se omite: {signal_id}")
                continue

            symbol = str(row.get("symbol", ""))
            model_name = str(row.get("model", "UNKNOWN"))
            volume_value = pd.to_numeric(pd.Series([row.get("volume_lots")]), errors="coerce").iloc[0]
            volume_lots = 0.0 if pd.isna(volume_value) else float(volume_value)
            live_sl_value = pd.to_numeric(pd.Series([row.get("live_sl_price")]), errors="coerce").iloc[0]
            live_tp_value = pd.to_numeric(pd.Series([row.get("live_tp_price")]), errors="coerce").iloc[0]
            sl_price = live_sl_value if pd.notna(live_sl_value) else pd.to_numeric(
                pd.Series([row.get("sl_price")]), errors="coerce"
            ).iloc[0]
            tp_price = live_tp_value if pd.notna(live_tp_value) else pd.to_numeric(
                pd.Series([row.get("tp_price")]), errors="coerce"
            ).iloc[0]

            base_record = {
                "signal_id": signal_id,
                "signal_time": pd.to_datetime(row.get("timestamp"), errors="coerce"),
                "execution_time": datetime.now().isoformat(),
                "release_id": row.get("release_id"),
                "strategy_profile": row.get("strategy_profile"),
                "symbol": symbol,
                "timeframe": row.get("timeframe"),
                "model": model_name,
                "signal": signal,
                "confidence": row.get("confidence"),
                "pred_return": row.get("pred_return"),
                "requested_entry_price": row.get("entry_price"),
                "requested_live_entry_price": row.get("live_entry_price"),
                "requested_sl_price": sl_price,
                "requested_tp_price": tp_price,
                "requested_plan_sl_price": row.get("sl_price"),
                "requested_plan_tp_price": row.get("tp_price"),
                "requested_volume_lots": volume_lots,
                "risk_amount": row.get("risk_amount"),
                "allocated_risk_budget": row.get("allocated_risk_budget"),
                "risk_per_pip_per_lot": row.get("risk_per_pip_per_lot"),
                "risk_per_lot_at_stop": row.get("risk_per_lot_at_stop"),
                "open_risk_amount": row.get("open_risk_amount"),
                "remaining_risk_budget_before_trade": row.get("remaining_risk_budget_before_trade"),
                "projected_total_open_risk_after_trade": row.get("projected_total_open_risk_after_trade"),
                "is_best_model": row.get("is_best_model"),
                "magic_number": live_cfg["magic_number"],
                "order_comment_prefix": live_cfg["order_comment_prefix"],
                "status": "PENDING_CONFIRMATION",
                "status_detail": "",
                "mt5_order_ticket": None,
                "mt5_deal_ticket": None,
                "mt5_position_id": None,
                "execution_price": None,
                "execution_retcode": None,
                "execution_comment": None,
                "applied_sl_price": None,
                "applied_tp_price": None,
                "protection_status": None,
                "protection_comment": None,
                "close_time": None,
                "close_price": None,
                "close_profit_net": None,
                "close_reason": None,
                "close_deal_ticket": None,
                "last_sync_time": None,
            }

            if daily_guard_state.get("halt_active"):
                base_record["status"] = "SKIPPED_DAILY_LOSS_LIMIT"
                base_record["status_detail"] = (
                    f"Kill switch diario activo: perdida {float(daily_guard_state.get('daily_loss_pct', 0.0)) * 100.0:.2f}% "
                    f"con limite {float(daily_guard_state.get('daily_loss_limit_pct', 0.0)) * 100.0:.2f}%."
                )
                execution_rows.append(base_record)
                existing_signal_ids.add(signal_id)
                continue

            if volume_lots <= 0:
                base_record["status"] = "SKIPPED_NO_VOLUME"
                base_record["status_detail"] = "El tamaño de posición calculado fue <= 0."
                execution_rows.append(base_record)
                existing_signal_ids.add(signal_id)
                continue

            valid_trade, validation_detail = self._validate_pre_trade_execution(
                row=row,
                balance=float(pd.to_numeric(pd.Series([row.get("account_balance")]), errors="coerce").iloc[0] or 0.0),
            )
            if not valid_trade:
                base_record["status"] = "SKIPPED_RISK_VALIDATION"
                base_record["status_detail"] = validation_detail
                execution_rows.append(base_record)
                existing_signal_ids.add(signal_id)
                continue

            if (
                not live_cfg["allow_multiple_positions"]
                and open_positions is not None
                and not open_positions.empty
                and "symbol" in open_positions.columns
            ):
                same_symbol = open_positions[open_positions["symbol"] == symbol]
                if not same_symbol.empty:
                    base_record["status"] = "SKIPPED_OPEN_POSITION"
                    base_record["status_detail"] = f"Ya existe una posición abierta para {symbol}."
                    execution_rows.append(base_record)
                    existing_signal_ids.add(signal_id)
                    continue

            result = mt5_client.open_market_order(
                symbol=symbol,
                volume=volume_lots,
                side=signal,
                comment=self._build_order_comment(model_name),
                sl=None if pd.isna(sl_price) else float(sl_price),
                tp=None if pd.isna(tp_price) else float(tp_price),
                deviation=live_cfg["order_deviation_points"],
                magic=live_cfg["magic_number"],
            )

            base_record["status"] = "OPEN" if result.get("success") else "FAILED"
            base_record["status_detail"] = result.get("comment")
            base_record["mt5_order_ticket"] = result.get("order")
            base_record["mt5_deal_ticket"] = result.get("deal")
            base_record["mt5_position_id"] = result.get("position_id")
            base_record["execution_price"] = result.get("price")
            base_record["execution_retcode"] = result.get("retcode")
            base_record["execution_comment"] = result.get("comment")
            protection = result.get("protection") or {}
            base_record["applied_sl_price"] = protection.get("applied_sl", result.get("sent_sl"))
            base_record["applied_tp_price"] = protection.get("applied_tp", result.get("sent_tp"))
            base_record["protection_status"] = "PROTECTED" if protection.get("success") else "UNPROTECTED"
            base_record["protection_comment"] = protection.get("comment")
            execution_rows.append(base_record)
            existing_signal_ids.add(signal_id)

            if result.get("success"):
                self.logger.info(
                    f"✅ Orden enviada: model={model_name} signal={signal} symbol={symbol} "
                    f"lots={volume_lots:.2f} position_id={result.get('position_id')} "
                    f"SL={base_record['applied_sl_price']} TP={base_record['applied_tp_price']} "
                    f"protection={base_record['protection_status']}"
                )
                open_positions = mt5_client.get_all_positions()
            else:
                self.logger.error(
                    f"❌ Falló la ejecución de {model_name} en {symbol}: {result.get('comment')} "
                    f"(retcode={result.get('retcode')})"
                )

        if execution_rows:
            df_exec = pd.DataFrame(execution_rows)
            self._append_rows_to_csv(paths["lifecycle"], df_exec)
            self._sync_live_trade_report()
    
    def _save_backtest_detail(self, model_name: str, df_bt: pd.DataFrame) -> None:
        """
        Guarda el detalle del mejor backtest para cada modelo.
        Crea CSV y, opcionalmente, Excel con señales, precios y pips.
        """
        if df_bt is None or df_bt.empty:
            return

        output_dir = self._get_backtest_output_dir()

        csv_path = output_dir / f"{model_name}_best_backtest_detail.csv"
        df_bt.to_csv(csv_path)
        self.logger.info(f"    💾 Detalle de backtest guardado en: {csv_path}")
        self._archive_backtest_artifact(csv_path)

        # Si quieres también Excel
        if "excel" in self.config.get("output", {}).get("formats", []):
            xlsx_path = output_dir / f"{model_name}_best_backtest_detail.xlsx"
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                df_bt.to_excel(writer, sheet_name="backtest_detail")
            self.logger.info(f"    💾 Detalle de backtest (Excel) guardado en: {xlsx_path}")
            self._archive_backtest_artifact(xlsx_path)

    
    def _load_config(self, config_path: str) -> tuple[Dict[str, Any], str]:
        """Carga configuración desde YAML"""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"El archivo de configuración no se encontró en: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print(f"Configuracion cargada desde: {config_path}")
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
            log_file = self._build_dated_log_path(log_file)
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
        output_root = self.config.get("output", {}).get("dir", "outputs")
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
        self._active_mode = str(mode).lower()
        
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
        elif mode == "sync_trades":
            self._run_sync_trades_mode()
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
        self._start_backtest_run()
        
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
        target_col = self.config.get("backtest", {}).get("target", "ReturnFwd_1")
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
        optimized_config_path = self._resolve_active_release_config_path()
        if not optimized_config_path.exists():
            self.logger.error("No se encontró 'config_optimizado.yaml'. Ejecute el backtest primero.")
            return
        
        # Crear un nuevo pipeline temporal para la validación
        validation_pipeline = TradingPipeline(config_path=str(optimized_config_path))
        
        # Preparar datos de test
        y_test = df_test[target_col]
        feature_cols = self._get_model_feature_columns(df_test, target_col)
        X_test = df_test[feature_cols]

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
        Modo BACKTEST:
        - Carga datos históricos
        - Genera features
        - (Opcional) Reserva un hold-out final según config['validation']
        - Ejecuta búsqueda de hiperparámetros SOLO sobre la parte in-sample
        - Guarda resultados y deja preparado self._df_features_last_backtest
          para reentrenar los modelos óptimos.
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("MODO: BACKTEST")
        self.logger.info("=" * 60 + "\n")
        self._start_backtest_run()

        # 1) Cargar y procesar datos
        df = self._load_data()
        df_clean = self._clean_data(df)
        df_features = self._generate_features(df_clean)

        target_col = self.config.get("backtest", {}).get("target", "ReturnFwd_1")

        if target_col not in df_features.columns:
            raise KeyError(
                f"El target '{target_col}' no existe en df_features. "
                f"Revisa tu FeatureEngineer y/o config."
            )

        # ✅ Limpiar NaNs de target + features (sin bfill para evitar leakage)
        feature_cols = self._get_model_feature_columns(df_features, target_col)
        df_features = df_features.dropna(subset=[target_col] + feature_cols).copy()

        # 2) Aplicar hold-out opcional (validation.mode) UNA sola vez
        val_cfg = self.config.get("validation", {})
        mode = str(val_cfg.get("mode", "none")).lower()
        n_holdout = int(val_cfg.get("n", 0))

        if mode == "last_n" and n_holdout > 0 and len(df_features) > n_holdout:
            df_bt = df_features.iloc[:-n_holdout].copy()
            df_holdout = df_features.iloc[-n_holdout:].copy()  # opcional, por si luego lo usas
            self.logger.info(
                f"🔒 Hold-out activado: se reservan los últimos {n_holdout} puntos "
                f"({len(df_bt)} usados para backtest)."
            )
        else:
            df_bt = df_features
            df_holdout = None
            if mode == "last_n" and n_holdout > 0:
                self.logger.warning(
                    f"validation.mode=last_n pero n={n_holdout} es mayor o igual "
                    f"al tamaño de la serie ({len(df_features)}). Se ignora hold-out."
                )
            else:
                self.logger.info("Sin hold-out: se usa toda la serie para backtest.")

        # 💾 Guardar features IN-SAMPLE para reentrenar modelos óptimos
        self._df_features_last_backtest = df_bt.copy()

        # 3) Ejecutar tuning de hiperparámetros sobre df_bt (in-sample)
        self._run_hyperparameter_tuning(df_bt)


        self.logger.info("\n✅ MODO BACKTEST COMPLETADO")


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

            if "params" in model_config:
                param_grid = model_config["params"]
            else:
                param_grid = model_config.get("param_grid", {})

            grid = ParameterGrid(param_grid)
            model_results = []
            selection_rows = []
            series_candidates = []

            for i, params in enumerate(grid):
                self.logger.info(f"  -> Probando combinación {i+1}/{len(grid)}: {params}")

                # Devuelve predicciones, valores reales, fechas y filtro opcional de confirmación.
                predictions, true_values, timestamps, trade_mask, confirmation_reasons = self._run_walk_forward_for_params(
                    df_features, model_name, params
                )

                if not predictions:
                    self.logger.warning("    No se generaron predicciones, saltando métricas.")
                    continue

                metrics = self._calculate_metrics(true_values, predictions, trade_mask=trade_mask)
                self.logger.info(f"    - Métricas: {metrics}")

                result_row = {"model": model_name, **params, **metrics}
                model_results.append(result_row)
                all_results.append(result_row)
                selection_rows.append({**result_row, "_artifact_idx": len(series_candidates)})
                series_candidates.append(
                    {
                        "dates": timestamps,
                        "y_true": true_values,
                        "y_pred": predictions,
                        "trade_mask": trade_mask,
                        "confirmation_reasons": confirmation_reasons,
                        "params": params,
                    }
                )

            # ==================== CAMBIO IMPORTANTE =====================
            # Guardamos y graficamos la serie del run ganador según
            # config.model_selection, no según una métrica hardcodeada.
            best_series = None
            if selection_rows:
                best_row = self._select_best_run(
                    pd.DataFrame(selection_rows),
                    model_name=model_name,
                    log_prefix="  -> Serie best ",
                )
                if best_row is not None and "_artifact_idx" in best_row.index:
                    best_series = series_candidates[int(best_row["_artifact_idx"])]

            if best_series is not None:
                # Guardar CSV con la serie de backtest del mejor run
                self._save_backtest_series(
                    model_name=model_name,
                    params=best_series["params"],
                    y_true=best_series["y_true"],
                    y_pred=best_series["y_pred"],
                    dates=best_series["dates"],  # <-- AQUÍ VA 'dates'
                    trade_mask=best_series.get("trade_mask"),
                    confirmation_reason=best_series.get("confirmation_reasons"),
                )

                # Generar gráfico para la mejor combinación de este modelo
                self._plot_predictions_series(
                    dates=best_series["dates"],
                    y_true=best_series["y_true"],
                    y_pred=best_series["y_pred"],
                    model_name=model_name,
                    params=best_series["params"],
                    suffix="_best",
                )
            # ==================== FIN CAMBIO IMPORTANTE =====================

            # Guardar reporte detallado para este modelo (como antes)
            if model_results:
                self._save_model_report(model_name, model_results)

        # Guardar resumen consolidado y config optimizada (como ya tenías)
        if all_results:
            self._save_consolidated_summary(all_results)
            self._find_and_save_best_params(all_results, df_features)


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

        target_col = self.config.get("backtest", {}).get("target", "ReturnFwd_1")

        df_processed = df_features.dropna(subset=[target_col]).bfill().ffill()
        if len(df_processed) <= n + 10:
            self.logger.error("No hay suficientes datos para una validación con last_n=%s", n)
            return

        df_train = df_processed.iloc[:-n]
        df_test = df_processed.iloc[-n:]

        features_cols = self._get_model_feature_columns(df_processed, target_col)
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
        trade_mask = []
        bt_rows = []
        close_prices = df_processed["Close"] if "Close" in df_processed.columns else None

        # Entrenamos una vez con df_train completo y vamos moviendo la ventana sobre df_test
        model_class_map = {
            "RandomWalk": RandomWalkModel,
            "RandomWalkModel": RandomWalkModel,
            "Momentum": MomentumModel,
            "MomentumModel": MomentumModel,
            "ARIMA": ArimaModel,
            "PROPHET": ProphetModel,
            "LSTM": LSTMModel,
            "RandomForestRegressor": RandomForestRegressorModel,
            "HistGradientBoosting": HistGradientBoostingRegressorModel,
        }
        model_class = model_class_map.get(model_name)
        if model_class is None:
            self.logger.error(f"Modelo '{model_name}' no soportado en modo test.")
            return

        # Entrenar modelo una vez con todo df_train
        model_instance = model_class(params=params, logger=self.logger)
        trading_cfg = self.config.get("trading", {}) or {}
        pip_size = float(self.config.get("backtest", {}).get("pip_size", 0.0001))
        min_pips_signal = float(
            trading_cfg.get(
                "min_pips_signal",
                self.config.get("backtest", {}).get("threshold_pips", 0.0),
            )
        )
        enable_confidence_filter = bool(trading_cfg.get("enable_confidence_filter", False))
        min_confidence = float(trading_cfg.get("min_confidence", 0.60))
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

            signal_info = build_signal_from_prediction(
            pred_return=pred,
            pip_size=pip_size,
            min_pips_signal=min_pips_signal,
            model_metrics={},
            min_confidence=min_confidence if enable_confidence_filter else 0.0,
            probability=None,
        )

            signal = str(signal_info["signal"])
            confidence = float(signal_info["confidence"])
            confirmation = self._evaluate_signal_confirmation(
                signal=signal,
                feature_row=X_te.iloc[0] if not X_te.empty else None,
            )
            trade_allowed = signal in {"BUY", "SELL"} and bool(confirmation.get("passed", True))

            true_sign = np.sign(true_val)
            pred_sign = np.sign(pred)

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
                "confidence": confidence,
                "trade_allowed": trade_allowed,
                "signal_filter_reason": confirmation.get("reason"),
                "price_prev": price_prev,
                "price_true": price_true,
                "price_pred": price_pred,
                "delta_price": delta_price,
            })
            trade_mask.append(trade_allowed)

        if not all_pred:
            self.logger.error("No se generaron predicciones en validación.")
            return

        # 7. Métricas de validación
        metrics = self._calculate_metrics(all_true, all_pred, trade_mask=trade_mask)
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
        
    def _find_and_save_best_params(self, all_results: list[dict], df_features: pd.DataFrame) -> None:
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
            "f1_score",
            "precision",
            "recall",
            "dm_stat",
            "dm_pvalue",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "profit_factor",
            "win_rate",
            "payoff_ratio",
            "consistency_ratio",
            "avg_trade_return",
            "n_test_points",
            "n_trades",
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
        selection_cfg = self._get_model_selection_settings()
        primary_metric = selection_cfg["primary_metric"]
        secondary_metric = selection_cfg["secondary_metric"]
        best_rows_for_global: list[dict[str, Any]] = []

        # 3. Por cada modelo (ARIMA, PROPHET, LSTM, RandomWalk, etc.) encontrar la mejor fila
        for model_name in df["model"].dropna().unique():
            model_df = df[df["model"] == model_name].copy()
            if model_df.empty:
                continue

            best_run = self._select_best_run(
                model_df,
                model_name=model_name,
                log_prefix="  -> Selección ",
            )
            if best_run is None:
                continue

            # Hiperparámetros = todas las columnas excepto métricas + 'model'
            param_cols = [c for c in model_df.columns if c not in metric_cols + ["model"]]
            raw_params = {k: best_run[k] for k in param_cols}

            clean_params = {k: to_native(v) for k, v in raw_params.items() if not is_nan(v)}

            # Log informativo
            p_val = best_run[primary_metric] if primary_metric in best_run.index else None
            s_val = best_run[secondary_metric] if secondary_metric in best_run.index else None
            t_val = best_run["n_trades"] if "n_trades" in best_run.index else None

            self.logger.info(
                f"  -> Mejor para {model_name}: "
                f"{primary_metric}={to_native(p_val)} | {secondary_metric}={to_native(s_val)} | "
                f"n_trades={to_native(t_val)} | params={clean_params}"
            )

            best_rows_for_global.append(best_run.to_dict())
            best_models.append({"name": model_name, "enabled": True, "params": clean_params})

        if not best_models:
            self.logger.warning("No se encontró ningún mejor modelo para guardar en config_optimizado.")
            return

        global_champion_name = None
        if best_rows_for_global:
            champion_row = self._select_best_run(
                pd.DataFrame(best_rows_for_global),
                log_prefix="  -> Campeón global ",
            )
            if champion_row is not None and "model" in champion_row.index:
                global_champion_name = str(champion_row["model"])
                self.logger.info(f"🏆 Modelo campeón global: {global_champion_name}")

        self._global_champion = global_champion_name
        for model_cfg in best_models:
            model_cfg["is_best"] = bool(
                global_champion_name and str(model_cfg.get("name")) == global_champion_name
            )

        release_id = self._ensure_backtest_run_label()

        # 4. Construir config optimizado: copiamos config actual y reemplazamos sólo la sección de modelos
        optimized_config = dict(self.config)
        optimized_config["models"] = best_models

        base_config_path = Path(self.config_path)
        versioned_config_path = base_config_path.parent / f"config_optimizado_{release_id}.yaml"
        self._write_yaml_atomic(versioned_config_path, optimized_config)

        self.logger.info(f"\n💾 Configuración optimizada versionada guardada en: {versioned_config_path}")

        # 5. Reentrenar y guardar modelos finales (si tenemos features del último backtest)
        if self._df_features_last_backtest is None:
            self.logger.warning(
                "    -> self._df_features_last_backtest es None. "
                "No se reentrenan ni se guardan modelos en disco."
            )
            return

        target_col = self.config.get("backtest", {}).get("target", "ReturnFwd_1")

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

        feature_cols = self._get_model_feature_columns(df_proc, target_col)
        X_full = df_proc[feature_cols]
        y_full = df_proc[target_col]

        models_dir = self._get_release_models_dir(release_id)

        model_class_map = {
            "RandomWalk": RandomWalkModel,
            "RandomWalkModel": RandomWalkModel,
            "Momentum": MomentumModel,
            "MomentumModel": MomentumModel,
            "ARIMA": ArimaModel,
            "PROPHET": ProphetModel,
            "LSTM": LSTMModel,
            "RandomForestRegressor": RandomForestRegressorModel,
            "HistGradientBoosting": HistGradientBoostingRegressorModel,
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

        self._publish_active_release(
            release_id=release_id,
            optimized_config=optimized_config,
            versioned_config_path=versioned_config_path,
            models_dir=models_dir,
            champion_name=global_champion_name,
            profile_name=self._get_strategy_profile_name(),
        )

        self.logger.info("\n✅ Proceso de optimización y guardado de modelos completado.")
        
    def _save_model_report(self, model_name: str, model_results: list[dict]) -> None:
        """Guarda el reporte detallado de un modelo en un archivo CSV."""
        if not model_results:
            return

        output_dir = self._get_backtest_output_dir()
        
        report_path = output_dir / f"report_{model_name}.csv"
        df_report = pd.DataFrame(model_results)
        
        # Ordenar usando criterios de selección (primario y secundario)
        ms = self.config.get("model_selection", {}) or {}
        primary = ms.get("primary_metric", "hit_rate")
        secondary = ms.get("secondary_metric", "rmse")
        primary_greater = bool(ms.get("primary_greater_is_better", True))
        secondary_greater = bool(ms.get("secondary_greater_is_better", False))

        # Asegurar numéricos para ordenar
        for c in [primary, secondary, "n_trades"]:
            if c in df_report.columns:
                df_report[c] = pd.to_numeric(df_report[c], errors="coerce")

        sort_cols = []
        ascending = []

        if primary in df_report.columns:
            sort_cols.append(primary)
            ascending.append(not primary_greater)  # hit_rate: desc

        if secondary in df_report.columns:
            sort_cols.append(secondary)
            ascending.append(not secondary_greater)  # rmse: asc

        # Desempate: preferir más trades si existe
        if "n_trades" in df_report.columns:
            sort_cols.append("n_trades")
            ascending.append(False)

        if sort_cols:
            df_report = df_report.sort_values(by=sort_cols, ascending=ascending, na_position="last")

            
        df_report.to_csv(report_path, index=False)
        self.logger.info(f"    💾 Reporte para {model_name} guardado en: {report_path}")
        self._archive_backtest_artifact(report_path)
        
    def _save_backtest_series(
        self,
        model_name: str,
        params: Dict[str, Any],
        y_true: List[float],
        y_pred: List[float],
        # CAMBIO: dates ahora es OPCIONAL (tiene default = None)
        dates: Optional[List[pd.Timestamp]] = None,
        trade_mask: Optional[List[bool]] = None,
        confirmation_reason: Optional[List[str]] = None,
    ) -> None:
        """
        Guarda la serie completa de backtest (y_true, y_pred, error y fechas opcionales)
        para poder graficar después los errores / predicciones.

        Se guarda en:
            outputs/backtest/{model_name}_{param_suffix}_series.csv
        """

        # ==================== NUEVO: construir DataFrame ====================
        data = {
            "y_true": y_true,
            "y_pred": y_pred,
        }

        # Si recibimos fechas y tienen la misma longitud, las incluimos
        if dates is not None and len(dates) == len(y_true):
            data["date"] = dates
        if trade_mask is not None and len(trade_mask) == len(y_true):
            data["trade_allowed"] = trade_mask
        if confirmation_reason is not None and len(confirmation_reason) == len(y_true):
            data["signal_filter_reason"] = confirmation_reason

        df_series = pd.DataFrame(data)
        df_series["error"] = df_series["y_true"] - df_series["y_pred"]
        # ===================================================================

        # ==================== IGUAL QUE ANTES: sufijo params ===============
        if params:
            # Convertir params en un sufijo legible para el nombre del archivo
            param_parts = []
            for k, v in params.items():
                param_parts.append(f"{k}-{str(v)}")
            param_suffix = "_".join(param_parts)
        else:
            param_suffix = "default"
        # ===================================================================

        # ==================== CAMBIO IMPORTANTE AQUÍ ========================
        # ANTES usábamos: self.paths["backtest_dir"]  -> pero self.paths NO existe
        # Usamos un directorio fijo dentro de 'outputs/backtest'
        backtest_dir = self._get_backtest_output_dir()

        file_name = f"{model_name}_{param_suffix}_series.csv"
        file_path = backtest_dir / file_name
        # ===================================================================

        # Guardar CSV
        df_series.to_csv(file_path, index=False)
        self.logger.info(f"      ↳ Serie completa guardada en: {file_path}")
        self._archive_backtest_artifact(file_path)


    def _save_consolidated_summary(self, all_results) -> None:
        """Guarda un resumen consolidado (best run por modelo).

        Soporta:
        - list[dict] (una fila por run)
        - dict[str, list[dict]] (modelo -> runs)
        """
        if not all_results:
            return

        output_dir = Path(self.config.get("output", {}).get("dir", "outputs")) / "backtest"
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Normalizar resultados a una lista de filas ---
        rows = []
        if isinstance(all_results, dict):
            for model_name, runs in all_results.items():
                if runs is None:
                    continue
                if isinstance(runs, dict):
                    runs = [runs]
                for r in runs:
                    if r is None:
                        continue
                    row = dict(r)
                    row.setdefault("model", model_name)
                    rows.append(row)
        elif isinstance(all_results, list):
            rows = [dict(r) for r in all_results if r is not None]
        else:
            try:
                rows = [dict(r) for r in list(all_results) if r is not None]
            except Exception as e:
                self.logger.error(f"No fue posible construir el resumen consolidado: {e}")
                return

        if not rows:
            return

        df = pd.DataFrame(rows)
        if df.empty:
            return

        if "model" not in df.columns:
            self.logger.warning("Resumen consolidado omitido: no existe la columna 'model' en los resultados.")
            return

        # Forzar numérico en métricas clave
        for c in {"rmse", "hit_rate", "n_trades", "n_test_points"}:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        best_rows = []
        for model_name in df["model"].dropna().unique():
            best_row = self._select_best_run(
                df[df["model"] == model_name].copy(),
                model_name=model_name,
                log_prefix="Resumen ",
            )
            if best_row is None:
                continue
            best_rows.append(best_row.to_dict())

        if not best_rows:
            return

        best_runs = pd.DataFrame(best_rows)
        champion_row = self._select_best_run(best_runs.copy(), log_prefix="Resumen global ")
        champion_model = None
        if champion_row is not None and "model" in champion_row.index:
            champion_model = str(champion_row["model"])
        if champion_model and "model" in best_runs.columns:
            best_runs["is_best"] = best_runs["model"].astype(str) == champion_model

        # Guardar outputs
        csv_path = output_dir / "summary_best_runs.csv"
        best_runs.to_csv(csv_path, index=False)
        self.logger.info(f"📄 Resumen consolidado guardado en: {csv_path}")
        csv_archive_path = self._archive_backtest_artifact(csv_path)
        self._latest_backtest_summary_paths["csv"] = csv_archive_path or csv_path

        xlsx_path = output_dir / "summary_best_runs.xlsx"
        try:
            best_runs.to_excel(xlsx_path, index=False)
            self.logger.info(f"📄 Resumen consolidado guardado en: {xlsx_path}")
            xlsx_archive_path = self._archive_backtest_artifact(xlsx_path)
            self._latest_backtest_summary_paths["xlsx"] = xlsx_archive_path or xlsx_path
        except Exception as e:
            self.logger.warning(f"No se pudo guardar XLSX del resumen: {e}")
            self._latest_backtest_summary_paths["xlsx"] = None
            
    def _run_walk_forward_for_params(
        self,
        df_features: pd.DataFrame,
        model_name: str,
        params: dict
    ) -> tuple[list, list, list]:
        """Ejecuta un backtest Walk-Forward para una configuración de modelo específica."""
        backtest_config = self.config.get("backtest", {})
        initial_train_size = int(backtest_config.get("initial_train", 800))
        step = int(backtest_config.get("step", 20))
        horizon = int(backtest_config.get("horizon", 1))
        target_col = backtest_config.get("target", "ReturnFwd_1")

        # 1) Definir features (basado en df_features)
        features_cols = self._get_model_feature_columns(df_features, target_col)

        # 2) Eliminar filas con NaNs en target + features (robusto para rolling indicators)
        df_processed = df_features.dropna(subset=[target_col] + features_cols).copy()

        if df_processed.empty:
            self.logger.warning("    -> df_processed quedó vacío tras dropna de target+features.")
            return [], [], [], [], []

        y = df_processed[target_col]
        X = df_processed[features_cols]

        if initial_train_size >= len(X):
            self.logger.warning(
                f"    -> No hay suficientes datos para el backtest con "
                f"initial_train_size={initial_train_size}. "
                f"Datos disponibles después de limpiar NaNs: {len(X)}. Saltando combinación."
            )
            return [], [], [], [], []

        # 3) Resolver timestamps: columna Date si existe, si no índice
        if "Date" in df_processed.columns:
            ts_all = pd.to_datetime(df_processed["Date"], errors="coerce")
        else:
            ts_all = pd.to_datetime(df_processed.index, errors="coerce")

        all_predictions: list = []
        all_true_values: list = []
        all_timestamps: list = []
        all_trade_mask: list[bool] = []
        all_confirmation_reasons: list[str] = []
        pip_size = float(backtest_config.get("pip_size", 0.0001))
        threshold_pips = float(backtest_config.get("threshold_pips", 0.0))

        for i in range(initial_train_size, len(X) - horizon + 1, step):
            train_end = i
            test_end = i + horizon

            X_train, X_test = X.iloc[:train_end], X.iloc[train_end:test_end]
            y_train, y_test = y.iloc[:train_end], y.iloc[train_end:test_end]

            if len(X_test) == 0:
                continue

            # Log de diagnóstico (igual que tenías)
            if self.logger.isEnabledFor(20):  # INFO
                nan_in_train = X_train.isnull().sum().sum()
                self.logger.info(
                    f"    -> Ventana {i-initial_train_size}: "
                    f"X_train shape={X_train.shape}, "
                    f"y_train len={len(y_train)}, "
                    f"NaNs en X_train={nan_in_train}"
                )

            prediction = self._train_and_predict(model_name, params, X_train, y_train, X_test)

            # Normalizar salida a lista con misma longitud que y_test
            if prediction is None:
                pred_list = [np.nan] * len(y_test)
            elif isinstance(prediction, (list, tuple, np.ndarray)):
                pred_list = list(prediction)
                if len(pred_list) != len(y_test):
                    # si devuelve algo raro, usamos el último valor como constante
                    last = pred_list[-1] if len(pred_list) else np.nan
                    pred_list = [last] * len(y_test)
            else:
                pred_list = [float(prediction)] * len(y_test)

            for pred_value, (_, x_row), true_value, y_index in zip(
                pred_list,
                X_test.iterrows(),
                y_test.values.tolist(),
                y_test.index.tolist(),
            ):
                candidate_signal = "HOLD"
                if pip_size > 0 and abs(float(pred_value) / pip_size) >= threshold_pips:
                    candidate_signal = "BUY" if float(pred_value) > 0 else "SELL"

                confirmation = self._evaluate_signal_confirmation(
                    signal=candidate_signal,
                    feature_row=x_row,
                )
                trade_allowed = candidate_signal in {"BUY", "SELL"} and bool(confirmation.get("passed", True))

                all_predictions.append(pred_value)
                all_true_values.append(true_value)
                all_timestamps.append(pd.to_datetime(y_index, errors="coerce"))
                all_trade_mask.append(trade_allowed)
                all_confirmation_reasons.append(str(confirmation.get("reason", "confirmation_disabled")))

        return all_predictions, all_true_values, all_timestamps, all_trade_mask, all_confirmation_reasons

    def _train_and_predict(self, model_name: str, params: dict, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> list | None:
        """Punto central para entrenar y predecir con un modelo específico."""
        
        model_map = {
            "RandomWalk": RandomWalkModel,
            "RandomWalkModel": RandomWalkModel,
            "Momentum": MomentumModel,
            "MomentumModel": MomentumModel,
            "ARIMA": ArimaModel,
            "PROPHET": ProphetModel,
            "LSTM": LSTMModel,
            "RandomForestRegressor": RandomForestRegressorModel,
            "HistGradientBoosting": HistGradientBoostingRegressorModel,
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

    def _calculate_metrics(self, y_true: list, y_pred: list, trade_mask: list[bool] | None = None) -> dict:
        """
        Calcula un conjunto de métricas de evaluación.

        - Calcula todas las métricas disponibles en utils.metrics_v2.calculate_all_metrics.
        - Aplica un umbral de pips (backtest.threshold_pips) para las métricas de TRADING.
        - Filtra las métricas a las listadas en config['backtest']['metrics'].
        """
        if not y_true or not y_pred:
            self.logger.warning("Listas de valores vacías para calcular métricas.")
            # Devolvemos también contadores en 0 para que las columnas existan en los CSV
            return {
                "rmse": np.nan,
                "mae": np.nan,
                "hit_rate": np.nan,
                "n_test_points": 0,
                "n_trades": 0,
            }

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
            active_mask_override=trade_mask,
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

    def _generate_backtest_plots_for_model(
        self,
        df_backtest: pd.DataFrame,
        y_true: list[float],
        y_pred: list[float],
        indices: list,
        model_name: str,
    ) -> None:
        """
        Genera los gráficos de backtest para el mejor run de un modelo:
        - Precio + puntos de entrada.
        - Curva de accuracy direccional.
        """
        if not y_true or not y_pred or not indices:
            self.logger.warning(f"No hay datos suficientes para graficar backtest de {model_name}.")
            return

        symbol = self.config.get("data", {}).get("symbol", "ASSET")
        price_col = self.config.get("eda", {}).get("price_col", "Close")

        output_root = Path(self.config.get("output", {}).get("dir", "outputs"))
        plot_dir = output_root / "backtest" / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Índices como Index de pandas
        idx = pd.Index(indices)

        # 1) Precio + puntos de entrada
        try:
            self._plot_price_with_entries(
                df_backtest=df_backtest,
                idx=idx,
                y_true=y_true,
                y_pred=y_pred,
                model_name=model_name,
                symbol=symbol,
                price_col=price_col,
                plot_dir=plot_dir,
            )
        except Exception as e:
            self.logger.warning(f"No se pudo generar gráfico de entradas para {model_name}: {e}")

        # 2) Curva de accuracy direccional
        try:
            self._plot_accuracy_curve(
                idx=idx,
                y_true=y_true,
                y_pred=y_pred,
                model_name=model_name,
                symbol=symbol,
                plot_dir=plot_dir,
            )
        except Exception as e:
            self.logger.warning(f"No se pudo generar curva de accuracy para {model_name}: {e}")

    def _plot_predictions_series(
        self,
        dates: list,
        y_true: list,
        y_pred: list,
        model_name: str,
        params: dict | None = None,
        suffix: str = ""
    ) -> None:
        """Genera y guarda un gráfico (en retornos o en precios rebajados) para un modelo."""
        if not dates or not y_true or not y_pred:
            self.logger.warning("No hay datos suficientes para graficar predicciones.")
            return

        output_dir = (
            Path(self.config.get("output", {}).get("dir", "outputs"))
            / "backtest"
            / "plots"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        df_plot = pd.DataFrame(
            {
                "y_true": y_true,
                "y_pred": y_pred,
            },
            index=pd.to_datetime(dates),
        )

        # --- NUEVO: decidir si graficamos retornos o “precios” ---
        plot_scale = self.config.get("backtest", {}).get("plot_scale", "returns")

        if plot_scale == "price":
            # Intentamos usar el precio de cierre real como base
            price_col = self.config.get("eda", {}).get("price_col", "Close")
            base_price = 1.0
            if hasattr(self, "df_clean") and price_col in self.df_clean.columns:
                try:
                    base_price = float(self.df_clean.loc[df_plot.index[0], price_col])
                except Exception as e:
                    self.logger.warning(
                        "No se pudo alinear el precio base (%s). Usando 1.0 como índice. Error: %s",
                        price_col,
                        e,
                    )

            # Construimos un “índice de precio” acumulando los retornos
            price_true = (1 + df_plot["y_true"]).cumprod() * base_price
            price_pred = (1 + df_plot["y_pred"]).cumprod() * base_price

            series_real = price_true
            series_pred = price_pred
            ylabel = f"Precio aproximado ({price_col}, rebajado)"
        else:
            # Comportamiento original: graficar retornos directamente
            series_real = df_plot["y_true"]
            series_pred = df_plot["y_pred"]
            ylabel = self.config.get("backtest", {}).get("target", "ReturnFwd_1")

        # --- Gráfico ---
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_plot.index, series_real, label="Real", alpha=0.8)
        ax.plot(df_plot.index, series_pred, label="Predicho", alpha=0.8)

        escala_txt = "precios" if plot_scale == "price" else "retornos"
        ax.set_title(f"{model_name} - Real vs Predicho{suffix} ({escala_txt})")
        ax.set_xlabel("Fecha")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

        # Nombre archivo
        if params:
            params_str = "_".join(f"{k}{v}" for k, v in params.items())
            fname = f"{model_name}_{params_str}{suffix}.png"
        else:
            fname = f"{model_name}{suffix}.png"

        plot_path = output_dir / fname
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        self.logger.info(f"📊 Gráfico de predicciones guardado en: {plot_path}")
        self._archive_backtest_artifact(plot_path)



    def _validate_model_on_test(self, model_name: str, params: dict, df_train: pd.DataFrame, y_test: pd.Series, X_test: pd.DataFrame):
        """Entrena un modelo con datos de train y lo valida contra test."""
        target_col = self.config.get("backtest", {}).get("target", "ReturnFwd_1")
        
        # Preparar datos de entrenamiento completos
        y_train = df_train[target_col]
        feature_cols = self._get_model_feature_columns(df_train, target_col)
        X_train = df_train[feature_cols]

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
        - Calcula niveles de entrada / SL / TP y tamaño de posición con base en la sección 'risk'
        - Guarda señales y, opcionalmente, ejecuta la orden real en MT5
        - Actualiza el reporte de ciclo de vida de trades cerrados
        """
        from utils.risk_utils import (
            calculate_position_size_for_risk_amount,
            compute_entry_sl_tp,
            estimate_position_risk_amount,
        )

        self.logger.info("\n" + "=" * 60)
        self.logger.info("MODO: PRODUCCIÓN")
        self.logger.info("=" * 60 + "\n")

        # 1) Cargar / limpiar / generar features
        self.logger.info("📥 Cargando datos para producción...")
        df_raw = self._load_data()
        df_clean = self._clean_data(df_raw)
        df_features = self._generate_features(df_clean)

        target_col = self.config.get("backtest", {}).get("target", "ReturnFwd_1")

        # Quitamos filas sin target ni features
        feature_cols = self._get_model_feature_columns(df_features, target_col)
        df_processed = df_features.dropna(subset=[target_col] + feature_cols)

        if df_processed.empty:
            self.logger.error("No hay datos suficientes después del procesamiento para producción.")
            return

        X_all = df_processed[feature_cols]
        y_all = df_processed[target_col]

        # Último valor de ATR (si existe) para gestión de riesgo basada en volatilidad
        atr_col = "ATR_14"
        if atr_col in df_processed.columns:
            atr_value = float(df_processed[atr_col].iloc[-1])
        else:
            atr_value = None

        # 2) Modelos habilitados en la config
        models_cfg = self.config.get("models", [])
        enabled_models_cfg = [m for m in models_cfg if m.get("enabled", True)]
        live_cfg = self._get_live_trading_settings()

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
                f"🏆 Modelo campeón global según backtest: {best_model_name}"
            )
        else:
            self.logger.warning(
                "No se pudo determinar un modelo campeón global con _get_best_model_from_config()."
            )

        if live_cfg["execute_best_model_only"] and best_model_config:
            enabled_models_cfg = [best_model_config]
            self.logger.info("🎯 Producción configurada para operar únicamente con el modelo campeón.")

        active_release = self._resolve_active_release_assets()
        active_release_id = active_release.get("release_id")
        strategy_profile_name = self._get_strategy_profile_name() or active_release.get("strategy_profile") or "default"
        if active_release_id:
            self.logger.info(
                "📦 Release activa de producción%s: %s (activada %s)",
                f" [{strategy_profile_name}]" if strategy_profile_name else "",
                active_release_id,
                active_release.get("activated_at"),
            )

        # 3) Métricas de backtest (summary_best_runs.csv)
        metrics_by_model: dict[str, dict[str, float]] = {}
        summary_path = Path(active_release.get("summary_csv"))

        if summary_path.exists():
            try:
                df_best = pd.read_csv(summary_path)
                metric_cols = [
                    "rmse",
                    "mae",
                    "hit_rate",
                    "accuracy",
                    "f1_score",
                    "precision",
                    "recall",
                    "dm_stat",
                    "dm_pvalue",
                    "sharpe",
                    "sortino",
                    "calmar",
                    "max_drawdown",
                    "profit_factor",
                    "win_rate",
                    "payoff_ratio",
                    "consistency_ratio",
                    "avg_trade_return",
                ]
                for _, row in df_best.iterrows():
                    model_name = str(row.get("model", "")).upper()
                    metrics_by_model[model_name] = {
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
            "RANDOMWALK": RandomWalkModel,
            "MOMENTUM": MomentumModel,
            "RANDOMFORESTREGRESSOR": RandomForestRegressorModel,
            "HISTGRADIENTBOOSTING": HistGradientBoostingRegressorModel,
        }

        # Directorio donde están los modelos guardados
        models_dir = Path(active_release.get("models_dir"))
        models_dir.mkdir(parents=True, exist_ok=True)

        # Datos comunes para todas las filas de salida
        last_row = df_processed.iloc[-1]
        price_now = float(last_row["Close"]) if "Close" in last_row else float("nan")

        # Info del símbolo desde config + MT5
        symbol = self.config.get("data", {}).get("symbol", "UNKNOWN")

        pip_size_cfg = self.config.get("backtest", {}).get("pip_size")
        pip_size = float(pip_size_cfg) if pip_size_cfg is not None else 0.0

        point = None
        digits = 5
        contract_size = None
        min_lot = 0.01
        lot_step = 0.01
        stops_level_points = 0
        freeze_level_points = 0
        market_tick = None

        try:
            if hasattr(self, "data_loader") and self.data_loader is not None:
                info = self.data_loader.get_symbol_info(symbol)
                if info:
                    point = float(info.get("point") or 0.0)
                    digits = int(info.get("digits") or digits)
                    contract_size = float(info.get("trade_contract_size") or 0.0)
                    # volúmenes mínimos / step
                    min_lot = float(info.get("volume_min") or min_lot)
                    lot_step = float(info.get("volume_step") or lot_step)
                    stops_level_points = int(info.get("trade_stops_level") or 0)
                    freeze_level_points = int(info.get("trade_freeze_level") or 0)
        except Exception as e:
            self.logger.warning(f"No se pudo obtener info detallada del símbolo desde MT5: {e}")

        # Fallbacks razonables para FX
        if pip_size <= 0.0:
            if point is not None and point > 0:
                pip_size = point
            else:
                pip_size = 0.0001
        if point is None or point <= 0:
            point = pip_size
        if contract_size is None or contract_size <= 0:
            contract_size = 100000.0  # típico FX 1 lote

        # Info de cuenta para tamaño de posición
        balance = None
        try:
            mt5_client = self._ensure_mt5_client()
            acc_info = mt5_client.get_account_info()
            if acc_info:
                balance = float(acc_info.get("balance", 0.0))
            market_tick = mt5_client.get_symbol_tick(symbol)
        except Exception as e:
            self.logger.warning(f"No se pudo obtener balance desde MT5: {e}")

        # Fallback: usar balance definido en config.risk.account_balance si existe
        risk_cfg_dict = self.config.get("risk", {}) or {}
        if balance is None:
            balance_cfg = risk_cfg_dict.get("account_balance")
            if balance_cfg is not None:
                try:
                    balance = float(balance_cfg)
                except Exception:
                    balance = None

        # Último fallback si no hay balance
        if balance is None:
            balance = 0.0

        risk_budget_cfg = self._get_risk_budget_settings()
        risk_per_trade_pct = float(risk_budget_cfg["risk_per_trade_pct"])
        max_total_open_risk_pct = float(risk_budget_cfg["max_total_open_risk_pct"])
        total_risk_budget = max(0.0, balance * max_total_open_risk_pct)
        per_trade_risk_budget = max(0.0, balance * risk_per_trade_pct)

        open_risk_snapshot = {
            "open_risk_amount": 0.0,
            "open_positions_count": 0,
            "positions_without_sl": 0,
        }
        if "mt5_client" in locals():
            try:
                open_risk_snapshot = self._estimate_open_positions_risk(
                    mt5_client=mt5_client,
                    open_positions=mt5_client.get_all_positions(),
                )
            except Exception as e:
                self.logger.warning(f"No se pudo estimar el riesgo abierto actual: {e}")

        open_risk_amount = float(open_risk_snapshot.get("open_risk_amount", 0.0))
        positions_without_sl = int(open_risk_snapshot.get("positions_without_sl", 0))

        if positions_without_sl > 0:
            self.logger.warning(
                "Hay %s posicion(es) abierta(s) sin SL valido. Riesgo abierto estimado=%.2f",
                positions_without_sl,
                open_risk_amount,
            )

        if total_risk_budget > 0:
            self.logger.info(
                "Presupuesto de riesgo: total=%.2f | abierto=%.2f | restante=%.2f | por_trade=%.2f",
                total_risk_budget,
                open_risk_amount,
                max(total_risk_budget - open_risk_amount, 0.0),
                per_trade_risk_budget,
            )

        # --- Parámetros de trading (umbral de pips para señal) ---
        trading_cfg = self.config.get("trading", {}) or {}
        min_pips_signal = float(
            trading_cfg.get(
                "min_pips_signal",
                self.config.get("backtest", {}).get("threshold_pips", 0.0),
            )
        )

        rows = []
        planned_additional_risk_amount = 0.0

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
                if hasattr(model_instance, "predict_loaded_with_context"):
                    prediction = model_instance.predict_loaded_with_context(X_all, y_all)
                else:
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

            # Precio objetivo y delta desde el cierre actual
            if not np.isnan(price_now):
                price_target = price_now * (1.0 + pred_return)
                delta_price = price_target - price_now
                pips = delta_price / pip_size
            else:
                price_target = float("nan")
                delta_price = float("nan")
                pips = float("nan")

            # Métricas históricas del modelo para score de confianza
            m_metrics = metrics_by_model.get(model_name_upper, {})

            enable_confidence_filter = bool(trading_cfg.get("enable_confidence_filter", False))
            min_confidence = float(trading_cfg.get("min_confidence", 0.60))

            signal_info = build_signal_from_prediction(
                pred_return=pred_return,
                pip_size=pip_size,
                min_pips_signal=min_pips_signal,
                model_metrics=m_metrics if 'm_metrics' in locals() else {},
                min_confidence=min_confidence if enable_confidence_filter else 0.0,
                probability=None,
            )
            signal = str(signal_info["signal"])
            confidence = float(signal_info["confidence"])
            confirmation = self._evaluate_signal_confirmation(
                signal=signal,
                feature_row=last_row,
            )
            if signal in {"BUY", "SELL"} and not confirmation.get("passed", True):
                self.logger.info(
                    "  -> Señal %s bloqueada por confirmación opcional: %s",
                    signal,
                    confirmation.get("reason"),
                )
                signal = "HOLD"

            # --- Gestión de riesgo: niveles planificados y niveles reales de mercado ---
            entry_price = float("nan")
            sl_price = float("nan")
            tp_price = float("nan")
            sl_pips = float("nan")
            tp_pips = float("nan")
            live_entry_price = float("nan")
            live_sl_price = float("nan")
            live_tp_price = float("nan")
            live_sl_pips = float("nan")
            live_tp_pips = float("nan")
            volume_lots = 0.0
            risk_amount = 0.0
            allocated_risk_budget = 0.0
            risk_per_pip_per_lot = 0.0
            risk_per_lot_at_stop = 0.0
            projected_total_open_risk_after_trade = max(open_risk_amount + planned_additional_risk_amount, 0.0)
            remaining_risk_budget_before_trade = max(
                total_risk_budget - open_risk_amount - planned_additional_risk_amount,
                0.0,
            )

            market_reference_price = float("nan")
            if isinstance(market_tick, dict):
                if signal == "BUY":
                    market_reference_price = float(market_tick.get("ask") or np.nan)
                elif signal == "SELL":
                    market_reference_price = float(market_tick.get("bid") or np.nan)

            if signal in ("BUY", "SELL") and not np.isnan(price_now):
                planned_levels = compute_entry_sl_tp(
                    side=signal,
                    close_price=price_now,
                    atr_value=atr_value,
                    pip_size=pip_size,
                    risk_cfg_dict=risk_cfg_dict,
                )
                entry_price = round(float(planned_levels["entry_price"]), digits)
                sl_price = round(float(planned_levels["sl_price"]), digits)
                tp_price = round(float(planned_levels["tp_price"]), digits)
                sl_pips = planned_levels["sl_pips"]
                tp_pips = planned_levels["tp_pips"]

                live_risk_cfg = dict(risk_cfg_dict)
                live_risk_cfg["entry_mode"] = "close"
                live_close_price = market_reference_price if not np.isnan(market_reference_price) else price_now
                live_levels = compute_entry_sl_tp(
                    side=signal,
                    close_price=live_close_price,
                    atr_value=atr_value,
                    pip_size=pip_size,
                    risk_cfg_dict=live_risk_cfg,
                )
                live_entry_price = round(float(live_levels["entry_price"]), digits)
                live_sl_price = round(float(live_levels["sl_price"]), digits)
                live_tp_price = round(float(live_levels["tp_price"]), digits)
                live_sl_pips = live_levels["sl_pips"]
                live_tp_pips = live_levels["tp_pips"]

                # Tamaño de posición coherente con la ejecución real de mercado.
                block_for_unprotected_positions = (
                    risk_budget_cfg["allow_multiple_positions"]
                    and risk_budget_cfg["block_new_entries_without_sl"]
                    and positions_without_sl > 0
                )
                available_risk_budget = remaining_risk_budget_before_trade
                if block_for_unprotected_positions:
                    available_risk_budget = 0.0

                allocated_risk_budget = min(per_trade_risk_budget, available_risk_budget)
                risk_per_pip_per_lot = max(contract_size * pip_size, 0.0)
                risk_per_lot_at_stop = estimate_position_risk_amount(
                    entry_price=live_entry_price,
                    sl_price=live_sl_price,
                    point=point,
                    contract_size=contract_size,
                    volume_lots=1.0,
                )
                volume_lots = calculate_position_size_for_risk_amount(
                    entry_price=live_entry_price,
                    sl_price=live_sl_price,
                    point=point,
                    contract_size=contract_size,
                    risk_amount=allocated_risk_budget,
                    min_lot=min_lot,
                    lot_step=lot_step,
                )
                if volume_lots > 0:
                    risk_amount = estimate_position_risk_amount(
                        entry_price=live_entry_price,
                        sl_price=live_sl_price,
                        point=point,
                        contract_size=contract_size,
                        volume_lots=volume_lots,
                    )
                    planned_additional_risk_amount += risk_amount
                    projected_total_open_risk_after_trade = open_risk_amount + planned_additional_risk_amount
                else:
                    projected_total_open_risk_after_trade = open_risk_amount + planned_additional_risk_amount

            # Métricas de backtest (si existen)
            m_metrics = metrics_by_model.get(model_name_upper, {})
            rmse = m_metrics.get("rmse")
            mae = m_metrics.get("mae")
            hit_rate = m_metrics.get("hit_rate")
            accuracy = m_metrics.get("accuracy")
            dm_stat = m_metrics.get("dm_stat")
            dm_pvalue = m_metrics.get("dm_pvalue")
            sharpe = m_metrics.get("sharpe")
            sortino = m_metrics.get("sortino")
            max_dd = m_metrics.get("max_drawdown")
            profit_factor = m_metrics.get("profit_factor")
            win_rate = m_metrics.get("win_rate")
            payoff_ratio = m_metrics.get("payoff_ratio")

            is_best = (model_name_upper == best_model_name)

            self.logger.info(
                f"  📈 Modelo {model_name} -> retorno={pred_return:.6f}, "
                f"pips={pips:.2f}, signal={signal}, confidence={confidence:.3f}, "
                f"confirm={confirmation.get('reason')}, "
                f"entry_plan={entry_price}, SL_plan={sl_price}, TP_plan={tp_price}, "
                f"entry_live={live_entry_price}, SL_live={live_sl_price}, TP_live={live_tp_price}, "
                f"lots={volume_lots:.2f}, balance={balance}, risk={risk_amount:.2f}, "
                f"sl_pips_live={live_sl_pips:.2f}, usd_per_pip_lot={risk_per_pip_per_lot:.2f}, "
                f"risk_per_lot_stop={risk_per_lot_at_stop:.2f}, open_risk={open_risk_amount:.2f}, "
                f"remaining_budget={remaining_risk_budget_before_trade:.2f}, "
                f"projected_open_risk={projected_total_open_risk_after_trade:.2f}"
            )

            row = {
                "timestamp": df_processed.index[-1],
                "release_id": active_release_id,
                "strategy_profile": strategy_profile_name,
                "magic_number": live_cfg["magic_number"],
                "order_comment_prefix": live_cfg["order_comment_prefix"],
                "symbol": symbol,
                "timeframe": self.config.get("data", {}).get("timeframe", "UNKNOWN"),
                "model": model_name,
                "pred_return": pred_return,
                "signal": signal,
                "confidence": confidence,
                "signal_confirmation_enabled": bool(confirmation.get("enabled", False)),
                "signal_confirmation_passed": bool(confirmation.get("passed", True)),
                "signal_confirmation_reason": confirmation.get("reason"),
                "momentum_feature": confirmation.get("momentum_column"),
                "momentum_value": confirmation.get("momentum_value"),
                "volume_feature": confirmation.get("volume_column"),
                "volume_value": confirmation.get("volume_value"),
                "regime_feature": confirmation.get("regime_column"),
                "regime_value": confirmation.get("regime_value"),
                "entry_price": entry_price,
                "planned_entry_price": entry_price,
                "price_now": price_now,
                "price_target": price_target,
                "delta_price": delta_price,
                "pips": pips,
                # Gestión de riesgo
                "sl_price": sl_price,
                "tp_price": tp_price,
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
                "market_reference_price": market_reference_price,
                "live_entry_price": live_entry_price,
                "live_sl_price": live_sl_price,
                "live_tp_price": live_tp_price,
                "live_sl_pips": live_sl_pips,
                "live_tp_pips": live_tp_pips,
                "symbol_digits": digits,
                "stops_level_points": stops_level_points,
                "freeze_level_points": freeze_level_points,
                "volume_lots": volume_lots,
                "account_balance": balance,
                "risk_per_trade_pct": risk_per_trade_pct,
                "risk_amount": risk_amount,
                "allocated_risk_budget": allocated_risk_budget,
                "risk_per_pip_per_lot": risk_per_pip_per_lot,
                "risk_per_lot_at_stop": risk_per_lot_at_stop,
                "max_total_open_risk_pct": max_total_open_risk_pct,
                "open_risk_amount": open_risk_amount,
                "remaining_risk_budget_before_trade": remaining_risk_budget_before_trade,
                "projected_total_open_risk_after_trade": projected_total_open_risk_after_trade,
                "positions_without_sl_open": positions_without_sl,
                "is_best_model": is_best,
                # Métricas de backtest
                "rmse_backtest": rmse,
                "mae_backtest": mae,
                "hit_rate_backtest": hit_rate,
                "accuracy_backtest": accuracy,
                "f1_score_backtest": m_metrics.get("f1_score"),
                "precision_backtest": m_metrics.get("precision"),
                "recall_backtest": m_metrics.get("recall"),
                "dm_stat_backtest": dm_stat,
                "dm_pvalue_backtest": dm_pvalue,
                "sharpe_backtest": sharpe,
                "sortino_backtest": sortino,
                "calmar_backtest": m_metrics.get("calmar"),
                "max_drawdown_backtest": max_dd,
                "profit_factor_backtest": profit_factor,
                "win_rate_backtest": win_rate,
                "payoff_ratio_backtest": payoff_ratio,
                "consistency_ratio_backtest": m_metrics.get("consistency_ratio"),
                "avg_trade_return_backtest": m_metrics.get("avg_trade_return"),
            }

            rows.append(row)

        if not rows:
            self.logger.error("No se generó ninguna señal de producción (todas fallaron).")
            self._sync_live_trade_report()
            return

        df_rows = pd.DataFrame(rows)

        # 7) Guardar señales
        output_paths = self._get_production_output_paths()
        self._append_rows_to_csv(output_paths["signals"], df_rows)
        self.logger.info(f"\n💾 Señales de producción guardadas en: {output_paths['signals']}")

        # 8) Ejecución real opcional + reconciliación del reporte de trades
        self._execute_live_orders(df_rows)
        self._sync_live_trade_report()
        self.logger.info("✅ MODO PRODUCCIÓN COMPLETADO\n")

    def _run_sync_trades_mode(self) -> None:
        """Sincroniza el reporte local con el estado real de posiciones/deals en MT5."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("MODO: SYNC TRADES")
        self.logger.info("=" * 60 + "\n")

        self._ensure_mt5_client()
        lifecycle = self._sync_live_trade_report()
        n_closed = 0
        if lifecycle is not None and not lifecycle.empty and "status" in lifecycle.columns:
            n_closed = int((lifecycle["status"].astype(str).str.upper() == "CLOSED").sum())

        self.logger.info(f"🔄 Sincronización completada. Trades cerrados registrados: {n_closed}")
        self.logger.info("✅ MODO SYNC TRADES COMPLETADO\n")

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
        runtime_mode = self._active_mode in {"production", "sync_trades"}
        use_cache = data_config.get("use_cache", True)
        cache_expiry_hours = data_config.get("cache_expiry_hours", 24)

        if runtime_mode:
            runtime_use_cache = data_config.get("runtime_use_cache")
            runtime_cache_expiry_minutes = data_config.get("runtime_cache_expiry_minutes")
            use_cache = bool(runtime_use_cache) if runtime_use_cache is not None else False
            if runtime_cache_expiry_minutes is not None:
                try:
                    cache_expiry_hours = float(runtime_cache_expiry_minutes) / 60.0
                except (TypeError, ValueError):
                    cache_expiry_hours = data_config.get("cache_expiry_hours", 24)
            self.logger.info(
                f"  -> Runtime data policy: use_cache={use_cache}, cache_expiry_hours={cache_expiry_hours}"
            )
        
        self.data_loader = DataLoader(mt5_config=mt5_config)
        df = self.data_loader.load_data(
            symbol=data_config.get("symbol", "EURUSD"),
            timeframe=data_config.get("timeframe", "D1"),
            n_bars=data_config.get("n_bars", 1000),
            use_cache=use_cache,
            cache_expiry_hours=cache_expiry_hours
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
        self.df_clean = df_clean.copy()
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
            indicators = features_config.get("technical_indicators", {}).get("indicators")
            df_features = FeatureEngineer.add_technical_indicators(df_features, indicators=indicators)
            self.logger.info("  -> Indicadores técnicos agregados.")

        # 3. Generar features rezagados (lags)
        if features_config.get("lag_features", {}).get("enabled", False):
            lag_config = features_config["lag_features"]
            for col in lag_config.get("columns", []):
                if col in df_features.columns:
                    df_features = FeatureEngineer.add_lag_features(df_features, col=col, lags=lag_config.get("lags", []))
                    self.logger.info(f"  -> Lags agregados para la columna: '{col}'")

        # 4. Aprendizaje no supervisado (regímenes de mercado)
        unsup_cfg = self.config.get("unsupervised", {}) or {}
        if unsup_cfg.get("enabled", False):
            try:
                method = str(unsup_cfg.get("method", "kmeans")).lower()
                if method == "kmeans":
                    n_clusters = int(unsup_cfg.get("n_clusters", 3))
                    self.regime_clusterer = MarketRegimeClusterer(n_clusters=n_clusters)
                    regime_result = self.regime_clusterer.fit_transform(df_features)
                    df_features = regime_result.features
                    self.logger.info(
                        f"  -> Regímenes de mercado agregados con KMeans (n_clusters={n_clusters})."
                    )
                else:
                    self.logger.warning(f"  -> Método no supervisado no soportado: {method}")
            except Exception as e:
                self.logger.warning(f"  -> No se pudieron agregar regímenes de mercado: {e}")

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
                # 1) Definir símbolo y columna de precio desde la config
        symbol = self.config.get("data", {}).get("symbol", "UNKNOWN")
        price_col = self.config.get("eda", {}).get("price_col", "Close")

        # 2) Definir directorio de salida para el EDA
        output_root = self.config.get("output", {}).get("dir", "outputs")
        eda_dir = Path(output_root) / "eda"

        # 3) Ejecutar el EDA con la clase actual (exploratory_analysis.py)
        self.eda = ExploratoryAnalysis(output_dir=str(eda_dir))
        self.eda.analyze(df, symbol=symbol, price_col=price_col)

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

def _plot_price_with_entries(
    self,
    df_features: pd.DataFrame,
    idx: pd.Index,
    y_true: list,
    y_pred: list,
    model_name: str,
    symbol: str,
    price_col: str,
    plot_dir: Path,
) -> str:
    """
    Grafica el precio del activo y marca los puntos de entrada del backtest.

    Asume que:
    - y_true e y_pred son retornos (o cambios) para el horizonte de backtest.
    - La señal de trading se basa en sign(y_pred): >0 = LONG, <0 = SHORT.
    """
    # Alinear longitudes
    n = min(len(y_true), len(y_pred), len(idx))
    y_true_arr = np.asarray(y_true[:n], dtype=float)
    y_pred_arr = np.asarray(y_pred[:n], dtype=float)
    idx = idx[:n]

    # Direcciones
    true_dir = np.sign(y_true_arr)
    pred_dir = np.sign(y_pred_arr)
    hits = true_dir == pred_dir

    # Serie de precios completa (para dar contexto)
    price_series = df_features[price_col]
    # Nos aseguramos de que idx esté dentro de price_series
    price_at_signals = price_series.loc[idx]

    # Construir DataFrame auxiliar
    df_trades = pd.DataFrame({
        "date": idx,
        "price": price_at_signals.values,
        "pred_dir": pred_dir,
        "hit": hits,
    })

    fig, ax = plt.subplots(figsize=(14, 6))

    # Precio completo
    ax.plot(price_series.index, price_series.values, label=f"Precio {symbol}", linewidth=1.5)

    # Puntos LONG
    long_hits = df_trades[(df_trades["pred_dir"] > 0) & (df_trades["hit"])]
    long_errors = df_trades[(df_trades["pred_dir"] > 0) & (~df_trades["hit"])]

    ax.scatter(
        long_hits["date"],
        long_hits["price"],
        marker="^",
        color="green",
        s=60,
        label="Entrada LONG (acierto)",
    )
    ax.scatter(
        long_errors["date"],
        long_errors["price"],
        marker="^",
        color="red",
        s=60,
        label="Entrada LONG (error)",
        alpha=0.7,
    )

    # Puntos SHORT
    short_hits = df_trades[(df_trades["pred_dir"] < 0) & (df_trades["hit"])]
    short_errors = df_trades[(df_trades["pred_dir"] < 0) & (~df_trades["hit"])]

    ax.scatter(
        short_hits["date"],
        short_hits["price"],
        marker="v",
        color="blue",
        s=60,
        label="Entrada SHORT (acierto)",
    )
    ax.scatter(
        short_errors["date"],
        short_errors["price"],
        marker="v",
        color="orange",
        s=60,
        label="Entrada SHORT (error)",
        alpha=0.7,
    )

    ax.set_title(f"{symbol} - {model_name}\nPuntos de entrada del backtest", fontsize=13, weight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel(price_col)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    fname = f"{symbol}_{model_name}_backtest_entries.png"
    path = plot_dir / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    self.logger.info(f"📈 Gráfico de entradas guardado en: {path}")
    return str(path)

def _plot_accuracy_curve(
    self,
    idx: pd.Index,
    y_true: list,
    y_pred: list,
    model_name: str,
    symbol: str,
    plot_dir: Path,
    window: int = 50,
) -> str:
    """
    Grafica la precisión direccional del modelo a lo largo del backtest:
    - Precisión acumulada.
    - Precisión móvil en ventana (rolling).
    """
    n = min(len(y_true), len(y_pred), len(idx))
    y_true_arr = np.asarray(y_true[:n], dtype=float)
    y_pred_arr = np.asarray(y_pred[:n], dtype=float)
    idx = idx[:n]

    true_dir = np.sign(y_true_arr)
    pred_dir = np.sign(y_pred_arr)
    hits = (true_dir == pred_dir).astype(int)

    hits_series = pd.Series(hits, index=idx)

    # Precisión acumulada
    cum_hits = hits_series.cumsum() / np.arange(1, len(hits_series) + 1)

    # Precisión rolling
    rolling_hits = hits_series.rolling(window).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Acumulada
    ax1.plot(cum_hits.index, cum_hits.values, linewidth=1.5, label="Precisión acumulada")
    ax1.axhline(0.5, linestyle="--", color="gray", linewidth=1, label="Azar (50%)")
    ax1.set_ylabel("Accuracy acumulado")
    ax1.set_title(f"{symbol} - {model_name}\nEvolución de la precisión direccional", fontsize=13, weight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    # Rolling
    ax2.plot(rolling_hits.index, rolling_hits.values, linewidth=1.5, label=f"Precisión móvil ({window} trades)")
    ax2.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    ax2.set_ylabel(f"Accuracy rolling ({window})")
    ax2.set_xlabel("Fecha")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.legend(loc="best")

    plt.tight_layout()
    fname = f"{symbol}_{model_name}_accuracy_curve.png"
    path = plot_dir / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    self.logger.info(f"📊 Curva de accuracy guardada en: {path}")
    return str(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Trading Algorítmico.")
    parser.add_argument("--mode", type=str, default="eda", 
                        choices=["eda", "train", "backtest","production", "test", "sync_trades", "clear_cache"],
                        help="Modo de ejecución del pipeline.")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Ruta al archivo de configuración YAML.")
    args = parser.parse_args()
    
    pipeline = TradingPipeline(config_path=args.config)
    pipeline.run(mode=args.mode)
