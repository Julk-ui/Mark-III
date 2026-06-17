#!/usr/bin/env python3
# main_pipeline.py
"""
Pipeline principal del proyecto de Trading AlgorÃ­tmico.
Integra todos los mÃ³dulos: ConexiÃ³n, Limpieza, EDA y Modelos.
"""


from __future__ import annotations
import debugpy
import sys, os
import re
import math
import logging
import matplotlib
matplotlib.use("Agg")   # <- importante en Windows para evitar Tkinter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Any
if os.getenv("DEBUGPY", "0") == "1":
    debugpy.listen(("localhost", 5680))
    print("Esperando debuggerâ€¦ ConÃ©ctate desde VS Code.")
    debugpy.wait_for_client()

# --- SupresiÃ³n de Warnings de librerÃ­as ---
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=ValueWarning)
except Exception:
    pass
import json
import yaml
import argparse
import shutil
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pandas.errors import EmptyDataError
from itertools import product
from copy import deepcopy
from sklearn.model_selection import ParameterGrid

# Imports de mÃ³dulos propios
from data.data_loader import DataLoader, DataValidator
from data.data_cleaner import DataCleaner, FeatureEngineer
from utils.metrics_v2 import calculate_all_metrics
from models.arima_model import ArimaModel
from models.prophet_model import ProphetModel
from models.lstm_model import LSTMModel # AsegÃºrate que este archivo exista
from models.random_walk_model import MomentumModel, RandomWalkModel
from models.tree_models import RandomForestRegressorModel, HistGradientBoostingRegressorModel
from models.linear_models import RidgeRegressorModel
from models.classifier_models import (
    LogisticRegressionClassifierModel,
    RandomForestClassifierModel,
    ExtraTreesClassifierModel,
    HistGradientBoostingClassifierModel,
)
from models.regime_model import MarketRegimeClusterer
from utils.decision_utils import (
    build_signal_from_prediction,
    build_signal_from_probabilities,
    build_signal_from_hybrid_prediction,
)

from eda.exploratory_analysis import ExploratoryAnalysis


def _configure_stdio_safely() -> None:
    """Evita que la consola de Windows rompa el proceso por caracteres no representables."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            continue


_configure_stdio_safely()


class TradingPipeline:
    """
    Orquestador principal del pipeline de trading
    """
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Ruta al archivo de configuraciÃ³n YAML
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
        """Normaliza la configuraciÃ³n usada para elegir los mejores runs."""
        selection_cfg = self.config.get("model_selection", {}) or {}
        return {
            "primary_metric": selection_cfg.get("primary_metric", "hit_rate"),
            "primary_greater_is_better": bool(selection_cfg.get("primary_greater_is_better", True)),
            "secondary_metric": selection_cfg.get("secondary_metric", "rmse"),
            "secondary_greater_is_better": bool(selection_cfg.get("secondary_greater_is_better", False)),
            "min_trades": int(selection_cfg.get("min_trades", 0) or 0),
            "min_test_points": int(selection_cfg.get("min_test_points", 0) or 0),
            "publish_requires_candidate_thresholds": bool(
                selection_cfg.get("publish_requires_candidate_thresholds", True)
            ),
        }

    def _filter_runs_by_selection_thresholds(
        self,
        df_runs: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Aplica los filtros minimos de trades y puntos de test."""
        if df_runs is None or df_runs.empty:
            return pd.DataFrame(), []

        selection = self._get_model_selection_settings()
        min_trades = selection["min_trades"]
        min_test_points = selection["min_test_points"]

        filtered = df_runs.copy()
        applied_filters: list[str] = []

        if min_trades > 0 and "n_trades" in filtered.columns:
            filtered = filtered[pd.to_numeric(filtered["n_trades"], errors="coerce").fillna(0) >= min_trades]
            applied_filters.append(f"n_trades >= {min_trades}")

        if min_test_points > 0 and "n_test_points" in filtered.columns:
            filtered = filtered[
                pd.to_numeric(filtered["n_test_points"], errors="coerce").fillna(0) >= min_test_points
            ]
            applied_filters.append(f"n_test_points >= {min_test_points}")

        return filtered, applied_filters

    def _select_best_run(
        self,
        df_runs: pd.DataFrame,
        model_name: str | None = None,
        log_prefix: str = "",
    ) -> pd.Series | None:
        """Selecciona el mejor run con la misma lÃ³gica usada en todo el pipeline."""
        if df_runs is None or df_runs.empty:
            return None

        selection = self._get_model_selection_settings()
        primary = selection["primary_metric"]
        primary_greater = selection["primary_greater_is_better"]
        secondary = selection["secondary_metric"]
        secondary_greater = selection["secondary_greater_is_better"]
        ranked = df_runs.copy()

        for col in {primary, secondary, "rmse", "hit_rate", "n_trades", "n_test_points"}:
            if col in ranked.columns:
                ranked[col] = pd.to_numeric(ranked[col], errors="coerce")

        filtered, applied_filters = self._filter_runs_by_selection_thresholds(ranked)

        if not filtered.empty:
            ranked = filtered
        elif applied_filters:
            scope = model_name or "corridas"
            self.logger.warning(
                f"{log_prefix}{scope}: no hay runs que cumplan {' y '.join(applied_filters)}. "
                "Se seleccionarÃ¡ sin esos filtros."
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

    def _rank_runs_for_selection(
        self,
        df_runs: pd.DataFrame,
        *,
        enforce_thresholds: bool = True,
    ) -> pd.DataFrame:
        """Ordena corridas con la misma logica de seleccion usada para el campeon."""
        if df_runs is None or df_runs.empty:
            return pd.DataFrame()

        selection = self._get_model_selection_settings()
        primary = selection["primary_metric"]
        primary_greater = selection["primary_greater_is_better"]
        secondary = selection["secondary_metric"]
        secondary_greater = selection["secondary_greater_is_better"]
        ranked = df_runs.copy()

        for col in {primary, secondary, "rmse", "hit_rate", "n_trades", "n_test_points"}:
            if col in ranked.columns:
                ranked[col] = pd.to_numeric(ranked[col], errors="coerce")

        if enforce_thresholds:
            filtered, _ = self._filter_runs_by_selection_thresholds(ranked)
            if not filtered.empty:
                ranked = filtered

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
            return ranked

        return ranked.sort_values(by=sort_cols, ascending=ascending, na_position="last")

    def _start_backtest_run(self) -> str:
        """Inicializa un identificador uniforme para los artefactos del run."""
        self._backtest_run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"Backtest run_id: {self._backtest_run_label}")
        return self._backtest_run_label

    def _ensure_backtest_run_label(self) -> str:
        """Devuelve el run_id actual o crea uno si todavÃ­a no existe."""
        if not self._backtest_run_label:
            return self._start_backtest_run()
        return self._backtest_run_label

    def _get_backtest_output_dir(self) -> Path:
        """Directorio estÃ¡ndar de salidas de backtest."""
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
        """Directorio raÃ­z de modelos persistidos."""
        models_dir = Path(self.config.get("output", {}).get("dir", "outputs")) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir

    def _get_release_models_dir(self, release_id: str) -> Path:
        """Directorio versionado para una release de modelos."""
        release_dir = self._get_models_output_dir() / "releases" / release_id
        release_dir.mkdir(parents=True, exist_ok=True)
        return release_dir

    def _get_active_release_manifest_path(self, profile_name: str | None = None) -> Path:
        """Ruta del puntero a la release activa usada por producciÃ³n."""
        profile_label = self._normalize_profile_label(profile_name)
        suffix = f"_{profile_label}" if profile_label else ""
        return self._get_config_dir() / f"active_release{suffix}.json"

    def _get_stable_optimized_config_path(self, profile_name: str | None = None) -> Path:
        """Alias estable de la configuraciÃ³n optimizada para un perfil."""
        profile_label = self._normalize_profile_label(profile_name)
        suffix = f"_{profile_label}" if profile_label else ""
        return self._get_config_dir() / f"config_optimizado{suffix}.yaml"

    def _write_yaml_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        """Escribe YAML de forma atÃ³mica para evitar lecturas parciales."""
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
        """Escribe JSON de forma atÃ³mica para publicar la release activa."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _write_dataframe_atomic(self, path: Path, df: pd.DataFrame) -> None:
        """Escribe un DataFrame CSV de forma atomica para evitar lecturas parciales."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        try:
            df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _copy_file_atomic(self, source: Path, destination: Path) -> None:
        """Copia un archivo a su alias estable usando replace atÃ³mico."""
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
        """Normaliza rutas leÃ­das del manifiesto."""
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

    def _load_yaml_dict(self, path: Path | None) -> dict[str, Any] | None:
        """Carga un YAML como dict, si existe y es valido."""
        if not isinstance(path, Path) or not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = yaml.safe_load(fh)
        except Exception as e:
            self.logger.warning("No se pudo leer YAML desde %s: %s", path, e)
            return None
        return payload if isinstance(payload, dict) else None

    def _merge_missing_mapping_keys(
        self,
        current: dict[str, Any] | None,
        template: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Completa claves faltantes de un mapping sin pisar la configuracion actual."""
        merged = deepcopy(current) if isinstance(current, dict) else {}
        if not isinstance(template, dict):
            return merged
        for key, template_value in template.items():
            current_value = merged.get(key)
            if key not in merged or current_value is None or current_value == {}:
                merged[key] = deepcopy(template_value)
            elif isinstance(current_value, dict) and isinstance(template_value, dict):
                merged[key] = self._merge_missing_mapping_keys(current_value, template_value)
        return merged

    def _get_release_operational_template(self, profile_name: str | None = None) -> tuple[dict[str, Any] | None, Path | None]:
        """Busca una configuracion canonica para heredar bloques operativos live."""
        config_dir = self._get_config_dir()
        candidates: list[Path] = []

        existing_manifest = self._load_active_release_manifest(profile_name=profile_name)
        if existing_manifest:
            resolved = self._resolve_manifest_path(existing_manifest.get("config_path"))
            if isinstance(resolved, Path):
                candidates.append(resolved)

        # Plantilla operativa canonica del live principal.
        candidates.append(config_dir / "config_optimizado_aggressive_hybrid_v1_3_tp5_sl3.yaml")
        candidates.append(config_dir / "config_optimizado.yaml")

        seen: set[str] = set()
        for candidate in candidates:
            candidate_key = str(candidate.resolve()) if candidate.exists() else str(candidate)
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            payload = self._load_yaml_dict(candidate)
            if isinstance(payload, dict):
                return payload, candidate
        return None, None

    def _inherit_release_operational_sections(
        self,
        optimized_config: dict[str, Any],
        *,
        profile_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Garantiza que toda release publicada herede la misma capa operativa live.

        Esto evita campeones publicables sin gestion de entrada/salida/riesgo
        solo porque el YAML de backtest no incluia esos bloques.
        """
        section_names = [
            "risk",
            "entry_management",
            "entry_grid",
            "entry_staging",
            "trade_management",
            "runtime_monitor",
            "logging",
            "scheduler",
        ]
        template_cfg, template_path = self._get_release_operational_template(profile_name=profile_name)
        if not isinstance(template_cfg, dict):
            return optimized_config

        merged = deepcopy(optimized_config)
        inherited_sections: list[str] = []
        for section_name in section_names:
            template_section = template_cfg.get(section_name)
            if not isinstance(template_section, dict):
                continue
            current_section = merged.get(section_name)
            if not isinstance(current_section, dict) or not current_section:
                merged[section_name] = deepcopy(template_section)
                inherited_sections.append(section_name)
                continue
            merged_section = self._merge_missing_mapping_keys(current_section, template_section)
            if merged_section != current_section:
                merged[section_name] = merged_section
                inherited_sections.append(section_name)

        if inherited_sections:
            self.logger.info(
                "Bloques operativos heredados para release%s: %s | template=%s",
                f" [{self._normalize_profile_label(profile_name)}]" if profile_name else "",
                ", ".join(inherited_sections),
                template_path if template_path is not None else "desconocido",
            )
        return merged

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
        """Asegura una conexiÃ³n MT5 reusable para producciÃ³n/sync."""
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

    def _get_market_pause_settings(self) -> dict[str, Any]:
        raw_cfg = self.config.get("market_pause", {}) or {}
        timeframe = str(self.config.get("data", {}).get("timeframe", "M5") or "M5")
        timeframe_delta = self._timeframe_to_timedelta(timeframe)
        default_stale_seconds = max(int(timeframe_delta.total_seconds() * 3), 900)
        return {
            "enabled": bool(raw_cfg.get("enabled", False)),
            "pause_production": bool(raw_cfg.get("pause_production", True)),
            "pause_sync_trades": bool(raw_cfg.get("pause_sync_trades", True)),
            "pause_runtime_monitor": bool(raw_cfg.get("pause_runtime_monitor", True)),
            "max_tick_staleness_seconds": max(
                int(raw_cfg.get("max_tick_staleness_seconds", default_stale_seconds) or 0),
                1,
            ),
            "require_positive_quotes": bool(raw_cfg.get("require_positive_quotes", True)),
        }

    def _get_market_pause_status(self, *, mode: str) -> dict[str, Any]:
        settings = self._get_market_pause_settings()
        mode_key = str(mode or "").strip().lower()
        mode_enabled = (
            (mode_key == "production" and settings["pause_production"])
            or (mode_key == "sync_trades" and settings["pause_sync_trades"])
            or (mode_key == "monitor_runtime" and settings["pause_runtime_monitor"])
        )
        if not settings["enabled"] or not mode_enabled:
            return {"paused": False, "reason": "disabled"}

        symbol = str(self.config.get("data", {}).get("symbol", "") or "").strip()
        if not symbol:
            return {"paused": False, "reason": "missing_symbol"}

        mt5_client = self._ensure_mt5_client()
        tick = mt5_client.get_symbol_tick(symbol)
        if not isinstance(tick, dict) or not tick:
            return {
                "paused": True,
                "reason": "no_tick",
                "symbol": symbol,
                "tick_age_seconds": float("inf"),
            }

        bid = pd.to_numeric(pd.Series([tick.get("bid")]), errors="coerce").iloc[0]
        ask = pd.to_numeric(pd.Series([tick.get("ask")]), errors="coerce").iloc[0]
        if settings["require_positive_quotes"] and (
            pd.isna(bid) or pd.isna(ask) or float(bid) <= 0.0 or float(ask) <= 0.0
        ):
            return {
                "paused": True,
                "reason": "invalid_quote",
                "symbol": symbol,
                "bid": None if pd.isna(bid) else float(bid),
                "ask": None if pd.isna(ask) else float(ask),
            }

        tick_time_msc = pd.to_numeric(pd.Series([tick.get("time_msc")]), errors="coerce").iloc[0]
        tick_time = pd.to_numeric(pd.Series([tick.get("time")]), errors="coerce").iloc[0]
        tick_timestamp = float("nan")
        if pd.notna(tick_time_msc) and float(tick_time_msc) > 0:
            tick_timestamp = float(tick_time_msc) / 1000.0
        elif pd.notna(tick_time) and float(tick_time) > 0:
            tick_timestamp = float(tick_time)
        if pd.isna(tick_timestamp) or float(tick_timestamp) <= 0.0:
            return {
                "paused": True,
                "reason": "missing_tick_time",
                "symbol": symbol,
            }

        tick_age_seconds = max(float(datetime.now().timestamp()) - float(tick_timestamp), 0.0)
        if tick_age_seconds > float(settings["max_tick_staleness_seconds"]):
            return {
                "paused": True,
                "reason": "stale_tick",
                "symbol": symbol,
                "tick_age_seconds": tick_age_seconds,
                "max_tick_staleness_seconds": float(settings["max_tick_staleness_seconds"]),
            }

        return {
            "paused": False,
            "reason": "market_open",
            "symbol": symbol,
            "tick_age_seconds": tick_age_seconds,
            "bid": None if pd.isna(bid) else float(bid),
            "ask": None if pd.isna(ask) else float(ask),
        }

    def _pause_if_market_closed(self, *, mode: str) -> bool:
        status = self._get_market_pause_status(mode=mode)
        if not bool(status.get("paused")):
            return False

        reason = str(status.get("reason") or "market_pause")
        symbol = str(status.get("symbol") or self.config.get("data", {}).get("symbol", "") or "").strip()
        if reason == "stale_tick":
            self.logger.warning(
                "Mercado en pausa para %s [%s]: ultimo tick demasiado viejo (%.0fs > %.0fs).",
                symbol,
                mode,
                float(status.get("tick_age_seconds") or 0.0),
                float(status.get("max_tick_staleness_seconds") or 0.0),
            )
        elif reason == "invalid_quote":
            self.logger.warning(
                "Mercado en pausa para %s [%s]: cotizacion invalida bid=%s ask=%s.",
                symbol,
                mode,
                status.get("bid"),
                status.get("ask"),
            )
        elif reason == "no_tick":
            self.logger.warning("Mercado en pausa para %s [%s]: no hay tick disponible.", symbol, mode)
        elif reason == "missing_tick_time":
            self.logger.warning("Mercado en pausa para %s [%s]: tick sin timestamp util.", symbol, mode)
        else:
            self.logger.warning("Mercado en pausa para %s [%s]: reason=%s.", symbol, mode, reason)
        return True

    def _get_entry_management_settings(self) -> dict[str, Any]:
        """Configuracion opcional para escalar la entrada con una orden LIMIT de mejora."""
        raw_cfg = self.config.get("entry_management", {}) or {}
        mode = str(raw_cfg.get("mode", "") or "").strip().lower()

        try:
            initial_fraction = float(raw_cfg.get("initial_market_fraction", 0.30) or 0.0)
        except Exception:
            initial_fraction = 0.30
        pending_raw = raw_cfg.get("pending_fraction")
        try:
            pending_fraction = (
                max(1.0 - initial_fraction, 0.0)
                if pending_raw is None
                else float(pending_raw or 0.0)
            )
        except Exception:
            pending_fraction = max(1.0 - initial_fraction, 0.0)
        try:
            retrace_fraction = float(raw_cfg.get("retrace_fraction_of_stop", 0.80) or 0.0)
        except Exception:
            retrace_fraction = 0.80

        initial_fraction = min(max(initial_fraction, 0.0), 1.0)
        pending_fraction = min(max(pending_fraction, 0.0), 1.0)
        total_fraction = initial_fraction + pending_fraction
        if total_fraction > 1.0 and total_fraction > 0.0:
            initial_fraction /= total_fraction
            pending_fraction /= total_fraction

        return {
            "enabled": bool(raw_cfg.get("enabled", False)) and mode == "split_retrace_limit",
            "mode": mode,
            "initial_market_fraction": initial_fraction,
            "pending_fraction": pending_fraction,
            "retrace_fraction_of_stop": min(max(retrace_fraction, 0.0), 1.0),
            "pending_order_type": str(raw_cfg.get("pending_order_type", "limit") or "limit").strip().lower(),
            "cancel_pending_after_bars": max(int(raw_cfg.get("cancel_pending_after_bars", 3) or 0), 1),
            "retrace_only_cancel_pending_after_bars": max(
                int(raw_cfg.get("retrace_only_cancel_pending_after_bars", 2) or 0),
                1,
            ),
            "cancel_pending_on_position_close": bool(raw_cfg.get("cancel_pending_on_position_close", True)),
            "cancel_pending_when_market_in_profit_enabled": bool(
                raw_cfg.get("cancel_pending_when_market_in_profit_enabled", True)
            ),
            "cancel_pending_when_market_in_profit_progress_min": min(
                max(float(raw_cfg.get("cancel_pending_when_market_in_profit_progress_min", 0.30) or 0.0), 0.0),
                1.0,
            ),
            "disable_pending_when_filter_hold": bool(raw_cfg.get("disable_pending_when_filter_hold", False)),
            "filter_hold_market_fraction": min(
                max(float(raw_cfg.get("filter_hold_market_fraction", 0.35) or 0.0), 0.0),
                1.0,
            ),
            "filter_hold_small_market_adjust_levels_enabled": bool(
                raw_cfg.get("filter_hold_small_market_adjust_levels_enabled", True)
            ),
            "filter_hold_small_market_allow_buy": bool(
                raw_cfg.get("filter_hold_small_market_allow_buy", False)
            ),
            "filter_hold_small_market_allow_sell": bool(
                raw_cfg.get("filter_hold_small_market_allow_sell", True)
            ),
            "filter_hold_small_market_retrace_on_mature_enabled": bool(
                raw_cfg.get("filter_hold_small_market_retrace_on_mature_enabled", False)
            ),
            "filter_hold_small_market_retrace_on_mature_sell_enabled": bool(
                raw_cfg.get("filter_hold_small_market_retrace_on_mature_sell_enabled", False)
            ),
            "filter_hold_small_market_confidence_min": max(
                float(raw_cfg.get("filter_hold_small_market_confidence_min", 0.88) or 0.0),
                0.0,
            ),
            "filter_hold_small_market_predicted_pips_min": max(
                float(raw_cfg.get("filter_hold_small_market_predicted_pips_min", 4.8) or 0.0),
                0.0,
            ),
            "filter_hold_small_market_require_aligned_context": bool(
                raw_cfg.get("filter_hold_small_market_require_aligned_context", True)
            ),
            "filter_hold_small_market_tp_min_pips": max(
                float(raw_cfg.get("filter_hold_small_market_tp_min_pips", 2.5) or 0.0),
                0.0,
            ),
            "filter_hold_small_market_sl_floor_pips": max(
                float(raw_cfg.get("filter_hold_small_market_sl_floor_pips", 3.0) or 0.0),
                0.0,
            ),
            "filter_hold_small_market_sl_max_tp_ratio": min(
                max(float(raw_cfg.get("filter_hold_small_market_sl_max_tp_ratio", 0.85) or 0.0), 0.1),
                1.0,
            ),
            "filter_hold_small_market_retrace_on_soft_contradiction": bool(
                raw_cfg.get("filter_hold_small_market_retrace_on_soft_contradiction", True)
            ),
            "filter_hold_small_market_retrace_on_market_rejection": bool(
                raw_cfg.get("filter_hold_small_market_retrace_on_market_rejection", True)
            ),
            "filter_hold_small_market_retrace_on_adverse_extreme": bool(
                raw_cfg.get("filter_hold_small_market_retrace_on_adverse_extreme", True)
            ),
            "split_retrace_filter_opposite_retrace_only_enabled": bool(
                raw_cfg.get("split_retrace_filter_opposite_retrace_only_enabled", True)
            ),
            "split_retrace_filter_opposite_range_vs_avg_column": str(
                raw_cfg.get("split_retrace_filter_opposite_range_vs_avg_column", "RangeVsAvg6") or "RangeVsAvg6"
            ),
            "split_retrace_filter_opposite_range_vs_avg_min": max(
                float(raw_cfg.get("split_retrace_filter_opposite_range_vs_avg_min", 1.20) or 0.0),
                0.0,
            ),
            "split_retrace_filter_opposite_range_pips_min": max(
                float(raw_cfg.get("split_retrace_filter_opposite_range_pips_min", 5.5) or 0.0),
                0.0,
            ),
            "split_retrace_filter_opposite_wick_ratio_min": min(
                max(float(raw_cfg.get("split_retrace_filter_opposite_wick_ratio_min", 0.35) or 0.0), 0.0),
                1.0,
            ),
            "split_retrace_filter_opposite_rejection_override": bool(
                raw_cfg.get("split_retrace_filter_opposite_rejection_override", True)
            ),
            "cluster_guard_enabled": bool(raw_cfg.get("cluster_guard_enabled", False)),
            "cluster_guard_symbol_side_max_open_positions": max(
                int(raw_cfg.get("cluster_guard_symbol_side_max_open_positions", 3) or 0),
                1,
            ),
            "cluster_guard_symbol_side_max_pending_orders": max(
                int(raw_cfg.get("cluster_guard_symbol_side_max_pending_orders", 1) or 0),
                0,
            ),
            "cluster_guard_cancel_pending_open_positions_min": max(
                int(raw_cfg.get("cluster_guard_cancel_pending_open_positions_min", 2) or 0),
                1,
            ),
            "cluster_guard_cancel_pending_progress_min": min(
                max(float(raw_cfg.get("cluster_guard_cancel_pending_progress_min", 0.45) or 0.0), 0.0),
                1.0,
            ),
            "cluster_guard_skip_new_entries_open_positions_min": max(
                int(raw_cfg.get("cluster_guard_skip_new_entries_open_positions_min", 2) or 0),
                1,
            ),
            "cluster_guard_skip_new_entries_pending_orders_min": max(
                int(raw_cfg.get("cluster_guard_skip_new_entries_pending_orders_min", 2) or 0),
                0,
            ),
            "cluster_guard_skip_new_entries_progress_min": min(
                max(float(raw_cfg.get("cluster_guard_skip_new_entries_progress_min", 0.55) or 0.0), 0.0),
                1.0,
            ),
            "cluster_guard_allow_market_on_strong_continuation": bool(
                raw_cfg.get("cluster_guard_allow_market_on_strong_continuation", True)
            ),
            "cluster_guard_disable_pending_on_strong_continuation": bool(
                raw_cfg.get("cluster_guard_disable_pending_on_strong_continuation", True)
            ),
            "cluster_guard_retrace_only_on_adverse_extreme": bool(
                raw_cfg.get("cluster_guard_retrace_only_on_adverse_extreme", False)
            ),
            "cluster_guard_retrace_only_requires_rejection": bool(
                raw_cfg.get("cluster_guard_retrace_only_requires_rejection", False)
            ),
            "cluster_guard_strong_continuation_primary_confidence_min": min(
                max(
                    float(raw_cfg.get("cluster_guard_strong_continuation_primary_confidence_min", 0.90) or 0.0),
                    0.0,
                ),
                1.0,
            ),
            "cluster_guard_strong_continuation_predicted_pips_min": max(
                float(raw_cfg.get("cluster_guard_strong_continuation_predicted_pips_min", 4.5) or 0.0),
                0.0,
            ),
            "comment_prefix": str(raw_cfg.get("comment_prefix", "EM")),
        }

    def _get_entry_grid_settings(self) -> dict[str, Any]:
        """Configuracion opcional para ejecutar una grilla/ladder de entrada en varias patas."""
        raw_cfg = self.config.get("entry_grid", {}) or {}
        mode = str(raw_cfg.get("mode", "") or "").strip().lower()

        def _as_float(value: Any, default: float) -> float:
            try:
                return float(value if value is not None else default)
            except Exception:
                return float(default)

        def _as_int(value: Any, default: int) -> int:
            try:
                return int(value if value is not None else default)
            except Exception:
                return int(default)

        legs_cfg = raw_cfg.get("legs", []) or []
        parsed_legs: list[dict[str, Any]] = []
        for idx, leg_cfg in enumerate(legs_cfg, start=1):
            leg_cfg = leg_cfg or {}
            parsed_legs.append(
                {
                    "leg_id": str(leg_cfg.get("leg_id", f"leg_{idx}") or f"leg_{idx}"),
                    "entry_type": str(leg_cfg.get("entry_type", "limit") or "limit").strip().lower(),
                    "volume_weight": max(_as_float(leg_cfg.get("volume_weight"), 0.0), 0.0),
                    "spacing_fraction_of_stop": min(
                        max(_as_float(leg_cfg.get("spacing_fraction_of_stop"), 0.0), 0.0),
                        1.0,
                    ),
                    "expiry_bars": max(_as_int(leg_cfg.get("expiry_bars"), 0), 0),
                }
            )

        apply_to_profiles = raw_cfg.get("apply_to_profiles", []) or []
        normalized_profiles = [
            str(profile or "").strip().lower()
            for profile in apply_to_profiles
            if str(profile or "").strip()
        ]

        runner_legs = max(_as_int(raw_cfg.get("runner_legs"), 1), 0)
        if runner_legs <= 0:
            runner_legs = 1

        return {
            "enabled": bool(raw_cfg.get("enabled", False)) and mode == "risk_based_ladder",
            "mode": mode,
            "apply_to_profiles": normalized_profiles or ["strong_trend", "normal_trend"],
            "require_confirmed_bundle": bool(raw_cfg.get("require_confirmed_bundle", True)),
            "allow_filter_hold_variant": bool(raw_cfg.get("allow_filter_hold_variant", False)),
            "runner_legs": runner_legs,
            "legs": parsed_legs,
            "comment_prefix": str(raw_cfg.get("comment_prefix", "EG")),
        }

    def _get_entry_staging_settings(self) -> dict[str, Any]:
        """Configuracion opcional para retener senales HOLD y esperar una mejor entrada."""
        raw_cfg = self.config.get("entry_staging", {}) or {}
        mode = str(raw_cfg.get("mode", "") or "").strip().lower()

        def _as_float(key: str, default: float) -> float:
            try:
                return float(raw_cfg.get(key, default) or 0.0)
            except Exception:
                return float(default)

        def _as_int(key: str, default: int) -> int:
            try:
                return int(raw_cfg.get(key, default) or 0)
            except Exception:
                return int(default)

        return {
            "enabled": bool(raw_cfg.get("enabled", False)) and mode == "candidate_retrace",
            "mode": mode,
            "max_stage_bars": max(_as_int("max_stage_bars", 2), 1),
            "min_primary_confidence": max(_as_float("min_primary_confidence", 0.68), 0.0),
            "min_abs_predicted_pips": max(_as_float("min_abs_predicted_pips", 3.2), 0.0),
            "allow_stage_on_filter_contradiction_alignment": bool(
                raw_cfg.get("allow_stage_on_filter_contradiction_alignment", True)
            ),
            "contradiction_stage_primary_confidence_min": max(
                _as_float("contradiction_stage_primary_confidence_min", 0.70),
                0.0,
            ),
            "contradiction_stage_predicted_pips_min": max(
                _as_float("contradiction_stage_predicted_pips_min", 3.5),
                0.0,
            ),
            "contradiction_stage_adx_min": max(
                _as_float("contradiction_stage_adx_min", 8.0),
                0.0,
            ),
            "contradiction_stage_roc_abs_min": max(
                _as_float("contradiction_stage_roc_abs_min", 0.00060),
                0.0,
            ),
            "contradiction_stage_directional_volume_abs_min": max(
                _as_float("contradiction_stage_directional_volume_abs_min", 0.50),
                0.0,
            ),
            "contradiction_stage_retrace_fraction": min(
                max(_as_float("contradiction_stage_retrace_fraction", 0.25), 0.0),
                1.0,
            ),
            "contradiction_stage_max_stage_bars": max(
                _as_int("contradiction_stage_max_stage_bars", 1),
                1,
            ),
            "contradiction_stage_breakout_partial_fraction": min(
                max(_as_float("contradiction_stage_breakout_partial_fraction", 0.20), 0.0),
                1.0,
            ),
            "allow_stage_on_strong_primary_filter_hold": bool(
                raw_cfg.get("allow_stage_on_strong_primary_filter_hold", True)
            ),
            "strong_primary_hold_confidence_min": max(
                _as_float("strong_primary_hold_confidence_min", 0.90),
                0.0,
            ),
            "strong_primary_hold_predicted_pips_min": max(
                _as_float("strong_primary_hold_predicted_pips_min", 4.0),
                0.0,
            ),
            "strong_primary_hold_adx_min": max(
                _as_float("strong_primary_hold_adx_min", 25.0),
                0.0,
            ),
            "strong_primary_hold_roc_abs_min": max(
                _as_float("strong_primary_hold_roc_abs_min", 0.00035),
                0.0,
            ),
            "strong_primary_hold_retrace_fraction": min(
                max(_as_float("strong_primary_hold_retrace_fraction", 0.25), 0.0),
                1.0,
            ),
            "strong_primary_hold_max_stage_bars": max(
                _as_int("strong_primary_hold_max_stage_bars", 1),
                1,
            ),
            "strong_primary_hold_breakout_partial_fraction": min(
                max(_as_float("strong_primary_hold_breakout_partial_fraction", 0.20), 0.0),
                1.0,
            ),
            "allow_stage_on_medium_primary_filter_hold": bool(
                raw_cfg.get("allow_stage_on_medium_primary_filter_hold", True)
            ),
            "medium_primary_hold_confidence_min": max(
                _as_float("medium_primary_hold_confidence_min", 0.70),
                0.0,
            ),
            "medium_primary_hold_predicted_pips_min": max(
                _as_float("medium_primary_hold_predicted_pips_min", 3.6),
                0.0,
            ),
            "medium_primary_hold_adx_min": max(
                _as_float("medium_primary_hold_adx_min", 22.0),
                0.0,
            ),
            "medium_primary_hold_roc_abs_min": max(
                _as_float("medium_primary_hold_roc_abs_min", 0.00015),
                0.0,
            ),
            "medium_primary_hold_roc_column": str(
                raw_cfg.get("medium_primary_hold_roc_column", "ROC_3") or "ROC_3"
            ),
            "medium_primary_hold_retrace_fraction": min(
                max(_as_float("medium_primary_hold_retrace_fraction", 0.25), 0.0),
                1.0,
            ),
            "medium_primary_hold_max_stage_bars": max(
                _as_int("medium_primary_hold_max_stage_bars", 1),
                1,
            ),
            "medium_primary_hold_breakout_partial_fraction": min(
                max(_as_float("medium_primary_hold_breakout_partial_fraction", 0.15), 0.0),
                1.0,
            ),
            "medium_primary_hold_immediate_market_enabled": bool(
                raw_cfg.get("medium_primary_hold_immediate_market_enabled", False)
            ),
            "medium_primary_hold_partial_enabled": bool(
                raw_cfg.get("medium_primary_hold_partial_enabled", False)
            ),
            "medium_primary_hold_partial_fraction": min(
                max(_as_float("medium_primary_hold_partial_fraction", 0.10), 0.0),
                1.0,
            ),
            "medium_primary_hold_activation_delay_bars": max(
                _as_int("medium_primary_hold_activation_delay_bars", 1),
                0,
            ),
            "medium_primary_hold_rebuild_on_activation": bool(
                raw_cfg.get("medium_primary_hold_rebuild_on_activation", True)
            ),
            "medium_primary_hold_grace_enabled": bool(
                raw_cfg.get("medium_primary_hold_grace_enabled", True)
            ),
            "medium_primary_hold_grace_max_bars": max(
                _as_int("medium_primary_hold_grace_max_bars", 4),
                1,
            ),
            "medium_primary_hold_grace_predicted_pips_min": max(
                _as_float("medium_primary_hold_grace_predicted_pips_min", 3.6),
                0.0,
            ),
            "medium_primary_hold_preserve_if_armed_on_confidence_drop": bool(
                raw_cfg.get("medium_primary_hold_preserve_if_armed_on_confidence_drop", True)
            ),
            "allow_stage_on_filter_lead_structural": bool(
                raw_cfg.get("allow_stage_on_filter_lead_structural", True)
            ),
            "filter_lead_structural_filter_confidence_min": max(
                _as_float("filter_lead_structural_filter_confidence_min", 0.60),
                0.0,
            ),
            "filter_lead_structural_primary_hold_confidence_min": max(
                _as_float("filter_lead_structural_primary_hold_confidence_min", 0.55),
                0.0,
            ),
            "filter_lead_structural_predicted_pips_floor": max(
                _as_float("filter_lead_structural_predicted_pips_floor", 3.6),
                0.0,
            ),
            "filter_lead_structural_short_momentum_column": str(
                raw_cfg.get("filter_lead_structural_short_momentum_column", "ROC_3") or "ROC_3"
            ),
            "filter_lead_structural_structure_score3_column": str(
                raw_cfg.get("filter_lead_structural_structure_score3_column", "StructureScore3")
                or "StructureScore3"
            ),
            "filter_lead_structural_structure_score6_column": str(
                raw_cfg.get("filter_lead_structural_structure_score6_column", "StructureScore6")
                or "StructureScore6"
            ),
            "filter_lead_structural_break_above_prev_high_column": str(
                raw_cfg.get("filter_lead_structural_break_above_prev_high_column", "BreakAbovePrevHigh")
                or "BreakAbovePrevHigh"
            ),
            "filter_lead_structural_break_below_prev_low_column": str(
                raw_cfg.get("filter_lead_structural_break_below_prev_low_column", "BreakBelowPrevLow")
                or "BreakBelowPrevLow"
            ),
            "filter_lead_structural_break_above_recent_high3_column": str(
                raw_cfg.get("filter_lead_structural_break_above_recent_high3_column", "BreakAboveRecentHigh3")
                or "BreakAboveRecentHigh3"
            ),
            "filter_lead_structural_break_below_recent_low3_column": str(
                raw_cfg.get("filter_lead_structural_break_below_recent_low3_column", "BreakBelowRecentLow3")
                or "BreakBelowRecentLow3"
            ),
            "filter_lead_structural_retrace_fraction": min(
                max(_as_float("filter_lead_structural_retrace_fraction", 0.20), 0.0),
                1.0,
            ),
            "filter_lead_structural_max_stage_bars": max(
                _as_int("filter_lead_structural_max_stage_bars", 1),
                1,
            ),
            "filter_lead_structural_breakout_partial_fraction": min(
                max(_as_float("filter_lead_structural_breakout_partial_fraction", 0.12), 0.0),
                1.0,
            ),
            "filter_lead_structural_short_roc_abs_min": max(
                _as_float("filter_lead_structural_short_roc_abs_min", 0.00015),
                0.0,
            ),
            "filter_lead_structural_adx_min": max(
                _as_float("filter_lead_structural_adx_min", 18.0),
                0.0,
            ),
            "filter_lead_structural_directional_volume_column": str(
                raw_cfg.get("filter_lead_structural_directional_volume_column", "DirectionalVolumeProxy_ZScore_20")
                or "DirectionalVolumeProxy_ZScore_20"
            ),
            "filter_lead_structural_directional_volume_abs_min": max(
                _as_float("filter_lead_structural_directional_volume_abs_min", 0.10),
                0.0,
            ),
            "filter_lead_structural_structure_score_abs_min": max(
                _as_float("filter_lead_structural_structure_score_abs_min", 0.20),
                0.0,
            ),
            "allow_stage_on_early_structural_reversal": bool(
                raw_cfg.get("allow_stage_on_early_structural_reversal", True)
            ),
            "early_reversal_short_momentum_column": str(
                raw_cfg.get("early_reversal_short_momentum_column", "ROC_3") or "ROC_3"
            ),
            "early_reversal_directional_volume_column": str(
                raw_cfg.get("early_reversal_directional_volume_column", "DirectionalVolumeProxy_ZScore_20")
                or "DirectionalVolumeProxy_ZScore_20"
            ),
            "early_reversal_structure_score3_column": str(
                raw_cfg.get("early_reversal_structure_score3_column", "StructureScore3") or "StructureScore3"
            ),
            "early_reversal_structure_score6_column": str(
                raw_cfg.get("early_reversal_structure_score6_column", "StructureScore6") or "StructureScore6"
            ),
            "early_reversal_break_above_prev_high_column": str(
                raw_cfg.get("early_reversal_break_above_prev_high_column", "BreakAbovePrevHigh")
                or "BreakAbovePrevHigh"
            ),
            "early_reversal_break_below_prev_low_column": str(
                raw_cfg.get("early_reversal_break_below_prev_low_column", "BreakBelowPrevLow")
                or "BreakBelowPrevLow"
            ),
            "early_reversal_break_above_recent_high3_column": str(
                raw_cfg.get("early_reversal_break_above_recent_high3_column", "BreakAboveRecentHigh3")
                or "BreakAboveRecentHigh3"
            ),
            "early_reversal_break_below_recent_low3_column": str(
                raw_cfg.get("early_reversal_break_below_recent_low3_column", "BreakBelowRecentLow3")
                or "BreakBelowRecentLow3"
            ),
            "early_reversal_retrace_fraction": min(
                max(_as_float("early_reversal_retrace_fraction", 0.20), 0.0),
                1.0,
            ),
            "early_reversal_max_stage_bars": max(
                _as_int("early_reversal_max_stage_bars", 1),
                1,
            ),
            "early_reversal_breakout_partial_fraction": min(
                max(_as_float("early_reversal_breakout_partial_fraction", 0.15), 0.0),
                1.0,
            ),
            "early_reversal_confidence_min": max(
                _as_float("early_reversal_confidence_min", 0.78),
                0.0,
            ),
            "early_reversal_predicted_pips_min": max(
                _as_float("early_reversal_predicted_pips_min", 4.5),
                0.0,
            ),
            "early_reversal_short_roc_abs_min": max(
                _as_float("early_reversal_short_roc_abs_min", 0.00015),
                0.0,
            ),
            "early_reversal_adx_min": max(
                _as_float("early_reversal_adx_min", 22.0),
                0.0,
            ),
            "early_reversal_directional_volume_abs_min": max(
                _as_float("early_reversal_directional_volume_abs_min", 0.25),
                0.0,
            ),
            "early_reversal_structure_score_abs_min": max(
                _as_float("early_reversal_structure_score_abs_min", 0.30),
                0.0,
            ),
            "early_reversal_relaxed_enabled": bool(
                raw_cfg.get("early_reversal_relaxed_enabled", True)
            ),
            "early_reversal_relaxed_confidence_min": max(
                _as_float("early_reversal_relaxed_confidence_min", 0.70),
                0.0,
            ),
            "early_reversal_relaxed_predicted_pips_min": max(
                _as_float("early_reversal_relaxed_predicted_pips_min", 3.6),
                0.0,
            ),
            "early_reversal_relaxed_adx_min": max(
                _as_float("early_reversal_relaxed_adx_min", 30.0),
                0.0,
            ),
            "early_reversal_relaxed_short_roc_abs_min": max(
                _as_float("early_reversal_relaxed_short_roc_abs_min", 0.00001),
                0.0,
            ),
            "early_reversal_relax_directional_volume_if_adx_strong": bool(
                raw_cfg.get("early_reversal_relax_directional_volume_if_adx_strong", True)
            ),
            "direct_confirmed_revalidate_on_activation": bool(
                raw_cfg.get("direct_confirmed_revalidate_on_activation", True)
            ),
            "direct_confirmed_cancel_if_primary_mismatch": bool(
                raw_cfg.get("direct_confirmed_cancel_if_primary_mismatch", True)
            ),
            "direct_confirmed_cancel_if_confidence_drops": bool(
                raw_cfg.get("direct_confirmed_cancel_if_confidence_drops", True)
            ),
            "direct_confirmed_cancel_if_predicted_move_drops": bool(
                raw_cfg.get("direct_confirmed_cancel_if_predicted_move_drops", True)
            ),
            "direct_confirmed_activation_confidence_min": max(
                _as_float("direct_confirmed_activation_confidence_min", 0.60),
                0.0,
            ),
            "direct_confirmed_activation_predicted_pips_min": max(
                _as_float("direct_confirmed_activation_predicted_pips_min", 3.5),
                0.0,
            ),
            "direct_confirmed_extend_if_aligned": bool(
                raw_cfg.get("direct_confirmed_extend_if_aligned", True)
            ),
            "direct_confirmed_extension_bars": max(
                _as_int("direct_confirmed_extension_bars", 1),
                1,
            ),
            "direct_confirmed_max_extensions": max(
                _as_int("direct_confirmed_max_extensions", 1),
                0,
            ),
            "direct_confirmed_extension_confidence_min": max(
                _as_float("direct_confirmed_extension_confidence_min", 0.75),
                0.0,
            ),
            "direct_confirmed_extension_predicted_pips_min": max(
                _as_float("direct_confirmed_extension_predicted_pips_min", 4.0),
                0.0,
            ),
            "direct_confirmed_hold_grace_enabled": bool(
                raw_cfg.get("direct_confirmed_hold_grace_enabled", True)
            ),
            "direct_confirmed_hold_grace_max_bars": max(
                _as_int("direct_confirmed_hold_grace_max_bars", 4),
                0,
            ),
            "direct_confirmed_hold_grace_predicted_pips_min": max(
                _as_float("direct_confirmed_hold_grace_predicted_pips_min", 3.5),
                0.0,
            ),
            "direct_confirmed_keep_better_active_candidate": bool(
                raw_cfg.get("direct_confirmed_keep_better_active_candidate", True)
            ),
            "direct_confirmed_strong_trend_retrace_fraction": min(
                max(_as_float("direct_confirmed_strong_trend_retrace_fraction", 0.20), 0.0),
                1.0,
            ),
            "direct_confirmed_normal_trend_retrace_fraction": min(
                max(_as_float("direct_confirmed_normal_trend_retrace_fraction", 0.25), 0.0),
                1.0,
            ),
            "direct_confirmed_weak_trend_retrace_fraction": min(
                max(_as_float("direct_confirmed_weak_trend_retrace_fraction", 0.40), 0.0),
                1.0,
            ),
            "direct_confirmed_max_entry_improvement_pct_of_predicted_pips": min(
                max(_as_float("direct_confirmed_max_entry_improvement_pct_of_predicted_pips", 0.30), 0.0),
                1.0,
            ),
            "direct_confirmed_max_entry_improvement_min_pips": max(
                _as_float("direct_confirmed_max_entry_improvement_min_pips", 0.8),
                0.0,
            ),
            "direct_confirmed_strong_trend_partial_enabled": bool(
                raw_cfg.get("direct_confirmed_strong_trend_partial_enabled", True)
            ),
            "direct_confirmed_strong_trend_partial_fraction": min(
                max(_as_float("direct_confirmed_strong_trend_partial_fraction", 0.15), 0.0),
                1.0,
            ),
            "direct_confirmed_strong_trend_confidence_min": max(
                _as_float("direct_confirmed_strong_trend_confidence_min", 0.85),
                0.0,
            ),
            "direct_confirmed_strong_trend_predicted_pips_min": max(
                _as_float("direct_confirmed_strong_trend_predicted_pips_min", 6.0),
                0.0,
            ),
            "direct_confirmed_normal_trend_partial_enabled": bool(
                raw_cfg.get("direct_confirmed_normal_trend_partial_enabled", True)
            ),
            "direct_confirmed_normal_trend_partial_fraction": min(
                max(_as_float("direct_confirmed_normal_trend_partial_fraction", 0.10), 0.0),
                1.0,
            ),
            "direct_confirmed_normal_trend_confidence_min": max(
                _as_float("direct_confirmed_normal_trend_confidence_min", 0.75),
                0.0,
            ),
            "direct_confirmed_normal_trend_predicted_pips_min": max(
                _as_float("direct_confirmed_normal_trend_predicted_pips_min", 4.0),
                0.0,
            ),
            "direct_confirmed_immediate_extreme_entry_guard_enabled": bool(
                raw_cfg.get("direct_confirmed_immediate_extreme_entry_guard_enabled", True)
            ),
            "direct_confirmed_immediate_extreme_entry_range_fraction": min(
                max(_as_float("direct_confirmed_immediate_extreme_entry_range_fraction", 0.20), 0.0),
                0.50,
            ),
            "direct_confirmed_immediate_extreme_entry_min_range_pips": max(
                _as_float("direct_confirmed_immediate_extreme_entry_min_range_pips", 2.0),
                0.0,
            ),
            "near_trigger_immediate_partial_enabled": bool(
                raw_cfg.get("near_trigger_immediate_partial_enabled", True)
            ),
            "near_trigger_immediate_partial_fraction": min(
                max(_as_float("near_trigger_immediate_partial_fraction", 0.20), 0.0),
                1.0,
            ),
            "near_trigger_immediate_partial_entry_improvement_pips_max": max(
                _as_float("near_trigger_immediate_partial_entry_improvement_pips_max", 1.6),
                0.0,
            ),
            "near_trigger_immediate_partial_confidence_min": max(
                _as_float("near_trigger_immediate_partial_confidence_min", 0.75),
                0.0,
            ),
            "near_trigger_immediate_partial_predicted_pips_min": max(
                _as_float("near_trigger_immediate_partial_predicted_pips_min", 3.8),
                0.0,
            ),
            "execution_confirmation_m1_enabled": bool(
                raw_cfg.get("execution_confirmation_m1_enabled", True)
            ),
            "execution_confirmation_m1_timeframe": str(
                raw_cfg.get("execution_confirmation_m1_timeframe", "M1") or "M1"
            ),
            "execution_confirmation_m1_n_bars": max(
                _as_int("execution_confirmation_m1_n_bars", 180),
                30,
            ),
            "execution_confirmation_m1_use_cache": bool(
                raw_cfg.get("execution_confirmation_m1_use_cache", False)
            ),
            "execution_confirmation_m1_cache_expiry_minutes": max(
                _as_float("execution_confirmation_m1_cache_expiry_minutes", 1.0),
                0.0,
            ),
            "execution_confirmation_m1_fail_open_on_missing": bool(
                raw_cfg.get("execution_confirmation_m1_fail_open_on_missing", True)
            ),
            "execution_confirmation_m1_apply_on_near_trigger_partial": bool(
                raw_cfg.get("execution_confirmation_m1_apply_on_near_trigger_partial", True)
            ),
            "execution_confirmation_m1_apply_on_stage_activation": bool(
                raw_cfg.get("execution_confirmation_m1_apply_on_stage_activation", True)
            ),
            "execution_confirmation_m1_breakout_lookback_bars": max(
                _as_int("execution_confirmation_m1_breakout_lookback_bars", 2),
                1,
            ),
            "execution_confirmation_m1_min_alignment_hits": max(
                _as_int("execution_confirmation_m1_min_alignment_hits", 3),
                1,
            ),
            "execution_confirmation_m1_min_score": min(
                max(_as_float("execution_confirmation_m1_min_score", 0.50), 0.0),
                1.0,
            ),
            "execution_confirmation_m1_min_roc1_pips": max(
                _as_float("execution_confirmation_m1_min_roc1_pips", 0.10),
                0.0,
            ),
            "execution_confirmation_m1_min_roc3_pips": max(
                _as_float("execution_confirmation_m1_min_roc3_pips", 0.25),
                0.0,
            ),
            "execution_confirmation_m1_min_directional_volume_abs": max(
                _as_float("execution_confirmation_m1_min_directional_volume_abs", 0.0),
                0.0,
            ),
            "execution_confirmation_m1_min_tick_volume_zscore": _as_float(
                "execution_confirmation_m1_min_tick_volume_zscore",
                -0.10,
            ),
            "execution_confirmation_m1_min_close_location_abs": min(
                max(_as_float("execution_confirmation_m1_min_close_location_abs", 0.20), 0.0),
                1.0,
            ),
            "execution_confirmation_m1_max_opposite_wick_ratio": min(
                max(_as_float("execution_confirmation_m1_max_opposite_wick_ratio", 0.60), 0.0),
                1.0,
            ),
            "execution_confirmation_m1_max_stretch_vs_avg_range": max(
                _as_float("execution_confirmation_m1_max_stretch_vs_avg_range", 1.40),
                0.0,
            ),
            "execution_confirmation_m1_strong_alignment_override_enabled": bool(
                raw_cfg.get("execution_confirmation_m1_strong_alignment_override_enabled", True)
            ),
            "execution_confirmation_m1_strong_alignment_min_hits": max(
                _as_int("execution_confirmation_m1_strong_alignment_min_hits", 5),
                1,
            ),
            "execution_confirmation_m1_strong_alignment_min_score": min(
                max(_as_float("execution_confirmation_m1_strong_alignment_min_score", 0.70), 0.0),
                1.0,
            ),
            "execution_confirmation_m1_strong_alignment_max_stretch_vs_avg_range": max(
                _as_float("execution_confirmation_m1_strong_alignment_max_stretch_vs_avg_range", 1.85),
                0.0,
            ),
            "execution_confirmation_m1_strong_alignment_require_breakout": bool(
                raw_cfg.get("execution_confirmation_m1_strong_alignment_require_breakout", True)
            ),
            "pilot_entry_enabled": bool(raw_cfg.get("pilot_entry_enabled", False)),
            "pilot_confidence_min": max(_as_float("pilot_confidence_min", 0.60), 0.0),
            "pilot_fraction_of_full_size": min(max(_as_float("pilot_fraction_of_full_size", 0.25), 0.0), 1.0),
            "pilot_allow_on_filter_contradiction": bool(raw_cfg.get("pilot_allow_on_filter_contradiction", True)),
            "pilot_convert_to_staged": bool(raw_cfg.get("pilot_convert_to_staged", True)),
            "pilot_market_fallback_enabled": bool(raw_cfg.get("pilot_market_fallback_enabled", False)),
            "pilot_retrace_fraction": min(
                max(_as_float("pilot_retrace_fraction", 0.50), 0.0),
                1.0,
            ),
            "pilot_allowed_profiles": {
                str(profile).strip().lower()
                for profile in (raw_cfg.get("pilot_allowed_profiles") or ["strong_trend", "normal_trend"])
                if str(profile).strip()
            },
            "pilot_revalidate_on_activation": bool(raw_cfg.get("pilot_revalidate_on_activation", True)),
            "pilot_cancel_if_primary_mismatch": bool(raw_cfg.get("pilot_cancel_if_primary_mismatch", True)),
            "pilot_cancel_if_filter_contradicted": bool(raw_cfg.get("pilot_cancel_if_filter_contradicted", True)),
            "pilot_cancel_if_confidence_drops": bool(raw_cfg.get("pilot_cancel_if_confidence_drops", True)),
            "pilot_cancel_if_predicted_move_drops": bool(raw_cfg.get("pilot_cancel_if_predicted_move_drops", True)),
            "adaptive_profile_enabled": bool(raw_cfg.get("adaptive_profile_enabled", True)),
            "adaptive_roc_column": str(raw_cfg.get("adaptive_roc_column", "ROC_6") or "ROC_6"),
            "adaptive_adx_column": str(raw_cfg.get("adaptive_adx_column", "ADX_14") or "ADX_14"),
            "adaptive_directional_volume_column": str(
                raw_cfg.get("adaptive_directional_volume_column", "DirectionalVolumeProxy_ZScore_20")
                or "DirectionalVolumeProxy_ZScore_20"
            ),
            "strong_trend_adx_min": _as_float("strong_trend_adx_min", 35.0),
            "normal_trend_adx_min": _as_float("normal_trend_adx_min", 22.0),
            "strong_trend_roc_abs_min": _as_float("strong_trend_roc_abs_min", 0.00045),
            "normal_trend_roc_abs_min": _as_float("normal_trend_roc_abs_min", 0.00020),
            "strong_trend_volume_abs_min": _as_float("strong_trend_volume_abs_min", 0.35),
            "normal_trend_volume_abs_min": _as_float("normal_trend_volume_abs_min", 0.05),
            "strong_trend_retrace_fraction": min(max(_as_float("strong_trend_retrace_fraction", 0.20), 0.0), 1.0),
            "normal_trend_retrace_fraction": min(max(_as_float("normal_trend_retrace_fraction", 0.35), 0.0), 1.0),
            "weak_trend_retrace_fraction": min(max(_as_float("weak_trend_retrace_fraction", 0.55), 0.0), 1.0),
            "strong_trend_max_stage_bars": max(_as_int("strong_trend_max_stage_bars", 1), 1),
            "normal_trend_max_stage_bars": max(_as_int("normal_trend_max_stage_bars", 2), 1),
            "weak_trend_max_stage_bars": max(_as_int("weak_trend_max_stage_bars", 2), 1),
            "strong_trend_breakout_partial_fraction": min(max(_as_float("strong_trend_breakout_partial_fraction", 0.35), 0.0), 1.0),
            "normal_trend_breakout_partial_fraction": min(max(_as_float("normal_trend_breakout_partial_fraction", 0.20), 0.0), 1.0),
            "weak_trend_breakout_partial_fraction": min(max(_as_float("weak_trend_breakout_partial_fraction", 0.0), 0.0), 1.0),
            "breakout_trigger_fraction_of_stop": min(max(_as_float("breakout_trigger_fraction_of_stop", 0.12), 0.0), 1.0),
            "breakout_min_trigger_pips": max(_as_float("breakout_min_trigger_pips", 0.8), 0.0),
            "allow_stage_on_filter_hold": bool(raw_cfg.get("allow_stage_on_filter_hold", True)),
            "block_stage_on_filter_contradiction": bool(raw_cfg.get("block_stage_on_filter_contradiction", True)),
            "soft_support_score_min": _as_float("soft_support_score_min", -0.02),
            "retrace_trigger_fraction_of_stop": min(
                max(_as_float("retrace_trigger_fraction_of_stop", 0.35), 0.0),
                1.0,
            ),
            "min_entry_improvement_pips": max(_as_float("min_entry_improvement_pips", 0.8), 0.0),
            "allow_breakout_if_filter_upgrades": bool(raw_cfg.get("allow_breakout_if_filter_upgrades", True)),
            "breakout_min_support_score": _as_float("breakout_min_support_score", 0.08),
            "cancel_on_opposite_primary_signal": bool(raw_cfg.get("cancel_on_opposite_primary_signal", True)),
            "cancel_on_filter_contradiction": bool(raw_cfg.get("cancel_on_filter_contradiction", True)),
            "convert_direct_filter_hold_to_staged": bool(raw_cfg.get("convert_direct_filter_hold_to_staged", False)),
            "convert_direct_confirmed_to_staged": bool(raw_cfg.get("convert_direct_confirmed_to_staged", False)),
            "context_guard_enabled": bool(raw_cfg.get("context_guard_enabled", True)),
            "context_guard_hard_block_direct": bool(raw_cfg.get("context_guard_hard_block_direct", True)),
            "context_guard_soft_disable_pending": bool(raw_cfg.get("context_guard_soft_disable_pending", True)),
            "context_guard_directional_volume_column": str(
                raw_cfg.get("context_guard_directional_volume_column", "DirectionalVolumeProxy_ZScore_20")
                or "DirectionalVolumeProxy_ZScore_20"
            ),
            "context_guard_close_location_column": str(
                raw_cfg.get("context_guard_close_location_column", "CloseLocationValue") or "CloseLocationValue"
            ),
            "context_guard_soft_close_location_abs_min": min(
                max(_as_float("context_guard_soft_close_location_abs_min", 0.45), 0.0),
                1.0,
            ),
            "context_guard_hard_close_location_abs_min": min(
                max(_as_float("context_guard_hard_close_location_abs_min", 0.80), 0.0),
                1.0,
            ),
            "context_guard_soft_directional_volume_abs_min": max(
                _as_float("context_guard_soft_directional_volume_abs_min", 0.35),
                0.0,
            ),
            "context_guard_hard_directional_volume_abs_min": max(
                _as_float("context_guard_hard_directional_volume_abs_min", 0.90),
                0.0,
            ),
            "context_guard_use_body_direction": bool(raw_cfg.get("context_guard_use_body_direction", True)),
            "context_guard_soft_disable_market_on_extreme_rejection": bool(
                raw_cfg.get("context_guard_soft_disable_market_on_extreme_rejection", True)
            ),
            "context_guard_market_entry_extreme_range_fraction": min(
                max(_as_float("context_guard_market_entry_extreme_range_fraction", 0.18), 0.0),
                0.5,
            ),
            "context_guard_market_entry_min_range_pips": max(
                _as_float("context_guard_market_entry_min_range_pips", 2.0),
                0.0,
            ),
            "context_guard_market_entry_rejection_close_location_abs_min": min(
                max(_as_float("context_guard_market_entry_rejection_close_location_abs_min", 0.0), 0.0),
                1.0,
            ),
            "entry_quality_enabled": bool(raw_cfg.get("entry_quality_enabled", True)),
            "entry_quality_min_score_for_market": min(
                max(_as_float("entry_quality_min_score_for_market", 0.62), 0.0),
                1.0,
            ),
            "entry_quality_min_score_for_retrace": min(
                max(_as_float("entry_quality_min_score_for_retrace", 0.38), 0.0),
                1.0,
            ),
            "entry_quality_skip_on_low": bool(raw_cfg.get("entry_quality_skip_on_low", True)),
            "entry_quality_force_retrace_on_medium": bool(
                raw_cfg.get("entry_quality_force_retrace_on_medium", True)
            ),
            "entry_quality_force_retrace_on_opposite_filter_buy_high_clv_enabled": bool(
                raw_cfg.get("entry_quality_force_retrace_on_opposite_filter_buy_high_clv_enabled", False)
            ),
            "entry_quality_opposite_filter_buy_high_clv_min": min(
                max(_as_float("entry_quality_opposite_filter_buy_high_clv_min", 0.90), 0.0),
                1.0,
            ),
            "entry_quality_force_retrace_on_opposite_filter_buy_weak_volume_enabled": bool(
                raw_cfg.get("entry_quality_force_retrace_on_opposite_filter_buy_weak_volume_enabled", False)
            ),
            "entry_quality_opposite_filter_buy_market_score_min": min(
                max(_as_float("entry_quality_opposite_filter_buy_market_score_min", 0.75), 0.0),
                1.0,
            ),
            "entry_quality_opposite_filter_buy_dirvol_min": _as_float(
                "entry_quality_opposite_filter_buy_dirvol_min",
                0.0,
            ),
            "entry_quality_force_retrace_on_same_side_reentry_enabled": bool(
                raw_cfg.get("entry_quality_force_retrace_on_same_side_reentry_enabled", False)
            ),
            "entry_quality_same_side_reentry_min_open_positions": max(
                int(_as_float("entry_quality_same_side_reentry_min_open_positions", 1) or 1),
                1,
            ),
            "entry_quality_same_side_reentry_market_score_min": min(
                max(_as_float("entry_quality_same_side_reentry_market_score_min", 0.78), 0.0),
                1.0,
            ),
            "entry_quality_same_side_reentry_dirvol_min": _as_float(
                "entry_quality_same_side_reentry_dirvol_min",
                0.0,
            ),
            "entry_quality_force_retrace_on_mature_sell_non_sell_filter_enabled": bool(
                raw_cfg.get("entry_quality_force_retrace_on_mature_sell_non_sell_filter_enabled", False)
            ),
            "entry_quality_force_retrace_on_mature_sell_non_sell_filter_always": bool(
                raw_cfg.get("entry_quality_force_retrace_on_mature_sell_non_sell_filter_always", False)
            ),
            "entry_quality_force_retrace_on_mature_non_aligned_filter_enabled": bool(
                raw_cfg.get("entry_quality_force_retrace_on_mature_non_aligned_filter_enabled", False)
            ),
            "entry_quality_force_retrace_on_mature_non_aligned_filter_always": bool(
                raw_cfg.get("entry_quality_force_retrace_on_mature_non_aligned_filter_always", False)
            ),
            "entry_quality_mature_non_aligned_filter_market_score_min": min(
                max(_as_float("entry_quality_mature_non_aligned_filter_market_score_min", 0.74), 0.0),
                1.0,
            ),
            "entry_quality_mature_non_aligned_filter_dirvol_min_abs": max(
                _as_float("entry_quality_mature_non_aligned_filter_dirvol_min_abs", 0.12),
                0.0,
            ),
            "entry_quality_mature_sell_non_sell_filter_market_score_min": min(
                max(_as_float("entry_quality_mature_sell_non_sell_filter_market_score_min", 0.72), 0.0),
                1.0,
            ),
            "entry_quality_mature_sell_non_sell_filter_dirvol_min_abs": max(
                _as_float("entry_quality_mature_sell_non_sell_filter_dirvol_min_abs", 0.20),
                0.0,
            ),
            "entry_quality_impulse_routing_enabled": bool(
                raw_cfg.get("entry_quality_impulse_routing_enabled", True)
            ),
            "entry_quality_impulse_birth_market_score_min": min(
                max(_as_float("entry_quality_impulse_birth_market_score_min", 0.72), 0.0),
                1.0,
            ),
            "entry_quality_impulse_birth_split_score_min": min(
                max(_as_float("entry_quality_impulse_birth_split_score_min", 0.48), 0.0),
                1.0,
            ),
            "entry_quality_impulse_exhausted_retrace_score_min": min(
                max(_as_float("entry_quality_impulse_exhausted_retrace_score_min", 0.55), 0.0),
                1.0,
            ),
            "entry_quality_impulse_exhausted_skip_score_min": min(
                max(_as_float("entry_quality_impulse_exhausted_skip_score_min", 0.78), 0.0),
                1.0,
            ),
            "entry_quality_stretch_abs_pips_min": max(
                _as_float("entry_quality_stretch_abs_pips_min", 4.0),
                0.0,
            ),
            "entry_quality_stretch_vs_avg_range_min": max(
                _as_float("entry_quality_stretch_vs_avg_range_min", 0.90),
                0.0,
            ),
            "entry_quality_news_range_vs_avg_min": max(
                _as_float("entry_quality_news_range_vs_avg_min", 1.35),
                0.0,
            ),
            "entry_quality_roc_fast_column": str(
                raw_cfg.get("entry_quality_roc_fast_column", "ROC_3") or "ROC_3"
            ),
            "entry_quality_roc_slow_column": str(
                raw_cfg.get("entry_quality_roc_slow_column", "ROC_6") or "ROC_6"
            ),
            "entry_quality_range_vs_avg_column": str(
                raw_cfg.get("entry_quality_range_vs_avg_column", "RangeVsAvg6") or "RangeVsAvg6"
            ),
            "entry_quality_ema20_slope_column": str(
                raw_cfg.get("entry_quality_ema20_slope_column", "EMA20SlopePips") or "EMA20SlopePips"
            ),
            "entry_quality_vwap_slope_column": str(
                raw_cfg.get("entry_quality_vwap_slope_column", "SessionVWAPSlopePips") or "SessionVWAPSlopePips"
            ),
            "entry_quality_signed_distance_to_ema20_column": str(
                raw_cfg.get("entry_quality_signed_distance_to_ema20_column", "SignedDistanceToEMA20Pips")
                or "SignedDistanceToEMA20Pips"
            ),
            "entry_quality_signed_distance_to_vwap_column": str(
                raw_cfg.get("entry_quality_signed_distance_to_vwap_column", "SignedDistanceToVWAPPips")
                or "SignedDistanceToVWAPPips"
            ),
            "entry_quality_ema20_stretch_column": str(
                raw_cfg.get("entry_quality_ema20_stretch_column", "EMA20StretchVsAvgRange")
                or "EMA20StretchVsAvgRange"
            ),
            "entry_quality_vwap_stretch_column": str(
                raw_cfg.get("entry_quality_vwap_stretch_column", "VWAPStretchVsAvgRange")
                or "VWAPStretchVsAvgRange"
            ),
            "entry_quality_break_above_recent_high3_column": str(
                raw_cfg.get("entry_quality_break_above_recent_high3_column", "BreakAboveRecentHigh3")
                or "BreakAboveRecentHigh3"
            ),
            "entry_quality_break_below_recent_low3_column": str(
                raw_cfg.get("entry_quality_break_below_recent_low3_column", "BreakBelowRecentLow3")
                or "BreakBelowRecentLow3"
            ),
            "direct_filter_hold_retrace_fraction": min(
                max(_as_float("direct_filter_hold_retrace_fraction", 0.50), 0.0),
                1.0,
            ),
            "direct_filter_hold_sell_small_market_bypass_enabled": bool(
                raw_cfg.get("direct_filter_hold_sell_small_market_bypass_enabled", False)
            ),
            "direct_filter_hold_sell_small_market_confidence_min": max(
                _as_float("direct_filter_hold_sell_small_market_confidence_min", 0.88),
                0.0,
            ),
            "direct_filter_hold_sell_small_market_predicted_pips_min": max(
                _as_float("direct_filter_hold_sell_small_market_predicted_pips_min", 4.8),
                0.0,
            ),
            "direct_filter_hold_revalidate_on_activation": bool(
                raw_cfg.get("direct_filter_hold_revalidate_on_activation", True)
            ),
            "direct_filter_hold_cancel_if_primary_mismatch": bool(
                raw_cfg.get("direct_filter_hold_cancel_if_primary_mismatch", True)
            ),
            "direct_filter_hold_cancel_if_primary_hold_weak": bool(
                raw_cfg.get("direct_filter_hold_cancel_if_primary_hold_weak", True)
            ),
            "direct_filter_hold_activation_hold_confidence_min": max(
                _as_float("direct_filter_hold_activation_hold_confidence_min", 0.55),
                0.0,
            ),
            "direct_filter_hold_activation_predicted_pips_min": max(
                _as_float("direct_filter_hold_activation_predicted_pips_min", 3.5),
                0.0,
            ),
            "direct_filter_hold_hold_grace_enabled": bool(
                raw_cfg.get("direct_filter_hold_hold_grace_enabled", True)
            ),
            "direct_filter_hold_hold_grace_max_bars": max(
                _as_int("direct_filter_hold_hold_grace_max_bars", 4),
                0,
            ),
            "direct_filter_hold_hold_grace_predicted_pips_min": max(
                _as_float(
                    "direct_filter_hold_hold_grace_predicted_pips_min",
                    _as_float("direct_filter_hold_activation_predicted_pips_min", 3.5),
                ),
                0.0,
            ),
            "direct_filter_hold_cancel_on_soft_context_contradiction": bool(
                raw_cfg.get("direct_filter_hold_cancel_on_soft_context_contradiction", True)
            ),
            "direct_filter_hold_stop_buffer_pips": max(
                _as_float("direct_filter_hold_stop_buffer_pips", 0.30),
                0.0,
            ),
            "dynamic_stop_atr_fraction": max(
                _as_float("dynamic_stop_atr_fraction", 0.15),
                0.0,
            ),
            "dynamic_stop_min_pips": max(
                _as_float("dynamic_stop_min_pips", 3.8),
                0.0,
            ),
            "require_directional_volume_activation": bool(
                raw_cfg.get("require_directional_volume_activation", False)
            ),
            "directional_volume_column": str(
                raw_cfg.get("directional_volume_column", "DirectionalVolumeProxy_ZScore_20")
            ),
            "directional_volume_buy_min": _as_float("directional_volume_buy_min", 0.10),
            "directional_volume_sell_max": _as_float("directional_volume_sell_max", -0.10),
            "comment_prefix": str(raw_cfg.get("comment_prefix", "ES")),
        }

    def _evaluate_directional_volume_activation(
        self,
        *,
        signal: str,
        feature_row: pd.Series | dict[str, Any] | None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        settings = settings or self._get_entry_staging_settings()
        column_name = str(settings.get("directional_volume_column", "") or "").strip()
        details = {
            "enabled": bool(settings.get("require_directional_volume_activation", False)),
            "column": column_name,
            "value": self._coerce_feature_value(feature_row, column_name),
            "passed": True,
            "reason": "directional_volume_disabled",
        }
        signal_upper = str(signal or "").upper()
        if signal_upper not in {"BUY", "SELL"}:
            details["passed"] = False
            details["reason"] = "signal_hold"
            return details
        if not details["enabled"]:
            return details
        if not column_name:
            details["passed"] = False
            details["reason"] = "missing_directional_volume_column"
            return details

        value = details["value"]
        if value is None:
            details["passed"] = False
            details["reason"] = f"missing_{column_name}"
            return details

        if signal_upper == "BUY":
            threshold = float(settings.get("directional_volume_buy_min", 0.10) or 0.0)
            if value < threshold:
                details["passed"] = False
                details["reason"] = f"{column_name}_below_buy_threshold"
                return details
        else:
            threshold = float(settings.get("directional_volume_sell_max", -0.10) or 0.0)
            if value > threshold:
                details["passed"] = False
                details["reason"] = f"{column_name}_above_sell_threshold"
                return details

        details["reason"] = "directional_volume_passed"
        return details

    def _evaluate_strong_primary_filter_hold_stage(
        self,
        *,
        signal: str,
        feature_row: pd.Series | dict[str, Any] | None,
        filter_signal: str,
        primary_confidence: float | None,
        predicted_pips: float | None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        settings = settings or self._get_entry_staging_settings()
        signal_upper = str(signal or "").upper()
        details = {
            "eligible": False,
            "reason": "strong_primary_hold_stage_disabled",
            "profile": "strong_primary_hold",
            "retrace_fraction": float(settings.get("strong_primary_hold_retrace_fraction", 0.25)),
            "max_stage_bars": int(settings.get("strong_primary_hold_max_stage_bars", 1)),
            "breakout_partial_fraction": float(
                settings.get("strong_primary_hold_breakout_partial_fraction", 0.20)
            ),
            "roc_value": self._coerce_feature_value(
                feature_row,
                str(settings.get("adaptive_roc_column", "ROC_6")),
            ),
            "adx_value": self._coerce_feature_value(
                feature_row,
                str(settings.get("adaptive_adx_column", "ADX_14")),
            ),
        }
        if not bool(settings.get("allow_stage_on_strong_primary_filter_hold", True)):
            return details
        if signal_upper not in {"BUY", "SELL"}:
            details["reason"] = "signal_hold"
            return details
        if primary_confidence is None or pd.isna(primary_confidence):
            details["reason"] = "missing_primary_confidence"
            return details
        if float(primary_confidence) < float(settings.get("strong_primary_hold_confidence_min", 0.90)):
            details["reason"] = "primary_confidence_below_strong_primary_hold_threshold"
            return details
        if predicted_pips is None or pd.isna(predicted_pips):
            details["reason"] = "missing_predicted_pips"
            return details
        if abs(float(predicted_pips)) < float(settings.get("strong_primary_hold_predicted_pips_min", 6.0)):
            details["reason"] = "predicted_move_below_strong_primary_hold_threshold"
            return details
        roc_value = details["roc_value"]
        if roc_value is None:
            details["reason"] = "missing_strong_primary_hold_roc"
            return details
        if signal_upper == "BUY":
            if float(roc_value) <= 0:
                details["reason"] = "roc_not_aligned_for_buy"
                return details
        else:
            if float(roc_value) >= 0:
                details["reason"] = "roc_not_aligned_for_sell"
                return details
        if abs(float(roc_value)) < float(settings.get("strong_primary_hold_roc_abs_min", 0.00035)):
            details["reason"] = "roc_below_strong_primary_hold_threshold"
            return details
        adx_value = details["adx_value"]
        if adx_value is None:
            details["reason"] = "missing_strong_primary_hold_adx"
            return details
        if abs(float(adx_value)) < float(settings.get("strong_primary_hold_adx_min", 25.0)):
            details["reason"] = "adx_below_strong_primary_hold_threshold"
            return details
        details["eligible"] = True
        details["reason"] = "strong_primary_hold_stage_eligible"
        return details

    def _evaluate_medium_primary_filter_hold_stage(
        self,
        *,
        signal: str,
        feature_row: pd.Series | dict[str, Any] | None,
        primary_confidence: float | None,
        predicted_pips: float | None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        settings = settings or self._get_entry_staging_settings()
        signal_upper = str(signal or "").upper()
        roc_column = str(settings.get("medium_primary_hold_roc_column", "ROC_3") or "ROC_3")
        details = {
            "eligible": False,
            "reason": "medium_primary_hold_stage_disabled",
            "profile": "medium_primary_hold",
            "retrace_fraction": float(settings.get("medium_primary_hold_retrace_fraction", 0.25)),
            "max_stage_bars": int(settings.get("medium_primary_hold_max_stage_bars", 1)),
            "breakout_partial_fraction": float(
                settings.get("medium_primary_hold_breakout_partial_fraction", 0.15)
            ),
            "partial_fraction": float(settings.get("medium_primary_hold_partial_fraction", 0.10)),
            "roc_value": self._coerce_feature_value(feature_row, roc_column),
            "adx_value": self._coerce_feature_value(
                feature_row,
                str(settings.get("adaptive_adx_column", "ADX_14")),
            ),
        }
        if not bool(settings.get("allow_stage_on_medium_primary_filter_hold", True)):
            return details
        if signal_upper not in {"BUY", "SELL"}:
            details["reason"] = "signal_hold"
            return details
        if primary_confidence is None or pd.isna(primary_confidence):
            details["reason"] = "missing_primary_confidence"
            return details
        if float(primary_confidence) < float(settings.get("medium_primary_hold_confidence_min", 0.75)):
            details["reason"] = "primary_confidence_below_medium_primary_hold_threshold"
            return details
        if predicted_pips is None or pd.isna(predicted_pips):
            details["reason"] = "missing_predicted_pips"
            return details
        if abs(float(predicted_pips)) < float(settings.get("medium_primary_hold_predicted_pips_min", 4.0)):
            details["reason"] = "predicted_move_below_medium_primary_hold_threshold"
            return details
        roc_value = details["roc_value"]
        if roc_value is None:
            details["reason"] = "missing_medium_primary_hold_roc"
            return details
        if signal_upper == "BUY":
            if float(roc_value) <= 0:
                details["reason"] = "medium_primary_hold_roc_not_aligned_for_buy"
                return details
        else:
            if float(roc_value) >= 0:
                details["reason"] = "medium_primary_hold_roc_not_aligned_for_sell"
                return details
        if abs(float(roc_value)) < float(settings.get("medium_primary_hold_roc_abs_min", 0.00015)):
            details["reason"] = "medium_primary_hold_roc_below_threshold"
            return details
        adx_value = details["adx_value"]
        if adx_value is None:
            details["reason"] = "missing_medium_primary_hold_adx"
            return details
        if abs(float(adx_value)) < float(settings.get("medium_primary_hold_adx_min", 22.0)):
            details["reason"] = "medium_primary_hold_adx_below_threshold"
            return details
        details["eligible"] = True
        details["reason"] = "medium_primary_hold_stage_eligible"
        return details

    def _evaluate_early_structural_reversal_stage(
        self,
        *,
        signal: str,
        feature_row: pd.Series | dict[str, Any] | None,
        primary_confidence: float | None,
        predicted_pips: float | None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        settings = settings or self._get_entry_staging_settings()
        signal_upper = str(signal or "").upper()
        details = {
            "eligible": False,
            "reason": "early_structural_reversal_stage_disabled",
            "profile": "early_structural_reversal",
            "retrace_fraction": float(settings.get("early_reversal_retrace_fraction", 0.20)),
            "max_stage_bars": int(settings.get("early_reversal_max_stage_bars", 1)),
            "breakout_partial_fraction": float(
                settings.get("early_reversal_breakout_partial_fraction", 0.15)
            ),
            "short_roc_value": self._coerce_feature_value(
                feature_row,
                str(settings.get("early_reversal_short_momentum_column", "ROC_3")),
            ),
            "adx_value": self._coerce_feature_value(
                feature_row,
                str(settings.get("adaptive_adx_column", "ADX_14")),
            ),
            "directional_volume_value": self._coerce_feature_value(
                feature_row,
                str(settings.get("early_reversal_directional_volume_column", "DirectionalVolumeProxy_ZScore_20")),
            ),
            "structure_score3_value": self._coerce_feature_value(
                feature_row,
                str(settings.get("early_reversal_structure_score3_column", "StructureScore3")),
            ),
            "structure_score6_value": self._coerce_feature_value(
                feature_row,
                str(settings.get("early_reversal_structure_score6_column", "StructureScore6")),
            ),
            "break_prev_value": None,
            "break_recent3_value": None,
        }
        if not bool(settings.get("allow_stage_on_early_structural_reversal", True)):
            return details
        if signal_upper not in {"BUY", "SELL"}:
            details["reason"] = "signal_hold"
            return details
        if primary_confidence is None or pd.isna(primary_confidence):
            details["reason"] = "missing_primary_confidence"
            return details
        if float(primary_confidence) < float(settings.get("early_reversal_confidence_min", 0.78)):
            details["reason"] = "primary_confidence_below_early_reversal_threshold"
            return details
        if predicted_pips is None or pd.isna(predicted_pips):
            details["reason"] = "missing_predicted_pips"
            return details
        if abs(float(predicted_pips)) < float(settings.get("early_reversal_predicted_pips_min", 4.5)):
            details["reason"] = "predicted_move_below_early_reversal_threshold"
            return details

        relaxed_reversal_allowed = bool(settings.get("early_reversal_relaxed_enabled", True)) and (
            float(primary_confidence)
            >= float(settings.get("early_reversal_relaxed_confidence_min", 0.70))
            and abs(float(predicted_pips))
            >= float(settings.get("early_reversal_relaxed_predicted_pips_min", 3.6))
        )

        short_roc_value = details["short_roc_value"]
        if short_roc_value is None:
            details["reason"] = "missing_early_reversal_short_roc"
            return details
        if signal_upper == "BUY":
            if float(short_roc_value) <= 0:
                details["reason"] = "short_roc_not_aligned_for_buy"
                return details
        else:
            if float(short_roc_value) >= 0:
                details["reason"] = "short_roc_not_aligned_for_sell"
                return details
        short_roc_abs_min = float(settings.get("early_reversal_short_roc_abs_min", 0.00015))
        relaxed_short_roc_abs_min = float(
            settings.get("early_reversal_relaxed_short_roc_abs_min", 0.00001)
        )
        if abs(float(short_roc_value)) < short_roc_abs_min:
            if not relaxed_reversal_allowed or abs(float(short_roc_value)) < relaxed_short_roc_abs_min:
                details["reason"] = "short_roc_below_early_reversal_threshold"
                return details
            details["relaxed_mode"] = True
            details["relaxed_reason"] = "short_roc_threshold_relaxed"

        adx_value = details["adx_value"]
        if adx_value is None:
            details["reason"] = "missing_early_reversal_adx"
            return details
        if abs(float(adx_value)) < float(settings.get("early_reversal_adx_min", 22.0)):
            details["reason"] = "adx_below_early_reversal_threshold"
            return details
        relaxed_reversal_allowed = relaxed_reversal_allowed and (
            abs(float(adx_value)) >= float(settings.get("early_reversal_relaxed_adx_min", 30.0))
        )

        directional_volume_value = details["directional_volume_value"]
        if directional_volume_value is None:
            if not (
                relaxed_reversal_allowed
                and bool(settings.get("early_reversal_relax_directional_volume_if_adx_strong", True))
            ):
                details["reason"] = "missing_early_reversal_directional_volume"
                return details
            details["relaxed_mode"] = True
            details["relaxed_reason"] = "directional_volume_missing_but_relaxed"
        directional_volume_abs_min = float(
            settings.get("early_reversal_directional_volume_abs_min", 0.25)
        )
        if directional_volume_value is not None:
            directional_volume_aligned = True
            if signal_upper == "BUY":
                directional_volume_aligned = float(directional_volume_value) > 0
                not_aligned_reason = "directional_volume_not_aligned_for_buy"
            else:
                directional_volume_aligned = float(directional_volume_value) < 0
                not_aligned_reason = "directional_volume_not_aligned_for_sell"

            if not directional_volume_aligned:
                if not (
                    relaxed_reversal_allowed
                    and bool(settings.get("early_reversal_relax_directional_volume_if_adx_strong", True))
                ):
                    details["reason"] = not_aligned_reason
                    return details
                details["relaxed_mode"] = True
                details["relaxed_reason"] = "directional_volume_alignment_relaxed"
            elif abs(float(directional_volume_value)) < directional_volume_abs_min:
                if not (
                    relaxed_reversal_allowed
                    and bool(settings.get("early_reversal_relax_directional_volume_if_adx_strong", True))
                ):
                    details["reason"] = "directional_volume_below_early_reversal_threshold"
                    return details
                details["relaxed_mode"] = True
                details["relaxed_reason"] = "directional_volume_threshold_relaxed"

        structure_score3_value = details["structure_score3_value"]
        structure_score6_value = details["structure_score6_value"]
        structure_score_abs_min = float(settings.get("early_reversal_structure_score_abs_min", 0.30))
        break_prev_column = (
            str(settings.get("early_reversal_break_above_prev_high_column", "BreakAbovePrevHigh"))
            if signal_upper == "BUY"
            else str(settings.get("early_reversal_break_below_prev_low_column", "BreakBelowPrevLow"))
        )
        break_recent3_column = (
            str(settings.get("early_reversal_break_above_recent_high3_column", "BreakAboveRecentHigh3"))
            if signal_upper == "BUY"
            else str(settings.get("early_reversal_break_below_recent_low3_column", "BreakBelowRecentLow3"))
        )
        break_prev_value = self._coerce_feature_value(feature_row, break_prev_column)
        break_recent3_value = self._coerce_feature_value(feature_row, break_recent3_column)
        details["break_prev_value"] = break_prev_value
        details["break_recent3_value"] = break_recent3_value

        break_ok = False
        if break_prev_value is not None and float(break_prev_value) > 0:
            break_ok = True
        if break_recent3_value is not None and float(break_recent3_value) > 0:
            break_ok = True

        structure_ok = False
        if structure_score3_value is not None:
            if signal_upper == "BUY" and float(structure_score3_value) >= structure_score_abs_min:
                structure_ok = True
            if signal_upper == "SELL" and float(structure_score3_value) <= -structure_score_abs_min:
                structure_ok = True
        if not structure_ok and structure_score6_value is not None:
            if signal_upper == "BUY" and float(structure_score6_value) >= structure_score_abs_min / 2.0:
                structure_ok = True
            if signal_upper == "SELL" and float(structure_score6_value) <= -structure_score_abs_min / 2.0:
                structure_ok = True

        if not break_ok and not structure_ok:
            details["reason"] = "structure_not_aligned_for_early_reversal"
            return details

        details["eligible"] = True
        details["reason"] = (
            "early_structural_reversal_stage_eligible_relaxed"
            if details.get("relaxed_mode")
            else "early_structural_reversal_stage_eligible"
        )
        return details

    def _evaluate_filter_lead_structural_stage(
        self,
        *,
        filter_signal: str,
        feature_row: pd.Series | dict[str, Any] | None,
        primary_signal: str,
        primary_confidence: float | None,
        filter_confidence: float | None,
        predicted_pips: float | None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        settings = settings or self._get_entry_staging_settings()
        signal_upper = str(filter_signal or "").upper()
        details = {
            "eligible": False,
            "reason": "filter_lead_structural_stage_disabled",
            "profile": "filter_lead_structural",
            "retrace_fraction": float(settings.get("filter_lead_structural_retrace_fraction", 0.20)),
            "max_stage_bars": int(settings.get("filter_lead_structural_max_stage_bars", 1)),
            "breakout_partial_fraction": float(
                settings.get("filter_lead_structural_breakout_partial_fraction", 0.12)
            ),
            "short_roc_value": self._coerce_feature_value(
                feature_row,
                str(settings.get("filter_lead_structural_short_momentum_column", "ROC_3")),
            ),
            "adx_value": self._coerce_feature_value(
                feature_row,
                str(settings.get("adaptive_adx_column", "ADX_14")),
            ),
            "directional_volume_value": self._coerce_feature_value(
                feature_row,
                str(
                    settings.get(
                        "filter_lead_structural_directional_volume_column",
                        "DirectionalVolumeProxy_ZScore_20",
                    )
                ),
            ),
            "structure_score3_value": self._coerce_feature_value(
                feature_row,
                str(
                    settings.get(
                        "filter_lead_structural_structure_score3_column",
                        "StructureScore3",
                    )
                ),
            ),
            "structure_score6_value": self._coerce_feature_value(
                feature_row,
                str(
                    settings.get(
                        "filter_lead_structural_structure_score6_column",
                        "StructureScore6",
                    )
                ),
            ),
            "break_prev_value": None,
            "break_recent3_value": None,
            "stage_predicted_pips": np.nan,
        }
        if not bool(settings.get("allow_stage_on_filter_lead_structural", True)):
            return details
        if primary_signal != "HOLD":
            details["reason"] = "primary_not_hold"
            return details
        if signal_upper not in {"BUY", "SELL"}:
            details["reason"] = "filter_signal_hold"
            return details
        if filter_confidence is None or pd.isna(filter_confidence):
            details["reason"] = "missing_filter_confidence"
            return details
        if float(filter_confidence) < float(settings.get("filter_lead_structural_filter_confidence_min", 0.60)):
            details["reason"] = "filter_confidence_below_filter_lead_threshold"
            return details
        if primary_confidence is None or pd.isna(primary_confidence):
            details["reason"] = "missing_primary_confidence"
            return details
        if float(primary_confidence) < float(settings.get("filter_lead_structural_primary_hold_confidence_min", 0.55)):
            details["reason"] = "primary_hold_confidence_below_filter_lead_threshold"
            return details

        short_roc_value = details["short_roc_value"]
        if short_roc_value is None:
            details["reason"] = "missing_filter_lead_short_roc"
            return details
        if signal_upper == "BUY":
            if float(short_roc_value) <= 0:
                details["reason"] = "filter_lead_short_roc_not_aligned_for_buy"
                return details
        else:
            if float(short_roc_value) >= 0:
                details["reason"] = "filter_lead_short_roc_not_aligned_for_sell"
                return details
        if abs(float(short_roc_value)) < float(settings.get("filter_lead_structural_short_roc_abs_min", 0.00015)):
            details["reason"] = "filter_lead_short_roc_below_threshold"
            return details

        adx_value = details["adx_value"]
        if adx_value is None:
            details["reason"] = "missing_filter_lead_adx"
            return details
        if abs(float(adx_value)) < float(settings.get("filter_lead_structural_adx_min", 18.0)):
            details["reason"] = "filter_lead_adx_below_threshold"
            return details

        directional_volume_value = details["directional_volume_value"]
        if directional_volume_value is None:
            details["reason"] = "missing_filter_lead_directional_volume"
            return details
        directional_volume_abs_min = float(
            settings.get("filter_lead_structural_directional_volume_abs_min", 0.10)
        )
        if signal_upper == "BUY":
            if float(directional_volume_value) <= 0:
                details["reason"] = "filter_lead_directional_volume_not_aligned_for_buy"
                return details
        else:
            if float(directional_volume_value) >= 0:
                details["reason"] = "filter_lead_directional_volume_not_aligned_for_sell"
                return details
        if abs(float(directional_volume_value)) < directional_volume_abs_min:
            details["reason"] = "filter_lead_directional_volume_below_threshold"
            return details

        structure_score3_value = details["structure_score3_value"]
        structure_score6_value = details["structure_score6_value"]
        structure_score_abs_min = float(settings.get("filter_lead_structural_structure_score_abs_min", 0.20))
        break_prev_column = (
            str(settings.get("filter_lead_structural_break_above_prev_high_column", "BreakAbovePrevHigh"))
            if signal_upper == "BUY"
            else str(settings.get("filter_lead_structural_break_below_prev_low_column", "BreakBelowPrevLow"))
        )
        break_recent3_column = (
            str(settings.get("filter_lead_structural_break_above_recent_high3_column", "BreakAboveRecentHigh3"))
            if signal_upper == "BUY"
            else str(settings.get("filter_lead_structural_break_below_recent_low3_column", "BreakBelowRecentLow3"))
        )
        break_prev_value = self._coerce_feature_value(feature_row, break_prev_column)
        break_recent3_value = self._coerce_feature_value(feature_row, break_recent3_column)
        details["break_prev_value"] = break_prev_value
        details["break_recent3_value"] = break_recent3_value

        break_ok = False
        if break_prev_value is not None and float(break_prev_value) > 0:
            break_ok = True
        if break_recent3_value is not None and float(break_recent3_value) > 0:
            break_ok = True

        structure_ok = False
        if structure_score3_value is not None:
            if signal_upper == "BUY" and float(structure_score3_value) >= structure_score_abs_min:
                structure_ok = True
            if signal_upper == "SELL" and float(structure_score3_value) <= -structure_score_abs_min:
                structure_ok = True
        if not structure_ok and structure_score6_value is not None:
            if signal_upper == "BUY" and float(structure_score6_value) >= structure_score_abs_min / 2.0:
                structure_ok = True
            if signal_upper == "SELL" and float(structure_score6_value) <= -structure_score_abs_min / 2.0:
                structure_ok = True

        if not break_ok and not structure_ok:
            details["reason"] = "filter_lead_structure_not_aligned"
            return details

        predicted_floor = float(settings.get("filter_lead_structural_predicted_pips_floor", 3.6))
        raw_predicted = 0.0
        if predicted_pips is not None and pd.notna(predicted_pips):
            raw_predicted = abs(float(predicted_pips))
        stage_predicted_pips = max(raw_predicted, predicted_floor)
        details["stage_predicted_pips"] = (
            stage_predicted_pips if signal_upper == "BUY" else -stage_predicted_pips
        )
        details["eligible"] = True
        details["reason"] = "filter_lead_structural_stage_eligible"
        return details

    def _evaluate_filter_contradiction_stage(
        self,
        *,
        signal: str,
        feature_row: pd.Series | dict[str, Any] | None,
        filter_contradicted: bool,
        primary_confidence: float | None,
        predicted_pips: float | None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        settings = settings or self._get_entry_staging_settings()
        signal_upper = str(signal or "").upper()
        details = {
            "eligible": False,
            "reason": "contradiction_stage_disabled",
            "profile": "contradiction_aligned",
            "retrace_fraction": float(settings.get("contradiction_stage_retrace_fraction", 0.25)),
            "max_stage_bars": int(settings.get("contradiction_stage_max_stage_bars", 1)),
            "breakout_partial_fraction": float(
                settings.get("contradiction_stage_breakout_partial_fraction", 0.20)
            ),
            "roc_value": self._coerce_feature_value(
                feature_row,
                str(settings.get("adaptive_roc_column", "ROC_6")),
            ),
            "adx_value": self._coerce_feature_value(
                feature_row,
                str(settings.get("adaptive_adx_column", "ADX_14")),
            ),
            "directional_volume_value": self._coerce_feature_value(
                feature_row,
                str(settings.get("adaptive_directional_volume_column", "DirectionalVolumeProxy_ZScore_20")),
            ),
        }
        if not bool(settings.get("allow_stage_on_filter_contradiction_alignment", True)):
            return details
        if signal_upper not in {"BUY", "SELL"}:
            details["reason"] = "signal_hold"
            return details
        if not bool(filter_contradicted):
            details["reason"] = "filter_not_contradicted"
            return details
        if primary_confidence is None or pd.isna(primary_confidence):
            details["reason"] = "missing_primary_confidence"
            return details
        if float(primary_confidence) < float(settings.get("contradiction_stage_primary_confidence_min", 0.70)):
            details["reason"] = "primary_confidence_below_contradiction_stage_threshold"
            return details
        if predicted_pips is None or pd.isna(predicted_pips):
            details["reason"] = "missing_predicted_pips"
            return details
        if abs(float(predicted_pips)) < float(settings.get("contradiction_stage_predicted_pips_min", 3.5)):
            details["reason"] = "predicted_move_below_contradiction_stage_threshold"
            return details

        roc_value = details["roc_value"]
        if roc_value is None:
            details["reason"] = "missing_contradiction_stage_roc"
            return details
        if signal_upper == "BUY":
            if float(roc_value) <= 0:
                details["reason"] = "roc_not_aligned_for_buy"
                return details
        else:
            if float(roc_value) >= 0:
                details["reason"] = "roc_not_aligned_for_sell"
                return details
        if abs(float(roc_value)) < float(settings.get("contradiction_stage_roc_abs_min", 0.00060)):
            details["reason"] = "roc_below_contradiction_stage_threshold"
            return details

        dirvol_value = details["directional_volume_value"]
        if dirvol_value is None:
            details["reason"] = "missing_contradiction_stage_directional_volume"
            return details
        if signal_upper == "BUY":
            if float(dirvol_value) <= 0:
                details["reason"] = "directional_volume_not_aligned_for_buy"
                return details
        else:
            if float(dirvol_value) >= 0:
                details["reason"] = "directional_volume_not_aligned_for_sell"
                return details
        if abs(float(dirvol_value)) < float(
            settings.get("contradiction_stage_directional_volume_abs_min", 0.50)
        ):
            details["reason"] = "directional_volume_below_contradiction_stage_threshold"
            return details

        adx_value = details["adx_value"]
        if adx_value is None:
            details["reason"] = "missing_contradiction_stage_adx"
            return details
        if abs(float(adx_value)) < float(settings.get("contradiction_stage_adx_min", 8.0)):
            details["reason"] = "adx_below_contradiction_stage_threshold"
            return details

        details["eligible"] = True
        details["reason"] = "contradiction_stage_alignment_passed"
        return details

    def _get_trade_management_settings(self) -> dict[str, Any]:
        """Configuracion opcional de gestion de posiciones abierta en sync_trades."""
        raw_cfg = self.config.get("trade_management", {}) or {}
        raw_stages = raw_cfg.get("break_even_stages")
        progressive_grid_enabled = bool(raw_cfg.get("progressive_grid_enabled", False))
        default_stages = [
            {
                "name": "progress_50_close_30",
                "trigger_progress_to_tp": 0.50,
                "partial_close_fraction": 0.30,
                "move_sl_to_break_even": True,
            },
            {
                "name": "progress_70_close_50",
                "trigger_progress_to_tp": 0.70,
                "partial_close_fraction": 0.50,
                "move_sl_to_break_even": False,
            },
            {
                "name": "progress_85_close_80",
                "trigger_progress_to_tp": 0.85,
                "partial_close_fraction": 0.80,
                "move_sl_to_break_even": False,
            },
        ]
        if progressive_grid_enabled:
            try:
                grid_start = float(raw_cfg.get("progressive_grid_start_progress_to_tp", 0.30) or 0.30)
            except Exception:
                grid_start = 0.30
            try:
                grid_step = float(raw_cfg.get("progressive_grid_step_progress_to_tp", 0.05) or 0.05)
            except Exception:
                grid_step = 0.05
            try:
                grid_end = float(raw_cfg.get("progressive_grid_end_progress_to_tp", 0.90) or 0.90)
            except Exception:
                grid_end = 0.90
            try:
                grid_close_fraction = float(raw_cfg.get("progressive_grid_partial_close_fraction", 0.10) or 0.10)
            except Exception:
                grid_close_fraction = 0.10
            try:
                grid_break_even_trigger = float(
                    raw_cfg.get("progressive_grid_break_even_progress_to_tp", grid_start) or grid_start
                )
            except Exception:
                grid_break_even_trigger = grid_start

            grid_start = min(max(grid_start, 0.0), 1.0)
            grid_step = min(max(grid_step, 0.01), 0.50)
            grid_end = min(max(grid_end, grid_start), 1.0)
            grid_close_fraction = min(max(grid_close_fraction, 0.01), 1.0)
            grid_break_even_trigger = min(max(grid_break_even_trigger, grid_start), grid_end)

            generated_stages: list[dict[str, Any]] = []
            progress = grid_start
            break_even_assigned = False
            while progress <= grid_end + 1e-9:
                stage_pct = int(round(progress * 100.0))
                close_pct = int(round(grid_close_fraction * 100.0))
                move_be = False
                if not break_even_assigned and progress + 1e-12 >= grid_break_even_trigger:
                    move_be = True
                    break_even_assigned = True
                generated_stages.append(
                    {
                        "name": f"grid_progress_{stage_pct:02d}_close_{close_pct:02d}",
                        "trigger_progress_to_tp": round(progress, 4),
                        "partial_close_fraction": grid_close_fraction,
                        "move_sl_to_break_even": move_be,
                    }
                )
                progress += grid_step
            raw_stages = generated_stages

        if not isinstance(raw_stages, list) or not raw_stages:
            raw_stages = default_stages

        stages: list[dict[str, Any]] = []
        for idx, raw_stage in enumerate(raw_stages):
            if not isinstance(raw_stage, dict):
                continue
            try:
                trigger = float(raw_stage.get("trigger_progress_to_tp", 0.0) or 0.0)
                partial_fraction = float(raw_stage.get("partial_close_fraction", 0.0) or 0.0)
            except Exception:
                continue
            if trigger <= 0 or partial_fraction <= 0:
                continue
            stages.append(
                {
                    "name": str(raw_stage.get("name") or f"stage_{idx + 1}"),
                    "trigger_progress_to_tp": min(max(trigger, 0.0), 1.0),
                    "partial_close_fraction": min(max(partial_fraction, 0.0), 1.0),
                    "move_sl_to_break_even": bool(raw_stage.get("move_sl_to_break_even", idx == 0)),
                }
            )
        stages.sort(key=lambda item: item["trigger_progress_to_tp"])

        return {
            "enabled": bool(raw_cfg.get("enabled", False)) and bool(stages),
            "stages": stages,
            "move_sl_to": str(raw_cfg.get("move_sl_to", "breakeven")).strip().lower(),
            "breakeven_buffer_pips": float(raw_cfg.get("breakeven_buffer_pips", 0.0) or 0.0),
            "only_once_per_stage": bool(raw_cfg.get("only_once_per_stage", True)),
            "skip_if_no_tp": bool(raw_cfg.get("skip_if_no_tp", True)),
            "max_stage_actions_per_cycle": max(int(raw_cfg.get("max_stage_actions_per_cycle", 1) or 1), 1),
            "full_close_when_partial_below_min_enabled": bool(
                raw_cfg.get("full_close_when_partial_below_min_enabled", True)
            ),
            "full_close_when_partial_below_min_progress_to_tp": min(
                max(float(raw_cfg.get("full_close_when_partial_below_min_progress_to_tp", 0.95) or 0.95), 0.0),
                1.0,
            ),
            "comment_prefix": str(raw_cfg.get("comment_prefix", "TM")),
        }

    def _get_runtime_monitor_settings(self) -> dict[str, Any]:
        """Configuracion de monitor liviano para staged/trades recientes en runtime."""
        raw_cfg = self.config.get("runtime_monitor", {}) or {}

        def _as_float(key: str, default: float) -> float:
            try:
                return float(raw_cfg.get(key, default) or 0.0)
            except Exception:
                return float(default)

        def _as_int(key: str, default: int) -> int:
            try:
                return int(raw_cfg.get(key, default) or 0)
            except Exception:
                return int(default)

        return {
            "enabled": bool(raw_cfg.get("enabled", False)),
            "recent_trade_max_bars": max(_as_int("recent_trade_max_bars", 3), 1),
            "min_bars_before_no_progress": max(_as_int("min_bars_before_no_progress", 2), 1),
            "min_progress_to_keep": max(_as_float("min_progress_to_keep", 0.20), 0.0),
            "min_profit_progress_to_manage": max(_as_float("min_profit_progress_to_manage", 0.0), 0.0),
            "partial_profit_progress_min": max(_as_float("partial_profit_progress_min", 0.25), 0.0),
            "protect_on_reversal": bool(raw_cfg.get("protect_on_reversal", True)),
            "protect_on_lateralization": bool(raw_cfg.get("protect_on_lateralization", True)),
            "manage_on_opposite_signal": bool(raw_cfg.get("manage_on_opposite_signal", False)),
            "opposite_signal_min_primary_confidence": max(
                _as_float("opposite_signal_min_primary_confidence", 0.70),
                0.0,
            ),
            "opposite_signal_max_age_bars": max(_as_int("opposite_signal_max_age_bars", 2), 0),
            "opposite_signal_close_if_nonnegative": bool(
                raw_cfg.get("opposite_signal_close_if_nonnegative", True)
            ),
            "opposite_signal_arm_exit_until_break_even": bool(
                raw_cfg.get("opposite_signal_arm_exit_until_break_even", True)
            ),
            "opposite_signal_reduce_to_min_if_losing": bool(
                raw_cfg.get("opposite_signal_reduce_to_min_if_losing", False)
            ),
            "opposite_signal_prioritize_child_positions": bool(
                raw_cfg.get("opposite_signal_prioritize_child_positions", True)
            ),
            "close_full_on_weakness": bool(raw_cfg.get("close_full_on_weakness", False)),
            "full_close_profit_progress_min": max(_as_float("full_close_profit_progress_min", 0.60), 0.0),
            "full_close_on_reversal": bool(raw_cfg.get("full_close_on_reversal", True)),
            "full_close_on_lateralization": bool(raw_cfg.get("full_close_on_lateralization", True)),
            "full_close_on_no_followthrough": bool(raw_cfg.get("full_close_on_no_followthrough", True)),
            "shock_reversal_enabled": bool(raw_cfg.get("shock_reversal_enabled", True)),
            "shock_reversal_roc_abs_min": max(_as_float("shock_reversal_roc_abs_min", 0.00045), 0.0),
            "shock_reversal_range_vs_avg_min": max(_as_float("shock_reversal_range_vs_avg_min", 1.6), 0.0),
            "shock_reversal_dirvol_abs_min": max(_as_float("shock_reversal_dirvol_abs_min", 0.75), 0.0),
            "shock_reversal_close_location_abs_min": min(
                max(_as_float("shock_reversal_close_location_abs_min", 0.55), 0.0),
                1.0,
            ),
            "shock_reversal_progress_min": max(_as_float("shock_reversal_progress_min", 0.05), 0.0),
            "regime_column": str(raw_cfg.get("regime_column", "ADX_14") or "ADX_14"),
            "momentum_column": str(raw_cfg.get("momentum_column", "ROC_6") or "ROC_6"),
            "lateralization_short_momentum_column": str(
                raw_cfg.get("lateralization_short_momentum_column", "ROC_3") or "ROC_3"
            ),
            "lateralization_small_range_column": str(
                raw_cfg.get("lateralization_small_range_column", "RangeVsAvg6") or "RangeVsAvg6"
            ),
            "directional_volume_column": str(
                raw_cfg.get("directional_volume_column", "DirectionalVolumeProxy_ZScore_20")
                or "DirectionalVolumeProxy_ZScore_20"
            ),
            "close_location_column": str(raw_cfg.get("close_location_column", "CloseLocationValue") or "CloseLocationValue"),
            "reversal_roc_abs_min": max(_as_float("reversal_roc_abs_min", 0.00025), 0.0),
            "lateralization_horizon_bars": max(_as_int("lateralization_horizon_bars", 2), 1),
            "lateralization_adx_max": max(_as_float("lateralization_adx_max", 18.0), 0.0),
            "lateralization_compact_adx_max": max(_as_float("lateralization_compact_adx_max", 24.0), 0.0),
            "lateralization_roc_abs_max": max(_as_float("lateralization_roc_abs_max", 0.00010), 0.0),
            "lateralization_short_roc_abs_max": max(_as_float("lateralization_short_roc_abs_max", 0.00008), 0.0),
            "lateralization_dirvol_abs_max": max(_as_float("lateralization_dirvol_abs_max", 0.10), 0.0),
            "lateralization_compact_dirvol_abs_max": max(
                _as_float("lateralization_compact_dirvol_abs_max", 0.20),
                0.0,
            ),
            "lateralization_small_range_max": max(_as_float("lateralization_small_range_max", 0.85), 0.0),
            "lateralization_close_location_center_abs_max": min(
                max(_as_float("lateralization_close_location_center_abs_max", 0.35), 0.0),
                1.0,
            ),
            "partial_close_fraction_on_weakness": min(max(_as_float("partial_close_fraction_on_weakness", 0.50), 0.0), 1.0),
            "comment_prefix": str(raw_cfg.get("comment_prefix", "RTM") or "RTM"),
        }

    def _get_target_mode(self) -> str:
        backtest_cfg = self.config.get("backtest", {}) or {}
        target_mode = str(backtest_cfg.get("target_mode", "") or "").strip().lower()
        if target_mode:
            return target_mode
        target_col = str(backtest_cfg.get("target", "ReturnFwd_1") or "")
        if target_col.startswith("BarrierReturn") or target_col.startswith("BarrierDir"):
            return "barrier_event"
        return "return_regression"

    def _collect_same_side_cluster_state(
        self,
        *,
        lifecycle: pd.DataFrame | None,
        row: pd.Series | dict[str, Any],
    ) -> dict[str, Any]:
        """Resume la exposicion del mismo perfil, simbolo y lado en lifecycle."""
        defaults = {
            "open_positions_count": 0,
            "market_open_count": 0,
            "active_pending_orders_count": 0,
            "max_progress_to_tp": 0.0,
        }
        if lifecycle is None or lifecycle.empty:
            return defaults

        data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
        symbol = str(data.get("symbol") or "").strip()
        side = str(data.get("signal") or "").strip().upper()
        if not symbol or side not in {"BUY", "SELL"}:
            return defaults

        cluster = lifecycle.copy()
        if "symbol" in cluster.columns:
            cluster = cluster[cluster["symbol"].astype(str).str.strip() == symbol]
        if cluster.empty:
            return defaults
        if "signal" in cluster.columns:
            cluster = cluster[cluster["signal"].astype(str).str.upper() == side]
        if cluster.empty:
            return defaults

        profile_name = str(
            data.get("profile_name")
            or (self.config.get("strategy_profile", {}) or {}).get("name")
            or ""
        ).strip()
        release_id = str(data.get("release_id") or self.release_timestamp or "").strip()
        if profile_name and "profile_name" in cluster.columns:
            cluster = cluster[cluster["profile_name"].astype(str).str.strip() == profile_name]
        elif release_id and "release_id" in cluster.columns:
            cluster = cluster[cluster["release_id"].astype(str).str.strip() == release_id]
        if cluster.empty:
            return defaults

        status_upper = (
            cluster["status"].astype(str).str.upper()
            if "status" in cluster.columns
            else pd.Series("", index=cluster.index)
        )
        entry_leg = (
            cluster["entry_leg"].astype(str).str.strip().str.lower()
            if "entry_leg" in cluster.columns
            else pd.Series("", index=cluster.index)
        )
        pending_status = (
            cluster["pending_order_status"].astype(str).str.upper()
            if "pending_order_status" in cluster.columns
            else pd.Series("", index=cluster.index)
        )
        progress = (
            pd.to_numeric(cluster["management_progress_to_tp"], errors="coerce")
            if "management_progress_to_tp" in cluster.columns
            else pd.Series(np.nan, index=cluster.index)
        )

        open_mask = status_upper.isin(["OPEN", "PENDING_CONFIRMATION"])
        market_open_mask = open_mask & ~entry_leg.eq("pending_limit")
        active_pending_mask = pending_status.isin(["ACTIVE", "PLACED", "PENDING"]) & (
            entry_leg.eq("pending_limit") | status_upper.eq("PENDING_LIMIT")
        )
        max_progress = progress[open_mask].dropna()

        return {
            "open_positions_count": int(open_mask.sum()),
            "market_open_count": int(market_open_mask.sum()),
            "active_pending_orders_count": int(active_pending_mask.sum()),
            "max_progress_to_tp": float(max_progress.max()) if not max_progress.empty else 0.0,
        }

    def _should_skip_live_entry_for_cluster_guard(
        self,
        *,
        lifecycle: pd.DataFrame | None,
        row: pd.Series | dict[str, Any],
    ) -> tuple[bool, str]:
        """Evita seguir apilando entradas si el cluster ya va cargado y avanzado."""
        settings = self._get_entry_management_settings()
        if not settings.get("cluster_guard_enabled"):
            return False, ""

        cluster = self._collect_same_side_cluster_state(lifecycle=lifecycle, row=row)
        data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
        symbol = str(data.get("symbol") or "").strip()
        side = str(data.get("signal") or "").strip().upper()
        strong_continuation = self._is_strong_continuation_signal_for_cluster_guard(row)

        if (
            cluster["open_positions_count"] >= int(settings["cluster_guard_skip_new_entries_open_positions_min"])
            and cluster["max_progress_to_tp"] >= float(settings["cluster_guard_skip_new_entries_progress_min"])
            and not strong_continuation
        ):
            return (
                True,
                f"Cluster guard: {symbol} {side} ya tiene {cluster['open_positions_count']} posiciones abiertas "
                f"con progreso maximo {cluster['max_progress_to_tp'] * 100.0:.1f}%.",
            )

        if (
            cluster["active_pending_orders_count"] >= int(settings["cluster_guard_skip_new_entries_pending_orders_min"])
            and cluster["open_positions_count"] >= 1
            and not strong_continuation
        ):
            return (
                True,
                f"Cluster guard: {symbol} {side} ya tiene {cluster['active_pending_orders_count']} pending orders activas.",
            )

        return False, ""

    def _is_strong_continuation_signal_for_cluster_guard(self, row: pd.Series | dict[str, Any]) -> bool:
        """Detecta continuacion fuerte para permitir seguir tendencia sin apilar pending redundantes."""
        settings = self._get_entry_management_settings()
        if not settings.get("cluster_guard_allow_market_on_strong_continuation"):
            return False

        data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
        signal = str(data.get("signal") or "").strip().upper()
        primary_signal = str(data.get("primary_signal") or signal).strip().upper()
        if signal not in {"BUY", "SELL"} or primary_signal not in {"BUY", "SELL"}:
            return False
        if primary_signal != signal:
            return False

        primary_conf = pd.to_numeric(pd.Series([data.get("primary_confidence")]), errors="coerce").iloc[0]
        if pd.isna(primary_conf):
            primary_conf = pd.to_numeric(pd.Series([data.get("confidence")]), errors="coerce").iloc[0]
        predicted_pips = pd.to_numeric(pd.Series([data.get("predicted_move_pips")]), errors="coerce").iloc[0]
        if pd.isna(predicted_pips):
            predicted_pips = pd.to_numeric(pd.Series([data.get("predicted_pips")]), errors="coerce").iloc[0]
        if pd.isna(predicted_pips):
            predicted_pips = pd.to_numeric(pd.Series([data.get("pips")]), errors="coerce").iloc[0]

        if pd.isna(primary_conf) or pd.isna(predicted_pips):
            return False

        return (
            float(primary_conf) >= float(settings["cluster_guard_strong_continuation_primary_confidence_min"])
            and abs(float(predicted_pips)) >= float(settings["cluster_guard_strong_continuation_predicted_pips_min"])
        )

    def _should_force_market_only_for_cluster_trend(
        self,
        *,
        lifecycle: pd.DataFrame | None,
        row: pd.Series | dict[str, Any],
    ) -> tuple[str, str]:
        """Permite seguir tendencia fuerte, pero sin agregar otra pending redundante."""
        settings = self._get_entry_management_settings()
        if not (
            settings.get("cluster_guard_enabled")
            and settings.get("cluster_guard_disable_pending_on_strong_continuation")
        ):
            return "", ""

        cluster = self._collect_same_side_cluster_state(lifecycle=lifecycle, row=row)
        if cluster["open_positions_count"] < 1:
            return "", ""
        if cluster["active_pending_orders_count"] < 1:
            return "", ""
        if not self._is_strong_continuation_signal_for_cluster_guard(row):
            return "", ""

        data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
        symbol = str(data.get("symbol") or "").strip()
        side = str(data.get("signal") or "").strip().upper()
        entry_in_adverse_extreme = self._coerce_boolish(
            data.get("entry_context_market_entry_adverse_extreme"),
            False,
        )
        entry_rejection = self._coerce_boolish(
            data.get("entry_context_market_entry_rejection"),
            False,
        )
        if settings.get("cluster_guard_retrace_only_on_adverse_extreme") and entry_in_adverse_extreme:
            requires_rejection = settings.get("cluster_guard_retrace_only_requires_rejection", False)
            if not requires_rejection or entry_rejection:
                return (
                    "retrace_only",
                    f"Cluster trend continuation: {symbol} {side} sigue fuerte, "
                    "pero el market cae en un extremo estructural; se prioriza retrace-only.",
                )
        return (
            "market_only",
            f"Cluster trend continuation: {symbol} {side} sigue fuerte; se permite market-only sin otra pending.",
        )

    def _get_prediction_stack_settings(self) -> dict[str, Any]:
        """Normaliza la configuracion opcional de un stack hibrido."""
        raw_cfg = self.config.get("prediction_stack", {}) or {}

        def _normalized_name_set(values: Any) -> set[str]:
            if not isinstance(values, list):
                return set()
            return {
                str(value).strip().upper()
                for value in values
                if str(value).strip()
            }

        mode = str(raw_cfg.get("mode", "") or "").strip().lower()
        primary_models = _normalized_name_set(raw_cfg.get("primary_models"))
        filter_models = _normalized_name_set(raw_cfg.get("filter_models"))
        return {
            "mode": mode,
            "enabled": mode == "hybrid_primary_plus_filter",
            "primary_models": primary_models,
            "filter_models": filter_models,
            "require_alignment": bool(raw_cfg.get("require_alignment", True)),
            "filter_gate_mode": str(raw_cfg.get("filter_gate_mode", "full_signal") or "full_signal").strip().lower(),
            "support_probability_threshold": (
                None
                if raw_cfg.get("support_probability_threshold") is None
                else float(raw_cfg.get("support_probability_threshold"))
            ),
            "support_probability_margin": (
                None
                if raw_cfg.get("support_probability_margin") is None
                else float(raw_cfg.get("support_probability_margin"))
            ),
            "support_score_min": (
                None
                if raw_cfg.get("support_score_min") is None
                else float(raw_cfg.get("support_score_min"))
            ),
            "contradiction_margin": (
                None
                if raw_cfg.get("contradiction_margin") is None
                else float(raw_cfg.get("contradiction_margin"))
            ),
            "top_k_primary_for_bundle_eval": max(
                int(raw_cfg.get("top_k_primary_for_bundle_eval", 3) or 1),
                1,
            ),
            "top_k_filter_for_bundle_eval": max(
                int(raw_cfg.get("top_k_filter_for_bundle_eval", 3) or 1),
                1,
            ),
        }

    def _is_hybrid_mode(self) -> bool:
        return bool(self._get_prediction_stack_settings()["enabled"])

    def _get_barrier_settings(self) -> dict[str, Any]:
        backtest_cfg = self.config.get("backtest", {}) or {}
        trading_cfg = self.config.get("trading", {}) or {}
        barrier_pips = float(backtest_cfg.get("barrier_pips", trading_cfg.get("min_pips_signal", 3.0)) or 3.0)
        horizon_bars = int(backtest_cfg.get("barrier_horizon_bars", 3) or 3)
        pip_size = float(backtest_cfg.get("pip_size", self.config.get("data", {}).get("pip_size", 0.0001)) or 0.0001)
        suffix = f"_{int(barrier_pips)}p_{int(horizon_bars)}b"
        hybrid_filter_target = backtest_cfg.get("filter_target")
        configured_target = hybrid_filter_target if self._is_hybrid_mode() and hybrid_filter_target else backtest_cfg.get("target")
        return {
            "barrier_pips": barrier_pips,
            "horizon_bars": horizon_bars,
            "pip_size": pip_size,
            "price_col": str(backtest_cfg.get("barrier_price_col", "Close")),
            "high_col": str(backtest_cfg.get("barrier_high_col", "High")),
            "low_col": str(backtest_cfg.get("barrier_low_col", "Low")),
            "probability_threshold": float(trading_cfg.get("barrier_probability_threshold", trading_cfg.get("min_confidence", 0.60)) or 0.60),
            "probability_margin": float(trading_cfg.get("barrier_probability_margin", 0.05) or 0.05),
            "target_col": configured_target or f"BarrierReturn{suffix}",
            "dir_col": f"BarrierDir{suffix}",
            "return_col": f"BarrierReturn{suffix}",
            "move_pips_col": f"BarrierMovePips{suffix}",
            "bars_to_touch_col": f"BarrierBarsToTouch{suffix}",
            "ambiguous_col": f"BarrierAmbiguous{suffix}",
            "mfe_col": f"MFEPips{suffix}",
            "mae_col": f"MAEPips{suffix}",
        }

    def _get_model_stack_role(
        self,
        *,
        model_name: str | None,
        model_cfg: dict[str, Any] | None = None,
    ) -> str:
        """Clasifica un modelo como primary/filter/standalone para el stack hibrido."""
        name_upper = str(model_name or "").strip().upper()
        if model_cfg is not None:
            explicit_role = str(model_cfg.get("stack_role", "") or "").strip().lower()
            if explicit_role in {"primary", "filter"}:
                return explicit_role

        if not self._is_hybrid_mode():
            return "standalone"

        stack_cfg = self._get_prediction_stack_settings()
        if name_upper in stack_cfg["primary_models"]:
            return "primary"
        if name_upper in stack_cfg["filter_models"]:
            return "filter"
        if "CLASSIFIER" in name_upper:
            return "filter"
        return "primary"

    def _get_model_target_column(
        self,
        *,
        model_name: str | None,
        model_cfg: dict[str, Any] | None = None,
    ) -> str:
        """Devuelve el target correcto para un modelo segun el modo activo."""
        if self._is_hybrid_mode() and self._get_model_stack_role(model_name=model_name, model_cfg=model_cfg) == "filter":
            return str(self._get_barrier_settings()["target_col"])
        return str(self.config.get("backtest", {}).get("target", "ReturnFwd_1"))

    def _get_model_selection_role(
        self,
        *,
        model_name: str | None = None,
        model_cfg: dict[str, Any] | None = None,
        row: pd.Series | dict[str, Any] | None = None,
    ) -> str:
        """Define si un modelo compite por campeÃ³n o solo actÃºa como baseline."""
        raw_role = None
        if model_cfg is not None:
            raw_role = model_cfg.get("selection_role")
        if raw_role is None and row is not None:
            if isinstance(row, pd.Series):
                raw_role = row.get("selection_role")
                if model_name is None:
                    model_name = row.get("model")
            else:
                row_dict = dict(row)
                raw_role = row_dict.get("selection_role")
                if model_name is None:
                    model_name = row_dict.get("model")

        if raw_role is not None:
            role = str(raw_role).strip().lower()
            if role:
                return role

        normalized_name = str(model_name or "").strip().upper()
        if normalized_name in {"MOMENTUM", "MOMENTUMMODEL", "RANDOMWALK", "RANDOMWALKMODEL"}:
            return "baseline"
        return "candidate"

    def _is_model_selection_candidate(
        self,
        *,
        model_name: str | None = None,
        model_cfg: dict[str, Any] | None = None,
        row: pd.Series | dict[str, Any] | None = None,
    ) -> bool:
        return self._get_model_selection_role(
            model_name=model_name,
            model_cfg=model_cfg,
            row=row,
        ) != "baseline"

    def _is_future_leakage_column(self, column_name: str, target_col: str | None = None) -> bool:
        """Identifica columnas que contienen informaciÃ³n futura y no deben ser features."""
        col = str(column_name or "")
        if target_col and col == str(target_col):
            return True
        future_prefixes = (
            "ReturnFwd_",
            "ReturnFwd",
            "BarrierDir",
            "BarrierReturn",
            "BarrierMovePips",
            "BarrierBarsToTouch",
            "BarrierAmbiguous",
            "MFEPips",
            "MAEPips",
        )
        return any(col.startswith(prefix) for prefix in future_prefixes)

    def _get_model_feature_columns(self, df: pd.DataFrame, target_col: str) -> list[str]:
        """Columnas vÃ¡lidas para modelado, excluyendo target y variables futuras."""
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
        """ConfiguraciÃ³n opcional de filtros hÃ­bridos para autorizar una seÃ±al."""
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

    def _get_live_feature_snapshot(self, last_row: pd.Series | dict[str, Any] | None) -> dict[str, float]:
        """Extrae un subconjunto estable de features live para staging y auditoria."""
        feature_columns = [
            "ROC_3",
            "ROC_6",
            "ADX_14",
            "CloseLocationValue",
            "DirectionalVolumeProxy",
            "DirectionalVolumeProxy_ZScore_20",
        ]
        snapshot: dict[str, float] = {}
        for column_name in feature_columns:
            value = self._coerce_feature_value(last_row, column_name)
            snapshot[column_name] = float(value) if value is not None else np.nan
        return snapshot

    def _derive_close_location_value(
        self,
        *,
        candle_high: float | None,
        candle_low: float | None,
        candle_close: float | None,
    ) -> float | None:
        if any(pd.isna(value) for value in [candle_high, candle_low, candle_close]):
            return None
        candle_high = float(candle_high)
        candle_low = float(candle_low)
        candle_close = float(candle_close)
        price_range = candle_high - candle_low
        if price_range <= 0:
            return 0.0
        return float(((candle_close - candle_low) - (candle_high - candle_close)) / price_range)

    def _derive_price_location_in_candle(
        self,
        *,
        candle_high: float | None,
        candle_low: float | None,
        price: float | None,
    ) -> float | None:
        if any(pd.isna(value) for value in [candle_high, candle_low, price]):
            return None
        candle_high = float(candle_high)
        candle_low = float(candle_low)
        price = float(price)
        price_range = candle_high - candle_low
        if price_range <= 0:
            return None
        return float(max(min((price - candle_low) / price_range, 1.0), 0.0))

    def _resolve_feature_datetime_index(self, df: pd.DataFrame) -> pd.DatetimeIndex | None:
        if isinstance(df.index, pd.DatetimeIndex):
            return df.index

        for column in ("Date", "Datetime", "datetime", "timestamp", "Timestamp", "Time", "time"):
            if column not in df.columns:
                continue
            try:
                dt_candidate = pd.to_datetime(df[column], errors="coerce")
            except Exception:
                continue
            if isinstance(dt_candidate, pd.Series) and dt_candidate.notna().any():
                return pd.DatetimeIndex(dt_candidate)
        return None

    def _add_entry_execution_context_features(
        self,
        df: pd.DataFrame,
        *,
        pip_size: float,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if not all(column in df.columns for column in ["Close", "High", "Low"]):
            return df

        df = df.copy()
        pip_size = max(float(pip_size or 0.0001), 1e-12)
        close = pd.to_numeric(df["Close"], errors="coerce")
        high = pd.to_numeric(df["High"], errors="coerce")
        low = pd.to_numeric(df["Low"], errors="coerce")
        open_ = pd.to_numeric(df["Open"], errors="coerce") if "Open" in df.columns else close.shift(1)

        ema_20 = (
            pd.to_numeric(df["EMA_20"], errors="coerce")
            if "EMA_20" in df.columns
            else close.ewm(span=20, adjust=False).mean()
        )
        df["EMA_20"] = ema_20
        df["EMA20SlopePips"] = (ema_20 - ema_20.shift(1)) / pip_size

        bar_range_pips = (high - low).abs() / pip_size
        avg_range_6_pips = bar_range_pips.shift(1).rolling(6).mean()
        df["AvgRange6Pips"] = avg_range_6_pips

        signed_distance_to_ema20 = (close - ema_20) / pip_size
        df["SignedDistanceToEMA20Pips"] = signed_distance_to_ema20
        df["DistanceToEMA20Pips"] = signed_distance_to_ema20.abs()
        df["EMA20StretchVsAvgRange"] = signed_distance_to_ema20.abs() / avg_range_6_pips.replace(0, np.nan)

        volume_series = None
        for volume_column in ("TickVolume", "Volume"):
            if volume_column not in df.columns:
                continue
            candidate = pd.to_numeric(df[volume_column], errors="coerce")
            if candidate.notna().any() and candidate.abs().sum() > 0:
                volume_series = candidate.fillna(0.0)
                break
        if volume_series is None:
            volume_series = pd.Series(1.0, index=df.index, dtype=float)

        typical_price = (high + low + close) / 3.0
        dt_index = self._resolve_feature_datetime_index(df)
        if dt_index is not None and len(dt_index) == len(df):
            session_keys = pd.Series(dt_index.normalize(), index=df.index)
            valid_mask = session_keys.notna() & typical_price.notna() & volume_series.notna()
            session_vwap = pd.Series(np.nan, index=df.index, dtype=float)
            if valid_mask.any():
                pv = typical_price[valid_mask] * volume_series[valid_mask]
                grouped_pv = pv.groupby(session_keys[valid_mask]).cumsum()
                grouped_vol = volume_series[valid_mask].groupby(session_keys[valid_mask]).cumsum()
                session_vwap.loc[valid_mask] = grouped_pv / grouped_vol.replace(0, np.nan)
        else:
            cumulative_pv = (typical_price * volume_series).cumsum()
            cumulative_vol = volume_series.cumsum()
            session_vwap = cumulative_pv / cumulative_vol.replace(0, np.nan)

        df["SessionVWAP"] = session_vwap
        df["SessionVWAPSlopePips"] = (session_vwap - session_vwap.shift(1)) / pip_size

        signed_distance_to_vwap = (close - session_vwap) / pip_size
        df["SignedDistanceToVWAPPips"] = signed_distance_to_vwap
        df["DistanceToVWAPPips"] = signed_distance_to_vwap.abs()
        df["VWAPStretchVsAvgRange"] = signed_distance_to_vwap.abs() / avg_range_6_pips.replace(0, np.nan)

        if "CloseLocationValue" not in df.columns:
            price_range = (high - low).replace(0, np.nan)
            df["CloseLocationValue"] = ((2.0 * close) - high - low) / price_range

        body = close - open_
        df["SignalBodyPips"] = body / pip_size

        return df

    def _get_execution_confirmation_m1_snapshot(
        self,
        *,
        runtime_ctx: dict[str, Any] | None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        settings = settings or self._get_entry_staging_settings()
        enabled = bool(settings.get("execution_confirmation_m1_enabled", True))
        timeframe = str(settings.get("execution_confirmation_m1_timeframe", "M1") or "M1").strip().upper()
        snapshot = {
            "enabled": enabled,
            "available": False,
            "timeframe": timeframe,
            "as_of": pd.NA,
            "reason": "m1_execution_confirmation_disabled",
            "row": {},
        }
        if not enabled:
            return snapshot

        if isinstance(runtime_ctx, dict):
            cached_snapshot = runtime_ctx.get("_entry_execution_confirmation_m1_snapshot")
            if isinstance(cached_snapshot, dict):
                return cached_snapshot

        pip_size = max(
            float(
                (runtime_ctx or {}).get("pip_size")
                or (self.config.get("data", {}) or {}).get("pip_size", 0.0001)
                or 0.0001
            ),
            1e-12,
        )
        n_bars = max(int(settings.get("execution_confirmation_m1_n_bars", 180) or 180), 30)
        use_cache = bool(settings.get("execution_confirmation_m1_use_cache", False))
        cache_expiry_minutes = float(settings.get("execution_confirmation_m1_cache_expiry_minutes", 1.0) or 1.0)
        cache_expiry_hours = max(cache_expiry_minutes / 60.0, 0.0)
        symbol = str((self.config.get("data", {}) or {}).get("symbol", "EURUSD") or "EURUSD").strip()

        try:
            if self.data_loader is None:
                self.data_loader = DataLoader(mt5_config=self.config.get("mt5", {}))
            df_m1 = self.data_loader.load_data(
                symbol=symbol,
                timeframe=timeframe,
                n_bars=n_bars,
                use_cache=use_cache,
                cache_expiry_hours=cache_expiry_hours,
            )
        except Exception as exc:
            snapshot["reason"] = f"m1_load_failed_{type(exc).__name__}"
            if isinstance(runtime_ctx, dict):
                runtime_ctx["_entry_execution_confirmation_m1_snapshot"] = snapshot
            return snapshot

        if df_m1 is None or df_m1.empty:
            snapshot["reason"] = "m1_no_data"
            if isinstance(runtime_ctx, dict):
                runtime_ctx["_entry_execution_confirmation_m1_snapshot"] = snapshot
            return snapshot

        df_m1 = self._add_entry_execution_context_features(df_m1.copy(), pip_size=pip_size)
        close = pd.to_numeric(df_m1.get("Close"), errors="coerce")
        high = pd.to_numeric(df_m1.get("High"), errors="coerce")
        low = pd.to_numeric(df_m1.get("Low"), errors="coerce")
        open_ = pd.to_numeric(df_m1.get("Open"), errors="coerce") if "Open" in df_m1.columns else close.shift(1)

        volume_series = None
        for volume_column in ("TickVolume", "Volume"):
            if volume_column not in df_m1.columns:
                continue
            candidate = pd.to_numeric(df_m1[volume_column], errors="coerce")
            if candidate.notna().any():
                volume_series = candidate.fillna(0.0)
                break
        if volume_series is None:
            volume_series = pd.Series(1.0, index=df_m1.index, dtype=float)

        tick_volume_mean = volume_series.shift(1).rolling(20).mean()
        tick_volume_std = volume_series.shift(1).rolling(20).std(ddof=0).replace(0, np.nan)
        tick_volume_zscore = (volume_series - tick_volume_mean) / tick_volume_std

        roc1_pips = (close - close.shift(1)) / pip_size
        roc3_pips = (close - close.shift(3)) / pip_size
        price_range = (high - low).replace(0, np.nan)
        close_location = pd.to_numeric(df_m1.get("CloseLocationValue"), errors="coerce")
        if close_location.isna().all():
            close_location = ((2.0 * close) - high - low) / price_range
        body_direction = np.sign((close - open_).fillna(0.0))
        directional_volume_proxy = body_direction * tick_volume_zscore.fillna(0.0) * close_location.abs().fillna(0.0)

        breakout_lookback_bars = max(
            int(settings.get("execution_confirmation_m1_breakout_lookback_bars", 2) or 2),
            1,
        )
        breakout_above_recent = high >= high.shift(1).rolling(breakout_lookback_bars).max()
        breakout_below_recent = low <= low.shift(1).rolling(breakout_lookback_bars).min()

        df_m1["ExecutionM1ROC1Pips"] = roc1_pips
        df_m1["ExecutionM1ROC3Pips"] = roc3_pips
        df_m1["ExecutionM1TickVolumeZScore20"] = tick_volume_zscore
        df_m1["ExecutionM1DirectionalVolumeZScore20"] = directional_volume_proxy
        df_m1["ExecutionM1BreakAboveRecent"] = breakout_above_recent.fillna(False)
        df_m1["ExecutionM1BreakBelowRecent"] = breakout_below_recent.fillna(False)

        if df_m1.empty:
            snapshot["reason"] = "m1_features_empty"
            if isinstance(runtime_ctx, dict):
                runtime_ctx["_entry_execution_confirmation_m1_snapshot"] = snapshot
            return snapshot

        last_row = df_m1.iloc[-1].to_dict()
        dt_index = self._resolve_feature_datetime_index(df_m1)
        snapshot["available"] = True
        snapshot["reason"] = "m1_ready"
        snapshot["row"] = last_row
        if dt_index is not None and len(dt_index) == len(df_m1):
            last_ts = dt_index[-1]
            snapshot["as_of"] = last_ts.isoformat() if pd.notna(last_ts) else pd.NA

        if isinstance(runtime_ctx, dict):
            runtime_ctx["_entry_execution_confirmation_m1_snapshot"] = snapshot
        return snapshot

    def _record_entry_execution_confirmation_details(
        self,
        *,
        details: dict[str, Any] | None,
        df_rows: pd.DataFrame | None = None,
        row_idx: int | None = None,
        staged: pd.DataFrame | None = None,
        staged_idx: int | None = None,
    ) -> None:
        details = details or {}
        if df_rows is not None and row_idx is not None:
            df_rows.at[row_idx, "entry_execution_confirmation_tf"] = details.get("timeframe")
            df_rows.at[row_idx, "entry_execution_confirmation_score"] = details.get("score")
            df_rows.at[row_idx, "entry_execution_confirmation_passed"] = details.get("passed")
            df_rows.at[row_idx, "entry_execution_confirmation_reason"] = details.get("reason")
            df_rows.at[row_idx, "entry_execution_confirmation_hits"] = details.get("hits")
            df_rows.at[row_idx, "entry_execution_confirmation_total"] = details.get("total")
            df_rows.at[row_idx, "entry_execution_confirmation_as_of"] = details.get("as_of")

        if staged is not None and staged_idx is not None and staged_idx in staged.index:
            staged.at[staged_idx, "last_execution_confirmation_timeframe"] = details.get("timeframe")
            staged.at[staged_idx, "last_execution_confirmation_score"] = details.get("score")
            staged.at[staged_idx, "last_execution_confirmation_passed"] = details.get("passed")
            staged.at[staged_idx, "last_execution_confirmation_reason"] = details.get("reason")
            staged.at[staged_idx, "last_execution_confirmation_hits"] = details.get("hits")
            staged.at[staged_idx, "last_execution_confirmation_total"] = details.get("total")
            staged.at[staged_idx, "last_execution_confirmation_as_of"] = details.get("as_of")

    def _evaluate_entry_execution_confirmation(
        self,
        *,
        signal: str,
        runtime_ctx: dict[str, Any] | None,
        settings: dict[str, Any] | None = None,
        purpose: str = "stage_activation",
    ) -> dict[str, Any]:
        settings = settings or self._get_entry_staging_settings()
        signal_upper = str(signal or "").upper()
        details = {
            "enabled": bool(settings.get("execution_confirmation_m1_enabled", True)),
            "timeframe": str(settings.get("execution_confirmation_m1_timeframe", "M1") or "M1").strip().upper(),
            "purpose": purpose,
            "passed": True,
            "reason": "m1_execution_confirmation_disabled",
            "score": np.nan,
            "hits": 0,
            "total": 0,
            "as_of": pd.NA,
        }
        if signal_upper not in {"BUY", "SELL"}:
            details["passed"] = False
            details["reason"] = "m1_signal_hold"
            return details
        if not details["enabled"]:
            return details

        snapshot = self._get_execution_confirmation_m1_snapshot(
            runtime_ctx=runtime_ctx,
            settings=settings,
        )
        details["timeframe"] = snapshot.get("timeframe", details["timeframe"])
        details["as_of"] = snapshot.get("as_of", pd.NA)
        if not snapshot.get("available", False):
            fail_open = bool(settings.get("execution_confirmation_m1_fail_open_on_missing", True))
            details["passed"] = fail_open
            details["reason"] = (
                str(snapshot.get("reason") or "m1_unavailable")
                if fail_open
                else f"{snapshot.get('reason') or 'm1_unavailable'}_blocked"
            )
            return details

        row = snapshot.get("row") or {}
        roc1_pips = self._coerce_feature_value(row, "ExecutionM1ROC1Pips")
        roc3_pips = self._coerce_feature_value(row, "ExecutionM1ROC3Pips")
        dirvol = self._coerce_feature_value(row, "ExecutionM1DirectionalVolumeZScore20")
        tick_volume_zscore = self._coerce_feature_value(row, "ExecutionM1TickVolumeZScore20")
        close_location = self._coerce_feature_value(row, "CloseLocationValue")
        ema_slope = self._coerce_feature_value(row, "EMA20SlopePips")
        vwap_slope = self._coerce_feature_value(row, "SessionVWAPSlopePips")
        ema_stretch = self._coerce_feature_value(row, "EMA20StretchVsAvgRange")
        vwap_stretch = self._coerce_feature_value(row, "VWAPStretchVsAvgRange")
        close_price = self._coerce_feature_value(row, "Close")
        high_price = self._coerce_feature_value(row, "High")
        low_price = self._coerce_feature_value(row, "Low")
        breakout_hit = bool(
            row.get("ExecutionM1BreakAboveRecent", False)
            if signal_upper == "BUY"
            else row.get("ExecutionM1BreakBelowRecent", False)
        )

        opposite_wick_ratio = np.nan
        if (
            pd.notna(close_price)
            and pd.notna(high_price)
            and pd.notna(low_price)
            and float(high_price) > float(low_price)
        ):
            candle_range = float(high_price) - float(low_price)
            if signal_upper == "BUY":
                opposite_wick_ratio = max(float(high_price) - float(close_price), 0.0) / candle_range
            else:
                opposite_wick_ratio = max(float(close_price) - float(low_price), 0.0) / candle_range

        min_roc1_pips = float(settings.get("execution_confirmation_m1_min_roc1_pips", 0.10) or 0.0)
        min_roc3_pips = float(settings.get("execution_confirmation_m1_min_roc3_pips", 0.25) or 0.0)
        min_dirvol_abs = float(settings.get("execution_confirmation_m1_min_directional_volume_abs", 0.0) or 0.0)
        min_tick_volume_zscore = float(settings.get("execution_confirmation_m1_min_tick_volume_zscore", -0.10) or -0.10)
        min_close_location_abs = float(settings.get("execution_confirmation_m1_min_close_location_abs", 0.20) or 0.20)
        max_stretch_vs_avg_range = float(settings.get("execution_confirmation_m1_max_stretch_vs_avg_range", 1.40) or 1.40)
        max_opposite_wick_ratio = float(settings.get("execution_confirmation_m1_max_opposite_wick_ratio", 0.60) or 0.60)

        slope_alignment_hit = (
            pd.notna(ema_slope)
            and pd.notna(vwap_slope)
            and (
                (signal_upper == "BUY" and float(ema_slope) >= 0 and float(vwap_slope) >= 0)
                or (signal_upper == "SELL" and float(ema_slope) <= 0 and float(vwap_slope) <= 0)
            )
        )
        roc1_hit = pd.notna(roc1_pips) and (
            float(roc1_pips) >= min_roc1_pips if signal_upper == "BUY" else float(roc1_pips) <= -min_roc1_pips
        )
        roc3_hit = pd.notna(roc3_pips) and (
            float(roc3_pips) >= min_roc3_pips if signal_upper == "BUY" else float(roc3_pips) <= -min_roc3_pips
        )
        dirvol_hit = pd.notna(dirvol) and (
            float(dirvol) >= min_dirvol_abs if signal_upper == "BUY" else float(dirvol) <= -min_dirvol_abs
        )
        close_location_hit = pd.notna(close_location) and (
            float(close_location) >= min_close_location_abs
            if signal_upper == "BUY"
            else float(close_location) <= -min_close_location_abs
        )
        tick_volume_hit = (
            pd.isna(tick_volume_zscore) or float(tick_volume_zscore) >= min_tick_volume_zscore
        )
        stretch_value = max(
            abs(float(ema_stretch)) if pd.notna(ema_stretch) else 0.0,
            abs(float(vwap_stretch)) if pd.notna(vwap_stretch) else 0.0,
        )
        stretch_ok = stretch_value <= max_stretch_vs_avg_range
        wick_ok = pd.isna(opposite_wick_ratio) or float(opposite_wick_ratio) <= max_opposite_wick_ratio

        checks = {
            "breakout": breakout_hit,
            "roc1": roc1_hit,
            "roc3": roc3_hit,
            "directional_volume": dirvol_hit,
            "close_location": close_location_hit,
            "slope_alignment": slope_alignment_hit,
            "tick_volume": tick_volume_hit,
        }
        hits = sum(1 for passed in checks.values() if passed)
        total = len(checks)
        score = float(hits / total) if total > 0 else float("nan")
        details["hits"] = hits
        details["total"] = total
        details["score"] = score

        min_hits = max(int(settings.get("execution_confirmation_m1_min_alignment_hits", 3) or 3), 1)
        min_score = float(settings.get("execution_confirmation_m1_min_score", 0.50) or 0.50)
        strong_alignment_override = False
        if (
            bool(settings.get("execution_confirmation_m1_strong_alignment_override_enabled", True))
            and not stretch_ok
            and wick_ok
        ):
            override_min_hits = max(
                int(settings.get("execution_confirmation_m1_strong_alignment_min_hits", 5) or 5),
                1,
            )
            override_min_score = float(
                settings.get("execution_confirmation_m1_strong_alignment_min_score", 0.70) or 0.70
            )
            override_max_stretch = float(
                settings.get(
                    "execution_confirmation_m1_strong_alignment_max_stretch_vs_avg_range",
                    1.85,
                )
                or 1.85
            )
            require_breakout_override = bool(
                settings.get("execution_confirmation_m1_strong_alignment_require_breakout", True)
            )
            strong_alignment_override = (
                hits >= override_min_hits
                and score >= override_min_score
                and stretch_value <= override_max_stretch
                and (breakout_hit if require_breakout_override else True)
                and (roc1_hit or roc3_hit)
                and dirvol_hit
            )
        passed = (
            hits >= min_hits and score >= min_score and stretch_ok and wick_ok
        ) or strong_alignment_override
        details["passed"] = passed
        if passed:
            details["reason"] = (
                "m1_execution_confirmed_breakout_override"
                if strong_alignment_override
                else "m1_execution_confirmed"
            )
        elif not stretch_ok:
            details["reason"] = "m1_stretch_too_high"
        elif not wick_ok:
            details["reason"] = "m1_opposite_wick_rejection"
        elif hits < min_hits:
            details["reason"] = "m1_alignment_insufficient"
        else:
            details["reason"] = "m1_score_below_threshold"
        return details

    def _is_immediate_entry_price_in_adverse_extreme(
        self,
        *,
        side: str,
        price: float | None,
        candle_high: float | None,
        candle_low: float | None,
        pip_size: float,
        settings: dict[str, Any] | None = None,
    ) -> tuple[bool, float | None, float | None]:
        settings = settings or self._get_entry_staging_settings()
        if not bool(settings.get("direct_confirmed_immediate_extreme_entry_guard_enabled", True)):
            return False, None, None
        if pip_size <= 0:
            return False, None, None
        location = self._derive_price_location_in_candle(
            candle_high=candle_high,
            candle_low=candle_low,
            price=price,
        )
        if location is None:
            return False, None, None
        try:
            candle_range_pips = abs(float(candle_high) - float(candle_low)) / float(pip_size)
        except Exception:
            candle_range_pips = None
        if candle_range_pips is None or candle_range_pips < float(
            settings.get("direct_confirmed_immediate_extreme_entry_min_range_pips", 2.0)
        ):
            return False, location, candle_range_pips
        adverse_fraction = float(
            settings.get("direct_confirmed_immediate_extreme_entry_range_fraction", 0.20)
        )
        side_upper = str(side or "").upper()
        if side_upper == "BUY":
            return bool(location >= 1.0 - adverse_fraction), location, candle_range_pips
        if side_upper == "SELL":
            return bool(location <= adverse_fraction), location, candle_range_pips
        return False, location, candle_range_pips

    def _evaluate_entry_context_guard(
        self,
        *,
        signal: str,
        feature_row: pd.Series | dict[str, Any] | None,
        candle_open: float | None,
        candle_high: float | None,
        candle_low: float | None,
        candle_close: float | None,
        market_entry_price: float | None = None,
        pip_size: float | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        settings = settings or self._get_entry_staging_settings()
        signal_upper = str(signal or "").upper()
        details = {
            "enabled": bool(settings.get("context_guard_enabled", True)),
            "soft_contradicted": False,
            "hard_contradicted": False,
            "reason": "entry_context_guard_disabled",
            "directional_volume_column": str(
                settings.get("context_guard_directional_volume_column", "DirectionalVolumeProxy_ZScore_20")
                or "DirectionalVolumeProxy_ZScore_20"
            ),
            "close_location_column": str(
                settings.get("context_guard_close_location_column", "CloseLocationValue") or "CloseLocationValue"
            ),
            "directional_volume_value": None,
            "close_location_value": None,
            "body_against_signal": False,
            "market_entry_location_value": None,
            "market_entry_range_pips": None,
            "market_entry_adverse_extreme": False,
            "market_entry_rejection": False,
            "market_entry_retrace_only": False,
            "quality_score": np.nan,
            "quality_decision": "unavailable",
            "quality_alignment_hits": 0,
            "quality_alignment_total": 0,
            "quality_force_retrace_only": False,
            "quality_force_split_only": False,
            "quality_skip": False,
            "quality_signed_distance_to_ema20_pips": None,
            "quality_signed_distance_to_vwap_pips": None,
            "quality_ema20_stretch_vs_avg_range": None,
            "quality_vwap_stretch_vs_avg_range": None,
            "quality_range_vs_avg_value": None,
            "quality_stretched_from_ema20": False,
            "quality_stretched_from_vwap": False,
            "quality_stretched_entry": False,
            "impulse_birth_score": np.nan,
            "impulse_exhaustion_score": np.nan,
            "impulse_state": "unavailable",
        }
        if signal_upper not in {"BUY", "SELL"}:
            details["reason"] = "signal_hold"
            return details
        if not details["enabled"]:
            return details

        close_location_value = self._coerce_feature_value(feature_row, details["close_location_column"])
        if close_location_value is None:
            close_location_value = self._derive_close_location_value(
                candle_high=candle_high,
                candle_low=candle_low,
                candle_close=candle_close,
            )
        details["close_location_value"] = close_location_value
        details["directional_volume_value"] = self._coerce_feature_value(
            feature_row,
            details["directional_volume_column"],
        )

        if close_location_value is None or details["directional_volume_value"] is None:
            details["reason"] = "missing_context_features"
            details["quality_score"] = 0.70
            details["quality_decision"] = "market"
            return details

        dirvol_value = float(details["directional_volume_value"])
        close_location_value = float(close_location_value)
        candle_body = None
        try:
            if pd.notna(candle_open) and pd.notna(candle_close):
                candle_body = float(candle_close) - float(candle_open)
        except Exception:
            candle_body = None
        body_against_signal = False
        if candle_body is not None and bool(settings.get("context_guard_use_body_direction", True)):
            body_against_signal = (
                (signal_upper == "BUY" and candle_body < 0)
                or (signal_upper == "SELL" and candle_body > 0)
            )
        details["body_against_signal"] = body_against_signal

        soft_clv = float(settings.get("context_guard_soft_close_location_abs_min", 0.45) or 0.45)
        hard_clv = float(settings.get("context_guard_hard_close_location_abs_min", 0.80) or 0.80)
        soft_dirvol = float(settings.get("context_guard_soft_directional_volume_abs_min", 0.35) or 0.35)
        hard_dirvol = float(settings.get("context_guard_hard_directional_volume_abs_min", 0.90) or 0.90)

        if signal_upper == "BUY":
            soft_clv_hit = close_location_value <= -soft_clv
            hard_clv_hit = close_location_value <= -hard_clv
            soft_dirvol_hit = dirvol_value <= -soft_dirvol
            hard_dirvol_hit = dirvol_value <= -hard_dirvol
        else:
            soft_clv_hit = close_location_value >= soft_clv
            hard_clv_hit = close_location_value >= hard_clv
            soft_dirvol_hit = dirvol_value >= soft_dirvol
            hard_dirvol_hit = dirvol_value >= hard_dirvol

        details["soft_contradicted"] = bool(soft_clv_hit and (soft_dirvol_hit or body_against_signal))
        details["hard_contradicted"] = bool(hard_clv_hit and hard_dirvol_hit)
        pip_size = float(
            pip_size
            or (self.config.get("data", {}) or {}).get("pip_size", 0.0001)
            or 0.0001
        )
        market_entry_price = pd.to_numeric(pd.Series([market_entry_price]), errors="coerce").iloc[0]
        if (
            bool(settings.get("context_guard_soft_disable_market_on_extreme_rejection", True))
            and pd.notna(market_entry_price)
            and pip_size > 0
        ):
            entry_location = self._derive_price_location_in_candle(
                candle_high=candle_high,
                candle_low=candle_low,
                price=float(market_entry_price),
            )
            details["market_entry_location_value"] = entry_location
            if entry_location is not None and not any(pd.isna(value) for value in [candle_high, candle_low]):
                try:
                    candle_range_pips = abs(float(candle_high) - float(candle_low)) / pip_size
                except Exception:
                    candle_range_pips = None
                details["market_entry_range_pips"] = candle_range_pips
                if (
                    candle_range_pips is not None
                    and candle_range_pips
                    >= float(settings.get("context_guard_market_entry_min_range_pips", 2.0) or 2.0)
                ):
                    adverse_fraction = float(
                        settings.get("context_guard_market_entry_extreme_range_fraction", 0.18) or 0.18
                    )
                    if signal_upper == "BUY":
                        details["market_entry_adverse_extreme"] = bool(entry_location >= 1.0 - adverse_fraction)
                    else:
                        details["market_entry_adverse_extreme"] = bool(entry_location <= adverse_fraction)
                    rejection_clv = float(
                        settings.get("context_guard_market_entry_rejection_close_location_abs_min", 0.0) or 0.0
                    )
                    if signal_upper == "BUY":
                        details["market_entry_rejection"] = bool(
                            body_against_signal and close_location_value <= -rejection_clv
                        )
                    else:
                        details["market_entry_rejection"] = bool(
                            body_against_signal and close_location_value >= rejection_clv
                        )
                    details["market_entry_retrace_only"] = bool(
                        details["market_entry_adverse_extreme"] and details["market_entry_rejection"]
                    )
        if bool(settings.get("entry_quality_enabled", True)):
            roc_fast = self._coerce_feature_value(
                feature_row,
                str(settings.get("entry_quality_roc_fast_column", "ROC_3") or "ROC_3"),
            )
            roc_slow = self._coerce_feature_value(
                feature_row,
                str(settings.get("entry_quality_roc_slow_column", "ROC_6") or "ROC_6"),
            )
            ema20_slope = self._coerce_feature_value(
                feature_row,
                str(settings.get("entry_quality_ema20_slope_column", "EMA20SlopePips") or "EMA20SlopePips"),
            )
            vwap_slope = self._coerce_feature_value(
                feature_row,
                str(settings.get("entry_quality_vwap_slope_column", "SessionVWAPSlopePips") or "SessionVWAPSlopePips"),
            )
            breakout_buy = self._coerce_feature_value(
                feature_row,
                str(
                    settings.get("entry_quality_break_above_recent_high3_column", "BreakAboveRecentHigh3")
                    or "BreakAboveRecentHigh3"
                ),
            )
            breakout_sell = self._coerce_feature_value(
                feature_row,
                str(
                    settings.get("entry_quality_break_below_recent_low3_column", "BreakBelowRecentLow3")
                    or "BreakBelowRecentLow3"
                ),
            )
            range_vs_avg = self._coerce_feature_value(
                feature_row,
                str(settings.get("entry_quality_range_vs_avg_column", "RangeVsAvg6") or "RangeVsAvg6"),
            )
            signed_distance_to_ema20 = self._coerce_feature_value(
                feature_row,
                str(
                    settings.get("entry_quality_signed_distance_to_ema20_column", "SignedDistanceToEMA20Pips")
                    or "SignedDistanceToEMA20Pips"
                ),
            )
            signed_distance_to_vwap = self._coerce_feature_value(
                feature_row,
                str(
                    settings.get("entry_quality_signed_distance_to_vwap_column", "SignedDistanceToVWAPPips")
                    or "SignedDistanceToVWAPPips"
                ),
            )
            ema20_stretch = self._coerce_feature_value(
                feature_row,
                str(settings.get("entry_quality_ema20_stretch_column", "EMA20StretchVsAvgRange") or "EMA20StretchVsAvgRange"),
            )
            vwap_stretch = self._coerce_feature_value(
                feature_row,
                str(settings.get("entry_quality_vwap_stretch_column", "VWAPStretchVsAvgRange") or "VWAPStretchVsAvgRange"),
            )

            details["quality_signed_distance_to_ema20_pips"] = signed_distance_to_ema20
            details["quality_signed_distance_to_vwap_pips"] = signed_distance_to_vwap
            details["quality_ema20_stretch_vs_avg_range"] = ema20_stretch
            details["quality_vwap_stretch_vs_avg_range"] = vwap_stretch
            details["quality_range_vs_avg_value"] = range_vs_avg

            def _directional_alignment(value: Any) -> bool | None:
                if value is None or pd.isna(value):
                    return None
                value = float(value)
                return value > 0 if signal_upper == "BUY" else value < 0

            alignment_hits = 0
            alignment_total = 0
            for candidate in (roc_fast, roc_slow, ema20_slope, vwap_slope):
                aligned = _directional_alignment(candidate)
                if aligned is None:
                    continue
                alignment_total += 1
                alignment_hits += int(aligned)

            breakout_aligned = None
            if signal_upper == "BUY" and breakout_buy is not None and not pd.isna(breakout_buy):
                breakout_aligned = float(breakout_buy) > 0.5
            elif signal_upper == "SELL" and breakout_sell is not None and not pd.isna(breakout_sell):
                breakout_aligned = float(breakout_sell) > 0.5
            if breakout_aligned is not None:
                alignment_total += 1
                alignment_hits += int(breakout_aligned)

            details["quality_alignment_hits"] = alignment_hits
            details["quality_alignment_total"] = alignment_total

            stretch_abs_pips_min = float(settings.get("entry_quality_stretch_abs_pips_min", 4.0) or 4.0)
            stretch_vs_avg_min = float(settings.get("entry_quality_stretch_vs_avg_range_min", 0.90) or 0.90)
            news_range_vs_avg_min = float(settings.get("entry_quality_news_range_vs_avg_min", 1.35) or 1.35)

            def _adverse_stretch(distance_pips: Any, stretch_value: Any) -> bool:
                if distance_pips is None or stretch_value is None or pd.isna(distance_pips) or pd.isna(stretch_value):
                    return False
                distance_pips = float(distance_pips)
                stretch_value = float(stretch_value)
                if stretch_value < stretch_vs_avg_min:
                    return False
                if signal_upper == "BUY":
                    return distance_pips >= stretch_abs_pips_min
                return distance_pips <= -stretch_abs_pips_min

            details["quality_stretched_from_ema20"] = _adverse_stretch(
                signed_distance_to_ema20,
                ema20_stretch,
            )
            details["quality_stretched_from_vwap"] = _adverse_stretch(
                signed_distance_to_vwap,
                vwap_stretch,
            )
            details["quality_stretched_entry"] = bool(
                details["quality_stretched_from_ema20"] or details["quality_stretched_from_vwap"]
            )

            score = 0.55
            score += 0.06 if not body_against_signal else -0.06
            score += 0.08 if not details["soft_contradicted"] else -0.10
            score -= 0.25 if details["hard_contradicted"] else 0.0

            if signal_upper == "BUY":
                if close_location_value >= 0.15:
                    score += 0.06
                elif close_location_value <= -0.15:
                    score -= 0.06
                if dirvol_value >= 0.10:
                    score += 0.06
                elif dirvol_value <= -0.10:
                    score -= 0.06
            else:
                if close_location_value <= -0.15:
                    score += 0.06
                elif close_location_value >= 0.15:
                    score -= 0.06
                if dirvol_value <= -0.10:
                    score += 0.06
                elif dirvol_value >= 0.10:
                    score -= 0.06

            score += min(alignment_hits * 0.06, 0.24)
            if alignment_total > 0 and (alignment_hits / alignment_total) < 0.40:
                score -= 0.08
            if details["market_entry_adverse_extreme"]:
                score -= 0.16
            if details["market_entry_rejection"]:
                score -= 0.14
            if details["quality_stretched_entry"]:
                score -= 0.18
                if range_vs_avg is not None and not pd.isna(range_vs_avg) and float(range_vs_avg) >= news_range_vs_avg_min:
                    score -= 0.08

            score = max(min(score, 1.0), 0.0)
            details["quality_score"] = score

            market_min = float(settings.get("entry_quality_min_score_for_market", 0.62) or 0.62)
            retrace_min = float(settings.get("entry_quality_min_score_for_retrace", 0.38) or 0.38)
            fast_aligned = _directional_alignment(roc_fast)
            slow_aligned = _directional_alignment(roc_slow)
            ema20_aligned = _directional_alignment(ema20_slope)
            vwap_aligned = _directional_alignment(vwap_slope)
            if signal_upper == "BUY":
                clv_supports_signal = close_location_value >= 0.15
                clv_shows_exhaustion = close_location_value <= 0.05
                dirvol_supports_signal = dirvol_value >= 0.10
            else:
                clv_supports_signal = close_location_value <= -0.15
                clv_shows_exhaustion = close_location_value >= -0.05
                dirvol_supports_signal = dirvol_value <= -0.10

            birth_score = 0.0
            birth_score += 0.18 if breakout_aligned else 0.0
            birth_score += 0.12 if fast_aligned is True else 0.0
            birth_score += 0.10 if slow_aligned is True else 0.0
            birth_score += 0.08 if ema20_aligned is True else 0.0
            birth_score += 0.08 if vwap_aligned is True else 0.0
            birth_score += 0.10 if clv_supports_signal else 0.0
            birth_score += 0.10 if dirvol_supports_signal else 0.0
            birth_score += 0.06 if not body_against_signal else 0.0
            birth_score += 0.08 if not details["quality_stretched_entry"] else 0.0
            if range_vs_avg is not None and not pd.isna(range_vs_avg) and float(range_vs_avg) >= 0.90:
                birth_score += 0.05
            if details["market_entry_adverse_extreme"]:
                birth_score -= 0.15
            if details["market_entry_rejection"]:
                birth_score -= 0.15
            if details["soft_contradicted"]:
                birth_score -= 0.18
            if details["hard_contradicted"]:
                birth_score -= 0.25
            birth_score = max(min(birth_score, 1.0), 0.0)

            exhaustion_score = 0.0
            if details["quality_stretched_entry"]:
                exhaustion_score += 0.22
            if details["market_entry_adverse_extreme"]:
                exhaustion_score += 0.18
            if details["market_entry_rejection"]:
                exhaustion_score += 0.18
            if details["soft_contradicted"]:
                exhaustion_score += 0.18
            if details["hard_contradicted"]:
                exhaustion_score += 0.22
            if clv_shows_exhaustion:
                exhaustion_score += 0.08
            if slow_aligned is True and fast_aligned is False:
                exhaustion_score += 0.14
            if alignment_total > 0 and (alignment_hits / alignment_total) < 0.50:
                exhaustion_score += 0.10
            if (
                range_vs_avg is not None
                and not pd.isna(range_vs_avg)
                and float(range_vs_avg) >= news_range_vs_avg_min
                and (body_against_signal or not dirvol_supports_signal)
            ):
                exhaustion_score += 0.12
            exhaustion_score = max(min(exhaustion_score, 1.0), 0.0)
            details["impulse_birth_score"] = birth_score
            details["impulse_exhaustion_score"] = exhaustion_score

            if bool(settings.get("entry_quality_impulse_routing_enabled", True)):
                birth_market_min = float(
                    settings.get("entry_quality_impulse_birth_market_score_min", 0.72) or 0.72
                )
                birth_split_min = float(
                    settings.get("entry_quality_impulse_birth_split_score_min", 0.48) or 0.48
                )
                exhausted_retrace_min = float(
                    settings.get("entry_quality_impulse_exhausted_retrace_score_min", 0.55) or 0.55
                )
                exhausted_skip_min = float(
                    settings.get("entry_quality_impulse_exhausted_skip_score_min", 0.78) or 0.78
                )
                if score < retrace_min:
                    details["quality_decision"] = "skip"
                    details["quality_skip"] = bool(settings.get("entry_quality_skip_on_low", True))
                    details["impulse_state"] = "exhausted"
                elif exhaustion_score >= exhausted_skip_min and score < market_min:
                    details["quality_decision"] = "skip"
                    details["quality_skip"] = bool(settings.get("entry_quality_skip_on_low", True))
                    details["impulse_state"] = "exhausted"
                elif details["market_entry_retrace_only"] or exhaustion_score >= exhausted_retrace_min:
                    details["quality_decision"] = "retrace_only"
                    details["quality_force_retrace_only"] = bool(
                        settings.get("entry_quality_force_retrace_on_medium", True)
                    )
                    details["impulse_state"] = "exhausted"
                elif birth_score >= birth_market_min and score >= market_min:
                    details["quality_decision"] = "market"
                    details["impulse_state"] = "birth"
                elif birth_score >= birth_split_min:
                    details["quality_decision"] = "split"
                    details["quality_force_split_only"] = True
                    details["impulse_state"] = "mature"
                elif score < market_min:
                    details["quality_decision"] = "retrace_only"
                    details["quality_force_retrace_only"] = bool(
                        settings.get("entry_quality_force_retrace_on_medium", True)
                    )
                    details["impulse_state"] = "mature"
                else:
                    details["quality_decision"] = "split"
                    details["quality_force_split_only"] = True
                    details["impulse_state"] = "mature"
            else:
                if score < retrace_min:
                    details["quality_decision"] = "skip"
                    details["quality_skip"] = bool(settings.get("entry_quality_skip_on_low", True))
                elif details["market_entry_retrace_only"] or details["quality_stretched_entry"] or score < market_min:
                    details["quality_decision"] = "retrace_only"
                    details["quality_force_retrace_only"] = bool(
                        settings.get("entry_quality_force_retrace_on_medium", True)
                    )
                else:
                    details["quality_decision"] = "market"
                details["impulse_state"] = (
                    "birth" if details["quality_decision"] == "market" else "mature"
                )
        if details["hard_contradicted"]:
            details["reason"] = "hard_candle_directional_contradiction"
        elif details["market_entry_retrace_only"]:
            details["reason"] = "market_entry_extreme_rejection"
        elif details["soft_contradicted"]:
            details["reason"] = "soft_candle_directional_contradiction"
        else:
            details["reason"] = "entry_context_aligned"
        if details["reason"] == "entry_context_aligned":
            if details["quality_decision"] == "market" and details["impulse_state"] == "birth":
                details["reason"] = "impulse_birth_market"
            elif details["quality_decision"] == "split":
                details["reason"] = "impulse_mature_split"
            elif details["quality_decision"] == "retrace_only" and details["impulse_state"] == "exhausted":
                details["reason"] = "impulse_exhausted_retrace_only"
            elif details["quality_decision"] == "skip" and details["impulse_state"] == "exhausted":
                details["reason"] = "impulse_exhausted_skip"
        return details

    def _candidate_comparison_price(self, candidate: dict[str, Any] | None) -> float | None:
        if not candidate:
            return None
        for key in ("trigger_price", "breakout_trigger_price", "reference_price"):
            try:
                value = pd.to_numeric(pd.Series([candidate.get(key)]), errors="coerce").iloc[0]
                if pd.notna(value):
                    return float(value)
            except Exception:
                continue
        return None

    def _is_more_favorable_entry_price(
        self,
        *,
        side: str,
        candidate_price: float | None,
        reference_price: float | None,
    ) -> bool:
        if candidate_price is None or reference_price is None:
            return False
        side_upper = str(side or "").upper()
        if side_upper == "BUY":
            return float(candidate_price) < float(reference_price)
        if side_upper == "SELL":
            return float(candidate_price) > float(reference_price)
        return False

    def _cap_retrace_candidate_entry_improvement(
        self,
        *,
        candidate: dict[str, Any] | None,
        side: str,
        predicted_pips: float | None,
        pip_size: float,
        digits: int,
        candle_high: float,
        candle_low: float,
        candle_close: float,
        max_improvement_fraction: float,
        min_improvement_pips: float,
    ) -> dict[str, Any] | None:
        if not candidate:
            return candidate
        side_upper = str(side or "").upper()
        if side_upper not in {"BUY", "SELL"} or pip_size <= 0 or predicted_pips is None or pd.isna(predicted_pips):
            return candidate

        predicted_abs = abs(float(predicted_pips))
        if not np.isfinite(predicted_abs) or predicted_abs <= 0:
            return candidate

        current_improvement = pd.to_numeric(
            pd.Series([candidate.get("entry_improvement_pips")]),
            errors="coerce",
        ).iloc[0]
        reference_price = pd.to_numeric(
            pd.Series([candidate.get("reference_price")]),
            errors="coerce",
        ).iloc[0]
        custom_stop_price = pd.to_numeric(
            pd.Series([candidate.get("custom_stop_price")]),
            errors="coerce",
        ).iloc[0]
        reference_stop_pips = pd.to_numeric(
            pd.Series([candidate.get("reference_stop_pips")]),
            errors="coerce",
        ).iloc[0]

        if any(pd.isna(value) for value in [current_improvement, reference_price, custom_stop_price, reference_stop_pips]):
            return candidate

        max_improvement_pips = max(
            float(min_improvement_pips or 0.0),
            predicted_abs * max(float(max_improvement_fraction or 0.0), 0.0),
        )
        if float(current_improvement) <= max_improvement_pips + 1e-9:
            return candidate

        reference_price = float(reference_price)
        custom_stop_price = float(custom_stop_price)
        candle_high = float(candle_high)
        candle_low = float(candle_low)
        candle_close = float(candle_close)
        epsilon = max(pip_size * 0.1, 1e-9)

        if side_upper == "BUY":
            capped_trigger_price = reference_price - max_improvement_pips * pip_size
            capped_trigger_price = min(capped_trigger_price, candle_close - epsilon)
            capped_trigger_price = max(capped_trigger_price, candle_low + epsilon)
            if not (candle_low < capped_trigger_price < candle_close) or not (custom_stop_price < capped_trigger_price):
                return candidate
        else:
            capped_trigger_price = reference_price + max_improvement_pips * pip_size
            capped_trigger_price = max(capped_trigger_price, candle_close + epsilon)
            capped_trigger_price = min(capped_trigger_price, candle_high - epsilon)
            if not (candle_close < capped_trigger_price < candle_high) or not (custom_stop_price > capped_trigger_price):
                return candidate

        capped_candidate = dict(candidate)
        capped_candidate["trigger_price"] = round(float(capped_trigger_price), int(max(digits, 0)))
        capped_candidate["entry_improvement_pips"] = float(
            abs(reference_price - capped_trigger_price) / pip_size
        )
        capped_candidate["reference_stop_pips"] = float(reference_stop_pips)
        return capped_candidate

    def _build_adaptive_entry_profile(
        self,
        *,
        signal: str,
        feature_row: pd.Series | dict[str, Any] | None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        settings = settings or self._get_entry_staging_settings()
        signal_upper = str(signal or "").upper()
        default_profile = {
            "profile": "weak_or_mixed",
            "retrace_fraction": float(settings.get("weak_trend_retrace_fraction", 0.55)),
            "max_stage_bars": int(settings.get("weak_trend_max_stage_bars", settings.get("max_stage_bars", 2))),
            "breakout_partial_fraction": float(settings.get("weak_trend_breakout_partial_fraction", 0.0)),
        }
        if not bool(settings.get("adaptive_profile_enabled", True)) or signal_upper not in {"BUY", "SELL"}:
            return default_profile

        roc_value = self._coerce_feature_value(feature_row, str(settings.get("adaptive_roc_column", "ROC_6")))
        adx_value = self._coerce_feature_value(feature_row, str(settings.get("adaptive_adx_column", "ADX_14")))
        dirvol_value = self._coerce_feature_value(
            feature_row,
            str(settings.get("adaptive_directional_volume_column", "DirectionalVolumeProxy_ZScore_20")),
        )

        adx_abs = abs(float(adx_value)) if adx_value is not None else 0.0
        roc_abs = abs(float(roc_value)) if roc_value is not None else 0.0
        dirvol_abs = abs(float(dirvol_value)) if dirvol_value is not None else 0.0
        roc_aligned = (
            roc_value is not None
            and ((signal_upper == "BUY" and float(roc_value) > 0) or (signal_upper == "SELL" and float(roc_value) < 0))
        )
        dirvol_aligned = (
            dirvol_value is not None
            and ((signal_upper == "BUY" and float(dirvol_value) > 0) or (signal_upper == "SELL" and float(dirvol_value) < 0))
        )

        if (
            adx_abs >= float(settings.get("strong_trend_adx_min", 35.0))
            and roc_aligned
            and roc_abs >= float(settings.get("strong_trend_roc_abs_min", 0.00045))
            and dirvol_aligned
            and dirvol_abs >= float(settings.get("strong_trend_volume_abs_min", 0.35))
        ):
            return {
                "profile": "strong_trend",
                "retrace_fraction": float(settings.get("strong_trend_retrace_fraction", 0.20)),
                "max_stage_bars": int(settings.get("strong_trend_max_stage_bars", 1)),
                "breakout_partial_fraction": float(settings.get("strong_trend_breakout_partial_fraction", 0.35)),
            }

        if (
            adx_abs >= float(settings.get("normal_trend_adx_min", 22.0))
            and roc_aligned
            and roc_abs >= float(settings.get("normal_trend_roc_abs_min", 0.00020))
            and (
                dirvol_value is None
                or (dirvol_aligned and dirvol_abs >= float(settings.get("normal_trend_volume_abs_min", 0.05)))
            )
        ):
            return {
                "profile": "normal_trend",
                "retrace_fraction": float(settings.get("normal_trend_retrace_fraction", 0.35)),
                "max_stage_bars": int(settings.get("normal_trend_max_stage_bars", 2)),
                "breakout_partial_fraction": float(settings.get("normal_trend_breakout_partial_fraction", 0.20)),
            }

        return default_profile

    def _get_price_reference_from_feature_row(
        self,
        feature_row: pd.Series | dict[str, Any] | None,
    ) -> float | None:
        """Obtiene el precio actual usado para convertir retornos esperados a pips."""
        for column_name in ("Close", "Open", "price_true", "price_prev"):
            value = self._coerce_feature_value(feature_row, column_name)
            if value is not None and value > 0:
                return value
        return None

    def _get_supported_model_class_map(self) -> dict[str, Any]:
        """Mapa centralizado de nombres de modelo soportados por el pipeline."""
        return {
            "RandomWalk": RandomWalkModel,
            "RandomWalkModel": RandomWalkModel,
            "Momentum": MomentumModel,
            "MomentumModel": MomentumModel,
            "ARIMA": ArimaModel,
            "PROPHET": ProphetModel,
            "LSTM": LSTMModel,
            "Ridge": RidgeRegressorModel,
            "RidgeRegressor": RidgeRegressorModel,
            "RidgeRegressorModel": RidgeRegressorModel,
            "RandomForestRegressor": RandomForestRegressorModel,
            "HistGradientBoosting": HistGradientBoostingRegressorModel,
            "LogisticRegressionClassifier": LogisticRegressionClassifierModel,
            "RandomForestClassifier": RandomForestClassifierModel,
            "ExtraTreesClassifier": ExtraTreesClassifierModel,
            "HistGradientBoostingClassifier": HistGradientBoostingClassifierModel,
        }

    def _normalize_prediction_output(
        self,
        raw_prediction: Any,
        *,
        expected_len: int,
    ) -> tuple[list[float], list[dict[str, Any] | None]]:
        """Normaliza salidas de regresiÃ³n o clasificaciÃ³n a una forma homogÃ©nea."""
        empty_detail_rows = [None] * expected_len

        if raw_prediction is None:
            return [np.nan] * expected_len, empty_detail_rows

        if isinstance(raw_prediction, dict):
            pred_list = list(raw_prediction.get("predictions", []))
            detail_keys = [k for k in raw_prediction.keys() if k != "predictions"]
            detail_rows: list[dict[str, Any] | None] = []

            for idx in range(len(pred_list)):
                row_detail: dict[str, Any] = {}
                for key in detail_keys:
                    values = raw_prediction.get(key, [])
                    if isinstance(values, (list, tuple, np.ndarray)) and idx < len(values):
                        row_detail[key] = values[idx]
                detail_rows.append(row_detail or None)

            if len(pred_list) != expected_len:
                last_pred = pred_list[-1] if pred_list else np.nan
                pred_list = [last_pred] * expected_len
                last_detail = detail_rows[-1] if detail_rows else None
                detail_rows = [deepcopy(last_detail) if last_detail is not None else None for _ in range(expected_len)]

            return [float(x) if pd.notna(x) else np.nan for x in pred_list], detail_rows

        if isinstance(raw_prediction, (list, tuple, np.ndarray)):
            pred_list = list(raw_prediction)
            if len(pred_list) != expected_len:
                last_pred = pred_list[-1] if pred_list else np.nan
                pred_list = [last_pred] * expected_len
            return [float(x) if pd.notna(x) else np.nan for x in pred_list], empty_detail_rows

        return [float(raw_prediction)] * expected_len, empty_detail_rows

    def _build_signal_target_levels(
        self,
        *,
        signal: str,
        entry_reference: float | None,
        pip_size: float,
        target_pips: float | None,
    ) -> dict[str, float]:
        """Construye niveles TP/SL objetivo de la seÃ±al para auditorÃ­a."""
        signal_upper = str(signal or "HOLD").upper()
        ref = float(entry_reference) if entry_reference is not None else float("nan")
        pip_size = float(pip_size or 0.0)
        target_pips_abs = abs(float(target_pips or 0.0))

        if signal_upper not in {"BUY", "SELL"} or np.isnan(ref) or ref <= 0 or pip_size <= 0 or target_pips_abs <= 0:
            return {
                "signal_target_tp_pips": float("nan"),
                "signal_target_sl_pips": float("nan"),
                "signal_target_tp_price": float("nan"),
                "signal_target_sl_price": float("nan"),
            }

        delta = target_pips_abs * pip_size
        if signal_upper == "BUY":
            return {
                "signal_target_tp_pips": target_pips_abs,
                "signal_target_sl_pips": target_pips_abs,
                "signal_target_tp_price": ref + delta,
                "signal_target_sl_price": ref - delta,
            }

        return {
            "signal_target_tp_pips": target_pips_abs,
            "signal_target_sl_pips": target_pips_abs,
            "signal_target_tp_price": ref - delta,
            "signal_target_sl_price": ref + delta,
        }

    def _flatten_series_details(
        self,
        detail_rows: list[dict[str, Any] | None] | None,
    ) -> dict[str, list[Any]] | None:
        """Convierte una lista de detalles por fila en columnas listas para CSV."""
        if not detail_rows:
            return None

        keys: list[str] = []
        for row in detail_rows:
            if not isinstance(row, dict):
                continue
            for key in row.keys():
                if key not in keys:
                    keys.append(key)

        if not keys:
            return None

        flattened: dict[str, list[Any]] = {key: [] for key in keys}
        for row in detail_rows:
            row_dict = row if isinstance(row, dict) else {}
            for key in keys:
                flattened[key].append(row_dict.get(key))
        return flattened

    def _build_param_suffix(self, params: Dict[str, Any] | None) -> str:
        """Convierte params en un sufijo de archivo legible y estable."""
        if not params:
            return "default"
        param_parts = []
        for k, v in params.items():
            param_parts.append(f"{k}-{str(v)}")
        return "_".join(param_parts)

    def _get_trade_audit_horizon_bars(self) -> int:
        """Horizonte usado para auditar visualmente un trade de backtest."""
        backtest_cfg = self.config.get("backtest", {}) or {}
        explicit_horizon = backtest_cfg.get("audit_horizon_bars")
        if explicit_horizon is not None:
            try:
                return max(int(explicit_horizon), 1)
            except Exception:
                pass

        if self._get_target_mode() == "barrier_event":
            return max(int(self._get_barrier_settings()["horizon_bars"]), 1)

        target_col = str(backtest_cfg.get("target", "ReturnFwd_1") or "")
        match = re.search(r"ReturnFwd[_]?(\d+)", target_col)
        if match:
            try:
                return max(int(match.group(1)), 1)
            except Exception:
                pass

        try:
            return max(int(backtest_cfg.get("horizon", 1) or 1), 1)
        except Exception:
            return 1

    def _get_trade_audit_followthrough_bars(self) -> int:
        """Horizonte extendido para medir si el precio llega al objetivo despues del primer stop."""
        backtest_cfg = self.config.get("backtest", {}) or {}
        explicit_followthrough = backtest_cfg.get("audit_followthrough_bars")
        if explicit_followthrough is not None:
            try:
                return max(int(explicit_followthrough), self._get_trade_audit_horizon_bars())
            except Exception:
                pass

        base_horizon = self._get_trade_audit_horizon_bars()
        return max(base_horizon * 3, base_horizon)

    def _build_backtest_trade_audit_dataframe(
        self,
        *,
        dates: list,
        y_true: list[float],
        y_pred: list[float],
        signal_details: list[dict[str, Any]] | None,
    ) -> pd.DataFrame:
        """Reconstruye el resultado de cada trade del mejor run usando High/Low reales."""
        if self.df_clean is None or not dates or not signal_details:
            return pd.DataFrame()

        price_col = str(self.config.get("eda", {}).get("price_col", "Close"))
        high_col = "High"
        low_col = "Low"
        pip_size = float(self.config.get("backtest", {}).get("pip_size", 0.0001) or 0.0001)
        horizon_bars = self._get_trade_audit_horizon_bars()
        followthrough_bars = self._get_trade_audit_followthrough_bars()

        clean_df = self.df_clean.copy()
        if price_col not in clean_df.columns or high_col not in clean_df.columns or low_col not in clean_df.columns:
            return pd.DataFrame()

        clean_df = clean_df.sort_index()
        clean_index = pd.DatetimeIndex(pd.to_datetime(clean_df.index))

        audit_rows: list[dict[str, Any]] = []
        for idx, ts_raw in enumerate(dates):
            if idx >= len(signal_details):
                break

            detail = signal_details[idx] if isinstance(signal_details[idx], dict) else {}
            signal = str(detail.get("signal", "HOLD") or "HOLD").upper()
            trade_allowed = bool(detail.get("trade_allowed", False))
            if not trade_allowed or signal not in {"BUY", "SELL"}:
                continue

            entry_ts = pd.to_datetime(ts_raw)
            loc = clean_index.get_indexer([entry_ts])
            if len(loc) == 0 or int(loc[0]) < 0:
                continue

            pos = int(loc[0])
            if pos >= len(clean_df) - 1:
                continue

            entry_price = pd.to_numeric(
                pd.Series([detail.get("entry_reference_price")]),
                errors="coerce",
            ).iloc[0]
            if pd.isna(entry_price) or float(entry_price) <= 0:
                entry_price = pd.to_numeric(
                    pd.Series([clean_df.iloc[pos][price_col]]),
                    errors="coerce",
                ).iloc[0]

            tp_price = pd.to_numeric(pd.Series([detail.get("signal_target_tp_price")]), errors="coerce").iloc[0]
            sl_price = pd.to_numeric(pd.Series([detail.get("signal_target_sl_price")]), errors="coerce").iloc[0]
            target_tp_pips = pd.to_numeric(pd.Series([detail.get("signal_target_tp_pips")]), errors="coerce").iloc[0]
            target_sl_pips = pd.to_numeric(pd.Series([detail.get("signal_target_sl_pips")]), errors="coerce").iloc[0]
            confidence = pd.to_numeric(pd.Series([detail.get("confidence")]), errors="coerce").iloc[0]
            predicted_pips = pd.to_numeric(pd.Series([detail.get("predicted_pips")]), errors="coerce").iloc[0]

            if pd.isna(entry_price) or pd.isna(tp_price) or pd.isna(sl_price):
                continue

            future_end_pos = min(pos + horizon_bars, len(clean_df) - 1)
            if future_end_pos <= pos:
                continue

            future_window = clean_df.iloc[pos + 1 : future_end_pos + 1]
            if future_window.empty:
                continue

            followthrough_end_pos = min(pos + followthrough_bars, len(clean_df) - 1)
            followthrough_window = clean_df.iloc[pos + 1 : followthrough_end_pos + 1]

            exit_ts = future_window.index[-1]
            exit_price = pd.to_numeric(pd.Series([future_window.iloc[-1][price_col]]), errors="coerce").iloc[0]
            exit_reason = "horizon"
            trade_result = "TIMEOUT_FLAT"
            bars_to_exit = int(len(future_window))

            favorable_excursion = float("nan")
            adverse_excursion = float("nan")
            first_tp_bar = np.nan
            first_sl_bar = np.nan
            first_ambiguous_bar = np.nan

            if signal == "BUY":
                favorable_excursion = ((future_window[high_col].max() - entry_price) / pip_size) if pip_size > 0 else np.nan
                adverse_excursion = ((future_window[low_col].min() - entry_price) / pip_size) if pip_size > 0 else np.nan
            else:
                favorable_excursion = ((entry_price - future_window[low_col].min()) / pip_size) if pip_size > 0 else np.nan
                adverse_excursion = ((entry_price - future_window[high_col].max()) / pip_size) if pip_size > 0 else np.nan

            for step_idx, (step_ts, step_row) in enumerate(future_window.iterrows(), start=1):
                bar_high = pd.to_numeric(pd.Series([step_row[high_col]]), errors="coerce").iloc[0]
                bar_low = pd.to_numeric(pd.Series([step_row[low_col]]), errors="coerce").iloc[0]
                bar_close = pd.to_numeric(pd.Series([step_row[price_col]]), errors="coerce").iloc[0]

                if pd.isna(bar_high) or pd.isna(bar_low):
                    continue

                if signal == "BUY":
                    tp_hit = bar_high >= tp_price
                    sl_hit = bar_low <= sl_price
                else:
                    tp_hit = bar_low <= tp_price
                    sl_hit = bar_high >= sl_price

                if tp_hit and pd.isna(first_tp_bar):
                    first_tp_bar = float(step_idx)
                if sl_hit and pd.isna(first_sl_bar):
                    first_sl_bar = float(step_idx)

                if tp_hit and sl_hit:
                    first_ambiguous_bar = float(step_idx)
                    exit_ts = step_ts
                    exit_price = bar_close
                    exit_reason = "ambiguous_touch"
                    trade_result = "AMBIGUOUS"
                    bars_to_exit = step_idx
                    break
                if tp_hit:
                    exit_ts = step_ts
                    exit_price = tp_price
                    exit_reason = "tp"
                    trade_result = "WIN"
                    bars_to_exit = step_idx
                    break
                if sl_hit:
                    exit_ts = step_ts
                    exit_price = sl_price
                    exit_reason = "sl"
                    trade_result = "LOSS"
                    bars_to_exit = step_idx
                    break

            realized_pnl_pips = float("nan")
            if pip_size > 0 and pd.notna(exit_price):
                if signal == "BUY":
                    realized_pnl_pips = (float(exit_price) - float(entry_price)) / pip_size
                else:
                    realized_pnl_pips = (float(entry_price) - float(exit_price)) / pip_size

            if exit_reason == "horizon" and pd.notna(realized_pnl_pips):
                if realized_pnl_pips > 0:
                    trade_result = "TIMEOUT_WIN"
                elif realized_pnl_pips < 0:
                    trade_result = "TIMEOUT_LOSS"
                else:
                    trade_result = "TIMEOUT_FLAT"

            eventual_tp_bar = np.nan
            eventual_sl_bar = np.nan
            if not followthrough_window.empty:
                for step_idx, (_, step_row) in enumerate(followthrough_window.iterrows(), start=1):
                    bar_high = pd.to_numeric(pd.Series([step_row[high_col]]), errors="coerce").iloc[0]
                    bar_low = pd.to_numeric(pd.Series([step_row[low_col]]), errors="coerce").iloc[0]
                    if pd.isna(bar_high) or pd.isna(bar_low):
                        continue

                    if signal == "BUY":
                        tp_hit = bar_high >= tp_price
                        sl_hit = bar_low <= sl_price
                    else:
                        tp_hit = bar_low <= tp_price
                        sl_hit = bar_high >= sl_price

                    if tp_hit and pd.isna(eventual_tp_bar):
                        eventual_tp_bar = float(step_idx)
                    if sl_hit and pd.isna(eventual_sl_bar):
                        eventual_sl_bar = float(step_idx)
                    if pd.notna(eventual_tp_bar) and pd.notna(eventual_sl_bar):
                        break

            tp_after_sl_within_followthrough = bool(
                pd.notna(eventual_tp_bar)
                and pd.notna(eventual_sl_bar)
                and float(eventual_tp_bar) > float(eventual_sl_bar)
            )
            sl_after_tp_within_followthrough = bool(
                pd.notna(eventual_tp_bar)
                and pd.notna(eventual_sl_bar)
                and float(eventual_sl_bar) > float(eventual_tp_bar)
            )

            audit_rows.append(
                {
                    "timestamp": entry_ts,
                    "signal": signal,
                    "confidence": confidence,
                    "y_true": float(y_true[idx]) if idx < len(y_true) else np.nan,
                    "y_pred": float(y_pred[idx]) if idx < len(y_pred) else np.nan,
                    "predicted_pips": predicted_pips,
                    "entry_price": float(entry_price),
                    "signal_target_tp_pips": target_tp_pips,
                    "signal_target_sl_pips": target_sl_pips,
                    "signal_target_tp_price": float(tp_price),
                    "signal_target_sl_price": float(sl_price),
                    "exit_timestamp": pd.to_datetime(exit_ts),
                    "exit_price": float(exit_price) if pd.notna(exit_price) else np.nan,
                    "exit_reason": exit_reason,
                    "trade_result": trade_result,
                    "bars_to_exit": bars_to_exit,
                    "realized_pnl_pips": realized_pnl_pips,
                    "favorable_excursion_pips": favorable_excursion,
                    "adverse_excursion_pips": adverse_excursion,
                    "audit_horizon_bars": horizon_bars,
                    "followthrough_horizon_bars": followthrough_bars,
                    "first_tp_bar": first_tp_bar,
                    "first_sl_bar": first_sl_bar,
                    "first_ambiguous_bar": first_ambiguous_bar,
                    "eventual_tp_bar_followthrough": eventual_tp_bar,
                    "eventual_sl_bar_followthrough": eventual_sl_bar,
                    "tp_after_sl_within_followthrough": tp_after_sl_within_followthrough,
                    "sl_after_tp_within_followthrough": sl_after_tp_within_followthrough,
                    "touch_probability": detail.get("touch_probability"),
                    "prob_up": detail.get("prob_up"),
                    "prob_hold": detail.get("prob_hold"),
                    "prob_down": detail.get("prob_down"),
                }
            )

        if not audit_rows:
            return pd.DataFrame()

        df_audit = pd.DataFrame(audit_rows)
        df_audit = df_audit.sort_values("timestamp").reset_index(drop=True)
        return df_audit

    def _save_backtest_trade_audit(
        self,
        *,
        model_name: str,
        params: Dict[str, Any],
        dates: list,
        y_true: list[float],
        y_pred: list[float],
        signal_details: list[dict[str, Any]] | None,
    ) -> pd.DataFrame:
        """Guarda una auditorÃ­a de trades del mejor run para revisiÃ³n visual y cuantitativa."""
        df_audit = self._build_backtest_trade_audit_dataframe(
            dates=dates,
            y_true=y_true,
            y_pred=y_pred,
            signal_details=signal_details,
        )
        if df_audit.empty:
            return df_audit

        backtest_dir = self._get_backtest_output_dir()
        param_suffix = self._build_param_suffix(params)

        audit_path = backtest_dir / f"{model_name}_{param_suffix}_trade_audit.csv"
        df_audit.to_csv(audit_path, index=False)
        self.logger.info(f"      â†³ AuditorÃ­a de trades guardada en: {audit_path}")
        self._archive_backtest_artifact(audit_path)

        outcomes = df_audit["trade_result"].astype(str)
        realized = pd.to_numeric(df_audit["realized_pnl_pips"], errors="coerce")
        signals = df_audit["signal"].astype(str).str.upper()

        def _profit_factor_from_series(series: pd.Series) -> float:
            series_num = pd.to_numeric(series, errors="coerce").dropna()
            gross_profit = float(series_num[series_num > 0].sum())
            gross_loss = float(-series_num[series_num < 0].sum())
            if gross_loss > 0:
                return gross_profit / gross_loss
            if gross_profit > 0:
                return float("inf")
            return np.nan

        summary_payload = {
            "model": model_name,
            "n_trades_audited": int(len(df_audit)),
            "wins_like": int((realized > 0).sum()),
            "losses_like": int((realized < 0).sum()),
            "flat_like": int((realized == 0).sum()),
            "tp_hits": int((df_audit["exit_reason"].astype(str) == "tp").sum()),
            "sl_hits": int((df_audit["exit_reason"].astype(str) == "sl").sum()),
            "ambiguous": int((outcomes == "AMBIGUOUS").sum()),
            "timeouts": int(outcomes.str.startswith("TIMEOUT").sum()),
            "avg_realized_pnl_pips": float(realized.mean()) if not realized.empty else np.nan,
            "median_realized_pnl_pips": float(realized.median()) if not realized.empty else np.nan,
            "avg_bars_to_exit": float(pd.to_numeric(df_audit["bars_to_exit"], errors="coerce").mean()),
            "eventual_tp_within_followthrough": int(pd.to_numeric(df_audit["eventual_tp_bar_followthrough"], errors="coerce").notna().sum()),
            "eventual_sl_within_followthrough": int(pd.to_numeric(df_audit["eventual_sl_bar_followthrough"], errors="coerce").notna().sum()),
            "tp_after_sl_within_followthrough": int(df_audit["tp_after_sl_within_followthrough"].fillna(False).astype(bool).sum()),
            "sl_after_tp_within_followthrough": int(df_audit["sl_after_tp_within_followthrough"].fillna(False).astype(bool).sum()),
            "audit_profit_factor": _profit_factor_from_series(realized),
        }
        summary_payload["tp_after_sl_rate_over_losses"] = (
            float(summary_payload["tp_after_sl_within_followthrough"]) / float(summary_payload["losses_like"])
            if summary_payload["losses_like"] > 0
            else np.nan
        )

        for side_name, prefix in (("BUY", "buy"), ("SELL", "sell")):
            side_mask = signals.eq(side_name)
            side_df = df_audit.loc[side_mask].copy()
            side_realized = pd.to_numeric(side_df.get("realized_pnl_pips"), errors="coerce")
            side_exit_reasons = side_df.get("exit_reason", pd.Series(dtype=object)).astype(str)
            side_losses_like = int((side_realized < 0).sum())
            side_tp_after_sl = int(
                side_df.get("tp_after_sl_within_followthrough", pd.Series(dtype=bool))
                .fillna(False)
                .astype(bool)
                .sum()
            )
            summary_payload[f"{prefix}_trades_audited"] = int(len(side_df))
            summary_payload[f"{prefix}_wins_like"] = int((side_realized > 0).sum())
            summary_payload[f"{prefix}_losses_like"] = side_losses_like
            summary_payload[f"{prefix}_flat_like"] = int((side_realized == 0).sum())
            summary_payload[f"{prefix}_tp_hits"] = int((side_exit_reasons == "tp").sum())
            summary_payload[f"{prefix}_sl_hits"] = int((side_exit_reasons == "sl").sum())
            summary_payload[f"{prefix}_avg_realized_pnl_pips"] = (
                float(side_realized.mean()) if not side_realized.empty else np.nan
            )
            summary_payload[f"{prefix}_median_realized_pnl_pips"] = (
                float(side_realized.median()) if not side_realized.empty else np.nan
            )
            summary_payload[f"{prefix}_audit_profit_factor"] = _profit_factor_from_series(side_realized)
            summary_payload[f"{prefix}_tp_after_sl_within_followthrough"] = side_tp_after_sl
            summary_payload[f"{prefix}_win_like_rate"] = (
                float(summary_payload[f"{prefix}_wins_like"]) / float(len(side_df)) * 100.0
                if len(side_df) > 0
                else np.nan
            )
            summary_payload[f"{prefix}_tp_after_sl_rate_over_losses"] = (
                float(side_tp_after_sl) / float(side_losses_like)
                if side_losses_like > 0
                else np.nan
            )
        buy_win_like_rate = pd.to_numeric(summary_payload.get("buy_win_like_rate"), errors="coerce")
        sell_win_like_rate = pd.to_numeric(summary_payload.get("sell_win_like_rate"), errors="coerce")
        buy_audit_pf = pd.to_numeric(summary_payload.get("buy_audit_profit_factor"), errors="coerce")
        sell_audit_pf = pd.to_numeric(summary_payload.get("sell_audit_profit_factor"), errors="coerce")
        buy_avg_realized = pd.to_numeric(summary_payload.get("buy_avg_realized_pnl_pips"), errors="coerce")
        sell_avg_realized = pd.to_numeric(summary_payload.get("sell_avg_realized_pnl_pips"), errors="coerce")
        buy_trades = pd.to_numeric(summary_payload.get("buy_trades_audited"), errors="coerce")
        sell_trades = pd.to_numeric(summary_payload.get("sell_trades_audited"), errors="coerce")
        summary_payload["directional_balance_score"] = (
            float(min(buy_win_like_rate, sell_win_like_rate))
            if pd.notna(buy_win_like_rate) and pd.notna(sell_win_like_rate)
            else np.nan
        )
        summary_payload["directional_pf_floor"] = (
            float(min(buy_audit_pf, sell_audit_pf))
            if pd.notna(buy_audit_pf) and pd.notna(sell_audit_pf)
            else np.nan
        )
        summary_payload["directional_avg_pnl_floor"] = (
            float(min(buy_avg_realized, sell_avg_realized))
            if pd.notna(buy_avg_realized) and pd.notna(sell_avg_realized)
            else np.nan
        )
        summary_payload["directional_trade_floor"] = (
            int(min(buy_trades, sell_trades))
            if pd.notna(buy_trades) and pd.notna(sell_trades)
            else 0
        )
        summary_df = pd.DataFrame([{**summary_payload, **(params or {})}])
        summary_path = backtest_dir / f"{model_name}_{param_suffix}_trade_audit_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        self.logger.info(f"      â†³ Resumen de auditorÃ­a guardado en: {summary_path}")
        self._archive_backtest_artifact(summary_path)

        return df_audit

    def _save_backtest_monthly_stability(
        self,
        *,
        model_name: str,
        params: Dict[str, Any],
        audit_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Guarda un agregado mensual del mejor run para revisar estabilidad temporal."""
        if audit_df is None or audit_df.empty:
            return pd.DataFrame()

        df_monthly = audit_df.copy()
        df_monthly["timestamp"] = pd.to_datetime(df_monthly["timestamp"], errors="coerce")
        df_monthly["exit_timestamp"] = pd.to_datetime(df_monthly["exit_timestamp"], errors="coerce")
        df_monthly["realized_pnl_pips"] = pd.to_numeric(df_monthly["realized_pnl_pips"], errors="coerce")
        df_monthly["month"] = df_monthly["timestamp"].dt.to_period("M").astype(str)
        if df_monthly["month"].isna().all():
            return pd.DataFrame()

        grouped = (
            df_monthly.groupby("month", dropna=False)
            .agg(
                trades=("timestamp", "count"),
                wins_like=("realized_pnl_pips", lambda s: int((s > 0).sum())),
                losses_like=("realized_pnl_pips", lambda s: int((s < 0).sum())),
                flat_like=("realized_pnl_pips", lambda s: int((s == 0).sum())),
                tp_hits=("exit_reason", lambda s: int((s.astype(str) == "tp").sum())),
                sl_hits=("exit_reason", lambda s: int((s.astype(str) == "sl").sum())),
                timeouts=("trade_result", lambda s: int(s.astype(str).str.startswith("TIMEOUT").sum())),
                ambiguous=("trade_result", lambda s: int((s.astype(str) == "AMBIGUOUS").sum())),
                net_realized_pnl_pips=("realized_pnl_pips", "sum"),
                avg_realized_pnl_pips=("realized_pnl_pips", "mean"),
                median_realized_pnl_pips=("realized_pnl_pips", "median"),
                avg_bars_to_exit=("bars_to_exit", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
                eventual_tp_within_followthrough=("eventual_tp_bar_followthrough", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
                eventual_sl_within_followthrough=("eventual_sl_bar_followthrough", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
                tp_after_sl_within_followthrough=("tp_after_sl_within_followthrough", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
                sl_after_tp_within_followthrough=("sl_after_tp_within_followthrough", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
            )
            .reset_index()
        )
        grouped["win_like_rate"] = np.where(
            grouped["trades"] > 0,
            grouped["wins_like"] / grouped["trades"] * 100.0,
            np.nan,
        )
        grouped["tp_after_sl_rate_over_losses"] = np.where(
            grouped["losses_like"] > 0,
            grouped["tp_after_sl_within_followthrough"] / grouped["losses_like"] * 100.0,
            np.nan,
        )

        for side_name, prefix in (("BUY", "buy"), ("SELL", "sell")):
            side_df = df_monthly[df_monthly["signal"].astype(str).str.upper().eq(side_name)].copy()
            if side_df.empty:
                grouped[f"{prefix}_trades"] = 0
                grouped[f"{prefix}_wins_like"] = 0
                grouped[f"{prefix}_losses_like"] = 0
                grouped[f"{prefix}_net_realized_pnl_pips"] = np.nan
                grouped[f"{prefix}_avg_realized_pnl_pips"] = np.nan
                grouped[f"{prefix}_median_realized_pnl_pips"] = np.nan
                grouped[f"{prefix}_tp_hits"] = 0
                grouped[f"{prefix}_sl_hits"] = 0
                grouped[f"{prefix}_tp_after_sl_within_followthrough"] = 0
                grouped[f"{prefix}_win_like_rate"] = np.nan
                grouped[f"{prefix}_tp_after_sl_rate_over_losses"] = np.nan
                continue

            side_grouped = (
                side_df.groupby("month", dropna=False)
                .agg(
                    trades=("timestamp", "count"),
                    wins_like=("realized_pnl_pips", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
                    losses_like=("realized_pnl_pips", lambda s: int((pd.to_numeric(s, errors="coerce") < 0).sum())),
                    net_realized_pnl_pips=("realized_pnl_pips", "sum"),
                    avg_realized_pnl_pips=("realized_pnl_pips", "mean"),
                    median_realized_pnl_pips=("realized_pnl_pips", "median"),
                    tp_hits=("exit_reason", lambda s: int((s.astype(str) == "tp").sum())),
                    sl_hits=("exit_reason", lambda s: int((s.astype(str) == "sl").sum())),
                    tp_after_sl_within_followthrough=(
                        "tp_after_sl_within_followthrough",
                        lambda s: int(pd.Series(s).fillna(False).astype(bool).sum()),
                    ),
                )
                .reset_index()
            )
            side_grouped[f"{prefix}_win_like_rate"] = np.where(
                side_grouped["trades"] > 0,
                side_grouped["wins_like"] / side_grouped["trades"] * 100.0,
                np.nan,
            )
            side_grouped[f"{prefix}_tp_after_sl_rate_over_losses"] = np.where(
                side_grouped["losses_like"] > 0,
                side_grouped["tp_after_sl_within_followthrough"] / side_grouped["losses_like"] * 100.0,
                np.nan,
            )
            side_grouped = side_grouped.rename(
                columns={
                    "trades": f"{prefix}_trades",
                    "wins_like": f"{prefix}_wins_like",
                    "losses_like": f"{prefix}_losses_like",
                    "net_realized_pnl_pips": f"{prefix}_net_realized_pnl_pips",
                    "avg_realized_pnl_pips": f"{prefix}_avg_realized_pnl_pips",
                    "median_realized_pnl_pips": f"{prefix}_median_realized_pnl_pips",
                    "tp_hits": f"{prefix}_tp_hits",
                    "sl_hits": f"{prefix}_sl_hits",
                    "tp_after_sl_within_followthrough": f"{prefix}_tp_after_sl_within_followthrough",
                }
            )
            grouped = grouped.merge(side_grouped, on="month", how="left")

        grouped["directional_balance_score"] = grouped[
            ["buy_win_like_rate", "sell_win_like_rate"]
        ].min(axis=1, skipna=False)
        grouped["directional_avg_pnl_floor"] = grouped[
            ["buy_avg_realized_pnl_pips", "sell_avg_realized_pnl_pips"]
        ].min(axis=1, skipna=False)
        grouped["directional_trade_floor"] = grouped[
            ["buy_trades", "sell_trades"]
        ].min(axis=1, skipna=False)

        grouped["model"] = model_name
        for key, value in (params or {}).items():
            grouped[key] = value

        backtest_dir = self._get_backtest_output_dir()
        param_suffix = self._build_param_suffix(params)
        monthly_path = backtest_dir / f"{model_name}_{param_suffix}_monthly_stability.csv"
        grouped.to_csv(monthly_path, index=False)
        self.logger.info(f"      â†³ Estabilidad mensual guardada en: {monthly_path}")
        self._archive_backtest_artifact(monthly_path)
        return grouped

    def _plot_trade_audit(
        self,
        *,
        audit_df: pd.DataFrame,
        model_name: str,
        params: Dict[str, Any] | None = None,
        suffix: str = "_trade_audit",
    ) -> None:
        """Grafica precio real con entrada, TP/SL objetivo y salida de cada trade auditado."""
        if audit_df is None or audit_df.empty or self.df_clean is None:
            return

        price_col = str(self.config.get("eda", {}).get("price_col", "Close"))
        clean_df = self.df_clean.copy().sort_index()
        if price_col not in clean_df.columns:
            return

        entry_times = pd.to_datetime(audit_df["timestamp"], errors="coerce")
        exit_times = pd.to_datetime(audit_df["exit_timestamp"], errors="coerce")
        start_ts = entry_times.min()
        end_ts = exit_times.max()
        if pd.isna(start_ts) or pd.isna(end_ts):
            return

        price_series = clean_df.loc[
            (pd.to_datetime(clean_df.index) >= start_ts) & (pd.to_datetime(clean_df.index) <= end_ts),
            price_col,
        ]
        if price_series.empty:
            return

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(price_series.index, price_series.values, color="steelblue", linewidth=1.4, label="Precio real")

        for _, row in audit_df.iterrows():
            entry_ts = pd.to_datetime(row.get("timestamp"), errors="coerce")
            exit_ts = pd.to_datetime(row.get("exit_timestamp"), errors="coerce")
            entry_price = pd.to_numeric(pd.Series([row.get("entry_price")]), errors="coerce").iloc[0]
            exit_price = pd.to_numeric(pd.Series([row.get("exit_price")]), errors="coerce").iloc[0]
            tp_price = pd.to_numeric(pd.Series([row.get("signal_target_tp_price")]), errors="coerce").iloc[0]
            sl_price = pd.to_numeric(pd.Series([row.get("signal_target_sl_price")]), errors="coerce").iloc[0]
            signal = str(row.get("signal", "HOLD")).upper()
            trade_result = str(row.get("trade_result", "TIMEOUT_FLAT")).upper()

            if pd.isna(entry_ts) or pd.isna(exit_ts) or pd.isna(entry_price) or pd.isna(exit_price):
                continue

            if "WIN" in trade_result:
                result_color = "green"
            elif "LOSS" in trade_result:
                result_color = "red"
            elif "AMBIGUOUS" in trade_result:
                result_color = "orange"
            else:
                result_color = "gray"

            entry_marker = "^" if signal == "BUY" else "v"
            ax.scatter(entry_ts, entry_price, marker=entry_marker, color="black", s=28, alpha=0.9)
            ax.scatter(exit_ts, exit_price, marker="o", color=result_color, s=24, alpha=0.9)
            ax.plot([entry_ts, exit_ts], [entry_price, exit_price], color=result_color, linewidth=0.9, alpha=0.55)

            if pd.notna(tp_price) and pd.notna(sl_price):
                ax.hlines(
                    y=[float(tp_price), float(sl_price)],
                    xmin=entry_ts,
                    xmax=exit_ts,
                    colors=["green", "red"],
                    linestyles="dotted",
                    linewidth=0.8,
                    alpha=0.14,
                )

        ax.set_title(f"{model_name} - AuditorÃ­a visual de trades{suffix}", fontsize=13, weight="bold")
        ax.set_xlabel("Fecha")
        ax.set_ylabel(price_col)
        ax.grid(True, alpha=0.25)

        legend_handles = [
            Line2D([0], [0], color="steelblue", lw=1.4, label="Precio real"),
            Line2D([0], [0], marker="^", color="black", linestyle="None", markersize=7, label="Entrada BUY"),
            Line2D([0], [0], marker="v", color="black", linestyle="None", markersize=7, label="Entrada SELL"),
            Line2D([0], [0], marker="o", color="green", linestyle="None", markersize=6, label="Salida ganadora"),
            Line2D([0], [0], marker="o", color="red", linestyle="None", markersize=6, label="Salida perdedora"),
            Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=6, label="Salida timeout"),
        ]
        ax.legend(handles=legend_handles, loc="best")

        plt.tight_layout()

        plot_dir = self._get_backtest_output_dir() / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        param_suffix = self._build_param_suffix(params)
        plot_path = plot_dir / f"{model_name}_{param_suffix}{suffix}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        self.logger.info(f"ðŸ“‰ GrÃ¡fico de auditorÃ­a guardado en: {plot_path}")
        self._archive_backtest_artifact(plot_path)

    def _build_trade_decisions_for_predictions(
        self,
        *,
        predictions: list[float],
        feature_rows: list[pd.Series | dict[str, Any] | None],
        price_references: list[float | None],
        prediction_details: list[dict[str, Any] | None] | None = None,
        model_metrics: dict[str, Any] | None = None,
        apply_confidence_filter: bool = True,
        model_name: str | None = None,
        model_cfg: dict[str, Any] | None = None,
    ) -> tuple[list[bool], list[str], list[str], list[float], list[dict[str, Any]]]:
        """Construye seÃ±ales/mascara de trades con la misma lÃ³gica usada en producciÃ³n."""
        trading_cfg = self.config.get("trading", {}) or {}
        pip_size = float(self.config.get("backtest", {}).get("pip_size", 0.0001) or 0.0001)
        min_pips_signal = float(
            trading_cfg.get(
                "min_pips_signal",
                self.config.get("backtest", {}).get("threshold_pips", 0.0),
            )
        )
        enable_confidence_filter = bool(trading_cfg.get("enable_confidence_filter", False))
        min_confidence = float(trading_cfg.get("min_confidence", 0.60))
        effective_min_confidence = (
            min_confidence if (apply_confidence_filter and enable_confidence_filter) else 0.0
        )

        trade_mask: list[bool] = []
        reasons: list[str] = []
        signals: list[str] = []
        confidences: list[float] = []
        signal_details: list[dict[str, Any]] = []
        metrics_payload = model_metrics or {}
        target_mode = self._get_target_mode()
        barrier_settings = self._get_barrier_settings()
        details_iterable = prediction_details or [None] * len(predictions)
        use_probability_mode = (
            target_mode == "barrier_event"
            or (
                self._is_hybrid_mode()
                and self._get_model_stack_role(model_name=model_name, model_cfg=model_cfg) == "filter"
            )
        )

        for pred_value, feature_row, price_reference, prediction_detail in zip(
            predictions,
            feature_rows,
            price_references,
            details_iterable,
        ):
            ref_price = price_reference
            if ref_price is None or ref_price <= 0:
                ref_price = self._get_price_reference_from_feature_row(feature_row)

            if use_probability_mode and prediction_detail:
                signal_info = build_signal_from_probabilities(
                    prob_up=float(prediction_detail.get("prob_up", 0.0) or 0.0),
                    prob_hold=float(prediction_detail.get("prob_hold", 0.0) or 0.0),
                    prob_down=float(prediction_detail.get("prob_down", 0.0) or 0.0),
                    barrier_pips=float(barrier_settings["barrier_pips"]),
                    min_confidence=effective_min_confidence,
                    probability_threshold=float(barrier_settings["probability_threshold"]),
                    probability_margin=float(barrier_settings["probability_margin"]),
                    model_metrics=metrics_payload,
                )
            else:
                signal_info = build_signal_from_prediction(
                    pred_return=float(pred_value),
                    pip_size=pip_size,
                    min_pips_signal=min_pips_signal,
                    model_metrics=metrics_payload,
                    min_confidence=effective_min_confidence,
                    probability=None,
                    price_reference=ref_price,
                )
            signal = str(signal_info["signal"])
            confidence = float(signal_info["confidence"])
            confirmation = self._evaluate_signal_confirmation(
                signal=signal,
                feature_row=feature_row,
            )
            trade_allowed = signal in {"BUY", "SELL"} and bool(confirmation.get("passed", True))
            signal_target = self._build_signal_target_levels(
                signal=signal,
                entry_reference=ref_price,
                pip_size=pip_size,
                target_pips=signal_info.get("signal_target_pips"),
            )

            signal_details.append(
                {
                    "signal": signal,
                    "confidence": confidence,
                    "entry_reference_price": ref_price,
                    "predicted_pips": float(signal_info.get("predicted_pips", np.nan)),
                    "signal_target_pips": (
                        float(signal_info["signal_target_pips"])
                        if signal_info.get("signal_target_pips") is not None
                        else float("nan")
                    ),
                    "touch_probability": signal_info.get("touch_probability"),
                    "prob_up": signal_info.get("prob_up"),
                    "prob_hold": signal_info.get("prob_hold"),
                    "prob_down": signal_info.get("prob_down"),
                    "trade_allowed": trade_allowed,
                    "signal_filter_reason": str(confirmation.get("reason", "confirmation_disabled")),
                    **signal_target,
                }
            )

            signals.append(signal)
            confidences.append(confidence)
            trade_mask.append(trade_allowed)
            reasons.append(str(confirmation.get("reason", "confirmation_disabled")))

        return trade_mask, reasons, signals, confidences, signal_details

    def _build_hybrid_trade_decisions_for_predictions(
        self,
        *,
        predictions: list[float],
        feature_rows: list[pd.Series | dict[str, Any] | None],
        price_references: list[float | None],
        filter_prediction_details: list[dict[str, Any] | None],
        primary_model_metrics: dict[str, Any] | None = None,
        filter_model_metrics: dict[str, Any] | None = None,
        apply_confidence_filter: bool = True,
    ) -> tuple[list[bool], list[str], list[str], list[float], list[dict[str, Any]]]:
        """Construye seÃ±ales hÃ­bridas usando un modelo principal y un filtro probabilÃ­stico."""
        trading_cfg = self.config.get("trading", {}) or {}
        pip_size = float(self.config.get("backtest", {}).get("pip_size", 0.0001) or 0.0001)
        min_pips_signal = float(
            trading_cfg.get(
                "min_pips_signal",
                self.config.get("backtest", {}).get("threshold_pips", 0.0),
            )
        )
        enable_confidence_filter = bool(trading_cfg.get("enable_confidence_filter", False))
        min_confidence = float(trading_cfg.get("min_confidence", 0.60))
        effective_min_confidence = (
            min_confidence if (apply_confidence_filter and enable_confidence_filter) else 0.0
        )
        barrier_settings = self._get_barrier_settings()
        hybrid_cfg = self._get_prediction_stack_settings()

        trade_mask: list[bool] = []
        reasons: list[str] = []
        signals: list[str] = []
        confidences: list[float] = []
        signal_details: list[dict[str, Any]] = []

        details_iterable = filter_prediction_details or [None] * len(predictions)

        for pred_value, feature_row, price_reference, filter_detail in zip(
            predictions,
            feature_rows,
            price_references,
            details_iterable,
        ):
            ref_price = price_reference
            if ref_price is None or ref_price <= 0:
                ref_price = self._get_price_reference_from_feature_row(feature_row)

            filter_detail = filter_detail if isinstance(filter_detail, dict) else {}
            signal_info = build_signal_from_hybrid_prediction(
                pred_return=float(pred_value),
                pip_size=pip_size,
                min_pips_signal=min_pips_signal,
                price_reference=ref_price,
                primary_model_metrics=primary_model_metrics or {},
                prob_up=float(filter_detail.get("prob_up", 0.0) or 0.0),
                prob_hold=float(filter_detail.get("prob_hold", 0.0) or 0.0),
                prob_down=float(filter_detail.get("prob_down", 0.0) or 0.0),
                barrier_pips=float(barrier_settings["barrier_pips"]),
                min_confidence=effective_min_confidence,
                probability_threshold=float(barrier_settings["probability_threshold"]),
                probability_margin=float(barrier_settings["probability_margin"]),
                filter_model_metrics=filter_model_metrics or {},
                require_alignment=bool(hybrid_cfg["require_alignment"]),
                filter_gate_mode=str(hybrid_cfg["filter_gate_mode"]),
                support_probability_threshold=hybrid_cfg["support_probability_threshold"],
                support_probability_margin=hybrid_cfg["support_probability_margin"],
                support_score_min=hybrid_cfg["support_score_min"],
                contradiction_margin=hybrid_cfg["contradiction_margin"],
            )

            signal = str(signal_info["signal"])
            confidence = float(signal_info["confidence"])
            confirmation = self._evaluate_signal_confirmation(
                signal=signal,
                feature_row=feature_row,
            )
            trade_allowed = signal in {"BUY", "SELL"} and bool(confirmation.get("passed", True))
            signal_target = self._build_signal_target_levels(
                signal=signal,
                entry_reference=ref_price,
                pip_size=pip_size,
                target_pips=signal_info.get("signal_target_pips"),
            )

            filter_gate_mode = str(signal_info.get("filter_gate_mode", "full_signal") or "full_signal")
            filter_reason = "filter_gate_blocked"
            if not signal_info.get("gate_passed", False):
                if filter_gate_mode == "full_signal":
                    filter_reason = "filter_hold"
                    if signal_info.get("filter_passed", False) and not signal_info.get("alignment_ok", True):
                        filter_reason = "primary_filter_mismatch"
                elif filter_gate_mode == "direction_support":
                    filter_reason = "direction_support_missing"
                elif filter_gate_mode == "primary_only":
                    filter_reason = "primary_only_gate"
                else:
                    filter_reason = (
                        "filter_contradiction"
                        if bool(signal_info.get("filter_contradicted", False))
                        else "support_score_below_min"
                    )
            elif signal not in {"BUY", "SELL"}:
                filter_reason = "hybrid_hold"

            reason = (
                str(confirmation.get("reason", "confirmation_disabled"))
                if trade_allowed or signal == "HOLD"
                else filter_reason
            )
            if signal in {"BUY", "SELL"} and not confirmation.get("passed", True):
                reason = str(confirmation.get("reason", "confirmation_blocked"))

            signal_details.append(
                {
                    "signal": signal,
                    "confidence": confidence,
                    "entry_reference_price": ref_price,
                    "predicted_pips": float(signal_info.get("predicted_pips", np.nan)),
                    "signal_target_pips": (
                        float(signal_info["signal_target_pips"])
                        if signal_info.get("signal_target_pips") is not None
                        else float("nan")
                    ),
                    "touch_probability": signal_info.get("touch_probability"),
                    "prob_up": signal_info.get("prob_up"),
                    "prob_hold": signal_info.get("prob_hold"),
                    "prob_down": signal_info.get("prob_down"),
                    "primary_signal": signal_info.get("primary_signal"),
                    "primary_confidence": signal_info.get("primary_confidence"),
                    "filter_signal": signal_info.get("filter_signal"),
                    "filter_confidence": signal_info.get("filter_confidence"),
                    "filter_passed": signal_info.get("filter_passed"),
                    "filter_gate_mode": signal_info.get("filter_gate_mode"),
                    "filter_dominant_side": signal_info.get("filter_dominant_side"),
                    "filter_dominant_prob": signal_info.get("filter_dominant_prob"),
                    "filter_support_passed": signal_info.get("filter_support_passed"),
                    "filter_support_score": signal_info.get("filter_support_score"),
                    "filter_same_side_prob": signal_info.get("filter_same_side_prob"),
                    "filter_opposite_side_prob": signal_info.get("filter_opposite_side_prob"),
                    "filter_support_score_passed": signal_info.get("filter_support_score_passed"),
                    "filter_contradicted": signal_info.get("filter_contradicted"),
                    "gate_passed": signal_info.get("gate_passed"),
                    "alignment_ok": signal_info.get("alignment_ok"),
                    "trade_allowed": trade_allowed,
                    "signal_filter_reason": reason,
                    **signal_target,
                }
            )

            signals.append(signal)
            confidences.append(confidence)
            trade_mask.append(trade_allowed)
            reasons.append(reason)

        return trade_mask, reasons, signals, confidences, signal_details

    def _evaluate_hybrid_bundle_candidates(
        self,
        *,
        best_artifacts_by_model: dict[str, dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        """Evalua bundles primary+filter usando las mejores corridas individuales ya calculadas."""
        if not self._is_hybrid_mode():
            return [], {}

        stack_cfg = self._get_prediction_stack_settings()
        candidate_artifacts = [
            artifact
            for artifact in best_artifacts_by_model.values()
            if artifact and self._is_model_selection_candidate(
                model_name=artifact.get("model_name"),
                model_cfg=artifact.get("model_cfg"),
            )
        ]
        if not candidate_artifacts:
            return [], {}

        primary_rows = []
        filter_rows = []
        for artifact in candidate_artifacts:
            role = self._get_model_stack_role(
                model_name=artifact.get("model_name"),
                model_cfg=artifact.get("model_cfg"),
            )
            row = dict(artifact.get("best_run", {}))
            row["model"] = artifact.get("model_name")
            if role == "primary":
                primary_rows.append(row)
            elif role == "filter":
                filter_rows.append(row)

        primary_ranked = self._rank_runs_for_selection(pd.DataFrame(primary_rows), enforce_thresholds=False)
        filter_ranked = self._rank_runs_for_selection(pd.DataFrame(filter_rows), enforce_thresholds=False)
        if primary_ranked.empty or filter_ranked.empty:
            return [], {}

        primary_ranked = primary_ranked.head(int(stack_cfg["top_k_primary_for_bundle_eval"]))
        filter_ranked = filter_ranked.head(int(stack_cfg["top_k_filter_for_bundle_eval"]))

        bundle_rows: list[dict[str, Any]] = []
        bundle_artifacts: dict[str, dict[str, Any]] = {}

        for _, primary_row in primary_ranked.iterrows():
            primary_name = str(primary_row.get("model", ""))
            primary_artifact = best_artifacts_by_model.get(primary_name.upper())
            if not primary_artifact:
                continue

            primary_frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(primary_artifact.get("dates", []), errors="coerce"),
                    "y_true": primary_artifact.get("y_true", []),
                    "y_pred": primary_artifact.get("y_pred", []),
                    "feature_row": primary_artifact.get("feature_rows", []),
                    "price_reference": primary_artifact.get("price_references", []),
                }
            ).dropna(subset=["timestamp"])
            if primary_frame.empty:
                continue

            for _, filter_row in filter_ranked.iterrows():
                filter_name = str(filter_row.get("model", ""))
                filter_artifact = best_artifacts_by_model.get(filter_name.upper())
                if not filter_artifact:
                    continue

                filter_frame = pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(filter_artifact.get("dates", []), errors="coerce"),
                        "filter_detail": filter_artifact.get("prediction_details", []),
                    }
                ).dropna(subset=["timestamp"])
                if filter_frame.empty:
                    continue

                aligned = primary_frame.merge(filter_frame, on="timestamp", how="inner")
                if aligned.empty:
                    continue

                aligned_dates = aligned["timestamp"].tolist()
                aligned_true = aligned["y_true"].tolist()
                aligned_pred = aligned["y_pred"].tolist()
                aligned_feature_rows = aligned["feature_row"].tolist()
                aligned_price_refs = aligned["price_reference"].tolist()
                aligned_filter_details = aligned["filter_detail"].tolist()

                trade_mask, confirmation_reasons, _, _, signal_details = self._build_hybrid_trade_decisions_for_predictions(
                    predictions=aligned_pred,
                    feature_rows=aligned_feature_rows,
                    price_references=aligned_price_refs,
                    filter_prediction_details=aligned_filter_details,
                    primary_model_metrics=dict(primary_artifact.get("best_run", {})),
                    filter_model_metrics=dict(filter_artifact.get("best_run", {})),
                    apply_confidence_filter=True,
                )
                metrics = self._calculate_metrics(aligned_true, aligned_pred, trade_mask=trade_mask)

                bundle_label = f"HYBRID__{primary_name}__{filter_name}"
                bundle_row = {
                    "model": bundle_label,
                    "selection_role": "candidate",
                    "bundle_mode": "hybrid_primary_plus_filter",
                    "primary_model": primary_name,
                    "filter_model": filter_name,
                    "primary_params_json": json.dumps(primary_artifact.get("params", {}), ensure_ascii=False, sort_keys=True),
                    "filter_params_json": json.dumps(filter_artifact.get("params", {}), ensure_ascii=False, sort_keys=True),
                    **metrics,
                }
                bundle_rows.append(bundle_row)
                bundle_artifacts[bundle_label] = {
                    "model_name": bundle_label,
                    "primary_model": primary_name,
                    "filter_model": filter_name,
                    "params": {
                        "primary_model": primary_name,
                        "filter_model": filter_name,
                    },
                    "dates": aligned_dates,
                    "y_true": aligned_true,
                    "y_pred": aligned_pred,
                    "feature_rows": aligned_feature_rows,
                    "price_references": aligned_price_refs,
                    "prediction_details": signal_details,
                    "signal_details": signal_details,
                    "trade_mask": trade_mask,
                    "confirmation_reasons": confirmation_reasons,
                    "best_run": bundle_row,
                }

        return bundle_rows, bundle_artifacts

    def _evaluate_signal_confirmation(
        self,
        *,
        signal: str,
        feature_row: pd.Series | dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Aplica filtros opcionales de momentum/volumen/rÃ©gimen a una seÃ±al ya propuesta."""
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
            "staged": output_dir / "staged_signal_report.csv",
            "entry_grid": output_dir / "entry_grid_legs_report.csv",
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

    def _load_cached_production_signals(self) -> pd.DataFrame:
        """Carga production_signals.csv con cache liviano para decisiones runtime."""
        path = self._get_production_output_paths()["signals"]
        defaults = pd.DataFrame()
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            mtime_ns = None

        cache = getattr(self, "_production_signals_cache", None)
        if (
            isinstance(cache, dict)
            and cache.get("path") == str(path)
            and cache.get("mtime_ns") == mtime_ns
            and isinstance(cache.get("df"), pd.DataFrame)
        ):
            return cache["df"]

        if not path.exists():
            self._production_signals_cache = {"path": str(path), "mtime_ns": mtime_ns, "df": defaults}
            return defaults

        try:
            df = pd.read_csv(path, low_memory=False)
        except (EmptyDataError, FileNotFoundError):
            df = pd.DataFrame()
        except Exception as e:
            self.logger.warning("No se pudo leer production_signals.csv para runtime monitor: %s", e)
            df = pd.DataFrame()

        if not df.empty:
            df = self._normalize_signal_history_dataframe(df, path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            elif "signal_time" in df.columns:
                df["timestamp"] = pd.to_datetime(df["signal_time"], errors="coerce")
            else:
                df["timestamp"] = pd.NaT
            if "strategy_profile" in df.columns:
                df["strategy_profile_norm"] = df["strategy_profile"].apply(
                    lambda value: self._normalize_profile_label(self._coerce_textish(value, ""))
                )
            else:
                df["strategy_profile_norm"] = None
            if "symbol" in df.columns:
                df["symbol_norm"] = df["symbol"].astype(str).str.strip().str.upper()
            else:
                df["symbol_norm"] = ""
            if "signal" in df.columns:
                df["signal_norm"] = df["signal"].astype(str).str.strip().str.upper()
            else:
                df["signal_norm"] = ""
            if "primary_signal" in df.columns:
                df["primary_signal_norm"] = df["primary_signal"].astype(str).str.strip().str.upper()
            else:
                df["primary_signal_norm"] = ""
            if "primary_confidence" in df.columns:
                df["primary_confidence_num"] = pd.to_numeric(df["primary_confidence"], errors="coerce")
            else:
                df["primary_confidence_num"] = np.nan

        self._production_signals_cache = {"path": str(path), "mtime_ns": mtime_ns, "df": df}
        return df

    def _get_latest_runtime_signal_snapshot(
        self,
        *,
        row: pd.Series,
        current_bar_timestamp: pd.Timestamp | None = None,
    ) -> dict[str, Any] | None:
        """Obtiene la ultima señal reciente del mismo perfil/símbolo para gestión live."""
        signals = self._load_cached_production_signals()
        if signals is None or signals.empty:
            return None

        symbol = self._coerce_textish(row.get("symbol"), "").strip().upper()
        if not symbol:
            return None

        raw_profile = self._coerce_textish(
            row.get("strategy_profile"),
            self._get_strategy_profile_name() or "",
        )
        profile_norm = self._normalize_profile_label(raw_profile)

        subset = signals[signals["symbol_norm"] == symbol].copy()
        if subset.empty:
            return None
        if profile_norm:
            subset = subset[subset["strategy_profile_norm"] == profile_norm]
            if subset.empty:
                return None

        subset = subset[subset["signal_norm"].isin(["BUY", "SELL", "HOLD"])].copy()
        if subset.empty:
            return None

        signal_time = pd.to_datetime(row.get("signal_time"), errors="coerce")
        if pd.notna(signal_time):
            subset = subset[subset["timestamp"] > signal_time]
            if subset.empty:
                return None

        subset = subset.sort_values(by=["timestamp"], ascending=True)
        directional_subset = subset[subset["signal_norm"].isin(["BUY", "SELL"])].copy()
        latest = directional_subset.iloc[-1] if not directional_subset.empty else subset.iloc[-1]
        latest_ts = pd.to_datetime(latest.get("timestamp"), errors="coerce")
        timeframe = self._coerce_textish(
            row.get("timeframe"),
            self.config.get("data", {}).get("timeframe", "M5"),
        )
        timeframe_delta = self._timeframe_to_timedelta(timeframe)
        age_bars = 0
        if (
            current_bar_timestamp is not None
            and pd.notna(latest_ts)
            and timeframe_delta.total_seconds() > 0
        ):
            age_bars = max(
                int((current_bar_timestamp - latest_ts).total_seconds() // timeframe_delta.total_seconds()),
                0,
            )

        return {
            "timestamp": latest_ts,
            "signal": self._coerce_textish(latest.get("signal_norm"), "").strip().upper(),
            "primary_signal": self._coerce_textish(latest.get("primary_signal_norm"), "").strip().upper(),
            "primary_confidence": pd.to_numeric(
                pd.Series([latest.get("primary_confidence_num")]), errors="coerce"
            ).iloc[0],
            "age_bars": age_bars,
        }

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
        """Agrega filas a un CSV unificando columnas con el histÃ³rico."""
        if df_rows is None or df_rows.empty:
            return None

        if path.exists():
            try:
                existing = pd.read_csv(path, low_memory=False)
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
                        f"No se pudo escribir {path} porque estÃ¡ en uso. Reintentando ({attempt + 1}/3)..."
                    )
                    time.sleep(1.0)
                    continue

                fallback_path = path.with_name(
                    f"{path.stem}_locked_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}"
                )
                df_to_save.to_csv(fallback_path, index=False)
                self.logger.error(
                    f"No se pudo escribir {path}. Probablemente estÃ¡ abierto en Excel o bloqueado por otro proceso. "
                    f"Se guardÃ³ una copia alternativa en: {fallback_path}"
                )
                return fallback_path

        return path

    def _coerce_boolish(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None or pd.isna(value):
            return False
        return str(value).strip().lower() in {"true", "1", "yes", "si"}

    def _coerce_textish(self, value: Any, default: str = "") -> str:
        if value is None or pd.isna(value):
            return default
        return str(value)

    def _should_force_filter_hold_retrace_only(
        self,
        *,
        signal: str,
        row: pd.Series | dict[str, Any],
        context_check: dict[str, Any] | None = None,
    ) -> bool:
        signal_upper = str(signal or "").strip().upper()
        if signal_upper not in {"BUY", "SELL"}:
            return False

        settings = self._get_entry_management_settings()
        if not bool(settings.get("disable_pending_when_filter_hold", False)):
            return False

        filter_signal = self._coerce_textish(getattr(row, "get", lambda *_: None)("filter_signal"), "").strip().upper()
        if filter_signal != "HOLD":
            return False

        soft_contradicted = (
            bool(context_check.get("soft_contradicted"))
            if context_check is not None
            else self._coerce_boolish(getattr(row, "get", lambda *_: None)("entry_context_soft_contradicted"))
        )
        adverse_extreme = (
            bool(context_check.get("market_entry_adverse_extreme"))
            if context_check is not None
            else self._coerce_boolish(getattr(row, "get", lambda *_: None)("entry_context_market_entry_adverse_extreme"))
        )
        retrace_only = (
            bool(context_check.get("market_entry_retrace_only"))
            if context_check is not None
            else self._coerce_boolish(getattr(row, "get", lambda *_: None)("entry_context_market_entry_retrace_only"))
        )
        market_rejection = (
            bool(context_check.get("market_entry_rejection"))
            if context_check is not None
            else self._coerce_boolish(getattr(row, "get", lambda *_: None)("entry_context_market_entry_rejection"))
        )
        context_reason = (
            self._coerce_textish(context_check.get("reason"), "")
            if context_check is not None
            else self._coerce_textish(getattr(row, "get", lambda *_: None)("entry_context_reason"), "")
        ).strip().lower()

        if context_reason == "soft_candle_directional_contradiction":
            soft_contradicted = True
        if context_reason == "market_entry_extreme_rejection":
            retrace_only = True
            market_rejection = True

        if retrace_only:
            return True
        if bool(settings.get("filter_hold_small_market_retrace_on_soft_contradiction", True)) and soft_contradicted:
            return True
        if bool(settings.get("filter_hold_small_market_retrace_on_market_rejection", True)) and market_rejection:
            return True
        if bool(settings.get("filter_hold_small_market_retrace_on_adverse_extreme", True)) and adverse_extreme:
            return True
        return False

    def _should_allow_filter_hold_small_market(
        self,
        *,
        signal: str,
        row: pd.Series | dict[str, Any],
        predicted_pips: float | None = None,
        context_check: dict[str, Any] | None = None,
    ) -> bool:
        signal_upper = str(signal or "").strip().upper()
        if signal_upper not in {"BUY", "SELL"}:
            return False

        settings = self._get_entry_management_settings()
        if not bool(settings.get("disable_pending_when_filter_hold", False)):
            return False

        filter_signal = self._coerce_textish(getattr(row, "get", lambda *_: None)("filter_signal"), "").strip().upper()
        if filter_signal != "HOLD":
            return False

        if signal_upper == "BUY" and not bool(settings.get("filter_hold_small_market_allow_buy", False)):
            return False
        if signal_upper == "SELL" and not bool(settings.get("filter_hold_small_market_allow_sell", True)):
            return False

        if self._should_force_filter_hold_retrace_only(signal=signal_upper, row=row, context_check=context_check):
            return False

        impulse_state = (
            self._coerce_textish(context_check.get("impulse_state"), "")
            if context_check is not None
            else self._coerce_textish(
                getattr(row, "get", lambda *_: None)("entry_context_impulse_state"),
                "",
            )
        ).strip().lower()
        if impulse_state == "mature":
            if bool(settings.get("filter_hold_small_market_retrace_on_mature_enabled", False)):
                return False
            if signal_upper == "SELL" and bool(
                settings.get("filter_hold_small_market_retrace_on_mature_sell_enabled", False)
            ):
                return False

        context_reason = (
            self._coerce_textish(context_check.get("reason"), "")
            if context_check is not None
            else self._coerce_textish(getattr(row, "get", lambda *_: None)("entry_context_reason"), "")
        ).strip().lower()
        if bool(settings.get("filter_hold_small_market_require_aligned_context", True)):
            if context_reason not in {"", "entry_context_aligned"}:
                return False

        primary_confidence = pd.to_numeric(
            pd.Series([getattr(row, "get", lambda *_: None)("primary_confidence")]),
            errors="coerce",
        ).iloc[0]
        if pd.isna(primary_confidence):
            primary_confidence = pd.to_numeric(
                pd.Series([getattr(row, "get", lambda *_: None)("confidence")]),
                errors="coerce",
            ).iloc[0]
        if pd.isna(primary_confidence):
            return False
        if float(primary_confidence) < float(settings.get("filter_hold_small_market_confidence_min", 0.88)):
            return False

        predicted_candidates = [predicted_pips]
        getter = getattr(row, "get", lambda *_: None)
        predicted_candidates.extend(
            [
                getter("signal_target_tp_pips"),
                getter("predicted_move_pips"),
                getter("predicted_pips"),
                getter("pips"),
                getter("expected_move_pips"),
            ]
        )
        candidate_pips = np.nan
        for value in predicted_candidates:
            numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.notna(numeric_value):
                candidate_pips = float(abs(numeric_value))
                break
        if pd.isna(candidate_pips):
            return False
        if float(candidate_pips) < float(settings.get("filter_hold_small_market_predicted_pips_min", 4.8)):
            return False
        return True

    def _get_staged_signal_defaults(self) -> dict[str, Any]:
        return {
            "candidate_id": pd.NA,
            "parent_signal_id": pd.NA,
            "release_id": pd.NA,
            "strategy_profile": pd.NA,
            "symbol": pd.NA,
            "timeframe": pd.NA,
            "model": pd.NA,
            "side": pd.NA,
            "created_at": pd.NA,
            "source_timestamp": pd.NA,
            "expires_at": pd.NA,
            "reference_price": np.nan,
            "trigger_price": np.nan,
            "breakout_trigger_price": np.nan,
            "reference_stop_pips": np.nan,
            "entry_improvement_pips": np.nan,
            "predicted_pips": np.nan,
            "pred_return": np.nan,
            "candidate_mode": pd.NA,
            "candidate_volume_scale": 1.0,
            "adaptive_profile": pd.NA,
            "adaptive_retrace_fraction": np.nan,
            "adaptive_breakout_fraction": np.nan,
            "signal_candle_open": np.nan,
            "signal_candle_high": np.nan,
            "signal_candle_low": np.nan,
            "signal_candle_close": np.nan,
            "custom_stop_price": np.nan,
            "primary_confidence": np.nan,
            "filter_signal": pd.NA,
            "filter_support_score": np.nan,
            "filter_contradicted": False,
            "last_directional_volume_column": pd.NA,
            "last_directional_volume_value": np.nan,
            "last_directional_volume_passed": False,
            "last_directional_volume_reason": pd.NA,
            "last_execution_confirmation_timeframe": pd.NA,
            "last_execution_confirmation_score": np.nan,
            "last_execution_confirmation_passed": False,
            "last_execution_confirmation_reason": pd.NA,
            "last_execution_confirmation_hits": np.nan,
            "last_execution_confirmation_total": np.nan,
            "last_execution_confirmation_as_of": pd.NA,
            "status": pd.NA,
            "status_reason": pd.NA,
            "last_evaluated_at": pd.NA,
            "refresh_action": pd.NA,
            "refresh_reason": pd.NA,
            "refresh_trigger_price_prev": np.nan,
            "refresh_trigger_price_new": np.nan,
            "refresh_breakout_trigger_price_prev": np.nan,
            "refresh_breakout_trigger_price_new": np.nan,
            "activation_timestamp": pd.NA,
            "activation_price": np.nan,
            "activation_reason": pd.NA,
            "cancel_timestamp": pd.NA,
            "cancel_reason": pd.NA,
        }

    def _load_staged_signal_report(self) -> pd.DataFrame:
        path = self._get_production_output_paths()["staged"]
        defaults = self._get_staged_signal_defaults()
        if not path.exists():
            return pd.DataFrame(columns=list(defaults.keys()))
        try:
            staged = pd.read_csv(path)
        except EmptyDataError:
            staged = pd.DataFrame()
        if staged.empty:
            staged = pd.DataFrame(columns=list(defaults.keys()))
        for col, default_value in defaults.items():
            if col not in staged.columns:
                staged[col] = default_value
        return staged

    def _save_staged_signal_report(self, staged: pd.DataFrame) -> None:
        path = self._get_production_output_paths()["staged"]
        defaults = self._get_staged_signal_defaults()
        if staged is None:
            staged = pd.DataFrame(columns=list(defaults.keys()))
        for col, default_value in defaults.items():
            if col not in staged.columns:
                staged[col] = default_value
        staged = staged.reindex(columns=list(defaults.keys()))
        self._write_dataframe_atomic(path, staged)

    def _serialize_entry_grid_plan(self, plan: dict[str, Any] | None) -> str | None:
        if not plan:
            return None
        try:
            return json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return None

    def _parse_entry_grid_plan_from_row(self, row: pd.Series | dict[str, Any]) -> dict[str, Any] | None:
        payload = row.get("entry_grid_plan") if isinstance(row, dict) else row.get("entry_grid_plan")
        if payload is None or (isinstance(payload, float) and pd.isna(payload)):
            return None
        text = str(payload).strip()
        if not text or text.lower() == "nan":
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _build_entry_grid_plan(
        self,
        *,
        signal: str,
        total_volume_lots: float,
        live_entry_price: float,
        live_sl_price: float,
        live_tp_price: float | None,
        digits: int,
        min_lot: float,
        lot_step: float,
        timeframe: str,
        signal_time: Any,
        adaptive_profile: str,
    ) -> dict[str, Any]:
        settings = self._get_entry_grid_settings()
        total_volume_lots = self._normalize_volume_to_step(
            total_volume_lots,
            min_lot=min_lot,
            lot_step=lot_step,
        )
        disabled_plan = {
            "enabled": False,
            "mode": settings["mode"],
            "group_id": None,
            "adaptive_profile": adaptive_profile,
            "runner_legs": int(settings["runner_legs"]),
            "legs": [],
            "total_volume_lots": total_volume_lots,
            "comment": "entry_grid_disabled",
        }
        side = str(signal or "").upper()
        if (
            not settings["enabled"]
            or side not in {"BUY", "SELL"}
            or total_volume_lots <= 0
            or any(pd.isna(value) for value in [live_entry_price, live_sl_price])
        ):
            return disabled_plan

        stop_distance = abs(float(live_entry_price) - float(live_sl_price))
        if stop_distance <= 0:
            disabled_plan["comment"] = "entry_grid_invalid_stop_distance"
            return disabled_plan

        raw_legs = settings["legs"]
        if len(raw_legs) < 2:
            disabled_plan["comment"] = "entry_grid_missing_legs"
            return disabled_plan

        weights = [max(float(leg.get("volume_weight") or 0.0), 0.0) for leg in raw_legs]
        total_weight = sum(weights)
        if total_weight <= 0:
            disabled_plan["comment"] = "entry_grid_invalid_weights"
            return disabled_plan
        normalized_weights = [weight / total_weight for weight in weights]

        leg_volumes: list[float] = []
        for weight in normalized_weights:
            leg_volumes.append(
                self._normalize_volume_to_step(
                    total_volume_lots * weight,
                    min_lot=min_lot,
                    lot_step=lot_step,
                )
            )

        assigned_volume = sum(leg_volumes)
        volume_gap = self._normalize_volume_to_step(
            max(total_volume_lots - assigned_volume, 0.0),
            min_lot=min_lot,
            lot_step=lot_step,
        )
        if volume_gap > 0 and len(leg_volumes) > 0:
            step_value = float(lot_step if lot_step > 0 else min_lot)
            leg_idx = len(leg_volumes) - 1
            while volume_gap + 1e-12 >= step_value and leg_idx >= 0:
                leg_volumes[leg_idx] = float(round(leg_volumes[leg_idx] + step_value, 8))
                volume_gap = float(round(max(volume_gap - step_value, 0.0), 8))
                leg_idx = len(leg_volumes) - 1 if leg_idx == 0 else leg_idx - 1

        if any(volume <= 0 for volume in leg_volumes):
            disabled_plan["comment"] = "entry_grid_leg_below_min_lot"
            return disabled_plan

        signal_ts = pd.to_datetime(signal_time, errors="coerce")
        timeframe_delta = self._timeframe_to_timedelta(timeframe)
        shared_sl = round(float(live_sl_price), int(max(digits, 0)))
        shared_tp = (
            round(float(live_tp_price), int(max(digits, 0)))
            if live_tp_price is not None and not pd.isna(live_tp_price)
            else None
        )

        legs: list[dict[str, Any]] = []
        for idx, (leg_cfg, weight, planned_volume) in enumerate(
            zip(raw_legs, normalized_weights, leg_volumes),
            start=1,
        ):
            entry_type = str(leg_cfg.get("entry_type", "limit") or "limit").strip().lower()
            spacing_fraction = float(leg_cfg.get("spacing_fraction_of_stop", 0.0) or 0.0)
            expiry_bars = int(leg_cfg.get("expiry_bars", 0) or 0)
            if idx == 1 or entry_type == "market_or_breakout":
                planned_entry_price = round(float(live_entry_price), int(max(digits, 0)))
                order_type = "MARKET"
                expiry_time = None
                entry_leg = "grid_market_1"
            else:
                if side == "BUY":
                    planned_entry_price = float(live_entry_price) - stop_distance * spacing_fraction
                    valid_price = float(live_sl_price) < planned_entry_price < float(live_entry_price)
                    order_type = "BUY_LIMIT"
                else:
                    planned_entry_price = float(live_entry_price) + stop_distance * spacing_fraction
                    valid_price = float(live_entry_price) < planned_entry_price < float(live_sl_price)
                    order_type = "SELL_LIMIT"
                planned_entry_price = round(float(planned_entry_price), int(max(digits, 0)))
                if not valid_price:
                    disabled_plan["comment"] = f"entry_grid_invalid_price_{leg_cfg.get('leg_id', idx)}"
                    return disabled_plan
                expiry_time = (
                    signal_ts + timeframe_delta * expiry_bars
                    if pd.notna(signal_ts) and expiry_bars > 0
                    else None
                )
                entry_leg = f"grid_limit_{idx}"

            legs.append(
                {
                    "leg_id": str(leg_cfg.get("leg_id", f"leg_{idx}") or f"leg_{idx}"),
                    "leg_rank": idx,
                    "entry_leg": entry_leg,
                    "entry_type": "market" if order_type == "MARKET" else "limit",
                    "order_type": order_type,
                    "volume_weight": weight,
                    "planned_volume_lots": planned_volume,
                    "planned_entry_price": planned_entry_price,
                    "planned_sl_price": shared_sl,
                    "planned_tp_price": shared_tp,
                    "spacing_fraction_of_stop": spacing_fraction,
                    "expiry_bars": expiry_bars,
                    "expiry_time": expiry_time.isoformat() if expiry_time is not None else None,
                    "grid_quality_rank": idx,
                    "runner_candidate": idx > max(len(raw_legs) - int(settings["runner_legs"]), 0),
                }
            )

        return {
            "enabled": True,
            "mode": settings["mode"],
            "group_id": None,
            "adaptive_profile": adaptive_profile,
            "runner_legs": int(settings["runner_legs"]),
            "legs": legs,
            "total_volume_lots": total_volume_lots,
            "market_legs": int(sum(1 for leg in legs if leg["entry_type"] == "market")),
            "pending_legs": int(sum(1 for leg in legs if leg["entry_type"] == "limit")),
            "comment": "risk_based_ladder",
        }

    def _apply_entry_grid_to_production_rows(
        self,
        *,
        df_rows: pd.DataFrame,
        runtime_ctx: dict[str, Any],
    ) -> pd.DataFrame:
        if df_rows is None or df_rows.empty:
            return df_rows

        settings = self._get_entry_grid_settings()
        df_rows = df_rows.copy()
        defaults = {
            "entry_grid_enabled": False,
            "entry_grid_group_id": pd.NA,
            "entry_grid_leg_count": 0,
            "entry_grid_market_legs": 0,
            "entry_grid_pending_legs": 0,
            "entry_grid_total_volume_lots": np.nan,
            "entry_grid_runner_legs": 0,
            "entry_grid_plan": pd.NA,
            "entry_grid_apply_profile": pd.NA,
            "entry_grid_comment": pd.NA,
        }
        for col, default_value in defaults.items():
            if col not in df_rows.columns:
                df_rows[col] = default_value
            else:
                df_rows[col] = df_rows[col].where(df_rows[col].notna(), default_value)

        if not settings["enabled"]:
            return df_rows

        staging_settings = self._get_entry_staging_settings()
        digits = int(max(runtime_ctx.get("digits", 5) or 5, 0))
        min_lot = float(runtime_ctx.get("min_lot") or 0.01)
        lot_step = float(runtime_ctx.get("lot_step") or min_lot or 0.01)
        timeframe = str(runtime_ctx.get("timeframe") or "M5")

        for idx, row in df_rows.iterrows():
            signal = str(row.get("signal", "") or "").upper()
            if signal not in {"BUY", "SELL"}:
                continue

            if self._coerce_boolish(row.get("entry_management_split_active")):
                df_rows.at[idx, "entry_grid_comment"] = "grid_skipped_split_active"
                continue

            staged_action = self._coerce_textish(row.get("staged_action"), "").strip().upper()
            if staged_action == "CREATED_WITH_IMMEDIATE_PARTIAL":
                df_rows.at[idx, "entry_grid_comment"] = "grid_skipped_active_staged_remainder"
                continue

            signal_confirmation_reason = self._coerce_textish(row.get("signal_confirmation_reason"), "")
            primary_signal = self._coerce_textish(row.get("primary_signal"), signal).upper()
            filter_signal = self._coerce_textish(row.get("filter_signal"), "").upper()
            gate_passed = self._coerce_boolish(row.get("gate_passed"))
            confirmed_bundle = primary_signal == signal and filter_signal == signal and (
                gate_passed or "gate_passed" not in row.index
            )
            staged_direct_confirmed = signal_confirmation_reason.startswith("entry_staging_direct_confirmed_")
            if settings["require_confirmed_bundle"] and not (confirmed_bundle or staged_direct_confirmed):
                df_rows.at[idx, "entry_grid_comment"] = "grid_skipped_not_confirmed_bundle"
                continue
            if not settings["allow_filter_hold_variant"] and filter_signal == "HOLD" and not staged_direct_confirmed:
                df_rows.at[idx, "entry_grid_comment"] = "grid_skipped_filter_hold"
                continue

            adaptive_profile = self._coerce_textish(row.get("staged_adaptive_profile"), "").strip().lower()
            if not adaptive_profile:
                adaptive_profile = str(
                    self._build_adaptive_entry_profile(
                        signal=signal,
                        feature_row=row,
                        settings=staging_settings,
                    ).get("profile", "")
                ).strip().lower()
            if adaptive_profile not in settings["apply_to_profiles"]:
                df_rows.at[idx, "entry_grid_comment"] = "grid_skipped_profile_not_allowed"
                continue

            total_volume_lots = pd.to_numeric(pd.Series([row.get("volume_lots")]), errors="coerce").iloc[0]
            live_entry_price = pd.to_numeric(pd.Series([row.get("live_entry_price")]), errors="coerce").iloc[0]
            live_sl_price = pd.to_numeric(pd.Series([row.get("live_sl_price")]), errors="coerce").iloc[0]
            live_tp_price = pd.to_numeric(pd.Series([row.get("live_tp_price")]), errors="coerce").iloc[0]
            if any(pd.isna(value) for value in [total_volume_lots, live_entry_price, live_sl_price]):
                df_rows.at[idx, "entry_grid_comment"] = "grid_skipped_missing_runtime_prices"
                continue

            grid_plan = self._build_entry_grid_plan(
                signal=signal,
                total_volume_lots=float(total_volume_lots),
                live_entry_price=float(live_entry_price),
                live_sl_price=float(live_sl_price),
                live_tp_price=None if pd.isna(live_tp_price) else float(live_tp_price),
                digits=digits,
                min_lot=min_lot,
                lot_step=lot_step,
                timeframe=timeframe,
                signal_time=row.get("timestamp"),
                adaptive_profile=adaptive_profile,
            )
            if not grid_plan.get("enabled") or not grid_plan.get("legs"):
                df_rows.at[idx, "entry_grid_comment"] = grid_plan.get("comment", "grid_skipped")
                continue

            signal_id = self._build_signal_id(row)
            grid_group_id = f"{signal_id}|GRID"
            grid_plan["group_id"] = grid_group_id
            market_leg = next((leg for leg in grid_plan["legs"] if leg["entry_type"] == "market"), None)
            if market_leg is None:
                df_rows.at[idx, "entry_grid_comment"] = "grid_skipped_missing_market_leg"
                continue

            df_rows.at[idx, "entry_management_mode"] = grid_plan["mode"]
            df_rows.at[idx, "entry_management_split_active"] = False
            df_rows.at[idx, "entry_management_initial_market_fraction"] = market_leg["volume_weight"]
            df_rows.at[idx, "entry_management_pending_fraction"] = max(1.0 - float(market_leg["volume_weight"]), 0.0)
            df_rows.at[idx, "entry_management_retrace_fraction_of_stop"] = np.nan
            df_rows.at[idx, "entry_management_total_volume_lots"] = grid_plan["total_volume_lots"]
            df_rows.at[idx, "initial_market_volume_lots"] = market_leg["planned_volume_lots"]
            df_rows.at[idx, "pending_order_volume_lots"] = 0.0
            df_rows.at[idx, "pending_order_price"] = np.nan
            df_rows.at[idx, "pending_order_type"] = pd.NA
            df_rows.at[idx, "pending_order_sl_price"] = np.nan
            df_rows.at[idx, "pending_order_tp_price"] = np.nan
            df_rows.at[idx, "pending_order_expiry_time"] = pd.NA
            df_rows.at[idx, "entry_management_comment"] = grid_plan["comment"]
            df_rows.at[idx, "entry_grid_enabled"] = True
            df_rows.at[idx, "entry_grid_group_id"] = grid_group_id
            df_rows.at[idx, "entry_grid_leg_count"] = len(grid_plan["legs"])
            df_rows.at[idx, "entry_grid_market_legs"] = grid_plan["market_legs"]
            df_rows.at[idx, "entry_grid_pending_legs"] = grid_plan["pending_legs"]
            df_rows.at[idx, "entry_grid_total_volume_lots"] = grid_plan["total_volume_lots"]
            df_rows.at[idx, "entry_grid_runner_legs"] = grid_plan["runner_legs"]
            df_rows.at[idx, "entry_grid_plan"] = self._serialize_entry_grid_plan(grid_plan)
            df_rows.at[idx, "entry_grid_apply_profile"] = adaptive_profile
            df_rows.at[idx, "entry_grid_comment"] = grid_plan["comment"]
            self.logger.info(
                "Entrada grid aplicada: side=%s symbol=%s profile=%s market_leg=%.2f pending_legs=%s group=%s",
                signal,
                row.get("symbol"),
                adaptive_profile,
                float(market_leg["planned_volume_lots"]),
                ",".join(
                    f"{leg['leg_id']}@{leg['planned_entry_price']}"
                    for leg in grid_plan["legs"]
                    if leg["entry_type"] == "limit"
                ),
                grid_group_id,
            )

        return df_rows

    def _apply_entry_context_guard_to_production_rows(
        self,
        *,
        df_rows: pd.DataFrame,
    ) -> pd.DataFrame:
        if df_rows is None or df_rows.empty:
            return df_rows

        settings = self._get_entry_staging_settings()
        if not bool(settings.get("context_guard_enabled", True)):
            return df_rows

        df_rows = df_rows.copy()
        for idx, row in df_rows.iterrows():
            signal = str(row.get("signal", "") or "").upper()
            if signal not in {"BUY", "SELL"}:
                continue

            context_check = self._evaluate_entry_context_guard(
                signal=signal,
                feature_row=row,
                candle_open=row.get("signal_candle_open"),
                candle_high=row.get("signal_candle_high"),
                candle_low=row.get("signal_candle_low"),
                candle_close=row.get("signal_candle_close"),
                market_entry_price=row.get("live_entry_price"),
                pip_size=float((self.config.get("data", {}) or {}).get("pip_size", 0.0001) or 0.0001),
                settings=settings,
            )
            df_rows.at[idx, "entry_context_directional_volume_value"] = context_check["directional_volume_value"]
            df_rows.at[idx, "entry_context_close_location_value"] = context_check["close_location_value"]
            df_rows.at[idx, "entry_context_soft_contradicted"] = context_check["soft_contradicted"]
            df_rows.at[idx, "entry_context_hard_contradicted"] = context_check["hard_contradicted"]
            df_rows.at[idx, "entry_context_reason"] = context_check["reason"]
            df_rows.at[idx, "entry_context_market_entry_location_value"] = context_check["market_entry_location_value"]
            df_rows.at[idx, "entry_context_market_entry_range_pips"] = context_check["market_entry_range_pips"]
            df_rows.at[idx, "entry_context_market_entry_adverse_extreme"] = context_check["market_entry_adverse_extreme"]
            df_rows.at[idx, "entry_context_market_entry_rejection"] = context_check["market_entry_rejection"]
            df_rows.at[idx, "entry_context_market_entry_retrace_only"] = context_check["market_entry_retrace_only"]
            df_rows.at[idx, "entry_quality_rank"] = context_check["quality_score"]
            df_rows.at[idx, "entry_context_quality_score"] = context_check["quality_score"]
            df_rows.at[idx, "entry_context_quality_decision"] = context_check["quality_decision"]
            df_rows.at[idx, "entry_context_quality_alignment_hits"] = context_check["quality_alignment_hits"]
            df_rows.at[idx, "entry_context_quality_alignment_total"] = context_check["quality_alignment_total"]
            df_rows.at[idx, "entry_context_quality_signed_distance_to_ema20_pips"] = context_check["quality_signed_distance_to_ema20_pips"]
            df_rows.at[idx, "entry_context_quality_signed_distance_to_vwap_pips"] = context_check["quality_signed_distance_to_vwap_pips"]
            df_rows.at[idx, "entry_context_quality_ema20_stretch_vs_avg_range"] = context_check["quality_ema20_stretch_vs_avg_range"]
            df_rows.at[idx, "entry_context_quality_vwap_stretch_vs_avg_range"] = context_check["quality_vwap_stretch_vs_avg_range"]
            df_rows.at[idx, "entry_context_quality_range_vs_avg_value"] = context_check["quality_range_vs_avg_value"]
            df_rows.at[idx, "entry_context_quality_stretched_from_ema20"] = context_check["quality_stretched_from_ema20"]
            df_rows.at[idx, "entry_context_quality_stretched_from_vwap"] = context_check["quality_stretched_from_vwap"]
            df_rows.at[idx, "entry_context_quality_stretched_entry"] = context_check["quality_stretched_entry"]
            df_rows.at[idx, "entry_context_impulse_birth_score"] = context_check["impulse_birth_score"]
            df_rows.at[idx, "entry_context_impulse_exhaustion_score"] = context_check["impulse_exhaustion_score"]
            df_rows.at[idx, "entry_context_impulse_state"] = context_check["impulse_state"]

            filter_signal_upper = self._coerce_textish(row.get("filter_signal"), "").strip().upper()
            market_entry_location_value = context_check.get("market_entry_location_value")
            if market_entry_location_value is None or pd.isna(market_entry_location_value):
                market_entry_location_value = context_check.get("close_location_value")
            force_opposite_filter_high_buy_retrace = (
                bool(settings.get("entry_quality_force_retrace_on_opposite_filter_buy_high_clv_enabled", False))
                and signal == "BUY"
                and filter_signal_upper in {"BUY", "SELL"}
                and filter_signal_upper != signal
                and market_entry_location_value is not None
                and not pd.isna(market_entry_location_value)
                and float(market_entry_location_value)
                >= float(settings.get("entry_quality_opposite_filter_buy_high_clv_min", 0.90) or 0.90)
            )
            dirvol_for_buy_confirmation = context_check.get("directional_volume_value")
            force_opposite_filter_weak_buy_retrace = (
                bool(settings.get("entry_quality_force_retrace_on_opposite_filter_buy_weak_volume_enabled", False))
                and signal == "BUY"
                and filter_signal_upper in {"BUY", "SELL"}
                and filter_signal_upper != signal
                and (
                    float(context_check.get("quality_score", np.nan))
                    < float(settings.get("entry_quality_opposite_filter_buy_market_score_min", 0.75) or 0.75)
                    or (
                        dirvol_for_buy_confirmation is not None
                        and not pd.isna(dirvol_for_buy_confirmation)
                        and float(dirvol_for_buy_confirmation)
                        < float(settings.get("entry_quality_opposite_filter_buy_dirvol_min", 0.0) or 0.0)
                    )
                )
            )
            cluster_open_positions_value = pd.to_numeric(
                pd.Series([row.get("cluster_open_positions_count")]),
                errors="coerce",
            ).iloc[0]
            cluster_open_positions_count = (
                int(cluster_open_positions_value)
                if pd.notna(cluster_open_positions_value)
                else 0
            )
            force_same_side_reentry_retrace = (
                bool(settings.get("entry_quality_force_retrace_on_same_side_reentry_enabled", False))
                and signal in {"BUY", "SELL"}
                and cluster_open_positions_count
                >= int(settings.get("entry_quality_same_side_reentry_min_open_positions", 1) or 1)
                and (
                    filter_signal_upper not in {"", signal, "HOLD"}
                    or float(context_check.get("quality_score", np.nan))
                    < float(settings.get("entry_quality_same_side_reentry_market_score_min", 0.78) or 0.78)
                    or (
                        context_check.get("directional_volume_value") is not None
                        and not pd.isna(context_check.get("directional_volume_value"))
                        and float(context_check.get("directional_volume_value"))
                        < float(settings.get("entry_quality_same_side_reentry_dirvol_min", 0.0) or 0.0)
                    )
                )
            )
            if force_opposite_filter_high_buy_retrace:
                context_check["quality_force_retrace_only"] = True
                context_check["quality_decision"] = "retrace_only"
                if context_check["reason"] == "entry_context_aligned":
                    context_check["reason"] = "entry_quality_opposite_filter_high_buy_retrace"
                df_rows.at[idx, "entry_context_reason"] = context_check["reason"]
                df_rows.at[idx, "entry_context_quality_decision"] = context_check["quality_decision"]
            elif force_opposite_filter_weak_buy_retrace:
                context_check["quality_force_retrace_only"] = True
                context_check["quality_decision"] = "retrace_only"
                if context_check["reason"] == "entry_context_aligned":
                    context_check["reason"] = "entry_quality_opposite_filter_weak_buy_retrace"
                df_rows.at[idx, "entry_context_reason"] = context_check["reason"]
                df_rows.at[idx, "entry_context_quality_decision"] = context_check["quality_decision"]
            elif force_same_side_reentry_retrace:
                context_check["quality_force_retrace_only"] = True
                context_check["quality_decision"] = "retrace_only"
                if context_check["reason"] == "entry_context_aligned":
                    context_check["reason"] = "same_side_reentry_retrace_only"
                df_rows.at[idx, "entry_context_reason"] = context_check["reason"]
                df_rows.at[idx, "entry_context_quality_decision"] = context_check["quality_decision"]

            staged_status = self._coerce_textish(row.get("staged_status"), "").strip().upper()
            staged_candidate_id = self._coerce_textish(row.get("staged_candidate_id"), "").strip()
            has_active_stage = staged_status == "ACTIVE" and bool(staged_candidate_id)

            if (
                (
                    bool(settings.get("context_guard_hard_block_direct", True))
                    and context_check["hard_contradicted"]
                )
                or bool(context_check["quality_skip"])
            ):
                skip_reason = (
                    "entry_context_hard_contradiction"
                    if context_check["hard_contradicted"]
                    else "entry_quality_low_skip"
                )
                df_rows.at[idx, "signal"] = "HOLD"
                df_rows.at[idx, "signal_confirmation_passed"] = False
                df_rows.at[idx, "signal_confirmation_reason"] = skip_reason
                df_rows.at[idx, "volume_lots"] = 0.0
                df_rows.at[idx, "risk_amount"] = 0.0
                df_rows.at[idx, "allocated_risk_budget"] = 0.0
                df_rows.at[idx, "entry_management_split_active"] = False
                if not has_active_stage:
                    df_rows.at[idx, "entry_management_total_volume_lots"] = 0.0
                df_rows.at[idx, "initial_market_volume_lots"] = 0.0
                df_rows.at[idx, "pending_order_volume_lots"] = 0.0
                df_rows.at[idx, "pending_order_price"] = np.nan
                df_rows.at[idx, "pending_order_type"] = pd.NA
                df_rows.at[idx, "pending_order_sl_price"] = np.nan
                df_rows.at[idx, "pending_order_tp_price"] = np.nan
                df_rows.at[idx, "pending_order_expiry_time"] = pd.NA
                df_rows.at[idx, "entry_management_comment"] = skip_reason
                if self._coerce_boolish(row.get("entry_grid_enabled")):
                    df_rows.at[idx, "entry_grid_comment"] = (
                        "grid_blocked_hard_context_contradiction"
                        if context_check["hard_contradicted"]
                        else "grid_blocked_low_entry_quality"
                    )
                self.logger.info(
                    "Entrada bloqueada por calidad/contexto: side=%s model=%s reason=%s score=%s dirvol=%s clv=%s",
                    signal,
                    row.get("model"),
                    skip_reason,
                    context_check["quality_score"],
                    context_check["directional_volume_value"],
                    context_check["close_location_value"],
                )
                continue

            total_volume_lots = pd.to_numeric(pd.Series([row.get("volume_lots")]), errors="coerce").iloc[0]
            total_volume_lots = float(total_volume_lots) if pd.notna(total_volume_lots) else 0.0
            force_filter_hold_retrace = self._should_force_filter_hold_retrace_only(
                signal=signal,
                row=row,
                context_check=context_check,
            )
            force_split_retrace_filter_opposite = self._should_force_split_retrace_filter_opposite_retrace_only(
                signal=signal,
                row=row,
                context_check=context_check,
            )
            force_quality_retrace = bool(context_check["quality_force_retrace_only"])
            force_quality_split = bool(context_check.get("quality_force_split_only"))
            filter_signal_upper = str(self._coerce_textish(row.get("filter_signal"), "")).upper()
            directional_volume_value = pd.to_numeric(
                pd.Series([context_check.get("directional_volume_value")]),
                errors="coerce",
            ).iloc[0]
            quality_score_value = pd.to_numeric(
                pd.Series([context_check.get("quality_score")]),
                errors="coerce",
            ).iloc[0]
            impulse_state_value = str(context_check.get("impulse_state") or "").strip().lower()
            mature_non_aligned_filter = (
                signal in {"BUY", "SELL"}
                and impulse_state_value == "mature"
                and filter_signal_upper not in {"", signal}
            )
            mature_non_aligned_filter_force_always = (
                bool(settings.get("entry_quality_force_retrace_on_mature_non_aligned_filter_always", False))
                and mature_non_aligned_filter
            )
            force_mature_non_aligned_filter_retrace = (
                bool(settings.get("entry_quality_force_retrace_on_mature_non_aligned_filter_enabled", False))
                and mature_non_aligned_filter
                and (
                    (
                        pd.notna(quality_score_value)
                        and float(quality_score_value)
                        < float(settings.get("entry_quality_mature_non_aligned_filter_market_score_min", 0.74))
                    )
                    or (
                        pd.notna(directional_volume_value)
                        and (
                            (
                                signal == "BUY"
                                and float(directional_volume_value)
                                < float(settings.get("entry_quality_mature_non_aligned_filter_dirvol_min_abs", 0.12))
                            )
                            or (
                                signal == "SELL"
                                and float(directional_volume_value)
                                > -float(settings.get("entry_quality_mature_non_aligned_filter_dirvol_min_abs", 0.12))
                            )
                        )
                    )
                )
            )
            mature_sell_non_sell_filter_force_always = (
                bool(settings.get("entry_quality_force_retrace_on_mature_sell_non_sell_filter_always", False))
                and signal == "SELL"
                and impulse_state_value == "mature"
                and filter_signal_upper != "SELL"
            )
            force_mature_sell_retrace = (
                bool(settings.get("entry_quality_force_retrace_on_mature_sell_non_sell_filter_enabled", False))
                and signal == "SELL"
                and impulse_state_value == "mature"
                and filter_signal_upper != "SELL"
                and (
                    (pd.notna(quality_score_value) and float(quality_score_value) < float(
                        settings.get("entry_quality_mature_sell_non_sell_filter_market_score_min", 0.72)
                    ))
                    or (
                        pd.notna(directional_volume_value)
                        and float(directional_volume_value)
                        > -float(settings.get("entry_quality_mature_sell_non_sell_filter_dirvol_min_abs", 0.20))
                    )
                )
            )
            if (
                mature_non_aligned_filter_force_always
                or force_mature_non_aligned_filter_retrace
                or mature_sell_non_sell_filter_force_always
                or force_mature_sell_retrace
            ):
                force_quality_retrace = True
                force_quality_split = False
                if context_check.get("reason") == "impulse_mature_split":
                    context_check["reason"] = (
                        "mature_non_aligned_filter_forced_retrace_only"
                        if mature_non_aligned_filter_force_always
                        else "mature_non_aligned_filter_retrace_only"
                        if force_mature_non_aligned_filter_retrace
                        else "mature_sell_non_sell_filter_forced_retrace_only"
                        if mature_sell_non_sell_filter_force_always
                        else "mature_sell_non_sell_filter_retrace_only"
                    )
                    df_rows.at[idx, "entry_context_reason"] = context_check["reason"]
                df_rows.at[idx, "entry_context_quality_decision"] = "retrace_only"
            planned_pending_volume = pd.to_numeric(
                pd.Series([row.get("pending_order_volume_lots")]),
                errors="coerce",
            ).iloc[0]
            keep_impulse_split = (
                force_quality_split
                and pd.notna(planned_pending_volume)
                and float(planned_pending_volume) > 0
                and not force_filter_hold_retrace
                and not force_split_retrace_filter_opposite
                and not force_quality_retrace
            )
            if keep_impulse_split:
                df_rows.at[idx, "entry_management_comment"] = "impulse_mature_split_retrace"
                if context_check["reason"] == "entry_context_aligned":
                    context_check["reason"] = "impulse_mature_split"
                    df_rows.at[idx, "entry_context_reason"] = context_check["reason"]
                df_rows.at[idx, "entry_context_quality_decision"] = "split"
            if (
                bool(settings.get("context_guard_soft_disable_market_on_extreme_rejection", True))
                and (
                    bool(context_check["market_entry_retrace_only"])
                    or force_filter_hold_retrace
                    or force_split_retrace_filter_opposite
                    or force_quality_retrace
                )
                and total_volume_lots > 0
            ):
                live_entry_price = pd.to_numeric(pd.Series([row.get("live_entry_price")]), errors="coerce").iloc[0]
                live_sl_price = pd.to_numeric(pd.Series([row.get("live_sl_price")]), errors="coerce").iloc[0]
                live_tp_price = pd.to_numeric(pd.Series([row.get("live_tp_price")]), errors="coerce").iloc[0]
                digits_raw = pd.to_numeric(pd.Series([row.get("digits")]), errors="coerce").iloc[0]
                digits = int(digits_raw) if pd.notna(digits_raw) else 5
                retrace_only_plan = self._build_retrace_only_entry_plan(
                    signal=signal,
                    total_volume_lots=total_volume_lots,
                    live_entry_price=float(live_entry_price) if pd.notna(live_entry_price) else float("nan"),
                    live_sl_price=float(live_sl_price) if pd.notna(live_sl_price) else float("nan"),
                    live_tp_price=float(live_tp_price) if pd.notna(live_tp_price) else float("nan"),
                    digits=digits,
                    timeframe=self._coerce_textish(row.get("timeframe"), self.config.get("data", {}).get("timeframe", "M5")),
                    signal_time=row.get("timestamp", row.get("signal_time", row.get("Time"))),
                    comment=(
                        "split_retrace_filter_opposite_retrace_only"
                        if force_split_retrace_filter_opposite
                        else
                        "entry_quality_opposite_filter_high_buy_retrace"
                        if force_opposite_filter_high_buy_retrace
                        else
                        "entry_quality_opposite_filter_weak_buy_retrace"
                        if force_opposite_filter_weak_buy_retrace
                        else
                        "same_side_reentry_retrace_only"
                        if force_same_side_reentry_retrace
                        else
                        "filter_hold_context_retrace_only"
                        if force_filter_hold_retrace and not bool(context_check["market_entry_retrace_only"])
                        else "entry_quality_retrace_only"
                        if force_quality_retrace and not bool(context_check["market_entry_retrace_only"])
                        else "candle_context_retrace_only"
                    ),
                )
                pending_volume = pd.to_numeric(
                    pd.Series([retrace_only_plan.get("pending_order_volume_lots")]),
                    errors="coerce",
                ).iloc[0]
                if pd.notna(pending_volume) and float(pending_volume) > 0:
                    df_rows.at[idx, "entry_management_mode"] = retrace_only_plan["entry_management_mode"]
                    df_rows.at[idx, "entry_management_split_active"] = retrace_only_plan["entry_management_split_active"]
                    df_rows.at[idx, "entry_management_initial_market_fraction"] = retrace_only_plan[
                        "entry_management_initial_market_fraction"
                    ]
                    df_rows.at[idx, "entry_management_pending_fraction"] = retrace_only_plan[
                        "entry_management_pending_fraction"
                    ]
                    df_rows.at[idx, "entry_management_retrace_fraction_of_stop"] = retrace_only_plan[
                        "entry_management_retrace_fraction_of_stop"
                    ]
                    df_rows.at[idx, "entry_management_total_volume_lots"] = retrace_only_plan[
                        "entry_management_total_volume_lots"
                    ]
                    df_rows.at[idx, "initial_market_volume_lots"] = retrace_only_plan["initial_market_volume_lots"]
                    df_rows.at[idx, "pending_order_volume_lots"] = retrace_only_plan["pending_order_volume_lots"]
                    df_rows.at[idx, "pending_order_price"] = retrace_only_plan["pending_order_price"]
                    df_rows.at[idx, "pending_order_type"] = retrace_only_plan["pending_order_type"]
                    df_rows.at[idx, "pending_order_sl_price"] = retrace_only_plan["pending_order_sl_price"]
                    df_rows.at[idx, "pending_order_tp_price"] = retrace_only_plan["pending_order_tp_price"]
                    df_rows.at[idx, "pending_order_expiry_time"] = retrace_only_plan["pending_order_expiry_time"]
                    df_rows.at[idx, "entry_management_comment"] = retrace_only_plan["entry_management_comment"]
                    self.logger.info(
                        "Pierna market degradada a retrace_only por contexto/calidad: side=%s model=%s entry=%s score=%s dirvol=%s clv=%s",
                        signal,
                        row.get("model"),
                        live_entry_price,
                        context_check["quality_score"],
                        context_check["directional_volume_value"],
                        context_check["close_location_value"],
                    )
                    continue

            pending_volume = pd.to_numeric(pd.Series([row.get("pending_order_volume_lots")]), errors="coerce").iloc[0]
            if (
                bool(settings.get("context_guard_soft_disable_pending", True))
                and not keep_impulse_split
                and context_check["soft_contradicted"]
                and pd.notna(pending_volume)
                and float(pending_volume) > 0
            ):
                market_only_plan = self._build_market_only_entry_plan(
                    total_volume_lots=total_volume_lots,
                    comment="candle_context_market_only",
                )
                df_rows.at[idx, "entry_management_mode"] = market_only_plan["entry_management_mode"]
                df_rows.at[idx, "entry_management_split_active"] = market_only_plan["entry_management_split_active"]
                df_rows.at[idx, "entry_management_initial_market_fraction"] = market_only_plan[
                    "entry_management_initial_market_fraction"
                ]
                df_rows.at[idx, "entry_management_pending_fraction"] = market_only_plan[
                    "entry_management_pending_fraction"
                ]
                df_rows.at[idx, "entry_management_retrace_fraction_of_stop"] = market_only_plan[
                    "entry_management_retrace_fraction_of_stop"
                ]
                df_rows.at[idx, "entry_management_total_volume_lots"] = market_only_plan[
                    "entry_management_total_volume_lots"
                ]
                df_rows.at[idx, "initial_market_volume_lots"] = market_only_plan["initial_market_volume_lots"]
                df_rows.at[idx, "pending_order_volume_lots"] = market_only_plan["pending_order_volume_lots"]
                df_rows.at[idx, "pending_order_price"] = market_only_plan["pending_order_price"]
                df_rows.at[idx, "pending_order_type"] = market_only_plan["pending_order_type"]
                df_rows.at[idx, "pending_order_sl_price"] = market_only_plan["pending_order_sl_price"]
                df_rows.at[idx, "pending_order_tp_price"] = market_only_plan["pending_order_tp_price"]
                df_rows.at[idx, "pending_order_expiry_time"] = market_only_plan["pending_order_expiry_time"]
                df_rows.at[idx, "entry_management_comment"] = market_only_plan["entry_management_comment"]
                if self._coerce_boolish(row.get("entry_grid_enabled")):
                    df_rows.at[idx, "entry_grid_comment"] = "grid_soft_context_market_only"
                self.logger.info(
                    "Pierna pending deshabilitada por contradiccion contextual: side=%s model=%s dirvol=%s clv=%s",
                    signal,
                    row.get("model"),
                    context_check["directional_volume_value"],
                    context_check["close_location_value"],
                )

        return df_rows

    def _get_entry_grid_report_defaults(self) -> dict[str, Any]:
        return {
            "grid_parent_signal_id": pd.NA,
            "grid_group_id": pd.NA,
            "leg_id": pd.NA,
            "leg_rank": np.nan,
            "side": pd.NA,
            "entry_type": pd.NA,
            "volume_weight": np.nan,
            "planned_volume_lots": np.nan,
            "planned_entry_price": np.nan,
            "planned_sl_price": np.nan,
            "planned_tp_price": np.nan,
            "expiry_time": pd.NA,
            "status": pd.NA,
            "status_reason": pd.NA,
            "mt5_order_ticket": np.nan,
            "mt5_position_id": np.nan,
            "execution_price": np.nan,
            "entry_quality_rank": np.nan,
            "is_runner_candidate": False,
        }

    def _save_entry_grid_legs_report_from_lifecycle(self, lifecycle: pd.DataFrame | None) -> None:
        path = self._get_production_output_paths()["entry_grid"]
        defaults = self._get_entry_grid_report_defaults()
        if lifecycle is None or lifecycle.empty or "grid_group_id" not in lifecycle.columns:
            self._write_dataframe_atomic(path, pd.DataFrame(columns=list(defaults.keys())))
            return

        grid_rows = lifecycle[lifecycle["grid_group_id"].notna()].copy()
        if grid_rows.empty:
            self._write_dataframe_atomic(path, pd.DataFrame(columns=list(defaults.keys())))
            return

        report = pd.DataFrame(
            {
                "grid_parent_signal_id": grid_rows.get("grid_parent_signal_id", grid_rows.get("parent_signal_id")),
                "grid_group_id": grid_rows.get("grid_group_id"),
                "leg_id": grid_rows.get("grid_leg_id"),
                "leg_rank": grid_rows.get("grid_leg_rank"),
                "side": grid_rows.get("signal"),
                "entry_type": grid_rows.get("grid_entry_type"),
                "volume_weight": grid_rows.get("grid_volume_weight"),
                "planned_volume_lots": grid_rows.get("requested_volume_lots"),
                "planned_entry_price": grid_rows.get("requested_live_entry_price"),
                "planned_sl_price": grid_rows.get("requested_sl_price"),
                "planned_tp_price": grid_rows.get("requested_tp_price"),
                "expiry_time": grid_rows.get("pending_order_expiry_time"),
                "status": grid_rows.get("status"),
                "status_reason": grid_rows.get("status_detail"),
                "mt5_order_ticket": grid_rows.get("mt5_order_ticket"),
                "mt5_position_id": grid_rows.get("mt5_position_id"),
                "execution_price": grid_rows.get("execution_price"),
                "entry_quality_rank": grid_rows.get("grid_quality_rank"),
                "is_runner_candidate": grid_rows.get("grid_runner_candidate"),
            }
        )
        for col, default_value in defaults.items():
            if col not in report.columns:
                report[col] = default_value
        report = report.reindex(columns=list(defaults.keys()))
        self._write_dataframe_atomic(path, report)

    def _build_candle_retrace_candidate(
        self,
        *,
        side: str,
        candle_high: float,
        candle_low: float,
        candle_close: float,
        reference_price: float,
        atr_value: float | None,
        pip_size: float,
        digits: int,
        retrace_fraction: float,
        stop_buffer_pips: float,
        stop_buffer_atr_fraction: float,
        min_stop_pips: float,
    ) -> dict[str, float] | None:
        side = str(side or "").upper()
        if side not in {"BUY", "SELL"} or pip_size <= 0:
            return None
        if any(pd.isna(value) for value in [candle_high, candle_low, candle_close, reference_price]):
            return None

        candle_high = float(candle_high)
        candle_low = float(candle_low)
        candle_close = float(candle_close)
        reference_price = float(reference_price)
        retrace_fraction = min(max(float(retrace_fraction or 0.0), 0.0), 1.0)
        buffer_from_pips = float(stop_buffer_pips or 0.0) * pip_size
        buffer_from_atr = 0.0
        if atr_value is not None and not pd.isna(atr_value):
            try:
                atr_abs = abs(float(atr_value))
                if np.isfinite(atr_abs):
                    buffer_from_atr = atr_abs * max(float(stop_buffer_atr_fraction or 0.0), 0.0)
            except Exception:
                buffer_from_atr = 0.0
        stop_buffer_price = max(buffer_from_pips, buffer_from_atr)
        min_stop_distance = max(float(min_stop_pips or 0.0), 0.0) * pip_size

        if side == "BUY":
            retrace_range = max(candle_close - candle_low, 0.0)
            trigger_price = candle_close - retrace_fraction * retrace_range
            if not (candle_low < trigger_price < candle_close):
                return None
            raw_stop_price = candle_low - stop_buffer_price
            custom_stop_price = min(raw_stop_price, trigger_price - min_stop_distance)
            if not custom_stop_price < trigger_price:
                return None
        else:
            retrace_range = max(candle_high - candle_close, 0.0)
            trigger_price = candle_close + retrace_fraction * retrace_range
            if not (candle_close < trigger_price < candle_high):
                return None
            raw_stop_price = candle_high + stop_buffer_price
            custom_stop_price = max(raw_stop_price, trigger_price + min_stop_distance)
            if not custom_stop_price > trigger_price:
                return None

        stop_pips_from_reference = abs(reference_price - custom_stop_price) / pip_size
        entry_improvement_pips = abs(reference_price - trigger_price) / pip_size

        return {
            "reference_price": round(reference_price, int(max(digits, 0))),
            "trigger_price": round(float(trigger_price), int(max(digits, 0))),
            "custom_stop_price": round(float(custom_stop_price), int(max(digits, 0))),
            "reference_stop_pips": float(stop_pips_from_reference),
            "entry_improvement_pips": float(entry_improvement_pips),
        }

    def _compute_breakout_trigger_price(
        self,
        *,
        side: str,
        reference_price: float,
        reference_stop_pips: float,
        pip_size: float,
        settings: dict[str, Any] | None = None,
    ) -> float:
        settings = settings or self._get_entry_staging_settings()
        if pip_size <= 0:
            return float("nan")
        trigger_pips = max(
            float(settings.get("breakout_min_trigger_pips", 0.8)),
            float(reference_stop_pips) * float(settings.get("breakout_trigger_fraction_of_stop", 0.12)),
        )
        if str(side or "").upper() == "BUY":
            return float(reference_price) + trigger_pips * pip_size
        if str(side or "").upper() == "SELL":
            return float(reference_price) - trigger_pips * pip_size
        return float("nan")

    def _compute_runtime_trade_payload_for_signal(
        self,
        *,
        signal: str,
        predicted_pips_signed: float | None,
        signal_time: Any,
        runtime_ctx: dict[str, Any],
        volume_scale: float = 1.0,
        disable_entry_management: bool = False,
        explicit_sl_price: float | None = None,
        market_only_comment: str = "pilot_market_only",
    ) -> dict[str, Any]:
        from utils.risk_utils import (
            calculate_position_size_for_risk_amount,
            compute_entry_sl_tp,
            compute_take_profit_pips,
            estimate_position_risk_amount,
        )

        side = str(signal or "").upper()
        if side not in {"BUY", "SELL"}:
            return {}

        price_now = float(runtime_ctx.get("price_now", np.nan))
        if np.isnan(price_now):
            return {}

        atr_value = runtime_ctx.get("atr_value")
        pip_size = float(runtime_ctx.get("pip_size") or 0.0)
        digits = int(runtime_ctx.get("digits") or 5)
        point = float(runtime_ctx.get("point") or pip_size or 0.0001)
        contract_size = float(runtime_ctx.get("contract_size") or 100000.0)
        min_lot = float(runtime_ctx.get("min_lot") or 0.01)
        lot_step = float(runtime_ctx.get("lot_step") or 0.01)
        balance = float(runtime_ctx.get("balance") or 0.0)
        risk_cfg_dict = dict(runtime_ctx.get("risk_cfg_dict", {}) or {})
        total_risk_budget = float(runtime_ctx.get("total_risk_budget") or 0.0)
        per_trade_risk_budget = float(runtime_ctx.get("per_trade_risk_budget") or 0.0)
        open_risk_amount = float(runtime_ctx.get("open_risk_amount") or 0.0)
        planned_additional_risk_amount = float(runtime_ctx.get("planned_additional_risk_amount") or 0.0)
        positions_without_sl = int(runtime_ctx.get("positions_without_sl") or 0)
        market_tick = runtime_ctx.get("market_tick")
        timeframe = str(runtime_ctx.get("timeframe") or self.config.get("data", {}).get("timeframe", "M5"))

        market_reference_price = float("nan")
        if isinstance(market_tick, dict):
            if side == "BUY":
                market_reference_price = float(market_tick.get("ask") or np.nan)
            else:
                market_reference_price = float(market_tick.get("bid") or np.nan)
        if np.isnan(market_reference_price):
            market_reference_price = price_now

        target_pips_value = abs(float(predicted_pips_signed)) if predicted_pips_signed is not None and not pd.isna(predicted_pips_signed) else None
        signal_target_levels = self._build_signal_target_levels(
            signal=side,
            entry_reference=price_now,
            pip_size=pip_size,
            target_pips=target_pips_value,
        )

        explicit_sl_value = None
        if explicit_sl_price is not None and not pd.isna(explicit_sl_price):
            try:
                explicit_sl_value = float(explicit_sl_price)
            except Exception:
                explicit_sl_value = None

        def _enforce_min_stop_distance(
            *,
            entry_ref_price: float,
            stop_price_value: float,
            min_stop_pips: float,
        ) -> float:
            if not np.isfinite(entry_ref_price) or not np.isfinite(stop_price_value):
                return float(stop_price_value)
            min_stop_pips = max(float(min_stop_pips or 0.0), 0.0)
            if min_stop_pips <= 0 or pip_size <= 0:
                return float(stop_price_value)

            min_distance = min_stop_pips * pip_size
            if side == "BUY":
                required_stop = float(entry_ref_price) - min_distance
                return float(min(float(stop_price_value), required_stop))

            required_stop = float(entry_ref_price) + min_distance
            return float(max(float(stop_price_value), required_stop))

        if explicit_sl_value is None:
            planned_levels = compute_entry_sl_tp(
                side=side,
                close_price=price_now,
                atr_value=atr_value,
                pip_size=pip_size,
                risk_cfg_dict=risk_cfg_dict,
                predicted_pips_target=predicted_pips_signed,
            )
            entry_price = round(float(planned_levels["entry_price"]), digits)
            sl_price = round(float(planned_levels["sl_price"]), digits)
            tp_price = round(float(planned_levels["tp_price"]), digits)
            sl_pips = float(planned_levels["sl_pips"])
            tp_pips = float(planned_levels["tp_pips"])

            live_risk_cfg = dict(risk_cfg_dict)
            live_risk_cfg["entry_mode"] = "close"
            live_levels = compute_entry_sl_tp(
                side=side,
                close_price=market_reference_price,
                atr_value=atr_value,
                pip_size=pip_size,
                risk_cfg_dict=live_risk_cfg,
                predicted_pips_target=predicted_pips_signed,
            )
            live_entry_price = round(float(live_levels["entry_price"]), digits)
            live_sl_price = round(float(live_levels["sl_price"]), digits)
            live_tp_price = round(float(live_levels["tp_price"]), digits)
            live_sl_pips = float(live_levels["sl_pips"])
            live_tp_pips = float(live_levels["tp_pips"])
        else:
            entry_price = round(float(price_now), digits)
            planned_dynamic_levels = compute_entry_sl_tp(
                side=side,
                close_price=price_now,
                atr_value=atr_value,
                pip_size=pip_size,
                risk_cfg_dict=risk_cfg_dict,
                predicted_pips_target=predicted_pips_signed,
            )
            planned_min_sl_pips = float(planned_dynamic_levels["sl_pips"])
            adjusted_planned_stop = _enforce_min_stop_distance(
                entry_ref_price=float(entry_price),
                stop_price_value=float(explicit_sl_value),
                min_stop_pips=planned_min_sl_pips,
            )
            sl_price = round(float(adjusted_planned_stop), digits)
            sl_distance = abs(float(entry_price) - float(sl_price))
            if sl_distance <= 0:
                return {}
            sl_pips = float(sl_distance / max(pip_size, 1e-9))
            tp_pips = compute_take_profit_pips(
                sl_pips=sl_pips,
                risk_cfg_dict=risk_cfg_dict,
                predicted_pips_target=predicted_pips_signed,
            )
            tp_distance = tp_pips * pip_size
            if side == "BUY":
                tp_price = round(float(entry_price) + tp_distance, digits)
            else:
                tp_price = round(float(entry_price) - tp_distance, digits)

            live_entry_price = round(float(market_reference_price), digits)
            live_dynamic_levels = compute_entry_sl_tp(
                side=side,
                close_price=market_reference_price,
                atr_value=atr_value,
                pip_size=pip_size,
                risk_cfg_dict=risk_cfg_dict,
                predicted_pips_target=predicted_pips_signed,
            )
            live_min_sl_pips = float(live_dynamic_levels["sl_pips"])
            adjusted_live_stop = _enforce_min_stop_distance(
                entry_ref_price=float(live_entry_price),
                stop_price_value=float(explicit_sl_value),
                min_stop_pips=live_min_sl_pips,
            )
            live_sl_price = round(float(adjusted_live_stop), digits)
            live_sl_distance = abs(float(live_entry_price) - float(live_sl_price))
            if live_sl_distance <= 0:
                return {}
            live_sl_pips = float(live_sl_distance / max(pip_size, 1e-9))
            live_tp_pips = compute_take_profit_pips(
                sl_pips=live_sl_pips,
                risk_cfg_dict=risk_cfg_dict,
                predicted_pips_target=predicted_pips_signed,
            )
            live_tp_distance = live_tp_pips * pip_size
            if side == "BUY":
                live_tp_price = round(float(live_entry_price) + live_tp_distance, digits)
            else:
                live_tp_price = round(float(live_entry_price) - live_tp_distance, digits)

        risk_budget_cfg = self._get_risk_budget_settings()
        available_risk_budget = max(total_risk_budget - open_risk_amount - planned_additional_risk_amount, 0.0)
        block_for_unprotected_positions = (
            risk_budget_cfg["allow_multiple_positions"]
            and risk_budget_cfg["block_new_entries_without_sl"]
            and positions_without_sl > 0
        )
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
        full_volume_lots = calculate_position_size_for_risk_amount(
            entry_price=live_entry_price,
            sl_price=live_sl_price,
            point=point,
            contract_size=contract_size,
            risk_amount=allocated_risk_budget,
            min_lot=min_lot,
            lot_step=lot_step,
        )
        volume_scale = min(max(float(volume_scale or 0.0), 0.0), 1.0)
        if volume_scale <= 0:
            volume_lots = 0.0
        elif volume_scale >= 0.999999:
            volume_lots = full_volume_lots
        else:
            volume_lots = self._normalize_volume_to_step(
                full_volume_lots * volume_scale,
                min_lot=min_lot,
                lot_step=lot_step,
            )
        risk_amount = 0.0
        projected_total_open_risk_after_trade = open_risk_amount + planned_additional_risk_amount
        if volume_lots > 0:
            risk_amount = estimate_position_risk_amount(
                entry_price=live_entry_price,
                sl_price=live_sl_price,
                point=point,
                contract_size=contract_size,
                volume_lots=volume_lots,
            )
            projected_total_open_risk_after_trade += risk_amount
        allocated_risk_budget = min(allocated_risk_budget * volume_scale, allocated_risk_budget)

        if disable_entry_management:
            entry_management_plan = self._build_market_only_entry_plan(
                total_volume_lots=volume_lots,
                comment=market_only_comment,
            )
        else:
            entry_management_plan = self._compute_entry_management_plan(
                signal=side,
                total_volume_lots=volume_lots,
                live_entry_price=live_entry_price,
                live_sl_price=live_sl_price,
                live_tp_price=live_tp_price,
                digits=digits,
                min_lot=min_lot,
                lot_step=lot_step,
                timeframe=timeframe,
                signal_time=signal_time,
            )

        return {
            "signal_target_tp_price": signal_target_levels["signal_target_tp_price"],
            "signal_target_sl_price": signal_target_levels["signal_target_sl_price"],
            "signal_target_tp_pips": signal_target_levels["signal_target_tp_pips"],
            "signal_target_sl_pips": signal_target_levels["signal_target_sl_pips"],
            "entry_price": entry_price,
            "planned_entry_price": entry_price,
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
            "volume_lots": volume_lots,
            "risk_amount": risk_amount,
            "allocated_risk_budget": allocated_risk_budget,
            "risk_per_pip_per_lot": risk_per_pip_per_lot,
            "risk_per_lot_at_stop": risk_per_lot_at_stop,
            "remaining_risk_budget_before_trade": max(total_risk_budget - open_risk_amount - planned_additional_risk_amount, 0.0),
            "projected_total_open_risk_after_trade": projected_total_open_risk_after_trade,
            "entry_management_plan": entry_management_plan,
        }

    def _apply_entry_staging_to_production_rows(
        self,
        *,
        df_rows: pd.DataFrame,
        runtime_ctx: dict[str, Any],
    ) -> pd.DataFrame:
        if df_rows is None or df_rows.empty:
            return df_rows

        settings = self._get_entry_staging_settings()
        df_rows = df_rows.copy()
        staging_defaults = {
            "entry_staging_enabled": bool(settings["enabled"]),
            "entry_staging_mode": settings["mode"],
            "pilot_entry_enabled": bool(settings["pilot_entry_enabled"]),
            "pilot_entry_applied": False,
            "pilot_entry_fraction": settings["pilot_fraction_of_full_size"],
            "pilot_entry_reason": pd.NA,
            "staged_candidate_id": pd.NA,
            "staged_status": pd.NA,
            "staged_action": pd.NA,
            "staged_reason": pd.NA,
            "staged_reference_price": np.nan,
            "staged_trigger_price": np.nan,
            "staged_breakout_trigger_price": np.nan,
            "staged_trigger_price_prev": np.nan,
            "staged_trigger_price_new": np.nan,
            "staged_breakout_trigger_price_prev": np.nan,
            "staged_breakout_trigger_price_new": np.nan,
            "staged_refresh_action": pd.NA,
            "staged_refresh_reason": pd.NA,
            "staged_expires_at": pd.NA,
            "staged_activation_reason": pd.NA,
            "staged_entry_improvement_pips": np.nan,
            "staged_adaptive_profile": pd.NA,
        }
        for col, default_value in staging_defaults.items():
            if col not in df_rows.columns:
                df_rows[col] = default_value
            else:
                df_rows[col] = df_rows[col].where(df_rows[col].notna(), default_value)

        if not settings["enabled"]:
            return df_rows

        staged = self._load_staged_signal_report()
        changed = False
        now_iso = datetime.now().isoformat()
        pip_size = float(runtime_ctx.get("pip_size") or 0.0)
        timeframe_delta = self._timeframe_to_timedelta(str(runtime_ctx.get("timeframe") or "M5"))

        for idx, row in df_rows.iterrows():
            timestamp = pd.to_datetime(row.get("timestamp"), errors="coerce")
            strategy_profile = str(row.get("strategy_profile", "") or "")
            symbol = str(row.get("symbol", "") or "")
            timeframe = str(row.get("timeframe", "") or "")
            model = str(row.get("model", "") or "")
            final_signal = str(row.get("signal", "") or "HOLD").upper()
            primary_signal = str(row.get("primary_signal", "") or final_signal).upper()
            filter_signal = str(row.get("filter_signal", "") or "").upper()
            support_score = pd.to_numeric(pd.Series([row.get("filter_support_score")]), errors="coerce").iloc[0]
            support_score = float(support_score) if pd.notna(support_score) else float("nan")
            primary_confidence = pd.to_numeric(pd.Series([row.get("primary_confidence")]), errors="coerce").iloc[0]
            primary_confidence = float(primary_confidence) if pd.notna(primary_confidence) else float("nan")
            filter_confidence = pd.to_numeric(pd.Series([row.get("filter_confidence")]), errors="coerce").iloc[0]
            filter_confidence = float(filter_confidence) if pd.notna(filter_confidence) else float("nan")
            predicted_pips = pd.to_numeric(
                pd.Series([row.get("pips", row.get("expected_move_pips"))]),
                errors="coerce",
            ).iloc[0]
            predicted_pips = float(predicted_pips) if pd.notna(predicted_pips) else float("nan")
            filter_contradicted = self._coerce_boolish(row.get("filter_contradicted"))
            current_price = pd.to_numeric(pd.Series([row.get("price_now")]), errors="coerce").iloc[0]
            current_price = float(current_price) if pd.notna(current_price) else float(runtime_ctx.get("price_now", np.nan))
            candle_open = pd.to_numeric(pd.Series([row.get("signal_candle_open")]), errors="coerce").iloc[0]
            candle_high = pd.to_numeric(pd.Series([row.get("signal_candle_high")]), errors="coerce").iloc[0]
            candle_low = pd.to_numeric(pd.Series([row.get("signal_candle_low")]), errors="coerce").iloc[0]
            candle_close = pd.to_numeric(pd.Series([row.get("signal_candle_close", row.get("price_now"))]), errors="coerce").iloc[0]
            candle_open = float(candle_open) if pd.notna(candle_open) else float("nan")
            candle_high = float(candle_high) if pd.notna(candle_high) else float("nan")
            candle_low = float(candle_low) if pd.notna(candle_low) else float("nan")
            candle_close = float(candle_close) if pd.notna(candle_close) else current_price
            contradiction_stage_check = self._evaluate_filter_contradiction_stage(
                signal=primary_signal,
                feature_row=row,
                filter_contradicted=filter_contradicted,
                primary_confidence=primary_confidence,
                predicted_pips=predicted_pips,
                settings=settings,
            )
            strong_primary_hold_check = self._evaluate_strong_primary_filter_hold_stage(
                signal=primary_signal,
                feature_row=row,
                filter_signal=filter_signal,
                primary_confidence=primary_confidence,
                predicted_pips=predicted_pips,
                settings=settings,
            )
            medium_primary_hold_check = self._evaluate_medium_primary_filter_hold_stage(
                signal=primary_signal,
                feature_row=row,
                primary_confidence=primary_confidence,
                predicted_pips=predicted_pips,
                settings=settings,
            )
            filter_lead_structural_check = self._evaluate_filter_lead_structural_stage(
                filter_signal=filter_signal,
                feature_row=row,
                primary_signal=primary_signal,
                primary_confidence=primary_confidence,
                filter_confidence=filter_confidence,
                predicted_pips=predicted_pips,
                settings=settings,
            )
            early_reversal_stage_check = self._evaluate_early_structural_reversal_stage(
                signal=primary_signal,
                feature_row=row,
                primary_confidence=primary_confidence,
                predicted_pips=predicted_pips,
                settings=settings,
            )

            def _current_exec_price_for(side_name: str) -> float:
                tick = runtime_ctx.get("market_tick")
                if isinstance(tick, dict):
                    if side_name == "BUY":
                        tick_price = pd.to_numeric(pd.Series([tick.get("ask")]), errors="coerce").iloc[0]
                    else:
                        tick_price = pd.to_numeric(pd.Series([tick.get("bid")]), errors="coerce").iloc[0]
                    if pd.notna(tick_price):
                        return float(tick_price)
                return current_price

            active_mask = (
                staged["strategy_profile"].astype(str).eq(strategy_profile)
                & staged["symbol"].astype(str).eq(symbol)
                & staged["timeframe"].astype(str).eq(timeframe)
                & staged["model"].astype(str).eq(model)
                & staged["status"].astype(str).str.upper().eq("ACTIVE")
            )
            active_candidates = staged[active_mask].copy()
            active_idx = None
            active_candidate = None
            if not active_candidates.empty:
                active_candidates["created_at_ts"] = pd.to_datetime(active_candidates["created_at"], errors="coerce")
                active_candidates = active_candidates.sort_values("created_at_ts", ascending=False)
                active_idx = active_candidates.index[0]
                active_candidate = active_candidates.iloc[0].to_dict()
                duplicate_idxs = active_candidates.index[1:].tolist()
                for dup_idx in duplicate_idxs:
                    staged.at[dup_idx, "status"] = "CANCELLED_SUPERSEDED"
                    staged.at[dup_idx, "cancel_timestamp"] = now_iso
                    staged.at[dup_idx, "cancel_reason"] = "duplicate_active_candidate"
                    staged.at[dup_idx, "status_reason"] = "duplicate_active_candidate"
                    staged.at[dup_idx, "last_evaluated_at"] = now_iso
                    changed = True

            if final_signal in {"BUY", "SELL"}:
                convert_direct_to_staged = False
                direct_stage_mode = ""
                direct_stage_reason = ""
                if (
                    settings["convert_direct_filter_hold_to_staged"]
                    and filter_signal == "HOLD"
                    and final_signal == primary_signal
                    and final_signal in {"BUY", "SELL"}
                    and pip_size > 0
                    and pd.notna(candle_close)
                ):
                    bypass_to_small_market = (
                        final_signal == "SELL"
                        and bool(settings.get("direct_filter_hold_sell_small_market_bypass_enabled", False))
                        and pd.notna(primary_confidence)
                        and float(primary_confidence)
                        >= float(settings.get("direct_filter_hold_sell_small_market_confidence_min", 0.70))
                        and pd.notna(predicted_pips)
                        and abs(float(predicted_pips))
                        >= float(settings.get("direct_filter_hold_sell_small_market_predicted_pips_min", 5.0))
                    )
                    if bypass_to_small_market:
                        convert_direct_to_staged = False
                    else:
                        convert_direct_to_staged = True
                        direct_stage_mode = "direct_filter_hold_candle_retrace"
                        direct_stage_reason = "filter_hold_wait_candle_retrace"
                elif (
                    settings["convert_direct_confirmed_to_staged"]
                    and filter_signal == final_signal
                    and final_signal == primary_signal
                    and final_signal in {"BUY", "SELL"}
                    and pip_size > 0
                    and pd.notna(candle_close)
                ):
                    convert_direct_to_staged = True
                    direct_stage_mode = "direct_confirmed_candle_retrace"
                    direct_stage_reason = "direct_confirmed_wait_candle_retrace"

                if convert_direct_to_staged:
                    reference_price = _current_exec_price_for(final_signal)
                    adaptive_profile = self._build_adaptive_entry_profile(
                        signal=final_signal,
                        feature_row=row,
                        settings=settings,
                    )
                    direct_profile_name = str(adaptive_profile.get("profile", "") or "").strip().lower()
                    direct_retrace_fraction = float(adaptive_profile["retrace_fraction"])
                    if direct_stage_mode == "direct_confirmed_candle_retrace":
                        if direct_profile_name == "strong_trend":
                            direct_retrace_fraction = min(
                                direct_retrace_fraction,
                                float(settings.get("direct_confirmed_strong_trend_retrace_fraction", 0.20)),
                            )
                        elif direct_profile_name == "normal_trend":
                            direct_retrace_fraction = min(
                                direct_retrace_fraction,
                                float(settings.get("direct_confirmed_normal_trend_retrace_fraction", 0.25)),
                            )
                        else:
                            direct_retrace_fraction = min(
                                direct_retrace_fraction,
                                float(settings.get("direct_confirmed_weak_trend_retrace_fraction", 0.40)),
                            )
                    candidate_retrace = self._build_candle_retrace_candidate(
                        side=final_signal,
                        candle_high=candle_high,
                        candle_low=candle_low,
                        candle_close=candle_close,
                        reference_price=reference_price,
                        atr_value=runtime_ctx.get("atr_value"),
                        pip_size=pip_size,
                        digits=int(max(runtime_ctx.get("digits", 5), 0)),
                        retrace_fraction=direct_retrace_fraction,
                        stop_buffer_pips=float(settings["direct_filter_hold_stop_buffer_pips"]),
                        stop_buffer_atr_fraction=float(settings["dynamic_stop_atr_fraction"]),
                        min_stop_pips=float(settings["dynamic_stop_min_pips"]),
                    )
                    if direct_stage_mode == "direct_confirmed_candle_retrace":
                        candidate_retrace = self._cap_retrace_candidate_entry_improvement(
                            candidate=candidate_retrace,
                            side=final_signal,
                            predicted_pips=predicted_pips,
                            pip_size=pip_size,
                            digits=int(max(runtime_ctx.get("digits", 5), 0)),
                            candle_high=candle_high,
                            candle_low=candle_low,
                            candle_close=candle_close,
                            max_improvement_fraction=float(
                                settings.get(
                                    "direct_confirmed_max_entry_improvement_pct_of_predicted_pips",
                                    0.30,
                                )
                            ),
                            min_improvement_pips=float(
                                settings.get("direct_confirmed_max_entry_improvement_min_pips", 0.8)
                            ),
                        )

                    if candidate_retrace is not None and pd.notna(reference_price):
                        if (
                            direct_stage_mode == "direct_confirmed_candle_retrace"
                            and active_idx is not None
                            and bool(settings.get("direct_confirmed_keep_better_active_candidate", True))
                            and str(active_candidate.get("side", "") or "").upper() == final_signal
                        ):
                            active_compare_price = self._candidate_comparison_price(active_candidate)
                            new_compare_price = self._candidate_comparison_price(candidate_retrace)
                            if self._is_more_favorable_entry_price(
                                side=final_signal,
                                candidate_price=active_compare_price,
                                reference_price=new_compare_price,
                            ):
                                staged.at[active_idx, "last_evaluated_at"] = now_iso
                                staged.at[active_idx, "status_reason"] = "better_active_candidate_retained"
                                changed = True
                                df_rows.at[idx, "signal"] = "HOLD"
                                df_rows.at[idx, "signal_confirmation_reason"] = "better_active_stage_retained"
                                df_rows.at[idx, "volume_lots"] = 0.0
                                df_rows.at[idx, "risk_amount"] = 0.0
                                df_rows.at[idx, "allocated_risk_budget"] = 0.0
                                df_rows.at[idx, "entry_management_split_active"] = False
                                df_rows.at[idx, "initial_market_volume_lots"] = 0.0
                                df_rows.at[idx, "pending_order_volume_lots"] = 0.0
                                df_rows.at[idx, "pending_order_price"] = np.nan
                                df_rows.at[idx, "pending_order_type"] = pd.NA
                                df_rows.at[idx, "pending_order_sl_price"] = np.nan
                                df_rows.at[idx, "pending_order_tp_price"] = np.nan
                                df_rows.at[idx, "pending_order_expiry_time"] = pd.NA
                                df_rows.at[idx, "entry_management_comment"] = "better_active_stage_retained"
                                df_rows.at[idx, "staged_candidate_id"] = active_candidate.get("candidate_id")
                                df_rows.at[idx, "staged_status"] = "ACTIVE"
                                df_rows.at[idx, "staged_action"] = "REUSED_ACTIVE"
                                df_rows.at[idx, "staged_reason"] = "better_active_stage_retained"
                                df_rows.at[idx, "staged_reference_price"] = active_candidate.get("reference_price")
                                df_rows.at[idx, "staged_trigger_price"] = active_candidate.get("trigger_price")
                                df_rows.at[idx, "staged_breakout_trigger_price"] = active_candidate.get("breakout_trigger_price")
                                df_rows.at[idx, "staged_expires_at"] = active_candidate.get("expires_at")
                                df_rows.at[idx, "staged_entry_improvement_pips"] = active_candidate.get("entry_improvement_pips")
                                df_rows.at[idx, "staged_adaptive_profile"] = active_candidate.get("adaptive_profile")
                                self.logger.info(
                                    "Candidata activa retenida por mejor punto: model=%s side=%s active_trigger=%s new_trigger=%s",
                                    model,
                                    final_signal,
                                    active_candidate.get("trigger_price"),
                                    candidate_retrace["trigger_price"],
                                )
                                continue

                        if active_idx is not None:
                            staged.at[active_idx, "status"] = "SUPERSEDED_BY_FILTER_HOLD_RETRACE"
                            staged.at[active_idx, "cancel_timestamp"] = now_iso
                            staged.at[active_idx, "cancel_reason"] = "filter_hold_retrace_refresh"
                            staged.at[active_idx, "status_reason"] = "filter_hold_retrace_refresh"
                            staged.at[active_idx, "last_evaluated_at"] = now_iso
                            changed = True

                        strong_trend_partial_payload = None
                        immediate_partial_fraction = 0.0
                        immediate_partial_comment = "direct_confirmed_partial_market_only"
                        immediate_partial_reason = "direct_confirmed_partial_market_and_stage"
                        immediate_partial_staged_reason = "direct_confirmed_partial_wait_candle_retrace"
                        immediate_partial_suppressed_reason = None
                        candidate_entry_improvement_pips_raw = pd.to_numeric(
                            pd.Series([candidate_retrace.get("entry_improvement_pips")]),
                            errors="coerce",
                        ).iloc[0]
                        candidate_entry_improvement_pips = (
                            float(candidate_entry_improvement_pips_raw)
                            if pd.notna(candidate_entry_improvement_pips_raw)
                            else float("nan")
                        )
                        if (
                            direct_stage_mode in {"direct_confirmed_candle_retrace", "direct_filter_hold_candle_retrace"}
                            and bool(settings.get("near_trigger_immediate_partial_enabled", True))
                            and pd.notna(candidate_entry_improvement_pips)
                            and candidate_entry_improvement_pips
                            <= float(settings.get("near_trigger_immediate_partial_entry_improvement_pips_max", 1.6))
                            and pd.notna(primary_confidence)
                            and float(primary_confidence)
                            >= float(settings.get("near_trigger_immediate_partial_confidence_min", 0.75))
                            and pd.notna(predicted_pips)
                            and abs(float(predicted_pips))
                            >= float(settings.get("near_trigger_immediate_partial_predicted_pips_min", 3.8))
                        ):
                            immediate_partial_fraction = min(
                                max(float(settings.get("near_trigger_immediate_partial_fraction", 0.20)), 0.0),
                                1.0,
                            )
                            if direct_stage_mode == "direct_filter_hold_candle_retrace":
                                immediate_partial_comment = "near_trigger_filter_hold_micro_market_only"
                                immediate_partial_reason = "near_trigger_filter_hold_micro_market_and_stage"
                                immediate_partial_staged_reason = (
                                    "near_trigger_filter_hold_micro_wait_candle_retrace"
                                )
                            else:
                                immediate_partial_comment = "near_trigger_direct_confirmed_micro_market_only"
                                immediate_partial_reason = "near_trigger_direct_confirmed_micro_market_and_stage"
                                immediate_partial_staged_reason = (
                                    "near_trigger_direct_confirmed_micro_wait_candle_retrace"
                                )
                        if (
                            direct_stage_mode == "direct_confirmed_candle_retrace"
                            and direct_profile_name == "strong_trend"
                            and bool(settings.get("direct_confirmed_strong_trend_partial_enabled", True))
                            and pd.notna(primary_confidence)
                            and float(primary_confidence)
                            >= float(settings.get("direct_confirmed_strong_trend_confidence_min", 0.90))
                            and pd.notna(predicted_pips)
                            and abs(float(predicted_pips))
                            >= float(settings.get("direct_confirmed_strong_trend_predicted_pips_min", 6.0))
                        ):
                            immediate_partial_fraction = max(
                                immediate_partial_fraction,
                                min(
                                    max(
                                        float(settings.get("direct_confirmed_strong_trend_partial_fraction", 0.25)),
                                        0.0,
                                    ),
                                    1.0,
                                ),
                            )
                            immediate_partial_comment = "direct_confirmed_strong_trend_partial_market_only"
                            immediate_partial_reason = "direct_confirmed_strong_trend_partial_market_and_stage"
                            immediate_partial_staged_reason = (
                                "direct_confirmed_strong_trend_partial_wait_candle_retrace"
                            )
                        elif (
                            direct_stage_mode == "direct_confirmed_candle_retrace"
                            and direct_profile_name == "normal_trend"
                            and bool(settings.get("direct_confirmed_normal_trend_partial_enabled", True))
                            and pd.notna(primary_confidence)
                            and float(primary_confidence)
                            >= float(settings.get("direct_confirmed_normal_trend_confidence_min", 0.75))
                            and pd.notna(predicted_pips)
                            and abs(float(predicted_pips))
                            >= float(settings.get("direct_confirmed_normal_trend_predicted_pips_min", 4.0))
                        ):
                            immediate_partial_fraction = max(
                                immediate_partial_fraction,
                                min(
                                    max(
                                        float(settings.get("direct_confirmed_normal_trend_partial_fraction", 0.25)),
                                        0.0,
                                    ),
                                    1.0,
                                ),
                            )
                            immediate_partial_comment = "direct_confirmed_normal_trend_partial_market_only"
                            immediate_partial_reason = "direct_confirmed_normal_trend_partial_market_and_stage"
                            immediate_partial_staged_reason = (
                                "direct_confirmed_normal_trend_partial_wait_candle_retrace"
                            )
                        remaining_stage_fraction = 1.0
                        direct_confirmed_immediate_partial = immediate_partial_fraction > 0.0
                        if direct_confirmed_immediate_partial:
                            adverse_extreme_hit, entry_location_in_candle, candle_range_pips = (
                                self._is_immediate_entry_price_in_adverse_extreme(
                                    side=final_signal,
                                    price=float(candidate_retrace["reference_price"]),
                                    candle_high=candle_high,
                                    candle_low=candle_low,
                                    pip_size=pip_size,
                                    settings=settings,
                                )
                            )
                            if adverse_extreme_hit:
                                immediate_partial_fraction = 0.0
                                direct_confirmed_immediate_partial = False
                                immediate_partial_suppressed_reason = (
                                    "direct_confirmed_immediate_extreme_entry_wait_candle_retrace"
                                )
                                self.logger.info(
                                    "Pata inmediata suprimida por entrada en extremo adverso: model=%s side=%s profile=%s ref=%s loc=%.3f range_pips=%.2f",
                                    model,
                                    final_signal,
                                    direct_profile_name,
                                    candidate_retrace["reference_price"],
                                    float(entry_location_in_candle) if entry_location_in_candle is not None else float("nan"),
                                    float(candle_range_pips) if candle_range_pips is not None else float("nan"),
                                )
                        if (
                            direct_confirmed_immediate_partial
                            and bool(settings.get("execution_confirmation_m1_apply_on_near_trigger_partial", True))
                        ):
                            m1_execution_check = self._evaluate_entry_execution_confirmation(
                                signal=final_signal,
                                runtime_ctx=runtime_ctx,
                                settings=settings,
                                purpose="near_trigger_partial",
                            )
                            self._record_entry_execution_confirmation_details(
                                details=m1_execution_check,
                                df_rows=df_rows,
                                row_idx=idx,
                            )
                            if not bool(m1_execution_check["passed"]):
                                immediate_partial_fraction = 0.0
                                direct_confirmed_immediate_partial = False
                                immediate_partial_suppressed_reason = str(
                                    m1_execution_check.get("reason") or "m1_execution_confirmation_waiting"
                                )
                                self.logger.info(
                                    "Pata inmediata retenida por confirmacion M1: model=%s side=%s profile=%s reason=%s score=%s hits=%s/%s",
                                    model,
                                    final_signal,
                                    direct_profile_name,
                                    m1_execution_check.get("reason"),
                                    m1_execution_check.get("score"),
                                    m1_execution_check.get("hits"),
                                    m1_execution_check.get("total"),
                                )
                        if direct_confirmed_immediate_partial:
                            strong_trend_partial_payload = self._compute_runtime_trade_payload_for_signal(
                                signal=final_signal,
                                predicted_pips_signed=predicted_pips,
                                signal_time=timestamp if pd.notna(timestamp) else row.get("timestamp"),
                                runtime_ctx=runtime_ctx,
                                volume_scale=immediate_partial_fraction,
                                disable_entry_management=True,
                                explicit_sl_price=candidate_retrace["custom_stop_price"],
                                market_only_comment=immediate_partial_comment,
                            )
                            if strong_trend_partial_payload and float(
                                strong_trend_partial_payload.get("volume_lots") or 0.0
                            ) > 0:
                                remaining_stage_fraction = max(1.0 - immediate_partial_fraction, 0.0)
                                full_volume_lots_raw = pd.to_numeric(
                                    pd.Series([row.get("volume_lots")]),
                                    errors="coerce",
                                ).iloc[0]
                                market_volume_lots_raw = pd.to_numeric(
                                    pd.Series([strong_trend_partial_payload.get("volume_lots")]),
                                    errors="coerce",
                                ).iloc[0]
                                if (
                                    remaining_stage_fraction > 0.0
                                    and immediate_partial_comment.endswith("_market_only")
                                    and pd.notna(full_volume_lots_raw)
                                    and float(full_volume_lots_raw) > 0.0
                                    and pd.notna(market_volume_lots_raw)
                                    and float(market_volume_lots_raw) > 0.0
                                ):
                                    pending_variant_comment = immediate_partial_comment.replace(
                                        "_market_only",
                                        "_market_and_pending",
                                    )
                                    partial_entry_plan = self._build_immediate_partial_pending_entry_plan(
                                        signal=final_signal,
                                        total_volume_lots=float(full_volume_lots_raw),
                                        market_volume_lots=float(market_volume_lots_raw),
                                        live_entry_price=float(strong_trend_partial_payload["live_entry_price"]),
                                        live_sl_price=float(strong_trend_partial_payload["live_sl_price"]),
                                        live_tp_price=float(strong_trend_partial_payload["live_tp_price"]),
                                        digits=int(max(runtime_ctx.get("digits", 5) or 5, 0)),
                                        min_lot=float(runtime_ctx.get("min_lot") or 0.01),
                                        lot_step=float(runtime_ctx.get("lot_step") or runtime_ctx.get("min_lot") or 0.01),
                                        timeframe=timeframe,
                                        signal_time=timestamp if pd.notna(timestamp) else row.get("timestamp"),
                                        market_only_comment=immediate_partial_comment,
                                        pending_comment=pending_variant_comment,
                                    )
                                    partial_pending_volume = pd.to_numeric(
                                        pd.Series([partial_entry_plan.get("pending_order_volume_lots")]),
                                        errors="coerce",
                                    ).iloc[0]
                                    if (
                                        self._coerce_boolish(partial_entry_plan.get("entry_management_split_active"))
                                        and pd.notna(partial_pending_volume)
                                        and float(partial_pending_volume) > 0.0
                                    ):
                                        strong_trend_partial_payload["entry_management_plan"] = partial_entry_plan
                                        immediate_partial_comment = pending_variant_comment
                                        immediate_partial_reason = pending_variant_comment
                                        remaining_stage_fraction = 0.0
                                        self.logger.info(
                                            "Entrada inmediata con pending real: model=%s side=%s market_lots=%.2f pending_lots=%.2f pending_price=%s",
                                            model,
                                            final_signal,
                                            float(market_volume_lots_raw),
                                            float(partial_pending_volume),
                                            partial_entry_plan.get("pending_order_price"),
                                        )
                            else:
                                strong_trend_partial_payload = None
                                remaining_stage_fraction = 1.0
                        breakout_trigger_price = (
                            self._compute_breakout_trigger_price(
                                side=final_signal,
                                reference_price=candidate_retrace["reference_price"],
                                reference_stop_pips=candidate_retrace["reference_stop_pips"],
                                pip_size=pip_size,
                                settings=settings,
                            )
                            if float(adaptive_profile["breakout_partial_fraction"]) > 0
                            else float("nan")
                        )
                        expires_at = (
                            timestamp + timeframe_delta * int(adaptive_profile["max_stage_bars"])
                            if pd.notna(timestamp)
                            else pd.NaT
                        )
                        candidate_suffix = "FH_STAGE" if direct_stage_mode == "direct_filter_hold_candle_retrace" else "DC_STAGE"
                        candidate_id = f"{self._build_signal_id(row)}|{candidate_suffix}"
                        if remaining_stage_fraction > 0:
                            staged = pd.concat(
                                [
                                    staged,
                                    pd.DataFrame(
                                        [
                                            {
                                                "candidate_id": candidate_id,
                                                "parent_signal_id": self._build_signal_id(row),
                                                "release_id": row.get("release_id"),
                                                "strategy_profile": strategy_profile,
                                                "symbol": symbol,
                                                "timeframe": timeframe,
                                                "model": model,
                                                "side": final_signal,
                                                "created_at": now_iso,
                                                "source_timestamp": timestamp.isoformat() if pd.notna(timestamp) else row.get("timestamp"),
                                                "expires_at": expires_at.isoformat() if pd.notna(expires_at) else pd.NA,
                                                "reference_price": candidate_retrace["reference_price"],
                                                "trigger_price": candidate_retrace["trigger_price"],
                                                "breakout_trigger_price": breakout_trigger_price,
                                                "reference_stop_pips": candidate_retrace["reference_stop_pips"],
                                                "entry_improvement_pips": candidate_retrace["entry_improvement_pips"],
                                                "predicted_pips": predicted_pips,
                                                "pred_return": row.get("pred_return"),
                                                "candidate_mode": direct_stage_mode,
                                                "candidate_volume_scale": remaining_stage_fraction,
                                                "adaptive_profile": adaptive_profile["profile"],
                                                "adaptive_retrace_fraction": adaptive_profile["retrace_fraction"],
                                                "adaptive_breakout_fraction": adaptive_profile["breakout_partial_fraction"],
                                                "signal_candle_open": candle_open,
                                                "signal_candle_high": candle_high,
                                                "signal_candle_low": candle_low,
                                                "signal_candle_close": candle_close,
                                                "custom_stop_price": candidate_retrace["custom_stop_price"],
                                                "primary_confidence": primary_confidence,
                                                "filter_signal": filter_signal,
                                                "filter_support_score": support_score,
                                                "filter_contradicted": filter_contradicted,
                                                "status": "ACTIVE",
                                                "status_reason": "created_filter_hold_retrace",
                                                "last_evaluated_at": now_iso,
                                                "hold_grace_count": 0,
                                                "activation_timestamp": pd.NA,
                                                "activation_price": np.nan,
                                                "activation_reason": pd.NA,
                                                "cancel_timestamp": pd.NA,
                                                "cancel_reason": pd.NA,
                                            }
                                        ]
                                    ),
                                ],
                                ignore_index=True,
                            )
                            changed = True

                        if strong_trend_partial_payload and float(
                            strong_trend_partial_payload.get("volume_lots") or 0.0
                        ) > 0:
                            df_rows.at[idx, "signal"] = final_signal
                            df_rows.at[idx, "confidence"] = (
                                primary_confidence if pd.notna(primary_confidence) else row.get("confidence")
                            )
                            df_rows.at[idx, "signal_confirmation_passed"] = True
                            df_rows.at[idx, "signal_confirmation_reason"] = (
                                immediate_partial_reason
                                if remaining_stage_fraction > 0
                                else immediate_partial_comment
                            )
                            df_rows.at[idx, "entry_price"] = strong_trend_partial_payload["entry_price"]
                            df_rows.at[idx, "planned_entry_price"] = strong_trend_partial_payload["planned_entry_price"]
                            df_rows.at[idx, "price_target"] = strong_trend_partial_payload["tp_price"]
                            df_rows.at[idx, "delta_price"] = (
                                strong_trend_partial_payload["tp_price"] - current_price
                                if final_signal == "BUY"
                                else current_price - strong_trend_partial_payload["tp_price"]
                            )
                            df_rows.at[idx, "sl_price"] = strong_trend_partial_payload["sl_price"]
                            df_rows.at[idx, "tp_price"] = strong_trend_partial_payload["tp_price"]
                            df_rows.at[idx, "sl_pips"] = strong_trend_partial_payload["sl_pips"]
                            df_rows.at[idx, "tp_pips"] = strong_trend_partial_payload["tp_pips"]
                            df_rows.at[idx, "market_reference_price"] = strong_trend_partial_payload["market_reference_price"]
                            df_rows.at[idx, "live_entry_price"] = strong_trend_partial_payload["live_entry_price"]
                            df_rows.at[idx, "live_sl_price"] = strong_trend_partial_payload["live_sl_price"]
                            df_rows.at[idx, "live_tp_price"] = strong_trend_partial_payload["live_tp_price"]
                            df_rows.at[idx, "live_sl_pips"] = strong_trend_partial_payload["live_sl_pips"]
                            df_rows.at[idx, "live_tp_pips"] = strong_trend_partial_payload["live_tp_pips"]
                            df_rows.at[idx, "volume_lots"] = strong_trend_partial_payload["volume_lots"]
                            df_rows.at[idx, "risk_amount"] = strong_trend_partial_payload["risk_amount"]
                            df_rows.at[idx, "allocated_risk_budget"] = strong_trend_partial_payload["allocated_risk_budget"]
                            df_rows.at[idx, "risk_per_pip_per_lot"] = strong_trend_partial_payload["risk_per_pip_per_lot"]
                            df_rows.at[idx, "risk_per_lot_at_stop"] = strong_trend_partial_payload["risk_per_lot_at_stop"]
                            df_rows.at[idx, "remaining_risk_budget_before_trade"] = strong_trend_partial_payload[
                                "remaining_risk_budget_before_trade"
                            ]
                            df_rows.at[idx, "projected_total_open_risk_after_trade"] = strong_trend_partial_payload[
                                "projected_total_open_risk_after_trade"
                            ]
                            partial_entry_plan = strong_trend_partial_payload["entry_management_plan"]
                            df_rows.at[idx, "entry_management_mode"] = partial_entry_plan["entry_management_mode"]
                            df_rows.at[idx, "entry_management_split_active"] = partial_entry_plan[
                                "entry_management_split_active"
                            ]
                            df_rows.at[idx, "entry_management_initial_market_fraction"] = partial_entry_plan[
                                "entry_management_initial_market_fraction"
                            ]
                            df_rows.at[idx, "entry_management_pending_fraction"] = partial_entry_plan[
                                "entry_management_pending_fraction"
                            ]
                            df_rows.at[idx, "entry_management_retrace_fraction_of_stop"] = partial_entry_plan[
                                "entry_management_retrace_fraction_of_stop"
                            ]
                            df_rows.at[idx, "entry_management_total_volume_lots"] = partial_entry_plan[
                                "entry_management_total_volume_lots"
                            ]
                            df_rows.at[idx, "initial_market_volume_lots"] = partial_entry_plan[
                                "initial_market_volume_lots"
                            ]
                            df_rows.at[idx, "pending_order_volume_lots"] = partial_entry_plan[
                                "pending_order_volume_lots"
                            ]
                            df_rows.at[idx, "pending_order_price"] = partial_entry_plan["pending_order_price"]
                            df_rows.at[idx, "pending_order_type"] = partial_entry_plan["pending_order_type"]
                            df_rows.at[idx, "pending_order_sl_price"] = partial_entry_plan["pending_order_sl_price"]
                            df_rows.at[idx, "pending_order_tp_price"] = partial_entry_plan["pending_order_tp_price"]
                            df_rows.at[idx, "pending_order_expiry_time"] = partial_entry_plan[
                                "pending_order_expiry_time"
                            ]
                            df_rows.at[idx, "entry_management_comment"] = partial_entry_plan[
                                "entry_management_comment"
                            ]
                            df_rows.at[idx, "staged_candidate_id"] = candidate_id if remaining_stage_fraction > 0 else pd.NA
                            df_rows.at[idx, "staged_status"] = "ACTIVE" if remaining_stage_fraction > 0 else pd.NA
                            df_rows.at[idx, "staged_action"] = (
                                "CREATED_WITH_IMMEDIATE_PARTIAL" if remaining_stage_fraction > 0 else pd.NA
                            )
                            df_rows.at[idx, "staged_reason"] = (
                                immediate_partial_staged_reason
                                if remaining_stage_fraction > 0
                                else pd.NA
                            )
                            df_rows.at[idx, "staged_reference_price"] = (
                                candidate_retrace["reference_price"] if remaining_stage_fraction > 0 else np.nan
                            )
                            df_rows.at[idx, "staged_trigger_price"] = (
                                candidate_retrace["trigger_price"] if remaining_stage_fraction > 0 else np.nan
                            )
                            df_rows.at[idx, "staged_breakout_trigger_price"] = (
                                breakout_trigger_price if remaining_stage_fraction > 0 else np.nan
                            )
                            df_rows.at[idx, "staged_expires_at"] = (
                                expires_at.isoformat() if remaining_stage_fraction > 0 and pd.notna(expires_at) else pd.NA
                            )
                            df_rows.at[idx, "staged_entry_improvement_pips"] = (
                                candidate_retrace["entry_improvement_pips"] if remaining_stage_fraction > 0 else np.nan
                            )
                            df_rows.at[idx, "staged_adaptive_profile"] = (
                                adaptive_profile["profile"] if remaining_stage_fraction > 0 else pd.NA
                            )
                            self.logger.info(
                                "âš¡ Entrada parcial inmediata confirmada: model=%s side=%s profile=%s market_fraction=%.2f staged_fraction=%.2f close=%s trigger=%s stop=%s",
                                model,
                                final_signal,
                                direct_profile_name,
                                immediate_partial_fraction,
                                remaining_stage_fraction,
                                candle_close,
                                candidate_retrace["trigger_price"],
                                candidate_retrace["custom_stop_price"],
                            )
                        else:
                            df_rows.at[idx, "signal"] = "HOLD"
                            df_rows.at[idx, "signal_confirmation_reason"] = (
                                immediate_partial_suppressed_reason or direct_stage_reason
                            )
                            df_rows.at[idx, "volume_lots"] = 0.0
                            df_rows.at[idx, "risk_amount"] = 0.0
                            df_rows.at[idx, "allocated_risk_budget"] = 0.0
                            df_rows.at[idx, "entry_management_split_active"] = False
                            df_rows.at[idx, "initial_market_volume_lots"] = 0.0
                            df_rows.at[idx, "pending_order_volume_lots"] = 0.0
                            df_rows.at[idx, "pending_order_price"] = np.nan
                            df_rows.at[idx, "pending_order_type"] = pd.NA
                            df_rows.at[idx, "pending_order_sl_price"] = np.nan
                            df_rows.at[idx, "pending_order_tp_price"] = np.nan
                            df_rows.at[idx, "pending_order_expiry_time"] = pd.NA
                            df_rows.at[idx, "entry_management_comment"] = (
                                immediate_partial_suppressed_reason or direct_stage_reason
                            )
                            df_rows.at[idx, "staged_candidate_id"] = candidate_id
                            df_rows.at[idx, "staged_status"] = "ACTIVE"
                            df_rows.at[idx, "staged_action"] = "CREATED"
                            df_rows.at[idx, "staged_reason"] = (
                                immediate_partial_suppressed_reason or direct_stage_reason
                            )
                            df_rows.at[idx, "staged_reference_price"] = candidate_retrace["reference_price"]
                            df_rows.at[idx, "staged_trigger_price"] = candidate_retrace["trigger_price"]
                            df_rows.at[idx, "staged_breakout_trigger_price"] = breakout_trigger_price
                            df_rows.at[idx, "staged_expires_at"] = expires_at.isoformat() if pd.notna(expires_at) else pd.NA
                            df_rows.at[idx, "staged_entry_improvement_pips"] = candidate_retrace["entry_improvement_pips"]
                            df_rows.at[idx, "staged_adaptive_profile"] = adaptive_profile["profile"]
                            self.logger.info(
                                "ðŸ•’ SeÃ±al directa retenida por staging: model=%s side=%s close=%s trigger=%s stop=%s",
                                model,
                                final_signal,
                                candle_close,
                                candidate_retrace["trigger_price"],
                                candidate_retrace["custom_stop_price"],
                            )
                        continue

                if active_idx is not None:
                    staged.at[active_idx, "status"] = "SUPERSEDED_BY_DIRECT_SIGNAL"
                    staged.at[active_idx, "cancel_timestamp"] = now_iso
                    staged.at[active_idx, "cancel_reason"] = "direct_bundle_signal"
                    staged.at[active_idx, "status_reason"] = "direct_bundle_signal"
                    staged.at[active_idx, "last_evaluated_at"] = now_iso
                    changed = True
                    df_rows.at[idx, "staged_candidate_id"] = active_candidate.get("candidate_id")
                    df_rows.at[idx, "staged_status"] = "SUPERSEDED_BY_DIRECT_SIGNAL"
                    df_rows.at[idx, "staged_action"] = "DIRECT_SIGNAL"
                    df_rows.at[idx, "staged_reason"] = "bundle_direct_signal"
                continue

            if (
                final_signal == "HOLD"
                and primary_signal in {"BUY", "SELL"}
                and early_reversal_stage_check["eligible"]
                and pip_size > 0
                and pd.notna(candle_close)
            ):
                if active_idx is not None:
                    staged.at[active_idx, "status"] = "SUPERSEDED_BY_EARLY_STRUCTURAL_REVERSAL_RETRACE"
                    staged.at[active_idx, "cancel_timestamp"] = now_iso
                    staged.at[active_idx, "cancel_reason"] = "early_structural_reversal_retrace_refresh"
                    staged.at[active_idx, "status_reason"] = "early_structural_reversal_retrace_refresh"
                    staged.at[active_idx, "last_evaluated_at"] = now_iso
                    changed = True

                reference_price = _current_exec_price_for(primary_signal)
                candidate_retrace = self._build_candle_retrace_candidate(
                    side=primary_signal,
                    candle_high=candle_high,
                    candle_low=candle_low,
                    candle_close=candle_close,
                    reference_price=reference_price,
                    atr_value=runtime_ctx.get("atr_value"),
                    pip_size=pip_size,
                    digits=int(max(runtime_ctx.get("digits", 5), 0)),
                    retrace_fraction=float(early_reversal_stage_check["retrace_fraction"]),
                    stop_buffer_pips=float(settings["direct_filter_hold_stop_buffer_pips"]),
                    stop_buffer_atr_fraction=float(settings["dynamic_stop_atr_fraction"]),
                    min_stop_pips=float(settings["dynamic_stop_min_pips"]),
                )
                if candidate_retrace is not None and pd.notna(reference_price):
                    breakout_trigger_price = (
                        self._compute_breakout_trigger_price(
                            side=primary_signal,
                            reference_price=candidate_retrace["reference_price"],
                            reference_stop_pips=candidate_retrace["reference_stop_pips"],
                            pip_size=pip_size,
                            settings=settings,
                        )
                        if float(early_reversal_stage_check["breakout_partial_fraction"]) > 0
                        else float("nan")
                    )
                    expires_at = (
                        timestamp + timeframe_delta * int(early_reversal_stage_check["max_stage_bars"])
                        if pd.notna(timestamp)
                        else pd.NaT
                    )
                    candidate_id = f"{self._build_signal_id(row)}|ESR_STAGE"
                    staged = pd.concat(
                        [
                            staged,
                            pd.DataFrame(
                                [
                                    {
                                        "candidate_id": candidate_id,
                                        "parent_signal_id": self._build_signal_id(row),
                                        "release_id": row.get("release_id"),
                                        "strategy_profile": strategy_profile,
                                        "symbol": symbol,
                                        "timeframe": timeframe,
                                        "model": model,
                                        "side": primary_signal,
                                        "created_at": now_iso,
                                        "source_timestamp": timestamp.isoformat() if pd.notna(timestamp) else row.get("timestamp"),
                                        "expires_at": expires_at.isoformat() if pd.notna(expires_at) else pd.NA,
                                        "reference_price": candidate_retrace["reference_price"],
                                        "trigger_price": candidate_retrace["trigger_price"],
                                        "breakout_trigger_price": breakout_trigger_price,
                                        "reference_stop_pips": candidate_retrace["reference_stop_pips"],
                                        "entry_improvement_pips": candidate_retrace["entry_improvement_pips"],
                                        "predicted_pips": predicted_pips,
                                        "pred_return": row.get("pred_return"),
                                        "candidate_mode": "early_structural_reversal_candle_retrace",
                                        "candidate_volume_scale": 1.0,
                                        "adaptive_profile": early_reversal_stage_check["profile"],
                                        "adaptive_retrace_fraction": early_reversal_stage_check["retrace_fraction"],
                                        "adaptive_breakout_fraction": early_reversal_stage_check["breakout_partial_fraction"],
                                        "signal_candle_open": candle_open,
                                        "signal_candle_high": candle_high,
                                        "signal_candle_low": candle_low,
                                        "signal_candle_close": candle_close,
                                        "custom_stop_price": candidate_retrace["custom_stop_price"],
                                        "primary_confidence": primary_confidence,
                                        "filter_signal": filter_signal,
                                        "filter_support_score": support_score,
                                        "filter_contradicted": filter_contradicted,
                                        "status": "ACTIVE",
                                        "status_reason": "created_early_structural_reversal_retrace",
                                        "last_evaluated_at": now_iso,
                                        "hold_grace_count": 0,
                                        "activation_timestamp": pd.NA,
                                        "activation_price": np.nan,
                                        "activation_reason": pd.NA,
                                        "cancel_timestamp": pd.NA,
                                        "cancel_reason": pd.NA,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
                    changed = True
                    df_rows.at[idx, "signal"] = "HOLD"
                    df_rows.at[idx, "signal_confirmation_reason"] = "early_structural_reversal_wait_candle_retrace"
                    df_rows.at[idx, "volume_lots"] = 0.0
                    df_rows.at[idx, "risk_amount"] = 0.0
                    df_rows.at[idx, "allocated_risk_budget"] = 0.0
                    df_rows.at[idx, "entry_management_split_active"] = False
                    df_rows.at[idx, "initial_market_volume_lots"] = 0.0
                    df_rows.at[idx, "pending_order_volume_lots"] = 0.0
                    df_rows.at[idx, "pending_order_price"] = np.nan
                    df_rows.at[idx, "pending_order_type"] = pd.NA
                    df_rows.at[idx, "pending_order_sl_price"] = np.nan
                    df_rows.at[idx, "pending_order_tp_price"] = np.nan
                    df_rows.at[idx, "pending_order_expiry_time"] = pd.NA
                    df_rows.at[idx, "entry_management_comment"] = "early_structural_reversal_wait_candle_retrace"
                    df_rows.at[idx, "staged_candidate_id"] = candidate_id
                    df_rows.at[idx, "staged_status"] = "ACTIVE"
                    df_rows.at[idx, "staged_action"] = "CREATED"
                    df_rows.at[idx, "staged_reason"] = "early_structural_reversal_wait_candle_retrace"
                    df_rows.at[idx, "staged_reference_price"] = candidate_retrace["reference_price"]
                    df_rows.at[idx, "staged_trigger_price"] = candidate_retrace["trigger_price"]
                    df_rows.at[idx, "staged_breakout_trigger_price"] = breakout_trigger_price
                    df_rows.at[idx, "staged_expires_at"] = expires_at.isoformat() if pd.notna(expires_at) else pd.NA
                    df_rows.at[idx, "staged_entry_improvement_pips"] = candidate_retrace["entry_improvement_pips"]
                    df_rows.at[idx, "staged_adaptive_profile"] = early_reversal_stage_check["profile"]
                    self.logger.info(
                        "Reversion estructural temprana a staging: model=%s side=%s close=%s trigger=%s stop=%s",
                        model,
                        primary_signal,
                        candle_close,
                        candidate_retrace["trigger_price"],
                        candidate_retrace["custom_stop_price"],
                    )
                continue

            if (
                final_signal == "HOLD"
                and primary_signal == "HOLD"
                and filter_signal in {"BUY", "SELL"}
                and filter_lead_structural_check["eligible"]
                and pip_size > 0
                and pd.notna(candle_close)
            ):
                candidate_side = filter_signal
                reference_price = _current_exec_price_for(candidate_side)
                candidate_retrace = self._build_candle_retrace_candidate(
                    side=candidate_side,
                    candle_high=candle_high,
                    candle_low=candle_low,
                    candle_close=candle_close,
                    reference_price=reference_price,
                    atr_value=runtime_ctx.get("atr_value"),
                    pip_size=pip_size,
                    digits=int(max(runtime_ctx.get("digits", 5), 0)),
                    retrace_fraction=float(filter_lead_structural_check["retrace_fraction"]),
                    stop_buffer_pips=float(settings["direct_filter_hold_stop_buffer_pips"]),
                    stop_buffer_atr_fraction=float(settings["dynamic_stop_atr_fraction"]),
                    min_stop_pips=float(settings["dynamic_stop_min_pips"]),
                )
                if candidate_retrace is not None and pd.notna(reference_price):
                    if (
                        active_idx is not None
                        and str(active_candidate.get("candidate_mode", "") or "").strip().lower()
                        == "filter_lead_structural_candle_retrace"
                        and str(active_candidate.get("side", "") or "").upper() == candidate_side
                    ):
                        active_compare_price = self._candidate_comparison_price(active_candidate)
                        new_compare_price = self._candidate_comparison_price(candidate_retrace)
                        if self._is_more_favorable_entry_price(
                            side=candidate_side,
                            candidate_price=active_compare_price,
                            reference_price=new_compare_price,
                        ):
                            staged.at[active_idx, "last_evaluated_at"] = now_iso
                            staged.at[active_idx, "status_reason"] = "better_filter_lead_structural_candidate_retained"
                            changed = True
                            df_rows.at[idx, "staged_candidate_id"] = active_candidate.get("candidate_id")
                            df_rows.at[idx, "staged_status"] = "ACTIVE"
                            df_rows.at[idx, "staged_action"] = "REUSED_ACTIVE"
                            df_rows.at[idx, "staged_reason"] = "better_filter_lead_structural_candidate_retained"
                            df_rows.at[idx, "staged_reference_price"] = active_candidate.get("reference_price")
                            df_rows.at[idx, "staged_trigger_price"] = active_candidate.get("trigger_price")
                            df_rows.at[idx, "staged_breakout_trigger_price"] = active_candidate.get("breakout_trigger_price")
                            df_rows.at[idx, "staged_expires_at"] = active_candidate.get("expires_at")
                            self.logger.info(
                                "Filter lead structural mantiene mejor candidata activa: model=%s side=%s active=%s new=%s",
                                model,
                                candidate_side,
                                active_compare_price,
                                new_compare_price,
                            )
                            continue
                    if active_idx is not None:
                        staged.at[active_idx, "status"] = "SUPERSEDED_BY_FILTER_LEAD_STRUCTURAL_RETRACE"
                        staged.at[active_idx, "cancel_timestamp"] = now_iso
                        staged.at[active_idx, "cancel_reason"] = "filter_lead_structural_retrace_refresh"
                        staged.at[active_idx, "status_reason"] = "filter_lead_structural_retrace_refresh"
                        staged.at[active_idx, "last_evaluated_at"] = now_iso
                        changed = True

                    breakout_trigger_price = (
                        self._compute_breakout_trigger_price(
                            side=candidate_side,
                            reference_price=candidate_retrace["reference_price"],
                            reference_stop_pips=candidate_retrace["reference_stop_pips"],
                            pip_size=pip_size,
                            settings=settings,
                        )
                        if float(filter_lead_structural_check["breakout_partial_fraction"]) > 0
                        else float("nan")
                    )
                    expires_at = (
                        timestamp + timeframe_delta * int(filter_lead_structural_check["max_stage_bars"])
                        if pd.notna(timestamp)
                        else pd.NaT
                    )
                    candidate_id = f"{self._build_signal_id(row)}|FLS_STAGE"
                    staged = pd.concat(
                        [
                            staged,
                            pd.DataFrame(
                                [
                                    {
                                        "candidate_id": candidate_id,
                                        "parent_signal_id": self._build_signal_id(row),
                                        "release_id": row.get("release_id"),
                                        "strategy_profile": strategy_profile,
                                        "symbol": symbol,
                                        "timeframe": timeframe,
                                        "model": model,
                                        "side": candidate_side,
                                        "created_at": now_iso,
                                        "source_timestamp": timestamp.isoformat() if pd.notna(timestamp) else row.get("timestamp"),
                                        "expires_at": expires_at.isoformat() if pd.notna(expires_at) else pd.NA,
                                        "reference_price": candidate_retrace["reference_price"],
                                        "trigger_price": candidate_retrace["trigger_price"],
                                        "breakout_trigger_price": breakout_trigger_price,
                                        "reference_stop_pips": candidate_retrace["reference_stop_pips"],
                                        "entry_improvement_pips": candidate_retrace["entry_improvement_pips"],
                                        "predicted_pips": filter_lead_structural_check["stage_predicted_pips"],
                                        "pred_return": row.get("pred_return"),
                                        "candidate_mode": "filter_lead_structural_candle_retrace",
                                        "candidate_volume_scale": 1.0,
                                        "adaptive_profile": filter_lead_structural_check["profile"],
                                        "adaptive_retrace_fraction": filter_lead_structural_check["retrace_fraction"],
                                        "adaptive_breakout_fraction": filter_lead_structural_check["breakout_partial_fraction"],
                                        "signal_candle_open": candle_open,
                                        "signal_candle_high": candle_high,
                                        "signal_candle_low": candle_low,
                                        "signal_candle_close": candle_close,
                                        "custom_stop_price": candidate_retrace["custom_stop_price"],
                                        "primary_confidence": primary_confidence,
                                        "filter_signal": filter_signal,
                                        "filter_support_score": support_score,
                                        "filter_contradicted": filter_contradicted,
                                        "status": "ACTIVE",
                                        "status_reason": "created_filter_lead_structural_retrace",
                                        "last_evaluated_at": now_iso,
                                        "hold_grace_count": 0,
                                        "activation_timestamp": pd.NA,
                                        "activation_price": np.nan,
                                        "activation_reason": pd.NA,
                                        "cancel_timestamp": pd.NA,
                                        "cancel_reason": pd.NA,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
                    changed = True
                    df_rows.at[idx, "signal"] = "HOLD"
                    df_rows.at[idx, "signal_confirmation_reason"] = "filter_lead_structural_wait_candle_retrace"
                    df_rows.at[idx, "volume_lots"] = 0.0
                    df_rows.at[idx, "risk_amount"] = 0.0
                    df_rows.at[idx, "allocated_risk_budget"] = 0.0
                    df_rows.at[idx, "entry_management_split_active"] = False
                    df_rows.at[idx, "initial_market_volume_lots"] = 0.0
                    df_rows.at[idx, "pending_order_volume_lots"] = 0.0
                    df_rows.at[idx, "pending_order_price"] = np.nan
                    df_rows.at[idx, "pending_order_type"] = pd.NA
                    df_rows.at[idx, "pending_order_sl_price"] = np.nan
                    df_rows.at[idx, "pending_order_tp_price"] = np.nan
                    df_rows.at[idx, "pending_order_expiry_time"] = pd.NA
                    df_rows.at[idx, "entry_management_comment"] = "filter_lead_structural_wait_candle_retrace"
                    df_rows.at[idx, "staged_candidate_id"] = candidate_id
                    df_rows.at[idx, "staged_status"] = "ACTIVE"
                    df_rows.at[idx, "staged_action"] = "CREATED"
                    df_rows.at[idx, "staged_reason"] = "filter_lead_structural_wait_candle_retrace"
                    df_rows.at[idx, "staged_reference_price"] = candidate_retrace["reference_price"]
                    df_rows.at[idx, "staged_trigger_price"] = candidate_retrace["trigger_price"]
                    df_rows.at[idx, "staged_breakout_trigger_price"] = breakout_trigger_price
                    df_rows.at[idx, "staged_expires_at"] = expires_at.isoformat() if pd.notna(expires_at) else pd.NA
                    df_rows.at[idx, "staged_entry_improvement_pips"] = candidate_retrace["entry_improvement_pips"]
                    df_rows.at[idx, "staged_adaptive_profile"] = filter_lead_structural_check["profile"]
                    self.logger.info(
                        "Filter lead structural a staging: model=%s side=%s close=%s trigger=%s stop=%s filter_conf=%.3f primary_conf=%.3f",
                        model,
                        candidate_side,
                        candle_close,
                        candidate_retrace["trigger_price"],
                        candidate_retrace["custom_stop_price"],
                        float(filter_confidence) if pd.notna(filter_confidence) else float("nan"),
                        float(primary_confidence) if pd.notna(primary_confidence) else float("nan"),
                    )
                continue

            if (
                final_signal == "HOLD"
                and primary_signal in {"BUY", "SELL"}
                and strong_primary_hold_check["eligible"]
                and pip_size > 0
                and pd.notna(candle_close)
            ):
                if active_idx is not None:
                    staged.at[active_idx, "status"] = "SUPERSEDED_BY_STRONG_PRIMARY_HOLD_RETRACE"
                    staged.at[active_idx, "cancel_timestamp"] = now_iso
                    staged.at[active_idx, "cancel_reason"] = "strong_primary_hold_retrace_refresh"
                    staged.at[active_idx, "status_reason"] = "strong_primary_hold_retrace_refresh"
                    staged.at[active_idx, "last_evaluated_at"] = now_iso
                    changed = True

                reference_price = _current_exec_price_for(primary_signal)
                candidate_retrace = self._build_candle_retrace_candidate(
                    side=primary_signal,
                    candle_high=candle_high,
                    candle_low=candle_low,
                    candle_close=candle_close,
                    reference_price=reference_price,
                    atr_value=runtime_ctx.get("atr_value"),
                    pip_size=pip_size,
                    digits=int(max(runtime_ctx.get("digits", 5), 0)),
                    retrace_fraction=float(strong_primary_hold_check["retrace_fraction"]),
                    stop_buffer_pips=float(settings["direct_filter_hold_stop_buffer_pips"]),
                    stop_buffer_atr_fraction=float(settings["dynamic_stop_atr_fraction"]),
                    min_stop_pips=float(settings["dynamic_stop_min_pips"]),
                )
                if candidate_retrace is not None and pd.notna(reference_price):
                    breakout_trigger_price = (
                        self._compute_breakout_trigger_price(
                            side=primary_signal,
                            reference_price=candidate_retrace["reference_price"],
                            reference_stop_pips=candidate_retrace["reference_stop_pips"],
                            pip_size=pip_size,
                            settings=settings,
                        )
                        if float(strong_primary_hold_check["breakout_partial_fraction"]) > 0
                        else float("nan")
                    )
                    expires_at = (
                        timestamp + timeframe_delta * int(strong_primary_hold_check["max_stage_bars"])
                        if pd.notna(timestamp)
                        else pd.NaT
                    )
                    candidate_id = f"{self._build_signal_id(row)}|SPH_STAGE"
                    staged = pd.concat(
                        [
                            staged,
                            pd.DataFrame(
                                [
                                    {
                                        "candidate_id": candidate_id,
                                        "parent_signal_id": self._build_signal_id(row),
                                        "release_id": row.get("release_id"),
                                        "strategy_profile": strategy_profile,
                                        "symbol": symbol,
                                        "timeframe": timeframe,
                                        "model": model,
                                        "side": primary_signal,
                                        "created_at": now_iso,
                                        "source_timestamp": timestamp.isoformat() if pd.notna(timestamp) else row.get("timestamp"),
                                        "expires_at": expires_at.isoformat() if pd.notna(expires_at) else pd.NA,
                                        "reference_price": candidate_retrace["reference_price"],
                                        "trigger_price": candidate_retrace["trigger_price"],
                                        "breakout_trigger_price": breakout_trigger_price,
                                        "reference_stop_pips": candidate_retrace["reference_stop_pips"],
                                        "entry_improvement_pips": candidate_retrace["entry_improvement_pips"],
                                        "predicted_pips": predicted_pips,
                                        "pred_return": row.get("pred_return"),
                                        "candidate_mode": "strong_primary_filter_hold_candle_retrace",
                                        "candidate_volume_scale": 1.0,
                                        "adaptive_profile": strong_primary_hold_check["profile"],
                                        "adaptive_retrace_fraction": strong_primary_hold_check["retrace_fraction"],
                                        "adaptive_breakout_fraction": strong_primary_hold_check["breakout_partial_fraction"],
                                        "signal_candle_open": candle_open,
                                        "signal_candle_high": candle_high,
                                        "signal_candle_low": candle_low,
                                        "signal_candle_close": candle_close,
                                        "custom_stop_price": candidate_retrace["custom_stop_price"],
                                        "primary_confidence": primary_confidence,
                                        "filter_signal": filter_signal,
                                        "filter_support_score": support_score,
                                        "filter_contradicted": filter_contradicted,
                                        "status": "ACTIVE",
                                        "status_reason": "created_strong_primary_filter_hold_retrace",
                                        "last_evaluated_at": now_iso,
                                        "activation_timestamp": pd.NA,
                                        "activation_price": np.nan,
                                        "activation_reason": pd.NA,
                                        "cancel_timestamp": pd.NA,
                                        "cancel_reason": pd.NA,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
                    changed = True
                    df_rows.at[idx, "signal"] = "HOLD"
                    df_rows.at[idx, "signal_confirmation_reason"] = "strong_primary_filter_hold_wait_candle_retrace"
                    df_rows.at[idx, "volume_lots"] = 0.0
                    df_rows.at[idx, "risk_amount"] = 0.0
                    df_rows.at[idx, "allocated_risk_budget"] = 0.0
                    df_rows.at[idx, "entry_management_split_active"] = False
                    df_rows.at[idx, "initial_market_volume_lots"] = 0.0
                    df_rows.at[idx, "pending_order_volume_lots"] = 0.0
                    df_rows.at[idx, "pending_order_price"] = np.nan
                    df_rows.at[idx, "pending_order_type"] = pd.NA
                    df_rows.at[idx, "pending_order_sl_price"] = np.nan
                    df_rows.at[idx, "pending_order_tp_price"] = np.nan
                    df_rows.at[idx, "pending_order_expiry_time"] = pd.NA
                    df_rows.at[idx, "entry_management_comment"] = "strong_primary_filter_hold_wait_candle_retrace"
                    df_rows.at[idx, "staged_candidate_id"] = candidate_id
                    df_rows.at[idx, "staged_status"] = "ACTIVE"
                    df_rows.at[idx, "staged_action"] = "CREATED"
                    df_rows.at[idx, "staged_reason"] = "strong_primary_filter_hold_wait_candle_retrace"
                    df_rows.at[idx, "staged_reference_price"] = candidate_retrace["reference_price"]
                    df_rows.at[idx, "staged_trigger_price"] = candidate_retrace["trigger_price"]
                    df_rows.at[idx, "staged_breakout_trigger_price"] = breakout_trigger_price
                    df_rows.at[idx, "staged_expires_at"] = expires_at.isoformat() if pd.notna(expires_at) else pd.NA
                    df_rows.at[idx, "staged_entry_improvement_pips"] = candidate_retrace["entry_improvement_pips"]
                    df_rows.at[idx, "staged_adaptive_profile"] = strong_primary_hold_check["profile"]
                    self.logger.info(
                        "Candidata staged por primario fuerte con filtro HOLD: model=%s side=%s close=%s trigger=%s breakout=%s stop=%s",
                        model,
                        primary_signal,
                        candle_close,
                        candidate_retrace["trigger_price"],
                        breakout_trigger_price,
                        candidate_retrace["custom_stop_price"],
                    )
                    continue

            if (
                final_signal == "HOLD"
                and primary_signal in {"BUY", "SELL"}
                and filter_signal == "HOLD"
                and medium_primary_hold_check["eligible"]
                and pip_size > 0
                and pd.notna(candle_close)
            ):
                if active_idx is not None:
                    staged.at[active_idx, "status"] = "SUPERSEDED_BY_MEDIUM_PRIMARY_HOLD_RETRACE"
                    staged.at[active_idx, "cancel_timestamp"] = now_iso
                    staged.at[active_idx, "cancel_reason"] = "medium_primary_hold_retrace_refresh"
                    staged.at[active_idx, "status_reason"] = "medium_primary_hold_retrace_refresh"
                    staged.at[active_idx, "last_evaluated_at"] = now_iso
                    changed = True

                reference_price = _current_exec_price_for(primary_signal)
                candidate_retrace = self._build_candle_retrace_candidate(
                    side=primary_signal,
                    candle_high=candle_high,
                    candle_low=candle_low,
                    candle_close=candle_close,
                    reference_price=reference_price,
                    atr_value=runtime_ctx.get("atr_value"),
                    pip_size=pip_size,
                    digits=int(max(runtime_ctx.get("digits", 5), 0)),
                    retrace_fraction=float(medium_primary_hold_check["retrace_fraction"]),
                    stop_buffer_pips=float(settings["direct_filter_hold_stop_buffer_pips"]),
                    stop_buffer_atr_fraction=float(settings["dynamic_stop_atr_fraction"]),
                    min_stop_pips=float(settings["dynamic_stop_min_pips"]),
                )
                if candidate_retrace is not None and pd.notna(reference_price):
                    immediate_partial_fraction = 0.0
                    if bool(settings.get("medium_primary_hold_immediate_market_enabled", False)) and bool(
                        settings.get("medium_primary_hold_partial_enabled", False)
                    ):
                        immediate_partial_fraction = min(
                            max(float(medium_primary_hold_check["partial_fraction"]), 0.0),
                            1.0,
                        )
                    immediate_partial_suppressed_reason = None
                    medium_partial_payload = None
                    remaining_stage_fraction = 1.0
                    if immediate_partial_fraction > 0.0:
                        adverse_extreme_hit, entry_location_in_candle, candle_range_pips = (
                            self._is_immediate_entry_price_in_adverse_extreme(
                                side=primary_signal,
                                price=float(candidate_retrace["reference_price"]),
                                candle_high=candle_high,
                                candle_low=candle_low,
                                pip_size=pip_size,
                                settings=settings,
                            )
                        )
                        if adverse_extreme_hit:
                            immediate_partial_fraction = 0.0
                            immediate_partial_suppressed_reason = (
                                "medium_primary_filter_hold_immediate_extreme_entry_wait_candle_retrace"
                            )
                            self.logger.info(
                                "Pata inmediata media suprimida por entrada en extremo adverso: model=%s side=%s ref=%s loc=%.3f range_pips=%.2f",
                                model,
                                primary_signal,
                                candidate_retrace["reference_price"],
                                float(entry_location_in_candle) if entry_location_in_candle is not None else float("nan"),
                                float(candle_range_pips) if candle_range_pips is not None else float("nan"),
                            )
                    if immediate_partial_fraction > 0.0:
                        medium_partial_payload = self._compute_runtime_trade_payload_for_signal(
                            signal=primary_signal,
                            predicted_pips_signed=predicted_pips,
                            signal_time=timestamp if pd.notna(timestamp) else row.get("timestamp"),
                            runtime_ctx=runtime_ctx,
                            volume_scale=immediate_partial_fraction,
                            disable_entry_management=True,
                            explicit_sl_price=candidate_retrace["custom_stop_price"],
                            market_only_comment="medium_primary_filter_hold_partial_market_only",
                        )
                        if medium_partial_payload and float(medium_partial_payload.get("volume_lots") or 0.0) > 0:
                            remaining_stage_fraction = max(1.0 - immediate_partial_fraction, 0.0)
                            full_volume_lots_raw = pd.to_numeric(
                                pd.Series([row.get("volume_lots")]),
                                errors="coerce",
                            ).iloc[0]
                            market_volume_lots_raw = pd.to_numeric(
                                pd.Series([medium_partial_payload.get("volume_lots")]),
                                errors="coerce",
                            ).iloc[0]
                            if (
                                remaining_stage_fraction > 0.0
                                and pd.notna(full_volume_lots_raw)
                                and float(full_volume_lots_raw) > 0.0
                                and pd.notna(market_volume_lots_raw)
                                and float(market_volume_lots_raw) > 0.0
                            ):
                                partial_entry_plan = self._build_immediate_partial_pending_entry_plan(
                                    signal=primary_signal,
                                    total_volume_lots=float(full_volume_lots_raw),
                                    market_volume_lots=float(market_volume_lots_raw),
                                    live_entry_price=float(medium_partial_payload["live_entry_price"]),
                                    live_sl_price=float(medium_partial_payload["live_sl_price"]),
                                    live_tp_price=float(medium_partial_payload["live_tp_price"]),
                                    digits=int(max(runtime_ctx.get("digits", 5) or 5, 0)),
                                    min_lot=float(runtime_ctx.get("min_lot") or 0.01),
                                    lot_step=float(runtime_ctx.get("lot_step") or runtime_ctx.get("min_lot") or 0.01),
                                    timeframe=timeframe,
                                    signal_time=timestamp if pd.notna(timestamp) else row.get("timestamp"),
                                    market_only_comment="medium_primary_filter_hold_partial_market_only",
                                    pending_comment="medium_primary_filter_hold_partial_market_and_pending",
                                )
                                partial_pending_volume = pd.to_numeric(
                                    pd.Series([partial_entry_plan.get("pending_order_volume_lots")]),
                                    errors="coerce",
                                ).iloc[0]
                                if (
                                    self._coerce_boolish(partial_entry_plan.get("entry_management_split_active"))
                                    and pd.notna(partial_pending_volume)
                                    and float(partial_pending_volume) > 0.0
                                ):
                                    medium_partial_payload["entry_management_plan"] = partial_entry_plan
                                    remaining_stage_fraction = 0.0
                                    self.logger.info(
                                        "Entrada media inmediata con pending real: model=%s side=%s market_lots=%.2f pending_lots=%.2f pending_price=%s",
                                        model,
                                        primary_signal,
                                        float(market_volume_lots_raw),
                                        float(partial_pending_volume),
                                        partial_entry_plan.get("pending_order_price"),
                                    )
                        else:
                            medium_partial_payload = None
                            remaining_stage_fraction = 1.0

                    breakout_trigger_price = (
                        self._compute_breakout_trigger_price(
                            side=primary_signal,
                            reference_price=candidate_retrace["reference_price"],
                            reference_stop_pips=candidate_retrace["reference_stop_pips"],
                            pip_size=pip_size,
                            settings=settings,
                        )
                        if float(medium_primary_hold_check["breakout_partial_fraction"]) > 0
                        else float("nan")
                    )
                    expires_at = (
                        timestamp + timeframe_delta * int(medium_primary_hold_check["max_stage_bars"])
                        if pd.notna(timestamp)
                        else pd.NaT
                    )
                    candidate_id = f"{self._build_signal_id(row)}|MPH_STAGE"
                    if remaining_stage_fraction > 0:
                        staged = pd.concat(
                            [
                                staged,
                                pd.DataFrame(
                                    [
                                        {
                                            "candidate_id": candidate_id,
                                            "parent_signal_id": self._build_signal_id(row),
                                            "release_id": row.get("release_id"),
                                            "strategy_profile": strategy_profile,
                                            "symbol": symbol,
                                            "timeframe": timeframe,
                                            "model": model,
                                            "side": primary_signal,
                                            "created_at": now_iso,
                                            "source_timestamp": timestamp.isoformat() if pd.notna(timestamp) else row.get("timestamp"),
                                            "expires_at": expires_at.isoformat() if pd.notna(expires_at) else pd.NA,
                                            "reference_price": candidate_retrace["reference_price"],
                                            "trigger_price": candidate_retrace["trigger_price"],
                                            "breakout_trigger_price": breakout_trigger_price,
                                            "reference_stop_pips": candidate_retrace["reference_stop_pips"],
                                            "entry_improvement_pips": candidate_retrace["entry_improvement_pips"],
                                            "predicted_pips": predicted_pips,
                                            "pred_return": row.get("pred_return"),
                                            "candidate_mode": "medium_primary_filter_hold_candle_retrace",
                                            "candidate_volume_scale": remaining_stage_fraction,
                                            "adaptive_profile": medium_primary_hold_check["profile"],
                                            "adaptive_retrace_fraction": medium_primary_hold_check["retrace_fraction"],
                                            "adaptive_breakout_fraction": medium_primary_hold_check["breakout_partial_fraction"],
                                            "signal_candle_open": candle_open,
                                            "signal_candle_high": candle_high,
                                            "signal_candle_low": candle_low,
                                            "signal_candle_close": candle_close,
                                            "custom_stop_price": candidate_retrace["custom_stop_price"],
                                            "primary_confidence": primary_confidence,
                                            "filter_signal": filter_signal,
                                            "filter_support_score": support_score,
                                            "filter_contradicted": filter_contradicted,
                                            "status": "ACTIVE",
                                            "status_reason": "created_medium_primary_filter_hold_retrace",
                                            "last_evaluated_at": now_iso,
                                            "hold_grace_count": 0,
                                            "activation_timestamp": pd.NA,
                                            "activation_price": np.nan,
                                            "activation_reason": pd.NA,
                                            "cancel_timestamp": pd.NA,
                                            "cancel_reason": pd.NA,
                                        }
                                    ]
                                ),
                            ],
                            ignore_index=True,
                        )
                        changed = True

                    if medium_partial_payload and float(medium_partial_payload.get("volume_lots") or 0.0) > 0:
                        df_rows.at[idx, "signal"] = primary_signal
                        df_rows.at[idx, "confidence"] = (
                            primary_confidence if pd.notna(primary_confidence) else row.get("confidence")
                        )
                        df_rows.at[idx, "signal_confirmation_passed"] = True
                        df_rows.at[idx, "signal_confirmation_reason"] = (
                            "medium_primary_filter_hold_partial_market_and_stage"
                            if remaining_stage_fraction > 0
                            else (
                                "medium_primary_filter_hold_partial_market_and_pending"
                                if self._coerce_boolish(
                                    medium_partial_payload["entry_management_plan"].get("entry_management_split_active")
                                )
                                else "medium_primary_filter_hold_partial_market_only"
                            )
                        )
                        df_rows.at[idx, "entry_price"] = medium_partial_payload["entry_price"]
                        df_rows.at[idx, "planned_entry_price"] = medium_partial_payload["planned_entry_price"]
                        df_rows.at[idx, "price_target"] = medium_partial_payload["tp_price"]
                        df_rows.at[idx, "delta_price"] = (
                            medium_partial_payload["tp_price"] - current_price
                            if primary_signal == "BUY"
                            else current_price - medium_partial_payload["tp_price"]
                        )
                        df_rows.at[idx, "sl_price"] = medium_partial_payload["sl_price"]
                        df_rows.at[idx, "tp_price"] = medium_partial_payload["tp_price"]
                        df_rows.at[idx, "sl_pips"] = medium_partial_payload["sl_pips"]
                        df_rows.at[idx, "tp_pips"] = medium_partial_payload["tp_pips"]
                        df_rows.at[idx, "market_reference_price"] = medium_partial_payload["market_reference_price"]
                        df_rows.at[idx, "live_entry_price"] = medium_partial_payload["live_entry_price"]
                        df_rows.at[idx, "live_sl_price"] = medium_partial_payload["live_sl_price"]
                        df_rows.at[idx, "live_tp_price"] = medium_partial_payload["live_tp_price"]
                        df_rows.at[idx, "live_sl_pips"] = medium_partial_payload["live_sl_pips"]
                        df_rows.at[idx, "live_tp_pips"] = medium_partial_payload["live_tp_pips"]
                        df_rows.at[idx, "volume_lots"] = medium_partial_payload["volume_lots"]
                        df_rows.at[idx, "risk_amount"] = medium_partial_payload["risk_amount"]
                        df_rows.at[idx, "allocated_risk_budget"] = medium_partial_payload["allocated_risk_budget"]
                        df_rows.at[idx, "risk_per_pip_per_lot"] = medium_partial_payload["risk_per_pip_per_lot"]
                        df_rows.at[idx, "risk_per_lot_at_stop"] = medium_partial_payload["risk_per_lot_at_stop"]
                        df_rows.at[idx, "remaining_risk_budget_before_trade"] = medium_partial_payload[
                            "remaining_risk_budget_before_trade"
                        ]
                        df_rows.at[idx, "projected_total_open_risk_after_trade"] = medium_partial_payload[
                            "projected_total_open_risk_after_trade"
                        ]
                        partial_entry_plan = medium_partial_payload["entry_management_plan"]
                        df_rows.at[idx, "entry_management_mode"] = partial_entry_plan["entry_management_mode"]
                        df_rows.at[idx, "entry_management_split_active"] = partial_entry_plan[
                            "entry_management_split_active"
                        ]
                        df_rows.at[idx, "entry_management_initial_market_fraction"] = partial_entry_plan[
                            "entry_management_initial_market_fraction"
                        ]
                        df_rows.at[idx, "entry_management_pending_fraction"] = partial_entry_plan[
                            "entry_management_pending_fraction"
                        ]
                        df_rows.at[idx, "entry_management_retrace_fraction_of_stop"] = partial_entry_plan[
                            "entry_management_retrace_fraction_of_stop"
                        ]
                        df_rows.at[idx, "entry_management_total_volume_lots"] = partial_entry_plan[
                            "entry_management_total_volume_lots"
                        ]
                        df_rows.at[idx, "initial_market_volume_lots"] = partial_entry_plan[
                            "initial_market_volume_lots"
                        ]
                        df_rows.at[idx, "pending_order_volume_lots"] = partial_entry_plan[
                            "pending_order_volume_lots"
                        ]
                        df_rows.at[idx, "pending_order_price"] = partial_entry_plan["pending_order_price"]
                        df_rows.at[idx, "pending_order_type"] = partial_entry_plan["pending_order_type"]
                        df_rows.at[idx, "pending_order_sl_price"] = partial_entry_plan["pending_order_sl_price"]
                        df_rows.at[idx, "pending_order_tp_price"] = partial_entry_plan["pending_order_tp_price"]
                        df_rows.at[idx, "pending_order_expiry_time"] = partial_entry_plan["pending_order_expiry_time"]
                        df_rows.at[idx, "entry_management_comment"] = partial_entry_plan["entry_management_comment"]
                        df_rows.at[idx, "staged_candidate_id"] = candidate_id if remaining_stage_fraction > 0 else pd.NA
                        df_rows.at[idx, "staged_status"] = "ACTIVE" if remaining_stage_fraction > 0 else pd.NA
                        df_rows.at[idx, "staged_action"] = (
                            "CREATED_WITH_IMMEDIATE_PARTIAL" if remaining_stage_fraction > 0 else pd.NA
                        )
                        df_rows.at[idx, "staged_reason"] = (
                            "medium_primary_filter_hold_partial_wait_candle_retrace"
                            if remaining_stage_fraction > 0
                            else pd.NA
                        )
                        df_rows.at[idx, "staged_reference_price"] = (
                            candidate_retrace["reference_price"] if remaining_stage_fraction > 0 else np.nan
                        )
                        df_rows.at[idx, "staged_trigger_price"] = (
                            candidate_retrace["trigger_price"] if remaining_stage_fraction > 0 else np.nan
                        )
                        df_rows.at[idx, "staged_breakout_trigger_price"] = (
                            breakout_trigger_price if remaining_stage_fraction > 0 else np.nan
                        )
                        df_rows.at[idx, "staged_expires_at"] = (
                            expires_at.isoformat() if remaining_stage_fraction > 0 and pd.notna(expires_at) else pd.NA
                        )
                        df_rows.at[idx, "staged_entry_improvement_pips"] = (
                            candidate_retrace["entry_improvement_pips"] if remaining_stage_fraction > 0 else np.nan
                        )
                        df_rows.at[idx, "staged_adaptive_profile"] = (
                            medium_primary_hold_check["profile"] if remaining_stage_fraction > 0 else pd.NA
                        )
                        self.logger.info(
                            "Entrada parcial inmediata media con filtro HOLD: model=%s side=%s market_fraction=%.2f staged_fraction=%.2f close=%s trigger=%s stop=%s",
                            model,
                            primary_signal,
                            immediate_partial_fraction,
                            remaining_stage_fraction,
                            candle_close,
                            candidate_retrace["trigger_price"],
                            candidate_retrace["custom_stop_price"],
                        )
                    else:
                        df_rows.at[idx, "signal"] = "HOLD"
                        df_rows.at[idx, "signal_confirmation_reason"] = (
                            immediate_partial_suppressed_reason
                            or "medium_primary_filter_hold_wait_candle_retrace"
                        )
                        df_rows.at[idx, "volume_lots"] = 0.0
                        df_rows.at[idx, "risk_amount"] = 0.0
                        df_rows.at[idx, "allocated_risk_budget"] = 0.0
                        df_rows.at[idx, "entry_management_split_active"] = False
                        df_rows.at[idx, "initial_market_volume_lots"] = 0.0
                        df_rows.at[idx, "pending_order_volume_lots"] = 0.0
                        df_rows.at[idx, "pending_order_price"] = np.nan
                        df_rows.at[idx, "pending_order_type"] = pd.NA
                        df_rows.at[idx, "pending_order_sl_price"] = np.nan
                        df_rows.at[idx, "pending_order_tp_price"] = np.nan
                        df_rows.at[idx, "pending_order_expiry_time"] = pd.NA
                        df_rows.at[idx, "entry_management_comment"] = (
                            immediate_partial_suppressed_reason
                            or "medium_primary_filter_hold_wait_candle_retrace"
                        )
                        df_rows.at[idx, "staged_candidate_id"] = candidate_id
                        df_rows.at[idx, "staged_status"] = "ACTIVE"
                        df_rows.at[idx, "staged_action"] = "CREATED"
                        df_rows.at[idx, "staged_reason"] = (
                            immediate_partial_suppressed_reason
                            or "medium_primary_filter_hold_wait_candle_retrace"
                        )
                        df_rows.at[idx, "staged_reference_price"] = candidate_retrace["reference_price"]
                        df_rows.at[idx, "staged_trigger_price"] = candidate_retrace["trigger_price"]
                        df_rows.at[idx, "staged_breakout_trigger_price"] = breakout_trigger_price
                        df_rows.at[idx, "staged_expires_at"] = (
                            expires_at.isoformat() if pd.notna(expires_at) else pd.NA
                        )
                        df_rows.at[idx, "staged_entry_improvement_pips"] = candidate_retrace["entry_improvement_pips"]
                        df_rows.at[idx, "staged_adaptive_profile"] = medium_primary_hold_check["profile"]
                        self.logger.info(
                            "Candidata media con filtro HOLD enviada a staging: model=%s side=%s close=%s trigger=%s stop=%s",
                            model,
                            primary_signal,
                            candle_close,
                            candidate_retrace["trigger_price"],
                            candidate_retrace["custom_stop_price"],
                        )
                    continue

            pilot_allowed = (
                bool(settings["pilot_entry_enabled"])
                and final_signal == "HOLD"
                and primary_signal in {"BUY", "SELL"}
                and pd.notna(primary_confidence)
                and primary_confidence >= float(settings["pilot_confidence_min"])
                and pd.notna(predicted_pips)
                and abs(predicted_pips) >= float(settings["min_abs_predicted_pips"])
                and (
                    bool(settings["pilot_allow_on_filter_contradiction"])
                    or not filter_contradicted
                )
            )
            if pilot_allowed:
                if active_idx is not None:
                    staged.at[active_idx, "status"] = "SUPERSEDED_BY_PILOT_ENTRY"
                    staged.at[active_idx, "cancel_timestamp"] = now_iso
                    staged.at[active_idx, "cancel_reason"] = "pilot_entry_applied"
                    staged.at[active_idx, "status_reason"] = "pilot_entry_applied"
                    staged.at[active_idx, "last_evaluated_at"] = now_iso
                    changed = True
                    df_rows.at[idx, "staged_candidate_id"] = active_candidate.get("candidate_id")
                    df_rows.at[idx, "staged_status"] = "SUPERSEDED_BY_PILOT_ENTRY"
                    df_rows.at[idx, "staged_action"] = "SUPERSEDED_BY_PILOT_ENTRY"
                    df_rows.at[idx, "staged_reason"] = "pilot_entry_applied"
                    active_idx = None
                    active_candidate = None

                if bool(settings["pilot_convert_to_staged"]):
                    reference_price = _current_exec_price_for(primary_signal)
                    adaptive_profile = self._build_adaptive_entry_profile(
                        signal=primary_signal,
                        feature_row=row,
                        settings=settings,
                    )
                    if (
                        settings["pilot_allowed_profiles"]
                        and str(adaptive_profile["profile"]).strip().lower() not in settings["pilot_allowed_profiles"]
                    ):
                        df_rows.at[idx, "signal"] = "HOLD"
                        df_rows.at[idx, "signal_confirmation_reason"] = "pilot_profile_not_allowed"
                        df_rows.at[idx, "pilot_entry_applied"] = False
                        df_rows.at[idx, "pilot_entry_reason"] = "pilot_profile_not_allowed"
                        df_rows.at[idx, "staged_status"] = "SKIPPED"
                        df_rows.at[idx, "staged_action"] = "SKIPPED"
                        df_rows.at[idx, "staged_reason"] = "pilot_profile_not_allowed"
                        self.logger.info(
                            "Entrada piloto omitida: model=%s side=%s profile=%s motivo=pilot_profile_not_allowed",
                            model,
                            primary_signal,
                            adaptive_profile["profile"],
                        )
                        continue
                    candidate_retrace = self._build_candle_retrace_candidate(
                        side=primary_signal,
                        candle_high=candle_high,
                        candle_low=candle_low,
                        candle_close=candle_close,
                        reference_price=reference_price,
                        atr_value=runtime_ctx.get("atr_value"),
                        pip_size=pip_size,
                        digits=int(max(runtime_ctx.get("digits", 5), 0)),
                        retrace_fraction=float(adaptive_profile["retrace_fraction"]),
                        stop_buffer_pips=float(settings["direct_filter_hold_stop_buffer_pips"]),
                        stop_buffer_atr_fraction=float(settings["dynamic_stop_atr_fraction"]),
                        min_stop_pips=float(settings["dynamic_stop_min_pips"]),
                    )
                    if candidate_retrace is not None:
                        breakout_trigger_price = (
                            self._compute_breakout_trigger_price(
                                side=primary_signal,
                                reference_price=candidate_retrace["reference_price"],
                                reference_stop_pips=candidate_retrace["reference_stop_pips"],
                                pip_size=pip_size,
                                settings=settings,
                            )
                            if float(adaptive_profile["breakout_partial_fraction"]) > 0
                            else float("nan")
                        )
                        expires_at = (
                            timestamp + timeframe_delta * int(adaptive_profile["max_stage_bars"])
                            if pd.notna(timestamp)
                            else pd.NaT
                        )
                        candidate_id = f"{self._build_signal_id(row)}|PL_STAGE"
                        staged = pd.concat(
                            [
                                staged,
                                pd.DataFrame(
                                    [
                                        {
                                            "candidate_id": candidate_id,
                                            "parent_signal_id": self._build_signal_id(row),
                                            "release_id": row.get("release_id"),
                                            "strategy_profile": strategy_profile,
                                            "symbol": symbol,
                                            "timeframe": timeframe,
                                            "model": model,
                                            "side": primary_signal,
                                            "created_at": now_iso,
                                            "source_timestamp": timestamp.isoformat() if pd.notna(timestamp) else row.get("timestamp"),
                                            "expires_at": expires_at.isoformat() if pd.notna(expires_at) else pd.NA,
                                            "reference_price": candidate_retrace["reference_price"],
                                            "trigger_price": candidate_retrace["trigger_price"],
                                            "breakout_trigger_price": breakout_trigger_price,
                                            "reference_stop_pips": candidate_retrace["reference_stop_pips"],
                                            "entry_improvement_pips": candidate_retrace["entry_improvement_pips"],
                                            "predicted_pips": predicted_pips,
                                            "pred_return": row.get("pred_return"),
                                            "candidate_mode": "pilot_candle_retrace",
                                            "candidate_volume_scale": float(settings["pilot_fraction_of_full_size"]),
                                            "adaptive_profile": adaptive_profile["profile"],
                                            "adaptive_retrace_fraction": adaptive_profile["retrace_fraction"],
                                            "adaptive_breakout_fraction": adaptive_profile["breakout_partial_fraction"],
                                            "signal_candle_open": candle_open,
                                            "signal_candle_high": candle_high,
                                            "signal_candle_low": candle_low,
                                            "signal_candle_close": candle_close,
                                            "custom_stop_price": candidate_retrace["custom_stop_price"],
                                            "primary_confidence": primary_confidence,
                                            "filter_signal": filter_signal,
                                            "filter_support_score": support_score,
                                            "filter_contradicted": filter_contradicted,
                                            "status": "ACTIVE",
                                            "status_reason": "created_pilot_candle_retrace",
                                            "last_evaluated_at": now_iso,
                                            "activation_timestamp": pd.NA,
                                            "activation_price": np.nan,
                                            "activation_reason": pd.NA,
                                            "cancel_timestamp": pd.NA,
                                            "cancel_reason": pd.NA,
                                        }
                                    ]
                                ),
                            ],
                            ignore_index=True,
                        )
                        changed = True
                        df_rows.at[idx, "signal"] = "HOLD"
                        df_rows.at[idx, "signal_confirmation_reason"] = "pilot_wait_candle_retrace"
                        df_rows.at[idx, "volume_lots"] = 0.0
                        df_rows.at[idx, "risk_amount"] = 0.0
                        df_rows.at[idx, "allocated_risk_budget"] = 0.0
                        df_rows.at[idx, "entry_management_split_active"] = False
                        df_rows.at[idx, "initial_market_volume_lots"] = 0.0
                        df_rows.at[idx, "pending_order_volume_lots"] = 0.0
                        df_rows.at[idx, "pending_order_price"] = np.nan
                        df_rows.at[idx, "pending_order_type"] = pd.NA
                        df_rows.at[idx, "pending_order_sl_price"] = np.nan
                        df_rows.at[idx, "pending_order_tp_price"] = np.nan
                        df_rows.at[idx, "pending_order_expiry_time"] = pd.NA
                        df_rows.at[idx, "entry_management_comment"] = "pilot_wait_candle_retrace"
                        df_rows.at[idx, "pilot_entry_applied"] = False
                        df_rows.at[idx, "pilot_entry_reason"] = "pilot_wait_candle_retrace"
                        df_rows.at[idx, "staged_candidate_id"] = candidate_id
                        df_rows.at[idx, "staged_status"] = "ACTIVE"
                        df_rows.at[idx, "staged_action"] = "CREATED"
                        df_rows.at[idx, "staged_reason"] = "pilot_wait_candle_retrace"
                        df_rows.at[idx, "staged_reference_price"] = candidate_retrace["reference_price"]
                        df_rows.at[idx, "staged_trigger_price"] = candidate_retrace["trigger_price"]
                        df_rows.at[idx, "staged_breakout_trigger_price"] = breakout_trigger_price
                        df_rows.at[idx, "staged_expires_at"] = expires_at.isoformat() if pd.notna(expires_at) else pd.NA
                        df_rows.at[idx, "staged_entry_improvement_pips"] = candidate_retrace["entry_improvement_pips"]
                        df_rows.at[idx, "staged_adaptive_profile"] = adaptive_profile["profile"]
                        self.logger.info(
                            "Entrada piloto staged: model=%s side=%s profile=%s close=%s trigger=%s breakout=%s stop=%s vol_scale=%0.2f",
                            model,
                            primary_signal,
                            adaptive_profile["profile"],
                            candle_close,
                            candidate_retrace["trigger_price"],
                            breakout_trigger_price,
                            candidate_retrace["custom_stop_price"],
                            float(settings["pilot_fraction_of_full_size"]),
                        )
                        continue

                    if not bool(settings["pilot_market_fallback_enabled"]):
                        df_rows.at[idx, "signal"] = "HOLD"
                        df_rows.at[idx, "signal_confirmation_reason"] = "pilot_retrace_unavailable"
                        df_rows.at[idx, "volume_lots"] = 0.0
                        df_rows.at[idx, "risk_amount"] = 0.0
                        df_rows.at[idx, "allocated_risk_budget"] = 0.0
                        df_rows.at[idx, "entry_management_split_active"] = False
                        df_rows.at[idx, "initial_market_volume_lots"] = 0.0
                        df_rows.at[idx, "pending_order_volume_lots"] = 0.0
                        df_rows.at[idx, "pending_order_price"] = np.nan
                        df_rows.at[idx, "pending_order_type"] = pd.NA
                        df_rows.at[idx, "pending_order_sl_price"] = np.nan
                        df_rows.at[idx, "pending_order_tp_price"] = np.nan
                        df_rows.at[idx, "pending_order_expiry_time"] = pd.NA
                        df_rows.at[idx, "entry_management_comment"] = "pilot_retrace_unavailable"
                        df_rows.at[idx, "pilot_entry_applied"] = False
                        df_rows.at[idx, "pilot_entry_reason"] = "pilot_retrace_unavailable"
                        df_rows.at[idx, "staged_status"] = "SKIPPED"
                        df_rows.at[idx, "staged_action"] = "SKIPPED"
                        df_rows.at[idx, "staged_reason"] = "pilot_retrace_unavailable"
                        self.logger.info(
                            "Entrada piloto omitida: model=%s side=%s motivo=pilot_retrace_unavailable",
                            model,
                            primary_signal,
                        )
                        continue

                pilot_payload = self._compute_runtime_trade_payload_for_signal(
                    signal=primary_signal,
                    predicted_pips_signed=predicted_pips,
                    signal_time=timestamp if pd.notna(timestamp) else row.get("timestamp"),
                    runtime_ctx=runtime_ctx,
                    volume_scale=float(settings["pilot_fraction_of_full_size"]),
                    disable_entry_management=True,
                )
                if pilot_payload and float(pilot_payload.get("volume_lots") or 0.0) > 0:
                    df_rows.at[idx, "signal"] = primary_signal
                    df_rows.at[idx, "confidence"] = primary_confidence
                    df_rows.at[idx, "signal_confirmation_passed"] = True
                    df_rows.at[idx, "signal_confirmation_reason"] = "entry_staging_pilot_primary_confidence"
                    df_rows.at[idx, "entry_price"] = pilot_payload["entry_price"]
                    df_rows.at[idx, "planned_entry_price"] = pilot_payload["planned_entry_price"]
                    df_rows.at[idx, "price_target"] = pilot_payload["tp_price"]
                    df_rows.at[idx, "delta_price"] = (
                        pilot_payload["tp_price"] - current_price
                        if primary_signal == "BUY"
                        else current_price - pilot_payload["tp_price"]
                    )
                    df_rows.at[idx, "signal_target_tp_price"] = pilot_payload["signal_target_tp_price"]
                    df_rows.at[idx, "signal_target_sl_price"] = pilot_payload["signal_target_sl_price"]
                    df_rows.at[idx, "signal_target_tp_pips"] = pilot_payload["signal_target_tp_pips"]
                    df_rows.at[idx, "signal_target_sl_pips"] = pilot_payload["signal_target_sl_pips"]
                    df_rows.at[idx, "sl_price"] = pilot_payload["sl_price"]
                    df_rows.at[idx, "tp_price"] = pilot_payload["tp_price"]
                    df_rows.at[idx, "sl_pips"] = pilot_payload["sl_pips"]
                    df_rows.at[idx, "tp_pips"] = pilot_payload["tp_pips"]
                    df_rows.at[idx, "market_reference_price"] = pilot_payload["market_reference_price"]
                    df_rows.at[idx, "live_entry_price"] = pilot_payload["live_entry_price"]
                    df_rows.at[idx, "live_sl_price"] = pilot_payload["live_sl_price"]
                    df_rows.at[idx, "live_tp_price"] = pilot_payload["live_tp_price"]
                    df_rows.at[idx, "live_sl_pips"] = pilot_payload["live_sl_pips"]
                    df_rows.at[idx, "live_tp_pips"] = pilot_payload["live_tp_pips"]
                    df_rows.at[idx, "volume_lots"] = pilot_payload["volume_lots"]
                    df_rows.at[idx, "risk_amount"] = pilot_payload["risk_amount"]
                    df_rows.at[idx, "allocated_risk_budget"] = pilot_payload["allocated_risk_budget"]
                    df_rows.at[idx, "risk_per_pip_per_lot"] = pilot_payload["risk_per_pip_per_lot"]
                    df_rows.at[idx, "risk_per_lot_at_stop"] = pilot_payload["risk_per_lot_at_stop"]
                    df_rows.at[idx, "remaining_risk_budget_before_trade"] = pilot_payload["remaining_risk_budget_before_trade"]
                    df_rows.at[idx, "projected_total_open_risk_after_trade"] = pilot_payload["projected_total_open_risk_after_trade"]
                    pilot_plan = pilot_payload["entry_management_plan"]
                    df_rows.at[idx, "entry_management_mode"] = pilot_plan["entry_management_mode"]
                    df_rows.at[idx, "entry_management_split_active"] = pilot_plan["entry_management_split_active"]
                    df_rows.at[idx, "entry_management_initial_market_fraction"] = pilot_plan["entry_management_initial_market_fraction"]
                    df_rows.at[idx, "entry_management_pending_fraction"] = pilot_plan["entry_management_pending_fraction"]
                    df_rows.at[idx, "entry_management_retrace_fraction_of_stop"] = pilot_plan["entry_management_retrace_fraction_of_stop"]
                    df_rows.at[idx, "entry_management_total_volume_lots"] = pilot_plan["entry_management_total_volume_lots"]
                    df_rows.at[idx, "initial_market_volume_lots"] = pilot_plan["initial_market_volume_lots"]
                    df_rows.at[idx, "pending_order_volume_lots"] = pilot_plan["pending_order_volume_lots"]
                    df_rows.at[idx, "pending_order_price"] = pilot_plan["pending_order_price"]
                    df_rows.at[idx, "pending_order_type"] = pilot_plan["pending_order_type"]
                    df_rows.at[idx, "pending_order_sl_price"] = pilot_plan["pending_order_sl_price"]
                    df_rows.at[idx, "pending_order_tp_price"] = pilot_plan["pending_order_tp_price"]
                    df_rows.at[idx, "pending_order_expiry_time"] = pilot_plan["pending_order_expiry_time"]
                    df_rows.at[idx, "entry_management_comment"] = pilot_plan["entry_management_comment"]
                    df_rows.at[idx, "pilot_entry_applied"] = True
                    df_rows.at[idx, "pilot_entry_reason"] = "primary_confidence_ge_threshold"
                    df_rows.at[idx, "staged_action"] = "PILOT_ENTRY"
                    df_rows.at[idx, "staged_reason"] = "pilot_entry_primary_confidence"
                    self.logger.info(
                        "âœ³ Entrada piloto aplicada: model=%s side=%s primary_conf=%0.3f pips=%0.2f lots=%0.2f",
                        model,
                        primary_signal,
                        primary_confidence,
                        predicted_pips,
                        float(pilot_payload["volume_lots"]),
                    )
                    continue

            if active_idx is not None:
                expires_at = pd.to_datetime(active_candidate.get("expires_at"), errors="coerce")
                candidate_side = str(active_candidate.get("side", "") or "").upper()
                candidate_mode = str(active_candidate.get("candidate_mode", "") or "").strip().lower()
                candidate_trigger = pd.to_numeric(pd.Series([active_candidate.get("trigger_price")]), errors="coerce").iloc[0]
                candidate_trigger = float(candidate_trigger) if pd.notna(candidate_trigger) else float("nan")
                candidate_breakout_trigger = pd.to_numeric(pd.Series([active_candidate.get("breakout_trigger_price")]), errors="coerce").iloc[0]
                candidate_breakout_trigger = float(candidate_breakout_trigger) if pd.notna(candidate_breakout_trigger) else float("nan")
                candidate_stop_price = pd.to_numeric(pd.Series([active_candidate.get("custom_stop_price")]), errors="coerce").iloc[0]
                candidate_stop_price = float(candidate_stop_price) if pd.notna(candidate_stop_price) else float("nan")
                candidate_volume_scale = pd.to_numeric(pd.Series([active_candidate.get("candidate_volume_scale")]), errors="coerce").iloc[0]
                candidate_volume_scale = float(candidate_volume_scale) if pd.notna(candidate_volume_scale) else 1.0
                candidate_breakout_fraction = pd.to_numeric(pd.Series([active_candidate.get("adaptive_breakout_fraction")]), errors="coerce").iloc[0]
                candidate_breakout_fraction = float(candidate_breakout_fraction) if pd.notna(candidate_breakout_fraction) else 0.0
                candidate_adaptive_profile = str(active_candidate.get("adaptive_profile", "") or "")
                candidate_source_timestamp = pd.to_datetime(
                    active_candidate.get("source_timestamp"),
                    errors="coerce",
                )
                hold_grace_count_raw = pd.to_numeric(
                    pd.Series([active_candidate.get("hold_grace_count")]),
                    errors="coerce",
                ).iloc[0]
                hold_grace_count = int(hold_grace_count_raw) if pd.notna(hold_grace_count_raw) else 0
                candidate_exec_price = _current_exec_price_for(candidate_side) if candidate_side in {"BUY", "SELL"} else current_price
                entry_context_check = self._evaluate_entry_context_guard(
                    signal=candidate_side,
                    feature_row=row,
                    candle_open=candle_open,
                    candle_high=candle_high,
                    candle_low=candle_low,
                    candle_close=candle_close,
                    settings=settings,
                )
                cancel_reason = None
                if pd.notna(expires_at) and pd.notna(timestamp) and timestamp > expires_at:
                    extension_count_raw = pd.to_numeric(
                        pd.Series([active_candidate.get("extension_count")]),
                        errors="coerce",
                    ).iloc[0]
                    extension_count = int(extension_count_raw) if pd.notna(extension_count_raw) else 0
                    extension_bars = max(int(settings.get("direct_confirmed_extension_bars", 1) or 1), 1)
                    if (
                        candidate_mode == "direct_confirmed_candle_retrace"
                        and bool(settings.get("direct_confirmed_extend_if_aligned", True))
                        and extension_count < int(settings.get("direct_confirmed_max_extensions", 1) or 1)
                        and primary_signal == candidate_side
                        and filter_signal == candidate_side
                        and self._coerce_boolish(row.get("gate_passed"))
                        and self._coerce_boolish(row.get("alignment_ok"))
                        and pd.notna(primary_confidence)
                        and float(primary_confidence)
                        >= float(settings.get("direct_confirmed_extension_confidence_min", 0.75))
                        and pd.notna(predicted_pips)
                        and abs(float(predicted_pips))
                        >= float(settings.get("direct_confirmed_extension_predicted_pips_min", 4.0))
                    ):
                        new_expires_at = timestamp + timeframe_delta * extension_bars
                        staged.at[active_idx, "expires_at"] = (
                            new_expires_at.isoformat() if pd.notna(new_expires_at) else active_candidate.get("expires_at")
                        )
                        staged.at[active_idx, "extension_count"] = extension_count + 1
                        staged.at[active_idx, "status"] = "ACTIVE"
                        staged.at[active_idx, "status_reason"] = "direct_confirmed_extended_alignment"
                        staged.at[active_idx, "last_evaluated_at"] = now_iso
                        changed = True
                        df_rows.at[idx, "staged_candidate_id"] = active_candidate.get("candidate_id")
                        df_rows.at[idx, "staged_status"] = "ACTIVE"
                        df_rows.at[idx, "staged_action"] = "EXTENDED"
                        df_rows.at[idx, "staged_reason"] = "direct_confirmed_extended_alignment"
                        df_rows.at[idx, "staged_expires_at"] = (
                            new_expires_at.isoformat() if pd.notna(new_expires_at) else pd.NA
                        )
                        self.logger.info(
                            "Staging direct_confirmed extendido por alineacion persistente: model=%s side=%s expires=%s ext=%s",
                            model,
                            candidate_side,
                            new_expires_at,
                            extension_count + 1,
                        )
                        continue
                    cancel_reason = "expired"
                elif (
                    candidate_mode == "direct_confirmed_candle_retrace"
                    and bool(settings.get("direct_confirmed_hold_grace_enabled", True))
                    and hold_grace_count < int(settings.get("direct_confirmed_hold_grace_max_bars", 1) or 1)
                    and primary_signal == "HOLD"
                    and pd.notna(predicted_pips)
                    and abs(float(predicted_pips))
                    >= float(settings.get("direct_confirmed_hold_grace_predicted_pips_min", 3.5))
                    and (
                        (candidate_side == "BUY" and float(predicted_pips) > 0)
                        or (candidate_side == "SELL" and float(predicted_pips) < 0)
                    )
                    and filter_signal in {candidate_side, "HOLD"}
                    and not bool(entry_context_check["hard_contradicted"])
                ):
                    new_expires_at = (
                        timestamp + timeframe_delta if pd.notna(timestamp) else expires_at
                    )
                    staged.at[active_idx, "expires_at"] = (
                        new_expires_at.isoformat() if pd.notna(new_expires_at) else active_candidate.get("expires_at")
                    )
                    staged.at[active_idx, "hold_grace_count"] = hold_grace_count + 1
                    staged.at[active_idx, "status"] = "ACTIVE"
                    staged.at[active_idx, "status_reason"] = "direct_confirmed_hold_grace"
                    staged.at[active_idx, "last_evaluated_at"] = now_iso
                    changed = True
                    df_rows.at[idx, "staged_candidate_id"] = active_candidate.get("candidate_id")
                    df_rows.at[idx, "staged_status"] = "ACTIVE"
                    df_rows.at[idx, "staged_action"] = "EXTENDED"
                    df_rows.at[idx, "staged_reason"] = "direct_confirmed_hold_grace"
                    df_rows.at[idx, "staged_expires_at"] = (
                        new_expires_at.isoformat() if pd.notna(new_expires_at) else pd.NA
                    )
                    self.logger.info(
                        "Staging direct_confirmed retenido por HOLD neutro: model=%s side=%s expires=%s grace=%s pips=%.2f",
                        model,
                        candidate_side,
                        new_expires_at,
                        hold_grace_count + 1,
                        float(predicted_pips),
                    )
                    continue
                elif (
                    candidate_mode == "medium_primary_filter_hold_candle_retrace"
                    and bool(settings.get("medium_primary_hold_grace_enabled", True))
                    and hold_grace_count
                    < int(settings.get("medium_primary_hold_grace_max_bars", 4) or 4)
                    and primary_signal == "HOLD"
                    and pd.notna(predicted_pips)
                    and abs(float(predicted_pips))
                    >= float(
                        settings.get(
                            "medium_primary_hold_grace_predicted_pips_min",
                            settings.get("medium_primary_hold_predicted_pips_min", 3.6),
                        )
                    )
                    and (
                        (candidate_side == "BUY" and float(predicted_pips) > 0)
                        or (candidate_side == "SELL" and float(predicted_pips) < 0)
                    )
                    and not bool(entry_context_check["hard_contradicted"])
                ):
                    new_expires_at = (
                        timestamp + timeframe_delta if pd.notna(timestamp) else expires_at
                    )
                    adaptive_retrace_fraction_raw = pd.to_numeric(
                        pd.Series([active_candidate.get("adaptive_retrace_fraction")]),
                        errors="coerce",
                    ).iloc[0]
                    adaptive_retrace_fraction = (
                        float(adaptive_retrace_fraction_raw)
                        if pd.notna(adaptive_retrace_fraction_raw)
                        else float(settings.get("medium_primary_hold_retrace_fraction", 0.25))
                    )
                    previous_trigger_price = pd.to_numeric(
                        pd.Series([active_candidate.get("trigger_price")]),
                        errors="coerce",
                    ).iloc[0]
                    previous_breakout_trigger_price = pd.to_numeric(
                        pd.Series([active_candidate.get("breakout_trigger_price")]),
                        errors="coerce",
                    ).iloc[0]
                    reevaluated_retrace = None
                    reevaluated_breakout_trigger_price = float("nan")
                    reevaluated_status_reason = "medium_primary_hold_grace_retained"
                    reevaluated_staged_action = "EXTENDED"
                    reference_price = (
                        _current_exec_price_for(candidate_side)
                        if candidate_side in {"BUY", "SELL"}
                        else candidate_exec_price
                    )
                    if candidate_side in {"BUY", "SELL"} and pd.notna(reference_price):
                        reevaluated_retrace = self._build_candle_retrace_candidate(
                            side=candidate_side,
                            candle_high=candle_high,
                            candle_low=candle_low,
                            candle_close=candle_close,
                            reference_price=reference_price,
                            atr_value=runtime_ctx.get("atr_value"),
                            pip_size=pip_size,
                            digits=int(max(runtime_ctx.get("digits", 5), 0)),
                            retrace_fraction=adaptive_retrace_fraction,
                            stop_buffer_pips=float(settings["direct_filter_hold_stop_buffer_pips"]),
                            stop_buffer_atr_fraction=float(settings["dynamic_stop_atr_fraction"]),
                            min_stop_pips=float(settings["dynamic_stop_min_pips"]),
                        )
                    if reevaluated_retrace is not None and pd.notna(reference_price):
                        reevaluated_breakout_trigger_price = (
                            self._compute_breakout_trigger_price(
                                side=candidate_side,
                                reference_price=reevaluated_retrace["reference_price"],
                                reference_stop_pips=reevaluated_retrace["reference_stop_pips"],
                                pip_size=pip_size,
                                settings=settings,
                            )
                            if float(candidate_breakout_fraction) > 0
                            else float("nan")
                        )
                        active_compare_price = self._candidate_comparison_price(active_candidate)
                        new_compare_price = self._candidate_comparison_price(reevaluated_retrace)
                        if not self._is_more_favorable_entry_price(
                            side=candidate_side,
                            candidate_price=active_compare_price,
                            reference_price=new_compare_price,
                        ):
                            staged.at[active_idx, "reference_price"] = reevaluated_retrace["reference_price"]
                            staged.at[active_idx, "trigger_price"] = reevaluated_retrace["trigger_price"]
                            staged.at[active_idx, "breakout_trigger_price"] = reevaluated_breakout_trigger_price
                            staged.at[active_idx, "reference_stop_pips"] = reevaluated_retrace["reference_stop_pips"]
                            staged.at[active_idx, "entry_improvement_pips"] = reevaluated_retrace["entry_improvement_pips"]
                            staged.at[active_idx, "custom_stop_price"] = reevaluated_retrace["custom_stop_price"]
                            staged.at[active_idx, "signal_candle_open"] = candle_open
                            staged.at[active_idx, "signal_candle_high"] = candle_high
                            staged.at[active_idx, "signal_candle_low"] = candle_low
                            staged.at[active_idx, "signal_candle_close"] = candle_close
                            reevaluated_status_reason = "medium_primary_hold_grace_rebuilt"
                            reevaluated_staged_action = "REBUILT"
                        else:
                            reevaluated_status_reason = (
                                "medium_primary_hold_grace_retained_better_existing"
                            )
                            reevaluated_staged_action = "REUSED_ACTIVE"
                    staged.at[active_idx, "expires_at"] = (
                        new_expires_at.isoformat() if pd.notna(new_expires_at) else active_candidate.get("expires_at")
                    )
                    staged.at[active_idx, "hold_grace_count"] = hold_grace_count + 1
                    staged.at[active_idx, "predicted_pips"] = predicted_pips
                    staged.at[active_idx, "pred_return"] = row.get("pred_return")
                    staged.at[active_idx, "primary_confidence"] = primary_confidence
                    staged.at[active_idx, "filter_signal"] = filter_signal
                    staged.at[active_idx, "filter_support_score"] = support_score
                    staged.at[active_idx, "filter_contradicted"] = filter_contradicted
                    staged.at[active_idx, "status"] = "ACTIVE"
                    staged.at[active_idx, "status_reason"] = reevaluated_status_reason
                    staged.at[active_idx, "last_evaluated_at"] = now_iso
                    staged.at[active_idx, "refresh_action"] = reevaluated_staged_action
                    staged.at[active_idx, "refresh_reason"] = reevaluated_status_reason
                    staged.at[active_idx, "refresh_trigger_price_prev"] = previous_trigger_price
                    staged.at[active_idx, "refresh_trigger_price_new"] = staged.at[active_idx, "trigger_price"]
                    staged.at[active_idx, "refresh_breakout_trigger_price_prev"] = previous_breakout_trigger_price
                    staged.at[active_idx, "refresh_breakout_trigger_price_new"] = staged.at[
                        active_idx, "breakout_trigger_price"
                    ]
                    changed = True
                    df_rows.at[idx, "staged_candidate_id"] = active_candidate.get("candidate_id")
                    df_rows.at[idx, "staged_status"] = "ACTIVE"
                    df_rows.at[idx, "staged_action"] = reevaluated_staged_action
                    df_rows.at[idx, "staged_reason"] = reevaluated_status_reason
                    df_rows.at[idx, "staged_expires_at"] = (
                        new_expires_at.isoformat() if pd.notna(new_expires_at) else pd.NA
                    )
                    df_rows.at[idx, "staged_reference_price"] = staged.at[active_idx, "reference_price"]
                    df_rows.at[idx, "staged_trigger_price"] = staged.at[active_idx, "trigger_price"]
                    df_rows.at[idx, "staged_breakout_trigger_price"] = staged.at[
                        active_idx, "breakout_trigger_price"
                    ]
                    df_rows.at[idx, "staged_trigger_price_prev"] = previous_trigger_price
                    df_rows.at[idx, "staged_trigger_price_new"] = staged.at[active_idx, "trigger_price"]
                    df_rows.at[idx, "staged_breakout_trigger_price_prev"] = previous_breakout_trigger_price
                    df_rows.at[idx, "staged_breakout_trigger_price_new"] = staged.at[
                        active_idx, "breakout_trigger_price"
                    ]
                    df_rows.at[idx, "staged_refresh_action"] = reevaluated_staged_action
                    df_rows.at[idx, "staged_refresh_reason"] = reevaluated_status_reason
                    self.logger.info(
                        "Staging medium_primary_filter_hold retenido por HOLD neutro: model=%s side=%s expires=%s grace=%s action=%s reason=%s trigger_prev=%s trigger_new=%s pips=%.2f",
                        model,
                        candidate_side,
                        new_expires_at,
                        hold_grace_count + 1,
                        reevaluated_staged_action,
                        reevaluated_status_reason,
                        previous_trigger_price,
                        staged.at[active_idx, "trigger_price"],
                        float(predicted_pips),
                    )
                    continue
                elif (
                    candidate_mode == "direct_filter_hold_candle_retrace"
                    and bool(settings.get("direct_filter_hold_hold_grace_enabled", True))
                    and hold_grace_count
                    < int(settings.get("direct_filter_hold_hold_grace_max_bars", 4) or 4)
                    and primary_signal == "HOLD"
                    and pd.notna(predicted_pips)
                    and abs(float(predicted_pips))
                    >= float(
                        settings.get(
                            "direct_filter_hold_hold_grace_predicted_pips_min",
                            settings.get("direct_filter_hold_activation_predicted_pips_min", 3.5),
                        )
                    )
                    and (
                        (candidate_side == "BUY" and float(predicted_pips) > 0)
                        or (candidate_side == "SELL" and float(predicted_pips) < 0)
                    )
                    and (
                        pd.isna(primary_confidence)
                        or float(primary_confidence)
                        >= float(settings.get("direct_filter_hold_activation_hold_confidence_min", 0.55))
                    )
                    and not bool(entry_context_check["hard_contradicted"])
                ):
                    new_expires_at = (
                        timestamp + timeframe_delta if pd.notna(timestamp) else expires_at
                    )
                    adaptive_retrace_fraction_raw = pd.to_numeric(
                        pd.Series([active_candidate.get("adaptive_retrace_fraction")]),
                        errors="coerce",
                    ).iloc[0]
                    adaptive_retrace_fraction = (
                        float(adaptive_retrace_fraction_raw)
                        if pd.notna(adaptive_retrace_fraction_raw)
                        else float(settings.get("direct_filter_hold_retrace_fraction", 0.50))
                    )
                    previous_trigger_price = pd.to_numeric(
                        pd.Series([active_candidate.get("trigger_price")]),
                        errors="coerce",
                    ).iloc[0]
                    previous_breakout_trigger_price = pd.to_numeric(
                        pd.Series([active_candidate.get("breakout_trigger_price")]),
                        errors="coerce",
                    ).iloc[0]
                    reevaluated_retrace = None
                    reevaluated_breakout_trigger_price = float("nan")
                    reevaluated_status_reason = "direct_filter_hold_grace_retained"
                    reevaluated_staged_action = "EXTENDED"
                    reference_price = (
                        _current_exec_price_for(candidate_side)
                        if candidate_side in {"BUY", "SELL"}
                        else candidate_exec_price
                    )
                    if candidate_side in {"BUY", "SELL"} and pd.notna(reference_price):
                        reevaluated_retrace = self._build_candle_retrace_candidate(
                            side=candidate_side,
                            candle_high=candle_high,
                            candle_low=candle_low,
                            candle_close=candle_close,
                            reference_price=reference_price,
                            atr_value=runtime_ctx.get("atr_value"),
                            pip_size=pip_size,
                            digits=int(max(runtime_ctx.get("digits", 5), 0)),
                            retrace_fraction=adaptive_retrace_fraction,
                            stop_buffer_pips=float(settings["direct_filter_hold_stop_buffer_pips"]),
                            stop_buffer_atr_fraction=float(settings["dynamic_stop_atr_fraction"]),
                            min_stop_pips=float(settings["dynamic_stop_min_pips"]),
                        )
                    if reevaluated_retrace is not None and pd.notna(reference_price):
                        reevaluated_breakout_trigger_price = (
                            self._compute_breakout_trigger_price(
                                side=candidate_side,
                                reference_price=reevaluated_retrace["reference_price"],
                                reference_stop_pips=reevaluated_retrace["reference_stop_pips"],
                                pip_size=pip_size,
                                settings=settings,
                            )
                            if float(candidate_breakout_fraction) > 0
                            else float("nan")
                        )
                        active_compare_price = self._candidate_comparison_price(active_candidate)
                        new_compare_price = self._candidate_comparison_price(reevaluated_retrace)
                        if not self._is_more_favorable_entry_price(
                            side=candidate_side,
                            candidate_price=active_compare_price,
                            reference_price=new_compare_price,
                        ):
                            staged.at[active_idx, "reference_price"] = reevaluated_retrace["reference_price"]
                            staged.at[active_idx, "trigger_price"] = reevaluated_retrace["trigger_price"]
                            staged.at[active_idx, "breakout_trigger_price"] = reevaluated_breakout_trigger_price
                            staged.at[active_idx, "reference_stop_pips"] = reevaluated_retrace["reference_stop_pips"]
                            staged.at[active_idx, "entry_improvement_pips"] = reevaluated_retrace["entry_improvement_pips"]
                            staged.at[active_idx, "custom_stop_price"] = reevaluated_retrace["custom_stop_price"]
                            staged.at[active_idx, "signal_candle_open"] = candle_open
                            staged.at[active_idx, "signal_candle_high"] = candle_high
                            staged.at[active_idx, "signal_candle_low"] = candle_low
                            staged.at[active_idx, "signal_candle_close"] = candle_close
                            reevaluated_status_reason = "direct_filter_hold_grace_rebuilt"
                            reevaluated_staged_action = "REBUILT"
                        else:
                            reevaluated_status_reason = (
                                "direct_filter_hold_grace_retained_better_existing"
                            )
                            reevaluated_staged_action = "REUSED_ACTIVE"
                    staged.at[active_idx, "expires_at"] = (
                        new_expires_at.isoformat() if pd.notna(new_expires_at) else active_candidate.get("expires_at")
                    )
                    staged.at[active_idx, "hold_grace_count"] = hold_grace_count + 1
                    staged.at[active_idx, "predicted_pips"] = predicted_pips
                    staged.at[active_idx, "pred_return"] = row.get("pred_return")
                    staged.at[active_idx, "primary_confidence"] = primary_confidence
                    staged.at[active_idx, "filter_signal"] = filter_signal
                    staged.at[active_idx, "filter_support_score"] = support_score
                    staged.at[active_idx, "filter_contradicted"] = filter_contradicted
                    staged.at[active_idx, "status"] = "ACTIVE"
                    staged.at[active_idx, "status_reason"] = reevaluated_status_reason
                    staged.at[active_idx, "last_evaluated_at"] = now_iso
                    staged.at[active_idx, "refresh_action"] = reevaluated_staged_action
                    staged.at[active_idx, "refresh_reason"] = reevaluated_status_reason
                    staged.at[active_idx, "refresh_trigger_price_prev"] = previous_trigger_price
                    staged.at[active_idx, "refresh_trigger_price_new"] = staged.at[active_idx, "trigger_price"]
                    staged.at[active_idx, "refresh_breakout_trigger_price_prev"] = previous_breakout_trigger_price
                    staged.at[active_idx, "refresh_breakout_trigger_price_new"] = staged.at[
                        active_idx, "breakout_trigger_price"
                    ]
                    changed = True
                    df_rows.at[idx, "staged_candidate_id"] = active_candidate.get("candidate_id")
                    df_rows.at[idx, "staged_status"] = "ACTIVE"
                    df_rows.at[idx, "staged_action"] = reevaluated_staged_action
                    df_rows.at[idx, "staged_reason"] = reevaluated_status_reason
                    df_rows.at[idx, "staged_expires_at"] = (
                        new_expires_at.isoformat() if pd.notna(new_expires_at) else pd.NA
                    )
                    df_rows.at[idx, "staged_reference_price"] = staged.at[active_idx, "reference_price"]
                    df_rows.at[idx, "staged_trigger_price"] = staged.at[active_idx, "trigger_price"]
                    df_rows.at[idx, "staged_breakout_trigger_price"] = staged.at[
                        active_idx, "breakout_trigger_price"
                    ]
                    df_rows.at[idx, "staged_trigger_price_prev"] = previous_trigger_price
                    df_rows.at[idx, "staged_trigger_price_new"] = staged.at[active_idx, "trigger_price"]
                    df_rows.at[idx, "staged_breakout_trigger_price_prev"] = previous_breakout_trigger_price
                    df_rows.at[idx, "staged_breakout_trigger_price_new"] = staged.at[
                        active_idx, "breakout_trigger_price"
                    ]
                    df_rows.at[idx, "staged_refresh_action"] = reevaluated_staged_action
                    df_rows.at[idx, "staged_refresh_reason"] = reevaluated_status_reason
                    self.logger.info(
                        "Staging direct_filter_hold retenido por HOLD neutro: model=%s side=%s expires=%s grace=%s action=%s reason=%s trigger_prev=%s trigger_new=%s pips=%.2f",
                        model,
                        candidate_side,
                        new_expires_at,
                        hold_grace_count + 1,
                        reevaluated_staged_action,
                        reevaluated_status_reason,
                        previous_trigger_price,
                        staged.at[active_idx, "trigger_price"],
                        float(predicted_pips),
                    )
                    continue
                elif (
                    candidate_mode == "direct_filter_hold_candle_retrace"
                    and bool(settings.get("direct_filter_hold_revalidate_on_activation", True))
                    and bool(settings.get("direct_filter_hold_cancel_if_primary_mismatch", True))
                    and primary_signal in {"BUY", "SELL"}
                    and primary_signal != candidate_side
                ):
                    cancel_reason = "direct_filter_hold_primary_mismatch"
                elif (
                    candidate_mode == "direct_filter_hold_candle_retrace"
                    and bool(settings.get("direct_filter_hold_revalidate_on_activation", True))
                    and (
                        pd.isna(predicted_pips)
                        or abs(float(predicted_pips))
                        < float(settings.get("direct_filter_hold_activation_predicted_pips_min", 3.5))
                        or (
                            candidate_side == "BUY"
                            and float(predicted_pips) <= 0
                        )
                        or (
                            candidate_side == "SELL"
                            and float(predicted_pips) >= 0
                        )
                    )
                ):
                    cancel_reason = "direct_filter_hold_move_dropped"
                elif (
                    candidate_mode == "direct_filter_hold_candle_retrace"
                    and bool(settings.get("direct_filter_hold_revalidate_on_activation", True))
                    and bool(settings.get("direct_filter_hold_cancel_if_primary_hold_weak", True))
                    and primary_signal == "HOLD"
                    and (
                        pd.isna(primary_confidence)
                        or float(primary_confidence)
                        < float(settings.get("direct_filter_hold_activation_hold_confidence_min", 0.55))
                    )
                ):
                    cancel_reason = "direct_filter_hold_primary_hold_weak"
                elif (
                    candidate_mode == "direct_filter_hold_candle_retrace"
                    and bool(settings.get("direct_filter_hold_cancel_on_soft_context_contradiction", True))
                    and bool(entry_context_check["soft_contradicted"])
                ):
                    cancel_reason = "direct_filter_hold_soft_context_contradicted"
                elif (
                    candidate_mode == "direct_filter_hold_candle_retrace"
                    and bool(settings.get("context_guard_hard_block_direct", True))
                    and bool(entry_context_check["hard_contradicted"])
                ):
                    cancel_reason = "direct_filter_hold_context_contradicted"
                elif (
                    candidate_mode == "direct_confirmed_candle_retrace"
                    and bool(settings["direct_confirmed_revalidate_on_activation"])
                    and bool(settings["direct_confirmed_cancel_if_primary_mismatch"])
                    and primary_signal != candidate_side
                ):
                    cancel_reason = "direct_confirmed_primary_mismatch"
                elif (
                    candidate_mode == "direct_confirmed_candle_retrace"
                    and bool(settings["direct_confirmed_revalidate_on_activation"])
                    and bool(settings["direct_confirmed_cancel_if_confidence_drops"])
                    and (
                        pd.isna(primary_confidence)
                        or float(primary_confidence)
                        < float(settings["direct_confirmed_activation_confidence_min"])
                    )
                ):
                    cancel_reason = "direct_confirmed_confidence_dropped"
                elif (
                    candidate_mode == "direct_confirmed_candle_retrace"
                    and bool(settings["direct_confirmed_revalidate_on_activation"])
                    and bool(settings["direct_confirmed_cancel_if_predicted_move_drops"])
                    and (
                        pd.isna(predicted_pips)
                        or abs(float(predicted_pips))
                        < float(settings["direct_confirmed_activation_predicted_pips_min"])
                    )
                ):
                    cancel_reason = "direct_confirmed_move_dropped"
                elif (
                    candidate_mode == "direct_confirmed_candle_retrace"
                    and bool(settings.get("context_guard_hard_block_direct", True))
                    and bool(entry_context_check["hard_contradicted"])
                ):
                    cancel_reason = "direct_confirmed_context_contradicted"
                elif (
                    candidate_mode == "pilot_candle_retrace"
                    and bool(settings["pilot_revalidate_on_activation"])
                    and bool(settings["pilot_cancel_if_primary_mismatch"])
                    and primary_signal != candidate_side
                ):
                    cancel_reason = "pilot_primary_mismatch"
                elif (
                    candidate_mode == "pilot_candle_retrace"
                    and bool(settings["pilot_revalidate_on_activation"])
                    and bool(settings["pilot_cancel_if_filter_contradicted"])
                    and filter_contradicted
                ):
                    cancel_reason = "pilot_filter_contradicted"
                elif (
                    candidate_mode == "pilot_candle_retrace"
                    and bool(settings["pilot_revalidate_on_activation"])
                    and bool(settings["pilot_cancel_if_confidence_drops"])
                    and (
                        pd.isna(primary_confidence)
                        or float(primary_confidence) < float(settings["pilot_confidence_min"])
                    )
                ):
                    cancel_reason = "pilot_confidence_dropped"
                elif (
                    candidate_mode == "pilot_candle_retrace"
                    and bool(settings["pilot_revalidate_on_activation"])
                    and bool(settings["pilot_cancel_if_predicted_move_drops"])
                    and (
                        pd.isna(predicted_pips)
                        or abs(float(predicted_pips)) < float(settings["min_abs_predicted_pips"])
                    )
                ):
                    cancel_reason = "pilot_move_dropped"
                elif (
                    candidate_mode == "filter_contradiction_aligned_candle_retrace"
                    and not contradiction_stage_check["eligible"]
                ):
                    cancel_reason = str(contradiction_stage_check["reason"] or "contradiction_stage_lost")
                elif (
                    candidate_mode == "strong_primary_filter_hold_candle_retrace"
                    and not strong_primary_hold_check["eligible"]
                ):
                    cancel_reason = str(
                        strong_primary_hold_check["reason"] or "strong_primary_hold_stage_lost"
                    )
                elif candidate_mode == "medium_primary_filter_hold_candle_retrace":
                    medium_primary_hold_reason = str(
                        medium_primary_hold_check["reason"] or "medium_primary_hold_stage_lost"
                    )
                    medium_primary_hold_armed_this_candle = False
                    if (
                        not medium_primary_hold_check["eligible"]
                        and bool(
                            settings.get(
                                "medium_primary_hold_preserve_if_armed_on_confidence_drop",
                                True,
                            )
                        )
                        and medium_primary_hold_reason
                        == "primary_confidence_below_medium_primary_hold_threshold"
                        and candidate_side in {"BUY", "SELL"}
                    ):
                        medium_retrace_touched_this_candle = False
                        if pd.notna(candidate_trigger):
                            if candidate_side == "BUY":
                                medium_retrace_touched_this_candle = candle_low <= candidate_trigger
                            else:
                                medium_retrace_touched_this_candle = candle_high >= candidate_trigger
                        medium_breakout_armed_this_candle = False
                        if pd.notna(candidate_breakout_trigger):
                            if candidate_side == "BUY":
                                medium_breakout_armed_this_candle = candle_high >= candidate_breakout_trigger
                            else:
                                medium_breakout_armed_this_candle = candle_low <= candidate_breakout_trigger
                        medium_primary_hold_armed_this_candle = (
                            medium_retrace_touched_this_candle or medium_breakout_armed_this_candle
                        )
                    if not medium_primary_hold_check["eligible"] and not medium_primary_hold_armed_this_candle:
                        cancel_reason = medium_primary_hold_reason
                    elif not medium_primary_hold_check["eligible"] and medium_primary_hold_armed_this_candle:
                        self.logger.info(
                            "Staging medium_primary_filter_hold retenido pese a caida marginal de confianza: model=%s side=%s reason=%s trigger=%s breakout=%s candle=[%s,%s]",
                            model,
                            candidate_side,
                            medium_primary_hold_reason,
                            candidate_trigger,
                            candidate_breakout_trigger,
                            candle_low,
                            candle_high,
                        )
                elif (
                    candidate_mode == "filter_lead_structural_candle_retrace"
                    and not filter_lead_structural_check["eligible"]
                ):
                    cancel_reason = str(
                        filter_lead_structural_check["reason"] or "filter_lead_structural_stage_lost"
                    )
                elif (
                    candidate_mode == "early_structural_reversal_candle_retrace"
                    and not early_reversal_stage_check["eligible"]
                ):
                    cancel_reason = str(
                        early_reversal_stage_check["reason"] or "early_structural_reversal_stage_lost"
                    )
                elif (
                    settings["cancel_on_opposite_primary_signal"]
                    and primary_signal in {"BUY", "SELL"}
                    and candidate_side in {"BUY", "SELL"}
                    and primary_signal != candidate_side
                ):
                    cancel_reason = "opposite_primary_signal"
                elif (
                    settings["cancel_on_filter_contradiction"]
                    and filter_contradicted
                    and candidate_mode not in {
                        "filter_contradiction_aligned_candle_retrace",
                        "early_structural_reversal_candle_retrace",
                        "strong_primary_filter_hold_candle_retrace",
                        "medium_primary_filter_hold_candle_retrace",
                        "filter_lead_structural_candle_retrace",
                    }
                ):
                    cancel_reason = "filter_contradiction"

                if cancel_reason:
                    cancelled_trigger_price = pd.to_numeric(
                        pd.Series([active_candidate.get("trigger_price")]),
                        errors="coerce",
                    ).iloc[0]
                    cancelled_breakout_trigger_price = pd.to_numeric(
                        pd.Series([active_candidate.get("breakout_trigger_price")]),
                        errors="coerce",
                    ).iloc[0]
                    staged.at[active_idx, "status"] = "CANCELLED"
                    staged.at[active_idx, "cancel_timestamp"] = now_iso
                    staged.at[active_idx, "cancel_reason"] = cancel_reason
                    staged.at[active_idx, "status_reason"] = cancel_reason
                    staged.at[active_idx, "last_evaluated_at"] = now_iso
                    staged.at[active_idx, "refresh_action"] = "CANCELLED"
                    staged.at[active_idx, "refresh_reason"] = cancel_reason
                    staged.at[active_idx, "refresh_trigger_price_prev"] = cancelled_trigger_price
                    staged.at[active_idx, "refresh_trigger_price_new"] = cancelled_trigger_price
                    staged.at[active_idx, "refresh_breakout_trigger_price_prev"] = cancelled_breakout_trigger_price
                    staged.at[active_idx, "refresh_breakout_trigger_price_new"] = cancelled_breakout_trigger_price
                    changed = True
                    df_rows.at[idx, "staged_candidate_id"] = active_candidate.get("candidate_id")
                    df_rows.at[idx, "staged_status"] = "CANCELLED"
                    df_rows.at[idx, "staged_action"] = "CANCELLED"
                    df_rows.at[idx, "staged_reason"] = cancel_reason
                    df_rows.at[idx, "staged_trigger_price_prev"] = cancelled_trigger_price
                    df_rows.at[idx, "staged_trigger_price_new"] = cancelled_trigger_price
                    df_rows.at[idx, "staged_breakout_trigger_price_prev"] = cancelled_breakout_trigger_price
                    df_rows.at[idx, "staged_breakout_trigger_price_new"] = cancelled_breakout_trigger_price
                    df_rows.at[idx, "staged_refresh_action"] = "CANCELLED"
                    df_rows.at[idx, "staged_refresh_reason"] = cancel_reason
                    active_idx = None
                    active_candidate = None
                else:
                    if (
                        candidate_mode == "medium_primary_filter_hold_candle_retrace"
                        and pd.notna(timestamp)
                    ):
                        activation_delay_bars = max(
                            int(settings.get("medium_primary_hold_activation_delay_bars", 1) or 0),
                            0,
                        )
                        if activation_delay_bars > 0 and pd.notna(candidate_source_timestamp):
                            activation_eligible_at = candidate_source_timestamp + timeframe_delta * activation_delay_bars
                            if pd.notna(activation_eligible_at) and timestamp < activation_eligible_at:
                                staged.at[active_idx, "status"] = "ACTIVE"
                                staged.at[active_idx, "status_reason"] = (
                                    "medium_primary_hold_activation_delay_waiting"
                                )
                                staged.at[active_idx, "last_evaluated_at"] = now_iso
                                changed = True
                                df_rows.at[idx, "staged_candidate_id"] = active_candidate.get("candidate_id")
                                df_rows.at[idx, "staged_status"] = "ACTIVE"
                                df_rows.at[idx, "staged_action"] = "WAITING_DELAY"
                                df_rows.at[idx, "staged_reason"] = (
                                    "medium_primary_hold_activation_delay_waiting"
                                )
                                df_rows.at[idx, "staged_reference_price"] = active_candidate.get("reference_price")
                                df_rows.at[idx, "staged_trigger_price"] = active_candidate.get("trigger_price")
                                df_rows.at[idx, "staged_breakout_trigger_price"] = active_candidate.get(
                                    "breakout_trigger_price"
                                )
                                df_rows.at[idx, "staged_expires_at"] = active_candidate.get("expires_at")
                                continue

                        previous_trigger_price = np.nan
                        previous_breakout_trigger_price = np.nan
                        stage_refresh_action = pd.NA
                        stage_refresh_reason = pd.NA
                        if (
                            bool(settings.get("medium_primary_hold_rebuild_on_activation", True))
                            and candidate_side in {"BUY", "SELL"}
                            and pd.notna(candidate_exec_price)
                        ):
                            previous_trigger_price = pd.to_numeric(
                                pd.Series([active_candidate.get("trigger_price")]),
                                errors="coerce",
                            ).iloc[0]
                            previous_breakout_trigger_price = pd.to_numeric(
                                pd.Series([active_candidate.get("breakout_trigger_price")]),
                                errors="coerce",
                            ).iloc[0]
                            stage_refresh_action = "REUSED_ACTIVE"
                            stage_refresh_reason = "activation_reuse_existing"
                            adaptive_retrace_fraction_raw = pd.to_numeric(
                                pd.Series([active_candidate.get("adaptive_retrace_fraction")]),
                                errors="coerce",
                            ).iloc[0]
                            adaptive_retrace_fraction = (
                                float(adaptive_retrace_fraction_raw)
                                if pd.notna(adaptive_retrace_fraction_raw)
                                else float(settings.get("medium_primary_hold_retrace_fraction", 0.25))
                            )
                            reevaluated_retrace = self._build_candle_retrace_candidate(
                                side=candidate_side,
                                candle_high=candle_high,
                                candle_low=candle_low,
                                candle_close=candle_close,
                                reference_price=candidate_exec_price,
                                atr_value=runtime_ctx.get("atr_value"),
                                pip_size=pip_size,
                                digits=int(max(runtime_ctx.get("digits", 5), 0)),
                                retrace_fraction=adaptive_retrace_fraction,
                                stop_buffer_pips=float(settings["direct_filter_hold_stop_buffer_pips"]),
                                stop_buffer_atr_fraction=float(settings["dynamic_stop_atr_fraction"]),
                                min_stop_pips=float(settings["dynamic_stop_min_pips"]),
                            )
                            if reevaluated_retrace is not None:
                                reevaluated_breakout_trigger_price = (
                                    self._compute_breakout_trigger_price(
                                        side=candidate_side,
                                        reference_price=reevaluated_retrace["reference_price"],
                                        reference_stop_pips=reevaluated_retrace["reference_stop_pips"],
                                        pip_size=pip_size,
                                        settings=settings,
                                    )
                                    if candidate_breakout_fraction > 0
                                    else float("nan")
                                )
                                active_compare_price = self._candidate_comparison_price(active_candidate)
                                new_compare_price = self._candidate_comparison_price(reevaluated_retrace)
                                if not self._is_more_favorable_entry_price(
                                    side=candidate_side,
                                    candidate_price=active_compare_price,
                                    reference_price=new_compare_price,
                                ):
                                    staged.at[active_idx, "reference_price"] = reevaluated_retrace["reference_price"]
                                    staged.at[active_idx, "trigger_price"] = reevaluated_retrace["trigger_price"]
                                    staged.at[active_idx, "breakout_trigger_price"] = (
                                        reevaluated_breakout_trigger_price
                                    )
                                    staged.at[active_idx, "reference_stop_pips"] = reevaluated_retrace[
                                        "reference_stop_pips"
                                    ]
                                    staged.at[active_idx, "entry_improvement_pips"] = reevaluated_retrace[
                                        "entry_improvement_pips"
                                    ]
                                    staged.at[active_idx, "custom_stop_price"] = reevaluated_retrace[
                                        "custom_stop_price"
                                    ]
                                    staged.at[active_idx, "signal_candle_open"] = candle_open
                                    staged.at[active_idx, "signal_candle_high"] = candle_high
                                    staged.at[active_idx, "signal_candle_low"] = candle_low
                                    staged.at[active_idx, "signal_candle_close"] = candle_close
                                    staged.at[active_idx, "last_evaluated_at"] = now_iso
                                    changed = True
                                    stage_refresh_action = "REBUILT"
                                    stage_refresh_reason = "activation_rebuilt_trigger"
                                    active_candidate = staged.loc[active_idx].to_dict()
                                    candidate_trigger = float(reevaluated_retrace["trigger_price"])
                                    candidate_breakout_trigger = (
                                        float(reevaluated_breakout_trigger_price)
                                        if pd.notna(reevaluated_breakout_trigger_price)
                                        else float("nan")
                                    )
                                    candidate_stop_price = float(reevaluated_retrace["custom_stop_price"])
                                    df_rows.at[idx, "staged_reference_price"] = reevaluated_retrace[
                                        "reference_price"
                                    ]
                                    df_rows.at[idx, "staged_trigger_price"] = reevaluated_retrace["trigger_price"]
                                    df_rows.at[idx, "staged_breakout_trigger_price"] = (
                                        reevaluated_breakout_trigger_price
                                    )
                                    df_rows.at[idx, "staged_entry_improvement_pips"] = reevaluated_retrace[
                                        "entry_improvement_pips"
                                    ]
                                    self.logger.info(
                                        "Staging medium_primary_filter_hold reevaluado antes de activar: model=%s side=%s ref=%s trigger_prev=%s trigger_new=%s stop=%s",
                                        model,
                                        candidate_side,
                                        reevaluated_retrace["reference_price"],
                                        previous_trigger_price,
                                        reevaluated_retrace["trigger_price"],
                                        reevaluated_retrace["custom_stop_price"],
                                    )
                                else:
                                    stage_refresh_action = "REUSED_ACTIVE"
                                    stage_refresh_reason = "activation_reused_better_existing"

                    retrace_triggered = False
                    if candidate_side == "BUY" and pd.notna(candidate_trigger):
                        retrace_triggered = candidate_exec_price <= candidate_trigger
                    elif candidate_side == "SELL" and pd.notna(candidate_trigger):
                        retrace_triggered = candidate_exec_price >= candidate_trigger
                    breakout_triggered = False
                    if not retrace_triggered and candidate_breakout_fraction > 0 and pd.notna(candidate_breakout_trigger):
                        if candidate_side == "BUY":
                            breakout_triggered = candidate_exec_price >= candidate_breakout_trigger
                        elif candidate_side == "SELL":
                            breakout_triggered = candidate_exec_price <= candidate_breakout_trigger
                    directional_volume_check = self._evaluate_directional_volume_activation(
                        signal=candidate_side,
                        feature_row=row,
                        settings=settings,
                    )

                    df_rows.at[idx, "staged_candidate_id"] = active_candidate.get("candidate_id")
                    df_rows.at[idx, "staged_reference_price"] = active_candidate.get("reference_price")
                    df_rows.at[idx, "staged_trigger_price"] = active_candidate.get("trigger_price")
                    df_rows.at[idx, "staged_breakout_trigger_price"] = active_candidate.get("breakout_trigger_price")
                    df_rows.at[idx, "staged_expires_at"] = active_candidate.get("expires_at")
                    df_rows.at[idx, "staged_entry_improvement_pips"] = active_candidate.get("entry_improvement_pips")
                    df_rows.at[idx, "staged_adaptive_profile"] = candidate_adaptive_profile
                    if pd.notna(previous_trigger_price):
                        staged.at[active_idx, "refresh_action"] = stage_refresh_action
                        staged.at[active_idx, "refresh_reason"] = stage_refresh_reason
                        staged.at[active_idx, "refresh_trigger_price_prev"] = previous_trigger_price
                        staged.at[active_idx, "refresh_trigger_price_new"] = active_candidate.get("trigger_price")
                        staged.at[active_idx, "refresh_breakout_trigger_price_prev"] = previous_breakout_trigger_price
                        staged.at[active_idx, "refresh_breakout_trigger_price_new"] = active_candidate.get(
                            "breakout_trigger_price"
                        )
                        df_rows.at[idx, "staged_trigger_price_prev"] = previous_trigger_price
                        df_rows.at[idx, "staged_trigger_price_new"] = active_candidate.get("trigger_price")
                        df_rows.at[idx, "staged_breakout_trigger_price_prev"] = previous_breakout_trigger_price
                        df_rows.at[idx, "staged_breakout_trigger_price_new"] = active_candidate.get(
                            "breakout_trigger_price"
                        )
                        df_rows.at[idx, "staged_refresh_action"] = stage_refresh_action
                        df_rows.at[idx, "staged_refresh_reason"] = stage_refresh_reason
                    df_rows.at[idx, "staged_directional_volume_column"] = directional_volume_check["column"]
                    df_rows.at[idx, "staged_directional_volume_value"] = directional_volume_check["value"]
                    df_rows.at[idx, "staged_directional_volume_passed"] = directional_volume_check["passed"]
                    df_rows.at[idx, "staged_directional_volume_reason"] = directional_volume_check["reason"]
                    staged.at[active_idx, "last_directional_volume_column"] = directional_volume_check["column"]
                    staged.at[active_idx, "last_directional_volume_value"] = directional_volume_check["value"]
                    staged.at[active_idx, "last_directional_volume_passed"] = directional_volume_check["passed"]
                    staged.at[active_idx, "last_directional_volume_reason"] = directional_volume_check["reason"]

                    if retrace_triggered or breakout_triggered:
                        if bool(settings["require_directional_volume_activation"]) and not directional_volume_check["passed"]:
                            df_rows.at[idx, "staged_status"] = "ACTIVE"
                            df_rows.at[idx, "staged_action"] = "WAITING_VOLUME_CONFIRMATION"
                            df_rows.at[idx, "staged_reason"] = directional_volume_check["reason"]
                            staged.at[active_idx, "status_reason"] = directional_volume_check["reason"]
                            staged.at[active_idx, "last_evaluated_at"] = now_iso
                            changed = True
                            continue
                        if bool(settings.get("execution_confirmation_m1_apply_on_stage_activation", True)):
                            m1_execution_check = self._evaluate_entry_execution_confirmation(
                                signal=candidate_side,
                                runtime_ctx=runtime_ctx,
                                settings=settings,
                                purpose="stage_activation",
                            )
                            self._record_entry_execution_confirmation_details(
                                details=m1_execution_check,
                                df_rows=df_rows,
                                row_idx=idx,
                                staged=staged,
                                staged_idx=active_idx,
                            )
                            if not bool(m1_execution_check["passed"]):
                                df_rows.at[idx, "staged_status"] = "ACTIVE"
                                df_rows.at[idx, "staged_action"] = "WAITING_M1_CONFIRMATION"
                                df_rows.at[idx, "staged_reason"] = m1_execution_check["reason"]
                                df_rows.at[idx, "staged_refresh_action"] = "WAITING_M1_CONFIRMATION"
                                df_rows.at[idx, "staged_refresh_reason"] = m1_execution_check["reason"]
                                staged.at[active_idx, "status"] = "ACTIVE"
                                staged.at[active_idx, "status_reason"] = m1_execution_check["reason"]
                                staged.at[active_idx, "last_evaluated_at"] = now_iso
                                staged.at[active_idx, "refresh_action"] = "WAITING_M1_CONFIRMATION"
                                staged.at[active_idx, "refresh_reason"] = m1_execution_check["reason"]
                                changed = True
                                self.logger.info(
                                    "Staging retenido por confirmacion M1: model=%s side=%s reason=%s score=%s hits=%s/%s",
                                    model,
                                    candidate_side,
                                    m1_execution_check.get("reason"),
                                    m1_execution_check.get("score"),
                                    m1_execution_check.get("hits"),
                                    m1_execution_check.get("total"),
                                )
                                continue
                        activation_reason = "retrace_trigger" if retrace_triggered else "breakout_partial"
                        activation_volume_scale = candidate_volume_scale
                        if breakout_triggered and candidate_breakout_fraction > 0:
                            activation_volume_scale = min(candidate_volume_scale, candidate_breakout_fraction)
                        payload_kwargs = {
                            "signal": candidate_side,
                            "predicted_pips_signed": active_candidate.get("predicted_pips"),
                            "signal_time": timestamp if pd.notna(timestamp) else row.get("timestamp"),
                            "runtime_ctx": runtime_ctx,
                        }
                        if candidate_mode in {
                            "direct_filter_hold_candle_retrace",
                            "direct_confirmed_candle_retrace",
                            "pilot_candle_retrace",
                            "filter_contradiction_aligned_candle_retrace",
                            "strong_primary_filter_hold_candle_retrace",
                            "medium_primary_filter_hold_candle_retrace",
                            "filter_lead_structural_candle_retrace",
                            "early_structural_reversal_candle_retrace",
                        } and pd.notna(candidate_stop_price):
                            payload_kwargs.update(
                                {
                                    "explicit_sl_price": candidate_stop_price,
                                    "disable_entry_management": True,
                                    "market_only_comment": (
                                        "filter_hold_candle_retrace_market_only"
                                        if candidate_mode == "direct_filter_hold_candle_retrace"
                                        else (
                                            "direct_confirmed_candle_retrace_market_only"
                                            if candidate_mode == "direct_confirmed_candle_retrace"
                                            else (
                                                "pilot_candle_retrace_market_only"
                                                if candidate_mode == "pilot_candle_retrace"
                                                else (
                                                    "filter_contradiction_aligned_candle_retrace_market_only"
                                                    if candidate_mode == "filter_contradiction_aligned_candle_retrace"
                                                    else (
                                                        "strong_primary_filter_hold_candle_retrace_market_only"
                                                        if candidate_mode == "strong_primary_filter_hold_candle_retrace"
                                                        else (
                                                            "medium_primary_filter_hold_candle_retrace_market_only"
                                                            if candidate_mode == "medium_primary_filter_hold_candle_retrace"
                                                            else (
                                                                "filter_lead_structural_candle_retrace_market_only"
                                                                if candidate_mode == "filter_lead_structural_candle_retrace"
                                                                else "early_structural_reversal_candle_retrace_market_only"
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    ),
                                    "volume_scale": activation_volume_scale,
                                }
                            )
                            activation_reason = (
                                "filter_hold_candle_retrace"
                                if candidate_mode == "direct_filter_hold_candle_retrace"
                                else (
                                    "direct_confirmed_candle_retrace"
                                    if candidate_mode == "direct_confirmed_candle_retrace"
                                    else (
                                        "pilot_candle_retrace"
                                        if candidate_mode == "pilot_candle_retrace"
                                        else (
                                            "filter_contradiction_aligned_candle_retrace"
                                            if candidate_mode == "filter_contradiction_aligned_candle_retrace"
                                            else (
                                                "strong_primary_filter_hold_candle_retrace"
                                                if candidate_mode == "strong_primary_filter_hold_candle_retrace"
                                                else (
                                                    "medium_primary_filter_hold_candle_retrace"
                                                    if candidate_mode == "medium_primary_filter_hold_candle_retrace"
                                                    else (
                                                        "filter_lead_structural_candle_retrace"
                                                        if candidate_mode == "filter_lead_structural_candle_retrace"
                                                        else "early_structural_reversal_candle_retrace"
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            if breakout_triggered:
                                activation_reason = f"{activation_reason}_breakout_partial"
                        activation_payload = self._compute_runtime_trade_payload_for_signal(**payload_kwargs)
                        if activation_payload and float(activation_payload.get("volume_lots") or 0.0) > 0:
                            df_rows.at[idx, "signal"] = candidate_side
                            df_rows.at[idx, "confidence"] = primary_confidence if pd.notna(primary_confidence) else row.get("confidence")
                            df_rows.at[idx, "signal_confirmation_passed"] = True
                            df_rows.at[idx, "signal_confirmation_reason"] = f"entry_staging_{activation_reason}"
                            df_rows.at[idx, "entry_price"] = activation_payload["entry_price"]
                            df_rows.at[idx, "planned_entry_price"] = activation_payload["planned_entry_price"]
                            df_rows.at[idx, "price_target"] = activation_payload["tp_price"]
                            df_rows.at[idx, "delta_price"] = (
                                activation_payload["tp_price"] - current_price
                                if candidate_side == "BUY"
                                else current_price - activation_payload["tp_price"]
                            )
                            df_rows.at[idx, "pips"] = active_candidate.get("predicted_pips")
                            df_rows.at[idx, "expected_move_pips"] = active_candidate.get("predicted_pips")
                            df_rows.at[idx, "signal_target_tp_price"] = activation_payload["signal_target_tp_price"]
                            df_rows.at[idx, "signal_target_sl_price"] = activation_payload["signal_target_sl_price"]
                            df_rows.at[idx, "signal_target_tp_pips"] = activation_payload["signal_target_tp_pips"]
                            df_rows.at[idx, "signal_target_sl_pips"] = activation_payload["signal_target_sl_pips"]
                            df_rows.at[idx, "sl_price"] = activation_payload["sl_price"]
                            df_rows.at[idx, "tp_price"] = activation_payload["tp_price"]
                            df_rows.at[idx, "sl_pips"] = activation_payload["sl_pips"]
                            df_rows.at[idx, "tp_pips"] = activation_payload["tp_pips"]
                            df_rows.at[idx, "market_reference_price"] = activation_payload["market_reference_price"]
                            df_rows.at[idx, "live_entry_price"] = activation_payload["live_entry_price"]
                            df_rows.at[idx, "live_sl_price"] = activation_payload["live_sl_price"]
                            df_rows.at[idx, "live_tp_price"] = activation_payload["live_tp_price"]
                            df_rows.at[idx, "live_sl_pips"] = activation_payload["live_sl_pips"]
                            df_rows.at[idx, "live_tp_pips"] = activation_payload["live_tp_pips"]
                            df_rows.at[idx, "volume_lots"] = activation_payload["volume_lots"]
                            df_rows.at[idx, "risk_amount"] = activation_payload["risk_amount"]
                            df_rows.at[idx, "allocated_risk_budget"] = activation_payload["allocated_risk_budget"]
                            df_rows.at[idx, "risk_per_pip_per_lot"] = activation_payload["risk_per_pip_per_lot"]
                            df_rows.at[idx, "risk_per_lot_at_stop"] = activation_payload["risk_per_lot_at_stop"]
                            df_rows.at[idx, "remaining_risk_budget_before_trade"] = activation_payload["remaining_risk_budget_before_trade"]
                            df_rows.at[idx, "projected_total_open_risk_after_trade"] = activation_payload["projected_total_open_risk_after_trade"]
                            entry_plan = activation_payload["entry_management_plan"]
                            digits_value = pd.to_numeric(
                                pd.Series([runtime_ctx.get("digits", row.get("symbol_digits", 5))]),
                                errors="coerce",
                            ).iloc[0]
                            digits = int(max(float(digits_value), 0.0)) if pd.notna(digits_value) else 5
                            allow_filter_hold_small_market = self._should_allow_filter_hold_small_market(
                                signal=final_signal,
                                row=row,
                                predicted_pips=predicted_pips,
                            )
                            if (
                                 self._get_entry_management_settings()["disable_pending_when_filter_hold"]
                                 and filter_signal == "HOLD"
                                and candidate_mode != "direct_filter_hold_candle_retrace"
                            ):
                                if not allow_filter_hold_small_market:
                                    retrace_only_plan = self._build_retrace_only_entry_plan(
                                        signal=final_signal,
                                        total_volume_lots=float(activation_payload["volume_lots"]),
                                        live_entry_price=float(activation_payload["live_entry_price"]),
                                        live_sl_price=float(activation_payload["live_sl_price"]),
                                        live_tp_price=float(activation_payload["live_tp_price"]),
                                        digits=digits,
                                        timeframe=self._coerce_textish(
                                            row.get("timeframe"),
                                            runtime_ctx.get("timeframe", self.config.get("data", {}).get("timeframe", "M5")),
                                        ),
                                        signal_time=row.get("timestamp", row.get("signal_time", row.get("Time"))),
                                        comment="filter_hold_context_retrace_only",
                                    )
                                    pending_volume = pd.to_numeric(
                                        pd.Series([retrace_only_plan.get("pending_order_volume_lots")]),
                                        errors="coerce",
                                    ).iloc[0]
                                    if pd.notna(pending_volume) and float(pending_volume) > 0:
                                        entry_plan = retrace_only_plan
                                        activation_payload["entry_management_plan"] = entry_plan
                                else:
                                    reduced_market_only = self._apply_reduced_market_only_to_payload(
                                        total_volume_lots=float(activation_payload["volume_lots"]),
                                        risk_amount=float(activation_payload["risk_amount"]),
                                        allocated_risk_budget=float(activation_payload["allocated_risk_budget"]),
                                        projected_total_open_risk_after_trade=float(
                                            activation_payload["projected_total_open_risk_after_trade"]
                                        ),
                                        min_lot=float(runtime_ctx.get("min_lot") or 0.01),
                                        lot_step=float(runtime_ctx.get("lot_step") or runtime_ctx.get("min_lot") or 0.01),
                                        market_fraction=float(
                                            self._get_entry_management_settings()["filter_hold_market_fraction"]
                                        ),
                                        comment="filter_hold_small_market_only",
                                    )
                                    entry_plan = reduced_market_only["entry_plan"]
                                    activation_payload["entry_management_plan"] = entry_plan
                                    activation_payload["volume_lots"] = reduced_market_only["volume_lots"]
                                    activation_payload["risk_amount"] = reduced_market_only["risk_amount"]
                                    activation_payload["allocated_risk_budget"] = reduced_market_only["allocated_risk_budget"]
                                    activation_payload["projected_total_open_risk_after_trade"] = reduced_market_only[
                                        "projected_total_open_risk_after_trade"
                                    ]
                                    activation_payload = self._apply_filter_hold_small_market_level_overrides(
                                        signal=final_signal,
                                        payload=activation_payload,
                                        predicted_pips=predicted_pips,
                                        pip_size=pip_size,
                                        digits=digits,
                                    )
                                    df_rows.at[idx, "signal_target_tp_price"] = activation_payload["signal_target_tp_price"]
                                    df_rows.at[idx, "signal_target_sl_price"] = activation_payload["signal_target_sl_price"]
                                    df_rows.at[idx, "signal_target_tp_pips"] = activation_payload["signal_target_tp_pips"]
                                    df_rows.at[idx, "signal_target_sl_pips"] = activation_payload["signal_target_sl_pips"]
                                    df_rows.at[idx, "sl_price"] = activation_payload["sl_price"]
                                    df_rows.at[idx, "tp_price"] = activation_payload["tp_price"]
                                    df_rows.at[idx, "sl_pips"] = activation_payload["sl_pips"]
                                    df_rows.at[idx, "tp_pips"] = activation_payload["tp_pips"]
                                    df_rows.at[idx, "live_sl_price"] = activation_payload["live_sl_price"]
                                    df_rows.at[idx, "live_tp_price"] = activation_payload["live_tp_price"]
                                    df_rows.at[idx, "live_sl_pips"] = activation_payload["live_sl_pips"]
                                    df_rows.at[idx, "live_tp_pips"] = activation_payload["live_tp_pips"]
                                    df_rows.at[idx, "volume_lots"] = activation_payload["volume_lots"]
                                    df_rows.at[idx, "risk_amount"] = activation_payload["risk_amount"]
                                    df_rows.at[idx, "allocated_risk_budget"] = activation_payload["allocated_risk_budget"]
                                    df_rows.at[idx, "risk_per_lot_at_stop"] = activation_payload["risk_per_lot_at_stop"]
                                    df_rows.at[idx, "projected_total_open_risk_after_trade"] = activation_payload[
                                        "projected_total_open_risk_after_trade"
                                    ]
                            df_rows.at[idx, "entry_management_mode"] = entry_plan["entry_management_mode"]
                            df_rows.at[idx, "entry_management_split_active"] = entry_plan["entry_management_split_active"]
                            df_rows.at[idx, "entry_management_initial_market_fraction"] = entry_plan["entry_management_initial_market_fraction"]
                            df_rows.at[idx, "entry_management_pending_fraction"] = entry_plan["entry_management_pending_fraction"]
                            df_rows.at[idx, "entry_management_retrace_fraction_of_stop"] = entry_plan["entry_management_retrace_fraction_of_stop"]
                            df_rows.at[idx, "entry_management_total_volume_lots"] = entry_plan["entry_management_total_volume_lots"]
                            df_rows.at[idx, "initial_market_volume_lots"] = entry_plan["initial_market_volume_lots"]
                            df_rows.at[idx, "pending_order_volume_lots"] = entry_plan["pending_order_volume_lots"]
                            df_rows.at[idx, "pending_order_price"] = entry_plan["pending_order_price"]
                            df_rows.at[idx, "pending_order_type"] = entry_plan["pending_order_type"]
                            df_rows.at[idx, "pending_order_sl_price"] = entry_plan["pending_order_sl_price"]
                            df_rows.at[idx, "pending_order_tp_price"] = entry_plan["pending_order_tp_price"]
                            df_rows.at[idx, "pending_order_expiry_time"] = entry_plan["pending_order_expiry_time"]
                            df_rows.at[idx, "entry_management_comment"] = entry_plan["entry_management_comment"]
                            df_rows.at[idx, "staged_status"] = "ACTIVATED"
                            df_rows.at[idx, "staged_action"] = "ACTIVATED"
                            df_rows.at[idx, "staged_reason"] = activation_reason
                            df_rows.at[idx, "staged_activation_reason"] = activation_reason
                            staged.at[active_idx, "status"] = "ACTIVATED"
                            staged.at[active_idx, "activation_timestamp"] = now_iso
                            staged.at[active_idx, "activation_price"] = candidate_exec_price
                            staged.at[active_idx, "activation_reason"] = activation_reason
                            staged.at[active_idx, "status_reason"] = activation_reason
                            staged.at[active_idx, "last_evaluated_at"] = now_iso
                            changed = True
                            self.logger.info(
                                "â³ Entrada staged activada: model=%s side=%s trigger=%s current_price=%s reason=%s",
                                model,
                                candidate_side,
                                candidate_trigger,
                                candidate_exec_price,
                                activation_reason,
                            )
                            continue
                        df_rows.at[idx, "staged_status"] = "ACTIVE"
                        df_rows.at[idx, "staged_action"] = "TRIGGERED_NO_VOLUME"
                        df_rows.at[idx, "staged_reason"] = "no_volume_after_trigger"
                    else:
                        df_rows.at[idx, "staged_status"] = "ACTIVE"
                        df_rows.at[idx, "staged_action"] = "WAITING"
                        df_rows.at[idx, "staged_reason"] = "waiting_retrace_or_upgrade"
                    staged.at[active_idx, "last_evaluated_at"] = now_iso
                    changed = True
                    continue

            if final_signal != "HOLD":
                continue

            if primary_signal not in {"BUY", "SELL"}:
                continue
            if pd.isna(primary_confidence) or primary_confidence < float(settings["min_primary_confidence"]):
                continue
            if pd.isna(predicted_pips) or abs(predicted_pips) < float(settings["min_abs_predicted_pips"]):
                continue
            if filter_signal == "HOLD" and not settings["allow_stage_on_filter_hold"]:
                continue
            if settings["block_stage_on_filter_contradiction"] and filter_contradicted and not contradiction_stage_check["eligible"]:
                continue
            if pd.notna(support_score) and support_score < float(settings["soft_support_score_min"]) and not contradiction_stage_check["eligible"]:
                continue
            if active_idx is not None:
                continue

            if contradiction_stage_check["eligible"]:
                reference_price = _current_exec_price_for(primary_signal)
                candidate_retrace = self._build_candle_retrace_candidate(
                    side=primary_signal,
                    candle_high=candle_high,
                    candle_low=candle_low,
                    candle_close=candle_close,
                    reference_price=reference_price,
                    atr_value=runtime_ctx.get("atr_value"),
                    pip_size=pip_size,
                    digits=int(max(runtime_ctx.get("digits", 5), 0)),
                    retrace_fraction=float(contradiction_stage_check["retrace_fraction"]),
                    stop_buffer_pips=float(settings["direct_filter_hold_stop_buffer_pips"]),
                    stop_buffer_atr_fraction=float(settings["dynamic_stop_atr_fraction"]),
                    min_stop_pips=float(settings["dynamic_stop_min_pips"]),
                )
                if candidate_retrace is None or not pd.notna(reference_price):
                    continue
                breakout_trigger_price = (
                    self._compute_breakout_trigger_price(
                        side=primary_signal,
                        reference_price=candidate_retrace["reference_price"],
                        reference_stop_pips=candidate_retrace["reference_stop_pips"],
                        pip_size=pip_size,
                        settings=settings,
                    )
                    if float(contradiction_stage_check["breakout_partial_fraction"]) > 0
                    else float("nan")
                )
                expires_at = (
                    timestamp + timeframe_delta * int(contradiction_stage_check["max_stage_bars"])
                    if pd.notna(timestamp)
                    else pd.NaT
                )
                candidate_id = f"{self._build_signal_id(row)}|FC_STAGE"
                staged = pd.concat(
                    [
                        staged,
                        pd.DataFrame(
                            [
                                {
                                    "candidate_id": candidate_id,
                                    "parent_signal_id": self._build_signal_id(row),
                                    "release_id": row.get("release_id"),
                                    "strategy_profile": strategy_profile,
                                    "symbol": symbol,
                                    "timeframe": timeframe,
                                    "model": model,
                                    "side": primary_signal,
                                    "created_at": now_iso,
                                    "source_timestamp": timestamp.isoformat() if pd.notna(timestamp) else row.get("timestamp"),
                                    "expires_at": expires_at.isoformat() if pd.notna(expires_at) else pd.NA,
                                    "reference_price": candidate_retrace["reference_price"],
                                    "trigger_price": candidate_retrace["trigger_price"],
                                    "breakout_trigger_price": breakout_trigger_price,
                                    "reference_stop_pips": candidate_retrace["reference_stop_pips"],
                                    "entry_improvement_pips": candidate_retrace["entry_improvement_pips"],
                                    "predicted_pips": predicted_pips,
                                    "pred_return": row.get("pred_return"),
                                    "candidate_mode": "filter_contradiction_aligned_candle_retrace",
                                    "candidate_volume_scale": 1.0,
                                    "adaptive_profile": contradiction_stage_check["profile"],
                                    "adaptive_retrace_fraction": contradiction_stage_check["retrace_fraction"],
                                    "adaptive_breakout_fraction": contradiction_stage_check["breakout_partial_fraction"],
                                    "signal_candle_open": candle_open,
                                    "signal_candle_high": candle_high,
                                    "signal_candle_low": candle_low,
                                    "signal_candle_close": candle_close,
                                    "custom_stop_price": candidate_retrace["custom_stop_price"],
                                    "primary_confidence": primary_confidence,
                                    "filter_signal": filter_signal,
                                    "filter_support_score": support_score,
                                    "filter_contradicted": filter_contradicted,
                                    "status": "ACTIVE",
                                    "status_reason": "created_filter_contradiction_aligned_retrace",
                                    "last_evaluated_at": now_iso,
                                    "activation_timestamp": pd.NA,
                                    "activation_price": np.nan,
                                    "activation_reason": pd.NA,
                                    "cancel_timestamp": pd.NA,
                                    "cancel_reason": pd.NA,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
                changed = True
                df_rows.at[idx, "staged_candidate_id"] = candidate_id
                df_rows.at[idx, "staged_status"] = "ACTIVE"
                df_rows.at[idx, "staged_action"] = "CREATED"
                df_rows.at[idx, "staged_reason"] = "filter_contradiction_aligned_wait_candle_retrace"
                df_rows.at[idx, "staged_reference_price"] = candidate_retrace["reference_price"]
                df_rows.at[idx, "staged_trigger_price"] = candidate_retrace["trigger_price"]
                df_rows.at[idx, "staged_breakout_trigger_price"] = breakout_trigger_price
                df_rows.at[idx, "staged_expires_at"] = expires_at.isoformat() if pd.notna(expires_at) else pd.NA
                df_rows.at[idx, "staged_entry_improvement_pips"] = candidate_retrace["entry_improvement_pips"]
                df_rows.at[idx, "staged_adaptive_profile"] = contradiction_stage_check["profile"]
                df_rows.at[idx, "signal_confirmation_reason"] = "filter_contradiction_aligned_wait_candle_retrace"
                self.logger.info(
                    "Ã°Å¸â€¢â€™ Candidata staged por contradiccion alineada: model=%s side=%s close=%s trigger=%s stop=%s",
                    model,
                    primary_signal,
                    candle_close,
                    candidate_retrace["trigger_price"],
                    candidate_retrace["custom_stop_price"],
                )
                continue

            candidate_trade = self._compute_runtime_trade_payload_for_signal(
                signal=primary_signal,
                predicted_pips_signed=predicted_pips,
                signal_time=timestamp if pd.notna(timestamp) else row.get("timestamp"),
                runtime_ctx=runtime_ctx,
            )
            reference_stop_pips = pd.to_numeric(
                pd.Series([candidate_trade.get("live_sl_pips")]),
                errors="coerce",
            ).iloc[0]
            reference_stop_pips = float(reference_stop_pips) if pd.notna(reference_stop_pips) else float("nan")
            reference_price = pd.to_numeric(
                pd.Series([candidate_trade.get("market_reference_price", row.get("price_now"))]),
                errors="coerce",
            ).iloc[0]
            reference_price = float(reference_price) if pd.notna(reference_price) else current_price
            if pd.isna(reference_stop_pips) or reference_stop_pips <= 0 or pip_size <= 0:
                continue

            improvement_pips = max(
                float(settings["min_entry_improvement_pips"]),
                float(settings["retrace_trigger_fraction_of_stop"]) * reference_stop_pips,
            )
            trigger_delta = improvement_pips * pip_size
            trigger_price = (
                reference_price - trigger_delta if primary_signal == "BUY" else reference_price + trigger_delta
            )
            expires_at = (
                timestamp + timeframe_delta * int(settings["max_stage_bars"])
                if pd.notna(timestamp)
                else pd.NaT
            )
            candidate_id = f"{self._build_signal_id(row)}|STAGE"
            staged = pd.concat(
                [
                    staged,
                    pd.DataFrame(
                        [
                            {
                                "candidate_id": candidate_id,
                                "parent_signal_id": self._build_signal_id(row),
                                "release_id": row.get("release_id"),
                                "strategy_profile": strategy_profile,
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "model": model,
                                "side": primary_signal,
                                "created_at": now_iso,
                                "source_timestamp": timestamp.isoformat() if pd.notna(timestamp) else row.get("timestamp"),
                                "expires_at": expires_at.isoformat() if pd.notna(expires_at) else pd.NA,
                                "reference_price": reference_price,
                                "trigger_price": round(float(trigger_price), int(max(runtime_ctx.get("digits", 5), 0))),
                                "reference_stop_pips": reference_stop_pips,
                                "entry_improvement_pips": improvement_pips,
                                "predicted_pips": predicted_pips,
                                "pred_return": row.get("pred_return"),
                                "primary_confidence": primary_confidence,
                                "filter_signal": filter_signal,
                                "filter_support_score": support_score,
                                "filter_contradicted": filter_contradicted,
                                "status": "ACTIVE",
                                "status_reason": "created",
                                "last_evaluated_at": now_iso,
                                "activation_timestamp": pd.NA,
                                "activation_price": np.nan,
                                "activation_reason": pd.NA,
                                "cancel_timestamp": pd.NA,
                                "cancel_reason": pd.NA,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            changed = True
            df_rows.at[idx, "staged_candidate_id"] = candidate_id
            df_rows.at[idx, "staged_status"] = "ACTIVE"
            df_rows.at[idx, "staged_action"] = "CREATED"
            df_rows.at[idx, "staged_reason"] = "candidate_created"
            df_rows.at[idx, "staged_reference_price"] = reference_price
            df_rows.at[idx, "staged_trigger_price"] = round(float(trigger_price), int(max(runtime_ctx.get("digits", 5), 0)))
            df_rows.at[idx, "staged_expires_at"] = expires_at.isoformat() if pd.notna(expires_at) else pd.NA
            df_rows.at[idx, "staged_entry_improvement_pips"] = improvement_pips
            self.logger.info(
                "ðŸ•’ Candidata staged creada: model=%s side=%s reference=%s trigger=%s expires_at=%s",
                model,
                primary_signal,
                reference_price,
                trigger_price,
                expires_at,
            )

        if changed:
            self._save_staged_signal_report(staged)

        return df_rows

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

    def _parse_managed_stage_ids(self, value: Any) -> set[str]:
        if value is None or pd.isna(value):
            return set()
        return {token.strip() for token in str(value).split("|") if token.strip()}

    def _serialize_managed_stage_ids(self, stage_ids: set[str]) -> str:
        if not stage_ids:
            return ""
        return "|".join(sorted(stage_ids))

    def _timeframe_to_timedelta(self, timeframe: str) -> timedelta:
        """Convierte un timeframe tipo M5/H1 a timedelta."""
        tf = str(timeframe or "").strip().upper()
        match = re.fullmatch(r"([MHDW])(\d+)", tf)
        if not match:
            if tf == "MN1":
                return timedelta(days=30)
            return timedelta(minutes=5)

        unit, raw_value = match.groups()
        value = max(int(raw_value or 0), 1)
        if unit == "M":
            return timedelta(minutes=value)
        if unit == "H":
            return timedelta(hours=value)
        if unit == "D":
            return timedelta(days=value)
        if unit == "W":
            return timedelta(weeks=value)
        return timedelta(minutes=5)

    def _normalize_volume_to_step(
        self,
        volume: float,
        *,
        min_lot: float,
        lot_step: float,
    ) -> float:
        volume = float(volume or 0.0)
        min_lot = float(min_lot or 0.0)
        lot_step = float(lot_step or 0.0)
        if volume <= 0:
            return 0.0
        if min_lot <= 0:
            min_lot = 0.01
        if lot_step <= 0:
            lot_step = min_lot

        units = math.floor((volume + 1e-12) / lot_step)
        normalized = units * lot_step
        if normalized + 1e-12 < min_lot:
            return 0.0

        step_str = f"{lot_step:.10f}".rstrip("0")
        precision = len(step_str.split(".")[1]) if "." in step_str else 0
        return float(round(normalized, precision))

    def _compute_entry_management_plan(
        self,
        *,
        signal: str,
        total_volume_lots: float,
        live_entry_price: float,
        live_sl_price: float,
        live_tp_price: float | None,
        digits: int,
        min_lot: float,
        lot_step: float,
        timeframe: str,
        signal_time: Any,
    ) -> dict[str, Any]:
        """Calcula un plan opcional de entrada escalonada: mercado + orden LIMIT de retroceso."""
        settings = self._get_entry_management_settings()
        total_volume_lots = self._normalize_volume_to_step(
            total_volume_lots,
            min_lot=min_lot,
            lot_step=lot_step,
        )
        base_plan = {
            "entry_management_enabled": bool(settings["enabled"]),
            "entry_management_mode": settings["mode"],
            "entry_management_split_active": False,
            "entry_management_initial_market_fraction": settings["initial_market_fraction"],
            "entry_management_pending_fraction": settings["pending_fraction"],
            "entry_management_retrace_fraction_of_stop": settings["retrace_fraction_of_stop"],
            "entry_management_total_volume_lots": total_volume_lots,
            "initial_market_volume_lots": total_volume_lots,
            "pending_order_volume_lots": 0.0,
            "pending_order_price": float("nan"),
            "pending_order_type": None,
            "pending_order_sl_price": float("nan"),
            "pending_order_tp_price": float("nan"),
            "pending_order_expiry_time": None,
            "entry_management_comment": "disabled",
        }

        side = str(signal or "").upper()
        if (
            not settings["enabled"]
            or side not in {"BUY", "SELL"}
            or total_volume_lots <= 0
            or any(pd.isna(value) for value in [live_entry_price, live_sl_price])
        ):
            return base_plan

        stop_distance = abs(float(live_entry_price) - float(live_sl_price))
        tp_distance = (
            abs(float(live_tp_price) - float(live_entry_price))
            if live_tp_price is not None and not pd.isna(live_tp_price)
            else float("nan")
        )
        if stop_distance <= 0:
            base_plan["entry_management_comment"] = "invalid_stop_distance"
            return base_plan

        initial_target = total_volume_lots * float(settings["initial_market_fraction"])
        market_volume = self._normalize_volume_to_step(
            initial_target,
            min_lot=min_lot,
            lot_step=lot_step,
        )
        if market_volume <= 0 and total_volume_lots > min_lot + 1e-12:
            market_volume = self._normalize_volume_to_step(
                min_lot,
                min_lot=min_lot,
                lot_step=lot_step,
            )
        if market_volume <= 0:
            base_plan["entry_management_comment"] = "market_fraction_below_min_lot"
            return base_plan

        pending_volume = self._normalize_volume_to_step(
            max(total_volume_lots - market_volume, 0.0),
            min_lot=min_lot,
            lot_step=lot_step,
        )
        if pending_volume <= 0:
            base_plan["initial_market_volume_lots"] = market_volume
            base_plan["entry_management_comment"] = "no_pending_volume_after_rounding"
            return base_plan

        retrace_fraction = float(settings["retrace_fraction_of_stop"])
        if side == "BUY":
            pending_price = float(live_entry_price) - stop_distance * retrace_fraction
            valid_price = float(live_sl_price) < pending_price < float(live_entry_price)
            pending_type = "BUY_LIMIT"
            pending_sl_price = pending_price - stop_distance
            pending_tp_price = pending_price + tp_distance if not pd.isna(tp_distance) else float("nan")
        else:
            pending_price = float(live_entry_price) + stop_distance * retrace_fraction
            valid_price = float(live_entry_price) < pending_price < float(live_sl_price)
            pending_type = "SELL_LIMIT"
            pending_sl_price = pending_price + stop_distance
            pending_tp_price = pending_price - tp_distance if not pd.isna(tp_distance) else float("nan")

        pending_price = round(float(pending_price), int(max(digits, 0)))
        pending_sl_price = round(float(pending_sl_price), int(max(digits, 0)))
        if not pd.isna(pending_tp_price):
            pending_tp_price = round(float(pending_tp_price), int(max(digits, 0)))
        if not valid_price:
            base_plan["initial_market_volume_lots"] = market_volume
            base_plan["entry_management_comment"] = "invalid_pending_price"
            return base_plan

        expiry_time = None
        signal_ts = pd.to_datetime(signal_time, errors="coerce")
        if pd.notna(signal_ts):
            expiry_time = signal_ts + (
                self._timeframe_to_timedelta(timeframe)
                * int(settings.get("retrace_only_cancel_pending_after_bars", settings["cancel_pending_after_bars"]))
            )

        base_plan.update(
            {
                "entry_management_split_active": True,
                "initial_market_volume_lots": market_volume,
                "pending_order_volume_lots": pending_volume,
                "pending_order_price": pending_price,
                "pending_order_type": pending_type,
                "pending_order_sl_price": pending_sl_price,
                "pending_order_tp_price": pending_tp_price,
                "pending_order_expiry_time": (
                    expiry_time.isoformat() if isinstance(expiry_time, pd.Timestamp) else expiry_time.isoformat()
                )
                if expiry_time is not None
                else None,
                "entry_management_comment": "split_retrace_limit",
            }
        )
        return base_plan

    def _build_immediate_partial_pending_entry_plan(
        self,
        *,
        signal: str,
        total_volume_lots: float,
        market_volume_lots: float,
        live_entry_price: float,
        live_sl_price: float,
        live_tp_price: float | None,
        digits: int,
        min_lot: float,
        lot_step: float,
        timeframe: str,
        signal_time: Any,
        market_only_comment: str,
        pending_comment: str,
    ) -> dict[str, Any]:
        """Construye micro market + pending real para capturar impulsos con retrace corto."""
        settings = self._get_entry_management_settings()
        total_volume_lots = self._normalize_volume_to_step(
            total_volume_lots,
            min_lot=min_lot,
            lot_step=lot_step,
        )
        market_volume_lots = self._normalize_volume_to_step(
            market_volume_lots,
            min_lot=min_lot,
            lot_step=lot_step,
        )
        base_plan = self._build_market_only_entry_plan(
            total_volume_lots=market_volume_lots,
            comment=market_only_comment,
        )

        side = str(signal or "").upper()
        if (
            not settings["enabled"]
            or side not in {"BUY", "SELL"}
            or total_volume_lots <= 0
            or market_volume_lots <= 0
            or any(pd.isna(value) for value in [live_entry_price, live_sl_price])
        ):
            return base_plan

        pending_volume = self._normalize_volume_to_step(
            max(total_volume_lots - market_volume_lots, 0.0),
            min_lot=min_lot,
            lot_step=lot_step,
        )
        if pending_volume <= 0:
            return base_plan

        stop_distance = abs(float(live_entry_price) - float(live_sl_price))
        tp_distance = (
            abs(float(live_tp_price) - float(live_entry_price))
            if live_tp_price is not None and not pd.isna(live_tp_price)
            else float("nan")
        )
        if stop_distance <= 0:
            return base_plan

        retrace_fraction = float(settings["retrace_fraction_of_stop"])
        if side == "BUY":
            pending_price = float(live_entry_price) - stop_distance * retrace_fraction
            valid_price = float(live_sl_price) < pending_price < float(live_entry_price)
            pending_type = "BUY_LIMIT"
            pending_sl_price = pending_price - stop_distance
            pending_tp_price = pending_price + tp_distance if not pd.isna(tp_distance) else float("nan")
        else:
            pending_price = float(live_entry_price) + stop_distance * retrace_fraction
            valid_price = float(live_entry_price) < pending_price < float(live_sl_price)
            pending_type = "SELL_LIMIT"
            pending_sl_price = pending_price + stop_distance
            pending_tp_price = pending_price - tp_distance if not pd.isna(tp_distance) else float("nan")

        pending_price = round(float(pending_price), int(max(digits, 0)))
        pending_sl_price = round(float(pending_sl_price), int(max(digits, 0)))
        if not pd.isna(pending_tp_price):
            pending_tp_price = round(float(pending_tp_price), int(max(digits, 0)))
        if not valid_price:
            return base_plan

        expiry_time = None
        signal_ts = pd.to_datetime(signal_time, errors="coerce")
        if pd.notna(signal_ts):
            expiry_time = signal_ts + (
                self._timeframe_to_timedelta(timeframe)
                * int(settings.get("retrace_only_cancel_pending_after_bars", settings["cancel_pending_after_bars"]))
            )

        market_fraction = (
            min(max(float(market_volume_lots) / float(total_volume_lots), 0.0), 1.0)
            if total_volume_lots > 0
            else 1.0
        )
        pending_fraction = (
            min(max(float(pending_volume) / float(total_volume_lots), 0.0), 1.0)
            if total_volume_lots > 0
            else 0.0
        )

        return {
            "entry_management_enabled": bool(settings["enabled"]),
            "entry_management_mode": settings["mode"],
            "entry_management_split_active": True,
            "entry_management_initial_market_fraction": market_fraction,
            "entry_management_pending_fraction": pending_fraction,
            "entry_management_retrace_fraction_of_stop": settings["retrace_fraction_of_stop"],
            "entry_management_total_volume_lots": total_volume_lots,
            "initial_market_volume_lots": market_volume_lots,
            "pending_order_volume_lots": pending_volume,
            "pending_order_price": pending_price,
            "pending_order_type": pending_type,
            "pending_order_sl_price": pending_sl_price,
            "pending_order_tp_price": pending_tp_price,
            "pending_order_expiry_time": (
                expiry_time.isoformat() if expiry_time is not None else None
            ),
            "entry_management_comment": pending_comment,
        }

    def _build_market_only_entry_plan(
        self,
        *,
        total_volume_lots: float,
        comment: str,
    ) -> dict[str, Any]:
        """Construye un plan sin pierna pending, usando todo el lote en mercado."""
        settings = self._get_entry_management_settings()
        total_volume_lots = float(total_volume_lots or 0.0)
        return {
            "entry_management_enabled": False,
            "entry_management_mode": settings["mode"],
            "entry_management_split_active": False,
            "entry_management_initial_market_fraction": 1.0,
            "entry_management_pending_fraction": 0.0,
            "entry_management_retrace_fraction_of_stop": settings["retrace_fraction_of_stop"],
            "entry_management_total_volume_lots": total_volume_lots,
            "initial_market_volume_lots": total_volume_lots,
            "pending_order_volume_lots": 0.0,
            "pending_order_price": float("nan"),
            "pending_order_type": None,
            "pending_order_sl_price": float("nan"),
            "pending_order_tp_price": float("nan"),
            "pending_order_expiry_time": None,
            "entry_management_comment": comment,
        }

    def _build_reduced_market_only_entry_plan(
        self,
        *,
        total_volume_lots: float,
        min_lot: float,
        lot_step: float,
        market_fraction: float,
        comment: str,
    ) -> dict[str, Any]:
        """Construye un market-only reducido para rutas menos confiables como filter_hold."""
        total_volume_lots = float(total_volume_lots or 0.0)
        market_fraction = min(max(float(market_fraction or 0.0), 0.0), 1.0)
        reduced_volume = self._normalize_volume_to_step(
            total_volume_lots * market_fraction,
            min_lot=float(min_lot or 0.01),
            lot_step=float(lot_step or min_lot or 0.01),
        )
        if reduced_volume <= 0.0 and total_volume_lots > 0.0:
            reduced_volume = self._normalize_volume_to_step(
                min(float(total_volume_lots), float(min_lot or 0.01)),
                min_lot=float(min_lot or 0.01),
                lot_step=float(lot_step or min_lot or 0.01),
            )
        return self._build_market_only_entry_plan(
            total_volume_lots=reduced_volume,
            comment=comment,
        )

    def _apply_reduced_market_only_to_payload(
        self,
        *,
        total_volume_lots: float,
        risk_amount: float,
        allocated_risk_budget: float,
        projected_total_open_risk_after_trade: float,
        min_lot: float,
        lot_step: float,
        market_fraction: float,
        comment: str,
    ) -> dict[str, Any]:
        """Reduce tamano y riesgo cuando una ruta market-only debe ejecutarse con menor agresividad."""
        total_volume_lots = float(total_volume_lots or 0.0)
        risk_amount = float(risk_amount or 0.0)
        allocated_risk_budget = float(allocated_risk_budget or 0.0)
        projected_total_open_risk_after_trade = float(projected_total_open_risk_after_trade or 0.0)

        entry_plan = self._build_reduced_market_only_entry_plan(
            total_volume_lots=total_volume_lots,
            min_lot=min_lot,
            lot_step=lot_step,
            market_fraction=market_fraction,
            comment=comment,
        )
        reduced_volume = float(entry_plan.get("entry_management_total_volume_lots") or 0.0)
        scale = 0.0 if total_volume_lots <= 0.0 else min(max(reduced_volume / total_volume_lots, 0.0), 1.0)
        reduced_risk_amount = risk_amount * scale
        base_open_risk = max(projected_total_open_risk_after_trade - risk_amount, 0.0)
        reduced_projected_open_risk = base_open_risk + reduced_risk_amount
        reduced_allocated_risk_budget = min(allocated_risk_budget * scale, reduced_risk_amount) if scale > 0 else 0.0
        return {
            "entry_plan": entry_plan,
            "volume_lots": reduced_volume,
            "risk_amount": reduced_risk_amount,
            "allocated_risk_budget": reduced_allocated_risk_budget,
            "projected_total_open_risk_after_trade": reduced_projected_open_risk,
        }

    def _should_force_split_retrace_filter_opposite_retrace_only(
        self,
        *,
        signal: str,
        row: pd.Series | dict[str, Any],
        context_check: dict[str, Any] | None = None,
    ) -> bool:
        signal_upper = str(signal or "").strip().upper()
        if signal_upper not in {"BUY", "SELL"}:
            return False

        settings = self._get_entry_management_settings()
        if not bool(settings.get("split_retrace_filter_opposite_retrace_only_enabled", True)):
            return False

        row_get = getattr(row, "get", lambda *_: None)
        entry_comment = self._coerce_textish(row_get("entry_management_comment"), "").strip().lower()
        entry_mode = self._coerce_textish(row_get("entry_management_mode"), "").strip().lower()
        if entry_comment and "split_retrace_limit" not in entry_comment and entry_mode != "split_retrace_limit":
            return False

        filter_signal = self._coerce_textish(row_get("filter_signal"), "").strip().upper()
        if filter_signal not in {"BUY", "SELL"} or filter_signal == signal_upper:
            return False

        candle_open = pd.to_numeric(pd.Series([row_get("signal_candle_open")]), errors="coerce").iloc[0]
        candle_high = pd.to_numeric(pd.Series([row_get("signal_candle_high")]), errors="coerce").iloc[0]
        candle_low = pd.to_numeric(pd.Series([row_get("signal_candle_low")]), errors="coerce").iloc[0]
        candle_close = pd.to_numeric(
            pd.Series([row_get("signal_candle_close", row_get("price_now"))]),
            errors="coerce",
        ).iloc[0]
        if any(pd.isna(value) for value in [candle_open, candle_high, candle_low, candle_close]):
            return False

        candle_range = float(candle_high) - float(candle_low)
        if candle_range <= 0:
            return False

        pip_size = float((self.config.get("data", {}) or {}).get("pip_size", 0.0001) or 0.0001)
        candle_range_pips = abs(candle_range) / pip_size if pip_size > 0 else 0.0
        range_vs_avg_value = self._coerce_feature_value(
            row,
            str(settings.get("split_retrace_filter_opposite_range_vs_avg_column", "RangeVsAvg6")),
        )
        range_vs_avg_ok = (
            range_vs_avg_value is not None
            and not pd.isna(range_vs_avg_value)
            and float(range_vs_avg_value) >= float(settings.get("split_retrace_filter_opposite_range_vs_avg_min", 1.20))
        )
        range_pips_ok = candle_range_pips >= float(settings.get("split_retrace_filter_opposite_range_pips_min", 5.5))
        if not (range_vs_avg_ok or range_pips_ok):
            return False

        opposing_wick = (
            float(candle_high) - max(float(candle_open), float(candle_close))
            if signal_upper == "SELL"
            else min(float(candle_open), float(candle_close)) - float(candle_low)
        )
        wick_ratio = max(opposing_wick, 0.0) / candle_range if candle_range > 0 else 0.0

        market_rejection = (
            bool(context_check.get("market_entry_rejection"))
            if context_check is not None
            else self._coerce_boolish(row_get("entry_context_market_entry_rejection"))
        )
        if market_rejection and bool(settings.get("split_retrace_filter_opposite_rejection_override", True)):
            return True

        return wick_ratio >= float(settings.get("split_retrace_filter_opposite_wick_ratio_min", 0.35))

    def _apply_filter_hold_small_market_level_overrides(
        self,
        *,
        signal: str,
        payload: dict[str, Any],
        predicted_pips: float | None,
        pip_size: float,
        digits: int,
    ) -> dict[str, Any]:
        """Ajusta SL/TP de entradas market reducidas para que respeten la prediccion esperada."""
        settings = self._get_entry_management_settings()
        if not bool(settings.get("filter_hold_small_market_adjust_levels_enabled", True)):
            return payload
        if pip_size <= 0:
            return payload
        if predicted_pips is None or pd.isna(predicted_pips):
            return payload

        target_tp_pips = max(
            abs(float(predicted_pips)),
            float(settings.get("filter_hold_small_market_tp_min_pips", 2.5)),
        )
        if target_tp_pips <= 0:
            return payload

        current_live_sl_pips = pd.to_numeric(pd.Series([payload.get("live_sl_pips")]), errors="coerce").iloc[0]
        current_planned_sl_pips = pd.to_numeric(pd.Series([payload.get("sl_pips")]), errors="coerce").iloc[0]
        if pd.isna(current_live_sl_pips) and pd.isna(current_planned_sl_pips):
            return payload

        current_sl_reference = current_live_sl_pips if pd.notna(current_live_sl_pips) else current_planned_sl_pips
        sl_floor_pips = float(settings.get("filter_hold_small_market_sl_floor_pips", 3.0))
        sl_max_tp_ratio = float(settings.get("filter_hold_small_market_sl_max_tp_ratio", 0.85))
        logical_sl_floor = min(sl_floor_pips, target_tp_pips)
        adjusted_sl_pips = min(
            float(current_sl_reference),
            max(logical_sl_floor, target_tp_pips * sl_max_tp_ratio),
        )
        if adjusted_sl_pips <= 0:
            return payload

        def _price_from_pips(entry_price: float, sl_pips_value: float, tp_pips_value: float) -> tuple[float, float]:
            if str(signal or "").upper() == "BUY":
                sl_price_value = entry_price - sl_pips_value * pip_size
                tp_price_value = entry_price + tp_pips_value * pip_size
            else:
                sl_price_value = entry_price + sl_pips_value * pip_size
                tp_price_value = entry_price - tp_pips_value * pip_size
            return round(float(sl_price_value), digits), round(float(tp_price_value), digits)

        entry_price = float(payload.get("entry_price") or payload.get("planned_entry_price") or 0.0)
        live_entry_price = float(payload.get("live_entry_price") or 0.0)
        market_reference_price = float(payload.get("market_reference_price") or live_entry_price or entry_price)

        planned_sl_price, planned_tp_price = _price_from_pips(entry_price, adjusted_sl_pips, target_tp_pips)
        live_sl_price, live_tp_price = _price_from_pips(live_entry_price, adjusted_sl_pips, target_tp_pips)
        signal_target_sl_price, signal_target_tp_price = _price_from_pips(
            market_reference_price,
            adjusted_sl_pips,
            target_tp_pips,
        )

        payload["signal_target_tp_price"] = signal_target_tp_price
        payload["signal_target_sl_price"] = signal_target_sl_price
        payload["signal_target_tp_pips"] = float(target_tp_pips)
        payload["signal_target_sl_pips"] = float(adjusted_sl_pips)
        payload["sl_price"] = planned_sl_price
        payload["tp_price"] = planned_tp_price
        payload["sl_pips"] = float(adjusted_sl_pips)
        payload["tp_pips"] = float(target_tp_pips)
        payload["live_sl_price"] = live_sl_price
        payload["live_tp_price"] = live_tp_price
        payload["live_sl_pips"] = float(adjusted_sl_pips)
        payload["live_tp_pips"] = float(target_tp_pips)

        risk_per_pip_per_lot = float(payload.get("risk_per_pip_per_lot") or 0.0)
        volume_lots = float(payload.get("volume_lots") or 0.0)
        prior_risk_amount = float(payload.get("risk_amount") or 0.0)
        projected_total_open_risk = float(payload.get("projected_total_open_risk_after_trade") or 0.0)
        allocated_risk_budget = float(payload.get("allocated_risk_budget") or 0.0)
        payload["risk_per_lot_at_stop"] = risk_per_pip_per_lot * float(adjusted_sl_pips)
        if risk_per_pip_per_lot > 0 and volume_lots > 0:
            updated_risk_amount = risk_per_pip_per_lot * float(adjusted_sl_pips) * volume_lots
            base_open_risk = max(projected_total_open_risk - prior_risk_amount, 0.0)
            payload["risk_amount"] = updated_risk_amount
            payload["allocated_risk_budget"] = min(allocated_risk_budget, updated_risk_amount)
            payload["projected_total_open_risk_after_trade"] = base_open_risk + updated_risk_amount

        return payload

    def _build_retrace_only_entry_plan(
        self,
        *,
        signal: str,
        total_volume_lots: float,
        live_entry_price: float,
        live_sl_price: float,
        live_tp_price: float | None,
        digits: int,
        timeframe: str,
        signal_time: Any,
        comment: str,
    ) -> dict[str, Any]:
        """Construye un plan sin pierna market, dejando la tesis solo en retrace LIMIT."""
        settings = self._get_entry_management_settings()
        side = str(signal or "").upper()
        total_volume_lots = float(total_volume_lots or 0.0)
        digits = int(max(digits or 0, 0))
        base_plan = self._build_market_only_entry_plan(total_volume_lots=0.0, comment=comment)
        base_plan.update(
            {
                "entry_management_enabled": False,
                "entry_management_mode": settings["mode"],
                "entry_management_split_active": False,
                "entry_management_initial_market_fraction": 0.0,
                "entry_management_pending_fraction": 1.0,
                "entry_management_retrace_fraction_of_stop": settings["retrace_fraction_of_stop"],
                "entry_management_total_volume_lots": total_volume_lots,
                "initial_market_volume_lots": 0.0,
                "pending_order_volume_lots": 0.0,
                "entry_management_comment": comment,
            }
        )
        if (
            side not in {"BUY", "SELL"}
            or total_volume_lots <= 0
            or any(pd.isna(value) for value in [live_entry_price, live_sl_price])
        ):
            base_plan["entry_management_comment"] = "retrace_only_unavailable"
            return base_plan

        stop_distance = abs(float(live_entry_price) - float(live_sl_price))
        tp_distance = (
            abs(float(live_tp_price) - float(live_entry_price))
            if live_tp_price is not None and not pd.isna(live_tp_price)
            else float("nan")
        )
        if stop_distance <= 0:
            base_plan["entry_management_comment"] = "retrace_only_invalid_stop_distance"
            return base_plan

        retrace_fraction = float(settings["retrace_fraction_of_stop"])
        if side == "BUY":
            pending_price = float(live_entry_price) - stop_distance * retrace_fraction
            valid_price = float(live_sl_price) < pending_price < float(live_entry_price)
            pending_type = "BUY_LIMIT"
            pending_sl_price = pending_price - stop_distance
            pending_tp_price = pending_price + tp_distance if not pd.isna(tp_distance) else float("nan")
        else:
            pending_price = float(live_entry_price) + stop_distance * retrace_fraction
            valid_price = float(live_entry_price) < pending_price < float(live_sl_price)
            pending_type = "SELL_LIMIT"
            pending_sl_price = pending_price + stop_distance
            pending_tp_price = pending_price - tp_distance if not pd.isna(tp_distance) else float("nan")

        pending_price = round(float(pending_price), digits)
        pending_sl_price = round(float(pending_sl_price), digits)
        if not pd.isna(pending_tp_price):
            pending_tp_price = round(float(pending_tp_price), digits)
        if not valid_price:
            base_plan["entry_management_comment"] = "retrace_only_invalid_pending_price"
            return base_plan

        expiry_time = None
        signal_ts = pd.to_datetime(signal_time, errors="coerce")
        if pd.notna(signal_ts):
            expiry_time = signal_ts + (
                self._timeframe_to_timedelta(timeframe) * int(settings["cancel_pending_after_bars"])
            )

        base_plan.update(
            {
                "entry_management_split_active": True,
                "pending_order_volume_lots": total_volume_lots,
                "pending_order_price": pending_price,
                "pending_order_type": pending_type,
                "pending_order_sl_price": pending_sl_price,
                "pending_order_tp_price": pending_tp_price,
                "pending_order_expiry_time": (
                    expiry_time.isoformat() if expiry_time is not None else None
                ),
                "entry_management_comment": comment,
            }
        )
        return base_plan

    def _calculate_break_even_price(
        self,
        *,
        entry_price: float,
        side: str,
        pip_size: float,
        move_sl_to: str,
        buffer_pips: float,
    ) -> float:
        side_upper = str(side or "").upper()
        be_price = float(entry_price)
        if move_sl_to == "breakeven_plus_costs" and pip_size > 0 and buffer_pips > 0:
            buffer_price = float(pip_size) * float(buffer_pips)
            if side_upper == "BUY":
                be_price += buffer_price
            elif side_upper == "SELL":
                be_price -= buffer_price
        return float(be_price)

    def _manage_open_position(
        self,
        *,
        lifecycle: pd.DataFrame,
        idx: int,
        row: pd.Series,
        pos_row: pd.Series,
        mt5_client,
    ) -> bool:
        settings = self._get_trade_management_settings()
        if not settings["enabled"]:
            return False

        side = str(row.get("signal", "")).upper()
        if side not in {"BUY", "SELL"}:
            return False

        symbol = str(pos_row.get("symbol") or row.get("symbol") or "")
        position_value = pd.to_numeric(pd.Series([row.get("mt5_position_id")]), errors="coerce").iloc[0]
        current_volume = pd.to_numeric(pd.Series([pos_row.get("volume")]), errors="coerce").iloc[0]
        entry_price = pd.to_numeric(pd.Series([pos_row.get("price_open")]), errors="coerce").iloc[0]
        price_now = pd.to_numeric(pd.Series([pos_row.get("price_current")]), errors="coerce").iloc[0]
        current_tp = pd.to_numeric(pd.Series([pos_row.get("tp")]), errors="coerce").iloc[0]
        applied_tp = pd.to_numeric(pd.Series([row.get("applied_tp_price")]), errors="coerce").iloc[0]
        requested_tp = pd.to_numeric(pd.Series([row.get("requested_tp_price")]), errors="coerce").iloc[0]
        if pd.notna(current_tp) and float(current_tp) > 0:
            target_tp = current_tp
        elif pd.notna(applied_tp) and float(applied_tp) > 0:
            target_tp = applied_tp
        else:
            target_tp = requested_tp

        if not symbol or pd.isna(position_value):
            return False
        if any(pd.isna(value) for value in [current_volume, entry_price, price_now]):
            return False
        if settings["skip_if_no_tp"] and (pd.isna(target_tp) or abs(float(target_tp)) <= 0.0):
            return False

        if side == "BUY":
            total_distance = float(target_tp) - float(entry_price)
            done_distance = float(price_now) - float(entry_price)
        else:
            total_distance = float(entry_price) - float(target_tp)
            done_distance = float(entry_price) - float(price_now)
        if total_distance <= 0:
            return False

        progress_to_tp = max(float(done_distance) / float(total_distance), 0.0)
        managed_stage_ids = self._parse_managed_stage_ids(row.get("managed_stage_ids"))

        symbol_spec = mt5_client.get_symbol_spec(symbol) or {}
        min_lot = float(symbol_spec.get("volume_min") or 0.01)
        lot_step = float(symbol_spec.get("volume_step") or 0.01)
        pip_size = float(self.config.get("data", {}).get("pip_size", 0.0001) or 0.0001)
        position_id = int(position_value)
        now_iso = datetime.now().isoformat()
        current_volume_estimate = float(current_volume)
        break_even_applied = bool(row.get("break_even_applied"))
        any_change = False
        max_stage_actions = max(int(settings.get("max_stage_actions_per_cycle", 1) or 1), 1)
        actions_applied = 0

        while actions_applied < max_stage_actions:
            pending_stage = None
            for stage in settings["stages"]:
                if settings["only_once_per_stage"] and stage["name"] in managed_stage_ids:
                    continue
                if progress_to_tp + 1e-12 >= float(stage["trigger_progress_to_tp"]):
                    pending_stage = stage
                    break
            if pending_stage is None:
                break

            lifecycle.at[idx, "management_progress_to_tp"] = progress_to_tp
            lifecycle.at[idx, "remaining_volume_lots_estimate"] = float(current_volume_estimate)

            if pending_stage.get("move_sl_to_break_even") and not break_even_applied:
                be_price = self._calculate_break_even_price(
                    entry_price=float(entry_price),
                    side=side,
                    pip_size=pip_size,
                    move_sl_to=settings["move_sl_to"],
                    buffer_pips=float(settings["breakeven_buffer_pips"]),
                )
                protection = mt5_client.ensure_position_protection(
                    symbol=symbol,
                    position_ticket=position_id,
                    side=side,
                    sl=be_price,
                    tp=None if pd.isna(target_tp) else float(target_tp),
                )
                if not protection.get("success"):
                    lifecycle.at[idx, "trade_management_comment"] = (
                        f"Fallo break-even {pending_stage['name']}: {protection.get('comment')}"
                    )
                    return True
                lifecycle.at[idx, "break_even_applied"] = True
                lifecycle.at[idx, "break_even_applied_time"] = now_iso
                lifecycle.at[idx, "break_even_sl_price"] = protection.get("applied_sl")
                lifecycle.at[idx, "applied_sl_price"] = protection.get("applied_sl")
                lifecycle.at[idx, "applied_tp_price"] = protection.get("applied_tp")
                break_even_applied = True

            max_close_volume = float(current_volume_estimate) - float(min_lot)
            desired_close_volume = float(current_volume_estimate) * float(pending_stage["partial_close_fraction"])
            close_volume = self._normalize_volume_to_step(
                min(desired_close_volume, max_close_volume),
                min_lot=min_lot,
                lot_step=lot_step,
            )

            if close_volume <= 0:
                should_full_close_remainder = (
                    bool(settings.get("full_close_when_partial_below_min_enabled", True))
                    and progress_to_tp >= float(settings.get("full_close_when_partial_below_min_progress_to_tp", 0.95))
                )
                full_close_volume = self._normalize_volume_to_step(
                    float(current_volume_estimate),
                    min_lot=min_lot,
                    lot_step=lot_step,
                )
                if should_full_close_remainder and full_close_volume > 0:
                    close_result = mt5_client.close_position_volume(
                        symbol=symbol,
                        position_ticket=position_id,
                        volume=full_close_volume,
                        side=side,
                        comment=f"{settings['comment_prefix']}_NTP_FULL",
                        deviation=self._get_live_trading_settings()["order_deviation_points"],
                    )
                    if not close_result.get("success"):
                        lifecycle.at[idx, "trade_management_comment"] = (
                            f"Fallo cierre total near TP {pending_stage['name']}: {close_result.get('comment')}"
                        )
                        return True

                    managed_stage_ids.add(str(pending_stage["name"]))
                    prev_partial_total = pd.to_numeric(
                        pd.Series([lifecycle.at[idx, "partial_close_total_volume"]]),
                        errors="coerce",
                    ).iloc[0]
                    prev_partial_total = 0.0 if pd.isna(prev_partial_total) else float(prev_partial_total)
                    lifecycle.at[idx, "managed_stage_ids"] = self._serialize_managed_stage_ids(managed_stage_ids)
                    lifecycle.at[idx, "last_management_time"] = now_iso
                    lifecycle.at[idx, "last_management_action"] = (
                        f"{pending_stage['name']}:full_close_remainder"
                    )
                    lifecycle.at[idx, "last_partial_close_volume"] = float(full_close_volume)
                    lifecycle.at[idx, "partial_close_total_volume"] = (
                        prev_partial_total + float(full_close_volume)
                    )
                    lifecycle.at[idx, "remaining_volume_lots_estimate"] = 0.0
                    lifecycle.at[idx, "trade_management_comment"] = (
                        f"Etapa {pending_stage['name']} cerro el remanente completo por progreso >= "
                        f"{float(settings.get('full_close_when_partial_below_min_progress_to_tp', 0.95)) * 100.0:.0f}% y parcial inferior al minimo."
                    )
                    self.logger.info(
                        "Trade management aplicado: symbol=%s position_id=%s stage=%s progress=%.2f%% action=full_close_remainder closed=%.2f",
                        symbol,
                        position_id,
                        pending_stage["name"],
                        progress_to_tp * 100.0,
                        float(full_close_volume),
                    )
                    current_volume_estimate = 0.0
                    any_change = True
                    actions_applied += 1
                    break

                managed_stage_ids.add(str(pending_stage["name"]))
                lifecycle.at[idx, "managed_stage_ids"] = self._serialize_managed_stage_ids(managed_stage_ids)
                lifecycle.at[idx, "last_management_time"] = now_iso
                lifecycle.at[idx, "last_management_action"] = f"{pending_stage['name']}:skip_partial_volume"
                lifecycle.at[idx, "trade_management_comment"] = (
                    f"Etapa {pending_stage['name']} sin cierre parcial: el volumen remanente caeria por debajo del minimo."
                )
                any_change = True
                actions_applied += 1
                continue

            close_result = mt5_client.close_position_volume(
                symbol=symbol,
                position_ticket=position_id,
                volume=close_volume,
                side=side,
                comment=f"{settings['comment_prefix']}_{pending_stage['name']}",
                deviation=self._get_live_trading_settings()["order_deviation_points"],
            )
            if not close_result.get("success"):
                lifecycle.at[idx, "trade_management_comment"] = (
                    f"Fallo cierre parcial {pending_stage['name']}: {close_result.get('comment')}"
                )
                return True

            managed_stage_ids.add(str(pending_stage["name"]))
            prev_partial_total = pd.to_numeric(
                pd.Series([lifecycle.at[idx, "partial_close_total_volume"]]),
                errors="coerce",
            ).iloc[0]
            prev_partial_total = 0.0 if pd.isna(prev_partial_total) else float(prev_partial_total)
            remaining_estimate = max(float(current_volume_estimate) - float(close_volume), 0.0)

            lifecycle.at[idx, "managed_stage_ids"] = self._serialize_managed_stage_ids(managed_stage_ids)
            lifecycle.at[idx, "last_management_time"] = now_iso
            lifecycle.at[idx, "last_management_action"] = f"{pending_stage['name']}:partial_close"
            lifecycle.at[idx, "last_partial_close_volume"] = float(close_volume)
            lifecycle.at[idx, "partial_close_total_volume"] = prev_partial_total + float(close_volume)
            lifecycle.at[idx, "remaining_volume_lots_estimate"] = remaining_estimate
            lifecycle.at[idx, "trade_management_comment"] = (
                f"Etapa {pending_stage['name']} aplicada: cierre parcial de {close_volume:.2f} lotes."
            )

            self.logger.info(
                "Trade management aplicado: symbol=%s position_id=%s stage=%s progress=%.2f%% partial_close=%.2f remaining=%.2f",
                symbol,
                position_id,
                pending_stage["name"],
                progress_to_tp * 100.0,
                float(close_volume),
                remaining_estimate,
            )
            current_volume_estimate = remaining_estimate
            any_change = True
            actions_applied += 1
            if current_volume_estimate <= float(min_lot) + 1e-12:
                break

        return any_change

    def _should_defer_opposite_signal_close_for_child_priority(
        self,
        *,
        lifecycle: pd.DataFrame,
        idx: int,
        row: pd.Series,
        side: str,
        pnl_nonnegative: bool,
        settings: dict[str, Any],
    ) -> tuple[bool, str]:
        if (
            lifecycle is None
            or lifecycle.empty
            or not bool(settings.get("opposite_signal_prioritize_child_positions", True))
            or not pnl_nonnegative
        ):
            return False, ""

        current_entry_leg = self._coerce_textish(row.get("entry_leg"), "market").strip().lower()
        if current_entry_leg == "pending_limit":
            return False, ""

        parent_signal_id = self._coerce_textish(
            row.get("parent_signal_id"),
            row.get("signal_id"),
        ).strip()
        if not parent_signal_id:
            return False, ""

        required_cols = {"parent_signal_id", "entry_leg", "status", "signal"}
        if any(col not in lifecycle.columns for col in required_cols):
            return False, ""

        sibling_mask = (
            pd.Series(lifecycle.index, index=lifecycle.index).ne(idx)
            & lifecycle["parent_signal_id"].astype(str).str.strip().eq(parent_signal_id)
            & lifecycle["entry_leg"].astype(str).str.strip().str.lower().eq("pending_limit")
            & lifecycle["status"].astype(str).str.strip().str.upper().isin(["OPEN", "PENDING_CONFIRMATION"])
            & lifecycle["signal"].astype(str).str.strip().str.upper().eq(side)
        )
        if not bool(sibling_mask.any()):
            return False, ""

        sibling_rows = lifecycle.loc[sibling_mask]
        sibling_ids = (
            pd.to_numeric(sibling_rows.get("mt5_position_id"), errors="coerce")
            if "mt5_position_id" in sibling_rows.columns
            else pd.Series(dtype=float)
        )
        sibling_count = int(sibling_ids.notna().sum()) if not sibling_ids.empty else int(len(sibling_rows))
        detail = (
            "Se difiere cierre por señal opuesta: se prioriza pierna hija abierta "
            f"del mismo cluster ({sibling_count})."
        )
        return True, detail

    def _apply_runtime_monitor_to_position(
        self,
        *,
        lifecycle: pd.DataFrame,
        idx: int,
        row: pd.Series,
        pos_row: pd.Series,
        mt5_client,
        feature_row: pd.Series | dict[str, Any] | None = None,
        current_bar_timestamp: pd.Timestamp | None = None,
    ) -> bool:
        settings = self._get_runtime_monitor_settings()
        if not settings["enabled"]:
            return False

        side = str(row.get("signal", "")).upper()
        if side not in {"BUY", "SELL"}:
            return False

        symbol = str(pos_row.get("symbol") or row.get("symbol") or "")
        position_value = pd.to_numeric(pd.Series([row.get("mt5_position_id")]), errors="coerce").iloc[0]
        current_volume = pd.to_numeric(pd.Series([pos_row.get("volume")]), errors="coerce").iloc[0]
        entry_price = pd.to_numeric(pd.Series([pos_row.get("price_open")]), errors="coerce").iloc[0]
        price_now = pd.to_numeric(pd.Series([pos_row.get("price_current")]), errors="coerce").iloc[0]
        current_tp = pd.to_numeric(pd.Series([pos_row.get("tp")]), errors="coerce").iloc[0]
        current_sl = pd.to_numeric(pd.Series([pos_row.get("sl")]), errors="coerce").iloc[0]
        applied_tp = pd.to_numeric(pd.Series([row.get("applied_tp_price")]), errors="coerce").iloc[0]
        requested_tp = pd.to_numeric(pd.Series([row.get("requested_tp_price")]), errors="coerce").iloc[0]
        signal_time = pd.to_datetime(row.get("signal_time"), errors="coerce")

        if pd.notna(current_tp) and float(current_tp) > 0:
            target_tp = current_tp
        elif pd.notna(applied_tp) and float(applied_tp) > 0:
            target_tp = applied_tp
        else:
            target_tp = requested_tp

        if not symbol or pd.isna(position_value):
            return False
        if any(pd.isna(value) for value in [current_volume, entry_price, price_now]):
            return False
        if pd.isna(target_tp) or abs(float(target_tp)) <= 0.0:
            return False

        timeframe = str(row.get("timeframe") or self.config.get("data", {}).get("timeframe", "M5"))
        timeframe_delta = self._timeframe_to_timedelta(timeframe)
        if pd.notna(signal_time) and current_bar_timestamp is not None and timeframe_delta.total_seconds() > 0:
            bars_open = max(
                int((current_bar_timestamp - signal_time).total_seconds() // timeframe_delta.total_seconds()),
                0,
            )
            if bars_open > int(settings["recent_trade_max_bars"]):
                return False
        else:
            bars_open = 0

        if side == "BUY":
            total_distance = float(target_tp) - float(entry_price)
            done_distance = float(price_now) - float(entry_price)
        else:
            total_distance = float(entry_price) - float(target_tp)
            done_distance = float(entry_price) - float(price_now)
        if total_distance <= 0:
            return False

        progress_to_tp = max(float(done_distance) / float(total_distance), 0.0)
        pnl_nonnegative = done_distance >= 0.0
        break_even_applied = bool(row.get("break_even_applied"))
        latest_signal_snapshot = self._get_latest_runtime_signal_snapshot(
            row=row,
            current_bar_timestamp=current_bar_timestamp,
        )
        opposite_side = "SELL" if side == "BUY" else "BUY"
        managed_stage_ids = self._parse_managed_stage_ids(row.get("managed_stage_ids"))
        opposite_arm_stage_id = f"{settings['comment_prefix']}_ARM_OPPOSITE_{opposite_side}_EXIT"
        opposite_signal_armed = opposite_arm_stage_id in managed_stage_ids
        opposite_signal_active = False
        opposite_reason_detail = "opposite_signal"
        if (
            bool(settings["manage_on_opposite_signal"])
            and isinstance(latest_signal_snapshot, dict)
            and latest_signal_snapshot.get("signal") == opposite_side
        ):
            latest_primary_conf = pd.to_numeric(
                pd.Series([latest_signal_snapshot.get("primary_confidence")]),
                errors="coerce",
            ).iloc[0]
            latest_primary_side = self._coerce_textish(
                latest_signal_snapshot.get("primary_signal"),
                latest_signal_snapshot.get("signal"),
            ).strip().upper()
            latest_age_bars = int(latest_signal_snapshot.get("age_bars") or 0)
            if (
                latest_primary_side == opposite_side
                and pd.notna(latest_primary_conf)
                and float(latest_primary_conf) >= float(settings["opposite_signal_min_primary_confidence"])
                and latest_age_bars <= int(settings["opposite_signal_max_age_bars"])
            ):
                opposite_signal_active = True
                opposite_reason_detail = (
                    f"opposite_signal_{opposite_side.lower()}_conf_{float(latest_primary_conf):.3f}"
                )

        if opposite_signal_active or opposite_signal_armed:
            symbol_spec = mt5_client.get_symbol_spec(symbol) or {}
            min_lot = float(symbol_spec.get("volume_min") or 0.01)
            lot_step = float(symbol_spec.get("volume_step") or 0.01)
            position_id = int(position_value)
            now_iso = datetime.now().isoformat()
            if opposite_signal_armed and not opposite_signal_active:
                opposite_reason_detail = f"armed_opposite_signal_{opposite_side.lower()}_be_exit"

            lifecycle.at[idx, "management_progress_to_tp"] = progress_to_tp
            lifecycle.at[idx, "remaining_volume_lots_estimate"] = float(current_volume)

            if (
                opposite_signal_active
                and not pnl_nonnegative
                and bool(settings.get("opposite_signal_arm_exit_until_break_even", True))
                and not opposite_signal_armed
            ):
                managed_stage_ids.add(opposite_arm_stage_id)
                lifecycle.at[idx, "managed_stage_ids"] = self._serialize_managed_stage_ids(managed_stage_ids)
                lifecycle.at[idx, "last_management_time"] = now_iso
                lifecycle.at[idx, "last_management_action"] = f"{opposite_arm_stage_id}:armed"
                lifecycle.at[idx, "trade_management_comment"] = (
                    f"{settings['comment_prefix']} armó salida first-to-exit por señal opuesta {opposite_side}; "
                    "si el trade vuelve a break-even o positivo, se cerrará completo."
                )
                self.logger.info(
                    "Runtime monitor: symbol=%s position_id=%s action=arm_exit_on_be reason=%s",
                    symbol,
                    position_id,
                    opposite_reason_detail,
                )
                return True

            if pnl_nonnegative and bool(settings["opposite_signal_close_if_nonnegative"]):
                defer_close, defer_detail = self._should_defer_opposite_signal_close_for_child_priority(
                    lifecycle=lifecycle,
                    idx=idx,
                    row=row,
                    side=side,
                    pnl_nonnegative=pnl_nonnegative,
                    settings=settings,
                )
                if defer_close:
                    lifecycle.at[idx, "trade_management_comment"] = defer_detail
                    self.logger.info(
                        "Runtime monitor: symbol=%s position_id=%s action=defer_close reason=%s",
                        symbol,
                        position_id,
                        opposite_reason_detail,
                    )
                    return False

                full_close_stage_id = f"{settings['comment_prefix']}_FULL_OPPOSITE_SIGNAL"
                if full_close_stage_id in managed_stage_ids:
                    return False
                close_volume = self._normalize_volume_to_step(
                    float(current_volume),
                    min_lot=min_lot,
                    lot_step=lot_step,
                )
                if close_volume > 0:
                    close_result = mt5_client.close_position_volume(
                        symbol=symbol,
                        position_ticket=position_id,
                        volume=close_volume,
                        side=side,
                        comment=f"{settings['comment_prefix']}_OPPOSITE"[:31],
                        deviation=self._get_live_trading_settings()["order_deviation_points"],
                    )
                    managed_stage_ids.add(full_close_stage_id)
                    lifecycle.at[idx, "managed_stage_ids"] = self._serialize_managed_stage_ids(managed_stage_ids)
                    lifecycle.at[idx, "last_management_time"] = now_iso
                    if not close_result.get("success"):
                        lifecycle.at[idx, "last_management_action"] = f"{full_close_stage_id}:full_close_failed"
                        lifecycle.at[idx, "trade_management_comment"] = (
                            f"{settings['comment_prefix']} no pudo cerrar por señal opuesta {opposite_side}: {close_result.get('comment')}"
                        )
                        return True

                    prev_partial_total = pd.to_numeric(
                        pd.Series([row.get("partial_close_total_volume")]),
                        errors="coerce",
                    ).iloc[0]
                    prev_partial_total = 0.0 if pd.isna(prev_partial_total) else float(prev_partial_total)
                    lifecycle.at[idx, "last_management_action"] = f"{full_close_stage_id}:full_close"
                    lifecycle.at[idx, "last_partial_close_volume"] = float(close_volume)
                    lifecycle.at[idx, "partial_close_total_volume"] = prev_partial_total + float(close_volume)
                    lifecycle.at[idx, "remaining_volume_lots_estimate"] = 0.0
                    lifecycle.at[idx, "trade_management_comment"] = (
                        f"{settings['comment_prefix']} cerro el trade en positivo por señal opuesta {opposite_side}."
                    )
                    self.logger.info(
                        "Runtime monitor: symbol=%s position_id=%s action=full_close reason=%s progress=%.2f%% closed=%.2f",
                        symbol,
                        position_id,
                        opposite_reason_detail,
                        progress_to_tp * 100.0,
                        float(close_volume),
                    )
                    return True

            if bool(settings["opposite_signal_reduce_to_min_if_losing"]):
                reduce_stage_id = f"{settings['comment_prefix']}_REDUCE_OPPOSITE_SIGNAL"
                if reduce_stage_id in managed_stage_ids:
                    return False
                max_close_volume = float(current_volume) - float(min_lot)
                close_volume = self._normalize_volume_to_step(
                    max_close_volume,
                    min_lot=min_lot,
                    lot_step=lot_step,
                )
                if close_volume > 0:
                    close_result = mt5_client.close_position_volume(
                        symbol=symbol,
                        position_ticket=position_id,
                        volume=close_volume,
                        side=side,
                        comment=f"{settings['comment_prefix']}_REDUCE_OPP"[:31],
                        deviation=self._get_live_trading_settings()["order_deviation_points"],
                    )
                    managed_stage_ids.add(reduce_stage_id)
                    lifecycle.at[idx, "managed_stage_ids"] = self._serialize_managed_stage_ids(managed_stage_ids)
                    lifecycle.at[idx, "last_management_time"] = now_iso
                    if not close_result.get("success"):
                        lifecycle.at[idx, "last_management_action"] = f"{reduce_stage_id}:partial_failed"
                        lifecycle.at[idx, "trade_management_comment"] = (
                            f"{settings['comment_prefix']} no pudo reducir por señal opuesta {opposite_side}: {close_result.get('comment')}"
                        )
                        return True

                    prev_partial_total = pd.to_numeric(
                        pd.Series([row.get("partial_close_total_volume")]),
                        errors="coerce",
                    ).iloc[0]
                    prev_partial_total = 0.0 if pd.isna(prev_partial_total) else float(prev_partial_total)
                    remaining_estimate = max(float(current_volume) - float(close_volume), 0.0)
                    lifecycle.at[idx, "last_management_action"] = f"{reduce_stage_id}:partial_close"
                    lifecycle.at[idx, "last_partial_close_volume"] = float(close_volume)
                    lifecycle.at[idx, "partial_close_total_volume"] = prev_partial_total + float(close_volume)
                    lifecycle.at[idx, "remaining_volume_lots_estimate"] = remaining_estimate
                    lifecycle.at[idx, "trade_management_comment"] = (
                        f"{settings['comment_prefix']} redujo la posicion al minimo por señal opuesta {opposite_side}."
                    )
                    self.logger.info(
                        "Runtime monitor: symbol=%s position_id=%s action=partial_close reason=%s progress=%.2f%% partial=%.2f remaining=%.2f",
                        symbol,
                        position_id,
                        opposite_reason_detail,
                        progress_to_tp * 100.0,
                        float(close_volume),
                        remaining_estimate,
                    )
                    return True

        if not pnl_nonnegative and not break_even_applied:
            return False
        if progress_to_tp < float(settings["min_profit_progress_to_manage"]) and not break_even_applied:
            return False

        roc_value = self._coerce_feature_value(feature_row, settings["momentum_column"])
        short_roc_value = self._coerce_feature_value(feature_row, settings["lateralization_short_momentum_column"])
        adx_value = self._coerce_feature_value(feature_row, settings["regime_column"])
        small_range_value = self._coerce_feature_value(feature_row, settings["lateralization_small_range_column"])
        dirvol_value = self._coerce_feature_value(feature_row, settings["directional_volume_column"])
        close_loc_value = self._coerce_feature_value(feature_row, settings["close_location_column"])

        roc_abs = abs(float(roc_value)) if roc_value is not None else 0.0
        short_roc_abs = abs(float(short_roc_value)) if short_roc_value is not None else 0.0
        adx_abs = abs(float(adx_value)) if adx_value is not None else 0.0
        dirvol_abs = abs(float(dirvol_value)) if dirvol_value is not None else 0.0
        close_loc_abs = abs(float(close_loc_value)) if close_loc_value is not None else 0.0
        roc_aligned = (
            roc_value is not None
            and ((side == "BUY" and float(roc_value) > 0) or (side == "SELL" and float(roc_value) < 0))
        )
        roc_opposite = (
            roc_value is not None
            and ((side == "BUY" and float(roc_value) < 0) or (side == "SELL" and float(roc_value) > 0))
        )
        dirvol_opposite = (
            dirvol_value is not None
            and ((side == "BUY" and float(dirvol_value) < 0) or (side == "SELL" and float(dirvol_value) > 0))
        )
        close_loc_opposite = (
            close_loc_value is not None
            and ((side == "BUY" and float(close_loc_value) < 0) or (side == "SELL" and float(close_loc_value) > 0))
        )

        shock_reversal_market = False
        if bool(settings.get("shock_reversal_enabled", True)):
            range_spike_ok = small_range_value is not None and float(small_range_value) >= float(
                settings["shock_reversal_range_vs_avg_min"]
            )
            dirvol_shock_ok = (
                dirvol_value is not None
                and dirvol_opposite
                and dirvol_abs >= float(settings["shock_reversal_dirvol_abs_min"])
            )
            close_loc_shock_ok = (
                close_loc_value is not None
                and close_loc_opposite
                and close_loc_abs >= float(settings["shock_reversal_close_location_abs_min"])
            )
            shock_reversal_market = (
                roc_opposite
                and roc_abs >= float(settings["shock_reversal_roc_abs_min"])
                and range_spike_ok
                and dirvol_shock_ok
                and close_loc_shock_ok
            )

        reversed_market = bool(settings["protect_on_reversal"]) and roc_opposite and roc_abs >= float(settings["reversal_roc_abs_min"])
        if reversed_market and dirvol_value is not None:
            reversed_market = reversed_market and dirvol_opposite
        if reversed_market and close_loc_value is not None:
            reversed_market = reversed_market and close_loc_opposite

        lateralized_market = False
        lateralization_detail = ""
        if bool(settings["protect_on_lateralization"]):
            baseline_lateralized_market = (
                adx_abs <= float(settings["lateralization_adx_max"])
                and roc_abs <= float(settings["lateralization_roc_abs_max"])
                and dirvol_abs <= float(settings["lateralization_dirvol_abs_max"])
            )
            compact_oscillation_lateralized_market = (
                bars_open >= int(settings["lateralization_horizon_bars"])
                and short_roc_value is not None
                and small_range_value is not None
                and close_loc_value is not None
                and short_roc_abs <= float(settings["lateralization_short_roc_abs_max"])
                and float(small_range_value) <= float(settings["lateralization_small_range_max"])
                and close_loc_abs <= float(settings["lateralization_close_location_center_abs_max"])
                and adx_abs <= float(settings["lateralization_compact_adx_max"])
                and dirvol_abs <= float(settings["lateralization_compact_dirvol_abs_max"])
            )
            lateralized_market = baseline_lateralized_market or compact_oscillation_lateralized_market
            if baseline_lateralized_market:
                lateralization_detail = "baseline"
            elif compact_oscillation_lateralized_market:
                lateralization_detail = "compact_oscillation"

        no_followthrough = (
            bars_open >= int(settings["min_bars_before_no_progress"])
            and progress_to_tp < float(settings["min_progress_to_keep"])
            and not roc_aligned
        )
        should_protect = shock_reversal_market or reversed_market or lateralized_market or no_followthrough
        if not should_protect:
            return False

        managed_stage_ids = self._parse_managed_stage_ids(row.get("managed_stage_ids"))
        symbol_spec = mt5_client.get_symbol_spec(symbol) or {}
        min_lot = float(symbol_spec.get("volume_min") or 0.01)
        lot_step = float(symbol_spec.get("volume_step") or 0.01)
        pip_size = float(self.config.get("data", {}).get("pip_size", 0.0001) or 0.0001)
        position_id = int(position_value)
        now_iso = datetime.now().isoformat()
        reason = (
            "shock_reversal"
            if shock_reversal_market
            else "reversal"
            if reversed_market
            else "lateralization"
            if lateralized_market
            else "no_followthrough"
        )
        reason_detail = lateralization_detail if reason == "lateralization" and lateralization_detail else reason

        lifecycle.at[idx, "management_progress_to_tp"] = progress_to_tp
        lifecycle.at[idx, "remaining_volume_lots_estimate"] = float(current_volume)

        if (
            shock_reversal_market
            and pnl_nonnegative
            and progress_to_tp >= float(settings.get("shock_reversal_progress_min", 0.05))
        ):
            full_close_stage_id = f"{settings['comment_prefix']}_FULL_SHOCK_REVERSAL"
            if full_close_stage_id not in managed_stage_ids:
                close_volume = self._normalize_volume_to_step(
                    float(current_volume),
                    min_lot=min_lot,
                    lot_step=lot_step,
                )
                if close_volume <= 0:
                    managed_stage_ids.add(full_close_stage_id)
                    lifecycle.at[idx, "managed_stage_ids"] = self._serialize_managed_stage_ids(managed_stage_ids)
                    lifecycle.at[idx, "last_management_time"] = now_iso
                    lifecycle.at[idx, "last_management_action"] = f"{full_close_stage_id}:skip_full_close_volume"
                    lifecycle.at[idx, "trade_management_comment"] = (
                        f"{settings['comment_prefix']} detecto shock_reversal, pero no pudo cerrar total por volumen minimo."
                    )
                    return True

                close_result = mt5_client.close_position_volume(
                    symbol=symbol,
                    position_ticket=position_id,
                    volume=close_volume,
                    side=side,
                    comment=f"{settings['comment_prefix']}_FULL_SHOCK_REVERSAL",
                    deviation=self._get_live_trading_settings()["order_deviation_points"],
                )
                if not close_result.get("success"):
                    lifecycle.at[idx, "trade_management_comment"] = (
                        f"{settings['comment_prefix']} fallo cierre total por shock_reversal: {close_result.get('comment')}"
                    )
                    return True

                managed_stage_ids.add(full_close_stage_id)
                lifecycle.at[idx, "managed_stage_ids"] = self._serialize_managed_stage_ids(managed_stage_ids)
                lifecycle.at[idx, "last_management_time"] = now_iso
                prev_partial_total = pd.to_numeric(pd.Series([row.get("partial_close_total_volume")]), errors="coerce").iloc[0]
                prev_partial_total = 0.0 if pd.isna(prev_partial_total) else float(prev_partial_total)
                lifecycle.at[idx, "last_management_action"] = f"{full_close_stage_id}:full_close"
                lifecycle.at[idx, "last_partial_close_volume"] = float(close_volume)
                lifecycle.at[idx, "partial_close_total_volume"] = prev_partial_total + float(close_volume)
                lifecycle.at[idx, "remaining_volume_lots_estimate"] = 0.0
                lifecycle.at[idx, "trade_management_comment"] = (
                    f"{settings['comment_prefix']} cerro el trade completo por shock_reversal con trade no negativo."
                )
                self.logger.info(
                    "Runtime monitor: symbol=%s position_id=%s action=full_close reason=%s progress=%.2f%% closed=%.2f",
                    symbol,
                    position_id,
                    "shock_reversal",
                    progress_to_tp * 100.0,
                    float(close_volume),
                )
                return True

        if not break_even_applied:
            be_stage_id = f"{settings['comment_prefix']}_BE_{reason}"
            if be_stage_id in managed_stage_ids:
                return False
            be_price = self._calculate_break_even_price(
                entry_price=float(entry_price),
                side=side,
                pip_size=pip_size,
                move_sl_to="breakeven",
                buffer_pips=float(self._get_trade_management_settings()["breakeven_buffer_pips"]),
            )
            protection = mt5_client.ensure_position_protection(
                symbol=symbol,
                position_ticket=position_id,
                side=side,
                sl=be_price,
                tp=None if pd.isna(target_tp) else float(target_tp),
            )
            managed_stage_ids.add(be_stage_id)
            lifecycle.at[idx, "managed_stage_ids"] = self._serialize_managed_stage_ids(managed_stage_ids)
            lifecycle.at[idx, "last_management_time"] = now_iso
            lifecycle.at[idx, "last_management_action"] = f"{be_stage_id}:move_break_even"
            if not protection.get("success"):
                lifecycle.at[idx, "trade_management_comment"] = (
                    f"{settings['comment_prefix']} no pudo mover a break-even por {reason_detail}: {protection.get('comment')}"
                )
                return True
            lifecycle.at[idx, "break_even_applied"] = True
            lifecycle.at[idx, "break_even_applied_time"] = now_iso
            lifecycle.at[idx, "break_even_sl_price"] = protection.get("applied_sl")
            lifecycle.at[idx, "applied_sl_price"] = protection.get("applied_sl")
            lifecycle.at[idx, "applied_tp_price"] = protection.get("applied_tp")
            lifecycle.at[idx, "trade_management_comment"] = (
                f"{settings['comment_prefix']} proteccion temprana aplicada por {reason_detail}: SL movido a break-even."
            )
            self.logger.info(
                "Runtime monitor: symbol=%s position_id=%s action=break_even reason=%s progress=%.2f%%",
                symbol,
                position_id,
                reason_detail,
                progress_to_tp * 100.0,
            )
            return True

        if progress_to_tp < float(settings["partial_profit_progress_min"]):
            return False

        should_full_close = False
        if bool(settings["close_full_on_weakness"]) and progress_to_tp >= float(settings["full_close_profit_progress_min"]):
            if reason == "reversal" and bool(settings["full_close_on_reversal"]):
                should_full_close = True
            elif reason == "lateralization" and bool(settings["full_close_on_lateralization"]):
                should_full_close = True
            elif reason == "no_followthrough" and bool(settings["full_close_on_no_followthrough"]):
                should_full_close = True

        if should_full_close:
            full_close_stage_id = f"{settings['comment_prefix']}_FULL_{reason}"
            if full_close_stage_id in managed_stage_ids:
                return False
            close_volume = self._normalize_volume_to_step(
                float(current_volume),
                min_lot=min_lot,
                lot_step=lot_step,
            )
            if close_volume <= 0:
                managed_stage_ids.add(full_close_stage_id)
                lifecycle.at[idx, "managed_stage_ids"] = self._serialize_managed_stage_ids(managed_stage_ids)
                lifecycle.at[idx, "last_management_time"] = now_iso
                lifecycle.at[idx, "last_management_action"] = f"{full_close_stage_id}:skip_full_close_volume"
                lifecycle.at[idx, "trade_management_comment"] = (
                    f"{settings['comment_prefix']} detecto {reason}, pero no pudo cerrar total por volumen minimo."
                )
                return True

            close_result = mt5_client.close_position_volume(
                symbol=symbol,
                position_ticket=position_id,
                volume=close_volume,
                side=side,
                comment=f"{settings['comment_prefix']}_FULL_{reason}"[:31],
                deviation=self._get_live_trading_settings()["order_deviation_points"],
            )
            managed_stage_ids.add(full_close_stage_id)
            lifecycle.at[idx, "managed_stage_ids"] = self._serialize_managed_stage_ids(managed_stage_ids)
            lifecycle.at[idx, "last_management_time"] = now_iso
            if not close_result.get("success"):
                lifecycle.at[idx, "last_management_action"] = f"{full_close_stage_id}:full_close_failed"
                lifecycle.at[idx, "trade_management_comment"] = (
                    f"{settings['comment_prefix']} no pudo cerrar total por {reason_detail}: {close_result.get('comment')}"
                )
                return True

            prev_partial_total = pd.to_numeric(pd.Series([row.get("partial_close_total_volume")]), errors="coerce").iloc[0]
            prev_partial_total = 0.0 if pd.isna(prev_partial_total) else float(prev_partial_total)
            lifecycle.at[idx, "last_management_action"] = f"{full_close_stage_id}:full_close"
            lifecycle.at[idx, "last_partial_close_volume"] = float(close_volume)
            lifecycle.at[idx, "partial_close_total_volume"] = prev_partial_total + float(close_volume)
            lifecycle.at[idx, "remaining_volume_lots_estimate"] = 0.0
            lifecycle.at[idx, "trade_management_comment"] = (
                f"{settings['comment_prefix']} cerro el remanente por {reason_detail} con trade en positivo."
            )
            self.logger.info(
                "Runtime monitor: symbol=%s position_id=%s action=full_close reason=%s progress=%.2f%% closed=%.2f",
                symbol,
                position_id,
                reason_detail,
                progress_to_tp * 100.0,
                float(close_volume),
            )
            return True

        partial_stage_id = f"{settings['comment_prefix']}_PARTIAL_{reason}"
        if partial_stage_id in managed_stage_ids:
            return False

        max_close_volume = float(current_volume) - float(min_lot)
        desired_close_volume = float(current_volume) * float(settings["partial_close_fraction_on_weakness"])
        close_volume = self._normalize_volume_to_step(
            min(desired_close_volume, max_close_volume),
            min_lot=min_lot,
            lot_step=lot_step,
        )
        if close_volume <= 0:
            managed_stage_ids.add(partial_stage_id)
            lifecycle.at[idx, "managed_stage_ids"] = self._serialize_managed_stage_ids(managed_stage_ids)
            lifecycle.at[idx, "last_management_time"] = now_iso
            lifecycle.at[idx, "last_management_action"] = f"{partial_stage_id}:skip_partial_volume"
            lifecycle.at[idx, "trade_management_comment"] = (
                f"{settings['comment_prefix']} detecto {reason}, pero no pudo cerrar parcial por volumen minimo."
            )
            return True

        close_result = mt5_client.close_position_volume(
            symbol=symbol,
            position_ticket=position_id,
            volume=close_volume,
            side=side,
            comment=f"{settings['comment_prefix']}_{reason}"[:31],
            deviation=self._get_live_trading_settings()["order_deviation_points"],
        )
        managed_stage_ids.add(partial_stage_id)
        lifecycle.at[idx, "managed_stage_ids"] = self._serialize_managed_stage_ids(managed_stage_ids)
        lifecycle.at[idx, "last_management_time"] = now_iso
        if not close_result.get("success"):
            lifecycle.at[idx, "last_management_action"] = f"{partial_stage_id}:partial_failed"
            lifecycle.at[idx, "trade_management_comment"] = (
                f"{settings['comment_prefix']} no pudo cerrar parcial por {reason_detail}: {close_result.get('comment')}"
            )
            return True

        prev_partial_total = pd.to_numeric(pd.Series([row.get("partial_close_total_volume")]), errors="coerce").iloc[0]
        prev_partial_total = 0.0 if pd.isna(prev_partial_total) else float(prev_partial_total)
        remaining_estimate = max(float(current_volume) - float(close_volume), 0.0)
        lifecycle.at[idx, "last_management_action"] = f"{partial_stage_id}:partial_close"
        lifecycle.at[idx, "last_partial_close_volume"] = float(close_volume)
        lifecycle.at[idx, "partial_close_total_volume"] = prev_partial_total + float(close_volume)
        lifecycle.at[idx, "remaining_volume_lots_estimate"] = remaining_estimate
        lifecycle.at[idx, "trade_management_comment"] = (
            f"{settings['comment_prefix']} redujo {close_volume:.2f} lotes por {reason_detail} con trade en positivo."
        )
        self.logger.info(
            "Runtime monitor: symbol=%s position_id=%s action=partial_close reason=%s progress=%.2f%% partial=%.2f remaining=%.2f",
            symbol,
            position_id,
            reason_detail,
            progress_to_tp * 100.0,
            float(close_volume),
            remaining_estimate,
        )
        return True

    def _sync_pending_entry_order_state(
        self,
        *,
        lifecycle: pd.DataFrame,
        idx: int,
        row: pd.Series,
        mt5_client,
        pending_orders: pd.DataFrame,
        position_is_open: bool,
        current_volume: float | None,
    ) -> bool:
        """Sincroniza/cancela la orden pendiente asociada a una entrada escalonada."""
        pending_ticket_value = pd.to_numeric(pd.Series([row.get("pending_order_ticket")]), errors="coerce").iloc[0]
        if pd.isna(pending_ticket_value):
            return False

        pending_ticket = int(pending_ticket_value)
        if pending_ticket <= 0:
            return False

        current_status = str(row.get("pending_order_status", "") or "").strip().upper()
        lifecycle_status = str(row.get("status", "") or "").strip().upper()
        now_dt = datetime.now()
        now_iso = now_dt.isoformat()
        expiry_ts = pd.to_datetime(row.get("pending_order_expiry_time"), errors="coerce")
        entry_leg = str(row.get("entry_leg", "") or "").strip().lower()
        parent_signal_id = str(row.get("parent_signal_id") or row.get("signal_id") or "").strip()
        parent_market_open = False
        if entry_leg != "market" and parent_signal_id and lifecycle is not None and not lifecycle.empty:
            parent_mask = (
                lifecycle["parent_signal_id"].astype(str).eq(parent_signal_id)
                & lifecycle["entry_leg"].astype(str).str.lower().eq("market")
            )
            if bool(parent_mask.any()):
                parent_statuses = lifecycle.loc[parent_mask, "status"].astype(str).str.upper()
                parent_market_open = bool(parent_statuses.isin(["OPEN", "PENDING_CONFIRMATION"]).any())

        matches = pd.DataFrame()
        if pending_orders is not None and not pending_orders.empty and "ticket" in pending_orders.columns:
            matches = pending_orders[
                pd.to_numeric(pending_orders["ticket"], errors="coerce").fillna(-1).astype(int) == pending_ticket
            ].copy()
        if matches.empty:
            try:
                direct_matches = mt5_client.get_pending_orders(ticket=pending_ticket)
            except Exception:
                direct_matches = pd.DataFrame()
            if direct_matches is not None and not direct_matches.empty:
                matches = direct_matches.copy()

        changed = False
        if not matches.empty:
            lifecycle.at[idx, "pending_order_last_sync_time"] = now_iso
            if current_status != "ACTIVE":
                lifecycle.at[idx, "pending_order_status"] = "ACTIVE"
                changed = True
            if lifecycle_status not in {"PENDING_LIMIT", "OPEN", "PENDING_CONFIRMATION"} and entry_leg == "pending_limit":
                lifecycle.at[idx, "status"] = "PENDING_LIMIT"
                lifecycle.at[idx, "status_detail"] = "Orden LIMIT pendiente reactivada desde sincronizacion."
                changed = True

            cancel_reason = None
            settings = self._get_entry_management_settings()
            cluster_state = self._collect_same_side_cluster_state(lifecycle=lifecycle, row=row)
            lifecycle.at[idx, "cluster_open_positions_count"] = cluster_state["open_positions_count"]
            lifecycle.at[idx, "cluster_active_pending_orders_count"] = cluster_state["active_pending_orders_count"]
            lifecycle.at[idx, "cluster_max_progress_to_tp"] = cluster_state["max_progress_to_tp"]
            if (
                not position_is_open
                and settings["cancel_pending_on_position_close"]
                and not parent_market_open
            ):
                cancel_reason = "CANCELLED_POSITION_CLOSED"
            elif (
                settings.get("cancel_pending_when_market_in_profit_enabled")
                and cluster_state["market_open_count"] >= 1
                and cluster_state["max_progress_to_tp"]
                >= float(settings["cancel_pending_when_market_in_profit_progress_min"])
            ):
                cancel_reason = "CANCELLED_MARKET_PROFIT_PROGRESS"
            elif (
                settings.get("cluster_guard_enabled")
                and cluster_state["open_positions_count"] >= int(settings["cluster_guard_cancel_pending_open_positions_min"])
                and cluster_state["max_progress_to_tp"] >= float(settings["cluster_guard_cancel_pending_progress_min"])
            ):
                cancel_reason = "CANCELLED_CLUSTER_PROGRESS"
            elif (
                settings.get("cluster_guard_enabled")
                and cluster_state["open_positions_count"] >= int(settings["cluster_guard_symbol_side_max_open_positions"])
                and cluster_state["active_pending_orders_count"] > int(settings["cluster_guard_symbol_side_max_pending_orders"])
            ):
                cancel_reason = "CANCELLED_CLUSTER_EXPOSURE"
            elif pd.notna(expiry_ts) and now_dt >= expiry_ts.to_pydatetime():
                cancel_reason = "CANCELLED_EXPIRED"

            if cancel_reason:
                cancel_result = mt5_client.cancel_pending_order(order_ticket=pending_ticket)
                lifecycle.at[idx, "pending_order_last_sync_time"] = now_iso
                if cancel_reason == "CANCELLED_CLUSTER_PROGRESS":
                    lifecycle.at[idx, "pending_order_comment"] = (
                        "Cluster guard: pending cancelada porque el movimiento ya avanzo y habia exposicion suficiente."
                    )
                elif cancel_reason == "CANCELLED_MARKET_PROFIT_PROGRESS":
                    lifecycle.at[idx, "pending_order_comment"] = (
                        "Pierna pending cancelada porque la pierna market ya iba en ganancia/progreso suficiente."
                    )
                elif cancel_reason == "CANCELLED_CLUSTER_EXPOSURE":
                    lifecycle.at[idx, "pending_order_comment"] = (
                        "Cluster guard: pending cancelada para no seguir apilando exposicion del mismo lado."
                    )
                else:
                    lifecycle.at[idx, "pending_order_comment"] = cancel_result.get("comment")
                lifecycle.at[idx, "pending_order_status"] = (
                    cancel_reason if cancel_result.get("success") else f"{cancel_reason}_FAILED"
                )
                return True
            return changed

        lifecycle.at[idx, "pending_order_last_sync_time"] = now_iso
        if current_status in {"ACTIVE", "PLACED", "PENDING"}:
            initial_volume = pd.to_numeric(
                pd.Series([row.get("initial_market_volume_lots")]),
                errors="coerce",
            ).iloc[0]
            if (
                position_is_open
                and current_volume is not None
                and not pd.isna(current_volume)
                and not pd.isna(initial_volume)
                and float(current_volume) > float(initial_volume) + 1e-12
            ):
                lifecycle.at[idx, "pending_order_status"] = "FILLED"
            elif pd.notna(expiry_ts) and now_dt >= expiry_ts.to_pydatetime():
                lifecycle.at[idx, "pending_order_status"] = "INACTIVE"
            else:
                lifecycle.at[idx, "pending_order_status"] = current_status
            changed = True
        return changed

    def _find_pending_activated_position(
        self,
        *,
        row: pd.Series,
        open_positions: pd.DataFrame,
        magic_number: int,
        claimed_position_ids: set[int] | None = None,
    ) -> pd.Series | None:
        """Busca una posicion abierta derivada de una LIMIT ya activada."""
        if open_positions is None or open_positions.empty:
            return None

        claimed_position_ids = claimed_position_ids or set()
        pending_ticket_value = pd.to_numeric(pd.Series([row.get("pending_order_ticket")]), errors="coerce").iloc[0]
        if pd.isna(pending_ticket_value):
            return None
        pending_ticket = int(pending_ticket_value)
        if pending_ticket <= 0:
            return None

        candidates = pd.DataFrame()
        for col in ["ticket", "identifier"]:
            if col in open_positions.columns:
                matches = open_positions[
                    pd.to_numeric(open_positions[col], errors="coerce").fillna(-1).astype(int) == pending_ticket
                ].copy()
                if not matches.empty:
                    candidates = matches
                    break

        if candidates.empty:
            candidates = open_positions.copy()
            symbol = str(row.get("symbol") or "")
            side = str(row.get("signal") or "").upper()
            pending_volume = pd.to_numeric(pd.Series([row.get("pending_order_volume_lots")]), errors="coerce").iloc[0]
            pending_entry = pd.to_numeric(pd.Series([row.get("pending_order_price")]), errors="coerce").iloc[0]
            pending_sl = pd.to_numeric(pd.Series([row.get("pending_order_sl_price")]), errors="coerce").iloc[0]
            side_value = 0 if side == "BUY" else 1 if side == "SELL" else None

            if claimed_position_ids:
                claimed_mask = pd.Series(False, index=candidates.index)
                for col in ["ticket", "identifier"]:
                    if col not in candidates.columns:
                        continue
                    candidate_ids = pd.to_numeric(candidates[col], errors="coerce").fillna(-1).astype(int)
                    claimed_mask = claimed_mask | candidate_ids.isin(claimed_position_ids)
                candidates = candidates.loc[~claimed_mask].copy()
            if symbol and "symbol" in candidates.columns:
                candidates = candidates[candidates["symbol"] == symbol]
            if "magic" in candidates.columns:
                candidates = candidates[
                    pd.to_numeric(candidates["magic"], errors="coerce").fillna(0).astype(int) == int(magic_number)
                ]
            if side_value is not None and "type" in candidates.columns:
                candidates = candidates[
                    pd.to_numeric(candidates["type"], errors="coerce").fillna(-1).astype(int) == int(side_value)
                ]
            if pd.notna(pending_volume) and "volume" in candidates.columns:
                volume_series = pd.to_numeric(candidates["volume"], errors="coerce")
                volume_tol = max(0.02, abs(float(pending_volume)) * 0.05)
                close_volume = candidates[(volume_series - float(pending_volume)).abs() <= volume_tol].copy()
                if not close_volume.empty:
                    candidates = close_volume
            if candidates.empty:
                return None

            if pd.notna(pending_entry) and "price_open" in candidates.columns:
                price_open_series = pd.to_numeric(candidates["price_open"], errors="coerce")
                stop_distance = (
                    abs(float(pending_entry) - float(pending_sl))
                    if pd.notna(pending_sl)
                    else np.nan
                )
                price_tol = max(
                    0.00025,
                    (float(stop_distance) * 0.35) if pd.notna(stop_distance) and float(stop_distance) > 0.0 else 0.0,
                )
                close_price = candidates[(price_open_series - float(pending_entry)).abs() <= price_tol].copy()
                if close_price.empty:
                    return None
                candidates = close_price

            signal_ts = pd.to_datetime(row.get("signal_time"), errors="coerce")
            expiry_ts = pd.to_datetime(row.get("pending_order_expiry_time"), errors="coerce")
            if "time" in candidates.columns and (pd.notna(signal_ts) or pd.notna(expiry_ts)):
                position_times = pd.to_datetime(
                    pd.to_numeric(candidates["time"], errors="coerce"),
                    unit="s",
                    errors="coerce",
                )
                time_mask = pd.Series(True, index=candidates.index)
                if pd.notna(signal_ts):
                    time_mask = time_mask & (position_times >= (signal_ts - timedelta(minutes=5)))
                if pd.notna(expiry_ts):
                    time_mask = time_mask & (position_times <= (expiry_ts + timedelta(minutes=15)))
                close_time = candidates.loc[time_mask].copy()
                if close_time.empty:
                    return None
                candidates = close_time

        if candidates.empty:
            return None

        sortable = candidates.copy()
        sortable["_sort_time"] = (
            pd.to_numeric(sortable["time"], errors="coerce").fillna(0)
            if "time" in sortable.columns
            else 0
        )
        sortable["_sort_ticket"] = (
            pd.to_numeric(sortable["ticket"], errors="coerce").fillna(0)
            if "ticket" in sortable.columns
            else 0
        )
        sortable = sortable.sort_values(by=["_sort_time", "_sort_ticket"], ascending=[False, False])
        return sortable.iloc[0]

    def _adopt_pending_filled_position(
        self,
        *,
        lifecycle: pd.DataFrame,
        idx: int,
        row: pd.Series,
        open_positions: pd.DataFrame,
        magic_number: int,
        claimed_position_ids: set[int] | None = None,
    ) -> bool:
        """Reasigna la fila lifecycle a la posicion abierta desde la LIMIT en cuentas hedging."""
        pos_row = self._find_pending_activated_position(
            row=row,
            open_positions=open_positions,
            magic_number=magic_number,
            claimed_position_ids=claimed_position_ids,
        )
        if pos_row is None:
            return False

        ticket_value = pd.to_numeric(pd.Series([pos_row.get("ticket")]), errors="coerce").iloc[0]
        if pd.isna(ticket_value):
            return False

        adopted_position_id = int(ticket_value)
        now_iso = datetime.now().isoformat()

        pending_entry = pd.to_numeric(pd.Series([row.get("pending_order_price")]), errors="coerce").iloc[0]
        pending_sl = pd.to_numeric(pd.Series([row.get("pending_order_sl_price")]), errors="coerce").iloc[0]
        pending_tp = pd.to_numeric(pd.Series([row.get("pending_order_tp_price")]), errors="coerce").iloc[0]
        current_sl = pd.to_numeric(pd.Series([pos_row.get("sl")]), errors="coerce").iloc[0]
        current_tp = pd.to_numeric(pd.Series([pos_row.get("tp")]), errors="coerce").iloc[0]
        current_volume = pd.to_numeric(pd.Series([pos_row.get("volume")]), errors="coerce").iloc[0]
        price_open = pd.to_numeric(pd.Series([pos_row.get("price_open")]), errors="coerce").iloc[0]

        lifecycle.at[idx, "status"] = "OPEN"
        lifecycle.at[idx, "status_detail"] = "Pending LIMIT activada como posicion independiente."
        lifecycle.at[idx, "mt5_position_id"] = adopted_position_id
        lifecycle.at[idx, "mt5_order_ticket"] = adopted_position_id
        lifecycle.at[idx, "execution_price"] = price_open
        lifecycle.at[idx, "requested_entry_price"] = pending_entry if pd.notna(pending_entry) else price_open
        lifecycle.at[idx, "requested_live_entry_price"] = pending_entry if pd.notna(pending_entry) else price_open
        lifecycle.at[idx, "requested_sl_price"] = (
            pending_sl if pd.notna(pending_sl) else (current_sl if pd.notna(current_sl) else row.get("requested_sl_price"))
        )
        lifecycle.at[idx, "requested_tp_price"] = (
            pending_tp if pd.notna(pending_tp) else (current_tp if pd.notna(current_tp) else row.get("requested_tp_price"))
        )
        lifecycle.at[idx, "requested_plan_sl_price"] = lifecycle.at[idx, "requested_sl_price"]
        lifecycle.at[idx, "requested_plan_tp_price"] = lifecycle.at[idx, "requested_tp_price"]
        lifecycle.at[idx, "applied_sl_price"] = current_sl if pd.notna(current_sl) else lifecycle.at[idx, "requested_sl_price"]
        lifecycle.at[idx, "applied_tp_price"] = current_tp if pd.notna(current_tp) else lifecycle.at[idx, "requested_tp_price"]
        lifecycle.at[idx, "protection_status"] = "PROTECTED"
        lifecycle.at[idx, "protection_comment"] = "Proteccion adoptada desde pending fill."
        lifecycle.at[idx, "close_time"] = pd.NA
        lifecycle.at[idx, "close_price"] = np.nan
        lifecycle.at[idx, "close_profit_net"] = np.nan
        lifecycle.at[idx, "close_reason"] = pd.NA
        lifecycle.at[idx, "close_deal_ticket"] = np.nan
        lifecycle.at[idx, "last_sync_time"] = now_iso
        lifecycle.at[idx, "managed_stage_ids"] = ""
        lifecycle.at[idx, "break_even_applied"] = False
        lifecycle.at[idx, "break_even_applied_time"] = pd.NA
        lifecycle.at[idx, "break_even_sl_price"] = np.nan
        lifecycle.at[idx, "last_management_time"] = pd.NA
        lifecycle.at[idx, "last_management_action"] = pd.NA
        lifecycle.at[idx, "last_partial_close_volume"] = 0.0
        lifecycle.at[idx, "partial_close_total_volume"] = 0.0
        lifecycle.at[idx, "remaining_volume_lots_estimate"] = current_volume if pd.notna(current_volume) else np.nan
        lifecycle.at[idx, "management_progress_to_tp"] = np.nan
        lifecycle.at[idx, "trade_management_comment"] = "Pending LIMIT adoptada para gestion."
        lifecycle.at[idx, "pending_order_status"] = "FILLED"
        lifecycle.at[idx, "pending_order_comment"] = "Pending LIMIT activada como posicion independiente."
        lifecycle.at[idx, "pending_order_last_sync_time"] = now_iso
        entry_mode = str(row.get("entry_management_mode", "") or "").strip().lower()
        lifecycle.at[idx, "entry_management_comment"] = (
            "risk_based_ladder_filled"
            if entry_mode == "risk_based_ladder"
            else "split_retrace_limit_filled"
        )
        return True

    def _dedupe_open_lifecycle_rows_by_position(
        self,
        lifecycle: pd.DataFrame,
        *,
        scope_mask: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, bool]:
        """Cierra filas OPEN duplicadas que apuntan al mismo mt5_position_id."""
        if lifecycle is None or lifecycle.empty or "mt5_position_id" not in lifecycle.columns:
            return lifecycle, False

        status_upper = lifecycle["status"].astype(str).str.upper()
        position_ids = pd.to_numeric(lifecycle["mt5_position_id"], errors="coerce")
        open_mask = status_upper.isin(["OPEN", "PENDING_CONFIRMATION"]) & position_ids.notna()
        if scope_mask is not None:
            aligned_scope = scope_mask.reindex(lifecycle.index, fill_value=False)
            open_mask = open_mask & aligned_scope
        if not bool(open_mask.any()):
            return lifecycle, False

        changed = False
        now_iso = datetime.now().isoformat()
        for _, idx_values in lifecycle.loc[open_mask].groupby(position_ids.loc[open_mask]).groups.items():
            idx_list = list(idx_values)
            if len(idx_list) <= 1:
                continue

            subset = lifecycle.loc[idx_list].copy()
            signal_times = (
                pd.to_datetime(subset["signal_time"], errors="coerce")
                if "signal_time" in subset.columns
                else pd.Series(pd.NaT, index=subset.index)
            )
            execution_prices = (
                pd.to_numeric(subset["execution_price"], errors="coerce")
                if "execution_price" in subset.columns
                else pd.Series(np.nan, index=subset.index)
            )
            pending_prices = (
                pd.to_numeric(subset["pending_order_price"], errors="coerce")
                if "pending_order_price" in subset.columns
                else pd.Series(np.nan, index=subset.index)
            )
            requested_entries = (
                pd.to_numeric(subset["requested_entry_price"], errors="coerce")
                if "requested_entry_price" in subset.columns
                else pd.Series(np.nan, index=subset.index)
            )
            entry_legs = (
                subset["entry_leg"].astype(str).str.lower()
                if "entry_leg" in subset.columns
                else pd.Series("", index=subset.index)
            )
            reference_prices = pending_prices.where(entry_legs.eq("pending_limit"), requested_entries)
            price_gap = (execution_prices - reference_prices).abs().fillna(np.inf)
            keep_order = pd.DataFrame(
                {
                    "idx": subset.index,
                    "price_gap": price_gap,
                    "signal_time": signal_times,
                }
            ).sort_values(by=["price_gap", "signal_time", "idx"], ascending=[True, False, False])
            keep_idx = int(keep_order.iloc[0]["idx"])

            for dup_idx in idx_list:
                if int(dup_idx) == keep_idx:
                    continue
                lifecycle.at[dup_idx, "status"] = "CLOSED"
                lifecycle.at[dup_idx, "status_detail"] = (
                    f"Fila lifecycle duplicada reconciliada al mt5_position_id={int(position_ids.at[dup_idx])}."
                )
                lifecycle.at[dup_idx, "close_time"] = now_iso
                lifecycle.at[dup_idx, "close_price"] = execution_prices.get(keep_idx, np.nan)
                lifecycle.at[dup_idx, "close_profit_net"] = 0.0
                lifecycle.at[dup_idx, "close_reason"] = "DEDUPLICATED_LIFECYCLE"
                lifecycle.at[dup_idx, "last_sync_time"] = now_iso
                lifecycle.at[dup_idx, "remaining_volume_lots_estimate"] = 0.0
                lifecycle.at[dup_idx, "trade_management_comment"] = (
                    "Fila duplicada cerrada administrativamente para evitar doble gestion."
                )
                if "pending_order_status" in lifecycle.columns:
                    lifecycle.at[dup_idx, "pending_order_status"] = "DUPLICATE_RECONCILED"
                changed = True

        return lifecycle, changed

    def _get_active_lifecycle_scope_mask(
        self,
        lifecycle: pd.DataFrame,
        *,
        magic_number: int | None = None,
        profile_name: str | None = None,
    ) -> pd.Series:
        """Devuelve la máscara de filas lifecycle que pertenecen al perfil live activo."""
        if lifecycle is None or lifecycle.empty:
            return pd.Series(dtype=bool)

        if magic_number is not None and "magic_number" in lifecycle.columns:
            magic_series = pd.to_numeric(lifecycle["magic_number"], errors="coerce").fillna(0).astype(int)
            magic_mask = magic_series.eq(int(magic_number))
            if bool(magic_mask.any()):
                return magic_mask

        profile_norm = self._normalize_profile_label(profile_name or self._get_strategy_profile_name())
        if profile_norm:
            for col in ["profile_name", "strategy_profile"]:
                if col not in lifecycle.columns:
                    continue
                profile_series = lifecycle[col].apply(
                    lambda value: self._normalize_profile_label(self._coerce_textish(value, ""))
                )
                profile_mask = profile_series.eq(profile_norm)
                if bool(profile_mask.any()):
                    return profile_mask

        return pd.Series(True, index=lifecycle.index)

    def _ensure_pending_child_rows(self, lifecycle: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """Migra filas legacy creando la fila hija de la pierna pending cuando falte."""
        if lifecycle is None or lifecycle.empty:
            return lifecycle, False

        new_rows: list[dict[str, Any]] = []
        changed = False

        for idx, row in lifecycle.iterrows():
            entry_mode = str(row.get("entry_management_mode", "") or "").strip().lower()
            entry_leg = str(row.get("entry_leg", "") or "market").strip().lower()
            if entry_mode != "split_retrace_limit" or entry_leg == "pending_limit":
                continue
            if str(row.get("entry_management_comment", "") or "").strip().lower() == "split_retrace_limit_filled":
                continue

            parent_signal_id = str(row.get("parent_signal_id") or row.get("signal_id") or "").strip()
            signal_id = str(row.get("signal_id") or "").strip()
            if not signal_id:
                continue

            pending_ticket_value = pd.to_numeric(pd.Series([row.get("pending_order_ticket")]), errors="coerce").iloc[0]
            if pd.isna(pending_ticket_value) or int(pending_ticket_value) <= 0:
                continue

            child_exists = False
            if "parent_signal_id" in lifecycle.columns and "entry_leg" in lifecycle.columns:
                mask = (
                    lifecycle["parent_signal_id"].astype(str) == parent_signal_id
                ) & (
                    lifecycle["entry_leg"].astype(str).str.lower() == "pending_limit"
                )
                child_exists = bool(mask.any())
            if child_exists:
                continue

            pending_signal_id = f"{signal_id}|LEG:PENDING"
            pending_volume = pd.to_numeric(pd.Series([row.get("pending_order_volume_lots")]), errors="coerce").iloc[0]
            pending_price = pd.to_numeric(pd.Series([row.get("pending_order_price")]), errors="coerce").iloc[0]
            pending_sl = pd.to_numeric(pd.Series([row.get("pending_order_sl_price")]), errors="coerce").iloc[0]
            pending_tp = pd.to_numeric(pd.Series([row.get("pending_order_tp_price")]), errors="coerce").iloc[0]

            child = row.to_dict()
            child.update(
                {
                    "signal_id": pending_signal_id,
                    "parent_signal_id": parent_signal_id,
                    "entry_leg": "pending_limit",
                    "status": "PENDING_LIMIT",
                    "status_detail": "Fila hija pending creada desde lifecycle legacy.",
                    "mt5_order_ticket": row.get("pending_order_ticket"),
                    "mt5_deal_ticket": np.nan,
                    "mt5_position_id": np.nan,
                    "execution_price": np.nan,
                    "execution_retcode": np.nan,
                    "execution_comment": row.get("pending_order_comment"),
                    "requested_entry_price": pending_price,
                    "requested_live_entry_price": pending_price,
                    "requested_sl_price": pending_sl,
                    "requested_tp_price": pending_tp,
                    "requested_plan_sl_price": pending_sl,
                    "requested_plan_tp_price": pending_tp,
                    "requested_volume_lots": pending_volume,
                    "applied_sl_price": pending_sl,
                    "applied_tp_price": pending_tp,
                    "protection_status": pd.NA,
                    "protection_comment": pd.NA,
                    "close_time": pd.NA,
                    "close_price": np.nan,
                    "close_profit_net": np.nan,
                    "close_reason": pd.NA,
                    "close_deal_ticket": np.nan,
                    "last_sync_time": pd.NA,
                    "managed_stage_ids": "",
                    "break_even_applied": False,
                    "break_even_applied_time": pd.NA,
                    "break_even_sl_price": np.nan,
                    "last_management_time": pd.NA,
                    "last_management_action": pd.NA,
                    "last_partial_close_volume": 0.0,
                    "partial_close_total_volume": 0.0,
                    "remaining_volume_lots_estimate": pending_volume if pd.notna(pending_volume) else np.nan,
                    "management_progress_to_tp": np.nan,
                    "trade_management_comment": pd.NA,
                    "entry_management_total_volume_lots": pending_volume if pd.notna(pending_volume) else np.nan,
                    "initial_market_volume_lots": pending_volume if pd.notna(pending_volume) else np.nan,
                    "pending_order_status": row.get("pending_order_status") or "ACTIVE",
                    "pending_order_comment": row.get("pending_order_comment"),
                    "pending_order_last_sync_time": row.get("pending_order_last_sync_time"),
                    "entry_management_comment": "split_retrace_limit_pending_leg",
                }
            )
            new_rows.append(child)

            lifecycle.at[idx, "pending_order_ticket"] = np.nan
            lifecycle.at[idx, "pending_order_status"] = "TRACKED_IN_CHILD"
            lifecycle.at[idx, "pending_order_comment"] = "Pierna pending registrada en fila separada."
            changed = True

        if new_rows:
            lifecycle = pd.concat([lifecycle, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
            changed = True
        return lifecycle, changed

    def _save_daily_trade_report(self, lifecycle: pd.DataFrame) -> pd.DataFrame:
        """Construye un agregado diario por fecha/sÃ­mbolo/modelo a partir del lifecycle."""
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

    def _sync_live_trade_report(
        self,
        *,
        apply_runtime_monitor: bool = False,
        runtime_feature_row: pd.Series | dict[str, Any] | None = None,
        current_bar_timestamp: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Actualiza el estado de trades ejecutados usando posiciones y deals de MT5."""
        paths = self._get_production_output_paths()
        lifecycle_path = paths["lifecycle"]

        if not lifecycle_path.exists():
            self._save_entry_grid_legs_report_from_lifecycle(pd.DataFrame())
            return pd.DataFrame()

        try:
            lifecycle = pd.read_csv(lifecycle_path)
        except EmptyDataError:
            self._save_entry_grid_legs_report_from_lifecycle(pd.DataFrame())
            return pd.DataFrame()
        if lifecycle.empty:
            paths["closed"].write_text("", encoding="utf-8")
            paths["daily"].write_text("", encoding="utf-8")
            self._save_entry_grid_legs_report_from_lifecycle(pd.DataFrame())
            return lifecycle

        lifecycle_defaults = {
            "parent_signal_id": pd.NA,
            "entry_leg": pd.NA,
            "grid_parent_signal_id": pd.NA,
            "grid_group_id": pd.NA,
            "grid_leg_id": pd.NA,
            "grid_leg_rank": np.nan,
            "grid_entry_type": pd.NA,
            "grid_volume_weight": np.nan,
            "grid_quality_rank": np.nan,
            "grid_runner_candidate": False,
            "managed_stage_ids": "",
            "break_even_applied": False,
            "break_even_applied_time": pd.NA,
            "break_even_sl_price": np.nan,
            "last_management_time": pd.NA,
            "last_management_action": pd.NA,
            "last_partial_close_volume": 0.0,
            "partial_close_total_volume": 0.0,
            "remaining_volume_lots_estimate": np.nan,
            "management_progress_to_tp": np.nan,
            "trade_management_comment": pd.NA,
            "entry_management_mode": pd.NA,
            "entry_management_split_active": False,
            "entry_management_initial_market_fraction": np.nan,
            "entry_management_pending_fraction": np.nan,
            "entry_management_retrace_fraction_of_stop": np.nan,
            "entry_management_total_volume_lots": np.nan,
            "initial_market_volume_lots": np.nan,
            "pending_order_volume_lots": 0.0,
            "pending_order_price": np.nan,
            "pending_order_type": pd.NA,
            "pending_order_sl_price": np.nan,
            "pending_order_tp_price": np.nan,
            "pending_order_ticket": np.nan,
            "pending_order_expiry_time": pd.NA,
            "pending_order_status": pd.NA,
            "pending_order_comment": pd.NA,
            "pending_order_last_sync_time": pd.NA,
            "entry_management_comment": pd.NA,
        }
        for col, default_value in lifecycle_defaults.items():
            if col not in lifecycle.columns:
                lifecycle[col] = default_value
        if "signal_id" in lifecycle.columns:
            lifecycle["parent_signal_id"] = lifecycle["parent_signal_id"].where(
                lifecycle["parent_signal_id"].notna(),
                lifecycle["signal_id"],
            )
        if "entry_leg" in lifecycle.columns:
            lifecycle["entry_leg"] = lifecycle["entry_leg"].where(
                lifecycle["entry_leg"].notna(),
                "market",
            )
        lifecycle, child_rows_changed = self._ensure_pending_child_rows(lifecycle)
        lifecycle["break_even_applied"] = lifecycle["break_even_applied"].apply(
            lambda value: value
            if isinstance(value, bool)
            else str(value).strip().lower() in {"true", "1", "yes", "si"}
        )

        try:
            mt5_client = self._ensure_mt5_client()
        except Exception as e:
            self.logger.warning(f"No se pudo conectar a MT5 para sincronizar trades: {e}")
            return lifecycle

        self._update_daily_loss_guard_state(mt5_client=mt5_client)

        live_cfg = self._get_live_trading_settings()
        active_scope_mask = self._get_active_lifecycle_scope_mask(
            lifecycle,
            magic_number=live_cfg["magic_number"],
            profile_name=self._get_strategy_profile_name(),
        )
        open_positions = mt5_client.get_all_positions()
        if open_positions is not None and not open_positions.empty and "magic" in open_positions.columns:
            open_positions = open_positions[
                pd.to_numeric(open_positions["magic"], errors="coerce").fillna(0).astype(int)
                == int(live_cfg["magic_number"])
            ].copy()
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
        pending_orders = mt5_client.get_pending_orders(magic=live_cfg["magic_number"])
        all_deals = mt5_client.get_history_deals(
            date_from=deals_date_from,
            date_to=deals_date_to,
        )

        lifecycle["status"] = lifecycle["status"].astype(str)
        lifecycle, duplicate_rows_changed = self._dedupe_open_lifecycle_rows_by_position(
            lifecycle,
            scope_mask=active_scope_mask,
        )
        now_iso = datetime.now().isoformat()
        changed = bool(child_rows_changed or duplicate_rows_changed)
        open_status_mask = (
            lifecycle["status"].astype(str).str.upper().isin(["OPEN", "PENDING_CONFIRMATION"])
            & active_scope_mask.reindex(lifecycle.index, fill_value=False)
        )
        claimed_open_position_ids = set(
            pd.to_numeric(lifecycle.loc[open_status_mask, "mt5_position_id"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )

        for idx, row in lifecycle.loc[active_scope_mask.reindex(lifecycle.index, fill_value=False)].iterrows():
            status = str(row.get("status", "")).upper()
            pending_ticket_value = pd.to_numeric(pd.Series([row.get("pending_order_ticket")]), errors="coerce").iloc[0]
            entry_mode = str(row.get("entry_management_mode", "") or "").strip().lower()
            entry_leg = str(row.get("entry_leg", "") or "").strip().lower()
            pending_status_upper = str(row.get("pending_order_status", "") or "").strip().upper()
            can_attempt_pending_recovery = (
                pd.notna(pending_ticket_value)
                and int(pending_ticket_value) > 0
                and (
                    (
                        entry_mode in {"split_retrace_limit", "risk_based_ladder"}
                        and status in {"OPEN", "PENDING_CONFIRMATION", "PENDING_LIMIT"}
                    )
                    or (
                        entry_leg == "pending_limit"
                        and pending_status_upper
                        not in {
                            "CANCELLED_POSITION_CLOSED",
                            "CANCELLED_EXPIRED",
                            "CANCELLED_CLUSTER_PROGRESS",
                            "CANCELLED_CLUSTER_EXPOSURE",
                        }
                    )
                )
            )
            if status not in {"OPEN", "PENDING_CONFIRMATION", "PENDING_LIMIT"} and not can_attempt_pending_recovery:
                continue

            position_value = pd.to_numeric(pd.Series([row.get("mt5_position_id")]), errors="coerce").iloc[0]
            position_id = None if pd.isna(position_value) else int(position_value)

            lifecycle.at[idx, "last_sync_time"] = now_iso
            changed = True

            if position_id is None and can_attempt_pending_recovery:
                adopted = self._adopt_pending_filled_position(
                    lifecycle=lifecycle,
                    idx=idx,
                    row=lifecycle.loc[idx],
                    open_positions=open_positions,
                    magic_number=live_cfg["magic_number"],
                    claimed_position_ids=claimed_open_position_ids,
                )
                if adopted:
                    row = lifecycle.loc[idx]
                    position_value = pd.to_numeric(pd.Series([row.get("mt5_position_id")]), errors="coerce").iloc[0]
                    if pd.notna(position_value):
                        position_id = int(position_value)
                        claimed_open_position_ids.add(position_id)
                    status = str(row.get("status", "")).upper()
                    changed = True
                else:
                    changed = self._sync_pending_entry_order_state(
                        lifecycle=lifecycle,
                        idx=idx,
                        row=lifecycle.loc[idx],
                        mt5_client=mt5_client,
                        pending_orders=pending_orders,
                        position_is_open=False,
                        current_volume=None,
                    ) or changed
                    pending_status = str(lifecycle.at[idx, "pending_order_status"] or "").upper()
                    if status == "PENDING_LIMIT" and pending_status in {"INACTIVE", "CANCELLED_POSITION_CLOSED", "CANCELLED_EXPIRED"}:
                        lifecycle.at[idx, "status"] = "CLOSED"
                        lifecycle.at[idx, "status_detail"] = "Pierna pending cerrada/cancelada sin posicion activa."
                        changed = True
                    continue

            if position_id is None:
                continue

            if (
                position_id not in open_position_ids
                and can_attempt_pending_recovery
                and status == "PENDING_LIMIT"
                and pending_status_upper in {"ACTIVE", "PLACED", "PENDING"}
            ):
                adopted = self._adopt_pending_filled_position(
                    lifecycle=lifecycle,
                    idx=idx,
                    row=lifecycle.loc[idx],
                    open_positions=open_positions,
                    magic_number=live_cfg["magic_number"],
                    claimed_position_ids=claimed_open_position_ids,
                )
                if adopted:
                    row = lifecycle.loc[idx]
                    position_value = pd.to_numeric(pd.Series([row.get("mt5_position_id")]), errors="coerce").iloc[0]
                    if pd.notna(position_value):
                        position_id = int(position_value)
                        claimed_open_position_ids.add(position_id)
                    status = str(row.get("status", "")).upper()
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
                        current_volume = pd.to_numeric(pd.Series([pos_row.get("volume")]), errors="coerce").iloc[0]
                        current_sl = pd.to_numeric(pd.Series([pos_row.get("sl")]), errors="coerce").iloc[0]
                        current_tp = pd.to_numeric(pd.Series([pos_row.get("tp")]), errors="coerce").iloc[0]
                        lifecycle.at[idx, "applied_sl_price"] = current_sl
                        lifecycle.at[idx, "applied_tp_price"] = current_tp
                        lifecycle.at[idx, "remaining_volume_lots_estimate"] = current_volume

                        changed = self._sync_pending_entry_order_state(
                            lifecycle=lifecycle,
                            idx=idx,
                            row=lifecycle.loc[idx],
                            mt5_client=mt5_client,
                            pending_orders=pending_orders,
                            position_is_open=True,
                            current_volume=None if pd.isna(current_volume) else float(current_volume),
                        ) or changed

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

                        stage_managed = self._manage_open_position(
                            lifecycle=lifecycle,
                            idx=idx,
                            row=lifecycle.loc[idx],
                            pos_row=pos_row,
                            mt5_client=mt5_client,
                        )
                        changed = stage_managed or changed
                        if apply_runtime_monitor and not stage_managed:
                            changed = self._apply_runtime_monitor_to_position(
                                lifecycle=lifecycle,
                                idx=idx,
                                row=lifecycle.loc[idx],
                                pos_row=pos_row,
                                mt5_client=mt5_client,
                                feature_row=runtime_feature_row,
                                current_bar_timestamp=current_bar_timestamp,
                            ) or changed
                continue

            if deals is None or deals.empty or "position_id" not in deals.columns:
                deals_pos = pd.DataFrame()
            else:
                deals_pos = deals[
                    pd.to_numeric(deals["position_id"], errors="coerce").fillna(-1).astype(int) == position_id
                ].copy()

            changed = self._sync_pending_entry_order_state(
                lifecycle=lifecycle,
                idx=idx,
                row=lifecycle.loc[idx],
                mt5_client=mt5_client,
                pending_orders=pending_orders,
                position_is_open=False,
                current_volume=None,
            ) or changed

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
        self._save_entry_grid_legs_report_from_lifecycle(lifecycle)
        return lifecycle

    def _execute_live_orders(self, df_rows: pd.DataFrame) -> None:
        """Ejecuta Ã³rdenes reales en MT5 para las seÃ±ales elegibles."""
        if df_rows is None or df_rows.empty:
            return

        live_cfg = self._get_live_trading_settings()
        if not live_cfg["auto_execute_orders"]:
            return

        try:
            mt5_client = self._ensure_mt5_client()
        except Exception as e:
            self.logger.error(f"No se pudo conectar a MT5 para ejecutar Ã³rdenes: {e}")
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
        if not lifecycle.empty:
            if "signal_id" in lifecycle.columns:
                existing_signal_ids.update(lifecycle["signal_id"].dropna().astype(str).tolist())
            if "parent_signal_id" in lifecycle.columns:
                existing_signal_ids.update(lifecycle["parent_signal_id"].dropna().astype(str).tolist())

        open_positions = mt5_client.get_all_positions()
        execution_rows: list[dict[str, Any]] = []

        for _, row in df_rows.iterrows():
            signal = str(row.get("signal", "HOLD")).upper()
            if signal not in {"BUY", "SELL"}:
                continue

            signal_id = self._build_signal_id(row)
            if signal_id in existing_signal_ids:
                self.logger.info(f"â­ SeÃ±al ya ejecutada anteriormente, se omite: {signal_id}")
                continue

            symbol = str(row.get("symbol", ""))
            model_name = str(row.get("model", "UNKNOWN"))
            volume_value = pd.to_numeric(pd.Series([row.get("volume_lots")]), errors="coerce").iloc[0]
            volume_lots = 0.0 if pd.isna(volume_value) else float(volume_value)
            market_volume_value = pd.to_numeric(
                pd.Series([row.get("initial_market_volume_lots")]),
                errors="coerce",
            ).iloc[0]
            market_volume_lots = float(market_volume_value) if pd.notna(market_volume_value) else volume_lots
            pending_volume_value = pd.to_numeric(
                pd.Series([row.get("pending_order_volume_lots")]),
                errors="coerce",
            ).iloc[0]
            pending_volume_lots = 0.0 if pd.isna(pending_volume_value) else float(pending_volume_value)
            pending_price_value = pd.to_numeric(
                pd.Series([row.get("pending_order_price")]),
                errors="coerce",
            ).iloc[0]
            pending_price = None if pd.isna(pending_price_value) else float(pending_price_value)
            pending_sl_value = pd.to_numeric(
                pd.Series([row.get("pending_order_sl_price")]),
                errors="coerce",
            ).iloc[0]
            pending_sl_price = None if pd.isna(pending_sl_value) else float(pending_sl_value)
            pending_tp_value = pd.to_numeric(
                pd.Series([row.get("pending_order_tp_price")]),
                errors="coerce",
            ).iloc[0]
            pending_tp_price = None if pd.isna(pending_tp_value) else float(pending_tp_value)
            pending_order_type = str(row.get("pending_order_type", "") or "").upper()
            pending_expiry_time = row.get("pending_order_expiry_time")
            digits_value = pd.to_numeric(
                pd.Series([row.get("symbol_digits", row.get("digits"))]),
                errors="coerce",
            ).iloc[0]
            digits = int(digits_value) if pd.notna(digits_value) else 5
            grid_enabled = self._coerce_boolish(row.get("entry_grid_enabled"))
            grid_plan = self._parse_entry_grid_plan_from_row(row) if grid_enabled else None
            grid_group_id = row.get("entry_grid_group_id")
            grid_market_leg = None
            grid_pending_legs: list[dict[str, Any]] = []
            if grid_plan and grid_plan.get("enabled"):
                grid_legs = grid_plan.get("legs", []) or []
                grid_market_leg = next(
                    (leg for leg in grid_legs if str(leg.get("entry_type", "")).strip().lower() == "market"),
                    None,
                )
                grid_pending_legs = [
                    leg
                    for leg in grid_legs
                    if str(leg.get("entry_type", "")).strip().lower() == "limit"
                ]
                if grid_market_leg is not None:
                    market_volume_lots = float(grid_market_leg.get("planned_volume_lots") or market_volume_lots)
                    pending_volume_lots = 0.0
                    pending_price = None
                    pending_sl_price = None
                    pending_tp_price = None
                    pending_order_type = ""
                    pending_expiry_time = None
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
                "parent_signal_id": signal_id,
                "entry_leg": "market",
                "grid_parent_signal_id": signal_id if grid_plan and grid_plan.get("enabled") else pd.NA,
                "grid_group_id": grid_group_id if grid_plan and grid_plan.get("enabled") else pd.NA,
                "grid_leg_id": grid_market_leg.get("leg_id") if grid_market_leg is not None else pd.NA,
                "grid_leg_rank": grid_market_leg.get("leg_rank") if grid_market_leg is not None else np.nan,
                "grid_entry_type": grid_market_leg.get("entry_type") if grid_market_leg is not None else pd.NA,
                "grid_volume_weight": grid_market_leg.get("volume_weight") if grid_market_leg is not None else np.nan,
                "grid_quality_rank": grid_market_leg.get("grid_quality_rank") if grid_market_leg is not None else np.nan,
                "grid_runner_candidate": bool(grid_market_leg.get("runner_candidate")) if grid_market_leg is not None else False,
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
                "requested_volume_lots": market_volume_lots if grid_market_leg is not None else volume_lots,
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
                "managed_stage_ids": "",
                "break_even_applied": False,
                "break_even_applied_time": None,
                "break_even_sl_price": None,
                "last_management_time": None,
                "last_management_action": None,
                "last_partial_close_volume": 0.0,
                "partial_close_total_volume": 0.0,
                "remaining_volume_lots_estimate": market_volume_lots,
                "management_progress_to_tp": None,
                "trade_management_comment": None,
                "entry_management_mode": row.get("entry_management_mode"),
                "entry_management_split_active": row.get("entry_management_split_active"),
                "entry_management_initial_market_fraction": row.get("entry_management_initial_market_fraction"),
                "entry_management_pending_fraction": row.get("entry_management_pending_fraction"),
                "entry_management_retrace_fraction_of_stop": row.get("entry_management_retrace_fraction_of_stop"),
                "entry_management_total_volume_lots": row.get("entry_management_total_volume_lots"),
                "initial_market_volume_lots": market_volume_lots,
                "pending_order_volume_lots": pending_volume_lots,
                "pending_order_price": pending_price,
                "pending_order_type": pending_order_type or None,
                "pending_order_sl_price": pending_sl_price,
                "pending_order_tp_price": pending_tp_price,
                "pending_order_ticket": None,
                "pending_order_expiry_time": pending_expiry_time,
                "pending_order_status": None,
                "pending_order_comment": None,
                "pending_order_last_sync_time": None,
                "entry_management_comment": row.get("entry_management_comment"),
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

            if market_volume_lots <= 0:
                base_record["status"] = "SKIPPED_NO_VOLUME"
                base_record["status_detail"] = "El tamaÃ±o de posiciÃ³n de mercado calculado fue <= 0."
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

            skip_cluster_entry, cluster_detail = self._should_skip_live_entry_for_cluster_guard(
                lifecycle=lifecycle,
                row=row,
            )
            if skip_cluster_entry:
                base_record["status"] = "SKIPPED_CLUSTER_GUARD"
                base_record["status_detail"] = cluster_detail
                execution_rows.append(base_record)
                existing_signal_ids.add(signal_id)
                continue

            cluster_override_mode, cluster_override_detail = self._should_force_market_only_for_cluster_trend(
                lifecycle=lifecycle,
                row=row,
            )
            if cluster_override_mode == "market_only":
                pending_volume_lots = 0.0
                pending_price = None
                pending_order_type = None
                pending_sl_price = None
                pending_tp_price = None
                pending_expiry_time = None
                base_record["pending_order_volume_lots"] = 0.0
                base_record["pending_order_price"] = None
                base_record["pending_order_type"] = None
                base_record["pending_order_sl_price"] = None
                base_record["pending_order_tp_price"] = None
                base_record["pending_order_expiry_time"] = None
                base_record["entry_management_comment"] = "cluster_trend_market_only"
                base_record["status_detail"] = cluster_override_detail
            elif cluster_override_mode == "retrace_only":
                retrace_only_plan = self._build_retrace_only_entry_plan(
                    signal=signal,
                    total_volume_lots=volume_lots,
                    live_entry_price=float(row.get("live_entry_price")) if pd.notna(pd.to_numeric(pd.Series([row.get("live_entry_price")]), errors="coerce").iloc[0]) else float("nan"),
                    live_sl_price=float(sl_price) if pd.notna(sl_price) else float("nan"),
                    live_tp_price=float(tp_price) if pd.notna(tp_price) else float("nan"),
                    digits=digits,
                    timeframe=self._coerce_textish(row.get("timeframe"), self.config.get("data", {}).get("timeframe", "M5")),
                    signal_time=row.get("timestamp", row.get("signal_time", row.get("Time"))),
                    comment="cluster_trend_retrace_only",
                )
                market_volume_lots = 0.0
                pending_volume_lots = float(retrace_only_plan.get("pending_order_volume_lots") or 0.0)
                pending_price_value = pd.to_numeric(
                    pd.Series([retrace_only_plan.get("pending_order_price")]),
                    errors="coerce",
                ).iloc[0]
                pending_price = None if pd.isna(pending_price_value) else float(pending_price_value)
                pending_order_type = str(retrace_only_plan.get("pending_order_type") or "").upper() or None
                pending_sl_value = pd.to_numeric(
                    pd.Series([retrace_only_plan.get("pending_order_sl_price")]),
                    errors="coerce",
                ).iloc[0]
                pending_sl_price = None if pd.isna(pending_sl_value) else float(pending_sl_value)
                pending_tp_value = pd.to_numeric(
                    pd.Series([retrace_only_plan.get("pending_order_tp_price")]),
                    errors="coerce",
                ).iloc[0]
                pending_tp_price = None if pd.isna(pending_tp_value) else float(pending_tp_value)
                pending_expiry_time = retrace_only_plan.get("pending_order_expiry_time")
                base_record["entry_leg"] = "pending_limit"
                base_record["requested_volume_lots"] = pending_volume_lots
                base_record["remaining_volume_lots_estimate"] = pending_volume_lots
                base_record["entry_management_mode"] = retrace_only_plan.get("entry_management_mode")
                base_record["entry_management_split_active"] = retrace_only_plan.get("entry_management_split_active")
                base_record["entry_management_initial_market_fraction"] = retrace_only_plan.get(
                    "entry_management_initial_market_fraction"
                )
                base_record["entry_management_pending_fraction"] = retrace_only_plan.get(
                    "entry_management_pending_fraction"
                )
                base_record["entry_management_retrace_fraction_of_stop"] = retrace_only_plan.get(
                    "entry_management_retrace_fraction_of_stop"
                )
                base_record["entry_management_total_volume_lots"] = retrace_only_plan.get(
                    "entry_management_total_volume_lots"
                )
                base_record["initial_market_volume_lots"] = 0.0
                base_record["pending_order_volume_lots"] = pending_volume_lots
                base_record["pending_order_price"] = pending_price
                base_record["pending_order_type"] = pending_order_type
                base_record["pending_order_sl_price"] = pending_sl_price
                base_record["pending_order_tp_price"] = pending_tp_price
                base_record["pending_order_expiry_time"] = pending_expiry_time
                base_record["entry_management_comment"] = str(
                    retrace_only_plan.get("entry_management_comment") or "cluster_trend_retrace_only"
                )
                base_record["status_detail"] = cluster_override_detail

            if (
                str(base_record.get("entry_management_comment") or row.get("entry_management_comment") or "").strip().lower()
                == "split_retrace_limit"
                and self._should_force_split_retrace_filter_opposite_retrace_only(
                    signal=signal,
                    row=row,
                )
            ):
                retrace_only_plan = self._build_retrace_only_entry_plan(
                    signal=signal,
                    total_volume_lots=volume_lots,
                    live_entry_price=float(row.get("live_entry_price"))
                    if pd.notna(pd.to_numeric(pd.Series([row.get("live_entry_price")]), errors="coerce").iloc[0])
                    else float("nan"),
                    live_sl_price=float(sl_price) if pd.notna(sl_price) else float("nan"),
                    live_tp_price=float(tp_price) if pd.notna(tp_price) else float("nan"),
                    digits=digits,
                    timeframe=self._coerce_textish(row.get("timeframe"), self.config.get("data", {}).get("timeframe", "M5")),
                    signal_time=row.get("timestamp", row.get("signal_time", row.get("Time"))),
                    comment="split_retrace_filter_opposite_retrace_only",
                )
                market_volume_lots = 0.0
                pending_volume_lots = float(retrace_only_plan.get("pending_order_volume_lots") or 0.0)
                pending_price_value = pd.to_numeric(
                    pd.Series([retrace_only_plan.get("pending_order_price")]),
                    errors="coerce",
                ).iloc[0]
                pending_price = None if pd.isna(pending_price_value) else float(pending_price_value)
                pending_order_type = str(retrace_only_plan.get("pending_order_type") or "").upper() or None
                pending_sl_value = pd.to_numeric(
                    pd.Series([retrace_only_plan.get("pending_order_sl_price")]),
                    errors="coerce",
                ).iloc[0]
                pending_sl_price = None if pd.isna(pending_sl_value) else float(pending_sl_value)
                pending_tp_value = pd.to_numeric(
                    pd.Series([retrace_only_plan.get("pending_order_tp_price")]),
                    errors="coerce",
                ).iloc[0]
                pending_tp_price = None if pd.isna(pending_tp_value) else float(pending_tp_value)
                pending_expiry_time = retrace_only_plan.get("pending_order_expiry_time")
                base_record["entry_leg"] = "pending_limit"
                base_record["requested_volume_lots"] = pending_volume_lots
                base_record["remaining_volume_lots_estimate"] = pending_volume_lots
                base_record["entry_management_mode"] = retrace_only_plan.get("entry_management_mode")
                base_record["entry_management_split_active"] = retrace_only_plan.get("entry_management_split_active")
                base_record["entry_management_initial_market_fraction"] = retrace_only_plan.get(
                    "entry_management_initial_market_fraction"
                )
                base_record["entry_management_pending_fraction"] = retrace_only_plan.get(
                    "entry_management_pending_fraction"
                )
                base_record["entry_management_retrace_fraction_of_stop"] = retrace_only_plan.get(
                    "entry_management_retrace_fraction_of_stop"
                )
                base_record["entry_management_total_volume_lots"] = retrace_only_plan.get(
                    "entry_management_total_volume_lots"
                )
                base_record["initial_market_volume_lots"] = 0.0
                base_record["pending_order_volume_lots"] = pending_volume_lots
                base_record["pending_order_price"] = pending_price
                base_record["pending_order_type"] = pending_order_type
                base_record["pending_order_sl_price"] = pending_sl_price
                base_record["pending_order_tp_price"] = pending_tp_price
                base_record["pending_order_expiry_time"] = pending_expiry_time
                base_record["entry_management_comment"] = str(
                    retrace_only_plan.get("entry_management_comment") or "split_retrace_filter_opposite_retrace_only"
                )
                base_record["status_detail"] = (
                    "Split retrace degradado a retrace-only por filtro opuesto y vela tipo spike/rechazo."
                )

            if (
                not live_cfg["allow_multiple_positions"]
                and open_positions is not None
                and not open_positions.empty
                and "symbol" in open_positions.columns
            ):
                same_symbol = open_positions[open_positions["symbol"] == symbol]
                if not same_symbol.empty:
                    base_record["status"] = "SKIPPED_OPEN_POSITION"
                    base_record["status_detail"] = f"Ya existe una posiciÃ³n abierta para {symbol}."
                execution_rows.append(base_record)
                existing_signal_ids.add(signal_id)
                continue

            pending_only_candidate = (
                market_volume_lots <= 0.0
                and pending_volume_lots > 0.0
                and pending_price is not None
                and pending_order_type in {"BUY_LIMIT", "SELL_LIMIT"}
            )
            if pending_only_candidate:
                pending_comment = f"{self._build_order_comment(model_name)}R"[:31]
                pending_result = mt5_client.open_pending_limit_order(
                    symbol=symbol,
                    volume=pending_volume_lots,
                    side=signal,
                    price=float(pending_price),
                    comment=pending_comment,
                    sl=pending_sl_price,
                    tp=pending_tp_price,
                    magic=live_cfg["magic_number"],
                )
                base_record["status"] = "PENDING_LIMIT" if pending_result.get("success") else "FAILED"
                base_record["status_detail"] = (
                    "Pending LIMIT registrada como entrada principal."
                    if pending_result.get("success")
                    else pending_result.get("comment")
                )
                base_record["mt5_order_ticket"] = pending_result.get("order")
                base_record["mt5_deal_ticket"] = pending_result.get("deal")
                base_record["mt5_position_id"] = None
                base_record["execution_price"] = None
                base_record["execution_retcode"] = pending_result.get("retcode")
                base_record["execution_comment"] = pending_result.get("comment")
                base_record["applied_sl_price"] = pending_sl_price
                base_record["applied_tp_price"] = pending_tp_price
                base_record["protection_status"] = None
                base_record["protection_comment"] = (
                    "Pending LIMIT registrada; la proteccion se aplicara cuando se active."
                    if pending_result.get("success")
                    else None
                )
                base_record["pending_order_ticket"] = pending_result.get("order")
                base_record["pending_order_status"] = "ACTIVE" if pending_result.get("success") else "FAILED"
                base_record["pending_order_comment"] = (
                    "Entrada principal en retrace registrada."
                    if pending_result.get("success")
                    else pending_result.get("comment")
                )
                execution_rows.append(base_record)
                existing_signal_ids.add(signal_id)
                continue

            result = mt5_client.open_market_order(
                symbol=symbol,
                volume=market_volume_lots,
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

            if result.get("success") and grid_plan and grid_plan.get("enabled") and grid_market_leg is not None:
                base_record["pending_order_status"] = "TRACKED_IN_CHILD" if grid_pending_legs else "INACTIVE"
                base_record["pending_order_comment"] = (
                    "Grid legs pendientes registradas en filas separadas."
                    if grid_pending_legs
                    else "Entry grid sin piernas pendientes."
                )
                execution_rows.append(base_record)
                existing_signal_ids.add(signal_id)

                for pending_leg in grid_pending_legs:
                    leg_signal_id = f"{signal_id}|GRID:{str(pending_leg.get('leg_id') or '').upper()}"
                    pending_result = mt5_client.open_pending_limit_order(
                        symbol=symbol,
                        volume=float(pending_leg.get("planned_volume_lots") or 0.0),
                        side=signal,
                        price=float(pending_leg.get("planned_entry_price")),
                        comment=f"{self._build_order_comment(model_name)}{pending_leg.get('leg_rank', 0)}"[:31],
                        sl=pending_leg.get("planned_sl_price"),
                        tp=pending_leg.get("planned_tp_price"),
                        magic=live_cfg["magic_number"],
                    )
                    pending_record = dict(base_record)
                    pending_record.update(
                        {
                            "signal_id": leg_signal_id,
                            "parent_signal_id": signal_id,
                            "entry_leg": pending_leg.get("entry_leg") or f"grid_limit_{pending_leg.get('leg_rank')}",
                            "grid_parent_signal_id": signal_id,
                            "grid_group_id": grid_group_id,
                            "grid_leg_id": pending_leg.get("leg_id"),
                            "grid_leg_rank": pending_leg.get("leg_rank"),
                            "grid_entry_type": pending_leg.get("entry_type"),
                            "grid_volume_weight": pending_leg.get("volume_weight"),
                            "grid_quality_rank": pending_leg.get("grid_quality_rank"),
                            "grid_runner_candidate": bool(pending_leg.get("runner_candidate")),
                            "status": "PENDING_LIMIT" if pending_result.get("success") else "FAILED",
                            "status_detail": (
                                "Grid LIMIT pendiente registrada."
                                if pending_result.get("success")
                                else pending_result.get("comment")
                            ),
                            "mt5_order_ticket": pending_result.get("order"),
                            "mt5_deal_ticket": pending_result.get("deal"),
                            "mt5_position_id": None,
                            "execution_price": None,
                            "execution_retcode": pending_result.get("retcode"),
                            "execution_comment": pending_result.get("comment"),
                            "requested_entry_price": pending_leg.get("planned_entry_price"),
                            "requested_live_entry_price": pending_leg.get("planned_entry_price"),
                            "requested_sl_price": pending_leg.get("planned_sl_price"),
                            "requested_tp_price": pending_leg.get("planned_tp_price"),
                            "requested_plan_sl_price": pending_leg.get("planned_sl_price"),
                            "requested_plan_tp_price": pending_leg.get("planned_tp_price"),
                            "requested_volume_lots": pending_leg.get("planned_volume_lots"),
                            "execution_time": None,
                            "applied_sl_price": pending_leg.get("planned_sl_price"),
                            "applied_tp_price": pending_leg.get("planned_tp_price"),
                            "protection_status": None,
                            "protection_comment": None,
                            "last_sync_time": None,
                            "break_even_applied": False,
                            "break_even_applied_time": None,
                            "break_even_sl_price": None,
                            "last_management_time": None,
                            "last_management_action": None,
                            "last_partial_close_volume": 0.0,
                            "partial_close_total_volume": 0.0,
                            "remaining_volume_lots_estimate": pending_leg.get("planned_volume_lots"),
                            "management_progress_to_tp": None,
                            "trade_management_comment": None,
                            "entry_management_total_volume_lots": row.get("entry_management_total_volume_lots"),
                            "initial_market_volume_lots": pending_leg.get("planned_volume_lots"),
                            "pending_order_volume_lots": pending_leg.get("planned_volume_lots"),
                            "pending_order_price": pending_leg.get("planned_entry_price"),
                            "pending_order_type": "BUY_LIMIT" if signal == "BUY" else "SELL_LIMIT",
                            "pending_order_sl_price": pending_leg.get("planned_sl_price"),
                            "pending_order_tp_price": pending_leg.get("planned_tp_price"),
                            "pending_order_ticket": pending_result.get("order"),
                            "pending_order_expiry_time": pending_leg.get("expiry_time"),
                            "pending_order_status": "ACTIVE" if pending_result.get("success") else "FAILED",
                            "pending_order_comment": pending_result.get("comment"),
                            "pending_order_last_sync_time": None,
                            "entry_management_comment": "risk_based_ladder_pending_leg",
                        }
                    )
                    execution_rows.append(pending_record)
                    existing_signal_ids.add(leg_signal_id)

                self.logger.info(
                    f"✅ Grid enviada: model={model_name} signal={signal} symbol={symbol} "
                    f"legs={len(grid_plan.get('legs', []))} market={market_volume_lots:.2f} "
                    f"pending={len(grid_pending_legs)} position_id={result.get('position_id')} "
                    f"group={grid_group_id} SL={base_record['applied_sl_price']} TP={base_record['applied_tp_price']}"
                )
                open_positions = mt5_client.get_all_positions()
                continue

            if (
                result.get("success")
                and pending_volume_lots > 0.0
                and pending_price is not None
                and pending_order_type in {"BUY_LIMIT", "SELL_LIMIT"}
            ):
                pending_comment = f"{self._build_order_comment(model_name)}P"[:31]
                pending_result = mt5_client.open_pending_limit_order(
                    symbol=symbol,
                    volume=pending_volume_lots,
                    side=signal,
                    price=float(pending_price),
                    comment=pending_comment,
                    sl=pending_sl_price,
                    tp=pending_tp_price,
                    magic=live_cfg["magic_number"],
                )
                if pending_result.get("success"):
                    base_record["pending_order_status"] = "TRACKED_IN_CHILD"
                    base_record["pending_order_comment"] = "Pierna pending registrada en fila separada."
                    base_record["pending_order_ticket"] = None
                else:
                    base_record["pending_order_ticket"] = pending_result.get("order")
                    base_record["pending_order_status"] = "FAILED"
                    base_record["pending_order_comment"] = pending_result.get("comment")
                    base_record["status_detail"] = (
                        f"{base_record['status_detail']} | pending_entry_failed: {pending_result.get('comment')}"
                    ).strip()
                    execution_rows.append(base_record)
                    existing_signal_ids.add(signal_id)
                    if result.get("success"):
                        self.logger.info(
                            f"âœ… Orden enviada: model={model_name} signal={signal} symbol={symbol} "
                            f"lots_market={market_volume_lots:.2f} lots_pending={pending_volume_lots:.2f} "
                            f"position_id={result.get('position_id')} pending_status={base_record['pending_order_status']} "
                            f"SL={base_record['applied_sl_price']} TP={base_record['applied_tp_price']} "
                            f"protection={base_record['protection_status']}"
                        )
                        open_positions = mt5_client.get_all_positions()
                    else:
                        self.logger.error(
                            f"âŒ FallÃ³ la ejecuciÃ³n de {model_name} en {symbol}: {result.get('comment')} "
                            f"(retcode={result.get('retcode')})"
                        )
                    continue

                pending_signal_id = f"{signal_id}|LEG:PENDING"
                pending_record = dict(base_record)
                pending_record.update(
                    {
                        "signal_id": pending_signal_id,
                        "parent_signal_id": signal_id,
                        "entry_leg": "pending_limit",
                        "status": "PENDING_LIMIT",
                        "status_detail": "Orden LIMIT pendiente registrada.",
                        "mt5_order_ticket": pending_result.get("order"),
                        "mt5_deal_ticket": pending_result.get("deal"),
                        "mt5_position_id": None,
                        "execution_price": None,
                        "execution_retcode": pending_result.get("retcode"),
                        "execution_comment": pending_result.get("comment"),
                        "requested_entry_price": pending_price,
                        "requested_live_entry_price": pending_price,
                        "requested_sl_price": pending_sl_price,
                        "requested_tp_price": pending_tp_price,
                        "requested_plan_sl_price": pending_sl_price,
                        "requested_plan_tp_price": pending_tp_price,
                        "requested_volume_lots": pending_volume_lots,
                        "risk_amount": row.get("risk_amount"),
                        "allocated_risk_budget": row.get("allocated_risk_budget"),
                        "execution_time": None,
                        "applied_sl_price": pending_sl_price,
                        "applied_tp_price": pending_tp_price,
                        "protection_status": None,
                        "protection_comment": None,
                        "last_sync_time": None,
                        "break_even_applied": False,
                        "break_even_applied_time": None,
                        "break_even_sl_price": None,
                        "last_management_time": None,
                        "last_management_action": None,
                        "last_partial_close_volume": 0.0,
                        "partial_close_total_volume": 0.0,
                        "remaining_volume_lots_estimate": pending_volume_lots,
                        "management_progress_to_tp": None,
                        "trade_management_comment": None,
                        "entry_management_total_volume_lots": pending_volume_lots,
                        "initial_market_volume_lots": pending_volume_lots,
                        "pending_order_volume_lots": pending_volume_lots,
                        "pending_order_price": pending_price,
                        "pending_order_type": pending_order_type,
                        "pending_order_sl_price": pending_sl_price,
                        "pending_order_tp_price": pending_tp_price,
                        "pending_order_ticket": pending_result.get("order"),
                        "pending_order_expiry_time": pending_expiry_time,
                        "pending_order_status": "ACTIVE",
                        "pending_order_comment": pending_result.get("comment"),
                        "pending_order_last_sync_time": None,
                        "entry_management_comment": "split_retrace_limit_pending_leg",
                    }
                )
                execution_rows.append(pending_record)
                existing_signal_ids.add(pending_signal_id)
            elif result.get("success") and pending_volume_lots > 0.0:
                base_record["pending_order_status"] = "SKIPPED"
                base_record["pending_order_comment"] = "Plan de entrada incompleto: pending sin precio/tipo valido."
            else:
                base_record["pending_order_status"] = "INACTIVE"
                base_record["pending_order_comment"] = "Entrada escalonada no aplicada."
            execution_rows.append(base_record)
            existing_signal_ids.add(signal_id)

            if result.get("success"):
                self.logger.info(
                    f"âœ… Orden enviada: model={model_name} signal={signal} symbol={symbol} "
                    f"lots_market={market_volume_lots:.2f} lots_pending={pending_volume_lots:.2f} "
                    f"position_id={result.get('position_id')} pending_status={base_record['pending_order_status']} "
                    f"SL={base_record['applied_sl_price']} TP={base_record['applied_tp_price']} "
                    f"protection={base_record['protection_status']}"
                )
                open_positions = mt5_client.get_all_positions()
            else:
                self.logger.error(
                    f"âŒ FallÃ³ la ejecuciÃ³n de {model_name} en {symbol}: {result.get('comment')} "
                    f"(retcode={result.get('retcode')})"
                )

        if execution_rows:
            df_exec = pd.DataFrame(execution_rows)
            self._append_rows_to_csv(paths["lifecycle"], df_exec)
            self._sync_live_trade_report()
    
    def _save_backtest_detail(self, model_name: str, df_bt: pd.DataFrame) -> None:
        """
        Guarda el detalle del mejor backtest para cada modelo.
        Crea CSV y, opcionalmente, Excel con seÃ±ales, precios y pips.
        """
        if df_bt is None or df_bt.empty:
            return

        output_dir = self._get_backtest_output_dir()

        csv_path = output_dir / f"{model_name}_best_backtest_detail.csv"
        df_bt.to_csv(csv_path)
        self.logger.info(f"    ðŸ’¾ Detalle de backtest guardado en: {csv_path}")
        self._archive_backtest_artifact(csv_path)

        # Si quieres tambiÃ©n Excel
        if "excel" in self.config.get("output", {}).get("formats", []):
            xlsx_path = output_dir / f"{model_name}_best_backtest_detail.xlsx"
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                df_bt.to_excel(writer, sheet_name="backtest_detail")
            self.logger.info(f"    ðŸ’¾ Detalle de backtest (Excel) guardado en: {xlsx_path}")
            self._archive_backtest_artifact(xlsx_path)

    
    def _load_config(self, config_path: str) -> tuple[Dict[str, Any], str]:
        """Carga configuraciÃ³n desde YAML"""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"El archivo de configuraciÃ³n no se encontrÃ³ en: {config_path}")
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
        self._quiet_runtime_monitor = os.environ.get("MARKIII_QUIET_RUNTIME_MONITOR", "0") == "1"
        if not self._quiet_runtime_monitor:
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
        if getattr(self, "_quiet_runtime_monitor", False):
            return
        
        self.logger.info("ðŸ“ Directorios de trabajo configurados")
    
    def run(self, mode: str = None) -> None:
        """
        Ejecuta el pipeline segÃºn el modo especificado
        
        Args:
            mode: "eda", "train", "backtest", "production"
                 Si es None, usa el modo del config
        """
        mode = mode or self.config.get("execution", {}).get("mode", "eda")
        self._active_mode = str(mode).lower()
        self._runtime_monitor_previous_level = None
        if self._active_mode == "monitor_runtime" and getattr(self, "_quiet_runtime_monitor", False):
            self._runtime_monitor_previous_level = self.logger.level
            self.logger.setLevel(max(self._runtime_monitor_previous_level, logging.WARNING))
        
        self.logger.info(f"ðŸš€ Ejecutando modo: {mode.upper()}")
        
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
        elif mode == "monitor_runtime":
            self._run_monitor_runtime_mode()
        elif mode == "clear_cache":
            self._run_clear_cache_mode()
        else:
            raise ValueError(f"Modo no soportado: {mode}")
    
    def _run_eda_mode(self) -> None:
        """
        Modo EDA: Carga â†’ Limpia â†’ Analiza
        Genera reportes estadÃ­sticos y grÃ¡ficos
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("MODO: ANÃLISIS EXPLORATORIO (EDA)")
        self.logger.info("="*60 + "\n")
        
        # 1. Cargar datos
        df_raw = self._load_data()
        
        # 2. Limpiar datos
        df_clean = self._clean_data(df_raw)
        
        # 3. Generar features (opcional para EDA)
        df_features = self._generate_features(df_clean)
        
        # 4. AnÃ¡lisis exploratorio
        self._perform_eda(df_features)
        
        # 5. Guardar datos en diferentes formatos
        self._save_processed_data(df_features)
        self._save_dataframes_to_excel({
            "Raw Data": df_raw,
            "Cleaned Data": df_clean,
            "Features Data": df_features
        })
        
        self.logger.info("\nâœ… MODO EDA COMPLETADO")
    
    def _run_train_mode(self) -> None:
        """
        Modo Train: Entrena modelos y guarda para producciÃ³n
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("MODO: ENTRENAMIENTO DE MODELOS")
        self.logger.info("="*60 + "\n")
        self._start_backtest_run()
        
        # --- PASO 1: Carga, Limpieza y GeneraciÃ³n de Features ---
        df = self._load_data()
        df_clean = self._clean_data(df)
        df_features = self._generate_features(df_clean)

        # --- PASO 2: DivisiÃ³n en Train y Test ---
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
        self.logger.info(f"âœ“ Datos de entrenamiento: {len(df_train)} filas")
        self.logger.info(f"âœ“ Datos de prueba (hold-out): {len(df_test)} filas")

        # --- PASO 3: BÃºsqueda de HiperparÃ¡metros (usando el set de TRAIN) ---
        self.logger.info("\n" + "="*60)
        self.logger.info("MODO: BÃšSQUEDA DE HIPERPARÃMETROS (SOBRE TRAIN SET)")
        self.logger.info("="*60 + "\n")
        self._run_hyperparameter_tuning(df_train)

        # --- PASO 4: ValidaciÃ³n Final (usando el set de TEST) ---
        self.logger.info("\n" + "="*60)
        self.logger.info("MODO: VALIDACIÃ“N FINAL (SOBRE TEST SET)")
        self.logger.info("="*60 + "\n")
        
        # Cargar la configuraciÃ³n reciÃ©n optimizada
        optimized_config_path = self._resolve_active_release_config_path()
        if not optimized_config_path.exists():
            self.logger.error("No se encontrÃ³ 'config_optimizado.yaml'. Ejecute el backtest primero.")
            return
        
        # Crear un nuevo pipeline temporal para la validaciÃ³n
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
            # AquÃ­ irÃ­a la lÃ³gica para cargar el modelo guardado (.h5, .joblib)
            # y predecir sobre df_test, luego calcular mÃ©tricas.
            # Por simplicidad, re-entrenamos y predecimos en un solo paso.
            self._validate_model_on_test(model_name, model_config.get("params", {}), df_train, y_test, X_test)
        
        self.logger.info("\nâœ… MODO TRAIN COMPLETADO")

    def _run_backtest_mode(self) -> None:
        """
        Modo BACKTEST:
        - Carga datos histÃ³ricos
        - Genera features
        - (Opcional) Reserva un hold-out final segÃºn config['validation']
        - Ejecuta bÃºsqueda de hiperparÃ¡metros SOLO sobre la parte in-sample
        - Guarda resultados y deja preparado self._df_features_last_backtest
          para reentrenar los modelos Ã³ptimos.
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

        # âœ… Limpiar NaNs de target + features (sin bfill para evitar leakage)
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
                f"ðŸ”’ Hold-out activado: se reservan los Ãºltimos {n_holdout} puntos "
                f"({len(df_bt)} usados para backtest)."
            )
        else:
            df_bt = df_features
            df_holdout = None
            if mode == "last_n" and n_holdout > 0:
                self.logger.warning(
                    f"validation.mode=last_n pero n={n_holdout} es mayor o igual "
                    f"al tamaÃ±o de la serie ({len(df_features)}). Se ignora hold-out."
                )
            else:
                self.logger.info("Sin hold-out: se usa toda la serie para backtest.")

        # ðŸ’¾ Guardar features IN-SAMPLE para reentrenar modelos Ã³ptimos
        self._df_features_last_backtest = df_bt.copy()

        # 3) Ejecutar tuning de hiperparÃ¡metros sobre df_bt (in-sample)
        self._run_hyperparameter_tuning(df_bt)


        self.logger.info("\nâœ… MODO BACKTEST COMPLETADO")


    def _run_hyperparameter_tuning(self, df_features: pd.DataFrame) -> None:
        """Orquesta el backtesting con bÃºsqueda de hiperparÃ¡metros."""
        self.logger.info("íŠœ PASO 4: INICIANDO BÃšSQUEDA DE HIPERPARÃMETROS")
        self.logger.info("-" * 60)

        all_results = []
        models_config = self.config.get("models", [])
        best_artifacts_by_model: dict[str, dict[str, Any]] = {}

        for model_config in models_config:
            if not model_config.get("enabled", False):
                continue

            model_name = model_config["name"]
            self.logger.info(f"\nðŸ”¥ Procesando modelo: {model_name}")

            if "params" in model_config:
                raw_param_grid = model_config["params"]
            else:
                raw_param_grid = model_config.get("param_grid", {})

            param_grid = {}
            for param_name, param_value in (raw_param_grid or {}).items():
                if isinstance(param_value, (list, tuple, np.ndarray)):
                    param_grid[param_name] = list(param_value)
                else:
                    # Backtest configs sometimes store fixed params as scalars.
                    param_grid[param_name] = [param_value]

            grid = ParameterGrid(param_grid)
            model_results = []
            selection_rows = []
            series_candidates = []

            for i, params in enumerate(grid):
                self.logger.info(f"  -> Probando combinaciÃ³n {i+1}/{len(grid)}: {params}")

                # Devuelve predicciones, valores reales, fechas y contexto para reconstruir seÃ±ales.
                predictions, true_values, timestamps, feature_rows, price_references, prediction_details = self._run_walk_forward_for_params(
                    df_features,
                    model_name,
                    params,
                    model_cfg=model_config,
                )

                if not predictions:
                    self.logger.warning("    No se generaron predicciones, saltando mÃ©tricas.")
                    continue

                provisional_trade_mask, _, _, _, _ = self._build_trade_decisions_for_predictions(
                    predictions=predictions,
                    feature_rows=feature_rows,
                    price_references=price_references,
                    prediction_details=prediction_details,
                    model_metrics={},
                    apply_confidence_filter=False,
                    model_name=model_name,
                    model_cfg=model_config,
                )
                provisional_metrics = self._calculate_metrics(
                    true_values,
                    predictions,
                    trade_mask=provisional_trade_mask,
                )
                trade_mask, confirmation_reasons, _, _, signal_details = self._build_trade_decisions_for_predictions(
                    predictions=predictions,
                    feature_rows=feature_rows,
                    price_references=price_references,
                    prediction_details=prediction_details,
                    model_metrics=provisional_metrics,
                    apply_confidence_filter=True,
                    model_name=model_name,
                    model_cfg=model_config,
                )
                metrics = self._calculate_metrics(true_values, predictions, trade_mask=trade_mask)
                self.logger.info(f"    - MÃ©tricas: {metrics}")

                result_row = {
                    "model": model_name,
                    "selection_role": self._get_model_selection_role(
                        model_name=model_name,
                        model_cfg=model_config,
                    ),
                    **params,
                    **metrics,
                }
                model_results.append(result_row)
                all_results.append(result_row)
                selection_rows.append({**result_row, "_artifact_idx": len(series_candidates)})
                series_candidates.append(
                    {
                        "dates": timestamps,
                        "y_true": true_values,
                        "y_pred": predictions,
                        "feature_rows": feature_rows,
                        "price_references": price_references,
                        "prediction_details": prediction_details,
                        "trade_mask": trade_mask,
                        "confirmation_reasons": confirmation_reasons,
                        "signal_details": signal_details,
                        "params": params,
                    }
                )

            # ==================== CAMBIO IMPORTANTE =====================
            # Guardamos y graficamos la serie del run ganador segÃºn
            # config.model_selection, no segÃºn una mÃ©trica hardcodeada.
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
                best_artifacts_by_model[model_name.upper()] = {
                    "model_name": model_name,
                    "model_cfg": model_config,
                    "best_run": best_row.to_dict() if best_row is not None else {},
                    "params": best_series["params"],
                    "dates": best_series["dates"],
                    "y_true": best_series["y_true"],
                    "y_pred": best_series["y_pred"],
                    "feature_rows": best_series.get("feature_rows", []),
                    "price_references": best_series.get("price_references", []),
                    "prediction_details": best_series.get("prediction_details", []),
                    "trade_mask": best_series.get("trade_mask"),
                    "confirmation_reasons": best_series.get("confirmation_reasons"),
                    "signal_details": best_series.get("signal_details"),
                }
                # Guardar CSV con la serie de backtest del mejor run
                self._save_backtest_series(
                    model_name=model_name,
                    params=best_series["params"],
                    y_true=best_series["y_true"],
                    y_pred=best_series["y_pred"],
                    dates=best_series["dates"],  # <-- AQUÃ VA 'dates'
                    trade_mask=best_series.get("trade_mask"),
                    confirmation_reason=best_series.get("confirmation_reasons"),
                    extra_columns=self._flatten_series_details(best_series.get("signal_details")),
                )

                # Generar grÃ¡fico para la mejor combinaciÃ³n de este modelo
                self._plot_predictions_series(
                    dates=best_series["dates"],
                    y_true=best_series["y_true"],
                    y_pred=best_series["y_pred"],
                    model_name=model_name,
                    params=best_series["params"],
                    suffix="_best",
                )

                audit_df = self._save_backtest_trade_audit(
                    model_name=model_name,
                    params=best_series["params"],
                    dates=best_series["dates"],
                    y_true=best_series["y_true"],
                    y_pred=best_series["y_pred"],
                    signal_details=best_series.get("signal_details"),
                )
                self._save_backtest_monthly_stability(
                    model_name=model_name,
                    params=best_series["params"],
                    audit_df=audit_df,
                )
                self._plot_trade_audit(
                    audit_df=audit_df,
                    model_name=model_name,
                    params=best_series["params"],
                    suffix="_trade_audit",
                )
            # ==================== FIN CAMBIO IMPORTANTE =====================

            # Guardar reporte detallado para este modelo (como antes)
            if model_results:
                self._save_model_report(model_name, model_results)

        # Guardar resumen consolidado y config optimizada (como ya tenÃ­as)
        bundle_results: list[dict[str, Any]] = []
        bundle_artifacts: dict[str, dict[str, Any]] = {}
        if all_results and self._is_hybrid_mode():
            bundle_results, bundle_artifacts = self._evaluate_hybrid_bundle_candidates(
                best_artifacts_by_model=best_artifacts_by_model,
            )
            if bundle_results:
                self._save_model_report("HYBRID_BUNDLE", bundle_results)
                best_bundle_row = self._select_best_run(
                    pd.DataFrame(bundle_results),
                    model_name="HYBRID_BUNDLE",
                    log_prefix="  -> Bundle best ",
                )
                if best_bundle_row is not None:
                    bundle_label = str(best_bundle_row.get("model", ""))
                    best_bundle_artifact = bundle_artifacts.get(bundle_label)
                    if best_bundle_artifact:
                        self._save_backtest_series(
                            model_name=bundle_label,
                            params=best_bundle_artifact["params"],
                            y_true=best_bundle_artifact["y_true"],
                            y_pred=best_bundle_artifact["y_pred"],
                            dates=best_bundle_artifact["dates"],
                            trade_mask=best_bundle_artifact.get("trade_mask"),
                            confirmation_reason=best_bundle_artifact.get("confirmation_reasons"),
                            extra_columns=self._flatten_series_details(best_bundle_artifact.get("signal_details")),
                        )
                        self._plot_predictions_series(
                            dates=best_bundle_artifact["dates"],
                            y_true=best_bundle_artifact["y_true"],
                            y_pred=best_bundle_artifact["y_pred"],
                            model_name=bundle_label,
                            params=best_bundle_artifact["params"],
                            suffix="_best",
                        )
                        audit_df = self._save_backtest_trade_audit(
                            model_name=bundle_label,
                            params=best_bundle_artifact["params"],
                            dates=best_bundle_artifact["dates"],
                            y_true=best_bundle_artifact["y_true"],
                            y_pred=best_bundle_artifact["y_pred"],
                            signal_details=best_bundle_artifact.get("signal_details"),
                        )
                        self._save_backtest_monthly_stability(
                            model_name=bundle_label,
                            params=best_bundle_artifact["params"],
                            audit_df=audit_df,
                        )
                        self._plot_trade_audit(
                            audit_df=audit_df,
                            model_name=bundle_label,
                            params=best_bundle_artifact["params"],
                            suffix="_trade_audit",
                        )

        if all_results:
            summary_rows = list(all_results) + list(bundle_results)
            self._save_consolidated_summary(summary_rows)
            self._find_and_save_best_params(
                all_results,
                df_features,
                bundle_results=bundle_results,
            )


    def _run_test_mode(self) -> None:
        """
        Modo TEST / VALIDACIÃ“N:
        Usa los mejores parÃ¡metros (config_optimizado) y evalÃºa en un hold-out final.
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("MODO: TEST / VALIDACIÃ“N")
        self.logger.info("="*60 + "\n")

        # 1-3. Cargar, limpiar y features
        df = self._load_data()
        df_clean = self._clean_data(df)
        df_features = self._generate_features(df_clean)

        # 4. Determinar segmento de validaciÃ³n
        val_cfg = self.config.get("validation", {})
        mode = val_cfg.get("mode", "last_n")
        n = int(val_cfg.get("n", 500))

        target_col = self.config.get("backtest", {}).get("target", "ReturnFwd_1")

        df_processed = df_features.dropna(subset=[target_col]).bfill().ffill()
        if len(df_processed) <= n + 10:
            self.logger.error("No hay suficientes datos para una validaciÃ³n con last_n=%s", n)
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
            self.logger.error("No se encontrÃ³ un modelo con 'params' en la configuraciÃ³n. "
                            "Ejecuta primero el modo backtest para generar config_optimizado.")
            return

        model_name = best_model_config["name"]
        params = best_model_config.get("params", {})
        self.logger.info(f"Usando mejor modelo '{model_name}' para validaciÃ³n, params={params}")

        # 6. ValidaciÃ³n tipo walk-forward sobre df_test
        all_pred = []
        all_true = []
        trade_mask = []
        bt_rows = []
        close_prices = df_processed["Close"] if "Close" in df_processed.columns else None

        # Entrenamos una vez con df_train completo y vamos moviendo la ventana sobre df_test
        model_class_map = self._get_supported_model_class_map()
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
        # Truco: usamos train_and_predict iterativamente con X_test de tamaÃ±o 1
        for ts in X_test_full.index:
            # Ventana de entrenamiento = todo hasta ts-1
            mask_train = df_processed.index < ts
            X_tr = df_processed.loc[mask_train, features_cols]
            y_tr = df_processed.loc[mask_train, target_col]
            X_te = df_processed.loc[[ts], features_cols]
            y_te = df_processed.loc[[ts], target_col]

            if self._get_target_mode() == "barrier_event" and hasattr(model_instance, "train_and_predict_details"):
                raw_prediction = model_instance.train_and_predict_details(y_tr, X_tr, X_te)
            else:
                raw_prediction = model_instance.train_and_predict(y_tr, X_tr, X_te)

            pred_list, detail_rows = self._normalize_prediction_output(
                raw_prediction,
                expected_len=1,
            )
            if pred_list is None or len(pred_list) == 0:
                continue

            pred = float(pred_list[0])
            true_val = float(y_te.iloc[0])
            probability_detail = detail_rows[0] if detail_rows else None

            all_pred.append(pred)
            all_true.append(true_val)

            if self._get_target_mode() == "barrier_event" and probability_detail:
                signal_info = build_signal_from_probabilities(
                    prob_up=float(probability_detail.get("prob_up", 0.0) or 0.0),
                    prob_hold=float(probability_detail.get("prob_hold", 0.0) or 0.0),
                    prob_down=float(probability_detail.get("prob_down", 0.0) or 0.0),
                    barrier_pips=float(self._get_barrier_settings()["barrier_pips"]),
                    min_confidence=min_confidence if enable_confidence_filter else 0.0,
                    probability_threshold=float(self._get_barrier_settings()["probability_threshold"]),
                    probability_margin=float(self._get_barrier_settings()["probability_margin"]),
                    model_metrics={},
                )
            else:
                signal_info = build_signal_from_prediction(
                    pred_return=pred,
                    pip_size=pip_size,
                    min_pips_signal=min_pips_signal,
                    model_metrics={},
                    min_confidence=min_confidence if enable_confidence_filter else 0.0,
                    probability=None,
                    price_reference=self._get_price_reference_from_feature_row(
                        X_te.iloc[0] if not X_te.empty else None
                    ),
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
            self.logger.error("No se generaron predicciones en validaciÃ³n.")
            return

        # 7. MÃ©tricas de validaciÃ³n
        metrics = self._calculate_metrics(all_true, all_pred, trade_mask=trade_mask)
        self.logger.info(f"ðŸ“Š MÃ©tricas de VALIDACIÃ“N para {model_name}: {metrics}")

        # 8. Guardar Excel consolidado (detalle + mÃ©tricas)
        output_dir = Path(self.config.get("output", {}).get("dir", "outputs")) / "validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        xlsx_path = output_dir / "validation_consolidated.xlsx"

        df_bt = pd.DataFrame(bt_rows).set_index("timestamp")
        df_metrics = pd.DataFrame([metrics])

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df_bt.to_excel(writer, sheet_name="detail")
            df_metrics.to_excel(writer, sheet_name="metrics", index=False)

        self.logger.info(f"ðŸ’¾ Archivo de validaciÃ³n guardado en: {xlsx_path}")
        self.logger.info("\nâœ… MODO TEST / VALIDACIÃ“N COMPLETADO")
        
    def _find_and_save_best_params(
        self,
        all_results: list[dict],
        df_features: pd.DataFrame,
        bundle_results: list[dict] | None = None,
    ) -> None:
        """
        A partir de todas las combinaciones evaluadas en el backtest:
        - Identifica la mejor por modelo usando las mÃ©tricas de model_selection.
        - Construye un config_optimizado.yaml con esos mejores modelos.
        - (Opcional) Reentrena y guarda los modelos finales en outputs/models.
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("ðŸ† ENCONTRANDO MEJORES HIPERPARÃMETROS")
        self.logger.info("=" * 60)

        if not all_results:
            self.logger.warning("No hay resultados en all_results; nada que optimizar.")
            return

        # 1. Pasar resultados a DataFrame
        df = pd.DataFrame(all_results)

        # Columnas de mÃ©tricas que NO son hiperparÃ¡metros
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

        # 2. ConfiguraciÃ³n de cÃ³mo se escoge el "mejor" modelo
        selection_cfg = self._get_model_selection_settings()
        primary_metric = selection_cfg["primary_metric"]
        secondary_metric = selection_cfg["secondary_metric"]
        best_rows_for_global: list[dict[str, Any]] = []
        hybrid_mode = self._is_hybrid_mode()

        # 3. Por cada modelo (ARIMA, PROPHET, LSTM, RandomWalk, etc.) encontrar la mejor fila
        for model_name in df["model"].dropna().unique():
            model_df = df[df["model"] == model_name].copy()
            if model_df.empty:
                continue

            best_run = self._select_best_run(
                model_df,
                model_name=model_name,
                log_prefix="  -> SelecciÃ³n ",
            )
            if best_run is None:
                continue

            # HiperparÃ¡metros = todas las columnas excepto mÃ©tricas + 'model'
            param_cols = [c for c in model_df.columns if c not in metric_cols + ["model", "selection_role"]]
            raw_params = {k: best_run[k] for k in param_cols}

            clean_params = {k: to_native(v) for k, v in raw_params.items() if not is_nan(v)}
            selection_role = self._get_model_selection_role(
                model_name=model_name,
                row=best_run,
            )

            # Log informativo
            p_val = best_run[primary_metric] if primary_metric in best_run.index else None
            s_val = best_run[secondary_metric] if secondary_metric in best_run.index else None
            t_val = best_run["n_trades"] if "n_trades" in best_run.index else None

            self.logger.info(
                f"  -> Mejor para {model_name}: "
                f"{primary_metric}={to_native(p_val)} | {secondary_metric}={to_native(s_val)} | "
                f"n_trades={to_native(t_val)} | role={selection_role} | params={clean_params}"
            )

            if selection_role != "baseline":
                best_rows_for_global.append(best_run.to_dict())
            else:
                self.logger.info(
                    f"  -> {model_name} se mantiene como baseline y no compite por campeÃ³n global."
                )
            best_models.append(
                {
                    "name": model_name,
                    "enabled": True,
                    "params": clean_params,
                    "selection_role": selection_role,
                }
            )

        if not best_models:
            self.logger.warning("No se encontrÃ³ ningÃºn mejor modelo para guardar en config_optimizado.")
            return

        model_params_map = {str(model_cfg["name"]): dict(model_cfg.get("params", {})) for model_cfg in best_models}
        trading_cfg = self.config.get("trading", {}) or {}
        barrier_settings = self._get_barrier_settings()

        selection_cfg = self._get_model_selection_settings()
        global_champion_name = None
        publish_release = True
        decision_bundle: dict[str, Any] | None = None
        if hybrid_mode:
            bundle_df = pd.DataFrame(bundle_results or [])
            if bundle_df.empty:
                publish_release = False
                self.logger.warning("No se publicarÃ¡ release activa: no se generaron bundles hÃ­bridos evaluables.")
            else:
                eligible_bundle_df, applied_filters = self._filter_runs_by_selection_thresholds(bundle_df)
                champion_row = None
                if not eligible_bundle_df.empty:
                    champion_row = self._select_best_run(
                        eligible_bundle_df,
                        model_name="HYBRID_BUNDLE",
                        log_prefix="  -> CampeÃ³n bundle ",
                    )
                elif selection_cfg.get("publish_requires_candidate_thresholds", True):
                    publish_release = False
                    self.logger.warning(
                        "No se publicarÃ¡ release activa: ningÃºn bundle hÃ­brido cumple %s.",
                        " y ".join(applied_filters) if applied_filters else "los filtros mÃ­nimos",
                    )
                else:
                    champion_row = self._select_best_run(
                        bundle_df,
                        model_name="HYBRID_BUNDLE",
                        log_prefix="  -> CampeÃ³n bundle ",
                    )

                if champion_row is not None and "model" in champion_row.index:
                    global_champion_name = str(champion_row["model"])
                    primary_model_name = str(champion_row.get("primary_model", ""))
                    filter_model_name = str(champion_row.get("filter_model", ""))
                    decision_bundle = {
                        "mode": "hybrid_primary_plus_filter",
                        "name": global_champion_name,
                        "primary_model": {
                            "name": primary_model_name,
                            "params": model_params_map.get(primary_model_name, {}),
                        },
                        "filter_model": {
                            "name": filter_model_name,
                            "params": model_params_map.get(filter_model_name, {}),
                        },
                        "require_alignment": bool(self._get_prediction_stack_settings()["require_alignment"]),
                        "filter_gate_mode": str(self._get_prediction_stack_settings()["filter_gate_mode"]),
                        "support_probability_threshold": self._get_prediction_stack_settings()["support_probability_threshold"],
                        "support_probability_margin": self._get_prediction_stack_settings()["support_probability_margin"],
                        "support_score_min": self._get_prediction_stack_settings()["support_score_min"],
                        "contradiction_margin": self._get_prediction_stack_settings()["contradiction_margin"],
                        "rules": {
                            "min_pips_signal": float(
                                trading_cfg.get(
                                    "min_pips_signal",
                                    self.config.get("backtest", {}).get("threshold_pips", 0.0),
                                )
                            ),
                            "min_confidence": float(trading_cfg.get("min_confidence", 0.60) or 0.60),
                            "barrier_probability_threshold": float(barrier_settings["probability_threshold"]),
                            "barrier_probability_margin": float(barrier_settings["probability_margin"]),
                            "barrier_pips": float(barrier_settings["barrier_pips"]),
                            "barrier_horizon_bars": int(barrier_settings["horizon_bars"]),
                        },
                    }
                    self.logger.info(f"Bundle campeÃ³n global: {global_champion_name}")
        elif best_rows_for_global:
            global_candidates_df = pd.DataFrame(best_rows_for_global)
            eligible_global_candidates, applied_filters = self._filter_runs_by_selection_thresholds(global_candidates_df)

            champion_row = None
            if not eligible_global_candidates.empty:
                champion_row = self._select_best_run(
                    eligible_global_candidates,
                    log_prefix="  -> CampeÃ³n global ",
                )
            elif selection_cfg.get("publish_requires_candidate_thresholds", True):
                publish_release = False
                self.logger.warning(
                    "âš ï¸ No se publicarÃ¡ release activa: ningÃºn modelo candidato cumple %s.",
                    " y ".join(applied_filters) if applied_filters else "los filtros mÃ­nimos",
                )
            else:
                champion_row = self._select_best_run(
                    global_candidates_df,
                    log_prefix="  -> CampeÃ³n global ",
                )

            if champion_row is not None and "model" in champion_row.index:
                global_champion_name = str(champion_row["model"])
                self.logger.info(f"ðŸ† Modelo campeÃ³n global: {global_champion_name}")
        else:
            publish_release = False
            self.logger.warning("âš ï¸ No hay modelos candidatos elegibles para publicar una release activa.")

        self._global_champion = global_champion_name
        for model_cfg in best_models:
            model_cfg["is_best"] = bool(
                (not hybrid_mode)
                and global_champion_name
                and str(model_cfg.get("name")) == global_champion_name
            )

        release_id = self._ensure_backtest_run_label()

        # 4. Construir config optimizado: copiamos config actual y reemplazamos sÃ³lo la secciÃ³n de modelos
        optimized_config = dict(self.config)
        optimized_config["models"] = best_models
        if hybrid_mode and decision_bundle:
            optimized_config["decision_bundle"] = decision_bundle
        elif "decision_bundle" in optimized_config:
            optimized_config.pop("decision_bundle", None)
        optimized_config = self._inherit_release_operational_sections(
            optimized_config,
            profile_name=self._get_strategy_profile_name(),
        )

        base_config_path = Path(self.config_path)
        versioned_config_path = base_config_path.parent / f"config_optimizado_{release_id}.yaml"
        self._write_yaml_atomic(versioned_config_path, optimized_config)

        self.logger.info(f"\nðŸ’¾ ConfiguraciÃ³n optimizada versionada guardada en: {versioned_config_path}")

        if not publish_release or not global_champion_name:
            self.logger.warning(
                "Se omitirÃ¡ la publicaciÃ³n de release activa y el reentrenamiento final porque no hubo campeÃ³n candidato robusto."
            )
            return

        # 5. Reentrenar y guardar modelos finales (si tenemos features del Ãºltimo backtest)
        if self._df_features_last_backtest is None:
            self.logger.warning(
                "    -> self._df_features_last_backtest es None. "
                "No se reentrenan ni se guardan modelos en disco."
            )
            return

        models_dir = self._get_release_models_dir(release_id)

        model_class_map = self._get_supported_model_class_map()
        

        self.logger.info("\nðŸ§  Reentrenando y guardando modelos Ã³ptimos...")

        for m in best_models:
            name = m["name"]
            params = m.get("params", {})

            model_class = model_class_map.get(name)
            if model_class is None:
                self.logger.warning(f"    -> Modelo '{name}' no estÃ¡ soportado para guardado. Se omite.")
                continue

            model = model_class(params=params, logger=self.logger)
            model_name = f"{name.lower()}_best"
            target_col = self._get_model_target_column(model_name=name, model_cfg=m)
            feature_cols = self._get_model_feature_columns(self._df_features_last_backtest, target_col)
            df_proc = (
                self._df_features_last_backtest
                .dropna(subset=[target_col] + feature_cols)
                .bfill()
                .ffill()
            )
            if df_proc.empty:
                self.logger.warning(
                    "    -> No se pudo reentrenar %s: serie vacÃ­a tras limpiar target=%s.",
                    name,
                    target_col,
                )
                continue

            X_full = df_proc[feature_cols]
            y_full = df_proc[target_col]

            try:
                model.train_and_save(
                    y_train=y_full,
                    X_train=X_full,
                    model_name=model_name,
                    models_dir=models_dir,
                )
                self.logger.info(
                    f"    âœ… Modelo {name} entrenado y guardado en carpeta: {models_dir} "
                    f"(nombre base: {model_name})"
                )
            except NotImplementedError:
                self.logger.warning(
                    f"    âš ï¸ El modelo {name} no implementa train_and_save(...). "
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

        self.logger.info("\nâœ… Proceso de optimizaciÃ³n y guardado de modelos completado.")
        
    def _save_model_report(self, model_name: str, model_results: list[dict]) -> None:
        """Guarda el reporte detallado de un modelo en un archivo CSV."""
        if not model_results:
            return

        output_dir = self._get_backtest_output_dir()
        
        report_path = output_dir / f"report_{model_name}.csv"
        df_report = pd.DataFrame(model_results)
        
        # Ordenar usando criterios de selecciÃ³n (primario y secundario)
        ms = self.config.get("model_selection", {}) or {}
        primary = ms.get("primary_metric", "hit_rate")
        secondary = ms.get("secondary_metric", "rmse")
        primary_greater = bool(ms.get("primary_greater_is_better", True))
        secondary_greater = bool(ms.get("secondary_greater_is_better", False))

        # Asegurar numÃ©ricos para ordenar
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

        # Desempate: preferir mÃ¡s trades si existe
        if "n_trades" in df_report.columns:
            sort_cols.append("n_trades")
            ascending.append(False)

        if sort_cols:
            df_report = df_report.sort_values(by=sort_cols, ascending=ascending, na_position="last")

            
        df_report.to_csv(report_path, index=False)
        self.logger.info(f"    ðŸ’¾ Reporte para {model_name} guardado en: {report_path}")
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
        extra_columns: Optional[Dict[str, List[Any]]] = None,
    ) -> None:
        """
        Guarda la serie completa de backtest (y_true, y_pred, error y fechas opcionales)
        para poder graficar despuÃ©s los errores / predicciones.

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
        if extra_columns:
            for column_name, values in extra_columns.items():
                if values is not None and len(values) == len(y_true):
                    data[column_name] = values

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

        # ==================== CAMBIO IMPORTANTE AQUÃ ========================
        # ANTES usÃ¡bamos: self.paths["backtest_dir"]  -> pero self.paths NO existe
        # Usamos un directorio fijo dentro de 'outputs/backtest'
        backtest_dir = self._get_backtest_output_dir()

        file_name = f"{model_name}_{param_suffix}_series.csv"
        file_path = backtest_dir / file_name
        # ===================================================================

        # Guardar CSV
        df_series.to_csv(file_path, index=False)
        self.logger.info(f"      â†³ Serie completa guardada en: {file_path}")
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

        # Forzar numÃ©rico en mÃ©tricas clave
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

        def _merge_trade_audit_summary_fields(best_runs_df: pd.DataFrame) -> pd.DataFrame:
            if best_runs_df.empty:
                return best_runs_df

            merged_rows: list[dict[str, Any]] = []
            backtest_dir = self._get_backtest_output_dir()
            for _, row in best_runs_df.iterrows():
                row_dict = row.to_dict()
                params = {
                    key: value
                    for key, value in row_dict.items()
                    if key not in {
                        "model",
                        "selection_role",
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
                    }
                    and not str(key).startswith("_")
                    and pd.notna(value)
                }
                param_suffix = self._build_param_suffix(params)
                audit_summary_path = backtest_dir / f"{row_dict['model']}_{param_suffix}_trade_audit_summary.csv"
                if audit_summary_path.exists():
                    try:
                        audit_summary_df = pd.read_csv(audit_summary_path)
                        if not audit_summary_df.empty:
                            audit_summary = audit_summary_df.iloc[0].to_dict()
                            for audit_key, audit_value in audit_summary.items():
                                if audit_key in {"model"}:
                                    continue
                                row_dict[audit_key] = audit_value
                    except Exception as exc:
                        self.logger.warning(
                            f"No se pudieron fusionar metricas de auditoria desde {audit_summary_path}: {exc}"
                        )
                merged_rows.append(row_dict)
            return pd.DataFrame(merged_rows)

        best_runs = _merge_trade_audit_summary_fields(best_runs)
        selection_cfg = self._get_model_selection_settings()
        candidate_best_runs = best_runs[
            best_runs.apply(
                lambda row: self._is_model_selection_candidate(row=row),
                axis=1,
            )
        ].copy()
        eligible_candidate_runs, applied_filters = self._filter_runs_by_selection_thresholds(candidate_best_runs)
        champion_row = None
        champion_model = None
        if not eligible_candidate_runs.empty:
            champion_row = self._select_best_run(eligible_candidate_runs, log_prefix="Resumen global ")
        elif candidate_best_runs.empty:
            self.logger.warning("Resumen global: no hay modelos candidatos habilitados para campeÃ³n.")
        elif selection_cfg.get("publish_requires_candidate_thresholds", True):
            self.logger.warning(
                "Resumen global: ningÃºn candidato cumple %s. No se marcarÃ¡ campeÃ³n en el resumen.",
                " y ".join(applied_filters) if applied_filters else "los filtros mÃ­nimos",
            )
        else:
            champion_row = self._select_best_run(candidate_best_runs, log_prefix="Resumen global ")
        champion_model = None
        if champion_row is not None and "model" in champion_row.index:
            champion_model = str(champion_row["model"])
        if champion_model and "model" in best_runs.columns:
            best_runs["is_best"] = best_runs["model"].astype(str) == champion_model
        elif "is_best" not in best_runs.columns:
            best_runs["is_best"] = False

        # Guardar outputs
        csv_path = output_dir / "summary_best_runs.csv"
        best_runs.to_csv(csv_path, index=False)
        self.logger.info(f"ðŸ“„ Resumen consolidado guardado en: {csv_path}")
        csv_archive_path = self._archive_backtest_artifact(csv_path)
        self._latest_backtest_summary_paths["csv"] = csv_archive_path or csv_path

        xlsx_path = output_dir / "summary_best_runs.xlsx"
        try:
            best_runs.to_excel(xlsx_path, index=False)
            self.logger.info(f"ðŸ“„ Resumen consolidado guardado en: {xlsx_path}")
            xlsx_archive_path = self._archive_backtest_artifact(xlsx_path)
            self._latest_backtest_summary_paths["xlsx"] = xlsx_archive_path or xlsx_path
        except Exception as e:
            self.logger.warning(f"No se pudo guardar XLSX del resumen: {e}")
            self._latest_backtest_summary_paths["xlsx"] = None
            
    def _run_walk_forward_for_params(
        self,
        df_features: pd.DataFrame,
        model_name: str,
        params: dict,
        model_cfg: dict[str, Any] | None = None,
    ) -> tuple[list, list, list, list, list, list]:
        """Ejecuta un backtest Walk-Forward para una configuraciÃ³n de modelo especÃ­fica."""
        backtest_config = self.config.get("backtest", {})
        initial_train_size = int(backtest_config.get("initial_train", 800))
        step = int(backtest_config.get("step", 20))
        horizon = int(backtest_config.get("horizon", 1))
        target_col = self._get_model_target_column(model_name=model_name, model_cfg=model_cfg)

        # 1) Definir features (basado en df_features)
        features_cols = self._get_model_feature_columns(df_features, target_col)

        # 2) Eliminar filas con NaNs en target + features (robusto para rolling indicators)
        df_processed = df_features.dropna(subset=[target_col] + features_cols).copy()

        if df_processed.empty:
            self.logger.warning("    -> df_processed quedÃ³ vacÃ­o tras dropna de target+features.")
            return [], [], [], [], [], []

        y = df_processed[target_col]
        X = df_processed[features_cols]

        if initial_train_size >= len(X):
            self.logger.warning(
                f"    -> No hay suficientes datos para el backtest con "
                f"initial_train_size={initial_train_size}. "
                f"Datos disponibles despuÃ©s de limpiar NaNs: {len(X)}. Saltando combinaciÃ³n."
            )
            return [], [], [], [], [], []

        # 3) Resolver timestamps: columna Date si existe, si no Ã­ndice
        if "Date" in df_processed.columns:
            ts_all = pd.to_datetime(df_processed["Date"], errors="coerce")
        else:
            ts_all = pd.to_datetime(df_processed.index, errors="coerce")

        all_predictions: list = []
        all_true_values: list = []
        all_timestamps: list = []
        all_feature_rows: list = []
        all_price_references: list[float | None] = []
        all_prediction_details: list[dict[str, Any] | None] = []

        for i in range(initial_train_size, len(X) - horizon + 1, step):
            train_end = i
            test_end = i + horizon

            X_train, X_test = X.iloc[:train_end], X.iloc[train_end:test_end]
            y_train, y_test = y.iloc[:train_end], y.iloc[train_end:test_end]

            if len(X_test) == 0:
                continue

            # Log de diagnÃ³stico (igual que tenÃ­as)
            if self.logger.isEnabledFor(20):  # INFO
                nan_in_train = X_train.isnull().sum().sum()
                self.logger.info(
                    f"    -> Ventana {i-initial_train_size}: "
                    f"X_train shape={X_train.shape}, "
                    f"y_train len={len(y_train)}, "
                    f"NaNs en X_train={nan_in_train}"
                )

            prediction = self._train_and_predict(
                model_name,
                params,
                X_train,
                y_train,
                X_test,
                model_cfg=model_cfg,
            )
            pred_list, detail_rows = self._normalize_prediction_output(
                prediction,
                expected_len=len(y_test),
            )

            # Normalizar salida a lista con misma longitud que y_test
            if True:
                pred_list = pred_list
            elif isinstance(prediction, (list, tuple, np.ndarray)):
                pred_list = list(prediction)
                if len(pred_list) != len(y_test):
                    # si devuelve algo raro, usamos el Ãºltimo valor como constante
                    last = pred_list[-1] if len(pred_list) else np.nan
                    pred_list = [last] * len(y_test)
            else:
                pred_list = [float(prediction)] * len(y_test)

            for pred_value, detail_row, (_, x_row), true_value, y_index in zip(
                pred_list,
                detail_rows,
                X_test.iterrows(),
                y_test.values.tolist(),
                y_test.index.tolist(),
            ):
                all_predictions.append(pred_value)
                all_true_values.append(true_value)
                all_timestamps.append(pd.to_datetime(y_index, errors="coerce"))
                all_feature_rows.append(x_row.copy())
                all_price_references.append(self._get_price_reference_from_feature_row(x_row))
                all_prediction_details.append(detail_row)

        return (
            all_predictions,
            all_true_values,
            all_timestamps,
            all_feature_rows,
            all_price_references,
            all_prediction_details,
        )

    def _train_and_predict(
        self,
        model_name: str,
        params: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        model_cfg: dict[str, Any] | None = None,
    ) -> list | dict | None:
        """Punto central para entrenar y predecir con un modelo especÃ­fico."""
        
        model_class = self._get_supported_model_class_map().get(model_name)
        
        if not model_class:
            self.logger.warning(f"Modelo '{model_name}' no reconocido. Saltando.")
            return None
        
        try:
            self.logger.debug(f"Instanciando modelo {model_name} con params: {params}")
            model_instance = model_class(params=params, logger=self.logger)
            
            use_probability_details = (
                self._get_target_mode() == "barrier_event"
                or (
                    self._is_hybrid_mode()
                    and self._get_model_stack_role(model_name=model_name, model_cfg=model_cfg) == "filter"
                )
            )
            if use_probability_details and hasattr(model_instance, "train_and_predict_details"):
                return model_instance.train_and_predict_details(y_train, X_train, X_test)

            return model_instance.train_and_predict(y_train, X_train, X_test)

        except Exception as e:
            self.logger.error(f"Error al ejecutar {model_name}: {e}")
            return None

    def _calculate_metrics(self, y_true: list, y_pred: list, trade_mask: list[bool] | None = None) -> dict:
        """
        Calcula un conjunto de mÃ©tricas de evaluaciÃ³n.

        - Calcula todas las mÃ©tricas disponibles en utils.metrics_v2.calculate_all_metrics.
        - Aplica un umbral de pips (backtest.threshold_pips) para las mÃ©tricas de TRADING.
        - Filtra las mÃ©tricas a las listadas en config['backtest']['metrics'].
        """
        if not y_true or not y_pred:
            self.logger.warning("Listas de valores vacÃ­as para calcular mÃ©tricas.")
            # Devolvemos tambiÃ©n contadores en 0 para que las columnas existan en los CSV
            return {
                "rmse": np.nan,
                "mae": np.nan,
                "hit_rate": np.nan,
                "n_test_points": 0,
                "n_trades": 0,
            }

        bt_cfg = self.config.get("backtest", {})

        # ParÃ¡metros opcionales para mÃ©tricas de trading
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
            threshold_pips=0.0 if trade_mask is not None else threshold_pips,
            active_mask_override=trade_mask,
        )

        # Lista de mÃ©tricas a usar segÃºn la configuraciÃ³n
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
        Genera los grÃ¡ficos de backtest para el mejor run de un modelo:
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

        # Ãndices como Index de pandas
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
            self.logger.warning(f"No se pudo generar grÃ¡fico de entradas para {model_name}: {e}")

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
        """Genera y guarda un grÃ¡fico (en retornos o en precios rebajados) para un modelo."""
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

        plot_scale = self.config.get("backtest", {}).get("plot_scale", "returns")
        price_col = self.config.get("eda", {}).get("price_col", "Close")
        base_close_series = None

        if plot_scale in {"price", "price_target"}:
            if hasattr(self, "df_clean") and price_col in self.df_clean.columns:
                try:
                    base_close_series = (
                        pd.Series(self.df_clean[price_col], index=pd.to_datetime(self.df_clean.index))
                        .reindex(df_plot.index)
                        .astype(float)
                        .ffill()
                        .bfill()
                    )
                    if base_close_series.isna().all():
                        base_close_series = None
                except Exception as e:
                    self.logger.warning(
                        "No se pudo alinear la serie base (%s). Se usaran retornos. Error: %s",
                        price_col,
                        e,
                    )
                    base_close_series = None

            if base_close_series is not None:
                series_real = base_close_series * (1.0 + df_plot["y_true"].astype(float))
                series_pred = base_close_series * (1.0 + df_plot["y_pred"].astype(float))
                ylabel = f"Precio objetivo implicito ({price_col})"
            else:
                plot_scale = "returns"
                series_real = df_plot["y_true"]
                series_pred = df_plot["y_pred"]
                ylabel = self.config.get("backtest", {}).get("target", "ReturnFwd_1")
        elif plot_scale == "rebased_price":
            price_col = self.config.get("eda", {}).get("price_col", "Close")
            base_price = 1.0
            if hasattr(self, "df_clean") and price_col in self.df_clean.columns:
                try:
                    base_price = float(self.df_clean.loc[df_plot.index[0], price_col])
                except Exception as e:
                    self.logger.warning(
                        "No se pudo alinear el precio base (%s). Usando 1.0 como indice. Error: %s",
                        price_col,
                        e,
                    )
            series_real = (1 + df_plot["y_true"]).cumprod() * base_price
            series_pred = (1 + df_plot["y_pred"]).cumprod() * base_price
            ylabel = f"Precio rebasado ({price_col})"
        else:
            series_real = df_plot["y_true"]
            series_pred = df_plot["y_pred"]
            ylabel = self.config.get("backtest", {}).get("target", "ReturnFwd_1")

        fig, ax = plt.subplots(figsize=(12, 6))
        if base_close_series is not None and plot_scale in {"price", "price_target"}:
            ax.plot(df_plot.index, base_close_series, label="Close actual", alpha=0.45, linestyle="--", color="gray")
        ax.plot(df_plot.index, series_real, label="Real", alpha=0.8)
        ax.plot(df_plot.index, series_pred, label="Predicho", alpha=0.8)

        escala_txt = {
            "price": "precio_objetivo",
            "price_target": "precio_objetivo",
            "rebased_price": "precio_rebasado",
        }.get(plot_scale, "retornos")
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

        self.logger.info(f"ðŸ“Š GrÃ¡fico de predicciones guardado en: {plot_path}")
        self._archive_backtest_artifact(plot_path)



    def _validate_model_on_test(self, model_name: str, params: dict, df_train: pd.DataFrame, y_test: pd.Series, X_test: pd.DataFrame):
        """Entrena un modelo con datos de train y lo valida contra test."""
        target_col = self.config.get("backtest", {}).get("target", "ReturnFwd_1")
        
        # Preparar datos de entrenamiento completos
        y_train = df_train[target_col]
        feature_cols = self._get_model_feature_columns(df_train, target_col)
        X_train = df_train[feature_cols]

        # Entrenar y predecir en el conjunto de test
        # Para una validaciÃ³n real, se cargarÃ­a el modelo guardado.
        # AquÃ­, re-entrenamos y predecimos para demostrar el flujo.
        raw_predictions = self._train_and_predict(model_name, params, X_train, y_train, X_test)
        predictions, _ = self._normalize_prediction_output(raw_predictions, expected_len=len(y_test))

        if predictions is None or len(predictions) != len(y_test):
            self.logger.error(f"No se pudieron generar predicciones para {model_name} en el set de validaciÃ³n.")
            return

        # Calcular y mostrar mÃ©tricas finales
        final_metrics = self._calculate_metrics(y_test.tolist(), predictions)
        self.logger.info(f"  -> MÃ©tricas finales para {model_name} en Test Set:")
        for metric, value in final_metrics.items():
            self.logger.info(f"    - {metric.upper()}: {value}")


    def _run_production_mode(self) -> None:
        """
        Modo ProducciÃ³n:
        - Carga datos recientes desde MT5
        - Genera features
        - Carga desde disco los modelos ganadores segÃºn la config (config_optimizado.yaml)
        - Genera una predicciÃ³n de retorno por modelo
        - Traduce cada predicciÃ³n a seÃ±al BUY/SELL/HOLD (aplicando un umbral en pips)
        - Calcula niveles de entrada / SL / TP y tamaÃ±o de posiciÃ³n con base en la secciÃ³n 'risk'
        - Guarda seÃ±ales y, opcionalmente, ejecuta la orden real en MT5
        - Actualiza el reporte de ciclo de vida de trades cerrados
        """
        from utils.risk_utils import (
            calculate_position_size_for_risk_amount,
            compute_entry_sl_tp,
            estimate_position_risk_amount,
        )

        self.logger.info("\n" + "=" * 60)
        self.logger.info("MODO: PRODUCCIÃ“N")
        self.logger.info("=" * 60 + "\n")
        if self._pause_if_market_closed(mode="production"):
            self.logger.info("MODO PRODUCCIÃ“N EN PAUSA.\n")
            return

        # 1) Cargar / limpiar / generar features
        self.logger.info("ðŸ“¥ Cargando datos para producciÃ³n...")
        df_raw = self._load_data()
        df_clean = self._clean_data(df_raw)
        df_features = self._generate_features(df_clean)

        target_col = self.config.get("backtest", {}).get("target", "ReturnFwd_1")
        target_mode = self._get_target_mode()
        barrier_settings = self._get_barrier_settings()

        # Quitamos filas sin target ni features
        feature_cols = self._get_model_feature_columns(df_features, target_col)
        df_infer = df_features.dropna(subset=feature_cols).copy()
        df_context = df_features.dropna(subset=[target_col] + feature_cols).copy()

        if df_infer.empty:
            self.logger.error("No hay datos suficientes con features completas para producciÃ³n.")
            return

        X_live = df_infer[feature_cols]
        X_context = df_context[feature_cols] if not df_context.empty else pd.DataFrame(columns=feature_cols)
        y_context = df_context[target_col] if not df_context.empty else pd.Series(dtype=float, name=target_col)

        # Ãšltimo valor de ATR (si existe) para gestiÃ³n de riesgo basada en volatilidad
        atr_col = "ATR_14"
        if atr_col in df_infer.columns:
            atr_value = float(df_infer[atr_col].iloc[-1])
        else:
            atr_value = None

        # 2) Modelos habilitados en la config
        models_cfg = self.config.get("models", [])
        enabled_models_cfg = [m for m in models_cfg if m.get("enabled", True)]
        live_cfg = self._get_live_trading_settings()
        hybrid_bundle_config = self._get_best_decision_bundle_from_config() if self._is_hybrid_mode() else None

        if not enabled_models_cfg:
            self.logger.error(
                "No hay modelos habilitados en la configuraciÃ³n. Revisa la secciÃ³n 'models' del YAML."
            )
            return

        # Determinar el modelo campeÃ³n global (usa la lÃ³gica existente)
        best_model_config = self._get_best_model_from_config()
        best_model_name = None
        if hybrid_bundle_config:
            best_model_name = str(hybrid_bundle_config.get("name", "")).upper()
            self.logger.info(
                f"ðŸ† Bundle campeÃ³n global segÃºn backtest: {best_model_name}"
            )
        elif best_model_config:
            best_model_name = str(best_model_config.get("name", "")).upper()
            self.logger.info(
                f"ðŸ† Modelo campeÃ³n global segÃºn backtest: {best_model_name}"
            )
        else:
            self.logger.warning(
                "No se pudo determinar un modelo campeÃ³n global con _get_best_model_from_config()."
            )

        if live_cfg["execute_best_model_only"] and hybrid_bundle_config:
            def _resolve_bundle_model_cfg(bundle_model: dict[str, Any]) -> dict[str, Any]:
                bundle_model_name = str(bundle_model.get("name", ""))
                bundle_model_params = dict(bundle_model.get("params", {}) or {})
                for cfg in models_cfg:
                    if str(cfg.get("name", "")).upper() == bundle_model_name.upper():
                        merged_cfg = dict(cfg)
                        merged_cfg["params"] = bundle_model_params or dict(cfg.get("params", {}) or {})
                        merged_cfg["enabled"] = True
                        return merged_cfg
                return {
                    "name": bundle_model_name,
                    "enabled": True,
                    "params": bundle_model_params,
                }

            enabled_models_cfg = [
                _resolve_bundle_model_cfg(hybrid_bundle_config.get("primary_model", {})),
                _resolve_bundle_model_cfg(hybrid_bundle_config.get("filter_model", {})),
            ]
            self.logger.info("ðŸŽ¯ ProducciÃ³n configurada para operar Ãºnicamente con el bundle campeÃ³n.")
        elif live_cfg["execute_best_model_only"] and best_model_config:
            enabled_models_cfg = [best_model_config]
            self.logger.info("ðŸŽ¯ ProducciÃ³n configurada para operar Ãºnicamente con el modelo campeÃ³n.")

        active_release = self._resolve_active_release_assets()
        active_release_id = active_release.get("release_id")
        strategy_profile_name = self._get_strategy_profile_name() or active_release.get("strategy_profile") or "default"
        if active_release_id:
            self.logger.info(
                "ðŸ“¦ Release activa de producciÃ³n%s: %s (activada %s)",
                f" [{strategy_profile_name}]" if strategy_profile_name else "",
                active_release_id,
                active_release.get("activated_at"),
            )

        # 3) MÃ©tricas de backtest (summary_best_runs.csv)
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
                    f"No se pudieron cargar mÃ©tricas desde {summary_path}: {e}"
                )
        else:
            self.logger.warning(
                f"No se encontrÃ³ {summary_path}; no se agregarÃ¡n mÃ©tricas de backtest al CSV de producciÃ³n."
            )

        # 4) Mapa nombre -> clase de modelo
        model_map = {str(name).upper(): cls for name, cls in self._get_supported_model_class_map().items()}

        # Directorio donde estÃ¡n los modelos guardados
        models_dir = Path(active_release.get("models_dir"))
        models_dir.mkdir(parents=True, exist_ok=True)

        # Datos comunes para todas las filas de salida
        last_row = df_infer.iloc[-1]
        price_now = float(last_row["Close"]) if "Close" in last_row else float("nan")

        # Info del sÃ­mbolo desde config + MT5
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
                    # volÃºmenes mÃ­nimos / step
                    min_lot = float(info.get("volume_min") or min_lot)
                    lot_step = float(info.get("volume_step") or lot_step)
                    stops_level_points = int(info.get("trade_stops_level") or 0)
                    freeze_level_points = int(info.get("trade_freeze_level") or 0)
        except Exception as e:
            self.logger.warning(f"No se pudo obtener info detallada del sÃ­mbolo desde MT5: {e}")

        # Fallbacks razonables para FX
        if pip_size <= 0.0:
            if point is not None and point > 0:
                pip_size = point
            else:
                pip_size = 0.0001
        if point is None or point <= 0:
            point = pip_size
        if contract_size is None or contract_size <= 0:
            contract_size = 100000.0  # tÃ­pico FX 1 lote

        # Info de cuenta para tamaÃ±o de posiciÃ³n
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

        # Ãšltimo fallback si no hay balance
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

        # --- ParÃ¡metros de trading (umbral de pips para seÃ±al) ---
        trading_cfg = self.config.get("trading", {}) or {}
        min_pips_signal = float(
            trading_cfg.get(
                "min_pips_signal",
                self.config.get("backtest", {}).get("threshold_pips", 0.0),
            )
        )

        rows = []
        planned_additional_risk_amount = 0.0
        hybrid_live_outputs: dict[str, dict[str, Any]] = {}

        self.logger.info("ðŸ”Ž Generando seÃ±ales de producciÃ³n para TODOS los modelos habilitados...\n")

        for m_cfg in enabled_models_cfg:
            model_name = str(m_cfg.get("name", "UNKNOWN"))
            params = m_cfg.get("params", {})
            model_name_upper = model_name.upper()

            self.logger.info(f"âž¡ Procesando modelo: {model_name} | params={params}")

            model_class = model_map.get(model_name_upper)
            if model_class is None:
                self.logger.error(f"  âœ— No hay clase asociada al modelo '{model_name}'. Se omite.")
                continue

            model_instance = model_class(params=params, logger=self.logger)

            # ConvenciÃ³n: LSTM -> .keras, resto -> .pkl
            file_prefix = f"{model_name.lower()}_best"
            if model_name_upper == "LSTM":
                model_path = models_dir / f"{file_prefix}.keras"
            else:
                model_path = models_dir / f"{file_prefix}.pkl"

            self.logger.info(f"  ðŸ’¾ Intentando cargar el modelo desde: {model_path}")

            if not hasattr(model_instance, "load_model") or not hasattr(model_instance, "predict_loaded"):
                self.logger.error(
                    f"  âœ— El modelo {model_name} no implementa 'load_model' o 'predict_loaded'. Se omite."
                )
                continue

            if not model_path.exists():
                self.logger.error(
                    f"  âœ— El archivo de modelo {model_path} no existe. Se omite."
                )
                continue

            # Cargar modelo
            try:
                model_instance.load_model(model_path)
            except Exception as e:
                self.logger.error(
                    f"  âœ— No se pudo cargar el modelo {model_name} desde disco: {e}"
                )
                continue

            # Predecir
            try:
                prediction_details = None
                is_hybrid_filter_model = (
                    self._is_hybrid_mode()
                    and self._get_model_stack_role(model_name=model_name, model_cfg=m_cfg) == "filter"
                )
                if (target_mode == "barrier_event" or is_hybrid_filter_model) and hasattr(model_instance, "predict_loaded_details"):
                    prediction = model_instance.predict_loaded_details(X_live)
                    prediction_details = prediction
                elif hasattr(model_instance, "predict_loaded_with_context"):
                    prediction = model_instance.predict_loaded_with_context(
                        X_all=X_context,
                        y_all=y_context,
                        X_live=X_live,
                    )
                else:
                    prediction = model_instance.predict_loaded(X_live)
            except Exception as e:
                self.logger.error(
                    f"  âœ— Error al predecir con el modelo cargado {model_name}: {e}"
                )
                continue

            pred_values, normalized_prediction_details = self._normalize_prediction_output(
                prediction,
                expected_len=1,
            )

            if pred_values is None or len(pred_values) == 0:
                self.logger.error(
                    f"  âœ— El modelo {model_name} no devolviÃ³ ninguna predicciÃ³n. Se omite."
                )
                continue

            # Tomamos la Ãºltima predicciÃ³n como "prÃ³ximo" retorno
            pred_return = float(pred_values[-1])
            probability_detail = normalized_prediction_details[-1] if normalized_prediction_details else None

            # MÃ©tricas histÃ³ricas del modelo para score de confianza
            m_metrics = metrics_by_model.get(model_name_upper, {})

            enable_confidence_filter = bool(trading_cfg.get("enable_confidence_filter", False))
            min_confidence = float(trading_cfg.get("min_confidence", 0.60))
            prob_up = float(probability_detail.get("prob_up", np.nan)) if probability_detail else float("nan")
            prob_hold = float(probability_detail.get("prob_hold", np.nan)) if probability_detail else float("nan")
            prob_down = float(probability_detail.get("prob_down", np.nan)) if probability_detail else float("nan")

            if live_cfg["execute_best_model_only"] and hybrid_bundle_config:
                hybrid_live_outputs[model_name_upper] = {
                    "model_name": model_name,
                    "params": params,
                    "pred_return": pred_return,
                    "probability_detail": probability_detail,
                    "metrics": m_metrics,
                    "prob_up": prob_up,
                    "prob_hold": prob_hold,
                    "prob_down": prob_down,
                }
                self.logger.info(
                    "  ðŸ“ˆ Modelo %s listo para bundle hÃ­brido -> retorno=%0.6f, prob_up=%0.3f, prob_hold=%0.3f, prob_down=%0.3f",
                    model_name,
                    pred_return,
                    prob_up if not np.isnan(prob_up) else float("nan"),
                    prob_hold if not np.isnan(prob_hold) else float("nan"),
                    prob_down if not np.isnan(prob_down) else float("nan"),
                )
                continue

            if target_mode == "barrier_event" and probability_detail:
                signal_info = build_signal_from_probabilities(
                    prob_up=float(probability_detail.get("prob_up", 0.0) or 0.0),
                    prob_hold=float(probability_detail.get("prob_hold", 0.0) or 0.0),
                    prob_down=float(probability_detail.get("prob_down", 0.0) or 0.0),
                    barrier_pips=float(barrier_settings["barrier_pips"]),
                    min_confidence=min_confidence if enable_confidence_filter else 0.0,
                    probability_threshold=float(barrier_settings["probability_threshold"]),
                    probability_margin=float(barrier_settings["probability_margin"]),
                    model_metrics=m_metrics,
                )
            else:
                signal_info = build_signal_from_prediction(
                    pred_return=pred_return,
                    pip_size=pip_size,
                    min_pips_signal=min_pips_signal,
                    model_metrics=m_metrics if 'm_metrics' in locals() else {},
                    min_confidence=min_confidence if enable_confidence_filter else 0.0,
                    probability=None,
                    price_reference=price_now if not np.isnan(price_now) else None,
                )

            pips = float(signal_info.get("predicted_pips", np.nan))
            if not np.isnan(price_now):
                delta_price = pips * pip_size
                price_target = price_now + delta_price
            else:
                price_target = float("nan")
                delta_price = float("nan")
            signal = str(signal_info["signal"])
            confidence = float(signal_info["confidence"])
            touch_probability = float(signal_info.get("touch_probability", np.nan))
            confirmation = self._evaluate_signal_confirmation(
                signal=signal,
                feature_row=last_row,
            )
            if signal in {"BUY", "SELL"} and not confirmation.get("passed", True):
                self.logger.info(
                    "  -> SeÃ±al %s bloqueada por confirmaciÃ³n opcional: %s",
                    signal,
                    confirmation.get("reason"),
                )
                signal = "HOLD"

            # --- GestiÃ³n de riesgo: niveles planificados y niveles reales de mercado ---
            entry_price = float("nan")
            sl_price = float("nan")
            tp_price = float("nan")
            sl_pips = float("nan")
            tp_pips = float("nan")
            signal_target_tp_pips = float("nan")
            signal_target_sl_pips = float("nan")
            signal_target_tp_price = float("nan")
            signal_target_sl_price = float("nan")
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

            signal_target_levels = self._build_signal_target_levels(
                signal=signal,
                entry_reference=price_now if not np.isnan(price_now) else None,
                pip_size=pip_size,
                target_pips=signal_info.get("signal_target_pips"),
            )
            signal_target_tp_pips = signal_target_levels["signal_target_tp_pips"]
            signal_target_sl_pips = signal_target_levels["signal_target_sl_pips"]
            signal_target_tp_price = signal_target_levels["signal_target_tp_price"]
            signal_target_sl_price = signal_target_levels["signal_target_sl_price"]

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

                # TamaÃ±o de posiciÃ³n coherente con la ejecuciÃ³n real de mercado.
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

            entry_management_plan = self._compute_entry_management_plan(
                signal=signal,
                total_volume_lots=volume_lots,
                live_entry_price=live_entry_price,
                live_sl_price=live_sl_price,
                live_tp_price=live_tp_price,
                digits=digits,
                min_lot=min_lot,
                lot_step=lot_step,
                timeframe=str(self.config.get("data", {}).get("timeframe", "M5")),
                signal_time=df_infer.index[-1],
            )

            # MÃ©tricas de backtest (si existen)
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
                f"  ðŸ“ˆ Modelo {model_name} -> retorno={pred_return:.6f}, "
                f"pips={pips:.2f}, signal={signal}, confidence={confidence:.3f}, "
                f"prob_up={prob_up if not np.isnan(prob_up) else float('nan'):.3f}, "
                f"prob_hold={prob_hold if not np.isnan(prob_hold) else float('nan'):.3f}, "
                f"prob_down={prob_down if not np.isnan(prob_down) else float('nan'):.3f}, "
                f"touch_prob={touch_probability if not np.isnan(touch_probability) else float('nan'):.3f}, "
                f"confirm={confirmation.get('reason')}, "
                f"signal_tp={signal_target_tp_price}, signal_sl={signal_target_sl_price}, "
                f"entry_plan={entry_price}, SL_plan={sl_price}, TP_plan={tp_price}, "
                f"entry_live={live_entry_price}, SL_live={live_sl_price}, TP_live={live_tp_price}, "
                f"lots={volume_lots:.2f}, balance={balance}, risk={risk_amount:.2f}, "
                f"entry_mgmt={entry_management_plan['entry_management_comment']}, "
                f"market_lots={entry_management_plan['initial_market_volume_lots']:.2f}, "
                f"pending_lots={entry_management_plan['pending_order_volume_lots']:.2f}, "
                f"pending_price={entry_management_plan['pending_order_price']}, "
                f"pending_sl={entry_management_plan['pending_order_sl_price']}, "
                f"pending_tp={entry_management_plan['pending_order_tp_price']}, "
                f"sl_pips_live={live_sl_pips:.2f}, usd_per_pip_lot={risk_per_pip_per_lot:.2f}, "
                f"risk_per_lot_stop={risk_per_lot_at_stop:.2f}, open_risk={open_risk_amount:.2f}, "
                f"remaining_budget={remaining_risk_budget_before_trade:.2f}, "
                f"projected_open_risk={projected_total_open_risk_after_trade:.2f}"
            )

            row = {
                "timestamp": df_infer.index[-1],
                "release_id": active_release_id,
                "strategy_profile": strategy_profile_name,
                "magic_number": live_cfg["magic_number"],
                "order_comment_prefix": live_cfg["order_comment_prefix"],
                "symbol": symbol,
                "timeframe": self.config.get("data", {}).get("timeframe", "UNKNOWN"),
                "model": model_name,
                "target_mode": target_mode,
                "pred_return": pred_return,
                "signal": signal,
                "confidence": confidence,
                "prob_up": prob_up,
                "prob_hold": prob_hold,
                "prob_down": prob_down,
                "touch_probability": touch_probability,
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
                "signal_candle_open": float(last_row["Open"]) if "Open" in last_row else np.nan,
                "signal_candle_high": float(last_row["High"]) if "High" in last_row else np.nan,
                "signal_candle_low": float(last_row["Low"]) if "Low" in last_row else np.nan,
                "signal_candle_close": float(last_row["Close"]) if "Close" in last_row else np.nan,
                "price_target": price_target,
                "delta_price": delta_price,
                "pips": pips,
                "expected_move_pips": pips,
                "target_barrier_pips": float(barrier_settings["barrier_pips"]) if target_mode == "barrier_event" else float("nan"),
                "target_horizon_bars": int(barrier_settings["horizon_bars"]) if target_mode == "barrier_event" else np.nan,
                "signal_target_tp_price": signal_target_tp_price,
                "signal_target_sl_price": signal_target_sl_price,
                "signal_target_tp_pips": signal_target_tp_pips,
                "signal_target_sl_pips": signal_target_sl_pips,
                # GestiÃ³n de riesgo
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
                "entry_management_mode": entry_management_plan["entry_management_mode"],
                "entry_management_split_active": entry_management_plan["entry_management_split_active"],
                "entry_management_initial_market_fraction": entry_management_plan["entry_management_initial_market_fraction"],
                "entry_management_pending_fraction": entry_management_plan["entry_management_pending_fraction"],
                "entry_management_retrace_fraction_of_stop": entry_management_plan["entry_management_retrace_fraction_of_stop"],
                "entry_management_total_volume_lots": entry_management_plan["entry_management_total_volume_lots"],
                "initial_market_volume_lots": entry_management_plan["initial_market_volume_lots"],
                "pending_order_volume_lots": entry_management_plan["pending_order_volume_lots"],
                "pending_order_price": entry_management_plan["pending_order_price"],
                "pending_order_type": entry_management_plan["pending_order_type"],
                "pending_order_sl_price": entry_management_plan["pending_order_sl_price"],
                "pending_order_tp_price": entry_management_plan["pending_order_tp_price"],
                "pending_order_expiry_time": entry_management_plan["pending_order_expiry_time"],
                "entry_management_comment": entry_management_plan["entry_management_comment"],
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
                # MÃ©tricas de backtest
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
            row.update(self._get_live_feature_snapshot(last_row))

            rows.append(row)

        if live_cfg["execute_best_model_only"] and hybrid_bundle_config:
            primary_model_cfg = hybrid_bundle_config.get("primary_model", {}) or {}
            filter_model_cfg = hybrid_bundle_config.get("filter_model", {}) or {}
            primary_name = str(primary_model_cfg.get("name", ""))
            filter_name = str(filter_model_cfg.get("name", ""))
            bundle_name = str(
                hybrid_bundle_config.get("name")
                or f"HYBRID__{primary_name}__{filter_name}"
            )

            primary_output = hybrid_live_outputs.get(primary_name.upper())
            filter_output = hybrid_live_outputs.get(filter_name.upper())

            if not primary_output or not filter_output:
                self.logger.error(
                    "No se pudo construir el bundle hÃƒÂ­brido de producciÃƒÂ³n. primary=%s listo=%s | filter=%s listo=%s",
                    primary_name,
                    bool(primary_output),
                    filter_name,
                    bool(filter_output),
                )
            else:
                primary_metrics = dict(primary_output.get("metrics") or {})
                filter_metrics = dict(filter_output.get("metrics") or {})
                signal_info = build_signal_from_hybrid_prediction(
                    pred_return=float(primary_output.get("pred_return", 0.0) or 0.0),
                    pip_size=pip_size,
                    min_pips_signal=min_pips_signal,
                    price_reference=price_now if not np.isnan(price_now) else None,
                    primary_model_metrics=primary_metrics,
                    prob_up=float(filter_output.get("prob_up", 0.0) or 0.0),
                    prob_hold=float(filter_output.get("prob_hold", 0.0) or 0.0),
                    prob_down=float(filter_output.get("prob_down", 0.0) or 0.0),
                    barrier_pips=float(barrier_settings["barrier_pips"]),
                    min_confidence=min_confidence if enable_confidence_filter else 0.0,
                    probability_threshold=float(barrier_settings["probability_threshold"]),
                    probability_margin=float(barrier_settings["probability_margin"]),
                    filter_model_metrics=filter_metrics,
                    require_alignment=bool(hybrid_bundle_config.get("require_alignment", True)),
                    filter_gate_mode=str(hybrid_bundle_config.get("filter_gate_mode", "full_signal") or "full_signal"),
                    support_probability_threshold=hybrid_bundle_config.get("support_probability_threshold"),
                    support_probability_margin=hybrid_bundle_config.get("support_probability_margin"),
                    support_score_min=hybrid_bundle_config.get("support_score_min"),
                    contradiction_margin=hybrid_bundle_config.get("contradiction_margin"),
                )

                pred_return = float(primary_output.get("pred_return", 0.0) or 0.0)
                prob_up = float(filter_output.get("prob_up", np.nan))
                prob_hold = float(filter_output.get("prob_hold", np.nan))
                prob_down = float(filter_output.get("prob_down", np.nan))
                pips = float(signal_info.get("predicted_pips", np.nan))
                if not np.isnan(price_now):
                    delta_price = pips * pip_size
                    price_target = price_now + delta_price
                else:
                    price_target = float("nan")
                    delta_price = float("nan")
                signal = str(signal_info["signal"])
                confidence = float(signal_info["confidence"])
                touch_probability = float(signal_info.get("touch_probability", np.nan))

                confirmation = self._evaluate_signal_confirmation(
                    signal=signal,
                    feature_row=last_row,
                )
                if signal in {"BUY", "SELL"} and not confirmation.get("passed", True):
                    self.logger.info(
                        "  -> Bundle hÃƒÂ­brido %s bloqueado por confirmaciÃƒÂ³n opcional: %s",
                        bundle_name,
                        confirmation.get("reason"),
                    )
                    signal = "HOLD"

                entry_price = float("nan")
                sl_price = float("nan")
                tp_price = float("nan")
                sl_pips = float("nan")
                tp_pips = float("nan")
                signal_target_tp_pips = float("nan")
                signal_target_sl_pips = float("nan")
                signal_target_tp_price = float("nan")
                signal_target_sl_price = float("nan")
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
                projected_total_open_risk_after_trade = max(
                    open_risk_amount + planned_additional_risk_amount,
                    0.0,
                )
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

                signal_target_levels = self._build_signal_target_levels(
                    signal=signal,
                    entry_reference=price_now if not np.isnan(price_now) else None,
                    pip_size=pip_size,
                    target_pips=signal_info.get("signal_target_pips"),
                )
                signal_target_tp_pips = signal_target_levels["signal_target_tp_pips"]
                signal_target_sl_pips = signal_target_levels["signal_target_sl_pips"]
                signal_target_tp_price = signal_target_levels["signal_target_tp_price"]
                signal_target_sl_price = signal_target_levels["signal_target_sl_price"]

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

                entry_management_plan = self._compute_entry_management_plan(
                    signal=signal,
                    total_volume_lots=volume_lots,
                    live_entry_price=live_entry_price,
                    live_sl_price=live_sl_price,
                    live_tp_price=live_tp_price,
                    digits=digits,
                    min_lot=min_lot,
                    lot_step=lot_step,
                    timeframe=str(self.config.get("data", {}).get("timeframe", "M5")),
                    signal_time=df_infer.index[-1],
                )
                if (
                    self._get_entry_management_settings()["disable_pending_when_filter_hold"]
                    and signal_info.get("filter_signal") == "HOLD"
                    and signal in {"BUY", "SELL"}
                ):
                    allow_filter_hold_small_market = self._should_allow_filter_hold_small_market(
                        signal=signal,
                        row=signal_info,
                    )
                    if not allow_filter_hold_small_market:
                        retrace_only_plan = self._build_retrace_only_entry_plan(
                            signal=signal,
                            total_volume_lots=volume_lots,
                            live_entry_price=live_entry_price,
                            live_sl_price=live_sl_price,
                            live_tp_price=live_tp_price,
                            digits=digits,
                            timeframe=str(self.config.get("data", {}).get("timeframe", "M5")),
                            signal_time=df_infer.index[-1],
                            comment="filter_hold_context_retrace_only",
                        )
                        pending_volume = pd.to_numeric(
                            pd.Series([retrace_only_plan.get("pending_order_volume_lots")]),
                            errors="coerce",
                        ).iloc[0]
                        if pd.notna(pending_volume) and float(pending_volume) > 0:
                            entry_management_plan = retrace_only_plan
                    else:
                        reduced_market_only = self._apply_reduced_market_only_to_payload(
                            total_volume_lots=volume_lots,
                            risk_amount=risk_amount,
                            allocated_risk_budget=allocated_risk_budget,
                            projected_total_open_risk_after_trade=projected_total_open_risk_after_trade,
                            min_lot=min_lot,
                            lot_step=lot_step,
                            market_fraction=float(self._get_entry_management_settings()["filter_hold_market_fraction"]),
                            comment="filter_hold_small_market_only",
                        )
                        entry_management_plan = reduced_market_only["entry_plan"]
                        volume_lots = reduced_market_only["volume_lots"]
                        risk_amount = reduced_market_only["risk_amount"]
                        allocated_risk_budget = reduced_market_only["allocated_risk_budget"]
                        projected_total_open_risk_after_trade = reduced_market_only[
                            "projected_total_open_risk_after_trade"
                        ]
                        direct_payload = self._apply_filter_hold_small_market_level_overrides(
                            signal=signal,
                            payload={
                                "signal_target_tp_price": signal_target_tp_price,
                                "signal_target_sl_price": signal_target_sl_price,
                                "signal_target_tp_pips": signal_target_tp_pips,
                                "signal_target_sl_pips": signal_target_sl_pips,
                                "entry_price": entry_price,
                                "planned_entry_price": entry_price,
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
                                "volume_lots": volume_lots,
                                "risk_amount": risk_amount,
                                "allocated_risk_budget": allocated_risk_budget,
                                "risk_per_pip_per_lot": risk_per_pip_per_lot,
                                "risk_per_lot_at_stop": risk_per_lot_at_stop,
                                "projected_total_open_risk_after_trade": projected_total_open_risk_after_trade,
                            },
                            predicted_pips=pips,
                            pip_size=pip_size,
                            digits=digits,
                        )
                        signal_target_tp_price = direct_payload["signal_target_tp_price"]
                        signal_target_sl_price = direct_payload["signal_target_sl_price"]
                        signal_target_tp_pips = direct_payload["signal_target_tp_pips"]
                        signal_target_sl_pips = direct_payload["signal_target_sl_pips"]
                        sl_price = direct_payload["sl_price"]
                        tp_price = direct_payload["tp_price"]
                        sl_pips = direct_payload["sl_pips"]
                        tp_pips = direct_payload["tp_pips"]
                        live_sl_price = direct_payload["live_sl_price"]
                        live_tp_price = direct_payload["live_tp_price"]
                        live_sl_pips = direct_payload["live_sl_pips"]
                        live_tp_pips = direct_payload["live_tp_pips"]
                        risk_amount = direct_payload["risk_amount"]
                        allocated_risk_budget = direct_payload["allocated_risk_budget"]
                        risk_per_lot_at_stop = direct_payload["risk_per_lot_at_stop"]
                        projected_total_open_risk_after_trade = direct_payload["projected_total_open_risk_after_trade"]

                if (
                    str(entry_management_plan.get("entry_management_comment") or "").strip().lower() == "split_retrace_limit"
                    and self._should_force_split_retrace_filter_opposite_retrace_only(
                        signal=signal,
                        row=signal_info,
                    )
                ):
                    retrace_only_plan = self._build_retrace_only_entry_plan(
                        signal=signal,
                        total_volume_lots=volume_lots,
                        live_entry_price=live_entry_price,
                        live_sl_price=live_sl_price,
                        live_tp_price=live_tp_price,
                        digits=digits,
                        timeframe=str(self.config.get("data", {}).get("timeframe", "M5")),
                        signal_time=df_infer.index[-1],
                        comment="split_retrace_filter_opposite_retrace_only",
                    )
                    pending_volume = pd.to_numeric(
                        pd.Series([retrace_only_plan.get("pending_order_volume_lots")]),
                        errors="coerce",
                    ).iloc[0]
                    if pd.notna(pending_volume) and float(pending_volume) > 0:
                        entry_management_plan = retrace_only_plan

                bundle_metrics = metrics_by_model.get(bundle_name.upper(), {})
                rmse = bundle_metrics.get("rmse")
                mae = bundle_metrics.get("mae")
                hit_rate = bundle_metrics.get("hit_rate")
                accuracy = bundle_metrics.get("accuracy")
                dm_stat = bundle_metrics.get("dm_stat")
                dm_pvalue = bundle_metrics.get("dm_pvalue")
                sharpe = bundle_metrics.get("sharpe")
                sortino = bundle_metrics.get("sortino")
                max_dd = bundle_metrics.get("max_drawdown")
                profit_factor = bundle_metrics.get("profit_factor")
                win_rate = bundle_metrics.get("win_rate")
                payoff_ratio = bundle_metrics.get("payoff_ratio")

                self.logger.info(
                    "  Ã°Å¸â€œË† Bundle %s -> retorno=%0.6f, pips=%0.2f, signal=%s, confidence=%0.3f, "
                    "primary=%s(%0.3f), filter=%s(%0.3f), prob_up=%0.3f, prob_hold=%0.3f, prob_down=%0.3f, "
                    "touch_prob=%0.3f, align=%s, confirm=%s, signal_tp=%s, signal_sl=%s, "
                    "entry_plan=%s, SL_plan=%s, TP_plan=%s, entry_live=%s, SL_live=%s, TP_live=%s, "
                    "lots=%0.2f, balance=%s, risk=%0.2f, entry_mgmt=%s, market_lots=%0.2f, pending_lots=%0.2f, "
                    "pending_price=%s, pending_sl=%s, pending_tp=%s",
                    bundle_name,
                    pred_return,
                    pips,
                    signal,
                    confidence,
                    signal_info.get("primary_signal"),
                    float(signal_info.get("primary_confidence", np.nan)),
                    signal_info.get("filter_signal"),
                    float(signal_info.get("filter_confidence", np.nan)),
                    prob_up if not np.isnan(prob_up) else float("nan"),
                    prob_hold if not np.isnan(prob_hold) else float("nan"),
                    prob_down if not np.isnan(prob_down) else float("nan"),
                    touch_probability if not np.isnan(touch_probability) else float("nan"),
                    bool(signal_info.get("alignment_ok", False)),
                    confirmation.get("reason"),
                    signal_target_tp_price,
                    signal_target_sl_price,
                    entry_price,
                    sl_price,
                    tp_price,
                    live_entry_price,
                    live_sl_price,
                    live_tp_price,
                    volume_lots,
                    balance,
                    risk_amount,
                    entry_management_plan["entry_management_comment"],
                    float(entry_management_plan["initial_market_volume_lots"]),
                    float(entry_management_plan["pending_order_volume_lots"]),
                    entry_management_plan["pending_order_price"],
                    entry_management_plan["pending_order_sl_price"],
                    entry_management_plan["pending_order_tp_price"],
                )

                rows.append(
                    {
                        "timestamp": df_infer.index[-1],
                        "release_id": active_release_id,
                        "strategy_profile": strategy_profile_name,
                        "magic_number": live_cfg["magic_number"],
                        "order_comment_prefix": live_cfg["order_comment_prefix"],
                        "symbol": symbol,
                        "timeframe": self.config.get("data", {}).get("timeframe", "UNKNOWN"),
                        "model": bundle_name,
                        "primary_model": primary_name,
                        "filter_model": filter_name,
                        "target_mode": "hybrid_primary_plus_filter",
                        "pred_return": pred_return,
                        "signal": signal,
                        "confidence": confidence,
                        "primary_signal": signal_info.get("primary_signal"),
                        "primary_confidence": signal_info.get("primary_confidence"),
                        "filter_signal": signal_info.get("filter_signal"),
                        "filter_confidence": signal_info.get("filter_confidence"),
                        "filter_passed": bool(signal_info.get("filter_passed", False)),
                        "filter_gate_mode": signal_info.get("filter_gate_mode"),
                        "filter_dominant_side": signal_info.get("filter_dominant_side"),
                        "filter_dominant_prob": signal_info.get("filter_dominant_prob"),
                        "filter_support_passed": signal_info.get("filter_support_passed"),
                        "filter_support_score": signal_info.get("filter_support_score"),
                        "filter_same_side_prob": signal_info.get("filter_same_side_prob"),
                        "filter_opposite_side_prob": signal_info.get("filter_opposite_side_prob"),
                        "filter_support_score_passed": signal_info.get("filter_support_score_passed"),
                        "filter_contradicted": bool(signal_info.get("filter_contradicted", False)),
                        "gate_passed": bool(signal_info.get("gate_passed", False)),
                        "alignment_ok": bool(signal_info.get("alignment_ok", False)),
                        "prob_up": prob_up,
                        "prob_hold": prob_hold,
                        "prob_down": prob_down,
                        "touch_probability": touch_probability,
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
                        "signal_candle_open": float(last_row["Open"]) if "Open" in last_row else np.nan,
                        "signal_candle_high": float(last_row["High"]) if "High" in last_row else np.nan,
                        "signal_candle_low": float(last_row["Low"]) if "Low" in last_row else np.nan,
                        "signal_candle_close": float(last_row["Close"]) if "Close" in last_row else np.nan,
                        "price_target": price_target,
                        "delta_price": delta_price,
                        "pips": pips,
                        "expected_move_pips": pips,
                        "target_barrier_pips": float(barrier_settings["barrier_pips"]),
                        "target_horizon_bars": int(barrier_settings["horizon_bars"]),
                        "signal_target_tp_price": signal_target_tp_price,
                        "signal_target_sl_price": signal_target_sl_price,
                        "signal_target_tp_pips": signal_target_tp_pips,
                        "signal_target_sl_pips": signal_target_sl_pips,
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
                        "entry_management_mode": entry_management_plan["entry_management_mode"],
                        "entry_management_split_active": entry_management_plan["entry_management_split_active"],
                        "entry_management_initial_market_fraction": entry_management_plan["entry_management_initial_market_fraction"],
                        "entry_management_pending_fraction": entry_management_plan["entry_management_pending_fraction"],
                        "entry_management_retrace_fraction_of_stop": entry_management_plan["entry_management_retrace_fraction_of_stop"],
                        "entry_management_total_volume_lots": entry_management_plan["entry_management_total_volume_lots"],
                        "initial_market_volume_lots": entry_management_plan["initial_market_volume_lots"],
                        "pending_order_volume_lots": entry_management_plan["pending_order_volume_lots"],
                        "pending_order_price": entry_management_plan["pending_order_price"],
                        "pending_order_type": entry_management_plan["pending_order_type"],
                        "pending_order_sl_price": entry_management_plan["pending_order_sl_price"],
                        "pending_order_tp_price": entry_management_plan["pending_order_tp_price"],
                        "pending_order_expiry_time": entry_management_plan["pending_order_expiry_time"],
                        "entry_management_comment": entry_management_plan["entry_management_comment"],
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
                        "is_best_model": True,
                        "rmse_backtest": rmse,
                        "mae_backtest": mae,
                        "hit_rate_backtest": hit_rate,
                        "accuracy_backtest": accuracy,
                        "f1_score_backtest": bundle_metrics.get("f1_score"),
                        "precision_backtest": bundle_metrics.get("precision"),
                        "recall_backtest": bundle_metrics.get("recall"),
                        "dm_stat_backtest": dm_stat,
                        "dm_pvalue_backtest": dm_pvalue,
                        "sharpe_backtest": sharpe,
                        "sortino_backtest": sortino,
                        "calmar_backtest": bundle_metrics.get("calmar"),
                        "max_drawdown_backtest": max_dd,
                        "profit_factor_backtest": profit_factor,
                        "win_rate_backtest": win_rate,
                        "payoff_ratio_backtest": payoff_ratio,
                        "consistency_ratio_backtest": bundle_metrics.get("consistency_ratio"),
                        "avg_trade_return_backtest": bundle_metrics.get("avg_trade_return"),
                        "primary_model_profit_factor_backtest": primary_metrics.get("profit_factor"),
                        "filter_model_profit_factor_backtest": filter_metrics.get("profit_factor"),
                    }
                )
                rows[-1].update(self._get_live_feature_snapshot(last_row))

        if not rows:
            self.logger.error("No se generÃ³ ninguna seÃ±al de producciÃ³n (todas fallaron).")
            self._sync_live_trade_report()
            return

        df_rows = pd.DataFrame(rows)
        staging_runtime_ctx = {
            "price_now": price_now,
            "atr_value": atr_value,
            "pip_size": pip_size,
            "digits": digits,
            "point": point,
            "contract_size": contract_size,
            "min_lot": min_lot,
            "lot_step": lot_step,
            "balance": balance,
            "risk_cfg_dict": risk_cfg_dict,
            "total_risk_budget": total_risk_budget,
            "per_trade_risk_budget": per_trade_risk_budget,
            "open_risk_amount": open_risk_amount,
            "planned_additional_risk_amount": 0.0,
            "positions_without_sl": positions_without_sl,
            "market_tick": market_tick,
            "timeframe": self.config.get("data", {}).get("timeframe", "UNKNOWN"),
        }
        staging_runtime_ctx["_entry_execution_confirmation_m1_snapshot"] = (
            self._get_execution_confirmation_m1_snapshot(
                runtime_ctx=staging_runtime_ctx,
                settings=self._get_entry_staging_settings(),
            )
        )
        df_rows = self._apply_entry_staging_to_production_rows(
            df_rows=df_rows,
            runtime_ctx=staging_runtime_ctx,
        )
        df_rows = self._apply_entry_grid_to_production_rows(
            df_rows=df_rows,
            runtime_ctx=staging_runtime_ctx,
        )
        df_rows = self._apply_entry_context_guard_to_production_rows(
            df_rows=df_rows,
        )

        # 7) Guardar seÃ±ales
        output_paths = self._get_production_output_paths()
        self._append_rows_to_csv(output_paths["signals"], df_rows)
        self.logger.info(f"\nðŸ’¾ SeÃ±ales de producciÃ³n guardadas en: {output_paths['signals']}")

        # 8) EjecuciÃ³n real opcional + reconciliaciÃ³n del reporte de trades
        self._execute_live_orders(df_rows)
        self._sync_live_trade_report()
        self.logger.info("âœ… MODO PRODUCCIÃ“N COMPLETADO\n")

    def _run_sync_trades_mode(self) -> None:
        """Sincroniza el reporte local con el estado real de posiciones/deals en MT5."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("MODO: SYNC TRADES")
        self.logger.info("=" * 60 + "\n")
        if self._pause_if_market_closed(mode="sync_trades"):
            self.logger.info("MODO SYNC TRADES EN PAUSA.\n")
            return

        self._ensure_mt5_client()
        lifecycle = self._sync_live_trade_report()
        n_closed = 0
        if lifecycle is not None and not lifecycle.empty and "status" in lifecycle.columns:
            n_closed = int((lifecycle["status"].astype(str).str.upper() == "CLOSED").sum())

        self.logger.info(f"ðŸ”„ SincronizaciÃ³n completada. Trades cerrados registrados: {n_closed}")
        self.logger.info("âœ… MODO SYNC TRADES COMPLETADO\n")

    def _run_monitor_runtime_mode(self) -> None:
        """Monitor liviano para trades recientes; no fuerza cierres en rojo."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("MODO: RUNTIME MONITOR")
        self.logger.info("=" * 60 + "\n")

        if not self._get_runtime_monitor_settings()["enabled"]:
            self.logger.info("-> Runtime monitor deshabilitado en config. Saltando.")
            return
        if self._pause_if_market_closed(mode="monitor_runtime"):
            self.logger.info("MODO RUNTIME MONITOR EN PAUSA.\n")
            return

        try:
            self._ensure_mt5_client()
            previous_level = getattr(self, "_runtime_monitor_previous_level", None)
            df_raw = self._load_data()
            df_clean = self._clean_data(df_raw)
            df_features = self._generate_features(df_clean)
            if previous_level is not None:
                self.logger.setLevel(previous_level)
            feature_row = df_features.iloc[-1] if df_features is not None and not df_features.empty else None
            current_bar_timestamp = df_features.index[-1] if df_features is not None and not df_features.empty else None

            lifecycle = self._sync_live_trade_report(
                apply_runtime_monitor=True,
                runtime_feature_row=feature_row,
                current_bar_timestamp=current_bar_timestamp,
            )
            n_open = 0
            if lifecycle is not None and not lifecycle.empty and "status" in lifecycle.columns:
                n_open = int((lifecycle["status"].astype(str).str.upper() == "OPEN").sum())

            self.logger.info(f"Ã°Å¸â€Â Runtime monitor completado. Trades abiertos revisados: {n_open}")
            self.logger.info("Ã¢Å“â€¦ MODO RUNTIME MONITOR COMPLETADO\n")
        except RuntimeError as exc:
            message = str(exc or "").lower()
            if "authorization failed" in message or "initialize failed" in message:
                self.logger.warning(
                    "Runtime monitor omitido por fallo de autorizacion/conexion MT5: %s",
                    exc,
                )
                return
            raise

    def _get_best_model_from_config(self) -> dict | None:
        """
        Identifica el mejor modelo segÃºn la config.
        Prioridad:
        1) Modelo con is_best: true y enabled.
        2) Primer modelo enabled que tenga 'params'.
        """
        models = self.config.get("models", [])

        # 1) Buscar marcado como is_best
        for m in models:
            if (
                m.get("enabled", True)
                and m.get("is_best", False)
                and self._is_model_selection_candidate(model_cfg=m, model_name=m.get("name"))
            ):
                return m

        # 2) Fallback: primer modelo enabled con params
        for m in models:
            if (
                m.get("enabled", True)
                and "params" in m
                and self._is_model_selection_candidate(model_cfg=m, model_name=m.get("name"))
            ):
                return m

        # 3) Ãšltimo fallback: cualquier modelo enabled con params
        for m in models:
            if m.get("enabled", True) and "params" in m:
                return m

        return None

    def _get_best_decision_bundle_from_config(self) -> dict[str, Any] | None:
        """Devuelve el bundle hÃ­brido optimizado, si existe en la config activa."""
        bundle = self.config.get("decision_bundle")
        if not isinstance(bundle, dict):
            return None
        mode = str(bundle.get("mode", "") or "").strip().lower()
        if mode != "hybrid_primary_plus_filter":
            return None
        primary_model = bundle.get("primary_model")
        filter_model = bundle.get("filter_model")
        if not isinstance(primary_model, dict) or not isinstance(filter_model, dict):
            return None
        return bundle

    def _run_clear_cache_mode(self) -> None:
        """
        Modo para limpiar los archivos de cachÃ© de datos.
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("MODO: LIMPIEZA DE CACHÃ‰")
        self.logger.info("="*60 + "\n")

        data_config = self.config.get("data", {})
        mt5_config = self.config.get("mt5", {})
        
        # No es necesario conectar a MT5, solo instanciar el loader
        # para acceder a su mÃ©todo de limpieza.
        data_loader = DataLoader(mt5_config=mt5_config)
        
        symbol_to_clear = data_config.get("symbol")
        self.logger.info(f"Limpiando cachÃ© para el sÃ­mbolo: {symbol_to_clear}...")
        data_loader.clear_cache(symbol=symbol_to_clear)
        self.logger.info("\nâœ… MODO LIMPIEZA DE CACHÃ‰ COMPLETADO")

    # --- MÃ‰TODOS AUXILIARES DEL PIPELINE ---

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
        
        # Mensajes para indicar de dÃ³nde vienen los parÃ¡metros
        if "symbol" in data_config:
            self.logger.info(f"  -> SÃ­mbolo '{data_config['symbol']}' cargado desde config/config.yaml")
        else:
            self.logger.info(f"  -> SÃ­mbolo '{df.attrs['symbol']}' (por defecto) usado, no especificado en config/config.yaml")
        if "timeframe" in data_config:
            self.logger.info(f"  -> Timeframe '{data_config['timeframe']}' cargado desde config/config.yaml")
        else:
            self.logger.info(f"  -> Timeframe '{df.attrs['timeframe']}' (por defecto) usado, no especificado en config/config.yaml")
        self.logger.info(f"âœ“ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas.")
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Paso 2: Limpiar datos usando DataCleaner."""
        self.logger.info("PASO 2: LIMPIANDO DATOS")
        self.logger.info("-" * 60)
        self.data_cleaner = DataCleaner(self.config.get("data_cleaning", {}))
        df_clean = self.data_cleaner.clean(df)
        self.logger.info(f"âœ“ Datos limpios: {df_clean.shape[0]} filas restantes.")
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
            self.logger.info(f"  -> Retornos agregados para perÃ­odos: {periods}")

        # 2. Generar indicadores tÃ©cnicos
        if features_config.get("technical_indicators", {}).get("enabled", False):
            indicators = features_config.get("technical_indicators", {}).get("indicators")
            df_features = FeatureEngineer.add_technical_indicators(df_features, indicators=indicators)
            self.logger.info("  -> Indicadores tecnicos agregados.")
        if features_config.get("price_action", {}).get("enabled", False):
            price_action_cfg = features_config.get("price_action", {}) or {}
            df_features = FeatureEngineer.add_price_action_features(
                df_features,
                pip_size=float((self.config.get("data", {}) or {}).get("pip_size", 0.0001) or 0.0001),
                features=price_action_cfg.get("features"),
            )
            self.logger.info("  -> Indicadores tÃ©cnicos agregados.")
        df_features = self._add_entry_execution_context_features(
            df_features,
            pip_size=float((self.config.get("data", {}) or {}).get("pip_size", 0.0001) or 0.0001),
        )
        self.logger.info("  -> Features de ejecucion/contexto agregadas (EMA20, SessionVWAP, stretch).")

        # 3. Generar features rezagados (lags)
        if features_config.get("lag_features", {}).get("enabled", False):
            lag_config = features_config["lag_features"]
            for col in lag_config.get("columns", []):
                if col in df_features.columns:
                    df_features = FeatureEngineer.add_lag_features(df_features, col=col, lags=lag_config.get("lags", []))
                    self.logger.info(f"  -> Lags agregados para la columna: '{col}'")

        # 4. Aprendizaje no supervisado (regÃ­menes de mercado)
        if self._get_target_mode() == "barrier_event" or self._is_hybrid_mode():
            barrier_cfg = self._get_barrier_settings()
            df_features = FeatureEngineer.add_barrier_targets(
                df_features,
                barrier_pips=float(barrier_cfg["barrier_pips"]),
                horizon_bars=int(barrier_cfg["horizon_bars"]),
                pip_size=float(barrier_cfg["pip_size"]),
                price_col=str(barrier_cfg["price_col"]),
                high_col=str(barrier_cfg["high_col"]),
                low_col=str(barrier_cfg["low_col"]),
            )
            self.logger.info(
                "  -> Targets de barrera agregados: %sp / %sb",
                int(barrier_cfg["barrier_pips"]),
                int(barrier_cfg["horizon_bars"]),
            )

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
                        f"  -> RegÃ­menes de mercado agregados con KMeans (n_clusters={n_clusters})."
                    )
                else:
                    self.logger.warning(f"  -> MÃ©todo no supervisado no soportado: {method}")
            except Exception as e:
                self.logger.warning(f"  -> No se pudieron agregar regÃ­menes de mercado: {e}")

        # --- NUEVO: Log para inspeccionar NaNs despuÃ©s de la generaciÃ³n ---
        nan_counts = df_features.isnull().sum()
        nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
        if not nan_counts.empty:
            self.logger.info("  -> Conteos de valores NaN generados por las features:")
            # Usamos print para asegurar que se muestre completo sin truncar
            print(nan_counts.to_string())
        else:
            self.logger.info("  -> No se generaron valores NaN en este paso.")
        # --- FIN NUEVO ---
        self.logger.info(f"âœ“ Features generadas. Total columnas: {df_features.shape[1]}.")
        return df_features

    def _perform_eda(self, df: pd.DataFrame) -> None:
        """Ejecuta el anÃ¡lisis exploratorio."""
        if not self.config.get("eda", {}).get("enabled", False):
            self.logger.info("-> AnÃ¡lisis Exploratorio (EDA) deshabilitado en config. Saltando.")
            return
            
        self.logger.info("PASO 4: REALIZANDO ANÃLISIS EXPLORATORIO (EDA)")
        self.logger.info("-" * 60)
                # 1) Definir sÃ­mbolo y columna de precio desde la config
        symbol = self.config.get("data", {}).get("symbol", "UNKNOWN")
        price_col = self.config.get("eda", {}).get("price_col", "Close")

        # 2) Definir directorio de salida para el EDA
        output_root = self.config.get("output", {}).get("dir", "outputs")
        eda_dir = Path(output_root) / "eda"

        # 3) Ejecutar el EDA con la clase actual (exploratory_analysis.py)
        self.eda = ExploratoryAnalysis(output_dir=str(eda_dir))
        self.eda.analyze(df, symbol=symbol, price_col=price_col)

        self.logger.info("âœ“ AnÃ¡lisis exploratorio completado.")

    def _save_processed_data(self, df: pd.DataFrame) -> None:
        """Guarda el dataframe procesado en los formatos especificados."""
        output_config = self.config.get("output", {})
        if not output_config.get("save_predictions", False): return

        output_dir = Path(output_config.get("dir", "outputs"))
        formats = output_config.get("formats", ["csv"])
        
        if "csv" in formats:
            df.to_csv(output_dir / "processed_data.csv")
            self.logger.info(f"ðŸ’¾ Datos procesados guardados en: {output_dir / 'processed_data.csv'}")

    def _save_dataframes_to_excel(self, dataframes: dict[str, pd.DataFrame]):
        """Guarda mÃºltiples dataframes en un solo archivo Excel."""
        output_config = self.config.get("output", {})
        if "excel" not in output_config.get("formats", []): return

        output_dir = Path(output_config.get("dir", "outputs"))
        excel_path = output_dir / "trading_data_analysis.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for sheet_name, df in dataframes.items():
                df.to_excel(writer, sheet_name=sheet_name, index=True)
        self.logger.info(f"ðŸ’¾ Reporte de datos guardado en: {excel_path}")

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
    - La seÃ±al de trading se basa en sign(y_pred): >0 = LONG, <0 = SHORT.
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
    # Nos aseguramos de que idx estÃ© dentro de price_series
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

    self.logger.info(f"ðŸ“ˆ GrÃ¡fico de entradas guardado en: {path}")
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
    Grafica la precisiÃ³n direccional del modelo a lo largo del backtest:
    - PrecisiÃ³n acumulada.
    - PrecisiÃ³n mÃ³vil en ventana (rolling).
    """
    n = min(len(y_true), len(y_pred), len(idx))
    y_true_arr = np.asarray(y_true[:n], dtype=float)
    y_pred_arr = np.asarray(y_pred[:n], dtype=float)
    idx = idx[:n]

    true_dir = np.sign(y_true_arr)
    pred_dir = np.sign(y_pred_arr)
    hits = (true_dir == pred_dir).astype(int)

    hits_series = pd.Series(hits, index=idx)

    # PrecisiÃ³n acumulada
    cum_hits = hits_series.cumsum() / np.arange(1, len(hits_series) + 1)

    # PrecisiÃ³n rolling
    rolling_hits = hits_series.rolling(window).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Acumulada
    ax1.plot(cum_hits.index, cum_hits.values, linewidth=1.5, label="PrecisiÃ³n acumulada")
    ax1.axhline(0.5, linestyle="--", color="gray", linewidth=1, label="Azar (50%)")
    ax1.set_ylabel("Accuracy acumulado")
    ax1.set_title(f"{symbol} - {model_name}\nEvoluciÃ³n de la precisiÃ³n direccional", fontsize=13, weight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    # Rolling
    ax2.plot(rolling_hits.index, rolling_hits.values, linewidth=1.5, label=f"PrecisiÃ³n mÃ³vil ({window} trades)")
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

    self.logger.info(f"ðŸ“Š Curva de accuracy guardada en: {path}")
    return str(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Trading AlgorÃ­tmico.")
    parser.add_argument("--mode", type=str, default="eda", 
                        choices=["eda", "train", "backtest","production", "test", "sync_trades", "monitor_runtime", "clear_cache"],
                        help="Modo de ejecuciÃ³n del pipeline.")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Ruta al archivo de configuraciÃ³n YAML.")
    args = parser.parse_args()
    
    pipeline = TradingPipeline(config_path=args.config)
    pipeline.run(mode=args.mode)

