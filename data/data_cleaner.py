# data/data_cleaner.py
"""
Módulo de limpieza y preprocesamiento de datos
Maneja missing values, outliers, duplicados y normalización
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from scipy import stats


class DataCleaner:
    """
    Pipeline de limpieza de datos para series temporales financieras
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Configuración de limpieza (thresholds, métodos, etc)
        """
        self.config = config or self._get_default_config()
        self.cleaning_report: Dict[str, Any] = {}
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Configuración por defecto"""
        return {
            "handle_missing": "ffill",  # ffill, bfill, interpolate, drop
            "outlier_method": "iqr",     # iqr, zscore, winsorize
            "outlier_threshold": 3.0,
            "handle_outliers": "cap",    # cap, remove, interpolate
            "remove_duplicates": True,
            "validate_ohlc": True,
            "min_valid_ratio": 0.90,     # Mínimo % de datos válidos
        }
    
    def clean(self, df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
        """
        Ejecuta pipeline completo de limpieza
        
        Args:
            df: DataFrame con datos OHLCV
            price_col: Columna principal de precio (para cálculos)
            
        Returns:
            DataFrame limpio
        """
        self.cleaning_report = {
            "original_shape": df.shape,
            "steps": []
        }
        
        df_clean = df.copy()
        
        # 1. Ordenar por índice
        df_clean = self._ensure_sorted_index(df_clean)
        
        # 2. Remover duplicados
        if self.config["remove_duplicates"]:
            df_clean = self._remove_duplicates(df_clean)
        
        # 3. Validar OHLC
        if self.config["validate_ohlc"]:
            df_clean = self._validate_ohlc(df_clean)
        
        # 4. Manejar valores faltantes
        df_clean = self._handle_missing_values(df_clean)
        
        # 5. Detectar y manejar outliers
        df_clean = self._handle_outliers(df_clean, price_col)
        
        # 6. Validación final
        valid_ratio = len(df_clean) / len(df)
        if valid_ratio < self.config["min_valid_ratio"]:
            raise ValueError(
                f"Demasiados datos inválidos removidos: "
                f"{(1-valid_ratio)*100:.1f}% perdido"
            )
        
        self.cleaning_report["final_shape"] = df_clean.shape
        self.cleaning_report["data_loss_pct"] = (1 - valid_ratio) * 100
        
        return df_clean
    
    def _ensure_sorted_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Asegura que el índice esté ordenado cronológicamente"""
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
            self.cleaning_report["steps"].append("Índice ordenado")
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remueve timestamps duplicados"""
        n_before = len(df)
        df = df[~df.index.duplicated(keep="first")]
        n_removed = n_before - len(df)
        
        if n_removed > 0:
            self.cleaning_report["steps"].append(
                f"Duplicados removidos: {n_removed}"
            )
        
        return df
    
    def _validate_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Valida y corrige relaciones OHLC inconsistentes
        - High debe ser >= Low
        - High debe ser >= Open y Close
        - Low debe ser <= Open y Close
        """
        if not all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
            return df
        
        n_invalid = 0
        
        # High < Low (intercambiar)
        mask = df["High"] < df["Low"]
        if mask.any():
            n_invalid += mask.sum()
            df.loc[mask, ["High", "Low"]] = df.loc[
                mask, ["Low", "High"]
            ].values
        
        # High < Open o Close (ajustar High)
        mask = df["High"] < df[["Open", "Close"]].max(axis=1)
        if mask.any():
            n_invalid += mask.sum()
            df.loc[mask, "High"] = df.loc[mask, ["Open", "Close"]].max(axis=1)
        
        # Low > Open o Close (ajustar Low)
        mask = df["Low"] > df[["Open", "Close"]].min(axis=1)
        if mask.any():
            n_invalid += mask.sum()
            df.loc[mask, "Low"] = df.loc[mask, ["Open", "Close"]].min(axis=1)
        
        if n_invalid > 0:
            self.cleaning_report["steps"].append(
                f"OHLC corregidos: {n_invalid} filas"
            )
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maneja valores faltantes según configuración"""
        n_missing_before = df.isnull().sum().sum()
        
        if n_missing_before == 0:
            return df
        
        method = self.config["handle_missing"]
        
        if method == "ffill":
            df = df.ffill()
        elif method == "bfill":
            df = df.bfill()
        elif method == "interpolate":
            df = df.interpolate(method="time")
        elif method == "drop":
            df = df.dropna()
        else:
            raise ValueError(f"Método no soportado: {method}")
        
        # Si aún quedan NaN, llenar con método alternativo
        if df.isnull().any().any():
            df = df.ffill().bfill()
        
        n_missing_after = df.isnull().sum().sum()
        
        self.cleaning_report["steps"].append(
            f"Missing values ({method}): {n_missing_before} → {n_missing_after}"
        )
        
        return df
    
    def _handle_outliers(
        self,
        df: pd.DataFrame,
        price_col: str
    ) -> pd.DataFrame:
        """Detecta y maneja outliers en la serie de precios"""
        if price_col not in df.columns:
            return df
        
        # Detectar outliers
        outlier_mask = self._detect_outliers(
            df[price_col],
            method=self.config["outlier_method"],
            threshold=self.config["outlier_threshold"]
        )
        
        n_outliers = outlier_mask.sum()
        
        if n_outliers == 0:
            return df
        
        # Manejar según configuración
        handle_method = self.config["handle_outliers"]
        
        if handle_method == "cap":
            # Winsorization: reemplazar con percentiles
            lower = df[price_col].quantile(0.01)
            upper = df[price_col].quantile(0.99)
            df.loc[outlier_mask, price_col] = df.loc[
                outlier_mask, price_col
            ].clip(lower, upper)
            
        elif handle_method == "interpolate":
            df.loc[outlier_mask, price_col] = np.nan
            df[price_col] = df[price_col].interpolate(method="time")
            
        elif handle_method == "remove":
            df = df[~outlier_mask]
        
        self.cleaning_report["steps"].append(
            f"Outliers ({handle_method}): {n_outliers} detectados"
        )
        
        return df
    
    @staticmethod
    def _detect_outliers(
        series: pd.Series,
        method: str = "iqr",
        threshold: float = 3.0
    ) -> pd.Series:
        """
        Detecta outliers en una serie
        
        Returns:
            Serie booleana (True = outlier)
        """
        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            return (series < lower) | (series > upper)
        
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(series.dropna()))
            mask = pd.Series(False, index=series.index)
            mask.loc[series.dropna().index] = z_scores > threshold
            return mask
        
        else:
            raise ValueError(f"Método no soportado: {method}")
    
    def get_report(self) -> str:
        """Genera reporte legible de limpieza"""
        if not self.cleaning_report:
            return "No se ha ejecutado limpieza"
        
        lines = [
            "=" * 60,
            "REPORTE DE LIMPIEZA DE DATOS",
            "=" * 60,
            f"Shape original: {self.cleaning_report['original_shape']}",
            f"Shape final:    {self.cleaning_report['final_shape']}",
            f"Pérdida datos:  {self.cleaning_report['data_loss_pct']:.2f}%",
            "",
            "Pasos ejecutados:",
        ]
        
        for step in self.cleaning_report["steps"]:
            lines.append(f"  ✓ {step}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


class FeatureEngineer:
    """
    Generador de features técnicos para modelos
    """
    
    @staticmethod
    def add_returns(
        df: pd.DataFrame,
        price_col: str = "Close",
        periods: list[int] = [1]
    ) -> pd.DataFrame:
        """
        Agrega retornos simples y logarítmicos
        
        Args:
            df: DataFrame con precios
            price_col: Columna de precio
            periods: Lista de periodos para calcular retornos
            
        Returns:
            DataFrame con nuevas columnas de retornos
        """
        df = df.copy()
        
        for period in periods:
            suffix = f"_{period}"
            # Retorno simple
            df[f"Return{suffix}"] = df[price_col].pct_change(period)
            
            # Retorno simple futuro a n pasos.
            # Ej: ReturnFwd_4 en fecha t = (P[t+4] / P[t]) - 1
            df[f"ReturnFwd{suffix}"] = (df[price_col].shift(-period) / df[price_col]) - 1.0

            
            # Retorno logarítmico
            df[f"LogReturn{suffix}"] = np.log(
                df[price_col] / df[price_col].shift(period)
            )
        
        return df
    
    @staticmethod
    def add_technical_indicators(
        df: pd.DataFrame,
        price_col: str = "Close",
        indicators: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Agrega indicadores técnicos básicos
        """
        df = df.copy()
        requested = {str(ind).strip().lower() for ind in (indicators or []) if str(ind).strip()}

        def wants(*names: str) -> bool:
            if not requested:
                return True
            normalized = {str(name).strip().lower() for name in names}
            return any(name in requested for name in normalized)

        def resolve_volume_column() -> str | None:
            """Prioriza TickVolume cuando Volume real no aporta informacion."""
            candidates = ["TickVolume", "Volume"]
            for column in candidates:
                if column not in df.columns:
                    continue
                series = pd.to_numeric(df[column], errors="coerce")
                if series.notna().any() and series.abs().sum() > 0:
                    return column
            return None
        
        # Medias móviles
        if wants("sma_20"):
            df["SMA_20"] = df[price_col].rolling(20).mean()
        if wants("sma_50"):
            df["SMA_50"] = df[price_col].rolling(50).mean()
        if wants("ema_12", "macd"):
            df["EMA_12"] = df[price_col].ewm(span=12, adjust=False).mean()
        if wants("ema_26", "macd"):
            df["EMA_26"] = df[price_col].ewm(span=26, adjust=False).mean()
        
        # RSI
        delta = df[price_col].diff()
        if wants("rsi_14"):
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            df["RSI_14"] = 100 - (100 / (1 + rs))
        
        # MACD
        if wants("macd"):
            if "EMA_12" not in df.columns:
                df["EMA_12"] = df[price_col].ewm(span=12, adjust=False).mean()
            if "EMA_26" not in df.columns:
                df["EMA_26"] = df[price_col].ewm(span=26, adjust=False).mean()
            df["MACD"] = df["EMA_12"] - df["EMA_26"]
            df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
            df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        
        # Bollinger Bands
        if wants("bollinger_bands", "bb_upper", "bb_lower", "bb_width"):
            sma_20 = df[price_col].rolling(20).mean()
            std_20 = df[price_col].rolling(20).std()
            df["BB_Upper"] = sma_20 + (2 * std_20)
            df["BB_Lower"] = sma_20 - (2 * std_20)
            df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]

        if wants("roc_3"):
            df["ROC_3"] = df[price_col].pct_change(3)
        if wants("roc_6"):
            df["ROC_6"] = df[price_col].pct_change(6)
        
        # ATR (si hay OHLC)
        if all(c in df.columns for c in ["High", "Low"]):
            high_low = df["High"] - df["Low"]
            high_close = np.abs(df["High"] - df[price_col].shift())
            low_close = np.abs(df["Low"] - df[price_col].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            tr_sum_14 = tr.rolling(14).sum()
            if wants("atr_14", "adx_14"):
                df["ATR_14"] = tr.rolling(14).mean()

            if wants("adx_14"):
                up_move = df["High"].diff()
                down_move = -df["Low"].diff()
                plus_dm = pd.Series(
                    np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                    index=df.index,
                )
                minus_dm = pd.Series(
                    np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                    index=df.index,
                )
                plus_di = 100.0 * plus_dm.rolling(14).sum() / tr_sum_14.replace(0, np.nan)
                minus_di = 100.0 * minus_dm.rolling(14).sum() / tr_sum_14.replace(0, np.nan)
                dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
                df["ADX_14"] = dx.rolling(14).mean()

        volume_column = resolve_volume_column()
        if volume_column is not None:
            volume_series = pd.to_numeric(df[volume_column], errors="coerce")
            if wants("tick_volume_roc_3"):
                df["TickVolume_ROC_3"] = volume_series.pct_change(3)
            if wants("tick_volume_zscore_20"):
                volume_mean_20 = volume_series.rolling(20).mean()
                volume_std_20 = volume_series.rolling(20).std()
                df["TickVolume_ZScore_20"] = (volume_series - volume_mean_20) / volume_std_20.replace(0, np.nan)
            if any(
                wants(name)
                for name in (
                    "close_location_value",
                    "directional_volume_proxy",
                    "directional_volume_proxy_zscore_20",
                )
            ) and all(c in df.columns for c in ["High", "Low", price_col]):
                close_location_value = (
                    ((2.0 * df[price_col]) - df["High"] - df["Low"])
                    / (df["High"] - df["Low"]).replace(0, np.nan)
                )
                directional_volume_proxy = volume_series * close_location_value
                if wants("close_location_value"):
                    df["CloseLocationValue"] = close_location_value
                if wants("directional_volume_proxy"):
                    df["DirectionalVolumeProxy"] = directional_volume_proxy
                if wants("directional_volume_proxy_zscore_20"):
                    proxy_mean_20 = directional_volume_proxy.rolling(20).mean()
                    proxy_std_20 = directional_volume_proxy.rolling(20).std()
                    df["DirectionalVolumeProxy_ZScore_20"] = (
                        directional_volume_proxy - proxy_mean_20
                    ) / proxy_std_20.replace(0, np.nan)

            if wants("mfi_14") and all(c in df.columns for c in ["High", "Low"]):
                typical_price = (df["High"] + df["Low"] + df[price_col]) / 3.0
                raw_money_flow = typical_price * volume_series
                tp_delta = typical_price.diff()
                positive_flow = raw_money_flow.where(tp_delta > 0, 0.0)
                negative_flow = raw_money_flow.where(tp_delta < 0, 0.0).abs()
                positive_mf = positive_flow.rolling(14).sum()
                negative_mf = negative_flow.rolling(14).sum()
                money_ratio = positive_mf / negative_mf.replace(0, np.nan)
                df["MFI_14"] = 100 - (100 / (1 + money_ratio))
        
        return df

    @staticmethod
    def add_price_action_features(
        df: pd.DataFrame,
        *,
        price_col: str = "Close",
        open_col: str = "Open",
        high_col: str = "High",
        low_col: str = "Low",
        pip_size: float = 0.0001,
        features: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Agrega features de microestructura / price action de bajo costo.

        Pensadas para targets intradia de barrera en series como EURUSD M5.
        """
        df = df.copy()
        requested = {str(item).strip().lower() for item in (features or []) if str(item).strip()}

        def wants(*names: str) -> bool:
            if not requested:
                return True
            normalized = {str(name).strip().lower() for name in names}
            return any(name in requested for name in normalized)

        if not all(col in df.columns for col in [price_col, open_col, high_col, low_col]):
            return df

        pip_size = max(float(pip_size or 0.0001), 1e-12)
        close = pd.to_numeric(df[price_col], errors="coerce")
        open_ = pd.to_numeric(df[open_col], errors="coerce")
        high = pd.to_numeric(df[high_col], errors="coerce")
        low = pd.to_numeric(df[low_col], errors="coerce")

        bar_range = (high - low).replace(0, np.nan)
        body = close - open_
        upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
        lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
        prior_high = high.shift(1)
        prior_low = low.shift(1)
        rolling_high_3 = high.shift(1).rolling(3).max()
        rolling_low_3 = low.shift(1).rolling(3).min()
        rolling_high_6 = high.shift(1).rolling(6).max()
        rolling_low_6 = low.shift(1).rolling(6).min()

        atr = pd.to_numeric(df["ATR_14"], errors="coerce") if "ATR_14" in df.columns else pd.Series(np.nan, index=df.index)
        atr_pips = (atr / pip_size).replace(0, np.nan)

        if wants("body_pips"):
            df["BodyPips"] = body / pip_size
        if wants("upper_wick_pips"):
            df["UpperWickPips"] = upper_wick / pip_size
        if wants("lower_wick_pips"):
            df["LowerWickPips"] = lower_wick / pip_size
        if wants("range_pips"):
            df["RangePips"] = bar_range / pip_size
        if wants("body_over_range"):
            df["BodyOverRange"] = body.abs() / bar_range
        if wants("close_location_in_bar"):
            df["CloseLocationInBar"] = (close - low) / bar_range
        if wants("range_over_atr"):
            df["RangeOverATR"] = (bar_range / pip_size) / atr_pips
        if wants("body_over_atr"):
            df["BodyOverATR"] = (body.abs() / pip_size) / atr_pips
        if wants("break_above_prev_high"):
            df["BreakAbovePrevHigh"] = (high > prior_high).astype(float)
        if wants("break_below_prev_low"):
            df["BreakBelowPrevLow"] = (low < prior_low).astype(float)
        if wants("break_above_recent_high_3"):
            df["BreakAboveRecentHigh3"] = (high > rolling_high_3).astype(float)
        if wants("break_below_recent_low_3"):
            df["BreakBelowRecentLow3"] = (low < rolling_low_3).astype(float)
        if wants("break_above_recent_high_6"):
            df["BreakAboveRecentHigh6"] = (high > rolling_high_6).astype(float)
        if wants("break_below_recent_low_6"):
            df["BreakBelowRecentLow6"] = (low < rolling_low_6).astype(float)
        if wants("breakout_margin_high_3_pips"):
            df["BreakoutMarginHigh3Pips"] = (close - rolling_high_3) / pip_size
        if wants("breakout_margin_low_3_pips"):
            df["BreakoutMarginLow3Pips"] = (rolling_low_3 - close) / pip_size
        if wants("breakout_margin_high_6_pips"):
            df["BreakoutMarginHigh6Pips"] = (close - rolling_high_6) / pip_size
        if wants("breakout_margin_low_6_pips"):
            df["BreakoutMarginLow6Pips"] = (rolling_low_6 - close) / pip_size
        if wants("higher_high_flag"):
            df["HigherHighFlag"] = (high > prior_high).astype(float)
        if wants("higher_low_flag"):
            df["HigherLowFlag"] = (low > prior_low).astype(float)
        if wants("lower_high_flag"):
            df["LowerHighFlag"] = (high < prior_high).astype(float)
        if wants("lower_low_flag"):
            df["LowerLowFlag"] = (low < prior_low).astype(float)

        structure_score_base = (
            (high > prior_high).astype(float)
            + (low > prior_low).astype(float)
            - (high < prior_high).astype(float)
            - (low < prior_low).astype(float)
        )
        if wants("structure_score_3"):
            df["StructureScore3"] = structure_score_base.rolling(3).sum() / 6.0
        if wants("structure_score_6"):
            df["StructureScore6"] = structure_score_base.rolling(6).sum() / 12.0
        if wants("range_vs_avg_6"):
            df["RangeVsAvg6"] = (bar_range / pip_size) / ((bar_range.shift(1).rolling(6).mean()) / pip_size)
        if wants("pullback_from_recent_high_6_over_atr"):
            df["PullbackFromRecentHigh6OverATR"] = ((rolling_high_6 - close) / pip_size) / atr_pips
        if wants("bounce_from_recent_low_6_over_atr"):
            df["BounceFromRecentLow6OverATR"] = ((close - rolling_low_6) / pip_size) / atr_pips

        sma_20 = pd.to_numeric(df["SMA_20"], errors="coerce") if "SMA_20" in df.columns else close.rolling(20).mean()
        sma_50 = pd.to_numeric(df["SMA_50"], errors="coerce") if "SMA_50" in df.columns else close.rolling(50).mean()
        if wants("distance_to_sma20_over_atr"):
            df["DistToSMA20OverATR"] = ((close - sma_20).abs() / pip_size) / atr_pips
        if wants("distance_to_sma50_over_atr"):
            df["DistToSMA50OverATR"] = ((close - sma_50).abs() / pip_size) / atr_pips
        if wants("ema12_26_spread_over_atr"):
            ema_12 = pd.to_numeric(df["EMA_12"], errors="coerce") if "EMA_12" in df.columns else close.ewm(span=12, adjust=False).mean()
            ema_26 = pd.to_numeric(df["EMA_26"], errors="coerce") if "EMA_26" in df.columns else close.ewm(span=26, adjust=False).mean()
            df["EMA1226SpreadOverATR"] = ((ema_12 - ema_26) / pip_size) / atr_pips

        if wants("realized_range_3"):
            df["RealizedRange3Pips"] = (high.rolling(3).max() - low.rolling(3).min()) / pip_size
        if wants("realized_range_6"):
            df["RealizedRange6Pips"] = (high.rolling(6).max() - low.rolling(6).min()) / pip_size

        dt_index = df.index if isinstance(df.index, pd.DatetimeIndex) else None
        if dt_index is None and "Date" in df.columns:
            dt_candidate = pd.to_datetime(df["Date"], errors="coerce")
            if dt_candidate.notna().any():
                dt_index = pd.DatetimeIndex(dt_candidate)

        if dt_index is not None:
            hours = pd.Series(dt_index.hour, index=df.index, dtype=float)
            radians = 2.0 * np.pi * hours / 24.0
            if wants("hour_sin"):
                df["HourSin"] = np.sin(radians)
            if wants("hour_cos"):
                df["HourCos"] = np.cos(radians)
            if wants("session_london"):
                df["SessionLondon"] = hours.between(7, 11, inclusive="both").astype(float)
            if wants("session_newyork"):
                df["SessionNewYork"] = hours.between(12, 16, inclusive="both").astype(float)
            if wants("session_overlap"):
                df["SessionOverlap"] = hours.between(12, 15, inclusive="both").astype(float)

        return df

    @staticmethod
    def add_barrier_targets(
        df: pd.DataFrame,
        *,
        price_col: str = "Close",
        high_col: str = "High",
        low_col: str = "Low",
        pip_size: float = 0.0001,
        barrier_pips: float = 3.0,
        horizon_bars: int = 3,
    ) -> pd.DataFrame:
        """
        Agrega targets tipo first-touch / triple-barrier simplificado.

        Genera columnas auxiliares y un target numérico compatible con el flujo actual:
        - BarrierDir_{pips}p_{bars}b: {-1, 0, +1}
        - BarrierReturn_{pips}p_{bars}b: retorno equivalente a tocar +/- barrera
        - BarrierMovePips_{pips}p_{bars}b: movimiento objetivo en pips {-pips, 0, +pips}
        - BarrierBarsToTouch_{pips}p_{bars}b: velas hasta el toque
        - BarrierAmbiguous_{pips}p_{bars}b: 1 si una vela tocó ambas barreras
        - MFEPips_{pips}p_{bars}b / MAEPips_{pips}p_{bars}b
        """
        df = df.copy()

        if price_col not in df.columns or high_col not in df.columns or low_col not in df.columns:
            return df

        pip_size = max(float(pip_size or 0.0001), 1e-12)
        barrier_pips = float(barrier_pips or 0.0)
        horizon_bars = int(horizon_bars or 0)
        if barrier_pips <= 0 or horizon_bars <= 0:
            return df

        suffix = f"_{int(barrier_pips)}p_{int(horizon_bars)}b"
        dir_col = f"BarrierDir{suffix}"
        ret_col = f"BarrierReturn{suffix}"
        move_col = f"BarrierMovePips{suffix}"
        touch_col = f"BarrierBarsToTouch{suffix}"
        ambiguous_col = f"BarrierAmbiguous{suffix}"
        mfe_col = f"MFEPips{suffix}"
        mae_col = f"MAEPips{suffix}"

        close = pd.to_numeric(df[price_col], errors="coerce")
        high = pd.to_numeric(df[high_col], errors="coerce")
        low = pd.to_numeric(df[low_col], errors="coerce")
        barrier_abs = barrier_pips * pip_size

        barrier_dir: list[float] = []
        barrier_return: list[float] = []
        barrier_move_pips: list[float] = []
        barrier_bars_to_touch: list[float] = []
        barrier_ambiguous: list[float] = []
        mfe_pips: list[float] = []
        mae_pips: list[float] = []

        n = len(df)
        for i in range(n):
            entry = close.iloc[i]
            if pd.isna(entry):
                barrier_dir.append(np.nan)
                barrier_return.append(np.nan)
                barrier_move_pips.append(np.nan)
                barrier_bars_to_touch.append(np.nan)
                barrier_ambiguous.append(np.nan)
                mfe_pips.append(np.nan)
                mae_pips.append(np.nan)
                continue

            future_end = min(i + horizon_bars, n - 1)
            if future_end <= i:
                barrier_dir.append(np.nan)
                barrier_return.append(np.nan)
                barrier_move_pips.append(np.nan)
                barrier_bars_to_touch.append(np.nan)
                barrier_ambiguous.append(np.nan)
                mfe_pips.append(np.nan)
                mae_pips.append(np.nan)
                continue

            upper = entry + barrier_abs
            lower = entry - barrier_abs

            future_high = high.iloc[i + 1 : future_end + 1]
            future_low = low.iloc[i + 1 : future_end + 1]
            if future_high.empty or future_low.empty:
                barrier_dir.append(np.nan)
                barrier_return.append(np.nan)
                barrier_move_pips.append(np.nan)
                barrier_bars_to_touch.append(np.nan)
                barrier_ambiguous.append(np.nan)
                mfe_pips.append(np.nan)
                mae_pips.append(np.nan)
                continue

            max_favorable = ((future_high.max() - entry) / pip_size) if future_high.notna().any() else np.nan
            max_adverse = ((future_low.min() - entry) / pip_size) if future_low.notna().any() else np.nan
            mfe_pips.append(float(max_favorable) if pd.notna(max_favorable) else np.nan)
            mae_pips.append(float(max_adverse) if pd.notna(max_adverse) else np.nan)

            label = 0.0
            label_return = 0.0
            label_move_pips = 0.0
            bars_to_touch = np.nan
            ambiguous = 0.0

            for step_idx, (bar_high, bar_low) in enumerate(
                zip(future_high.tolist(), future_low.tolist()),
                start=1,
            ):
                if pd.isna(bar_high) or pd.isna(bar_low):
                    continue

                upper_hit = bar_high >= upper
                lower_hit = bar_low <= lower

                if upper_hit and lower_hit:
                    ambiguous = 1.0
                    label = 0.0
                    label_return = 0.0
                    label_move_pips = 0.0
                    bars_to_touch = float(step_idx)
                    break
                if upper_hit:
                    label = 1.0
                    label_return = barrier_abs / max(float(entry), 1e-12)
                    label_move_pips = barrier_pips
                    bars_to_touch = float(step_idx)
                    break
                if lower_hit:
                    label = -1.0
                    label_return = -barrier_abs / max(float(entry), 1e-12)
                    label_move_pips = -barrier_pips
                    bars_to_touch = float(step_idx)
                    break

            barrier_dir.append(label)
            barrier_return.append(label_return)
            barrier_move_pips.append(label_move_pips)
            barrier_bars_to_touch.append(bars_to_touch)
            barrier_ambiguous.append(ambiguous)

        df[dir_col] = barrier_dir
        df[ret_col] = barrier_return
        df[move_col] = barrier_move_pips
        df[touch_col] = barrier_bars_to_touch
        df[ambiguous_col] = barrier_ambiguous
        df[mfe_col] = mfe_pips
        df[mae_col] = mae_pips

        return df
    
    @staticmethod
    def add_lag_features(
        df: pd.DataFrame,
        col: str,
        lags: list[int]
    ) -> pd.DataFrame:
        """Agrega versiones rezagadas de una columna"""
        df = df.copy()
        
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        
        return df


# Ejemplo de uso
if __name__ == "__main__":
    # Crear datos de ejemplo
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "Open": np.random.randn(100).cumsum() + 100,
        "High": np.random.randn(100).cumsum() + 102,
        "Low": np.random.randn(100).cumsum() + 98,
        "Close": np.random.randn(100).cumsum() + 100,
        "Volume": np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Introducir problemas artificiales
    df.loc[df.index[10], "Close"] = np.nan  # Missing value
    df.loc[df.index[20], "High"] = df.loc[df.index[20], "Low"] - 1  # OHLC inválido
    df.loc[df.index[30], "Close"] = df["Close"].mean() * 5  # Outlier
    
    print("DataFrame original:")
    print(df.head(10))
    print(f"\nNaN count: {df.isnull().sum().sum()}")
    
    # Limpiar
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df, price_col="Close")
    
    print("\n" + cleaner.get_report())
    
    # Agregar features
    engineer = FeatureEngineer()
    df_features = engineer.add_returns(df_clean)
    df_features = engineer.add_technical_indicators(df_features)
    
    print("\nColumnas finales:")
    print(df_features.columns.tolist())
    print("\nDataFrame con features:")
    print(df_features.tail())
