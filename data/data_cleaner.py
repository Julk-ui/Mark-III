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
