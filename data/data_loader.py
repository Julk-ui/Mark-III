# data/data_loader.py
"""
Módulo de carga de datos desde MetaTrader 5
Gestiona la conexión, descarga y cache de datos históricos
"""

from __future__ import annotations
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from conexion.easy_Trading import Basic_funcs


class DataLoader:
    """
    Gestor de carga de datos desde MT5 con soporte de cache
    """
    
    def __init__(self, mt5_config: Dict[str, Any], cache_dir: str = "data/cache"):
        """
        Args:
            mt5_config: Diccionario con login, password, server, path
            cache_dir: Directorio para almacenar cache de datos
        """
        self.mt5_config = mt5_config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mt5_client: Optional[Basic_funcs] = None
        
    def connect(self) -> None:
        """Establece conexión con MT5"""
        if self.mt5_client is None:
            self.mt5_client = Basic_funcs(
                login=self.mt5_config["login"],
                password=self.mt5_config["password"],
                server=self.mt5_config["server"],
                path=self.mt5_config.get("path")
            )
            print("✅ Conexión establecida con MetaTrader 5")
    
    def disconnect(self) -> None:
        """Cierra conexión con MT5"""
        if self.mt5_client is not None:
            del self.mt5_client
            self.mt5_client = None
            print("🛑 Conexión cerrada")
    
    def load_data(
        self,
        symbol: str,
        timeframe: str,
        n_bars: int,
        use_cache: bool = True,
        cache_expiry_hours: int = 24
    ) -> pd.DataFrame:
        """
        Carga datos históricos con soporte de cache
        
        Args:
            symbol: Símbolo a descargar (ej: EURUSD)
            timeframe: Temporalidad (D1, H1, M15, etc)
            n_bars: Número de barras a descargar
            use_cache: Usar datos en cache si existen
            cache_expiry_hours: Horas antes de considerar cache obsoleto
            
        Returns:
            DataFrame con columnas: Open, High, Low, Close, Volume
        """
        cache_file = self._get_cache_path(symbol, timeframe, n_bars)
        
        # Intentar cargar desde cache
        if use_cache and cache_file.exists():
            if self._is_cache_valid(cache_file, cache_expiry_hours):
                print(f"📦 Cargando {symbol} desde cache...")
                return self._load_from_cache(cache_file)
        
        # Descargar datos frescos
        print(f"📥 Descargando {symbol} ({timeframe}) - {n_bars} barras...")
        self.connect()
        
        df = self.mt5_client.get_data_for_bt(
            timeframe=timeframe,
            symbol=symbol,
            count=n_bars
        )
        
        # Guardar en cache
        self._save_to_cache(df, cache_file)
        
        return df
    
    def load_multiple_symbols(
        self,
        symbols: list[str],
        timeframe: str,
        n_bars: int,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Carga múltiples símbolos de manera eficiente
        
        Returns:
            Diccionario {symbol: DataFrame}
        """
        data = {}
        self.connect()
        
        for symbol in symbols:
            try:
                df = self.load_data(symbol, timeframe, n_bars, use_cache)
                data[symbol] = df
                print(f"✅ {symbol}: {len(df)} barras cargadas")
            except Exception as e:
                print(f"❌ Error cargando {symbol}: {e}")
                
        return data
    
    def _get_cache_path(self, symbol: str, timeframe: str, n_bars: int) -> Path:
        """Genera ruta del archivo de cache"""
        filename = f"{symbol}_{timeframe}_{n_bars}.pkl"
        return self.cache_dir / filename
    
    def _is_cache_valid(self, cache_file: Path, expiry_hours: int) -> bool:
        """Verifica si el cache aún es válido"""
        if not cache_file.exists():
            return False
        
        file_age_hours = (
            datetime.now().timestamp() - cache_file.stat().st_mtime
        ) / 3600
        
        return file_age_hours < expiry_hours
    
    def _load_from_cache(self, cache_file: Path) -> pd.DataFrame:
        """Carga datos desde archivo pickle"""
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    def _save_to_cache(self, df: pd.DataFrame, cache_file: Path) -> None:
        """Guarda datos en cache"""
        with open(cache_file, "wb") as f:
            pickle.dump(df, f)
        print(f"💾 Datos guardados en cache: {cache_file.name}")
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información del símbolo (spread, tick size, etc)
        
        Returns:
            Diccionario con información del símbolo o None si falla
        """
        self.connect()
        
        try:
            info = self.mt5_client.mt5.symbol_select(symbol, True)
            info = self.mt5_client.mt5.symbol_info(symbol)
            
            if info is None:
                return None
            
            return {
                "symbol": symbol,
                "description": info.description,
                "point": info.point,
                "digits": info.digits,
                "spread": info.spread,
                "trade_contract_size": info.trade_contract_size,
                "volume_min": info.volume_min,
                "volume_max": info.volume_max,
                "volume_step": info.volume_step,
            }
        except Exception as e:
            print(f"❌ Error obteniendo info de {symbol}: {e}")
            return None
    
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Limpia archivos de cache
        
        Args:
            symbol: Si se especifica, solo limpia ese símbolo. 
                   Si es None, limpia todo el cache
        """
        if symbol:
            pattern = f"{symbol}_*.pkl"
        else:
            pattern = "*.pkl"
        
        deleted_count = 0
        for file in self.cache_dir.glob(pattern):
            file.unlink()
            deleted_count += 1
        
        print(f"🗑️  {deleted_count} archivo(s) de cache eliminado(s)")


class DataValidator:
    """
    Validador de calidad de datos cargados
    """
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida estructura y calidad del DataFrame
        
        Returns:
            Diccionario con resultado de validaciones
        """
        issues = []
        warnings = []
        
        # Validar estructura básica
        required_cols = ["Open", "High", "Low", "Close"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            issues.append(f"Columnas faltantes: {missing_cols}")
        
        # Validar índice datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append("El índice no es DatetimeIndex")
        
        # Validar valores faltantes
        missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
        for col, pct in missing_pct.items():
            if pct > 0:
                warnings.append(f"{col}: {pct:.2f}% valores faltantes")
        
        # Validar precios negativos
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (df[col] < 0).any():
                issues.append(f"{col} contiene valores negativos")
        
        # Validar High >= Low
        if "High" in df.columns and "Low" in df.columns:
            invalid_hl = (df["High"] < df["Low"]).sum()
            if invalid_hl > 0:
                issues.append(f"{invalid_hl} filas con High < Low")
        
        # Validar duplicados en índice
        if df.index.duplicated().any():
            n_dups = df.index.duplicated().sum()
            warnings.append(f"{n_dups} timestamps duplicados")
        
        return {
            "is_valid": len(issues) == 0,
            "n_rows": len(df),
            "date_range": (df.index.min(), df.index.max()),
            "issues": issues,
            "warnings": warnings,
            "missing_values": missing_pct
        }
    
    @staticmethod
    def detect_outliers(
        series: pd.Series,
        method: str = "iqr",
        threshold: float = 3.0
    ) -> pd.Series:
        """
        Detecta outliers en una serie
        
        Args:
            series: Serie a analizar
            method: "iqr" (rango intercuartil) o "zscore"
            threshold: Umbral para considerar outlier
            
        Returns:
            Serie booleana indicando outliers
        """
        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            return (series < lower) | (series > upper)
        
        elif method == "zscore":
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > threshold
        
        else:
            raise ValueError(f"Método no soportado: {method}")


# Ejemplo de uso
if __name__ == "__main__":
    # Configuración de ejemplo
    config = {
        "login": 68238343,
        "password": "Colombia123*",
        "server": "RoboForex-Pro",
        "path": r"C:\Program Files\RoboForex - MetaTrader 5\terminal64.exe"
    }
    
    # Inicializar loader
    loader = DataLoader(mt5_config=config)
    
    try:
        # Cargar datos
        df = loader.load_data(
            symbol="EURUSD",
            timeframe="D1",
            n_bars=1000,
            use_cache=True
        )
        
        # Validar datos
        validator = DataValidator()
        validation = validator.validate_dataframe(df)
        
        print("\n" + "="*60)
        print("RESULTADO DE VALIDACIÓN")
        print("="*60)
        print(f"✓ Válido: {validation['is_valid']}")
        print(f"✓ Filas: {validation['n_rows']}")
        print(f"✓ Rango: {validation['date_range'][0]} → {validation['date_range'][1]}")
        
        if validation['issues']:
            print("\n⚠️  PROBLEMAS:")
            for issue in validation['issues']:
                print(f"  • {issue}")
        
        if validation['warnings']:
            print("\n⚠️  ADVERTENCIAS:")
            for warning in validation['warnings']:
                print(f"  • {warning}")
        
        print("\n" + df.head().to_string())
        
    finally:
        loader.disconnect()