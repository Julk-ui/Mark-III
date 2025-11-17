# eda/exploratory_analysis.py
"""
Módulo de Análisis Exploratorio de Datos (EDA)
Genera estadísticas, gráficos y reportes para series financieras
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose


# Configurar estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


class ExploratoryAnalysis:
    """
    Generador de análisis exploratorio completo para series financieras
    """
    
    def __init__(self, output_dir: str = "outputs/eda"):
        """
        Args:
            output_dir: Directorio para guardar gráficos y reportes
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats_report: Dict[str, Any] = {}
    
    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str,
        price_col: str = "Close"
    ) -> Dict[str, Any]:
        """
        Ejecuta análisis exploratorio completo
        
        Args:
            df: DataFrame con datos OHLCV
            symbol: Nombre del símbolo (para títulos y nombres de archivo)
            price_col: Columna principal de precio
            
        Returns:
            Diccionario con todas las estadísticas y rutas de gráficos
        """
        print(f"\n{'='*60}")
        print(f"ANÁLISIS EXPLORATORIO: {symbol}")
        print(f"{'='*60}\n")
        
        # 1. Estadísticas descriptivas
        desc_stats = self._descriptive_statistics(df, price_col)
        
        # 2. Análisis de retornos
        returns_stats = self._returns_analysis(df, price_col)
        
        # 3. Tests de estacionariedad
        stationarity = self._stationarity_tests(df[price_col])
        
        # 4. Análisis de autocorrelación
        autocorr = self._autocorrelation_analysis(df, price_col)
        
        # 5. Descomposición de serie temporal
        decomp = self._seasonal_decomposition(df, price_col)
        
        # 6. Generación de gráficos
        plots = self._generate_plots(df, symbol, price_col)
        
        # Consolidar reporte
        self.stats_report = {
            "symbol": symbol,
            "period": {
                "start": df.index.min(),
                "end": df.index.max(),
                "n_observations": len(df)
            },
            "descriptive_stats": desc_stats,
            "returns_stats": returns_stats,
            "stationarity": stationarity,
            "autocorrelation": autocorr,
            "decomposition": decomp,
            "plots": plots
        }
        
        # Imprimir resumen
        self._print_summary()
        
        # Guardar reporte en Excel
        self._save_excel_report(symbol)
        
        return self.stats_report
    
    def _descriptive_statistics(
        self,
        df: pd.DataFrame,
        price_col: str
    ) -> Dict[str, Any]:
        """Calcula estadísticas descriptivas del precio"""
        price = df[price_col].dropna()
        
        stats_dict = {
            "count": int(price.count()),
            "mean": float(price.mean()),
            "std": float(price.std()),
            "min": float(price.min()),
            "25%": float(price.quantile(0.25)),
            "50%": float(price.median()),
            "75%": float(price.quantile(0.75)),
            "max": float(price.max()),
            "range": float(price.max() - price.min()),
            "cv": float(price.std() / price.mean()),  # Coeficiente de variación
        }
        
        print("📊 ESTADÍSTICAS DESCRIPTIVAS (Precio)")
        for key, value in stats_dict.items():
            print(f"  {key:>10s}: {value:,.4f}")
        
        return stats_dict
    
    def _returns_analysis(
        self,
        df: pd.DataFrame,
        price_col: str
    ) -> Dict[str, Any]:
        """Analiza distribución de retornos"""
        # Calcular retornos
        returns = df[price_col].pct_change().dropna()
        log_returns = np.log(df[price_col] / df[price_col].shift(1)).dropna()
        
        # Estadísticas
        stats_dict = {
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std()),
            "mean_log_return": float(log_returns.mean()),
            "std_log_return": float(log_returns.std()),
            "skewness": float(log_returns.skew()),
            "kurtosis": float(log_returns.kurtosis()),
            "min_return": float(returns.min()),
            "max_return": float(returns.max()),
            "sharpe_ratio": float(
                returns.mean() / returns.std() * np.sqrt(252)
            ) if returns.std() > 0 else 0.0,
        }
        
        # Test de normalidad (Jarque-Bera)
        jb_stat, jb_pval = stats.jarque_bera(log_returns)
        stats_dict["jb_statistic"] = float(jb_stat)
        stats_dict["jb_pvalue"] = float(jb_pval)
        stats_dict["is_normal"] = jb_pval > 0.05
        
        # VaR y CVaR (95%)
        stats_dict["var_95"] = float(np.percentile(returns, 5))
        stats_dict["cvar_95"] = float(
            returns[returns <= np.percentile(returns, 5)].mean()
        )
        
        print("\n📈 ANÁLISIS DE RETORNOS")
        print(f"  Media (diaria):     {stats_dict['mean_return']:,.6f}")
        print(f"  Volatilidad:        {stats_dict['std_return']:,.6f}")
        print(f"  Sharpe Ratio:       {stats_dict['sharpe_ratio']:,.2f}")
        print(f"  Asimetría:          {stats_dict['skewness']:,.2f}")
        print(f"  Curtosis:           {stats_dict['kurtosis']:,.2f}")
        print(f"  Normalidad (JB):    {'✓ Normal' if stats_dict['is_normal'] else '✗ No normal'} (p={jb_pval:.4f})")
        print(f"  VaR 95%:            {stats_dict['var_95']:,.4%}")
        print(f"  CVaR 95%:           {stats_dict['cvar_95']:,.4%}")
        
        return stats_dict
    
    def _stationarity_tests(self, series: pd.Series) -> Dict[str, Any]:
        """Ejecuta tests de estacionariedad"""
        series_clean = series.dropna()
        
        # Augmented Dickey-Fuller
        adf_result = adfuller(series_clean, autolag="AIC")
        
        # KPSS
        kpss_result = kpss(series_clean, regression="c", nlags="auto")
        
        results = {
            "adf": {
                "statistic": float(adf_result[0]),
                "pvalue": float(adf_result[1]),
                "critical_values": {k: float(v) for k, v in adf_result[4].items()},
                "is_stationary": adf_result[1] < 0.05
            },
            "kpss": {
                "statistic": float(kpss_result[0]),
                "pvalue": float(kpss_result[1]),
                "critical_values": {k: float(v) for k, v in kpss_result[3].items()},
                "is_stationary": kpss_result[1] > 0.05
            }
        }
        
        print("\n🔍 TESTS DE ESTACIONARIEDAD")
        print(f"  ADF:")
        print(f"    Estadístico:  {results['adf']['statistic']:,.4f}")
        print(f"    P-value:      {results['adf']['pvalue']:,.4f}")
        print(f"    Resultado:    {'✓ Estacionaria' if results['adf']['is_stationary'] else '✗ No estacionaria'}")
        print(f"  KPSS:")
        print(f"    Estadístico:  {results['kpss']['statistic']:,.4f}")
        print(f"    P-value:      {results['kpss']['pvalue']:,.4f}")
        print(f"    Resultado:    {'✓ Estacionaria' if results['kpss']['is_stationary'] else '✗ No estacionaria'}")
        
        return results
    
    def _autocorrelation_analysis(
        self,
        df: pd.DataFrame,
        price_col: str,
        max_lag: int = 40
    ) -> Dict[str, Any]:
        """Analiza autocorrelación de retornos"""
        log_returns = np.log(df[price_col] / df[price_col].shift(1)).dropna()
        
        # Calcular autocorrelación
        acf_values = [log_returns.autocorr(lag=i) for i in range(1, max_lag + 1)]
        
        # Test de Ljung-Box
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(log_returns, lags=[10, 20], return_df=True)
        
        results = {
            "acf_values": acf_values,
            "ljung_box": {
                "lag_10": {
                    "statistic": float(lb_test.loc[10, "lb_stat"]),
                    "pvalue": float(lb_test.loc[10, "lb_pvalue"])
                },
                "lag_20": {
                    "statistic": float(lb_test.loc[20, "lb_stat"]),
                    "pvalue": float(lb_test.loc[20, "lb_pvalue"])
                }
            }
        }
        
        print("\n🔄 AUTOCORRELACIÓN")
        print(f"  Ljung-Box (lag 10): p={results['ljung_box']['lag_10']['pvalue']:.4f}")
        print(f"  Ljung-Box (lag 20): p={results['ljung_box']['lag_20']['pvalue']:.4f}")
        
        return results
    
    def _seasonal_decomposition(
        self,
        df: pd.DataFrame,
        price_col: str
    ) -> Optional[Dict[str, Any]]:
        """Descomposición estacional de la serie"""
        try:
            # Determinar periodo basado en frecuencia
            freq = pd.infer_freq(df.index)
            if freq is None:
                period = 7  # Default semanal
            elif freq.startswith("D"):
                period = 7  # Semanal para datos diarios
            elif freq.startswith("H"):
                period = 24  # Diario para datos horarios
            else:
                period = 7
            
            # Ejecutar descomposición
            decomposition = seasonal_decompose(
                df[price_col].dropna(),
                model="multiplicative",
                period=period,
                extrapolate_trend="freq"
            )
            
            return {
                "period": period,
                "trend_strength": float(
                    1 - (decomposition.resid.var() / 
                         (decomposition.trend + decomposition.resid).var())
                ),
                "seasonal_strength": float(
                    1 - (decomposition.resid.var() / 
                         (decomposition.seasonal + decomposition.resid).var())
                )
            }
        except Exception as e:
            print(f"  ⚠️  Descomposición fallida: {e}")
            return None
    
    def _generate_plots(
        self,
        df: pd.DataFrame,
        symbol: str,
        price_col: str
    ) -> Dict[str, str]:
        """Genera todos los gráficos del análisis"""
        plots = {}
        
        # 1. Serie de tiempo
        plots["price_series"] = self._plot_price_series(df, symbol, price_col)
        
        # 2. Distribución de retornos
        plots["returns_dist"] = self._plot_returns_distribution(df, symbol, price_col)
        
        # 3. QQ Plot
        plots["qq_plot"] = self._plot_qq(df, symbol, price_col)
        
        # 4. ACF y PACF
        plots["acf"], plots["pacf"] = self._plot_acf_pacf(df, symbol, price_col)
        
        # 5. Volatilidad rolling
        plots["rolling_vol"] = self._plot_rolling_volatility(df, symbol, price_col)
        
        # 6. Descomposición
        plots["decomposition"] = self._plot_decomposition(df, symbol, price_col)
        
        print(f"\n📁 Gráficos guardados en: {self.output_dir}")
        
        return plots
    
    def _plot_price_series(self, df, symbol, price_col) -> str:
        """Gráfico de serie de tiempo con volumen"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        # Precio
        ax1.plot(df.index, df[price_col], linewidth=1.5, color='#2E86AB')
        ax1.set_title(f"{symbol} - Serie de Precios", fontsize=14, weight='bold')
        ax1.set_ylabel("Precio", fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Volumen (si existe)
        if "Volume" in df.columns:
            ax2.bar(df.index, df["Volume"], color='#A23B72', alpha=0.6, width=1)
            ax2.set_ylabel("Volumen", fontsize=11)
            ax2.set_xlabel("Fecha", fontsize=11)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = self.output_dir / f"{symbol}_01_price_series.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def _plot_returns_distribution(self, df, symbol, price_col) -> str:
        """Histograma y KDE de retornos"""
        returns = df[price_col].pct_change().dropna()
        log_returns = np.log(df[price_col] / df[price_col].shift(1)).dropna()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Retornos simples
        axes[0].hist(returns, bins=50, density=True, alpha=0.6, 
                     color='#2E86AB', edgecolor='black')
        returns.plot(kind='kde', ax=axes[0], color='#A23B72', linewidth=2)
        axes[0].axvline(returns.mean(), color='red', linestyle='--', 
                       label=f'Media: {returns.mean():.4f}')
        axes[0].set_title("Distribución de Retornos Simples", fontsize=12, weight='bold')
        axes[0].set_xlabel("Retorno")
        axes[0].set_ylabel("Densidad")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Log-retornos
        axes[1].hist(log_returns, bins=50, density=True, alpha=0.6,
                     color='#2E86AB', edgecolor='black')
        log_returns.plot(kind='kde', ax=axes[1], color='#A23B72', linewidth=2)
        axes[1].axvline(log_returns.mean(), color='red', linestyle='--',
                       label=f'Media: {log_returns.mean():.4f}')
        axes[1].set_title("Distribución de Log-Retornos", fontsize=12, weight='bold')
        axes[1].set_xlabel("Log-Retorno")
        axes[1].set_ylabel("Densidad")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = self.output_dir / f"{symbol}_02_returns_distribution.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def _plot_qq(self, df, symbol, price_col) -> str:
        """QQ Plot para verificar normalidad"""
        log_returns = np.log(df[price_col] / df[price_col].shift(1)).dropna()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        stats.probplot(log_returns, dist="norm", plot=ax)
        ax.set_title(f"{symbol} - QQ Plot (Log-Retornos vs Normal)", 
                     fontsize=12, weight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = self.output_dir / f"{symbol}_03_qq_plot.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def _plot_acf_pacf(self, df, symbol, price_col, lags=40) -> Tuple[str, str]:
        """ACF y PACF de log-retornos"""
        log_returns = np.log(df[price_col] / df[price_col].shift(1)).dropna()
        
        # ACF
        fig, ax = plt.subplots(figsize=(12, 4))
        plot_acf(log_returns, lags=lags, ax=ax, color='#2E86AB')
        ax.set_title(f"{symbol} - Autocorrelación (ACF)", fontsize=12, weight='bold')
        plt.tight_layout()
        path_acf = self.output_dir / f"{symbol}_04_acf.png"
        plt.savefig(path_acf, dpi=150, bbox_inches='tight')
        plt.close()
        
        # PACF
        fig, ax = plt.subplots(figsize=(12, 4))
        plot_pacf(log_returns, lags=lags, ax=ax, method='ywm', color='#A23B72')
        ax.set_title(f"{symbol} - Autocorrelación Parcial (PACF)", 
                     fontsize=12, weight='bold')
        plt.tight_layout()
        path_pacf = self.output_dir / f"{symbol}_05_pacf.png"
        plt.savefig(path_pacf, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(path_acf), str(path_pacf)
    
    def _plot_rolling_volatility(self, df, symbol, price_col) -> str:
        """Volatilidad móvil"""
        returns = df[price_col].pct_change()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for window in [20, 60, 120]:
            vol = returns.rolling(window).std() * np.sqrt(252)
            ax.plot(df.index, vol, label=f'{window} días', linewidth=1.5)
        
        ax.set_title(f"{symbol} - Volatilidad Móvil (Anualizada)", 
                     fontsize=12, weight='bold')
        ax.set_ylabel("Volatilidad", fontsize=11)
        ax.set_xlabel("Fecha", fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = self.output_dir / f"{symbol}_06_rolling_volatility.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def _plot_decomposition(self, df, symbol, price_col) -> str:
        """Descomposición estacional"""
        try:
            decomposition = seasonal_decompose(
                df[price_col].dropna(),
                model="multiplicative",
                period=7,
                extrapolate_trend="freq"
            )
            
            fig, axes = plt.subplots(4, 1, figsize=(14, 10))
            
            decomposition.observed.plot(ax=axes[0], color='#2E86AB')
            axes[0].set_ylabel("Observado")
            axes[0].set_title(f"{symbol} - Descomposición Estacional", 
                             fontsize=12, weight='bold')
            
            decomposition.trend.plot(ax=axes[1], color='#A23B72')
            axes[1].set_ylabel("Tendencia")
            
            decomposition.seasonal.plot(ax=axes[2], color='#F18F01')
            axes[2].set_ylabel("Estacional")
            
            decomposition.resid.plot(ax=axes[3], color='#C73E1D')
            axes[3].set_ylabel("Residuos")
            axes[3].set_xlabel("Fecha")
            
            for ax in axes:
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            path = self.output_dir / f"{symbol}_07_decomposition.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(path)
        except Exception as e:
            print(f"  ⚠️  Gráfico de descomposición omitido: {e}")
            return ""
    
    def _print_summary(self) -> None:
        """Imprime resumen ejecutivo"""
        print(f"\n{'='*60}")
        print("RESUMEN EJECUTIVO")
        print(f"{'='*60}")
        
        period = self.stats_report["period"]
        print(f"Periodo: {period['start'].date()} → {period['end'].date()}")
        print(f"Observaciones: {period['n_observations']:,}")
        
        ret = self.stats_report["returns_stats"]
        print(f"\nRentabilidad media diaria: {ret['mean_return']:.4%}")
        print(f"Volatilidad diaria: {ret['std_return']:.4%}")
        print(f"Sharpe Ratio (anualizado): {ret['sharpe_ratio']:.2f}")
        
        adf = self.stats_report["stationarity"]["adf"]
        print(f"\nEstacionariedad (ADF): {'✓ Sí' if adf['is_stationary'] else '✗ No'}")
        
        print(f"\n{'='*60}\n")
    
    def _save_excel_report(self, symbol: str) -> None:
        """Guarda reporte completo en Excel"""
        excel_path = self.output_dir / f"{symbol}_EDA_report.xlsx"
        
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # Resumen general
            summary = pd.DataFrame([{
                "Symbol": self.stats_report["symbol"],
                "Start Date": self.stats_report["period"]["start"],
                "End Date": self.stats_report["period"]["end"],
                "N Observations": self.stats_report["period"]["n_observations"]
            }])
            summary.to_excel(writer, sheet_name="Summary", index=False)
            
            # Estadísticas descriptivas
            desc = pd.DataFrame([self.stats_report["descriptive_stats"]])
            desc.to_excel(writer, sheet_name="Descriptive Stats", index=False)
            
            # Estadísticas de retornos
            ret = pd.DataFrame([self.stats_report["returns_stats"]])
            ret.to_excel(writer, sheet_name="Returns Stats", index=False)
            
            # Tests de estacionariedad
            stat_data = []
            for test_name, test_data in self.stats_report["stationarity"].items():
                stat_data.append({
                    "Test": test_name.upper(),
                    "Statistic": test_data["statistic"],
                    "P-Value": test_data["pvalue"],
                    "Is Stationary": test_data["is_stationary"]
                })
            stat_df = pd.DataFrame(stat_data)
            stat_df.to_excel(writer, sheet_name="Stationarity", index=False)
        
        print(f"📄 Reporte Excel guardado: {excel_path}")


# Ejemplo de uso
if __name__ == "__main__":
    # Crear datos de ejemplo
    dates = pd.date_range("2020-01-01", periods=1000, freq="D")
    df = pd.DataFrame({
        "Open": np.random.randn(1000).cumsum() + 100,
        "High": np.random.randn(1000).cumsum() + 102,
        "Low": np.random.randn(1000).cumsum() + 98,
        "Close": np.random.randn(1000).cumsum() + 100,
        "Volume": np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # Ejecutar análisis
    eda = ExploratoryAnalysis(output_dir="outputs/eda")
    report = eda.analyze(df, symbol="EURUSD", price_col="Close")
    
    print("\n✅ Análisis completado")
    print(f"📊 Gráficos generados: {len(report['plots'])}")