# utils/metrics.py
from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats  # ya usas scipy en data_cleaner, así que entra en el mismo requirements

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_error(y_true, y_pred)

def calculate_hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el Hit Rate (% de aciertos de dirección).
    Compara el signo de la predicción con el signo del valor real.
    """
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)

    # ignorar casos donde la predicción es 0 (no posición)
    mask = pred_sign != 0
    if mask.sum() == 0:
        return 0.0

    hits = (true_sign[mask] == pred_sign[mask]).mean()
    return float(hits * 100.0)

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Accuracy direccional en escala 0-100 (igual que HitRate, pero separado por claridad).
    """
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    return np.mean(true_sign == pred_sign) * 100

def diebold_mariano(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_benchmark: np.ndarray,
    power: int = 2
) -> tuple[float, float]:
    """
    Test Diebold-Mariano (versión simple, horizonte h=1).

    Compara el modelo vs un benchmark:
    - y_true: valores reales
    - y_pred_model: predicciones de tu modelo
    - y_pred_benchmark: predicciones del modelo benchmark
    - power: 1 -> pérdida tipo MAE; 2 -> tipo MSE (por defecto)

    Devuelve: (dm_stat, p_value) usando aproximación normal estándar.
    """
    import math

    y_true = np.asarray(y_true, dtype=float)
    y_pred_model = np.asarray(y_pred_model, dtype=float)
    y_pred_benchmark = np.asarray(y_pred_benchmark, dtype=float)

    # Aseguramos misma longitud
    n = min(len(y_true), len(y_pred_model), len(y_pred_benchmark))
    if n < 5:
        return np.nan, np.nan

    y_true = y_true[-n:]
    y_pred_model = y_pred_model[-n:]
    y_pred_benchmark = y_pred_benchmark[-n:]

    # Errores
    e_model = y_true - y_pred_model
    e_bench = y_true - y_pred_benchmark

    # Diferencias de loss (MAE o MSE)
    if power == 1:
        d = np.abs(e_model) - np.abs(e_bench)
    else:  # power=2
        d = e_model**2 - e_bench**2

    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    if d_var == 0 or np.isnan(d_var):
        return np.nan, np.nan

    dm_stat = d_mean / math.sqrt(d_var / n)

    # p-valor usando normal estándar
    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    p_value = 2.0 * (1.0 - norm_cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)

# =========================
#  MÉTRICAS DE TRADING
# =========================

def _compute_pnl_series(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Construye la serie de PnL por periodo a partir de:
    - Señal = sign(y_pred)
    - Retorno real = y_true
    - PnL = señal * retorno_real
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    signal = np.sign(y_pred)
    pnl = signal * y_true
    return pnl


def calculate_sharpe_ratio(pnl: np.ndarray, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calcula el Sharpe Ratio anualizado.
    Asume que pnl es una serie de retornos por periodo.
    """
    if len(pnl) < 2:
        return 0.0

    excess = pnl - risk_free / periods_per_year
    mean_excess = excess.mean()
    std_excess = excess.std(ddof=1)

    if std_excess == 0:
        return 0.0

    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_sortino_ratio(pnl: np.ndarray, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calcula el Sortino Ratio anualizado (solo penaliza volatilidad negativa).
    """
    if len(pnl) < 2:
        return 0.0

    excess = pnl - risk_free / periods_per_year
    downside = excess[excess < 0]

    if len(downside) == 0:
        return 0.0

    downside_std = downside.std(ddof=1)
    if downside_std == 0:
        return 0.0

    mean_excess = excess.mean()
    sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)
    return float(sortino)


def calculate_max_drawdown(pnl: np.ndarray) -> float:
    """
    Calcula el máximo drawdown sobre el equity curve acumulado.
    Devuelve un valor negativo (por ejemplo -0.25 = -25%).
    """
    if len(pnl) == 0:
        return 0.0

    equity = pnl.cumsum()
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity - running_max

    max_dd = drawdowns.min()  # es negativo
    return float(max_dd)


def calculate_profit_factor(pnl: np.ndarray) -> float:
    """
    Calcula el Profit Factor = (suma ganancias) / (|suma pérdidas|).
    """
    if len(pnl) == 0:
        return 0.0

    gains = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()

    if losses == 0:
        return float("inf") if gains > 0 else 0.0

    return float(gains / abs(losses))


def calculate_win_rate_and_payoff(pnl: np.ndarray) -> tuple[float, float]:
    """
    Calcula:
    - win_rate: % de trades ganadores
    - payoff_ratio: promedio ganancia / promedio pérdida (en valor absoluto)
    """
    if len(pnl) == 0:
        return 0.0, 0.0

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    n_trades = len(pnl[pnl != 0])
    if n_trades == 0:
        return 0.0, 0.0

    win_rate = len(wins) / n_trades * 100.0

    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0

    if avg_loss == 0:
        payoff = float("inf") if avg_win > 0 else 0.0
    else:
        payoff = avg_win / abs(avg_loss)

    return float(win_rate), float(payoff)


def calculate_all_metrics(
    true_values,
    predicted_values,
    benchmark_values=None,   # opcional: benchmark explícito
    risk_free: float = 0.0,  # tasa libre de riesgo anual (para Sharpe/Sortino)
    periods_per_year: int = 252,  # frecuencia (252 para diario, 12 para mensual, etc.)
    pip_size: float = 0.0001,
    threshold_pips: float = 0.0,
) -> dict[str, float]:
    """
    Calcula varias métricas de un solo modelo:

    Error de predicción (SIEMPRE sobre la predicción completa):
    - rmse
    - mae

    Métrica direccional:
    - hit_rate  (aplicando umbral de pips, si se define)

    Comparación con benchmark (Diebold-Mariano):
    - dm_stat
    - dm_pvalue

    Métricas de trading basadas en la dirección de la predicción:
    - sharpe
    - sortino
    - max_drawdown
    - profit_factor
    - win_rate
    - payoff_ratio

    Notas:
    - Se asume que true_values son retornos (p.ej. Return_1).
    - La señal se construye como sign(predicted_values) tras aplicar threshold_pips.
    - Si no se pasa benchmark_values, se usa un benchmark ingenuo: predicción = 0.
    """

    true = np.array(true_values, dtype=float)
    pred = np.array(predicted_values, dtype=float)

    if true.shape != pred.shape:
        raise ValueError("true_values y predicted_values deben tener la misma forma.")

    metrics: dict[str, float] = {}

    # =========================
    #  MÉTRICAS DE ERROR (sin umbral)
    # =========================
    metrics["rmse"] = calculate_rmse(true, pred)
    metrics["mae"] = calculate_mae(true, pred)

    # =========================
    #  APLICAR UMBRAL EN PIPS PARA SEÑALES DE TRADING
    # =========================
    # Si threshold_pips > 0, forzamos HOLD cuando la predicción sea "pequeña"
    if pip_size <= 0:
        pip_size = 1.0  # fallback defensivo

    if threshold_pips > 0.0:
        # Aproximación: retorno mínimo equivalente a X pips
        threshold_return = threshold_pips * pip_size
        pred_for_signals = pred.copy()
        small = np.abs(pred_for_signals) < threshold_return
        pred_for_signals[small] = 0.0
    else:
        pred_for_signals = pred

    # =========================
    #  HIT RATE (con umbral aplicado)
    # =========================
    metrics["hit_rate"] = calculate_hit_rate(true, pred_for_signals)

    # =========================
    #  DIEBOLD–MARIANO (sin umbral, usa la predicción cruda)
    # =========================
    if benchmark_values is None:
        bench = np.zeros_like(true)
    else:
        bench = np.array(benchmark_values, dtype=float)
        if bench.shape != true.shape:
            raise ValueError("benchmark_values debe tener la misma forma que true_values.")

    try:
        dm_stat, dm_pvalue = diebold_mariano(true, pred, bench, power=2)
    except Exception:
        dm_stat, dm_pvalue = np.nan, np.nan

    metrics["dm_stat"] = dm_stat
    metrics["dm_pvalue"] = dm_pvalue

    # =========================
    #  MÉTRICAS DE TRADING (sobre señales filtradas por umbral)
    # =========================
    pnl = _compute_pnl_series(true, pred_for_signals)

    sharpe = calculate_sharpe_ratio(pnl, risk_free=risk_free, periods_per_year=periods_per_year)
    sortino = calculate_sortino_ratio(pnl, risk_free=risk_free, periods_per_year=periods_per_year)
    max_dd = calculate_max_drawdown(pnl)
    profit_factor = calculate_profit_factor(pnl)
    win_rate, payoff = calculate_win_rate_and_payoff(pnl)

    metrics["sharpe"] = sharpe
    metrics["sortino"] = sortino
    metrics["max_drawdown"] = max_dd
    metrics["profit_factor"] = profit_factor
    metrics["win_rate"] = win_rate
    metrics["payoff_ratio"] = payoff

    return metrics
