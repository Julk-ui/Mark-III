from __future__ import annotations

"""Métricas extendidas para backtesting y selección de estrategias.

Diseñado como reemplazo compatible de utils.metrics.calculate_all_metrics.
Calcula métricas de error, clasificación direccional y trading.

Compatibilidad híbrida con el módulo legacy:
- hit_rate excluye señales HOLD (predicción 0 tras umbral)
- dm_stat conserva el signo histórico del proyecto
"""

from dataclasses import dataclass
from typing import Any

import math
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


EPS = 1e-12


@dataclass
class TradeSeries:
    strategy_returns: np.ndarray
    active_mask: np.ndarray
    trade_returns: np.ndarray
    signals: np.ndarray


def _to_numpy(values: list | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def calculate_hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    mask = pred_sign != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(true_sign[mask] == pred_sign[mask]) * 100.0)


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_cls = np.sign(y_true)
    y_pred_cls = np.sign(y_pred)
    return float(accuracy_score(y_true_cls, y_pred_cls) * 100.0)


def calculate_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_cls = np.sign(y_true)
    y_pred_cls = np.sign(y_pred)
    return float(f1_score(y_true_cls, y_pred_cls, average="macro", zero_division=0))


def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_cls = np.sign(y_true)
    y_pred_cls = np.sign(y_pred)
    return float(precision_score(y_true_cls, y_pred_cls, average="macro", zero_division=0))


def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_cls = np.sign(y_true)
    y_pred_cls = np.sign(y_pred)
    return float(recall_score(y_true_cls, y_pred_cls, average="macro", zero_division=0))


def _build_trade_series(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pip_size: float = 0.0001,
    threshold_pips: float = 0.0,
    active_mask_override: np.ndarray | None = None,
) -> TradeSeries:
    pip_size = float(pip_size or 0.0001)
    threshold_pips = float(threshold_pips or 0.0)

    predicted_pips = y_pred / max(pip_size, EPS)
    active_mask = np.abs(predicted_pips) >= threshold_pips if threshold_pips > 0 else np.ones_like(y_pred, dtype=bool)
    if active_mask_override is not None:
        override_mask = np.asarray(active_mask_override, dtype=bool).reshape(-1)
        if override_mask.shape != active_mask.shape:
            raise ValueError("active_mask_override debe tener la misma forma que y_pred.")
        active_mask = active_mask & override_mask
    signals = np.sign(y_pred)
    trade_returns = signals * y_true
    strategy_returns = np.where(active_mask, trade_returns, 0.0)

    return TradeSeries(
        strategy_returns=strategy_returns,
        active_mask=active_mask.astype(bool),
        trade_returns=trade_returns,
        signals=signals,
    )


def calculate_max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return float("nan")
    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(running_max, EPS) - 1.0
    return float(np.min(drawdown))


def calculate_sharpe(returns: np.ndarray, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    if returns.size < 2:
        return float("nan")
    rf_period = float(risk_free) / max(periods_per_year, 1)
    excess = returns - rf_period
    std = float(np.std(excess, ddof=1))
    if abs(std) < EPS:
        return float("nan")
    return float(np.sqrt(periods_per_year) * np.mean(excess) / std)


def calculate_sortino(returns: np.ndarray, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    if returns.size < 2:
        return float("nan")
    rf_period = float(risk_free) / max(periods_per_year, 1)
    excess = returns - rf_period
    downside = excess[excess < 0]
    if downside.size < 2:
        return float("nan")
    downside_std = float(np.std(downside, ddof=1))
    if abs(downside_std) < EPS:
        return float("nan")
    return float(np.sqrt(periods_per_year) * np.mean(excess) / downside_std)


def calculate_calmar(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if returns.size == 0:
        return float("nan")
    avg_period = float(np.mean(returns))
    annual_return = (1.0 + avg_period) ** periods_per_year - 1.0
    max_dd = abs(calculate_max_drawdown(returns))
    if max_dd < EPS:
        return float("nan")
    return float(annual_return / max_dd)


def calculate_profit_factor(trade_returns: np.ndarray) -> float:
    if trade_returns.size == 0:
        return float("nan")
    gross_profit = float(np.sum(trade_returns[trade_returns > 0]))
    gross_loss = float(np.sum(np.abs(trade_returns[trade_returns < 0])))
    if gross_loss < EPS:
        return float("inf") if gross_profit > 0 else float("nan")
    return float(gross_profit / gross_loss)


def calculate_win_rate(trade_returns: np.ndarray) -> float:
    if trade_returns.size == 0:
        return float("nan")
    return float(np.mean(trade_returns > 0) * 100.0)


def calculate_payoff_ratio(trade_returns: np.ndarray) -> float:
    wins = trade_returns[trade_returns > 0]
    losses = np.abs(trade_returns[trade_returns < 0])
    if wins.size == 0:
        return float("nan")
    if losses.size == 0:
        return float("inf")
    return float(np.mean(wins) / max(np.mean(losses), EPS))


def calculate_consistency_ratio(strategy_returns: np.ndarray) -> float:
    """Porcentaje de ventanas rolling con retorno acumulado positivo."""
    if strategy_returns.size < 10:
        return float("nan")
    window = min(20, strategy_returns.size)
    rolling = np.convolve(strategy_returns, np.ones(window), mode="valid")
    return float(np.mean(rolling > 0))


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    benchmark_values: np.ndarray | None = None,
    power: int = 2,
) -> tuple[float, float]:
    """Versión ligera del test Diebold-Mariano.

    Usa pérdida absoluta o cuadrática y aproximación normal.
    """
    if benchmark_values is None:
        benchmark_values = np.zeros_like(y_true)
    benchmark_values = _to_numpy(benchmark_values)

    e_model = y_true - y_pred
    e_bench = y_true - benchmark_values

    if power == 1:
        d = np.abs(e_bench) - np.abs(e_model)
    else:
        d = (e_bench ** 2) - (e_model ** 2)

    if d.size < 3:
        return float("nan"), float("nan")

    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))
    if var_d < EPS:
        return float("nan"), float("nan")

    dm_stat = mean_d / math.sqrt(var_d / d.size)
    # Aproximación normal estándar sin scipy
    pvalue = math.erfc(abs(dm_stat) / math.sqrt(2.0))
    return float(dm_stat), float(pvalue)


def calculate_all_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    benchmark_values: list | np.ndarray | None = None,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
    pip_size: float = 0.0001,
    threshold_pips: float = 0.0,
    active_mask_override: list | np.ndarray | None = None,
) -> dict[str, Any]:
    """Calcula métricas de error, clasificación y trading.

    Compatible con llamadas como:
    calculate_all_metrics(y_true, y_pred, benchmark_values=None,
                          risk_free=0.0, periods_per_year=252,
                          pip_size=0.0001, threshold_pips=10)
    """
    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)

    if y_true_np.shape != y_pred_np.shape:
        raise ValueError("Los arrays de y_true y y_pred deben tener la misma forma.")

    ts = _build_trade_series(
        y_true=y_true_np,
        y_pred=y_pred_np,
        pip_size=pip_size,
        threshold_pips=threshold_pips,
        active_mask_override=None if active_mask_override is None else np.asarray(active_mask_override, dtype=bool),
    )

    pred_for_trading = np.where(ts.active_mask, y_pred_np, 0.0)
    active_trade_returns = ts.trade_returns[ts.active_mask]
    dm_stat, dm_pvalue = diebold_mariano_test(y_true_np, y_pred_np, benchmark_values=benchmark_values)
    # Compatibilidad con el signo legacy: negativo = mejor que benchmark ingenuo.
    if not np.isnan(dm_stat):
        dm_stat = -dm_stat

    max_dd = calculate_max_drawdown(ts.strategy_returns)

    return {
        "rmse": calculate_rmse(y_true_np, y_pred_np),
        "mae": calculate_mae(y_true_np, y_pred_np),
        "hit_rate": calculate_hit_rate(y_true_np, pred_for_trading),
        "accuracy": calculate_accuracy(y_true_np, y_pred_np),
        "f1_score": calculate_f1(y_true_np, y_pred_np),
        "precision": calculate_precision(y_true_np, y_pred_np),
        "recall": calculate_recall(y_true_np, y_pred_np),
        "dm_stat": dm_stat,
        "dm_pvalue": dm_pvalue,
        "sharpe": calculate_sharpe(ts.strategy_returns, risk_free=risk_free, periods_per_year=periods_per_year),
        "sortino": calculate_sortino(ts.strategy_returns, risk_free=risk_free, periods_per_year=periods_per_year),
        "calmar": calculate_calmar(ts.strategy_returns, periods_per_year=periods_per_year),
        "max_drawdown": max_dd,
        "profit_factor": calculate_profit_factor(active_trade_returns),
        "win_rate": calculate_win_rate(active_trade_returns),
        "payoff_ratio": calculate_payoff_ratio(active_trade_returns),
        "consistency_ratio": calculate_consistency_ratio(ts.strategy_returns),
        "avg_trade_return": float(np.mean(active_trade_returns)) if active_trade_returns.size else float("nan"),
        "n_test_points": int(y_true_np.size),
        "n_trades": int(np.sum(ts.active_mask)),
    }
