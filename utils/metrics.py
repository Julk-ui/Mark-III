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
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    return np.mean(true_sign == pred_sign) * 100

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Accuracy direccional en escala 0-100 (igual que HitRate, pero separado por claridad).
    """
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    return np.mean(true_sign == pred_sign) * 100

def diebold_mariano(y_true: np.ndarray,
                    y_pred_model: np.ndarray,
                    y_pred_benchmark: np.ndarray,
                    power: int = 2) -> tuple[float, float]:
    """
    Test de Diebold-Mariano comparando el modelo vs un benchmark.
    Devuelve (estadístico DM, p-value). Implementación simplificada.
    """
    if len(y_true) < 5:
        return np.nan, np.nan

    # Pérdidas (por defecto, error^2)
    e_model = np.abs(y_true - y_pred_model) ** power
    e_bench = np.abs(y_true - y_pred_benchmark) ** power
    d = e_model - e_bench

    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    n = len(d)
    if var_d == 0 or n <= 1:
        return np.nan, np.nan

    dm_stat = mean_d / np.sqrt(var_d / n)
    # Aproximamos con t-student de n-1 grados de libertad
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n-1))
    return float(dm_stat), float(p_value)

def calculate_all_metrics(y_true: list | np.ndarray,
                          y_pred: list | np.ndarray) -> dict[str, float]:
    y_true_np = np.array(y_true, dtype=float)
    y_pred_np = np.array(y_pred, dtype=float)

    if y_true_np.shape != y_pred_np.shape:
        raise ValueError("Los arrays de y_true y y_pred deben tener la misma forma.")

    rmse = calculate_rmse(y_true_np, y_pred_np)
    mae = calculate_mae(y_true_np, y_pred_np)
    hit = calculate_hit_rate(y_true_np, y_pred_np)
    acc = calculate_accuracy(y_true_np, y_pred_np)

    # Benchmark naive: "random walk" de retorno siguiente = retorno de hoy
    if len(y_true_np) > 1:
        y_bench = np.roll(y_true_np, 1)
        y_bench[0] = y_true_np[0]
        dm_stat, dm_p = diebold_mariano(y_true_np[1:], y_pred_np[1:], y_bench[1:])
    else:
        dm_stat, dm_p = np.nan, np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "hit_rate": hit,
        "accuracy": acc,
        "dm_stat": dm_stat,
        "dm_pvalue": dm_p,
    }
