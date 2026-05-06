from __future__ import annotations

"""Utilidades para convertir una predicción en señal operable con score de confianza."""

from typing import Any

import math


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def compute_decision_confidence(
    pred_return: float,
    pip_size: float,
    min_pips_signal: float,
    model_metrics: dict[str, Any] | None = None,
    probability: float | None = None,
) -> float:
    """Combina fuerza de señal y calidad histórica del modelo en [0, 1]."""
    model_metrics = model_metrics or {}

    predicted_pips = abs(pred_return / max(pip_size, 1e-12))
    strength = min(predicted_pips / max(min_pips_signal, 1e-12), 2.0) / 2.0

    hit_rate = _safe_float(model_metrics.get("hit_rate"), 50.0) / 100.0
    win_rate = _safe_float(model_metrics.get("win_rate"), 50.0) / 100.0
    sharpe = _safe_float(model_metrics.get("sharpe"), 0.0)
    profit_factor = _safe_float(model_metrics.get("profit_factor"), 1.0)

    sharpe_score = 1.0 / (1.0 + math.exp(-sharpe))
    pf_score = min(max((profit_factor - 1.0) / 2.0, 0.0), 1.0)
    hist_quality = 0.35 * hit_rate + 0.25 * win_rate + 0.20 * sharpe_score + 0.20 * pf_score

    if probability is not None:
        prob_score = max(0.0, min(float(probability), 1.0))
        confidence = 0.45 * prob_score + 0.30 * strength + 0.25 * hist_quality
    else:
        confidence = 0.55 * strength + 0.45 * hist_quality

    return max(0.0, min(confidence, 1.0))


def build_signal_from_prediction(
    pred_return: float,
    pip_size: float,
    min_pips_signal: float,
    model_metrics: dict[str, Any] | None = None,
    min_confidence: float = 0.60,
    probability: float | None = None,
) -> dict[str, float | str]:
    if pip_size <= 0:
        pip_size = 0.0001

    predicted_pips = pred_return / pip_size
    confidence = compute_decision_confidence(
        pred_return=pred_return,
        pip_size=pip_size,
        min_pips_signal=min_pips_signal,
        model_metrics=model_metrics,
        probability=probability,
    )

    if abs(predicted_pips) < min_pips_signal:
        signal = "HOLD"
    elif confidence < min_confidence:
        signal = "HOLD"
    else:
        signal = "BUY" if predicted_pips > 0 else "SELL"

    return {
        "signal": signal,
        "confidence": confidence,
        "predicted_pips": predicted_pips,
    }
