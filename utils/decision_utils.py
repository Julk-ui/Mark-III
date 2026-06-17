from __future__ import annotations

"""Utilidades para convertir una predicción en señal operable con score de confianza."""

from typing import Any

import math


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        parsed = float(value)
        if not math.isfinite(parsed):
            return default
        return parsed
    except Exception:
        return default


def predicted_return_to_pips(
    pred_return: float,
    pip_size: float,
    price_reference: float | None = None,
) -> float:
    """Convierte un retorno esperado en pips usando el precio de referencia actual."""
    pip_size = max(_safe_float(pip_size, 0.0001), 1e-12)
    price_ref = _safe_float(price_reference, 0.0)

    if price_ref > 0.0:
        predicted_price_change = _safe_float(pred_return, 0.0) * price_ref
    else:
        # Fallback legacy para no romper llamadas antiguas sin precio disponible.
        predicted_price_change = _safe_float(pred_return, 0.0)

    return predicted_price_change / pip_size


def compute_decision_confidence(
    pred_return: float,
    pip_size: float,
    min_pips_signal: float,
    model_metrics: dict[str, Any] | None = None,
    probability: float | None = None,
    price_reference: float | None = None,
) -> float:
    """Combina fuerza de señal y calidad histórica del modelo en [0, 1]."""
    model_metrics = model_metrics or {}

    predicted_pips = abs(
        predicted_return_to_pips(
            pred_return=pred_return,
            pip_size=pip_size,
            price_reference=price_reference,
        )
    )
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
    price_reference: float | None = None,
) -> dict[str, float | str]:
    if pip_size <= 0:
        pip_size = 0.0001

    predicted_pips = predicted_return_to_pips(
        pred_return=pred_return,
        pip_size=pip_size,
        price_reference=price_reference,
    )
    confidence = compute_decision_confidence(
        pred_return=pred_return,
        pip_size=pip_size,
        min_pips_signal=min_pips_signal,
        model_metrics=model_metrics,
        probability=probability,
        price_reference=price_reference,
    )

    if abs(predicted_pips) < min_pips_signal:
        signal = "HOLD"
    elif confidence < min_confidence:
        signal = "HOLD"
    else:
        signal = "BUY" if predicted_pips > 0 else "SELL"

    signal_target_pips = 0.0
    if signal in {"BUY", "SELL"}:
        signal_target_pips = abs(predicted_pips)

    return {
        "signal": signal,
        "confidence": confidence,
        "predicted_pips": predicted_pips,
        "signal_target_pips": signal_target_pips,
        "touch_probability": None,
        "prob_up": None,
        "prob_hold": None,
        "prob_down": None,
    }


def build_signal_from_probabilities(
    *,
    prob_up: float,
    prob_hold: float,
    prob_down: float,
    barrier_pips: float,
    min_confidence: float = 0.60,
    probability_threshold: float | None = None,
    probability_margin: float = 0.05,
    model_metrics: dict[str, Any] | None = None,
) -> dict[str, float | str]:
    """
    Convierte probabilidades de first-touch en señal operable.

    Devuelve:
    - signal: BUY / SELL / HOLD
    - confidence: score en [0,1]
    - predicted_pips: desplazamiento esperado firmado
    - signal_target_pips: pips del objetivo direccional si hay señal
    - prob_up / prob_hold / prob_down
    """
    model_metrics = model_metrics or {}
    prob_up = max(0.0, min(_safe_float(prob_up, 0.0), 1.0))
    prob_hold = max(0.0, min(_safe_float(prob_hold, 0.0), 1.0))
    prob_down = max(0.0, min(_safe_float(prob_down, 0.0), 1.0))
    total = prob_up + prob_hold + prob_down
    if total > 0:
        prob_up /= total
        prob_hold /= total
        prob_down /= total

    barrier_pips = abs(_safe_float(barrier_pips, 0.0))
    touch_probability = prob_up + prob_down
    dominant_prob = max(prob_up, prob_down)
    dominant_side = "BUY" if prob_up > prob_down else "SELL" if prob_down > prob_up else "HOLD"
    side_margin = abs(prob_up - prob_down)
    probability_threshold = (
        max(0.0, min(_safe_float(probability_threshold, 0.0), 1.0))
        if probability_threshold is not None
        else max(0.50, min(_safe_float(min_confidence, 0.60), 0.95))
    )
    probability_margin = max(0.0, min(_safe_float(probability_margin, 0.05), 1.0))

    hit_rate = _safe_float(model_metrics.get("hit_rate"), 50.0) / 100.0
    win_rate = _safe_float(model_metrics.get("win_rate"), 50.0) / 100.0
    sharpe = _safe_float(model_metrics.get("sharpe"), 0.0)
    profit_factor = _safe_float(model_metrics.get("profit_factor"), 1.0)
    sharpe_score = 1.0 / (1.0 + math.exp(-sharpe))
    pf_score = min(max((profit_factor - 1.0) / 2.0, 0.0), 1.0)
    hist_quality = 0.35 * hit_rate + 0.25 * win_rate + 0.20 * sharpe_score + 0.20 * pf_score

    confidence = 0.55 * dominant_prob + 0.20 * side_margin + 0.10 * touch_probability + 0.15 * hist_quality
    confidence = max(0.0, min(confidence, 1.0))

    signal = "HOLD"
    if dominant_side != "HOLD" and dominant_prob >= probability_threshold and side_margin >= probability_margin:
        signal = dominant_side if confidence >= min_confidence else "HOLD"

    expected_move_pips = barrier_pips * (prob_up - prob_down)
    signal_target_pips = 0.0
    if signal == "BUY":
        signal_target_pips = barrier_pips
    elif signal == "SELL":
        signal_target_pips = -barrier_pips

    return {
        "signal": signal,
        "confidence": confidence,
        "predicted_pips": expected_move_pips,
        "signal_target_pips": signal_target_pips,
        "touch_probability": touch_probability,
        "prob_up": prob_up,
        "prob_hold": prob_hold,
        "prob_down": prob_down,
    }


def build_signal_from_hybrid_prediction(
    *,
    pred_return: float,
    pip_size: float,
    min_pips_signal: float,
    price_reference: float | None,
    primary_model_metrics: dict[str, Any] | None = None,
    prob_up: float,
    prob_hold: float,
    prob_down: float,
    barrier_pips: float,
    min_confidence: float = 0.60,
    probability_threshold: float | None = None,
    probability_margin: float = 0.05,
    filter_model_metrics: dict[str, Any] | None = None,
    require_alignment: bool = True,
    filter_gate_mode: str = "full_signal",
    support_probability_threshold: float | None = None,
    support_probability_margin: float | None = None,
    support_score_min: float | None = None,
    contradiction_margin: float | None = None,
) -> dict[str, float | str | bool]:
    """
    Combina un modelo principal de retorno con un filtro probabilistico.

    La direccion y magnitud vienen del modelo principal. El filtro de barrera
    valida si esa idea es suficientemente operable segun probabilidades de
    first-touch.
    """
    primary_info = build_signal_from_prediction(
        pred_return=pred_return,
        pip_size=pip_size,
        min_pips_signal=min_pips_signal,
        model_metrics=primary_model_metrics or {},
        min_confidence=0.0,
        probability=None,
        price_reference=price_reference,
    )
    filter_info = build_signal_from_probabilities(
        prob_up=prob_up,
        prob_hold=prob_hold,
        prob_down=prob_down,
        barrier_pips=barrier_pips,
        min_confidence=0.0,
        probability_threshold=probability_threshold,
        probability_margin=probability_margin,
        model_metrics=filter_model_metrics or {},
    )

    primary_signal = str(primary_info.get("signal", "HOLD"))
    filter_signal = str(filter_info.get("signal", "HOLD"))
    primary_confidence = _safe_float(primary_info.get("confidence"), 0.0)
    filter_confidence = _safe_float(filter_info.get("confidence"), 0.0)
    touch_probability = _safe_float(filter_info.get("touch_probability"), 0.0)
    predicted_pips = _safe_float(primary_info.get("predicted_pips"), 0.0)
    prob_up_norm = _safe_float(filter_info.get("prob_up"), 0.0)
    prob_down_norm = _safe_float(filter_info.get("prob_down"), 0.0)
    side_margin = abs(
        prob_up_norm
        - prob_down_norm
    )
    dominant_side = "BUY" if prob_up_norm > prob_down_norm else "SELL" if prob_down_norm > prob_up_norm else "HOLD"
    dominant_prob = max(prob_up_norm, prob_down_norm)
    filter_gate_mode = str(filter_gate_mode or "full_signal").strip().lower()
    if filter_gate_mode not in {"full_signal", "direction_support", "support_score", "primary_only"}:
        filter_gate_mode = "full_signal"
    effective_support_threshold = (
        _safe_float(support_probability_threshold, -1.0)
        if support_probability_threshold is not None
        else _safe_float(probability_threshold, 0.0)
    )
    effective_support_margin = (
        _safe_float(support_probability_margin, -1.0)
        if support_probability_margin is not None
        else _safe_float(probability_margin, 0.0)
    )
    if effective_support_threshold < 0.0:
        effective_support_threshold = _safe_float(probability_threshold, 0.0)
    if effective_support_margin < 0.0:
        effective_support_margin = _safe_float(probability_margin, 0.0)
    effective_support_score_min = (
        _safe_float(support_score_min, -1.0)
        if support_score_min is not None
        else -1.0
    )
    if effective_support_score_min < 0.0:
        effective_support_score_min = max(effective_support_margin, 0.0)
    effective_contradiction_margin = (
        _safe_float(contradiction_margin, -1.0)
        if contradiction_margin is not None
        else -1.0
    )
    if effective_contradiction_margin < 0.0:
        effective_contradiction_margin = max(effective_support_margin, 0.0)

    combined_confidence = (
        0.50 * primary_confidence
        + 0.35 * filter_confidence
        + 0.10 * min(touch_probability, 1.0)
        + 0.05 * min(side_margin * 2.0, 1.0)
    )
    if filter_gate_mode == "primary_only":
        combined_confidence = primary_confidence
    combined_confidence = max(0.0, min(combined_confidence, 1.0))

    signal = "HOLD"
    filter_passed = filter_signal in {"BUY", "SELL"}
    alignment_ok = (
        primary_signal in {"BUY", "SELL"}
        and filter_signal in {"BUY", "SELL"}
        and primary_signal == filter_signal
    )
    support_passed = (
        primary_signal in {"BUY", "SELL"}
        and dominant_side == primary_signal
        and dominant_prob >= max(effective_support_threshold, 0.0)
        and side_margin >= max(effective_support_margin, 0.0)
    )
    same_side_prob = 0.0
    opposite_side_prob = 0.0
    if primary_signal == "BUY":
        same_side_prob = prob_up_norm
        opposite_side_prob = prob_down_norm
    elif primary_signal == "SELL":
        same_side_prob = prob_down_norm
        opposite_side_prob = prob_up_norm
    support_score = same_side_prob - opposite_side_prob
    contradicted = support_score <= -max(effective_contradiction_margin, 0.0)
    support_score_passed = (
        primary_signal in {"BUY", "SELL"}
        and support_score >= max(effective_support_score_min, 0.0)
        and not contradicted
    )

    if filter_gate_mode == "primary_only":
        gate_passed = primary_signal in {"BUY", "SELL"}
    elif filter_gate_mode == "full_signal":
        gate_passed = filter_passed
    elif filter_gate_mode == "direction_support":
        gate_passed = support_passed
    else:
        gate_passed = support_score_passed

    if primary_signal in {"BUY", "SELL"}:
        if not gate_passed:
            signal = "HOLD"
        elif require_alignment and filter_gate_mode == "full_signal" and not alignment_ok:
            signal = "HOLD"
        elif combined_confidence >= max(_safe_float(min_confidence, 0.0), 0.0):
            signal = primary_signal

    signal_target_pips = 0.0
    if signal in {"BUY", "SELL"}:
        signal_target_pips = abs(predicted_pips)

    return {
        "signal": signal,
        "confidence": combined_confidence,
        "predicted_pips": predicted_pips,
        "signal_target_pips": signal_target_pips,
        "touch_probability": touch_probability,
        "prob_up": filter_info.get("prob_up"),
        "prob_hold": filter_info.get("prob_hold"),
        "prob_down": filter_info.get("prob_down"),
        "primary_signal": primary_signal,
        "primary_confidence": primary_confidence,
        "filter_signal": filter_signal,
        "filter_confidence": filter_confidence,
        "filter_passed": filter_passed,
        "filter_gate_mode": filter_gate_mode,
        "filter_dominant_side": dominant_side,
        "filter_dominant_prob": dominant_prob,
        "filter_support_passed": support_passed,
        "filter_support_score": support_score,
        "filter_same_side_prob": same_side_prob,
        "filter_opposite_side_prob": opposite_side_prob,
        "filter_support_score_passed": support_score_passed,
        "filter_contradicted": contradicted,
        "gate_passed": gate_passed,
        "alignment_ok": alignment_ok,
    }
