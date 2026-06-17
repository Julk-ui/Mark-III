"""
utils/risk_utils.py

Funciones de apoyo para gestion de riesgo:
- calculo de SL/TP a partir de config de riesgo y ATR/pips
- calculo de tamano de posicion dado balance y distancia al stop
- estimacion de riesgo monetario de posiciones abiertas o planificadas
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np


Side = Literal["BUY", "SELL"]


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.01
    sl_mode: str = "atr"
    fixed_sl_pips: float = 30.0
    atr_sl_multiplier: float = 1.5
    atr_sl_min_pips: float = 10.0
    atr_sl_max_pips: float = 80.0
    tp_rr_ratio: float = 2.0
    tp_use_predicted_move: bool = False
    tp_min_pips: float = 0.0
    tp_max_pips: float = 0.0
    entry_mode: str = "close"
    atr_entry_mult: float = 0.5


def _get_risk_config_from_dict(cfg: Dict) -> RiskConfig:
    if cfg is None:
        return RiskConfig()
    return RiskConfig(
        risk_per_trade_pct=float(cfg.get("risk_per_trade_pct", 0.01)),
        sl_mode=str(cfg.get("sl_mode", "atr")),
        fixed_sl_pips=float(cfg.get("fixed_sl_pips", 30.0)),
        atr_sl_multiplier=float(cfg.get("atr_sl_multiplier", 1.5)),
        atr_sl_min_pips=float(cfg.get("atr_sl_min_pips", 10.0)),
        atr_sl_max_pips=float(cfg.get("atr_sl_max_pips", 80.0)),
        tp_rr_ratio=float(cfg.get("tp_rr_ratio", 2.0)),
        tp_use_predicted_move=bool(cfg.get("tp_use_predicted_move", False)),
        tp_min_pips=float(cfg.get("tp_min_pips", 0.0)),
        tp_max_pips=float(cfg.get("tp_max_pips", 0.0)),
        entry_mode=str(cfg.get("entry_mode", "close")),
        atr_entry_mult=float(cfg.get("atr_entry_mult", 0.5)),
    )


def compute_take_profit_pips(
    *,
    sl_pips: float,
    risk_cfg_dict: Optional[Dict] = None,
    predicted_pips_target: Optional[float] = None,
) -> float:
    """Calcula take-profit en pips respetando RR y opcionalmente el movimiento esperado."""
    cfg = _get_risk_config_from_dict(risk_cfg_dict or {})

    rr_base_tp = max(float(sl_pips), 0.0) * max(float(cfg.tp_rr_ratio), 0.1)
    tp_pips = rr_base_tp

    if cfg.tp_use_predicted_move and predicted_pips_target is not None and np.isfinite(predicted_pips_target):
        tp_pips = max(tp_pips, abs(float(predicted_pips_target)))

    if cfg.tp_min_pips > 0:
        tp_pips = max(tp_pips, float(cfg.tp_min_pips))
    if cfg.tp_max_pips > 0:
        tp_pips = min(tp_pips, float(cfg.tp_max_pips))

    return float(tp_pips)


def compute_entry_sl_tp(
    side: Side,
    close_price: float,
    atr_value: Optional[float],
    pip_size: float,
    risk_cfg_dict: Optional[Dict] = None,
    predicted_pips_target: Optional[float] = None,
) -> Dict[str, float]:
    """Calcula entry, stop-loss y take-profit recomendados."""
    cfg = _get_risk_config_from_dict(risk_cfg_dict or {})
    if pip_size <= 0:
        pip_size = 1e-4

    entry_price = float(close_price)
    if cfg.entry_mode == "atr_pullback" and atr_value is not None and np.isfinite(atr_value):
        if side == "BUY":
            entry_price = close_price - cfg.atr_entry_mult * atr_value
        else:
            entry_price = close_price + cfg.atr_entry_mult * atr_value

    if cfg.sl_mode == "fixed" or atr_value is None or not np.isfinite(atr_value):
        sl_pips = cfg.fixed_sl_pips
    else:
        sl_pips = cfg.atr_sl_multiplier * (atr_value / pip_size)
        sl_pips = max(cfg.atr_sl_min_pips, min(sl_pips, cfg.atr_sl_max_pips))

    tp_pips = compute_take_profit_pips(
        sl_pips=sl_pips,
        risk_cfg_dict=risk_cfg_dict,
        predicted_pips_target=predicted_pips_target,
    )

    if side == "BUY":
        sl_price = entry_price - sl_pips * pip_size
        tp_price = entry_price + tp_pips * pip_size
    else:
        sl_price = entry_price + sl_pips * pip_size
        tp_price = entry_price - tp_pips * pip_size

    return {
        "entry_price": float(entry_price),
        "sl_price": float(sl_price),
        "tp_price": float(tp_price),
        "sl_pips": float(sl_pips),
        "tp_pips": float(tp_pips),
    }


def estimate_position_risk_amount(
    entry_price: float,
    sl_price: float,
    point: float,
    contract_size: float,
    volume_lots: float,
) -> float:
    """Estima el riesgo monetario de una posicion abierta o planificada."""
    if volume_lots <= 0:
        return 0.0
    if point <= 0 or contract_size <= 0:
        return 0.0

    price_risk = abs(entry_price - sl_price)
    if price_risk <= 0:
        return 0.0

    ticks = price_risk / point
    tick_value_per_lot = contract_size * point
    return float(ticks * tick_value_per_lot * volume_lots)


def calculate_position_size_for_risk_amount(
    entry_price: float,
    sl_price: float,
    point: float,
    contract_size: float,
    risk_amount: float,
    min_lot: float = 0.01,
    lot_step: float = 0.01,
) -> float:
    """Calcula lotes para un riesgo monetario fijo.

    Si el presupuesto no alcanza para el lote minimo del broker, devuelve 0.0.
    """
    if risk_amount <= 0:
        return 0.0
    if point <= 0 or contract_size <= 0:
        return 0.0

    price_risk = abs(entry_price - sl_price)
    if price_risk <= 0:
        return 0.0

    ticks = price_risk / point
    tick_value_per_lot = contract_size * point
    denom = ticks * tick_value_per_lot
    if denom <= 0:
        return 0.0

    raw_lots = risk_amount / denom
    if raw_lots < min_lot:
        return 0.0

    lots = round(raw_lots / lot_step) * lot_step
    if lots < min_lot:
        return 0.0
    return float(lots)


def calculate_position_size(
    balance: float,
    entry_price: float,
    sl_price: float,
    point: float,
    contract_size: float,
    risk_per_trade_pct: float,
    min_lot: float = 0.01,
    lot_step: float = 0.01,
) -> float:
    """Calcula el tamano de posicion en lotes.

    Mantiene compatibilidad con la semantica historica del proyecto:
    si el presupuesto de riesgo queda por debajo del lote minimo, fuerza el
    lote minimo permitido.
    """
    if balance <= 0 or risk_per_trade_pct <= 0:
        return 0.0

    risk_amount = balance * risk_per_trade_pct
    strict_lots = calculate_position_size_for_risk_amount(
        entry_price=entry_price,
        sl_price=sl_price,
        point=point,
        contract_size=contract_size,
        risk_amount=risk_amount,
        min_lot=min_lot,
        lot_step=lot_step,
    )
    if strict_lots > 0:
        return strict_lots

    if point <= 0 or contract_size <= 0:
        return 0.0

    price_risk = abs(entry_price - sl_price)
    if price_risk <= 0:
        return 0.0

    ticks = price_risk / point
    tick_value_per_lot = contract_size * point
    denom = ticks * tick_value_per_lot
    if denom <= 0:
        return 0.0

    raw_lots = risk_amount / denom
    lots = max(min_lot, raw_lots)
    lots = round(lots / lot_step) * lot_step
    return float(lots)
