"""
utils/risk_utils.py

Funciones de apoyo para gestión de riesgo:
- cálculo de SL/TP a partir de config de riesgo y ATR/pips
- cálculo de tamaño de posición dado balance y distancia al stop
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Dict

import numpy as np


Side = Literal["BUY", "SELL"]


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.01
    sl_mode: str = "atr"  # "atr" o "fixed"
    fixed_sl_pips: float = 30.0
    atr_sl_multiplier: float = 1.5
    atr_sl_min_pips: float = 10.0
    atr_sl_max_pips: float = 80.0
    tp_rr_ratio: float = 2.0
    entry_mode: str = "close"  # "close" o "atr_pullback"
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
        entry_mode=str(cfg.get("entry_mode", "close")),
        atr_entry_mult=float(cfg.get("atr_entry_mult", 0.5)),
    )


def compute_entry_sl_tp(
    side: Side,
    close_price: float,
    atr_value: Optional[float],
    pip_size: float,
    risk_cfg_dict: Optional[Dict] = None,
) -> Dict[str, float]:
    """Calcula entry, stop-loss y take-profit recomendados.

    Devuelve un diccionario con:
    - entry_price
    - sl_price
    - tp_price
    - sl_pips
    - tp_pips
    """
    cfg = _get_risk_config_from_dict(risk_cfg_dict or {})
    if pip_size <= 0:
        pip_size = 1e-4

    # --- Entry ---
    entry_price = float(close_price)
    if cfg.entry_mode == "atr_pullback" and atr_value is not None and np.isfinite(atr_value):
        # entrada ligeramente mejor que el cierre actual
        if side == "BUY":
            entry_price = close_price - cfg.atr_entry_mult * atr_value
        else:
            entry_price = close_price + cfg.atr_entry_mult * atr_value

    # --- Stop loss en pips ---
    if cfg.sl_mode == "fixed" or atr_value is None or not np.isfinite(atr_value):
        sl_pips = cfg.fixed_sl_pips
    else:
        sl_pips = cfg.atr_sl_multiplier * (atr_value / pip_size)
        sl_pips = max(cfg.atr_sl_min_pips, min(sl_pips, cfg.atr_sl_max_pips))

    # --- Precios de SL / TP ---
    if side == "BUY":
        sl_price = entry_price - sl_pips * pip_size
        tp_pips = sl_pips * cfg.tp_rr_ratio
        tp_price = entry_price + tp_pips * pip_size
    else:
        sl_price = entry_price + sl_pips * pip_size
        tp_pips = sl_pips * cfg.tp_rr_ratio
        tp_price = entry_price - tp_pips * pip_size

    return {
        "entry_price": float(entry_price),
        "sl_price": float(sl_price),
        "tp_price": float(tp_price),
        "sl_pips": float(sl_pips),
        "tp_pips": float(tp_pips),
    }


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
    """Calcula el tamaño de posición (lotes) dado:

    - balance: saldo de la cuenta
    - entry_price: precio de entrada
    - sl_price: precio de stop loss
    - point: tamaño de tick (por ejemplo 0.0001 en EURUSD)
    - contract_size: tamaño de contrato por 1.0 lote (por ejemplo 100000)
    - risk_per_trade_pct: porcentaje de balance a arriesgar (0.01 = 1%)
    - min_lot, lot_step: restricciones del broker

    Aproximación: valor de 1 pip por lote ≈ contract_size * point.
    """
    if balance <= 0 or risk_per_trade_pct <= 0:
        return 0.0
    if point <= 0 or contract_size <= 0:
        return 0.0

    price_risk = abs(entry_price - sl_price)
    if price_risk <= 0:
        return 0.0

    # número de ticks (pips "finos") entre entrada y stop
    ticks = price_risk / point

    # valor de 1 tick por 1 lote en divisa de la cuenta (aprox)
    pip_value_per_lot = contract_size * point

    risk_amount = balance * risk_per_trade_pct
    denom = ticks * pip_value_per_lot
    if denom <= 0:
        return 0.0

    raw_lots = risk_amount / denom

    # aplicar mínimos y pasos
    lots = max(min_lot, raw_lots)
    lots = round(lots / lot_step) * lot_step  # redondeo a múltiplo de lot_step
    return float(lots)
