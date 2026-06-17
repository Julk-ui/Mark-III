# ==========================================
# conexion/easy_Trading.py  (versión unificada y corregida)
# ==========================================
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

# ----------------- Timeframe map -----------------
_TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3,
    "M4": mt5.TIMEFRAME_M4, "M5": mt5.TIMEFRAME_M5, "M6": mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10, "M12": mt5.TIMEFRAME_M12, "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20, "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1, "H2": mt5.TIMEFRAME_H2, "H3": mt5.TIMEFRAME_H3,
    "H4": mt5.TIMEFRAME_H4, "H6": mt5.TIMEFRAME_H6, "H8": mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1,
}

def _tf_to_mt5(tf: str) -> int:
    t = str(tf).upper()
    if t not in _TIMEFRAME_MAP:
        raise ValueError(f"Timeframe no soportado: {tf}")
    return _TIMEFRAME_MAP[t]

# =================================================
#                   Basic_funcs
# =================================================
class Basic_funcs:
    """
    Clase de utilidades para conexión y operaciones con MetaTrader 5.
    Se conecta en __init__ y NO vuelve a inicializar dentro de cada método.
    """

    def __init__(self, login: int, password: str, server: str, path: Optional[str] = None):
        self.login = int(login) if login is not None else None
        self.password = str(password) if password is not None else None
        self.server = str(server) if server is not None else None
        self.path = path
        self.mt5 = mt5
        self._connected = False
        self._connect()

    # ---------- conexión ----------
    def _connect(self):
        if self._connected:
            return
        ok = mt5.initialize(path=self.path, login=self.login, password=self.password, server=self.server) \
             if self.path else \
             mt5.initialize(login=self.login, password=self.password, server=self.server)
        if not ok:
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        if not mt5.login(self.login, password=self.password, server=self.server):
            raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
        self._connected = True

    def __del__(self):
        try:
            mt5.shutdown()
        except Exception:
            pass

    # ---------- datos ----------
    def get_data_for_bt(self, timeframe: str, symbol: str, count: int) -> pd.DataFrame:
        """
        Devuelve OHLCV con columnas: ['Date','Open','High','Low','Close','TickVolume','Volume']
        Index = Date (tz-naive), orden ascendente.
        """
        tf = _tf_to_mt5(timeframe)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, int(count))
        if rates is None:
            raise RuntimeError(f"No se obtuvieron rates de {symbol} {timeframe}: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        # MT5 entrega 'time' en epoch seconds (UTC). Convertimos a naive (sin tz).
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
        df = df.rename(columns={
            "time": "Date",
            "open": "Open", "high": "High", "low": "Low", "close": "Close",
            "tick_volume": "TickVolume", "real_volume": "Volume"
        })
        # uniformamos columnas mínimas
        cols = ["Date","Open","High","Low","Close","TickVolume","Volume"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        df = df[cols].sort_values("Date").set_index("Date")
        return df

    def get_data_from_dates(self, year_ini, month_ini, day_ini,
                            year_fin, month_fin, day_fin,
                            symbol: str, timeframe: str, for_bt: bool = False) -> pd.DataFrame:
        """
        Extrae datos por rango de fechas. Si for_bt=True, devuelve columnas estandarizadas.
        """
        tf = _tf_to_mt5(timeframe)
        from_date = datetime(year_ini, month_ini, day_ini)
        to_date   = datetime(year_fin, month_fin, day_fin)
        rates = mt5.copy_rates_range(symbol, tf, from_date, to_date)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
        if for_bt:
            df = df.rename(columns={
                "time": "Date",
                "open": "Open", "high": "High", "low": "Low", "close": "Close",
                "tick_volume": "TickVolume", "real_volume": "Volume"
            })
            df = df[["Date","Open","High","Low","Close","TickVolume","Volume"]].sort_values("Date").set_index("Date")
        return df

    # ---------- órdenes ----------
    def modify_orders(self, symb: str, ticket: int,
                      stop_loss: float = None, take_profit: float = None,
                      type_order=mt5.ORDER_TYPE_BUY) -> dict:
        req = {
            'action': mt5.TRADE_ACTION_SLTP,
            'symbol': symb,
            'position': ticket,
            'type': type_order,
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK
        }
        if stop_loss is not None:
            req['sl'] = stop_loss
        if take_profit is not None:
            req['tp'] = take_profit
        res = mt5.order_send(req)
        if res is None:
            return {
                "success": False,
                "retcode": None,
                "comment": f"order_send retornó None: {mt5.last_error()}",
                "request": req,
            }
        raw = res._asdict()
        return {
            "success": raw.get("retcode") == mt5.TRADE_RETCODE_DONE,
            "retcode": raw.get("retcode"),
            "comment": raw.get("comment"),
            "request": req,
        }

    def open_operations(self, par: str, volumen: float, tipo_operacion,
                        nombre_bot: str, sl: float = None, tp: float = None) -> None:
        orden = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": par,
            "volume": volumen,
            "type": tipo_operacion,
            "magic": 202204,
            "comment": nombre_bot,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK
        }
        if sl is not None: orden["sl"] = sl
        if tp is not None: orden["tp"] = tp
        res = mt5.order_send(orden)
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Error al enviar orden: {res.retcode}, mensaje: {res.comment}")
        else:
            print(f"✅ Orden ejecutada. Ticket: {res.order}")

    def get_account_info(self) -> dict:
        """Devuelve información resumida de la cuenta conectada."""
        self._connect()
        info = mt5.account_info()
        if info is None:
            return {}
        return info._asdict()

    def get_symbol_tick(self, symbol: str) -> Optional[dict]:
        """Devuelve bid/ask/last del símbolo si está disponible."""
        self._connect()
        mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return tick._asdict()

    def get_symbol_spec(self, symbol: str) -> dict:
        """Devuelve especificaciones útiles del símbolo para validar órdenes."""
        self._connect()
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)
        if info is None:
            return {}
        return {
            "symbol": symbol,
            "digits": int(getattr(info, "digits", 5) or 5),
            "point": float(getattr(info, "point", 0.0) or 0.0),
            "trade_stops_level": int(getattr(info, "trade_stops_level", 0) or 0),
            "trade_freeze_level": int(getattr(info, "trade_freeze_level", 0) or 0),
            "volume_min": float(getattr(info, "volume_min", 0.01) or 0.01),
            "volume_step": float(getattr(info, "volume_step", 0.01) or 0.01),
            "trade_contract_size": float(getattr(info, "trade_contract_size", 0.0) or 0.0),
        }

    def _normalize_price(self, price: float | None, digits: int) -> float | None:
        if price is None:
            return None
        return round(float(price), int(max(digits, 0)))

    def _normalize_volume(self, volume: float | None, symbol_spec: dict) -> float:
        volume_value = float(volume or 0.0)
        if volume_value <= 0:
            return 0.0

        min_lot = float(symbol_spec.get("volume_min", 0.01) or 0.01)
        lot_step = float(symbol_spec.get("volume_step", 0.01) or 0.01)
        if lot_step <= 0:
            lot_step = min_lot

        units = int((volume_value + 1e-12) / lot_step)
        normalized = units * lot_step
        if normalized + 1e-12 < min_lot:
            return 0.0

        step_str = f"{lot_step:.10f}".rstrip("0")
        precision = len(step_str.split(".")[1]) if "." in step_str else 0
        return float(round(normalized, precision))

    def _sanitize_protection_levels(
        self,
        side: str,
        reference_price: float,
        sl: float | None,
        tp: float | None,
        symbol_spec: dict,
    ) -> dict:
        digits = int(symbol_spec.get("digits", 5) or 5)
        point = float(symbol_spec.get("point", 0.0) or 0.0)
        point = point if point > 0 else 0.0001
        stops_level = int(symbol_spec.get("trade_stops_level", 0) or 0)
        min_distance = max(stops_level * point, point)

        side_upper = str(side).upper()
        clean_sl = None if sl is None else float(sl)
        clean_tp = None if tp is None else float(tp)

        if side_upper == "BUY":
            if clean_sl is not None:
                clean_sl = min(clean_sl, reference_price - min_distance)
                if clean_sl >= reference_price:
                    clean_sl = reference_price - min_distance
            if clean_tp is not None:
                clean_tp = max(clean_tp, reference_price + min_distance)
                if clean_tp <= reference_price:
                    clean_tp = reference_price + min_distance
        else:
            if clean_sl is not None:
                clean_sl = max(clean_sl, reference_price + min_distance)
                if clean_sl <= reference_price:
                    clean_sl = reference_price + min_distance
            if clean_tp is not None:
                clean_tp = min(clean_tp, reference_price - min_distance)
                if clean_tp >= reference_price:
                    clean_tp = reference_price - min_distance

        return {
            "sl": self._normalize_price(clean_sl, digits),
            "tp": self._normalize_price(clean_tp, digits),
            "min_distance": float(min_distance),
            "digits": digits,
            "point": point,
            "trade_stops_level": stops_level,
        }

    def get_position_by_ticket(self, ticket: int) -> Optional[dict]:
        """Busca una posición abierta por ticket."""
        self._connect()
        try:
            positions = mt5.positions_get(ticket=int(ticket))
        except TypeError:
            positions = None

        if positions:
            return positions[0]._asdict()

        df = self.get_all_positions()
        if df.empty or "ticket" not in df.columns:
            return None

        matches = df[pd.to_numeric(df["ticket"], errors="coerce").fillna(-1).astype(int) == int(ticket)]
        if matches.empty:
            return None
        return matches.iloc[0].to_dict()

    def ensure_position_protection(
        self,
        *,
        symbol: str,
        position_ticket: int,
        side: str,
        sl: float | None,
        tp: float | None,
    ) -> dict:
        """Asegura que la posición abierta quede protegida con SL/TP."""
        symbol_spec = self.get_symbol_spec(symbol)
        tick = self.get_symbol_tick(symbol)
        if not symbol_spec or tick is None:
            return {
                "success": False,
                "comment": f"No se pudo obtener spec/tick para {symbol}",
                "applied_sl": sl,
                "applied_tp": tp,
            }

        side_upper = str(side).upper()
        reference_price = float(tick["ask"] if side_upper == "BUY" else tick["bid"])
        sanitized = self._sanitize_protection_levels(
            side=side_upper,
            reference_price=reference_price,
            sl=sl,
            tp=tp,
            symbol_spec=symbol_spec,
        )
        order_type = mt5.ORDER_TYPE_BUY if side_upper == "BUY" else mt5.ORDER_TYPE_SELL
        result = self.modify_orders(
            symb=symbol,
            ticket=int(position_ticket),
            stop_loss=sanitized["sl"],
            take_profit=sanitized["tp"],
            type_order=order_type,
        )
        return {
            "success": bool(result.get("success")),
            "comment": result.get("comment"),
            "retcode": result.get("retcode"),
            "applied_sl": sanitized["sl"],
            "applied_tp": sanitized["tp"],
            "min_distance": sanitized["min_distance"],
        }

    def open_market_order(
        self,
        symbol: str,
        volume: float,
        side: str,
        comment: str,
        sl: float | None = None,
        tp: float | None = None,
        deviation: int = 20,
        magic: int = 202204,
    ) -> dict:
        """Abre una orden de mercado y devuelve un resultado normalizado."""
        self._connect()

        side_upper = str(side).upper()
        if side_upper not in {"BUY", "SELL"}:
            raise ValueError(f"Side no soportado: {side}")

        mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
        symbol_spec = self.get_symbol_spec(symbol)
        if tick is None:
            return {
                "success": False,
                "retcode": None,
                "comment": f"No hay tick disponible para {symbol}",
                "order": None,
                "deal": None,
                "position_id": None,
                "price": None,
            }

        order_type = mt5.ORDER_TYPE_BUY if side_upper == "BUY" else mt5.ORDER_TYPE_SELL
        digits = int(symbol_spec.get("digits", 5) or 5)
        price = self._normalize_price(
            float(tick.ask if side_upper == "BUY" else tick.bid),
            digits,
        )
        protection = self._sanitize_protection_levels(
            side=side_upper,
            reference_price=float(price),
            sl=sl,
            tp=tp,
            symbol_spec=symbol_spec,
        )

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "deviation": int(deviation),
            "magic": int(magic),
            "comment": str(comment)[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        if protection["sl"] is not None:
            request["sl"] = float(protection["sl"])
        if protection["tp"] is not None:
            request["tp"] = float(protection["tp"])

        res = mt5.order_send(request)
        if res is None:
            return {
                "success": False,
                "retcode": None,
                "comment": f"order_send retornó None: {mt5.last_error()}",
                "order": None,
                "deal": None,
                "position_id": None,
                "price": price,
            }

        raw = res._asdict()
        invalid_stops_code = getattr(mt5, "TRADE_RETCODE_INVALID_STOPS", 10016)

        if raw.get("retcode") == invalid_stops_code and (sl is not None or tp is not None):
            retry_request = dict(request)
            retry_request.pop("sl", None)
            retry_request.pop("tp", None)
            retry_res = mt5.order_send(retry_request)
            if retry_res is not None:
                raw = retry_res._asdict()
                request = retry_request

        position_id = None
        position = None

        try:
            deals = mt5.history_deals_get(
                datetime.now() - timedelta(hours=12),
                datetime.now() + timedelta(days=1),
            )
            if deals:
                for deal in deals:
                    deal_dict = deal._asdict()
                    if deal_dict.get("ticket") == raw.get("deal"):
                        position_id = deal_dict.get("position_id")
                        break
        except Exception:
            position_id = None

        if position_id is not None:
            position = self.get_position_by_ticket(int(position_id))

        if position is None:
            try:
                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    candidates = [p._asdict() for p in positions]
                    side_value = 0 if side_upper == "BUY" else 1
                    candidates = [p for p in candidates if int(p.get("type", -1)) == side_value]
                    if candidates:
                        candidates.sort(key=lambda p: (p.get("time") or 0, p.get("ticket") or 0))
                        position = candidates[-1]
                        position_id = int(position.get("ticket"))
            except Exception:
                position = None

        protection_result = {
            "success": False,
            "comment": "No se aplicó verificación de protección.",
            "applied_sl": protection["sl"],
            "applied_tp": protection["tp"],
            "retcode": None,
        }

        if raw.get("retcode") == mt5.TRADE_RETCODE_DONE and position_id is not None:
            current_sl = None if position is None else float(position.get("sl") or 0.0)
            current_tp = None if position is None else float(position.get("tp") or 0.0)
            sl_missing = protection["sl"] is not None and (not current_sl or abs(current_sl - float(protection["sl"])) > 1e-9)
            tp_missing = protection["tp"] is not None and (not current_tp or abs(current_tp - float(protection["tp"])) > 1e-9)
            if sl_missing or tp_missing:
                protection_result = self.ensure_position_protection(
                    symbol=symbol,
                    position_ticket=int(position_id),
                    side=side_upper,
                    sl=protection["sl"],
                    tp=protection["tp"],
                )
            else:
                protection_result = {
                    "success": True,
                    "comment": "SL/TP presentes en la posición.",
                    "applied_sl": current_sl,
                    "applied_tp": current_tp,
                    "retcode": raw.get("retcode"),
                }

        return {
            "success": raw.get("retcode") == mt5.TRADE_RETCODE_DONE,
            "retcode": raw.get("retcode"),
            "comment": raw.get("comment"),
            "order": raw.get("order"),
            "deal": raw.get("deal"),
            "position_id": position_id if position_id is not None else raw.get("order"),
            "price": price,
            "request": request,
            "requested_sl": sl,
            "requested_tp": tp,
            "sent_sl": protection["sl"],
            "sent_tp": protection["tp"],
            "protection": protection_result,
            "symbol_spec": symbol_spec,
        }

    def open_pending_limit_order(
        self,
        *,
        symbol: str,
        volume: float,
        side: str,
        price: float,
        comment: str,
        sl: float | None = None,
        tp: float | None = None,
        magic: int = 202204,
    ) -> dict:
        """Coloca una orden pendiente LIMIT y devuelve un resultado normalizado."""
        self._connect()

        side_upper = str(side).upper()
        if side_upper not in {"BUY", "SELL"}:
            raise ValueError(f"Side no soportado para pending limit: {side}")

        mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
        symbol_spec = self.get_symbol_spec(symbol)
        if tick is None or not symbol_spec:
            return {
                "success": False,
                "retcode": None,
                "comment": f"No hay tick/spec disponible para {symbol}",
                "order": None,
                "price": None,
                "request": None,
            }

        normalized_volume = self._normalize_volume(volume, symbol_spec)
        if normalized_volume <= 0:
            return {
                "success": False,
                "retcode": None,
                "comment": f"Volumen invalido para pending limit en {symbol}: {volume}",
                "order": None,
                "price": None,
                "request": None,
            }

        digits = int(symbol_spec.get("digits", 5) or 5)
        point = float(symbol_spec.get("point", 0.0) or 0.0)
        point = point if point > 0 else 0.0001
        stops_level = int(symbol_spec.get("trade_stops_level", 0) or 0)
        min_distance = max(stops_level * point, point)

        current_bid = float(tick.bid)
        current_ask = float(tick.ask)
        limit_price = self._normalize_price(float(price), digits)

        if side_upper == "BUY":
            if limit_price >= current_ask - min_distance:
                return {
                    "success": False,
                    "retcode": None,
                    "comment": (
                        f"BUY LIMIT invalida para {symbol}: price={limit_price} debe quedar "
                        f"por debajo del ask actual {current_ask:.{digits}f}"
                    ),
                    "order": None,
                    "price": limit_price,
                    "request": None,
                }
            order_type = mt5.ORDER_TYPE_BUY_LIMIT
        else:
            if limit_price <= current_bid + min_distance:
                return {
                    "success": False,
                    "retcode": None,
                    "comment": (
                        f"SELL LIMIT invalida para {symbol}: price={limit_price} debe quedar "
                        f"por encima del bid actual {current_bid:.{digits}f}"
                    ),
                    "order": None,
                    "price": limit_price,
                    "request": None,
                }
            order_type = mt5.ORDER_TYPE_SELL_LIMIT

        protection = self._sanitize_protection_levels(
            side=side_upper,
            reference_price=float(limit_price),
            sl=sl,
            tp=tp,
            symbol_spec=symbol_spec,
        )

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": float(normalized_volume),
            "type": order_type,
            "price": float(limit_price),
            "magic": int(magic),
            "comment": str(comment)[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": getattr(mt5, "ORDER_FILLING_RETURN", mt5.ORDER_FILLING_FOK),
        }
        if protection["sl"] is not None:
            request["sl"] = float(protection["sl"])
        if protection["tp"] is not None:
            request["tp"] = float(protection["tp"])

        res = mt5.order_send(request)
        if res is None:
            return {
                "success": False,
                "retcode": None,
                "comment": f"order_send retornó None: {mt5.last_error()}",
                "order": None,
                "price": limit_price,
                "request": request,
            }

        raw = res._asdict()
        return {
            "success": raw.get("retcode") == mt5.TRADE_RETCODE_DONE,
            "retcode": raw.get("retcode"),
            "comment": raw.get("comment"),
            "order": raw.get("order"),
            "deal": raw.get("deal"),
            "price": limit_price,
            "request": request,
            "requested_sl": sl,
            "requested_tp": tp,
            "sent_sl": protection["sl"],
            "sent_tp": protection["tp"],
            "symbol_spec": symbol_spec,
        }

    def obtener_ordenes_pendientes(self) -> pd.DataFrame:
        try:
            ordenes = mt5.orders_get()
            if not ordenes:
                return pd.DataFrame()
            return pd.DataFrame(list(ordenes), columns=ordenes[0]._asdict().keys())
        except Exception:
            return pd.DataFrame()

    def get_pending_orders(
        self,
        *,
        symbol: str | None = None,
        magic: int | None = None,
        ticket: int | None = None,
    ) -> pd.DataFrame:
        """Devuelve ordenes pendientes activas opcionalmente filtradas."""
        self._connect()
        df = self.obtener_ordenes_pendientes()
        if df.empty:
            return df

        if symbol is not None and "symbol" in df.columns:
            df = df[df["symbol"] == symbol]
        if magic is not None and "magic" in df.columns:
            df = df[pd.to_numeric(df["magic"], errors="coerce").fillna(0).astype(int) == int(magic)]
        if ticket is not None and "ticket" in df.columns:
            df = df[pd.to_numeric(df["ticket"], errors="coerce").fillna(-1).astype(int) == int(ticket)]
        return df.reset_index(drop=True)

    def cancel_pending_order(self, *, order_ticket: int) -> dict:
        """Cancela una orden pendiente por ticket."""
        self._connect()
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": int(order_ticket),
        }
        res = mt5.order_send(request)
        if res is None:
            return {
                "success": False,
                "retcode": None,
                "comment": f"order_send retornó None: {mt5.last_error()}",
                "request": request,
            }
        raw = res._asdict()
        return {
            "success": raw.get("retcode") == mt5.TRADE_RETCODE_DONE,
            "retcode": raw.get("retcode"),
            "comment": raw.get("comment"),
            "request": request,
        }

    def remover_operacion_pendiente(self, nom_est: str) -> None:
        df = self.obtener_ordenes_pendientes()
        if df.empty: return
        for ticket in df.loc[df['comment'] == nom_est, 'ticket'].unique().tolist():
            req = {"action": mt5.TRADE_ACTION_REMOVE, "order": ticket, "type_filling": mt5.ORDER_FILLING_IOC}
            mt5.order_send(req)

    def close_all_open_operations(self, data: pd.DataFrame) -> None:
        if data is None or data.empty:
            return
        for ticket in data['ticket'].unique().tolist():
            row = data.loc[data['ticket'] == ticket].iloc[0]
            symb = row['symbol']
            vol  = row['volume']
            side = row['type']  # 0=buy, 1=sell
            close_type = mt5.ORDER_TYPE_SELL if side == 0 else mt5.ORDER_TYPE_BUY
            req = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': symb,
                'volume': vol,
                'type': close_type,
                'position': ticket,
                'comment': 'Cerrar posiciones',
                'type_filling': mt5.ORDER_FILLING_FOK
            }
            mt5.order_send(req)

    def close_position_volume(
        self,
        *,
        symbol: str,
        position_ticket: int,
        volume: float,
        side: str,
        comment: str = "PartialClose",
        deviation: int = 20,
    ) -> dict:
        """Cierra parcialmente una posicion abierta y devuelve un resultado normalizado."""
        self._connect()

        side_upper = str(side).upper()
        if side_upper not in {"BUY", "SELL"}:
            raise ValueError(f"Side no soportado para cierre parcial: {side}")

        symbol_spec = self.get_symbol_spec(symbol)
        tick = self.get_symbol_tick(symbol)
        if not symbol_spec or tick is None:
            return {
                "success": False,
                "retcode": None,
                "comment": f"No se pudo obtener spec/tick para {symbol}",
                "request": None,
                "closed_volume": 0.0,
            }

        normalized_volume = self._normalize_volume(volume, symbol_spec)
        if normalized_volume <= 0:
            return {
                "success": False,
                "retcode": None,
                "comment": f"Volumen parcial invalido para {symbol}: {volume}",
                "request": None,
                "closed_volume": 0.0,
            }

        digits = int(symbol_spec.get("digits", 5) or 5)
        order_type = mt5.ORDER_TYPE_SELL if side_upper == "BUY" else mt5.ORDER_TYPE_BUY
        price = self._normalize_price(
            float(tick["bid"] if side_upper == "BUY" else tick["ask"]),
            digits,
        )

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(normalized_volume),
            "type": order_type,
            "position": int(position_ticket),
            "price": price,
            "deviation": int(deviation),
            "comment": str(comment)[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        res = mt5.order_send(request)
        if res is None:
            return {
                "success": False,
                "retcode": None,
                "comment": f"order_send retorno None: {mt5.last_error()}",
                "request": request,
                "closed_volume": 0.0,
            }

        raw = res._asdict()
        return {
            "success": raw.get("retcode") == mt5.TRADE_RETCODE_DONE,
            "retcode": raw.get("retcode"),
            "comment": raw.get("comment"),
            "order": raw.get("order"),
            "deal": raw.get("deal"),
            "request": request,
            "closed_volume": float(normalized_volume),
        }

    def get_opened_positions(self, par: Optional[str] = None):
        try:
            pos = mt5.positions_get()
            if not pos:
                return 0, pd.DataFrame()
            df = pd.DataFrame(list(pos), columns=pos[0]._asdict().keys())
            if par:
                df = df[df['symbol'] == par]
            return len(df), df
        except Exception:
            return 0, pd.DataFrame()

    def get_all_positions(self) -> pd.DataFrame:
        try:
            pos = mt5.positions_get()
            if not pos: return pd.DataFrame()
            return pd.DataFrame(list(pos), columns=pos[0]._asdict().keys())
        except Exception:
            return pd.DataFrame()

    def get_history_deals(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        symbol: Optional[str] = None,
        magic: Optional[int] = None,
        position_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """Obtiene deals históricos normalizados desde MT5."""
        self._connect()

        # El servidor MT5 puede estar adelantado respecto al reloj local.
        # Si no se expande la ventana, los deals mas recientes pueden quedar fuera.
        date_to = date_to or (datetime.now() + timedelta(days=1))
        date_from = date_from or (date_to - timedelta(days=30))

        deals = mt5.history_deals_get(date_from, date_to)
        if not deals:
            return pd.DataFrame()

        df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())

        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)

        if symbol is not None and "symbol" in df.columns:
            df = df[df["symbol"] == symbol]

        if magic is not None and "magic" in df.columns:
            df = df[pd.to_numeric(df["magic"], errors="coerce").fillna(0).astype(int) == int(magic)]

        if position_id is not None and "position_id" in df.columns:
            df = df[pd.to_numeric(df["position_id"], errors="coerce").fillna(-1).astype(int) == int(position_id)]

        entry_map = {0: "IN", 1: "OUT", 2: "INOUT", 3: "OUT_BY"}
        reason_map = {
            0: "CLIENT",
            1: "MOBILE",
            2: "WEB",
            3: "EXPERT",
            4: "SL",
            5: "TP",
            6: "SO",
            7: "ROLLOVER",
            8: "VMARGIN",
            9: "SPLIT",
        }

        if "entry" in df.columns:
            df["entry_label"] = df["entry"].map(entry_map).fillna(df["entry"])
        if "reason" in df.columns:
            df["reason_label"] = df["reason"].map(reason_map).fillna(df["reason"])

        if "time" in df.columns:
            df = df.sort_values("time")

        return df.reset_index(drop=True)

    def send_to_breakeven(self, df_pos: pd.DataFrame, perc_rec: float) -> None:
        """
        Lleva a break-even las operaciones que ya recorrieron perc_rec% hacia su TP.
        """
        if df_pos is None or df_pos.empty:
            print('No hay operaciones abiertas')
            return
        for ticket in df_pos['ticket'].tolist():
            row = df_pos.loc[df_pos['ticket'] == ticket].iloc[0]
            symb = row['symbol']
            price_open = row['price_open']
            tp = row['tp']
            price_now = row['price_current']
            side = row['type']  # 0=buy, 1=sell
            # progreso hacia TP
            if side == 0:  # buy
                total = tp - price_open
                done = price_now - price_open
            else:          # sell
                total = price_open - tp
                done = price_open - price_now
            if total <= 0: 
                continue
            progreso = (done / total) * 100.0
            if progreso >= perc_rec:
                # mueve SL a BE
                type_order = mt5.ORDER_TYPE_BUY if side == 0 else mt5.ORDER_TYPE_SELL
                self.modify_orders(symb, ticket, stop_loss=price_open, take_profit=tp, type_order=type_order)

    def calculate_position_size(self, symbol: str, price_sl: float, risk_pct: float) -> float:
        """
        Calcula lotaje en función de distancia al SL y % de riesgo.
        price_sl: precio del stop loss
        risk_pct: 0.02 => 2%
        """
        mt5.symbol_select(symbol, True)
        sym_tick = mt5.symbol_info_tick(symbol)
        sym_info = mt5.symbol_info(symbol)
        if sym_tick is None or sym_info is None:
            return 0.01
        mid = (sym_tick.bid + sym_tick.ask) / 2
        tick_size = sym_info.trade_tick_size
        tick_value = sym_info.trade_tick_value
        balance = mt5.account_info().balance
        ticks_at_risk = abs(mid - price_sl) / max(tick_size, 1e-12)
        if ticks_at_risk <= 0 or tick_value <= 0:
            return 0.01
        pos_size = (balance * risk_pct) / (ticks_at_risk * tick_value)
        return round(max(pos_size, 0.01), 2)

    # ---------- calendario (web scraping sencillo) ----------
    def get_today_calendar(self) -> pd.DataFrame:
        """Regresa un DataFrame con columnas: currency, time, intensity (0..3)"""
        r = Request('https://es.investing.com/economic-calendar/', headers={'User-Agent': 'Mozilla/5.0'})
        response = urlopen(r).read()
        soup = BeautifulSoup(response, "html.parser")
        table = soup.find_all(class_="js-event-item")
        base = {}
        for bl in table:
            try:
                time = bl.find(class_="first left time js-time").text.strip()
                currency = bl.find(class_="left flagCur noWrap").text.split(' ')[1]
                full = bl.find_all(class_="left textNum sentiment noWrap")
                intensity = 0
                for ele in full:
                    bulls = ele.find_all(class_="grayFullBullishIcon")
                    if len(bulls) == 1: intensity = max(intensity, 1)
                    elif len(bulls) == 2: intensity = max(intensity, 2)
                    elif len(bulls) == 3: intensity = max(intensity, 3)
                base[f"{currency}_{time}"] = {"currency": currency, "time": time, "intensity": intensity}
            except Exception:
                continue
        return pd.DataFrame.from_dict(base, orient="index").reset_index(drop=True)
