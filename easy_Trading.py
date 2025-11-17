# ==========================================
# conexion/easy_Trading.py  (versión unificada y corregida)
# ==========================================
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
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
                      type_order=mt5.ORDER_TYPE_BUY) -> None:
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
        mt5.order_send(req)

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

    def obtener_ordenes_pendientes(self) -> pd.DataFrame:
        try:
            ordenes = mt5.orders_get()
            if not ordenes:
                return pd.DataFrame()
            return pd.DataFrame(list(ordenes), columns=ordenes[0]._asdict().keys())
        except Exception:
            return pd.DataFrame()

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
