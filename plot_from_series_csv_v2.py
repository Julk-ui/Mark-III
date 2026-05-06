
"""
plot_from_series_csv_v2.py

Uso:
  python plot_from_series_csv_v2.py --csv outputs/backtest/series/ARIMA_xxx_series.csv
  python plot_from_series_csv_v2.py --csv outputs/backtest/series/LSTM_xxx_series.csv --out outputs/backtest/plots

Qué hace:
- Siempre grafica retornos: y_true vs y_pred
- Si existen columnas de precio (Close_t / Close_t1_real / Close_t1_pred) grafica:
    1) Precio t+1 real vs predicho (one-step, punto a punto)
    2) Trayectoria acumulada usando retornos (opcional) para ver “camino” completo
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _safe_dt(s):
    return pd.to_datetime(s, errors="coerce")


def _plot_returns(df: pd.DataFrame, out_path: Path, title: str):
    d = df.copy()
    d["Date"] = _safe_dt(d["Date"]) if "Date" in d.columns else pd.NaT
    d = d.sort_values("Date")

    plt.figure(figsize=(12, 5))
    plt.plot(d["Date"], d["y_true"], label="ReturnFwd_1 real")
    plt.plot(d["Date"], d["y_pred"], label="ReturnFwd_1 pred")
    plt.axhline(0, linewidth=1)
    plt.title(f"{title} | Retornos (ReturnFwd_1)")
    plt.xlabel("Fecha")
    plt.ylabel("Retorno")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_price_point(df: pd.DataFrame, out_path: Path, title: str):
    # Punto a punto (one-step): Close_t1_real vs Close_t1_pred (si existe)
    d = df.copy()
    if "Date_t1" in d.columns and d["Date_t1"].notna().any():
        x = _safe_dt(d["Date_t1"])
    else:
        x = _safe_dt(d["Date"])

    plt.figure(figsize=(12, 5))
    plt.plot(x, d["Close_t1_real"], label="Precio real (t+1)")
    plt.plot(x, d["Close_t1_pred"], label="Precio pred (t+1)")
    plt.title(f"{title} | Precio (t+1) punto a punto")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_price_cumulative(df: pd.DataFrame, out_path: Path, title: str):
    # Trayectoria acumulada partiendo del primer Close_t y encadenando retornos
    d = df.copy()
    d["Date"] = _safe_dt(d["Date"]) if "Date" in d.columns else pd.NaT
    d = d.sort_values("Date")

    if "Close_t" not in d.columns or d["Close_t"].isna().all():
        return

    start_price = float(d["Close_t"].dropna().iloc[0])

    # Construye fechas para trayectoria: [Date_0] + [Date_t1] o, si no existe, Date
    if "Date_t1" in d.columns and d["Date_t1"].notna().any():
        dates_path = [_safe_dt(d["Date"].iloc[0])] + list(_safe_dt(d["Date_t1"]))
    else:
        # aprox: usar Date como eje, dejando misma longitud
        dates_path = list(d["Date"])

    # Encadenar retornos
    real_path = [start_price]
    pred_path = [start_price]
    for r_true, r_pred in zip(d["y_true"].tolist(), d["y_pred"].tolist()):
        # Si viene NaN, no avanzamos (mantener último)
        if pd.isna(r_true):
            real_path.append(real_path[-1])
        else:
            real_path.append(real_path[-1] * (1.0 + float(r_true)))

        if pd.isna(r_pred):
            pred_path.append(pred_path[-1])
        else:
            pred_path.append(pred_path[-1] * (1.0 + float(r_pred)))

    # Ajuste si fechas y series no calzan
    min_len = min(len(dates_path), len(real_path), len(pred_path))
    dates_path = dates_path[:min_len]
    real_path = real_path[:min_len]
    pred_path = pred_path[:min_len]

    plt.figure(figsize=(12, 5))
    plt.plot(dates_path, real_path, label="Precio real (acumulado)")
    plt.plot(dates_path, pred_path, label="Precio pred (acumulado)")
    plt.title(f"{title} | Trayectoria de precio (acumulada)")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Ruta al *_series.csv")
    ap.add_argument("--out", default=None, help="Carpeta de salida (default: misma carpeta del CSV)")
    ap.add_argument("--prefix", default=None, help="Prefijo para nombre de archivos (default: nombre base del CSV)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe: {csv_path}")

    out_dir = Path(args.out) if args.out else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    base = args.prefix if args.prefix else csv_path.stem.replace("_series", "")
    title = base

    # 1) Returns
    out_returns = out_dir / f"{base}_returns.png"
    _plot_returns(df, out_returns, title)

    # 2) Price (si hay columnas)
    has_price = all(c in df.columns for c in ["Close_t1_real", "Close_t1_pred"]) and (
        df["Close_t1_real"].notna().any() or df["Close_t1_pred"].notna().any()
    )
    if has_price:
        out_price = out_dir / f"{base}_price_point.png"
        _plot_price_point(df, out_price, title)

    # 3) Cumulative path (si hay Close_t)
    if "Close_t" in df.columns and df["Close_t"].notna().any():
        out_cum = out_dir / f"{base}_price_cumulative.png"
        _plot_price_cumulative(df, out_cum, title)

    print("OK. Generados:")
    print(f" - {out_returns}")
    if has_price:
        print(f" - {out_price}")
    if "Close_t" in df.columns and df["Close_t"].notna().any():
        print(f" - {out_cum}")


if __name__ == "__main__":
    main()
