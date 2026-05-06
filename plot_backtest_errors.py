"""
plot_backtest_errors.py

Lee un archivo *_series.csv generado por la pipeline de backtest
(p.ej. outputs/backtest/ARIMA_d-0_p-5_q-2_series.csv) y genera:

1) <base>_returns_with_errors.png
   - Arriba: y_true vs y_pred (retornos)
   - Abajo: error y |error|

2) <base>_prices_with_errors.png
   - Arriba: índice de precio rebajado real vs predicho
   - Abajo: error de precio y |error|

3) <base>_errors_only.png
   - Error y |error| sobre el tiempo (retornos)
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_series(series_path: str) -> pd.DataFrame:
    """Carga el *_series.csv y prepara columnas de error y de 'precio rebajado'."""
    df = pd.read_csv(series_path)

    # Detectar columna de fechas
    if "date" in df.columns:
        dt_col = "date"
    elif "timestamp" in df.columns:
        dt_col = "timestamp"
    else:
        raise ValueError("El archivo no tiene columna 'date' ni 'timestamp'.")

    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).set_index(dt_col)

    # Asegurar columnas y_true / y_pred
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError("El archivo debe tener columnas 'y_true' y 'y_pred'.")

    # Error en retornos
    if "error" not in df.columns:
        df["error"] = df["y_true"] - df["y_pred"]
    df["abs_error"] = df["error"].abs()

    # Índice de precios rebajado (base 1)
    base_price = 1.0
    df["price_true"] = (1.0 + df["y_true"]).cumprod() * base_price
    df["price_pred"] = (1.0 + df["y_pred"]).cumprod() * base_price
    df["price_error"] = df["price_true"] - df["price_pred"]
    df["price_abs_error"] = df["price_error"].abs()

    return df


def plot_returns_and_errors(df: pd.DataFrame, out_path: str) -> None:
    """Gráfico 1: retornos reales vs predichos + errores en la misma figura."""
    mae = df["abs_error"].mean()
    rmse = np.sqrt((df["error"] ** 2).mean())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Arriba: serie de retornos
    ax1.plot(df.index, df["y_true"], label="Real", alpha=0.8)
    ax1.plot(df.index, df["y_pred"], label="Predicho", alpha=0.8)
    ax1.set_title("Real vs Predicho (retornos)")
    ax1.set_ylabel("Return_1")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Abajo: errores en retornos
    ax2.plot(df.index, df["error"], label="Error (Real - Predicho)", alpha=0.8)
    ax2.plot(df.index, df["abs_error"], "--", label="|Error|", alpha=0.8)
    ax2.axhline(0.0, color="black", linewidth=1, linestyle=":")
    ax2.axhline(mae, color="grey", linestyle="--", label=f"MAE = {mae:.6f}")
    ax2.axhline(rmse, color="grey", linestyle="-.", label=f"RMSE = {rmse:.6f}")
    ax2.set_xlabel("Fecha")
    ax2.set_ylabel("Error")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prices_and_errors(df: pd.DataFrame, out_path: str) -> None:
    """Gráfico 2: índice de precios rebajado real vs predicho + errores."""
    mae_p = df["price_abs_error"].mean()
    rmse_p = np.sqrt((df["price_error"] ** 2).mean())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Arriba: "precios" rebajados (índice)
    ax1.plot(df.index, df["price_true"], label="Precio real (rebajado)", alpha=0.8)
    ax1.plot(df.index, df["price_pred"], label="Precio predicho (rebajado)", alpha=0.8)
    ax1.set_title("Real vs Predicho (precio rebajado)")
    ax1.set_ylabel("Índice de precio (base = 1)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Abajo: errores de precio
    ax2.plot(df.index, df["price_error"], label="Error de precio", alpha=0.8)
    ax2.plot(
        df.index,
        df["price_abs_error"],
        "--",
        label="|Error precio|",
        alpha=0.8,
    )
    ax2.axhline(0.0, color="black", linewidth=1, linestyle=":")
    ax2.axhline(mae_p, color="grey", linestyle="--", label=f"MAE precio = {mae_p:.6f}")
    ax2.axhline(
        rmse_p, color="grey", linestyle="-.", label=f"RMSE precio = {rmse_p:.6f}"
    )
    ax2.set_xlabel("Fecha")
    ax2.set_ylabel("Error precio")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_errors_only(df: pd.DataFrame, out_path: str) -> None:
    """Gráfico 3: sólo errores en retornos (como el que ya tenías, pero aislado)."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df["error"], label="Error (Real - Predicho)", alpha=0.8)
    ax.plot(df.index, df["abs_error"], "--", label="|Error|", alpha=0.8)
    ax.axhline(0.0, color="black", linewidth=1, linestyle=":")
    ax.set_title("Errores de backtest (retornos)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    # Ruta por defecto si no se pasa argumento:
    default_series = os.path.join(
        "outputs", "backtest", "ARIMA_d-0_p-5_q-2_series.csv"
    )
    series_path = sys.argv[1] if len(sys.argv) > 1 else default_series

    df = load_series(series_path)

    base_name = os.path.splitext(os.path.basename(series_path))[0]
    out_dir = os.path.dirname(series_path)

    path_returns = os.path.join(out_dir, base_name + "_returns_with_errors.png")
    path_prices = os.path.join(out_dir, base_name + "_prices_with_errors.png")
    path_errors = os.path.join(out_dir, base_name + "_errors_only.png")

    plot_returns_and_errors(df, path_returns)
    plot_prices_and_errors(df, path_prices)
    plot_errors_only(df, path_errors)

    print("Gráficos guardados en:")
    print("  -", path_returns)
    print("  -", path_prices)
    print("  -", path_errors)


if __name__ == "__main__":
    main()
