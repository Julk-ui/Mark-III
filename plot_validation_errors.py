import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_validation_errors(
    path: str = "outputs/validation/validation_consolidated.xlsx"
):
    # Leer detalle de validación
    df = pd.read_excel(path, sheet_name="detail")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    # Errores en retornos
    df["error"]    = df["y_true"] - df["y_pred"]
    df["abs_err"]  = df["error"].abs()
    df["sq_err"]   = df["error"] ** 2

    mae  = df["abs_err"].mean()
    rmse = np.sqrt(df["sq_err"].mean())

    print(f"MAE  (cálculo directo) = {mae:.6f}")
    print(f"RMSE (cálculo directo) = {rmse:.6f}")

    # Gráfico: arriba retornos, abajo errores
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # 1) Retornos
    ax1.plot(df.index, df["y_true"], label="Retorno real", alpha=0.8)
    ax1.plot(df.index, df["y_pred"], label="Retorno predicho", alpha=0.8)
    ax1.set_title("Validación - Return_1 real vs predicho")
    ax1.set_ylabel("Return_1")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2) Errores absolutos
    ax2.plot(df.index, df["abs_err"], label="|error_t| = |y_t - ŷ_t|")
    ax2.axhline(mae,  linestyle="--", label=f"MAE = {mae:.6f}")
    ax2.axhline(rmse, linestyle=":",  label=f"RMSE = {rmse:.6f}")
    ax2.set_ylabel("Error en retornos")
    ax2.set_xlabel("Fecha")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    out_path = Path("outputs/validation/errors_validation.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico guardado en: {out_path}")

if __name__ == "__main__":
    plot_validation_errors()
