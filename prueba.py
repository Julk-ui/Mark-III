import pandas as pd
import numpy as np

df = pd.read_csv("outputs/backtest/ARIMA_d-0_p-5_q-2_series.csv")

y_true = df["y_true"].values
y_pred = df["y_pred"].values

# mismo umbral que uses en config (en retornos, no en pips)
threshold = 0.0  

pred_for_trading = y_pred.copy()
mask_small = np.abs(pred_for_trading) < threshold
pred_for_trading[mask_small] = 0.0

true_sign = np.sign(y_true)
pred_sign = np.sign(pred_for_trading)

mask_valid = pred_sign != 0
hit_rate = (true_sign[mask_valid] == pred_sign[mask_valid]).mean() * 100

print("Hit rate manual:", hit_rate)
