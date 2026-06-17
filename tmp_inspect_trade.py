import pandas as pd
from pathlib import Path
import math
p = Path('data/cache/EURUSD_M5_8000.pkl')
df = pd.read_pickle(p)
df = df.copy()
df.index = pd.to_datetime(df.index)
entry_ts = pd.Timestamp('2026-05-14 16:55:00')
window = df.loc[entry_ts - pd.Timedelta(minutes=15): entry_ts + pd.Timedelta(minutes=45), ['Open','High','Low','Close']]
print(window.to_string())
