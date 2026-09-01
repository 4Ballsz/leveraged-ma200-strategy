import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
D = lambda _f: _os.path.join(_ROOT, "data", _f)      # price data (inputs)
R = lambda _f: _os.path.join(_ROOT, "results", _f)   # generated outputs

import yfinance as yf
import pandas as pd
import numpy as np

# --- Download real data ---
print("Downloading QQQ...")
qqq_raw = yf.download("QQQ", start="1993-01-01", end="2026-08-25", auto_adjust=False)
qqq_raw['Close'] = qqq_raw['Adj Close']
if isinstance(qqq_raw.columns, pd.MultiIndex):
    qqq_raw.columns = qqq_raw.columns.get_level_values(0)
qqq = qqq_raw[['Open', 'High', 'Low', 'Close', 'Volume']]
qqq.index.name = "Date"

print("Downloading TQQQ...")
tqqq_raw = yf.download("TQQQ", start="1993-01-01", end="2026-08-25", auto_adjust=False)
tqqq_raw['Close'] = tqqq_raw['Adj Close']
if isinstance(tqqq_raw.columns, pd.MultiIndex):
    tqqq_raw.columns = tqqq_raw.columns.get_level_values(0)
tqqq = tqqq_raw[['Open', 'High', 'Low', 'Close', 'Volume']]
tqqq.index.name = "Date"

print(f"QQQ: {qqq.index.min()} to {qqq.index.max()} ({len(qqq)} rows)")
print(f"TQQQ: {tqqq.index.min()} to {tqqq.index.max()} ({len(tqqq)} rows)")

# --- Build synthetic pre-inception TQQQ ---
leverage = 3
expense_ratio_annual = 0.0095
financing_spread_annual = 0.01
daily_drag = (expense_ratio_annual + financing_spread_annual) / 252

qqq_returns = qqq['Close'].pct_change()
synthetic_tqqq_returns = leverage * qqq_returns - daily_drag

tqqq_start_date = tqqq.index.min()
tqqq_start_price = tqqq['Close'].iloc[0]
print(f"\nTQQQ actual inception: {tqqq_start_date}, starting price: {tqqq_start_price:.4f}")

pre_inception_dates = qqq.index[qqq.index < tqqq_start_date]
pre_inception_returns = synthetic_tqqq_returns.loc[pre_inception_dates]

# Cumulative product from each day forward to inception, then anchor backward from the real start price
cum_factor = (1 + pre_inception_returns.iloc[::-1]).cumprod().iloc[::-1]
synthetic_prices = tqqq_start_price / cum_factor

# Build synthetic OHLC (approximate High/Low/Open as scaled from Close since we don't have real intraday synthetic data)
synthetic_df = pd.DataFrame(index=pre_inception_dates)
synthetic_df['Close'] = synthetic_prices
synthetic_df['Open'] = synthetic_df['Close'].shift(1).fillna(synthetic_df['Close'].iloc[0])
synthetic_df['High'] = synthetic_df[['Open', 'Close']].max(axis=1)
synthetic_df['Low'] = synthetic_df[['Open', 'Close']].min(axis=1)
synthetic_df['Volume'] = 0

# --- Splice synthetic + real ---
tqqq_extended = pd.concat([synthetic_df, tqqq]).sort_index()
tqqq_extended = tqqq_extended[~tqqq_extended.index.duplicated(keep='last')]

print(f"\nExtended TQQQ series: {tqqq_extended.index.min()} to {tqqq_extended.index.max()} ({len(tqqq_extended)} rows)")
print("\nAround the splice point:")
print(tqqq_extended.loc[tqqq_start_date - pd.Timedelta(days=5):tqqq_start_date + pd.Timedelta(days=5)])

# --- Save everything ---
qqq.to_csv(D("qqq_yf.csv"))
tqqq.to_csv(D("tqqq_yf.csv"))
tqqq_extended.to_csv(D("tqqq_extended_yf.csv"))

print("\nSaved: qqq_yf.csv, tqqq_yf.csv, tqqq_extended_yf.csv")