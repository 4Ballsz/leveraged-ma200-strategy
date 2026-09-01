import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
D = lambda _f: _os.path.join(_ROOT, "data", _f)      # price data (inputs)
R = lambda _f: _os.path.join(_ROOT, "results", _f)   # generated outputs

import os
import pandas as pd
import numpy as np
import yfinance as yf

def load_or_download(filepath, ticker, start="1993-01-01"):
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath, index_col="Date", parse_dates=True)
        last_date = existing.index.max()
        days_old = (pd.Timestamp.today().normalize() - last_date).days
        if days_old <= 3:
            print(f"{filepath} is up to date. Loading from disk...")
            return existing
        else:
            print(f"{filepath} is stale. Re-downloading...")

    print(f"Downloading {ticker}...")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=False)
    raw['Close'] = raw['Adj Close']
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    data = raw[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.index.name = "Date"
    data.to_csv(filepath)
    return data

spy = load_or_download(D("spy_yf.csv"), "SPY")
spxl = load_or_download(D("spxl_yf.csv"), "SPXL")

spy = spy.dropna(subset=['Close'])
spxl = spxl.dropna(subset=['Close'])

# --- Build synthetic pre-inception SPXL ---
leverage = 3
expense_ratio_annual = 0.0091   # SPXL's actual expense ratio (~0.91%)
financing_spread_annual = 0.01  # same rough financing-cost estimate used for TQQQ
daily_drag = (expense_ratio_annual + financing_spread_annual) / 252

spy_returns = spy['Close'].pct_change()
synthetic_spxl_returns = leverage * spy_returns - daily_drag

spxl_start_date = spxl.index.min()
spxl_start_price = spxl['Close'].iloc[0]
print(f"\nSPXL actual inception: {spxl_start_date}, starting price: {spxl_start_price:.4f}")

pre_inception_dates = spy.index[spy.index < spxl_start_date]
pre_inception_returns = synthetic_spxl_returns.loc[pre_inception_dates].dropna()

# Cumulative product from each day forward to inception, anchored backward from real start price
cum_factor = (1 + pre_inception_returns.iloc[::-1]).cumprod().iloc[::-1]
synthetic_prices = spxl_start_price / cum_factor

synthetic_df = pd.DataFrame(index=synthetic_prices.index)
synthetic_df['Close'] = synthetic_prices
synthetic_df['Open'] = synthetic_df['Close'].shift(1).fillna(synthetic_df['Close'].iloc[0])
synthetic_df['High'] = synthetic_df[['Open', 'Close']].max(axis=1)
synthetic_df['Low'] = synthetic_df[['Open', 'Close']].min(axis=1)
synthetic_df['Volume'] = 0

# --- Splice synthetic + real ---
spxl_extended = pd.concat([synthetic_df, spxl]).sort_index()
spxl_extended = spxl_extended[~spxl_extended.index.duplicated(keep='last')]
spxl_extended = spxl_extended.dropna(subset=['Close'])

print(f"\nExtended SPXL series: {spxl_extended.index.min()} to {spxl_extended.index.max()} ({len(spxl_extended)} rows)")
print("\nAround the splice point:")
print(spxl_extended.loc[spxl_start_date - pd.Timedelta(days=5):spxl_start_date + pd.Timedelta(days=5)])

spxl_extended.to_csv(D("spxl_extended_yf.csv"))
print("\nSaved: spxl_extended_yf.csv")