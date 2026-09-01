import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
D = lambda _f: _os.path.join(_ROOT, "data", _f)      # price data (inputs)
R = lambda _f: _os.path.join(_ROOT, "results", _f)   # generated outputs

import pandas as pd
import numpy as np

# --- Load existing files ---
qqq = pd.read_csv(D("qqq_yf.csv"), index_col="Date", parse_dates=True).dropna(subset=['Close'])
spy = pd.read_csv(D("spy_yf.csv"), index_col="Date", parse_dates=True).dropna(subset=['Close'])
qld = pd.read_csv(D("qld_yf.csv"), index_col="Date", parse_dates=True).dropna(subset=['Close'])
sso = pd.read_csv(D("sso_yf.csv"), index_col="Date", parse_dates=True).dropna(subset=['Close'])

def build_synthetic_extension(underlying_df, leveraged_df, leverage, expense_ratio_annual, financing_spread_annual, label):
    daily_drag = (expense_ratio_annual + financing_spread_annual) / 252
    underlying_returns = underlying_df['Close'].pct_change()
    synthetic_returns = leverage * underlying_returns - daily_drag

    lev_start_date = leveraged_df.index.min()
    lev_start_price = leveraged_df['Close'].iloc[0]
    print(f"{label} actual inception: {lev_start_date}, starting price: {lev_start_price:.4f}")

    pre_inception_dates = underlying_df.index[underlying_df.index < lev_start_date]
    pre_inception_returns = synthetic_returns.loc[pre_inception_dates].dropna()

    cum_factor = (1 + pre_inception_returns.iloc[::-1]).cumprod().iloc[::-1]
    synthetic_prices = lev_start_price / cum_factor

    synthetic_df = pd.DataFrame(index=synthetic_prices.index)
    synthetic_df['Close'] = synthetic_prices
    synthetic_df['Open'] = synthetic_df['Close'].shift(1).fillna(synthetic_df['Close'].iloc[0])
    synthetic_df['High'] = synthetic_df[['Open', 'Close']].max(axis=1)
    synthetic_df['Low'] = synthetic_df[['Open', 'Close']].min(axis=1)
    synthetic_df['Volume'] = 0

    extended = pd.concat([synthetic_df, leveraged_df]).sort_index()
    extended = extended[~extended.index.duplicated(keep='last')]
    extended = extended.dropna(subset=['Close'])

    print(f"{label} extended series: {extended.index.min()} to {extended.index.max()} ({len(extended)} rows)")
    return extended

# QLD: 2x QQQ
qld_extended = build_synthetic_extension(
    qqq, qld, leverage=2, expense_ratio_annual=0.0095, financing_spread_annual=0.01, label="QLD"
)
qld_extended.to_csv(D("qld_extended_yf.csv"))

# SSO: 2x SPY
sso_extended = build_synthetic_extension(
    spy, sso, leverage=2, expense_ratio_annual=0.0091, financing_spread_annual=0.01, label="SSO"
)
sso_extended.to_csv(D("sso_extended_yf.csv"))

print("\nSaved: qld_extended_yf.csv, sso_extended_yf.csv")