import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
D = lambda _f: _os.path.join(_ROOT, "data", _f)      # price data (inputs)
R = lambda _f: _os.path.join(_ROOT, "results", _f)   # generated outputs

import os
import pandas as pd
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

qld = load_or_download(D("qld_yf.csv"), "QLD")
sso = load_or_download(D("sso_yf.csv"), "SSO")

print(f"\nQLD: {qld.index.min()} to {qld.index.max()} ({len(qld)} rows)")
print(f"SSO: {sso.index.min()} to {sso.index.max()} ({len(sso)} rows)")