import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
D = lambda _f: _os.path.join(_ROOT, "data", _f)      # price data (inputs)
R = lambda _f: _os.path.join(_ROOT, "results", _f)   # generated outputs

import os
import pandas as pd
import numpy as np
import pandas_ta_classic as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# Load data
# ============================================================
def load_best_available(extended_path, plain_path, label):
    if os.path.exists(extended_path):
        print(f"{label}: using extended series ({extended_path})")
        return pd.read_csv(extended_path, index_col="Date", parse_dates=True)
    elif os.path.exists(plain_path):
        print(f"{label}: WARNING — no extended series, using real-only data ({plain_path})")
        return pd.read_csv(plain_path, index_col="Date", parse_dates=True)
    else:
        raise FileNotFoundError(f"Neither {extended_path} nor {plain_path} found for {label}.")

spy = pd.read_csv(D("spy_yf.csv"), index_col="Date", parse_dates=True).dropna(subset=['Close'])
qqq = pd.read_csv(D("qqq_yf.csv"), index_col="Date", parse_dates=True).dropna(subset=['Close'])
tqqq = load_best_available(D("tqqq_extended_yf.csv"), D("tqqq_yf.csv"), "TQQQ").dropna(subset=['Close'])
spxl = load_best_available(D("spxl_extended_yf.csv"), D("spxl_yf.csv"), "SPXL").dropna(subset=['Close'])
qld = load_best_available(D("qld_extended_yf.csv"), D("qld_yf.csv"), "QLD").dropna(subset=['Close'])
sso = load_best_available(D("sso_extended_yf.csv"), D("sso_yf.csv"), "SSO").dropna(subset=['Close'])

# ============================================================
# Align all six assets to a single shared date range
# ============================================================
common_dates = spy.index.intersection(qqq.index).intersection(tqqq.index) \
                  .intersection(spxl.index).intersection(qld.index).intersection(sso.index)
print(f"\nShared common date range across all 6 assets: {common_dates.min().date()} to {common_dates.max().date()} ({len(common_dates)} rows)")

spy_a = spy.loc[common_dates]
qqq_a = qqq.loc[common_dates]
tqqq_a = tqqq.loc[common_dates]
spxl_a = spxl.loc[common_dates]
qld_a = qld.loc[common_dates]
sso_a = sso.loc[common_dates]

ma_period = 200
entry_pct = 0.04
exit_pct = 0.03
deleverage_pct = 0.30
dca_trading_days = 126
init_cash = 10_000.0

spy_sma = ta.sma(spy_a['Close'], length=ma_period)
qqq_sma = ta.sma(qqq_a['Close'], length=ma_period)

# ============================================================
# Generic signal-based state-machine strategy
# ============================================================
def run_signal_strategy(signal_price, signal_sma, leveraged_price, fallback_price, fallback_sma_for_deleverage):
    cash = init_cash
    lev_shares = 0.0
    fallback_shares = 0.0
    state = "CASH"
    dca_day_counter = 0
    equity_curve = []

    for i in range(len(signal_price)):
        s_price = signal_price.iloc[i]
        l_price = leveraged_price.iloc[i]
        f_price = fallback_price.iloc[i]
        s_sma_val = signal_sma.iloc[i]
        deleverage_ref_val = fallback_sma_for_deleverage.iloc[i]

        if pd.isna(s_sma_val) or pd.isna(deleverage_ref_val):
            equity_curve.append(cash + lev_shares * l_price + fallback_shares * f_price)
            continue

        entry_threshold = s_sma_val * (1 + entry_pct)
        exit_threshold = s_sma_val * (1 - exit_pct)
        deleverage_threshold = deleverage_ref_val * (1 + deleverage_pct)

        if state == "CASH":
            if s_price > entry_threshold:
                lev_shares = cash / l_price
                cash = 0.0
                state = "FULL_LEV"

        elif state == "FULL_LEV":
            if f_price > deleverage_threshold:
                cash = lev_shares * l_price
                lev_shares = 0.0
                fallback_shares = cash / f_price
                cash = 0.0
                state = "DELEVERAGED"
            elif s_price < exit_threshold:
                cash = lev_shares * l_price
                lev_shares = 0.0
                state = "DCA"
                dca_day_counter = 0

        elif state == "DCA":
            if s_price > entry_threshold:
                cash += fallback_shares * f_price
                fallback_shares = 0.0
                lev_shares = cash / l_price
                cash = 0.0
                state = "FULL_LEV"
            else:
                dca_day_counter += 1
                if cash > 0:
                    remaining_days = max(1, dca_trading_days - dca_day_counter + 1)
                    daily_buy = min(cash / remaining_days, cash)
                    fallback_shares += daily_buy / f_price
                    cash -= daily_buy
                if dca_day_counter >= dca_trading_days:
                    state = "DELEVERAGED"

        elif state == "DELEVERAGED":
            if s_price > entry_threshold:
                cash = fallback_shares * f_price
                fallback_shares = 0.0
                lev_shares = cash / l_price
                cash = 0.0
                state = "FULL_LEV"

        equity_curve.append(cash + lev_shares * l_price + fallback_shares * f_price)

    return pd.Series(equity_curve, index=signal_price.index)

def calc_drawdown(equity):
    return (equity - equity.cummax()) / equity.cummax() * 100

def summarize(equity, dd, label):
    ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    print(f"{label:<32}{ret:>14.2f}{dd.min():>14.2f}")

# ============================================================
# Run strategies: 3x variants + new 2x variants
# ============================================================
eq_qqq_signal_tqqq = run_signal_strategy(qqq_a['Close'], qqq_sma, tqqq_a['Close'], qqq_a['Close'], qqq_sma)
eq_spy_signal_spxl = run_signal_strategy(spy_a['Close'], spy_sma, spxl_a['Close'], spy_a['Close'], spy_sma)
eq_qqq_signal_qld = run_signal_strategy(qqq_a['Close'], qqq_sma, qld_a['Close'], qqq_a['Close'], qqq_sma)
eq_spy_signal_sso = run_signal_strategy(spy_a['Close'], spy_sma, sso_a['Close'], spy_a['Close'], spy_sma)

bh_tqqq = (init_cash / tqqq_a['Close'].iloc[0]) * tqqq_a['Close']
bh_spxl = (init_cash / spxl_a['Close'].iloc[0]) * spxl_a['Close']
bh_qld = (init_cash / qld_a['Close'].iloc[0]) * qld_a['Close']
bh_sso = (init_cash / sso_a['Close'].iloc[0]) * sso_a['Close']

results = [
    (eq_qqq_signal_tqqq, "QQQ-Signal (TQQQ/QQQ) [3x]"),
    (eq_spy_signal_spxl, "SPY-Signal (SPXL/SPY) [3x]"),
    (eq_qqq_signal_qld, "QQQ-Signal (QLD/QQQ) [2x]"),
    (eq_spy_signal_sso, "SPY-Signal (SSO/SPY) [2x]"),
    (bh_tqqq, "Buy & Hold TQQQ"),
    (bh_spxl, "Buy & Hold SPXL"),
    (bh_qld, "Buy & Hold QLD"),
    (bh_sso, "Buy & Hold SSO"),
]

print(f"\nAll series start: {common_dates.min().date()}")
print(f"{'Strategy':<32}{'Return %':>14}{'Max DD %':>14}")
for eq, label in results:
    dd = calc_drawdown(eq)
    summarize(eq, dd, label)

# ============================================================
# Chart
# ============================================================
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.7, 0.3], vertical_spacing=0.05,
    subplot_titles=("2x vs 3x Strategy Comparison (log scale)", "Drawdown Comparison")
)

colors = ["steelblue", "darkred", "seagreen", "goldenrod", "lightblue", "salmon", "lightgreen", "khaki"]
for (eq, label), color in zip(results, colors):
    dd = calc_drawdown(eq)
    dash = "dot" if "Buy & Hold" in label else "solid"
    fig.add_trace(go.Scatter(x=eq.index, y=eq, name=label, line=dict(color=color, width=1.5, dash=dash)), row=1, col=1)
    if "Buy & Hold" not in label:
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=f"{label} DD", line=dict(color=color, width=1)), row=2, col=1)

fig.update_yaxes(type="log", row=1, col=1)
fig.update_layout(
    title="2x (QLD/SSO) vs 3x (TQQQ/SPXL) Strategy Comparison",
    height=850, width=1400, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig.show()