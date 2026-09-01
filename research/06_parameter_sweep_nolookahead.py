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
        return pd.read_csv(extended_path, index_col="Date", parse_dates=True)
    elif os.path.exists(plain_path):
        print(f"{label}: WARNING — using real-only data")
        return pd.read_csv(plain_path, index_col="Date", parse_dates=True)
    else:
        raise FileNotFoundError(f"Neither {extended_path} nor {plain_path} found for {label}.")

spy = pd.read_csv(D("spy_yf.csv"), index_col="Date", parse_dates=True)
qqq = pd.read_csv(D("qqq_yf.csv"), index_col="Date", parse_dates=True)
tqqq = load_best_available(D("tqqq_extended_yf.csv"), D("tqqq_yf.csv"), "TQQQ")
spxl = load_best_available(D("spxl_extended_yf.csv"), D("spxl_yf.csv"), "SPXL")

spy = spy.dropna(subset=['Open', 'Close'])
qqq = qqq.dropna(subset=['Open', 'Close'])
tqqq = tqqq.dropna(subset=['Open', 'Close'])
spxl = spxl.dropna(subset=['Open', 'Close'])

common_dates = spy.index.intersection(qqq.index).intersection(tqqq.index).intersection(spxl.index)
print(f"Shared common date range: {common_dates.min().date()} to {common_dates.max().date()} ({len(common_dates)} rows)")

spy = spy.loc[common_dates]
qqq = qqq.loc[common_dates]
tqqq = tqqq.loc[common_dates]
spxl = spxl.loc[common_dates]

ma_period = 200
init_cash = 10_000.0

spy_sma = ta.sma(spy['Close'], length=ma_period)
qqq_sma = ta.sma(qqq['Close'], length=ma_period)

# ============================================================
# Bias-corrected strategy WITH a hard safety check:
# entry_threshold >= exit_threshold  <=>  entry_pct + exit_pct >= 0
# ============================================================
def run_strategy_no_lookahead(signal_close, signal_sma, leveraged_open, fallback_open,
                                entry_pct, exit_pct, dca_days):
    if entry_pct + exit_pct < -1e-9:
        return None, None

    s_close_lagged = signal_close.shift(1).values
    s_sma_lagged = signal_sma.shift(1).values
    l_open = leveraged_open.values
    f_open = fallback_open.values

    cash = init_cash
    lev_shares = 0.0
    fallback_shares = 0.0
    state = "CASH"
    dca_day_counter = 0
    equity_curve = np.empty(len(s_close_lagged))
    trade_count = 0

    for i in range(len(s_close_lagged)):
        s_price = s_close_lagged[i]
        s_sma_val = s_sma_lagged[i]
        l_price = l_open[i]
        f_price = f_open[i]

        if np.isnan(s_sma_val) or np.isnan(s_price) or np.isnan(l_price) or np.isnan(f_price):
            equity_curve[i] = cash + lev_shares * (l_price if not np.isnan(l_price) else 0) + fallback_shares * (f_price if not np.isnan(f_price) else 0)
            continue

        entry_threshold = s_sma_val * (1 + entry_pct)
        exit_threshold = s_sma_val * (1 - exit_pct)

        if state == "CASH":
            if s_price > entry_threshold:
                lev_shares = cash / l_price; cash = 0.0; state = "FULL_LEV"
                trade_count += 1
        elif state == "FULL_LEV":
            if s_price < exit_threshold:
                cash = lev_shares * l_price; lev_shares = 0.0
                state = "DCA"; dca_day_counter = 0
                trade_count += 1
        elif state == "DCA":
            if s_price > entry_threshold:
                cash += fallback_shares * f_price; fallback_shares = 0.0
                lev_shares = cash / l_price; cash = 0.0
                state = "FULL_LEV"
                trade_count += 1
            else:
                dca_day_counter += 1
                if cash > 0:
                    remaining_days = max(1, dca_days - dca_day_counter + 1)
                    daily_buy = min(cash / remaining_days, cash)
                    fallback_shares += daily_buy / f_price
                    cash -= daily_buy
                    trade_count += 1
                if dca_day_counter >= dca_days:
                    state = "DELEVERAGED"
        elif state == "DELEVERAGED":
            if s_price > entry_threshold:
                cash = fallback_shares * f_price; fallback_shares = 0.0
                lev_shares = cash / l_price; cash = 0.0
                state = "FULL_LEV"
                trade_count += 1

        equity_curve[i] = cash + lev_shares * l_price + fallback_shares * f_price

    equity = pd.Series(equity_curve, index=signal_close.index)
    return equity, trade_count

def calc_stats(equity):
    if equity is None:
        return np.nan, np.nan
    equity_clean = equity.dropna()
    if len(equity_clean) < 2:
        return np.nan, np.nan
    ret = (equity_clean.iloc[-1] / equity_clean.iloc[0] - 1) * 100
    dd = ((equity_clean - equity_clean.cummax()) / equity_clean.cummax() * 100).min()
    return ret, dd

# ============================================================
# Sweep grid — single correct condition: entry_pct + exit_pct >= 0
# ============================================================
entry_range = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
exit_range = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
dca_range = [126, 189, 252, 315, 378, 441, 504]

def run_sweep(signal_close, signal_sma, leveraged_open, fallback_open, label):
    results = []
    skipped = 0
    for dca_days in dca_range:
        for entry_pct in entry_range:
            for exit_pct in exit_range:
                is_valid = entry_pct + exit_pct >= -1e-9
                if not is_valid:
                    results.append({
                        "entry_pct": entry_pct, "exit_pct": exit_pct, "dca_days": dca_days,
                        "return_pct": np.nan, "max_dd_pct": np.nan, "num_trades": np.nan
                    })
                    skipped += 1
                    continue
                eq, trades = run_strategy_no_lookahead(
                    signal_close, signal_sma, leveraged_open, fallback_open,
                    entry_pct, exit_pct, dca_days
                )
                ret, dd = calc_stats(eq)
                results.append({
                    "entry_pct": entry_pct, "exit_pct": exit_pct, "dca_days": dca_days,
                    "return_pct": ret, "max_dd_pct": dd, "num_trades": trades if trades is not None else np.nan
                })
    print(f"{label}: {len(entry_range)*len(exit_range)*len(dca_range) - skipped} valid combos, {skipped} skipped")
    return pd.DataFrame(results)

print("\nRunning sweep for TQQQ/QQQ...")
sweep_tqqq = run_sweep(qqq['Close'], qqq_sma, tqqq['Open'], qqq['Open'], "TQQQ/QQQ")

print("Running sweep for SPXL/SPY...")
sweep_spxl = run_sweep(spy['Close'], spy_sma, spxl['Open'], spy['Open'], "SPXL/SPY")

sweep_tqqq.to_csv(R("sweep_tqqq_qqq_nolookahead.csv"), index=False)
sweep_spxl.to_csv(R("sweep_spxl_spy_nolookahead.csv"), index=False)

# ============================================================
# Combined heatmaps
# ============================================================
def plot_combined_heatmaps(sweep_df, title_prefix):
    metrics = ["return_pct", "max_dd_pct", "num_trades"]
    metric_labels = ["Return %", "Max Drawdown %", "# Trades"]

    fig = make_subplots(
        rows=len(metrics), cols=len(dca_range),
        subplot_titles=[f"DCA={d}d" for d in dca_range] * len(metrics),
        shared_yaxes=True,
        vertical_spacing=0.06,
        row_titles=metric_labels
    )

    for row_idx, metric in enumerate(metrics, start=1):
        for col_idx, dca_days in enumerate(dca_range, start=1):
            subset = sweep_df[sweep_df['dca_days'] == dca_days]
            pivot = subset.pivot(index='exit_pct', columns='entry_pct', values=metric)
            fig.add_trace(
                go.Heatmap(
                    z=pivot.values, x=pivot.columns, y=pivot.index,
                    colorscale="RdYlGn",
                    showscale=(col_idx == len(dca_range)),
                    colorbar=dict(
                        len=0.28,
                        y=1 - (row_idx - 0.5) / len(metrics),
                        title=metric_labels[row_idx - 1]
                    ) if col_idx == len(dca_range) else None,
                    connectgaps=False
                ),
                row=row_idx, col=col_idx
            )

    fig.update_layout(
        title=f"{title_prefix} — Combined Sensitivity (entry_pct + exit_pct >= 0)",
        height=350 * len(metrics), width=1900
    )
    for col_idx in range(1, len(dca_range) + 1):
        fig.update_xaxes(title_text="Entry %", row=len(metrics), col=col_idx)
    for row_idx in range(1, len(metrics) + 1):
        fig.update_yaxes(title_text="Exit %", row=row_idx, col=1)

    fig.show()

plot_combined_heatmaps(sweep_tqqq, "TQQQ/QQQ")
plot_combined_heatmaps(sweep_spxl, "SPXL/SPY")

def show_top_results(sweep_df, label, n=10):
    valid = sweep_df.dropna(subset=['return_pct'])

    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")

    print(f"\n--- Top {n} by Return ---")
    top_return = valid.sort_values('return_pct', ascending=False).head(n)
    print(top_return[['entry_pct','exit_pct','dca_days','return_pct','max_dd_pct','num_trades']].to_string(index=False))

    print(f"\n--- Top {n} by Drawdown (least negative = safest) ---")
    top_dd = valid.sort_values('max_dd_pct', ascending=False).head(n)
    print(top_dd[['entry_pct','exit_pct','dca_days','return_pct','max_dd_pct','num_trades']].to_string(index=False))

    print(f"\n--- Top {n} by Return-per-unit-Drawdown (rough risk-adjusted proxy) ---")
    valid = valid.copy()
    valid['return_per_dd'] = valid['return_pct'] / valid['max_dd_pct'].abs()
    top_riskadj = valid.sort_values('return_per_dd', ascending=False).head(n)
    print(top_riskadj[['entry_pct','exit_pct','dca_days','return_pct','max_dd_pct','num_trades','return_per_dd']].to_string(index=False))

    print(f"\n--- Lowest {n} trade count (among top-half return performers, for robustness) ---")
    median_return = valid['return_pct'].median()
    above_median = valid[valid['return_pct'] >= median_return]
    low_trades = above_median.sort_values('num_trades').head(n)
    print(low_trades[['entry_pct','exit_pct','dca_days','return_pct','max_dd_pct','num_trades']].to_string(index=False))

show_top_results(sweep_tqqq, "TQQQ/QQQ")
show_top_results(sweep_spxl, "SPXL/SPY")