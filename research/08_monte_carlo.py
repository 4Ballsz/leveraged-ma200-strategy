import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
D = lambda _f: _os.path.join(_ROOT, "data", _f)      # price data (inputs)
R = lambda _f: _os.path.join(_ROOT, "results", _f)   # generated outputs

import os
import time
import pandas as pd
import numpy as np
import pandas_ta_classic as ta
import plotly.graph_objects as go

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

qqq = pd.read_csv(D("qqq_yf.csv"), index_col="Date", parse_dates=True).dropna(subset=['Open', 'Close'])
tqqq = load_best_available(D("tqqq_extended_yf.csv"), D("tqqq_yf.csv"), "TQQQ").dropna(subset=['Open', 'Close'])

common_dates = qqq.index.intersection(tqqq.index)
qqq = qqq.loc[common_dates]
tqqq = tqqq.loc[common_dates]

ma_period = 200
init_cash = 10_000.0

entry_range = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
exit_range = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
dca_range = [126, 189, 252, 315]

IN_SAMPLE_YEARS = 5
OUT_SAMPLE_YEARS = 1
TRADING_DAYS_PER_YEAR = 252
in_sample_len = IN_SAMPLE_YEARS * TRADING_DAYS_PER_YEAR
out_sample_len = OUT_SAMPLE_YEARS * TRADING_DAYS_PER_YEAR
MIN_NEIGHBORS_REQUIRED = 3

# ============================================================
# Core strategy function (position-continuous)
# ============================================================
def run_strategy(signal_close_lagged, signal_sma_lagged, leveraged_open, fallback_open,
                  entry_pct, exit_pct, dca_days,
                  start_cash=init_cash,
                  init_state="CASH", init_lev_shares=0.0, init_fallback_shares=0.0,
                  init_dca_day_counter=0):
    if entry_pct + exit_pct < -1e-9:
        return None, None, None

    s_close_lagged = signal_close_lagged.values
    s_sma_lagged = signal_sma_lagged.values
    l_open = leveraged_open.values
    f_open = fallback_open.values

    cash = start_cash
    lev_shares = init_lev_shares
    fallback_shares = init_fallback_shares
    state = init_state
    dca_day_counter = init_dca_day_counter
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

    equity = pd.Series(equity_curve, index=signal_close_lagged.index)
    final_state = {
        "state": state, "cash": cash, "lev_shares": lev_shares,
        "fallback_shares": fallback_shares, "dca_day_counter": dca_day_counter,
    }
    return equity, trade_count, final_state

def calc_stats(equity):
    if equity is None:
        return np.nan, np.nan
    equity_clean = equity.dropna()
    if len(equity_clean) < 2:
        return np.nan, np.nan
    ret = (equity_clean.iloc[-1] / equity_clean.iloc[0] - 1) * 100
    dd = ((equity_clean - equity_clean.cummax()) / equity_clean.cummax() * 100).min()
    return ret, dd

def optimize_window(qqq_close_lag_sub, qqq_sma_lag_sub, tqqq_sub, qqq_sub):
    """Grid-search entry%/exit%/dca on this in-sample window, then pick the
    cell whose local neighborhood (adjacent entry%, exit%, dca) scores best
    on average — a plateau, not an isolated spike. See chat explanation."""
    raw_scores = {}
    for dca_days in dca_range:
        grid = np.full((len(entry_range), len(exit_range)), np.nan)
        for ei, entry_pct in enumerate(entry_range):
            for xi, exit_pct in enumerate(exit_range):
                if entry_pct + exit_pct < -1e-9:
                    continue
                eq, _, _ = run_strategy(qqq_close_lag_sub, qqq_sma_lag_sub, tqqq_sub['Open'], qqq_sub['Open'],
                                         entry_pct, exit_pct, dca_days)
                ret, dd = calc_stats(eq)
                if np.isnan(ret) or np.isnan(dd) or dd == 0:
                    continue
                grid[ei, xi] = ret / abs(dd)
        raw_scores[dca_days] = grid

    best_smoothed = -np.inf
    best_params = None
    for di, dca_days in enumerate(dca_range):
        neighbor_grids = [raw_scores[dca_range[k]] for k in (di - 1, di, di + 1) if 0 <= k < len(dca_range)]
        for ei in range(len(entry_range)):
            for xi in range(len(exit_range)):
                if entry_range[ei] + exit_range[xi] < -1e-9:
                    continue
                vals = []
                for g in neighbor_grids:
                    for dei in (-1, 0, 1):
                        for dxi in (-1, 0, 1):
                            nei, nxi = ei + dei, xi + dxi
                            if 0 <= nei < len(entry_range) and 0 <= nxi < len(exit_range):
                                v = g[nei, nxi]
                                if not np.isnan(v):
                                    vals.append(v)
                if len(vals) >= MIN_NEIGHBORS_REQUIRED:
                    smoothed = float(np.mean(vals))
                    if smoothed > best_smoothed:
                        best_smoothed = smoothed
                        best_params = (entry_range[ei], exit_range[xi], dca_days)

    if best_params is None:
        best_raw = -np.inf
        for dca_days, grid in raw_scores.items():
            if np.all(np.isnan(grid)):
                continue
            idx = np.unravel_index(np.nanargmax(grid), grid.shape)
            if grid[idx] > best_raw:
                best_raw = grid[idx]
                best_params = (entry_range[idx[0]], exit_range[idx[1]], dca_days)
    return best_params

# ============================================================
# Full walk-forward pipeline as a reusable function — takes any
# qqq/tqqq OHLC dataframe (real OR synthetic) and runs the exact
# same optimize-then-execute loop. Used for both the baseline
# (real data) run and every simulated path in the nested Monte
# Carlo below, so the simulations are apples-to-apples with the
# real pipeline.
# ============================================================
def run_full_walkforward(qqq_df, tqqq_df, verbose=False):
    qqq_sma = ta.sma(qqq_df['Close'], length=ma_period)
    qqq_close_lagged_full = qqq_df['Close'].shift(1)
    qqq_sma_lagged_full = qqq_sma.shift(1)
    dates = qqq_df.index

    walk_forward_segments = []
    window_log = []
    carry_cash = init_cash
    carry_state = "CASH"
    carry_lev_shares = 0.0
    carry_fallback_shares = 0.0
    carry_dca_day_counter = 0

    start_idx = 0
    while start_idx + in_sample_len + out_sample_len <= len(dates):
        in_sample_dates = dates[start_idx: start_idx + in_sample_len]
        out_sample_dates = dates[start_idx + in_sample_len: start_idx + in_sample_len + out_sample_len]

        qqq_in = qqq_df.loc[in_sample_dates]
        tqqq_in = tqqq_df.loc[in_sample_dates]
        qqq_close_lag_in = qqq_close_lagged_full.loc[in_sample_dates]
        qqq_sma_lag_in = qqq_sma_lagged_full.loc[in_sample_dates]

        best_params = optimize_window(qqq_close_lag_in, qqq_sma_lag_in, tqqq_in, qqq_in)
        if best_params is None:
            start_idx += out_sample_len
            continue
        entry_p, exit_p, dca_d = best_params

        qqq_out = qqq_df.loc[out_sample_dates]
        tqqq_out = tqqq_df.loc[out_sample_dates]
        qqq_close_lag_out = qqq_close_lagged_full.loc[out_sample_dates]
        qqq_sma_lag_out = qqq_sma_lagged_full.loc[out_sample_dates]

        eq_out, _, final_state = run_strategy(
            qqq_close_lag_out, qqq_sma_lag_out, tqqq_out['Open'], qqq_out['Open'],
            entry_p, exit_p, dca_d,
            start_cash=carry_cash, init_state=carry_state,
            init_lev_shares=carry_lev_shares, init_fallback_shares=carry_fallback_shares,
            init_dca_day_counter=carry_dca_day_counter,
        )
        if eq_out is None:
            start_idx += out_sample_len
            continue

        carry_cash = final_state["cash"]
        carry_state = final_state["state"]
        carry_lev_shares = final_state["lev_shares"]
        carry_fallback_shares = final_state["fallback_shares"]
        carry_dca_day_counter = final_state["dca_day_counter"]

        walk_forward_segments.append(eq_out.dropna())
        window_log.append({
            "out_sample_start": out_sample_dates[0], "out_sample_end": out_sample_dates[-1],
            "entry_pct": entry_p, "exit_pct": exit_p, "dca_days": dca_d,
        })
        start_idx += out_sample_len

    if not walk_forward_segments:
        return None, None, None

    wf_equity = pd.concat(walk_forward_segments)
    wf_return = (wf_equity.iloc[-1] / wf_equity.iloc[0] - 1) * 100
    wf_dd = ((wf_equity - wf_equity.cummax()) / wf_equity.cummax() * 100).min()

    if verbose:
        print(pd.DataFrame(window_log).to_string(index=False))
        print(f"\nWalk-Forward Total Return: {wf_return:.2f}%")
        print(f"Walk-Forward Max Drawdown: {wf_dd:.2f}%")

    return wf_equity, wf_return, wf_dd

# ============================================================
# BASELINE: run the real pipeline once on actual data
# ============================================================
print("=== Baseline walk-forward on real data ===")
walk_forward_equity, wf_total_return, wf_dd = run_full_walkforward(qqq, tqqq, verbose=True)

# ============================================================
# CHEAP Monte Carlo: block bootstrap on the REALIZED strategy
# daily returns. Fast, but only measures sequence-of-returns
# risk (same building blocks, different order) — the parameters
# are frozen at whatever the real run picked. See chat explanation.
# ============================================================
def block_bootstrap_returns_mc(equity_series, n_sims=2000, block_size=21, seed=42):
    rng = np.random.default_rng(seed)
    daily_ret = equity_series.pct_change().dropna().values
    n = len(daily_ret)
    sim_finals = np.empty(n_sims)
    sim_dds = np.empty(n_sims)
    for s in range(n_sims):
        picks = []
        while len(picks) < n:
            start = rng.integers(0, max(1, n - block_size + 1))
            picks.extend(daily_ret[start:start + block_size].tolist())
        picks = np.array(picks[:n])
        growth = np.cumprod(1 + picks)
        sim_finals[s] = growth[-1]
        running_max = np.maximum.accumulate(growth)
        sim_dds[s] = ((growth - running_max) / running_max).min() * 100
    return sim_finals, sim_dds

sim_finals_cheap, sim_dds_cheap = block_bootstrap_returns_mc(walk_forward_equity)
pctiles = [5, 25, 50, 75, 95]
print(f"\n=== Cheap Monte Carlo (returns-only, sequence risk) ===")
print(f"{'Percentile':<12}{'Total Return %':>18}{'Max Drawdown %':>18}")
for p, r, d in zip(pctiles, np.percentile((sim_finals_cheap - 1) * 100, pctiles), np.percentile(sim_dds_cheap, pctiles)):
    print(f"{p:>10}%{r:>18.2f}{d:>18.2f}")

# Histogram — FIXED. go.Histogram bins in LINEAR space regardless of the
# axis display type, so xaxis type="log" alone just squashes/loses the bars.
# Instead, log10-transform the data ourselves, histogram THAT, and relabel
# the ticks to show human-readable numbers.
def log_histogram(values, actual_value, title):
    log_vals = np.log10(values)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=log_vals, nbinsx=50, name="Simulated final growth of $1"))
    fig.add_vline(x=np.log10(actual_value), line=dict(color="darkred", width=2, dash="dash"), annotation_text="Actual realized")
    tick_min = int(np.floor(log_vals.min()))
    tick_max = int(np.ceil(log_vals.max()))
    tickvals = list(range(tick_min, tick_max + 1))
    def fmt(v):
        val = 10 ** v
        if val >= 1e6:
            return f"{val/1e6:.0f}M"
        elif val >= 1e3:
            return f"{val/1e3:.0f}k"
        return f"{val:.0f}"
    fig.update_xaxes(title="Final growth of $1 (log scale)", tickmode="array", tickvals=tickvals, ticktext=[fmt(v) for v in tickvals])
    fig.update_layout(title=title, yaxis_title="Count", height=500, width=1000)
    return fig

actual_final_cheap = walk_forward_equity.iloc[-1] / walk_forward_equity.iloc[0]
fig_hist_cheap = log_histogram(sim_finals_cheap, actual_final_cheap, "Cheap MC: Distribution of Simulated Final Values (log x-axis)")
fig_hist_cheap.show()

# ============================================================
# RIGOROUS Monte Carlo: block-bootstrap the RAW PRICE returns
# (jointly for QQQ and TQQQ, preserving the day's simultaneous
# open/close relationship and the leverage relationship between
# the two assets), reconstruct a synthetic-but-internally-consistent
# price history, then re-run the ENTIRE pipeline — including
# parameter optimization — on that synthetic history. This captures
# parameter-selection risk that the cheap version above cannot.
#
# WARNING: this reruns a full grid-search walk-forward per simulated
# path, which is expensive. Start with a small N_SIMS_NESTED and
# scale up once you've confirmed the runtime per sim.
# ============================================================
N_SIMS_NESTED = 30      # start small; each sim reruns the full grid search
NESTED_BLOCK_SIZE = 21

qqq_open_ret = qqq['Open'].pct_change()
qqq_c2o_ratio = qqq['Close'] / qqq['Open']
tqqq_open_ret = tqqq['Open'].pct_change()
tqqq_c2o_ratio = tqqq['Close'] / tqqq['Open']

joint = np.column_stack([
    qqq_open_ret.values[1:], qqq_c2o_ratio.values[1:],
    tqqq_open_ret.values[1:], tqqq_c2o_ratio.values[1:],
])
n_days = joint.shape[0]

def synthesize_price_paths(rng, block_size=NESTED_BLOCK_SIZE):
    """Block-bootstrap the joint daily (open-to-open return, close/open ratio)
    tuples for QQQ and TQQQ together, then rebuild consistent OHLC-style
    price series from them. Resampling QQQ and TQQQ jointly (same day index)
    keeps the ~3x leverage relationship intact within each block, since
    that's literally how those two prices co-moved on any real historical day."""
    picks_idx = []
    while len(picks_idx) < n_days:
        start = rng.integers(0, max(1, n_days - block_size + 1))
        picks_idx.extend(range(start, min(start + block_size, n_days)))
    picks_idx = np.array(picks_idx[:n_days])
    sampled = joint[picks_idx]

    qqq_open_sim = np.empty(n_days + 1)
    qqq_close_sim = np.empty(n_days + 1)
    tqqq_open_sim = np.empty(n_days + 1)
    tqqq_close_sim = np.empty(n_days + 1)

    qqq_open_sim[0] = qqq['Open'].iloc[0]
    qqq_close_sim[0] = qqq['Close'].iloc[0]
    tqqq_open_sim[0] = tqqq['Open'].iloc[0]
    tqqq_close_sim[0] = tqqq['Close'].iloc[0]

    for i in range(n_days):
        qo_ret, qc2o, to_ret, tc2o = sampled[i]
        qqq_open_sim[i + 1] = qqq_open_sim[i] * (1 + qo_ret)
        qqq_close_sim[i + 1] = qqq_open_sim[i + 1] * qc2o
        tqqq_open_sim[i + 1] = tqqq_open_sim[i] * (1 + to_ret)
        tqqq_close_sim[i + 1] = tqqq_open_sim[i + 1] * tc2o

    qqq_sim = pd.DataFrame({'Open': qqq_open_sim, 'Close': qqq_close_sim}, index=qqq.index)
    tqqq_sim = pd.DataFrame({'Open': tqqq_open_sim, 'Close': tqqq_close_sim}, index=tqqq.index)
    return qqq_sim, tqqq_sim

print(f"\n=== Rigorous nested Monte Carlo: {N_SIMS_NESTED} sims, each re-running the full walk-forward ===")
print("This can take a while — timing the first sim to give you an estimate...")

rng = np.random.default_rng(123)
nested_returns = []
nested_dds = []
t0 = time.time()
for s in range(N_SIMS_NESTED):
    t_sim_start = time.time()
    qqq_sim, tqqq_sim = synthesize_price_paths(rng)
    _, sim_return, sim_dd = run_full_walkforward(qqq_sim, tqqq_sim, verbose=False)
    if sim_return is not None:
        nested_returns.append(sim_return)
        nested_dds.append(sim_dd)
    elapsed = time.time() - t_sim_start
    if s == 0:
        print(f"  sim 1 took {elapsed:.1f}s -> est. total ~{elapsed * N_SIMS_NESTED / 60:.1f} min for {N_SIMS_NESTED} sims")
    print(f"  sim {s + 1}/{N_SIMS_NESTED} done ({elapsed:.1f}s)")

nested_returns = np.array(nested_returns)
nested_dds = np.array(nested_dds)

print(f"\nTotal nested MC runtime: {(time.time() - t0)/60:.1f} min")
print(f"\n=== Rigorous Nested Monte Carlo Results ({len(nested_returns)} successful sims) ===")
print(f"{'Percentile':<12}{'Total Return %':>18}{'Max Drawdown %':>18}")
for p, r, d in zip(pctiles, np.percentile(nested_returns, pctiles), np.percentile(nested_dds, pctiles)):
    print(f"{p:>10}%{r:>18.2f}{d:>18.2f}")
print(f"\nActual realized path:  Return {wf_total_return:.2f}%  |  Max DD {wf_dd:.2f}%")

fig_hist_nested = log_histogram(1 + nested_returns / 100, 1 + wf_total_return / 100,
                                 f"Rigorous Nested MC ({len(nested_returns)} sims, re-optimized per path)")
fig_hist_nested.show()