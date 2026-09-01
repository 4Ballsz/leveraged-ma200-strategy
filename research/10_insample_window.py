import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
D = lambda _f: _os.path.join(_ROOT, "data", _f)      # price data (inputs)
R = lambda _f: _os.path.join(_ROOT, "results", _f)   # generated outputs

import os
import pandas as pd
import numpy as np

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
MIN_NEIGHBORS_REQUIRED = 3
OUT_SAMPLE_YEARS = 1
TRADING_DAYS_PER_YEAR = 252
out_sample_len = OUT_SAMPLE_YEARS * TRADING_DAYS_PER_YEAR
COST_BPS = 0.0

qqq_sma = qqq['Close'].rolling(window=ma_period).mean()  # plain pandas, no numba/llvmlite dependency
qqq_close_lagged_full = qqq['Close'].shift(1)
qqq_sma_lagged_full = qqq_sma.shift(1)

def apply_cost(notional, cost_bps):
    return notional * (1 - cost_bps / 10000.0)

# ============================================================
# Core strategy — linear DCA, position-continuous across windows
# ============================================================
def run_strategy(signal_close_lagged, signal_sma_lagged, leveraged_open, fallback_open,
                  entry_pct, exit_pct, dca_days,
                  start_cash=init_cash, cost_bps=COST_BPS,
                  init_state="CASH", init_lev_shares=0.0, init_fallback_shares=0.0, init_dca_day_counter=0):
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
                lev_shares = apply_cost(cash, cost_bps) / l_price; cash = 0.0; state = "FULL_LEV"
                trade_count += 1
        elif state == "FULL_LEV":
            if s_price < exit_threshold:
                cash = apply_cost(lev_shares * l_price, cost_bps); lev_shares = 0.0
                state = "DCA"; dca_day_counter = 0
                trade_count += 1
        elif state == "DCA":
            if s_price > entry_threshold:
                cash += apply_cost(fallback_shares * f_price, cost_bps); fallback_shares = 0.0
                lev_shares = apply_cost(cash, cost_bps) / l_price; cash = 0.0
                state = "FULL_LEV"
                trade_count += 1
            else:
                dca_day_counter += 1
                if cash > 0:
                    remaining_days = max(1, dca_days - dca_day_counter + 1)
                    daily_buy = min(cash / remaining_days, cash)
                    fallback_shares += apply_cost(daily_buy, cost_bps) / f_price
                    cash -= daily_buy
                    trade_count += 1
                if dca_day_counter >= dca_days:
                    state = "DELEVERAGED"
        elif state == "DELEVERAGED":
            if s_price > entry_threshold:
                cash = apply_cost(fallback_shares * f_price, cost_bps); fallback_shares = 0.0
                lev_shares = apply_cost(cash, cost_bps) / l_price; cash = 0.0
                state = "FULL_LEV"
                trade_count += 1

        equity_curve[i] = cash + lev_shares * l_price + fallback_shares * f_price

    equity = pd.Series(equity_curve, index=signal_close_lagged.index)
    final_state = {"state": state, "cash": cash, "lev_shares": lev_shares,
                    "fallback_shares": fallback_shares, "dca_day_counter": dca_day_counter}
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

def annualize(total_return_pct, years):
    base = max(1 + total_return_pct / 100, 1e-6)  # guard against <=-100% total return
    return (base ** (1.0 / years) - 1) * 100

# ============================================================
# Walk-forward with configurable IN_SAMPLE_YEARS, tracking each
# window's IN-SAMPLE return (annualized, for the chosen params)
# alongside its OUT-OF-SAMPLE return, so we can measure how much
# of the in-sample edge actually survived into unseen data.
# ============================================================
def run_walkforward_sweep(qqq_df, tqqq_df, in_sample_years):
    in_sample_len = in_sample_years * TRADING_DAYS_PER_YEAR
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

        # Re-run the chosen params once on the in-sample data itself, purely
        # to record how good they looked where they were picked — this is
        # the number we compare the out-of-sample result against.
        eq_in, _, _ = run_strategy(qqq_close_lag_in, qqq_sma_lag_in, tqqq_in['Open'], qqq_in['Open'],
                                    entry_p, exit_p, dca_d)
        in_sample_ret, in_sample_dd = calc_stats(eq_in)
        in_sample_ret_annualized = annualize(in_sample_ret, in_sample_years)

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

        out_sample_ret, out_sample_dd = calc_stats(eq_out.dropna())

        carry_cash = final_state["cash"]
        carry_state = final_state["state"]
        carry_lev_shares = final_state["lev_shares"]
        carry_fallback_shares = final_state["fallback_shares"]
        carry_dca_day_counter = final_state["dca_day_counter"]

        walk_forward_segments.append(eq_out.dropna())
        window_log.append({
            "out_sample_start": out_sample_dates[0], "out_sample_end": out_sample_dates[-1],
            "entry_pct": entry_p, "exit_pct": exit_p, "dca_days": dca_d,
            "in_sample_return_annualized_pct": in_sample_ret_annualized,
            "out_sample_return_pct": out_sample_ret,
        })
        start_idx += out_sample_len

    if not walk_forward_segments:
        return None, None, None, None

    wf_equity = pd.concat(walk_forward_segments)
    wf_return = (wf_equity.iloc[-1] / wf_equity.iloc[0] - 1) * 100
    wf_dd = ((wf_equity - wf_equity.cummax()) / wf_equity.cummax() * 100).min()
    window_log_df = pd.DataFrame(window_log)
    return wf_equity, wf_return, wf_dd, window_log_df

# ============================================================
# Regime coverage (reused from the earlier diagnostic)
# ============================================================
def in_sample_buyhold_dd(qqq_df, in_sample_dates):
    close = qqq_df.loc[in_sample_dates, 'Close']
    return ((close - close.cummax()) / close.cummax() * 100).min()

def coverage_stats(qqq_df, in_sample_years, threshold=-25):
    in_sample_len = in_sample_years * TRADING_DAYS_PER_YEAR
    dates = qqq_df.index
    dds = []
    start_idx = 0
    while start_idx + in_sample_len + out_sample_len <= len(dates):
        in_sample_dates = dates[start_idx: start_idx + in_sample_len]
        dds.append(in_sample_buyhold_dd(qqq_df, in_sample_dates))
        start_idx += out_sample_len
    dds = pd.Series(dds)
    return {
        "pct_windows_saw_dd_worse_than_25": (dds <= threshold).mean() * 100,
        "worst_case_blind_window_dd": dds.max(),
    }

# ============================================================
# Cheap Monte Carlo (block bootstrap on realized daily returns)
# ============================================================
def block_bootstrap_mc(equity_series, n_sims=2000, block_size=21, seed=42):
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

# ============================================================
# Consolidated test: for each IN_SAMPLE_YEARS candidate, run the
# full validated pipeline (walk-forward -> efficiency -> regime
# coverage -> Monte Carlo) and report everything together.
# ============================================================
IN_SAMPLE_YEARS_LIST = [3, 5, 7, 10, 15]
MC_PCTILES = [5, 25, 50, 75, 95]

print("=== Full consolidated test across IN_SAMPLE_YEARS candidates ===")
print("(walk-forward -> efficiency -> regime coverage -> 2000-sim Monte Carlo, each)\n")

all_results = []
for in_sample_years in IN_SAMPLE_YEARS_LIST:
    print(f"--- IN_SAMPLE_YEARS = {in_sample_years} ---")
    wf_equity, wf_return, wf_dd, window_log_df = run_walkforward_sweep(qqq, tqqq, in_sample_years)
    if wf_equity is None:
        print("  not enough history for even one fold — skipped\n")
        continue

    n_folds = len(window_log_df)
    avg_is = window_log_df["in_sample_return_annualized_pct"].mean()
    avg_oos = window_log_df["out_sample_return_pct"].mean()
    efficiency = (avg_oos / avg_is * 100) if abs(avg_is) > 1e-9 else np.nan

    cov = coverage_stats(qqq, in_sample_years)

    sim_finals, sim_dds = block_bootstrap_mc(wf_equity)
    sim_returns_pct = (sim_finals - 1) * 100
    mc_ret_pctiles = dict(zip(MC_PCTILES, np.percentile(sim_returns_pct, MC_PCTILES)))
    mc_dd_pctiles = dict(zip(MC_PCTILES, np.percentile(sim_dds, MC_PCTILES)))
    actual_return_rank = (sim_returns_pct < wf_return).mean() * 100
    actual_dd_rank = (sim_dds > wf_dd).mean() * 100  # % of sims with WORSE (more negative) drawdown than actual

    print(f"  total_return={wf_return:>14,.2f}% | max_dd={wf_dd:>7.2f}% | folds={n_folds:>2} | efficiency={efficiency:>6.1f}%")
    print(f"  regime coverage: {cov['pct_windows_saw_dd_worse_than_25']:>5.1f}% of windows saw -25%+ DD | mildest blind window: {cov['worst_case_blind_window_dd']:>6.2f}%")
    print(f"  MC return  5/25/50/75/95%: {mc_ret_pctiles[5]:>10,.0f}% / {mc_ret_pctiles[25]:>10,.0f}% / {mc_ret_pctiles[50]:>10,.0f}% / {mc_ret_pctiles[75]:>10,.0f}% / {mc_ret_pctiles[95]:>10,.0f}%")
    print(f"  MC max_dd  5/25/50/75/95%: {mc_dd_pctiles[5]:>7.2f}% / {mc_dd_pctiles[25]:>7.2f}% / {mc_dd_pctiles[50]:>7.2f}% / {mc_dd_pctiles[75]:>7.2f}% / {mc_dd_pctiles[95]:>7.2f}%")
    print(f"  actual result sits at: return={actual_return_rank:.0f}th pctile | drawdown milder than {actual_dd_rank:.0f}% of sims\n")

    all_results.append({
        "in_sample_years": in_sample_years, "n_folds": n_folds,
        "total_return_pct": wf_return, "max_dd_pct": wf_dd, "efficiency_pct": efficiency,
        "pct_windows_saw_real_bear": cov["pct_windows_saw_dd_worse_than_25"],
        "worst_case_blind_window_dd": cov["worst_case_blind_window_dd"],
        "mc_median_return_pct": mc_ret_pctiles[50], "mc_p25_return_pct": mc_ret_pctiles[25],
        "mc_median_dd_pct": mc_dd_pctiles[50], "mc_p25_dd_pct": mc_dd_pctiles[25],
        "actual_return_percentile_rank": actual_return_rank, "actual_dd_percentile_rank": actual_dd_rank,
    })

print("=== FULL SUMMARY ===")
results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))

print("""
How to read this for a final decision:
- efficiency_pct: higher = more of the in-sample edge survived out-of-sample (>100% means OOS
  beat what IS predicted; well below 100% is a real warning sign).
- pct_windows_saw_real_bear / worst_case_blind_window_dd: coverage — did the optimizer actually
  get tested against a real correction before picking parameters, or was it working blind.
- mc_median_return_pct / mc_median_dd_pct: the CENTER of 2000 resampled alternate histories —
  a better summary of "typical" performance than the single realized number alone.
- mc_p25_dd_pct: a realistic PLANNING number for drawdown risk (25th percentile = a genuinely
  bad-but-plausible outcome, not the best case).
- actual_dd_percentile_rank: if this is very high (90+), the realized backtest got a lucky
  drawdown draw at that setting and the headline max_dd understates real risk; if it's closer
  to 50, the realized result is a fair, typical representative of that setting's risk profile.
Look for the setting that's solid across ALL of these, not just the best on any single column.
""")