import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
D = lambda _f: _os.path.join(_ROOT, "data", _f)      # price data (inputs)
R = lambda _f: _os.path.join(_ROOT, "results", _f)   # generated outputs

import os
import pandas as pd
import numpy as np
import pandas_ta_classic as ta

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

IN_SAMPLE_YEARS = 8  # was 5; raised per in-sample-window analysis (regime coverage 84%, 19 folds, common-window Sharpe unchanged)
OUT_SAMPLE_YEARS = 1
TRADING_DAYS_PER_YEAR = 252
in_sample_len = IN_SAMPLE_YEARS * TRADING_DAYS_PER_YEAR
out_sample_len = OUT_SAMPLE_YEARS * TRADING_DAYS_PER_YEAR
MIN_NEIGHBORS_REQUIRED = 3
COST_BPS = 0.0  # held at 0 to isolate the DCA-shape question from the cost question already answered separately

qqq_sma = ta.sma(qqq['Close'], length=ma_period)
qqq_close_lagged_full = qqq['Close'].shift(1)
qqq_sma_lagged_full = qqq_sma.shift(1)

def apply_cost(notional, cost_bps):
    return notional * (1 - cost_bps / 10000.0)

# ============================================================
# DCA shapes.
# "linear", "front_loaded", "back_loaded" are STATIC schedules —
# decided purely by which day of the DCA window it is, computed
# once as a cumulative-fraction-of-pool curve.
# "dip_scaled" is DYNAMIC — it looks at today's actual fallback
# price relative to the price when the DCA period started, and
# buys proportionally more on days where the price has fallen
# further, less when it's flat or has recovered. This is the one
# that responds to what the market is actually doing, not just
# to elapsed time.
# ============================================================
_cum_frac_cache = {}
def get_cum_frac(dca_days, shape):
    key = (dca_days, shape)
    if key in _cum_frac_cache:
        return _cum_frac_cache[key]
    idx = np.arange(1, dca_days + 1, dtype=float)
    if shape == "linear":
        w = np.ones(dca_days)
    elif shape == "front_loaded":
        w = (dca_days - idx + 1)
    elif shape == "back_loaded":
        w = idx.copy()
    else:
        raise ValueError(f"get_cum_frac not defined for shape {shape}")
    cum = np.cumsum(w)
    cum_frac = cum / cum[-1]
    _cum_frac_cache[key] = cum_frac
    return cum_frac

def compute_daily_buy(dca_shape, dca_day_counter, dca_days, cash, dca_pool_initial, dca_spent_so_far,
                       f_price, dca_entry_price, dip_sensitivity=2.0):
    if dca_shape in ("linear", "front_loaded", "back_loaded"):
        cum_frac = get_cum_frac(dca_days, dca_shape)
        k = min(dca_day_counter, dca_days)
        target_spend = cum_frac[k - 1] * dca_pool_initial
        return max(0.0, min(target_spend - dca_spent_so_far, cash))
    elif dca_shape == "dip_scaled":
        remaining_days = max(1, dca_days - dca_day_counter + 1)
        discount = max(0.0, (dca_entry_price - f_price) / dca_entry_price) if dca_entry_price > 0 else 0.0
        weight = 1.0 + dip_sensitivity * discount
        base = cash / remaining_days
        return min(cash, base * weight)
    else:
        raise ValueError(f"unknown dca_shape {dca_shape}")

# ============================================================
# Core strategy function — now shape-aware and cost-aware, with
# the extra DCA-tracking state (pool size, amount spent so far,
# price when DCA began) carried across window boundaries just
# like the other state variables.
# ============================================================
def run_strategy(signal_close_lagged, signal_sma_lagged, leveraged_open, fallback_open,
                  entry_pct, exit_pct, dca_days,
                  start_cash=init_cash, cost_bps=COST_BPS, dca_shape="linear", dip_sensitivity=2.0,
                  init_state="CASH", init_lev_shares=0.0, init_fallback_shares=0.0,
                  init_dca_day_counter=0, init_dca_pool_initial=0.0, init_dca_spent_so_far=0.0,
                  init_dca_entry_price=0.0):
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
    dca_pool_initial = init_dca_pool_initial
    dca_spent_so_far = init_dca_spent_so_far
    dca_entry_price = init_dca_entry_price
    equity_curve = np.empty(len(s_close_lagged))
    signal_trades = 0
    dca_daily_trades = 0

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
                signal_trades += 1
        elif state == "FULL_LEV":
            if s_price < exit_threshold:
                cash = apply_cost(lev_shares * l_price, cost_bps); lev_shares = 0.0
                state = "DCA"; dca_day_counter = 0
                dca_pool_initial = cash; dca_spent_so_far = 0.0; dca_entry_price = f_price
                signal_trades += 1
        elif state == "DCA":
            if s_price > entry_threshold:
                cash += apply_cost(fallback_shares * f_price, cost_bps); fallback_shares = 0.0
                lev_shares = apply_cost(cash, cost_bps) / l_price; cash = 0.0
                state = "FULL_LEV"
                dca_pool_initial = 0.0; dca_spent_so_far = 0.0; dca_entry_price = 0.0
                signal_trades += 1
            else:
                dca_day_counter += 1
                daily_buy = compute_daily_buy(dca_shape, dca_day_counter, dca_days, cash,
                                               dca_pool_initial, dca_spent_so_far, f_price, dca_entry_price,
                                               dip_sensitivity=dip_sensitivity)
                if daily_buy > 0:
                    fallback_shares += apply_cost(daily_buy, cost_bps) / f_price
                    cash -= daily_buy
                    dca_spent_so_far += daily_buy
                    dca_daily_trades += 1
                if dca_day_counter >= dca_days:
                    state = "DELEVERAGED"
        elif state == "DELEVERAGED":
            if s_price > entry_threshold:
                cash = apply_cost(fallback_shares * f_price, cost_bps); fallback_shares = 0.0
                lev_shares = apply_cost(cash, cost_bps) / l_price; cash = 0.0
                state = "FULL_LEV"
                dca_pool_initial = 0.0; dca_spent_so_far = 0.0; dca_entry_price = 0.0
                signal_trades += 1

        equity_curve[i] = cash + lev_shares * l_price + fallback_shares * f_price

    equity = pd.Series(equity_curve, index=signal_close_lagged.index)
    final_state = {
        "state": state, "cash": cash, "lev_shares": lev_shares, "fallback_shares": fallback_shares,
        "dca_day_counter": dca_day_counter, "dca_pool_initial": dca_pool_initial,
        "dca_spent_so_far": dca_spent_so_far, "dca_entry_price": dca_entry_price,
        "signal_trades": signal_trades, "dca_daily_trades": dca_daily_trades,
    }
    return equity, signal_trades + dca_daily_trades, final_state

def calc_stats(equity):
    if equity is None:
        return np.nan, np.nan
    equity_clean = equity.dropna()
    if len(equity_clean) < 2:
        return np.nan, np.nan
    ret = (equity_clean.iloc[-1] / equity_clean.iloc[0] - 1) * 100
    dd = ((equity_clean - equity_clean.cummax()) / equity_clean.cummax() * 100).min()
    return ret, dd

def optimize_window(qqq_close_lag_sub, qqq_sma_lag_sub, tqqq_sub, qqq_sub, dca_shape, dip_sensitivity=2.0):
    raw_scores = {}
    for dca_days in dca_range:
        grid = np.full((len(entry_range), len(exit_range)), np.nan)
        for ei, entry_pct in enumerate(entry_range):
            for xi, exit_pct in enumerate(exit_range):
                if entry_pct + exit_pct < -1e-9:
                    continue
                eq, _, _ = run_strategy(qqq_close_lag_sub, qqq_sma_lag_sub, tqqq_sub['Open'], qqq_sub['Open'],
                                         entry_pct, exit_pct, dca_days, dca_shape=dca_shape, dip_sensitivity=dip_sensitivity)
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

def run_full_walkforward(qqq_df, tqqq_df, dca_shape, dip_sensitivity=2.0, verbose=False):
    dates = qqq_df.index
    walk_forward_segments = []
    window_log = []
    carry = dict(cash=init_cash, state="CASH", lev_shares=0.0, fallback_shares=0.0,
                 dca_day_counter=0, dca_pool_initial=0.0, dca_spent_so_far=0.0, dca_entry_price=0.0)
    total_signal_trades = 0
    total_dca_daily_trades = 0

    start_idx = 0
    while start_idx + in_sample_len + out_sample_len <= len(dates):
        in_sample_dates = dates[start_idx: start_idx + in_sample_len]
        out_sample_dates = dates[start_idx + in_sample_len: start_idx + in_sample_len + out_sample_len]

        qqq_in = qqq_df.loc[in_sample_dates]
        tqqq_in = tqqq_df.loc[in_sample_dates]
        qqq_close_lag_in = qqq_close_lagged_full.loc[in_sample_dates]
        qqq_sma_lag_in = qqq_sma_lagged_full.loc[in_sample_dates]

        best_params = optimize_window(qqq_close_lag_in, qqq_sma_lag_in, tqqq_in, qqq_in, dca_shape, dip_sensitivity=dip_sensitivity)
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
            entry_p, exit_p, dca_d, dca_shape=dca_shape, dip_sensitivity=dip_sensitivity,
            start_cash=carry["cash"], init_state=carry["state"],
            init_lev_shares=carry["lev_shares"], init_fallback_shares=carry["fallback_shares"],
            init_dca_day_counter=carry["dca_day_counter"], init_dca_pool_initial=carry["dca_pool_initial"],
            init_dca_spent_so_far=carry["dca_spent_so_far"], init_dca_entry_price=carry["dca_entry_price"],
        )
        if eq_out is None:
            start_idx += out_sample_len
            continue

        carry = {k: final_state[k] for k in ["state", "cash", "lev_shares", "fallback_shares",
                                              "dca_day_counter", "dca_pool_initial", "dca_spent_so_far", "dca_entry_price"]}
        total_signal_trades += final_state["signal_trades"]
        total_dca_daily_trades += final_state["dca_daily_trades"]

        walk_forward_segments.append(eq_out.dropna())
        window_log.append({"out_sample_start": out_sample_dates[0], "out_sample_end": out_sample_dates[-1],
                            "entry_pct": entry_p, "exit_pct": exit_p, "dca_days": dca_d})
        start_idx += out_sample_len

    if not walk_forward_segments:
        return None, None, None, None

    wf_equity = pd.concat(walk_forward_segments)
    wf_return = (wf_equity.iloc[-1] / wf_equity.iloc[0] - 1) * 100
    wf_dd = ((wf_equity - wf_equity.cummax()) / wf_equity.cummax() * 100).min()
    trade_summary = {"signal_trades": total_signal_trades, "dca_daily_trades": total_dca_daily_trades}
    if verbose:
        print(pd.DataFrame(window_log).to_string(index=False))
    return wf_equity, wf_return, wf_dd, trade_summary

# ============================================================
# Compare shapes
# ============================================================

# ============================================================
# Sensitivity sweep for dip_scaled, with linear and front_loaded
# included as fixed reference points. Each level re-runs the
# full walk-forward + optimization (dca_days grid still capped
# at 126+ here — this is testing sensitivity strength, not
# window length, which is the separate test running in parallel).
# ============================================================
DIP_SENSITIVITY_LEVELS = [1.0, 2.0, 4.0, 8.0, 15.0]

print("=== Reference shapes ===")
results = []
for shape in ["linear", "front_loaded"]:
    wf_equity, wf_return, wf_dd, _ = run_full_walkforward(qqq, tqqq, dca_shape=shape, verbose=False)
    print(f"{shape:<14} | return={wf_return:>14,.2f}% | max_dd={wf_dd:>7.2f}%")
    results.append({"variant": shape, "total_return_pct": wf_return, "max_dd_pct": wf_dd})

print("\n=== dip_scaled sensitivity sweep ===")
for sens in DIP_SENSITIVITY_LEVELS:
    wf_equity, wf_return, wf_dd, _ = run_full_walkforward(qqq, tqqq, dca_shape="dip_scaled", dip_sensitivity=sens, verbose=False)
    label = f"dip_scaled(k={sens})"
    print(f"{label:<20} | return={wf_return:>14,.2f}% | max_dd={wf_dd:>7.2f}%")
    results.append({"variant": label, "total_return_pct": wf_return, "max_dd_pct": wf_dd})

results_df = pd.DataFrame(results)
print("\n=== Summary ===")
print(results_df.to_string(index=False))
print("\nIf return/max_dd keep improving as k increases, sensitivity was indeed too low before —")
print("push k higher still. If it peaks partway through the sweep and then gets WORSE at high k,")
print("that's the sweet spot — pushing sensitivity too far starts overpaying into dips that keep falling.")
print("If it barely moves across the whole range, sensitivity wasn't the limiting factor at all —")
print("the dip_scaled mechanism itself (or its reference to dca_entry_price) may need rethinking.")