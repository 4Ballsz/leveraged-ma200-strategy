"""
daily_advisor.py  --  live "what's my next move" for the leveraged MA200
                      state-machine strategy (QQQ signal / TQQQ leveraged / QQQ fallback).

WHAT IT DOES  (run once a day, after the US close)
  1. Refreshes QQQ + TQQQ daily bars from Yahoo Finance (cached to the same
     CSVs the backtests use; only re-downloads when the local copy is stale).
  2. Rebuilds the synthetic-extended TQQQ series (pre-2010 modelled from QQQ)
     used only for the parameter optimization.
  3. Walk-forward parameter optimization: every REOPTIMIZE_EVERY_TRADING_DAYS
     (~1 year) it re-runs the neighbourhood-smoothed grid search on the
     trailing IN_SAMPLE_YEARS of data and adopts the new entry/exit/DCA
     parameters going forward. The full history of parameter sets is cached
     in advisor_state.json so the slow grid search only runs when a new
     re-optimization date has actually passed.
  4. Replays the strategy state machine over that parameter history up to the
     last completed signal bar, so it knows your current position without you
     telling it (assumes you have been following the recommendations).
  5. Prints today's market reading -- including how far price is from the
     200-day SMA in percent -- and your next move, to be executed at the
     NEXT session's open (the strategy decides on yesterday's close, trades
     at the next open; this tool follows the same convention).
  6. Appends a row to signal_log.csv.

USAGE
  python daily_advisor.py                 # normal daily run
  python daily_advisor.py --update        # force a fresh data download
  python daily_advisor.py --no-update     # never hit the network, use cached CSVs
  python daily_advisor.py --optimize-now  # force a re-optimization this run
  python daily_advisor.py --reset         # wipe parameter history and rebuild from scratch
  python daily_advisor.py --set-state FULL_LEV [--dca-day 40]
                                          # correct the tracked position, effective today
  python daily_advisor.py --history 20    # print the last 20 logged days and exit

NOTES / LIMITATIONS
  * Advice only. It does not place orders. It assumes you execute each
    recommendation at the next open, in full (all-in / all-out), matching the
    backtest. Partial fills, taxes, and slippage are not modelled.
  * "DCA X% of remaining cash" is a fraction, because the tool does not track
    your dollar balances -- only the position STATE and the DCA day counter.
  * If the latest bar is today and the US market has not closed yet, today's
    close is provisional -- re-run after 16:00 ET.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
SIGNAL_TICKER    = "QQQ"     # index whose 200-day SMA generates the signal
LEVERAGED_TICKER = "TQQQ"    # held when FULL_LEV (3x)
FALLBACK_TICKER  = "QQQ"     # DCA'd into / held when DELEVERAGED (1x)

MA_PERIOD                      = 200
IN_SAMPLE_YEARS                = 8      # trailing window for each re-optimization
REOPTIMIZE_EVERY_TRADING_DAYS  = 252    # ~1 year between walk-forward re-opts
TRADING_DAYS_PER_YEAR          = 252

ENTRY_RANGE = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
EXIT_RANGE  = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
DCA_RANGE   = [126, 189, 252, 315]
MIN_NEIGHBORS_REQUIRED = 3

# synthetic pre-inception leverage model (matches tqqqextended.py)
SYNTH_LEVERAGE         = 3
SYNTH_EXPENSE_RATIO    = 0.0095
SYNTH_FINANCING_SPREAD = 0.01

DATA_START = "1999-01-01"

HERE = os.path.dirname(os.path.abspath(__file__))

# Price CSVs live next to this script (flat "live" layout) unless a sibling
# ../data directory exists (repo layout), or ADVISOR_DATA_DIR overrides both.
# Runtime artifacts (state + log) always sit next to the script.
_SIBLING_DATA = os.path.join(HERE, os.pardir, "data")
DATA_DIR = os.environ.get("ADVISOR_DATA_DIR") or (
    _SIBLING_DATA if os.path.isdir(_SIBLING_DATA) else HERE)

SIGNAL_CSV     = os.path.join(DATA_DIR, f"{SIGNAL_TICKER.lower()}_yf.csv")
LEVERAGED_CSV  = os.path.join(DATA_DIR, f"{LEVERAGED_TICKER.lower()}_yf.csv")
LEVERAGED_EXT  = os.path.join(DATA_DIR, f"{LEVERAGED_TICKER.lower()}_extended_yf.csv")
STATE_JSON     = os.path.join(HERE, "advisor_state.json")
LOG_CSV        = os.path.join(HERE, "signal_log.csv")

IN_LEN = IN_SAMPLE_YEARS * TRADING_DAYS_PER_YEAR

STATES = ("CASH", "FULL_LEV", "DCA", "DELEVERAGED")


# ============================================================
# DATA
# ============================================================
def _download(ticker):
    import yfinance as yf
    raw = yf.download(ticker, start=DATA_START, auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"empty download for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.copy()
    raw["Close"] = raw["Adj Close"]
    data = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    data.index.name = "Date"
    return data


def refresh(ticker, path, mode):
    """mode: 'auto' | 'force' | 'never'. Returns an OHLC DataFrame (Date index)."""
    cached = None
    if os.path.exists(path):
        cached = pd.read_csv(path, index_col="Date", parse_dates=True)
    if mode == "never":
        if cached is None:
            sys.exit(f"ERROR: --no-update but no cached data at {path}")
        return cached
    stale = True
    if cached is not None and len(cached):
        age_days = (pd.Timestamp.now().normalize() - cached.index.max().normalize()).days
        stale = age_days > 2
    if mode == "auto" and not stale:
        return cached
    try:
        data = _download(ticker)
        data.to_csv(path)
        print(f"  {ticker}: downloaded {len(data)} rows through {data.index.max().date()}")
        return data
    except Exception as e:  # offline / rate-limited / API change
        if cached is not None:
            print(f"  {ticker}: download failed ({e}); using cached data through {cached.index.max().date()}")
            return cached
        raise


def build_extended(underlying, leveraged):
    """Synthetic pre-inception leveraged series spliced onto the real one
    (same construction as tqqqextended.py)."""
    daily_drag = (SYNTH_EXPENSE_RATIO + SYNTH_FINANCING_SPREAD) / TRADING_DAYS_PER_YEAR
    synth_ret = SYNTH_LEVERAGE * underlying["Close"].pct_change() - daily_drag

    lev_start = leveraged.index.min()
    lev_start_px = leveraged["Close"].iloc[0]
    pre_dates = underlying.index[underlying.index < lev_start]
    pre_ret = synth_ret.loc[pre_dates].dropna()
    if len(pre_ret) == 0:
        return leveraged.copy()

    cum = (1 + pre_ret.iloc[::-1]).cumprod().iloc[::-1]
    synth_px = lev_start_px / cum

    s = pd.DataFrame(index=synth_px.index)
    s["Close"] = synth_px
    s["Open"] = s["Close"].shift(1).fillna(s["Close"].iloc[0])
    s["High"] = s[["Open", "Close"]].max(axis=1)
    s["Low"] = s[["Open", "Close"]].min(axis=1)
    s["Volume"] = 0.0

    ext = pd.concat([s, leveraged]).sort_index()
    ext = ext[~ext.index.duplicated(keep="last")].dropna(subset=["Close"])
    return ext


# ============================================================
# STRATEGY ENGINE
# ============================================================
def run_strategy(s_close_lag, s_sma_lag, lev_open, fb_open, entry_pct, exit_pct, dca_days):
    """Backtest engine used only to SCORE parameter combos during optimization.
    Signal is pre-lagged by the caller; fills at the passed Open series."""
    if entry_pct + exit_pct < -1e-9:
        return None
    sc = np.asarray(s_close_lag, float); sm = np.asarray(s_sma_lag, float)
    lo = np.asarray(lev_open, float); fo = np.asarray(fb_open, float)
    cash = 10_000.0; lev = 0.0; fb = 0.0
    state = "CASH"; dcnt = 0
    eq = np.empty(len(sc))
    for i in range(len(sc)):
        sp, sv, lp, fp = sc[i], sm[i], lo[i], fo[i]
        if np.isnan(sv) or np.isnan(sp) or np.isnan(lp) or np.isnan(fp):
            eq[i] = cash + lev * (0 if np.isnan(lp) else lp) + fb * (0 if np.isnan(fp) else fp)
            continue
        et = sv * (1 + entry_pct); xt = sv * (1 - exit_pct)
        if state == "CASH":
            if sp > et: lev = cash / lp; cash = 0.0; state = "FULL_LEV"
        elif state == "FULL_LEV":
            if sp < xt: cash = lev * lp; lev = 0.0; state = "DCA"; dcnt = 0
        elif state == "DCA":
            if sp > et:
                cash += fb * fp; fb = 0.0; lev = cash / lp; cash = 0.0; state = "FULL_LEV"
            else:
                dcnt += 1
                if cash > 0:
                    rem = max(1, dca_days - dcnt + 1)
                    buy = min(cash / rem, cash)
                    fb += buy / fp; cash -= buy
                if dcnt >= dca_days:
                    state = "DELEVERAGED"
        elif state == "DELEVERAGED":
            if sp > et:
                cash = fb * fp; fb = 0.0; lev = cash / lp; cash = 0.0; state = "FULL_LEV"
        eq[i] = cash + lev * lp + fb * fp
    return pd.Series(eq, index=getattr(s_close_lag, "index", None))


def _score(equity):
    if equity is None:
        return np.nan
    e = equity.dropna()
    if len(e) < 2:
        return np.nan
    ret = e.iloc[-1] / e.iloc[0] - 1
    dd = ((e - e.cummax()) / e.cummax()).min()
    if dd == 0 or np.isnan(dd) or np.isnan(ret):
        return np.nan
    return (ret * 100) / abs(dd * 100)


def optimize_window(qc_lag, qs_lag, lev_open, fb_open):
    """Neighbourhood-smoothed grid search -> (entry_pct, exit_pct, dca_days).
    Identical logic to insample.py / walk_foward.py optimize_window."""
    raw = {}
    for dd_ in DCA_RANGE:
        grid = np.full((len(ENTRY_RANGE), len(EXIT_RANGE)), np.nan)
        for ei, ep in enumerate(ENTRY_RANGE):
            for xi, xp in enumerate(EXIT_RANGE):
                if ep + xp < -1e-9:
                    continue
                grid[ei, xi] = _score(run_strategy(qc_lag, qs_lag, lev_open, fb_open, ep, xp, dd_))
        raw[dd_] = grid

    best_s, best = -np.inf, None
    for di, dd_ in enumerate(DCA_RANGE):
        ng = [raw[DCA_RANGE[k]] for k in (di - 1, di, di + 1) if 0 <= k < len(DCA_RANGE)]
        for ei in range(len(ENTRY_RANGE)):
            for xi in range(len(EXIT_RANGE)):
                if ENTRY_RANGE[ei] + EXIT_RANGE[xi] < -1e-9:
                    continue
                vals = []
                for g in ng:
                    for de in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ni, nx = ei + de, xi + dx
                            if 0 <= ni < len(ENTRY_RANGE) and 0 <= nx < len(EXIT_RANGE):
                                v = g[ni, nx]
                                if not np.isnan(v):
                                    vals.append(v)
                if len(vals) >= MIN_NEIGHBORS_REQUIRED:
                    m = float(np.mean(vals))
                    if m > best_s:
                        best_s, best = m, (ENTRY_RANGE[ei], EXIT_RANGE[xi], dd_)
    if best is None:
        br = -np.inf
        for dd_, g in raw.items():
            if np.all(np.isnan(g)):
                continue
            idx = np.unravel_index(np.nanargmax(g), g.shape)
            if g[idx] > br:
                br, best = g[idx], (ENTRY_RANGE[idx[0]], EXIT_RANGE[idx[1]], dd_)
    return best


def step(state, dcnt, sp, et, xt, dca_days):
    """One-day state-machine transition. `sp` is the signal close that the
    decision is based on; the resulting action happens at the NEXT open."""
    if state == "CASH":
        if sp > et:
            return "FULL_LEV", 0
    elif state == "FULL_LEV":
        if sp < xt:
            return "DCA", 0
    elif state == "DCA":
        if sp > et:
            return "FULL_LEV", 0
        dcnt += 1
        if dcnt >= dca_days:
            return "DELEVERAGED", dcnt
        return "DCA", dcnt
    elif state == "DELEVERAGED":
        if sp > et:
            return "FULL_LEV", 0
    return state, dcnt


# ============================================================
# STATE FILE
# ============================================================
def load_state():
    if os.path.exists(STATE_JSON):
        with open(STATE_JSON) as f:
            return json.load(f)
    return {"version": 1, "config": {}, "params_history": [], "override": None, "last_run": None}


def save_state(st):
    st["config"] = {"signal": SIGNAL_TICKER, "leveraged": LEVERAGED_TICKER, "fallback": FALLBACK_TICKER,
                    "ma": MA_PERIOD, "in_sample_years": IN_SAMPLE_YEARS,
                    "reopt_days": REOPTIMIZE_EVERY_TRADING_DAYS}
    st["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(STATE_JSON, "w") as f:
        json.dump(st, f, indent=2)


def config_changed(st):
    c = st.get("config") or {}
    return c and (c.get("in_sample_years") != IN_SAMPLE_YEARS
                  or c.get("ma") != MA_PERIOD
                  or c.get("reopt_days") != REOPTIMIZE_EVERY_TRADING_DAYS
                  or c.get("signal") != SIGNAL_TICKER
                  or c.get("leveraged") != LEVERAGED_TICKER)


# ============================================================
# CORE
# ============================================================
def reopt_indices(n_bars):
    """Bar indices at which parameters are (re)chosen: first once IN_LEN bars
    of history exist, then every REOPTIMIZE_EVERY_TRADING_DAYS."""
    return list(range(IN_LEN, n_bars, REOPTIMIZE_EVERY_TRADING_DAYS))


def ensure_params(st, sig_close, sig_sma, lev_open, fb_open, dates, force_now=False):
    qc_lag = sig_close.shift(1)
    qs_lag = sig_sma.shift(1)
    history = st["params_history"]
    have = {h["as_of"] for h in history}
    idxs = reopt_indices(len(dates))

    if force_now and len(dates) > IN_LEN:
        idxs = sorted(set(idxs) | {len(dates) - 1})

    new = 0
    for i in idxs:
        as_of = dates[i].strftime("%Y-%m-%d")
        if as_of in have:
            continue
        sl = slice(i - IN_LEN, i)
        bp = optimize_window(qc_lag.iloc[sl], qs_lag.iloc[sl], lev_open.iloc[sl], fb_open.iloc[sl])
        if bp is None:
            continue
        history.append({"as_of": as_of, "entry_pct": float(bp[0]),
                        "exit_pct": float(bp[1]), "dca_days": int(bp[2])})
        have.add(as_of)
        new += 1
        print(f"  re-optimized as of {as_of}:  entry={bp[0]:+.0%}  exit={bp[1]:+.0%}  dca_days={bp[2]}")
    history.sort(key=lambda h: h["as_of"])
    if new == 0:
        print("  parameters already current (no new re-optimization date has passed)")
    return history


def params_at(history, d):
    chosen = None
    for h in history:
        if pd.Timestamp(h["as_of"]) <= d:
            chosen = h
        else:
            break
    return chosen


def replay(history, sig_close, sig_sma, dates, through_idx, override):
    """Advance the state machine to `through_idx` (inclusive). Returns
    (state, dca_day_counter, effective_from_date)."""
    state, dcnt = "CASH", 0
    start_i = 0
    if override:
        ov_date = pd.Timestamp(override["as_of"])
        state = override["state"]
        dcnt = int(override.get("dca_day_counter", 0))
        start_i = int(dates.searchsorted(ov_date))
    else:
        if history:
            start_i = int(dates.searchsorted(pd.Timestamp(history[0]["as_of"])))

    sc = sig_close.values
    sm = sig_sma.values
    for i in range(start_i, through_idx + 1):
        p = params_at(history, dates[i])
        if p is None or np.isnan(sc[i]) or np.isnan(sm[i]):
            continue
        et = sm[i] * (1 + p["entry_pct"])
        xt = sm[i] * (1 - p["exit_pct"])
        state, dcnt = step(state, dcnt, sc[i], et, xt, p["dca_days"])
    return state, dcnt


# ============================================================
# REPORT
# ============================================================
def pct(a, b):
    return (a / b - 1.0) * 100.0


def recommend(state, dcnt, close, sma, p, last_date):
    entry_t = sma * (1 + p["entry_pct"])
    exit_t = sma * (1 - p["exit_pct"])
    L, F = LEVERAGED_TICKER, FALLBACK_TICKER
    lines = []
    code = ""

    if state == "CASH":
        if close > entry_t:
            code = "BUY"
            lines.append(f"BUY  ->  go all-in {L} at the next open.")
            lines.append(f"     {SIGNAL_TICKER} closed {close:,.2f}, above the entry trigger "
                         f"{entry_t:,.2f} (+{pct(close, entry_t):.2f}%).")
        else:
            code = "HOLD_CASH"
            lines.append("STAY IN CASH.")
            lines.append(f"     Enter {L} only when {SIGNAL_TICKER} CLOSES above {entry_t:,.2f} "
                         f"(needs {pct(entry_t, close):+.2f}% from here).")

    elif state == "FULL_LEV":
        if close < exit_t:
            code = "SELL_START_DCA"
            lines.append(f"SELL  ->  move 100% {L} -> cash at the next open, then BEGIN DCA.")
            lines.append(f"     {SIGNAL_TICKER} closed {close:,.2f}, below the exit trigger "
                         f"{exit_t:,.2f} ({pct(close, exit_t):.2f}%).")
            lines.append(f"     Then deploy that cash into {F} over {p['dca_days']} trading days "
                         f"(~{100.0 / p['dca_days']:.2f}% of the pool per day),")
            lines.append(f"     unless {SIGNAL_TICKER} closes back above {entry_t:,.2f} first "
                         f"(then go all-in {L}).")
        else:
            code = "HOLD_LEV"
            lines.append(f"HOLD {L}.")
            lines.append(f"     Exit only if {SIGNAL_TICKER} CLOSES below {exit_t:,.2f} "
                         f"(cushion {pct(close, exit_t):+.2f}% right now).")

    elif state == "DCA":
        if close > entry_t:
            code = "DCA_REENTER"
            lines.append(f"RE-ENTER  ->  sell your {F} holdings and go all-in {L} at the next open.")
            lines.append(f"     {SIGNAL_TICKER} closed {close:,.2f}, above the entry trigger "
                         f"{entry_t:,.2f} (+{pct(close, entry_t):.2f}%).")
        else:
            code = "DCA_BUY"
            buys_left = p["dca_days"] - dcnt            # buys remaining incl. tomorrow's
            frac = 1.0 / max(1, buys_left)
            done_date = np.busday_offset(last_date.date(), buys_left, roll="forward")
            lines.append(f"CONTINUE DCA  ->  buy {F} at the next open with 1/{buys_left} "
                         f"~ {frac * 100:.2f}% of your REMAINING cash.")
            lines.append(f"     DCA day {dcnt + 1} of {p['dca_days']}  ({buys_left - 1} buys left after this one).")
            lines.append(f"     Abort DCA and go all-in {L} if {SIGNAL_TICKER} closes above "
                         f"{entry_t:,.2f} ({pct(entry_t, close):+.2f}% away).")
            lines.append(f"     If it runs to completion (~{done_date}) you'll be fully in {F} "
                         f"(DELEVERAGED) awaiting a re-entry signal.")

    elif state == "DELEVERAGED":
        if close > entry_t:
            code = "DELEV_REENTER"
            lines.append(f"RE-ENTER  ->  sell {F} and go all-in {L} at the next open.")
            lines.append(f"     {SIGNAL_TICKER} closed {close:,.2f}, above the entry trigger "
                         f"{entry_t:,.2f} (+{pct(close, entry_t):.2f}%).")
        else:
            code = "HOLD_DELEV"
            lines.append(f"HOLD {F}  (unleveraged).")
            lines.append(f"     Re-enter {L} when {SIGNAL_TICKER} CLOSES above {entry_t:,.2f} "
                         f"(needs {pct(entry_t, close):+.2f}% from here).")

    return code, lines, entry_t, exit_t


def print_report(dates, sig_close, sig_sma, state, dcnt, history, provisional):
    last = len(dates) - 1
    d = dates[last]
    close = float(sig_close.iloc[last])
    sma = float(sig_sma.iloc[last])
    p = params_at(history, d) or history[-1]
    dist = pct(close, sma)

    # 5-day change in distance-from-SMA, as a trend hint
    trend = ""
    if last >= 5 and not np.isnan(sig_sma.iloc[last - 5]):
        prev = pct(float(sig_close.iloc[last - 5]), float(sig_sma.iloc[last - 5]))
        trend = f"   (5 sessions ago: {prev:+.2f}%  ->  {dist - prev:+.2f} pts)"

    code, rec, entry_t, exit_t = recommend(state, dcnt, close, sma, p, d)

    n_bars = len(dates)
    idxs = reopt_indices(n_bars)
    last_opt_date = pd.Timestamp(history[-1]["as_of"]) if history else None
    next_opt_bar = None
    for i in idxs + [idxs[-1] + REOPTIMIZE_EVERY_TRADING_DAYS if idxs else IN_LEN]:
        if i >= n_bars:
            next_opt_bar = i
            break
    days_to_next = (next_opt_bar - (n_bars - 1)) if next_opt_bar else None

    W = 64
    print()
    print("=" * W)
    print(f"  LEVERAGED MA{MA_PERIOD} ADVISOR   ({SIGNAL_TICKER} signal / {LEVERAGED_TICKER} / {FALLBACK_TICKER})")
    print(f"  run {datetime.now():%Y-%m-%d %H:%M}   |   data through {d:%Y-%m-%d} close")
    if provisional:
        print("  ! latest bar is TODAY - if the market has not closed, this close")
        print("    is provisional; re-run after 16:00 ET.")
    print("=" * W)
    print(f"  MARKET  ({SIGNAL_TICKER})")
    print(f"    close .................... {close:>12,.2f}")
    print(f"    {MA_PERIOD}-day SMA ............. {sma:>12,.2f}")
    print(f"    price vs SMA ............ {dist:>+11.2f}%   {'ABOVE' if dist >= 0 else 'BELOW'}{trend}")
    print()
    print(f"  ACTIVE PARAMETERS   (walk-forward re-opt as of {p['as_of']})")
    print(f"    entry buffer ........... {p['entry_pct']:>+11.0%}   -> enter trigger  {entry_t:>10,.2f}"
          f"   ({pct(close, entry_t):+.2f}% vs close)")
    print(f"    exit buffer ............ {p['exit_pct']:>+11.0%}   -> exit trigger   {exit_t:>10,.2f}"
          f"   ({pct(close, exit_t):+.2f}% vs close)")
    print(f"    DCA length ............. {p['dca_days']:>8d} d")
    print()
    print("  POSITION")
    pos = state if state != "DCA" else f"DCA  (day {dcnt} of {p['dca_days']})"
    print(f"    tracked state .......... {pos}")
    hold = {"CASH": "cash", "FULL_LEV": LEVERAGED_TICKER, "DCA": f"cash draining into {FALLBACK_TICKER}",
            "DELEVERAGED": FALLBACK_TICKER}[state]
    print(f"    currently holding ...... {hold}")
    print()
    print("  NEXT MOVE   (execute at the next session's open)")
    for ln in rec:
        print(f"    {ln}")
    print()
    print("  RE-OPTIMIZATION")
    print(f"    last ................... {last_opt_date:%Y-%m-%d}" if last_opt_date is not None else
          "    last ................... (none yet)")
    if days_to_next is not None:
        print(f"    next .................. in ~{days_to_next} trading days")
    print("=" * W)
    return d, close, sma, dist, state, dcnt, code, p


def append_log(run_date, data_date, close, sma, dist, state, dcnt, code, p):
    row = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_date": data_date.strftime("%Y-%m-%d"),
        "close": round(close, 4), "sma": round(sma, 4), "pct_from_sma": round(dist, 4),
        "state": state, "dca_day": dcnt, "action": code,
        "entry_pct": p["entry_pct"], "exit_pct": p["exit_pct"], "dca_days": p["dca_days"],
    }
    df = pd.DataFrame([row])
    if os.path.exists(LOG_CSV):
        old = pd.read_csv(LOG_CSV)
        old = old[old["data_date"] != row["data_date"]]        # replace same-day rerun
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(LOG_CSV, index=False)


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Daily signal for the leveraged MA200 strategy.")
    ap.add_argument("--update", action="store_true", help="force a fresh data download")
    ap.add_argument("--no-update", action="store_true", help="never download; use cached CSVs")
    ap.add_argument("--optimize-now", action="store_true", help="force a re-optimization this run")
    ap.add_argument("--reset", action="store_true", help="wipe parameter history and rebuild")
    ap.add_argument("--set-state", choices=STATES, help="override the tracked position, effective today")
    ap.add_argument("--dca-day", type=int, default=0, help="with --set-state DCA: the current DCA day counter")
    ap.add_argument("--history", type=int, metavar="N", help="print the last N logged days and exit")
    args = ap.parse_args()

    if args.history:
        if not os.path.exists(LOG_CSV):
            sys.exit("no signal_log.csv yet")
        print(pd.read_csv(LOG_CSV).tail(args.history).to_string(index=False))
        return

    mode = "force" if args.update else ("never" if args.no_update else "auto")

    print("Refreshing data ...")
    sig = refresh(SIGNAL_TICKER, SIGNAL_CSV, mode).dropna(subset=["Open", "Close"])
    lev_real = refresh(LEVERAGED_TICKER, LEVERAGED_CSV, mode).dropna(subset=["Open", "Close"])
    lev_ext = build_extended(sig, lev_real)
    lev_ext.to_csv(LEVERAGED_EXT)

    common = sig.index.intersection(lev_ext.index)
    sig = sig.loc[common]
    lev_ext = lev_ext.loc[common]
    dates = sig.index

    sig_close = sig["Close"]
    sig_sma = sig["Close"].rolling(MA_PERIOD).mean()
    lev_open = lev_ext["Open"]
    fb_open = sig["Open"]

    if len(dates) < IN_LEN + 5:
        sys.exit(f"ERROR: only {len(dates)} bars; need > {IN_LEN}")

    st = load_state()
    if args.reset:
        st = {"version": 1, "config": {}, "params_history": [], "override": None, "last_run": None}
        print("  parameter history wiped (--reset)")
    if config_changed(st):
        print("  ! CONFIG CHANGED since last run (e.g. IN_SAMPLE_YEARS). Run with --reset "
              "to rebuild the parameter history on the new config.")

    if args.set_state:
        st["override"] = {"as_of": dates[-1].strftime("%Y-%m-%d"),
                          "state": args.set_state, "dca_day_counter": args.dca_day}
        print(f"  position override set: {args.set_state} (dca_day={args.dca_day}) as of {dates[-1].date()}")

    print("Checking parameters ...")
    history = ensure_params(st, sig_close, sig_sma, lev_open, fb_open, dates,
                            force_now=args.optimize_now)
    save_state(st)

    # last bar may be an unclosed session today
    provisional = dates[-1].date() == datetime.now().date()

    # replay through the last COMPLETED signal bar (the one before the bar we
    # base the recommendation on); the recommendation itself uses the last bar
    through = len(dates) - 2
    state, dcnt = replay(history, sig_close, sig_sma, dates, through, st.get("override"))

    d, close, sma, dist, state_r, dcnt_r, code, p = print_report(
        dates, sig_close, sig_sma, state, dcnt, history, provisional)
    append_log(datetime.now(), d, close, sma, dist, state_r, dcnt_r, code, p)
    print(f"  logged -> {os.path.basename(LOG_CSV)}")


if __name__ == "__main__":
    main()
