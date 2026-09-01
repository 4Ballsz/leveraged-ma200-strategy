"""
insample_riskadjusted.py

Companion to insample.py. For each candidate IN_SAMPLE_YEARS it runs the same
walk-forward (grid-search re-optimization every fold) and then reports the
metrics insample.py leaves out:

  - CAGR, annualized vol, Sharpe (rf = 0), Sortino, Calmar
  - the SAME metrics restricted to the window every candidate shares
    (fixes the confound that a shorter in-sample window makes the
     out-of-sample track START earlier, so longer total return / higher
     Sharpe is partly just "caught more good years", not "better params")
  - regime coverage: what fraction of the optimization windows had actually
    lived through a >=25% QQQ drawdown before parameters were chosen

Takeaway from the run on current data: risk-adjusted performance is flat
across IN_SAMPLE_YEARS 5-10 (Sharpe spread << 1 standard error), so the
choice comes down to regime coverage vs fold count -> 8 years.

Writes insample_sharpe_cagr.csv.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
D = lambda _f: _os.path.join(_ROOT, "data", _f)      # price data (inputs)
R = lambda _f: _os.path.join(_ROOT, "results", _f)   # generated outputs

import os
import numpy as np
import pandas as pd

qqq = pd.read_csv(D("qqq_yf.csv"), index_col="Date", parse_dates=True).dropna(subset=['Open', 'Close'])


def load_best_available(ext, plain, label):
    if os.path.exists(ext):
        return pd.read_csv(ext, index_col="Date", parse_dates=True)
    print(f"{label}: WARNING real-only")
    return pd.read_csv(plain, index_col="Date", parse_dates=True)


tqqq = load_best_available(D("tqqq_extended_yf.csv"), D("tqqq_yf.csv"), "TQQQ").dropna(subset=['Open', 'Close'])
common = qqq.index.intersection(tqqq.index)
qqq, tqqq = qqq.loc[common], tqqq.loc[common]

ma_period = 200
init_cash = 10_000.0
entry_range = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
exit_range = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
dca_range = [126, 189, 252, 315]
MIN_NEIGHBORS_REQUIRED = 3
OUT_SAMPLE_YEARS = 1
TRADING_DAYS_PER_YEAR = 252
out_sample_len = OUT_SAMPLE_YEARS * TRADING_DAYS_PER_YEAR

YEARS = [3, 5, 6, 7, 8, 9, 10, 15]

qqq_sma = qqq['Close'].rolling(window=ma_period).mean()
qqq_close_lagged_full = qqq['Close'].shift(1)
qqq_sma_lagged_full = qqq_sma.shift(1)


def run_strategy(s_close_lag, s_sma_lag, lev_open, fb_open, entry_pct, exit_pct, dca_days,
                 start_cash=init_cash, init_state="CASH", init_lev_shares=0.0,
                 init_fallback_shares=0.0, init_dca_day_counter=0):
    if entry_pct + exit_pct < -1e-9:
        return None, None, None
    sc = s_close_lag.values; sm = s_sma_lag.values; lo = lev_open.values; fo = fb_open.values
    cash = start_cash; lev = init_lev_shares; fb = init_fallback_shares
    state = init_state; dcnt = init_dca_day_counter
    eq = np.empty(len(sc))
    for i in range(len(sc)):
        sp, sv, lp, fp = sc[i], sm[i], lo[i], fo[i]
        if np.isnan(sv) or np.isnan(sp) or np.isnan(lp) or np.isnan(fp):
            eq[i] = cash + lev*(lp if not np.isnan(lp) else 0) + fb*(fp if not np.isnan(fp) else 0)
            continue
        et = sv*(1+entry_pct); xt = sv*(1-exit_pct)
        if state == "CASH":
            if sp > et: lev = cash/lp; cash = 0.0; state = "FULL_LEV"
        elif state == "FULL_LEV":
            if sp < xt: cash = lev*lp; lev = 0.0; state = "DCA"; dcnt = 0
        elif state == "DCA":
            if sp > et:
                cash += fb*fp; fb = 0.0; lev = cash/lp; cash = 0.0; state = "FULL_LEV"
            else:
                dcnt += 1
                if cash > 0:
                    rem = max(1, dca_days - dcnt + 1)
                    buy = min(cash/rem, cash)
                    fb += buy/fp; cash -= buy
                if dcnt >= dca_days: state = "DELEVERAGED"
        elif state == "DELEVERAGED":
            if sp > et:
                cash = fb*fp; fb = 0.0; lev = cash/lp; cash = 0.0; state = "FULL_LEV"
        eq[i] = cash + lev*lp + fb*fp
    equity = pd.Series(eq, index=s_close_lag.index)
    fs = {"state": state, "cash": cash, "lev_shares": lev, "fallback_shares": fb, "dca_day_counter": dcnt}
    return equity, 0, fs


def calc_stats(equity):
    if equity is None: return np.nan, np.nan
    e = equity.dropna()
    if len(e) < 2: return np.nan, np.nan
    ret = (e.iloc[-1]/e.iloc[0] - 1)*100
    dd = ((e - e.cummax())/e.cummax()*100).min()
    return ret, dd


def optimize_window(qc, qs, tq_open, qq_open):
    raw = {}
    for dd_ in dca_range:
        grid = np.full((len(entry_range), len(exit_range)), np.nan)
        for ei, ep in enumerate(entry_range):
            for xi, xp in enumerate(exit_range):
                if ep + xp < -1e-9: continue
                eq, _, _ = run_strategy(qc, qs, tq_open, qq_open, ep, xp, dd_)
                r, d = calc_stats(eq)
                if np.isnan(r) or np.isnan(d) or d == 0: continue
                grid[ei, xi] = r/abs(d)
        raw[dd_] = grid
    best_s = -np.inf; best = None
    for di, dd_ in enumerate(dca_range):
        ng = [raw[dca_range[k]] for k in (di-1, di, di+1) if 0 <= k < len(dca_range)]
        for ei in range(len(entry_range)):
            for xi in range(len(exit_range)):
                if entry_range[ei] + exit_range[xi] < -1e-9: continue
                vals = []
                for g in ng:
                    for de in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ni, nx = ei+de, xi+dx
                            if 0 <= ni < len(entry_range) and 0 <= nx < len(exit_range):
                                v = g[ni, nx]
                                if not np.isnan(v): vals.append(v)
                if len(vals) >= MIN_NEIGHBORS_REQUIRED:
                    sm_ = float(np.mean(vals))
                    if sm_ > best_s: best_s = sm_; best = (entry_range[ei], exit_range[xi], dd_)
    if best is None:
        br = -np.inf
        for dd_, g in raw.items():
            if np.all(np.isnan(g)): continue
            idx = np.unravel_index(np.nanargmax(g), g.shape)
            if g[idx] > br: br = g[idx]; best = (entry_range[idx[0]], exit_range[idx[1]], dd_)
    return best


def walkforward(in_sample_years):
    in_len = in_sample_years * TRADING_DAYS_PER_YEAR
    dates = qqq.index
    segs = []; carry = dict(cash=init_cash, state="CASH", lev=0.0, fb=0.0, dcnt=0)
    si = 0; nfolds = 0
    while si + in_len + out_sample_len <= len(dates):
        isd = dates[si:si+in_len]
        osd = dates[si+in_len: si+in_len+out_sample_len]
        bp = optimize_window(qqq_close_lagged_full.loc[isd], qqq_sma_lagged_full.loc[isd],
                             tqqq.loc[isd]['Open'], qqq.loc[isd]['Open'])
        if bp is None:
            si += out_sample_len; continue
        ep, xp, dd_ = bp
        eq, _, fs = run_strategy(
            qqq_close_lagged_full.loc[osd], qqq_sma_lagged_full.loc[osd],
            tqqq.loc[osd]['Open'], qqq.loc[osd]['Open'], ep, xp, dd_,
            start_cash=carry["cash"], init_state=carry["state"],
            init_lev_shares=carry["lev"], init_fallback_shares=carry["fb"],
            init_dca_day_counter=carry["dcnt"])
        if eq is None:
            si += out_sample_len; continue
        carry = dict(cash=fs["cash"], state=fs["state"], lev=fs["lev_shares"],
                     fb=fs["fallback_shares"], dcnt=fs["dca_day_counter"])
        segs.append(eq.dropna()); nfolds += 1
        si += out_sample_len
    if not segs: return None
    return pd.concat(segs).dropna(), nfolds


def metrics(wf):
    e = wf.dropna()
    daily = e.pct_change().dropna()
    years = len(e) / TRADING_DAYS_PER_YEAR
    total_ret = e.iloc[-1]/e.iloc[0] - 1
    cagr = (e.iloc[-1]/e.iloc[0])**(1/years) - 1
    sharpe = daily.mean()/daily.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    downside = daily[daily < 0].std(ddof=1)
    sortino = daily.mean()/downside * np.sqrt(TRADING_DAYS_PER_YEAR)
    ann_vol = daily.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    dd = ((e - e.cummax())/e.cummax()).min()
    return dict(years=years, total_ret=total_ret*100, cagr=cagr*100, ann_vol=ann_vol*100,
                sharpe=sharpe, sortino=sortino, max_dd=dd*100, calmar=cagr/abs(dd))


def coverage_stats(in_sample_years, threshold=-25.0):
    in_len = in_sample_years * TRADING_DAYS_PER_YEAR
    dates = qqq.index
    dds = []
    si = 0
    while si + in_len + out_sample_len <= len(dates):
        c = qqq.loc[dates[si:si+in_len], 'Close']
        dds.append(((c - c.cummax())/c.cummax()*100).min())
        si += out_sample_len
    dds = pd.Series(dds)
    return (dds <= threshold).mean()*100, dds.max()


if __name__ == "__main__":
    wfs = {y: walkforward(y) for y in YEARS}
    common_start = max(wf.index[0] for wf, _ in wfs.values())
    print(f"common overlap window starts {common_start.date()}\n")

    rows = []
    for y in YEARS:
        wf, nf = wfs[y]
        m = metrics(wf); c = metrics(wf.loc[common_start:])
        cov_pct, mildest_blind = coverage_stats(y)
        m.update(in_sample_years=y, n_folds=nf, regime_cov_pct=cov_pct, mildest_blind_dd=mildest_blind)
        for k, v in c.items():
            m["common_" + k] = v
        rows.append(m)
        print(f"IS={y:>2}y f={nf:>2} | FULL CAGR={m['cagr']:6.2f}% Sharpe={m['sharpe']:.3f} "
              f"maxDD={m['max_dd']:6.1f}% || COMMON CAGR={c['cagr']:6.2f}% Sharpe={c['sharpe']:.3f} "
              f"Sortino={c['sortino']:.3f} maxDD={c['max_dd']:6.1f}% Calmar={c['calmar']:.3f} || "
              f"regime_cov={cov_pct:5.1f}% mildest_blind={mildest_blind:6.1f}%")

    df = pd.DataFrame(rows)[["in_sample_years", "n_folds", "regime_cov_pct", "mildest_blind_dd",
                             "years", "total_ret", "cagr", "ann_vol", "sharpe", "sortino",
                             "max_dd", "calmar", "common_cagr", "common_ann_vol", "common_sharpe",
                             "common_sortino", "common_max_dd", "common_calmar"]]
    df.to_csv(R("insample_sharpe_cagr.csv"), index=False)
    print("\nsaved insample_sharpe_cagr.csv")
    print(df.to_string(index=False))
