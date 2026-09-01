"""
12_make_figures.py

Regenerates every PNG in ../figures/ from committed data + results:

  sweep_tqqq_qqq.png    entry x exit heatmaps per dca_days  (return + drawdown)
  sweep_spxl_spy.png    same, SPXL/SPY
  insample_window.png   common-window Sharpe / regime coverage / folds vs IN_SAMPLE_YEARS
  walkforward_equity.png  strategy (IS=8) vs buy & hold TQQQ, with drawdown panel

Pure matplotlib, no browser, no kaleido. The equity figure re-runs one
walk-forward (~40 s); the rest are instant.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
D = lambda _f: _os.path.join(_ROOT, "data", _f)
R = lambda _f: _os.path.join(_ROOT, "results", _f)
FIG = lambda _f: _os.path.join(_ROOT, "figures", _f)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

_os.makedirs(_os.path.join(_ROOT, "figures"), exist_ok=True)
plt.rcParams.update({"figure.facecolor": "white", "savefig.dpi": 110, "font.size": 9})


# ============================================================
# 1-2. sweep heatmaps
# ============================================================
def sweep_heatmaps(csv, title, out):
    df = pd.read_csv(csv)
    dcas = sorted(df["dca_days"].unique())
    fig, axes = plt.subplots(2, len(dcas), figsize=(2.6 * len(dcas), 6.8),
                             squeeze=False, constrained_layout=True)

    rnorm = LogNorm(vmin=np.nanmin(df["return_pct"]), vmax=np.nanmax(df["return_pct"]))
    for col, dca in enumerate(dcas):
        sub = df[df["dca_days"] == dca]
        ret = sub.pivot(index="exit_pct", columns="entry_pct", values="return_pct")
        dd = sub.pivot(index="exit_pct", columns="entry_pct", values="max_dd_pct")

        ax = axes[0][col]
        im_r = ax.imshow(ret.values, cmap="viridis", norm=rnorm, origin="lower", aspect="auto")
        ax.set_title(f"dca={dca}d", fontsize=9)
        if col == 0:
            ax.set_ylabel("RETURN %\n\nexit_pct")
        _ticks(ax, ret)

        ax = axes[1][col]
        im_d = ax.imshow(dd.values, cmap="RdYlGn", vmin=-90, vmax=-40, origin="lower", aspect="auto")
        if col == 0:
            ax.set_ylabel("MAX DRAWDOWN %\n\nexit_pct")
        ax.set_xlabel("entry_pct")
        _ticks(ax, dd)

    fig.suptitle(f"{title}  —  parameter sweep (full history, look-ahead corrected)\n"
                 "white cells = entry_pct + exit_pct < 0, not evaluated", fontsize=10)
    fig.colorbar(im_r, ax=axes[0].tolist(), shrink=0.85, label="total return % (log)", pad=0.01)
    fig.colorbar(im_d, ax=axes[1].tolist(), shrink=0.85, label="max drawdown %", pad=0.01)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", _os.path.relpath(out, _ROOT))


def _ticks(ax, piv):
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c:+.0%}" for c in piv.columns], fontsize=6, rotation=90)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{i:+.0%}" for i in piv.index], fontsize=6)


# ============================================================
# 3. in-sample window analysis
# ============================================================
def insample_window(out):
    df = pd.read_csv(R("insample_sharpe_cagr.csv")).sort_values("in_sample_years")
    x = df["in_sample_years"]
    se = np.sqrt((1 + 0.5 * df["common_sharpe"] ** 2) / df["years"])  # ~Sharpe standard error

    fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax[0].fill_between(x, df["common_sharpe"] - se, df["common_sharpe"] + se,
                       alpha=0.15, color="tab:blue", label="±1 std err")
    ax[0].plot(x, df["common_sharpe"], "o-", color="tab:blue", label="common-window Sharpe")
    ax[0].axvline(8, color="tab:red", ls="--", lw=1, label="_nolegend_")
    ax[0].text(8.15, df["common_sharpe"].min(), "chosen: 8y", color="tab:red", fontsize=8)
    ax[0].set_ylabel("Sharpe (2014→now, rf=0)")
    ax[0].set_title("Risk-adjusted performance is flat across 5–10y (spread << 1 std err)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax2 = ax[1]
    l1, = ax2.plot(x, df["regime_cov_pct"], "s-", color="tab:green", label="regime coverage %")
    ax2.axvline(8, color="tab:red", ls="--", lw=1, label="_nolegend_")
    ax2.set_ylabel("% of windows that saw a ≥25% drawdown", color="tab:green")
    ax2.set_xlabel("IN_SAMPLE_YEARS")
    ax2.grid(alpha=0.3)
    ax3 = ax2.twinx()
    l2, = ax3.plot(x, df["n_folds"], "^:", color="tab:orange", label="walk-forward folds")
    ax3.set_ylabel("folds", color="tab:orange")
    ax2.set_title("The two non-noisy criteria trade off smoothly → 8y is the midpoint")
    ax2.legend([l1, l2], [l1.get_label(), l2.get_label()], fontsize=8, loc="center right")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", _os.path.relpath(out, _ROOT))


# ============================================================
# 4. walk-forward equity  (IS=8)  vs buy & hold TQQQ
# ============================================================
MA, TDY, IN_LEN = 200, 252, 8 * 252
E_RANGE = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
X_RANGE = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
DCA_RANGE = [126, 189, 252, 315]
MIN_NB = 3


def _load(ext, plain):
    return pd.read_csv(D(ext) if _os.path.exists(D(ext)) else D(plain),
                       index_col="Date", parse_dates=True)


def _run(sc, sm, lo, fo, ep, xp, dd, cash=1e4, st="CASH", lv=0.0, fb=0.0, dcnt=0):
    if ep + xp < -1e-9:
        return None, None
    sc, sm, lo, fo = (np.asarray(a, float) for a in (sc, sm, lo, fo))
    eq = np.empty(len(sc))
    for i in range(len(sc)):
        sp, sv, lp, fp = sc[i], sm[i], lo[i], fo[i]
        if np.isnan(sv) or np.isnan(sp) or np.isnan(lp) or np.isnan(fp):
            eq[i] = cash + lv * (0 if np.isnan(lp) else lp) + fb * (0 if np.isnan(fp) else fp)
            continue
        et, xt = sv * (1 + ep), sv * (1 - xp)
        if st == "CASH":
            if sp > et: lv, cash, st = cash / lp, 0.0, "FULL_LEV"
        elif st == "FULL_LEV":
            if sp < xt: cash, lv, st, dcnt = lv * lp, 0.0, "DCA", 0
        elif st == "DCA":
            if sp > et:
                cash += fb * fp; fb = 0.0; lv = cash / lp; cash = 0.0; st = "FULL_LEV"
            else:
                dcnt += 1
                if cash > 0:
                    b = min(cash / max(1, dd - dcnt + 1), cash); fb += b / fp; cash -= b
                if dcnt >= dd: st = "DELEVERAGED"
        elif st == "DELEVERAGED":
            if sp > et:
                cash = fb * fp; fb = 0.0; lv = cash / lp; cash = 0.0; st = "FULL_LEV"
        eq[i] = cash + lv * lp + fb * fp
    return pd.Series(eq, index=None), dict(cash=cash, st=st, lv=lv, fb=fb, dcnt=dcnt)


def _score(eq):
    if eq is None or eq.dropna().shape[0] < 2:
        return np.nan
    e = eq.dropna()
    r = e.iloc[-1] / e.iloc[0] - 1
    d = ((e - e.cummax()) / e.cummax()).min()
    return np.nan if d == 0 else (r * 100) / abs(d * 100)


def _optimize(qc, qs, lo, fo):
    raw = {}
    for dd in DCA_RANGE:
        g = np.full((len(E_RANGE), len(X_RANGE)), np.nan)
        for ei, ep in enumerate(E_RANGE):
            for xi, xp in enumerate(X_RANGE):
                if ep + xp < -1e-9:
                    continue
                eq, _ = _run(qc, qs, lo, fo, ep, xp, dd)
                if eq is not None:
                    eq.index = qc.index
                g[ei, xi] = _score(eq)
        raw[dd] = g
    bs, best = -np.inf, None
    for di, dd in enumerate(DCA_RANGE):
        ng = [raw[DCA_RANGE[k]] for k in (di - 1, di, di + 1) if 0 <= k < len(DCA_RANGE)]
        for ei in range(len(E_RANGE)):
            for xi in range(len(X_RANGE)):
                if E_RANGE[ei] + X_RANGE[xi] < -1e-9:
                    continue
                v = [g[a, b] for g in ng for a in (ei-1, ei, ei+1) for b in (xi-1, xi, xi+1)
                     if 0 <= a < len(E_RANGE) and 0 <= b < len(X_RANGE) and not np.isnan(g[a, b])]
                if len(v) >= MIN_NB and np.mean(v) > bs:
                    bs, best = np.mean(v), (E_RANGE[ei], X_RANGE[xi], dd)
    return best


def walkforward_equity(out):
    qqq = _load("qqq_yf.csv", "qqq_yf.csv").dropna(subset=["Open", "Close"])
    tqqq = _load("tqqq_extended_yf.csv", "tqqq_yf.csv").dropna(subset=["Open", "Close"])
    idx = qqq.index.intersection(tqqq.index)
    qqq, tqqq = qqq.loc[idx], tqqq.loc[idx]
    sma = qqq["Close"].rolling(MA).mean()
    qcl, qsl = qqq["Close"].shift(1), sma.shift(1)
    dates = qqq.index

    segs, carry, si = [], dict(cash=1e4, st="CASH", lv=0.0, fb=0.0, dcnt=0), 0
    while si + IN_LEN + TDY <= len(dates):
        isd = dates[si:si + IN_LEN]
        osd = dates[si + IN_LEN: si + IN_LEN + TDY]
        bp = _optimize(qcl.loc[isd], qsl.loc[isd], tqqq.loc[isd, "Open"], qqq.loc[isd, "Open"])
        if bp:
            eq, fs = _run(qcl.loc[osd], qsl.loc[osd], tqqq.loc[osd, "Open"], qqq.loc[osd, "Open"],
                          *bp, cash=carry["cash"], st=carry["st"], lv=carry["lv"],
                          fb=carry["fb"], dcnt=carry["dcnt"])
            eq.index = osd
            carry = fs
            segs.append(eq)
        si += TDY

    strat = pd.concat(segs).dropna()
    bh = tqqq.loc[strat.index, "Close"]
    bh = bh / bh.iloc[0] * strat.iloc[0]

    def dd(s):
        return (s - s.cummax()) / s.cummax() * 100

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[2.4, 1])
    ax[0].plot(strat.index, strat, label="strategy (walk-forward, IS=8y)", color="tab:blue", lw=1.2)
    ax[0].plot(bh.index, bh, label="buy & hold TQQQ", color="tab:gray", lw=1, alpha=0.8)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("equity  ($10k start, log)")
    ax[0].legend()
    ax[0].grid(alpha=0.3, which="both")
    ax[0].set_title("Out-of-sample walk-forward: strategy vs buy & hold TQQQ")

    ax[1].fill_between(strat.index, dd(strat), 0, color="tab:blue", alpha=0.4, label="strategy")
    ax[1].fill_between(bh.index, dd(bh), 0, color="tab:gray", alpha=0.3, label="buy & hold")
    ax[1].set_ylabel("drawdown %")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    cagr = (strat.iloc[-1] / strat.iloc[0]) ** (TDY / len(strat)) - 1
    mdd = dd(strat).min()
    ax[0].text(0.01, 0.97, f"CAGR {cagr*100:.1f}%   max DD {mdd:.0f}%   "
               f"(plan for ~-75%)", transform=ax[0].transAxes, va="top", fontsize=9,
               bbox=dict(boxstyle="round", fc="white", ec="tab:blue", alpha=0.8))

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", _os.path.relpath(out, _ROOT))


if __name__ == "__main__":
    sweep_heatmaps(R("sweep_tqqq_qqq_nolookahead.csv"), "TQQQ / QQQ", FIG("sweep_tqqq_qqq.png"))
    sweep_heatmaps(R("sweep_spxl_spy_nolookahead.csv"), "SPXL / SPY", FIG("sweep_spxl_spy.png"))
    insample_window(FIG("insample_window.png"))
    walkforward_equity(FIG("walkforward_equity.png"))
    print("done")
