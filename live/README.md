# Live Advisor — daily operating folder

Everything needed to run the leveraged MA200 strategy day to day. Nothing here
depends on the research folder.

```
daily_advisor.py        the program
qqq_yf.csv              signal data   (auto-refreshed)
tqqq_yf.csv             leveraged ETF (auto-refreshed)
tqqq_extended_yf.csv    synthetic pre-2010 TQQQ, rebuilt each run
advisor_state.json      parameter history + position override   <- do not delete
signal_log.csv          one row per run
```

## Daily use

Run once per US trading day, **after the 16:00 ET close**:

```bash
python daily_advisor.py
```

It prints the market reading, your tracked position, and the next move —
**to be executed at the next session's open**.

Normal days take ~1 second. Once every ~252 trading days it re-optimizes
parameters on the trailing 8 years, which takes ~40 seconds.

## Reading the output

```
  MARKET  (QQQ)
    close ....................       716.76
    200-day SMA .............       654.68
    price vs SMA ............       +9.48%   ABOVE   (5 sessions ago: +8.30%  ->  +1.18 pts)
```
How far price sits from the 200-day SMA, and whether that gap widened or
narrowed over the last week.

```
  ACTIVE PARAMETERS   (walk-forward re-opt as of 2026-03-30)
    entry buffer ...........         +4%   -> enter trigger      680.87
    exit buffer ............         -2%   -> exit trigger       667.77
    DCA length .............      189 d
```
The entry/exit triggers are the SMA offset by the buffers. **The signal is a
CLOSE**, not an intraday touch.

```
  NEXT MOVE   (execute at the next session's open)
    HOLD TQQQ.
         Exit only if QQQ CLOSES below 667.77 (cushion +7.34% right now).
```

## The four states

| State | You hold | Leaves when |
|---|---|---|
| `CASH` | cash | QQQ closes above the entry trigger → all-in TQQQ |
| `FULL_LEV` | TQQQ | QQQ closes below the exit trigger → sell, begin DCA |
| `DCA` | cash draining into QQQ | DCA completes → `DELEVERAGED`, or QQQ closes above entry → all-in TQQQ |
| `DELEVERAGED` | QQQ | QQQ closes above the entry trigger → all-in TQQQ |

## Flags

```bash
python daily_advisor.py --update         # force a fresh download
python daily_advisor.py --no-update      # offline, cached data only
python daily_advisor.py --optimize-now   # force re-optimization now
python daily_advisor.py --history 20     # last 20 logged days
python daily_advisor.py --reset          # rebuild parameter history from scratch
python daily_advisor.py --set-state FULL_LEV
python daily_advisor.py --set-state DCA --dca-day 40
```

`--set-state` corrects the tracked position when reality drifts from the log
(you skipped a trade, filled partially, started mid-stream). Use it — the
recommendations are wrong if the tracked state is wrong.

Re-run with `--reset` after changing any constant at the top of the script
(`IN_SAMPLE_YEARS`, `MA_PERIOD`, tickers); the cached parameter history was
built on the old settings.

## Scheduling it (Windows)

Point Task Scheduler at `run_advisor.bat` (it locates a working Python itself),
or register the job directly — replace `<DIR>` with this folder's full path:

```bash
schtasks /create /tn "MA200 Advisor" /sc weekly /d MON,TUE,WED,THU,FRI /st 16:30 /tr "<DIR>\run_advisor.bat"
```

Adjust `/st` to ~30 minutes after your local equivalent of the 16:00 ET close.
The scheduled run is silent — read `signal_log.csv` afterward, or append
`>> advisor_output.txt 2>&1` inside the `.bat` to keep the printed report.

## How much can you skip?

The signal is evaluated on every daily close, so in principle run it daily.
In practice:

- **`FULL_LEV` with a wide cushion** — a missed day changes nothing, and the
  next run replays every bar since the last one.
- **Cushion under ~2-3%, or mid-`DCA`** — don't skip. A DCA day is a buy, and a
  trigger can be crossed on any single close.

## Requirements

```bash
pip install pandas numpy yfinance
```

## What this is not

Advice only — it places no orders. It assumes you execute each recommendation
in full at the next open, matching the backtest. Taxes, spreads, and slippage
are not modelled. "DCA ~X% of remaining cash" is a fraction because the tool
tracks position *state* and the DCA day counter, not your dollar balances.

**Plan for a ~-75% drawdown.** The backtest realized -57%, but that sits at only
the ~13th percentile of bootstrapped alternate histories — it was a lucky draw,
not a ceiling. This is a 3x leveraged strategy with ~48% annualized volatility.
