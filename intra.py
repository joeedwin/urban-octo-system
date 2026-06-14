"""
Intraday Bank Nifty options strategy (PE/CE).

Three things in one file:
  1. BACKTEST  - runs the full state machine over past 5-min index data,
                 MODELLING the option premium with Black-Scholes (approximate).
  2. LIVE PAPER - runs the same rules on the live market via Zerodha or Groww,
                 reading REAL option premiums, simulating fills (no real orders).
  3. Strike selection has two modes (override):
        STRIKE_MODE = "delta"  -> pick strike nearest TARGET_DELTA (default ~0.45)
        STRIKE_MODE = "otm"    -> flat OTM_POINTS away (your original 500-OTM rule)

Switch everything with env vars or the constants below.

Locked rules (spec + our decisions):
  - Skip first 5 five-min candles (no trade before 09:40).
  - PE: BN high >= prev-day high, red candle, osc < 0; mark candle low; enter PE
        when BN trades 30 pts below it; stop = day HIGH + 1.
  - CE: mirror (prev-day low, green candle, osc > 0; stop = day LOW - 1).
  - Oscillator: signed Widner; bar colour = sign.
  - Size: qty = floor(RISK / (delta * stop_distance)); lots = floor(qty/LOT).
  - T1: nearest classic floor pivot >= 100 pts away; sell ceil(lots/2); trail rest to breakeven.
  - T2: two candles against the option + matching osc bar -> exit rest.
  - Square off all at 15:20.

NOTE on the live broker calls: I can't test these without your credentials, and the
option-chain field names (esp. Groww) may differ from the SDK version you have. Every
such spot is marked  # VERIFY .  Real order placement is intentionally NOT implemented.
"""

import os
import sys
import json
import math
import time as _time
from dataclasses import dataclass, field
from datetime import time as dtime, datetime, timedelta

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(line_buffering=True)   # show prints immediately (no buffering)
except Exception:
    pass

print("[boot] strategy file loaded", flush=True)   # if you never see this, the file isn't being run

# ==========================================
# CREDENTIALS  -- fill these in ONLY in your private copy / Groww Cloud editor.
# Keep this file PRIVATE: never share it, commit it to git, screenshot it, or post it.
# On your own computer you can leave these blank and use environment variables instead.
# ==========================================

# Groww TOTP flow (your "API key" = TOTP token, plus the TOTP secret):
GROWW_TOTP_TOKEN="eyJraWQiOiJaTUtjVXciLCJhbGciOiJFUzI1NiJ9.eyJleHAiOjI1Njk3NjYzNDYsImlhdCI6MTc4MTM2NjM0NiwibmJmIjoxNzgxMzY2MzQ2LCJzdWIiOiJ7XCJ0b2tlblJlZklkXCI6XCI5MTVjNjliZS02NTliLTQ3NzQtYjliOC0xMmNmOTZlNWY4YjlcIixcInZlbmRvckludGVncmF0aW9uS2V5XCI6XCJlMzFmZjIzYjA4NmI0MDZjODg3NGIyZjZkODQ5NTMxM1wiLFwidXNlckFjY291bnRJZFwiOlwiMTg2NTEzZDMtNmY4ZC00NWJiLThkOGItMjA2YTZhODc5NDk5XCIsXCJkZXZpY2VJZFwiOlwiMTI4Y2YxYzMtMTY5OS01OWRjLTk2MDItYjhmNWE1YmIzM2UwXCIsXCJzZXNzaW9uSWRcIjpcIjU1NTM2Y2RhLThhNGUtNDU5OC04NzQxLWEyOTEwYjkyYWUxMlwiLFwiYWRkaXRpb25hbERhdGFcIjpcIno1NC9NZzltdjE2WXdmb0gvS0EwYkVSTlY3TE1vUHE1Nk9MeU8rdFZnU2xSTkczdTlLa2pWZDNoWjU1ZStNZERhWXBOVi9UOUxIRmtQejFFQisybTdRPT1cIixcInJvbGVcIjpcImF1dGgtdG90cFwiLFwic291cmNlSXBBZGRyZXNzXCI6XCI0OS4zNy4yMTguMjQ0LDEzNi4yMjYuMjQzLjE2LDE2Mi4xNTguNTUuMTAyLDM1LjI0MS4yMy4xMjNcIixcInR3b0ZhRXhwaXJ5VHNcIjoyNTY5NzY2MzQ2MzkyLFwidmVuZG9yTmFtZVwiOlwiZ3Jvd3dBcGlcIn0iLCJpc3MiOiJhcGV4LWF1dGgtcHJvZC1hcHAifQ.TH5gyWFl7VyBRNfpKghdFGUfST5hfyx0EEw1-wkf54aOAsRsSyQky6AsJCB61OI5uFY7wFNIYfTFhy-dHUI-3Q"
GROWW_TOTP_SECRET="RYZOPWY4BVDYTKTQPN2T6DZMO27Y2IXM"

# Zerodha (only if LIVE_BROKER = "zerodha"):
ZERODHA_API_KEY      = ""
ZERODHA_ACCESS_TOKEN = ""


# ==========================================
# WHAT TO RUN  (edit these, then just run the file -- no env vars, no command line)
# ==========================================

RUN_MODE = "realbt"             # "realbt" = real-data backtest | "live" = live paper | "off"
BACKTEST_START = "2025-01-01"   # backtest from this date (YYYY-MM-DD)
BACKTEST_END   = "2025-03-31"   # backtest to this date (YYYY-MM-DD)


# ==========================================
# CONFIG
# ==========================================

RISK_PER_TRADE = float(os.environ.get("RISK", 3000))
LOT_SIZE = int(os.environ.get("LOT_SIZE", 30))            # verify current NSE BNF lot
STRIKE_MODE = os.environ.get("STRIKE_MODE", "otm")        # "delta" | "otm"  <-- OVERRIDE
OTM_POINTS = int(os.environ.get("OTM_POINTS", 500))       # used by "otm" mode + delta fallback
TARGET_DELTA = float(os.environ.get("TARGET_DELTA", 0.45))
STRIKE_STEP = 100
T1_MIN_PTS = 100
ENTRY_TRIGGER_PTS = 30        # CE first-trade only: 1m must break setup high + 30
STOP_BUFFER = 1.0
SKIP_CANDLES = 9             # no setup before 09:40 (first 5 5-min candles)
SQUAREOFF = dtime(15, 20)

# --- YCloseBounce setup rules (match the stored procedures) ---
BOUNCE_BAND = 150            # trigger must be within this many pts of yesterday's (or today's running) level
TRIGGER_WINDOW_MIN = 10      # 1m trigger must appear within 10 min of the 5m setup candle
LUNCH_CE = (dtime(11, 30), dtime(12, 30))   # CE first-trade dead zone
LUNCH_PE = (dtime(11, 0), dtime(12, 30))    # PE / second-trade dead zone
PIVOT_OFFSET = 100           # SELL1 pivot must be at least this far from entry
ALLOW_MULTIPLE_TRADES = True # subsequent trades use today's running high/low (SecondTrade SPs)
ONE_TRADE_PER_DAY = not ALLOW_MULTIPLE_TRADES   # used by the older modelled backtest / live-paper paths
DEBUG_SKIPS = os.environ.get("DEBUG_SKIPS", "0") == "1"   # print near-miss setups that didn't fire
PAPER_BUNDLE = os.environ.get("PAPER_BUNDLE", "paper_bundle.json")  # live-paper -> dashboard
_SKIPS = []                # collected near-miss setups (for the dashboard bundle)
LAST_RUN = {}              # last backtest result, used by export_bundle()

def _skip(msg):
    _SKIPS.append(msg)
    if DEBUG_SKIPS:
        print(msg, flush=True)

# --- premium-based sizing guards (my recommendation) ---
MAX_LOTS = int(os.environ.get("MAX_LOTS", 10))            # hard cap on lots per trade
MIN_BUFFER_FRAC = float(os.environ.get("MIN_BUFFER_FRAC", 0.15))  # buffer floored at 15% of entry premium

# Black-Scholes assumptions (BACKTEST modelling only; live uses real premiums)
ASSUMED_IV = 0.18
RISK_FREE = 0.07
OSC_LAYERS = 10
OSC_LOOKBACK = 10

# Live
LIVE_BROKER = os.environ.get("LIVE_BROKER", "zerodha")    # "zerodha" | "groww"
UNDERLYING = "BANKNIFTY"
INDEX_KITE_SYMBOL = "NIFTY BANK"                          # Kite NSE index tradingsymbol
INDEX_KITE_EXCHANGE = "NSE"
INDEX_GROWW_SYMBOL = os.environ.get("INDEX_GROWW_SYMBOL", "NSE-BANKNIFTY")   # confirm ticker via Get Instruments CSV
LIVE_STATE_PATH = "intraday_paper_state.json"
LIVE_POLL_SECS = 60


# ==========================================
# COST MODEL
# ==========================================

@dataclass
class OptCost:
    slippage_pts: float = 0.5
    brokerage_per_order: float = 20.0
    txn_pct: float = 0.0003503
    stt_sell_pct: float = 0.000625
    sebi_pct: float = 0.000001
    stamp_buy_pct: float = 0.00003
    gst_pct: float = 0.18

    def charge(self, side, premium, qty):
        turnover = premium * qty
        broker = self.brokerage_per_order
        txn = self.txn_pct * turnover
        sebi = self.sebi_pct * turnover
        gst = self.gst_pct * (broker + txn)
        stt = self.stt_sell_pct * turnover if side == "SELL" else 0.0
        stamp = self.stamp_buy_pct * turnover if side == "BUY" else 0.0
        return broker + txn + sebi + gst + stt + stamp

    def fill(self, side, premium):
        return premium + self.slippage_pts if side == "BUY" else max(0.05, premium - self.slippage_pts)


# ==========================================
# INDICATORS / MATH
# ==========================================

def widner_oscillator(close, layers=OSC_LAYERS, period=2, lookback=OSC_LOOKBACK):
    """Rainbow Oscillator — matches the user's reference script exactly.

    Ribbon: r1 = Close, r_i = period-SMA of r_{i-1} for i = 2..layers
    (so `layers` lines total, INCLUDING raw Close as r1).
    hist = 100 * (Close - mean(ribbon)) / (max(ribbon) - min(ribbon)),
    where max/min/mean are taken ACROSS the ribbon lines at each bar.
    Sign drives the bar colour: > 0 green (bullish), < 0 red (bearish).
    (`lookback` governs only the over/under display bands, unused here.)
    """
    close = pd.Series(close).astype(float)
    cols, r = [close], close                 # r1 = Close
    for _ in range(layers - 1):              # r2..r_layers = recursive smoothings
        r = r.rolling(period).mean()
        cols.append(r)
    ribbon = pd.concat(cols, axis=1)
    avAv = ribbon.mean(axis=1)
    rng = (ribbon.max(axis=1) - ribbon.min(axis=1)).replace(0, np.nan)
    return (100 * (close - avAv) / rng).fillna(0.0)


def floor_pivots(h, l, c):
    p = (h + l + c) / 3
    return {"P": p, "R1": 2 * p - l, "S1": 2 * p - h,
            "R2": p + (h - l), "S2": p - (h - l),
            "R3": h + 2 * (p - l), "S3": l - 2 * (h - p)}


def pick_t1(entry_spot, opt, pivots):
    levels = list(pivots.values())
    if opt == "PE":
        below = [v for v in levels if v <= entry_spot - T1_MIN_PTS]
        return max(below) if below else entry_spot - T1_MIN_PTS
    above = [v for v in levels if v >= entry_spot + T1_MIN_PTS]
    return min(above) if above else entry_spot + T1_MIN_PTS


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(S, K, T, r, sigma, opt):
    """Return (premium, abs_delta)."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(0.0, S - K) if opt == "CE" else max(0.0, K - S)
        d = 1.0 if (opt == "CE" and S > K) or (opt == "PE" and S < K) else 0.0
        return intrinsic, d
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt == "CE":
        price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        delta = _norm_cdf(d1)
    else:
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1
    return price, abs(delta)


def implied_vol(premium, S, K, T, r, opt):
    """Back-solve IV from a real premium (bisection). Used live to get delta."""
    if premium <= 0 or T <= 0:
        return ASSUMED_IV
    lo, hi = 1e-3, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        price, _ = bs(S, K, T, r, mid, opt)
        if price > premium:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def delta_from_premium(premium, S, K, T, r, opt):
    iv = implied_vol(premium, S, K, T, r, opt)
    _, d = bs(S, K, T, r, iv, opt)
    return d, iv


def select_strike(spot, opt, T, iv):
    """STRIKE_MODE: 'otm' = flat OTM_POINTS away; 'delta' = nearest TARGET_DELTA."""
    if STRIKE_MODE == "otm":
        k = spot - OTM_POINTS if opt == "PE" else spot + OTM_POINTS
        K = round(k / STRIKE_STEP) * STRIKE_STEP
        _, d = bs(spot, K, T, RISK_FREE, iv, opt)
        return K, d

    base = round(spot / STRIKE_STEP) * STRIKE_STEP
    best, best_err = None, 1e9
    for step in range(0, 40):
        k = base - step * STRIKE_STEP if opt == "PE" else base + step * STRIKE_STEP
        if (opt == "PE" and k >= spot) or (opt == "CE" and k <= spot):
            continue
        _, d = bs(spot, k, T, RISK_FREE, iv, opt)
        if abs(d - TARGET_DELTA) < best_err:
            best, best_err = k, abs(d - TARGET_DELTA)
        if d < TARGET_DELTA - 0.15:
            break
    if best is None:
        k = spot - OTM_POINTS if opt == "PE" else spot + OTM_POINTS
        best = round(k / STRIKE_STEP) * STRIKE_STEP
    _, d = bs(spot, best, T, RISK_FREE, iv, opt)
    return best, d


def size_position(delta, stop_distance_pts):
    per_qty_risk = delta * stop_distance_pts
    if per_qty_risk <= 0:
        return 0, 0
    qty = math.floor(RISK_PER_TRADE / per_qty_risk)
    lots = qty // LOT_SIZE
    return lots, lots * LOT_SIZE


def days_to_monthly_expiry(ts):
    import calendar
    y, m = ts.year, ts.month
    last = calendar.monthrange(y, m)[1]
    d = pd.Timestamp(y, m, last)
    while d.weekday() != 1:                      # Tuesday; VERIFY current BNF expiry weekday
        d -= pd.Timedelta(days=1)
    return max(1, (d.normalize() - ts.normalize()).days)


# ==========================================
# BACKTEST ENGINE (BS-modelled option leg)  -- unchanged logic
# ==========================================

@dataclass
class Trade:
    date: str
    opt: str
    strike: float
    lots: int
    entry_spot: float
    entry_prem: float
    legs: list = field(default_factory=list)
    entry_time: str = ""
    stop: float = 0.0
    t1: float = 0.0
    reason: str = ""
    mfe: float = 0.0
    mae: float = 0.0

    @property
    def pnl(self):
        return sum(l[3] for l in self.legs)


def backtest(df: pd.DataFrame, cost: OptCost = None, iv: float = ASSUMED_IV):
    cost = cost or OptCost()
    df = df.copy()
    df["osc"] = widner_oscillator(df["Close"])
    df["date"] = df.index.normalize()
    sessions = list(df.groupby("date"))
    trades = []

    for di in range(1, len(sessions)):
        date, day = sessions[di]
        _, prev = sessions[di - 1]
        PDH, PDL, PDC = prev["High"].max(), prev["Low"].min(), prev["Close"].iloc[-1]
        piv = floor_pivots(PDH, PDL, PDC)
        T = days_to_monthly_expiry(date) / 365.0
        state, armed, ref = "WAIT", None, None
        pos = None
        day_hi, day_lo = -1e18, 1e18
        done = False
        bars = list(day.iterrows())

        for i, (ts, row) in enumerate(bars):
            day_hi = max(day_hi, row["High"]); day_lo = min(day_lo, row["Low"])
            green = row["Close"] > row["Open"]; red = row["Close"] < row["Open"]
            osc = row["osc"]
            if i < SKIP_CANDLES:
                continue

            if pos is not None:
                opt, K = pos["opt"], pos["strike"]
                cur_prem, _ = bs(row["Close"], K, T, RISK_FREE, iv, opt)

                def exit_lots(n_lots, ref_spot, reason):
                    if n_lots <= 0:
                        return
                    prem, _ = bs(ref_spot, K, T, RISK_FREE, iv, opt)
                    fillp = cost.fill("SELL", prem)
                    q = n_lots * LOT_SIZE
                    fee = cost.charge("SELL", fillp, q)
                    pnl = (fillp - pos["entry_fill"]) * q - fee
                    pos["trade"].legs.append((reason, n_lots, round(fillp, 2), round(pnl, 2)))
                    pos["lots"] -= n_lots

                hit_stop = (not pos["half"]) and (
                    (opt == "PE" and row["High"] >= pos["stop_spot"]) or
                    (opt == "CE" and row["Low"] <= pos["stop_spot"]))
                t1_touch = (not pos["half"]) and (
                    (opt == "PE" and row["Low"] <= pos["t1"]) or
                    (opt == "CE" and row["High"] >= pos["t1"]))

                if ts.time() >= SQUAREOFF:
                    exit_lots(pos["lots"], row["Close"], "TIME"); pos = None; done = True
                elif hit_stop:
                    exit_lots(pos["lots"], pos["stop_spot"], "STOP"); pos = None; done = True
                elif t1_touch:
                    exit_lots(math.ceil(pos["lots"] / 2), pos["t1"], "T1")
                    pos["half"] = True
                    if pos["lots"] <= 0:
                        pos = None; done = True
                else:
                    if pos["half"]:
                        if cur_prem <= pos["entry_fill"]:
                            exit_lots(pos["lots"], row["Close"], "BE_STOP"); pos = None; done = True
                        else:
                            against = green if opt == "PE" else red
                            osc_ok = (osc > 0) if opt == "PE" else (osc < 0)
                            if against and pos["prev_against"] and osc_ok:
                                exit_lots(pos["lots"], row["Close"], "T2"); pos = None; done = True
                    if pos is not None:
                        pos["prev_against"] = green if opt == "PE" else red
                if pos is None:
                    continue

            if pos is None and not (ONE_TRADE_PER_DAY and done):
                if state != "ARMED":
                    if row["High"] >= PDH and red and osc < 0:
                        state, armed, ref = "ARMED", "PE", row["Low"]
                    elif row["Low"] <= PDL and green and osc > 0:
                        state, armed, ref = "ARMED", "CE", row["High"]
                if state == "ARMED":
                    trig = ((armed == "PE" and row["Low"] <= ref - ENTRY_TRIGGER_PTS) or
                            (armed == "CE" and row["High"] >= ref + ENTRY_TRIGGER_PTS))
                    if trig:
                        entry_spot = (ref - ENTRY_TRIGGER_PTS) if armed == "PE" else (ref + ENTRY_TRIGGER_PTS)
                        stop_spot = (day_hi + STOP_BUFFER) if armed == "PE" else (day_lo - STOP_BUFFER)
                        K, delta = select_strike(entry_spot, armed, T, iv)
                        lots, qty = size_position(delta, abs(entry_spot - stop_spot))
                        if lots >= 1:
                            prem, _ = bs(entry_spot, K, T, RISK_FREE, iv, armed)
                            fillp = cost.fill("BUY", prem)
                            fee = cost.charge("BUY", fillp, qty)
                            tr = Trade(str(date.date()), armed, K, lots, round(entry_spot, 1), round(fillp, 2))
                            tr.legs.append(("ENTRY_FEE", 0, round(fillp, 2), -round(fee, 2)))
                            pos = {"opt": armed, "strike": K, "lots": lots, "entry_fill": fillp,
                                   "stop_spot": stop_spot, "t1": pick_t1(entry_spot, armed, piv),
                                   "half": False, "prev_against": False, "trade": tr}
                            trades.append(tr)
                        state = "WAIT"

        if pos is not None:
            prem, _ = bs(bars[-1][1]["Close"], pos["strike"], T, RISK_FREE, iv, pos["opt"])
            fillp = cost.fill("SELL", prem); q = pos["lots"] * LOT_SIZE
            fee = cost.charge("SELL", fillp, q)
            pos["trade"].legs.append(("EOD", pos["lots"], round(fillp, 2),
                                      round((fillp - pos["entry_fill"]) * q - fee, 2)))
    return trades


def summarise(trades):
    if not trades:
        print("No trades."); return

    def leg_time(l):  return l[4] if len(l) > 4 else ""
    def leg_spot(l):  return l[5] if len(l) > 5 else None
    def leg_detail(l): return l[6] if len(l) > 6 else ""

    print(f"\n{'='*72}")
    print(f"TRADE-BY-TRADE  (mode={STRIKE_MODE}, underlying={UNDERLYING}, lot={LOT_SIZE})")
    print(f"{'='*72}")

    reason_counts = {}
    gross_win = gross_loss = 0.0
    wins = losses = 0

    for n, t in enumerate(trades, 1):
        et = t.entry_time.split()[-1][:8] if t.entry_time else "?"
        print(f"\n#{n}  {t.date}  {t.opt}  strike {t.strike:.0f}   {t.lots} lot(s)")
        if t.reason:
            print(f"    why   : {t.reason}")
        print(f"    entry : {et}  spot {t.entry_spot:.0f}  premium {t.entry_prem:.1f}"
              f"   stop {t.stop:.0f}   T1 {t.t1:.0f}")
        for l in t.legs:
            reason, nl, prem, pnl = l[0], l[1], l[2], l[3]
            if reason == "ENTRY_FEE":
                print(f"    fees  : entry charges {pnl:>9.0f}")
                continue
            tm = (leg_time(l).split()[-1][:8] if leg_time(l) else "")
            sp = leg_spot(l)
            sp_s = f"spot {sp:.0f}  " if sp is not None else ""
            det = leg_detail(l)
            det_s = f"   <- {det}" if det else ""
            print(f"    {reason:<7}: {tm:<9} {sp_s}{nl} lot(s) @ {prem:.1f}   P&L {pnl:>9.0f}{det_s}")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if t.mfe or t.mae:
            print(f"    excur : best +{t.mfe:.1f} / worst {t.mae:.1f} premium pts while held")
        mark = "WIN " if t.pnl > 0 else ("LOSS" if t.pnl < 0 else "FLAT")
        print(f"    ----> NET {t.pnl:>9.0f}   [{mark}]")
        if t.pnl > 0:
            wins += 1; gross_win += t.pnl
        elif t.pnl < 0:
            losses += 1; gross_loss += t.pnl

    total = sum(t.pnl for t in trades)
    pf = (gross_win / abs(gross_loss)) if gross_loss < 0 else float("inf")
    best = max(trades, key=lambda x: x.pnl).pnl
    worst = min(trades, key=lambda x: x.pnl).pnl
    avg_w = gross_win / wins if wins else 0.0
    avg_l = gross_loss / losses if losses else 0.0

    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"  Trades        : {len(trades)}   (wins {wins} / losses {losses} / "
          f"flat {len(trades)-wins-losses})")
    print(f"  Win rate      : {wins/len(trades)*100:.1f}%")
    print(f"  Net P&L       : {total:,.0f}")
    print(f"  Gross win/loss: {gross_win:,.0f} / {gross_loss:,.0f}")
    print(f"  Profit factor : {pf:.2f}" if pf != float('inf') else "  Profit factor : inf (no losses)")
    print(f"  Avg win/loss  : {avg_w:,.0f} / {avg_l:,.0f}")
    print(f"  Best / worst  : {best:,.0f} / {worst:,.0f}")
    rc = "  ".join(f"{k}:{v}" for k, v in sorted(reason_counts.items()))
    print(f"  Exit reasons  : {rc}")
    print(f"{'='*72}\n")


# ==========================================
# OPTION BROKERS (live)  -- Zerodha fully wired; Groww best-effort, VERIFY marked
# ==========================================

class OptionBroker:
    def index_5min(self, days=3): raise NotImplementedError
    def index_1min(self, days=2): raise NotImplementedError          # for the YCloseBounce 1m trigger
    def option_1min(self, symbol, days=1): raise NotImplementedError  # for the pre-entry premium low
    def monthly_expiry(self): raise NotImplementedError
    def option_symbol(self, strike, opt, expiry): raise NotImplementedError
    def option_ltp(self, symbol): raise NotImplementedError
    def place_order(self, *a, **k):
        raise NotImplementedError("Real order placement is disabled (paper mode).")


class ZerodhaOptionBroker(OptionBroker):
    def __init__(self):
        from kiteconnect import KiteConnect
        key = ZERODHA_API_KEY or os.environ.get("ZERODHA_API_KEY")
        tok = ZERODHA_ACCESS_TOKEN or os.environ.get("ZERODHA_ACCESS_TOKEN")
        if not key or not tok:
            raise SystemExit("Fill ZERODHA_API_KEY + ZERODHA_ACCESS_TOKEN at the top of the file, "
                             "or set them as environment variables.")
        self.kite = KiteConnect(api_key=key); self.kite.set_access_token(tok)
        self._nse = None; self._nfo = None; self._idx_token = None

    def _nfo_dump(self):
        if self._nfo is None:
            self._nfo = self.kite.instruments("NFO")
        return self._nfo

    def _index_token(self):
        if self._idx_token is None:
            for ins in self.kite.instruments(INDEX_KITE_EXCHANGE):
                if ins["tradingsymbol"] == INDEX_KITE_SYMBOL:
                    self._idx_token = ins["instrument_token"]; break
        if self._idx_token is None:
            raise LookupError(f"{INDEX_KITE_SYMBOL} not found")
        return self._idx_token

    def index_5min(self, days=3):
        end = datetime.now(); start = end - timedelta(days=days)
        data = self.kite.historical_data(self._index_token(), start, end, "5minute")
        df = pd.DataFrame(data).rename(columns={"date": "Date", "open": "Open", "high": "High",
                                                "low": "Low", "close": "Close"})
        return df.set_index(pd.to_datetime(df["Date"]))[["Open", "High", "Low", "Close"]]

    def index_1min(self, days=2):
        end = datetime.now(); start = end - timedelta(days=days)
        data = self.kite.historical_data(self._index_token(), start, end, "minute")
        df = pd.DataFrame(data).rename(columns={"date": "Date", "open": "Open", "high": "High",
                                                "low": "Low", "close": "Close"})
        return df.set_index(pd.to_datetime(df["Date"]))[["Open", "High", "Low", "Close"]]

    def _option_token(self, symbol):
        for i in self._nfo_dump():
            if i["tradingsymbol"] == symbol:
                return i["instrument_token"]
        raise LookupError(f"option token not found for {symbol}")

    def option_1min(self, symbol, days=1):
        end = datetime.now(); start = end - timedelta(days=days)
        data = self.kite.historical_data(self._option_token(symbol), start, end, "minute")
        df = pd.DataFrame(data).rename(columns={"date": "Date", "open": "Open", "high": "High",
                                                "low": "Low", "close": "Close"})
        return df.set_index(pd.to_datetime(df["Date"]))[["Open", "High", "Low", "Close"]]

    def monthly_expiry(self):
        today = pd.Timestamp.now().normalize().date()
        exps = sorted({i["expiry"] for i in self._nfo_dump()
                       if i["name"] == UNDERLYING and i["expiry"] >= today})
        if not exps:
            raise LookupError("No BANKNIFTY expiries found")
        return exps[0]                                    # nearest (monthly, since no weeklies)

    def option_symbol(self, strike, opt, expiry):
        for i in self._nfo_dump():
            if (i["name"] == UNDERLYING and i["instrument_type"] == opt
                    and int(i["strike"]) == int(strike) and i["expiry"] == expiry):
                return i["tradingsymbol"]
        raise LookupError(f"No {UNDERLYING} {strike}{opt} {expiry}")

    def option_ltp(self, symbol):
        key = f"NFO:{symbol}"
        return self.kite.ltp(key)[key]["last_price"]


class GrowwOptionBroker(OptionBroker):
    """Best-effort. Option-chain field names vary by SDK version -- see # VERIFY."""
    def __init__(self):
        from growwapi import GrowwAPI
        key = GROWW_TOTP_TOKEN or os.environ.get("GROWW_TOTP_TOKEN")       # the TOTP token
        secret = GROWW_TOTP_SECRET or os.environ.get("GROWW_TOTP_SECRET")  # the TOTP secret
        tok = GROWW_API_TOKEN or os.environ.get("GROWW_API_TOKEN")         # OR a daily access token
        if key and secret:
            import pyotp                                # pip install pyotp
            totp = pyotp.TOTP(secret).now()
            tok = GrowwAPI.get_access_token(api_key=key, totp=totp)
        if not tok:
            raise SystemExit("Fill GROWW_TOTP_TOKEN + GROWW_TOTP_SECRET at the top of the file "
                             "(or GROWW_API_TOKEN), or set them as environment variables.")
        self.g = GrowwAPI(tok)

    def index_5min(self, days=3):
        end = datetime.now(); start = end - timedelta(days=days)
        resp = self.g.get_historical_candles(                      # VERIFY interval constant name
            exchange=self.g.EXCHANGE_NSE, segment=self.g.SEGMENT_CASH,
            groww_symbol=INDEX_GROWW_SYMBOL,
            start_time=start.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
            candle_interval=getattr(self.g, "CANDLE_INTERVAL_MIN_5"))
        rows = resp.get("candles", []) if isinstance(resp, dict) else (resp or [])
        names = ["ts", "Open", "High", "Low", "Close", "Volume", "OI"][:len(rows[0])]
        df = pd.DataFrame(rows, columns=names)
        ts = df["ts"]
        if pd.api.types.is_numeric_dtype(ts):
            df.index = pd.to_datetime(ts, unit="ms" if float(ts.iloc[0]) > 1e12 else "s")
        else:
            df.index = pd.to_datetime(ts)
        return df[["Open", "High", "Low", "Close"]].astype(float).sort_index()

    def _candles(self, segment, symbol, days, interval_const):
        end = datetime.now(); start = end - timedelta(days=days)
        resp = self.g.get_historical_candles(
            exchange=self.g.EXCHANGE_NSE, segment=segment, groww_symbol=symbol,
            start_time=start.strftime("%Y-%m-%d %H:%M:%S"), end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
            candle_interval=getattr(self.g, interval_const))
        rows = resp.get("candles", []) if isinstance(resp, dict) else (resp or [])
        if not rows:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
        names = ["ts", "Open", "High", "Low", "Close", "Volume", "OI"][:len(rows[0])]
        df = pd.DataFrame(rows, columns=names); ts = df["ts"]
        if pd.api.types.is_numeric_dtype(ts):
            df.index = pd.to_datetime(ts, unit="ms" if float(ts.iloc[0]) > 1e12 else "s")
        else:
            df.index = pd.to_datetime(ts)
        return df[["Open", "High", "Low", "Close"]].astype(float).sort_index()

    def index_1min(self, days=2):
        return self._candles(self.g.SEGMENT_CASH, INDEX_GROWW_SYMBOL, days, "CANDLE_INTERVAL_MIN_1")

    def option_1min(self, symbol, days=1):
        return self._candles(self.g.SEGMENT_FNO, symbol, days, "CANDLE_INTERVAL_MIN_1")

    def monthly_expiry(self):
        now = pd.Timestamp.now()

        def fetch(y, mo):
            resp = self.g.get_expiries(exchange=self.g.EXCHANGE_NSE,
                                       underlying_symbol=UNDERLYING, year=y, month=mo)
            return resp["expiries"] if isinstance(resp, dict) and "expiries" in resp else resp

        cand = []
        for off in (0, 1):                       # this month, then next if this month's expiry passed
            t = now + pd.DateOffset(months=off)
            for e in fetch(t.year, t.month):
                try:
                    dt = pd.to_datetime(e, format="%d%b%y")
                except Exception:
                    dt = pd.to_datetime(e)
                cand.append(dt.normalize())
            future = [d for d in cand if d >= now.normalize()]
            if future:
                return min(future)
        return min(cand)

    def option_symbol(self, strike, opt, expiry):
        # Per Groww docs: groww_symbol = NSE-BANKNIFTY-DDMmmYY-STRIKE-CE/PE
        exp = pd.Timestamp(expiry).strftime("%d%b%y")
        return f"NSE-{UNDERLYING}-{exp}-{int(strike)}-{opt}"

    def option_ltp(self, symbol):
        resp = self.g.get_ltp(segment=self.g.SEGMENT_FNO,                       # VERIFY segment/shape
                              exchange_trading_symbols=symbol)
        if isinstance(resp, dict):
            v = next(iter(resp.values()))
            return v.get("ltp", v.get("last_price", v)) if isinstance(v, dict) else v
        return resp


def build_option_broker():
    return ZerodhaOptionBroker() if LIVE_BROKER == "zerodha" else GrowwOptionBroker()


# ==========================================
# LIVE PAPER RUNNER (no real orders)
# ==========================================

def _log(m): print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def _load_state():
    if os.path.exists(LIVE_STATE_PATH):
        with open(LIVE_STATE_PATH) as f:
            return json.load(f)
    return {"day": None, "phase": "WAIT", "armed": None, "ref": None,
            "pos": None, "day_hi": -1e18, "day_lo": 1e18, "last_ts": None,
            "prev_against": False, "done": False, "trades": []}


def _save_state(s):
    with open(LIVE_STATE_PATH, "w") as f:
        json.dump(s, f, indent=2, default=str)


def _paper_bundle(s, candles5):
    """Convert live-paper state into the same bundle schema the dashboard reads,
    plus an open-position summary and realized P&L."""
    candles = []
    if candles5 is not None and len(candles5):
        cc = candles5.reset_index(); tcol = cc.columns[0]
        for _, r in cc.iterrows():
            candles.append({"dt": str(r[tcol]), "Open": float(r["Open"]), "High": float(r["High"]),
                            "Low": float(r["Low"]), "Close": float(r["Close"]), "osc": float(r.get("osc", 0.0))})
    trades = []; cur = None                                          # group fills into trades
    for f in s.get("trades", []):
        if f["side"] == "BUY":
            cur = {"date": s.get("day", ""), "opt": f["opt"], "strike": f["strike"], "lots": f["lots"],
                   "entry_spot": f.get("spot"), "entry_prem": f["prem"], "entry_time": f["ts"],
                   "stop": f.get("stop", 0), "t1": f.get("t1", 0), "reason": "live paper",
                   "mfe": 0, "mae": 0,
                   "legs": [{"reason": "ENTRY_FEE", "lots": 0, "prem": f["prem"], "pnl": f["pnl"],
                             "time": f["ts"], "spot": f.get("spot"), "detail": "buy"}]}
            trades.append(cur)
        elif cur is not None:
            cur["legs"].append({"reason": f["side"], "lots": f["lots"], "prem": f["prem"], "pnl": f["pnl"],
                                "time": f["ts"], "spot": f.get("spot"), "detail": ""})
    for t in trades:
        t["pnl"] = round(sum(l["pnl"] for l in t["legs"]), 2)
    return {"underlying": UNDERLYING, "lot_size": LOT_SIZE, "strike_mode": "live-paper",
            "trades": trades, "skips": [], "candles": candles,
            "open_position": s.get("pos"), "realized_pnl": round(sum(t["pnl"] for t in trades), 2),
            "updated": str(pd.Timestamp.now())}


def _write_paper_bundle(s, candles5, path=None):
    import json
    try:
        with open(path or PAPER_BUNDLE, "w") as f:
            json.dump(_paper_bundle(s, candles5), f)
    except Exception as e:
        _log(f"paper-bundle write failed: {e}")


def run_live_paper():
    """Live PAPER trading on the SAME YCloseBounce brain as the backtest.
    Only the source (live broker polls) and destination (paper log) differ from the
    backtest; every decision routes through the shared _setup_side / _find_trigger /
    _bounce_ok / _choose_strike / _size_premium / _exit_decision functions.
    No real orders are ever placed."""
    cost = OptCost(); broker = build_option_broker()
    _log(f"PAPER intraday {UNDERLYING} on {LIVE_BROKER} (YCloseBounce, shared engine). No real orders.")
    s = _load_state()

    while True:
        try:
            idx5 = broker.index_5min(days=5)
            idx5 = idx5[~idx5.index.duplicated(keep="last")].sort_index()
            idx5["osc"] = widner_oscillator(idx5["Close"])
            idx1 = broker.index_1min(days=2)
            idx1 = idx1[~idx1.index.duplicated(keep="last")].sort_index()

            by5 = list(idx5.groupby(idx5.index.normalize()))
            if len(by5) < 2:
                _time.sleep(LIVE_POLL_SECS); continue
            today, day5 = by5[-1]; _, prev5 = by5[-2]
            day1 = idx1[idx1.index.normalize() == today]

            if s.get("day") != str(today.date()):                 # new session -> reset
                s = {"day": str(today.date()), "pos": None, "trade_no": 0,
                     "armed": None, "last_t2_5m": None, "trades": s.get("trades", [])}

            PDH, PDL, PDC = prev5["High"].max(), prev5["Low"].min(), prev5["Close"].iloc[-1]
            piv = floor_pivots(PDH, PDL, PDC)
            expiry = broker.monthly_expiry()

            completed = day5.iloc[:-1] if len(day5) > 1 else day5  # drop the still-forming candle
            if completed.empty:
                _save_state(s); _time.sleep(LIVE_POLL_SECS); continue
            run_lo = float(completed["Low"].min()); run_hi = float(completed["High"].max())
            spot = float(day1["Close"].iloc[-1]) if len(day1) else float(completed["Close"].iloc[-1])
            now_t = pd.Timestamp.now().time()
            pos = s.get("pos")

            # ---------- manage an open paper position ----------
            if pos:
                prem = broker.option_ltp(pos["symbol"])           # REAL live premium
                last2 = None
                last5_ts = str(completed.index[-1])
                if pos["half"] and s.get("last_t2_5m") != last5_ts and len(completed) >= 2:
                    last2 = (completed.iloc[-2], completed.iloc[-1]); s["last_t2_5m"] = last5_ts
                action, n_lots, detail = _exit_decision(pos["opt"], pos, pos["t1"], now_t, spot, prem, last2)
                if action:
                    fillp = cost.fill("SELL", prem); q = n_lots * LOT_SIZE
                    fee = cost.charge("SELL", fillp, q)
                    pnl = (fillp - pos["entry_fill"]) * q - fee
                    s["trades"].append({"ts": str(pd.Timestamp.now()), "opt": pos["opt"],
                                        "strike": pos["strike"], "side": action, "lots": n_lots,
                                        "prem": round(fillp, 2), "pnl": round(pnl, 2),
                                        "spot": round(float(spot), 1)})
                    _log(f"SELL {n_lots} ({action}) {pos['opt']} {pos['strike']} @ {fillp:.1f}  "
                         f"pnl {pnl:.0f}  <- {detail}")
                    pos["lots"] -= n_lots
                    if action == "T1":
                        pos["half"] = True; pos["stop_prem"] = pos["entry_fill"]
                    if pos["lots"] <= 0 or action in ("TIME", "STOP", "BE_STOP", "T2"):
                        s["pos"] = None; s["trade_no"] = s.get("trade_no", 0) + 1
                    else:
                        s["pos"] = pos
                else:
                    s["pos"] = pos

            # ---------- look for a new entry ----------
            elif (ALLOW_MULTIPLE_TRADES or s.get("trade_no", 0) == 0) and now_t < SQUAREOFF:
                first = (s.get("trade_no", 0) == 0)
                setup_ts = completed.index[-1]; srow = completed.iloc[-1]
                side = _setup_side(setup_ts, srow["Open"], srow["Close"], srow["osc"], first)
                # arm on the freshest qualifying setup candle
                if side and (s.get("armed") is None or s["armed"]["ts"] != str(setup_ts)):
                    s["armed"] = {"ts": str(setup_ts), "hi": float(srow["High"]),
                                  "lo": float(srow["Low"]), "side": side, "first": first}
                armed = s.get("armed")
                if armed:
                    a_ts = pd.Timestamp(armed["ts"])
                    if now_t > (a_ts + pd.Timedelta(minutes=TRIGGER_WINDOW_MIN)).time():
                        s["armed"] = None                          # window expired, give up on it
                    else:
                        trig_ts, entry_spot = _find_trigger(day1, a_ts, armed["hi"], armed["lo"],
                                                            armed["side"], armed["first"])
                        level_lo = PDL if armed["first"] else run_lo
                        level_hi = PDH if armed["first"] else run_hi
                        if trig_ts is not None and _bounce_ok(entry_spot, level_lo, level_hi, armed["side"]):
                            sd = armed["side"]; K = _choose_strike(entry_spot, sd)
                            sym = broker.option_symbol(K, sd, expiry)
                            entry_prem = broker.option_ltp(sym)
                            try:
                                opt1 = broker.option_1min(sym, days=1)
                                pre = opt1[opt1.index <= pd.Timestamp.now()]
                                premium_stop = float(pre["Low"].min()) if len(pre) else entry_prem * (1 - MIN_BUFFER_FRAC)
                            except Exception:
                                premium_stop = entry_prem * (1 - MIN_BUFFER_FRAC)
                            lots, buf, floored, capped = _size_premium(entry_prem, premium_stop)
                            if lots >= 1:
                                fillp = cost.fill("BUY", entry_prem)
                                fee = cost.charge("BUY", fillp, lots * LOT_SIZE)
                                t1 = pick_t1(entry_spot, sd, piv)
                                s["trades"].append({"ts": str(pd.Timestamp.now()), "opt": sd, "strike": K,
                                                    "side": "BUY", "lots": lots, "prem": round(fillp, 2),
                                                    "pnl": -round(fee, 2), "spot": round(float(entry_spot), 1),
                                                    "stop": round(premium_stop, 1), "t1": round(t1, 1)})
                                s["pos"] = {"opt": sd, "symbol": sym, "strike": K, "lots": lots,
                                            "entry_fill": fillp, "stop_prem": premium_stop, "t1": t1,
                                            "half": False}
                                _log(f"BUY {lots} {sd} {K} @ {fillp:.1f}  stop {premium_stop:.1f}  "
                                     f"t1 {t1:.0f}  (trade #{s.get('trade_no',0)+1}{' CAPPED' if capped else ''})")
                                s["armed"] = None

            _save_state(s)
            _write_paper_bundle(s, day5)                            # feed the dashboard
        except Exception as e:                                     # keep the loop alive
            _log(f"error: {e}")
        _time.sleep(LIVE_POLL_SECS)


# ==========================================
# REAL-DATA BACKTEST via Groww historical FNO  (run locally; uses real option prices)
# Reads data only -- never places an order. Use your NEW (regenerated) keys.
# ==========================================

def _groww_login():
    from growwapi import GrowwAPI
    key = GROWW_TOTP_TOKEN or os.environ.get("GROWW_TOTP_TOKEN")
    secret = GROWW_TOTP_SECRET or os.environ.get("GROWW_TOTP_SECRET")
    tok = GROWW_API_TOKEN or os.environ.get("GROWW_API_TOKEN")
    if key and secret:
        import pyotp
        tok = GrowwAPI.get_access_token(api_key=key, totp=pyotp.TOTP(secret).now())
    if not tok:
        raise SystemExit("Fill GROWW_TOTP_TOKEN + GROWW_TOTP_SECRET (or GROWW_API_TOKEN).")
    return GrowwAPI(tok)


GROWW_5MIN_MAX_DAYS = 28   # Groww caps 5-min history at ~30 days per request
GROWW_1MIN_MAX_DAYS = 5    # 1-min history caps tighter; chunk small (adjust if Groww errors)
GROWW_MIN_GAP = float(os.environ.get("GROWW_MIN_GAP", 0.5))   # min seconds between API calls
GROWW_MAX_RETRIES = 6
_LAST_API_CALL = [0.0]


def _throttled(fn):
    """Space out Groww API calls and retry with backoff when rate-limited."""
    gap = _time.time() - _LAST_API_CALL[0]
    if gap < GROWW_MIN_GAP:
        _time.sleep(GROWW_MIN_GAP - gap)
    for attempt in range(GROWW_MAX_RETRIES):
        try:
            r = fn()
            _LAST_API_CALL[0] = _time.time()
            return r
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < GROWW_MAX_RETRIES - 1:
                wait = min(2 ** attempt, 30)
                print(f"[backtest] rate limited; waiting {wait}s then retrying...", flush=True)
                _time.sleep(wait)
                continue
            raise
    _LAST_API_CALL[0] = _time.time()


def _g_candles(g, symbol, segment, start, end, interval="5m"):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if interval == "1m":
        const, max_days = "CANDLE_INTERVAL_MIN_1", GROWW_1MIN_MAX_DAYS
    else:
        const, max_days = "CANDLE_INTERVAL_MIN_5", GROWW_5MIN_MAX_DAYS
    frames = []
    cur = start
    while cur <= end:                                   # fetch in limited-size windows
        chunk_end = min(cur + pd.Timedelta(days=max_days), end)
        resp = _throttled(lambda c=cur, ce=chunk_end: g.get_historical_candles(
            exchange=g.EXCHANGE_NSE, segment=segment, groww_symbol=symbol,
            start_time=c.strftime("%Y-%m-%d 00:00:00"),
            end_time=ce.strftime("%Y-%m-%d 23:59:59"),
            candle_interval=getattr(g, const)))
        rows = resp.get("candles", []) if isinstance(resp, dict) else (resp or [])
        if rows:
            names = ["ts", "Open", "High", "Low", "Close", "Volume", "OI"][:len(rows[0])]
            frames.append(pd.DataFrame(rows, columns=names))
        cur = chunk_end + pd.Timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    df = pd.concat(frames, ignore_index=True)
    ts = df["ts"]
    if pd.api.types.is_numeric_dtype(ts):
        df.index = pd.to_datetime(ts, unit="ms" if float(ts.iloc[0]) > 1e12 else "s")
    else:
        df.index = pd.to_datetime(ts)
    df = df[["Open", "High", "Low", "Close"]].astype(float)
    return df[~df.index.duplicated(keep="last")].sort_index()


_EXPIRY_CACHE = {}


def _g_monthly_expiry(g, on_date):
    on = pd.Timestamp(on_date); cand = []
    for off in (0, 1):
        t = on + pd.DateOffset(months=off)
        ckey = (t.year, t.month)
        if ckey not in _EXPIRY_CACHE:
            resp = _throttled(lambda yy=t.year, mm=t.month: g.get_expiries(
                exchange=g.EXCHANGE_NSE, underlying_symbol=UNDERLYING, year=yy, month=mm))
            exps = resp["expiries"] if isinstance(resp, dict) and "expiries" in resp else resp
            parsed = []
            for e in exps:
                try:
                    dt = pd.to_datetime(e, format="%d%b%y")
                except Exception:
                    dt = pd.to_datetime(e)
                parsed.append(dt.normalize())
            _EXPIRY_CACHE[ckey] = parsed
        cand += _EXPIRY_CACHE[ckey]
        fut = [d for d in cand if d >= on.normalize()]
        if fut:
            return min(fut)
    return min(cand)


def _in_lunch(t, side, first):
    lo, hi = LUNCH_CE if (side == "CE" and first) else LUNCH_PE
    return lo <= t <= hi


def _size_by_premium(entry_prem, premium_stop):
    """Premium-buffer sizing with a min-buffer floor and a hard lots cap."""
    buf = max(entry_prem - premium_stop + 1.0, MIN_BUFFER_FRAC * max(entry_prem, 1e-9))
    if buf <= 0:
        return 0
    return max(0, min(int(RISK_PER_TRADE / buf / LOT_SIZE), MAX_LOTS))


def _find_trigger(day1, setup_ts, setup_hi, setup_lo, side, first):
    """First 1-min candle within TRIGGER_WINDOW_MIN after the 5m setup candle that
    breaks the level (matches the SP trigger rules). Returns (ts, entry_spot)."""
    win_end = setup_ts + pd.Timedelta(minutes=TRIGGER_WINDOW_MIN)
    w = day1[(day1.index > setup_ts) & (day1.index <= win_end)]
    for tts, r in w.iterrows():
        o, h, l, c = r["Open"], r["High"], r["Low"], r["Close"]
        if side == "CE":
            if first:
                if h >= setup_hi + ENTRY_TRIGGER_PTS:      # first CE: +30 breakout
                    return tts, float(h)
            elif c > o and h > setup_hi:                   # second CE: green break of high
                return tts, float(h)
        else:                                              # PE: red 1m closing below setup low
            if c < o and l < setup_lo:
                return tts, float(l)
    return None, None


# ============================================================
# SHARED STRATEGY BRAIN  -- the SAME decisions for backtest and live paper.
# These are pure functions: no data fetching, no order placement. The backtest
# feeds them historical bars; the live runner feeds them broker polls. Source and
# destination differ; the rules below do not.
# ============================================================

def _setup_side(ts, open_, close_, osc, first):
    """A 5-min candle's setup side: CE (green + osc>0), PE (red + osc<0), else None.
    Respects the 09:40 start and the lunch dead-zone."""
    t = ts.time()
    if t < dtime(9, 40):
        return None
    green, red = close_ > open_, close_ < open_
    if green and osc > 0 and not _in_lunch(t, "CE", first):
        return "CE"
    if red and osc < 0 and not _in_lunch(t, "PE", first):
        return "PE"
    return None


def _bounce_ok(entry_spot, level_lo, level_hi, side):
    """Trigger price must be within BOUNCE_BAND of yesterday's (or today's running) level."""
    if side == "CE":
        return entry_spot <= level_lo + BOUNCE_BAND
    return entry_spot >= level_hi - BOUNCE_BAND


def _choose_strike(entry_spot, side):
    """500-point OTM strike, rounded to the strike step."""
    pts = OTM_POINTS if side == "CE" else -OTM_POINTS
    return round((entry_spot + pts) / STRIKE_STEP) * STRIKE_STEP


def _size_premium(entry_prem, premium_stop):
    """Premium-buffer sizing. Returns (lots, buffer, floored?, capped?)."""
    raw_buf = entry_prem - premium_stop + 1.0
    floor_buf = MIN_BUFFER_FRAC * max(entry_prem, 1e-9)
    buf = max(raw_buf, floor_buf)
    floored = raw_buf < floor_buf
    raw_lots = int(RISK_PER_TRADE / buf / LOT_SIZE) if buf > 0 else 0
    lots = max(0, min(raw_lots, MAX_LOTS))
    return lots, buf, floored, (raw_lots > MAX_LOTS)


def _size_by_premium(entry_prem, premium_stop):   # backward-compatible thin wrapper
    return _size_premium(entry_prem, premium_stop)[0]


def _exit_decision(side, pos, t1, now_time, spot, prem, last2_5m):
    """Decide the exit action for the current tick. Returns (action, n_lots, detail);
    action is one of None/TIME/STOP/BE_STOP/T1/T2. The caller books the fill and,
    for T1, sets pos['half']=True and trails pos['stop_prem'] to breakeven."""
    if now_time >= SQUAREOFF:
        return ("TIME", pos["lots"], "15:20 square-off")
    if prem is not None and prem <= pos["stop_prem"]:
        return ("BE_STOP" if pos["half"] else "STOP", pos["lots"],
                f"prem {prem:.1f} <= stop {pos['stop_prem']:.1f}")
    if not pos["half"]:
        hit = (spot >= t1) if side == "CE" else (spot <= t1)
        if hit and prem is not None:
            return ("T1", math.ceil(pos["lots"] / 2), f"spot {spot:.0f} reached pivot {t1:.0f}")
    elif last2_5m is not None and prem is not None:        # T2 only after T1, on a completed-5m basis
        c0, c1 = last2_5m
        if side == "PE":
            against = (c1["Close"] > c1["Open"]) and (c0["Close"] > c0["Open"]); osc_ok = c1["osc"] > 0; col = "green"
        else:
            against = (c1["Close"] < c1["Open"]) and (c0["Close"] < c0["Open"]); osc_ok = c1["osc"] < 0; col = "red"
        if against and osc_ok:
            return ("T2", pos["lots"], f"2 {col} 5m + {col} osc")
    return (None, 0, "")


def backtest_real_groww(start, end, cost=None, g=None):
    """Real-data backtest matching the YCloseBounce stored procedures
    (5-min setup + 1-min trigger bounce near yesterday's / today's level),
    with the live bot's premium-based sizing and exits. Reads only; never trades.
    Pass an authenticated `g` (GrowwAPI) to reuse an existing login (e.g. from the dashboard)."""
    print("[backtest] logging in to Groww...", flush=True)
    _SKIPS.clear(); g = g or _groww_login(); cost = cost or OptCost()
    print(f"[backtest] fetching {UNDERLYING} 5-min index {start} -> {end} ...", flush=True)
    idx5 = _g_candles(g, INDEX_GROWW_SYMBOL, g.SEGMENT_CASH, start, end, "5m")
    if idx5.empty:
        print("No 5-min index data (check INDEX_GROWW_SYMBOL / dates)."); return []
    print(f"[backtest] fetching {UNDERLYING} 1-min index (for triggers) ...", flush=True)
    idx1 = _g_candles(g, INDEX_GROWW_SYMBOL, g.SEGMENT_CASH, start, end, "1m")
    if idx1.empty:
        print("No 1-min index data (triggers need 1-min). If Groww errors on the "
              "range, lower GROWW_1MIN_MAX_DAYS."); return []
    print(f"[backtest] {len(idx5)} 5m bars, {len(idx1)} 1m bars, "
          f"{idx5.index.normalize().nunique()} sessions", flush=True)
    idx5["osc"] = widner_oscillator(idx5["Close"])
    LAST_RUN["candles"] = idx5
    idx5["date"] = idx5.index.normalize()
    idx1["date"] = idx1.index.normalize()
    five_by_day = dict(list(idx5.groupby("date")))
    one_by_day = dict(list(idx1.groupby("date")))
    days = sorted(five_by_day)
    trades = []
    opt_cache = {}

    def opt_data(date, sym):
        key = (str(date.date()), sym)
        if key not in opt_cache:
            opt_cache[key] = _g_candles(g, sym, g.SEGMENT_FNO, date,
                                        date + pd.Timedelta(days=1), "1m")
        return opt_cache[key]

    for di in range(1, len(days)):
        date = days[di]; prev = five_by_day[days[di - 1]]
        day5 = five_by_day[date]; day1 = one_by_day.get(date)
        if day1 is None or day1.empty:
            continue
        PDH, PDL, PDC = prev["High"].max(), prev["Low"].min(), prev["Close"].iloc[-1]
        piv = floor_pivots(PDH, PDL, PDC)
        expiry = _g_monthly_expiry(g, date)
        T = max(1, (pd.Timestamp(expiry) - date).days) / 365.0

        five = list(day5.iterrows())
        run_lo = day5["Low"].cummin().values
        run_hi = day5["High"].cummax().values
        trade_no = 0; i = 0

        while i < len(five):
            setup_ts, s = five[i]
            if setup_ts.time() < dtime(9, 40):
                i += 1; continue
            osc = s["osc"]
            green = s["Close"] > s["Open"]; red = s["Close"] < s["Open"]
            first = (trade_no == 0)
            level_lo = PDL if first else run_lo[i]
            level_hi = PDH if first else run_hi[i]

            side = _setup_side(setup_ts, s["Open"], s["Close"], osc, first)
            if side is None:
                i += 1; continue

            trig_ts, entry_spot = _find_trigger(day1, setup_ts, s["High"], s["Low"], side, first)
            if trig_ts is None:
                _skip(f"[skip] {date.date()} {setup_ts.time()} {side} setup (osc {osc:+.0f}) "
                          f"- no 1m trigger within {TRIGGER_WINDOW_MIN}m")
                i += 1; continue
            ref_name = (("yest-low" if first else "today-low") if side == "CE"
                        else ("yest-high" if first else "today-high"))
            ref_lvl = level_lo if side == "CE" else level_hi
            if not _bounce_ok(entry_spot, level_lo, level_hi, side):
                sgn = "+" if side == "CE" else "-"
                _skip(f"[skip] {date.date()} {trig_ts.time()} {side} entry {entry_spot:.0f} outside "
                      f"band ({ref_name} {ref_lvl:.0f}{sgn}{BOUNCE_BAND})")
                i += 1; continue

            K = _choose_strike(entry_spot, side)
            sym = f"NSE-{UNDERLYING}-{pd.Timestamp(expiry).strftime('%d%b%y')}-{int(K)}-{side}"
            opt = opt_data(date, sym)
            if opt.empty:
                _skip(f"[skip] {date.date()} {trig_ts.time()} {side} - no option data {sym}")
                i += 1; continue
            ep = opt["Close"].reindex([trig_ts], method="ffill")
            if ep.empty or pd.isna(ep.iloc[0]):
                _skip(f"[skip] {date.date()} {trig_ts.time()} {side} - no option premium at entry")
                i += 1; continue
            entry_prem = float(ep.iloc[0])
            pre = opt[opt.index <= trig_ts]
            premium_stop = float(pre["Low"].min()) if len(pre) else entry_prem * (1 - MIN_BUFFER_FRAC)

            lots, buf, floored, capped = _size_premium(entry_prem, premium_stop)
            if lots < 1:
                _skip(f"[skip] {date.date()} {trig_ts.time()} {side} - buffer {buf:.1f} too wide "
                          f"for 1 lot (prem {entry_prem:.1f}, stop {premium_stop:.1f})")
                i += 1; continue

            entry_fill = cost.fill("BUY", entry_prem)
            fee = cost.charge("BUY", entry_fill, lots * LOT_SIZE)
            t1 = pick_t1(entry_spot, side, piv)
            brk = (entry_spot - s["High"]) if side == "CE" else (s["Low"] - entry_spot)
            dist = (entry_spot - level_lo) if side == "CE" else (level_hi - entry_spot)
            nth = {0: "1st", 1: "2nd", 2: "3rd"}.get(trade_no, f"{trade_no + 1}th")
            size_note = (" [CAPPED]" if capped else "") + (" [floored]" if floored else "")
            reason = (f"{nth} trade | setup {setup_ts.time().strftime('%H:%M')} "
                      f"{'green' if side == 'CE' else 'red'} osc {osc:+.0f} | "
                      f"{dist:+.0f}pt from {ref_name} {ref_lvl:.0f} (band {BOUNCE_BAND}) | "
                      f"1m {trig_ts.time().strftime('%H:%M')} broke setup "
                      f"{'high' if side == 'CE' else 'low'} by {brk:+.0f} | "
                      f"{int(K)}{side} {OTM_POINTS}OTM | prem {entry_prem:.1f} stop {premium_stop:.1f} "
                      f"buf {buf:.1f} -> {lots} lot{size_note}")
            tr = Trade(str(date.date()), side, K, lots, round(entry_spot, 1), round(entry_fill, 2),
                       entry_time=str(trig_ts), stop=round(premium_stop, 1), t1=round(t1, 1),
                       reason=reason)
            tr.legs.append(("ENTRY_FEE", 0, round(entry_fill, 2), -round(fee, 2),
                            str(trig_ts), round(float(entry_spot), 1), "buy"))
            trades.append(tr)

            pos = {"lots": lots, "half": False, "stop_prem": premium_stop, "entry_fill": entry_fill}
            exit_ts = None; mfe = 0.0; mae = 0.0

            def book(n, reason_lbl, prem, tts, spot, detail=""):
                if n <= 0:
                    return
                fp = cost.fill("SELL", prem); q = n * LOT_SIZE; fee2 = cost.charge("SELL", fp, q)
                tr.legs.append((reason_lbl, n, round(fp, 2),
                                round((fp - pos["entry_fill"]) * q - fee2, 2),
                                str(tts), round(float(spot), 1), detail))
                pos["lots"] -= n

            for tts, r1 in day1[day1.index > trig_ts].iterrows():
                spot_now = float(r1["Close"])
                pe = opt["Close"].reindex([tts], method="ffill")
                prem = float(pe.iloc[0]) if len(pe) and not pd.isna(pe.iloc[0]) else None
                if prem is not None:                              # excursions in premium points
                    mfe = max(mfe, prem - pos["entry_fill"]); mae = min(mae, prem - pos["entry_fill"])

                last2 = None
                if pos["half"] and tts.minute % 5 == 0:
                    done5 = day5[day5.index < tts]
                    if len(done5) >= 2:
                        last2 = (done5.iloc[-2], done5.iloc[-1])
                action, n_lots, detail = _exit_decision(side, pos, t1, tts.time(), spot_now, prem, last2)
                if action == "T1":
                    book(n_lots, "T1", prem, tts, spot_now, detail)
                    pos["half"] = True; pos["stop_prem"] = pos["entry_fill"]
                    if pos["lots"] <= 0:
                        exit_ts = tts; break
                    continue
                if action:
                    book(n_lots, action, prem if prem is not None else pos["entry_fill"],
                         tts, spot_now, detail)
                    exit_ts = tts; break

            if pos["lots"] > 0 and exit_ts is None:
                last_ts = day1.index[-1]
                pe = opt["Close"].reindex([last_ts], method="ffill")
                prem = float(pe.iloc[0]) if len(pe) and not pd.isna(pe.iloc[0]) else pos["entry_fill"]
                book(pos["lots"], "EOD", prem, last_ts, float(day1["Close"].iloc[-1]), "session end")
                exit_ts = last_ts
            tr.mfe = round(mfe, 1); tr.mae = round(mae, 1)

            trade_no += 1
            if not ALLOW_MULTIPLE_TRADES:
                break
            while i < len(five) and five[i][0] <= (exit_ts or setup_ts):     # resume after exit
                i += 1
    LAST_RUN["trades"] = trades; LAST_RUN["skips"] = list(_SKIPS)
    return trades


def _trade_to_dict(t):
    return {"date": t.date, "opt": t.opt, "strike": t.strike, "lots": t.lots,
            "entry_spot": t.entry_spot, "entry_prem": t.entry_prem, "entry_time": t.entry_time,
            "stop": t.stop, "t1": t.t1, "reason": t.reason, "mfe": t.mfe, "mae": t.mae,
            "pnl": round(t.pnl, 2),
            "legs": [{"reason": l[0], "lots": l[1], "prem": l[2], "pnl": l[3],
                      "time": l[4] if len(l) > 4 else "", "spot": l[5] if len(l) > 5 else None,
                      "detail": l[6] if len(l) > 6 else ""} for l in t.legs]}


def bundle_dict():
    """Return the dashboard bundle (candles + trades + skips) for the last backtest run."""
    c = LAST_RUN.get("candles")
    candles = []
    if c is not None and len(c):
        cc = c.reset_index()
        tcol = cc.columns[0]
        for _, r in cc.iterrows():
            candles.append({"dt": str(r[tcol]), "Open": float(r["Open"]), "High": float(r["High"]),
                            "Low": float(r["Low"]), "Close": float(r["Close"]),
                            "osc": float(r.get("osc", 0.0))})
    return {"underlying": UNDERLYING, "lot_size": LOT_SIZE, "strike_mode": STRIKE_MODE,
            "trades": [_trade_to_dict(t) for t in LAST_RUN.get("trades", [])],
            "skips": LAST_RUN.get("skips", []), "candles": candles}


def export_bundle(path="bundle.json"):
    """Write a self-contained JSON the Streamlit dashboard reads (candles + trades + skips)."""
    import json
    bundle = bundle_dict()
    with open(path, "w") as f:
        json.dump(bundle, f)
    print(f"[export] wrote {path}: {len(bundle['trades'])} trades, {len(bundle['candles'])} candles, "
          f"{len(bundle['skips'])} skips. Download this file and upload it to the dashboard.", flush=True)
    return path


def _dispatch():
    print(f"[startup] RUN_MODE={RUN_MODE}", flush=True)
    if RUN_MODE == "live":
        run_live_paper()
    elif RUN_MODE == "realbt":
        print(f"[startup] real-data backtest {BACKTEST_START} -> {BACKTEST_END}", flush=True)
        summarise(backtest_real_groww(BACKTEST_START, BACKTEST_END))
        export_bundle("bundle.json")
    else:
        print("RUN_MODE is 'off' -- set it to 'realbt' or 'live' at the top of the file.", flush=True)


# Run when executed as a script. The unbuffered stdout above is what makes
# output show up in Groww Cloud's logs.
if __name__ == "__main__":
    _dispatch()
