"""
Bank Nifty strategy dashboard (online, no DB, no Groww login).

Reads a `bundle.json` produced by intraday_bnf_options.py (RUN_MODE="realbt" writes it),
and shows: candlestick + Rainbow oscillator, pivots, entry/exit markers with the full
"why" reason, a trades table with MFE/MAE, the skip log, and summary stats.

Run locally:        streamlit run app.py
Deploy online:      push app.py + requirements.txt to a public GitHub repo, then
                    https://share.streamlit.io -> New app -> pick the repo -> Deploy.
Then just upload your bundle.json in the sidebar.
"""

import json
from datetime import time as dtime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------- pure helpers (importable / testable, no Streamlit) ----------

def floor_pivots(h, l, c):
    p = (h + l + c) / 3
    return {"P": p, "R1": 2 * p - l, "S1": 2 * p - h,
            "R2": p + (h - l), "S2": p - (h - l),
            "R3": h + 2 * (p - l), "S3": l - 2 * (h - p)}


def load_bundle(raw):
    """raw: dict (already parsed) or JSON string/bytes."""
    if isinstance(raw, (str, bytes, bytearray)):
        raw = json.loads(raw)
    cdf = pd.DataFrame(raw.get("candles", []))
    if not cdf.empty:
        cdf["dt"] = pd.to_datetime(cdf["dt"])
        cdf["date"] = cdf["dt"].dt.date.astype(str)
        cdf["color"] = cdf["osc"].apply(lambda x: "#26a69a" if x > 0 else ("#ef5350" if x < 0 else "#9e9e9e"))
    return {"meta": {k: raw.get(k) for k in ("underlying", "lot_size", "strike_mode")},
            "candles": cdf, "trades": raw.get("trades", []), "skips": raw.get("skips", [])}


def day_pivots(cdf, day):
    """Floor pivots for `day` from the previous available session in the candles."""
    days = sorted(cdf["date"].unique())
    if day not in days or days.index(day) == 0:
        return None
    prev = cdf[cdf["date"] == days[days.index(day) - 1]]
    if prev.empty:
        return None
    return floor_pivots(prev["High"].max(), prev["Low"].min(), prev["Close"].iloc[-1])


def build_day_figure(cdf, trades, day, show_pivots=True, setups=None):
    d = cdf[cdf["date"] == day]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.04,
                        subplot_titles=(f"Bank Nifty  {day}", "Rainbow oscillator"))
    fig.add_trace(go.Candlestick(x=d["dt"], open=d["Open"], high=d["High"], low=d["Low"],
                                 close=d["Close"], name="BNF",
                                 increasing_line_color="#26a69a", decreasing_line_color="#ef5350"),
                  row=1, col=1)
    if show_pivots:
        piv = day_pivots(cdf, day)
        if piv:
            for k, v in piv.items():
                fig.add_hline(y=v, line_dash="dot", line_color="#9aa0a6", opacity=0.6,
                              annotation_text=k, annotation_position="right", row=1, col=1)

    for s in (setups or []):                          # candidate setup candles (live mode)
        fig.add_trace(go.Scatter(x=[s[0]], y=[s[1]], mode="markers",
                                 marker=dict(symbol="diamond-open", size=11,
                                             color="#2e7d32" if s[2] == "CE" else "#c62828"),
                                 name=f"{s[2]} setup", hovertext=f"candidate {s[2]} setup",
                                 hoverinfo="text", showlegend=False), row=1, col=1)

    for t in [t for t in trades if t.get("date") == day]:
        et = pd.to_datetime(t["entry_time"]) if t.get("entry_time") else None
        if et is not None:
            sym = "triangle-up" if t["opt"] == "CE" else "triangle-down"
            fig.add_trace(go.Scatter(x=[et], y=[t["entry_spot"]], mode="markers",
                                     marker=dict(symbol=sym, size=14, color="#1565c0",
                                                 line=dict(width=1, color="white")),
                                     name=f"{t['opt']} entry",
                                     hovertext=f"ENTRY {t['opt']} {t['strike']:.0f}<br>{t.get('reason','')}",
                                     hoverinfo="text"), row=1, col=1)
        for l in t.get("legs", []):
            if l["reason"] == "ENTRY_FEE" or not l.get("time"):
                continue
            lt = pd.to_datetime(l["time"]); sp = l.get("spot")
            fig.add_trace(go.Scatter(x=[lt], y=[sp], mode="markers",
                                     marker=dict(symbol="x", size=11, color="#6a1b9a"),
                                     name=l["reason"],
                                     hovertext=f"{l['reason']}  {l['lots']} lot @ {l['prem']}"
                                               f"<br>P&L {l['pnl']:.0f}<br>{l.get('detail','')}",
                                     hoverinfo="text", showlegend=False), row=1, col=1)

    fig.add_trace(go.Bar(x=d["dt"], y=d["osc"], marker_color=list(d["color"]),
                         name="osc", showlegend=False), row=2, col=1)
    fig.update_layout(height=760, template="plotly_white", xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=40, b=10), hovermode="closest")
    return fig


def detect_setups(cdf_day):
    """Candidate 5-min setup candles (colour + oscillator sign) for live mode.
    Approximate: ignores the 1-min trigger and the yesterday-level bounce band."""
    out = []
    for _, r in cdf_day.iterrows():
        t = r["dt"].time()
        if t < dtime(9, 40) or (dtime(11, 0) <= t <= dtime(12, 30)):
            continue
        green = r["Close"] > r["Open"]; red = r["Close"] < r["Open"]
        if green and r["osc"] > 0:
            out.append((r["dt"], r["High"], "CE"))
        elif red and r["osc"] < 0:
            out.append((r["dt"], r["Low"], "PE"))
    return out


def live_fetch(token, secret, symbol, start, end, interval="5m"):
    """Log into Groww with the entered keys, fetch index candles, compute the oscillator.
    Reuses the strategy module's fetch + oscillator so it matches the backtest exactly."""
    import pyotp
    from growwapi import GrowwAPI
    import intraday_bnf_options as strat
    g = GrowwAPI(GrowwAPI.get_access_token(api_key=token, totp=pyotp.TOTP(secret).now()))
    df = strat._g_candles(g, symbol, g.SEGMENT_CASH, start, end, interval)
    if df.empty:
        return df
    df = df.copy()
    df["osc"] = strat.widner_oscillator(df["Close"])
    df = df.reset_index().rename(columns={df.reset_index().columns[0]: "dt"})
    df["dt"] = pd.to_datetime(df["dt"])
    df["date"] = df["dt"].dt.date.astype(str)
    df["color"] = df["osc"].apply(lambda x: "#26a69a" if x > 0 else ("#ef5350" if x < 0 else "#9e9e9e"))
    return df


def trade_rows(trades, day=None):
    rows = []
    for t in trades:
        if day and t.get("date") != day:
            continue
        exits = " / ".join(f"{l['reason']}({l['pnl']:.0f})"
                           for l in t.get("legs", []) if l["reason"] != "ENTRY_FEE")
        rows.append({"date": t["date"], "side": t["opt"], "strike": t["strike"], "lots": t["lots"],
                     "entry": (t.get("entry_time", "") or "")[-8:], "spot": t["entry_spot"],
                     "prem": t["entry_prem"], "stop": t["stop"], "T1": t["t1"],
                     "MFE": t.get("mfe", 0), "MAE": t.get("mae", 0), "net": t.get("pnl", 0),
                     "exits": exits, "why": t.get("reason", "")})
    return pd.DataFrame(rows)


def stats(trades):
    if not trades:
        return {}
    pnls = [t.get("pnl", 0) for t in trades]
    wins = [p for p in pnls if p > 0]; losses = [p for p in pnls if p < 0]
    gl = sum(losses)
    return {"Trades": len(trades), "Win rate": f"{len(wins)/len(trades)*100:.0f}%",
            "Net P&L": f"{sum(pnls):,.0f}", "Profit factor": (f"{sum(wins)/abs(gl):.2f}" if gl else "inf"),
            "Best": f"{max(pnls):,.0f}", "Worst": f"{min(pnls):,.0f}"}


# ---------- Streamlit UI ----------

def main():
    import streamlit as st
    st.set_page_config(page_title="Bank Nifty Strategy Dashboard", layout="wide")
    st.title("Bank Nifty — Strategy Dashboard")

    with st.sidebar:
        st.header("Data source")
        src = st.radio("", ["Upload bundle", "Live (Groww login)"], label_visibility="collapsed")
        show_pivots = st.checkbox("Show pivot lines", value=True)

    b = None
    live = False

    if src == "Upload bundle":
        st.caption("Reads bundle.json from the backtest. No database, no Groww login.")
        up = st.sidebar.file_uploader("Upload bundle.json", type=["json"])
        pasted = st.sidebar.text_area("…or paste bundle JSON", height=100)
        raw = up.read() if up is not None else (pasted if pasted.strip() else None)
        if raw is None:
            st.info("Upload the `bundle.json` your backtest wrote (RUN_MODE='realbt'), or paste it.")
            return
        try:
            b = load_bundle(raw)
        except Exception as e:
            st.error(f"Could not read bundle: {e}"); return

    else:  # Live (Groww login)
        live = True
        st.caption("Logs into Groww with the keys you enter (kept only for this session) and "
                   "fetches index candles. Shows candles + oscillator + pivots + candidate setups. "
                   "Full trades/MFE/skips still come from a backtest bundle. Safest run locally.")
        with st.sidebar:
            tok = st.text_input("Groww TOTP token", type="password")
            sec = st.text_input("Groww TOTP secret", type="password")
            symbol = st.text_input("Index symbol", value="NSE-BANKNIFTY")
            c1, c2 = st.columns(2)
            start = c1.text_input("Start", value="2026-05-01")
            end = c2.text_input("End", value="2026-05-07")
            go_btn = st.button("Connect & fetch")
        if go_btn:
            if not tok or not sec:
                st.error("Enter both the TOTP token and secret."); return
            with st.spinner("Logging into Groww and fetching candles…"):
                try:
                    df = live_fetch(tok, sec, symbol, start, end, "5m")
                except Exception as e:
                    st.error(f"Groww fetch failed: {e}\n\nIf this is a geo/IP block from a cloud host, "
                             f"run the dashboard locally instead."); return
            if df is None or df.empty:
                st.warning("No candles returned (check symbol/dates)."); return
            st.session_state["live_df"] = df
        df = st.session_state.get("live_df")
        if df is None:
            st.info("Enter your keys and a date range, then **Connect & fetch**.")
            return
        b = {"meta": {"underlying": symbol, "lot_size": "-", "strike_mode": "live"},
             "candles": df, "trades": [], "skips": []}

    m = b["meta"]
    st.write(f"**{m.get('underlying')}**  ·  lot {m.get('lot_size')}  ·  mode `{m.get('strike_mode')}`")

    s = stats(b["trades"])
    if s:
        cols = st.columns(len(s))
        for c, (k, v) in zip(cols, s.items()):
            c.metric(k, v)

    cdf = b["candles"]
    days = sorted(set(t["date"] for t in b["trades"]) |
                  (set(cdf["date"].unique()) if not cdf.empty else set()))
    if not days:
        st.warning("No dates found."); return
    day = st.selectbox("Day", days, index=len(days) - 1)

    tab1, tab2, tab3, tab4 = st.tabs(["Chart", "Trades", "Skips", "Raw candles"])

    with tab1:
        if cdf.empty:
            st.warning("No candle data.")
        else:
            setups = detect_setups(cdf[cdf["date"] == day]) if live else None
            st.plotly_chart(build_day_figure(cdf, b["trades"], day, show_pivots, setups),
                            use_container_width=True)
            if live:
                st.caption("◇ diamonds = candidate setups (5-min colour + oscillator only; the "
                           "1-min trigger and yesterday-level bounce band aren't applied here).")

    with tab2:
        if live:
            st.info("Live mode shows data + oscillator + candidate setups. For executed trades "
                    "with entries, exits, MFE/MAE and P&L, load a backtest bundle.")
        else:
            st.subheader(f"Trades on {day}")
            st.dataframe(trade_rows(b["trades"], day), use_container_width=True)
            for t in [t for t in b["trades"] if t["date"] == day]:
                with st.expander(f"{t['opt']} {t['strike']:.0f} — net {t.get('pnl',0):.0f}"):
                    st.write(t.get("reason", ""))
                    st.table(pd.DataFrame(t.get("legs", [])))
            st.divider(); st.subheader("All trades")
            st.dataframe(trade_rows(b["trades"]), use_container_width=True)

    with tab3:
        if live:
            st.info("Skip log comes from a backtest bundle (run with DEBUG_SKIPS=1).")
        else:
            sk = [x for x in b["skips"] if day in x]
            st.subheader(f"Skipped setups on {day}  ({len(sk)})")
            st.code("\n".join(sk) if sk else "none recorded for this day")

    with tab4:
        if not cdf.empty:
            dd = cdf[cdf["date"] == day][["dt", "Open", "High", "Low", "Close", "osc"]]
            st.dataframe(dd, use_container_width=True)
            st.download_button("Download this day (CSV)", dd.to_csv(index=False),
                               file_name=f"banknifty_{day}.csv", mime="text/csv")


if __name__ == "__main__":
    main()
