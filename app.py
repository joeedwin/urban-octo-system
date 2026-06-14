import json
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:                      # auto-load a local .env so GROWW_TOTP_TOKEN / GROWW_TOTP_SECRET are picked up
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


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
            "candles": cdf, "trades": raw.get("trades", []), "skips": raw.get("skips", []),
            "open_position": raw.get("open_position"), "realized_pnl": raw.get("realized_pnl"),
            "updated": raw.get("updated")}


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

def build_g(token, secret):
    import pyotp
    from growwapi import GrowwAPI
    return GrowwAPI(GrowwAPI.get_access_token(api_key=token, totp=pyotp.TOTP(secret).now()))


def main():
    import streamlit as st
    st.set_page_config(page_title="Bank Nifty Strategy Dashboard", layout="wide")
    st.title("Bank Nifty \u2014 Strategy Dashboard")

    with st.sidebar:
        st.header("Data source")
        src = st.radio("Data source", ["Live (Groww login)", "Live paper", "Upload bundle"],
                       label_visibility="collapsed")
        show_pivots = st.checkbox("Show pivot lines", value=True)

    b = None
    auto_refresh = False

    if src == "Upload bundle":
        st.caption("Reads bundle.json written by the backtest. No login.")
        up = st.sidebar.file_uploader("Upload bundle.json", type=["json"])
        pasted = st.sidebar.text_area("\u2026or paste bundle JSON", height=100)
        raw = up.read() if up is not None else (pasted if pasted.strip() else None)
        if raw is None:
            st.info("Upload the bundle.json your backtest wrote, or paste it.")
            return
        try:
            b = load_bundle(raw)
        except Exception as e:
            st.error(f"Could not read bundle: {e}")
            return

    elif src == "Live paper":
        st.caption("Watches the paper_bundle.json your live-paper runner writes (RUN_MODE='live'). "
                   "Same machine: give the file path and turn on auto-refresh. Remote: upload the file.")
        pmode = st.sidebar.radio("Paper source", ["File path", "Upload"], horizontal=True)
        raw = None
        if pmode == "File path":
            path = st.sidebar.text_input("paper_bundle.json path", value="paper_bundle.json")
            auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
            try:
                with open(path) as f:
                    raw = f.read()
            except Exception as e:
                st.warning(f"Can't read {path}: {e}\n\nStart the runner with RUN_MODE='live' so it "
                           "writes this file, and point to it (same machine).")
                return
        else:
            up = st.sidebar.file_uploader("Upload paper_bundle.json", type=["json"])
            raw = up.read() if up is not None else None
            if raw is None:
                st.info("Upload the paper_bundle.json your runner is writing.")
                return
        try:
            b = load_bundle(raw)
        except Exception as e:
            st.error(f"Could not read paper bundle: {e}")
            return

    else:  # Live (Groww login) -- runs the FULL strategy live
        st.caption("Logs into Groww and runs the full strategy live: trades, entries/exits, "
                   "MFE/MAE, skips, oscillator, pivots. Use a SHORT date range (it fetches 1-min + "
                   "option data, so it is slow). Best run locally \u2014 your keys go into the app.")
        from datetime import date, timedelta
        env_tok = os.environ.get("GROWW_TOTP_TOKEN")
        env_sec = os.environ.get("GROWW_TOTP_SECRET")
        with st.sidebar:
            if env_tok and env_sec:
                st.success("Using Groww keys from the environment.")
                tok, sec = env_tok, env_sec
            else:
                tok = st.text_input("Groww TOTP token", type="password")
                sec = st.text_input("Groww TOTP secret", type="password")
                st.caption("Tip: set GROWW_TOTP_TOKEN and GROWW_TOTP_SECRET (e.g. in a local .env) "
                           "to skip typing these.")
            c1, c2 = st.columns(2)
            start = c1.date_input("Start", value=date.today() - timedelta(days=4))
            end = c2.date_input("End", value=date.today())
            run_btn = st.button("Run live backtest")
        if run_btn:
            if not tok or not sec:
                st.error("Enter both the TOTP token and secret.")
                return
            try:
                import intra as strat
            except ImportError as e:
                st.error("Live mode needs **intraday_bnf_options.py next to app.py** (the strategy "
                         f"engine), plus growwapi + pyotp. Missing: {e}")
                return
            with st.spinner("Logging in and running the strategy live (can take a minute)\u2026"):
                try:
                    g = build_g(tok, sec)
                    strat.backtest_real_groww(str(start), str(end), g=g)
                    st.session_state["live_bundle"] = strat.bundle_dict()
                except Exception as e:
                    st.error(f"Live run failed: {e}\n\nIf this is a geo/IP block from a cloud host, "
                             "run the dashboard locally.")
                    return
        bundle = st.session_state.get("live_bundle")
        if bundle is None:
            st.info("Enter your keys and a date range, then **Run live backtest**.")
            return
        b = load_bundle(bundle)

    # ---- unified rendering (same for live and bundle) ----
    m = b["meta"]
    st.write(f"**{m.get('underlying')}**  \u00b7  lot {m.get('lot_size')}  \u00b7  mode `{m.get('strike_mode')}`")

    if b.get("open_position") is not None or b.get("realized_pnl") is not None:   # live-paper status
        pos = b.get("open_position")
        c1, c2, c3 = st.columns(3)
        if pos:
            c1.metric("Open position", f"{pos['opt']} {pos['strike']:.0f}",
                      f"{pos['lots']} lot(s)")
            c2.metric("Stop / T1", f"{pos.get('stop_prem','?')}", f"T1 {pos.get('t1','?')}")
        else:
            c1.metric("Open position", "Flat")
        c3.metric("Realized P&L", f"{b.get('realized_pnl', 0):,.0f}")
        if b.get("updated"):
            st.caption(f"Last updated: {b['updated']}")

    s = stats(b["trades"])
    if s:
        cols = st.columns(len(s))
        for c, (k, v) in zip(cols, s.items()):
            c.metric(k, v)

    cdf = b["candles"]
    days = sorted(set(t["date"] for t in b["trades"]) |
                  (set(cdf["date"].unique()) if not cdf.empty else set()))
    if not days:
        st.warning("No dates found.")
        return
    day = st.selectbox("Day", days, index=len(days) - 1)

    tab1, tab2, tab3, tab4 = st.tabs(["Chart", "Trades", "Skips", "Raw candles"])
    with tab1:
        if cdf.empty:
            st.warning("No candle data.")
        else:
            st.plotly_chart(build_day_figure(cdf, b["trades"], day, show_pivots),
                            use_container_width=True)
    with tab2:
        st.subheader(f"Trades on {day}")
        st.dataframe(trade_rows(b["trades"], day), use_container_width=True)
        for t in [t for t in b["trades"] if t["date"] == day]:
            with st.expander(f"{t['opt']} {t['strike']:.0f} \u2014 net {t.get('pnl',0):.0f}"):
                st.write(t.get("reason", ""))
                st.table(pd.DataFrame(t.get("legs", [])))
        st.divider(); st.subheader("All trades")
        st.dataframe(trade_rows(b["trades"]), use_container_width=True)
    with tab3:
        sk = [x for x in b["skips"] if day in x]
        st.subheader(f"Skipped setups on {day}  ({len(sk)})")
        st.code("\n".join(sk) if sk else "none recorded for this day")
    with tab4:
        if not cdf.empty:
            dd = cdf[cdf["date"] == day][["dt", "Open", "High", "Low", "Close", "osc"]]
            st.dataframe(dd, use_container_width=True)
            st.download_button("Download this day (CSV)", dd.to_csv(index=False),
                               file_name=f"banknifty_{day}.csv", mime="text/csv")

    if auto_refresh:                       # live-paper: re-read the file every 30s
        import time as _t
        _t.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
