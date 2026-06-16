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


def load_engine():
    """Import the strategy engine regardless of its filename (intraday_bnf_options.py or intra.py)."""
    import importlib
    for name in ("intraday_bnf_options", "intra"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise ImportError("engine not found \u2014 expected intraday_bnf_options.py or intra.py next to app.py")


# ---------- pure helpers (importable / testable, no Streamlit) ----------

def floor_pivots(h, l, c):
    p = (h + l + c) / 3
    return {"P": p, "R1": 2 * p - l, "S1": 2 * p - h,
            "R2": p + (h - l), "S2": p - (h - l),
            "R3": h + 2 * (p - l), "S3": l - 2 * (h - p)}


# ---------- modern Material theme (dark trading console) ----------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Roboto+Mono:wght@400;500&display=swap');
:root{
  --bg:#0E1116; --surface:#161B22; --surface2:#1C2230; --line:#2A3340;
  --text:#E6EDF3; --muted:#8B98A5; --primary:#4F8CFF; --bull:#26A69A; --bear:#EF5350; --warn:#E3B341;
}
.stApp{background:var(--bg); color:var(--text); font-family:'Roboto',sans-serif;}
section.main > div{padding-top:1rem;}
h1,h2,h3,h4{font-family:'Roboto',sans-serif; font-weight:700; letter-spacing:.2px;}
.mat-card{background:var(--surface); border:1px solid var(--line); border-radius:14px;
  padding:16px 18px; margin:10px 0; box-shadow:0 1px 2px rgba(0,0,0,.4),0 4px 16px rgba(0,0,0,.25);}
.mat-eyebrow{font-family:'Roboto Mono',monospace; font-size:11px; letter-spacing:1.5px;
  text-transform:uppercase; color:var(--muted); margin-bottom:8px;}
.mat-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:10px;}
.mat-cell{background:var(--surface2); border-radius:10px; padding:10px 12px; border:1px solid var(--line);}
.mat-k{font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.8px;}
.mat-v{font-family:'Roboto Mono',monospace; font-size:20px; font-weight:500; margin-top:2px;}
.mat-formula{font-family:'Roboto Mono',monospace; font-size:12px; color:var(--muted); margin-top:2px;}
.mat-row{display:flex; justify-content:space-between; gap:12px; padding:6px 0; border-bottom:1px dashed var(--line);
  font-family:'Roboto Mono',monospace; font-size:13px;}
.mat-row:last-child{border-bottom:none;}
.mat-name{color:var(--text); font-weight:500;} .mat-calc{color:var(--muted);} .mat-out{color:var(--primary);}
.bull{color:var(--bull);} .bear{color:var(--bear);}
.mat-chip{display:inline-block; padding:3px 10px; border-radius:999px; font-family:'Roboto Mono',monospace;
  font-size:12px; font-weight:500; border:1px solid var(--line);}
.chip-flat{background:#20262E; color:var(--muted);} .chip-armed{background:rgba(227,179,65,.15); color:var(--warn); border-color:var(--warn);}
.chip-pos{background:rgba(79,140,255,.15); color:var(--primary); border-color:var(--primary);}
</style>
"""


def _card(eyebrow, body):
    return f'<div class="mat-card"><div class="mat-eyebrow">{eyebrow}</div>{body}</div>'


def pivots_card(pdh, pdl, pdc, src):
    p = (pdh + pdl + pdc) / 3
    rng = pdh - pdl
    rows = [
        ("P",  f"(H+L+C)/3 = ({pdh:.0f}+{pdl:.0f}+{pdc:.0f})/3", p),
        ("R1", f"2P-L = 2\u00d7{p:.1f}-{pdl:.0f}", 2 * p - pdl),
        ("S1", f"2P-H = 2\u00d7{p:.1f}-{pdh:.0f}", 2 * p - pdh),
        ("R2", f"P+(H-L) = {p:.1f}+{rng:.0f}", p + rng),
        ("S2", f"P-(H-L) = {p:.1f}-{rng:.0f}", p - rng),
        ("R3", f"H+2(P-L) = {pdh:.0f}+2\u00d7{p - pdl:.1f}", pdh + 2 * (p - pdl)),
        ("S3", f"L-2(H-P) = {pdl:.0f}-2\u00d7{pdh - p:.1f}", pdl - 2 * (pdh - p)),
    ]
    body = (f'<div class="mat-formula">from yesterday ({src}): H {pdh:.0f} \u00b7 L {pdl:.0f} \u00b7 C {pdc:.0f}</div>')
    for name, calc, out in rows:
        body += (f'<div class="mat-row"><span class="mat-name">{name}</span>'
                 f'<span class="mat-calc">{calc}</span><span class="mat-out">{out:.1f}</span></div>')
    return _card("pivots \u00b7 classic floor", body)


def oscillator_card(closes, layers, last_osc):
    """Recompute the rainbow ribbon for the latest bar and show the breakdown."""
    s = pd.Series([float(x) for x in closes], dtype=float)
    ribbon = [s]
    for _ in range(layers - 1):
        ribbon.append(ribbon[-1].rolling(2).mean())
    rib = pd.concat(ribbon, axis=1)
    last = rib.iloc[-1].dropna()
    avav = last.mean(); hi = last.max(); lo = last.min(); rng = (hi - lo) or 1e-9
    close = s.iloc[-1]; hist = 100 * (close - avav) / rng
    over = float(pd.Series([100 * (s.iloc[i] - rib.iloc[i].dropna().mean()) /
                            ((rib.iloc[i].dropna().max() - rib.iloc[i].dropna().min()) or 1e-9)
                            for i in range(max(0, len(s) - 10), len(s))]).abs().max())
    cls = "bull" if hist > 0 else "bear"
    body = (f'<div class="mat-grid">'
            f'<div class="mat-cell"><div class="mat-k">close</div><div class="mat-v">{close:.1f}</div></div>'
            f'<div class="mat-cell"><div class="mat-k">avAv (ribbon mean)</div><div class="mat-v">{avav:.1f}</div></div>'
            f'<div class="mat-cell"><div class="mat-k">ribbon spread</div><div class="mat-v">{rng:.1f}</div>'
            f'<div class="mat-formula">{hi:.1f} \u2212 {lo:.1f}</div></div>'
            f'<div class="mat-cell"><div class="mat-k">hist (osc)</div><div class="mat-v {cls}">{hist:.1f}</div></div>'
            f'<div class="mat-cell"><div class="mat-k">Over / Under</div><div class="mat-v">\u00b1{over:.1f}</div></div>'
            f'</div>'
            f'<div class="mat-formula" style="margin-top:8px;">'
            f'ribbon = Close + {layers - 1} recursive 2-SMA lines \u00b7 '
            f'hist = 100\u00d7(Close\u2212avAv)/(max\u2212min) \u00b7 Over = max|hist| over 10 bars</div>')
    return _card("rainbow oscillator", body)


def rules_card(cfg):
    et = cfg.get("entry_trigger_pts", 30); bb = cfg.get("bounce_band", 150)
    tw = cfg.get("trigger_window_min", 10); otm = cfg.get("otm_points", 500)
    lce, lpe = cfg.get("lunch_ce", ["11:30", "12:30"]), cfg.get("lunch_pe", ["11:00", "12:30"])
    body = (f'<div class="mat-row"><span class="mat-name bull">CE</span>'
            f'<span class="mat-calc">green 5m + osc&gt;0, \u226509:40, not lunch {lce[0][:5]}\u2013{lce[1][:5]} \u00b7 '
            f'1m high \u2265 setup high +{et} within {tw}m \u00b7 near PDL +{bb}</span></div>'
            f'<div class="mat-row"><span class="mat-name bear">PE</span>'
            f'<span class="mat-calc">red 5m + osc&lt;0, \u226509:40, not lunch {lpe[0][:5]}\u2013{lpe[1][:5]} \u00b7 '
            f'1m red break below setup low within {tw}m \u00b7 near PDH \u2212{bb}</span></div>'
            f'<div class="mat-row"><span class="mat-name">strike</span>'
            f'<span class="mat-calc">{otm} OTM, rounded to 100</span></div>')
    return _card("setup \u00b7 trigger \u00b7 strike rules", body)


def sizing_card(entry, cfg):
    prem = entry.get("prem", 0); stop = entry.get("stop", 0); lots = entry.get("lots", 0)
    risk = cfg.get("risk", 3000); lot = cfg.get("lot_size", 30); mx = cfg.get("max_lots", 10)
    mbf = cfg.get("min_buffer_frac", 0.15)
    buf = max(prem - stop + 1.0, mbf * max(prem, 1e-9))
    per_lot = buf * lot
    raw = int(risk / per_lot) if per_lot else 0
    body = (f'<div class="mat-row"><span class="mat-name">buffer</span>'
            f'<span class="mat-calc">max(entry\u2212stop+1, {int(mbf*100)}%\u00d7entry) = '
            f'max({prem:.1f}\u2212{stop:.1f}+1, {mbf*prem:.1f})</span><span class="mat-out">{buf:.1f}</span></div>'
            f'<div class="mat-row"><span class="mat-name">risk / lot</span>'
            f'<span class="mat-calc">buffer \u00d7 lot = {buf:.1f} \u00d7 {lot}</span><span class="mat-out">\u20b9{per_lot:.0f}</span></div>'
            f'<div class="mat-row"><span class="mat-name">lots</span>'
            f'<span class="mat-calc">min(\u230a{risk:.0f}/{per_lot:.0f}\u230b, {mx}) = min({raw}, {mx})</span>'
            f'<span class="mat-out">{lots}</span></div>')
    return _card("position sizing", body)


def load_bundle(raw):
    """raw: dict (already parsed) or JSON string/bytes."""
    if isinstance(raw, (str, bytes, bytearray)):
        raw = json.loads(raw)
    cdf = pd.DataFrame(raw.get("candles", []))
    if not cdf.empty:
        cdf["dt"] = pd.to_datetime(cdf["dt"])
        cdf["date"] = cdf["dt"].dt.date.astype(str)
        cdf["color"] = cdf["osc"].apply(lambda x: "#26a69a" if x > 0 else ("#ef5350" if x < 0 else "#9e9e9e"))
        cdf["over"] = cdf["osc"].abs().rolling(10, min_periods=1).max()    # Over band (HHV |osc|, 10)
        cdf["under"] = -cdf["over"]                                         # Under = -Over
    return {"meta": {k: raw.get(k) for k in ("underlying", "lot_size", "strike_mode")},
            "candles": cdf, "trades": raw.get("trades", []), "skips": raw.get("skips", []),
            "open_position": raw.get("open_position"), "realized_pnl": raw.get("realized_pnl"),
            "last_decision": raw.get("last_decision"), "decisions": raw.get("decisions", []),
            "config": raw.get("config", {}), "updated": raw.get("updated")}


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

    if "over" in d.columns:                              # Over/Under bands (filled envelope, like the platform)
        fig.add_trace(go.Scatter(x=d["dt"], y=d["over"], mode="lines", line=dict(width=0),
                                 fill="tozeroy", fillcolor="rgba(38,166,154,0.20)",
                                 name="Over", hoverinfo="skip", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=d["dt"], y=d["under"], mode="lines", line=dict(width=0),
                                 fill="tozeroy", fillcolor="rgba(239,83,80,0.20)",
                                 name="Under", hoverinfo="skip", showlegend=False), row=2, col=1)
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
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.title("Bank Nifty \u2014 Strategy Dashboard")

    with st.sidebar:
        st.header("Data source")
        src = st.radio("Data source", ["Live (Groww login)", "Live paper (run here)",
                                       "Live paper (file)", "Upload bundle"],
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

    elif src == "Live paper (run here)":
        st.caption("Runs the paper loop INSIDE this app \u2014 each refresh does one poll "
                   "(fetch \u2192 strategy step \u2192 update state). No separate runner, no file. "
                   "Needs Groww reachable from wherever this app runs (a cloud host's foreign "
                   "IP may be geo-blocked \u2014 if so, run the dashboard locally).")
        env_tok = os.environ.get("GROWW_TOTP_TOKEN")
        env_sec = os.environ.get("GROWW_TOTP_SECRET")
        with st.sidebar:
            if env_tok and env_sec:
                st.success("Using Groww keys from the environment.")
                tok, sec = env_tok, env_sec
            else:
                tok = st.text_input("Groww TOTP token", type="password")
                sec = st.text_input("Groww TOTP secret", type="password")
                st.caption("Tip: set GROWW_TOTP_TOKEN / GROWW_TOTP_SECRET in a local .env.")
            live_on = st.toggle("Live polling", value=False)
        if not live_on:
            st.info("Enter your keys and flip **Live polling** on to start paper trading in-app. "
                    "Each refresh = one poll; nothing is ever ordered for real.")
            return
        if not tok or not sec:
            st.error("Enter both the TOTP token and secret.")
            return
        try:
            strat = load_engine()
        except ImportError as e:
            st.error(f"Needs **intraday_bnf_options.py next to app.py**, plus growwapi + pyotp. Missing: {e}")
            return
        if "paper_broker" not in st.session_state:                 # log in once, then reuse
            try:
                os.environ["GROWW_TOTP_TOKEN"] = tok
                os.environ["GROWW_TOTP_SECRET"] = sec
                os.environ.setdefault("LIVE_BROKER", "groww")
                with st.spinner("Logging into Groww\u2026"):
                    st.session_state["paper_broker"] = strat.build_option_broker()
                    st.session_state["paper_cost"] = strat.OptCost()
                    st.session_state["paper_state"] = strat._load_state()
            except Exception as e:
                st.error(f"Login/broker failed: {e}\n\nIf this is a geo/IP block from a cloud host, "
                         "run the dashboard locally (Indian IP).")
                return
        try:                                                       # one poll per render
            s, day5 = strat.live_paper_step(st.session_state["paper_broker"],
                                            st.session_state["paper_cost"],
                                            st.session_state["paper_state"])
            st.session_state["paper_state"] = s
            b = load_bundle(json.dumps(strat._paper_bundle(s, day5)))
        except Exception as e:
            st.error(f"Poll failed: {e}")
            return
        auto_refresh = True                                        # drive the next poll

    elif src == "Live paper (file)":
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
                strat = load_engine()
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

    ld = b.get("last_decision")
    if ld and ld.get("candle"):                                   # rich live "now" panel
        st.markdown("#### What the strategy sees now")
        st.markdown(f"**State:** `{ld.get('state','FLAT')}`  \u2014  {ld.get('action','')}")
        cc = ld.get("candle", {}); lv = ld.get("levels", {})
        a, c, e = st.columns(3)
        a.metric("Spot", f"{ld.get('spot','?')}")
        a.caption(f"setup candle: {cc.get('colour','?')} \u00b7 "
                  f"O{cc.get('o')} H{cc.get('h')} L{cc.get('l')} C{cc.get('c')}")
        c.metric("Oscillator", f"{cc.get('osc','?')}", cc.get("osc_colour", ""))
        c.caption(f"candle {cc.get('time','')[-8:]}")
        e.metric(f"Yesterday (src: {lv.get('src','?')})", f"H {lv.get('PDH','?')}")
        e.caption(f"L {lv.get('PDL','?')} \u00b7 C {lv.get('PDC','?')} \u00b7 "
                  f"today {lv.get('run_lo','?')}\u2013{lv.get('run_hi','?')}")
        piv = lv.get("pivots", {})
        if piv:
            st.caption("Pivots \u2014 " + "   ".join(f"{k} {piv[k]}"
                       for k in ("R3", "R2", "R1", "P", "S1", "S2", "S3") if k in piv))
        ck = ld.get("checks", {})
        if "armed" in ck:
            am = ck["armed"]
            st.info(f"**ARMED {am['side']}** \u2014 waiting for **{am['waiting_for']}**, "
                    f"bounce {am['bounce_level']}, expires {am['expires']}")
        if "position" in ck:
            ps = ck["position"]
            st.info(f"**IN POSITION {ps['opt']} {ps['strike']} \u00d7{ps['lots']}** \u2014 "
                    f"prem now {ps['prem_now']} vs stop {ps['stop_prem']}, T1 {ps['t1']}"
                    + ("  (half booked)" if ps.get("half") else ""))

        st.markdown("#### Calculations")                          # Material cards: how everything is computed
        cfg = b.get("config", {})
        cards = []
        if lv.get("PDH") is not None:
            cards.append(pivots_card(lv["PDH"], lv["PDL"], lv["PDC"], lv.get("src", "?")))
        cdf_all = b["candles"]
        if cdf_all is not None and not cdf_all.empty:
            try:
                cards.append(oscillator_card(cdf_all["Close"].tolist(),
                                             int(cfg.get("osc_layers", 10)), cc.get("osc")))
            except Exception:
                pass
        if cfg:
            cards.append(rules_card(cfg))
        if "entry" in ck:
            cards.append(sizing_card(ck["entry"], cfg))
        cols = st.columns(2)
        for k, html in enumerate(cards):
            cols[k % 2].markdown(html, unsafe_allow_html=True)

    decs = b.get("decisions", [])                                 # decision log (live AND backtest)
    if decs:
        label = "Live decision log" if (ld and ld.get("candle")) else "Decision log (why each setup did/didn't trade)"
        with st.expander(f"{label} \u2014 {len(decs)} entries (latest first)"):
            for d in reversed(decs[-80:]):
                st.markdown(f"`{str(d.get('ts',''))[11:19]}` **{d.get('state','')}** "
                            f"\u2014 {d.get('action','')}")
    if ld and ld.get("candle"):
        with st.expander("Latest decision \u2014 raw detail"):
            st.json(ld)

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
