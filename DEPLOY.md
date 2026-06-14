# Bank Nifty dashboard — deploy online (free)

You need 3 files in a public GitHub repo: `app.py`, `requirements.txt`, and (optionally) a sample `bundle.json`.

## 1. Produce a bundle
Run the backtest with `RUN_MODE = "realbt"`. It writes `bundle.json`
(candles + trades + skips). Download that file.
- Run it locally (your India IP, full Groww data access), or on Groww Cloud.
- The skip log fills in only if `DEBUG_SKIPS=1` (env var) when the backtest runs.

## 2. Put the app on GitHub
Create a public repo and add `app.py` + `requirements.txt`. (No keys, no bundle needed in the repo.)

## 3. Deploy on Streamlit Community Cloud (free)
1. Go to https://share.streamlit.io and sign in with GitHub.
2. New app -> pick your repo -> main file `app.py` -> Deploy.
3. You get a public URL (open on any browser or phone).

## 4. Use it
In the sidebar, **upload your bundle.json** (or paste its contents). Pick a day to see:
- candlestick + pivot lines + entry/exit markers (hover for the full "why"),
- the Rainbow oscillator histogram,
- the trades table with MFE/MAE and per-leg exits,
- the skip log, and a raw-candle CSV download.

## Alternative host
Hugging Face Spaces (choose the Streamlit SDK) works the same way — push the two files, it builds and serves a public URL.

## Note on static IP
Not needed here. The static-IP rule is only for **placing orders**. This dashboard only
reads/visualises data, so it never touches that requirement.

## Live-login mode (optional)
The dashboard has a second data source: **Live (Groww login)**. You enter your TOTP
token + secret in the sidebar (kept only for that session), pick a date range, and it
fetches index candles and shows candles + Rainbow oscillator + pivots + candidate setups.
It does NOT run the full options backtest live — executed trades, MFE/MAE and the skip log
still come from a bundle.

Requirements for live mode:
- `intraday_bnf_options.py` must be in the same repo/folder (the app reuses its fetch + oscillator).
- `growwapi` and `pyotp` must be installed (already in requirements.txt).

Safety: live mode puts your keys into the running app. On a PUBLIC deployment that is a real
exposure — prefer running live mode **locally** (`streamlit run app.py`). Keep the deployed
copy on the upload-bundle source.

## Deploy on Render
Render is a generic web host, so deploy the dashboard as a **Web Service**. The one thing
that matters: Streamlit must listen on Render's `$PORT` and on 0.0.0.0.

Easiest: commit the included `render.yaml` to your repo, then in Render choose
**New + > Blueprint** and point it at the repo — it reads render.yaml and sets everything up.

Manual (no render.yaml):
1. Push `app.py`, `requirements.txt` (and `intraday_bnf_options.py` if you want live mode) to a GitHub repo.
2. Render dashboard -> New + -> Web Service -> connect the repo.
3. Runtime: Python.  Instance: Free.
4. Build command:  pip install -r requirements.txt
5. Start command:
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
6. Create. You get a URL like https://banknifty-dashboard.onrender.com

Notes:
- Free tier spins down after ~15 min idle, so the first load after that takes ~30-60s (cold start). Normal.
- On Render (public + servers outside India), use the **Upload bundle** source. Don't use live-login here:
  your Groww keys would sit in a public app, and the live fetch may be geo-blocked from Render's IP.
  Use live mode locally instead.
