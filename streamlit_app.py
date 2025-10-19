# streamlit_app.py
"""
Multi-TF Crypto Dashboard (final)
- Live Binance klines (1m,5m,15m,30m,1h,2h,4h,1d)
- Top timeframe selector
- Per-timeframe weights & tolerances in sidebar
- Composite signals (Bullish/Neutral/Bearish)
- Dedicated reversal detection boxes per timeframe with reasons
- Mocked on-chain metrics (placeholder)
- Robust handling of pandas_ta return shapes
"""
import streamlit as st
import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict

st.set_page_config(layout="wide", page_title="Multi-TF Crypto Dashboard (Final)")

st.title("Multi-Timeframe Crypto Dashboard — Final")

# ----------------------------
# Constants
# ----------------------------
BINANCE_BASE = "https://api.binance.com/api/v3/klines"
TF_LIST = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d']
DEFAULT_SYMBOL = "BTCUSDT"

# ----------------------------
# Binance helper
# ----------------------------
@st.cache_data(ttl=20)
def fetch_binance_klines(symbol: str, interval: str, limit: int = 800) -> pd.DataFrame:
    params = {'symbol': symbol.upper(), 'interval': interval, 'limit': limit}
    r = requests.get(BINANCE_BASE, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume","close_time",
        "quote_asset_volume","num_trades","taker_buy_base_vol","taker_buy_quote_vol","ignore"
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df[['open','high','low','close','volume']].astype(float)
    df = df.rename(columns={'close':'price'})
    return df

# ----------------------------
# Mocked on-chain metrics
# ----------------------------
@st.cache_data(ttl=300)
def fetch_mock_onchain(days=90):
    dates = pd.date_range(end=datetime.utcnow(), periods=days, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'NUPL': np.random.uniform(0.2, 0.8, size=len(dates)),
        'Supply_in_Profit_pct': np.random.uniform(50, 90, size=len(dates)),
        'Miner_Netflow': np.random.uniform(-2000, 2000, size=len(dates)),
        'Exchange_Netflow': np.random.uniform(-5000, 5000, size=len(dates)),
        'Funding_Rate': np.random.uniform(-0.01, 0.01, size=len(dates))
    })
    df = df.set_index('date')
    return df

# ----------------------------
# pandas_ta robust wrapper
# ----------------------------
def safe_series(fn, *args, prefer=None, **kwargs):
    idx = None
    if args and isinstance(args[0], (pd.Series, pd.DataFrame)):
        idx = args[0].index
    try:
        res = fn(*args, **kwargs)
        if isinstance(res, pd.Series):
            return res.reindex(idx)
        if isinstance(res, pd.DataFrame):
            # prefer column name substring
            if prefer:
                for c in res.columns:
                    if prefer.lower() in str(c).lower():
                        return res[c].reindex(idx)
            return res.iloc[:, 0].reindex(idx)
        if isinstance(res, (list, np.ndarray)):
            return pd.Series(res, index=idx)
        return pd.Series([np.nan] * (len(idx) if idx is not None else 0), index=idx)
    except Exception:
        return pd.Series([np.nan] * (len(idx) if idx is not None else 0), index=idx)

# ----------------------------
# Indicators computation
# ----------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    price = out['price']
    vol = out.get('volume', pd.Series(1, index=out.index))

    out['EMA9'] = safe_series(ta.ema, price, length=9)
    out['EMA21'] = safe_series(ta.ema, price, length=21)
    out['EMA50'] = safe_series(ta.ema, price, length=50)
    out['EMA200'] = safe_series(ta.ema, price, length=200)
    out['RSI'] = safe_series(ta.rsi, price, length=14)

    # MACD: try to get both columns
    try:
        macd_df = ta.macd(price)
        if isinstance(macd_df, pd.DataFrame) and not macd_df.empty:
            out['MACD'] = macd_df.iloc[:,0].reindex(out.index)
            out['MACD_signal'] = macd_df.iloc[:,1].reindex(out.index) if macd_df.shape[1] > 1 else safe_series(ta.ema, out['MACD'].fillna(method='ffill'), length=9)
        else:
            raise Exception("macd empty")
    except Exception:
        out['MACD'] = safe_series(ta.ema, price, length=12) - safe_series(ta.ema, price, length=26)
        out['MACD_signal'] = safe_series(ta.ema, out['MACD'].fillna(method='ffill'), length=9)

    out['ATR'] = safe_series(ta.atr, price, price, price, length=14)
    out['OBV'] = safe_series(ta.obv, price, vol)

    try:
        adx_df = ta.adx(price, price, price, length=14)
        if isinstance(adx_df, pd.DataFrame) and not adx_df.empty:
            col = next((c for c in adx_df.columns if 'adx' in c.lower()), adx_df.columns[-1])
            out['ADX'] = adx_df[col].reindex(out.index)
        else:
            out['ADX'] = np.nan
    except Exception:
        out['ADX'] = np.nan

    try:
        sar = ta.sar(price, price)
        if isinstance(sar, pd.Series):
            out['SAR'] = sar.reindex(out.index)
        elif isinstance(sar, pd.DataFrame) and not sar.empty:
            out['SAR'] = sar.iloc[:,0].reindex(out.index)
        else:
            out['SAR'] = np.nan
    except Exception:
        out['SAR'] = np.nan

    try:
        pv = (price * vol).fillna(0).cumsum()
        cv = vol.fillna(1).cumsum()
        out['VWAP'] = (pv / cv).reindex(out.index)
    except Exception:
        out['VWAP'] = np.nan

    return out

# ----------------------------
# Per-timeframe default settings
# ----------------------------
def default_tf_settings():
    return {
        'weights': {'ema': 0.30, 'macd': 0.20, 'rsi': 0.15, 'adx': 0.05, 'obv': 0.05, 'nupl': 0.10, 'supply': 0.10},
        'tolerances': {
            'ema': 1.0, 'macd': 0.01, 'rsi_overbought':70, 'rsi_oversold':30,
            'adx_strong':25, 'nupl_top':0.5, 'nupl_bottom':0.25, 'profit_top':80, 'profit_bottom':60,
            'bull_thresh':0.3
        }
    }

if 'tf_settings' not in st.session_state:
    st.session_state.tf_settings = {tf: default_tf_settings() for tf in TF_LIST}

# ----------------------------
# Top controls (timeframe selector + edit TF selector)
# ----------------------------
tf_container = st.container()
with tf_container:
    cols = st.columns([1,1,1,3])
    main_tf = cols[0].selectbox("Main chart timeframe", TF_LIST, index=TF_LIST.index('2h'))
    symbol = cols[1].text_input("Symbol (Binance)", value=DEFAULT_SYMBOL)
    edit_tf = cols[2].selectbox("Edit controls for timeframe", TF_LIST, index=TF_LIST.index('2h'))
    cols[3].markdown("Use 'Edit controls for timeframe' to tune weights & tolerances per timeframe.")

# ----------------------------
# Sidebar: per-timeframe weights & tolerances
# ----------------------------
st.sidebar.header("Per-timeframe Controls")
st.sidebar.markdown(f"Editing controls for timeframe **{edit_tf}**")

settings = st.session_state.tf_settings.get(edit_tf, default_tf_settings())
w = settings['weights']
t = settings['tolerances']

st.sidebar.subheader("Weights")
w['ema'] = st.sidebar.slider("EMA weight", 0.0, 1.0, float(w['ema']), 0.05, key=f"{edit_tf}_w_ema")
w['macd'] = st.sidebar.slider("MACD weight", 0.0, 1.0, float(w['macd']), 0.05, key=f"{edit_tf}_w_macd")
w['rsi'] = st.sidebar.slider("RSI weight", 0.0, 1.0, float(w['rsi']), 0.05, key=f"{edit_tf}_w_rsi")
w['adx'] = st.sidebar.slider("ADX weight", 0.0, 1.0, float(w['adx']), 0.05, key=f"{edit_tf}_w_adx")
w['obv'] = st.sidebar.slider("OBV weight", 0.0, 1.0, float(w['obv']), 0.05, key=f"{edit_tf}_w_obv")
w['nupl'] = st.sidebar.slider("NUPL weight", 0.0, 1.0, float(w['nupl']), 0.05, key=f"{edit_tf}_w_nupl")
w['supply'] = st.sidebar.slider("Supply weight", 0.0, 1.0, float(w['supply']), 0.05, key=f"{edit_tf}_w_supply")

st.sidebar.subheader("Tolerances")
t['ema'] = st.sidebar.slider("EMA sensitivity (price units)", 0.01, 20.0, float(t['ema']), 0.01, key=f"{edit_tf}_t_ema")
t['macd'] = st.sidebar.slider("MACD threshold", 0.0005, 0.05, float(t['macd']), 0.0005, key=f"{edit_tf}_t_macd")
t['rsi_overbought'] = st.sidebar.slider("RSI overbought", 55, 90, int(t['rsi_overbought']), key=f"{edit_tf}_t_rsi_high")
t['rsi_oversold'] = st.sidebar.slider("RSI oversold", 10, 45, int(t['rsi_oversold']), key=f"{edit_tf}_t_rsi_low")
t['adx_strong'] = st.sidebar.slider("ADX strong threshold", 10, 50, int(t['adx_strong']), key=f"{edit_tf}_t_adx")
t['nupl_top'] = st.sidebar.slider("NUPL top", 0.3, 0.9, float(t['nupl_top']), 0.01, key=f"{edit_tf}_t_nupl_top")
t['nupl_bottom'] = st.sidebar.slider("NUPL bottom", 0.01, 0.3, float(t['nupl_bottom']), 0.01, key=f"{edit_tf}_t_nupl_bot")
t['profit_top'] = st.sidebar.slider("Supply profit high %", 60, 95, int(t['profit_top']), key=f"{edit_tf}_t_profit_top")
t['profit_bottom'] = st.sidebar.slider("Supply profit low %", 10, 70, int(t['profit_bottom']), key=f"{edit_tf}_t_profit_bot")
t['bull_thresh'] = st.sidebar.slider("Composite bull threshold", 0.1, 0.6, float(t['bull_thresh']), 0.01, key=f"{edit_tf}_t_bull_thresh")

st.session_state.tf_settings[edit_tf] = {'weights': w, 'tolerances': t}

# ----------------------------
# Fetch data for all timeframes (cached)
# ----------------------------
st.info("Fetching live price data from Binance for all timeframes (cached).")

tf_dfs = {}
for tf in TF_LIST:
    try:
        kdf = fetch_binance_klines(symbol, tf, limit=800)
        kdf = compute_indicators(kdf)
        tf_dfs[tf] = kdf
    except Exception as e:
        tf_dfs[tf] = pd.DataFrame()
        st.warning(f"Failed to fetch {tf}: {e}")

# mocked on-chain data
onchain = fetch_mock_onchain()

# ----------------------------
# Evaluate composite signal + reversal detection per TF
# ----------------------------
def evaluate_tf(df: pd.DataFrame, tf: str):
    if df.empty:
        return {'label':'Neutral','score':0.0,'breakdown':{}, 'latest':pd.Series(), 'reversal':False, 'reversal_reasons':[]}
    latest = df.iloc[-1]
    settings = st.session_state.tf_settings.get(tf, default_tf_settings())
    weights = settings['weights']; tol = settings['tolerances']

    score = 0.0; total_w = 0.0; breakdown = {}

    def push(name, sig, w, reason):
        nonlocal score, total_w, breakdown
        breakdown[name] = (sig, reason)
        if sig is None:
            return
        score += w * sig
        total_w += w

    # EMA cross
    ema9 = latest.get('EMA9', np.nan); ema21 = latest.get('EMA21', np.nan)
    if pd.notna(ema9) and pd.notna(ema21):
        diff = ema9 - ema21
        if diff > tol['ema']: push('EMA_cross', 1, weights['ema'], f"EMA9>EMA21 by {diff:.3f}")
        elif diff < -tol['ema']: push('EMA_cross', -1, weights['ema'], f"EMA9<EMA21 by {diff:.3f}")
        else: push('EMA_cross', 0, weights['ema'], f"EMA diff {diff:.3f}")
    else:
        push('EMA_cross', None, weights['ema'], "missing")

    # MACD
    macd = latest.get('MACD', np.nan); macds = latest.get('MACD_signal', np.nan)
    if pd.notna(macd) and pd.notna(macds):
        d = macd - macds
        if d > tol['macd']: push('MACD', 1, weights['macd'], f"MACD diff {d:.4f}")
        elif d < -tol['macd']: push('MACD', -1, weights['macd'], f"MACD diff {d:.4f}")
        else: push('MACD', 0, weights['macd'], f"MACD neutral {d:.4f}")
    else:
        push('MACD', None, weights['macd'], "missing")

    # RSI
    rsi = latest.get('RSI', np.nan)
    if pd.notna(rsi):
        if rsi >= tol['rsi_overbought']: push('RSI', -1, weights['rsi'], f"RSI {rsi:.1f} overbought")
        elif rsi <= tol['rsi_oversold']: push('RSI', 1, weights['rsi'], f"RSI {rsi:.1f} oversold")
        else: push('RSI', 0, weights['rsi'], f"RSI {rsi:.1f} neutral")
    else:
        push('RSI', None, weights['rsi'], "missing")

    # ADX (trend strength)
    adx = latest.get('ADX', np.nan); ema50 = latest.get('EMA50', np.nan); price = latest.get('price', np.nan)
    if pd.notna(adx):
        if adx >= tol['adx_strong']:
            if pd.notna(price) and pd.notna(ema50):
                push('ADX', 1 if price > ema50 else -1, weights['adx'], f"ADX {adx:.1f} strong")
            else:
                push('ADX', 0, weights['adx'], f"ADX {adx:.1f} strong")
        else:
            push('ADX', 0, weights['adx'], f"ADX {adx:.1f} weak")
    else:
        push('ADX', None, weights['adx'], "missing")

    # OBV stub
    obv = latest.get('OBV', np.nan)
    push('OBV', 0 if pd.notna(obv) else None, weights['obv'], "OBV snapshot")

    # on-chain (daily mocked)
    try:
        chain_latest = onchain.iloc[-1]
    except Exception:
        chain_latest = pd.Series()
    nupl = chain_latest.get('NUPL', np.nan)
    if pd.notna(nupl):
        if nupl >= tol['nupl_top']: push('NUPL', -1, weights['nupl'], f"NUPL {nupl:.2f} high")
        elif nupl <= tol['nupl_bottom']: push('NUPL', 1, weights['nupl'], f"NUPL {nupl:.2f} low")
        else: push('NUPL', 0, weights['nupl'], f"NUPL {nupl:.2f}")
    else:
        push('NUPL', None, weights['nupl'], "missing")

    supply = chain_latest.get('Supply_in_Profit_pct', np.nan)
    if pd.notna(supply):
        if supply >= tol['profit_top']: push('Supply_in_Profit_pct', -1, weights['supply'], f"{supply:.1f}% high")
        elif supply <= tol['profit_bottom']: push('Supply_in_Profit_pct', 1, weights['supply'], f"{supply:.1f}% low")
        else: push('Supply_in_Profit_pct', 0, weights['supply'], f"{supply:.1f}%")
    else:
        push('Supply_in_Profit_pct', None, weights['supply'], "missing")

    final_score = (score / total_w) if total_w > 0 else 0.0
    label = 'Neutral'
    if final_score >= tol['bull_thresh']: label = 'Bullish'
    elif final_score <= -tol['bull_thresh']: label = 'Bearish'

    # Reversal detection (EMA cross / MACD cross / RSI flip within last N candles)
    reversal = False; rev_reasons = []
    try:
        recent = df[['EMA9','EMA21','MACD','MACD_signal','RSI']].dropna().tail(6)
        if len(recent) >= 3:
            # EMA cross detection
            s = np.sign(recent['EMA9'] - recent['EMA21'])
            if len(np.unique(s)) > 1 and s.iloc[-1] != s.iloc[-2]:
                reversal = True; rev_reasons.append("Recent EMA cross")
            # MACD cross detection
            m = np.sign(recent['MACD'] - recent['MACD_signal'])
            if len(np.unique(m)) > 1 and m.iloc[-1] != m.iloc[-2]:
                reversal = True; rev_reasons.append("Recent MACD cross")
            # RSI flip
            r = recent['RSI']
            if (r.iloc[-2] >= tol['rsi_overbought'] and r.iloc[-1] < tol['rsi_overbought']) or (r.iloc[-2] <= tol['rsi_oversold'] and r.iloc[-1] > tol['rsi_oversold']):
                reversal = True; rev_reasons.append("RSI flip")
    except Exception:
        pass

    # primary reason
    primary = next((f"{k}: {v[1]}" for k, v in breakdown.items() if v[1] not in ("missing","N/A")), "No strong signals")

    return {'label': label, 'score': final_score, 'breakdown': breakdown, 'latest': latest, 'reversal': reversal, 'reversal_reasons': rev_reasons, 'primary': primary}

# compute for all TFs
tf_results = {tf: evaluate_tf(tf_dfs.get(tf, pd.DataFrame()), tf) for tf in TF_LIST}

# ----------------------------
# Main chart for selected timeframe
# ----------------------------
st.subheader(f"{symbol} — Price ({main_tf})")
main_df = tf_dfs.get(main_tf, pd.DataFrame())
if main_df.empty:
    st.error("No data for this symbol/timeframe.")
else:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=main_df.index, y=main_df['price'], mode='lines', name='Price'))
    if 'EMA9' in main_df.columns: fig.add_trace(go.Scatter(x=main_df.index, y=main_df['EMA9'], mode='lines', name='EMA9', line=dict(width=1)))
    if 'EMA21' in main_df.columns: fig.add_trace(go.Scatter(x=main_df.index, y=main_df['EMA21'], mode='lines', name='EMA21', line=dict(width=1)))
    fig.update_layout(height=520, margin=dict(l=20,r=20,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Multi-TF cards
# ----------------------------
st.subheader("Multi-timeframe Overview")
cards_per_row = 4
rows = (len(TF_LIST) + cards_per_row - 1) // cards_per_row
for r in range(rows):
    start = r * cards_per_row
    end = min(start + cards_per_row, len(TF_LIST))
    cols = st.columns(end - start, gap="large")
    for i, tf in enumerate(TF_LIST[start:end]):
        res = tf_results[tf]
        label = res['label']; score = res['score']; latest = res['latest']
        color = "#D3D3D3"
        if label == 'Bullish': color = "#2ECC71"
        elif label == 'Bearish': color = "#E74C3C"
        display = f"{latest['price']:.2f}" if isinstance(latest, pd.Series) and 'price' in latest and pd.notna(latest['price']) else f"score {score:.2f}"
        reason = res.get('primary','')
        cols[i].markdown(
            f"""
            <div style="background:{color};padding:14px;border-radius:10px;min-height:110px;">
              <div style="font-weight:800;font-size:15px">{tf} — {label}</div>
              <div style="font-size:20px;font-weight:800;margin-top:8px">{display}</div>
              <div style="font-size:12px;margin-top:8px;opacity:0.95">{reason}</div>
            </div>
            """, unsafe_allow_html=True
        )

# ----------------------------
# Reversal boxes (explicit)
# ----------------------------
st.subheader("Reversal Signals (explicit)")
cols = st.columns(len(TF_LIST), gap="large")
for i, tf in enumerate(TF_LIST):
    res = tf_results[tf]
    rev = res['reversal']; reasons = res['reversal_reasons']
    if rev:
        color = "#F39C12"  # orange for 'possible reversal' (distinct from bull/red)
        label = "Potential Reversal"
        reason_text = "; ".join(reasons) if reasons else res.get('primary','')
    else:
        color = "#D3D3D3"
        label = "No reversal"
        reason_text = res.get('primary','')
    cols[i].markdown(
        f"""
        <div style="background:{color};padding:12px;border-radius:8px;text-align:center;min-height:80px;">
          <div style="font-weight:700">{tf}</div>
          <div style="font-size:16px;margin-top:6px">{label}</div>
          <div style="font-size:12px;margin-top:6px;opacity:0.95">{reason_text}</div>
        </div>
        """, unsafe_allow_html=True
    )

# ----------------------------
# On-chain static boxes
# ----------------------------
st.subheader("On-chain (static demo)")
chain_row = onchain.iloc[-1] if not onchain.empty else pd.Series()
onchain_metrics = [
    ('NUPL', chain_row.get('NUPL', np.nan)),
    ('Supply_in_Profit_pct', chain_row.get('Supply_in_Profit_pct', np.nan)),
    ('Miner_Netflow', chain_row.get('Miner_Netflow', np.nan)),
    ('Exchange_Netflow', chain_row.get('Exchange_Netflow', np.nan)),
    ('Funding_Rate', chain_row.get('Funding_Rate', np.nan)),
]
per_row = 3
rows = (len(onchain_metrics) + per_row - 1) // per_row
for r in range(rows):
    start = r * per_row
    end = min(start + per_row, len(onchain_metrics))
    cols = st.columns(end - start, gap="large")
    for i, (name, val) in enumerate(onchain_metrics[start:end]):
        if pd.isna(val):
            color = "#D3D3D3"; text = "Unavailable"
        else:
            if name == 'NUPL':
                if val >= st.session_state.tf_settings[main_tf]['tolerances']['nupl_top']:
                    color = "#E74C3C"; text = f"NUPL {val:.2f} (high)"
                elif val <= st.session_state.tf_settings[main_tf]['tolerances']['nupl_bottom']:
                    color = "#2ECC71"; text = f"NUPL {val:.2f} (low)"
                else:
                    color = "#D3D3D3"; text = f"NUPL {val:.2f} neutral"
            elif name == 'Supply_in_Profit_pct':
                if val >= st.session_state.tf_settings[main_tf]['tolerances']['profit_top']:
                    color = "#E74C3C"; text = f"{val:.1f}% (high)"
                elif val <= st.session_state.tf_settings[main_tf]['tolerances']['profit_bottom']:
                    color = "#2ECC71"; text = f"{val:.1f}% (low)"
                else:
                    color = "#D3D3D3"; text = f"{val:.1f}% (neutral)"
            elif name in ('Miner_Netflow','Exchange_Netflow'):
                if val > 0:
                    color = "#E74C3C"; text = f"{val:.0f} inflow (selling)"
                elif val < 0:
                    color = "#2ECC71"; text = f"{abs(val):.0f} outflow (buying)"
                else:
                    color = "#D3D3D3"; text = "neutral"
            else:
                color = "#D3D3D3"; text = f"{val:.4f}"
        cols[i].markdown(
            f"""<div style="background:{color};padding:12px;border-radius:10px;text-align:center;min-height:90px">
                <div style="font-weight:700">{name}</div>
                <div style="font-size:18px;margin-top:6px">{'' if pd.isna(val) else f'{val:.3f}'}</div>
                <div style="font-size:12px;opacity:0.9;margin-top:6px">{text}</div>
               </div>""", unsafe_allow_html=True
        )

# ----------------------------
# Forecast placeholder for main_tf
# ----------------------------
st.subheader(f"30-day simple forecast (placeholder) — {main_tf}")
main_latest = tf_dfs.get(main_tf, pd.DataFrame()).iloc[-1] if not tf_dfs.get(main_tf, pd.DataFrame()).empty else pd.Series()
last_price = main_latest.get('price', np.nan)
if pd.isna(last_price):
    st.info("No price available for forecast.")
else:
    days = 30
    rng = np.random.default_rng(seed=42)
    shocks = rng.normal(loc=0.0, scale=0.01, size=days)
    prices = [last_price]
    for s in shocks:
        prices.append(prices[-1] * (1 + s))
    fc = pd.DataFrame({'date':[datetime.utcnow().date() + timedelta(days=i) for i in range(1, days+1)], 'price': prices[1:]})
    st.line_chart(fc.set_index('date')['price'])
