# streamlit_live_crypto_dashboard_resilient.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import pandas_ta as ta
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Crypto Dashboard — Resilient Indicators")

st.title("Crypto & On-Chain Dashboard — Resilient Indicators (red/green/gray boxes)")

# ---------------------- Helpers ----------------------
@st.cache_data(ttl=300)
def fetch_price_data(coin='bitcoin', vs_currency='usd', days=90):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {'vs_currency': vs_currency, 'days': days}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    j = resp.json()
    df = pd.DataFrame(j['prices'], columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df.drop('timestamp', axis=1, inplace=True)
    return df

@st.cache_data(ttl=300)
def fetch_blockchain_metrics():
    # Mock blockchain metrics: replace with real API calls if desired
    dates = pd.date_range(end=datetime.today(), periods=90)
    df = pd.DataFrame({
        'date': dates,
        'NUPL': np.random.uniform(0.2, 0.8, size=len(dates)),
        'Supply_in_Profit_pct': np.random.uniform(50, 90, size=len(dates)),
        'Miner_Netflow': np.random.uniform(-2000, 2000, size=len(dates)),
        'Exchange_Netflow': np.random.uniform(-5000, 5000, size=len(dates)),
        'Funding_Rate': np.random.uniform(-0.01, 0.01, size=len(dates)),
        'Miner_Reserve': np.random.uniform(1e5, 2e6, size=len(dates)),
        'Exchange_Reserve': np.random.uniform(1e6, 3e6, size=len(dates)),
        # If you have a volume series from a proper data source, include it.
    })
    df.set_index('date', inplace=True)
    return df

# Robust wrapper to call pandas_ta functions and normalize outputs
def _safe_ta_call(fn, *args, df_index=None, prefer_col_contains=None, **kwargs):
    """
    Call a pandas_ta function (fn) with args/kwargs.
    Returns a pd.Series aligned to df_index (if provided) or indexed from args[0].
    prefer_col_contains: list or str of substrings to prefer when picking a column from DataFrame results.
    On any failure, returns a Series of NaN with appropriate index.
    """
    # determine index to use for fallback NaN
    idx = df_index
    if idx is None and len(args) > 0 and isinstance(args[0], (pd.Series, pd.DataFrame)):
        idx = args[0].index
    try:
        res = fn(*args, **kwargs)
        # If Series, return directly
        if isinstance(res, pd.Series):
            return res.reindex(idx) if idx is not None else res
        # If DataFrame, try to pick sensible column
        if isinstance(res, pd.DataFrame):
            # If user specified substrings to prefer, try those first
            if prefer_col_contains:
                prefs = [prefer_col_contains] if isinstance(prefer_col_contains, str) else list(prefer_col_contains)
                for p in prefs:
                    for col in res.columns:
                        if p in str(col):
                            return res[col].reindex(idx) if idx is not None else res[col]
            # common pandas_ta naming heuristics
            for candidate in ['MACD_12_26_9','MACD','MACDh_12_26_9','MACDs_12_26_9','VOSC_14_28','VOSC','BBU_20_2.0','BBL_20_2.0','ADX_14','SAR_0.02_0.2']:
                if candidate in res.columns:
                    return res[candidate].reindex(idx) if idx is not None else res[candidate]
            # fallback: return first column
            return res.iloc[:,0].reindex(idx) if idx is not None else res.iloc[:,0]
        # Some pandas_ta functions may return dict-like or numpy arrays -> try to convert to Series
        if isinstance(res, (list, np.ndarray)):
            s = pd.Series(res, index=idx)
            return s
        # otherwise fallback to NaN series
        return pd.Series([np.nan]*len(idx), index=idx)
    except Exception:
        # On any error, return NaN series with correct index
        if idx is not None:
            return pd.Series([np.nan]*len(idx), index=idx)
        else:
            return pd.Series([])  # empty fallback

# ---------------------- Indicator computation (robust) ----------------------
def compute_indicators(df):
    df = df.copy()
    idx = df.index

    # EMAs
    df['EMA9']  = _safe_ta_call(ta.ema, df['price'], length=9, df_index=idx)
    df['EMA21'] = _safe_ta_call(ta.ema, df['price'], length=21, df_index=idx)
    df['EMA50'] = _safe_ta_call(ta.ema, df['price'], length=50, df_index=idx)
    df['EMA200']= _safe_ta_call(ta.ema, df['price'], length=200, df_index=idx)

    # RSI
    df['RSI'] = _safe_ta_call(ta.rsi, df['price'], length=14, df_index=idx)

    # MACD (may return DataFrame)
    macd_series = _safe_ta_call(ta.macd, df['price'], df_index=idx, prefer_col_contains=['MACD','MACDh','MACDs'])
    # macd_series will pick a single column (main MACD). To get MACD signal, call again and try to extract signal column if available:
    macd_full = None
    try:
        macd_full = ta.macd(df['price'])
    except Exception:
        macd_full = None
    if isinstance(macd_full, pd.DataFrame):
        # try to fetch signal column names
        sig_col = next((c for c in macd_full.columns if 'signal' in str(c).lower() or 'macds' in str(c).lower()), None)
        if sig_col is None:
            # try known names
            for ctry in ['MACDs_12_26_9','MACDs','MACD_signal','MACDh_12_26_9']:
                if ctry in macd_full.columns:
                    sig_col = ctry
                    break
        df['MACD'] = macd_full[macd_full.columns[0]].reindex(idx)
        if sig_col:
            df['MACD_signal'] = macd_full[sig_col].reindex(idx)
        else:
            # try to compute a signal as ema of MACD if not present
            df['MACD_signal'] = _safe_ta_call(ta.ema, df.get('MACD', df['price']), length=9, df_index=idx)
    else:
        # fallback - use the single series result and create a signal as EMA(9) of it
        df['MACD'] = macd_series
        df['MACD_signal'] = _safe_ta_call(ta.ema, df['MACD'].fillna(method='ffill'), length=9, df_index=idx)

    # Bollinger Bands
    bb = _safe_ta_call(ta.bbands, df['price'], df_index=idx)
    if isinstance(bb, pd.Series):
        # rare: single series - put it in both upper/lower as NaN
        df['BB_upper'] = pd.Series(np.nan, index=idx)
        df['BB_lower'] = pd.Series(np.nan, index=idx)
    elif isinstance(bb, pd.DataFrame) and not bb.empty:
        # pick columns heuristically
        upper_col = next((c for c in bb.columns if 'bbu' in str(c).lower() or 'upper' in str(c).lower()), None)
        lower_col = next((c for c in bb.columns if 'bbl' in str(c).lower() or 'lower' in str(c).lower()), None)
        if upper_col:
            df['BB_upper'] = bb[upper_col].reindex(idx)
        else:
            df['BB_upper'] = bb.iloc[:,0].reindex(idx)
        if lower_col and lower_col in bb.columns:
            df['BB_lower'] = bb[lower_col].reindex(idx)
        elif bb.shape[1] > 1:
            df['BB_lower'] = bb.iloc[:,1].reindex(idx)
        else:
            df['BB_lower'] = pd.Series(np.nan, index=idx)
    else:
        df['BB_upper'] = pd.Series(np.nan, index=idx)
        df['BB_lower'] = pd.Series(np.nan, index=idx)

    # ATR
    df['ATR'] = _safe_ta_call(ta.atr, df['price'], df['price'], df['price'], length=14, df_index=idx)

    # OBV (needs volume). If no volume column, fall back to ones
    volume_series = df.get('volume', pd.Series(1, index=idx))
    df['OBV'] = _safe_ta_call(ta.obv, df['price'], volume_series, df_index=idx)

    # Volume Oscillator - use ta.vosc if present, else fallback
    try:
        vosc = ta.vosc(volume_series, length_fast=14, length_slow=28)
        # vosc may be dataframe or series
        if isinstance(vosc, pd.Series):
            df['VO'] = vosc.reindex(idx)
        elif isinstance(vosc, pd.DataFrame) and not vosc.empty:
            # pick column containing 'vosc' or first column
            col = next((c for c in vosc.columns if 'vosc' in str(c).lower()), vosc.columns[0])
            df['VO'] = vosc[col].reindex(idx)
        else:
            df['VO'] = pd.Series(np.nan, index=idx)
    except Exception:
        # fallback: simple diff of short/long moving averages of volume
        df['VO'] = (volume_series.rolling(14).mean() - volume_series.rolling(28).mean()).reindex(idx)

    # ADX
    try:
        adx_df = ta.adx(df['price'], df['price'], df['price'], length=14)
        if isinstance(adx_df, pd.DataFrame) and not adx_df.empty:
            adx_col = next((c for c in adx_df.columns if 'adx' in str(c).lower()), adx_df.columns[-1])
            df['ADX'] = adx_df[adx_col].reindex(idx)
        else:
            df['ADX'] = pd.Series(np.nan, index=idx)
    except Exception:
        df['ADX'] = pd.Series(np.nan, index=idx)

    # SAR
    try:
        sar = ta.sar(df['price'], df['price'])
        if isinstance(sar, pd.Series):
            df['SAR'] = sar.reindex(idx)
        elif isinstance(sar, pd.DataFrame) and not sar.empty:
            df['SAR'] = sar.iloc[:,0].reindex(idx)
        else:
            df['SAR'] = pd.Series(np.nan, index=idx)
    except Exception:
        df['SAR'] = pd.Series(np.nan, index=idx)

    # VWAP (approx) - needs volume ideally; use cumprice/cumcount as proxy if no volume
    try:
        vol = df.get('volume', pd.Series(1, index=idx)).fillna(1)
        cum_pv = (df['price'] * vol).cumsum()
        cum_v  = vol.cumsum()
        df['VWAP'] = (cum_pv / cum_v).reindex(idx)
    except Exception:
        df['VWAP'] = pd.Series(np.nan, index=idx)

    # CMF placeholder if no typical price + volume; try pandas_ta.cmf if available
    try:
        cmf = ta.cmf(df['price'], df['price'], df['price'], volume_series, length=20)
        if isinstance(cmf, pd.Series):
            df['CMF'] = cmf.reindex(idx)
        elif isinstance(cmf, pd.DataFrame) and not cmf.empty:
            df['CMF'] = cmf.iloc[:,0].reindex(idx)
        else:
            df['CMF'] = pd.Series(np.nan, index=idx)
    except Exception:
        df['CMF'] = pd.Series(np.nan, index=idx)

    # keep df aligned
    df = df.reindex(idx)
    return df

# ---------------------- Signal helpers ----------------------
def safe_get(latest, col_name):
    return latest[col_name] if col_name in latest and pd.notna(latest[col_name]) else np.nan

def get_signal_color(value, metric, latest_row=None):
    # returns color and explanation
    if pd.isna(value):
        return 'gray', f"{metric}: data unavailable"
    if metric == 'EMA_cross':
        if value > 0.5:
            return 'green', f"EMA9 above EMA21 (+{value:.2f})"
        elif value < -0.5:
            return 'red', f"EMA9 below EMA21 ({value:.2f})"
        else:
            return 'gray', f"Neutral EMA cross ({value:.2f})"
    if metric == 'RSI':
        if value > 60: return 'red', f"Overbought ({value:.1f})"
        if value < 40: return 'green', f"Oversold ({value:.1f})"
        return 'gray', f"RSI neutral ({value:.1f})"
    if metric == 'MACD':
        if value > 0.01: return 'green', f"MACD bullish ({value:.3f})"
        if value < -0.01: return 'red', f"MACD bearish ({value:.3f})"
        return 'gray', f"MACD neutral ({value:.3f})"
    if metric == 'NUPL':
        if value > 0.5: return 'red', f"NUPL high (market top-ish) {value:.2f}"
        if value < 0.25: return 'green', f"NUPL low (market bottom-ish) {value:.2f}"
        return 'gray', f"NUPL neutral ({value:.2f})"
    if metric == 'Supply_in_Profit_pct':
        if value > 80: return 'red', f"High supply in profit {value:.1f}%"
        if value < 60: return 'green', f"Low supply in profit {value:.1f}%"
        return 'gray', f"Supply in profit neutral {value:.1f}%"
    if metric in ('Miner_Netflow','Exchange_Netflow'):
        if value > 0: return 'red', f"Selling pressure {value:.0f}"
        if value < 0: return 'green', f"Accumulation {value:.0f}"
        return 'gray', f"Netflow neutral {value:.0f}"
    if metric == 'Funding_Rate':
        if value > 0.002: return 'red', f"Long pressure {value:.4f}"
        if value < -0.002: return 'green', f"Short pressure {value:.4f}"
        return 'gray', f"Funding neutral {value:.4f}"
    if metric == 'ATR':
        # treat higher ATR as bearish (volatility) and low ATR as neutral/bullish depending on use-case
        if value > np.nanpercentile([v for v in latest_row.get('ATR_series', [np.nan]) if not pd.isna(v)], 75): 
            return 'red', f"High ATR {value:.4f}"
        return 'gray', f"ATR {value:.4f}"
    # Generic fallback
    return 'gray', f"{metric}: {value}"

# ---------------------- Sidebar ----------------------
st.sidebar.header("Settings")
coin = st.sidebar.selectbox("Select coin", ['bitcoin', 'ethereum'])
days = st.sidebar.selectbox("Timeframe (days)", [30, 60, 90, 180])

# ---------------------- Data ----------------------
price_df = fetch_price_data(coin, days=days)
chain_df = fetch_blockchain_metrics()
df = price_df.join(chain_df, how='left')
df = compute_indicators(df)
latest = df.iloc[-1]

# ---------------------- Price Chart ----------------------
st.subheader(f"{coin.capitalize()} Price Chart")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['price'], mode='lines', name='Price'))
fig.update_layout(height=520, margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig, use_container_width=True)

# ---------------------- Red/Green/Gray Boxes ----------------------
st.subheader("Latest Signals & Metrics")

metrics_list = [
    'EMA_cross','RSI','MACD','NUPL','Supply_in_Profit_pct',
    'Miner_Netflow','Exchange_Netflow','Funding_Rate',
    'ATR','ADX','SAR','VWAP','CMF','OBV','VO'
]

# Show boxes in responsive rows (max 6 per row)
per_row = 6
rows = (len(metrics_list) + per_row - 1) // per_row
for r in range(rows):
    start = r*per_row
    end = min(start+per_row, len(metrics_list))
    cols = st.columns(end-start)
    for i, m in enumerate(metrics_list[start:end]):
        # handle EMA_cross specially
        if m == 'EMA_cross':
            a = safe_get(latest, 'EMA9')
            b = safe_get(latest, 'EMA21')
            val = np.nan if pd.isna(a) or pd.isna(b) else (a - b)
        else:
            val = safe_get(latest, m)
        color, explanation = get_signal_color(val, m, latest_row=latest)
        cols[i].markdown(
            f"<div style='background-color:{color};padding:12px;border-radius:8px;text-align:center;'>"
            f"<div style='font-weight:600'>{m}</div>"
            f"<div style='margin-top:6px;font-size:14px'>{explanation}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

# ---------------------- Forecast (simple placeholder) ----------------------
st.subheader("30-Day Simple Forecast (Scenario)")
last_price = latest['price'] if 'price' in latest else np.nan
forecast_days = 30
proj = pd.DataFrame({'date':[datetime.now()+timedelta(days=i) for i in range(1,forecast_days+1)]})
proj['price'] = last_price * (1 + np.cumsum(np.random.normal(0, 0.01, size=forecast_days)))
st.line_chart(proj.set_index('date')['price'])
