# streamlit_multi_tf_dashboard.py
"""
Multi-timeframe Crypto Dashboard (Binance live prices + static on-chain)
- Live price data: Binance public REST API (klines)
- Timeframes: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 1d
- Indicators: EMA(9,21,50,200), RSI(14), MACD, ATR, OBV, ADX (best-effort via pandas_ta)
- User controls: weights & tolerances in the sidebar
- On-chain metrics: static/mocked (placeholder)
- Output: price chart + multi-timeframe colored cards (green/gray/red) with value & reason
"""
import streamlit as st
import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict

st.set_page_config(layout="wide", page_title="Multi-Timeframe Crypto Dashboard (Live)")

st.title("Multi-Timeframe Crypto Dashboard — Live prices (Binance)")

# ---------------------------
# Requirements (for developer)
# ---------------------------
# pip install streamlit pandas numpy requests pandas_ta plotly

# ---------------------------
# Helper: Binance klines fetch
# ---------------------------
BINANCE_BASE = "https://api.binance.com/api/v3/klines"

# intervals mapping (valid Binance intervals)
VALID_INTERVALS = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d']

@st.cache_data(ttl=30)  # short cache so UI updates frequently but avoid hammering Binance
def fetch_binance_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """
    Fetch klines from Binance public API and return a DataFrame with columns:
    ['open','high','low','close','volume'] indexed by datetime
    """
    params = {'symbol': symbol.upper(), 'interval': interval, 'limit': limit}
    resp = requests.get(BINANCE_BASE, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume","close_time",
        "quote_asset_volume","num_trades","taker_buy_base_vol","taker_buy_quote_vol","ignore"
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df[['open','high','low','close','volume']].astype(float)
    df.rename(columns={'close':'price'}, inplace=True)
    return df

# ---------------------------
# Robust TA helpers
# ---------------------------
def safe_series(fn, *args, prefer_col_contains=None, **kwargs) -> pd.Series:
    """
    Call pandas_ta function fn on args and return a pd.Series aligned to args[0].index.
    If the function returns DataFrame, choose a sensible column.
    On exception, returns a NaN Series of the right length.
    """
    idx = None
    if len(args) > 0 and isinstance(args[0], (pd.Series, pd.DataFrame)):
        idx = args[0].index
    try:
        res = fn(*args, **kwargs)
        if isinstance(res, pd.Series):
            return res.reindex(idx)
        if isinstance(res, pd.DataFrame):
            # prefer a column containing substrings in prefer_col_contains
            if prefer_col_contains:
                prefs = [prefer_col_contains] if isinstance(prefer_col_contains, str) else list(prefer_col_contains)
                for p in prefs:
                    for c in res.columns:
                        if p.lower() in str(c).lower():
                            return res[c].reindex(idx)
            # common fallback names
            for candidate in ['MACD_12_26_9','MACD','MACDh_12_26_9','MACDs_12_26_9','VOSC_14_28','VOSC','BBU_20_2.0','BBL_20_2.0','ADX_14','SAR_0.02_0.2']:
                if candidate in res.columns:
                    return res[candidate].reindex(idx)
            # otherwise pick first numeric column
            return res.iloc[:,0].reindex(idx)
        if isinstance(res, (list, np.ndarray)):
            return pd.Series(res, index=idx)
        # unknown return -> NaN series
        return pd.Series([np.nan]* (len(idx) if idx is not None else 0), index=idx)
    except Exception:
        return pd.Series([np.nan]* (len(idx) if idx is not None else 0), index=idx)

# ---------------------------
# Compute indicators for a kline DF
# ---------------------------
def compute_indicators_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a selection of indicators to df (must include 'price' and optionally 'volume').
    Returns df with new columns (EMA9, EMA21, EMA50, EMA200, RSI, MACD, MACD_signal, ATR, OBV, ADX, SAR, VWAP)
    """
    if df.empty:
        return df
    df2 = df.copy()
    idx = df2.index

    # price series
    price = df2['price']

    # EMAs
    df2['EMA9'] = safe_series(ta.ema, price, length=9)
    df2['EMA21'] = safe_series(ta.ema, price, length=21)
    df2['EMA50'] = safe_series(ta.ema, price, length=50)
    df2['EMA200'] = safe_series(ta.ema, price, length=200)

    # RSI
    df2['RSI'] = safe_series(ta.rsi, price, length=14)

    # MACD (prefer MACD column)
    macd = None
    try:
        macd = ta.macd(price)
    except Exception:
        macd = None
    if isinstance(macd, pd.DataFrame) and not macd.empty:
        # place first column as MACD, second as signal if possible
        df2['MACD'] = macd.iloc[:,0].reindex(idx)
        if macd.shape[1] > 1:
            df2['MACD_signal'] = macd.iloc[:,1].reindex(idx)
        else:
            df2['MACD_signal'] = safe_series(ta.ema, df2['MACD'].fillna(method='ffill'), length=9)
    else:
        df2['MACD'] = safe_series(ta.ema, price, length=12) - safe_series(ta.ema, price, length=26)
        df2['MACD_signal'] = safe_series(ta.ema, df2['MACD'].fillna(method='ffill'), length=9)

    # ATR (using price as high/low/close fallback)
    df2['ATR'] = safe_series(ta.atr, price, price, price, length=14)

    # OBV (needs volume)
    vol = df2.get('volume', pd.Series(1, index=idx))
    df2['OBV'] = safe_series(ta.obv, price, vol)

    # ADX (requires high/low/close generally; use price as placeholder)
    try:
        adx_df = ta.adx(price, price, price, length=14)
        if isinstance(adx_df, pd.DataFrame) and not adx_df.empty:
            # try find ADX column
            adx_col = next((c for c in adx_df.columns if 'adx' in str(c).lower()), adx_df.columns[-1])
            df2['ADX'] = adx_df[adx_col].reindex(idx)
        else:
            df2['ADX'] = np.nan
    except Exception:
        df2['ADX'] = np.nan

    # SAR
    try:
        sar = ta.sar(price, price)
        if isinstance(sar, pd.Series):
            df2['SAR'] = sar.reindex(idx)
        elif isinstance(sar, pd.DataFrame) and not sar.empty:
            df2['SAR'] = sar.iloc[:,0].reindex(idx)
        else:
            df2['SAR'] = np.nan
    except Exception:
        df2['SAR'] = np.nan

    # VWAP approx (price*volume cumulative / volume cumulative) - requires volume
    try:
        pv = (price * vol).fillna(0).cumsum()
        cv = vol.fillna(1).cumsum()
        df2['VWAP'] = (pv / cv).reindex(idx)
    except Exception:
        df2['VWAP'] = np.nan

    return df2

# ---------------------------
# Composite signal calculation
# ---------------------------
def compute_tf_signal(latest_row: pd.Series, tolerances: Dict, weights: Dict) -> Dict:
    """
    Evaluate multiple indicators in latest_row using tolerances & weights.
    Returns a dict with:
      - score: weighted score between -1 and +1
      - label: 'Bullish'/'Neutral'/'Bearish'
      - breakdown: dict per indicator -> (signal int, reason string)
    """

    breakdown = {}
    total_w = 0.0
    score = 0.0

    def add_indicator(name, s, w, reason=""):
        nonlocal score, total_w, breakdown
        if s is None or (isinstance(s, float) and np.isnan(s)):
            breakdown[name] = (0, "N/A")
            return
        breakdown[name] = (s, reason)
        score += w * s
        total_w += w

    # EMA cross (use EMA9 - EMA21)
    ema9 = latest_row.get('EMA9', np.nan)
    ema21 = latest_row.get('EMA21', np.nan)
    if pd.notna(ema9) and pd.notna(ema21):
        diff = ema9 - ema21
        tol = tolerances.get('ema', 1.0)
        if diff > tol:
            s = 1
            reason = f"EMA9 > EMA21 by {diff:.3f}"
        elif diff < -tol:
            s = -1
            reason = f"EMA9 < EMA21 by {diff:.3f}"
        else:
            s = 0
            reason = f"EMA diff {diff:.3f} within tol {tol}"
        add_indicator('EMA_cross', s, weights.get('ema', 0.25), reason)
    else:
        add_indicator('EMA_cross', None, weights.get('ema', 0.25), "missing")

    # MACD
    macd = latest_row.get('MACD', np.nan)
    macd_sig = latest_row.get('MACD_signal', np.nan)
    macd_diff = np.nan
    if pd.notna(macd) and pd.notna(macd_sig):
        macd_diff = macd - macd_sig
        mt = tolerances.get('macd', 0.01)
        if macd_diff > mt:
            add_indicator('MACD', 1, weights.get('macd',0.2), f"MACD>{mt:.4f} ({macd_diff:.4f})")
        elif macd_diff < -mt:
            add_indicator('MACD', -1, weights.get('macd',0.2), f"MACD<{ -mt:.4f} ({macd_diff:.4f})")
        else:
            add_indicator('MACD', 0, weights.get('macd',0.2), f"MACD neutral ({macd_diff:.4f})")
    else:
        add_indicator('MACD', None, weights.get('macd',0.2), "missing")

    # RSI
    rsi = latest_row.get('RSI', np.nan)
    if pd.notna(rsi):
        if rsi >= tolerances.get('rsi_overbought', 70):
            add_indicator('RSI', -1, weights.get('rsi',0.15), f"RSI {rsi:.1f} overbought")
        elif rsi <= tolerances.get('rsi_oversold', 30):
            add_indicator('RSI', 1, weights.get('rsi',0.15), f"RSI {rsi:.1f} oversold")
        else:
            add_indicator('RSI', 0, weights.get('rsi',0.15), f"RSI {rsi:.1f} neutral")
    else:
        add_indicator('RSI', None, weights.get('rsi',0.15), "missing")

    # ADX (trend strength - if strong and price above EMA50 treat as bullish)
    adx = latest_row.get('ADX', np.nan)
    ema50 = latest_row.get('EMA50', np.nan)
    price = latest_row.get('price', np.nan)
    if pd.notna(adx):
        if adx >= tolerances.get('adx_strong', 25):
            # determine trend direction using price vs EMA50 if available
            if pd.notna(price) and pd.notna(ema50) and price > ema50:
                add_indicator('ADX', 1, weights.get('adx',0.05), f"ADX {adx:.1f} strong uptrend")
            elif pd.notna(price) and pd.notna(ema50) and price < ema50:
                add_indicator('ADX', -1, weights.get('adx',0.05), f"ADX {adx:.1f} strong downtrend")
            else:
                add_indicator('ADX', 0, weights.get('adx',0.05), f"ADX {adx:.1f} strong but dir unclear")
        else:
            add_indicator('ADX', 0, weights.get('adx',0.05), f"ADX {adx:.1f} weak")
    else:
        add_indicator('ADX', None, weights.get('adx',0.05), "missing")

    # OBV momentum (compare latest to rolling mean)
    obv = latest_row.get('OBV', np.nan)
    if pd.notna(obv):
        # we don't have historical here in latest_row context; treat positive obv as bullish if above previous mean
        add_indicator('OBV', 0, weights.get('obv',0.05), "OBV snapshot (not full time series used)")
    else:
        add_indicator('OBV', None, weights.get('obv',0.05), "missing")

    # On-chain: NUPL
    nupl = latest_row.get('NUPL', np.nan)
    if pd.notna(nupl):
        if nupl >= tolerances.get('nupl_top', 0.5):
            add_indicator('NUPL', -1, weights.get('nupl',0.15), f"NUPL high {nupl:.2f}")
        elif nupl <= tolerances.get('nupl_bottom', 0.25):
            add_indicator('NUPL', 1, weights.get('nupl',0.15), f"NUPL low {nupl:.2f}")
        else:
            add_indicator('NUPL',_
