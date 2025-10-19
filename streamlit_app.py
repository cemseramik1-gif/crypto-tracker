import streamlit as st
import requests
import pandas as pd
import pandas_ta as ta 
from datetime import datetime, timezone, timedelta
import math
import uuid
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# --- 1. CONFIGURATION AND CONSTANTS ---

# Kraken API endpoints
KRAKEN_API_URL = "https://api.kraken.com/0/public/"
BLOCKCYPHER_API_URL = "https://api.blockcypher.com/v1/btc/main"

# Timezone configuration
UTC_OFFSET_HOTHOURS = 11

# Presets for quick configuration
TRADING_PRESETS = {
    "Scalper": {"htf": "15 min", "etf": "5 min", "bars": 200},
    "Day Trader": {"htf": "4 hour", "etf": "1 hour", "bars": 250},
    "Swing Trader": {"htf": "1 day", "etf": "4 hour", "bars": 300},
}

# Alert thresholds
ALERT_CONFIG = {
    "confluence_bullish": 2,  # Alert when score >= 2
    "confluence_bearish": -2,  # Alert when score <= -2
    "rsi_oversold": 30,
    "rsi_overbought": 70,
}

# Indicator Glossary Data
INDICATOR_GLOSSARY = {
    "RSI": {"title": "Relative Strength Index (RSI)", "description": "Measures speed and change of price, identifying overbought (>70) or oversold (<30) conditions. Signal: Bullish > 55, Bearish < 45.", "type": "Momentum"},
    "MACD": {"title": "Moving Average Convergence Divergence (MACD)", "description": "Trend-following momentum indicator. Bullish signal when the MACD Line crosses above the Signal Line.", "type": "Momentum"},
    "StochOsc": {"title": "Stochastic Oscillator", "description": "Compares closing price to its price range. Bullish when %K crosses %D line above 20 (oversold zone).", "type": "Momentum"},
    "CCI": {"title": "Commodity Channel Index (CCI)", "description": "Identifies new trends or extremes. Bullish when CCI crosses above +100; Extreme Overbought > 200.", "type": "Momentum"},
    "WilliamsR": {"title": "Williams %R", "description": "Momentum indicator measuring overbought (0 to -20) and oversold (-80 to -100) levels. Inversely scaled.", "type": "Momentum"},
    "ADX": {"title": "Average Directional Index (ADX)", "description": "Measures the strength of a trend. ADX > 25 indicates a strong trend. +DI and -DI show direction.", "type": "Trend"},
    "EMA": {"title": "Exponential Moving Average (EMA)", "description": "Weighted moving average used to determine trend direction. Crossover of 50/200 EMAs signal major trend shifts.", "type": "Trend"},
    "Ichimoku": {"title": "Ichimoku Cloud", "description": "A comprehensive system defining support, resistance, trend, and momentum. Price above the Cloud (Kumo) is bullish.", "type": "Trend"},
    "VWAP": {"title": "Volume Weighted Average Price (VWAP)", "description": "Average price weighted by volume. Price above VWAP suggests buying pressure; below suggests selling pressure.", "type": "Volume & Flow"},
    "OBV": {"title": "On-Balance Volume (OBV)", "description": "Relates volume change to price change. Rising OBV with rising price confirms trend (Bullish).", "type": "Volume & Flow"},
    "MFI": {"title": "Money Flow Index (MFI)", "description": "Volume-weighted RSI. Measures the rate money flows in/out. Bullish when rising from oversold (<20).", "type": "Volume & Flow"},
    "CMF": {"title": "Chaikin Money Flow (CMF)", "description": "Measures accumulation (>0) or distribution (<0) volume over a period.", "type": "Volume & Flow"},
    "BBANDS": {"title": "Bollinger Bands (BBANDS)", "description": "A volatility indicator. Price touching the Upper Band is Overbought; touching the Lower Band is Oversold. Band width signals volatility.", "type": "Volatility"},
    "ATR": {"title": "Average True Range (ATR)", "description": "A measure of volatility. Used to set stop-losses. Expanding ATR suggests high volatility.", "type": "Volatility"},
    "BlockchainHeight": {"title": "Blockchain Height", "description": "The current number of blocks in the chain. Confirms network health and processing.", "type": "On-Chain"},
    "HashRate": {"title": "Network Hash Rate", "description": "Total combined computational power used to mine Bitcoin. Rising Hash Rate confirms network security and miner confidence (Bullish).", "type": "On-Chain"},
}

# --- 2. UTILITY FUNCTIONS ---

def safe_column_lookup(df, prefix):
    """Safely looks up a column by prefix."""
    try:
        matching_cols = df.columns[df.columns.str.startswith(prefix)]
        if not matching_cols.empty:
            return matching_cols[-1]
        return None
    except Exception:
        return None

def lookup_value(df, prefix, index=-1):
    """Robustly looks up an indicator value by prefix and index."""
    col_name = safe_column_lookup(df, prefix)
    if col_name is None:
        return None
    try:
        val = df[col_name].iloc[index]
        if pd.isna(val) or isinstance(val, (int, float)) and math.isnan(val):
            return None
        return float(val)
    except Exception:
        return None

def get_kraken_interval_code(interval_label):
    """Maps human-readable interval to Kraken API code."""
    interval_map = {
        "1 min": 1, "5 min": 5, "15 min": 15, "30 min": 30, 
        "1 hour": 60, "4 hour": 240, "1 day": 1440, "1 week": 10080,
    }
    return interval_map.get(interval_label, 60)

def get_html_color_class(signal):
    """Maps signal status to Tailwind CSS color classes."""
    if "Bullish" in signal or "Accumulation" in signal or "Oversold" in signal or "Above" in signal or "Up" in signal or "Rising" in signal:
        return "bg-green-100 border-green-400 text-green-800", "text-green-600"
    elif "Bearish" in signal or "Distribution" in signal or "Overbought" in signal or "Below" in signal or "Down" in signal or "Falling" in signal:
        return "bg-red-100 border-red-400 text-red-800", "text-red-600"
    elif "Strong Trend" in signal or "Expansion" in signal or "Extreme" in signal or "Neutral" not in signal:
        return "bg-yellow-100 border-yellow-400 text-yellow-800", "text-yellow-600"
    else:
        return "bg-gray-100 border-gray-400 text-gray-700", "text-gray-500"

def calculate_support_resistance(df, lookback=20):
    """Calculate support and resistance levels."""
    if df.empty or len(df) < lookback:
        return None, None
    
    highs = df['high'].tail(lookback)
    lows = df['low'].tail(lookback)
    
    resistance = highs.quantile(0.95)
    support = lows.quantile(0.05)
    
    return support, resistance

def detect_divergence(df, price_col='close', indicator_col='RSI_14', lookback=14):
    """Detect bullish/bearish divergence between price and indicator."""
    if df.empty or len(df) < lookback * 2:
        return "No Divergence"
    
    try:
        recent_df = df.tail(lookback)
        
        price_trend = recent_df[price_col].iloc[-1] - recent_df[price_col].iloc[0]
        indicator_trend = recent_df[indicator_col].iloc[-1] - recent_df[indicator_col].iloc[0]
        
        # Bullish divergence: price falling, indicator rising
        if price_trend < 0 and indicator_trend > 0:
            return "Bullish Divergence"
        # Bearish divergence: price rising, indicator falling
        elif price_trend > 0 and indicator_trend < 0:
            return "Bearish Divergence"
        else:
            return "No Divergence"
    except:
        return "No Divergence"

# --- 3. API FETCHING FUNCTIONS ---

@st.cache_data(ttl=60)
def fetch_kraken_time():
    """Fetches the official Kraken server time."""
    try:
        response = requests.get(KRAKEN_API_URL + "Time", timeout=5)
        response.raise_for_status()
        server_time_unix = response.json()['result']['unixtime']
        dt_utc = datetime.fromtimestamp(server_time_unix, tz=timezone.utc)
        dt_local = dt_utc.astimezone(timezone(timedelta(hours=UTC_OFFSET_HOTHOURS)))
        return dt_local.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        return f"Error fetching time: {e}"

@st.cache_data(ttl=15)
def fetch_btc_data():
    """Fetches the latest Bitcoin price and volume from Kraken."""
    try:
        response = requests.get(KRAKEN_API_URL + "Ticker?pair=XBTUSD", timeout=5)
        response.raise_for_status()
        data = response.json()['result']
        btc_key = next(k for k in data.keys() if 'XBT' in k)
        price = float(data[btc_key]['c'][0])
        volume = float(data[btc_key]['v'][1])
        return price, volume
    except Exception as e:
        print(f"Could not fetch live Bitcoin data from Kraken: {e}") 
        return None, None

@st.cache_data(ttl=15)
def fetch_historical_data(interval_code, count, label):
    """Fetches OHLC data from Kraken for TA."""
    params = {'pair': 'XBTUSD', 'interval': interval_code} 
    try:
        response = requests.get(KRAKEN_API_URL + "OHLC", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()['result']
        data_key = next(k for k in data.keys() if k != 'last')
        candles = data[data_key]
        df = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
        for col in ['open', 'high', 'low', 'close', 'vwap', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        return df.tail(count).copy()
    except Exception as e:
        st.error(f"Kraken OHLC API reported an error for {label}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_on_chain_data():
    """Fetches Bitcoin blockchain height and hash rate from BlockCypher."""
    try:
        response = requests.get(BLOCKCYPHER_API_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        block_height = data.get('height')
        hash_rate_ghs = data.get('hashrate')
        hash_rate_phs = hash_rate_ghs / 1_000_000 if hash_rate_ghs else None

        return block_height, hash_rate_phs
    except Exception as e:
        st.error(f"Error fetching On-Chain data: {e}")
        return None, None

@st.cache_data(ttl=300)
def run_all_checks():
    """Runs all data feed health checks."""
    checks = [
        {"name": "Kraken Time", "url": KRAKEN_API_URL + "Time"},
        {"name": "BlockCypher On-Chain", "url": BLOCKCYPHER_API_URL},
        {"name": "Kraken Ticker", "url": KRAKEN_API_URL + "Ticker?pair=XBTUSD"},
    ]
    
    results = []
    
    for check in checks:
        result = {"name": check["name"], "status": "FAIL", "latency_ms": "N/A", "error": ""}
        try:
            start_time = datetime.now()
            response = requests.get(check["url"], timeout=5)
            response.raise_for_status()
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds() * 1000
            
            result["status"] = "OK"
            result["latency_ms"] = f"{latency:.0f} ms"
        except requests.exceptions.Timeout:
            result["status"] = "TIMEOUT"
            result["error"] = "Request timed out."
        except Exception as e:
            result["error"] = f"{e}"
            
        results.append(result)
        
    return results

# --- 4. TECHNICAL ANALYSIS LOGIC ---

def calculate_ta(df):
    """Calculates all required TA indicators."""
    if df.empty or len(df) < 200: 
        return df
    
    try: 
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
        df.ta.cci(length=20, append=True)
        df.ta.williamsr(length=14, append=True)
        df.ta.ema(length=[9, 21, 50, 200], append=True)
        df.ta.adx(length=14, append=True)
        df.ta.ichimoku(append=True)
        df.ta.vwap(append=True) 
        df.ta.obv(append=True)
        df.ta.mfi(length=14, append=True)
        df.ta.cmf(length=20, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.atr(length=14, append=True)
    except Exception as e:
        st.warning(f"Technical Analysis calculation failed: {e}")
        return df 

    return df

def get_indicator_signal(df):
    """Calculates detailed signals for all indicators with divergence detection."""
    signals = {
        "Momentum": [],
        "Trend": [],
        "Volume & Flow": [],
        "Volatility": [],
        "Divergence": []
    }
    
    if df.empty or len(df) < 200:
        error_item = {'name': 'Data Error', 'value': 'N/A', 'signal': 'Data Insufficient'}
        for group in ["Momentum", "Trend", "Volume & Flow", "Volatility"]:
            signals[group].append(error_item)
        return signals

    df = calculate_ta(df)
    latest_close = lookup_value(df, 'close')
    
    # --- MOMENTUM ---
    rsi_val = lookup_value(df, 'RSI_14')
    rsi_signal = "Neutral"
    if rsi_val is not None:
        if rsi_val > 70: rsi_signal = "Bearish (Overbought)"
        elif rsi_val < 30: rsi_signal = "Bullish (Oversold)"
        elif rsi_val > 55: rsi_signal = "Bullish (Rising)"
        elif rsi_val < 45: rsi_signal = "Bearish (Falling)"
        signals["Momentum"].append({'name': 'RSI (14)', 'value': f"{rsi_val:,.1f}", 'signal': rsi_signal})
    
    macd_hist = lookup_value(df, 'MACDH_12_26_9')
    macd_signal = "Neutral"
    if macd_hist is not None:
        if macd_hist > 0: macd_signal = "Bullish (Momentum Up)"
        elif macd_hist < 0: macd_signal = "Bearish (Momentum Down)"
        signals["Momentum"].append({'name': 'MACD Histogram', 'value': f"H:{macd_hist:,.2f}", 'signal': macd_signal})
        
    stoch_k = lookup_value(df, 'STOCHk_14_3_3')
    stoch_d = lookup_value(df, 'STOCHd_14_3_3')
    stoch_signal = "Neutral"
    if stoch_k is not None and stoch_d is not None:
        if stoch_k > stoch_d and stoch_d < 20: stoch_signal = "Bullish (Oversold Buy)"
        elif stoch_k < stoch_d and stoch_d > 80: stoch_signal = "Bearish (Overbought Sell)"
        elif stoch_k > stoch_d: stoch_signal = "Bullish (K Crosses D)"
        elif stoch_k < stoch_d: stoch_signal = "Bearish (D Crosses K)"
        signals["Momentum"].append({'name': 'Stoch K/D', 'value': f"K:{stoch_k:,.1f}", 'signal': stoch_signal})

    cci_val = lookup_value(df, 'CCI_20_0.015')
    cci_signal = "Neutral"
    if cci_val is not None:
        if cci_val > 100: cci_signal = "Bullish (New Trend)"
        elif cci_val < -100: cci_signal = "Bearish (New Trend)"
        signals["Momentum"].append({'name': 'CCI (20)', 'value': f"{cci_val:,.1f}", 'signal': cci_signal})

    williamsr_val = lookup_value(df, 'WPR_14')
    williamsr_signal = "Neutral"
    if williamsr_val is not None:
        if williamsr_val > -20: williamsr_signal = "Bearish (Overbought)"
        elif williamsr_val < -80: williamsr_signal = "Bullish (Oversold)"
        signals["Momentum"].append({'name': 'Williams %R', 'value': f"{williamsr_val:,.1f}", 'signal': williamsr_signal})

    # --- TREND ---
    ema200 = lookup_value(df, 'EMA_200')
    ema_signal = "Neutral"
    if ema200 is not None and latest_close is not None:
        if latest_close > ema200: ema_signal = "Bullish (Long-Term Up)"
        elif latest_close < ema200: ema_signal = "Bearish (Long-Term Down)"
        signals["Trend"].append({'name': 'Close vs 200 EMA', 'value': f"C:{latest_close:,.0f}", 'signal': ema_signal})

    ema50 = lookup_value(df, 'EMA_50')
    ema50_prev = lookup_value(df, 'EMA_50', index=-2)
    ema200_prev = lookup_value(df, 'EMA_200', index=-2)
    cross_signal = "Neutral"
    if ema50 is not None and ema200 is not None and ema50_prev is not None and ema200_prev is not None:
        if ema50 > ema200 and ema50_prev < ema200_prev:
            cross_signal = "Bullish (Golden Cross)"
        elif ema50 < ema200 and ema50_prev > ema200_prev:
            cross_signal = "Bearish (Death Cross)"
        signals["Trend"].append({'name': '50/200 EMA Cross', 'value': f"Gap:{(ema50-ema200):,.0f}", 'signal': cross_signal})

    adx_val = lookup_value(df, 'ADX_14')
    adx_di_plus = lookup_value(df, 'DMP_14')
    adx_di_minus = lookup_value(df, 'DMN_14')
    adx_signal = "Neutral Trend"
    if adx_val is not None and adx_di_plus is not None and adx_di_minus is not None:
        if adx_val > 25:
            adx_signal = "Strong Trend"
            if adx_di_plus > adx_di_minus: adx_signal += " (Bullish)"
            else: adx_signal += " (Bearish)"
        signals["Trend"].append({'name': 'ADX (14)', 'value': f"{adx_val:,.1f}", 'signal': adx_signal})

    ichimoku_span_b = lookup_value(df, 'ICSB_26_52_9')
    ichimoku_signal = "Neutral"
    if ichimoku_span_b is not None and latest_close is not None:
        if latest_close > ichimoku_span_b: ichimoku_signal = "Bullish (Above Cloud)"
        elif latest_close < ichimoku_span_b: ichimoku_signal = "Bearish (Below Cloud)"
        signals["Trend"].append({'name': 'Ichimoku Cloud', 'value': f"vs Span B", 'signal': ichimoku_signal})

    # --- VOLUME & FLOW ---
    vwap_val = lookup_value(df, 'VWAP')
    vwap_signal = "Neutral"
    if vwap_val is not None and latest_close is not None:
        if latest_close > vwap_val: vwap_signal = "Bullish (Above VWAP)"
        elif latest_close < vwap_val: vwap_signal = "Bearish (Below VWAP)"
        signals["Volume & Flow"].append({'name': 'VWAP', 'value': f"${vwap_val:,.0f}", 'signal': vwap_signal})

    cmf_val = lookup_value(df, 'CMF_20')
    cmf_signal = "Neutral"
    if cmf_val is not None:
        if cmf_val > 0.05: cmf_signal = "Bullish (Accumulation)"
        elif cmf_val < -0.05: cmf_signal = "Bearish (Distribution)"
        signals["Volume & Flow"].append({'name': 'Chaikin Money Flow', 'value': f"{cmf_val:,.2f}", 'signal': cmf_signal})

    mfi_val = lookup_value(df, 'MFI_14')
    mfi_signal = "Neutral"
    if mfi_val is not None:
        if mfi_val > 80: mfi_signal = "Bearish (Extreme Overbought)"
        elif mfi_val < 20: mfi_signal = "Bullish (Extreme Oversold)"
        signals["Volume & Flow"].append({'name': 'MFI (14)', 'value': f"{mfi_val:,.1f}", 'signal': mfi_signal})

    obv_current = lookup_value(df, 'OBV', index=-1)
    obv_prev = lookup_value(df, 'OBV', index=-2)
    obv_signal = "Neutral"
    if obv_current is not None and obv_prev is not None:
        if obv_current > obv_prev: obv_signal = "Bullish (Volume In)"
        elif obv_current < obv_prev: obv_signal = "Bearish (Volume Out)"
        signals["Volume & Flow"].append({'name': 'OBV', 'value': f"Δ:{(obv_current-obv_prev):,.0f}", 'signal': obv_signal})

    # --- VOLATILITY ---
    upper_band = lookup_value(df, 'BBU_20_2')
    lower_band = lookup_value(df, 'BBL_20_2')
    bb_signal = "Neutral"
    if upper_band is not None and lower_band is not None and latest_close is not None:
        if latest_close > upper_band: bb_signal = "Bearish (Upper Band)"
        elif latest_close < lower_band: bb_signal = "Bullish (Lower Band)"
        signals["Volatility"].append({'name': 'Bollinger Bands', 'value': "Band Test", 'signal': bb_signal})

    atr_val = lookup_value(df, 'ATR_14')
    atr_signal = "Neutral"
    if atr_val is not None:
        atr_col = safe_column_lookup(df, 'ATR_14')
        if atr_col and len(df) >= 6:
            atr_avg_5 = df[atr_col].iloc[-6:-1].mean()
            if atr_val > atr_avg_5 * 1.1: atr_signal = "Expansion (High Vol)"
            elif atr_val < atr_avg_5 * 0.9: atr_signal = "Contraction (Low Vol)"
        signals["Volatility"].append({'name': 'ATR (14)', 'value': f"${atr_val:,.2f}", 'signal': atr_signal})
    
    # --- DIVERGENCE DETECTION ---
    rsi_col = safe_column_lookup(df, 'RSI_14')
    macd_col = safe_column_lookup(df, 'MACDH_12_26_9')
    
    if rsi_col:
        rsi_div = detect_divergence(df, 'close', rsi_col)
        if "Divergence" in rsi_div:
            signals["Divergence"].append({'name': 'RSI/Price', 'value': 'Detected', 'signal': rsi_div})
    
    if macd_col:
        macd_div = detect_divergence(df, 'close', macd_col)
        if "Divergence" in macd_div:
            signals["Divergence"].append({'name': 'MACD/Price', 'value': 'Detected', 'signal': macd_div})
    
    if not signals["Divergence"]:
        signals["Divergence"].append({'name': 'Status', 'value': 'None', 'signal': 'No Divergence'})
        
    return signals

def get_confluence_score(htf_df, etf_df, htf_label, etf_label):
    """Calculates confluence score with enhanced logic."""
    if htf_df.empty or etf_df.empty or len(htf_df) < 200 or len(etf_df) < 200:
        return 0, "N/A", ["Insufficient data"], "bg-gray-400"

    calculate_ta(htf_df) 
    calculate_ta(etf_df) 

    score = 0
    factors = []

    # HTF TREND (200 EMA)
    htf_close = lookup_value(htf_df, 'close')
    htf_ema200 = lookup_value(htf_df, 'EMA_200')
    if htf_ema200 is not None and htf_close is not None:
        if htf_close > htf_ema200:
            score += 1
            factors.append(f"**+1 | HTF Trend:** Bullish ({htf_label} > 200 EMA)")
        elif htf_close < htf_ema200:
            score -= 1
            factors.append(f"**-1 | HTF Trend:** Bearish ({htf_label} < 200 EMA)")
    
    # ETF MOMENTUM (RSI)
    etf_rsi = lookup_value(etf_df, 'RSI_14')
    if etf_rsi is not None:
        if etf_rsi > 55:
            score += 1
            factors.append(f"**+1 | ETF Momentum:** Bullish ({etf_label} RSI > 55)")
        elif etf_rsi < 45:
            score -= 1
            factors.append(f"**-1 | ETF Momentum:** Bearish ({etf_label} RSI < 45)")

    # ETF FLOW (VWAP)
    etf_close = lookup_value(etf_df, 'close')
    etf_vwap = lookup_value(etf_df, 'VWAP')
    if etf_vwap is not None and etf_close is not None:
        if etf_close > etf_vwap:
            score += 1
            factors.append(f"**+1 | ETF Flow:** Bullish ({etf_label} > VWAP)")
        elif etf_close < etf_vwap:
            score -= 1
            factors.append(f"**-1 | ETF Flow:** Bearish ({etf_label} < VWAP)")
            
    if score >= 2:
        signal = "Strong Bullish Confluence"
        color = "bg-green-600"
    elif score == 1:
        signal = "Weak Bullish Confluence"
        color = "bg-green-400"
    elif score <= -2:
        signal = "Strong Bearish Confluence"
        color = "bg-red-600"
    elif score == -1:
        signal = "Weak Bearish Confluence"
        color = "bg-red-400"
    else:
        signal = "Neutral/Conflicting"
        color = "bg-yellow-500"

    return score, signal, factors, color

def calculate_risk_management(df, entry_price=None):
    """Calculate position sizing and stop-loss recommendations."""
    if df.empty or len(df) < 50:
        return None
    
    atr_val = lookup_value(df, 'ATR_14')
    lower_band = lookup_value(df, 'BBL_20_2')
    upper_band = lookup_value(df, 'BBU_20_2')
    current_price = lookup_value(df, 'close')
    
    if entry_price is None:
        entry_price = current_price
    
    risk_metrics = {}
    
    if atr_val and entry_price:
        # Stop loss: 2x ATR below entry
        risk_metrics['stop_loss'] = entry_price - (2 * atr_val)
        # Take profit: 3x ATR above entry (1.5:1 RR)
        risk_metrics['take_profit'] = entry_price + (3 * atr_val)
        risk_metrics['risk_reward'] = 1.5
        risk_metrics['atr'] = atr_val
    
    if lower_band:
        risk_metrics['support_bb'] = lower_band
    
    if upper_band:
        risk_metrics['resistance_bb'] = upper_band
    
    # Calculate position size for 2% risk
    if atr_val and entry_price:
        risk_per_unit = 2 * atr_val
        risk_metrics['risk_per_unit'] = risk_per_unit
    
    return risk_metrics

# --- 5. CHART GENERATION ---

def create_interactive_chart(df, title="BTC/USD Chart"):
    """Creates an interactive Plotly candlestick chart with indicators."""
    if df.empty or len(df) < 50:
        return None
    
    df = calculate_ta(df)
    
    # Create subplots: price chart + volume + RSI
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(title, 'Volume', 'RSI')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    # EMAs
    for ema in [9, 21, 50, 200]:
        ema_col = safe_column_lookup(df, f'EMA_{ema}')
        if ema_col:
            colors = {9: 'orange', 21: 'blue', 50: 'purple', 200: 'red'}
            fig.add_trace(
                go.Scatter(
                    x=df['datetime'],
                    y=df[ema_col],
                    name=f'EMA {ema}',
                    line=dict(color=colors.get(ema, 'gray'), width=1)
                ),
                row=1, col=1
            )
    
    # Bollinger Bands
    bb_upper = safe_column_lookup(df, 'BBU_20_2')
    bb_lower = safe_column_lookup(df, 'BBL_20_2')
    bb_mid = safe_column_lookup(df, 'BBM_20_2')
    
    if bb_upper and bb_lower and bb_mid:
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df[bb_upper],
                name='BB Upper',
                line=dict(color='rgba(128,128,128,0.3)', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df[bb_lower],
                name='BB Lower',
                line=dict(color='rgba(128,128,128,0.3)', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # VWAP
    vwap_col = safe_column_lookup(df, 'VWAP')
    if vwap_col:
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df[vwap_col],
                name='VWAP',
                line=dict(color='cyan', width=2, dash='dot')
            ),
            row=1, col=1
        )
    
    # Volume bars
    colors = ['red' if row['close'] < row['open'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df['datetime'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # RSI
    rsi_col = safe_column_lookup(df, 'RSI_14')
    if rsi_col:
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df[rsi_col],
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=3, col=1
        )
        
        # RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    return fig

def create_confluence_history_chart(history):
    """Creates a line chart showing confluence score over time."""
    if not history:
        return None
    
    df_history = pd.DataFrame(history)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df_history['timestamp'],
            y=df_history['score'],
            mode='lines+markers',
            name='Confluence Score',
            line=dict(color='blue', width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(0,100,255,0.1)'
        )
    )
    
    # Reference lines
    fig.add_hline(y=2, line_dash="dash", line_color="green", annotation_text="Strong Bullish")
    fig.add_hline(y=-2, line_dash="dash", line_color="red", annotation_text="Strong Bearish")
    fig.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Neutral")
    
    fig.update_layout(
        title="Confluence Score History (Last 24 Hours)",
        xaxis_title="Time",
        yaxis_title="Score",
        height=300,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

# --- 6. ALERT SYSTEM ---

def check_alerts(df, confluence_score):
    """Check for alert conditions and return triggered alerts."""
    alerts = []
    
    # Confluence alerts
    if confluence_score >= ALERT_CONFIG['confluence_bullish']:
        alerts.append({
            'type': 'success',
            'message': f'🚀 Strong Bullish Confluence: Score {confluence_score}/3'
        })
    elif confluence_score <= ALERT_CONFIG['confluence_bearish']:
        alerts.append({
            'type': 'error',
            'message': f'🔻 Strong Bearish Confluence: Score {confluence_score}/3'
        })
    
    # RSI alerts
    rsi_val = lookup_value(df, 'RSI_14')
    if rsi_val:
        if rsi_val <= ALERT_CONFIG['rsi_oversold']:
            alerts.append({
                'type': 'info',
                'message': f'📉 RSI Oversold: {rsi_val:.1f} (Potential reversal)'
            })
        elif rsi_val >= ALERT_CONFIG['rsi_overbought']:
            alerts.append({
                'type': 'warning',
                'message': f'📈 RSI Overbought: {rsi_val:.1f} (Potential pullback)'
            })
    
    # Divergence alerts
    rsi_col = safe_column_lookup(df, 'RSI_14')
    if rsi_col:
        divergence = detect_divergence(df, 'close', rsi_col)
        if "Bullish" in divergence:
            alerts.append({
                'type': 'success',
                'message': f'🔄 {divergence} detected - Potential bottom'
            })
        elif "Bearish" in divergence:
            alerts.append({
                'type': 'warning',
                'message': f'🔄 {divergence} detected - Potential top'
            })
    
    return alerts

# --- 7. SESSION STATE MANAGEMENT ---

def init_session_state():
    """Initialize session state variables."""
    if 'confluence_history' not in st.session_state:
        st.session_state['confluence_history'] = []
    if 'dark_mode' not in st.session_state:
        st.session_state['dark_mode'] = False
    if 'last_update' not in st.session_state:
        st.session_state['last_update'] = datetime.now()
    if 'alert_sound' not in st.session_state:
        st.session_state['alert_sound'] = True

def store_confluence_score(score, signal):
    """Store confluence score in history with 24-hour retention."""
    current_time = datetime.now()
    
    # Add new score
    st.session_state['confluence_history'].append({
        'timestamp': current_time,
        'score': score,
        'signal': signal
    })
    
    # Remove entries older than 24 hours
    cutoff_time = current_time - timedelta(hours=24)
    st.session_state['confluence_history'] = [
        entry for entry in st.session_state['confluence_history']
        if entry['timestamp'] > cutoff_time
    ]

# --- 8. STREAMLIT UI ---

st.set_page_config(
    page_title="Enhanced Crypto Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_session_state()

# Apply styling
st.markdown(
    """
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
    .stApp { background-color: #f8f9fa; }
    h1 { color: #1e3a8a; }
    .stButton>button { border-radius: 0.5rem; border: 1px solid #3b82f6; }
    
    .ta-tile {
        padding: 0.5rem;
        border-radius: 0.75rem;
        border-width: 1px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .tile-value {
        font-size: 1.25rem;
        font-weight: 700;
        line-height: 1.5;
        text-align: center;
    }
    .tile-signal {
        font-size: 0.9rem;
        font-weight: 600;
        text-align: center;
        padding-top: 0.25rem;
    }
    .tile-name {
        font-size: 0.8rem;
        font-weight: 500;
        text-align: center;
        opacity: 0.8;
    }
    .confluence-box {
        padding: 20px;
        border-radius: 12px;
        color: white;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .confluence-header {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .confluence-score {
        font-size: 3rem;
        font-weight: 900;
        line-height: 1;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .signal-summary {
        padding: 1.5rem;
        border-radius: 1rem;
        border: 3px solid;
        margin-bottom: 1rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .risk-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        color: white;
        margin: 0.5rem 0;
    }
    
    @media (max-width: 768px) {
        .ta-tile { min-height: 100px; }
        .tile-value { font-size: 1rem; }
        .tile-signal { font-size: 0.8rem; }
        .confluence-score { font-size: 2rem; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://placehold.co/150x50/1e3a8a/ffffff?text=XBT+Pro")
    st.markdown("## 📊 Configuration")
    
    # Preset selector
    st.markdown("### Quick Presets")
    preset_cols = st.columns(3)
    for idx, (preset_name, preset_config) in enumerate(TRADING_PRESETS.items()):
        if preset_cols[idx].button(preset_name, use_container_width=True):
            st.session_state['selected_htf'] = preset_config['htf']
            st.session_state['selected_etf'] = preset_config['etf']
            st.session_state['bar_count'] = preset_config['bars']
    
    st.markdown("---")
    st.markdown("### Custom Settings")
    
    timeframe_options = ["1 min", "5 min", "15 min", "30 min", "1 hour", "4 hour", "1 day"]
    
    selected_htf_label = st.selectbox(
        "High Time Frame (Context)",
        options=timeframe_options,
        index=timeframe_options.index(st.session_state.get('selected_htf', '4 hour'))
    )
    
    selected_etf_label = st.selectbox(
        "Entry Time Frame (Signal)",
        options=timeframe_options,
        index=timeframe_options.index(st.session_state.get('selected_etf', '1 hour'))
    )
    
    bar_count = st.slider(
        "Historical Bars",
        min_value=100,
        max_value=400,
        value=st.session_state.get('bar_count', 250),
        step=50
    )
    
    st.markdown("---")
    st.markdown("### Alert Settings")
    
    alert_enabled = st.checkbox("Enable Alerts", value=True)
    
    if alert_enabled:
        ALERT_CONFIG['confluence_bullish'] = st.slider("Bullish Confluence Threshold", 1, 3, 2)
        ALERT_CONFIG['confluence_bearish'] = st.slider("Bearish Confluence Threshold", -3, -1, -2)
    
    st.markdown("---")
    
    # Fetch data
    htf_code = get_kraken_interval_code(selected_htf_label)
    etf_code = get_kraken_interval_code(selected_etf_label)
    
    with st.spinner('Fetching data...'):
        htf_df = fetch_historical_data(htf_code, bar_count, selected_htf_label)
        etf_df = fetch_historical_data(etf_code, bar_count, selected_etf_label)
    
    st.markdown("### System Health")
    col_health1, col_health2 = st.columns(2)
    
    if col_health1.button("🔍 Check APIs", use_container_width=True):
        with st.spinner('Running checks...'):
            st.session_state['health_check_result'] = run_all_checks()
    
    if col_health2.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    if 'health_check_result' in st.session_state:
        for result in st.session_state['health_check_result']:
            status_emoji = "✅" if result['status'] == "OK" else "❌"
            st.caption(f"{status_emoji} {result['name']}: {result['latency_ms']}")

# --- MAIN DASHBOARD ---

st.title("🚀 Enhanced Crypto TA Dashboard")

# Calculate confluence and store in history
score, signal, factors, color_class = get_confluence_score(
    htf_df.copy(), 
    etf_df.copy(), 
    selected_htf_label, 
    selected_etf_label
)
store_confluence_score(score, signal)

# Check for alerts
if alert_enabled:
    alerts = check_alerts(etf_df.copy(), score)
    for alert in alerts:
        if alert['type'] == 'success':
            st.success(alert['message'])
        elif alert['type'] == 'error':
            st.error(alert['message'])
        elif alert['type'] == 'warning':
            st.warning(alert['message'])
        else:
            st.info(alert['message'])

st.markdown("---")

# --- SIGNAL SUMMARY CARD ---
st.header("📍 Trade Signal Summary")

summary_color = "border-green-500 bg-green-50" if score >= 2 else \
                "border-red-500 bg-red-50" if score <= -2 else \
                "border-yellow-500 bg-yellow-50"

# Determine action
if score >= 2:
    action = "🟢 STRONG BUY"
    confidence = "High"
    reasoning = f"All 3 pillars aligned bullish on {selected_etf_label}"
elif score == 1:
    action = "🟡 WEAK BUY"
    confidence = "Low"
    reasoning = f"Mixed signals on {selected_etf_label} - wait for confirmation"
elif score <= -2:
    action = "🔴 STRONG SELL"
    confidence = "High"
    reasoning = f"All 3 pillars aligned bearish on {selected_etf_label}"
elif score == -1:
    action = "🟡 WEAK SELL"
    confidence = "Low"
    reasoning = f"Mixed signals on {selected_etf_label} - wait for confirmation"
else:
    action = "⚪ HOLD"
    confidence = "N/A"
    reasoning = f"No clear direction on {selected_etf_label}"

st.markdown(f"""
<div class="signal-summary {summary_color}">
    <div style="font-size: 2rem;">{action}</div>
    <div style="font-size: 1rem; margin-top: 0.5rem;">Confidence: {confidence} | Score: {score}/3</div>
    <div style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.8;">{reasoning}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- LIVE METRICS ---
st.header("1️⃣ Live Market Data")

col1, col2, col3, col4, col5 = st.columns(5)

live_price, live_volume = fetch_btc_data()
block_height, hash_rate_phs = fetch_on_chain_data()
support, resistance = calculate_support_resistance(etf_df)

with col1:
    st.metric("BTC Price", f"${live_price:,.2f}" if live_price else "N/A")

with col2:
    st.metric("24h Volume", f"{live_volume:,.0f} BTC" if live_volume else "N/A")

with col3:
    st.metric("Block Height", f"{block_height:,.0f}" if block_height else "N/A")

with col4:
    st.metric("Hash Rate", f"{hash_rate_phs:,.0f} PH/s" if hash_rate_phs else "N/A")

with col5:
    last_update = datetime.now().strftime("%H:%M:%S")
    st.metric("Last Update", last_update)

# Support/Resistance levels
if support and resistance:
    col_sr1, col_sr2 = st.columns(2)
    col_sr1.metric("📊 Support Level", f"${support:,.2f}")
    col_sr2.metric("📊 Resistance Level", f"${resistance:,.2f}")

st.markdown("---")

# --- INTERACTIVE CHARTS ---
st.header("2️⃣ Interactive Price Charts")

tab1, tab2 = st.tabs([f"📈 {selected_etf_label} Chart", f"📊 {selected_htf_label} Chart"])

with tab1:
    chart_etf = create_interactive_chart(etf_df.copy(), f"BTC/USD - {selected_etf_label}")
    if chart_etf:
        st.plotly_chart(chart_etf, use_container_width=True)

with tab2:
    chart_htf = create_interactive_chart(htf_df.copy(), f"BTC/USD - {selected_htf_label}")
    if chart_htf:
        st.plotly_chart(chart_htf, use_container_width=True)

st.markdown("---")

# --- CONFLUENCE SECTION ---
st.header("3️⃣ Multi-Time Frame Confluence Analysis")

col_conf1, col_conf2 = st.columns([1, 2])

with col_conf1:
    st.markdown(f"""
        <div class="confluence-box {color_class}">
            <p class="confluence-header">CONFLUENCE SCORE</p>
            <p class="confluence-score">{score}/3</p>
            <p style="font-size: 1.1rem; font-weight: 600;">{signal}</p>
        </div>
    """, unsafe_allow_html=True)

with col_conf2:
    st.subheader("Pillar Breakdown")
    for factor in factors:
        st.markdown(f"- {factor}")

# Confluence history chart
if len(st.session_state['confluence_history']) > 1:
    st.subheader("Confluence Score Trend")
    history_chart = create_confluence_history_chart(st.session_state['confluence_history'])
    if history_chart:
        st.plotly_chart(history_chart, use_container_width=True)

st.markdown("---")

# --- RISK MANAGEMENT ---
st.header("4️⃣ Risk Management Calculator")

risk_metrics = calculate_risk_management(etf_df.copy())

if risk_metrics:
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    
    with col_r1:
        st.markdown('<div class="risk-box">', unsafe_allow_html=True)
        st.markdown(f"**Stop Loss (2x ATR)**")
        st.markdown(f"### ${risk_metrics.get('stop_loss', 0):,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_r2:
        st.markdown('<div class="risk-box">', unsafe_allow_html=True)
        st.markdown(f"**Take Profit (3x ATR)**")
        st.markdown(f"### ${risk_metrics.get('take_profit', 0):,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_r3:
        st.markdown('<div class="risk-box">', unsafe_allow_html=True)
        st.markdown(f"**Risk/Reward Ratio**")
        st.markdown(f"### {risk_metrics.get('risk_reward', 0):.1f}:1")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_r4:
        st.markdown('<div class="risk-box">', unsafe_allow_html=True)
        st.markdown(f"**Current ATR (14)**")
        st.markdown(f"### ${risk_metrics.get('atr', 0):,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.info("💡 **Tip**: Use 2% of your portfolio as risk per trade. Position size = (Portfolio × 0.02) / Risk per unit")

st.markdown("---")

# --- TA SIGNAL MATRIX ---
st.header("5️⃣ Technical Analysis Matrix")

ta_signals_grouped = get_indicator_signal(etf_df.copy())

for group_name, signals in ta_signals_grouped.items():
    if group_name == "Divergence" and signals[0]['signal'] == "No Divergence":
        continue
        
    st.subheader(f"📊 {group_name}")
    
    num_signals = len(signals)
    num_columns = min(num_signals, 5)
    
    for i in range(0, num_signals, 5):
        row_signals = signals[i:i+5]
        cols = st.columns(len(row_signals))
        
        for j, item in enumerate(row_signals):
            tile_class, value_class = get_html_color_class(item['signal'])
            
            cols[j].markdown(f"""
                <div class='ta-tile {tile_class}'>
                    <p class='tile-name'>{item['name']}</p>
                    <p class='tile-value {value_class}'>{item['value']}</p>
                    <p class='tile-signal'>{item['signal']}</p>
                </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# --- MARKET CONTEXT ---
st.header("6️⃣ Market Context Panel")

col_ctx1, col_ctx2, col_ctx3 = st.columns(3)

with col_ctx1:
    st.markdown("### 📍 Key Levels")
    if support and resistance:
        st.write(f"**Support:** ${support:,.2f}")
        st.write(f"**Resistance:** ${resistance:,.2f}")
        if live_price:
            dist_support = ((live_price - support) / support) * 100
            dist_resistance = ((resistance - live_price) / live_price) * 100
            st.write(f"**Distance to Support:** {dist_support:.2f}%")
            st.write(f"**Distance to Resistance:** {dist_resistance:.2f}%")

with col_ctx2:
    st.markdown("### 🎯 Volatility Status")
    atr_val = lookup_value(etf_df, 'ATR_14')
    if atr_val and live_price:
        volatility_pct = (atr_val / live_price) * 100
        st.write(f"**ATR:** ${atr_val:,.2f}")
        st.write(f"**Volatility:** {volatility_pct:.2f}%")
        if volatility_pct > 3:
            st.write("**Status:** 🔴 High Volatility")
        elif volatility_pct < 1.5:
            st.write("**Status:** 🟢 Low Volatility")
        else:
            st.write("**Status:** 🟡 Normal Volatility")

with col_ctx3:
    st.markdown("### 📊 Trend Strength")
    adx_val = lookup_value(etf_df, 'ADX_14')
    if adx_val:
        st.write(f"**ADX:** {adx_val:.1f}")
        if adx_val > 40:
            st.write("**Strength:** 🟢 Very Strong Trend")
        elif adx_val > 25:
            st.write("**Strength:** 🟡 Strong Trend")
        else:
            st.write("**Strength:** 🔴 Weak/No Trend")

st.markdown("---")

# --- GLOSSARY ---
with st.expander("📚 Technical Indicator Glossary"):
    glossary_grouped = {}
    for key, data in INDICATOR_GLOSSARY.items():
        group_name = data['type']
        if group_name not in glossary_grouped:
            glossary_grouped[group_name] = []
        glossary_grouped[group_name].append(data)
    
    for group_name, items in glossary_grouped.items():
        st.markdown(f"### {group_name}")
        for item in items:
            st.markdown(f"**{item['title']}**")
            st.caption(item['description'])
            st.markdown("---")

# --- AUTO-REFRESH ---
# Add auto-refresh functionality
import time

if 'auto_refresh' not in st.session_state:
    st.session_state['auto_refresh'] = False

with st.sidebar:
    st.markdown("---")
    st.markdown("### ⚡ Auto-Refresh")
    auto_refresh = st.toggle("Enable Auto-Refresh (30s)", value=st.session_state.get('auto_refresh', False))
    st.session_state['auto_refresh'] = auto_refresh
    
    if auto_refresh:
        st.info("Dashboard will refresh in 30 seconds")
        time.sleep(30)
        st.rerun()

# Footer
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
col_f1.caption(f"🕐 Last Update: {datetime.now().strftime('%H:%M:%S')}")
col_f2.caption(f"📊 Data Sources: Kraken API, BlockCypher")
col_f3.caption(f"⚙️ Timeframes: HTF {selected_htf_label} | ETF {selected_etf_label}")

# Export functionality
st.markdown("---")
st.markdown("### 📥 Export Data")

export_col1, export_col2, export_col3 = st.columns(3)

# Export TA signals
with export_col1:
    if st.button("Export TA Signals (CSV)", use_container_width=True):
        all_signals = []
        for group_name, signals in ta_signals_grouped.items():
            for signal in signals:
                all_signals.append({
                    'Category': group_name,
                    'Indicator': signal['name'],
                    'Value': signal['value'],
                    'Signal': signal['signal'],
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
        
        df_export = pd.DataFrame(all_signals)
        csv = df_export.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"ta_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Export confluence history
with export_col2:
    if st.button("Export Confluence History", use_container_width=True):
        if st.session_state['confluence_history']:
            df_history = pd.DataFrame(st.session_state['confluence_history'])
            df_history['timestamp'] = df_history['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            csv = df_history.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"confluence_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No history data available yet")

# Export OHLC data
with export_col3:
    if st.button("Export OHLC Data", use_container_width=True):
        df_ohlc = etf_df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
        df_ohlc['datetime'] = df_ohlc['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        csv = df_ohlc.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"ohlc_{selected_etf_label.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
