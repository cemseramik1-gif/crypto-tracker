import streamlit as st
import requests
import pandas as pd
import pandas_ta as ta 
from datetime import datetime, timezone, timedelta
import math
import uuid
import numpy as np

# --- 1. CONFIGURATION AND CONSTANTS ---

# Kraken API endpoints
KRAKEN_API_URL = "https://api.kraken.com/0/public/"
# Using BlockCypher for basic on-chain metrics (Block Height, Hash Rate)
BLOCKCYPHER_API_URL = "https://api.blockcypher.com/v1/btc/main"

# Timezone configuration (e.g., AEST/AEDT is UTC+10 or UTC+11)
UTC_OFFSET_HOTHOURS = 11

# Indicator Glossary Data (Expanded to include advanced On-Chain placeholders)
INDICATOR_GLOSSARY = {
    # --- MOMENTUM ---
    "RSI": {"title": "Relative Strength Index (RSI)", "description": "Measures speed and change of price, identifying overbought (>70) or oversold (<30) conditions. Signal: Bullish > 55, Bearish < 45.", "type": "Momentum"},
    "MACD": {"title": "Moving Average Convergence Divergence (MACD)", "description": "Trend-following momentum indicator. Bullish signal when the MACD Line crosses above the Signal Line.", "type": "Momentum"},
    "StochOsc": {"title": "Stochastic Oscillator", "description": "Compares closing price to its price range. Bullish when %K crosses %D line above 20 (oversold zone).", "type": "Momentum"},
    "CCI": {"title": "Commodity Channel Index (CCI)", "description": "Identifies new trends or extremes. Bullish when CCI crosses above +100; Extreme Overbought > 200.", "type": "Momentum"},
    "WilliamsR": {"title": "Williams %R", "description": "Momentum indicator measuring overbought (0 to -20) and oversold (-80 to -100) levels. Inversely scaled.", "type": "Momentum"},
    
    # --- TREND ---
    "ADX": {"title": "Average Directional Index (ADX)", "description": "Measures the strength of a trend. ADX > 25 indicates a strong trend. +DI and -DI show direction.", "type": "Trend"},
    "EMA": {"title": "Exponential Moving Average (EMA)", "description": "Weighted moving average used to determine trend direction. Crossover of 50/200 EMAs signal major trend shifts.", "type": "Trend"},
    "Ichimoku": {"title": "Ichimoku Cloud", "description": "A comprehensive system defining support, resistance, trend, and momentum. Price above the Cloud (Kumo) is bullish.", "type": "Trend"},

    # --- VOLUME & FLOW ---
    "VWAP": {"title": "Volume Weighted Average Price (VWAP)", "description": "Average price weighted by volume. Price above VWAP suggests buying pressure; below suggests selling pressure.", "type": "Volume & Flow"},
    "OBV": {"title": "On-Balance Volume (OBV)", "description": "Relates volume change to price change. Rising OBV with rising price confirms trend (Bullish).", "type": "Volume & Flow"},
    "MFI": {"title": "Money Flow Index (MFI)", "description": "Volume-weighted RSI. Measures the rate money flows in/out. Bullish when rising from oversold (<20).", "type": "Volume & Flow"},
    "CMF": {"title": "Chaikin Money Flow (CMF)", "description": "Measures accumulation (>0) or distribution (<0) volume over a period.", "type": "Volume & Flow"},
    
    # --- VOLATILITY ---
    "BBANDS": {"title": "Bollinger Bands (BBANDS)", "description": "A volatility indicator. Price touching the Upper Band is Overbought; touching the Lower Band is Oversold. Band width signals volatility.", "type": "Volatility"},
    "ATR": {"title": "Average True Range (ATR)", "description": "A measure of volatility. Used to set stop-losses. Expanding ATR suggests high volatility.", "type": "Volatility"},
    
    # --- ON-CHAIN (EXPANDED) ---
    "BlockchainHeight": {"title": "Blockchain Height", "description": "The current number of blocks in the chain. Confirms network health and processing.", "type": "On-Chain"},
    "HashRate": {"title": "Network Hash Rate", "description": "Total combined computational power used to mine Bitcoin. Rising Hash Rate confirms network security and miner confidence (Bullish).", "type": "On-Chain"},
    "MVRV": {"title": "Market Value to Realized Value (MVRV)", "description": "ADVANCED: Ratio of market cap vs. the price coins were last moved at. Used to find market tops and bottoms. (Needs dedicated API)", "type": "On-Chain (Placeholder)"},
    "ExchangeNetFlow": {"title": "Exchange Net Flow", "description": "ADVANCED: Net amount of coins moving in or out of exchanges. Large positive inflow is Bearish (selling pressure). (Needs dedicated API)", "type": "On-Chain (Placeholder)"},
}

# --- 2. UTILITY FUNCTIONS (Enhanced for Robustness) ---

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
    """Maps signal status to Tailwind CSS color classes for the tiles."""
    if "Bullish" in signal or "Accumulation" in signal or "Oversold" in signal or "Above" in signal or "Up" in signal or "Rising" in signal:
        return "bg-green-100 border-green-400 text-green-800", "text-green-600"
    elif "Bearish" in signal or "Distribution" in signal or "Overbought" in signal or "Below" in signal or "Down" in signal or "Falling" in signal:
        return "bg-red-100 border-red-400 text-red-800", "text-red-600"
    elif "Strong Trend" in signal or "Expansion" in signal or "Extreme" in signal or "Neutral" not in signal:
        return "bg-yellow-100 border-yellow-400 text-yellow-800", "text-yellow-600"
    else:
        return "bg-gray-100 border-gray-400 text-gray-700", "text-gray-500"

# --- 3. API FETCHING FUNCTIONS (Including On-Chain) ---

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

@st.cache_data(ttl=120)
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
        # Hash rate is often returned in Gh/s or Th/s. Convert to PetaHashes/s (PH/s) for readability
        hash_rate_ghs = data.get('hashrate') # BlockCypher returns Gh/s
        
        # 1 PH/s = 1,000,000 Gh/s
        hash_rate_phs = hash_rate_ghs / 1_000_000 if hash_rate_ghs else None

        return block_height, hash_rate_phs
    except Exception as e:
        st.error(f"Error fetching On-Chain data: {e}")
        return None, None
    
# Placeholder health check functions (using the provided logic)
def check_http_status(url):
    start_time = datetime.now()
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    end_time = datetime.now()
    latency = (end_time - start_time).total_seconds() * 1000
    return "OK", latency, response.json()

def check_kraken_time(url):
    start_time = datetime.now()
    response = requests.get(url, timeout=5)
    response_json = response.json()
    end_time = datetime.now()
    latency = (end_time - start_time).total_seconds() * 1000
    if 'result' in response_json and 'unixtime' in response_json['result']:
        return "OK (Kraken Time OK)", latency, response_json
    else:
        raise Exception("Time field missing in Kraken response.")

@st.cache_data(ttl=300)
def run_all_checks():
    """Runs all data feed health checks."""
    checks = [
        {"name": "Kraken Time (Latency Check)", "url": KRAKEN_API_URL + "Time", "checker": check_kraken_time},
        {"name": "BlockCypher (On-Chain Data)", "url": BLOCKCYPHER_API_URL, "checker": check_http_status},
        {"name": "Kraken Ticker (Live Price)", "url": KRAKEN_API_URL + "Ticker?pair=XBTUSD", "checker": check_http_status},
    ]
    
    results = []
    
    for check in checks:
        result = {"name": check["name"], "status": "FAIL", "latency_ms": "N/A", "error": ""}
        try:
            status, latency, _ = check["checker"](check["url"])
            result["status"] = status
            result["latency_ms"] = f"{latency:.0f} ms"
            result["error"] = ""
        except requests.exceptions.Timeout:
            result["status"] = "TIMEOUT"
            result["error"] = "Request timed out."
        except requests.exceptions.RequestException as e:
            result["error"] = f"HTTP Error: {e}"
        except Exception as e:
            result["error"] = f"Internal Check Error: {e}"
            
        results.append(result)
        
    return results

# --- 4. TECHNICAL ANALYSIS LOGIC (FIXED & MTFA) ---

def calculate_ta(df):
    """
    Explicitly calculates all required TA indicators.
    """
    if df.empty or len(df) < 200: 
        return df
    
    try: 
        # 1. Momentum
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
        df.ta.cci(length=20, append=True)
        df.ta.williamsr(length=14, append=True) # Fixed/Verified: pandas-ta uses williamsr

        # 2. Trend
        df.ta.ema(length=[9, 21, 50, 200], append=True)
        df.ta.adx(length=14, append=True)
        df.ta.ichimoku(append=True)

        # 3. Volume & Flow
        df.ta.vwap(append=True) 
        df.ta.obv(append=True)
        df.ta.mfi(length=14, append=True)
        df.ta.cmf(length=20, append=True)

        # 4. Volatility
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.atr(length=14, append=True)
    except Exception as e:
        st.warning(f"Technical Analysis calculation failed: {e}. Data may be incomplete.")
        return df 

    return df

def get_indicator_signal(df):
    """
    Calculates detailed signals for all available indicators (14 total).
    """
    df = calculate_ta(df)
    
    if df.empty or len(df) < 200:
        return {} 

    latest_close = lookup_value(df, 'close')
    
    # Initialize groups for all 14 indicators
    signals = {
        "Momentum": [],
        "Trend": [],
        "Volume & Flow": [],
        "Volatility": []
    }

    # --- MOMENTUM INDICATORS ---
    
    # RSI
    rsi_val = lookup_value(df, 'RSI_14')
    rsi_signal = "Neutral"
    if rsi_val is not None:
        if rsi_val > 70: rsi_signal = "Bearish (Overbought)"
        elif rsi_val < 30: rsi_signal = "Bullish (Oversold)"
        elif rsi_val > 55: rsi_signal = "Bullish (Rising)"
        elif rsi_val < 45: rsi_signal = "Bearish (Falling)"
        signals["Momentum"].append({'name': 'RSI (14)', 'value': f"{rsi_val:,.1f}", 'signal': rsi_signal})
    
    # MACD Histogram
    macd_hist = lookup_value(df, 'MACDH_12_26_9')
    macd_signal = "Neutral"
    if macd_hist is not None:
        if macd_hist > 0: macd_signal = "Bullish (Momentum Up)"
        elif macd_hist < 0: macd_signal = "Bearish (Momentum Down)"
        signals["Momentum"].append({'name': 'MACD Histogram', 'value': f"H:{macd_hist:,.2f}", 'signal': macd_signal})
        
    # Stochastic Oscillator
    stoch_k = lookup_value(df, 'STOCHk_14_3_3')
    stoch_d = lookup_value(df, 'STOCHd_14_3_3')
    stoch_signal = "Neutral"
    if stoch_k is not None and stoch_d is not None:
        if stoch_k > stoch_d and stoch_d < 20: stoch_signal = "Bullish (Oversold Buy)"
        elif stoch_k < stoch_d and stoch_d > 80: stoch_signal = "Bearish (Overbought Sell)"
        elif stoch_k > stoch_d: stoch_signal = "Bullish (K Crosses D)"
        elif stoch_k < stoch_d: stoch_signal = "Bearish (D Crosses K)"
        signals["Momentum"].append({'name': 'Stoch K/D', 'value': f"K:{stoch_k:,.1f}", 'signal': stoch_signal})

    # CCI
    cci_val = lookup_value(df, 'CCI_20_0.015')
    cci_signal = "Neutral"
    if cci_val is not None:
        if cci_val > 100: cci_signal = "Bullish (New Trend)"
        elif cci_val < -100: cci_signal = "Bearish (New Trend)"
        signals["Momentum"].append({'name': 'CCI (20)', 'value': f"{cci_val:,.1f}", 'signal': cci_signal})

    # Williams %R (The previously failing indicator)
    williamsr_val = lookup_value(df, 'WPR_14')
    williamsr_signal = "Neutral"
    if williamsr_val is not None:
        if williamsr_val > -20: williamsr_signal = "Bearish (Overbought)"
        elif williamsr_val < -80: williamsr_signal = "Bullish (Oversold)"
        signals["Momentum"].append({'name': 'Williams %R', 'value': f"{williamsr_val:,.1f}", 'signal': williamsr_signal})


    # --- TREND INDICATORS ---
    
    # 200 EMA Check
    ema200 = lookup_value(df, 'EMA_200')
    ema_signal = "Neutral"
    if ema200 is not None and latest_close is not None:
        if latest_close > ema200: ema_signal = "Bullish (Long-Term Trend Up)"
        elif latest_close < ema200: ema_signal = "Bearish (Long-Term Trend Down)"
        signals["Trend"].append({'name': 'Close vs 200 EMA', 'value': f"C:{latest_close:,.0f}", 'signal': ema_signal})

    # 50/200 EMA Cross
    ema50 = lookup_value(df, 'EMA_50')
    ema50_prev = lookup_value(df, 'EMA_50', index=-2)
    ema200_prev = lookup_value(df, 'EMA_200', index=-2)
    cross_signal = "Neutral"
    if ema50 is not None and ema200 is not None and ema50_prev is not None and ema200_prev is not None:
        if ema50 > ema200 and ema50_prev < ema200_prev:
            cross_signal = "Bullish (Golden Cross/Shift)"
        elif ema50 < ema200 and ema50_prev > ema200_prev:
            cross_signal = "Bearish (Death Cross/Shift)"
        signals["Trend"].append({'name': '50/200 EMA Cross', 'value': f"50/200 Gap:{(ema50-ema200):,.0f}", 'signal': cross_signal})

    # ADX/DI
    adx_val = lookup_value(df, 'ADX_14')
    adx_di_plus = lookup_value(df, 'DMP_14')
    adx_di_minus = lookup_value(df, 'DMN_14')
    adx_signal = "Neutral Trend Strength"
    if adx_val is not None and adx_di_plus is not None and adx_di_minus is not None:
        if adx_val > 25:
            adx_signal = "Strong Trend Detected"
            if adx_di_plus > adx_di_minus: adx_signal += " (Bullish Direction)"
            else: adx_signal += " (Bearish Direction)"
        signals["Trend"].append({'name': 'ADX (14)', 'value': f"ADX:{adx_val:,.1f}", 'signal': adx_signal})

    # Ichimoku Cloud (Kumo)
    ichimoku_span_b = lookup_value(df, 'ICSB_26_52_9')
    ichimoku_signal = "Neutral"
    if ichimoku_span_b is not None and latest_close is not None:
        if latest_close > ichimoku_span_b: ichimoku_signal = "Bullish (Above Cloud)"
        elif latest_close < ichimoku_span_b: ichimoku_signal = "Bearish (Below Cloud)"
        signals["Trend"].append({'name': 'Ichimoku Cloud', 'value': f"Close vs Span B", 'signal': ichimoku_signal})


    # --- VOLUME & FLOW INDICATORS ---

    # VWAP
    vwap_val = lookup_value(df, 'VWAP')
    vwap_signal = "Neutral"
    if vwap_val is not None and latest_close is not None:
        if latest_close > vwap_val: vwap_signal = "Bullish (Above VWAP)"
        elif latest_close < vwap_val: vwap_signal = "Bearish (Below VWAP)"
        signals["Volume & Flow"].append({'name': 'VWAP', 'value': f"${vwap_val:,.0f}", 'signal': vwap_signal})

    # CMF
    cmf_val = lookup_value(df, 'CMF_20')
    cmf_signal = "Neutral"
    if cmf_val is not None:
        if cmf_val > 0.05: cmf_signal = "Bullish (Accumulation)"
        elif cmf_val < -0.05: cmf_signal = "Bearish (Distribution)"
        signals["Volume & Flow"].append({'name': 'Chaikin Money Flow', 'value': f"CMF:{cmf_val:,.2f}", 'signal': cmf_signal})

    # MFI
    mfi_val = lookup_value(df, 'MFI_14')
    mfi_signal = "Neutral"
    if mfi_val is not None:
        if mfi_val > 80: mfi_signal = "Bearish (Extreme Overbought)"
        elif mfi_val < 20: mfi_signal = "Bullish (Extreme Oversold)"
        signals["Volume & Flow"].append({'name': 'Money Flow Index (MFI)', 'value': f"{mfi_val:,.1f}", 'signal': mfi_signal})

    # OBV
    obv_current = lookup_value(df, 'OBV', index=-1)
    obv_prev = lookup_value(df, 'OBV', index=-2)
    obv_signal = "Neutral"
    if obv_current is not None and obv_prev is not None:
        if obv_current > obv_prev: obv_signal = "Bullish (Volume Flowing In)"
        elif obv_current < obv_prev: obv_signal = "Bearish (Volume Flowing Out)"
        signals["Volume & Flow"].append({'name': 'On-Balance Volume', 'value': f"Change:{(obv_current-obv_prev):,.0f}", 'signal': obv_signal})


    # --- VOLATILITY INDICATORS ---

    # Bollinger Bands
    upper_band = lookup_value(df, 'BBU_20_2')
    lower_band = lookup_value(df, 'BBL_20_2')
    bb_signal = "Neutral"
    if upper_band is not None and lower_band is not None and latest_close is not None:
        if latest_close > upper_band: bb_signal = "Bearish (Upper Band Test)"
        elif latest_close < lower_band: bb_signal = "Bullish (Lower Band Test)"
        signals["Volatility"].append({'name': 'Bollinger Bands', 'value': "U/L Band Test", 'signal': bb_signal})

    # ATR
    atr_val = lookup_value(df, 'ATR_14')
    atr_signal = "Neutral"
    if atr_val is not None:
        # Simple check for expanding vs contracting volatility based on last 5 bars
        atr_avg_5 = df[safe_column_lookup(df, 'ATR_14')].iloc[-6:-1].mean()
        if atr_val > atr_avg_5 * 1.1: atr_signal = "Expansion (High Volatility)"
        elif atr_val < atr_avg_5 * 0.9: atr_signal = "Contraction (Low Volatility)"
        signals["Volatility"].append({'name': 'ATR (14)', 'value': f"${atr_val:,.2f}", 'signal': atr_signal})
        
    return signals

# ... (get_confluence_score remains unchanged) ...

def get_confluence_score(htf_df, etf_df, htf_label, etf_label):
    """
    Calculates a confluence score based on MTFA across three pillars.
    Score: -3 (Strong Bearish) to +3 (Strong Bullish).
    """
    if htf_df.empty or etf_df.empty or len(htf_df) < 200 or len(etf_df) < 200:
        return 0, "N/A", [f"Insufficient data in one or both time frames ({htf_label} and {etf_label}). Need 200 bars."], "bg-gray-400"

    # Ensure both DFs have required indicators calculated
    calculate_ta(htf_df) 
    calculate_ta(etf_df) 

    score = 0
    factors = []

    # --- PILLAR 1: HTF TREND (200 EMA) ---
    htf_close = lookup_value(htf_df, 'close')
    htf_ema200 = lookup_value(htf_df, 'EMA_200')
    if htf_ema200 is not None and htf_close is not None:
        if htf_close > htf_ema200:
            score += 1
            factors.append(f"**+1 | HTF Trend:** Bullish ({htf_label} close > 200 EMA)")
        elif htf_close < htf_ema200:
            score -= 1
            factors.append(f"**-1 | HTF Trend:** Bearish ({htf_label} close < 200 EMA)")
        else:
            factors.append(f"**0 | HTF Trend:** Neutral ({htf_label} close near 200 EMA)")
    
    # --- PILLAR 2: ETF MOMENTUM (RSI 55/45) ---
    etf_rsi = lookup_value(etf_df, 'RSI_14')
    if etf_rsi is not None:
        if etf_rsi > 55:
            score += 1
            factors.append(f"**+1 | ETF Momentum:** Bullish ({etf_label} RSI > 55)")
        elif etf_rsi < 45:
            score -= 1
            factors.append(f"**-1 | ETF Momentum:** Bearish ({etf_label} RSI < 45)")
        else:
            factors.append(f"**0 | ETF Momentum:** Neutral ({etf_label} RSI 45-55)")

    # --- PILLAR 3: ETF PRICE ACTION / FLOW (Close vs VWAP) ---
    etf_close = lookup_value(etf_df, 'close')
    etf_vwap = lookup_value(etf_df, 'VWAP')
    if etf_vwap is not None and etf_close is not None:
        if etf_close > etf_vwap:
            score += 1
            factors.append(f"**+1 | ETF Flow:** Bullish ({etf_label} Close > VWAP)")
        elif etf_close < etf_vwap:
            score -= 1
            factors.append(f"**-1 | ETF Flow:** Bearish ({etf_label} Close < VWAP)")
        else:
            factors.append(f"**0 | ETF Flow:** Neutral ({etf_label} Close near VWAP)")
            
    # Determine primary signal from score
    if score >= 2:
        signal = "Strong Bullish Confluence (Entry Safe)"
        color = "bg-green-600"
    elif score == 1:
        signal = "Weak Bullish Confluence (Caution)"
        color = "bg-green-400"
    elif score <= -2:
        signal = "Strong Bearish Confluence (Short Safe)"
        color = "bg-red-600"
    elif score == -1:
        signal = "Weak Bearish Confluence (Caution)"
        color = "bg-red-400"
    else:
        signal = "Neutral/Conflicting Confluence"
        color = "bg-yellow-500"

    return score, signal, factors, color

# --- 5. STREAMLIT UI LAYOUT ---

st.set_page_config(
    page_title="Crypto TA & On-Chain Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global styling
st.markdown(
    """
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
    /* ... (CSS styles for stApp, h1, buttons, and ta-tile) ... */
    .stApp { background-color: #f8f9fa; }
    h1 { color: #1e3a8a; } /* Dark Blue Title */
    .stButton>button { border-radius: 0.5rem; border: 1px solid #3b82f6; }
    
    .ta-tile {
        padding: 0.5rem;
        border-radius: 0.75rem;
        border-width: 1px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.06);
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
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar (Configuration & Instructions) ---
with st.sidebar:
    st.image("https://placehold.co/150x50/1e3a8a/ffffff?text=XBT+Monitor")
    st.markdown("## Multi-Time Frame Configuration")

    # HTF Selector (Context)
    timeframe_options = ["1 min", "5 min", "15 min", "30 min", "1 hour", "4 hour", "1 day"]
    
    selected_htf_label = st.selectbox(
        "1. High Time Frame (HTF) - Context",
        options=timeframe_options,
        index=5 # Default to 4 hour
    )
    
    # ETF Selector (Entry)
    selected_etf_label = st.selectbox(
        "2. Entry Time Frame (ETF) - Signal",
        options=timeframe_options,
        index=4 # Default to 1 hour
    )
    
    # Bar Count Slider
    bar_count = st.slider(
        "Historical Bars for TA (Max 720)",
        min_value=100,
        max_value=400,
        value=250,
        step=50
    )

    st.markdown("---")
    
    # Fetch data based on config
    htf_code = get_kraken_interval_code(selected_htf_label)
    etf_code = get_kraken_interval_code(selected_etf_label)
    
    # Fetch DataFrames for both time frames
    htf_df = fetch_historical_data(htf_code, bar_count, selected_htf_label)
    etf_df = fetch_historical_data(etf_code, bar_count, selected_etf_label)

    # Display Instructions and Glossary
    st.markdown("## Instructions")
    st.info("The **Confluence Score** checks for alignment across the HTF and ETF. Use this as your primary trade filter.")
    st.markdown(f"""
        - **Asset:** Bitcoin (XBT/USD)
        - **Exchange:** Kraken
        - **Timezone:** UTC+{UTC_OFFSET_HOTHOURS} (Australian Time)
    """)
    
    st.markdown("---")
    
    # Health Check Button
    if st.button("Run Data Feed Health Check", use_container_width=True):
        with st.spinner('Running API health checks...'):
            st.session_state['health_check_result'] = run_all_checks()

    # Display Health Check Results
    if 'health_check_result' in st.session_state:
        st.subheader("Latest Health Check Results")
        df_results = pd.DataFrame(st.session_state['health_check_result'])
        
        def style_status(val):
            color = 'green' if 'OK' in val else 'red' if 'FAIL' in val or 'TIMEOUT' in val else 'orange'
            return f'color: {color}; font-weight: bold'

        st.dataframe(
            df_results[['name', 'status', 'latency_ms', 'error']].style.applymap(style_status, subset=['status']),
            use_container_width=True,
            hide_index=True
        )
    

# --- Main Dashboard Layout ---

st.title("Automated Crypto TA & Confluence Dashboard")
st.markdown(f"<p class='text-gray-500'>Dashboard monitoring **{selected_etf_label}** data, framed by **{selected_htf_label}** context.</p>", unsafe_allow_html=True)


# 1. LIVE METRICS
st.header("1. Live Price, Volume, and Network Health")

col1_price, col2_vol, col3_block, col4_hash = st.columns([1, 1, 1, 1])

# Fetch live data
live_price, live_volume = fetch_btc_data()
block_height, hash_rate_phs = fetch_on_chain_data()

# Price & Volume
col1_price.metric("Latest XBT Price (USD)", f"${live_price:,.2f}" if live_price else "N/A")
col2_vol.metric("24h Volume (XBT)", f"{live_volume:,.0f} XBT" if live_volume else "N/A")

# On-Chain Metrics (New additions)
col3_block.metric("Current Block Height", f"{block_height:,.0f}" if block_height else "N/A")
col4_hash.metric("Network Hash Rate", f"{hash_rate_phs:,.0f} PH/s" if hash_rate_phs else "N/A")

# ---
st.markdown("---")
# ---

# 2. MTFA CONFLUENCE SCORE
st.header(f"2. Multi-Time Frame Confluence Score")

score, signal, factors, color_class = get_confluence_score(htf_df.copy(), etf_df.copy(), selected_htf_label, selected_etf_label)

col_score, col_details = st.columns([1, 2])

with col_score:
    st.markdown(f"""
        <div class="confluence-box {color_class}">
            <p class="confluence-header">OVERALL CONFLUENCE</p>
            <p class="confluence-score">{score}/3</p>
            <p class="confluence-signal text-lg font-semibold">{signal}</p>
        </div>
    """, unsafe_allow_html=True)

with col_details:
    st.subheader("Pillar Confirmation Details")
    st.markdown("<p class='text-sm text-gray-700'>The score is calculated based on alignment across Trend (HTF 200 EMA), Momentum (ETF RSI), and Flow (ETF Close vs VWAP).</p>", unsafe_allow_html=True)
    for factor in factors:
        st.markdown(f"- {factor}")


# ---
st.markdown("---")
# ---

# 3. AUTOMATED TA SIGNAL MATRIX (Uses Entry Time Frame)
st.header(f"3. Automated TA Signal Matrix ({selected_etf_label})")

if not etf_df.empty and len(etf_df) >= 200:
    ta_signals_grouped = get_indicator_signal(etf_df.copy())
    
    for group_name, signals in ta_signals_grouped.items():
        st.subheader(f"📊 {group_name} Indicators")
        
        num_signals = len(signals)
        num_columns = min(num_signals, 5)
        
        for i in range(0, num_signals, 5):
            row_signals = signals[i:i+5]
            cols = st.columns(len(row_signals)) 
            
            for j, item in enumerate(row_signals):
                tile_class, value_class = get_html_color_class(item['signal'])
                
                # Render the tile
                cols[j].markdown(f"""
                    <div class='ta-tile {tile_class}'>
                        <p class='tile-name'>{item['name']}</p>
                        <p class='tile-value {value_class}'>{item['value']}</p>
                        <p class='tile-signal'>{item['signal']}</p>
                    </div>
                """, unsafe_allow_html=True)
else:
    st.warning(f"Not enough historical data available for the Entry Time Frame ({selected_etf_label}) or bar count ({bar_count}). Need at least 200 bars for robust TA.")


# ---
st.markdown("---")
# ---

# 4. TECHNICAL INDICATOR GLOSSARY 
st.header("4. Technical Indicator Glossary")
st.info("Expand the sections below to review the purpose and signaling logic for each indicator.")

# Group glossary data by type for cleaner display
glossary_grouped = {}
for key, data in INDICATOR_GLOSSARY.items():
    # Merge On-Chain and Placeholder On-Chain for display
    group_name = data['type'].replace(" (Placeholder)", "")
    if group_name not in glossary_grouped:
        glossary_grouped[group_name] = []
    glossary_grouped[group_name].append(data)

# Display grouped glossary
for group_name, items in glossary_grouped.items():
    with st.expander(f"**{group_name}** ({len(items)})"):
        for item in items:
            is_placeholder = "Placeholder" in item['type']
            title = item['title'] + (" (Advanced Placeholder)" if is_placeholder else "")
            
            st.markdown(f"### {title}")
            description_style = 'text-gray-700' if not is_placeholder else 'text-red-600 font-semibold'
            st.markdown(f"<p class='{description_style}'>{item['description']}</p>", unsafe_allow_html=True)
            st.markdown("---")
