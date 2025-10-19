import streamlit as st
import requests
import pandas as pd
import pandas_ta as ta # CRITICAL: Ensures the .ta accessor is available
from datetime import datetime, timezone, timedelta
import math
import uuid
import numpy as np

# --- 1. CONFIGURATION AND CONSTANTS ---

# Kraken API endpoints
KRAKEN_API_URL = "https://api.kraken.com/0/public/"
BLOCKCYPHER_API_URL = "https://api.blockcypher.com/v1/btc/main"

# Timezone configuration (e.g., AEST/AEDT is UTC+10 or UTC+11)
# Set to UTC+11 for Australian time.
UTC_OFFSET_HOTHOURS = 11

# CRITICAL FIX: Define an explicit strategy to bypass the problematic df.ta.strategy("All") call.
# This ensures a more stable initialization of the indicators we need.
CryptoStrategy = ta.Strategy(
    name="Crypto TA Strategy",
    ta=[
        # Momentum Indicators
        {"kind": "rsi", "length": 14},
        {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
        {"kind": "stoch", "k": 14, "d": 3, "smooth_k": 3},
        {"kind": "cci", "length": 20},
        {"kind": "williamsr", "length": 14},
        # Trend Indicators
        {"kind": "ema", "length": [9, 21, 50, 200]},
        {"kind": "adx", "length": 14},
        {"kind": "ichimoku"}, # Defaults 9, 26, 52
        # Volume/Flow Indicators
        {"kind": "obv"},
        {"kind": "mfi", "length": 14},
        {"kind": "cmf", "length": 20},
        {"kind": "vwap"},
        # Volatility Indicators
        {"kind": "bbands", "length": 20, "std": 2},
        {"kind": "atr", "length": 14},
    ]
)


# Indicator Glossary Data (Used for the interactive reference section)
INDICATOR_GLOSSARY = {
    "RSI": {
        "title": "Relative Strength Index (RSI)",
        "description": "A momentum oscillator that measures the speed and change of price movements, identifying overbought or oversold conditions. Signals shifts in momentum: Bullish (crossing above 55), Bearish (crossing below 45).",
        "type": "Momentum"
    },
    "MACD": {
        "title": "Moving Average Convergence Divergence (MACD)",
        "description": "A trend-following momentum indicator that shows the relationship between two moving averages (12/26 periods). Bullish signal when the MACD Line crosses above the Signal Line; Bearish when it crosses below.",
        "type": "Momentum"
    },
    "StochOsc": {
        "title": "Stochastic Oscillator",
        "description": "A momentum indicator comparing a particular closing price to a range of its prices over a certain period. Bullish when %K crosses %D line above 20 (oversold); Bearish when %K crosses %D line below 80 (overbought).",
        "type": "Momentum"
    },
    "CCI": {
        "title": "Commodity Channel Index (CCI)",
        "description": "A momentum-based oscillator used to identify new trends or extreme conditions. Bullish when CCI crosses above +100; Bearish when CCI crosses below -100.",
        "type": "Momentum"
    },
    "WilliamsR": {
        "title": "Williams %R",
        "description": "A momentum indicator that measures overbought and oversold levels. It moves inversely to the scale: readings between 0 and -20 are considered Overbought; -80 and -100 are Oversold.",
        "type": "Momentum"
    },
    "ADX": {
        "title": "Average Directional Index (ADX)",
        "description": "A non-directional indicator that measures the strength of a price trend (from 0 to 100). Readings above 25 indicate a strong trend. The +DI and -DI lines show trend direction.",
        "type": "Trend"
    },
    "EMA": {
        "title": "Exponential Moving Average (EMA)",
        "description": "A type of moving average that places a greater weight and significance on the most recent data points. Used to determine trend direction (long: 50/200 EMA cross) and dynamic support/resistance.",
        "type": "Trend"
    },
    "Ichimoku": {
        "title": "Ichimoku Cloud",
        "description": "A comprehensive trend-following system that uses multiple lines to define support, resistance, trend direction, and momentum. The 'Cloud' (Kumo) is central to its signals.",
        "type": "Trend"
    },
    "VWAP": {
        "title": "Volume Weighted Average Price (VWAP)",
        "description": "The average price weighted by volume. Used as a benchmark by institutional traders. Price trading above VWAP is often bullish; below is bearish.",
        "type": "Volume"
    },
    "OBV": {
        "title": "On-Balance Volume (OBV)",
        "description": "A volume indicator that relates volume change to price change. Used to confirm trends: if price rises and OBV rises, the trend is supported by volume (Bullish).",
        "type": "Volume"
    },
    "MFI": {
        "title": "Money Flow Index (MFI)",
        "description": "A volume-weighted momentum oscillator (similar to RSI). Measures the rate at which money is flowing into or out of an asset. Bullish when MFI is rising from oversold (<20).",
        "type": "Volume"
    },
    "CMF": {
        "title": "Chaikin Money Flow (CMF)",
        "description": "Measures the amount of money flow volume over a period. Values above 0 suggest accumulation (buying pressure); values below 0 suggest distribution (selling pressure).",
        "type": "Volume"
    },
    "BBANDS": {
        "title": "Bollinger Bands (BBANDS)",
        "description": "A volatility indicator. Price hitting the Upper Band suggests Overbought, while price hitting the Lower Band suggests Oversold. Band width indicates volatility.",
        "type": "Volatility"
    },
    "ATR": {
        "title": "Average True Range (ATR)",
        "description": "A volatility measure that shows the average size of the price range over a specified period. Used to set stop-losses and determine if volatility is expanding (increasing) or contracting (decreasing).",
        "type": "Volatility"
    },
    "BlockchainHeight": {
        "title": "Blockchain Height (On-Chain)",
        "description": "The current block number of the Bitcoin blockchain. A consistently growing height confirms network health and processing activity.",
        "type": "On-Chain"
    }
}

# --- 2. UTILITY FUNCTIONS (Enhanced for Robustness) ---

def safe_column_lookup(df, prefix):
    """
    Safely looks up a column by prefix (e.g., 'RSI_14_') to prevent KeyError.
    Returns the full column name if found, otherwise returns None.
    """
    try:
        # Filter columns that start with the required prefix
        matching_cols = df.columns[df.columns.str.startswith(prefix)]
        if not matching_cols.empty:
            # Return the last matching column (e.g., if multiple versions exist)
            return matching_cols[-1]
        return None
    except Exception:
        return None

def lookup_value(df, prefix, index=-1):
    """
    Robustly looks up an indicator value by prefix and index.
    Returns the float value if found and not NaN, otherwise returns None.
    """
    col_name = safe_column_lookup(df, prefix)
    if col_name is None:
        return None
    try:
        # Check if the column exists after lookup
        if col_name not in df.columns:
            return None
            
        val = df[col_name].iloc[index]
        # Check for NaN and pandas' internal NA
        if pd.isna(val) or isinstance(val, (int, float)) and math.isnan(val):
            return None
        return float(val) # Ensure it's a float for comparisons
    except Exception:
        # Catch any remaining KeyError/IndexError/TypeErrors
        return None

def get_kraken_interval_code(interval_label):
    """Maps human-readable interval to Kraken API code."""
    interval_map = {
        "1 min": 1,
        "5 min": 5,
        "15 min": 15,
        "30 min": 30,
        "1 hour": 60,
        "4 hour": 240,
        "1 day": 1440,
        "1 week": 10080,
    }
    return interval_map.get(interval_label, 60) # Default to 1 hour

def get_html_color_class(signal):
    """Maps signal status to Tailwind CSS color classes for the tiles."""
    if "Bullish" in signal or "Accumulation" in signal or "Strong Long" in signal or "Oversold" in signal:
        return "bg-green-100 border-green-400 text-green-800", "text-green-600"
    elif "Bearish" in signal or "Distribution" in signal or "Strong Short" in signal or "Overbought" in signal:
        return "bg-red-100 border-red-400 text-red-800", "text-red-600"
    elif "Strong Trend" in signal or "Expansion" in signal:
        return "bg-yellow-100 border-yellow-400 text-yellow-800", "text-yellow-600"
    else:
        return "bg-gray-100 border-gray-400 text-gray-700", "text-gray-500"

def get_glossary_html(glossary):
    """
    Generates a series of HTML elements for a collapsible glossary.
    """
    html_parts = []
    
    for key, data in glossary.items():
        unique_id = uuid.uuid4().hex[:8] 

        header = f"""
        <div id="header-{unique_id}" data-id="{unique_id}" class="indicator-header flex justify-between items-center p-3 md:p-4 bg-indigo-50 hover:bg-indigo-100 transition duration-150 ease-in-out select-none rounded-t-lg shadow-md cursor-pointer mb-0 border border-indigo-200">
            <span class="font-semibold text-sm text-indigo-700">{data['title']} <span class="text-xs font-medium px-2 py-0.5 ml-2 rounded-full bg-indigo-200 text-indigo-800">{data['type']}</span></span>
            <svg class="h-5 w-5 text-indigo-500 transform transition-transform duration-300" data-arrow="{unique_id}" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
            </svg>
        </div>
        """
        content = f"""
        <div id="content-{unique_id}" class="indicator-content p-4 bg-white border border-t-0 border-gray-300 rounded-b-lg shadow-inner hidden mb-4">
            <p class="text-xs text-gray-600 leading-relaxed">{data['description']}</p>
        </div>
        """
        html_parts.append(header)
        html_parts.append(content)

    # JavaScript for the collapse/expand behavior
    js_script = """
    <script>
        function setupGlossaryToggle() {
            document.querySelectorAll('.indicator-header').forEach(header => {
                header.removeEventListener('click', toggleContent); 
                header.addEventListener('click', toggleContent);
            });
        }

        function toggleContent(event) {
            const uniqueId = event.currentTarget.getAttribute('data-id');
            const content = document.getElementById(`content-${uniqueId}`);
            const arrow = document.querySelector(`[data-arrow='${uniqueId}']`);
            const header = event.currentTarget;

            if (content.classList.contains('hidden')) {
                content.classList.remove('hidden');
                header.classList.remove('rounded-b-lg');
                header.classList.add('rounded-none');
                arrow.classList.add('rotate-180');
            } else {
                content.classList.add('hidden');
                header.classList.remove('rounded-none');
                header.classList.add('rounded-b-lg');
                arrow.classList.remove('rotate-180');
            }
        }

        window.onload = setupGlossaryToggle;
        setTimeout(setupGlossaryToggle, 500); // Rerun for dynamic content
    </script>
    """

    return "\n".join(html_parts) + js_script

# --- 3. API FETCHING FUNCTIONS ---

@st.cache_data(ttl=60)
def fetch_kraken_time():
    """Fetches the official Kraken server time."""
    try:
        response = requests.get(KRAKEN_API_URL + "Time", timeout=5)
        response.raise_for_status()
        server_time_unix = response.json()['result']['unixtime']
        
        # Convert to local time zone (UTC+11)
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
        st.error(f"Could not fetch live Bitcoin data from Kraken: {e}. Bitcoin data not available from Kraken API.")
        return None, None

@st.cache_data(ttl=120)
def fetch_historical_data(interval_code, count):
    """Fetches OHLC data from Kraken for TA."""
    params = {'pair': 'XBTUSD', 'interval': interval_code} 
    
    try:
        response = requests.get(KRAKEN_API_URL + "OHLC", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()['result']
        
        data_key = next(k for k in data.keys() if k != 'last')
        candles = data[data_key]
        
        # Convert to DataFrame
        df = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # Explicitly ensure OHLC columns are numeric floats
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
        # Only keep the last 'count' number of bars (Kraken provides max 720)
        return df.tail(count).copy()
    
    except Exception as e:
        st.error(f"Kraken OHLC API reported an error: {e}")
        return pd.DataFrame()

# --- 4. HEALTH CHECK FUNCTIONS ---

@st.cache_data(ttl=300)
def run_all_checks():
    """Runs health checks on all major data feeds."""
    api_configs = [
        {'name': 'Kraken API (Live Ticker)', 'url': KRAKEN_API_URL + "Ticker?pair=XBTUSD", 'type': 'Exchange', 'check_func': check_http_status},
        {'name': 'Kraken API (Stable Time)', 'url': KRAKEN_API_URL + "Time", 'type': 'Exchange', 'check_func': check_kraken_time},
        {'name': 'Blockchain (BTC BlockCypher)', 'url': BLOCKCYPHER_API_URL, 'type': 'Blockchain', 'check_func': check_http_status},
    ]

    for i in range(len(api_configs)):
        name = api_configs[i]['name']
        url = api_configs[i]['url']
        check_func = api_configs[i]['check_func']
        
        try:
            status, latency, data = check_func(url)
            api_configs[i]['status'] = "UP"
            api_configs[i]['latency'] = f"{latency:.0f}ms"
            # Attempt to extract a concise data point for display
            if 'unixtime' in data.get('result', {}):
                 api_configs[i]['status_detail'] = f"Time: {data['result']['unixtime']}"
            elif isinstance(data, dict) and 'hash' in data:
                api_configs[i]['status_detail'] = f"Block: {data['height']}"
            else:
                api_configs[i]['status_detail'] = status
            api_configs[i]['last_data'] = "Check OK"
        except Exception as e:
            api_configs[i]['status'] = "Error"
            api_configs[i]['latency'] = "N/A"
            api_configs[i]['status_detail'] = f"Failed: {e}"
            api_configs[i]['last_data'] = "Check logs"
    
    return api_configs

def check_http_status(url):
    """Generic check for any HTTP endpoint."""
    start_time = datetime.now()
    response = requests.get(url, timeout=5)
    end_time = datetime.now()
    latency = (end_time - start_time).total_seconds() * 1000
    response.raise_for_status()
    return "OK", latency, response.json()

def check_kraken_time(url):
    """Specific check for Kraken Time endpoint integrity."""
    start_time = datetime.now()
    response = requests.get(url, timeout=5)
    end_time = datetime.now()
    latency = (end_time - start_time).total_seconds() * 1000
    
    response_json = response.json()
    if 'result' in response_json and 'unixtime' in response_json['result']:
        return "OK (Kraken Time OK)", latency, response_json
    else:
        raise Exception("Time field missing in Kraken response.")

# --- 5. TECHNICAL ANALYSIS LOGIC (Using Robust Lookups) ---

def get_indicator_signal(df):
    """Calculates all indicators and determines Bullish/Bearish/Neutral signals."""
    global CryptoStrategy # Access the globally defined strategy

    # Ensure there is enough data for robust TA, typically 200 bars is safe for EMAs/Ichimoku
    if df.empty or len(df) < 200:
        return {} 

    try:
        # APPLY FIX: Use the explicit ta.call() method with the custom strategy
        df.ta.call(CryptoStrategy)
    except Exception as e:
        st.error(f"Error calculating indicators with pandas_ta using custom strategy: {e}")
        return {}
    
    latest_close = lookup_value(df, 'close')
    signals = {}

    # --- MOMENTUM INDICATORS ---

    # 1. RSI (14)
    rsi_val = lookup_value(df, 'RSI_14')
    rsi_signal = "Neutral"
    if rsi_val is not None:
        if rsi_val > 70: rsi_signal = "Bearish (Overbought)"
        elif rsi_val > 55: rsi_signal = "Bullish (Momentum Up)"
        elif rsi_val < 30: rsi_signal = "Bullish (Oversold)"
        elif rsi_val < 45: rsi_signal = "Bearish (Momentum Down)"
    
    # 2. MACD (12, 26, 9)
    # Note: MACD column prefixes are typically MACD, MACDH, MACDs
    macd_line = lookup_value(df, 'MACD_12_26_9')
    macd_hist = lookup_value(df, 'MACDH_12_26_9')
    macd_signal_line = lookup_value(df, 'MACDs_12_26_9')
    macd_signal = "Neutral"
    if macd_line is not None and macd_signal_line is not None:
        if macd_hist > 0 and macd_line > macd_signal_line: macd_signal = "Bullish Crossover"
        elif macd_hist < 0 and macd_line < macd_signal_line: macd_signal = "Bearish Crossover"
        
    # 3. Stochastic Oscillator (14, 3, 3)
    k_line = lookup_value(df, 'STOCHk_14_3_3')
    d_line = lookup_value(df, 'STOCHd_14_3_3')
    stoch_signal = "Neutral"
    if k_line is not None and d_line is not None:
        if k_line > d_line and k_line < 20: stoch_signal = "Bullish Crossover (Oversold)"
        elif k_line < d_line and k_line > 80: stoch_signal = "Bearish Crossover (Overbought)"

    # 4. Commodity Channel Index (CCI) (20)
    cci_val = lookup_value(df, 'CCI_20')
    cci_signal = "Neutral"
    if cci_val is not None:
        if cci_val > 100: cci_signal = "Bullish (Strong Move)"
        elif cci_val < -100: cci_signal = "Bearish (Strong Move)"

    # 5. Williams %R (14)
    wpr_val = lookup_value(df, 'WPR_14')
    wpr_signal = "Neutral"
    if wpr_val is not None:
        if wpr_val < -80: wpr_signal = "Bullish (Oversold)"
        elif wpr_val > -20: wpr_signal = "Bearish (Overbought)"
        
    # --- TREND INDICATORS ---

    # 6. EMA (9, 21, 50, 200)
    ema9 = lookup_value(df, 'EMA_9')
    ema200 = lookup_value(df, 'EMA_200')
    ema_signal = "Neutral"
    if ema9 is not None and ema200 is not None and latest_close is not None:
        if latest_close > ema9 and ema9 > ema200 and latest_close > ema200: ema_signal = "Bullish (Strong Trend)"
        elif latest_close < ema9 and ema9 < ema200 and latest_close < ema200: ema_signal = "Bearish (Strong Trend)"
        
    # 7. Average Directional Index (ADX) (14)
    adx_val = lookup_value(df, 'ADX_14')
    di_plus = lookup_value(df, 'DMP_14')
    di_minus = lookup_value(df, 'DMN_14')
    adx_signal = "Neutral"
    if adx_val is not None and di_plus is not None and di_minus is not None:
        strength = "Weak/No Trend"
        if adx_val > 25: strength = "Strong Trend"
        
        if di_plus > di_minus: adx_signal = f"Bullish ({strength})"
        elif di_minus > di_plus: adx_signal = f"Bearish ({strength})"
        else: adx_signal = f"Neutral ({strength})"

    # 8. Ichimoku Cloud (9, 26, 52)
    # SSA (Span A) and SSB (Span B) define the cloud boundaries (Kumo)
    ssa_val = lookup_value(df, 'ISA_9_26_52') 
    ssb_val = lookup_value(df, 'ISB_9_26_52') 
    ichimoku_signal = "Neutral"
    if ssa_val is not None and ssb_val is not None and latest_close is not None:
        cloud_top = max(ssa_val, ssb_val)
        cloud_bottom = min(ssa_val, ssb_val)
        
        if latest_close > cloud_top: ichimoku_signal = "Bullish (Above Cloud)"
        elif latest_close < cloud_bottom: ichimoku_signal = "Bearish (Below Cloud)"
        
    # --- VOLUME/FLOW INDICATORS ---

    # 9. On-Balance Volume (OBV)
    obv_val = lookup_value(df, 'OBV')
    obv_prev = lookup_value(df, 'OBV', index=-2) # Get previous value
    obv_signal = "Neutral"
    if obv_val is not None and obv_prev is not None:
        if obv_val > obv_prev: obv_signal = "Bullish (Volume Confirmation)"
        elif obv_val < obv_prev: obv_signal = "Bearish (Volume Divergence)"

    # 10. Money Flow Index (MFI) (14)
    mfi_val = lookup_value(df, 'MFI_14')
    mfi_signal = "Neutral"
    if mfi_val is not None:
        if mfi_val > 80: mfi_signal = "Bearish (Overbought)"
        elif mfi_val < 20: mfi_signal = "Bullish (Oversold)"
        
    # 11. Chaikin Money Flow (CMF) (20)
    cmf_val = lookup_value(df, 'CMF_20')
    cmf_signal = "Neutral"
    if cmf_val is not None:
        if cmf_val > 0.20: cmf_signal = "Bullish (Strong Accumulation)"
        elif cmf_val > 0: cmf_signal = "Bullish (Accumulation)"
        elif cmf_val < -0.20: cmf_signal = "Bearish (Strong Distribution)"
        elif cmf_val < 0: cmf_signal = "Bearish (Distribution)"

    # 12. Volume Weighted Average Price (VWAP)
    # Note: VWAP column name is just 'VWAP'
    vwap_val = lookup_value(df, 'VWAP')
    vwap_signal = "Neutral"
    if vwap_val is not None and latest_close is not None:
        if latest_close > vwap_val: vwap_signal = "Bullish (Above VWAP)"
        elif latest_close < vwap_val: vwap_signal = "Bearish (Below VWAP)"
        
    # --- VOLATILITY INDICATORS ---

    # 13. Bollinger Bands (20, 2)
    upper_band = lookup_value(df, 'BBU_20_2')
    median_band = lookup_value(df, 'BBM_20_2')
    lower_band = lookup_value(df, 'BBL_20_2')
    bb_signal = "Neutral"
    if upper_band is not None and lower_band is not None and latest_close is not None:
        if latest_close > upper_band: bb_signal = "Bearish (Overbought/Upper Band Test)"
        elif latest_close < lower_band: bb_signal = "Bullish (Oversold/Lower Band Test)"
        
    # 14. Average True Range (ATR)
    atr_val = lookup_value(df, 'ATR_14')
    atr_prev = lookup_value(df, 'ATR_14', index=-2)
    atr_signal = "Neutral"
    if atr_val is not None and atr_prev is not None:
        if atr_val > atr_prev: atr_signal = "Expansion (Volatility Rising)"
        else: atr_signal = "Contraction (Volatility Falling)"

    # --- Format Values for Display ---
    
    # VWAP
    vwap_val_str = f"${vwap_val:,.0f}" if vwap_val is not None else "N/A"

    # BBANDS (Formatted to show U | M | L)
    if upper_band is not None and median_band is not None and lower_band is not None:
        bb_val_str = f"U:{upper_band:,.0f} | M:{median_band:,.0f} | L:{lower_band:,.0f}"
    else:
        bb_val_str = "N/A"
        
    # MACD (Formatted to show M | S | H)
    if macd_line is not None and macd_signal_line is not None and macd_hist is not None:
        macd_val_str = f"M:{macd_line:,.2f} | S:{macd_signal_line:,.2f} | H:{macd_hist:,.2f}"
    else:
        macd_val_str = "N/A"

    # ADX (Formatted to show ADX | +DI | -DI)
    if adx_val is not None and di_plus is not None and di_minus is not None:
        adx_val_str = f"ADX:{adx_val:,.1f} | +DI:{di_plus:,.1f} | -DI:{di_minus:,.1f}"
    else:
        adx_val_str = "N/A"

    # Stoch (Formatted to show %K | %D)
    stoch_val_str = f"%K:{k_line:,.1f} | %D:{d_line:,.1f}" if k_line is not None and d_line is not None else "N/A"

    # ATR (Formatted to show current ATR value)
    atr_val_str = f"${atr_val:,.2f}" if atr_val is not None else "N/A"
        
    # CMF/MFI/WPR/CCI/RSI (Formatted to show simple value)
    cmf_val_str = f"{cmf_val:,.2f}" if cmf_val is not None else "N/A"
    mfi_val_str = f"{mfi_val:,.1f}" if mfi_val is not None else "N/A"
    wpr_val_str = f"{wpr_val:,.1f}" if wpr_val is not None else "N/A"
    cci_val_str = f"{cci_val:,.0f}" if cci_val is not None else "N/A"
    rsi_val_str = f"{rsi_val:,.1f}" if rsi_val is not None else "N/A"
    
    # Handle latest close if it failed
    latest_close_str = f"{latest_close:,.0f}" if latest_close is not None else "N/A"

    # Assemble final signals dictionary, grouped by type
    signals = {
        "Momentum": [
            {'name': 'RSI', 'value': rsi_val_str, 'signal': rsi_signal},
            {'name': 'MACD', 'value': macd_val_str, 'signal': macd_signal},
            {'name': 'Stoch Oscillator', 'value': stoch_val_str, 'signal': stoch_signal},
            {'name': 'Williams %R', 'value': wpr_val_str, 'signal': wpr_signal},
            {'name': 'CCI', 'value': cci_val_str, 'signal': cci_signal},
        ],
        "Trend": [
            {'name': 'EMA 9/21/50/200', 'value': f"Last Close: {latest_close_str}", 'signal': ema_signal},
            {'name': 'ADX', 'value': adx_val_str, 'signal': adx_signal},
            {'name': 'Ichimoku Cloud', 'value': "Cloud Status", 'signal': ichimoku_signal},
        ],
        "Volume": [
            {'name': 'VWAP', 'value': vwap_val_str, 'signal': vwap_signal},
            {'name': 'OBV', 'value': "Volume Flow", 'signal': obv_signal},
            {'name': 'MFI', 'value': mfi_val_str, 'signal': mfi_signal},
            {'name': 'CMF', 'value': cmf_val_str, 'signal': cmf_signal},
        ],
        "Volatility": [
            {'name': 'Bollinger Bands', 'value': bb_val_str, 'signal': bb_signal},
            {'name': 'ATR (14)', 'value': atr_val_str, 'signal': atr_signal},
        ]
    }
    
    return signals

def get_divergence_alerts(df):
    """Provides actionable long/short alerts based on current indicator values."""
    if df.empty or len(df) < 200:
        return []

    alerts = []
    
    # Use the robust lookup_value utility
    latest_close = lookup_value(df, 'close')
    
    # Check if necessary data exists before proceeding with complex checks
    if latest_close is None:
        return []

    mfi_val = lookup_value(df, 'MFI_14')
    mfi_prev = lookup_value(df, 'MFI_14', index=-2)
    cmf_val = lookup_value(df, 'CMF_20')
    cmf_prev = lookup_value(df, 'CMF_20', index=-2)
    atr_val = lookup_value(df, 'ATR_14')
    close_prev = lookup_value(df, 'close', index=-2)


    # 1. MFI Extreme Reversal Alert (Tactical Reversal)
    if all(v is not None for v in [mfi_val, mfi_prev, close_prev]):
        if mfi_val < 30 and mfi_prev < 30 and close_prev < latest_close:
            alerts.append({"type": "Long", "message": f"MFI ({mfi_val:,.0f}) is **Oversold** and showing Bullish reversal on volume. Potential bounce.", "color": "green"})
        elif mfi_val > 70 and mfi_prev > 70 and close_prev > latest_close:
            alerts.append({"type": "Short", "message": f"MFI ({mfi_val:,.0f}) is **Overbought** and showing Bearish reversal on volume. Potential drop.", "color": "red"})

    # 2. CMF Zero Cross Alert
    if all(v is not None for v in [cmf_val, cmf_prev]):
        if cmf_prev < 0 and cmf_val >= 0:
            alerts.append({"type": "Long", "message": f"CMF ({cmf_val:,.2f}) crossed **ZERO** (Accumulation) line. Shift from distribution to buying pressure.", "color": "green"})
        elif cmf_prev > 0 and cmf_val <= 0:
            alerts.append({"type": "Short", "message": f"CMF ({cmf_val:,.2f}) crossed **ZERO** (Distribution) line. Shift from accumulation to selling pressure.", "color": "red"})

    # 3. Volume/Price Disparity Alert (Absorption/Exhaustion)
    if atr_val is not None and cmf_val is not None and 'volume' in df.columns:
        
        # Calculate moving average volume over last 20 bars
        if len(df) >= 20:
            avg_volume = df['volume'].iloc[-20:].mean()
            current_volume = df['volume'].iloc[-1]
            current_range = df['high'].iloc[-1] - df['low'].iloc[-1]
            
            # High volume (150% of average) and small range (less than 50% of ATR)
            if current_volume > (avg_volume * 1.5) and current_range < (atr_val * 0.5):
                if cmf_val > 0.1: # High Volume + Tight Range + Accumulation (Bullish Absorption)
                    alerts.append({"type": "Long", "message": f"VOLUME/PRICE DISPARITY: High volume absorption with tight range (ATR:{atr_val:,.0f}). Demand is strong, supply is being absorbed.", "color": "green"})
                elif cmf_val < -0.1: # High Volume + Tight Range + Distribution (Bearish Exhaustion)
                    alerts.append({"type": "Short", "message": f"VOLUME/PRICE DISPARITY: High volume exhaustion with tight range (ATR:{atr_val:,.0f}). Supply is strong, demand is exhausting.", "color": "red"})

    return alerts

# --- 6. STREAMLIT UI LAYOUT ---

# Set the page configuration for a wide view and a professional look
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
    /* General Styling */
    .stApp { background-color: #f8f9fa; }
    h1 { color: #1e3a8a; } /* Dark Blue Title */
    .stButton>button { border-radius: 0.5rem; border: 1px solid #3b82f6; }
    
    /* Custom Styling for the TA Tiles */
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
    .rotate-180 {
        transform: rotate(180deg);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar (Configuration & Instructions) ---
with st.sidebar:
    st.image("https://placehold.co/150x50/1e3a8a/ffffff?text=XBT+Monitor")
    st.markdown("## Configuration")

    # Timeframe Selector
    timeframe_options = ["1 min", "5 min", "15 min", "30 min", "1 hour", "4 hour", "1 day", "1 week"]
    selected_timeframe = st.selectbox(
        "Timeframe (Interval)",
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
    interval_code = get_kraken_interval_code(selected_timeframe)
    # The function returns a DataFrame modified in-place by pandas_ta after the first run
    historical_df = fetch_historical_data(interval_code, bar_count)

    # Display Instructions and Glossary
    st.markdown("## Instructions")
    st.info("Signals are calculated on the **current candle** using the selected timeframe. Use the **Glossary** section below for indicator definitions.")
    st.markdown(f"""
        - **Asset:** Bitcoin (XBT/USD)
        - **Exchange:** Kraken
        - **Timezone:** UTC+{UTC_OFFSET_HOTHOURS} (Australian Time)
    """)
    
    st.markdown("---")
    
    # Health Check Button
    if st.button("Run Data Feed Health Check", use_container_width=True):
        st.session_state['health_check_result'] = run_all_checks()
    

# --- Main Dashboard Layout ---

st.title("Automated Crypto TA & On-Chain Signal Dashboard")
st.markdown(f"<p class='text-gray-500'>Dashboard running on **{selected_timeframe}** data (last {len(historical_df)} bars).</p>", unsafe_allow_html=True)


# 1. LIVE METRICS (Three columns)
st.header("1. Live Price & Volume Metrics")

col1_price, col2_vol, col3_time = st.columns([1, 1, 1])

# Fetch live data
live_price, live_volume = fetch_btc_data()

if live_price and live_volume:
    col1_price.metric("Latest XBT Price (USD)", f"${live_price:,.2f}")
    col2_vol.metric("24h Volume (XBT)", f"{live_volume:,.0f} XBT")
else:
    col1_price.metric("Latest XBT Price (USD)", "N/A")
    col2_vol.metric("24h Volume (XBT)", "N/A")

# Fetch Kraken time
kraken_time = fetch_kraken_time()
col3_time.metric("Kraken Server Time", kraken_time)

# 2. TACTICAL REVERSAL ALERTS (Highest priority signals)
st.markdown("---")
st.header("2. Tactical Reversal Alerts")

if not historical_df.empty and len(historical_df) >= 200:
    # We run the TA calculation here so that both alerts and matrix use the same calculated columns
    ta_signals_df = historical_df.copy()
    # Ensure the strategy is applied to the temporary df for stability
    ta_signals_df.ta.call(CryptoStrategy) 
    
    alerts = get_divergence_alerts(ta_signals_df)
    if alerts:
        # Use columns up to a max of 4 for small screen readability
        num_alerts = len(alerts)
        alert_cols = st.columns(min(num_alerts, 4)) 
        for i, alert in enumerate(alerts):
            # Fallback to the same column index if we run out of columns
            col_index = i % 4 
            color = alert['color']
            alert_cols[col_index].markdown(f"""
                <div style="background-color: {'#d1e7dd' if color == 'green' else '#f8d7da'}; 
                             border: 1px solid {'#a3cfbb' if color == 'green' else '#f5c6cb'}; 
                             padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 15px;">
                    <p style="font-weight: bold; color: {'#0f5132' if color == 'green' else '#842029'}; margin: 0; font-size: 1rem;">
                        {alert['type']} OPPORTUNITY
                    </p>
                    <p style="color: {'#0f5132' if color == 'green' else '#842029'}; margin: 0; font-size: 0.85rem;">
                        {alert['message']}
                    </p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No immediate tactical reversal signals detected. Market is stable or lacking strong divergence patterns.")
else:
    st.warning("Insufficient data to generate Tactical Reversal Alerts. Need at least 200 bars for robust calculation.")


# 3. AUTOMATED TA SIGNAL MATRIX
st.markdown("---")
st.header(f"3. Automated TA Signal Matrix ({selected_timeframe})")

if not historical_df.empty and len(historical_df) >= 200:
    # Pass the already calculated DataFrame to the signal function
    ta_signals_grouped = get_indicator_signal(ta_signals_df)
    
    for group_name, signals in ta_signals_grouped.items():
        st.subheader(f"📊 {group_name} Indicators")
        # Use 5 columns for a tight, dashboard look
        cols = st.columns(5) 
        
        for i, item in enumerate(signals):
            # Get color class based on signal
            tile_class, value_class = get_html_color_class(item['signal'])
            
            # Render the tile
            cols[i].markdown(f"""
                <div class='ta-tile {tile_class}'>
                    <p class='tile-name'>{item['name']}</p>
                    <p class='tile-value {value_class}'>{item['value']}</p>
                    <p class='tile-signal'>{item['signal']}</p>
                </div>
            """, unsafe_allow_html=True)
else:
    st.warning(f"Not enough historical data available for the selected timeframe ({selected_timeframe}) or bar count ({bar_count}). Need at least 200 bars for robust TA.")


# 4. DATA FEED HEALTH MONITOR
st.markdown("---")
st.header("4. Data Feed Health Monitor")

if 'health_check_result' in st.session_state:
    results = st.session_state['health_check_result']
    data = []
    for item in results:
        data.append({
            'API Feed': item['name'],
            'Type': item['type'],
            'Status': item['status'],
            'Latency': item['latency'],
            'Status Detail': item['status_detail'],
            'Last Check': datetime.now(timezone(timedelta(hours=UTC_OFFSET_HOTHOURS))).strftime("%H:%M:%S")
        })
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
else:
    st.info("Click 'Run Data Feed Health Check' in the sidebar to monitor connections.")

# 5. GLOSSARY OF INDICATORS (Interactive Section)
st.markdown("---")
st.header("5. Glossary of Indicators")
st.markdown("<p class='text-sm text-gray-500 mb-4'>Click to expand definitions for all Technical and On-Chain concepts.</p>", unsafe_allow_html=True)

# Generate and display the interactive HTML glossary
st.markdown(get_glossary_html(INDICATOR_GLOSSARY), unsafe_allow_html=True)
