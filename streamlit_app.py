import streamlit as st
import requests
import time
import json
import pandas as pd
from datetime import datetime, timezone, timedelta
import pandas_ta as ta # Import the technical analysis library

# --- Configuration and Constants ---
MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1
# ... (rest of configuration constants remain the same)

# TIMEZONE CONFIG: Set target display timezone to UTC+11
UTC_OFFSET_HOURS = 11 
UTC_PLUS_11 = timezone(timedelta(hours=UTC_OFFSET_HOURS))

# KRAKEN API CONFIG
KRAKEN_API_URL = "https://api.kraken.com"
KRAKEN_TICKER_ENDPOINT = f"{KRAKEN_API_URL}/0/public/Ticker?pair=XBTUSD" 
KRAKEN_OHLC_ENDPOINT = f"{KRAKEN_API_URL}/0/public/OHLC"

# Mapping for Kraken OHLC intervals (in minutes)
KRAKEN_INTERVALS = {
    "1 minute": 1,
    "5 minute": 5,
    "15 minute": 15,
    "30 minute": 30,
    "1 hour": 60,
    "4 hour": 240,
    "1 day": 1440,
    "1 week": 10080,
}

# The INITIAL_CONFIG list for the health check section (Section 1)
INITIAL_CONFIG = [
    {"id": 1, "name": "Blockchain (BTC BlockCypher)", "url": "https://api.blockcypher.com/v1/btc/main", "status": "Pending", "last_check": None, "last_result": ""},
    {"id": 2, "name": "Kraken API (Stable Time)", "url": f"{KRAKEN_API_URL}/0/public/Time", "status": "Pending", "last_check": None, "last_result": ""},
    {"id": 3, "name": "Kraken API (Live Ticker)", "url": KRAKEN_TICKER_ENDPOINT, "status": "Pending", "last_check": None, "last_result": ""},
]


# Configure the Streamlit page layout and title
st.set_page_config(
    page_title="Bitcoin TA Signal Matrix & Feed Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State Initialization ---
if 'api_configs' not in st.session_state:
    st.session_state.api_configs = INITIAL_CONFIG
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Time', 'Feed', 'Status', 'Response Time (ms)'])

# --- Historical Data Fetcher (Kraken OHLC) ---

@st.cache_data(ttl=300) # Cache historical data for 5 minutes
def fetch_historical_data(interval_minutes, count=300):
    """
    Fetches historical OHLC data from Kraken.
    """
    try:
        seconds_per_interval = interval_minutes * 60
        approx_start_time = int(time.time() - (count * seconds_per_interval * 1.5))

        params = {
            'pair': 'XBTUSD',
            'interval': interval_minutes,
            'since': approx_start_time
        }

        response = requests.get(KRAKEN_OHLC_ENDPOINT, params=params, timeout=10)
        response.raise_for_status() 
        data = response.json()
        
        if data.get('error'):
            error_msg = data['error'][0] if data['error'] else "Unknown Kraken API Error"
            st.error(f"Kraken OHLC API reported an error: {error_msg}")
            return None

        result_pairs = {k: v for k, v in data.get('result', {}).items() if k != 'last'}
        pair_key = next(iter(result_pairs), None)
        
        if not pair_key:
            return None

        ohlc_data = result_pairs.get(pair_key)
        
        # Convert list of lists to DataFrame
        df = pd.DataFrame(ohlc_data, columns=[
            'Time', 'Open', 'High', 'Low', 'Close', 'VWAP', 'Volume', 'Count'
        ])
        
        # Convert types and set index
        df['Time'] = pd.to_datetime(df['Time'], unit='s', utc=True)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Limit to the most recent 'count' bars and drop the last (potentially partial) bar
        df = df.iloc[:-1].tail(count).set_index('Time')
        
        return df

    except Exception as e:
        st.error(f"Could not fetch historical data from Kraken: {e}")
        return None

# --- Technical Analysis (TA) Signal Logic ---

def get_indicator_signal(df):
    """
    Calculates all specified indicators and determines a bullish/bearish/neutral signal.
    Returns (signal, detail, value_string) for each indicator.
    """
    
    # Needs at least 52 bars for Ichimoku to be calculated properly.
    if df is None or df.empty or len(df) < 52: 
        return {}
    
    # --- 1. Calculate ALL indicators using pandas_ta ---
    # These calls populate the DataFrame with new columns like 'RSI_14', 'MACDh_12_26_9', etc.
    df.ta.rsi(append=True)
    df.ta.macd(append=True)
    df.ta.stoch(append=True) 
    df.ta.cci(append=True) 
    df.ta.willr(append=True) 
    df.ta.adx(append=True) 
    df.ta.bbands(append=True) 
    df.ta.obv(append=True)
    df.ta.mfi(append=True)
    df.ta.ichimoku(append=True)

    # --- 2. Define Signal Logic (Based on standard rules) ---
    signals = {}
    close = df['Close'].iloc[-1]
    close_str = f"{close:,.0f}" # Formatted for price indicators
    
    # --- RSI Signal (Momentum) ---
    rsi_val = df['RSI_14'].iloc[-1]
    rsi_val_str = f"{rsi_val:.2f}"
    if rsi_val > 70:
        signals['RSI (14)'] = ('Bearish', f"Overbought ({rsi_val_str})", rsi_val_str)
    elif rsi_val < 30:
        signals['RSI (14)'] = ('Bullish', f"Oversold ({rsi_val_str})", rsi_val_str)
    else:
        signals['RSI (14)'] = ('Neutral', f"Mid-Range ({rsi_val_str})", rsi_val_str)

    # --- MACD Signal (Momentum/Trend) ---
    macd_hist = df['MACDh_12_26_9'].iloc[-1]
    macd_hist_str = f"{macd_hist:.4f}"
    if macd_hist > 0:
        signals['MACD (12,26,9)'] = ('Bullish', f"Histogram > 0 ({macd_hist_str})", macd_hist_str)
    elif macd_hist < 0:
        signals['MACD (12,26,9)'] = ('Bearish', f"Histogram < 0 ({macd_hist_str})", macd_hist_str)
    else:
        signals['MACD (12,26,9)'] = ('Neutral', f"Histogram ≈ 0", macd_hist_str)
        
    # --- Bollinger Bands Signal (Volatility) ---
    try:
        upper_band_col = next(col for col in df.columns if col.startswith('BBU_'))
        lower_band_col = next(col for col in df.columns if col.startswith('BBL_'))
        
        upper_band = df[upper_band_col].iloc[-1]
        lower_band = df[lower_band_col].iloc[-1]
        
        if close > upper_band:
            signals['Bollinger Bands (20,2)'] = ('Bearish', f"Price above Upper Band ({close_str} > {upper_band:.0f})", close_str)
        elif close < lower_band:
            signals['Bollinger Bands (20,2)'] = ('Bullish', f"Price below Lower Band ({close_str} < {lower_band:.0f})", close_str)
        else:
            signals['Bollinger Bands (20,2)'] = ('Neutral', f"Price inside Bands ({close_str})", close_str)
            
    except StopIteration:
        signals['Bollinger Bands (20,2)'] = ('Neutral', 'Error: BB Column not found.', 'N/A')
        
    # --- Stochastic Oscillator Signal (Momentum) ---
    try:
        k = df['STOCHk_14_3_3'].iloc[-1]
        d = df['STOCHd_14_3_3'].iloc[-1]
        k_str = f"{k:.2f}"
        
        if k > 80:
            signals['Stochastic Oscillator (14,3,3)'] = ('Bearish', f"Overbought (%K={k_str})", k_str)
        elif k < 20:
            signals['Stochastic Oscillator (14,3,3)'] = ('Bullish', f"Oversold (%K={k_str})", k_str)
        else:
            if d > 80 and k < d: # In overbought and K crosses below D
                 signals['Stochastic Oscillator (14,3,3)'] = ('Bearish', f"K crossing D, top range", k_str)
            elif d < 20 and k > d: # In oversold and K crosses above D
                 signals['Stochastic Oscillator (14,3,3)'] = ('Bullish', f"K crossing D, bottom range", k_str)
            else:
                 signals['Stochastic Oscillator (14,3,3)'] = ('Neutral', f"Mid-Range (%K={k_str})", k_str)
    except KeyError:
        signals['Stochastic Oscillator (14,3,3)'] = ('Neutral', 'Error: Column not found.', 'N/A')

    # --- Commodity Channel Index (CCI) Signal ---
    try:
        cci_val = df[next(col for col in df.columns if col.startswith('CCI_'))].iloc[-1]
        cci_val_str = f"{cci_val:.2f}"
        if cci_val > 100:
            signals['Commodity Channel Index (CCI)'] = ('Bearish', f"Extreme Overbought ({cci_val_str})", cci_val_str)
        elif cci_val < -100:
            signals['Commodity Channel Index (CCI)'] = ('Bullish', f"Extreme Oversold ({cci_val_str})", cci_val_str)
        else:
            signals['Commodity Channel Index (CCI)'] = ('Neutral', f"Between -100 and +100 ({cci_val_str})", cci_val_str)
    except (KeyError, StopIteration):
        signals['Commodity Channel Index (CCI)'] = ('Neutral', 'Error: Column not found.', 'N/A')

    # --- Money Flow Index (MFI) Signal ---
    try:
        mfi_val = df[next(col for col in df.columns if col.startswith('MFI_'))].iloc[-1]
        mfi_val_str = f"{mfi_val:.2f}"
        if mfi_val > 80:
            signals['Money Flow Index (MFI)'] = ('Bearish', f"Overbought ({mfi_val_str})", mfi_val_str)
        elif mfi_val < 20:
            signals['Money Flow Index (MFI)'] = ('Bullish', f"Oversold ({mfi_val_str})", mfi_val_str)
        else:
            signals['Money Flow Index (MFI)'] = ('Neutral', f"Mid-Range ({mfi_val_str})", mfi_val_str)
    except (KeyError, StopIteration):
        signals['Money Flow Index (MFI)'] = ('Neutral', 'Error: Column not found.', 'N/A')

    # --- Williams %R Signal ---
    try:
        willr_val = df[next(col for col in df.columns if col.startswith('WMR_') or col.startswith('WILLR_'))].iloc[-1]
        willr_val_str = f"{willr_val:.2f}"
        if willr_val > -20: # -20 is the overbought threshold
            signals['Williams %R (14)'] = ('Bearish', f"Overbought ({willr_val_str})", willr_val_str)
        elif willr_val < -80: # -80 is the oversold threshold
            signals['Williams %R (14)'] = ('Bullish', f"Oversold ({willr_val_str})", willr_val_str)
        else:
            signals['Williams %R (14)'] = ('Neutral', f"Mid-Range ({willr_val_str})", willr_val_str)
    except (KeyError, StopIteration):
        signals['Williams %R (14)'] = ('Neutral', 'Error: Column not found.', 'N/A')

    # --- Average Directional Index (ADX) Signal (Trend Strength & Direction) ---
    try:
        adx = df[next(col for col in df.columns if col.startswith('ADX_'))].iloc[-1]
        di_plus = df[next(col for col in df.columns if col.startswith('DMP_'))].iloc[-1]
        di_minus = df[next(col for col in df.columns if col.startswith('DMM_'))].iloc[-1]
        
        adx_str = f"{adx:.2f}"
        strength = "Weak"
        if adx > 25: strength = "Strong"
        elif adx > 20: strength = "Developing"
            
        if di_plus > di_minus:
            signals['Average Directional Index (ADX)'] = ('Bullish', f"{strength} Bull Trend (+DI > -DI)", adx_str)
        elif di_minus > di_plus:
            signals['Average Directional Index (ADX)'] = ('Bearish', f"{strength} Bear Trend (-DI > +DI)", adx_str)
        else:
            signals['Average Directional Index (ADX)'] = ('Neutral', f"No Clear Trend Direction ({strength})", adx_str)

    except (KeyError, StopIteration):
        signals['Average Directional Index (ADX)'] = ('Neutral', 'Error: Column not found.', 'N/A')

    # --- On-Balance Volume (OBV) Signal (Simple trend check) ---
    try:
        obv_val = df['OBV'].iloc[-1]
        obv_prev = df['OBV'].iloc[-5] # Compare to 5 periods ago
        obv_val_str = f"{obv_val:,.0f}"
        
        if obv_val > obv_prev:
            signals['On-Balance Volume (OBV)'] = ('Bullish', "Volume increasing (OBV rising)", obv_val_str)
        elif obv_val < obv_prev:
            signals['On-Balance Volume (OBV)'] = ('Bearish', "Volume decreasing (OBV falling)", obv_val_str)
        else:
            signals['On-Balance Volume (OBV)'] = ('Neutral', "Volume flat or range-bound", obv_val_str)
    except KeyError:
        signals['On-Balance Volume (OBV)'] = ('Neutral', 'Error: Column not found.', 'N/A')

    # --- Ichimoku Cloud Signal (Simplistic Cloud Position) ---
    try:
        span_a = df[next(col for col in df.columns if col.startswith('ISA_'))].iloc[-1]
        span_b = df[next(col for col in df.columns if col.startswith('ISB_'))].iloc[-1]
        
        # Determine the cloud top and bottom
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        if close > cloud_top:
            signals['Ichimoku Cloud'] = ('Bullish', f"Price above Kumo Cloud ({close_str})", close_str)
        elif close < cloud_bottom:
            signals['Ichimoku Cloud'] = ('Bearish', f"Price below Kumo Cloud ({close_str})", close_str)
        else:
            signals['Ichimoku Cloud'] = ('Neutral', f"Price inside Kumo Cloud ({close_str})", close_str)

    except (KeyError, StopIteration):
        signals['Ichimoku Cloud'] = ('Neutral', 'Error: Column not found.', 'N/A')

    return signals


# --- Live Bitcoin Data Fetcher (Kraken Ticker) ---
@st.cache_data(ttl=15) 
def fetch_btc_data():
    """Fetches live Bitcoin price, 24h volume, and 24h change from Kraken public API."""
    try:
        response = requests.get(KRAKEN_TICKER_ENDPOINT, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('error'):
            error_msg = data['error'][0] if data['error'] else "Unknown Kraken API Error"
            st.error(f"Kraken API reported an error: {error_msg}")
            return None, None, None

        result_pairs = data.get('result')
        if not result_pairs:
            st.error("Kraken response 'result' field was empty.")
            return None, None, None
        
        pair_key = next(iter(result_pairs), None)
        if not pair_key:
            st.error("Kraken response 'result' field contained no market data.")
            return None, None, None

        btc_data = result_pairs.get(pair_key)
        
        price = float(btc_data['c'][0])
        # Volume is in the base asset (XBT)
        volume_xbt = float(btc_data['v'][1]) 
        open_price = float(btc_data['o'])
        change_percent = ((price - open_price) / open_price) * 100 if open_price else 0
        
        return price, volume_xbt, change_percent
    except Exception as e:
        st.error(f"Could not fetch live Bitcoin data from Kraken: {e}")
        return None, None, None

# --- Existing Core Logic: API Check with Exponential Backoff ---

def check_single_api(url, attempt=0):
    """
    Performs the API health check with exponential backoff.
    """
    start_time = time.time()
    
    if attempt > 0:
        delay = BASE_DELAY_SECONDS * (2 ** attempt)
        time.sleep(delay)

    try:
        response = requests.get(url, timeout=10)
        response_time_ms = round((time.time() - start_time) * 1000)

        if response.status_code == 200:
            data = response.json()
            
            if data.get('error') and data['error']:
                error_msg = data['error'][0] if data['error'] else "Unknown Kraken API Error"
                status = "API Error"
                result_detail = f"Kraken Error: {error_msg}"
                return status, result_detail, response_time_ms, {"error": error_msg}

            status = "UP"
            result_detail = f"OK ({response_time_ms}ms)"
            if isinstance(data, dict):
                # Check for the BlockCypher blockchain API
                if 'block_height' in data:
                    result_detail += f", Block: {data['block_height']:,}"
                elif 'unixtime' in data.get('result', {}):
                    result_detail += f", Kraken Time OK"
                
            return status, result_detail, response_time_ms, data
        
        status = "HTTP Error"
        if response.status_code == 451:
            result_detail = f"HTTP 451 - Geo-Blocked"
        else:
            result_detail = f"HTTP {response.status_code} ({response_time_ms}ms)"
            
        return status, result_detail, response_time_ms, {"error": f"HTTP {response.status_code}"}

    except requests.exceptions.Timeout:
        status = "Timeout"
        result_detail = "Request timed out."
        return status, result_detail, -1, {"error": result_detail}
    except requests.exceptions.ConnectionError:
        status = "Connection Failed"
        result_detail = "DNS or Network error."
        return status, result_detail, -1, {"error": result_detail}
    except requests.exceptions.RequestException as e:
        status = "Request Error"
        result_detail = str(e)
        return status, result_detail, -1, {"error": result_detail}
    except json.JSONDecodeError:
        status = "Invalid JSON"
        result_detail = "Response was not valid JSON."
        return status, result_detail, -1, {"error": result_detail}
    except Exception as e:
        status = "Unknown Error"
        result_detail = str(e)
        return status, result_detail, -1, {"error": result_detail}

def run_all_checks():
    """Iterates through all configured APIs, running the check with retries."""
    new_history_entries = []
    
    with st.status("Running all data feed checks...", expanded=True) as status_box:
        
        for i, config in enumerate(st.session_state.api_configs):
            
            st.write(f"Checking **{config['name']}** ({config['url']})...")
            
            final_status = "BROKEN"
            final_result = "Failed after all retries."
            final_time = -1
            final_data = {}
            
            for attempt in range(MAX_RETRIES + 1):
                status_box.update(label=f"Checking all data feed checks... (Attempt {attempt+1}/{MAX_RETRIES+1} for {config['name']})", state="running")
                
                status, result, res_time, data = check_single_api(config['url'], attempt)

                if status in ["UP", "HTTP Error", "Invalid JSON", "API Error"]:
                    final_status = status
                    final_result = result
                    final_time = res_time
                    final_data = data
                    break 
                
                if attempt == MAX_RETRIES:
                    final_status = "BROKEN"
                    final_result = result
                    final_time = -1
                    final_data = data
            
            st.session_state.api_configs[i]['status'] = final_status
            st.session_state.api_configs[i]['last_check'] = datetime.now()
            st.session_state.api_configs[i]['last_result'] = final_result
            st.session_state.api_configs[i]['last_data'] = final_data
            
            new_history_entries.append({
                'Time': datetime.now(), 
                'Feed': config['name'], 
                'Status': final_status, 
                'Response Time (ms)': final_time if final_time > 0 else None
            })

    if new_history_entries:
        new_df = pd.DataFrame(new_history_entries)
        st.session_state.history = pd.concat([new_df, st.session_state.history]).reset_index(drop=True)
        st.session_state.history = st.session_state.history.head(50)
        
    st.toast("✅ All Health Checks Complete.")


# --- Streamlit UI Layout ---

st.title("Bitcoin TA Signal Matrix & Feed Monitor")
st.markdown("Live data pulled from **Kraken API**. Health monitored with **exponential backoff**.")

# --- 0. Live Bitcoin Tracking ---
st.header("0. Live Bitcoin Metrics (Kraken API)")
btc_price, btc_volume_xbt, btc_change = fetch_btc_data()

if btc_price is not None:
    
    col_price, col_24h_change, col_volume = st.columns(3)
    
    # Format volume for readability (XBT)
    if btc_volume_xbt >= 1_000_000:
        volume_str = f"{btc_volume_xbt / 1_000_000:,.2f}M"
    elif btc_volume_xbt >= 1_000:
        volume_str = f"{btc_volume_xbt / 1_000:,.2f}K"
    else:
        volume_str = f"{btc_volume_xbt:,.0f}"

    btc_change_str = f"{btc_change:+.2f}%"

    col_price.metric(
        label="Current Price (BTC/USD)", 
        value=f"${btc_price:,.2f}", 
        delta=btc_change_str 
    )
    
    col_24h_change.metric(
        label="24h Change (%)",
        value=f"{btc_change:+.2f}%",
        delta_color="normal"
    )

    # FIX: Correct unit display for volume
    col_volume.metric(
        label="24h Volume (XBT)", 
        value=volume_str,
    )
    
    now_utc_plus_11 = datetime.now(UTC_PLUS_11).strftime('%H:%M:%S')
    st.caption(f"Data source: {KRAKEN_API_URL}. Fetched at {now_utc_plus_11} (UTC+{UTC_OFFSET_HOURS})")
else:
    st.warning("Bitcoin data not available from Kraken API. Check the connection or API health.")

st.markdown("---")

# --- 0.5 TA Configuration Controls (Sidebar) ---
with st.sidebar:
    st.header("Configuration")
    st.info("The configuration is currently in-memory and will reset when the Streamlit app is stopped and restarted.")

    st.subheader("TA Data Parameters")
    
    # User control for Timeframe (Interval)
    selected_timeframe_str = st.selectbox(
        "Timeframe (Interval)",
        options=list(KRAKEN_INTERVALS.keys()),
        index=4 # Default to '1 hour'
    )
    selected_interval_minutes = KRAKEN_INTERVALS[selected_timeframe_str]

    # User control for number of bars (Count)
    selected_bar_count = st.slider(
        "Number of Historical Bars",
        min_value=52, # Minimum needed for Ichimoku
        max_value=1000,
        value=300,
        step=50
    )


# --- Action Button ---
st.button("Run All Health Checks Now", on_click=run_all_checks, use_container_width=True, type="primary")

st.markdown("---")

# --- 3. Automated TA Signals (Tiled Matrix) ---
st.header(f"3. Automated TA Signal Matrix ({selected_timeframe_str})")
st.markdown("Consolidated signals (Bullish/Bearish/Neutral) for rapid analysis, showing current indicator value.")

historical_df = fetch_historical_data(selected_interval_minutes, selected_bar_count)
ta_signals = get_indicator_signal(historical_df)

if ta_signals:
    # Use columns to create the tile layout (4 tiles per row)
    num_signals = len(ta_signals)
    cols = st.columns(min(4, num_signals))
    
    # Define color map for the tiles
    color_map = {
        'Bullish': '#198754',  # Green
        'Bearish': '#dc3545',  # Red
        'Neutral': '#6c757d'   # Grey
    }
    
    # Sort signals alphabetically for consistent display
    sorted_signals = sorted(ta_signals.items(), key=lambda item: item[0])

    for i, (indicator_name, (signal, detail, value_str)) in enumerate(sorted_signals):
        
        # Determine the color and icon
        bg_color = color_map.get(signal, '#6c757d')
        icon = "🔺" if signal == "Bullish" else "🔻" if signal == "Bearish" else "⚫"
        
        # Apply custom HTML/Markdown for the colored tile effect
        tile_html = f"""
        <div style="
            background-color: {bg_color};
            padding: 15px;
            border-radius: 8px;
            color: white;
            text-align: center;
            margin-bottom: 10px;
            height: 120px; /* Increased height for better fit */
        ">
            <h5 style="margin: 0; font-size: 14px; opacity: 0.8;">{indicator_name}</h5>
            <p style="
                margin: 5px 0 5px 0; /* Adjusted vertical margin for spacing */
                font-weight: bold; 
                font-size: 20px; /* Slightly reduced font size for value */
            ">{value_str}</p>
            <p style="margin: 0; font-weight: bold; font-size: 16px;">{icon} {signal.upper()}</p>
        </div>
        """
        
        # Place the tile in the appropriate column
        with cols[i % 4]:
            st.markdown(tile_html, unsafe_allow_html=True)
            
else:
    st.warning("Not enough historical data to calculate indicators (min 52 bars). Try increasing the bar count or check data fetch status.")


st.markdown("---")

# --- 1. Feed Status Overview (Monitoring) ---
st.header("1. Critical Data Feed Health Check")

# Calculate metrics
total_checks = len(st.session_state.api_configs)
up_count = sum(1 for c in st.session_state.api_configs if c['status'] == 'UP')
broken_count = sum(1 for c in st.session_state.api_configs if c['status'] in ['BROKEN', 'Timeout', 'Connection Failed', 'Request Error', 'Unknown Error'])
error_count = sum(1 for c in st.session_state.api_configs if c['status'] in ['HTTP Error', 'Invalid JSON', 'API Error'])

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Monitored Feeds", total_checks)
col2.metric("Online (UP)", up_count, delta=f"{round(up_count/total_checks*100 if total_checks else 0)}%", delta_color="normal")
col3.metric("Critical (BROKEN)", broken_count, delta=f"{round(broken_count/total_checks*100 if total_checks else 0)}%", delta_color="inverse")
col4.metric("Error/Warning", error_count, delta_color="off")

st.markdown("### Latest Check Results")

# Display results in a stylish dataframe
df_status = pd.DataFrame(st.session_state.api_configs)
df_display = df_status[['name', 'status', 'last_result', 'last_check']].rename(
    columns={'name': 'Feed Name', 'status': 'Status', 'last_result': 'Details', 'last_check': 'Last Check Time'}
)

# Function to apply color based on status
def color_status(val):
    if val == 'UP':
        color = 'background-color: #d1e7dd' # Green-lite
    elif val in ['BROKEN', 'Timeout', 'Connection Failed', 'Request Error', 'Unknown Error', 'HTTP 451 - Geo-Blocked']:
        color = 'background-color: #f8d7da' # Red-lite
    elif val in ['HTTP Error', 'Invalid JSON', 'API Error']:
        color = 'background-color: #fff3cd' # Yellow-lite
    else:
        color = ''
    return color

# Style the dataframe for visual impact
st.dataframe(
    df_display.style.applymap(color_status, subset=['Status']),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# --- 2. Response Time History ---
st.header("2. Response Time History")
st.markdown("Displays the response time trend for all successful checks over the last 50 attempts.")

history_df = st.session_state.history[st.session_state.history['Response Time (ms)'] > 0]

if not history_df.empty:
    # Use st.line_chart for visualization
    st.line_chart(
        history_df,
        x='Time',
        y='Response Time (ms)',
        color='Feed' # Group lines by feed name
    )
else:
    st.info("No successful checks recorded yet. Run the checks to populate history.")

st.markdown("---")

# --- Configuration and Details (Sidebar) ---
with st.sidebar:
    
    st.subheader("API Details & Debug")
    
    selected_id = st.selectbox(
        "Select Feed for JSON Detail",
        options=[c['id'] for c in st.session_state.api_configs],
        format_func=lambda x: next(c['name'] for c in st.session_state.api_configs if c['id'] == x)
    )

    selected_config = next(c for c in st.session_state.api_configs if c['id'] == selected_id)
    
    st.markdown(f"**URL:** `{selected_config['url']}`")
    st.markdown(f"**Last Status:** `{selected_config['status']}`")
    
    st.markdown("#### Last Raw API Data/Error")
    
    last_data = selected_config.get('last_data', {})
    if last_data:
        st.json(last_data)
    else:
        st.code("No data available yet. Run the check.")
