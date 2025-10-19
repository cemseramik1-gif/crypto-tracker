import streamlit as st
import requests
import time
import json
import pandas as pd
from datetime import datetime, timezone, timedelta

# --- Configuration and Constants ---
MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1

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
    "1 day": 1440, # Default based on user request
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
    page_title="Bitcoin TA Prep & Feed Monitor",
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
    Kraken uses a 'since' parameter (start time) instead of a 'limit' (end time/count).
    We estimate the required 'since' timestamp to fetch the required bar count.
    """
    st.info(f"Fetching {count} bars of {interval_minutes}-minute data...")
    try:
        # Estimate the required start time (fetch 50% more data to ensure we get 'count' valid bars)
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

        # Find the market data key (e.g., 'XXBTZUSD') dynamically
        result_pairs = {k: v for k, v in data.get('result', {}).items() if k != 'last'}
        pair_key = next(iter(result_pairs), None)
        
        if not pair_key:
            st.warning("Kraken OHLC response contained no market data.")
            return None

        ohlc_data = result_pairs.get(pair_key)
        
        # Convert list of lists to DataFrame
        df = pd.DataFrame(ohlc_data, columns=[
            'Time', 'Open', 'High', 'Low', 'Close', 'VWAP', 'Volume', 'Count'
        ])
        
        # Convert types and set index
        df['Time'] = pd.to_datetime(df['Time'], unit='s', utc=True)
        # Convert OHLC values to float
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Limit to the most recent 'count' bars and drop the last (potentially partial) bar
        df = df.iloc[:-1].tail(count).set_index('Time')
        
        return df

    except Exception as e:
        st.error(f"Could not fetch historical data from Kraken: {e}")
        return None

# --- Technical Analysis (TA) Calculation ---

def calculate_rsi(df, period=14):
    """
    Calculates the Relative Strength Index (RSI) using the standard 14-period 
    Exponentially Weighted Moving Average (EWMA) method.
    """
    if df.empty or 'Close' not in df.columns or len(df) < period:
        return df
        
    # 1. Calculate price change
    df['Price Change'] = df['Close'].diff()
    
    # 2. Separate gains and losses
    df['Gain'] = df['Price Change'].apply(lambda x: x if x > 0 else 0)
    df['Loss'] = df['Price Change'].apply(lambda x: abs(x) if x < 0 else 0)
    
    # 3. Calculate Average Gain and Average Loss (Smoothed using EWMA - Wilder's method)
    # Pandas ewm(com=period-1, adjust=False) is equivalent to Wilder's smoothing
    df['Avg Gain'] = df['Gain'].ewm(com=period-1, adjust=False).mean()
    df['Avg Loss'] = df['Loss'].ewm(com=period-1, adjust=False).mean()
    
    # 4. Calculate Relative Strength (RS)
    df['RS'] = df['Avg Gain'] / df['Avg Loss']
    
    # 5. Calculate RSI
    df['RSI'] = 100 - (100 / (1 + df['RS']))
    
    return df

# --- Live Bitcoin Data Fetcher (Kraken Ticker) ---

@st.cache_data(ttl=15) # Cache price data for 15 seconds to be closer to "live"
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
        volume = float(btc_data['v'][1])
        open_price = float(btc_data['o'])
        change_percent = ((price - open_price) / open_price) * 100 if open_price else 0
        
        return price, volume, change_percent
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

st.title("Bitcoin TA Prep & Feed Monitor")
st.markdown("Live data pulled from **Kraken API**. Health monitored with **exponential backoff**.")

# --- 0. Live Bitcoin Tracking ---
st.header("0. Live Bitcoin Metrics (Kraken API)")
btc_price, btc_volume, btc_change = fetch_btc_data()

if btc_price is not None:
    
    col_price, col_24h_change, col_volume = st.columns(3)
    
    # Format volume for readability
    if btc_volume >= 1_000_000_000:
        volume_str = f"${btc_volume / 1_000_000_000:,.2f}B"
    elif btc_volume >= 1_000_000:
        volume_str = f"${btc_volume / 1_000_000:,.2f}M"
    else:
        volume_str = f"${btc_volume:,.0f}"

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

    col_volume.metric(
        label="24h Volume (BTC)", 
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
        index=6 # Default to '1 day'
    )
    selected_interval_minutes = KRAKEN_INTERVALS[selected_timeframe_str]

    # User control for number of bars (Count)
    selected_bar_count = st.slider(
        "Number of Historical Bars",
        min_value=50,
        max_value=1000,
        value=300,
        step=50
    )


# --- Action Button ---
st.button("Run All Health Checks Now", on_click=run_all_checks, use_container_width=True, type="primary")

st.markdown("---")


# --- 3. Technical Analysis ---
st.header(f"3. Technical Analysis ({selected_timeframe_str} Bars)")

# Fetch historical data based on user input
historical_df = fetch_historical_data(selected_interval_minutes, selected_bar_count)

if historical_df is not None and not historical_df.empty:
    
    # Run TA calculation
    ta_df = calculate_rsi(historical_df)
    
    # Extract the latest RSI value
    latest_rsi = ta_df['RSI'].iloc[-1]
    latest_close = ta_df['Close'].iloc[-1]
    
    col_rsi, col_data_count, col_chart = st.columns([1, 1, 3])
    
    with col_rsi:
        # Determine color for RSI metric
        delta_color = "off"
        if latest_rsi >= 70:
            delta_color = "inverse" # Red for overbought
        elif latest_rsi <= 30:
            delta_color = "normal"  # Green for oversold

        st.metric(
            label=f"Current RSI (14-period)",
            value=f"{latest_rsi:,.2f}",
            delta=f"Close: ${latest_close:,.2f}",
            delta_color=delta_color
        )
        st.caption("RSI > 70 is Overbought. RSI < 30 is Oversold.")

    with col_data_count:
        st.metric(
            label="Bars Analyzed",
            value=f"{len(ta_df)}",
            delta=f"Timeframe: {selected_timeframe_str}",
            delta_color="off"
        )
        st.caption(f"Data ends at {ta_df.index[-1].tz_convert(UTC_PLUS_11).strftime('%Y-%m-%d %H:%M:%S')} (UTC+{UTC_OFFSET_HOURS})")


    with col_chart:
        # Plot the closing price and RSI
        st.line_chart(ta_df[['Close', 'RSI']])
        st.caption("The top line is the closing price. The bottom line is the RSI.")
    
    with st.expander("Show Raw Historical Data and RSI Table"):
        st.dataframe(ta_df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']].tail(15), use_container_width=True)

else:
    st.warning(f"Could not load historical data for the selected parameters: {selected_timeframe_str}, {selected_bar_count} bars.")

st.markdown("---")

# --- 1. Feed Status Overview (Remaining sections preserved) ---
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

# --- 3. Configuration and Details (Sidebar) ---
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
