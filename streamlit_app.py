import streamlit as st
import requests
import time
import json
import pandas as pd
from datetime import datetime

# --- Configuration and Constants ---
MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1

# Using the most stable Binance endpoint for live data fetching
BINANCE_API_URL = "https://api.binance.com"

# The INITIAL_CONFIG list for the health check section (Section 1)
INITIAL_CONFIG = [
    # 1. Critical free blockchain data API (BlockCypher)
    {"id": 1, "name": "Blockchain (BTC BlockCypher)", "url": "https://api.blockcypher.com/v1/btc/main", "status": "Pending", "last_check": None, "last_result": ""},
    # 2. Stable Binance API endpoint check
    {"id": 2, "name": "Binance API (Stable)", "url": f"{BINANCE_API_URL}/api/v3/ping", "status": "Pending", "last_check": None, "last_result": ""},
    # 3. Mock data feed example for general system health check
    {"id": 3, "name": "Financial (Mock Data)", "url": "https://jsonplaceholder.typicode.com/todos/1", "status": "Pending", "last_check": None, "last_result": ""},
]
# NOTE: To use a faster endpoint (e.g., api1), change BINANCE_API_URL above:
# BINANCE_API_URL = "https://api1.binance.com"


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

# --- New: Bitcoin Data Fetcher (Binance 24hr Ticker) ---

@st.cache_data(ttl=15) # Cache price data for 15 seconds to be closer to "live"
def fetch_btc_data():
    """Fetches live Bitcoin price, 24h volume, and 24h change from Binance public API."""
    ticker_url = f"{BINANCE_API_URL}/api/v3/ticker/24hr?symbol=BTCUSDT"
    try:
        response = requests.get(ticker_url, timeout=5)
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        
        # Extract required fields for TA preparation
        price = float(data.get('lastPrice', 0))
        volume = float(data.get('volume', 0))
        change_percent = float(data.get('priceChangePercent', 0))
        
        return price, volume, change_percent
    except Exception as e:
        st.error(f"Could not fetch live Bitcoin data from Binance: {e}")
        return None, None, None

# --- Existing Core Logic: API Check with Exponential Backoff ---

def check_single_api(url, attempt=0):
    """
    Performs the API health check with exponential backoff and returns the result.
    This is a pure function that does not interact with Streamlit UI directly.
    """
    start_time = time.time()
    
    # Wait using time.sleep() for backoff 
    if attempt > 0:
        delay = BASE_DELAY_SECONDS * (2 ** attempt)
        time.sleep(delay)

    try:
        # Send the request with a timeout
        response = requests.get(url, timeout=10)
        response_time_ms = round((time.time() - start_time) * 1000)

        if response.status_code == 200:
            # Success: Data feed is UP
            data = response.json()
            
            # Extract relevant info for status update
            status = "UP"
            result_detail = f"OK ({response_time_ms}ms)"
            if isinstance(data, dict):
                # Specific check for the BlockCypher blockchain API
                if 'block_height' in data:
                    result_detail += f", Block: {data['block_height']:,}"
                elif 'title' in data: # Example for JSONPlaceholder
                    result_detail += f", Title: {data['title'][:20]}..."
                
            return status, result_detail, response_time_ms, data
        
        # API responded, but with an error status (e.g., 404, 500)
        status = "HTTP Error"
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
        # Binance /api/v3/ping returns an empty dictionary, which json.JSONDecodeError handles
        # We manually check the status code for success on the PING endpoint
        if url.endswith("/api/v3/ping") and response.status_code == 200:
             status = "UP"
             result_detail = f"OK (Ping Success, {response_time_ms}ms)"
             return status, result_detail, response_time_ms, {}
        
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
    
    # Use st.status for a persistent progress report during the long operation
    with st.status("Running all data feed checks...", expanded=True) as status_box:
        
        for i, config in enumerate(st.session_state.api_configs):
            
            st.write(f"Checking **{config['name']}** ({config['url']})...")
            
            final_status = "BROKEN"
            final_result = "Failed after all retries."
            final_time = -1
            final_data = {}
            
            # --- Exponential Backoff Logic for one API ---
            for attempt in range(MAX_RETRIES + 1):
                status_box.update(label=f"Checking all data feed checks... (Attempt {attempt+1}/{MAX_RETRIES+1} for {config['name']})", state="running")
                
                status, result, res_time, data = check_single_api(config['url'], attempt)

                if status in ["UP", "HTTP Error", "Invalid JSON"]:
                    # An acceptable response (either good or a clear HTTP error) was received
                    final_status = status
                    final_result = result
                    final_time = res_time
                    final_data = data
                    break # Exit retry loop early on success or clear error
                
                # If status is Timeout or Connection Failed, continue to next retry (if available)
                if attempt == MAX_RETRIES:
                    final_status = "BROKEN"
                    final_result = result # Last error message
                    final_time = -1
                    final_data = data # Last error dict
            
            # --- Update Session State and History ---
            st.session_state.api_configs[i]['status'] = final_status
            st.session_state.api_configs[i]['last_check'] = datetime.now()
            st.session_state.api_configs[i]['last_result'] = final_result
            st.session_session.api_configs[i]['last_data'] = final_data
            
            # Add to history
            new_history_entries.append({
                'Time': datetime.now(), 
                'Feed': config['name'], 
                'Status': final_status, 
                'Response Time (ms)': final_time if final_time > 0 else None
            })

        # --- Final Status Update ---
        # The 'with st.status' block exiting successfully sets the final state.
        
    # Append new history and manage size
    if new_history_entries:
        new_df = pd.DataFrame(new_history_entries)
        st.session_state.history = pd.concat([new_df, st.session_state.history]).reset_index(drop=True)
        # Keep only the last 50 entries
        st.session_state.history = st.session_state.history.head(50)
        
    # Provide a toast notification to confirm completion
    st.toast("✅ All Health Checks Complete.")


# --- Streamlit UI Layout ---

st.title("Bitcoin Data Dashboard (TA Prep & Feed Health)")
st.markdown("Live data pulled from **Binance API**. Health monitored with **exponential backoff**.")

# --- 0. Live Bitcoin Tracking ---
st.header("0. Live Bitcoin Metrics (Binance API)")
btc_price, btc_volume, btc_change = fetch_btc_data()

if btc_price is not None:
    
    # Use three columns for metrics
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
        label="Current Price (BTCUSDT)", 
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
    
    st.caption(f"Data source: {BINANCE_API_URL}. Fetched at {datetime.now().strftime('%H:%M:%S')}")
else:
    st.warning("Bitcoin data not available from Binance API. Check the connection or API health.")

st.markdown("---")

# --- Action Button ---
st.button("Run All Health Checks Now", on_click=run_all_checks, use_container_width=True, type="primary")

st.markdown("---")

# --- 1. Feed Status Overview ---
st.header("1. Critical Data Feed Health Check")

# Calculate metrics
total_checks = len(st.session_state.api_configs)
up_count = sum(1 for c in st.session_state.api_configs if c['status'] == 'UP')
broken_count = sum(1 for c in st.session_state.api_configs if c['status'] in ['BROKEN', 'Timeout', 'Connection Failed', 'Request Error', 'Unknown Error'])
error_count = sum(1 for c in st.session_state.api_configs if c['status'] in ['HTTP Error', 'Invalid JSON'])

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
    elif val in ['BROKEN', 'Timeout', 'Connection Failed', 'Request Error', 'Unknown Error']:
        color = 'background-color: #f8d7da' # Red-lite
    elif val in ['HTTP Error', 'Invalid JSON']:
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
    st.header("Configuration")
    st.info("The configuration is currently in-memory and will reset when the Streamlit app is stopped and restarted.")

    # Display configuration details for inspection/debugging
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
    
    # Display the raw response data or error message
    last_data = selected_config.get('last_data', {})
    if last_data:
        st.json(last_data)
    else:
        st.code("No data available yet. Run the check.")
