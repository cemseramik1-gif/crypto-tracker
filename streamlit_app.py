import streamlit as st
import requests
import time
import json
import pandas as pd
from datetime import datetime

# --- Configuration and Constants ---
MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1
# Note: The 'Blockchain (BTC BlockCypher)' is the free blockchain data API check.
INITIAL_CONFIG = [
    {"id": 1, "name": "Blockchain (BTC BlockCypher)", "url": "https://api.blockcypher.com/v1/btc/main", "status": "Pending", "last_check": None, "last_result": ""},
    {"id": 2, "name": "Financial (Mock Data)", "url": "https://jsonplaceholder.typicode.com/todos/1", "status": "Pending", "last_check": None, "last_result": ""},
    {"id": 3, "name": "Custom Feed Example (Good)", "url": "https://httpstat.us/200", "status": "Pending", "last_check": None, "last_result": ""},
]

# Configure the Streamlit page layout and title
st.set_page_config(
    page_title="Bitcoin & Data Feed Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State Initialization ---
if 'api_configs' not in st.session_state:
    st.session_state.api_configs = INITIAL_CONFIG
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Time', 'Feed', 'Status', 'Response Time (ms)'])

# --- New: Bitcoin Price Fetcher ---

@st.cache_data(ttl=60) # Cache price data for 60 seconds to avoid hitting API limits
def fetch_btc_price():
    """Fetches live Bitcoin price from CoinGecko's public API."""
    # Using the simple price endpoint for only Bitcoin
    price_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true"
    try:
        response = requests.get(price_url, timeout=5)
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except Exception as e:
        st.error(f"Could not fetch live Bitcoin price: {e}")
        return None

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
            st.session_state.api_configs[i]['last_data'] = final_data
            
            # Add to history
            new_history_entries.append({
                'Time': datetime.now(), 
                'Feed': config['name'], 
                'Status': final_status, 
                'Response Time (ms)': final_time if final_time > 0 else None
            })

        # --- Final Status Update ---
        # The line below caused a TypeError in the Streamlit environment.
        # status_box.update(label="All Checks Complete.", state="complete", icon="🎉")
        # We rely on the 'with st.status' block exiting successfully for the final state.
        
    # Append new history and manage size
    if new_history_entries:
        new_df = pd.DataFrame(new_history_entries)
        st.session_state.history = pd.concat([new_df, st.session_state.history]).reset_index(drop=True)
        # Keep only the last 50 entries
        st.session_state.history = st.session_state.history.head(50)
        
    # Provide a toast notification to confirm completion
    st.toast("✅ All Health Checks Complete.")


# --- Streamlit UI Layout ---

st.title("Bitcoin Data Dashboard (Price & Feed Health)")
st.markdown("A unified view for **live Bitcoin pricing** and critical **Bitcoin Blockchain API health statuses** monitored with exponential backoff.")

# --- 0. Live Bitcoin Tracking ---
st.header("0. Live Bitcoin Tracking (Price API)")
crypto_data = fetch_btc_price()

if crypto_data and isinstance(crypto_data, dict):
    
    # Use two columns for metrics and one for the timestamp
    col_price, col_24h_change, col_time = st.columns(3)
    
    # Extract Bitcoin data
    btc = crypto_data.get('bitcoin', {})
    btc_price = btc.get('usd')
    btc_change = btc.get('usd_24h_change')
    
    # Format delta
    btc_delta_str = f"{btc_change:.2f}%" if btc_change is not None else "N/A"

    if btc_price is not None:
        col_price.metric(
            label="Bitcoin (BTC) Price", 
            value=f"${btc_price:,.2f}", 
            delta=btc_delta_str 
        )
    
    if btc_change is not None:
        col_24h_change.metric(
            label="24h Change",
            value=f"{btc_change:.2f}%",
            delta_color="normal"
        )
    else:
        col_24h_change.metric(label="24h Change", value="N/A", delta_color="off")


    col_time.metric("Last Price Update", datetime.now().strftime("%H:%M:%S"))
    st.caption("Live price fetched from CoinGecko's public API.")
else:
    st.warning("Bitcoin price data not available. Check the upstream API status.")

st.markdown("---")

# --- Action Button ---
st.button("Run All Health Checks Now", on_click=run_all_checks, use_container_width=True, type="primary")

st.markdown("---")

# --- 1. Feed Status Overview ---
st.header("1. Critical Data Feed Health Check (Blockchain API)")

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
