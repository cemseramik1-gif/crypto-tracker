import streamlit as st
import requests # <-- ADDED: Need this for API calls
from datetime import datetime, timedelta, timezone

# --- GLOBAL CONFIGURATION ---

# 1. Kraken API Endpoint for Time
KRAKEN_API_URL = "https://api.kraken.com/0/public/Time"

# 2. Timezone Configuration for Australia (UTC+11)
UTC_OFFSET_HOURS = 11
UTC_OFFSET_HOTHOURS = f"{UTC_OFFSET_HOURS:02d}" # Formatted as "11"
UTC_PLUS_11 = timezone(timedelta(hours=UTC_OFFSET_HOURS)) # Target timezone object

# --- Time Fetching Function ---

@st.cache_data(ttl=15) # Cache the result for 15 seconds to avoid unnecessary calls
def fetch_kraken_time():
    """
    Fetches the official server time (unixtime) from Kraken,
    converts it to a human-readable string, and adjusts it to the UTC+11 timezone.
    """
    try:
        # 1. Make the API request
        response = requests.get(KRAKEN_API_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # 2. Extract Unix timestamp
        unixtime = data['result']['unixtime']
        
        # 3. Convert Unix timestamp to UTC datetime object
        # Note: fromtimestamp expects a POSIX timestamp (which is implicitly UTC)
        utc_dt = datetime.fromtimestamp(unixtime, tz=timezone.utc)
        
        # 4. Convert UTC datetime to the target timezone (UTC+11)
        target_dt = utc_dt.astimezone(UTC_PLUS_11)
        
        # 5. Format for display
        time_str = target_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        return time_str, True

    except Exception as e:
        # Handle API errors, connection issues, or JSON parsing problems
        return f"Error fetching Kraken time: {e}", False

# --- STREAMLIT APPLICATION LOGIC ---

def crypto_tracker_app():
    """Main function for the Crypto Tracker application."""
    st.set_page_config(
        page_title="Crypto Price Tracker",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.title("Simple Crypto Tracker")
    st.markdown("This is a placeholder to demonstrate the fix for the `NameError` and display live Kraken time.")
    
    # --- Fetch Live Time ---
    kraken_time_str, success = fetch_kraken_time()
    
    if not success:
        st.error("Could not fetch live Kraken server time.")
    
    # --- Simulate Main App Content ---
    
    st.sidebar.header("Settings")
    st.sidebar.selectbox("Select Asset", ["BTC/USD", "ETH/USD", "SOL/USD"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="BTC Price (Kraken)", value="$65,000.00", delta="+1.2%", delta_color="normal")
        st.metric(label="24h Volume", value="12,345 BTC")

    with col2:
        st.metric(label="ETH Price (Kraken)", value="$3,500.00", delta="-0.5%", delta_color="inverse")
        st.metric(label="Market Cap", value="$420B")

    # --- Footer/Caption (Now displays fetched Kraken time in UTC+11) ---
    
    st.markdown("---")
    st.caption(
        f"Data source: **Kraken Public API**. "
        f"Kraken Server Time (UTC+{UTC_OFFSET_HOTHOURS}): **{kraken_time_str}**"
    )
    
    st.markdown("---")
    st.info("Remember: In Python, variables must be defined before they are used!")


if __name__ == "__main__":
    crypto_tracker_app()
