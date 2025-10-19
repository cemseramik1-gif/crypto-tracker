import streamlit as st
import requests
import time
import json
import pandas as pd
import numpy as np
import math
from datetime import datetime, timezone, timedelta
import pandas_ta as ta

# --- Configuration and Constants ---
MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1

# TIMEZONE CONFIG: Set target display timezone to UTC+11 (Australia/Sydney)
UTC_OFFSET_HOURS = 11 
UTC_PLUS_11 = timezone(timedelta(hours=UTC_OFFSET_HOURS)) # Target timezone object
UTC_OFFSET_HOTHOURS = f"{UTC_OFFSET_HOURS:02d}" # Formatted as a string "11"

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

# The INITIAL_CONFIG list for the health check section
INITIAL_CONFIG = [
    {"id": 1, "name": "Blockchain (BTC BlockCypher)", "url": "https://api.blockcypher.com/v1/btc/main", "status": "Pending", "last_check": None, "last_result": ""},
    {"id": 2, "name": "Kraken API (Stable Time)", "url": f"{KRAKEN_API_URL}/0/public/Time", "status": "Pending", "last_check": None, "last_result": ""},
    {"id": 3, "name": "Kraken API (Live Ticker)", "url": KRAKEN_TICKER_ENDPOINT, "status": "Pending", "last_check": None, "last_result": ""},
]

# --- Comprehensive Indicator Glossary Data (Python Structure) ---
INDICATOR_GLOSSARY = [
    # --- Technical Analysis (TA) Indicators ---
    {"name": "Relative Strength Index (RSI)", "type": "TA", "description": "Measures the speed and change of price movements. Primarily identifies overbought (>70) and oversold (<30) conditions. Bullish signal now triggers above 55 to catch rising momentum sooner.", "calculation": "RSI = 100 - [100 / (1 + Average Gain / Average Loss)]", "purpose": "Momentum & Reversal Signals"},
    {"name": "Moving Average Convergence Divergence (MACD)", "type": "TA", "description": "A trend-following momentum indicator showing the relationship between two EMAs (12 and 26). The **Histogram** (MACD Line - Signal Line) crossing zero is the key signal.", "calculation": "MACD Line = 12-period EMA - 26-period EMA; Signal Line = 9-period EMA of MACD Line", "purpose": "Trend Direction & Momentum"},
    {"name": "Bollinger Bands (BB)", "type": "TA", "description": "A volatility indicator consisting of a moving average (Median Band) and two standard deviation bands. Tight bands suggest low volatility; price outside bands suggests high volatility or extremes.", "calculation": "Middle Band: 20-day SMA; Outer Bands: SMA ± (2 * Standard Deviation)", "purpose": "Volatility & Price Extremes"},
    {"name": "Exponential Moving Average (EMA)", "type": "TA", "description": "A type of moving average that gives more weight to recent data points, making it react faster to price changes than an SMA. Used for trend smoothing and dynamic support/resistance.", "calculation": "EMA = (Price * Multiplier) + (EMA yesterday * (1 - Multiplier))", "purpose": "Trend Smoothing & Speed"},
    {"name": "Stochastic Oscillator", "type": "TA", "description": "A momentum indicator comparing a particular closing price of a security to a range of its prices over a certain period of time. Used to identify overbought (>80) and oversold (<20) conditions.", "calculation": "%K = 100 * ((Close - Low) / (High - Low))", "purpose": "Momentum & Reversal Signals"},
    {"name": "Commodity Channel Index (CCI)", "type": "TA", "description": "Measures the current price level relative to an average price level over a given period. Used to identify price extremes: +100 and -100 are common thresholds for strength/weakness.", "calculation": "CCI = (Price - MA) / (0.015 * Mean Deviation)", "purpose": "Trend Strength & Price Extremes"},
    {"name": "Williams %R", "type": "TA", "description": "A momentum indicator that measures overbought and oversold levels. It is similar to the Stochastic Oscillator but inverted, typically ranging from 0 to -100. Readings between -80 and -100 are oversold.", "calculation": "%R = ((Highest High - Close) / (Highest High - Lowest Low)) * -100", "purpose": "Momentum & Reversal Signals"},
    {"name": "Average Directional Index (ADX)", "type": "TA", "description": "Measures the **strength** of a trend, not its direction. ADX values above 25 indicate a strong trend. The direction (+DI or -DI) tells you if the trend is bullish or bearish.", "calculation": "Uses the smoothed averages of Positive Directional Indicator (+DI) and Negative Directional Indicator (-DI).", "purpose": "Trend Strength"},
    {"name": "Ichimoku Cloud", "type": "TA", "description": "A comprehensive, all-in-one indicator that shows support and resistance, momentum, and trend direction. The Cloud (Kumo) acts as a powerful dynamic support/resistance zone.", "calculation": "Composed of five lines, including the Conversion Line, Base Line, Lagging Span, Leading Span A, and Leading Span B.", "purpose": "Trend Direction, Momentum, S/R"},
    {"name": "Chaikin Money Flow (CMF)", "type": "TA", "description": "Measures the amount of Money Flow Volume over a specific period. Positive suggests accumulation (buying pressure); negative suggests distribution (selling pressure).", "calculation": "CMF = 20-period sum of Money Flow Volume / 20-period sum of Volume", "purpose": "Volume & Accumulation/Distribution"},
    {"name": "On-Balance Volume (OBV)", "type": "TA", "description": "Measures buying and selling pressure cumulatively. It is a running total of volume that tracks whether that volume is flowing into or out of a security. Rising OBV confirms a price increase; divergence signals weakness.", "calculation": "OBV = OBV previous + Volume (if close > close previous) or - Volume (if close < close previous)", "purpose": "Volume Confirmation"},
    {"name": "Volume-Weighted Average Price (VWAP)", "type": "TA", "description": "The average price of an asset weighted by its trading volume. It often serves as a benchmark for institutional investors. Price above/below VWAP is Bullish/Bearish from an institutional perspective.", "calculation": "VWAP = Sum(Price * Volume) / Sum(Volume)", "purpose": "Institutional Benchmark & Intraday Trend"},
    {"name": "Average True Range (ATR)", "type": "TA", "description": "Measures market **volatility**. It shows the average range between High, Low, and previous Close over a period. It's non-directional but essential for setting stop losses and target profit levels.", "calculation": "ATR = Smoothed average of True Range (TR)", "purpose": "Volatility & Risk Management"},
    {"name": "Money Flow Index (MFI)", "type": "TA", "description": "A momentum indicator that uses both price and volume to identify overbought and oversold conditions. Similar to RSI, but incorporates volume data, making it a stronger measure of buying/selling pressure.", "calculation": "MFI = 100 - [100 / (1 + Money Flow Ratio)]", "purpose": "Volume-weighted Momentum & Reversal Signals"},
    
    # --- Blockchain / On-Chain Indicators ---
    {"name": "Block Height & Block Time", "type": "On-Chain", "description": "The **Block Height** is the total number of blocks in the blockchain. The time between blocks (Block Time) is a measure of network health and confirmation speed. Rising block height confirms continuous network operation, which underpins the asset's existence.", "calculation": "Simple count of blocks. Time is calculated from block timestamps.", "purpose": "Network Health & Confirmation"},
    {"name": "NVT Ratio (Network Value to Transactions)", "type": "On-Chain", "description": "Often called the 'P/E Ratio for Crypto,' NVT compares the asset's **Market Cap (Network Value)** to its daily **Transaction Volume**. High NVT suggests overvalued (low network usage), low NVT suggests undervalued (high network usage).", "calculation": "NVT Ratio = Market Capitalization / Daily Transferred Value (in USD)", "purpose": "Valuation, Overbought/Oversold"},
    {"name": "MVRV Z-Score (Market Value to Realized Value)", "type": "On-Chain", "description": "MVRV Z-Score is a cyclical indicator. **Market Value (MV)** is Market Cap; **Realized Value (RV)** is the sum of all money paid for coins when they last moved. High Z-Score (far above 0) signals a market top; low Z-Score (far below 0) signals a market bottom.", "calculation": "MVRV Z-Score = (Market Value - Realized Value) / Standard Deviation of Market Value", "purpose": "Cyclical Market Tops/Bottoms"},
    {"name": "SOPR (Spent Output Profit Ratio)", "type": "On-Chain", "description": "Compares the value of coins when they were last moved (Realized Value) to their value when they are being spent (Market Value). SOPR > 1 means holders are selling at a profit; SOPR < 1 means they are selling at a loss.", "calculation": "SOPR = (Current Price / Price when the UTXO was created)", "purpose": "Market Profitability and Sentiment"},
    {"name": "Active Addresses / Entities", "type": "On-Chain", "description": "The number of unique blockchain addresses or estimated entities (users) active in a given period. Rising active addresses indicates increasing network adoption, utility, and demand.", "calculation": "Count of unique addresses/entities that participated in a transaction.", "purpose": "Network Utility & Adoption"},
    {"name": "Exchange Netflow", "type": "On-Chain", "description": "The net difference between total coins moving **onto** exchanges (selling pressure) and total coins moving **off** exchanges (accumulation). Negative netflow is typically bullish.", "calculation": "Coins Inflow - Coins Outflow", "purpose": "Short-Term Selling Pressure & Liquidity"},
    {"name": "Whale Wallet Balances", "type": "On-Chain", "description": "Tracks the aggregated balance of wallets holding a significant threshold of the asset. Accumulation by whales suggests confidence; distribution suggests caution.", "calculation": "Sum of balances across high-value addresses.", "purpose": "Large Investor Sentiment & Supply Shock"}
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

# --- Utility Function for Robust Column Lookup ---

def safe_column_lookup(df, prefix):
    """Safely finds the first column in the DataFrame that starts with the given prefix."""
    try:
        # Use a generator expression to find the column
        return next(col for col in df.columns if col.startswith(prefix))
    except StopIteration:
        # Return None if no column is found
        return None

# --- HTML Generator for Interactive Glossary ---

def get_glossary_html(indicator_data):
    """
    Generates the complete HTML, CSS, and JavaScript for the interactive glossary.
    Embeds the indicator data as a JSON string for client-side processing.
    """
    # Dump Python list to JSON string for JS consumption
    json_data = json.dumps(indicator_data)

    html_content = f"""
    <!-- Load Lucide Icons for collapse/expand arrows -->
    <script src="https://unpkg.com/lucide@latest"></script>
    
    <div class="max-w-full mx-auto bg-white rounded-xl shadow-lg p-4 md:p-6 mb-8">

        <header class="mb-6 border-b-2 border-indigo-400 pb-3">
            <h2 class="text-2xl font-bold text-gray-800">Comprehensive TA & On-Chain Glossary</h2>
            <p class="text-sm text-gray-500">Includes Technical Analysis (TA) and fundamental Blockchain/On-Chain metrics.</p>
        </header>

        <!-- Search and Filter Controls -->
        <div class="mb-6 space-y-4 sm:space-y-0 sm:flex sm:gap-4">
            <input type="text" id="search-input" placeholder="Search indicator name or description..." class="flex-grow p-3 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500 shadow-sm">
            <select id="filter-select" class="p-3 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500 shadow-sm w-full sm:w-auto">
                <option value="all">Filter by Type (All)</option>
                <option value="TA">Technical Analysis (TA)</option>
                <option value="On-Chain">Blockchain/On-Chain</option>
            </select>
        </div>

        <!-- Indicator List Container -->
        <div id="indicator-list" class="space-y-3">
            <!-- Indicators inserted by JS -->
        </div>
    </div>

    <script>
        // --- Indicator Data Embedded from Python ---
        const indicators = {json_data};
        
        // --- Core Functions (Optimized for Streamlit embedding) ---

        const listContainer = document.getElementById('indicator-list');
        const searchInput = document.getElementById('search-input');
        const filterSelect = document.getElementById('filter-select');

        function renderIndicators(list) {{
            listContainer.innerHTML = '';
            if (list.length === 0) {{
                listContainer.innerHTML = '<p class="text-center text-gray-500 p-4 text-lg">No indicators found matching your criteria.</p>';
                return;
            }}

            list.forEach((indicator) => {{
                // Generate a unique ID for the dynamic header/content pair
                const uniqueId = indicator.name.replace(/[^a-zA-Z0-9]/g, '');
                
                const typeClass = indicator.type === 'TA' ? 'bg-indigo-100 text-indigo-700' : 'bg-green-100 text-green-700';
                const borderColor = indicator.type === 'TA' ? 'border-indigo-500' : 'border-green-500';

                const element = document.createElement('div');
                element.className = 'bg-white border border-gray-200 rounded-lg shadow-sm overflow-hidden';
                element.innerHTML = `
                    <div id="header-${uniqueId}" data-id="${uniqueId}" class="indicator-header flex justify-between items-center p-3 md:p-4 bg-gray-50 hover:bg-gray-100 transition duration-150 ease-in-out select-none rounded-lg">
                        <div class="flex items-center space-x-3">
                            <span class="px-2 py-0.5 text-xs font-semibold rounded-full ${typeClass}">
                                ${indicator.type}
                            </span>
                            <h3 class="text-base font-semibold text-gray-800">${indicator.name}</h3>
                        </div>
                        <svg data-lucide="chevron-down" class="text-gray-500 w-5 h-5 transition-transform duration-300"></svg>
                    </div>

                    <div id="content-${uniqueId}" class="indicator-content px-4 md:px-5">
                        <div class="text-gray-700 space-y-3">
                            <p>
                                <span class="font-semibold text-gray-900 block mb-1">Description:</span>
                                ${indicator.description}
                            </p>
                            <p class="border-l-4 ${borderColor} pl-3 py-1 bg-gray-50 rounded-r text-sm">
                                <span class="font-bold text-gray-900 block">Primary Purpose:</span>
                                ${indicator.purpose}
                            </p>
                            <p class="text-sm">
                                <span class="font-semibold text-gray-900 block mb-1">Calculation/Concept:</span>
                                <code>${indicator.calculation}</code>
                            </p>
                        </div>
                    </div>
                `;
                listContainer.appendChild(element);
            }});

            // Set up click listeners after all elements are created
            document.querySelectorAll('.indicator-header').forEach(header => {{
                const id = header.getAttribute('data-id');
                const content = document.getElementById(`content-${id}`);
                const icon = header.querySelector('svg');

                if (content && icon) {{
                    // Add event listener for collapse/expand
                    header.addEventListener('click', () => {{
                        const isExpanded = content.classList.toggle('active');
                        icon.style.transform = isExpanded ? 'rotate(180deg)' : 'rotate(0deg)';
                    }});
                }}
            }});
            
            // Re-render Lucide icons (MUST be called after elements are added to DOM)
            if (typeof lucide !== 'undefined') {{
                lucide.createIcons();
            }}
        }}

        function filterAndSearch() {{
            const searchTerm = searchInput.value.toLowerCase().trim();
            const filterType = filterSelect.value;

            const filteredList = indicators.filter(indicator => {{
                const matchesSearch = indicator.name.toLowerCase().includes(searchTerm) ||
                                      indicator.description.toLowerCase().includes(searchTerm);
                const matchesFilter = filterType === 'all' || indicator.type === filterType;

                return matchesSearch && matchesFilter;
            }});

            renderIndicators(filteredList);
        }}

        // Initial setup on script load
        // Check if elements are available before binding events (important for Streamlit render cycle)
        if (listContainer && searchInput && filterSelect) {{
            renderIndicators(indicators);
            searchInput.addEventListener('input', filterAndSearch);
            filterSelect.addEventListener('change', filterAndSearch);
        }}
    </script>
    
    <!-- Inline CSS required for Streamlit custom components -->
    <style>
        .indicator-header {{
            cursor: pointer;
            transition: background-color 0.2s;
        }}
        .indicator-header:hover {{
            background-color: #e2e8f0; /* Slate 200 */
        }}
        .indicator-content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.4s ease-in-out, padding 0.4s ease-in-out;
            background-color: #f9fafb; /* Lightest gray background */
        }}
        .indicator-content.active {{
            max-height: 1000px; 
            padding-top: 1rem;
            padding-bottom: 1rem;
        }}
        code {{
            background-color: #eef2ff; /* Indigo 50 */
            padding: 2px 4px;
            border-radius: 4px;
            color: #4338ca; /* Indigo 700 */
            font-size: 0.85rem;
        }}
    </style>
    """
    return html_content


# --- Historical Data Fetcher (Kraken OHLC) ---

@st.cache_data(ttl=300) # Cache historical data for 5 minutes
def fetch_historical_data(interval_minutes, count=300):
    """
    Fetches historical OHLC data from Kraken.
    """
    try:
        seconds_per_interval = interval_minutes * 60
        # Fetch slightly more data to ensure we have enough for 200 EMA and Ichimoku lookbacks
        approx_start_time = int(time.time() - (count * seconds_per_interval * 2)) 

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
        for col in ['Open', 'High', 'Low', 'Low', 'Close', 'Volume', 'VWAP']: # Added VWAP here to ensure it's a float
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
    Calculates all specified indicators, groups them by type, and determines a signal.
    Returns a dictionary of dictionaries: {Group: {Indicator: (signal, detail, value)}}
    """
    
    # Needs at least 52 bars for Ichimoku to be calculated properly.
    if df is None or df.empty or len(df) < 52: 
        return None
    
    # --- 1. Calculate ALL indicators using pandas_ta ---
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
    df.ta.cmf(append=True) 
    df.ta.atr(append=True) 
    
    # EMA calculations
    df.ta.ema(length=9, append=True)
    df.ta.ema(length=21, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    
    # --- 2. Define Signal Logic (Grouped by type) ---
    signals = {
        'Momentum': {},
        'Trend': {},
        'Volume': {},
        'Volatility': {}
    }
    
    close = df['Close'].iloc[-1]
    close_str = f"${close:,.0f}" # Formatted for price indicators
    
    # --- Momentum Indicators ---
    
    # RSI
    rsi_col = safe_column_lookup(df, 'RSI_')
    if rsi_col:
        rsi_val = df[rsi_col].iloc[-1]
        
        if math.isnan(rsi_val):
            signals['Momentum']['RSI (14)'] = ('Neutral', 'N/A (Data Not Ready)', 'N/A')
        else:
            rsi_val_str = f"{rsi_val:.2f}"
            
            if rsi_val > 70:
                signals['Momentum']['RSI (14)'] = ('Bearish', f"Overbought (>70) ({rsi_val_str})", rsi_val_str)
            elif rsi_val < 30:
                signals['Momentum']['RSI (14)'] = ('Bullish', f"Oversold (<30) ({rsi_val_str})", rsi_val_str)
            elif rsi_val >= 55: 
                signals['Momentum']['RSI (14)'] = ('Bullish', f"Strong Momentum (>55) ({rsi_val_str})", rsi_val_str)
            elif rsi_val <= 45: 
                signals['Momentum']['RSI (14)'] = ('Bearish', f"Weak Momentum (<45) ({rsi_val_str})", rsi_val_str)
            else:
                signals['Momentum']['RSI (14)'] = ('Neutral', f"Mid-Range ({rsi_val_str})", rsi_val_str)
    else:
        signals['Momentum']['RSI (14)'] = ('Neutral', 'N/A (Error)', 'N/A')

    # Stochastic Oscillator
    stoch_k_col = safe_column_lookup(df, 'STOCHk_')
    if stoch_k_col:
        k = df[stoch_k_col].iloc[-1]
        
        if math.isnan(k):
             signals['Momentum']['Stochastic (14,3,3)'] = ('Neutral', 'N/A (Data Not Ready)', 'N/A')
        else:
            k_str = f"{k:.2f}"
            if k > 80:
                signals['Momentum']['Stochastic (14,3,3)'] = ('Bearish', f"Overbought (%K={k_str})", k_str)
            elif k < 20:
                signals['Momentum']['Stochastic (14,3,3)'] = ('Bullish', f"Oversold (%K={k_str})", k_str)
            else:
                signals['Momentum']['Stochastic (14,3,3)'] = ('Neutral', f"Mid-Range (%K={k_str})", k_str)
    else:
        signals['Momentum']['Stochastic (14,3,3)'] = ('Neutral', 'N/A (Error)', 'N/A')
        
    # CCI
    cci_col = safe_column_lookup(df, 'CCI_')
    if cci_col:
        cci_val = df[cci_col].iloc[-1]
        
        if math.isnan(cci_val):
             signals['Momentum']['CCI (14)'] = ('Neutral', 'N/A (Data Not Ready)', 'N/A')
        else:
            cci_val_str = f"{cci_val:.2f}"
            if cci_val > 100:
                signals['Momentum']['CCI (14)'] = ('Bearish', f"Extreme Overbought ({cci_val_str})", cci_val_str)
            elif cci_val < -100:
                signals['Momentum']['CCI (14)'] = ('Bullish', f"Extreme Oversold ({cci_val_str})", cci_val_str)
            else:
                signals['Momentum']['CCI (14)'] = ('Neutral', f"Between -100 and +100 ({cci_val_str})", cci_val_str)
    else:
        signals['Momentum']['CCI (14)'] = ('Neutral', 'N/A (Error)', 'N/A')
        
    # MFI
    mfi_col = safe_column_lookup(df, 'MFI_')
    if mfi_col:
        mfi_val = df[mfi_col].iloc[-1]
        
        if math.isnan(mfi_val):
             signals['Momentum']['MFI (14)'] = ('Neutral', 'N/A (Data Not Ready)', 'N/A')
        else:
            mfi_val_str = f"{mfi_val:.2f}"
            if mfi_val > 80:
                signals['Momentum']['MFI (14)'] = ('Bearish', f"Overbought ({mfi_val_str})", mfi_val_str)
            elif mfi_val < 20:
                signals['Momentum']['MFI (14)'] = ('Bullish', f"Oversold ({mfi_val_str})", mfi_val_str)
            else:
                signals['Momentum']['MFI (14)'] = ('Neutral', f"Mid-Range ({mfi_val_str})", mfi_val_str)
    else:
        signals['Momentum']['MFI (14)'] = ('Neutral', 'N/A (Error)', 'N/A')

    # Williams %R
    willr_col = safe_column_lookup(df, 'WMR_') or safe_column_lookup(df, 'WILLR_')
    if willr_col:
        willr_val = df[willr_col].iloc[-1]
        
        if math.isnan(willr_val):
             signals['Momentum']['Williams %R (14)'] = ('Neutral', 'N/A (Data Not Ready)', 'N/A')
        else:
            willr_val_str = f"{willr_val:.2f}"
            if willr_val > -20: 
                signals['Momentum']['Williams %R (14)'] = ('Bearish', f"Overbought ({willr_val_str})", willr_val_str)
            elif willr_val < -80: 
                signals['Momentum']['Williams %R (14)'] = ('Bullish', f"Oversold ({willr_val_str})", willr_val_str)
            else:
                signals['Momentum']['Williams %R (14)'] = ('Neutral', f"Mid-Range ({willr_val_str})", willr_val_str)
    else:
        signals['Momentum']['Williams %R (14)'] = ('Neutral', 'N/A (Error)', 'N/A')


    # --- Trend Indicators ---

    # MACD (NOW DISPLAYS ALL 3 VALUES)
    macd_hist_col = safe_column_lookup(df, 'MACDh_')
    macd_line_col = safe_column_lookup(df, 'MACD_')
    macd_signal_col = safe_column_lookup(df, 'MACDs_')
    
    if macd_hist_col and macd_line_col and macd_signal_col:
        macd_hist = df[macd_hist_col].iloc[-1]
        macd_line = df[macd_line_col].iloc[-1]
        macd_signal = df[macd_signal_col].iloc[-1]
        
        if math.isnan(macd_hist) or math.isnan(macd_line) or math.isnan(macd_signal):
             signals['Trend']['MACD (12,26,9)'] = ('Neutral', 'N/A (Data Not Ready)', 'N/A')
        else:
            # Format the combined string for the Value column (M: MACD Line, S: Signal Line, H: Histogram)
            macd_value_str = f"M:{macd_line:,.4f} | S:{macd_signal:,.4f} | H:{macd_hist:,.4f}"
            
            # Signal logic remains based on Histogram
            if macd_hist > 0:
                signals['Trend']['MACD (12,26,9)'] = ('Bullish', f"Histogram > 0 (Bullish Cross)", macd_value_str)
            elif macd_hist < 0:
                signals['Trend']['MACD (12,26,9)'] = ('Bearish', f"Histogram < 0 (Bearish Cross)", macd_value_str)
            else:
                signals['Trend']['MACD (12,26,9)'] = ('Neutral', f"Histogram ≈ 0 (Neutral/Flat)", macd_value_str)
    else:
        signals['Trend']['MACD (12,26,9)'] = ('Neutral', 'N/A (Error in calculation)', 'N/A')


    # ADX
    adx_col = safe_column_lookup(df, 'ADX_')
    di_plus_col = safe_column_lookup(df, 'DMP_')
    di_minus_col = safe_column_lookup(df, 'DMM_')

    if adx_col and di_plus_col and di_minus_col:
        adx = df[adx_col].iloc[-1]
        di_plus = df[di_plus_col].iloc[-1]
        di_minus = df[di_minus_col].iloc[-1]
        
        if math.isnan(adx) or math.isnan(di_plus) or math.isnan(di_minus):
            signals['Trend']['ADX (14)'] = ('Neutral', 'N/A (Data Not Ready)', 'N/A')
        else:
            adx_str = f"{adx:.2f}"
            strength = "Weak"
            if adx > 25: strength = "Strong"
            elif adx > 20: strength = "Developing"
            
            if di_plus > di_minus:
                signals['Trend']['ADX (14)'] = ('Bullish', f"{strength} Bull Trend (+DI > -DI)", adx_str)
            elif di_minus > di_plus:
                signals['Trend']['ADX (14)'] = ('Bearish', f"{strength} Bear Trend (-DI > +DI)", adx_str)
            else:
                signals['Trend']['ADX (14)'] = ('Neutral', f"No Clear Trend Direction ({strength})", adx_str)
    else:
        signals['Trend']['ADX (14)'] = ('Neutral', 'N/A (Error in calculation)', 'N/A')


    # Ichimoku Cloud
    span_a_col = safe_column_lookup(df, 'ISA_')
    span_b_col = safe_column_lookup(df, 'ISB_')

    if span_a_col and span_b_col:
        span_a = df[span_a_col].iloc[-1]
        span_b = df[span_b_col].iloc[-1]
        
        if math.isnan(span_a) or math.isnan(span_b):
             signals['Trend']['Ichimoku Cloud'] = ('Neutral', 'N/A (Data Not Ready)', 'N/A')
        else:
            cloud_top = max(span_a, span_b)
            cloud_bottom = min(span_a, span_b)
            if close > cloud_top:
                signals['Trend']['Ichimoku Cloud'] = ('Bullish', f"Price above Kumo Cloud ({close_str})", close_str)
            elif close < cloud_bottom:
                signals['Trend']['Ichimoku Cloud'] = ('Bearish', f"Price below Kumo Cloud ({close_str})", close_str)
            else:
                signals['Trend']['Ichimoku Cloud'] = ('Neutral', f"Price inside Kumo Cloud ({close_str})", close_str)
    else:
        signals['Trend']['Ichimoku Cloud'] = ('Neutral', 'N/A (Error)', 'N/A')
    
    # EMA Signals
    ema_lengths = [9, 21, 50, 200]
    for length in ema_lengths:
        try:
            ema_col = f'EMA_{length}'
            ema_val = df[ema_col].iloc[-1]
            
            if math.isnan(ema_val):
                signals['Trend'][f'EMA ({length})'] = ('Neutral', 'N/A (Data Not Ready)', 'N/A')
            else:
                ema_val_str = f"${ema_val:,.0f}"
                if close > ema_val:
                    signals['Trend'][f'EMA ({length})'] = ('Bullish', f"Price above EMA ({close_str})", ema_val_str)
                elif close < ema_val:
                    signals['Trend'][f'EMA ({length})'] = ('Bearish', f"Price below EMA ({close_str})", ema_val_str)
                else:
                    signals['Trend'][f'EMA ({length})'] = ('Neutral', f"Price at EMA ({close_str})", ema_val_str)
        except KeyError:
            signals['Trend'][f'EMA ({length})'] = ('Neutral', 'Error: Column not found.', 'N/A')

    # --- Volume Indicators ---
    
    # CMF
    cmf_col = safe_column_lookup(df, 'CMF_')
    if cmf_col:
        cmf_val = df[cmf_col].iloc[-1]
        
        if math.isnan(cmf_val):
             signals['Volume']['CMF (20)'] = ('Neutral', 'N/A (Data Not Ready)', 'N/A')
        else:
            cmf_val_str = f"{cmf_val:.3f}"
            if cmf_val > 0.20:
                signals['Volume']['CMF (20)'] = ('Bullish', f"Strong Accumulation ({cmf_val_str})", cmf_val_str)
            elif cmf_val < -0.20:
                signals['Volume']['CMF (20)'] = ('Bearish', f"Strong Distribution ({cmf_val_str})", cmf_val_str)
            else:
                signals['Volume']['CMF (20)'] = ('Neutral', f"Equilibrium/Weak Trend ({cmf_val_str})", cmf_val_str)
    else:
        signals['Volume']['CMF (20)'] = ('Neutral', 'N/A (Error)', 'N/A')


    # OBV
    try:
        obv_val = df['OBV'].iloc[-1]
        obv_prev = df['OBV'].iloc[-5] 
        
        if math.isnan(obv_val) or math.isnan(obv_prev):
             signals['Volume']['OBV'] = ('Neutral', 'N/A (Data Not Ready)', 'N/A')
        else:
            obv_val_str = f"{obv_val:,.0f}"
            if obv_val > obv_prev:
                signals['Volume']['OBV'] = ('Bullish', "Volume increasing (OBV rising)", obv_val_str)
            elif obv_val < obv_prev:
                signals['Volume']['OBV'] = ('Bearish', "Volume decreasing (OBV falling)", obv_val_str)
            else:
                signals['Volume']['OBV'] = ('Neutral', "Volume flat or range-bound", obv_val_str)
    except KeyError:
        signals['Volume']['OBV'] = ('Neutral', 'N/A (Error)', 'N/A')
    except IndexError:
        signals['Volume']['OBV'] = ('Neutral', 'N/A (Not enough bars)', 'N/A')

    # VWAP
    try:
        vwap_val = df['VWAP'].iloc[-1]
        
        if math.isnan(vwap_val):
            vwap_val_str = "N/A"
            signals['Volume']['VWAP'] = ('Neutral', 'N/A (Data Not Ready)', 'N/A')
        else:
            vwap_val_str = f"${vwap_val:,.0f}"
            if close > vwap_val:
                signals['Volume']['VWAP'] = ('Bullish', f"Price above VWAP ({close_str})", vwap_val_str)
            elif close < vwap_val:
                signals['Volume']['VWAP'] = ('Bearish', f"Price below VWAP ({close_str})", vwap_val_str)
            else:
                signals['Volume']['VWAP'] = ('Neutral', f"Price at VWAP ({close_str})", vwap_val_str)
                
    except KeyError:
        signals['Volume']['VWAP'] = ('Neutral', 'N/A (Error)', 'N/A')


    # --- Volatility Indicators ---

    # Bollinger Bands (NOW DISPLAYS ALL 3 VALUES)
    upper_band_col = safe_column_lookup(df, 'BBU_')
    median_band_col = safe_column_lookup(df, 'BBM_') # Middle band
    lower_band_col = safe_column_lookup(df, 'BBL_')
    
    if upper_band_col and median_band_col and lower_band_col:
        upper_band = df[upper_band_col].iloc[-1]
        median_band = df[median_band_col].iloc[-1]
        lower_band = df[lower_band_col].iloc[-1]
        
        if math.isnan(upper_band) or math.isnan(median_band) or math.isnan(lower_band):
             signals['Volatility']['Bollinger Bands (20,2)'] = ('Neutral', 'N/A (Data Not Ready)', 'N/A')
        else:
            # Format all three bands for the 'value' column: U|M|L
            bb_val_str = f"U:{upper_band:,.0f} | M:{median_band:,.0f} | L:{lower_band:,.0f}"
            
            if close > upper_band:
                signals['Volatility']['Bollinger Bands (20,2)'] = ('Bearish', f"Price above Upper Band (Squeeze Risk) ({close_str})", bb_val_str)
            elif close < lower_band:
                signals['Volatility']['Bollinger Bands (20,2)'] = ('Bullish', f"Price below Lower Band (Snap-back Risk) ({close_str})", bb_val_str)
            else:
                # Inside the bands: signal is determined by position relative to the Median Band (BBM)
                if close > median_band:
                    detail = f"Price above Median Band ({close_str})"
                    signal_type = 'Bullish'
                else:
                    detail = f"Price below Median Band ({close_str})"
                    signal_type = 'Bearish'
                    
                signals['Volatility']['Bollinger Bands (20,2)'] = (signal_type, detail, bb_val_str)
    else:
        signals['Volatility']['Bollinger Bands (20,2)'] = ('Neutral', 'N/A (Error)', 'N/A')

    # ATR
    atr_col = safe_column_lookup(df, 'ATR_')
    if atr_col:
        atr_val = df[atr_col].iloc[-1]
        
        if math.isnan(atr_val):
             signals['Volatility']['ATR (14)'] = ('Neutral', 'N/A (Data Not Ready)', 'N/A')
        else:
            atr_val_str = f"${atr_val:,.2f}"
            
            # Simple volatility expansion/contraction check
            try:
                atr_prev_avg = df[atr_col].iloc[-5:-1].mean() # Average of the 4 bars before the current one
                if atr_val > atr_prev_avg * 1.2:
                    detail = f"Volatility Expanding (+20% vs 4-bar avg)"
                elif atr_val < atr_prev_avg * 0.8:
                    detail = f"Volatility Contracting (-20% vs 4-bar avg)"
                else:
                    detail = f"Average bar range: {atr_val_str}"
            except IndexError:
                 detail = f"Average bar range: {atr_val_str}" # Fallback if not enough history for average
                 
            signals['Volatility']['ATR (14)'] = ('Neutral', detail, atr_val_str)
    else:
        signals['Volatility']['ATR (14)'] = ('Neutral', 'N/A (Error)', 'N/A')
        
    return signals


def get_divergence_alerts(df):
    """
    Checks for immediate reversal/trend-change alerts based on MFI, CMF, and Volume/Price Disparity.
    Returns a list of alert dictionaries.
    """
    if df is None or df.empty or len(df) < 5: 
        return []
    
    alerts = []
    
    # Ensure indicators are calculated
    df.ta.mfi(append=True)
    df.ta.cmf(append=True)
    df.ta.atr(append=True)

    # Use the last two bars (current/previous)
    current = df.iloc[-1]
    previous = df.iloc[-2]
    
    # --------------------------------------------------------
    # 1. MFI & CMF Cross Alerts (Immediate Reversal/Flow Change)
    # --------------------------------------------------------
    mfi_col = safe_column_lookup(df, 'MFI_')
    cmf_col = safe_column_lookup(df, 'CMF_')
    atr_col = safe_column_lookup(df, 'ATR_')

    if mfi_col and not math.isnan(current[mfi_col]):
        mfi_current = current[mfi_col]
        mfi_previous = previous[mfi_col]
        # Bullish Reversal: Was oversold (<20) and is now rising (>30)
        if mfi_previous < 20 and mfi_current > 30:
             alerts.append({
                'Indicator': 'MFI (14) Reversal',
                'Signal': 'BULLISH LONG',
                'Message': f"MFI reversed sharply from oversold (<20) to {mfi_current:.2f}. Potential bottom/Long entry.",
                'Direction': 'Long',
                'Value': f"{mfi_current:.2f}"
            })
        # Bearish Reversal: Was overbought (>80) and is now falling (<70)
        elif mfi_previous > 80 and mfi_current < 70:
            alerts.append({
                'Indicator': 'MFI (14) Reversal',
                'Signal': 'BEARISH SHORT',
                'Message': f"MFI reversed sharply from overbought (>80) to {mfi_current:.2f}. Potential top/Short entry.",
                'Direction': 'Short',
                'Value': f"{mfi_current:.2f}"
            })
    
    if cmf_col and not math.isnan(current[cmf_col]):
        cmf_current = current[cmf_col]
        cmf_previous = previous[cmf_col]
        # CMF Zero Crosses (Bullish/Bearish)
        if cmf_previous < 0 and cmf_current > 0:
            alerts.append({
                'Indicator': 'CMF (20) Zero Cross',
                'Signal': 'BULLISH LONG',
                'Message': f"CMF crossed above zero ({cmf_current:.3f}). Accumulation starting to dominate distribution.",
                'Direction': 'Long',
                'Value': f"{cmf_current:.3f}"
            })
        elif cmf_previous > 0 and cmf_current < 0:
            alerts.append({
                'Indicator': 'CMF (20) Zero Cross',
                'Signal': 'BEARISH SHORT',
                'Message': f"CMF crossed below zero ({cmf_current:.3f}). Distribution starting to dominate accumulation.",
                'Direction': 'Short',
                'Value': f"{cmf_current:.3f}"
            })

    # --------------------------------------------------------
    # 2. Volume/Price Disparity (High Volume, Low Price Movement)
    # --------------------------------------------------------
    
    # 5-Bar analysis
    recent_df = df.tail(5)
    
    # Check for required columns before proceeding
    if atr_col and cmf_col and 'Volume' in current and 'High' in current and 'Low' in current and not math.isnan(current[atr_col]):
        
        # Calculate key metrics
        avg_volume = recent_df['Volume'].mean()
        current_volume = current['Volume']
        current_range = current['High'] - current['Low']
        
        atr_val = current[atr_col]
        cmf_val = current[cmf_col]
    
        # Define threshold constants (can be adjusted)
        VOLUME_SPIKE_FACTOR = 1.5  # Current volume is 1.5x the 5-bar average
        RANGE_COMPRESSION_FACTOR = 0.5 # Current range is less than 50% of ATR
    
        if current_volume > avg_volume * VOLUME_SPIKE_FACTOR and current_range < atr_val * RANGE_COMPRESSION_FACTOR and not math.isnan(cmf_val):
            
            # High volume, low price movement (disparity detected)
            vol_str = f"{current_volume/1000:,.0f}K"
            
            # Check CMF for directional bias
            if cmf_val > 0.10: # Accumulation dominant, suggests demand is meeting supply but unable to push price higher
                alerts.append({
                    'Indicator': 'Vol/Price Disparity',
                    'Signal': 'BEARISH SHORT',
                    'Message': f"High Accumulation (CMF: {cmf_val:.3f}) with low volatility. Supply is absorbing heavy demand (potential top/trap). Vol: {vol_str}",
                    'Direction': 'Short',
                    'Value': f"Vol: {vol_str}"
                })
            elif cmf_val < -0.10: # Distribution dominant, suggests selling is met by floor buying but unable to break down
                alerts.append({
                    'Indicator': 'Vol/Price Disparity',
                    'Signal': 'BULLISH LONG',
                    'Message': f"High Distribution (CMF: {cmf_val:.3f}) with low volatility. Demand is absorbing heavy supply (potential bottom/Long entry). Vol: {vol_str}",
                    'Direction': 'Long',
                    'Value': f"Vol: {vol_str}"
                })

    return alerts

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
            return None, None, None, None

        result_pairs = data.get('result')
        if not result_pairs:
            st.error("Kraken response 'result' field was empty.")
            return None, None, None, None
        
        pair_key = next(iter(result_pairs), None)
        if not pair_key:
            st.error("Kraken response 'result' field contained no market data.")
            return None, None, None, None

        btc_data = result_pairs.get(pair_key)
        
        # Current time in UTC (best effort from the system running the script)
        current_utc_time = datetime.now(timezone.utc)
        
        price = float(btc_data['c'][0])
        # Volume is in the base asset (XBT)
        volume_xbt = float(btc_data['v'][1]) 
        open_price = float(btc_data['o'])
        change_percent = ((price - open_price) / open_price) * 100 if open_price else 0
        
        return price, volume_xbt, change_percent, current_utc_time
    except Exception as e:
        st.error(f"Could not fetch live Bitcoin data from Kraken: {e}")
        return None, None, None, None

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
            
            if url.endswith("/Time"): # Special handling for Kraken Time endpoint
                unixtime = data['result']['unixtime']
                utc_dt = datetime.fromtimestamp(unixtime, tz=timezone.utc)
                target_dt = utc_dt.astimezone(UTC_PLUS_11)
                time_str = target_dt.strftime('%H:%M:%S')
                
                status = "UP"
                result_detail = f"OK ({response_time_ms}ms, Time: {time_str})"
                return status, result_detail, response_time_ms, data

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

# --- Live Bitcoin Tracking ---
st.header("Live Bitcoin Metrics")
btc_price, btc_volume_xbt, btc_change, current_utc_time = fetch_btc_data()

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

    col_volume.metric(
        label="24h Volume (XBT)", 
        value=volume_str,
    )
    
    # Apply the time zone fix for the caption
    now_utc_plus_11 = current_utc_time.astimezone(UTC_PLUS_11).strftime('%H:%M:%S')
    st.caption(f"Data source: {KRAKEN_API_URL}. Fetched at {now_utc_plus_11} (UTC+{UTC_OFFSET_HOTHOURS})")
else:
    st.warning("Bitcoin data not available from Kraken API. Check the connection or API health.")

st.markdown("---")

# --- Configuration Controls (Sidebar) ---
with st.sidebar:
    st.header("App Controls")
    st.markdown("Use these controls to adjust the data used for the Technical Analysis Signal Matrix.")

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
        "Number of Historical Bars (Min 52 for Ichimoku)",
        min_value=52, # Minimum needed for Ichimoku
        max_value=1000,
        value=300,
        step=50
    )
    
    st.markdown("---")
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


# --- Action Button ---
st.button("Run All Health Checks Now", on_click=run_all_checks, use_container_width=True, type="primary")

st.markdown("---")

# --- Tactical Reversal Alerts ---
st.header("Tactical Reversal & Disparity Alerts")
st.markdown("Immediate Long/Short signals based on MFI/CMF crosses and **Volume/Price Disparity** (VPD).")

historical_df = fetch_historical_data(selected_interval_minutes, selected_bar_count)
alerts = get_divergence_alerts(historical_df)

if alerts:
    for alert in alerts:
        signal = alert['Signal']
        message = alert['Message']
        indicator = alert['Indicator']
        value = alert['Value']
        
        # Determine the color and icon
        if alert['Direction'] == 'Long':
            alert_style = 'success'
            icon = "🚀"
        else:
            alert_style = 'error'
            icon = "📉"
            
        # Use st.warning/st.success/st.error for colored background alerts
        st.markdown(f"""
        <div style="
            padding: 10px;
            border-radius: 8px;
            border: 1px solid {'#198754' if alert_style == 'success' else '#dc3545'};
            background-color: {'#d1e7dd' if alert_style == 'success' else '#f8d7da'};
            margin-bottom: 10px;
        ">
            <p style="margin: 0; font-weight: bold; color: {'#0f5132' if alert_style == 'success' else '#842029'}; font-size: 16px;">
                {icon} {signal} - {indicator} ({value})
            </p>
            <p style="margin: 0; font-size: 14px; color: {'#0f5132' if alert_style == 'success' else '#842029'};">
                {message}
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("No immediate reversal or disparity alerts detected on the current timeframe.")

st.markdown("---")


# --- Automated TA Signals (Tiled Matrix) ---
st.header(f"Automated TA Signal Matrix ({selected_timeframe_str})")
st.markdown("Consolidated signals grouped by indicator type for efficient market assessment.")

ta_signals_grouped = get_indicator_signal(historical_df)

if ta_signals_grouped:
    
    # Define the order of the groups
    group_order = ['Trend', 'Momentum', 'Volume', 'Volatility']
    
    # Define color map for the tiles
    color_map = {
        'Bullish': '#198754',  # Green
        'Bearish': '#dc3545',  # Red
        'Neutral': '#6c757d'   # Grey
    }
    
    for group_name in group_order:
        if group_name in ta_signals_grouped and ta_signals_grouped[group_name]:
            
            st.subheader(f"📊 {group_name} Indicators")
            signals = ta_signals_grouped[group_name]
            
            # Use columns for tiles (4 tiles per row)
            num_signals = len(signals)
            cols = st.columns(min(4, num_signals))
            
            # Sort signals alphabetically within the group
            sorted_signals = sorted(signals.items(), key=lambda item: item[0])

            for i, (indicator_name, (signal, detail, value_str)) in enumerate(sorted_signals):
                
                # Determine the color and icon
                bg_color = color_map.get(signal, '#6c757d')
                icon = "▲" if signal == "Bullish" else "▼" if signal == "Bearish" else "●"
                
                # Apply custom HTML/Markdown for the colored tile effect
                tile_html = f"""
                <div style="
                    background-color: {bg_color};
                    padding: 15px;
                    border-radius: 8px;
                    color: white;
                    text-align: center;
                    margin-bottom: 15px;
                    height: 125px; 
                ">
                    <h5 style="margin: 0; font-size: 14px; opacity: 0.8;">{indicator_name}</h5>
                    <p style="
                        margin: 5px 0 5px 0; 
                        font-weight: bold; 
                        font-size: 20px; 
                    ">{value_str}</p>
                    <p style="margin: 0; font-weight: bold; font-size: 16px;">{icon} {signal.upper()}</p>
                </div>
                """
                
                # Place the tile in the appropriate column
                with cols[i % 4]:
                    st.markdown(tile_html, unsafe_allow_html=True)
            
            st.markdown("---") # Separator after each group
            
else:
    st.warning("Not enough historical data to calculate indicators (min 52 bars). Try increasing the bar count or check data fetch status.")


# --- Interactive Glossary Section ---
st.markdown(get_glossary_html(INDICATOR_GLOSSARY), unsafe_allow_html=True)

# --- Feed Status Overview (Monitoring) ---
st.header("Critical Data Feed Health Check")

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

# --- Response Time History ---
st.header("Response Time History")
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
