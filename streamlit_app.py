import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
import warnings

# Suppress pandas_ta warnings
warnings.filterwarnings('ignore')

# --- Page configuration ---
st.set_page_config(
    layout="wide", 
    page_title="Crypto Alpha Dashboard",
    page_icon="📊"
)

# --- Custom CSS for better styling and mobile responsiveness ---
st.markdown("""
<style>
    /* Main Streamlit container adjustments */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: #0d6efd; /* Blue color */
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    
    .metric-card {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease-in-out;
    }

    .metric-card:hover {
        transform: translateY(-2px);
    }

    /* Sentiment specific styling */
    .bullish {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .bearish {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .neutral {
        background-color: #e2e3e5;
        border-left: 5px solid #6c757d;
    }
    
    /* Sidebar adjustments */
    .sidebar .sidebar-content {
        padding: 1rem;
    }
    h2 { /* Subheader fix for clarity */
        border-bottom: 1px solid #ddd;
        padding-bottom: 5px;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚀 Crypto Alpha Dashboard — Multi-Asset Analytics Platform</div>', unsafe_allow_html=True)

# ---------------------------
# --------- SETTINGS --------
# ---------------------------
# Initial default indicator configuration
INDICATOR_DEFAULTS = {
    "EMA Cross": {"weight": 0.12, "params": {"fast": 9, "slow": 21}},
    "MACD": {"weight": 0.10, "params": {"fast": 12, "slow": 26, "signal": 9}},
    "RSI": {"weight": 0.08, "params": {"period": 14}},
    "ADX": {"weight": 0.08, "params": {"period": 14}},
    "Bollinger Bands": {"weight": 0.05, "params": {"period": 20, "std": 2}},
    "Stochastic": {"weight": 0.06, "params": {"k": 14, "d": 3}},
    "CMF": {"weight": 0.04, "params": {"period": 20}}
}

# Available assets and their Binance symbols
CRYPTO_ASSETS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT", 
    "ADA": "ADAUSDT",
    "DOT": "DOTUSDT",
    "SOL": "SOLUSDT",
    "MATIC": "MATICUSDT",
    "LINK": "LINKUSDT"
}

# Binance API endpoints
BINANCE_BASE_URL = "https://api.binance.com/api/v3"

# ---------------------------
# --------- SIDEBAR ---------
# ---------------------------
st.sidebar.header("🎛️ Dashboard Controls")

# Multi-asset selection
st.sidebar.subheader("Asset Selection")
selected_assets = st.sidebar.multiselect(
    "Choose Assets for Analysis", 
    list(CRYPTO_ASSETS.keys()), 
    default=["BTC", "ETH"]
)

# Timeframe selection for Binance
st.sidebar.subheader("Timeframe Configuration")
selected_timeframe = st.sidebar.selectbox(
    "Select Timeframe", 
    ["1h", "4h", "1d", "1w"], 
    index=0
)

# Map timeframe to Binance intervals and data limits
TIMEFRAME_MAP = {
    "1h": {"interval": "1h", "limit": 500, "days_back": 21},
    "4h": {"interval": "4h", "limit": 500, "days_back": 84},
    "1d": {"interval": "1d", "limit": 500, "days_back": 500},
    "1w": {"interval": "1w", "limit": 500, "days_back": 2600}
}

selected_config = TIMEFRAME_MAP[selected_timeframe]

# Risk settings
st.sidebar.subheader("Risk Management")
risk_tolerance = st.sidebar.select_slider(
    "Risk Tolerance", 
    options=["Conservative", "Moderate", "Aggressive"],
    value="Moderate"
)

# ---------------------------
# ----- INDICATOR CONTROLS ---
# ---------------------------
st.sidebar.header("⚖️ Weighted Indicator Settings")
st.sidebar.caption("Adjust weights (0.0 - 0.2) for the combined sentiment score.")

indicator_weights = {}
for ind, config in INDICATOR_DEFAULTS.items():
    # Use a shorter title for the sidebar slider
    slider_title = f"{ind} Weight"
    
    w = st.sidebar.slider(
        slider_title, 
        0.0, 0.2, config["weight"], 0.01, 
        key=f"weight_{ind}",
        help=f"Weight applied to {ind} signal (Score * Weight)"
    )
    
    # Store dynamic weight
    indicator_weights[ind] = {
        "weight": w, 
        "params": config["params"]
    }

# ---------------------------
# ------- DATA FETCH --------
# ---------------------------
@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_binance_klines(symbol, interval, limit=500):
    """Fetch OHLCV data from Binance API"""
    try:
        url = f"{BINANCE_BASE_URL}/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('datetime')
        
        return df[['open', 'high', 'low', 'close', 'volume']].reset_index()
        
    except Exception as e:
        st.error(f"Error fetching Binance data for {symbol}: {e}")
        return create_sample_data_binance(interval, limit)

def create_sample_data_binance(interval, limit):
    """Create sample data when API fails"""
    freq_map = {"1h": "H", "4h": "4H", "1d": "D", "1w": "W"}
    freq = freq_map.get(interval, "H")
    
    dates = pd.date_range(end=datetime.now(), periods=limit, freq=freq)
    prices = 20000 + np.cumsum(np.random.randn(limit) * 50)
    data = pd.DataFrame({
        'datetime': dates,
        'open': prices.shift(1).fillna(prices.iloc[0]),
        'high': prices * (1 + np.random.rand(limit) * 0.01),
        'low': prices * (1 - np.random.rand(limit) * 0.01),
        'close': prices,
        'volume': np.random.randint(1000, 10000, limit)
    })
    return data

@st.cache_data(ttl=60)
def fetch_current_prices_binance(assets):
    """Fetch current prices and 24h change for multiple assets from Binance"""
    try:
        prices_data = {}
        for asset in assets:
            symbol = CRYPTO_ASSETS[asset]
            url = f"{BINANCE_BASE_URL}/ticker/24hr"
            params = {'symbol': symbol}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            prices_data[asset] = {
                'price': float(data['lastPrice']),
                'change_24h': float(data['priceChangePercent']),
                'volume': float(data['volume']),
                'high_24h': float(data['highPrice']),
                'low_24h': float(data['lowPrice'])
            }
        
        return prices_data
    except Exception as e:
        st.error(f"Error fetching current prices: {e}")
        return {}

@st.cache_data(ttl=3600)
def fetch_market_sentiment_binance():
    """Fetch general market sentiment indicators using Binance data"""
    try:
        # Get BTC dominance and overall market trend from top cryptocurrencies
        top_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]
        price_changes = []
        volumes = []
        
        for symbol in top_symbols:
            url = f"{BINANCE_BASE_URL}/ticker/24hr"
            params = {'symbol': symbol}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            price_changes.append(float(data['priceChangePercent']))
            volumes.append(float(data['volume']))
        
        # Calculate weighted average price change (market sentiment proxy)
        total_volume = sum(volumes)
        if total_volume > 0:
            weighted_change = sum(p * v for p, v in zip(price_changes, volumes)) / total_volume
        else:
            weighted_change = sum(price_changes) / len(price_changes)
        
        # Fear & Greed Index Proxy based on market performance
        fg_index = np.clip(50 + weighted_change * 2, 5, 95)
        
        # Determine market trend
        if weighted_change > 2:
            market_trend = "Strong Bullish"
        elif weighted_change > 0.5:
            market_trend = "Bullish"
        elif weighted_change < -2:
            market_trend = "Strong Bearish"
        elif weighted_change < -0.5:
            market_trend = "Bearish"
        else:
            market_trend = "Neutral"
        
        sentiment_data = {
            "market_change": weighted_change,
            "fear_greed_index": fg_index,
            "market_trend": market_trend,
            "total_volume": total_volume
        }
        return sentiment_data
    except Exception as e:
        st.error(f"Error fetching market sentiment: {e}")
        return {
            "market_change": 0.0,
            "fear_greed_index": 50,
            "market_trend": "Unknown",
            "total_volume": 0
        }

# ---------------------------
# ---- ENHANCED INDICATORS ---
# ---------------------------
def compute_enhanced_indicators(df, params):
    """Compute technical indicators using pandas_ta"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # EMAs
    ema_fast = params["EMA Cross"]["params"]["fast"]
    ema_slow = params["EMA Cross"]["params"]["slow"]
    df[f"EMA{ema_fast}"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    df[f"EMA{ema_slow}"] = df["close"].ewm(span=ema_slow, adjust=False).mean()
    
    # MACD
    macd_params = params["MACD"]["params"]
    macd = ta.macd(df["close"], **macd_params) 
    if not macd.empty:
        macd_cols = [col for col in macd.columns if 'MACD_' in col and 'MACDs_' not in col and 'MACDh_' not in col]
        signal_cols = [col for col in macd.columns if 'MACDs_' in col]
        if macd_cols:
            df["MACD"] = macd[macd_cols[0]]
        if signal_cols:
            df["MACD_Signal"] = macd[signal_cols[0]]
        
    # RSI
    rsi_period = params["RSI"]["params"]["period"]
    df["RSI"] = ta.rsi(df["close"], length=rsi_period)
    
    # Bollinger Bands
    bb_params = params["Bollinger Bands"]["params"]
    bbands = ta.bbands(df["close"], length=bb_params["period"], std=bb_params["std"])
    if not bbands.empty:
        upper_cols = [col for col in bbands.columns if 'BBU_' in col]
        lower_cols = [col for col in bbands.columns if 'BBL_' in col]
        if upper_cols: df["BB_upper"] = bbands[upper_cols[0]]
        if lower_cols: df["BB_lower"] = bbands[lower_cols[0]]
        
    # Stochastic
    stoch_params = params["Stochastic"]["params"]
    stoch = ta.stoch(df["high"], df["low"], df["close"], **stoch_params)
    if not stoch.empty:
        k_cols = [col for col in stoch.columns if 'STOCHk_' in col]
        d_cols = [col for col in stoch.columns if 'STOCHd_' in col]
        if k_cols: df["Stoch_K"] = stoch[k_cols[0]]
        if d_cols: df["Stoch_D"] = stoch[d_cols[0]]
    
    # ADX and CMF
    try:
        df["ADX"] = ta.adx(df["high"], df["low"], df["close"])["ADX_14"]
    except Exception:
        df["ADX"] = 0
    
    try:
        df["CMF"] = ta.cmf(df["high"], df["low"], df["close"], df["volume"])
    except Exception:
        df["CMF"] = 0
    
    return df

# ---------------------------
# ---- SIGNAL GENERATION ----
# ---------------------------
def generate_trading_signals(df, weights):
    """Generate comprehensive trading signals based on indicators"""
    if df.empty or len(df) < 50: # Require sufficient data for stable indicators
        return {}
    
    last = df.iloc[-1]
    signals = {}
    
    # --- EMA Cross Signal ---
    ema_fast = weights["EMA Cross"]["params"]["fast"]
    ema_slow = weights["EMA Cross"]["params"]["slow"]
    fast_col = f"EMA{ema_fast}"
    slow_col = f"EMA{ema_slow}"
    
    if fast_col in df.columns and slow_col in df.columns and not pd.isna(last[fast_col]):
        if last[fast_col] > last[slow_col]:
            signals["EMA Cross"] = ("Bullish", f"EMA{ema_fast} cross above EMA{ema_slow}")
        elif last[fast_col] < last[slow_col]:
            signals["EMA Cross"] = ("Bearish", f"EMA{ema_fast} cross below EMA{ema_slow}")
        else:
            signals["EMA Cross"] = ("Neutral", "EMAs converging")

    # --- RSI Signal ---
    if "RSI" in df.columns and not pd.isna(last["RSI"]):
        rsi = last["RSI"]
        if rsi > 70: 
             signals["RSI"] = ("Bearish", f"Overbought ({rsi:.1f})")
        elif rsi < 30: 
             signals["RSI"] = ("Bullish", f"Oversold ({rsi:.1f})")
        else:
             signals["RSI"] = ("Neutral", f"RSI in mid-range ({rsi:.1f})")
    
    # --- MACD Signal ---
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        macd = last["MACD"] if not pd.isna(last["MACD"]) else 0
        macd_signal = last["MACD_Signal"] if not pd.isna(last["MACD_Signal"]) else 0
        if macd > macd_signal and macd > 0:
            signals["MACD"] = ("Bullish", "MACD cross up (Positive)")
        elif macd < macd_signal and macd < 0:
            signals["MACD"] = ("Bearish", "MACD cross down (Negative)")
        else:
            signals["MACD"] = ("Neutral", "MACD/Signal line near zero or crossing near zero")

    # --- Bollinger Bands Signal (Breakout/Reversal) ---
    if "BB_upper" in df.columns and "BB_lower" in df.columns:
        price = last["close"]
        bb_upper = last["BB_upper"] if not pd.isna(last["BB_upper"]) else price * 1.1
        bb_lower = last["BB_lower"] if not pd.isna(last["BB_lower"]) else price * 0.9
        
        if price <= bb_lower:
            signals["Bollinger Bands"] = ("Bullish", "Price below lower band (Reversal opportunity)")
        elif price >= bb_upper:
            signals["Bollinger Bands"] = ("Bearish", "Price above upper band (Reversal opportunity)")
        else:
            signals["Bollinger Bands"] = ("Neutral", "Price within bands")
    
    # --- Stochastic Signal ---
    if "Stoch_K" in df.columns and "Stoch_D" in df.columns:
        stoch_k = last["Stoch_K"] if not pd.isna(last["Stoch_K"]) else 50
        stoch_d = last["Stoch_D"] if not pd.isna(last["Stoch_D"]) else 50
        
        if stoch_k < 20 and stoch_k > stoch_d:
            signals["Stochastic"] = ("Bullish", f"Oversold cross K:{stoch_k:.1f}")
        elif stoch_k > 80 and stoch_k < stoch_d:
            signals["Stochastic"] = ("Bearish", f"Overbought cross K:{stoch_k:.1f}")
        else:
            signals["Stochastic"] = ("Neutral", "K/D in mid-range")
            
    # --- ADX Signal (Trend Strength) ---
    if "ADX" in df.columns and not pd.isna(last["ADX"]):
        adx = last["ADX"]
        if adx > 25:
             signals["ADX"] = ("Bullish", f"Strong Trend ({adx:.1f})") # Trend is strong, direction must be confirmed by other signals
        elif adx < 20:
             signals["ADX"] = ("Neutral", f"Weak/No Trend ({adx:.1f})")
        else:
             signals["ADX"] = ("Neutral", f"Developing Trend ({adx:.1f})")
    
    # --- CMF Signal (Money Flow) ---
    if "CMF" in df.columns and not pd.isna(last["CMF"]):
        cmf = last["CMF"]
        if cmf > 0.2:
            signals["CMF"] = ("Bullish", f"Strong Inflow ({cmf:.2f})")
        elif cmf < -0.2:
            signals["CMF"] = ("Bearish", f"Strong Outflow ({cmf:.2f})")
        else:
            signals["CMF"] = ("Neutral", f"Money Flow Neutral ({cmf:.2f})")
    
    return signals

def calculate_combined_score(signals, weights):
    """Calculate overall sentiment score based on weighted signals"""
    score_map = {"Bullish": 1, "Neutral": 0, "Bearish": -1}
    total_weight = 0
    weighted_score = 0
    
    for indicator, signal in signals.items():
        if indicator in weights:
            weight = weights[indicator]["weight"]
            score = score_map.get(signal[0], 0)
            weighted_score += score * weight
            total_weight += weight
    
    if total_weight > 0:
        final_score = weighted_score / total_weight
    else:
        final_score = 0
    
    # Define sentiment thresholds
    if final_score > 0.15:
        return "Strong Bullish", final_score
    elif final_score > 0.05:
        return "Bullish", final_score
    elif final_score < -0.15:
        return "Strong Bearish", final_score
    elif final_score < -0.05:
        return "Bearish", final_score
    else:
        return "Neutral", final_score

# ---------------------------
# ----- VISUALIZATION -------
# ---------------------------
def create_price_chart(df, asset):
    """Create a price chart with primary technical indicators."""
    if df.empty: return go.Figure()
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f'{asset} Price & Trend', 'RSI (Momentum)', 'MACD (Trend Reversal)')
    )
    
    # --- Row 1: Price and Overlays ---
    fig.add_trace(
        go.Candlestick(
            x=df['datetime'],
            open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name='Price', showlegend=False
        ), row=1, col=1
    )
    
    # EMAs
    ema_fast = indicator_weights["EMA Cross"]["params"]["fast"]
    ema_slow = indicator_weights["EMA Cross"]["params"]["slow"]
    
    if f'EMA{ema_fast}' in df.columns:
        fig.add_trace(go.Scatter(x=df['datetime'], y=df[f'EMA{ema_fast}'], name=f'EMA{ema_fast}', line=dict(color='orange', width=1), legendgroup='price'), row=1, col=1)
    if f'EMA{ema_slow}' in df.columns:
        fig.add_trace(go.Scatter(x=df['datetime'], y=df[f'EMA{ema_slow}'], name=f'EMA{ema_slow}', line=dict(color='red', width=1.5), legendgroup='price'), row=1, col=1)
    
    # --- Row 2: RSI ---
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['RSI'], name='RSI', line=dict(color='purple'), showlegend=False),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # --- Row 3: MACD ---
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['MACD'], name='MACD Line', line=dict(color='blue'), legendgroup='macd'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['MACD_Signal'], name='Signal Line', line=dict(color='red'), legendgroup='macd'), row=3, col=1)
        
    
    # --- Layout and Aesthetics ---
    fig.update_layout(
        height=750,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        title_font_size=18
    )
    
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    
    return fig

# ---------------------------
# ----- MAIN DASHBOARD ------
# ---------------------------
def main():
    
    if not selected_assets:
        st.warning("Please select at least one asset from the sidebar to begin analysis.")
        return

    # Fetch data for all selected assets
    data_dict = {}
    for asset in selected_assets:
        symbol = CRYPTO_ASSETS[asset]
        data_dict[asset] = fetch_binance_klines(
            symbol=symbol,
            interval=selected_config["interval"],
            limit=selected_config["limit"]
        )

    # Fetch market data
    current_prices = fetch_current_prices_binance(selected_assets)
    sentiment_data = fetch_market_sentiment_binance()
    
    # --- Market Overview Metrics ---
    st.header("🌎 Global Market & Sentiment Snapshot")
    
    # Determine F&G Sentiment
    fg = sentiment_data['fear_greed_index']
    fg_status = "Extreme Greed" if fg >= 80 else ("Greed" if fg >= 60 else ("Neutral" if fg >= 40 else ("Fear" if fg >= 20 else "Extreme Fear")))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Market 24h Change", 
            f"{sentiment_data['market_change']:.2f}%", 
            delta=f"{sentiment_data['market_change']:.2f}%"
        )
    with col2:
        st.metric("Fear & Greed Index (Proxy)", f"{fg:.0f}", help="0=Extreme Fear, 100=Extreme Greed")
    
    with col3:
        # Style the Fear & Greed status
        style_color = "red" if fg < 40 else ("green" if fg > 60 else "gray")
        st.markdown(f"**Sentiment Status:** <span style='color:{style_color}'>{fg_status}</span>", unsafe_allow_html=True)
        
    with col4:
        st.metric("Market Trend", sentiment_data['market_trend'])

    st.markdown("---")
    
    # --- Asset-Specific Analysis ---
    st.header(f"💰 Asset Deep Dive ({selected_timeframe})")
    
    for asset in selected_assets:
        df = data_dict[asset]
        if df.empty or len(df) < 50:
             st.warning(f"Insufficient historical data for {asset} to run full technical analysis.")
             continue

        # 1. Compute Indicators & Signals
        df_indicators = compute_enhanced_indicators(df, indicator_weights)
        signals = generate_trading_signals(df_indicators, indicator_weights)
        overall_sentiment, score = calculate_combined_score(signals, indicator_weights)
        
        # 2. Get Price/Change Data
        price_data = current_prices.get(asset, {'price': df['close'].iloc[-1], 'change_24h': 0.0})
        price = price_data['price']
        change_24h = price_data.get('change_24h', 0.0)

        # 3. Display Main Sentiment Card
        sentiment_color = {
            "Strong Bullish": "bullish", "Bullish": "bullish",
            "Strong Bearish": "bearish", "Bearish": "bearish",
            "Neutral": "neutral"
        }.get(overall_sentiment, "neutral")

        st.markdown(
            f"""
            <div class='metric-card {sentiment_color}'>
                <div style='display:flex; justify-content: space-between; align-items: center;'>
                    <div style='flex-grow: 1;'>
                        <h2>{asset} ({selected_timeframe} Signal)</h2>
                        <p style='font-size: 1.5em; font-weight: bold;'>{overall_sentiment}</p>
                        <p style='font-size: 0.9em;'>Confidence Score: {score:.3f}</p>
                    </div>
                    <div style='text-align: right;'>
                        <p style='font-size: 1.8em; font-weight: bold;'>${price:,.2f}</p>
                        <p style='color: {'green' if change_24h >= 0 else 'red'};'>{change_24h:+.2f}% (24h)</p>
                    </div>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # 4. Drilldown Panel
        with st.expander(f"Drilldown: Detailed Signal Breakdown & Charts for {asset}", expanded=False):
            
            # --- Chart Visualization ---
            st.subheader("Chart Overview")
            fig = create_price_chart(df_indicators, asset)
            st.plotly_chart(fig, use_container_width=True)
            
            # --- Signals Table ---
            st.subheader("Signal Breakdown")
            if signals:
                signals_df = pd.DataFrame([
                    {"Indicator": ind, "Signal": sig[0], "Reason": sig[1]} 
                    for ind, sig in signals.items()
                ])
                # Add score (weight) column
                signals_df['Weight'] = signals_df['Indicator'].apply(lambda x: indicator_weights.get(x, {}).get('weight', 0.0))
                signals_df['Weighted Score'] = signals_df.apply(
                    lambda row: row['Weight'] * (1 if row['Signal'] == 'Bullish' else (-1 if row['Signal'] == 'Bearish' else 0)), axis=1
                )
                
                st.dataframe(signals_df.set_index('Indicator').sort_values(by='Weighted Score', ascending=False), use_container_width=True)
            else:
                st.info("No technical signals generated for this asset.")
                
        st.markdown("---")

    
    # --- How to Use Section (Required) ---
    st.header("📚 How to Use This Alpha Dashboard")
    st.markdown("""
    This dashboard provides a robust, multi-factor analysis platform designed to assess short- to medium-term sentiment for major crypto assets using **Binance API data**.
    
    ### 1. **Data & Timeframe**
    - **Asset Selection (Sidebar)**: Choose the cryptocurrencies you wish to analyze.
    - **Timeframe (Sidebar)**: Select the interval (1h, 4h, 1d, 1w) for the underlying data from Binance. All indicators and signals are calculated based on this selected timeframe.
    
    ### 2. **Weighted Indicator Scoring**
    The **Overall Sentiment** (Bullish/Neutral/Bearish) is derived from a custom weighted average:
    
    $$
    \\text{Final Score} = \\frac{\sum (\\text{Signal Score} \times \\text{Weight})}{\sum \\text{Weight}}
    $$
    
    - **Signal Score**: $\\{+1, 0, -1\\}$ for Bullish, Neutral, or Bearish signals from a specific indicator (e.g., RSI < 30 is Bullish, > 70 is Bearish).
    - **Weight (Sidebar)**: You control the importance of each indicator (e.g., set a higher weight for EMA Cross if you prioritize trend following).
    
    ### 3. **Interpreting Results**
    - **Global Snapshot**: Provides macro context using market-wide price changes and the Fear & Greed Index (proxy).
    - **Main Sentiment Card**: Displays the final combined signal and confidence score for the selected asset/timeframe.
    - **Drilldown Panel**: Expand this section to view the supporting price chart and the contribution of each individual indicator to the final score.
    
    **Data Source**: This dashboard uses real-time data from Binance API.
    **Disclaimer**: This tool is for educational and informational purposes only. It is not financial advice. Always perform your own research and manage your risk accordingly.
    """)

if __name__ == "__main__":
    main()
