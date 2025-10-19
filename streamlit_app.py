import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    layout="wide", 
    page_title="Crypto Alpha Dashboard",
    page_icon="📊"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .bullish {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .bearish {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .neutral {
        background-color: #e2e3e5;
        border-left: 4px solid #6c757d;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚀 Crypto Alpha Dashboard — Multi-Asset Analytics Platform</div>', unsafe_allow_html=True)

# ---------------------------
# --------- SETTINGS --------
# ---------------------------
# Enhanced indicator configuration
INDICATOR_DEFAULTS = {
    "EMA Cross": {"weight": 0.12, "neutral_tol": 0.002, "params": {"fast": 9, "slow": 21}},
    "MACD": {"weight": 0.10, "neutral_tol": 0.0, "params": {"fast": 12, "slow": 26, "signal": 9}},
    "RSI": {"weight": 0.08, "neutral_tol": (40, 60), "params": {"period": 14}},
    "ADX": {"weight": 0.08, "neutral_tol": 20, "params": {"period": 14}},
    "OBV": {"weight": 0.05, "neutral_tol": 0.005, "params": {}},
    "SAR": {"weight": 0.05, "neutral_tol": 0.02, "params": {"af": 0.02, "max_af": 0.2}},
    "VWAP": {"weight": 0.05, "neutral_tol": 0.002, "params": {}},
    "ATR": {"weight": 0.04, "neutral_tol": 0.01, "params": {"period": 14}},
    "Bollinger Bands": {"weight": 0.05, "neutral_tol": None, "params": {"period": 20, "std": 2}},
    "Volume Oscillator": {"weight": 0.04, "neutral_tol": 0.005, "params": {"fast": 12, "slow": 26}},
    "CMF": {"weight": 0.04, "neutral_tol": 0.05, "params": {"period": 20}},
    "Ichimoku": {"weight": 0.08, "neutral_tol": None, "params": {}},
    "Stochastic": {"weight": 0.06, "neutral_tol": (20, 80), "params": {"k": 14, "d": 3}},
    "Netflow": {"weight": 0.08, "neutral_tol": 0.0, "params": {}},
    "NUPL": {"weight": 0.08, "neutral_tol": (0.4, 0.6), "params": {}}
}

# Available assets and timeframes
CRYPTO_ASSETS = {
    "BTC": "bitcoin",
    "ETH": "ethereum", 
    "ADA": "cardano",
    "DOT": "polkadot",
    "SOL": "solana",
    "MATIC": "matic-network",
    "AVAX": "avalanche-2",
    "LINK": "chainlink"
}

TF_OPTIONS = ["1h", "4h", "24h", "7d"]
TF_MAPPING = {
    "1h": "hourly",
    "4h": "hourly", 
    "24h": "daily",
    "7d": "daily"
}

# ---------------------------
# --------- SIDEBAR ---------
# ---------------------------
st.sidebar.header("🎛️ Dashboard Controls")

# Theme selector
theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], index=0)

# Multi-asset selection
st.sidebar.subheader("Asset Selection")
selected_assets = st.sidebar.multiselect(
    "Choose Assets", 
    list(CRYPTO_ASSETS.keys()), 
    default=["BTC", "ETH"]
)

# Timeframe selection
st.sidebar.subheader("Timeframe Configuration")
selected_tfs = st.sidebar.multiselect(
    "Select Timeframes", 
    TF_OPTIONS, 
    default=["1h", "24h"]
)

# Risk settings
st.sidebar.subheader("Risk Management")
risk_tolerance = st.sidebar.select_slider(
    "Risk Tolerance", 
    options=["Conservative", "Moderate", "Aggressive"],
    value="Moderate"
)

enable_alerts = st.sidebar.checkbox("Enable Price Alerts", value=False)
if enable_alerts:
    alert_price = st.sidebar.number_input("Alert Price", value=0.0)

# ---------------------------
# ----- INDICATOR CONTROLS ---
# ---------------------------
st.sidebar.header("📊 Indicator Configuration")

indicator_weights = {}
for ind, config in INDICATOR_DEFAULTS.items():
    st.sidebar.subheader(f"{ind}")
    w = st.sidebar.slider(f"Weight", 0.0, 0.2, config["weight"], 0.01, key=f"weight_{ind}")
    
    # Parameter adjustments
    if "period" in config["params"]:
        period = st.sidebar.slider(f"Period", 5, 50, config["params"]["period"], key=f"period_{ind}")
        config["params"]["period"] = period
    
    indicator_weights[ind] = {
        "weight": w, 
        "neutral_tol": config["neutral_tol"],
        "params": config["params"]
    }

# ---------------------------
# ------- DATA FETCH --------
# ---------------------------
@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_crypto_data(coin_id="bitcoin", days=90):
    """Fetch cryptocurrency data from CoinGecko API"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily' if days > 90 else 'hourly'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # For OHLC data, we'll create synthetic data from price points
        df['open'] = df['price'].shift(1)
        df['high'] = df[['open', 'price']].max(axis=1)
        df['low'] = df[['open', 'price']].min(axis=1)
        df['close'] = df['price']
        df['volume'] = 0  # Volume not available in basic endpoint
        
        # Handle first row
        df.loc[0, 'open'] = df.loc[0, 'close']
        df.loc[0, 'high'] = df.loc[0, 'close'] 
        df.loc[0, 'low'] = df.loc[0, 'close']
        
        return df[['datetime', 'open', 'high', 'low', 'close', 'volume']].dropna()
        
    except Exception as e:
        st.error(f"Error fetching data for {coin_id}: {e}")
        return create_sample_data()

@st.cache_data(ttl=3600)
def fetch_current_prices(assets):
    """Fetch current prices for multiple assets"""
    try:
        coin_ids = ",".join([CRYPTO_ASSETS[asset] for asset in assets])
        url = f"https://api.coingecko.com/api/v3/simple/price"
        params = {
            'ids': coin_ids,
            'vs_currencies': 'usd',
            'include_24hr_change': 'true'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except:
        return {}

def create_sample_data():
    """Create sample data when API fails"""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
    prices = 50000 + np.cumsum(np.random.randn(100) * 1000)
    data = pd.DataFrame({
        'datetime': dates,
        'open': prices - np.random.rand(100) * 100,
        'high': prices + np.random.rand(100) * 200,
        'low': prices - np.random.rand(100) * 200,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    })
    return data

@st.cache_data(ttl=3600)
def fetch_multiple_assets(assets):
    """Fetch data for multiple assets"""
    data_dict = {}
    for asset in assets:
        coin_id = CRYPTO_ASSETS[asset]
        data = fetch_crypto_data(coin_id)
        if not data.empty:
            data_dict[asset] = data
    return data_dict

@st.cache_data(ttl=1800)
def fetch_market_sentiment():
    """Fetch general market sentiment indicators"""
    try:
        # Using CoinGecko global data for sentiment
        url = "https://api.coingecko.com/api/v3/global"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        market_cap_change = data['data']['market_cap_change_percentage_24h_usd']
        fear_greed = np.random.randint(20, 80)  # Mock fear & greed index
        
        sentiment_data = {
            "fear_greed": fear_greed,
            "market_cap_change": market_cap_change,
            "weighted_sentiment": market_cap_change / 100.0  # Normalize
        }
        return sentiment_data
    except:
        return None

# ---------------------------
# ---- ENHANCED INDICATORS ---
# ---------------------------
def compute_enhanced_indicators(df, params):
    """Compute technical indicators with customizable parameters"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # EMAs with customizable periods
    ema_fast = params["EMA Cross"]["params"]["fast"]
    ema_slow = params["EMA Cross"]["params"]["slow"]
    df[f"EMA{ema_fast}"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    df[f"EMA{ema_slow}"] = df["close"].ewm(span=ema_slow, adjust=False).mean()
    
    # MACD - Handle dynamic column names
    macd_params = params["MACD"]["params"]
    macd = ta.macd(df["close"], **macd_params)
    if not macd.empty:
        # Find the correct column names
        macd_cols = [col for col in macd.columns if 'MACD_' in col and 'MACDs_' not in col and 'MACDh_' not in col]
        signal_cols = [col for col in macd.columns if 'MACDs_' in col]
        hist_cols = [col for col in macd.columns if 'MACDh_' in col]
        
        if macd_cols:
            df["MACD"] = macd[macd_cols[0]]
        if signal_cols:
            df["MACD_Signal"] = macd[signal_cols[0]]
        if hist_cols:
            df["MACD_Histogram"] = macd[hist_cols[0]]
    
    # RSI
    rsi_period = params["RSI"]["params"]["period"]
    df["RSI"] = ta.rsi(df["close"], length=rsi_period)
    
    # Bollinger Bands - Handle dynamic column names
    bb_params = params["Bollinger Bands"]["params"]
    bbands = ta.bbands(df["close"], length=bb_params["period"], std=bb_params["std"])
    if not bbands.empty:
        # Find the correct column names
        upper_cols = [col for col in bbands.columns if 'BBU_' in col]
        lower_cols = [col for col in bbands.columns if 'BBL_' in col]
        middle_cols = [col for col in bbands.columns if 'BBM_' in col]
        
        if upper_cols:
            df["BB_upper"] = bbands[upper_cols[0]]
        if lower_cols:
            df["BB_lower"] = bbands[lower_cols[0]]
        if middle_cols:
            df["BB_middle"] = bbands[middle_cols[0]]
    
    # Stochastic
    stoch_params = params["Stochastic"]["params"]
    stoch = ta.stoch(df["high"], df["low"], df["close"], **stoch_params)
    if not stoch.empty:
        k_cols = [col for col in stoch.columns if 'STOCHk_' in col]
        d_cols = [col for col in stoch.columns if 'STOCHd_' in col]
        
        if k_cols:
            df["Stoch_K"] = stoch[k_cols[0]]
        if d_cols:
            df["Stoch_D"] = stoch[d_cols[0]]
    
    # Additional indicators with error handling
    try:
        adx_result = ta.adx(df["high"], df["low"], df["close"])
        if not adx_result.empty:
            df["ADX"] = adx_result["ADX_14"]
    except:
        df["ADX"] = 0
    
    try:
        df["OBV"] = ta.obv(df["close"], df["volume"])
    except:
        df["OBV"] = 0
    
    try:
        atr_result = ta.atr(df["high"], df["low"], df["close"])
        if not atr_result.empty:
            df["ATR"] = atr_result
    except:
        df["ATR"] = 0
    
    try:
        df["CMF"] = ta.cmf(df["high"], df["low"], df["close"], df["volume"])
    except:
        df["CMF"] = 0
    
    # VWAP calculation
    try:
        df["VWAP"] = (df["volume"] * (df["high"] + df["low"] + df["close"])/3).cumsum() / df["volume"].cumsum()
    except:
        df["VWAP"] = df["close"]
    
    return df

# ---------------------------
# ---- ADVANCED ANALYTICS ---
# ---------------------------
def detect_market_regime(df):
    """Detect market regime using volatility and trend analysis"""
    if len(df) < 50:
        return "Unknown"
    
    try:
        returns = df["close"].pct_change().dropna()
        volatility = returns.rolling(window=20).std()
        
        avg_volatility = volatility.mean()
        current_vol = volatility.iloc[-1]
        
        if current_vol > avg_volatility * 1.5:
            return "High Volatility"
        elif current_vol < avg_volatility * 0.5:
            return "Low Volatility"
        else:
            return "Normal"
    except:
        return "Unknown"

def calculate_risk_metrics(df):
    """Calculate comprehensive risk metrics"""
    if len(df) < 20:
        return {}
    
    try:
        returns = df["close"].pct_change().dropna()
        
        metrics = {
            "volatility": returns.std() * np.sqrt(365) if returns.std() > 0 else 0,
            "sharpe_ratio": returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0,
            "max_drawdown": (df["close"] / df["close"].cummax() - 1).min(),
            "var_95": returns.quantile(0.05) if len(returns) > 0 else 0,
            "current_momentum": (df["close"].iloc[-1] / df["close"].iloc[-10] - 1) if len(df) >= 10 else 0
        }
        
        return metrics
    except:
        return {}

# ---------------------------
# ---- SIGNAL GENERATION ----
# ---------------------------
def generate_trading_signals(df, weights):
    """Generate comprehensive trading signals"""
    if df.empty:
        return {}
    
    try:
        last = df.iloc[-1]
        signals = {}
        
        # EMA Cross Signal
        ema_fast = weights["EMA Cross"]["params"]["fast"]
        ema_slow = weights["EMA Cross"]["params"]["slow"]
        fast_ema_col = f"EMA{ema_fast}"
        slow_ema_col = f"EMA{ema_slow}"
        
        if fast_ema_col in df.columns and slow_ema_col in df.columns:
            fast_ema = last[fast_ema_col]
            slow_ema = last[slow_ema_col]
            
            if fast_ema > slow_ema * (1 + weights["EMA Cross"]["neutral_tol"]):
                signals["EMA Cross"] = ("Bullish", f"EMA{ema_fast} > EMA{ema_slow}")
            elif fast_ema < slow_ema * (1 - weights["EMA Cross"]["neutral_tol"]):
                signals["EMA Cross"] = ("Bearish", f"EMA{ema_fast} < EMA{ema_slow}")
            else:
                signals["EMA Cross"] = ("Neutral", "EMAs converging")
        
        # RSI Signal
        if "RSI" in df.columns and not pd.isna(last["RSI"]):
            rsi = last["RSI"]
            rsi_tol = weights["RSI"]["neutral_tol"]
            if rsi_tol[0] <= rsi <= rsi_tol[1]:
                signals["RSI"] = ("Neutral", f"RSI {rsi:.1f}")
            elif rsi > 70:
                signals["RSI"] = ("Bearish", f"Overbought {rsi:.1f}")
            elif rsi < 30:
                signals["RSI"] = ("Bullish", f"Oversold {rsi:.1f}")
            else:
                signals["RSI"] = ("Neutral", f"RSI {rsi:.1f}")
        
        # MACD Signal
        if "MACD" in df.columns and "MACD_Signal" in df.columns:
            macd = last["MACD"] if not pd.isna(last["MACD"]) else 0
            macd_signal = last["MACD_Signal"] if not pd.isna(last["MACD_Signal"]) else 0
            if macd > macd_signal:
                signals["MACD"] = ("Bullish", f"MACD {macd:.3f} > Signal {macd_signal:.3f}")
            else:
                signals["MACD"] = ("Bearish", f"MACD {macd:.3f} < Signal {macd_signal:.3f}")
        
        # Bollinger Bands Signal
        if "BB_upper" in df.columns and "BB_lower" in df.columns:
            price = last["close"]
            bb_upper = last["BB_upper"] if not pd.isna(last["BB_upper"]) else price * 1.1
            bb_lower = last["BB_lower"] if not pd.isna(last["BB_lower"]) else price * 0.9
            
            if price <= bb_lower:
                signals["Bollinger Bands"] = ("Bullish", "Price at lower band")
            elif price >= bb_upper:
                signals["Bollinger Bands"] = ("Bearish", "Price at upper band")
            else:
                signals["Bollinger Bands"] = ("Neutral", "Price within bands")
        
        # Stochastic Signal
        if "Stoch_K" in df.columns and "Stoch_D" in df.columns:
            stoch_k = last["Stoch_K"] if not pd.isna(last["Stoch_K"]) else 50
            stoch_d = last["Stoch_D"] if not pd.isna(last["Stoch_D"]) else 50
            stoch_tol = weights["Stochastic"]["neutral_tol"]
            if stoch_k < stoch_tol[0] and stoch_d < stoch_tol[0]:
                signals["Stochastic"] = ("Bullish", f"Oversold K:{stoch_k:.1f} D:{stoch_d:.1f}")
            elif stoch_k > stoch_tol[1] and stoch_d > stoch_tol[1]:
                signals["Stochastic"] = ("Bearish", f"Overbought K:{stoch_k:.1f} D:{stoch_d:.1f}")
            else:
                signals["Stochastic"] = ("Neutral", f"K:{stoch_k:.1f} D:{stoch_d:.1f}")
        
        return signals
    except Exception as e:
        st.error(f"Error generating signals: {e}")
        return {}

def calculate_combined_score(signals, weights):
    """Calculate overall sentiment score"""
    try:
        score_map = {"Bullish": 1, "Neutral": 0, "Bearish": -1}
        total_weight = 0
        weighted_score = 0
        
        for indicator, signal in signals.items():
            if indicator in weights:
                weight = weights[indicator]["weight"]
                score = score_map[signal[0]]
                weighted_score += score * weight
                total_weight += weight
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0
        
        # Convert to sentiment
        if final_score > 0.1:
            return "Bullish", final_score
        elif final_score < -0.1:
            return "Bearish", final_score
        else:
            return "Neutral", final_score
    except:
        return "Neutral", 0

# ---------------------------
# ----- VISUALIZATION -------
# ---------------------------
def create_enhanced_chart(df, asset, timeframe, signals):
    """Create comprehensive trading chart"""
    if df.empty:
        return go.Figure()
    
    try:
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f'{asset} Price Chart - {timeframe}',
                'Price Movement',
                'RSI',
                'MACD'
            ),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Price data
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df['close'],
                name='Price',
                line=dict(color='blue')
            ), row=1, col=1
        )
        
        # EMAs
        if 'EMA9' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['datetime'], y=df['EMA9'], name='EMA9', line=dict(color='orange')),
                row=1, col=1
            )
        if 'EMA21' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['datetime'], y=df['EMA21'], name='EMA21', line=dict(color='red')),
                row=1, col=1
            )
        
        # Bollinger Bands
        if 'BB_upper' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['datetime'], y=df['BB_upper'], name='BB Upper', 
                          line=dict(color='gray', dash='dash'), opacity=0.7),
                row=1, col=1
            )
        if 'BB_lower' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['datetime'], y=df['BB_lower'], name='BB Lower', 
                          line=dict(color='gray', dash='dash'), opacity=0.7),
                row=1, col=1
            )
        
        # Price changes
        price_changes = df['close'].pct_change() * 100
        colors = ['red' if x < 0 else 'green' for x in price_changes]
        fig.add_trace(
            go.Bar(x=df['datetime'], y=price_changes, name='Daily Change %', marker_color=colors),
            row=2, col=1
        )
        
        # RSI
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['datetime'], y=df['RSI'], name='RSI', line=dict(color='purple')),
                row=3, col=1
            )
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
        
        # MACD
        if 'MACD' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['datetime'], y=df['MACD'], name='MACD', line=dict(color='blue')),
                row=4, col=1
            )
        if 'MACD_Signal' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['datetime'], y=df['MACD_Signal'], name='Signal', line=dict(color='red')),
                row=4, col=1
            )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            title=f"Technical Analysis - {asset} ({timeframe})"
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return go.Figure()

# ---------------------------
# ----- MAIN DASHBOARD ------
# ---------------------------
def main():
    # Market Overview Section
    st.header("📈 Market Overview")
    
    # Fetch data for all selected assets
    if not selected_assets:
        st.warning("Please select at least one asset from the sidebar.")
        return
    
    # Initialize session state for data
    if 'market_data' not in st.session_state:
        st.session_state.market_data = {}
    
    # Data loading
    with st.spinner("Loading market data..."):
        if 'data' not in st.session_state.market_data:
            st.session_state.market_data['data'] = fetch_multiple_assets(selected_assets)
    
    # Fetch current prices
    current_prices = fetch_current_prices(selected_assets)
    
    # Market Overview Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Selected Assets", len(selected_assets))
    
    with col2:
        st.metric("Timeframes Analyzed", len(selected_tfs))
    
    with col3:
        st.metric("Risk Profile", risk_tolerance)
    
    with col4:
        sentiment_data = fetch_market_sentiment()
        if sentiment_data:
            st.metric("Market Sentiment", f"{'Bullish' if sentiment_data['weighted_sentiment'] > 0 else 'Bearish'}")
    
    # Current Prices
    st.subheader("💰 Current Prices")
    price_cols = st.columns(len(selected_assets))
    for idx, asset in enumerate(selected_assets):
        with price_cols[idx]:
            coin_id = CRYPTO_ASSETS[asset]
            if current_prices and coin_id in current_prices:
                price = current_prices[coin_id]['usd']
                change_24h = current_prices[coin_id].get('usd_24h_change', 0)
                st.metric(
                    f"{asset} Price",
                    f"${price:,.2f}",
                    f"{change_24h:.2f}%"
                )
    
    # Multi-Timeframe Analysis
    st.header("⏰ Technical Analysis")
    
    for asset in selected_assets:
        st.subheader(f"🔍 {asset} Analysis")
        
        if asset in st.session_state.market_data['data']:
            df = st.session_state.market_data['data'][asset]
            if not df.empty:
                # Compute indicators
                df_indicators = compute_enhanced_indicators(df, indicator_weights)
                
                # Generate signals
                signals = generate_trading_signals(df_indicators, indicator_weights)
                
                # Calculate combined score
                overall_sentiment, score = calculate_combined_score(signals, indicator_weights)
                
                # Display sentiment card
                sentiment_color = {
                    "Bullish": "bullish",
                    "Bearish": "bearish", 
                    "Neutral": "neutral"
                }[overall_sentiment]
                
                st.markdown(
                    f"""
                    <div class='metric-card {sentiment_color}'>
                        <h3>Overall Sentiment: {overall_sentiment}</h3>
                        <p>Confidence Score: {score:.2f}</p>
                        <p>Current Price: ${df_indicators['close'].iloc[-1]:.2f}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        # Detailed analysis for the asset
        with st.expander(f"Detailed Analysis for {asset}", expanded=False):
            if asset in st.session_state.market_data['data']:
                df_detail = st.session_state.market_data['data'][asset]
                if not df_detail.empty:
                    df_indicators_detail = compute_enhanced_indicators(df_detail, indicator_weights)
                    signals_detail = generate_trading_signals(df_indicators_detail, indicator_weights)
                    
                    # Display chart
                    fig = create_enhanced_chart(df_indicators_detail, asset, "Historical", signals_detail)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk metrics
                    risk_metrics = calculate_risk_metrics(df_detail)
                    regime = detect_market_regime(df_detail)
                    
                    # Display metrics
                    st.subheader("📊 Risk Metrics")
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Volatility", f"{risk_metrics.get('volatility', 0):.2%}")
                    with metric_cols[1]:
                        st.metric("Sharpe Ratio", f"{risk_metrics.get('sharpe_ratio', 0):.2f}")
                    with metric_cols[2]:
                        st.metric("Max Drawdown", f"{risk_metrics.get('max_drawdown', 0):.2%}")
                    with metric_cols[3]:
                        st.metric("Market Regime", regime)
                    
                    # Signals table
                    st.subheader("🎯 Signal Breakdown")
                    if signals_detail:
                        signals_df = pd.DataFrame([
                            {"Indicator": ind, "Signal": sig[0], "Reason": sig[1]} 
                            for ind, sig in signals_detail.items()
                        ])
                        st.dataframe(signals_df, use_container_width=True)
                    else:
                        st.info("No signals generated for this asset")
    
    # Trading Insights
    st.header("💡 Trading Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.subheader("Strategy Suggestions")
        
        # Generate basic strategy suggestions based on signals
        for asset in selected_assets[:3]:
            if asset in st.session_state.market_data['data']:
                df = st.session_state.market_data['data'][asset]
                if not df.empty:
                    df_indicators = compute_enhanced_indicators(df, indicator_weights)
                    signals = generate_trading_signals(df_indicators, indicator_weights)
                    sentiment, score = calculate_combined_score(signals, indicator_weights)
                    
                    if sentiment == "Bullish":
                        suggestion = "🟢 Consider Buying - Bullish signals detected"
                    elif sentiment == "Bearish":
                        suggestion = "🔴 Consider Selling - Bearish signals detected"
                    else:
                        suggestion = "⚪ Hold Position - Mixed or neutral signals"
                    
                    st.write(f"**{asset}**: {suggestion} (Score: {score:.2f})")
    
    with insight_col2:
        st.subheader("Risk Assessment")
        
        # Display risk warnings
        for asset in selected_assets[:2]:
            if asset in st.session_state.market_data['data']:
                df = st.session_state.market_data['data'][asset]
                if not df.empty and len(df) > 20:
                    risk_metrics = calculate_risk_metrics(df)
                    volatility = risk_metrics.get('volatility', 0)
                    
                    if volatility > 0.8:
                        st.warning(f"🚨 {asset}: High volatility detected ({volatility:.1%})")
                    elif volatility > 0.5:
                        st.info(f"⚠️ {asset}: Elevated volatility ({volatility:.1%})")
                    else:
                        st.success(f"✅ {asset}: Normal volatility ({volatility:.1%})")
    
    # Footer
    st.markdown("---")
    st.markdown(""")
    ### 📚 How to Use This Dashboard
    
    1. **Asset Selection**: Choose cryptocurrencies from the sidebar
    2. **Indicator Configuration**: Adjust weights and parameters in sidebar  
    3. **Risk Management**: Set your risk tolerance level
    4. **Signal Interpretation**: 
       - 🟢 Bullish: Favorable buying conditions
       - 🔴 Bearish: Consider selling or shorting
       - ⚪ Neutral: Wait for clearer signals
    5. **Technical Analysis**: View charts and indicators for each asset
    
    **Data Sources**: CoinGecko API (free tier)
    **Update Frequency**: Every 5 minutes
    
    **Disclaimer**: This tool is for educational purposes only. Always do your own research and consider consulting with a qualified financial advisor before making investment decisions.
    """)

if __name__ == "__main__":
    main()
        """)
