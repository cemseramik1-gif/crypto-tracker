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

# ---------------------------
# DEPENDENCY HANDLING
# ---------------------------
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    st.error("⚠️ yfinance not installed. Installing now...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance==0.2.18"])
    import yfinance as yf
    YFINANCE_AVAILABLE = True

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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
    "BTC": "BTC-USD",
    "ETH": "ETH-USD", 
    "ADA": "ADA-USD",
    "DOT": "DOT-USD",
    "SOL": "SOL-USD",
    "MATIC": "MATIC-USD",
    "AVAX": "AVAX-USD",
    "LINK": "LINK-USD"
}

TF_OPTIONS = ["5m", "15m", "30m", "1h", "4h", "Daily", "Weekly"]

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
    default=["1h", "4h", "Daily"]
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
def fetch_crypto_data(symbol="BTC-USD", period="60d", interval="1h"):
    """Fetch cryptocurrency data from Yahoo Finance"""
    try:
        if not YFINANCE_AVAILABLE:
            st.error("yfinance not available. Using fallback data.")
            return create_sample_data()
            
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        if data.empty:
            st.warning(f"No data returned for {symbol}. Using sample data.")
            return create_sample_data()
        
        data = data.reset_index()
        data.columns = [col.lower() for col in data.columns]
        return data[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return create_sample_data()

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
def fetch_multiple_assets(assets, timeframe="1h"):
    """Fetch data for multiple assets"""
    data_dict = {}
    for asset in assets:
        symbol = CRYPTO_ASSETS[asset]
        data = fetch_crypto_data(symbol, interval=timeframe)
        if not data.empty:
            data_dict[asset] = data
    return data_dict

@st.cache_data(ttl=1800)
def fetch_market_sentiment():
    """Fetch general market sentiment indicators"""
    try:
        # Mock sentiment data - replace with actual API calls
        sentiment_data = {
            "fear_greed": np.random.randint(0, 100),
            "social_volume": np.random.randint(1000, 50000),
            "weighted_sentiment": np.random.uniform(-1, 1)
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
    
    # Ichimoku Cloud
    try:
        ichimoku = ta.ichimoku(df["high"], df["low"], df["close"])
        if ichimoku and len(ichimoku) > 0:
            df["Ichimoku_Base"] = ichimoku[0]["ITS_9"]
            df["Ichimoku_Conversion"] = ichimoku[0]["ITA_9"]
            df["Ichimoku_Span_A"] = ichimoku[0]["ISA_9"]
            df["Ichimoku_Span_B"] = ichimoku[0]["ISB_26"]
    except Exception as e:
        st.warning(f"Ichimoku calculation failed: {e}")
        # Set default values for Ichimoku columns
        df["Ichimoku_Base"] = df["close"]
        df["Ichimoku_Conversion"] = df["close"]
        df["Ichimoku_Span_A"] = df["close"]
        df["Ichimoku_Span_B"] = df["close"]
    
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
    
    # Volume oscillator
    try:
        vo_params = params["Volume Oscillator"]["params"]
        vo_result = ta.pvo(df["volume"], **vo_params)
        if not vo_result.empty:
            pvo_cols = [col for col in vo_result.columns if 'PVO_' in col]
            if pvo_cols:
                df["VO"] = vo_result[pvo_cols[0]]
    except:
        df["VO"] = 0
    
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
            "volatility": returns.std() * np.sqrt(365) if returns.std() > 0 else 0,  # Annualized
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
        
        # Ichimoku Signal
        if all(col in df.columns for col in ["Ichimoku_Span_A", "Ichimoku_Span_B"]):
            price = last["close"]
            span_a = last["Ichimoku_Span_A"] if not pd.isna(last["Ichimoku_Span_A"]) else price
            span_b = last["Ichimoku_Span_B"] if not pd.isna(last["Ichimoku_Span_B"]) else price
            
            if price > span_a and price > span_b and span_a > span_b:
                signals["Ichimoku"] = ("Bullish", "Strong uptrend")
            elif price < span_a and price < span_b and span_a < span_b:
                signals["Ichimoku"] = ("Bearish", "Strong downtrend")
            else:
                signals["Ichimoku"] = ("Neutral", "Consolidation")
        
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
                'Volume',
                'RSI & Stochastic',
                'MACD'
            ),
            row_heights=[0.5, 0.15, 0.15, 0.2]
        )
        
        # Price data
        fig.add_trace(
            go.Candlestick(
                x=df['datetime'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ), row=1, col=1
        )
        
        # EMAs
        if 'EMA9' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['datetime'], y=df['EMA9'], name='EMA9', line=dict(color='blue')),
                row=1, col=1
            )
        if 'EMA21' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['datetime'], y=df['EMA21'], name='EMA21', line=dict(color='orange')),
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
        
        # Volume
        colors = ['red' if row['close'] < row['open'] else 'green' for _, row in df.iterrows()]
        fig.add_trace(
            go.Bar(x=df['datetime'], y=df['volume'], name='Volume', marker_color=colors),
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
        
        # Stochastic
        if 'Stoch_K' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['datetime'], y=df['Stoch_K'], name='Stoch %K', line=dict(color='blue')),
                row=3, col=1
            )
        if 'Stoch_D' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['datetime'], y=df['Stoch_D'], name='Stoch %D', line=dict(color='red')),
                row=3, col=1
            )
        
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
        # MACD Histogram
        if 'MACD_Histogram' in df.columns:
            colors_macd = ['red' if x < 0 else 'green' for x in df['MACD_Histogram']]
            fig.add_trace(
                go.Bar(x=df['datetime'], y=df['MACD_Histogram'], name='MACD Histogram', 
                       marker_color=colors_macd, opacity=0.6),
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
    # Display dependency status
    if not YFINANCE_AVAILABLE:
        st.error("⚠️ yfinance is not available. Some features may be limited.")
    
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
        for tf in selected_tfs:
            if tf not in st.session_state.market_data:
                st.session_state.market_data[tf] = fetch_multiple_assets(selected_assets, tf)
    
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
    
    # Multi-Timeframe Analysis
    st.header("⏰ Multi-Timeframe Analysis")
    
    for asset in selected_assets:
        st.subheader(f"🔍 {asset} Analysis")
        
        asset_cols = st.columns(len(selected_tfs))
        
        for idx, tf in enumerate(selected_tfs):
            with asset_cols[idx]:
                if asset in st.session_state.market_data[tf]:
                    df = st.session_state.market_data[tf][asset]
                    if not df.empty:
                        # Compute indicators
                        df_indicators = compute_enhanced_indicators(df, indicator_weights)
                        
                        # Generate signals
                        signals = generate_trading_signals(df_indicators, indicator_weights)
                        
                        # Calculate combined score
                        overall_sentiment, score = calculate_combined_score(signals, indicator_weights)
                        
                        # Display timeframe card
                        sentiment_color = {
                            "Bullish": "bullish",
                            "Bearish": "bearish", 
                            "Neutral": "neutral"
                        }[overall_sentiment]
                        
                        st.markdown(
                            f"""
                            <div class='metric-card {sentiment_color}'>
                                <h4>{tf}</h4>
                                <h3>{overall_sentiment}</h3>
                                <p>Score: {score:.2f}</p>
                                <p>Price: ${df_indicators['close'].iloc[-1]:.2f}</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
        
        # Detailed analysis for the asset
        with st.expander(f"Detailed Analysis for {asset}", expanded=False):
            selected_tf_detail = st.selectbox(
                f"Select timeframe for detailed chart", 
                selected_tfs, 
                key=f"detail_{asset}"
            )
            
            if asset in st.session_state.market_data[selected_tf_detail]:
                df_detail = st.session_state.market_data[selected_tf_detail][asset]
                if not df_detail.empty:
                    df_indicators_detail = compute_enhanced_indicators(df_detail, indicator_weights)
                    signals_detail = generate_trading_signals(df_indicators_detail, indicator_weights)
                    
                    # Display chart
                    fig = create_enhanced_chart(df_indicators_detail, asset, selected_tf_detail, signals_detail)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk metrics
                    risk_metrics = calculate_risk_metrics(df_detail)
                    regime = detect_market_regime(df_detail)
                    
                    # Display metrics
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
                    st.subheader("📊 Signal Breakdown")
                    signals_df = pd.DataFrame([
                        {"Indicator": ind, "Signal": sig[0], "Reason": sig[1]} 
                        for ind, sig in signals_detail.items()
                    ])
                    st.dataframe(signals_df, use_container_width=True)
    
    # Correlation Analysis
    if len(selected_assets) > 1:
        st.header("🔄 Correlation Analysis")
        
        # Calculate correlations
        correlation_data = []
        for tf in selected_tfs[:1]:  # Use first timeframe for correlation
            prices = {}
            for asset in selected_assets:
                if asset in st.session_state.market_data[tf]:
                    df = st.session_state.market_data[tf][asset]
                    if not df.empty and len(df) > 10:
                        prices[asset] = df['close'].pct_change().dropna()
            
            if len(prices) > 1:
                corr_df = pd.DataFrame(prices).corr()
                
                # Plot correlation heatmap
                fig_corr = px.imshow(
                    corr_df,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title=f"Asset Correlation Matrix ({tf})"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                break
    
    # Trading Insights
    st.header("💡 Trading Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.subheader("Strategy Suggestions")
        
        # Generate basic strategy suggestions based on signals
        for asset in selected_assets[:3]:  # Limit to first 3 assets
            all_tf_signals = []
            for tf in selected_tfs:
                if asset in st.session_state.market_data[tf]:
                    df = st.session_state.market_data[tf][asset]
                    if not df.empty:
                        df_indicators = compute_enhanced_indicators(df, indicator_weights)
                        signals = generate_trading_signals(df_indicators, indicator_weights)
                        sentiment, score = calculate_combined_score(signals, indicator_weights)
                        all_tf_signals.append((tf, sentiment, score))
            
            if all_tf_signals:
                bullish_count = sum(1 for _, sentiment
