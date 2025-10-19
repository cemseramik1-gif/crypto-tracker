import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas_ta as ta
import asyncio
import aiohttp
from textblob import TextBlob
import tweepy
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
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
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        if data.empty:
            return pd.DataFrame()
        
        data = data.reset_index()
        data.columns = [col.lower() for col in data.columns]
        return data[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

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
    
    # MACD
    macd_params = params["MACD"]["params"]
    macd = ta.macd(df["close"], **macd_params)
    df["MACD"] = macd[f"MACD_{macd_params['fast']}_{macd_params['slow']}_{macd_params['signal']}"]
    df["MACD_Signal"] = macd[f"MACDs_{macd_params['fast']}_{macd_params['slow']}_{macd_params['signal']}"]
    df["MACD_Histogram"] = macd[f"MACDh_{macd_params['fast']}_{macd_params['slow']}_{macd_params['signal']}"]
    
    # RSI
    rsi_period = params["RSI"]["params"]["period"]
    df["RSI"] = ta.rsi(df["close"], length=rsi_period)
    
    # Bollinger Bands
    bb_params = params["Bollinger Bands"]["params"]
    bbands = ta.bbands(df["close"], length=bb_params["period"], std=bb_params["std"])
    df["BB_upper"] = bbands[f"BBU_{bb_params['period']}_{bb_params['std']}.0"]
    df["BB_lower"] = bbands[f"BBL_{bb_params['period']}_{bb_params['std']}.0"]
    df["BB_middle"] = bbands[f"BBM_{bb_params['period']}_{bb_params['std']}.0"]
    
    # Stochastic
    stoch_params = params["Stochastic"]["params"]
    stoch = ta.stoch(df["high"], df["low"], df["close"], **stoch_params)
    df["Stoch_K"] = stoch[f"STOCHk_{stoch_params['k']}_{stoch_params['d']}"]
    df["Stoch_D"] = stoch[f"STOCHd_{stoch_params['k']}_{stoch_params['d']}"]
    
    # Ichimoku Cloud
    ichimoku = ta.ichimoku(df["high"], df["low"], df["close"])
    df["Ichimoku_Base"] = ichimoku["its_26"]
    df["Ichimoku_Conversion"] = ichimoku["ita_9"]
    df["Ichimoku_Span_A"] = ichimoku["isa_9"]
    df["Ichimoku_Span_B"] = ichimoku["isb_26"]
    
    # Additional indicators
    df["ADX"] = ta.adx(df["high"], df["low"], df["close"])["ADX_14"]
    df["OBV"] = ta.obv(df["close"], df["volume"])
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"])
    df["CMF"] = ta.cmf(df["high"], df["low"], df["close"], df["volume"])
    df["VWAP"] = (df["volume"] * (df["high"] + df["low"] + df["close"])/3).cumsum() / df["volume"].cumsum()
    
    # Volume indicators
    vo_params = params["Volume Oscillator"]["params"]
    df["VO"] = ta.pvo(df["volume"], **vo_params)["PVO_12_26_9"]
    
    return df

# ---------------------------
# ---- ADVANCED ANALYTICS ---
# ---------------------------
def detect_market_regime(df):
    """Detect market regime using volatility and trend analysis"""
    if len(df) < 50:
        return "Unknown"
    
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

def calculate_risk_metrics(df):
    """Calculate comprehensive risk metrics"""
    if len(df) < 20:
        return {}
    
    returns = df["close"].pct_change().dropna()
    
    metrics = {
        "volatility": returns.std() * np.sqrt(365),  # Annualized
        "sharpe_ratio": returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0,
        "max_drawdown": (df["close"] / df["close"].cummax() - 1).min(),
        "var_95": returns.quantile(0.05),
        "current_momentum": (df["close"].iloc[-1] / df["close"].iloc[-10] - 1) if len(df) >= 10 else 0
    }
    
    return metrics

# ---------------------------
# ---- SIGNAL GENERATION ----
# ---------------------------
def generate_trading_signals(df, weights):
    """Generate comprehensive trading signals"""
    if df.empty:
        return {}
    
    last = df.iloc[-1]
    signals = {}
    
    # EMA Cross Signal
    ema_fast = weights["EMA Cross"]["params"]["fast"]
    ema_slow = weights["EMA Cross"]["params"]["slow"]
    fast_ema = last[f"EMA{ema_fast}"]
    slow_ema = last[f"EMA{ema_slow}"]
    
    if fast_ema > slow_ema * (1 + weights["EMA Cross"]["neutral_tol"]):
        signals["EMA Cross"] = ("Bullish", f"EMA{ema_fast} > EMA{ema_slow}")
    elif fast_ema < slow_ema * (1 - weights["EMA Cross"]["neutral_tol"]):
        signals["EMA Cross"] = ("Bearish", f"EMA{ema_fast} < EMA{ema_slow}")
    else:
        signals["EMA Cross"] = ("Neutral", "EMAs converging")
    
    # RSI Signal
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
    macd = last["MACD"]
    macd_signal = last["MACD_Signal"]
    if macd > macd_signal:
        signals["MACD"] = ("Bullish", f"MACD {macd:.3f} > Signal {macd_signal:.3f}")
    else:
        signals["MACD"] = ("Bearish", f"MACD {macd:.3f} < Signal {macd_signal:.3f}")
    
    # Bollinger Bands Signal
    price = last["close"]
    bb_upper = last["BB_upper"]
    bb_lower = last["BB_lower"]
    if price <= bb_lower:
        signals["Bollinger Bands"] = ("Bullish", "Price at lower band")
    elif price >= bb_upper:
        signals["Bollinger Bands"] = ("Bearish", "Price at upper band")
    else:
        signals["Bollinger Bands"] = ("Neutral", "Price within bands")
    
    # Stochastic Signal
    stoch_k = last["Stoch_K"]
    stoch_d = last["Stoch_D"]
    stoch_tol = weights["Stochastic"]["neutral_tol"]
    if stoch_k < stoch_tol[0] and stoch_d < stoch_tol[0]:
        signals["Stochastic"] = ("Bullish", f"Oversold K:{stoch_k:.1f} D:{stoch_d:.1f}")
    elif stoch_k > stoch_tol[1] and stoch_d > stoch_tol[1]:
        signals["Stochastic"] = ("Bearish", f"Overbought K:{stoch_k:.1f} D:{stoch_d:.1f}")
    else:
        signals["Stochastic"] = ("Neutral", f"K:{stoch_k:.1f} D:{stoch_d:.1f}")
    
    # Ichimoku Signal
    price = last["close"]
    span_a = last["Ichimoku_Span_A"]
    span_b = last["Ichimoku_Span_B"]
    if price > span_a and price > span_b and span_a > span_b:
        signals["Ichimoku"] = ("Bullish", "Strong uptrend")
    elif price < span_a and price < span_b and span_a < span_b:
        signals["Ichimoku"] = ("Bearish", "Strong downtrend")
    else:
        signals["Ichimoku"] = ("Neutral", "Consolidation")
    
    return signals

def calculate_combined_score(signals, weights):
    """Calculate overall sentiment score"""
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

# ---------------------------
# ----- VISUALIZATION -------
# ---------------------------
def create_enhanced_chart(df, asset, timeframe, signals):
    """Create comprehensive trading chart"""
    if df.empty:
        return go.Figure()
    
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
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['EMA9'], name='EMA9', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['EMA21'], name='EMA21', line=dict(color='orange')),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['BB_upper'], name='BB Upper', 
                  line=dict(color='gray', dash='dash'), opacity=0.7),
        row=1, col=1
    )
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
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['RSI'], name='RSI', line=dict(color='purple')),
        row=3, col=1
    )
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
    
    # Stochastic
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['Stoch_K'], name='Stoch %K', line=dict(color='blue')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['Stoch_D'], name='Stoch %D', line=dict(color='red')),
        row=3, col=1
    )
    
    # MACD
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['MACD'], name='MACD', line=dict(color='blue')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['MACD_Signal'], name='Signal', line=dict(color='red')),
        row=4, col=1
    )
    # MACD Histogram
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
                bullish_count = sum(1 for _, sentiment, _ in all_tf_signals if sentiment == "Bullish")
                total_count = len(all_tf_signals)
                
                if bullish_count == total_count:
                    suggestion = "🟢 Strong Buy - Bullish across all timeframes"
                elif bullish_count >= total_count * 0.7:
                    suggestion = "🟡 Cautious Buy - Mostly bullish"
                elif bullish_count <= total_count * 0.3:
                    suggestion = "🔴 Consider Sell - Mostly bearish"
                else:
                    suggestion = "⚪ Hold - Mixed signals"
                
                st.write(f"**{asset}**: {suggestion}")
    
    with insight_col2:
        st.subheader("Risk Assessment")
        
        # Display risk warnings
        for asset in selected_assets[:2]:
            for tf in selected_tfs[-1:]:  # Use highest timeframe for risk assessment
                if asset in st.session_state.market_data[tf]:
                    df = st.session_state.market_data[tf][asset]
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
    st.markdown("""
    ### 📚 How to Use This Dashboard
    
    1. **Asset Selection**: Choose cryptocurrencies from the sidebar
    2. **Timeframe Analysis**: View multiple timeframes simultaneously
    3. **Indicator Configuration**: Adjust weights and parameters in sidebar
    4. **Risk Management**: Set your risk tolerance level
    5. **Signal Interpretation**: 
       - 🟢 Bullish: Favorable buying conditions
       - 🔴 Bearish: Consider selling or shorting
       - ⚪ Neutral: Wait for clearer signals
    
    **Disclaimer**: This tool is for educational purposes only. Always do your own research and consider consulting with a qualified financial advisor before making investment decisions.
    """)

if __name__ == "__main__":
    main()
