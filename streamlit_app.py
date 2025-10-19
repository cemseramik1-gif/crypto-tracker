import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import plotly.graph_objects as go
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

st.markdown('<div class="main-header">🚀 Crypto Alpha Dashboard</div>', unsafe_allow_html=True)

# Settings
INDICATOR_DEFAULTS = {
    "EMA Cross": {"weight": 0.12, "neutral_tol": 0.002, "params": {"fast": 9, "slow": 21}},
    "MACD": {"weight": 0.10, "neutral_tol": 0.0, "params": {"fast": 12, "slow": 26, "signal": 9}},
    "RSI": {"weight": 0.08, "neutral_tol": (40, 60), "params": {"period": 14}},
    "ADX": {"weight": 0.08, "neutral_tol": 20, "params": {"period": 14}},
    "Bollinger Bands": {"weight": 0.05, "neutral_tol": None, "params": {"period": 20, "std": 2}},
    "Stochastic": {"weight": 0.06, "neutral_tol": (20, 80), "params": {"k": 14, "d": 3}}
}

CRYPTO_ASSETS = {
    "BTC": "bitcoin",
    "ETH": "ethereum", 
    "ADA": "cardano",
    "SOL": "solana"
}

# Sidebar
st.sidebar.header("Dashboard Controls")

selected_assets = st.sidebar.multiselect(
    "Choose Assets", 
    list(CRYPTO_ASSETS.keys()), 
    default=["BTC", "ETH"]
)

risk_tolerance = st.sidebar.select_slider(
    "Risk Tolerance", 
    options=["Conservative", "Moderate", "Aggressive"],
    value="Moderate"
)

# Indicator Controls
st.sidebar.header("Indicator Configuration")

indicator_weights = {}
for ind, config in INDICATOR_DEFAULTS.items():
    st.sidebar.subheader(f"{ind}")
    w = st.sidebar.slider(f"Weight", 0.0, 0.2, config['weight'], 0.01, key=f"weight_{ind}")
    
    indicator_weights[ind] = {
        "weight": w, 
        "neutral_tol": config['neutral_tol'],
        "params": config['params']
    }

# Data Fetch
@st.cache_data(ttl=300)
def fetch_crypto_data(coin_id="bitcoin", days=90):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily' if days > 90 else 'hourly'
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        df['open'] = df['price'].shift(1)
        df['high'] = df[['open', 'price']].max(axis=1)
        df['low'] = df[['open', 'price']].min(axis=1)
        df['close'] = df['price']
        df['volume'] = 0
        
        df.loc[0, 'open'] = df.loc[0, 'close']
        df.loc[0, 'high'] = df.loc[0, 'close'] 
        df.loc[0, 'low'] = df.loc[0, 'close']
        
        return df[['datetime', 'open', 'high', 'low', 'close', 'volume']].dropna()
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return create_sample_data()

@st.cache_data(ttl=3600)
def fetch_current_prices(assets):
    try:
        coin_ids = ",".join([CRYPTO_ASSETS[asset] for asset in assets])
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            'ids': coin_ids,
            'vs_currencies': 'usd',
            'include_24hr_change': 'true'
        }
        
        response = requests.get(url, params=params, timeout=10)
        return response.json()
    except Exception:
        return {}

def create_sample_data():
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
    data_dict = {}
    for asset in assets:
        coin_id = CRYPTO_ASSETS[asset]
        data = fetch_crypto_data(coin_id, days=90)
        if not data.empty:
            data_dict[asset] = data
    return data_dict

# Enhanced Indicators
def compute_enhanced_indicators(df, params):
    if df.empty:
        return df
    
    df = df.copy()
    
    ema_fast = params["EMA Cross"]['params']['fast']
    ema_slow = params["EMA Cross"]['params']['slow']
    df[f"EMA{ema_fast}"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    df[f"EMA{ema_slow}"] = df["close"].ewm(span=ema_slow, adjust=False).mean()
    
    macd_params = params["MACD"]['params']
    macd = ta.macd(df["close"], **macd_params)
    if not macd.empty:
        macd_cols = [col for col in macd.columns if 'MACD_' in col and 'MACDs_' not in col and 'MACDh_' not in col]
        signal_cols = [col for col in macd.columns if 'MACDs_' in col]
        if macd_cols:
            df["MACD"] = macd[macd_cols[0]]
        if signal_cols:
            df["MACD_Signal"] = macd[signal_cols[0]]
    
    rsi_period = params["RSI"]['params']['period']
    df["RSI"] = ta.rsi(df["close"], length=rsi_period)
    
    bb_params = params["Bollinger Bands"]['params']
    bbands = ta.bbands(df["close"], length=bb_params['period'], std=bb_params['std'])
    if not bbands.empty:
        upper_cols = [col for col in bbands.columns if 'BBU_' in col]
        lower_cols = [col for col in bbands.columns if 'BBL_' in col]
        if upper_cols:
            df["BB_upper"] = bbands[upper_cols[0]]
        if lower_cols:
            df["BB_lower"] = bbands[lower_cols[0]]
    
    stoch_params = params["Stochastic"]['params']
    stoch = ta.stoch(df["high"], df["low"], df["close"], **stoch_params)
    if not stoch.empty:
        k_cols = [col for col in stoch.columns if 'STOCHk_' in col]
        d_cols = [col for col in stoch.columns if 'STOCHd_' in col]
        if k_cols:
            df["Stoch_K"] = stoch[k_cols[0]]
        if d_cols:
            df["Stoch_D"] = stoch[d_cols[0]]
    
    return df

# Signal Generation
def generate_trading_signals(df, weights):
    if df.empty or len(df) < 2:
        return {}
    
    try:
        last = df.iloc[-1]
        signals = {}
        
        ema_fast = weights["EMA Cross"]['params']['fast']
        ema_slow = weights["EMA Cross"]['params']['slow']
        fast_ema_col = f"EMA{ema_fast}"
        slow_ema_col = f"EMA{ema_slow}"
        
        if fast_ema_col in df.columns and slow_ema_col in df.columns:
            fast_ema = last[fast_ema_col]
            slow_ema = last[slow_ema_col]
            
            if fast_ema > slow_ema:
                signals["EMA Cross"] = ("Bullish", f"EMA{ema_fast} > EMA{ema_slow}")
            else:
                signals["EMA Cross"] = ("Bearish", f"EMA{ema_fast} < EMA{ema_slow}")
        
        if "RSI" in df.columns:
            rsi = last["RSI"]
            if rsi > 70:
                signals["RSI"] = ("Bearish", f"Overbought ({rsi:.1f})")
            elif rsi < 30:
                signals["RSI"] = ("Bullish", f"Oversold ({rsi:.1f})")
            else:
                signals["RSI"] = ("Neutral", f"RSI ({rsi:.1f})")
        
        if "MACD" in df.columns and "MACD_Signal" in df.columns:
            macd = last["MACD"]
            macd_signal = last["MACD_Signal"]
            if macd > macd_signal:
                signals["MACD"] = ("Bullish", "MACD > Signal")
            else:
                signals["MACD"] = ("Bearish", "MACD < Signal")

        if "BB_upper" in df.columns and "BB_lower" in df.columns:
            price = last["close"]
            bb_upper = last["BB_upper"]
            bb_lower = last["BB_lower"]
            
            if price <= bb_lower:
                signals["Bollinger Bands"] = ("Bullish", "Price at lower band")
            elif price >= bb_upper:
                signals["Bollinger Bands"] = ("Bearish", "Price at upper band")
            else:
                signals["Bollinger Bands"] = ("Neutral", "Price within bands")
        
        return signals
    except Exception as e:
        return {}

def calculate_combined_score(signals, weights):
    try:
        score_map = {"Bullish": 1, "Neutral": 0, "Bearish": -1}
        total_weight = 0
        weighted_score = 0
        
        for indicator, signal in signals.items():
            if indicator in weights:
                weight = weights[indicator]['weight']
                score = score_map.get(signal[0], 0)
                weighted_score += score * weight
                total_weight += weight
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0
        
        if final_score > 0.1:
            return "Bullish", final_score
        elif final_score < -0.1:
            return "Bearish", final_score
        else:
            return "Neutral", final_score
    except Exception:
        return "Neutral", 0

# Visualization
def create_enhanced_chart(df, asset):
    if df.empty:
        return go.Figure()
    
    try:
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f'{asset} Price Chart',
                'RSI',
                'MACD'
            ),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=df['close'],
                name='Price',
                line=dict(color='#1f77b4', width=2)
            ), row=1, col=1
        )
        
        ema_fast = indicator_weights["EMA Cross"]['params']['fast']
        ema_slow = indicator_weights["EMA Cross"]['params']['slow']

        if f'EMA{ema_fast}' in df.columns:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df[f'EMA{ema_fast}'], name=f'EMA{ema_fast}', line=dict(color='orange')), row=1, col=1)
        if f'EMA{ema_slow}' in df.columns:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df[f'EMA{ema_slow}'], name=f'EMA{ema_slow}', line=dict(color='red')), row=1, col=1)
        
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['datetime'], y=df['RSI'], name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        if 'MACD' in df.columns:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
        if 'MACD_Signal' in df.columns:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['MACD_Signal'], name='Signal', line=dict(color='red')), row=3, col=1)

        fig.update_layout(
            height=700,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    except Exception as e:
        return go.Figure()

# Main Dashboard
def main():
    if not selected_assets:
        st.warning("Please select at least one asset from the sidebar.")
        return
    
    if 'market_data' not in st.session_state:
        st.session_state.market_data = {}
    
    with st.spinner("Loading market data..."):
        st.session_state.market_data['data'] = fetch_multiple_assets(selected_assets)
    
    current_prices = fetch_current_prices(selected_assets)
    
    st.header("Market Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Selected Assets", len(selected_assets))
    with col2:
        st.metric("Risk Profile", risk_tolerance)
    with col3:
        st.metric("Data Points", "Live")
    
    st.subheader("Current Prices")
    price_cols = st.columns(len(selected_assets))
    for idx, asset in enumerate(selected_assets):
        with price_cols[idx]:
            coin_id = CRYPTO_ASSETS[asset]
            price = 0.0
            change_24h = 0.0
            
            if current_prices and coin_id in current_prices:
                price = current_prices[coin_id]['usd']
                change_24h = current_prices[coin_id].get('usd_24h_change', 0)
                
            st.metric(
                f"{asset} Price",
                f"${price:,.2f}",
                f"{change_24h:.2f}%"
            )
    
    st.header("Technical Analysis")
    
    for asset in selected_assets:
        st.subheader(f"{asset} Analysis")
        
        if asset not in st.session_state.market_data['data']:
            continue

        df = st.session_state.market_data['data'][asset]
        if df.empty:
            continue

        df_indicators = compute_enhanced_indicators(df, indicator_weights)
        signals = generate_trading_signals(df_indicators, indicator_weights)
        overall_sentiment, score = calculate_combined_score(signals, indicator_weights)
        
        sentiment_color = {
            "Bullish": "bullish",
            "Bearish": "bearish", 
            "Neutral": "neutral"
        }.get(overall_sentiment, "neutral")
        
        st.markdown(
            f"""
            <div class='metric-card {sentiment_color}'>
                <h3>Overall Sentiment: {overall_sentiment}</h3>
                <p>Confidence Score: {score:.3f}</p>
                <p>Current Price: ${df_indicators['close'].iloc[-1]:,.2f}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        with st.expander(f"Detailed Analysis for {asset}"):
            fig = create_enhanced_chart(df_indicators, asset)
            st.plotly_chart(fig, use_container_width=True)
            
            if signals:
                signals_df = pd.DataFrame([
                    {"Indicator": ind, "Signal": sig[0], "Reason": sig[1]} 
                    for ind, sig in signals.items()
                ])
                st.dataframe(signals_df, use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    ### How to Use This Dashboard
    
    1. **Asset Selection**: Choose cryptocurrencies from the sidebar
    2. **Indicator Configuration**: Adjust weights in sidebar 
    3. **Signal Interpretation**: 
        - 🟢 Bullish: Favorable buying conditions
        - 🔴 Bearish: Consider selling
        - ⚪ Neutral: Wait for clearer signals
    
    **Data Sources**: CoinGecko API
    **Disclaimer**: For educational purposes only.
    """)

if __name__ == "__main__":
    main()
