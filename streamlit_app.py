# streamlit_live_crypto_dashboard_weights.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import pandas_ta as ta
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Crypto Dashboard — Weighted Indicators")

st.title("Crypto & On-Chain Dashboard — Weighted Indicators")

# ---------------------- Helpers ----------------------
@st.cache_data(ttl=300)
def fetch_price_data(coin='bitcoin', vs_currency='usd', days=90):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {'vs_currency': vs_currency, 'days': days}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    j = resp.json()
    df = pd.DataFrame(j['prices'], columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df.drop('timestamp', axis=1, inplace=True)
    return df

@st.cache_data(ttl=300)
def fetch_blockchain_metrics():
    dates = pd.date_range(end=datetime.today(), periods=90)
    df = pd.DataFrame({
        'date': dates,
        'NUPL': np.random.uniform(0.2, 0.8, size=len(dates)),
        'Supply_in_Profit_pct': np.random.uniform(50, 90, size=len(dates)),
        'Miner_Netflow': np.random.uniform(-2000, 2000, size=len(dates)),
        'Exchange_Netflow': np.random.uniform(-5000, 5000, size=len(dates)),
        'Funding_Rate': np.random.uniform(-0.01, 0.01, size=len(dates))
    })
    df.set_index('date', inplace=True)
    return df

def safe_ta(fn, *args, **kwargs):
    try:
        r = fn(*args, **kwargs)
        if isinstance(r, pd.DataFrame): return r.iloc[:, 0]
        return r
    except Exception:
        return pd.Series(np.nan, index=args[0].index)

def compute_indicators(df):
    df = df.copy()
    df['EMA9'] = safe_ta(ta.ema, df['price'], length=9)
    df['EMA21'] = safe_ta(ta.ema, df['price'], length=21)
    df['RSI'] = safe_ta(ta.rsi, df['price'], length=14)
    macd_df = ta.macd(df['price'])
    if isinstance(macd_df, pd.DataFrame):
        df['MACD'] = macd_df.iloc[:, 0]
        df['MACD_signal'] = macd_df.iloc[:, 1]
    else:
        df['MACD'], df['MACD_signal'] = np.nan, np.nan
    df['ATR'] = safe_ta(ta.atr, df['price'], df['price'], df['price'])
    df['ADX'] = safe_ta(ta.adx, df['price'], df['price'], df['price'])
    return df

# ---------------------- Sidebar Controls ----------------------
st.sidebar.header("🔧 Indicator Controls")

st.sidebar.subheader("EMA & MACD Sensitivity")
ema_sensitivity = st.sidebar.slider("EMA Cross Sensitivity", 0.1, 5.0, 1.0, 0.1)
macd_threshold = st.sidebar.slider("MACD Signal Threshold", 0.001, 0.05, 0.01, 0.001)

st.sidebar.subheader("RSI Thresholds")
rsi_overbought = st.sidebar.slider("RSI Overbought", 60, 90, 70)
rsi_oversold = st.sidebar.slider("RSI Oversold", 10, 50, 30)

st.sidebar.subheader("On-Chain Metrics")
nupl_top = st.sidebar.slider("NUPL Top Threshold", 0.4, 0.9, 0.5)
nupl_bottom = st.sidebar.slider("NUPL Bottom Threshold", 0.1, 0.4, 0.25)
profit_top = st.sidebar.slider("Supply in Profit High (%)", 60, 95, 80)
profit_bottom = st.sidebar.slider("Supply in Profit Low (%)", 30, 80, 60)
funding_thresh = st.sidebar.slider("Funding Rate Sensitivity", 0.001, 0.02, 0.002, 0.001)

tolerances = {
    'ema': ema_sensitivity,
    'macd': macd_threshold,
    'rsi_overbought': rsi_overbought,
    'rsi_oversold': rsi_oversold,
    'nupl_top': nupl_top,
    'nupl_bottom': nupl_bottom,
    'profit_top': profit_top,
    'profit_bottom': profit_bottom,
    'funding': funding_thresh,
}

# ---------------------- Data ----------------------
coin = st.sidebar.selectbox("Select coin", ['bitcoin', 'ethereum'])
days = st.sidebar.selectbox("Timeframe (days)", [30, 60, 90, 180])
price_df = fetch_price_data(coin, days=days)
chain_df = fetch_blockchain_metrics()
df = price_df.join(chain_df, how='left')
df = compute_indicators(df)
latest = df.iloc[-1]

# ---------------------- Signal Evaluation ----------------------
def get_signal_color(value, metric, tol):
    if pd.isna(value): return 'gray', f"{metric}: N/A"
    if metric == 'EMA_cross':
        if value > tol['ema']: return 'green', f"EMA9 above EMA21 ({value:.2f})"
        if value < -tol['ema']: return 'red', f"EMA9 below EMA21 ({value:.2f})"
        return 'gray', f"Neutral ({value:.2f})"
    if metric == 'RSI':
        if value > tol['rsi_overbought']: return 'red', f"Overbought ({value:.1f})"
        if value < tol['rsi_oversold']: return 'green', f"Oversold ({value:.1f})"
        return 'gray', f"Neutral ({value:.1f})"
    if metric == 'MACD':
        if value > tol['macd']: return 'green', f"Bullish ({value:.3f})"
        if value < -tol['macd']: return 'red', f"Bearish ({value:.3f})"
        return 'gray', f"Flat ({value:.3f})"
    if metric == 'NUPL':
        if value > tol['nupl_top']: return 'red', f"High ({value:.2f})"
        if value < tol['nupl_bottom']: return 'green', f"Low ({value:.2f})"
        return 'gray', f"Neutral ({value:.2f})"
    if metric == 'Supply_in_Profit_pct':
        if value > tol['profit_top']: return 'red', f"High ({value:.1f}%)"
        if value < tol['profit_bottom']: return 'green', f"Low ({value:.1f}%)"
        return 'gray', f"Neutral ({value:.1f}%)"
    if metric == 'Funding_Rate':
        if value > tol['funding']: return 'red', f"Longs dominant ({value:.4f})"
        if value < -tol['funding']: return 'green', f"Shorts dominant ({value:.4f})"
        return 'gray', f"Neutral ({value:.4f})"
    return 'gray', f"{metric}: {value}"

# ---------------------- Layout: Price Chart ----------------------
st.subheader(f"{coin.capitalize()} Price")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['price'], mode='lines', name='Price'))
fig.update_layout(height=480, margin=dict(l=20,r=20,t=40,b=10))
st.plotly_chart(fig, use_container_width=True)

# ---------------------- Indicator Boxes ----------------------
st.subheader("Indicators & On-Chain Metrics")

metrics = [
    ('EMA_cross', (latest['EMA9'] - latest['EMA21'])),
    ('RSI', latest['RSI']),
    ('MACD', latest['MACD'] - latest['MACD_signal']),
    ('NUPL', latest['NUPL']),
    ('Supply_in_Profit_pct', latest['Supply_in_Profit_pct']),
    ('Funding_Rate', latest['Funding_Rate']),
]

cols_per_row = 4
rows = (len(metrics) + cols_per_row - 1) // cols_per_row
for r in range(rows):
    start, end = r * cols_per_row, min((r + 1) * cols_per_row, len(metrics))
    cols = st.columns(cols_per_row, gap="large")
    for i, (metric, val) in enumerate(metrics[start:end]):
        color, text = get_signal_color(val, metric, tolerances)
        cols[i].markdown(
            f"""
            <div style="
                background-color:{color};
                padding:18px;
                border-radius:12px;
                text-align:center;
                height:120px;
                box-shadow:0 2px 4px rgba(0,0,0,0.15);
                transition: all 0.3s ease;
            ">
                <div style="font-size:17px;font-weight:700;">{metric}</div>
                <div style="font-size:22px;margin-top:4px;font-weight:700;">
                    {'' if pd.isna(val) else f"{val:.3f}"}
                </div>
                <div style="font-size:13px;opacity:0.85;margin-top:4px;">
                    {text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------- Forecast Placeholder ----------------------
st.subheader("30-Day Scenario Forecast")
last_price = latest['price']
forecast = pd.DataFrame({
    'date': [datetime.now() + timedelta(days=i) for i in range(1, 31)],
    'price': last_price * (1 + np.cumsum(np.random.normal(0, 0.01, 30))),
})
st.line_chart(forecast.set_index('date')['price'])
