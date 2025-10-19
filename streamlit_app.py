# streamlit_live_crypto_dashboard_full.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import pandas_ta as ta
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Crypto Dashboard — Price & On-Chain")

st.title("Crypto & On-Chain Dashboard — Red/Green/Gray Signals")

# ---------------------- Helpers ----------------------
@st.cache_data(ttl=300)
def fetch_price_data(coin='bitcoin', vs_currency='usd', days=90):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {'vs_currency': vs_currency, 'days': days}
    resp = requests.get(url, params=params).json()
    df = pd.DataFrame(resp['prices'], columns=['timestamp', 'price'])
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
        'Funding_Rate': np.random.uniform(-0.01, 0.01, size=len(dates)),
    })
    df.set_index('date', inplace=True)
    return df

# ---------------------- Indicators ----------------------
def compute_indicators(df):
    df = df.copy()
    df['EMA9'] = ta.ema(df['price'], length=9)
    df['EMA21'] = ta.ema(df['price'], length=21)
    df['EMA50'] = ta.ema(df['price'], length=50)
    df['EMA200'] = ta.ema(df['price'], length=200)
    df['RSI'] = ta.rsi(df['price'], length=14)
    macd = ta.macd(df['price'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    bbands = ta.bbands(df['price'])
    for col in bbands.columns:
        if 'BBU' in col: df['BB_upper'] = bbands[col]
        elif 'BBL' in col: df['BB_lower'] = bbands[col]
    df['ATR'] = ta.atr(df['price'], df['price'], df['price'], length=14)
    df['OBV'] = ta.obv(df['price'], df.get('volume', pd.Series(1, index=df.index)))
    return df

# ---------------------- Signals & Colors ----------------------
def get_signal_color(value, metric):
    # return (color, explanation)
    if metric == 'EMA_cross':
        diff = latest['EMA9'] - latest['EMA21']
        if diff > 0.5: return 'green', f"EMA9 above EMA21 (+{diff:.2f})"
        elif diff < -0.5: return 'red', f"EMA9 below EMA21 ({diff:.2f})"
        else: return 'gray', f"EMA neutral ({diff:.2f})"
    elif metric == 'RSI':
        if value > 60: return 'red', f"Overbought ({value:.1f})"
        elif value < 40: return 'green', f"Oversold ({value:.1f})"
        else: return 'gray', f"Neutral ({value:.1f})"
    elif metric == 'MACD':
        diff = latest['MACD'] - latest['MACD_signal']
        if diff > 0.01: return 'green', f"Bullish ({diff:.3f})"
        elif diff < -0.01: return 'red', f"Bearish ({diff:.3f})"
        else: return 'gray', f"Neutral ({diff:.3f})"
    elif metric == 'NUPL':
        if value > 0.5: return 'red', f"Market top-ish ({value:.2f})"
        elif value < 0.25: return 'green', f"Market bottom-ish ({value:.2f})"
        else: return 'gray', f"Neutral ({value:.2f})"
    elif metric == 'Supply_in_Profit_pct':
        if value > 80: return 'red', f"High supply in profit ({value:.1f}%)"
        elif value < 60: return 'green', f"Low supply in profit ({value:.1f}%)"
        else: return 'gray', f"Neutral ({value:.1f}%)"
    elif metric == 'Miner_Netflow':
        if value > 0: return 'red', f"Miner selling pressure ({value:.0f})"
        elif value < 0: return 'green', f"Miner accumulation ({value:.0f})"
        else: return 'gray', f"Neutral ({value:.0f})"
    elif metric == 'Exchange_Netflow':
        if value > 0: return 'red', f"Exchange inflow selling ({value:.0f})"
        elif value < 0: return 'green', f"Exchange outflow buying ({value:.0f})"
        else: return 'gray', f"Neutral ({value:.0f})"
    elif metric == 'Funding_Rate':
        if value > 0.002: return 'red', f"Long pressure ({value:.4f})"
        elif value < -0.002: return 'green', f"Short pressure ({value:.4f})"
        else: return 'gray', f"Neutral ({value:.4f})"
    else:
        return 'gray', f"{metric}: {value}"

# ---------------------- UI ----------------------
st.sidebar.header("Settings")
coin = st.sidebar.selectbox("Select coin", ['bitcoin', 'ethereum'])
days = st.sidebar.selectbox("Timeframe (days)", [30, 60, 90, 180])

# Fetch data
price_df = fetch_price_data(coin, days=days)
chain_df = fetch_blockchain_metrics()
df = price_df.join(chain_df, how='left')
df = compute_indicators(df)
latest = df.iloc[-1]

# ---------------------- Price Chart ----------------------
st.subheader(f"{coin.capitalize()} Price Chart")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['price'], mode='lines', name='Price'))
fig.update_layout(height=500, margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig, use_container_width=True)

# ---------------------- Red/Green/Gray Boxes ----------------------
st.subheader("Latest Signals & Metrics")
metrics_list = ['EMA_cross','RSI','MACD','NUPL','Supply_in_Profit_pct','Miner_Netflow','Exchange_Netflow','Funding_Rate']
cols = st.columns(len(metrics_list))
for i, m in enumerate(metrics_list):
    val = latest[m.replace('_cross','')] if '_cross' in m else latest[m]
    color, explanation = get_signal_color(val, m)
    cols[i].markdown(f"<div style='background-color:{color};padding:10px;border-radius:5px;text-align:center'>"
                     f"<b>{m}</b><br>{explanation}</div>", unsafe_allow_html=True)

# ---------------------- Forecast ----------------------
st.subheader("30-Day Simple Forecast (Scenario)")
last_price = latest['price']
forecast_days = 30
proj = pd.DataFrame({'date':[datetime.now()+timedelta(days=i) for i in range(1,forecast_days+1)]})
proj['price'] = last_price * (1 + np.cumsum(np.random.normal(0, 0.01, size=forecast_days)))
st.line_chart(proj.set_index('date')['price'])
