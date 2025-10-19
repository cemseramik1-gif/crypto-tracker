# streamlit_live_crypto_dashboard_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Live Crypto Dashboard")

st.title("Live Crypto & On-Chain Dashboard — Public Data Only")

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
    # Mock blockchain metrics: NUPL & Supply in Profit
    dates = pd.date_range(end=datetime.today(), periods=90)
    df = pd.DataFrame({
        'date': dates,
        'NUPL': np.random.uniform(0.2, 0.8, size=len(dates)),
        'Supply_in_Profit_pct': np.random.uniform(50, 90, size=len(dates))
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

    # Bollinger Bands: dynamic column names
    bbands = ta.bbands(df['price'])
    for col in bbands.columns:
        if 'BBU' in col:
            df['BB_upper'] = bbands[col]
        elif 'BBL' in col:
            df['BB_lower'] = bbands[col]

    df['ATR'] = ta.atr(df['price'], df['price'], df['price'], length=14)
    df['OBV'] = ta.obv(df['price'], df.get('volume', pd.Series(1, index=df.index)))
    return df

# ---------------------- Signal Logic ----------------------
def compute_signal(row):
    score = 0
    weight_total = 0

    def tier(val, bullish, neutral_low, neutral_high, bearish):
        if val >= bullish: return 1
        elif neutral_low <= val <= neutral_high: return 0
        else: return -1

    # EMA cross signals
    ema_cross = row['EMA9'] - row['EMA21']
    signal_ema = tier(ema_cross, 0.5, -0.5, 0.5, -0.5)
    score += 0.3 * signal_ema
    weight_total += 0.3

    # RSI
    signal_rsi = tier(row['RSI'], 60, 40, 60, 40)
    score += 0.1 * signal_rsi
    weight_total += 0.1

    # MACD
    signal_macd = tier(row['MACD'] - row['MACD_signal'], 0.01, -0.01, 0.01, -0.01)
    score += 0.1 * signal_macd
    weight_total += 0.1

    # Blockchain NUPL
    signal_nupl = tier(row.get('NUPL', 0.5), 0.5, 0.25, 0.5, 0.25)
    score += 0.2 * signal_nupl
    weight_total += 0.2

    # Supply in profit
    supply_pct = row.get('Supply_in_Profit_pct', 70)
    signal_supply = tier(supply_pct, 80, 60, 80, 60)
    score += 0.2 * signal_supply
    weight_total += 0.2

    overall = score / weight_total if weight_total else 0
    if overall >= 0.3: return "Bullish"
    elif overall <= -0.3: return "Bearish"
    else: return "Neutral"

# ---------------------- UI ----------------------
st.sidebar.header("Settings")
coin = st.sidebar.selectbox("Select coin", ['bitcoin', 'ethereum'])
days = st.sidebar.selectbox("Timeframe (days)", [30, 60, 90, 180])

# Fetch data
price_df = fetch_price_data(coin, days=days)
chain_df = fetch_blockchain_metrics()

# Merge
df = price_df.join(chain_df, how='left')
df = compute_indicators(df)
df['Signal'] = df.apply(compute_signal, axis=1)

# ---------------------- Charts ----------------------
st.subheader(f"{coin.capitalize()} Price & Indicators")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['price'], mode='lines', name='Price'))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], mode='lines', name='EMA9'))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], mode='lines', name='EMA21'))
fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], line=dict(dash='dash'), name='BB Upper'))
fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], line=dict(dash='dash'), name='BB Lower'))

# Signal markers
bullish = df[df['Signal']=='Bullish']
neutral = df[df['Signal']=='Neutral']
bearish = df[df['Signal']=='Bearish']
fig.add_trace(go.Scatter(x=bullish.index, y=bullish['price'], mode='markers', marker=dict(color='green', size=8), name='Bullish'))
fig.add_trace(go.Scatter(x=neutral.index, y=neutral['price'], mode='markers', marker=dict(color='gray', size=6), name='Neutral'))
fig.add_trace(go.Scatter(x=bearish.index, y=bearish['price'], mode='markers', marker=dict(color='red', size=8), name='Bearish'))

fig.update_layout(height=600, margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig, use_container_width=True)

# Latest signals table
st.subheader("Latest Signals & Metrics")
st.dataframe(df.tail(10)[['price','EMA9','EMA21','RSI','MACD','NUPL','Supply_in_Profit_pct','Signal']])

# Simple 30-day forecast
st.subheader("Scenario Forecast (Next 30 Days)")
last_price = df['price'].iloc[-1]
forecast_days = 30
proj = pd.DataFrame({'date':[datetime.now()+timedelta(days=i) for i in range(1,forecast_days+1)]})
proj['price'] = last_price * (1 + np.cumsum(np.random.normal(0, 0.01, size=forecast_days)))
st.line_chart(proj.set_index('date')['price'])
