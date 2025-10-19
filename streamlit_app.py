# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Crypto Tracker (On-Chain)")

st.title("Crypto Tracker — Price & On-Chain Dashboard (robust CSV parsing)")

# ---------- Helpers ----------

@st.cache_data(ttl=600)
def clean_headers(df):
    df.columns = [c.strip().replace(' ', '_').replace('-', '_').replace("'", "").replace("\ufeff","") for c in df.columns]
    return df

@st.cache_data(ttl=600)
def parse_price_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = clean_headers(df)
    # find date column
    date_col = next((c for c in df.columns if 'date' in c.lower()), df.columns[0])
    df.rename(columns={date_col: 'date'}, inplace=True)
    # find price column
    price_col = next((c for c in df.columns[::-1] if any(k in c.lower() for k in ['close','price','last','adj'])), df.columns[-1])
    df.rename(columns={price_col: 'close'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    df['close'] = pd.to_numeric(df['close'].astype(str).str.replace('[$,]','', regex=True), errors='coerce')
    df = df.dropna(subset=['date','close']).sort_values('date').reset_index(drop=True)
    return df

@st.cache_data(ttl=600)
def parse_chain_csv_historical(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = clean_headers(df)
    # find date column
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col is None:
        raise ValueError(f"Historical CSV must have a Date column. Found columns: {df.columns.tolist()}")
    df.rename(columns={date_col: 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    numeric_cols = df.columns.drop('date')
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace('[$,]', '', regex=True), errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)
    return df

def merge_by_date(price_df, chain_df):
    if chain_df is None or chain_df.shape[0] == 0:
        return price_df.copy()
    merged = pd.merge(price_df, chain_df, on='date', how='left')
    return merged

def compute_ocpi_from_row(r):
    def get(r, keys, default=0.0):
        for k in keys:
            if k in r and pd.notna(r[k]):
                return r[k]
        return default

    netflow = get(r, ['Exchange_Netflow_Total','ExchangeNetflowTotal','Exchange_Netflow','ExchangeNetflow'], 0.0)
    minerflow = get(r, ['Miner_to_Exchange_Flow_Total','MinertoExchangeFlowTotal','MinerToExchangeFlow','miner_to_exchange_flow_total'], 0.0)
    funding = get(r, ['Funding_Rate','FundingRate','Funding'], 0.0)
    nupl = get(r, ['NUPL','n_up_l','Net_Unrealized_Profit_Loss','NetUnrealizedProfitLoss'], 0.0)
    supply_pct = get(r, ['Supply_in_Profit_pct','SupplyinProfitpct','Supply_in_Profit','SupplyInProfit'], 0.0)

    exchange_reserve = get(r, ['Exchange_Reserve','ExchangeReserve','ExchangeReserveUSD'], 1.0)
    try:
        netflow_ratio = abs(netflow) / max(1.0, float(exchange_reserve))
    except:
        netflow_ratio = 0.0
    miner_ratio = abs(minerflow) / max(1.0, float(exchange_reserve) if exchange_reserve else 1.0)
    funding_norm = np.tanh(max(0.0, funding) * 50)
    nupl_norm = np.clip(nupl if isinstance(nupl, (int,float)) else 0.0, 0, 1)
    supply_norm = np.clip(supply_pct/100.0 if isinstance(supply_pct, (int,float)) else 0.0, 0, 1)

    netflow_score = np.tanh(netflow_ratio * 500)
    miner_score = np.tanh(miner_ratio * 500)

    netflow_dir = netflow_score if netflow > 0 else -netflow_score
    miner_dir = miner_score if minerflow > 0 else -miner_score

    w = {'netflow':0.30, 'miner':0.25, 'funding':0.15, 'nupl':0.15, 'supply':0.15}
    ocpi_raw = (w['netflow']*netflow_dir + w['miner']*miner_dir + w['funding']*funding_norm +
                w['nupl']*nupl_norm + w['supply']*supply_norm)

    ocpi = (ocpi_raw + 1.0) / 2.0
    ocpi = float(max(0.0, min(1.0, ocpi)))
    return ocpi, {'netflow': netflow, 'minerflow': minerflow, 'funding': funding, 'nupl': nupl, 'supply_pct': supply_pct}

def generate_scenario_series(last_price, scenario_key, ocpi_value, days=30):
    results = []
    price = float(last_price)
    for d in range(1, days+1):
        shock = 0.0
        if scenario_key == 'miner_selling': shock = -0.008
        if scenario_key == 'inflow_surge' and d == 1: shock = -0.12
        if scenario_key == 'inflow_surge' and 1 < d <= 10: shock = -0.12 * (1 - (d-1)/10.0)
        if scenario_key == 'funding_unwind' and d == 1: shock = -0.02
        ocpi_tilt = 0.002 * (0.5 - ocpi_value)
        daily_return = shock + ocpi_tilt
        price = price * (1 + daily_return)
        results.append({'date': (datetime.now().date() + timedelta(days=d)).isoformat(), 'price': round(price,2)})
    return pd.DataFrame(results)

# ---------- UI Layout ----------
st.sidebar.header("Data Upload")
price_file = st.sidebar.file_uploader("Upload price CSV (Date, Close)", type=['csv'])
chain_file = st.sidebar.file_uploader("Upload historical on-chain CSV (Date + metrics)", type=['csv'])

st.sidebar.markdown("**Actions**")
upload_note = st.sidebar.empty()

price_df = None
chain_df = None
if price_file:
    try:
        price_df = parse_price_csv(price_file)
        upload_note.info(f"Price CSV loaded. Rows: {len(price_df)}")
    except Exception as e:
        st.sidebar.error(f"Failed to parse price CSV: {e}")

if chain_file:
    try:
        chain_df = parse_chain_csv_historical(chain_file)
        upload_note.info(f"Historical on-chain CSV loaded. Rows: {len(chain_df)}")
    except Exception as e:
        st.sidebar.error(f"Failed to parse on-chain CSV: {e}")

if price_df is None:
    st.info("Please upload a Price CSV (Date, Close).")
else:
    merged = merge_by_date(price_df, chain_df) if chain_df is not None else price_df.copy()

    # Compute OCPI per row if chain_df exists
    if chain_df is not None:
        merged['OCPI'], merged['OCPI_components'] = zip(*merged.apply(lambda r: compute_ocpi_from_row(r)[0:2], axis=1))
    else:
        merged['OCPI'] = np.nan

    st.markdown(f"### Price (last {min(200,len(price_df))})")
    fig = px.line(price_df.tail(200), x='date', y='close', title='Price (close) — last 200 days')
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Latest merged row & OCPI")
    st.write(merged.tail(1))

    if chain_df is not None:
        st.markdown("### On-Chain Metrics (latest 10 rows)")
        st.dataframe(merged.tail(10).drop(columns=['close']))

    st.markdown("### Quick scenarios (downloadable CSVs)")
    col1, col2, col3 = st.columns(3)
    last_price = price_df['close'].iloc[-1]
    ocpi_val = merged['OCPI'].iloc[-1] if 'OCPI' in merged.columns else 0.5
    with col1:
        if st.button("Miner selling (-0.8%/day)"):
            df_s = generate_scenario_series(last_price, 'miner_selling', ocpi_val, days=30)
            st.download_button("Download miner_selling.csv", df_s.to_csv(index=False).encode('utf-8'), file_name="miner_selling.csv")
            st.write(df_s.head())
    with col2:
        if st.button("Exchange inflow surge (-12% day1)"):
            df_s = generate_scenario_series(last_price, 'inflow_surge', ocpi_val, days=30)
            st.download_button("Download inflow_surge.csv", df_s.to_csv(index=False).encode('utf-8'), file_name="inflow_surge.csv")
            st.write(df_s.head())
    with col3:
        if st.button("Funding unwind (-2% day1)"):
            df_s = generate_scenario_series(last_price, 'funding_unwind', ocpi_val, days=30)
            st.download_button("Download funding_unwind.csv", df_s.to_csv(index=False).encode('utf-8'), file_name="funding_unwind.csv")
            st.write(df_s.head())

    st.markdown("### Export merged data")
    if st.button("Download merged CSV"):
        st.download_button("Download merged.csv", merged.to_csv(index=False).encode('utf-8'), file_name="merged.csv")

st.markdown("---")
st.caption("Note: OCPI is a heuristic index based on snapshot metrics. Historical on-chain CSVs enable proper time series tracking.")
