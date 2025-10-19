# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Crypto Tracker (On-Chain)")

st.title("Crypto Tracker — Price & On-Chain Dashboard (no live API)")

# ---------- Helpers ----------
@st.cache_data(ttl=600)
def parse_price_csv(uploaded_file):
    df = pd.read_csv(uploaded_file, dtype=str)
    # detect date and price columns
    date_col = next((c for c in df.columns if 'date' in c.lower()), df.columns[0])
    price_col = next((c for c in df.columns[::-1] if any(k in c.lower() for k in ['close','price','last','adj'])), df.columns[-1])
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[price_col] = df[price_col].astype(str).str.replace('[$,]','', regex=True)
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df = df.dropna(subset=[date_col, price_col])
    out = df[[date_col, price_col]].rename(columns={date_col: 'date', price_col: 'close'})
    out = out.sort_values('date').reset_index(drop=True)
    out['date'] = pd.to_datetime(out['date']).dt.date
    return out

@st.cache_data(ttl=600)
def parse_chain_csv(uploaded_file):
    df = pd.read_csv(uploaded_file, dtype=str)
    date_col = next((c for c in df.columns if 'date' in c.lower()), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).reset_index(drop=True)
    out_rows = []
    for _, row in df.iterrows():
        r = {}
        r['date'] = pd.to_datetime(row[date_col]).date()
        for c in df.columns:
            if c == date_col: continue
            key = c.strip().replace(' ', '_').replace('-', '_')
            raw = row[c]
            if pd.isna(raw) or raw == '':
                r[key] = np.nan
                continue
            try:
                num = float(str(raw).replace('$','').replace(',',''))
                r[key] = num
            except:
                # non-numeric, drop or set NaN
                r[key] = np.nan
        out_rows.append(r)
    out = pd.DataFrame(out_rows)
    out = out.sort_values('date').reset_index(drop=True)
    return out

def merge_by_date(price_df, chain_df):
    if chain_df is None or chain_df.shape[0] == 0:
        return price_df.copy()
    chain_idx = chain_df.set_index('date')
    merged = price_df.copy()
    merged = merged.join(chain_idx, on='date', how='left', rsuffix='_chain')
    return merged

def compute_ocpi_from_row(r):
    # tolerant keys (possible variants)
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

    # normalization and heuristics
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

    # direction: inflows (netflow>0) are selling pressure -> positive; outflows (netflow<0) reduce pressure (invert)
    netflow_dir = netflow_score if netflow > 0 else -netflow_score
    miner_dir = miner_score if minerflow > 0 else -miner_score

    w = {'netflow':0.30, 'miner':0.25, 'funding':0.15, 'nupl':0.15, 'supply':0.15}
    ocpi_raw = (w['netflow']*netflow_dir + w['miner']*miner_dir + w['funding']*funding_norm +
                w['nupl']*nupl_norm + w['supply']*supply_norm)

    # map ocpi_raw from roughly [-1..1] to [0..1]
    ocpi = (ocpi_raw + 1.0) / 2.0
    ocpi = float(max(0.0, min(1.0, ocpi)))
    return ocpi, {'netflow': netflow, 'minerflow': minerflow, 'funding': funding, 'nupl': nupl, 'supply_pct': supply_pct}

def generate_scenario_series(last_price, scenario_key, ocpi_value, days=30):
    results = []
    price = float(last_price)
    for d in range(1, days+1):
        shock = 0.0
        if scenario_key == 'miner_selling': shock = -0.008  # -0.8% daily
        if scenario_key == 'inflow_surge' and d == 1: shock = -0.12
        if scenario_key == 'inflow_surge' and d > 1 and d <= 10:
            frac = (d-1)/10.0
            shock = -0.12 * (1 - frac)
        if scenario_key == 'funding_unwind' and d == 1: shock = -0.02
        # small ocpi tilt (negative = accumulation bias, positive = selling bias)
        ocpi_tilt = 0.002 * (0.5 - ocpi_value)
        daily_return = shock + ocpi_tilt
        price = price * (1 + daily_return)
        results.append({'date': (datetime.now().date() + timedelta(days=d)).isoformat(), 'price': round(price,2)})
    return pd.DataFrame(results)

# ---------- UI Layout ----------
st.sidebar.header("Data Upload")
price_file = st.sidebar.file_uploader("Upload price CSV (Date, Close)", type=['csv'])
chain_file = st.sidebar.file_uploader("Upload on-chain CSV (Date + metrics)", type=['csv'])

st.sidebar.markdown("**Actions**")
upload_note = st.sidebar.empty()

# Parse data
price_df = None
chain_df = None
if price_file:
    try:
        price_df = parse_price_csv(price_file)
        upload_note.info("Price CSV loaded. Rows: {}".format(len(price_df)))
    except Exception as e:
        st.sidebar.error(f"Failed to parse price CSV: {e}")

if chain_file:
    try:
        chain_df = parse_chain_csv(chain_file)
        upload_note.info("On-chain CSV loaded. Rows: {}".format(len(chain_df)))
    except Exception as e:
        st.sidebar.error(f"Failed to parse on-chain CSV: {e}")

if price_df is None:
    st.info("Please upload a Price CSV (Date, Close). A sample file is included in the repository.")
else:
    # Merge
    merged = merge_by_date(price_df, chain_df) if chain_df is not None else price_df.copy()
    st.markdown("### Price (last {})".format(min(200, len(price_df))))
    fig = px.line(price_df.tail(200), x='date', y='close', title='Price (close) — last 200 days')
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Show merged latest and OCPI
    st.markdown("### Latest merged row & OCPI")
    latest = merged.iloc[-1].to_dict()
    st.write(latest)

    ocpi_val = None
    ocpi_components = None
    if chain_df is not None and len(chain_df) > 0:
        # choose last chain row with numeric values
        last_chain_row = chain_df.dropna(axis=1, how='all').iloc[-1].to_dict()
        ocpi_val, ocpi_components = compute_ocpi_from_row(last_chain_row)
        st.metric("OCPI (0=accumulation, 1=selling pressure)", f"{ocpi_val:.3f}")

        st.write("OCPI components (raw):", ocpi_components)
    else:
        st.info("Upload an on-chain CSV to compute OCPI.")

    # Quick on-chain table if provided
    if chain_df is not None:
        st.markdown("### On-Chain (latest 10 rows)")
        st.dataframe(chain_df.tail(10).reset_index(drop=True))

    # Scenario generator
    st.markdown("### Quick scenarios (downloadable CSVs)")
    col1, col2, col3 = st.columns(3)
    last_price = price_df['close'].iloc[-1]
    with col1:
        if st.button("Miner selling (-0.8%/day)"):
            df_s = generate_scenario_series(last_price, 'miner_selling', ocpi_val if ocpi_val is not None else 0.5, days=30)
            csv = df_s.to_csv(index=False).encode('utf-8')
            st.download_button("Download miner_selling.csv", csv, file_name="miner_selling.csv", mime='text/csv')
            st.write(df_s.head())
    with col2:
        if st.button("Exchange inflow surge (-12% day1)"):
            df_s = generate_scenario_series(last_price, 'inflow_surge', ocpi_val if ocpi_val is not None else 0.5, days=30)
            csv = df_s.to_csv(index=False).encode('utf-8')
            st.download_button("Download inflow_surge.csv", csv, file_name="inflow_surge.csv", mime='text/csv')
            st.write(df_s.head())
    with col3:
        if st.button("Funding unwind (-2% day1)"):
            df_s = generate_scenario_series(last_price, 'funding_unwind', ocpi_val if ocpi_val is not None else 0.5, days=30)
            csv = df_s.to_csv(index=False).encode('utf-8')
            st.download_button("Download funding_unwind.csv", csv, file_name="funding_unwind.csv", mime='text/csv')
            st.write(df_s.head())

    # Export merged CSV
    st.markdown("### Export merged data")
    if st.button("Download merged CSV"):
        csv = merged.to_csv(index=False).encode('utf-8')
        st.download_button("Download merged.csv", csv, file_name="merged.csv", mime='text/csv')

st.markdown("---")
st.caption("Note: OCPI is a heuristic index based on snapshot metrics. Provide historical on-chain time series to build causal models.")
