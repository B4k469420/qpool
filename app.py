import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import time
import ccxt
import numpy as np
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import base64
import random

# Configuration
GITHUB_RAW_URL = "http://66.179.92.83/data/qpool_V1.csv"
REFRESH_INTERVAL = 1  # seconds

# Encode the cat image to base64
cat_image_path = "data/matilda.jpg"
with open(cat_image_path, "rb") as img_file:
    encoded_cat = base64.b64encode(img_file.read()).decode("utf-8")

# Random initial position
top = random.randint(10, 80)
left = random.randint(10, 80)

# Random animation movement
move_x = random.randint(-100, 100)
move_y = random.randint(-100, 100)
duration = random.randint(6, 12)

# Generate random position and animation duration
top = random.randint(10, 80)
left = random.randint(10, 80)
duration = random.randint(5, 15)

@st.cache_data(ttl=300)
def fetch_current_qubic_price():
    try:
        mexc = ccxt.mexc()
        ticker = mexc.fetch_ticker("QUBIC/USDT")
        return float(ticker["last"])
    except Exception as e:
        st.warning(f"Failed to fetch current QUBIC price: {e}")
        return None


# Setup page
st.set_page_config(
    page_title="Qubic Monero Pool Dashboard",
    page_icon="⛏️",
    layout="wide"
)



# Custom CSS
st.markdown("""
<style>
    
    .element-container:has(.js-plotly-plot) {
        padding-bottom: 10 !important;
        margin-bottom: -10px;
    }

    body, .main, .block-container {
        background-color: #1a252f !important;
        color: white !important;
    }

    .stButton > button {
        background-color: #26303A;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #3ba8d0;
        border-color: #3ba8d0;
    }

      /* Remove blank space at top and bottom */ 
    .block-container {
       padding-top: 3rem;
       padding-bottom: 2rem;
    }
   
   /* Remove blank space at the center canvas */ 
   .st-emotion-cache-z5fcl4 {
       position: relative;
       top: -62px;
       }
   
   /* Make the toolbar transparent and the content below it clickable */ 
   .st-emotion-cache-18ni7ap {
       pointer-events: none;
       background: rgb(255 255 255 / 0%)
       }
   .st-emotion-cache-zq5wmm {
       pointer-events: auto;
       background: rgb(255 255 255);
       border-radius: 5px;
       }

    .metric-card {
    background-color: rgba(255, 255, 255, 0.05);
    padding: 1rem;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    .metric-title {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-bottom: 0.25rem;
    }
    
    .metric-value {
        font-size: 1.3rem;
        font-weight: bold;
        color: white;
    }
    /* Style for streamlit tab labels */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #999;  /* Inactive */
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        color: #fff !important;  /* Active */
        border-bottom: 2px solid #4cc9f0 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=REFRESH_INTERVAL, show_spinner="Loading data...")
def load_data():
    """Load and preprocess CSV data."""
    try:
        df = pd.read_csv(f"{GITHUB_RAW_URL}?t={int(time.time())}")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        if df['timestamp'].isna().any():
            raise ValueError("Invalid timestamp values in CSV")
        df.sort_values('timestamp', inplace=True)
        df['pool_hashrate_mhs'] = df['pool_hashrate'] / 1e6
        df['network_hashrate_ghs'] = df['network_hashrate'] / 1e9
        df['block_found'] = df['pool_blocks_found'].diff().fillna(0) > 0
        # Ensure price columns are numeric
        for col in ['qubic_usdt', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Log data summary for debugging
        # st.write(f"Data loaded: {len(df)} rows, Columns: {list(df.columns)}")
        # st.write(f"NaN in close: {df['close'].isna().sum()}, qubic_usdt: {df['qubic_usdt'].isna().sum()}")
        return df
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return pd.DataFrame()

def format_hashrate(h):
    """Format hashrate values for display."""
    if pd.isna(h):
        return "N/A"
    if h >= 1e9: return f"{h/1e9:.2f} GH/s"
    if h >= 1e6: return f"{h/1e6:.2f} MH/s"
    if h >= 1e3: return f"{h/1e3:.2f} KH/s"
    return f"{h:.2f} H/s"

def format_timespan(delta):
    """Format time delta for display."""
    if pd.isna(delta):
        return "N/A"
    if delta.days > 0: return f"{delta.days}d {delta.seconds//3600}h ago"
    return f"{delta.seconds//3600}h {(delta.seconds%3600)//60}m ago"

def downsample(df, interval='5min'):
    """Downsample DataFrame while preserving key points (ATH, blocks)."""
    if df.empty:
        return df
    ath = df['pool_hashrate'].idxmax()
    blocks = df[df['block_found']].index

    df_resampled = df.resample(interval, on='timestamp').agg({
        'pool_hashrate': 'mean',
        'pool_hashrate_mhs': 'mean',
        'network_hashrate': 'mean',
        'network_hashrate_ghs': 'mean',
        'pool_blocks_found': 'last',
        'block_found': 'any',
        'qubic_usdt': 'last',
        'close': 'last'
    }).reset_index()

    extra_points = pd.concat([df.loc[[ath]]] + [df.loc[[i]] for i in blocks if i not in df_resampled.index])
    df_combined = pd.concat([df_resampled, extra_points]).sort_values('timestamp').drop_duplicates('timestamp')
    
    # Fill forward any missing prices
    df_combined[['qubic_usdt', 'close']] = df_combined[['qubic_usdt', 'close']].ffill()
    
    # Recalculate block_found based on pool_blocks_found diff
    df_combined['block_found'] = df_combined['pool_blocks_found'].diff().fillna(0) > 0

    return df_combined
    
@st.cache_data(ttl=REFRESH_INTERVAL, show_spinner="Loading burn data…")
def load_burn_data():
    """
    Load the tracker-generated CSV and do light normalisation.

    Expected CSV columns:
        timestamp_utc | tickNumber | amount | source | hash |
        qubic_usdt | tx_value_usdt | qubic_epoch
    """
    try:
        df = pd.read_csv("http://66.179.92.83/data/qubic_burns2.csv")

        # --- rename to legacy column names your UI already expects ----------
        df.rename(
            columns={
                "timestamp_utc": "timestamp",
                "amount": "qubic_amount",
                "tx_value_usdt": "usdt_value",
                "hash": "txid",
            },
            inplace=True,
        )

        # --- typing ---------------------------------------------------------
        df["timestamp"]     = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["qubic_amount"]  = pd.to_numeric(df["qubic_amount"], errors="coerce")
        df["usdt_value"]    = pd.to_numeric(df["usdt_value"],   errors="coerce")
        df["qubic_usdt"]    = pd.to_numeric(df["qubic_usdt"],   errors="coerce")
        df["qubic_epoch"]   = pd.to_numeric(df["qubic_epoch"],  errors="coerce").astype("Int64")

        return df.sort_values("timestamp")

    except Exception as e:
        st.error(f"Failed to load burn data: {e}")
        return pd.DataFrame()

# Load data
df = load_data()

st.markdown("""
<div style="text-align: left; margin-bottom: 2rem;">
<svg width="95" height="26" viewBox="0 0 95 26" fill="none" xmlns="http://www.w3.org/2000/svg" class="cursor-pointer"><path d="M5.25 2H0.75C0.335786 2 0 2.33579 0 2.75V19.25C0 19.6642 0.335786 20 0.75 20H5.25C5.66421 20 6 19.6642 6 19.25V2.75C6 2.33579 5.66421 2 5.25 2Z" fill="white"></path><path d="M13.25 2H8.75C8.33579 2 8 2.33579 8 2.75V25.25C8 25.6642 8.33579 26 8.75 26H13.25C13.6642 26 14 25.6642 14 25.25V2.75C14 2.33579 13.6642 2 13.25 2Z" fill="white"></path><path d="M78.2335 20.5641C77.0029 20.5641 75.8848 20.3041 74.8795 19.7841C73.8915 19.2641 73.1028 18.5101 72.5135 17.5221C71.9415 16.5341 71.6555 15.3467 71.6555 13.9601V13.6221C71.6555 12.2354 71.9415 11.0567 72.5135 10.0861C73.1028 9.09807 73.8915 8.34407 74.8795 7.82407C75.8848 7.28673 77.0029 7.01807 78.2335 7.01807C79.4642 7.01807 80.5128 7.2434 81.3795 7.69407C82.2462 8.14473 82.9395 8.74273 83.4595 9.48807C83.9969 10.2334 84.3435 11.0567 84.4995 11.9581L81.8995 12.5041C81.8129 11.9321 81.6308 11.4121 81.3535 10.9441C81.0762 10.4761 80.6862 10.1034 80.1835 9.82607C79.6809 9.54873 79.0482 9.41007 78.2855 9.41007C77.5402 9.41007 76.8642 9.5834 76.2575 9.93007C75.6682 10.2594 75.2002 10.7447 74.8535 11.3861C74.5068 12.0101 74.3335 12.7727 74.3335 13.6741V13.9081C74.3335 14.8094 74.5068 15.5807 74.8535 16.2221C75.2002 16.8634 75.6682 17.3487 76.2575 17.6781C76.8642 18.0074 77.5402 18.1721 78.2855 18.1721C79.4122 18.1721 80.2702 17.8861 80.8595 17.3141C81.4488 16.7247 81.8215 15.9794 81.9775 15.0781L84.5775 15.6761C84.3695 16.5601 83.9969 17.3747 83.4595 18.1201C82.9395 18.8654 82.2462 19.4634 81.3795 19.9141C80.5128 20.3474 79.4642 20.5641 78.2335 20.5641Z" fill="white"></path><path d="M67.4473 20.2V7.382H70.1252V20.2H67.4473ZM68.7992 5.64C68.2792 5.64 67.8372 5.47533 67.4732 5.146C67.1266 4.79933 66.9532 4.35733 66.9532 3.82C66.9532 3.28267 67.1266 2.84933 67.4732 2.52C67.8372 2.17333 68.2792 2 68.7992 2C69.3366 2 69.7786 2.17333 70.1252 2.52C70.4719 2.84933 70.6452 3.28267 70.6452 3.82C70.6452 4.35733 70.4719 4.79933 70.1252 5.146C69.7786 5.47533 69.3366 5.64 68.7992 5.64Z" fill="white"></path><path d="M60.021 20.564C58.773 20.564 57.811 20.3387 57.135 19.888C56.4763 19.4373 55.9823 18.9347 55.653 18.38H55.237V20.2H52.611V2H55.289V9.124H55.705C55.913 8.77733 56.1903 8.448 56.537 8.136C56.8836 7.80667 57.343 7.538 57.915 7.33C58.487 7.122 59.189 7.018 60.021 7.018C61.0956 7.018 62.0836 7.278 62.985 7.798C63.8863 8.318 64.6056 9.072 65.143 10.06C65.6803 11.048 65.949 12.2267 65.949 13.596V13.986C65.949 15.3727 65.6716 16.56 65.117 17.548C64.5796 18.5187 63.8603 19.264 62.959 19.784C62.075 20.304 61.0956 20.564 60.021 20.564ZM59.241 18.224C60.4023 18.224 61.3556 17.8513 62.101 17.106C62.8636 16.3607 63.245 15.2947 63.245 13.908V13.674C63.245 12.3047 62.8723 11.2473 62.127 10.502C61.3816 9.75667 60.4196 9.384 59.241 9.384C58.097 9.384 57.1436 9.75667 56.381 10.502C55.6356 11.2473 55.263 12.3047 55.263 13.674V13.908C55.263 15.2947 55.6356 16.3607 56.381 17.106C57.1436 17.8513 58.097 18.224 59.241 18.224Z" fill="white"></path><path d="M43.3742 20.4341C42.4035 20.4341 41.5369 20.2174 40.7742 19.7841C40.0115 19.3507 39.4135 18.7354 38.9802 17.9381C38.5469 17.1407 38.3302 16.1874 38.3302 15.0781V7.38208H41.0082V14.8961C41.0082 16.0054 41.2855 16.8287 41.8402 17.3661C42.3949 17.8861 43.1662 18.1461 44.1542 18.1461C45.2462 18.1461 46.1215 17.7821 46.7802 17.0541C47.4562 16.3087 47.7942 15.2427 47.7942 13.8561V7.38208H50.4722V20.2001H47.8462V18.2761H47.4302C47.1875 18.7961 46.7542 19.2901 46.1302 19.7581C45.5062 20.2087 44.5875 20.4341 43.3742 20.4341Z" fill="white"></path><path d="M33.66 25.4001V18.4581H33.244C33.0533 18.8047 32.776 19.1427 32.412 19.4721C32.048 19.7841 31.58 20.0441 31.008 20.2521C30.4533 20.4601 29.76 20.5641 28.928 20.5641C27.8533 20.5641 26.8653 20.3041 25.964 19.7841C25.0627 19.2641 24.3433 18.5187 23.806 17.5481C23.2687 16.5601 23 15.3727 23 13.9861V13.5961C23 12.2094 23.2687 11.0307 23.806 10.0601C24.3607 9.07207 25.0887 8.31807 25.99 7.79807C26.8913 7.27807 27.8707 7.01807 28.928 7.01807C30.176 7.01807 31.1293 7.2434 31.788 7.69407C32.464 8.14473 32.9667 8.65607 33.296 9.22807H33.712V7.38207H36.338V25.4001H33.66ZM29.682 18.2241C30.8607 18.2241 31.8227 17.8514 32.568 17.1061C33.3133 16.3607 33.686 15.2947 33.686 13.9081V13.6741C33.686 12.3047 33.3047 11.2474 32.542 10.5021C31.7967 9.75673 30.8433 9.38407 29.682 9.38407C28.538 9.38407 27.5847 9.75673 26.822 10.5021C26.0767 11.2474 25.704 12.3047 25.704 13.6741V13.9081C25.704 15.2947 26.0767 16.3607 26.822 17.1061C27.5847 17.8514 28.538 18.2241 29.682 18.2241Z" fill="white"></path></svg>
</div>
""", unsafe_allow_html=True)


# Metric Cards (Top Row)
if not df.empty:
    latest = df.iloc[-1]
    six_hr = df[df['timestamp'] >= (df['timestamp'].max() - timedelta(hours=6))]
    day_av = df[df['timestamp'] >= (df['timestamp'].max() - timedelta(hours=24))]
    mean_hash_6h = six_hr['pool_hashrate'].mean() / 1e6 if not six_hr.empty else 0
    mean_hash_24h = day_av['pool_hashrate'].mean() / 1e6 if not day_av.empty else 0
    ath_val = df['pool_hashrate'][:-1].max() if len(df) > 1 else latest['pool_hashrate']
    ath_time = df[df['pool_hashrate'] == ath_val]['timestamp'].iloc[0].strftime('%Y-%m-%d') if not df.empty else "N/A"
    last_block = df[df['block_found']]['timestamp'].iloc[-1] if df['block_found'].any() else None
    time_since_block = format_timespan(latest['timestamp'] - last_block) if last_block else "No block"
    
    # Count blocks in the last 24 hours
    blocks_last_24h = df[df['timestamp'] >= (df['timestamp'].max() - timedelta(hours=24))]
    blocks_24h_count = blocks_last_24h['block_found'].sum()
    # Calculate mean time between blocks (in minutes) for the last 24h
    block_times = blocks_last_24h[blocks_last_24h['block_found']]['timestamp']
    if len(block_times) > 1:
        time_deltas = block_times.diff().dropna()
        mean_block_time_min = time_deltas.mean().total_seconds() / 60
    else:
        mean_block_time_min = None

    # Get max blocks found per epoch
    epoch_blocks = df.groupby('qubic_epoch')['pool_blocks_found'].max()
    
    # Calculate number of blocks per epoch by diff
    blocks_per_epoch = epoch_blocks.diff().fillna(epoch_blocks.iloc[0]).astype(int)
    
    # Get last two epochs
    current_epoch = epoch_blocks.index[-1]
    previous_epoch = epoch_blocks.index[-2] if len(epoch_blocks) > 1 else None

    # Calculate delta in blocks found
    df["blocks_delta"] = df["pool_blocks_found"].diff().fillna(0)
    
    #tab1, tab2, tab3, tab4 = st.tabs(["Pool Stats", "QUBIC/XMR", "Token Burns", "Hall of Fame"])
    tab1, tab2, tab3 = st.tabs(["Pool Stats", "QUBIC/XMR", "Token Burns"])
    with tab1: 
        col1, col2 = st.columns([1,3])
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Pool Hashrate</div>
                <div class="metric-value">{format_hashrate(latest['pool_hashrate'])}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Total Blocks Found</div>
                <div class="metric-value">{int(latest['pool_blocks_found'])}</div>
            </div>
            """, unsafe_allow_html=True)
            
            col1a, col1b = st.columns(2)
            with col1a: 
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Current epoch ({current_epoch})</div>
                    <div class="metric-value">{blocks_per_epoch.loc[current_epoch]}</div>
                </div>
                """, unsafe_allow_html=True)
            with col1b: 
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Previous epoch ({previous_epoch}) </div>
                    <div class="metric-value">{blocks_per_epoch.loc[previous_epoch]}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Avg Block Interval (24h)</div>
                <div class="metric-value">
                    {f"{mean_block_time_min:.1f} min" if mean_block_time_min else "N/A"}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Pool Hashrate ATH ({ath_time})</div>
                    <div class="metric-value">{format_hashrate(ath_val)}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Mean Hashrate (last 24h)</div>
                    <div class="metric-value">{mean_hash_24h:.2f} MH/s</div>
                </div>
                """, unsafe_allow_html=True)
            with col2b:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Mean Hashrate (last 6h)</div>
                    <div class="metric-value">{mean_hash_6h:.2f} MH/s</div>
                </div>
                """, unsafe_allow_html=True)
            with col2c:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Network Hashrate</div>
                    <div class="metric-value">{format_hashrate(latest['network_hashrate'])}</div>
                </div>
                """, unsafe_allow_html=True)

            # Hashrate Chart
            # Toggle for log scale
            use_log_scale = st.toggle("Use Log Scale", value=False)
            
            if not df.empty:
                df_chart = downsample(df)

                # Sanitaze for log
                df_chart['pool_hashrate_mhs'] = df_chart['pool_hashrate_mhs'].clip(lower=1e-1)
                df_chart['network_hashrate_ghs'] = df_chart['network_hashrate_ghs'].clip(lower=1e-1)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_chart['timestamp'],
                    y=df_chart['pool_hashrate_mhs'],
                    name='Pool Hashrate (MH/s)',
                    line=dict(color='#4cc9f0'),
                    hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Pool: %{y:.2f} MH/s<extra></extra>'
                ))
                fig.add_trace(go.Scatter(
                    x=df_chart['timestamp'],
                    y=df_chart['network_hashrate_ghs'],
                    name='Network Hashrate (GH/s)',
                    line=dict(color='#f72585', dash='dot'),
                    yaxis='y2',
                    hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Network: %{y:.2f} GH/s<extra></extra>'
                ))

                

                blocks = df_chart[df_chart['block_found']]
                fig.add_trace(go.Scatter(
                    x=blocks['timestamp'],
                    y=blocks['pool_hashrate_mhs'],
                    mode='markers',
                    name='Block Found',
                    marker=dict(symbol='star', size=12, color='gold', line=dict(width=1, color='black')),
                    hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Block Found<extra></extra>'
                ))
                
                # Convert to MH/s
                ath_val_mhs = ath_val / 1e6
                
                if not use_log_scale:
                    fig.add_hline(
                        y=ath_val_mhs,
                        line_dash="longdash",
                        line_color="gold",
                        annotation_text=f"ATH: {ath_val_mhs:,.0f} MH/s",
                        annotation_position="top left",
                        annotation_font_color="gold"
                    )

                # Calculate the time range for the last 24 hours
                end_time = df_chart['timestamp'].max()
                start_time = end_time - timedelta(hours=24)
            
                fig.update_layout(
                    xaxis=dict(
                        title='Time',
                        gridcolor='rgba(255,255,255,0.1)',
                        range=[start_time, end_time],
                        rangeslider=dict(visible=True, thickness=0.1),
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1h", step="hour", stepmode="backward"),
                                dict(count=6, label="6h", step="hour", stepmode="backward"),
                                dict(count=12, label="12h", step="hour", stepmode="backward"),
                                dict(count=24, label="24h", step="hour", stepmode="backward"),
                                dict(step="all", label="All")
                            ]),
                            bgcolor='rgba(32, 46, 60, 0.9)',
                            font=dict(color='white'),
                            activecolor='#4cc9f0'
                        ),
                        type='date'
                    ),
                    yaxis=dict(
                        title='Pool Hashrate (MH/s)',
                        gridcolor='rgba(255,255,255,0.1)',
                        type='log' if use_log_scale else 'linear'
                    ),
                    yaxis2=dict(
                        title='Network Hashrate (GH/s)',
                        overlaying='y',
                        side='right',
                        gridcolor='rgba(255,255,255,0.1)',
                        type='log' if use_log_scale else 'linear'
                    ),
                    margin=dict(t=5, b=10, l=10, r=10),
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    showlegend=True,
                    legend=dict(x=0.5, y=1, orientation='h'),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hashrate data available.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            tol1, tol2 = st.columns([1,3])
            with tol1:
            
                with tol1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">QUBIC/USDT</div>
                        <div class="metric-value">${df_chart['qubic_usdt'].iloc[-1]:.9f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">XMR/USDT</div>
                        <div class="metric-value">${df_chart['close'].iloc[-1]:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)

        
                with tol2:
                    # Hashrate Chart
                    # Price Chart with Stacked Subplots
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    if not df_chart.empty:
                        # Create price chart
                        fig_prices = go.Figure()
                        
                        # Add XMR price
                        fig_prices.add_trace(go.Scatter(
                            x=df_chart['timestamp'],
                            y=df_chart['close'],
                            mode='lines',
                            name='XMR Price (USD)',
                            line=dict(color='limegreen', width=2),
                            yaxis='y1'
                        ))
                        
                        # Add QUBIC price (on secondary axis)
                        fig_prices.add_trace(go.Scatter(
                            x=df_chart['timestamp'],
                            y=df_chart['qubic_usdt'],
                            mode='lines',
                            name='QUBIC Price (USD)',
                            line=dict(color='magenta', width=2),
                            yaxis='y2'
                        ))
                        
                        # Calculate the time range for the last 24 hours
                        end_time = df_chart['timestamp'].max()
                        start_time = end_time - timedelta(hours=24)
                        
                        # Layout with dual y-axes, range slider, and range selector
                        fig_prices.update_layout(
                            title='XMR & QUBIC Prices (24h)',
                            yaxis=dict(
                                title='XMR Price (USD)',
                                tickformat='$.2f',
                                side='left',
                                showgrid=False
                            ),
                            yaxis2=dict(
                                title='QUBIC Price (USD)',
                                tickformat='$.9f',
                                overlaying='y',
                                side='right',
                                showgrid=False
                            ),
                            legend=dict(
                                orientation='h',
                                yanchor='bottom',
                                y=1.02,
                                xanchor='right',
                                x=1
                            ),
                            margin=dict(l=40, r=40, t=40, b=40),
                            height=350,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        
                        st.plotly_chart(fig_prices, use_container_width=True)
                    else:
                        st.warning("No price data available to display.")
                    st.markdown('</div>', unsafe_allow_html=True)                    
        with tab3:
        df_burn = load_burn_data()
        current_qubic_price = fetch_current_qubic_price()
        
        if not df_burn.empty:
            total_qubic_burned = df_burn['qubic_amount'].sum()
            total_usdt_burned = df_burn['usdt_value'].sum()
            last_burn = df_burn.iloc[-1]
            
            # Compute current value of all historical burns
            if current_qubic_price:
                df_burn["current_usdt"] = df_burn["qubic_amount"] * current_qubic_price
                total_current_value = df_burn["current_usdt"].sum()
            else:
                df_burn["current_usdt"] = np.nan
                total_current_value = np.nan
    
            st.markdown("### 🔥 Token Burn Summary")
            colb1, colb2, colb3 = st.columns(3)
            with colb1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Total QUBIC Burned</div>
                    <div class="metric-value">{total_qubic_burned:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            with colb2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">USD Equivalent Burned</div>
                    <div class="metric-value">
                        <div style="font-size: 0.9rem;">At Burn: ${total_usdt_burned:,.0f}</div>
                        <div style="font-size: 0.9rem;">Now: ${total_current_value:,.0f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with colb3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Last Burn</div>
                    <div class="metric-value">{last_burn['timestamp'].strftime('%Y-%m-%d')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 📈 Burn History (Last 30 Days)")
    
            h2_days_ago = pd.Timestamp.utcnow() - timedelta(days=120)
            recent_burns = df_burn[df_burn["timestamp"] > h2_days_ago].copy()
    
            if not recent_burns.empty:
                burn_by_epoch = (
                    recent_burns.groupby("qubic_epoch")["qubic_amount"].sum().reset_index()
                )
                burn_by_epoch["qubic_epoch"] = burn_by_epoch["qubic_epoch"].astype(int).astype(str)
    
                fig_burn = go.Figure()
                fig_burn.add_trace(
                    go.Bar(
                        x=burn_by_epoch["qubic_epoch"],
                        y=burn_by_epoch["qubic_amount"],
                        name="QUBIC Burned",
                        marker_color="crimson",
                        hovertemplate="Epoch %{x}<br>%{y:,.0f} QUBIC<extra></extra>",
                    )
                )
                fig_burn.update_layout(
                    barmode="stack",
                    xaxis_title="Epoch",
                    yaxis_title="QUBIC Burned",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    xaxis=dict(type="category"),
                    margin=dict(l=20, r=20, t=30, b=30),
                    height=300,
                )
                st.plotly_chart(fig_burn, use_container_width=True)
            else:
                st.info("No burn transactions found in the last 120 days.")
            
            st.markdown("### 📋 All Burn Transactions")
            df_display_burns = df_burn.copy()
            df_display_burns["TX_URL"] = "https://explorer.qubic.org/network/tx/" + df_display_burns["txid"]
            df_display_burns = df_display_burns.sort_values("timestamp", ascending=False)
    
            st.dataframe(
                df_display_burns,
                column_order=(
                    "timestamp",
                    "qubic_epoch",
                    "TX_URL",
                    "qubic_amount",
                    "usdt_value",
                    "current_usdt",
                ),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn(label="Timestamp", format="YYYY-MM-DD HH:mm:ss"),
                    "qubic_epoch": st.column_config.NumberColumn(label="Epoch", format="%.0f"),
                    "TX_URL": st.column_config.LinkColumn(
                        label="TX ID",
                        display_text=r"https://explorer\.qubic\.org/network/tx/(.+)",
                    ),
                    "qubic_amount": st.column_config.NumberColumn(label="QUBIC (amount)", format="%.0f"),
                    "usdt_value": st.column_config.NumberColumn(label="Value at Burn ($USDT)", format="$%.2f"),
                    "current_usdt": st.column_config.NumberColumn(label="Current Value ($)", format="$%.2f"),
                    "txid": None,
                },
            )
        else:
            st.info("No burn data available.")



bcol1, bcol2 = st.columns(2)
with bcol1:
    # Manual Refresh Button
    if st.button("🔄 Refresh Data", key="refresh"):
        st.cache_data.clear()
        st.rerun()

with bcol2:
    # Toggle button
    if st.button("🐱 Release The Beast", key="toggle_beast"):
        st.markdown(f"""
            <style>
            @keyframes floatCat {{
                0% {{ transform: translate(0, 0); }}
                100% {{ transform: translate({move_x}px, {move_y}px); }}
            }}
        
            .floating-cat {{
                position: fixed;
                top: {top}%;
                left: {left}%;
                width: 100px;
                z-index: 9999;
                animation: floatCat {duration}s ease-in-out infinite alternate;
                cursor: pointer;
                border-radius: 50%;
                object-fit: cover;
            }}
        
            .cat-message {{
                display: none;
                position: fixed;
                top: {top + 5}%;
                left: {left + 5}%;
                font-size: 20px;
                background-color: #fff7dc;
                padding: 10px 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px #999;
                z-index: 9999;
            }}
            </style>
        
            <a href="https://matildaonqubic.com/" target="_blank">
            <img src="data:image/png;base64,{encoded_cat}" class="floating-cat"
               title="Hello! I'm Matilda the Satoshi’s Cat. In my idle time, I chase Monero blocks. 🐱" />
            </a>
        """, unsafe_allow_html=True)


# Footer
st.markdown("""
<div style="margin-top: 1.5em; font-size: 0.85em; color: #8b95a1;">
📊 <strong>Data Source:</strong> <a href="https://xmr-stats.qubic.org/" target="_blank">xmr-stats.qubic.org</a><br>
💰 <strong>Price Data:</strong> MEXC (via CCXT)<br>
💌 <strong>Inspired by:</strong> <a href="https://qubic-xmr.vercel.app/" target="_blank">qubic-xmr.vercel.app</a>
</div>
""", unsafe_allow_html=True)
