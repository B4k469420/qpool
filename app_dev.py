import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import base64
import random
import psycopg2
import os

# --- Configuration ---
REFRESH_INTERVAL = 3  # seconds

# --- Database Connection and Data Loading ---
@st.cache_data(ttl=REFRESH_INTERVAL, show_spinner="Fetching latest data from database...")
def load_data_from_postgres():
    """
    Connects to the PostgreSQL database using st.secrets,
    fetches the data, and returns a preprocessed pandas DataFrame.
    """
    try:
        # Establish a connection using credentials from st.secrets
        conn = psycopg2.connect(**st.secrets["database"])
        
        # Query uses the correct table (qubic_stats) and column (xmr_usdt)
        query = """
        SELECT 
            timestamp, 
            pool_hashrate, 
            network_hashrate, 
            pool_blocks_found,
            xmr_usdt, 
            qubic_usdt, 
            qubic_epoch
        FROM qubic_stats 
        ORDER BY timestamp DESC 
        LIMIT 2500;
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # --- Data Preprocessing ---
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.sort_values('timestamp', inplace=True)
        df['pool_hashrate_mhs'] = df['pool_hashrate'] / 1e6
        df['network_hashrate_ghs'] = df['network_hashrate'] / 1e9
        df['block_found'] = df['pool_blocks_found'].diff().fillna(0) > 0
        
        # Ensure price columns are numeric
        for col in ['qubic_usdt', 'xmr_usdt']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    except psycopg2.OperationalError as e:
        st.error(f"DB Connection Error: Could not connect to the database. Check credentials and server status. Details: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=REFRESH_INTERVAL, show_spinner="Loading burn data...")
def load_burn_data():
    """Loads token burn data from its separate CSV source."""
    try:
        # This URL may need to be updated if the source changes
        df = pd.read_csv("http://66.179.92.83/data/qubic_burns.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['qubic_amount'] = pd.to_numeric(df['qubic_amount'], errors='coerce')
        df['usdt_value'] = pd.to_numeric(df['usdt_value'], errors='coerce')
        return df.sort_values('timestamp')
    except Exception as e:
        st.error(f"Failed to load burn data: {str(e)}")
        return pd.DataFrame()

# --- Helper Functions ---
def format_hashrate(h):
    """Formats hashrate values for display."""
    if pd.isna(h): return "N/A"
    if h >= 1e9: return f"{h/1e9:.2f} GH/s"
    if h >= 1e6: return f"{h/1e6:.2f} MH/s"
    if h >= 1e3: return f"{h/1e3:.2f} KH/s"
    return f"{h:.2f} H/s"

def format_timespan(delta):
    """Formats a timedelta for display."""
    if pd.isna(delta) or not isinstance(delta, timedelta): return "N/A"
    days = delta.days
    hours, rem = divmod(delta.seconds, 3600)
    minutes, _ = divmod(rem, 60)
    if days > 0: return f"{days}d {hours}h ago"
    return f"{hours}h {minutes}m ago"

def downsample(df, interval='5T'):
    """Downsamples DataFrame while preserving key points like ATH and block finds."""
    if df.empty: return df

    ath_idx = df['pool_hashrate'].idxmax()
    original_block_indices = df[df['block_found']].index

    df_resampled = df.resample(interval, on='timestamp').agg({
        'pool_hashrate': 'mean',
        'pool_hashrate_mhs': 'mean',
        'network_hashrate': 'mean',
        'network_hashrate_ghs': 'mean',
        'pool_blocks_found': 'last',
        'block_found': 'any',
        'qubic_usdt': 'last',
        'xmr_usdt': 'last'
    }).reset_index()

    extra_points_list = []
    if not df.empty:
        ath_df = df.loc[[ath_idx]].copy()
        extra_points_list.append(ath_df)
        actual_block_df = df.loc[original_block_indices].copy()
        if not actual_block_df.empty:
            actual_block_df = actual_block_df.drop(ath_idx, errors='ignore')
            extra_points_list.append(actual_block_df)
    
    df_combined = pd.concat([df_resampled] + extra_points_list) if extra_points_list else df_resampled.copy()
    
    df_combined.sort_values(by=['timestamp', 'block_found'], ascending=[True, False], inplace=True)
    df_combined.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
    df_combined.sort_values(by=['timestamp'], inplace=True)
    df_combined[['qubic_usdt', 'xmr_usdt']] = df_combined[['qubic_usdt', 'xmr_usdt']].ffill()
    
    return df_combined

# --- Page Setup and Styling ---
st.set_page_config(page_title="Qubic Monero Pool Dashboard", page_icon="⛏️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    body, .main, .block-container {
        background-color: #1a252f !important;
        color: white !important;
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
    .stTabs [data-baseweb="tab"] { font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Beast Mode (Floating Cat)
cat_image_path = "data/matilda.jpg"
encoded_cat = ""
if os.path.exists(cat_image_path):
    with open(cat_image_path, "rb") as img_file:
        encoded_cat = base64.b64encode(img_file.read()).decode("utf-8")

if "beast_visible" not in st.session_state:
    st.session_state.beast_visible = True

if st.session_state.beast_visible and encoded_cat:
    top, left = random.randint(10, 80), random.randint(10, 80)
    move_x, move_y = random.randint(-100, 100), random.randint(-100, 100)
    duration = random.randint(6, 12)
    st.markdown(f"""
    <style>
        @keyframes floatCat {{
            0% {{ transform: translate(0, 0); }}
            100% {{ transform: translate({move_x}px, {move_y}px); }}
        }}
        .floating-cat {{
            position: fixed; top: {top}%; left: {left}%; width: 100px; z-index: 9999;
            animation: floatCat {duration}s ease-in-out infinite alternate;
            cursor: pointer; border-radius: 50%; object-fit: cover;
        }}
    </style>
    <a href="https://matildaonqubic.com/" target="_blank">
        <img src="data:image/png;base64,{encoded_cat}" class="floating-cat"
             title="Hello! I'm Matilda the Satoshi’s Cat. In my idle time, I chase Monero blocks. 🐱" />
    </a>
    """, unsafe_allow_html=True)

# --- Main App ---
# Header
st.markdown("""
<div style="text-align: left; margin-bottom: 2rem;">
<svg width="95" height="26" viewBox="0 0 95 26" fill="none" xmlns="http://www.w3.org/2000/svg" class="cursor-pointer"><path d="M5.25 2H0.75C0.335786 2 0 2.33579 0 2.75V19.25C0 19.6642 0.335786 20 0.75 20H5.25C5.66421 20 6 19.6642 6 19.25V2.75C6 2.33579 5.66421 2 5.25 2Z" fill="white"></path><path d="M13.25 2H8.75C8.33579 2 8 2.33579 8 2.75V25.25C8 25.6642 8.33579 26 8.75 26H13.25C13.6642 26 14 25.6642 14 25.25V2.75C14 2.33579 13.6642 2 13.25 2Z" fill="white"></path><path d="M78.2335 20.5641C77.0029 20.5641 75.8848 20.3041 74.8795 19.7841C73.8915 19.2641 73.1028 18.5101 72.5135 17.5221C71.9415 16.5341 71.6555 15.3467 71.6555 13.9601V13.6221C71.6555 12.2354 71.9415 11.0567 72.5135 10.0861C73.1028 9.09807 73.8915 8.34407 74.8795 7.82407C75.8848 7.28673 77.0029 7.01807 78.2335 7.01807C79.4642 7.01807 80.5128 7.2434 81.3795 7.69407C82.2462 8.14473 82.9395 8.74273 83.4595 9.48807C83.9969 10.2334 84.3435 11.0567 84.4995 11.9581L81.8995 12.5041C81.8129 11.9321 81.6308 11.4121 81.3535 10.9441C81.0762 10.4761 80.6862 10.1034 80.1835 9.82607C79.6809 9.54873 79.0482 9.41007 78.2855 9.41007C77.5402 9.41007 76.8642 9.5834 76.2575 9.93007C75.6682 10.2594 75.2002 10.7447 74.8535 11.3861C74.5068 12.0101 74.3335 12.7727 74.3335 13.6741V13.9081C74.3335 14.8094 74.5068 15.5807 74.8535 16.2221C75.2002 16.8634 75.6682 17.3487 76.2575 17.6781C76.8642 18.0074 77.5402 18.1721 78.2855 18.1721C79.4122 18.1721 80.2702 17.8861 80.8595 17.3141C81.4488 16.7247 81.8215 15.9794 81.9775 15.0781L84.5775 15.6761C84.3695 16.5601 83.9969 17.3747 83.4595 18.1201C82.9395 18.8654 82.2462 19.4634 81.3795 19.9141C80.5128 20.3474 79.4642 20.5641 78.2335 20.5641Z" fill="white"></path><path d="M67.4473 20.2V7.382H70.1252V20.2H67.4473ZM68.7992 5.64C68.2792 5.64 67.8372 5.47533 67.4732 5.146C67.1266 4.79933 66.9532 4.35733 66.9532 3.82C66.9532 3.28267 67.1266 2.84933 67.4732 2.52C67.8372 2.17333 68.2792 2 68.7992 2C69.3366 2 69.7786 2.17333 70.1252 2.52C70.4719 2.84933 70.6452 3.28267 70.6452 3.82C70.6452 4.35733 70.4719 4.79933 70.1252 5.146C69.7786 5.47533 69.3366 5.64 68.7992 5.64Z" fill="white"></path><path d="M60.021 20.564C58.773 20.564 57.811 20.3387 57.135 19.888C56.4763 19.4373 55.9823 18.9347 55.653 18.38H55.237V20.2H52.611V2H55.289V9.124H55.705C55.913 8.77733 56.1903 8.448 56.537 8.136C56.8836 7.80667 57.343 7.538 57.915 7.33C58.487 7.122 59.189 7.018 60.021 7.018C61.0956 7.018 62.0836 7.278 62.985 7.798C63.8863 8.318 64.6056 9.072 65.143 10.06C65.6803 11.048 65.949 12.2267 65.949 13.596V13.986C65.949 15.3727 65.6716 16.56 65.117 17.548C64.5796 18.5187 63.8603 19.264 62.959 19.784C62.075 20.304 61.0956 20.564 60.021 20.564ZM59.241 18.224C60.4023 18.224 61.3556 17.8513 62.101 17.106C62.8636 16.3607 63.245 15.2947 63.245 13.908V13.674C63.245 12.3047 62.8723 11.2473 62.127 10.502C61.3816 9.75667 60.4196 9.384 59.241 9.384C58.097 9.384 57.1436 9.75667 56.381 10.502C55.6356 11.2473 55.263 12.3047 55.263 13.674V13.908C55.263 15.2947 55.6356 16.3607 56.381 17.106C57.1436 17.8513 58.097 18.224 59.241 18.224Z" fill="white"></path><path d="M43.3742 20.4341C42.4035 20.4341 41.5369 20.2174 40.7742 19.7841C40.0115 19.3507 39.4135 18.7354 38.9802 17.9381C38.5469 17.1407 38.3302 16.1874 38.3302 15.0781V7.38208H41.0082V14.8961C41.0082 16.0054 41.2855 16.8287 41.8402 17.3661C42.3949 17.8861 43.1662 18.1461 44.1542 18.1461C45.2462 18.1461 46.1215 17.7821 46.7802 17.0541C47.4562 16.3087 47.7942 15.2427 47.7942 13.8561V7.38208H50.4722V20.2001H47.8462V18.2761H47.4302C47.1875 18.7961 46.7542 19.2901 46.1302 19.7581C45.5062 20.2087 44.5875 20.4341 43.3742 20.4341Z" fill="white"></path><path d="M33.66 25.4001V18.4581H33.244C33.0533 18.8047 32.776 19.1427 32.412 19.4721C32.048 19.7841 31.58 20.0441 31.008 20.2521C30.4533 20.4601 29.76 20.5641 28.928 20.5641C27.8533 20.5641 26.8653 20.3041 25.964 19.7841C25.0627 19.2641 24.3433 18.5187 23.806 17.5481C23.2687 16.5601 23 15.3727 23 13.9861V13.5961C23 12.2094 23.2687 11.0307 23.806 10.0601C24.3607 9.07207 25.0887 8.31807 25.99 7.79807C26.8913 7.27807 27.8707 7.01807 28.928 7.01807C30.176 7.01807 31.1293 7.2434 31.788 7.69407C32.464 8.14473 32.9667 8.65607 33.296 9.22807H33.712V7.38207H36.338V25.4001H33.66ZM29.682 18.2241C30.8607 18.2241 31.8227 17.8514 32.568 17.1061C33.3133 16.3607 33.686 15.2947 33.686 13.9081V13.6741C33.686 12.3047 33.3047 11.2474 32.542 10.5021C31.7967 9.75673 30.8433 9.38407 29.682 9.38407C28.538 9.38407 27.5847 9.75673 26.822 10.5021C26.0767 11.2474 25.704 12.3047 25.704 13.6741V13.9081C25.704 15.2947 26.0767 16.3607 26.822 17.1061C27.5847 17.8514 28.538 18.2241 29.682 18.2241Z" fill="white"></path></svg>
</div>
""", unsafe_allow_html=True)

# Load data
df = load_data_from_postgres()

# Main dashboard UI
if not df.empty:
    df_chart = downsample(df)
    latest = df.iloc[-1]
    
    # Calculate metrics
    six_hr = df[df['timestamp'] >= (df['timestamp'].max() - timedelta(hours=6))]
    day_av = df[df['timestamp'] >= (df['timestamp'].max() - timedelta(hours=24))]
    mean_hash_6h = six_hr['pool_hashrate_mhs'].mean()
    mean_hash_24h = day_av['pool_hashrate_mhs'].mean()
    ath_val = df['pool_hashrate'].max()
    ath_time = df.loc[df['pool_hashrate'].idxmax()]['timestamp'].strftime('%Y-%m-%d')
    last_block_time = df[df['block_found']]['timestamp'].iloc[-1] if df['block_found'].any() else None
    time_since_block = format_timespan(datetime.now(last_block_time.tz) - last_block_time) if last_block_time else "No block"
    
    # Calculate blocks per epoch
    epoch_blocks = df.groupby('qubic_epoch')['pool_blocks_found'].max()
    blocks_per_epoch = epoch_blocks.diff().fillna(epoch_blocks.iloc[0]).astype(int)
    current_epoch = epoch_blocks.index[-1]
    previous_epoch = epoch_blocks.index[-2] if len(epoch_blocks) > 1 else current_epoch
    
    # Calculate mean time between blocks (24h)
    blocks_24h = day_av[day_av['block_found']]
    mean_block_time_min = blocks_24h['timestamp'].diff().mean().total_seconds() / 60 if len(blocks_24h) > 1 else None
    
    tab1, tab2, tab3 = st.tabs(["Pool Stats", "QUBIC/XMR", "Token Burns"])

    with tab1:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"""
            <div class="metric-card"><div class="metric-title">Pool Hashrate</div><div class="metric-value">{format_hashrate(latest['pool_hashrate'])}</div></div>
            <div class="metric-card"><div class="metric-title">Total Blocks Found</div><div class="metric-value">{int(latest['pool_blocks_found'])}</div></div>
            """, unsafe_allow_html=True)
            col1a, col1b = st.columns(2)
            with col1a:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Current epoch ({current_epoch})</div><div class="metric-value">{blocks_per_epoch.get(current_epoch, 0)}</div></div>', unsafe_allow_html=True)
            with col1b:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Previous epoch ({previous_epoch})</div><div class="metric-value">{blocks_per_epoch.get(previous_epoch, 0)}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="metric-title">Avg Block Interval (24h)</div><div class="metric-value">{f"{mean_block_time_min:.1f} min" if mean_block_time_min else "N/A"}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="metric-title">Pool Hashrate ATH ({ath_time})</div><div class="metric-value">{format_hashrate(ath_val)}</div></div>', unsafe_allow_html=True)

        with col2:
            c2a, c2b, c2c = st.columns(3)
            c2a.markdown(f'<div class="metric-card"><div class="metric-title">Mean Hashrate (24h)</div><div class="metric-value">{mean_hash_24h:.2f} MH/s</div></div>', unsafe_allow_html=True)
            c2b.markdown(f'<div class="metric-card"><div class="metric-title">Mean Hashrate (6h)</div><div class="metric-value">{mean_hash_6h:.2f} MH/s</div></div>', unsafe_allow_html=True)
            c2c.markdown(f'<div class="metric-card"><div class="metric-title">Network Hashrate</div><div class="metric-value">{format_hashrate(latest["network_hashrate"])}</div></div>', unsafe_allow_html=True)

            use_log_scale = st.toggle("Use Log Scale for Hashrate", value=True)
            fig_hash = make_subplots(specs=[[{"secondary_y": True}]])
            fig_hash.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['pool_hashrate_mhs'], name='Pool (MH/s)', line=dict(color='#4cc9f0')), secondary_y=False)
            fig_hash.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['network_hashrate_ghs'], name='Network (GH/s)', line=dict(color='#f72585', dash='dot')), secondary_y=True)
            fig_hash.add_trace(go.Scatter(x=df_chart[df_chart['block_found']]['timestamp'], y=df_chart[df_chart['block_found']]['pool_hashrate_mhs'], mode='markers', name='Block Found', marker=dict(symbol='star', size=12, color='gold')), secondary_y=False)
            fig_hash.update_layout(
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)', rangeslider=dict(visible=True, thickness=0.1)),
                yaxis=dict(title='Pool Hashrate (MH/s)', gridcolor='rgba(255,255,255,0.1)', type='log' if use_log_scale else 'linear'),
                yaxis2=dict(title='Network Hashrate (GH/s)', overlaying='y', side='right', showgrid=False, type='log' if use_log_scale else 'linear'),
                height=330, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), hovermode='x unified', margin=dict(t=5, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_hash, use_container_width=True)
            
    with tab2:
        colp1, colp2 = st.columns([1,3])
        with colp1:
             st.markdown(f"""
            <div class="metric-card"><div class="metric-title">QUBIC/USDT</div><div class="metric-value">${latest['qubic_usdt']:.7f}</div></div>
            <div class="metric-card"><div class="metric-title">XMR/USDT</div><div class="metric-value">${latest['xmr_usdt']:.2f}</div></div>
            """, unsafe_allow_html=True)
        with colp2:
            fig_prices = make_subplots(specs=[[{"secondary_y": True}]])
            fig_prices.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['xmr_usdt'], name='XMR ($)', line=dict(color='limegreen')), secondary_y=False)
            fig_prices.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['qubic_usdt'], name='QUBIC ($)', line=dict(color='magenta')), secondary_y=True)
            fig_prices.update_layout(
                yaxis=dict(title='XMR Price ($)', showgrid=False),
                yaxis2=dict(title='QUBIC Price ($)', overlaying='y', side='right', showgrid=False),
                height=450, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), hovermode='x unified', margin=dict(t=5, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_prices, use_container_width=True)

    with tab3:
        df_burn = load_burn_data()
        if not df_burn.empty:
            total_qubic_burned = df_burn['qubic_amount'].sum()
            total_usdt_burned = df_burn['usdt_value'].sum()
            
            bc1, bc2, bc3 = st.columns(3)
            bc1.markdown(f'<div class="metric-card"><div class="metric-title">Total QUBIC Burned</div><div class="metric-value">{total_qubic_burned:,.0f}</div></div>', unsafe_allow_html=True)
            bc2.markdown(f'<div class="metric-card"><div class="metric-title">USD Equivalent Burned</div><div class="metric-value">${total_usdt_burned:,.0f}</div></div>', unsafe_allow_html=True)
            bc3.markdown(f'<div class="metric-card"><div class="metric-title">Last Burn</div><div class="metric-value">{df_burn["timestamp"].iloc[-1].strftime("%Y-%m-%d")}</div></div>', unsafe_allow_html=True)
            
            st.markdown("### 📈 Burn History (by Day)")
            burn_by_day = df_burn.set_index('timestamp').resample('D')['qubic_amount'].sum().reset_index()
            fig_burn = go.Figure(data=[go.Bar(x=burn_by_day['timestamp'], y=burn_by_day['qubic_amount'], marker_color='crimson')])
            fig_burn.update_layout(
                xaxis_title="Date", yaxis_title="QUBIC Burned", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='white', margin=dict(l=20, r=20, t=30, b=30), height=300
            )
            st.plotly_chart(fig_burn, use_container_width=True)

            st.markdown("### 📋 Recent Burn Transactions")
            df_display_burns = df_burn.copy()
            df_display_burns['Current Value ($)'] = df_display_burns['qubic_amount'] * latest['qubic_usdt']
            df_display_burns['TX_URL'] = "https://explorer.qubic.org/network/tx/" + df_display_burns['tx']
            st.dataframe(
                df_display_burns.sort_values('timestamp', ascending=False),
                column_order=("timestamp", "TX_URL", "qubic_amount", "usdt_value", "Current Value ($)"),
                use_container_width=True, hide_index=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
                    "TX_URL": st.column_config.LinkColumn("TX ID", display_text=r"https://explorer\.qubic\.org/network/tx/(.+)"),
                    "qubic_amount": st.column_config.NumberColumn("QUBIC Burned", format="%.0f"),
                    "usdt_value": st.column_config.NumberColumn("Value at Burn ($)", format="$%.2f"),
                    "Current Value ($)": st.column_config.NumberColumn("Current Value ($)", format="$%.2f"),
                }
            )
        else:
            st.warning("No token burn data available.")

else:
    st.warning("Data is currently being loaded. If this message persists, please check the database connection.")

# --- Footer and Controls ---
st.markdown("---")
bcol1, bcol2 = st.columns(2)
if bcol1.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

if bcol2.button("🐱 Hide The Beast" if st.session_state.beast_visible else "🐱 Release The Beast", use_container_width=True):
    st.session_state.beast_visible = not st.session_state.beast_visible
    st.rerun()

st.markdown("""
<div style="text-align: center; margin-top: 2em; font-size: 0.85em; color: #8b95a1;">
    📊 Data Source: <a href="https://xmr-stats.qubic.org/" target="_blank">xmr-stats.qubic.org</a> | 
    💰 Price Data: MEXC | 
     inspirado por <a href="https://qubic-xmr.vercel.app/" target="_blank">qubic-xmr.vercel.app</a>
</div>
""", unsafe_allow_html=True)
