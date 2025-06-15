import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import base64
import random
import psycopg2
import os

# --- Page Configuration ---
st.set_page_config(layout="wide")

# --- Constants ---
REFRESH_INTERVAL = 30  # seconds

# --- Styling ---
st.markdown("""
<style>
/* Optional: Reduce vertical space of st.metric */
[data-testid="stMetric"] {
    background-color: #222b38;
    border: 1px solid #3a4a60;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
}
[data-testid="stMetricLabel"] {
  font-size: 0.9rem;
  color: #a0b0c0;
}
</style>
""", unsafe_allow_html=True)


# --- Database Connection and Data Loading ---

### NEW ###
@st.cache_data(ttl=600, show_spinner="Fetching overall statistics...")
def get_overall_stats():
    """Fetches key overall metrics (total blocks, ATH) directly from the database."""
    try:
        conn = psycopg2.connect(**st.secrets["database"])
        query = """
        SELECT
            MAX(pool_blocks_found) AS total_blocks,
            MAX(pool_hashrate) AS ath_hashrate
        FROM pool_statistics;
        """
        stats_df = pd.read_sql_query(query, conn)
        conn.close()
        return stats_df.iloc[0]
    except Exception as e:
        st.error(f"Failed to fetch overall stats: {e}")
        return pd.Series({'total_blocks': 0, 'ath_hashrate': 0})

### NEW ###
@st.cache_data(ttl=3600, show_spinner="Fetching epoch history...")
def get_blocks_by_epoch():
    """Fetches the total blocks found at the end of each epoch for historical analysis."""
    try:
        conn = psycopg2.connect(**st.secrets["database"])
        query = """
        SELECT
            qubic_epoch,
            MAX(pool_blocks_found) as blocks_at_epoch_end
        FROM pool_statistics
        GROUP BY qubic_epoch
        ORDER BY qubic_epoch;
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        df['blocks_in_epoch'] = df['blocks_at_epoch_end'].diff().fillna(df['blocks_at_epoch_end'].iloc[0]).astype(int)
        return df[['qubic_epoch', 'blocks_in_epoch']]
    except Exception as e:
        st.error(f"Failed to fetch epoch block data: {e}")
        return pd.DataFrame(columns=['qubic_epoch', 'blocks_in_epoch'])


### UPDATED ###
def preprocess_pool_data(df):
    """Preprocesses the raw pool statistics DataFrame."""
    if df.empty:
        return df
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.sort_values('timestamp', inplace=True)
    df['pool_hashrate_mhs'] = df['pool_hashrate'] / 1e6
    df['network_hashrate_ghs'] = df['network_hashrate'] / 1e9
    df['block_found'] = df['pool_blocks_found'].diff().fillna(0) > 0
    return df

### UPDATED ###
@st.cache_data(ttl=REFRESH_INTERVAL, show_spinner="Fetching latest data...")
def load_recent_data():
    """Connects to PostgreSQL and fetches recent pool statistics."""
    try:
        conn = psycopg2.connect(**st.secrets["database"])
        query = """
        SELECT timestamp, pool_hashrate, network_hashrate, pool_blocks_found, qubic_epoch
        FROM pool_statistics
        ORDER BY timestamp DESC
        LIMIT 50000;
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return preprocess_pool_data(df)
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
    if pd.isna(h) or h == 0: return "N/A"
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
        'pool_hashrate': 'mean', 'pool_hashrate_mhs': 'mean',
        'network_hashrate': 'mean', 'network_hashrate_ghs': 'mean',
        'pool_blocks_found': 'last', 'block_found': 'any'
    }).reset_index()

    extra_points_list = [df.loc[[ath_idx]]]
    if not original_block_indices.empty:
        extra_points_list.append(df.loc[original_block_indices])
    
    df_combined = pd.concat([df_resampled] + extra_points_list, ignore_index=True)
    df_combined.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
    df_combined.sort_values(by='timestamp', inplace=True)
    return df_combined

### NEW ###
def display_floating_cat():
    """Generates and displays the HTML/CSS for the floating cat animation."""
    # Define animation parameters
    top = random.randint(10, 80)
    left = random.randint(10, 80)
    duration = random.randint(10, 20)
    move_x = random.randint(-40, 40)
    move_y = random.randint(-40, 40)
    
    # Placeholder for a base64 encoded cat image
    # In a real app, you would load this from a file
    encoded_cat = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" # 1x1 transparent pixel
    
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


# --- Main Application ---
st.markdown("""
<div style="text-align: left; margin-bottom: 2rem;">
<svg width="95" height="26" viewBox="0 0 95 26" fill="none" xmlns="http://www.w3.org/2000/svg" class="cursor-pointer"><path d="M5.25 2H0.75C0.335786 2 0 2.33579 0 2.75V19.25C0 19.6642 0.335786 20 0.75 20H5.25C5.66421 20 6 19.6642 6 19.25V2.75C6 2.33579 5.66421 2 5.25 2Z" fill="white"></path><path d="M13.25 2H8.75C8.33579 2 8 2.33579 8 2.75V25.25C8 25.6642 8.33579 26 8.75 26H13.25C13.6642 26 14 25.6642 14 25.25V2.75C14 2.33579 13.6642 2 13.25 2Z" fill="white"></path><path d="M78.2335 20.5641C77.0029 20.5641 75.8848 20.3041 74.8795 19.7841C73.8915 19.2641 73.1028 18.5101 72.5135 17.5221C71.9415 16.5341 71.6555 15.3467 71.6555 13.9601V13.6221C71.6555 12.2354 71.9415 11.0567 72.5135 10.0861C73.1028 9.09807 73.8915 8.34407 74.8795 7.82407C75.8848 7.28673 77.0029 7.01807 78.2335 7.01807C79.4642 7.01807 80.5128 7.2434 81.3795 7.69407C82.2462 8.14473 82.9395 8.74273 83.4595 9.48807C83.9969 10.2334 84.3435 11.0567 84.4995 11.9581L81.8995 12.5041C81.8129 11.9321 81.6308 11.4121 81.3535 10.9441C81.0762 10.4761 80.6862 10.1034 80.1835 9.82607C79.6809 9.54873 79.0482 9.41007 78.2855 9.41007C77.5402 9.41007 76.8642 9.5834 76.2575 9.93007C75.6682 10.2594 75.2002 10.7447 74.8535 11.3861C74.5068 12.0101 74.3335 12.7727 74.3335 13.6741V13.9081C74.3335 14.8094 74.5068 15.5807 74.8535 16.2221C75.2002 16.8634 75.6682 17.3487 76.2575 17.6781C76.8642 18.0074 77.5402 18.1721 78.2855 18.1721C79.4122 18.1721 80.2702 17.8861 80.8595 17.3141C81.4488 16.7247 81.8215 15.9794 81.9775 15.0781L84.5775 15.6761C84.3695 16.5601 83.9969 17.3747 83.4595 18.1201C82.9395 18.8654 82.2462 19.4634 81.3795 19.9141C80.5128 20.3474 79.4642 20.5641 78.2335 20.5641Z" fill="white"></path><path d="M67.4473 20.2V7.382H70.1252V20.2H67.4473ZM68.7992 5.64C68.2792 5.64 67.8372 5.47533 67.4732 5.146C67.1266 4.79933 66.9532 4.35733 66.9532 3.82C66.9532 3.28267 67.1266 2.84933 67.4732 2.52C67.8372 2.17333 68.2792 2 68.7992 2C69.3366 2 69.7786 2.17333 70.1252 2.52C70.4719 2.84933 70.6452 3.28267 70.6452 3.82C70.6452 4.35733 70.4719 4.79933 70.1252 5.146C69.7786 5.47533 69.3366 5.64 68.7992 5.64Z" fill="white"></path><path d="M60.021 20.564C58.773 20.564 57.811 20.3387 57.135 19.888C56.4763 19.4373 55.9823 18.9347 55.653 18.38H55.237V20.2H52.611V2H55.289V9.124H55.705C55.913 8.77733 56.1903 8.448 56.537 8.136C56.8836 7.80667 57.343 7.538 57.915 7.33C58.487 7.122 59.189 7.018 60.021 7.018C61.0956 7.018 62.0836 7.278 62.985 7.798C63.8863 8.318 64.6056 9.072 65.143 10.06C65.6803 11.048 65.949 12.2267 65.949 13.596V13.986C65.949 15.3727 65.6716 16.56 65.117 17.548C64.5796 18.5187 63.8603 19.264 62.959 19.784C62.075 20.304 61.0956 20.564 60.021 20.564ZM59.241 18.224C60.4023 18.224 61.3556 17.8513 62.101 17.106C62.8636 16.3607 63.245 15.2947 63.245 13.908V13.674C63.245 12.3047 62.8723 11.2473 62.127 10.502C61.3816 9.75667 60.4196 9.384 59.241 9.384C58.097 9.384 57.1436 9.75667 56.381 10.502C55.6356 11.2473 55.263 12.3047 55.263 13.674V13.908C55.263 15.2947 55.6356 16.3607 56.381 17.106C57.1436 17.8513 58.097 18.224 59.241 18.224Z" fill="white"></path><path d="M43.3742 20.4341C42.4035 20.4341 41.5369 20.2174 40.7742 19.7841C40.0115 19.3507 39.4135 18.7354 38.9802 17.9381C38.5469 17.1407 38.3302 16.1874 38.3302 15.0781V7.38208H41.0082V14.8961C41.0082 16.0054 41.2855 16.8287 41.8402 17.3661C42.3949 17.8861 43.1662 18.1461 44.1542 18.1461C45.2462 18.1461 46.1215 17.7821 46.7802 17.0541C47.4562 16.3087 47.7942 15.2427 47.7942 13.8561V7.38208H50.4722V20.2001H47.8462V18.2761H47.4302C47.1875 18.7961 46.7542 19.2901 46.1302 19.7581C45.5062 20.2087 44.5875 20.4341 43.3742 20.4341Z" fill="white"></path><path d="M33.66 25.4001V18.4581H33.244C33.0533 18.8047 32.776 19.1427 32.412 19.4721C32.048 19.7841 31.58 20.0441 31.008 20.2521C30.4533 20.4601 29.76 20.5641 28.928 20.5641C27.8533 20.5641 26.8653 20.3041 25.964 19.7841C25.0627 19.2641 24.3433 18.5187 23.806 17.5481C23.2687 16.5601 23 15.3727 23 13.9861V13.5961C23 12.2094 23.2687 11.0307 23.806 10.0601C24.3607 9.07207 25.0887 8.31807 25.99 7.79807C26.8913 7.27807 27.8707 7.01807 28.928 7.01807C30.176 7.01807 31.1293 7.2434 31.788 7.69407C32.464 8.14473 32.9667 8.65607 33.296 9.22807H33.712V7.38207H36.338V25.4001H33.66ZM29.682 18.2241C30.8607 18.2241 31.8227 17.8514 32.568 17.1061C33.3133 16.3607 33.686 15.2947 33.686 13.9081V13.6741C33.686 12.3047 33.3047 11.2474 32.542 10.5021C31.7967 9.75673 30.8433 9.38407 29.682 9.38407C28.538 9.38407 27.5847 9.75673 26.822 10.5021C26.0767 11.2474 25.704 12.3047 25.704 13.6741V13.9081C25.704 15.2947 26.0767 16.3607 26.822 17.1061C27.5847 17.8514 28.538 18.2241 29.682 18.2241Z" fill="white"></path></svg>
</div>
""", unsafe_allow_html=True)

# Load all data
df = load_recent_data()
overall_stats = get_overall_stats()

tab1, tab2 = st.tabs(["Pool Stats", "Token Burns"])

with tab1:
    if not df.empty:
        latest = df.iloc[-1]
        
        # --- Metric Calculations ---
        six_hr = df[df['timestamp'] >= (df['timestamp'].max() - timedelta(hours=6))]
        day_av = df[df['timestamp'] >= (df['timestamp'].max() - timedelta(hours=24))]
        
        mean_hash_6h = six_hr['pool_hashrate'].mean()
        mean_hash_24h = day_av['pool_hashrate'].mean()
        
        last_block_time = df[df['block_found']]['timestamp'].iloc[-1] if df['block_found'].any() else None
        time_since_block = format_timespan(latest['timestamp'] - last_block_time) if last_block_time else "No recent blocks"

        block_times = day_av[day_av['block_found']]['timestamp']
        mean_block_time_min = (block_times.diff().mean().total_seconds() / 60) if len(block_times) > 1 else None

        epoch_block_history = get_blocks_by_epoch()
        if not epoch_block_history.empty:
            current_epoch_blocks = epoch_block_history.iloc[-1]
            previous_epoch_blocks = epoch_block_history.iloc[-2] if len(epoch_block_history) > 1 else None
            epoch_delta = (current_epoch_blocks['blocks_in_epoch'] - previous_epoch_blocks['blocks_in_epoch']) if previous_epoch_blocks is not None else None
        else:
            current_epoch_blocks = previous_epoch_blocks = epoch_delta = None

        # --- UI Layout ---
        st.subheader("Live Pool Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pool Hashrate", format_hashrate(latest['pool_hashrate']))
        c2.metric("Network Hashrate", format_hashrate(latest['network_hashrate']))
        c3.metric("Total Blocks Found", f"{int(overall_stats['total_blocks']):,}")
        c4.metric("Last Block Found", time_since_block)
        
        st.divider()
        
        st.subheader("Performance Averages & Epochs")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("All-Time High Hashrate", format_hashrate(overall_stats['ath_hashrate']))
        c2.metric("Mean Hashrate (24h)", format_hashrate(mean_hash_24h))
        c3.metric("Avg. Block Time (24h)", f"{mean_block_time_min:.1f} min" if mean_block_time_min else "N/A")
        if current_epoch_blocks is not None:
             c4.metric(
                f"Current Epoch ({current_epoch_blocks['qubic_epoch']}) Blocks",
                value=f"{current_epoch_blocks['blocks_in_epoch']:,}",
                delta=f"{int(epoch_delta):,}" if epoch_delta is not None else None,
                delta_color="normal"
            )
        
        st.divider()

        # --- Charts ---
        chart_col, epoch_col = st.columns([3, 1])
        with chart_col:
            st.subheader("Pool & Network Hashrate (Last 24 Hours)")
            use_log_scale = st.toggle("Use Log Scale for Hashrate Chart", value=False, key="log_hash")

            df_chart = downsample(df)
            df_chart['pool_hashrate_mhs'] = df_chart['pool_hashrate_mhs'].clip(lower=1e-1)
            df_chart['network_hashrate_ghs'] = df_chart['network_hashrate_ghs'].clip(lower=1e-1)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['pool_hashrate_mhs'], name='Pool (MH/s)', line=dict(color='#4cc9f0'), hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Pool: %{y:.2f} MH/s<extra></extra>'))
            fig.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['network_hashrate_ghs'], name='Network (GH/s)', line=dict(color='#f72585', dash='dot'), yaxis='y2', hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Network: %{y:.2f} GH/s<extra></extra>'))
            blocks = df_chart[df_chart['block_found']]
            fig.add_trace(go.Scatter(x=blocks['timestamp'], y=blocks['pool_hashrate_mhs'], mode='markers', name='Block Found', marker=dict(symbol='star', size=12, color='gold', line=dict(width=1, color='black')), hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Block Found<extra></extra>'))
            
            ath_val_mhs = overall_stats['ath_hashrate'] / 1e6
            if not use_log_scale:
                fig.add_hline(y=ath_val_mhs, line_dash="longdash", line_color="gold", annotation_text=f"ATH: {ath_val_mhs:,.0f} MH/s", annotation_position="top left", annotation_font_color="gold")

            end_time = df_chart['timestamp'].max()
            start_time = end_time - timedelta(hours=24)
            
            fig.update_layout(
                xaxis=dict(title='Time', gridcolor='rgba(255,255,255,0.1)', range=[start_time, end_time], rangeslider=dict(visible=True, thickness=0.05)),
                yaxis=dict(title='Pool Hashrate (MH/s)', gridcolor='rgba(255,255,255,0.1)', type='log' if use_log_scale else 'linear'),
                yaxis2=dict(title='Network Hashrate (GH/s)', overlaying='y', side='right', showgrid=False, type='log' if use_log_scale else 'linear'),
                margin=dict(t=20, b=10, l=10, r=10), height=450, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'), legend=dict(x=0.01, y=0.99, orientation='h'), hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with epoch_col:
            st.subheader("Blocks per Epoch")
            if not epoch_block_history.empty:
                fig_epoch = go.Figure(go.Bar(
                    x=epoch_block_history['qubic_epoch'].astype(str),
                    y=epoch_block_history['blocks_in_epoch'],
                    marker_color='#3a0ca3'
                ))
                fig_epoch.update_layout(
                    xaxis_title="Epoch", yaxis_title="Blocks Found", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'), margin=dict(l=0, r=0, t=0, b=0), height=450,
                    xaxis=dict(type='category', gridcolor='rgba(255,255,255,0.1)'), yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                )
                st.plotly_chart(fig_epoch, use_container_width=True)
            else:
                st.info("Epoch history data is not available.")
    else:
        st.warning("Could not load pool data. Please check the database connection or try again later.")

with tab2:
    df_burn = load_burn_data()
    if not df_burn.empty:
        total_qubic_burned = df_burn['qubic_amount'].sum()
        total_usdt_burned = df_burn['usdt_value'].sum()
        
        st.subheader("🔥 Token Burn Summary")
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Total QUBIC Burned", f"{total_qubic_burned:,.0f}")
        bc2.metric("USD Equivalent at Burn", f"${total_usdt_burned:,.2f}")
        bc3.metric("Total Burn Transactions", f"{len(df_burn):,}")

        st.divider()
        
        st.subheader("📋 Recent Burn Transactions")
        df_display_burns = df_burn.copy()

        ### NEW ###
        # Placeholder for live price. In a real app, you'd fetch this.
        # Example: import ccxt; mexc = ccxt.mexc(); price = mexc.fetch_ticker('QUBIC/USDT')['last']
        live_qubic_price = 0.000007  # Replace with actual live price fetch
        st.info(f"Note: 'Current Value' is calculated using a placeholder price of ${live_qubic_price}. For a live value, a real-time price feed (e.g., from an exchange API) would be needed.")
        
        df_display_burns['Current Value ($)'] = df_display_burns['qubic_amount'] * live_qubic_price
        df_display_burns['TX_URL'] = "https://explorer.qubic.org/network/tx/" + df_display_burns['tx']
        df_display_burns = df_display_burns.sort_values('timestamp', ascending=False)
        
        st.dataframe(
            df_display_burns,
            column_order=("timestamp", "TX_URL", "qubic_amount", "usdt_value", "Current Value ($)"),
            use_container_width=True, hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
                "TX_URL": st.column_config.LinkColumn("TX ID", display_text=r"https://explorer\.qubic\.org/network/tx/(.+)"),
                "qubic_amount": st.column_config.NumberColumn("QUBIC Amount", format="%.0f"),
                "usdt_value": st.column_config.NumberColumn("Value at Burn ($)", format="$%.2f"),
                "Current Value ($)": st.column_config.NumberColumn("Current Value ($)", format="$%.2f")
            }
        )
    else:
        st.warning("No token burn data available.")

# --- Footer and Controls ---
st.divider()
bcol1, bcol2 = st.columns([1, 4])
with bcol1:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with bcol2:
    if st.button("🐱 Release The Beast", use_container_width=True):
        ### UPDATED ###
        display_floating_cat()

st.markdown("""
<div style="margin-top: 2em; text-align: center; font-size: 0.85em; color: #8b95a1;">
    📊 <strong>Data Source:</strong> <a href="https://xmr-stats.qubic.org/" target="_blank" style="color: #4cc9f0;">xmr-stats.qubic.org</a>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    💰 <strong>Price Data:</strong> Placeholder (formerly MEXC)
    &nbsp;&nbsp;|&nbsp;&nbsp;
    💌 <strong>Inspired by:</strong> <a href="https://qubic-xmr.vercel.app/" target="_blank" style="color: #4cc9f0;">qubic-xmr.vercel.app</a>
</div>
""", unsafe_allow_html=True)
