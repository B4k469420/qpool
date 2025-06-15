import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import psycopg2

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Qubic Mining Dashboard")

# --- Styling ---
st.markdown("""
<style>
    /* Main layout and theme */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Custom Metric Card Style */
    [data-testid="stMetric"] {
        background-color: #222b38;
        border: 1px solid #3a4a60;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    [data-testid="stMetricLabel"] {
      font-size: 1rem;
      color: #a0b0c0;
    }
    [data-testid="stMetricValue"] {
      font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Database Connection & Data Loading ---

@st.cache_data(ttl=60, show_spinner="Fetching latest pool data...")
def load_recent_xmr_data():
    """Fetches recent XMR pool statistics for live charts and metrics."""
    try:
        conn = psycopg2.connect(**st.secrets["database"])
        # Fetch more data to ensure calculations for 24h are accurate
        query = "SELECT * FROM pool_statistics ORDER BY timestamp DESC LIMIT 60000;"
        df = pd.read_sql_query(query, conn)
        conn.close()
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.sort_values('timestamp', inplace=True)
            df['pool_hashrate_mhs'] = df['pool_hashrate'] / 1e6
            df['block_found'] = df['pool_blocks_found'].diff() > 0
        return df
    except Exception as e:
        st.error(f"Failed to load XMR pool data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner="Fetching epoch history...")
def get_xmr_blocks_by_epoch():
    """Fetches and calculates the number of XMR blocks found in each epoch."""
    try:
        conn = psycopg2.connect(**st.secrets["database"])
        # Filter for relevant epochs to avoid calculation errors with initial data
        query = """
        SELECT qubic_epoch, MAX(pool_blocks_found) as blocks_at_epoch_end
        FROM pool_statistics
        WHERE qubic_epoch >= 160
        GROUP BY qubic_epoch
        ORDER BY qubic_epoch;
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        # Correctly calculate the diff for blocks within an epoch
        df['blocks_in_epoch'] = df['blocks_at_epoch_end'].diff()
        # For the first epoch in our dataset, the count is its own total
        df.loc[df.index[0], 'blocks_in_epoch'] = df.loc[df.index[0], 'blocks_at_epoch_end']
        return df[['qubic_epoch', 'blocks_in_epoch']].astype(int)
    except Exception as e:
        st.error(f"Failed to fetch XMR epoch data: {e}")
        return pd.DataFrame(columns=['qubic_epoch', 'blocks_in_epoch'])

@st.cache_data(ttl=60, show_spinner="Fetching Tari block data...")
def load_tari_data():
    """Fetches all Tari block data."""
    try:
        conn = psycopg2.connect(**st.secrets["database"])
        query = "SELECT * FROM tari_blocks ORDER BY block_height DESC;"
        df = pd.read_sql_query(query, conn)
        conn.close()
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Failed to load Tari block data: {e}")
        return pd.DataFrame()

# --- Helper Functions ---
def format_hashrate(h):
    if pd.isna(h) or h == 0: return "N/A"
    if h >= 1e9: return f"{h/1e9:.2f} GH/s"
    if h >= 1e6: return f"{h/1e6:.2f} MH/s"
    return f"{h:.2f} H/s"

def format_timespan(latest_timestamp, event_timestamp):
    if pd.isna(event_timestamp): return "N/A"
    delta = latest_timestamp - event_timestamp
    days = delta.days
    hours, rem = divmod(delta.seconds, 3600)
    minutes, _ = divmod(rem, 60)
    if days > 0: return f"{days}d {hours}h ago"
    if hours > 0: return f"{hours}h {minutes}m ago"
    return f"{minutes}m ago"

# --- Main Application ---
st.markdown("""
<div style="text-align: left; margin-bottom: 2rem;">
<svg width="95" height="26" viewBox="0 0 95 26" fill="none" xmlns="http://www.w3.org/2000/svg" class="cursor-pointer"><path d="M5.25 2H0.75C0.335786 2 0 2.33579 0 2.75V19.25C0 19.6642 0.335786 20 0.75 20H5.25C5.66421 20 6 19.6642 6 19.25V2.75C6 2.33579 5.66421 2 5.25 2Z" fill="white"></path><path d="M13.25 2H8.75C8.33579 2 8 2.33579 8 2.75V25.25C8 25.6642 8.33579 26 8.75 26H13.25C13.6642 26 14 25.6642 14 25.25V2.75C14 2.33579 13.6642 2 13.25 2Z" fill="white"></path><path d="M78.2335 20.5641C77.0029 20.5641 75.8848 20.3041 74.8795 19.7841C73.8915 19.2641 73.1028 18.5101 72.5135 17.5221C71.9415 16.5341 71.6555 15.3467 71.6555 13.9601V13.6221C71.6555 12.2354 71.9415 11.0567 72.5135 10.0861C73.1028 9.09807 73.8915 8.34407 74.8795 7.82407C75.8848 7.28673 77.0029 7.01807 78.2335 7.01807C79.4642 7.01807 80.5128 7.2434 81.3795 7.69407C82.2462 8.14473 82.9395 8.74273 83.4595 9.48807C83.9969 10.2334 84.3435 11.0567 84.4995 11.9581L81.8995 12.5041C81.8129 11.9321 81.6308 11.4121 81.3535 10.9441C81.0762 10.4761 80.6862 10.1034 80.1835 9.82607C79.6809 9.54873 79.0482 9.41007 78.2855 9.41007C77.5402 9.41007 76.8642 9.5834 76.2575 9.93007C75.6682 10.2594 75.2002 10.7447 74.8535 11.3861C74.5068 12.0101 74.3335 12.7727 74.3335 13.6741V13.9081C74.3335 14.8094 74.5068 15.5807 74.8535 16.2221C75.2002 16.8634 75.6682 17.3487 76.2575 17.6781C76.8642 18.0074 77.5402 18.1721 78.2855 18.1721C79.4122 18.1721 80.2702 17.8861 80.8595 17.3141C81.4488 16.7247 81.8215 15.9794 81.9775 15.0781L84.5775 15.6761C84.3695 16.5601 83.9969 17.3747 83.4595 18.1201C82.9395 18.8654 82.2462 19.4634 81.3795 19.9141C80.5128 20.3474 79.4642 20.5641 78.2335 20.5641Z" fill="white"></path><path d="M67.4473 20.2V7.382H70.1252V20.2H67.4473ZM68.7992 5.64C68.2792 5.64 67.8372 5.47533 67.4732 5.146C67.1266 4.79933 66.9532 4.35733 66.9532 3.82C66.9532 3.28267 67.1266 2.84933 67.4732 2.52C67.8372 2.17333 68.2792 2 68.7992 2C69.3366 2 69.7786 2.17333 70.1252 2.52C70.4719 2.84933 70.6452 3.28267 70.6452 3.82C70.6452 4.35733 70.4719 4.79933 70.1252 5.146C69.7786 5.47533 69.3366 5.64 68.7992 5.64Z" fill="white"></path><path d="M60.021 20.564C58.773 20.564 57.811 20.3387 57.135 19.888C56.4763 19.4373 55.9823 18.9347 55.653 18.38H55.237V20.2H52.611V2H55.289V9.124H55.705C55.913 8.77733 56.1903 8.448 56.537 8.136C56.8836 7.80667 57.343 7.538 57.915 7.33C58.487 7.122 59.189 7.018 60.021 7.018C61.0956 7.018 62.0836 7.278 62.985 7.798C63.8863 8.318 64.6056 9.072 65.143 10.06C65.6803 11.048 65.949 12.2267 65.949 13.596V13.986C65.949 15.3727 65.6716 16.56 65.117 17.548C64.5796 18.5187 63.8603 19.264 62.959 19.784C62.075 20.304 61.0956 20.564 60.021 20.564ZM59.241 18.224C60.4023 18.224 61.3556 17.8513 62.101 17.106C62.8636 16.3607 63.245 15.2947 63.245 13.908V13.674C63.245 12.3047 62.8723 11.2473 62.127 10.502C61.3816 9.75667 60.4196 9.384 59.241 9.384C58.097 9.384 57.1436 9.75667 56.381 10.502C55.6356 11.2473 55.263 12.3047 55.263 13.674V13.908C55.263 15.2947 55.6356 16.3607 56.381 17.106C57.1436 17.8513 58.097 18.224 59.241 18.224Z" fill="white"></path><path d="M43.3742 20.4341C42.4035 20.4341 41.5369 20.2174 40.7742 19.7841C40.0115 19.3507 39.4135 18.7354 38.9802 17.9381C38.5469 17.1407 38.3302 16.1874 38.3302 15.0781V7.38208H41.0082V14.8961C41.0082 16.0054 41.2855 16.8287 41.8402 17.3661C42.3949 17.8861 43.1662 18.1461 44.1542 18.1461C45.2462 18.1461 46.1215 17.7821 46.7802 17.0541C47.4562 16.3087 47.7942 15.2427 47.7942 13.8561V7.38208H50.4722V20.2001H47.8462V18.2761H47.4302C47.1875 18.7961 46.7542 19.2901 46.1302 19.7581C45.5062 20.2087 44.5875 20.4341 43.3742 20.4341Z" fill="white"></path><path d="M33.66 25.4001V18.4581H33.244C33.0533 18.8047 32.776 19.1427 32.412 19.4721C32.048 19.7841 31.58 20.0441 31.008 20.2521C30.4533 20.4601 29.76 20.5641 28.928 20.5641C27.8533 20.5641 26.8653 20.3041 25.964 19.7841C25.0627 19.2641 24.3433 18.5187 23.806 17.5481C23.2687 16.5601 23 15.3727 23 13.9861V13.5961C23 12.2094 23.2687 11.0307 23.806 10.0601C24.3607 9.07207 25.0887 8.31807 25.99 7.79807C26.8913 7.27807 27.8707 7.01807 28.928 7.01807C30.176 7.01807 31.1293 7.2434 31.788 7.69407C32.464 8.14473 32.9667 8.65607 33.296 9.22807H33.712V7.38207H36.338V25.4001H33.66ZM29.682 18.2241C30.8607 18.2241 31.8227 17.8514 32.568 17.1061C33.3133 16.3607 33.686 15.2947 33.686 13.9081V13.6741C33.686 12.3047 33.3047 11.2474 32.542 10.5021C31.7967 9.75673 30.8433 9.38407 29.682 9.38407C28.538 9.38407 27.5847 9.75673 26.822 10.5021C26.0767 11.2474 25.704 12.3047 25.704 13.6741V13.9081C25.704 15.2947 26.0767 16.3607 26.822 17.1061C27.5847 17.8514 28.538 18.2241 29.682 18.2241Z" fill="white"></path></svg>
</div>
""", unsafe_allow_html=True)

# Load data required for all tabs
xmr_df = load_recent_xmr_data()
tari_df = load_tari_data()
xmr_epoch_history = get_xmr_blocks_by_epoch()

# Main Tabs
overall_tab, xmr_tab, tari_tab = st.tabs(["📊 Overall", " H/s XMR Pool", "⛏️ Tari Pool"])

# --- Overall Tab ---
with overall_tab:
    st.header("Overall Mining Performance")
    st.markdown("A high-level view of blocks found by each mining pool across Qubic epochs.")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total XMR Pool Blocks", f"{xmr_epoch_history['blocks_in_epoch'].sum():,}" if not xmr_epoch_history.empty else "0")
    with c2:
        st.metric("Total Tari Pool Blocks", f"{len(tari_df):,}" if not tari_df.empty else "0")

    st.divider()
    st.subheader("Blocks per Epoch Heatmap")

    # Prepare data for heatmap
    if not tari_df.empty:
        tari_epoch_history = tari_df.groupby('epoch')['block_height'].count().reset_index()
        tari_epoch_history.rename(columns={'block_height': 'blocks_in_epoch', 'epoch': 'qubic_epoch'}, inplace=True)
        
        # Merge dataframes
        heatmap_df = pd.merge(xmr_epoch_history, tari_epoch_history, on='qubic_epoch', how='outer').fillna(0)
        heatmap_df.rename(columns={'blocks_in_epoch_x': 'XMR Pool', 'blocks_in_epoch_y': 'Tari Pool'}, inplace=True)
        heatmap_df = heatmap_df.sort_values('qubic_epoch')
        
        # Create Heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
                   z=[heatmap_df['XMR Pool'], heatmap_df['Tari Pool']],
                   x=heatmap_df['qubic_epoch'],
                   y=['XMR Pool', 'Tari Pool'],
                   hoverongaps=False,
                   colorscale='Viridis',
                   hovertemplate='<b>Epoch:</b> %{x}<br><b>Blocks:</b> %{z}<extra></extra>'))
        
        fig_heatmap.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis_title="Qubic Epoch",
            yaxis_title="Mining Pool",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Not enough data to display heatmap.")


# --- XMR Pool Tab ---
with xmr_tab:
    st.header("Monero (XMR) Pool Statistics")
    if not xmr_df.empty:
        latest_xmr = xmr_df.iloc[-1]
        
        # Metric Calculations
        day_ago_xmr = xmr_df['timestamp'].max() - timedelta(hours=24)
        day_av_xmr = xmr_df[xmr_df['timestamp'] >= day_ago_xmr]
        mean_hash_24h = day_av_xmr['pool_hashrate'].mean()
        
        last_block_time_xmr = xmr_df[xmr_df['block_found']]['timestamp'].iloc[-1] if xmr_df['block_found'].any() else None
        
        # UI Layout
        st.subheader("Live Performance")
        c1, c2, c3 = st.columns(3)
        c1.metric("Pool Hashrate", format_hashrate(latest_xmr['pool_hashrate']))
        c2.metric("Network Hashrate", format_hashrate(latest_xmr['network_hashrate']))
        c3.metric("Last Block Found", format_timespan(latest_xmr['timestamp'], last_block_time_xmr))
        
        st.divider()
        
        # Charts
        chart_col, epoch_col = st.columns([3, 1])
        with chart_col:
            st.subheader("Pool & Network Hashrate (Last 24 Hours)")
            df_chart = xmr_df[xmr_df['timestamp'] >= day_ago_xmr]
            
            fig_xmr = go.Figure()
            fig_xmr.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['pool_hashrate_mhs'], name='Pool (MH/s)', line=dict(color='#4cc9f0')))
            fig_xmr.add_trace(go.Scatter(x=df_chart[df_chart['block_found']]['timestamp'], y=df_chart[df_chart['block_found']]['pool_hashrate_mhs'], mode='markers', name='Block Found', marker=dict(symbol='star', size=12, color='gold')))
            
            fig_xmr.update_layout(
                xaxis_title='Time', yaxis_title='Pool Hashrate (MH/s)',
                margin=dict(t=20, b=10, l=10, r=10), height=400, plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                legend=dict(x=0.01, y=0.99, orientation='h'), hovermode='x unified'
            )
            st.plotly_chart(fig_xmr, use_container_width=True)

        with epoch_col:
            st.subheader("Blocks per Epoch")
            fig_epoch_xmr = go.Figure(go.Bar(
                x=xmr_epoch_history['qubic_epoch'].astype(str),
                y=xmr_epoch_history['blocks_in_epoch'],
                marker_color='#3a0ca3'
            ))
            fig_epoch_xmr.update_layout(
                xaxis_title="Epoch", yaxis_title="Blocks Found", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='white', margin=dict(l=0, r=0, t=0, b=0), height=400,
                xaxis=dict(type='category')
            )
            st.plotly_chart(fig_epoch_xmr, use_container_width=True)
    else:
        st.warning("Could not load XMR pool data.")


# --- Tari Pool Tab ---
with tari_tab:
    st.header("Tari (XTM) Pool Statistics")
    if not tari_df.empty:
        latest_tari = tari_df.iloc[0]
        
        # Metrics
        st.subheader("Live Performance")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Blocks Found", f"{len(tari_df):,}")
        c2.metric("Total Rewards", f"{tari_df['reward'].sum():,.2f} XTM")
        c3.metric("Last Block Found", format_timespan(datetime.now(timezone.utc), latest_tari['timestamp']))

        st.divider()

        # Charts and Data
        epoch_col, data_col = st.columns([1, 2])
        with epoch_col:
            st.subheader("Blocks per Epoch")
            tari_epoch_history = tari_df.groupby('epoch')['block_height'].count().reset_index()
            tari_epoch_history.rename(columns={'block_height': 'blocks_in_epoch', 'epoch':'qubic_epoch'}, inplace=True)
            
            fig_epoch_tari = go.Figure(go.Bar(
                x=tari_epoch_history['qubic_epoch'].astype(str),
                y=tari_epoch_history['blocks_in_epoch'],
                marker_color='#f72585'
            ))
            fig_epoch_tari.update_layout(
                xaxis_title="Epoch", yaxis_title="Blocks Found", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='white', margin=dict(l=0, r=0, t=0, b=0), height=400,
                xaxis=dict(type='category')
            )
            st.plotly_chart(fig_epoch_tari, use_container_width=True)
            
        with data_col:
            st.subheader("Recent Blocks Found")
            tari_df['url'] = "https://textexplore.tari.com/blocks/" + tari_df['block_height'].astype(str)
            st.dataframe(
                tari_df.head(20),
                column_order=("timestamp", "block_height", "reward", "epoch", "url"),
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Timestamp (UTC)", format="YYYY-MM-DD HH:mm"),
                    "block_height": "Block Height",
                    "reward": st.column_config.NumberColumn("Reward (XTM)", format="%.2f"),
                    "epoch": "Qubic Epoch",
                    "url": st.column_config.LinkColumn("Explorer Link", display_text="View Block →")
                },
                hide_index=True,
                use_container_width=True,
                height=465
            )
    else:
        st.warning("Could not load Tari pool data.")

# --- Footer and Controls ---
st.divider()
if st.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
