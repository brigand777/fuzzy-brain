import streamlit as st
import pandas as pd
import os

from auth import login_and_get_status
from utils.plots import (
    plot_portfolio_dashboard,
    plot_historical_assets,
    plot_asset_cumulative_returns,
    plot_gauge_charts
)

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
authenticator, authentication_status, username = login_and_get_status()

st.title("📊 Portfolio Dashboard")

# --- Load asset data ---
@st.cache_data
def load_data():
    return pd.read_parquet("Portfolio Tuner/App/data/prices.parquet")

def ensure_utc(dt):
    dt = pd.to_datetime(dt)
    return dt if dt.tzinfo else dt.tz_localize("UTC")

data = load_data()
available_assets = data.columns.tolist()

# --- Load saved portfolio ---
if authentication_status:
    portfolio_path = f"Portfolio Tuner/App/portfolios/{username}_portfolio.csv"
    if os.path.exists(portfolio_path):
        portfolio_df = pd.read_csv(portfolio_path)
    else:
        st.warning("No saved portfolio found. Please create one in the Portfolio Editor.")
        st.stop()
else:
    st.warning("Please log in to view saved portfolio data.")
    st.stop()

# --- Date range selector ---
with st.expander("📅 Select Date Range"):
    max_date = data.index.max()
    min_date = data.index.min()
    default_start = max_date - pd.Timedelta(days=100)
    date_range = st.date_input(
        "Select date range for Portfolio Dashboard:",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date
    )

# --- Dashboard Visualization ---
selected_assets = portfolio_df["Asset"].dropna().unique().tolist()
if selected_assets:
    metrics_fig, heatmap_fig = plot_portfolio_dashboard(
        data, selected_assets, date_range=date_range
    )

    # --- Benchmark selection ---
    benchmark = st.selectbox(
        "Select a benchmark for comparison:",
        options=["None"] + available_assets,
        index=available_assets.index("BTC") + 1 if "BTC" in available_assets else 0
    )
    benchmark = None if benchmark == "None" else benchmark

    start_date, end_date = date_range
    start_date = ensure_utc(start_date)
    end_date = ensure_utc(end_date)

    # --- Cumulative Portfolio Value Chart (Centered) ---
    st.markdown("### 📊 Portfolio Value Over Time")
    custom_width_percent = 75  # 👈 Control this to adjust chart width as % of screen

    with st.container():
        st.markdown(
            f"""
            <div style='width: {custom_width_percent}%; margin: 0 auto;'>
            """,
            unsafe_allow_html=True
        )

        cumulative_chart = plot_asset_cumulative_returns(
            data, selected_assets,
            benchmark=None,
            start=start_date, end=end_date,
            portfolio_df=portfolio_df
        )
        st.altair_chart(cumulative_chart, use_container_width=False)

        st.markdown("</div>", unsafe_allow_html=True)

    # --- Needle Charts (6 Porsche-inspired gauges) ---
    st.markdown("### 🧭 Portfolio Metrics ")
    if metrics_fig:  # This is your list of 6 Plotly figures
        cols = st.columns(6)
        for col, fig in zip(cols, metrics_fig):
            with col:
                st.plotly_chart(fig, use_container_width=True)

    # --- Comparison Charts (2-column layout) ---
    st.markdown("### 🔍 Portfolio Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Correlation Heatmap")
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)

    with col2:
        st.subheader("Cumulative Return vs. Benchmark")
        if benchmark:
            benchmark_chart = plot_asset_cumulative_returns(
                data, selected_assets,
                benchmark=benchmark,
                start=start_date, end=end_date,
                portfolio_df=portfolio_df
            )
            st.altair_chart(benchmark_chart, use_container_width=True)
        else:
            st.info("Select a benchmark to display comparison.")
else:
    st.warning("No valid assets found in your portfolio.")

# --- Optional historical charts toggle ---
if "show_plot" not in st.session_state:
    st.session_state.show_plot = False

if st.button("📊 Show Historical Asset Performance"):
    st.session_state.show_plot = not st.session_state.show_plot

if st.session_state.show_plot:
    if selected_assets:
        plot_historical_assets(data, selected_assets, portfolio_df=portfolio_df)
    else:
        st.warning("No assets found to plot.")

# --- Navigation ---
st.markdown("---")
st.markdown("[← Back to Portfolio Editor](1_My_Portfolio.py)")
