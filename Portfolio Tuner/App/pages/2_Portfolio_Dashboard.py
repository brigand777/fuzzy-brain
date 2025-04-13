import streamlit as st
import pandas as pd
import os

from auth import login_and_get_status
from utils.plots import (
    plot_portfolio_dashboard,
    plot_historical_assets,
    plot_asset_cumulative_returns,
    plot_gauge_charts,
    plot_portfolio_absolute_value
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

    # --- Benchmark Selector (controlled via session_state) ---
    if "selected_benchmark" not in st.session_state:
        st.session_state.selected_benchmark = "BTC" if "BTC" in available_assets else "None"



    # --- Date range selector ---
    with st.expander("🕝 Select Date Range"):
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
    benchmark = st.session_state.selected_benchmark

    if selected_assets:
        try:
            metrics_fig, heatmap_fig = plot_portfolio_dashboard(
                data, selected_assets,
                portfolio_df=portfolio_df,
                date_range=date_range,
                benchmark=benchmark if benchmark != "None" else None
            )
        except Exception as e:
            st.error(f"⚠️ Error in plot_portfolio_dashboard: {type(e).__name__} — {e}")
            st.stop()

        start_date, end_date = date_range
        start_date = ensure_utc(start_date)
        end_date = ensure_utc(end_date)

        # --- Cumulative Portfolio Value Chart (Centered) ---
        st.subheader("📊 Portfolio Value Over Time")
        cumulative_chart = plot_portfolio_absolute_value(
            data, selected_assets,
            start=start_date, end=end_date,
            portfolio_df=portfolio_df)
        
        st.selectbox(
            "🔍 Select a benchmark for your portfolio comparison:",
            options=["None"] + available_assets,
            index=(available_assets.index(st.session_state.selected_benchmark) + 1) if st.session_state.selected_benchmark in available_assets else 0,
            key="selected_benchmark"
        )
        col1, col2, col3 = st.columns([0.2, 8, 1])
        with col2:
            st.altair_chart(cumulative_chart, use_container_width=True)

        # --- Needle Charts (6 Porsche-inspired gauges) ---
        st.markdown("### 🧭 Portfolio Metrics")
        if metrics_fig and len(metrics_fig) == 6:
            row1 = st.columns(2)
            for col, fig in zip(row1, metrics_fig[:2]):
                with col:
                    st.plotly_chart(fig, use_container_width=True)

            row2 = st.columns(4)
            for col, fig in zip(row2, metrics_fig[2:]):
                with col:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Expected 6 metrics for layout, but received a different number.")

        # --- Comparison Charts (2-column layout) ---
        st.markdown("### 🔍 Portfolio Comparison")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Correlation Heatmap")
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)

        with col2:
            st.subheader("Cumulative Return vs. Benchmark")
            benchmark_input = st.session_state.selected_benchmark

            if benchmark_input != "None":
                benchmark_chart = plot_asset_cumulative_returns(
                    data, selected_assets,
                    benchmark=benchmark_input,
                    start=start_date, end=end_date,
                    portfolio_df=portfolio_df
                )
                st.altair_chart(benchmark_chart, use_container_width=True)
            else:
                st.info("Select a benchmark to display comparison.")

        # --- Optional historical charts toggle ---
        if "show_plot" not in st.session_state:
            st.session_state.show_plot = False

        if st.button("📊 Show Historical Asset Performance"):
            st.session_state.show_plot = not st.session_state.show_plot

        if st.session_state.show_plot:
            try:
                plot_historical_assets(
                    data,
                    selected_assets,
                    portfolio_df=portfolio_df,
                    date_range_default=(start_date, end_date)
                )
            except Exception as e:
                import traceback
                st.error("⚠️ An error occurred in `plot_historical_assets()`")
                st.code(traceback.format_exc())
                st.stop()
    else:
        st.warning("No valid assets found in your portfolio.")
else:
    st.warning("Please log in to view saved portfolio data.")

# --- Navigation ---
st.markdown("---")
st.markdown("[← Back to Portfolio Editor](1_My_Portfolio.py)")
