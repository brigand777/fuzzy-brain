import streamlit as st
import pandas as pd
import os

from auth import login_and_get_status
from utils.plots import (
    plot_portfolio_dashboard,
    plot_historical_assets
)

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
authenticator, authentication_status, username = login_and_get_status()

st.title("📊 Portfolio Dashboard")

# --- Load asset data ---
@st.cache_data
def load_data():
    return pd.read_parquet("Portfolio Tuner/App/data/prices.parquet")

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
    needle_fig, heatmap_fig = plot_portfolio_dashboard(
        data, selected_assets, date_range=date_range
    )
    if needle_fig:
        st.plotly_chart(needle_fig, use_container_width=True)
    if heatmap_fig:
        st.plotly_chart(heatmap_fig, use_container_width=True)
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
