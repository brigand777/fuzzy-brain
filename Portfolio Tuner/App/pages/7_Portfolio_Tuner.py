import streamlit as st
import pandas as pd
import os
import pytz
from datetime import datetime

from auth import login_and_get_status
from utils.api_client import call_fastapi_optimizer
from optimizer import run_optimizers
from utils.plots import (
    plotly_pie_allocation, plotly_bar_allocation,
    plot_cumulative_returns, plot_rolling_sharpe, plot_drawdowns,
    plot_allocations_per_method, add_interactivity,
    generate_styled_summary_table
)
from utils.utils import downsample_results_dict
from utils.backtest import dynamic_backtest_portfolio, dynamic_backtest_portfolio_user_fixed_shares
from components.portfolio_input import edit_portfolio
from user_input import get_optimization_methods, get_backtest_settings

# --- Page Setup ---
st.set_page_config(page_title="All-in-One Portfolio Tool", layout="wide")
authenticator, authentication_status, username = login_and_get_status()
st.title("💼 Portfolio Lab")

# --- Style Helper ---
def narrative(text, color="#1F77B4"):
    st.markdown(
        f"""<div style=\"background-color: rgba(31, 119, 180, 0.1); padding: 10px; border-left: 4px solid {color}; font-size: 18px; margin-bottom: 10px;\">
        {text}
        </div>""",
        unsafe_allow_html=True
    )

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_parquet("Portfolio Tuner/App/data/prices.parquet")

with st.spinner("Loading price data..."):
    try:
        data = load_data()
        available_assets = data.columns.tolist()
        available_dates = data.index.sort_values()
    except Exception as e:
        st.error(f"❌ Failed to load price data: {e}")
        st.stop()

# --- Step 1: Portfolio Setup ---
st.markdown("## Step 1: 📁 Select Portfolio")
input_mode = st.radio("Where is your portfolio coming from?", ["Use My Saved Portfolio", "Build Portfolio Now"])
portfolio_df = None
persistent = False

if input_mode == "Use My Saved Portfolio":
    if authentication_status:
        portfolio_path = f"Portfolio Tuner/App/portfolios/{username}_portfolio.csv"
        if os.path.exists(portfolio_path):
            portfolio_df = pd.read_csv(portfolio_path)
            st.success("📂 Loaded your saved portfolio.")
            persistent = True
        else:
            st.warning("⚠️ No saved portfolio found. Please switch to 'Build Portfolio Now'.")
            st.stop()
    else:
        st.warning("🔐 Login required to access saved portfolios.")
        st.stop()
else:
    portfolio_df = edit_portfolio(available_assets, data, persistent=False)

if portfolio_df.empty or "Asset" not in portfolio_df.columns:
    st.warning("🚫 Your portfolio is empty. Please add assets.")
    st.stop()

st.dataframe(portfolio_df, use_container_width=True)

# ------------------- BACKTESTING SECTION -------------------

st.markdown("## Step 2: 🧪 Backtest Your Portfolio")
col1, col2 = st.columns(2)

with col1:
    start_naive = st.date_input("📅 Start Date", value=available_dates[-252].date())
with col2:
    end_naive = st.date_input("📅 End Date", value=available_dates[-1].date())

start_date = pd.Timestamp(datetime.combine(start_naive, datetime.min.time()), tz="UTC")
end_date = pd.Timestamp(datetime.combine(end_naive, datetime.min.time()), tz="UTC")

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

with st.expander("ℹ️ Strategy Explanations"):
    st.markdown("""
    - **Equal Weight**: Equal investment in each asset.
    - **Smart Spread (Mean Variance)**: Historical optimization.
    - **HRB**: Risk-balanced by group.
    - **User Portfolio**: Your custom mix.
    """)

default_methods = ["Equal Weight", "Mean Variance", "HRB", "User Portfolio"]
selected_methods = st.multiselect("🧠 Backtest Strategies:", options=default_methods, default=default_methods)

with st.expander("🛠️ Advanced Settings"):
    rebalance_days = st.slider("🔁 Rebalance Frequency (days)", 7, 90, 30, step=7)
    lookback_days = st.slider("📊 Lookback Period (days)", 30, 365, 90, step=30)
    nonnegative_toggle = st.toggle("📉 Disallow short-selling?", value=True)

simulation_data = data.loc[start_date:end_date, [col for col in portfolio_df['Asset'] if col in data.columns]]

if simulation_data.empty:
    st.error("❌ No data for selected period.")
    st.stop()

if st.button("Run Backtest"):
    with st.spinner("Running simulation..."):
        try:
            latest_prices = data.iloc[-1]
            values = portfolio_df.apply(lambda row: row["Amount"] * latest_prices.get(row["Asset"], 0), axis=1)
            total_value = values.sum()
            user_weights = {
                row["Asset"]: (row["Amount"] * latest_prices.get(row["Asset"], 0)) / total_value
                for _, row in portfolio_df.iterrows()
                if latest_prices.get(row["Asset"], 0) > 0
            }
            lookback_window = data.loc[pd.to_datetime(start_date) - pd.Timedelta(days=lookback_days):start_date]
            initial_allocations = run_optimizers(lookback_window[user_weights.keys()], nonnegative_mvo=nonnegative_toggle)
            initial_allocations["User Portfolio"] = user_weights

            results_dict = {}
            for method in selected_methods:
                if method == "User Portfolio":
                    user_shares = {row["Asset"]: row["Amount"] for _, row in portfolio_df.iterrows()}
                    res = dynamic_backtest_portfolio_user_fixed_shares(simulation_data, asset_amounts=user_shares)
                else:
                    res = dynamic_backtest_portfolio(simulation_data, method, lookback_days, rebalance_days, nonnegative_toggle)
                results_dict[method] = res

            downsampled = downsample_results_dict(results_dict, start_date, end_date)

            st.session_state.backtest_results = results_dict
            st.session_state.downsampled = downsampled
            st.session_state.selected_methods = selected_methods

        except Exception as e:
            st.error(f"💥 Error: {e}")

if "downsampled" in st.session_state:
    st.markdown("## 📊 Backtest Results")
    downsampled = st.session_state.downsampled
    selected_methods = st.session_state.selected_methods
    st.altair_chart(add_interactivity(plot_cumulative_returns(downsampled), x_field="date", y_field="cumulative"), use_container_width=True)
    st.altair_chart(add_interactivity(plot_rolling_sharpe(downsampled), x_field="date", y_field="rolling_sharpe"), use_container_width=True)
    st.altair_chart(add_interactivity(plot_drawdowns(downsampled), x_field="date", y_field="drawdown"), use_container_width=True)

    for method in selected_methods:
        with st.expander(f"{method} Allocations Over Time"):
            st.altair_chart(
                add_interactivity(plot_allocations_per_method(downsampled[method]["allocations"], method), x_field="date", y_field="Allocation"),
                use_container_width=True
            )

    st.dataframe(generate_styled_summary_table(downsampled), use_container_width=True)

# ------------------- OPTIMIZER SECTION -------------------

st.markdown("## 📌 Today's Optimal Allocations")
narrative("These are strategy-based recommendations based on recent price trends.", color="#4CAF50")

lookback = st.selectbox("📆 Lookback period for optimizer", [30, 60, 90, 180, 365], index=2)
optimize_now = st.button("Run Optimizer")

if optimize_now:
    try:
        latest_prices = data.iloc[-1]
        values = portfolio_df.apply(lambda row: row["Amount"] * latest_prices.get(row["Asset"], 0), axis=1)
        total_value = values.sum()
        user_weights = {
            row["Asset"]: (row["Amount"] * latest_prices.get(row["Asset"], 0)) / total_value
            for _, row in portfolio_df.iterrows()
            if latest_prices.get(row["Asset"], 0) > 0
        }

        lookback_df = data[user_weights.keys()].tail(lookback)
        all_allocations = run_optimizers(lookback_df, nonnegative_mvo=True)
        all_allocations["User Portfolio"] = pd.Series(user_weights)

        st.session_state.optimizer_allocations = all_allocations
        st.session_state.optimizer_methods = selected_methods

        st.success("✅ Allocation analysis complete.")

    except Exception as e:
        st.error("❌ Optimization failed.")
        st.error(f"Details: {e}")

if "optimizer_allocations" in st.session_state:
    st.markdown("### 🥧 Pie Charts (Investment Mix)")
    pie_cols = st.columns(len(st.session_state.optimizer_methods))
    for i, method in enumerate(st.session_state.optimizer_methods):
        weights = st.session_state.optimizer_allocations[method]
        fig = plotly_pie_allocation(weights, title=f"{method} Allocation")
        with pie_cols[i]:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📈 Bar Charts (Compare Strategies)")
    bar_cols = st.columns(2)
    for i, method in enumerate(st.session_state.optimizer_methods):
        weights = st.session_state.optimizer_allocations[method]
        fig = plotly_bar_allocation(weights, title=f"{method} Allocation Breakdown")
        with bar_cols[i % 2]:
            st.plotly_chart(fig, use_container_width=True)
