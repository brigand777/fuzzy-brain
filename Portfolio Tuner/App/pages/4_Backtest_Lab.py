import streamlit as st
import pandas as pd
import altair as alt
import os
from datetime import timedelta
from datetime import datetime
import pytz

from auth import login_and_get_status
from utils.backtest import dynamic_backtest_portfolio, dynamic_backtest_portfolio_user_fixed_shares
from utils.plots import (
    plot_cumulative_returns,
    plot_rolling_sharpe,
    plot_drawdowns,
    plot_allocations_per_method,
    add_interactivity,
    plot_historical_assets,
    generate_styled_summary_table
)
from utils.utils import downsample_results_dict
from components.portfolio_input import edit_portfolio
from user_input import get_backtest_settings, get_optimization_methods
from optimizer import run_optimizers

# --- Page Setup ---
st.set_page_config(page_title="Backtest Lab", layout="wide")
authenticator, authentication_status, username = login_and_get_status()
st.title("⏳ Backtest Lab")

# --- Style Helper ---
def narrative(text):
    st.markdown(
        f"""<div style="background-color: rgba(255, 235, 59, 0.2); padding: 10px; border-left: 4px solid #FBC02D; font-size: 18px; margin-bottom: 10px;">
        {text}
        </div>""",
        unsafe_allow_html=True
    )

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_parquet("Portfolio Tuner/App/data/prices.parquet")

data = load_data()
available_dates = data.index.sort_values()
available_assets = data.columns.tolist()

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


utc = pytz.UTC

st.markdown("## Step 2: ⚙️ Customize Your Backtest")
narrative("Select the period you want to test and adjust optional strategy settings.")

# --- Dates (always shown) ---
col1, col2 = st.columns(2)

with col1:
    start_naive = st.date_input(
        "📅 Start Date",
        value=available_dates[-252].date(),
        min_value=available_dates[0].date(),
        max_value=available_dates[-2].date()
    )

with col2:
    end_naive = st.date_input(
        "📅 End Date",
        value=available_dates[-1].date(),
        min_value=start_naive,
        max_value=available_dates[-1].date()
    )

# Convert to UTC-aware timestamps
start_date = pd.Timestamp(datetime.combine(start_naive, datetime.min.time()), tz="UTC")
end_date = pd.Timestamp(datetime.combine(end_naive, datetime.min.time()), tz="UTC")

# --- Validate Date Range ---
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()
st.markdown(
    """
    <style>
    .streamlit-expanderHeader {
        font-size: 20px !important;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Advanced Settings ---
# Strategy descriptions (placed BEFORE advanced settings)
with st.expander("ℹ️ What do these strategies mean?", expanded=False):
    st.markdown("""
    - **Equal Weight**: Every asset gets the same amount of money.
    - **Smart Spread (Mean Variance / MVO)**: Uses past price data to find what *would have worked best* historically.
    - **Group & Balance (HRB)**: Groups similar investments and distributes risk evenly across them.
    - **Your Portfolio**: Your custom investment mix.
    """)

# --- Advanced Settings ---
with st.expander("🛠️ <span style='font-size: 20px;'>Show Advanced Strategy Settings</span>", expanded=False):
    rebalance_days = st.slider(
        "🔁 How often should the portfolio be rebalanced?",
        min_value=7,
        max_value=90,
        step=7,
        value=30,
        help="How frequently we recalculate and adjust the portfolio"
    )

    lookback_days = st.slider(
        "📊 Lookback window (in days)",
        min_value=30,
        max_value=365,
        step=30,
        value=90,
        help="More days = more historical data, but may be slower to adapt"
    )

    nonnegative_toggle = st.toggle(
        "📉 Disallow negative weights (no short-selling)?",
        value=True
    )

    st.markdown("#### 🧠 Choose Strategies to Compare")

    default_methods = ["Equal Weight", "Mean Variance", "HRB", "User Portfolio"]

    selected_methods = st.multiselect(
        "🧠 Select optimization strategies to compare during the backtest:",
        options=default_methods,
        default=default_methods,
        help="Choose one or more strategies to simulate and compare"
    )


if simulation_data.empty:
    st.error("❌ No data available for the selected backtest period.")
    st.stop()

# --- Step 3: Run Backtest ---
st.markdown("## Step 3: 🚀 Simulate Portfolio")

run_test = st.button("Run Backtest")
if run_test:
    with st.spinner("Running simulation..."):
        try:
            # Portfolio weights
            latest_prices = data.iloc[-1]
            portfolio_df = portfolio_df.dropna(subset=["Asset", "Amount"])
            values = portfolio_df.apply(lambda row: row["Amount"] * latest_prices.get(row["Asset"], 0), axis=1)
            total_value = values.sum()
            user_weights = {
                row["Asset"]: (row["Amount"] * latest_prices.get(row["Asset"], 0)) / total_value
                for _, row in portfolio_df.iterrows()
                if latest_prices.get(row["Asset"], 0) > 0
            }

            # Backtest allocations
            lookback_window = data.loc[pd.to_datetime(start_date) - pd.Timedelta(days=lookback_days):start_date]
            initial_allocations = run_optimizers(lookback_window[selected_assets], nonnegative_mvo=nonnegative_toggle)
            initial_allocations["User Portfolio"] = user_weights
            # Use the user-selected methods from the advanced settings
            selected_methods = [m for m in selected_methods if m in initial_allocations]


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
            st.error("💥 Something went wrong during the simulation.")
            st.error(f"Error details: {e}")

# --- Step 4: Show Results ---
if st.session_state.get("downsampled"):
    downsampled = st.session_state.downsampled
    selected_methods = st.session_state.selected_methods

    st.markdown("## Step 4: 📊 Review Results")

    st.markdown("### 🚀 Cumulative Returns")
    st.altair_chart(add_interactivity(plot_cumulative_returns(downsampled), x_field="date", y_field="cumulative"), use_container_width=True)

    st.markdown("### 📈 Rolling Sharpe Ratio")
    st.altair_chart(add_interactivity(plot_rolling_sharpe(downsampled), x_field="date", y_field="rolling_sharpe"), use_container_width=True)

    st.markdown("### 📉 Max Drawdowns")
    st.altair_chart(add_interactivity(plot_drawdowns(downsampled), x_field="date", y_field="drawdown"), use_container_width=True)

    st.markdown("### 🧩 Allocations Over Time")
    for method in selected_methods:
        with st.expander(f"{method} Asset Allocations Over Time"):
            st.altair_chart(
                add_interactivity(
                    plot_allocations_per_method(downsampled[method]["allocations"], method),
                    x_field="date",
                    y_field="Allocation"
                ),
                use_container_width=True
            )

    # Summary
    st.markdown("### 📋 Backtest Summary")
    st.write(f"**Date Range:** {start_date} to {end_date}")
    st.write(f"**Rebalance Every:** {rebalance_days} days")
    st.write(f"**Lookback Window:** {lookback_days} days before each rebalance")
    st.markdown("---")

    styled_table = generate_styled_summary_table(st.session_state.downsampled)
    st.markdown("### 🧾 Performance Table")
    st.dataframe(styled_table, use_container_width=True)
