import streamlit as st
import pandas as pd
import altair as alt
import os
from datetime import timedelta

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

st.set_page_config(page_title="Backtest Lab", layout="wide")

# --- Auth ---
authenticator, authentication_status, username = login_and_get_status()
st.title("⏳ Backtest Lab")

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_parquet("Portfolio Tuner/App/data/prices.parquet")

data = load_data()
available_dates = data.index.sort_values()
available_assets = data.columns.tolist()

# --- Portfolio Input ---
input_mode = st.radio("Choose Portfolio Input Method", ["Use My Portfolio", "Build Portfolio Here"])
portfolio_df = None
persistent = False

if input_mode == "Use My Portfolio":
    if authentication_status:
        portfolio_path = f"Portfolio Tuner/App/portfolios/{username}_portfolio.csv"
        if os.path.exists(portfolio_path):
            portfolio_df = pd.read_csv(portfolio_path)
            st.success("Loaded your saved portfolio.")
            persistent = True
        else:
            st.warning("No saved portfolio found. Please add assets in 'My Portfolio'.")
            st.stop()
    else:
        st.warning("Login required to use saved portfolio.")
        st.stop()
else:
    portfolio_df = edit_portfolio(available_assets, data, persistent=False)

if portfolio_df.empty or "Asset" not in portfolio_df.columns:
    st.warning("Your portfolio is empty. Please add assets.")
    st.stop()

# --- Backtest Settings ---
start_date, end_date, lookback_days, rebalance_days, nonnegative_toggle = get_backtest_settings(available_dates)

selected_assets = portfolio_df["Asset"].dropna().unique().tolist()
data = data[[col for col in selected_assets if col in data.columns]]
simulation_data = data.loc[start_date:end_date]

if simulation_data.empty:
    st.error("No data available for the selected backtest period.")
    st.stop()

# --- State Setup ---
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None
    st.session_state.downsampled = None
    st.session_state.selected_methods = None

# --- Run Backtest ---
st.markdown("## 📈 Backtest Portfolio vs. Strategies")

if st.button("Run Backtest"):
    with st.spinner("Running backtest..."):
        try:
            # Prepare user weights
            latest_prices = data.iloc[-1]
            values = portfolio_df.apply(lambda row: row["Amount"] * latest_prices.get(row["Asset"], 0), axis=1)
            total_value = values.sum()
            user_weights = {
                row["Asset"]: (row["Amount"] * latest_prices.get(row["Asset"], 0)) / total_value
                for _, row in portfolio_df.iterrows()
                if latest_prices.get(row["Asset"], 0) > 0
            }

            lookback_window = data.loc[pd.to_datetime(start_date) - pd.Timedelta(days=lookback_days):start_date]
            initial_allocations = run_optimizers(lookback_window[selected_assets], nonnegative_mvo=nonnegative_toggle)
            initial_allocations["User Portfolio"] = user_weights
            selected_methods = get_optimization_methods(initial_allocations)

            results_dict = {}
            for method in selected_methods:
                if method == "User Portfolio":
                    user_shares = {row["Asset"]: row["Amount"] for _, row in portfolio_df.iterrows()}
                    res = dynamic_backtest_portfolio_user_fixed_shares(simulation_data, asset_amounts=user_shares)
                else:
                    res = dynamic_backtest_portfolio(simulation_data, method, lookback_days, rebalance_days, nonnegative_toggle)
                results_dict[method] = res

            downsampled = downsample_results_dict(results_dict, start_date, end_date)

            # Store in session
            st.session_state.backtest_results = results_dict
            st.session_state.downsampled = downsampled
            st.session_state.selected_methods = selected_methods

        except Exception as e:
            st.error("An error occurred during backtesting.")
            st.error(f"Details: {e}")

# --- Render Results ---
if st.session_state.downsampled:
    downsampled = st.session_state.downsampled
    selected_methods = st.session_state.selected_methods

    st.markdown("### 📊 Cumulative Returns")
    st.altair_chart(add_interactivity(plot_cumulative_returns(downsampled), x_field="date", y_field="cumulative"), use_container_width=True)

    st.markdown("### 📈 Rolling Annualized Sharpe Ratio")
    st.altair_chart(add_interactivity(plot_rolling_sharpe(downsampled), x_field="date", y_field="rolling_sharpe"), use_container_width=True)

    st.markdown("### 📉 Drawdowns")
    st.altair_chart(add_interactivity(plot_drawdowns(downsampled), x_field="date", y_field="drawdown"), use_container_width=True)

    st.markdown("### 🧩 Dynamic Asset Allocations")
    for method in selected_methods:
        with st.expander(f"{method} Allocations", expanded=False):
            st.altair_chart(
                add_interactivity(plot_allocations_per_method(downsampled[method]["allocations"], method), x_field="date", y_field="Allocation"),
                use_container_width=True
            )

    st.markdown("### 📌 Backtest Summary")
    st.write(f"**Backtest Period:** {start_date} to {end_date}")
    st.write(f"**Rebalance Frequency:** Every {rebalance_days} days")
    st.write(f"**Lookback Window:** {lookback_days} days")

    # Prepare summary data
    styled_table = generate_styled_summary_table(st.session_state.downsampled)
    st.markdown("### 📋 Performance Summary")
    st.dataframe(styled_table, use_container_width=True)


