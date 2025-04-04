# pages/2_portfolio_optimizer.py

import streamlit as st
import pandas as pd
import altair as alt
import os
from datetime import timedelta

from auth import login_and_get_status
from utils.api_client import call_fastapi_optimizer
from utils.backtest import dynamic_backtest_portfolio, dynamic_backtest_portfolio_user
from user_input import get_backtest_settings, get_optimization_methods
from optimizer import run_optimizers 
from plots import (
    plot_asset_cumulative_returns,
    plot_cumulative_returns,  # ← this one is already built for result dicts
    plot_rolling_sharpe,
    plot_drawdowns,
    plot_allocations_per_method,
    plot_asset_returns,
    plot_asset_prices,
    pie_chart_allocation,
    add_interactivity,
    plot_historical_assets
)
from utils.utils import downsample_results_dict
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# Helper function to display narrative text with a light, primary-colored background.
def narrative(text):
    st.markdown(
        f"""<div style="background-color: rgba(31, 119, 180, 0.2); padding: 10px; border-left: 4px solid #1F77B4; font-size: 18px; margin-bottom: 10px;">
        {text}
        </div>""",
        unsafe_allow_html=True
    )
def run_optimizer_locally(price_df, asset_weights, lookback_days, nonnegative):
    lookback = price_df.tail(lookback_days)
    allocations = run_optimizers(lookback, nonnegative_mvo=nonnegative)
    return {method: alloc.to_dict() for method, alloc in allocations.items()}

# --- Auth ---
authenticator, authentication_status, username = login_and_get_status()
st.title("Portfolio Optimizer")
narrative("Welcome to the Portfolio Optimizer! Log in to access your saved portfolio or choose to build one temporarily. This tool helps you compare your portfolio against optimized strategies.")

# --- Load data ---
@st.cache_data
def load_data():
    return pd.read_parquet("Portfolio Tuner/App/data/prices.parquet")

data = load_data()
available_dates = data.index.sort_values()
available_assets = data.columns.tolist()
narrative("Historical price data for various assets has been loaded. This data will drive the performance simulation over time.")

# --- Portfolio Selection ---
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
    narrative("Build a temporary portfolio by selecting from the available assets.")
    from components.portfolio_input import edit_portfolio
    portfolio_df = edit_portfolio(available_assets, persistent=False)

if portfolio_df.empty or "Asset" not in portfolio_df.columns:
    st.warning("Your portfolio is empty. Please add assets.")
    st.stop()
narrative("Your portfolio is now set up and will determine the assets included in the simulation.")

# --- Backtest settings ---
start_date, end_date, lookback_days, rebalance_days, nonnegative_toggle = get_backtest_settings(available_dates)
narrative("Set your backtest parameters below, including the start and end dates, lookback period for analysis, and how often you want the portfolio rebalanced.")

# --- Filter simulation data ---
selected_assets = portfolio_df["Asset"].dropna().unique().tolist()
data = data[[col for col in selected_assets if col in data.columns]]
simulation_data = data.loc[start_date:end_date]

if simulation_data.empty:
    st.error("No data available for the selected backtest period.")
    st.stop()
narrative("The dataset has been filtered to match your portfolio and selected time period.")

# --- Plot Underlying Asset Data ---
st.markdown("## Underlying Asset Data")
narrative("Review the historical price charts of your selected assets to get a sense of their past performance.")
if "show_asset_plots" not in st.session_state:
    st.session_state.show_asset_plots = False
if st.button("📊 Plot Assets"):
    st.session_state.show_asset_plots = not st.session_state.show_asset_plots
if st.session_state.show_asset_plots:
    plot_historical_assets(simulation_data, selected_assets, portfolio_df=portfolio_df)

# --- Optimizer ---
st.markdown("## Dynamic Backtest Results")
narrative("Click the **Optimize Portfolio** button below to run simulations. This step compares your portfolio with other optimization methods to help you understand performance differences.")
optimize_button = st.button("Optimize Portfolio")

if optimize_button:
    try:
        # --- Begin Backtest Simulation ---
        # Prepare the lookback window used for calculating initial allocations.
        lookback_window = data.loc[pd.to_datetime(start_date) - pd.Timedelta(days=lookback_days):start_date]
        # Normalize user input weights (absolute amounts) into percentages.
        total = portfolio_df["Amount"].sum()
        user_weights = {row["Asset"]: row["Amount"] / total for _, row in portfolio_df.iterrows()}
        
        narrative("Processing simulation:\n\n"
                  "1. Normalizing your portfolio weights into percentages.\n"
                  "2. Calculating initial allocations using external optimization methods.\n"
                  "3. Including your own portfolio for direct comparison.")
        
        # Call external optimizer for other methods.
        initial_allocations = run_optimizer_locally(
            price_df=lookback_window[selected_assets],
            asset_weights=user_weights,
            lookback_days=lookback_days,
            nonnegative=nonnegative_toggle
        )
        # Add the user's portfolio.
        initial_allocations["User Portfolio"] = user_weights
        selected_methods = get_optimization_methods(initial_allocations)
        
        # --- Initial Allocations Display ---
        narrative("### Initial Allocations (Pie Charts)\nBelow are pie charts showing the initial asset allocations for each method, including your own portfolio.")
        pie_charts = [
            pie_chart_allocation(pd.Series(initial_allocations[method]).round(4), method)
            for method in selected_methods
        ]
        st.altair_chart(alt.hconcat(*pie_charts), use_container_width=True)
        
        # --- Run Backtest ---
        narrative("Running simulation: calculating key performance metrics (cumulative returns, rolling Sharpe ratios, and drawdowns) for each strategy.")
        results_dict = {}
        for method in selected_methods:
            if method == "User Portfolio":
                res = dynamic_backtest_portfolio_user(simulation_data, user_weights, lookback_days, rebalance_days, nonnegative_toggle)
            else:
                res = dynamic_backtest_portfolio(simulation_data, method, lookback_days, rebalance_days, nonnegative_toggle)
            results_dict[method] = res
        
        narrative("Simulation complete. Downsampling results for clearer visualization.")
        downsampled_results = downsample_results_dict(results_dict, start_date, end_date)
        
        # --- Cumulative Returns Chart ---
        narrative("### Cumulative Returns\nThis chart shows how each method's portfolio grows over time.")
        cumulative_chart = plot_cumulative_returns(downsampled_results)
        st.altair_chart(add_interactivity(cumulative_chart, x_field="date", y_field="cumulative"), use_container_width=True)
        
        # --- Rolling Sharpe Ratio Chart ---
        narrative("### Rolling Annualized Sharpe Ratio\nThe rolling Sharpe ratio chart illustrates the risk-adjusted performance of each method over time.")
        st.altair_chart(add_interactivity(plot_rolling_sharpe(downsampled_results), x_field="date", y_field="rolling_sharpe"), use_container_width=True)
        
        # --- Rolling Drawdown Chart ---
        narrative("### Rolling Maximum Drawdown\nThis chart shows the worst decline from a peak to a trough, helping you assess potential risk.")
        st.altair_chart(add_interactivity(plot_drawdowns(downsampled_results), x_field="date", y_field="drawdown"), use_container_width=True)
        
        # --- Dynamic Allocations Chart ---
        narrative("### Dynamic Asset Allocations Per Method\nThis chart displays how each strategy allocates assets over time. Note that your portfolio's allocations remain flat between rebalances.")
        for method in selected_methods:
            st.altair_chart(add_interactivity(plot_allocations_per_method(downsampled_results[method]["allocations"], method), x_field="date", y_field="allocation"), use_container_width=True)
        
        # --- Summary Metrics ---
        narrative("### Summary Metrics by Method\nBelow is a summary of the key performance metrics for each optimization method over the backtest period.")
        st.write(f"**Backtest period:** {pd.to_datetime(start_date).date()} to {pd.to_datetime(end_date).date()}")
        st.write(f"**Rebalance Frequency:** Every {rebalance_days} days")
        st.write(f"**Data Lookback Period:** {lookback_days} days")
        for method, res in downsampled_results.items():
            st.subheader(method)
            st.write(f"**Final Annualized Sharpe Ratio:** {res['sharpe']:.2f}")
            st.write(f"**Maximum Drawdown:** {res['drawdown']:.2%}")
        
        narrative("### Final Thoughts\nThis summary allows you to compare your portfolio with other optimization methods. Use this information to evaluate performance and potential risk. Thank you for using the Portfolio Optimizer!")
    except Exception as e:
        st.error("An error occurred during dynamic backtesting.")
        st.error(f"Details: {e}")
else:
    st.info("Click the 'Optimize Portfolio' button to run the backtest and view results.")
