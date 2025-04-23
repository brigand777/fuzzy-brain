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
from utils.glossary import chart_with_tooltip, add_info_icon, section_heading, inject_tooltip_css

# --- Page Setup ---
st.set_page_config(page_title="All-in-One Portfolio Tool", layout="wide")
inject_tooltip_css()
authenticator, authentication_status, username = login_and_get_status()
st.title("💼 Portfolio Tuner")
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,400;1,400&display=swap" rel="stylesheet">
<style>
    .metric-card {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)
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
#st.markdown("## Step 1: 📁 Select Portfolio")
render_final_tooltip_heading("Step 1: 📁 Select Portfolio", short_description="We can either pick what we built in Portfolio Builder or make something entirely new..yippie!", level=2)

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

# Calculate portfolio values and allocations based on latest prices
max_date = data.index.max()
latest_prices = data.loc[max_date]
values = portfolio_df.apply(lambda row: row["Amount"] * latest_prices.get(row["Asset"], 0), axis=1)
total_value = values.sum()
portfolio_df["Value ($)"] = values
portfolio_df["Allocation (%)"] = (values / total_value * 100).round(2)

col1, col2 = st.columns([2, 3])  # Split layout for table and pie chart
with col1:
    st.dataframe(portfolio_df, use_container_width=True)
with col2:
    # Visualize allocations using plotly_pie_allocation
    weights = pd.Series(portfolio_df["Allocation (%)"].values, index=portfolio_df["Asset"])
    fig = plotly_pie_allocation(weights, title="📊 Portfolio Allocation", show_legend=True)
    st.plotly_chart(fig, use_container_width=True)

# ------------------- BACKTESTING SECTION -------------------
# Step 2: Define Backtest Period
section_heading("Step 2: 🧪 Backtest Your Portfolio", short_description="'Past performance is not an indication of future returns'..but it doesn't hurt knowing", level=3)
 
st.markdown("### 📅 Step 2.1: Select Backtest Date Range")
col1, col2 = st.columns(2)

with col1:
    start_naive = st.date_input("Start Date", value=available_dates[-252].date())

with col2:
    end_naive = st.date_input("End Date", value=available_dates[-1].date())

start_date = pd.Timestamp(datetime.combine(start_naive, datetime.min.time()), tz="UTC")
end_date = pd.Timestamp(datetime.combine(end_naive, datetime.min.time()), tz="UTC")

if start_date >= end_date:
    st.error("🚫 Start date must be before end date.")
    st.stop()

# Step 2.2: Choose Strategy Types
section_heading("🧠 Step 2.2: Choose Strategies to Backtest", short_description="Oooh this is the technical stuff, Mean variance is optimal in the past, while HRB tries to find the structure and spread your risk mathematically..hard to chose I love looking at them all :p", level=3)

with st.expander("ℹ️ Strategy Descriptions", expanded=False):
    st.markdown("""
    - **Equal Weight**: Equal investment in each asset.
    - **Mean Variance (Smart Spread)**: Risk-efficient via MVO.
    - **HRB**: Hierarchical risk-based grouping.
    - **User Portfolio**: Your custom allocation.
    """)

default_methods = ["Equal Weight", "Mean Variance", "HRB", "User Portfolio"]
selected_methods = st.multiselect(
    "Select strategies to include:",
    options=default_methods,
    default=default_methods
)

# Step 2.3: Configure Advanced Settings
section_heading("⚙️ Step 2.3: Advanced Settings (Optional)", short_description="""This is the real geeky stuff..don't say I didn't warn ya! Rebalance frequency is how often you want to
 recalculate the strategy, lookback is how nearsighted (looking back) you want the strategy to optimize for, and not allowing short selling means you only buy (rather than sell) as part of a strategy""", level=3)

with st.expander("🔧 Show Advanced Settings", expanded=False):
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
    #cum_fig = add_interactivity(plot_cumulative_returns(downsampled), x_field="date", y_field="cumulative")
    #sha_fig = add_interactivity(plot_rolling_sharpe(downsampled), x_field="date", y_field="rolling_sharpe")
    #dra_fig = add_interactivity(plot_drawdowns(downsampled), x_field="date", y_field="drawdown")
    chart_with_tooltip(
        title="📈 Cumulative Returns",
        short_desc="Tracks the portfolio value over time relative to its starting point.",
        chart_func=lambda: plot_cumulative_returns(downsampled),
        term="Cumulative Return",
        interactive=True,
        x_field="date",
        y_field="cumulative"
    )


    chart_with_tooltip(
        title="📈 Rolling Sharpe Ratio",
        short_desc="Sharpe Ratio over a moving window. Shows changing risk-adjusted performance.",
        chart_func=lambda: plot_rolling_sharpe(downsampled),
        term="Rolling Sharpe Ratio",
        interactive=True,
        x_field="date",
        y_field="rolling_sharpe",
    )
    chart_with_tooltip(
        title="📉 Drawdowns",
        short_desc="Maximum dips from previous highs — a measure of worst-case losses.",
        chart_func=lambda: plot_drawdowns(downsampled),
        term="Max Drawdown",
        interactive=True,
        x_field="date",
        y_field="drawdown",
    )
    
    with st.expander(f"Strategy Allocations Over Time"):
        for method in selected_methods:
            st.altair_chart(
                add_interactivity(plot_allocations_per_method(downsampled[method]["allocations"], method), x_field="date", y_field="Allocation"),
                use_container_width=True
            )
    section_heading("📋 Summary Chart", short_description="""Check out which strategies performed best (green) and worst (red). If you want to leverage trade max drawdown is critical, if you are risk averse prioritize volatility, if you want to HODL prioritize sharpe.""", level=3)

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
    strategies = st.session_state.optimizer_methods
    all_allocations = st.session_state.optimizer_allocations

    # Collect all unique assets
    all_assets = set()
    for w in all_allocations.values():
        all_assets.update(w.index)
    asset_list = sorted(all_assets)

    # Define color map (professional, colorblind-safe)
    from plotly.colors import qualitative
    palette = qualitative.Safe
    asset_color_map = {asset: palette[i % len(palette)] for i, asset in enumerate(asset_list)}

    # --- Toggle ---
    chart_type = st.radio("Choose how to visualize allocations:", ["🥧 Pie Charts", "📈 Bar Charts"], horizontal=True)

    if chart_type == "🥧 Pie Charts":
        st.markdown("### 🥧 Investment Mix by Strategy")
        pie_cols = st.columns(len(strategies))
        for i, method in enumerate(strategies):
            weights = all_allocations[method]
            fig = plotly_pie_allocation(
                weights,
                title=f"{method} Allocation",
                show_legend=False,
                color_map=asset_color_map
            )
            with pie_cols[i]:
                st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "📈 Bar Charts":
        st.markdown("### 📈 Allocation Comparison by Strategy")
        bar_cols = st.columns(2)

        # Shared Y-axis and asset order
        global_max = max(w.max() for w in all_allocations.values()) * 1.1
        x_order = asset_list

        for i, method in enumerate(strategies):
            weights = all_allocations[method]
            fig = plotly_bar_allocation(
                weights,
                title=f"{method} Allocation Breakdown",
                yaxis_max=global_max,
                x_order=x_order,
                color_map=asset_color_map
            )
            with bar_cols[i % 2]:
                st.plotly_chart(fig, use_container_width=True)


# ------------------- MONTE CARLO SIMULATION SECTION -------------------

st.markdown("## 🎲 Monte Carlo Forecast (Multi-Strategy)")
narrative("Simulate future portfolio performance under random market conditions.", color="#9C27B0")



with st.expander("🛠️ Simulation Settings"):
    horizon_days = st.slider("⏳ Forecast Horizon (days)", 30, 365, 180, step=30)
    n_sims = st.slider("🎯 Number of Simulations", 100, 2000, 500, step=100)
    corr_mode = st.selectbox("📊 Correlation Assumption", ["shrinkage", "historical", "independent"], index=0)
    
    dynamic_rebal_toggle = st.toggle("Enable Rebalancing in MC Simulation", value=False)
    rebalance_interval_mc = st.slider("🔁 Rebalance Interval (days)", 10, 90, 30, step=10, disabled=not dynamic_rebal_toggle)

run_mc = st.button("Run Monte Carlo Simulation")

if run_mc:
    try:
        import numpy as np
        from fitter import Fitter
        from scipy.stats import norm, t, johnsonsu
        from sklearn.covariance import LedoitWolf
        import plotly.graph_objects as go
        from utils.simulation import run_monte_carlo_multi_strategy, run_monte_carlo_with_rebalancing

        if "optimizer_allocations" not in st.session_state:
            st.warning("⚠️ Run the optimizer first to generate strategies for simulation.")
        else:
            strategies = {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in st.session_state.optimizer_allocations.items()
            }
            mc_data = data[[asset for asset in portfolio_df["Asset"] if asset in data.columns]].dropna()

            if dynamic_rebal_toggle:
                mc_result = run_monte_carlo_with_rebalancing(
                    strategies=strategies,
                    price_data=mc_data,
                    horizon_days=horizon_days,
                    n_sims=n_sims,
                    rebalance_interval=rebalance_interval_mc,
                    correlation_strategy=corr_mode
                )
            else:
                mc_result = run_monte_carlo_multi_strategy(
                    strategies=strategies,
                    price_data=mc_data,
                    horizon_days=horizon_days,
                    n_sims=n_sims,
                    correlation_strategy=corr_mode
                )

            #st.plotly_chart(mc_result["chart"], use_container_width=True)
            chart_with_tooltip(
                title="🎲 Monte Carlo Projection",
                short_desc="Simulates thousands of future price paths based on asset statistics and strategy weights.",
                chart_func=lambda: mc_result["chart"],
            )

            #st.markdown("### 📊 Strategy Risk Metrics")
            #st.plotly_chart(mc_result["metric_plot"], use_container_width=True)
            chart_with_tooltip(
                title="📊 Strategy Risk Metrics",
                short_desc="Shows volatility, Sharpe ratio, and drawdowns across simulations.",
                chart_func=lambda: mc_result["metric_plot"],
                term="Sharpe Ratio",
                glossary_url="#sharpe-ratio"
            )

            summary_df = pd.DataFrame(mc_result["summary"]).T[
                ["sharpe", "volatility", "max_drawdown"]
            ]

            # Identify best and worst values (before rounding)
            best_values = {
                "sharpe": summary_df["sharpe"].max(),
                "volatility": summary_df["volatility"].min(),
                "max_drawdown": summary_df["max_drawdown"].max()
            }
            worst_values = {
                "sharpe": summary_df["sharpe"].min(),
                "volatility": summary_df["volatility"].max(),
                "max_drawdown": summary_df["max_drawdown"].min()
            }

            # Style function (color text only)
            def color_text(val, col):
                if pd.isna(val):
                    return ""
                if val == best_values[col]:
                    return "color: green;"
                elif val == worst_values[col]:
                    return "color: red;"
                return ""

            # Apply styling and format display
            styled_df = summary_df.style\
                .apply(lambda row: [color_text(row[col], col) for col in summary_df.columns], axis=1)\
                .format("{:.2f}")  # ✅ Round display values

            #st.markdown("### 📋 Simulation Summary (Median Sharpe, Volatility, Drawdown)")
            section_heading("📋 Simulation Summary", short_description="""Check out which strategies performed best (green) and worst (red). If you want to leverage trade max drawdown is critical, if you are risk averse prioritize volatility, if you want to HODL prioritize sharpe.""", level=3)

            st.dataframe(styled_df, use_container_width=True)





    except Exception as e:
        st.error("❌ Monte Carlo simulation failed.")
        st.error(f"Details: {e}")
