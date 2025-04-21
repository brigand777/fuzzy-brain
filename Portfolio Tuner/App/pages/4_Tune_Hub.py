import streamlit as st
import pandas as pd
import os
import pytz
from datetime import datetime
import plotly.express as px
import uuid

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
from plotly.colors import qualitative

# --- Page Setup ---
st.set_page_config(page_title="Crypto Portfolio Tuner", layout="wide")
authenticator, authentication_status, username = login_and_get_status()
st.title("💸 Crypto Portfolio Tuner")

# --- Style Helper ---
def narrative(text, color="#1F77B4"):
    st.markdown(
        f"""<div style="background-color: rgba(31, 119, 180, 0.1); padding: 10px; border-left: 4px solid {color}; font-size: 16px; margin-bottom: 10px;" aria-label="{text}">
        {text}
        </div>""",
        unsafe_allow_html=True
    )

# --- Load Data ---
@st.cache_resource
def load_data():
    return pd.read_parquet("Portfolio Tuner/App/data/prices.parquet")

with st.spinner("Loading crypto price data..."):
    try:
        data = load_data()
        available_assets = data.columns.tolist()
        available_dates = data.index.sort_values()
    except Exception as e:
        st.error(f"❌ Failed to load price data: {e}")
        st.stop()

# --- Welcome Banner ---
if "welcome_seen" not in st.session_state:
    with st.expander("Welcome to Crypto Portfolio Tuner! 🚀", expanded=True):
        st.markdown("1. Add cryptos → 2. Test performance → 3. Optimize → 4. Forecast")
        if st.button("Get Started", key="welcome"):
            st.session_state.welcome_seen = True

# --- Tabs for Workflow ---
tabs = st.tabs(["📁 Portfolio", "🧪 Backtest", "📈 Optimize", "🎲 Forecast"])

# --- Step 1: Portfolio Setup ---
with tabs[0]:
    st.markdown("### 📁 Build Your Crypto Portfolio")
    narrative("Add cryptocurrencies to analyze performance and get optimization tips.", color="#1F77B4")
    
    # Portfolio Health Snapshot
    if "portfolio_df" in st.session_state and not st.session_state.portfolio_df.empty:
        portfolio_df = st.session_state.portfolio_df
        latest_prices = data.loc[data.index.max()]
        values = portfolio_df.apply(lambda row: row["Amount"] * latest_prices.get(row["Asset"], 0), axis=1)
        total_value = values.sum()
        st.markdown("#### Portfolio Health")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Value", f"${total_value:,.2f}")
        col2.metric("Top Crypto", portfolio_df["Asset"].iloc[0])
        col3.metric("Diversification", f"{len(portfolio_df)} Coins")
    
    # Input Mode and Presets
    input_mode = st.radio("Portfolio Source", ["Use Saved Portfolio", "Build New Portfolio"], key="input_mode")
    preset = st.selectbox("Start with a Template", ["Custom", "Blue Chip", "DeFi"], key="preset")
    
    if input_mode == "Use Saved Portfolio":
        if authentication_status:
            portfolio_path = f"Portfolio Tuner/App/portfolios/{username}_portfolio.csv"
            if os.path.exists(portfolio_path):
                portfolio_df = pd.read_csv(portfolio_path)
                st.success("📂 Loaded saved portfolio.")
            else:
                st.warning("⚠️ No saved portfolio found. Switch to 'Build New Portfolio'.")
                st.stop()
        else:
            st.warning("🔐 Login required for saved portfolios.")
            st.stop()
    else:
        if preset == "Blue Chip":
            portfolio_df = pd.DataFrame({"Asset": ["BTC", "ETH"], "Amount": [1, 10]})
        elif preset == "DeFi":
            portfolio_df = pd.DataFrame({"Asset": ["LINK", "AAVE"], "Amount": [100, 5]})
        else:
            portfolio_df = edit_portfolio(available_assets, data, persistent=False)
    
    if portfolio_df.empty or "Asset" not in portfolio_df.columns:
        st.warning("🚫 Add cryptos to continue.")
        st.stop()
    
    # Calculate and Display Portfolio
    max_date = data.index.max()
    latest_prices = data.loc[max_date]
    values = portfolio_df.apply(lambda row: row["Amount"] * latest_prices.get(row["Asset"], 0), axis=1)
    total_value = values.sum()
    portfolio_df["Value ($)"] = values
    portfolio_df["Allocation (%)"] = (values / total_value * 100).round(2)
    st.session_state.portfolio_df = portfolio_df
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(portfolio_df, use_container_width=True, height=300)
        if len(portfolio_df) < 3:
            st.warning("⚠️ Add more coins for better diversification.")
    with col2:
        weights = pd.Series(portfolio_df["Allocation (%)"].values, index=portfolio_df["Asset"])
        fig = plotly_pie_allocation(weights, title="Portfolio Mix", show_legend=True)
        st.plotly_chart(fig, use_container_width=True)

# --- Step 2: Backtesting ---
with tabs[1]:
    st.markdown("### 🧪 Backtest Your Portfolio")
    narrative("Test how your portfolio would have performed historically.", color="#FF9900")
    
    # Date Range
    st.markdown("#### 📅 Select Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_naive = st.date_input("Start Date", value=available_dates[-252].date(), key="start_date")
    with col2:
        end_naive = st.date_input("End Date", value=available_dates[-1].date(), key="end_date")
    
    start_date = pd.Timestamp(datetime.combine(start_naive, datetime.min.time()), tz="UTC")
    end_date = pd.Timestamp(datetime.combine(end_naive, datetime.min.time()), tz="UTC")
    
    if start_date >= end_date:
        st.error("🚫 Start date must be before end date.")
        st.stop()
    
    # Strategies
    st.markdown("#### 🧠 Select Strategies")
    with st.expander("ℹ️ Strategy Info", expanded=False):
        st.markdown("""
        - **Equal Weight**: Equal investment in each crypto.
        - **Risk-Optimized**: Balances risk and return.
        - **Balanced Risk**: Groups assets by risk.
        - **Your Portfolio**: Your custom allocation.
        """)
    
    default_methods = ["Equal Weight", "Risk-Optimized", "Balanced Risk", "Your Portfolio"]
    selected_methods = st.multiselect("Strategies", default_methods, default=["Equal Weight", "Your Portfolio"], key="strategies")
    
    # Advanced Settings
    with st.expander("⚙️ Advanced Settings", expanded=False):
        rebalance_days = st.slider("Rebalance Frequency (days)", 7, 90, 30, step=7, key="rebalance")
        lookback_days = st.slider("Historical Data Range (days)", 30, 365, 90, step=30, key="lookback")
        nonnegative_toggle = st.toggle("No Short-Selling", value=True, key="no_short")
        if st.button("Reset to Recommended"):
            rebalance_days, lookback_days, nonnegative_toggle = 30, 90, True
    
    simulation_data = data.loc[start_date:end_date, [col for col in portfolio_df['Asset'] if col in data.columns]]
    
    if simulation_data.empty:
        st.error("❌ No data for selected period.")
        st.stop()
    
    if st.button("Run Backtest", key="run_backtest"):
        with st.spinner("Running backtest..."):
            try:
                latest_prices = data.iloc[-1]
                values = portfolio_df.apply(lambda row: row["Amount"] * latest_prices.get(row["Asset"], 0), axis=1)
                total_value = values.sum()
                user_weights = {
                    row["Asset"]: (row["Amount"] * latest_prices.get(row["Asset"], 0)) / total_value
                    for _, row in portfolio_df.iterrows() if latest_prices.get(row["Asset"], 0) > 0
                }
                lookback_window = data.loc[pd.to_datetime(start_date) - pd.Timedelta(days=lookback_days):start_date]
                initial_allocations = run_optimizers(lookback_window[user_weights.keys()], nonnegative_mvo=nonnegative_toggle)
                initial_allocations["Your Portfolio"] = user_weights
                
                results_dict = {}
                for method in selected_methods:
                    method_key = "Risk-Optimized" if method == "Mean Variance" else "Balanced Risk" if method == "HRB" else method
                    if method_key == "Your Portfolio":
                        user_shares = {row["Asset"]: row["Amount"] for _, row in portfolio_df.iterrows()}
                        res = dynamic_backtest_portfolio_user_fixed_shares(simulation_data, asset_amounts=user_shares)
                    else:
                        res = dynamic_backtest_portfolio(simulation_data, method_key, lookback_days, rebalance_days, nonnegative_toggle)
                    results_dict[method_key] = res
                
                downsampled = downsample_results_dict(results_dict, start_date, end_date)
                st.session_state.backtest_results = results_dict
                st.session_state.downsampled = downsampled
                st.session_state.selected_methods = selected_methods
                st.success("✅ Backtest complete!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    if "downsampled" in st.session_state:
        st.markdown("#### 📊 Backtest Results")
        downsampled = st.session_state.downsampled
        selected_methods = st.session_state.selected_methods
        
        # Interactive Dashboard
        fig = px.line(downsampled, x="date", y="cumulative", color="strategy", title="Portfolio Performance")
        fig.update_layout(updatemenus=[{
            "buttons": [
                {"label": "Returns", "method": "update", "args": [{"y": [downsampled["cumulative"]]}]},
                {"label": "Sharpe Ratio", "method": "update", "args": [{"y": [downsampled["rolling_sharpe"]]}]},
                {"label": "Drawdowns", "method": "update", "args": [{"y": [downsampled["drawdown"]]}]}
            ]
        }])
        st.plotly_chart(fig, use_container_width=True)
        
        for method in selected_methods:
            with st.expander(f"{method} Allocations"):
                st.altair_chart(
                    add_interactivity(plot_allocations_per_method(downsampled[method]["allocations"], method), x_field="date", y_field="Allocation"),
                    use_container_width=True
                )
        
        st.dataframe(generate_styled_summary_table(downsampled), use_container_width=True)
        
        if st.button("Share Results", key="share_backtest"):
            st.write(f"My crypto portfolio returned {downsampled['cumulative'].iloc[-1]:.2%}! #CryptoInvesting")

# --- Step 3: Optimizer ---
with tabs[2]:
    st.markdown("### 📈 Optimize Your Portfolio")
    narrative("Get today's recommended allocations based on recent trends.", color="#4CAF50")
    
    lookback = st.selectbox("Historical Data Range", [30, 60, 90, 180, 365], index=2, key="opt_lookback")
    if st.button("Run Optimizer", key="run_optimizer"):
        try:
            latest_prices = data.iloc[-1]
            values = portfolio_df.apply(lambda row: row["Amount"] * latest_prices.get(row["Asset"], 0), axis=1)
            total_value = values.sum()
            user_weights = {
                row["Asset"]: (row["Amount"] * latest_prices.get(row["Asset"], 0)) / total_value
                for _, row in portfolio_df.iterrows() if latest_prices.get(row["Asset"], 0) > 0
            }
            lookback_df = data[user_weights.keys()].tail(lookback)
            all_allocations = run_optimizers(lookback_df, nonnegative_mvo=True)
            all_allocations["Your Portfolio"] = pd.Series(user_weights)
            
            st.session_state.optimizer_allocations = all_allocations
            st.session_state.optimizer_methods = selected_methods
            st.success("✅ Optimization complete!")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    if "optimizer_allocations" in st.session_state:
        strategies = st.session_state.optimizer_methods
        all_allocations = st.session_state.optimizer_allocations
        
        all_assets = set()
        for w in all_allocations.values():
            all_assets.update(w.index)
        asset_list = sorted(all_assets)
        palette = qualitative.Safe
        asset_color_map = {asset: palette[i % len(palette)] for i, asset in enumerate(asset_list)}
        
        chart_type = st.radio("Visualization", ["Pie Charts", "Bar Charts"], horizontal=True, key="chart_type")
        
        if chart_type == "Pie Charts":
            st.markdown("#### Investment Mix")
            pie_cols = st.columns(len(strategies))
            for i, method in enumerate(strategies):
                weights = all_allocations[method]
                fig = plotly_pie_allocation(weights, title=f"{method}", show_legend=False, color_map=asset_color_map)
                with pie_cols[i]:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("#### Allocation Comparison")
            bar_cols = st.columns(2)
            global_max = max(w.max() for w in all_allocations.values()) * 1.1
            x_order = asset_list
            for i, method in enumerate(strategies):
                weights = all_allocations[method]
                fig = plotly_bar_allocation(weights, title=f"{method}", yaxis_max=global_max, x_order=x_order, color_map=asset_color_map)
                with bar_cols[i % 2]:
                    st.plotly_chart(fig, use_container_width=True)

# --- Step 4: Monte Carlo Simulation ---
with tabs[3]:
    st.markdown("### 🎲 Monte Carlo Forecast")
    narrative("Simulate future performance under random market conditions.", color="#9C27B0")
    
    with st.expander("🛠️ Settings", expanded=False):
        horizon_days = st.slider("Forecast Horizon (days)", 30, 365, 180, step=30, key="horizon")
        n_sims = st.slider("Simulations", 100, 2000, 500, step=100, key="n_sims")
        corr_mode = st.selectbox("Correlation", ["shrinkage", "historical", "independent"], key="corr_mode")
        dynamic_rebal_toggle = st.toggle("Enable Rebalancing", value=False, key="rebal_toggle")
        rebalance_interval_mc = st.slider("Rebalance Interval (days)", 10, 90, 30, step=10, disabled=not dynamic_rebal_toggle, key="rebal_interval")
    
    if st.button("Run Simulation", key="run_mc"):
        try:
            from fitter import Fitter
            from scipy.stats import norm, t, johnsonsu
            from sklearn.covariance import LedoitWolf
            from utils.simulation import run_monte_carlo_multi_strategy, run_monte_carlo_with_rebalancing
            
            if "optimizer_allocations" not in st.session_state:
                st.warning("⚠️ Run optimizer first.")
            else:
                strategies = {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in st.session_state.optimizer_allocations.items()}
                mc_data = data[[asset for asset in portfolio_df["Asset"] if asset in data.columns]].dropna()
                
                if dynamic_rebal_toggle:
                    mc_result = run_monte_carlo_with_rebalancing(
                        strategies=strategies, price_data=mc_data, horizon_days=horizon_days, n_sims=n_sims,
                        rebalance_interval=rebalance_interval_mc, correlation_strategy=corr_mode
                    )
                else:
                    mc_result = run_monte_carlo_multi_strategy(
                        strategies=strategies, price_data=mc_data, horizon_days=horizon_days, n_sims=n_sims,
                        correlation_strategy=corr_mode
                    )
                
                st.plotly_chart(mc_result["chart"], use_container_width=True)
                st.markdown("#### Risk Metrics")
                st.plotly_chart(mc_result["metric_plot"], use_container_width=True)
                
                summary_df = pd.DataFrame(mc_result["summary"]).T[["sharpe", "volatility", "max_drawdown"]]
                best_values = {"sharpe": summary_df["sharpe"].max(), "volatility": summary_df["volatility"].min(), "max_drawdown": summary_df["max_drawdown"].max()}
                worst_values = {"sharpe": summary_df["sharpe"].min(), "volatility": summary_df["volatility"].max(), "max_drawdown": summary_df["max_drawdown"].min()}
                
                def color_text(val, col):
                    if pd.isna(val):
                        return ""
                    if val == best_values[col]:
                        return "color: green;"
                    elif val == worst_values[col]:
                        return "color: red;"
                    return ""
                
                styled_df = summary_df.style.apply(lambda row: [color_text(row[col], col) for col in summary_df.columns], axis=1).format("{:.2f}")
                st.markdown("#### Summary")
                st.dataframe(styled_df, use_container_width=True)
                
                if st.button("Share Forecast", key="share_mc"):
                    st.write(f"Forecasted my crypto portfolio with {n_sims} simulations! #CryptoInvesting")
        except Exception as e:
            st.error(f"❌ Error: {e}")