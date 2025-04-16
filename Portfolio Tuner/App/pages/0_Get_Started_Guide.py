import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
from datetime import timedelta

from auth import login_and_get_status
from components.portfolio_input import edit_portfolio
from utils.utils import ensure_utc
from utils.plots import (
    plot_cumulative_returns,
    plot_single_gauge,
    pie_chart_allocation,
    generate_styled_summary_table,
    add_interactivity
)
from utils.simulation import run_smart_monte_carlo_simulation
from optimizer import run_optimizers
from utils.backtest import dynamic_backtest_portfolio, dynamic_backtest_portfolio_user_fixed_shares
from utils.utils import downsample_results_dict

# --- Config ---
st.set_page_config(page_title="Get Started Guide", layout="wide")
authenticator, authentication_status, username = login_and_get_status()

# --- Load data ---
@st.cache_data
def load_data():
    return pd.read_parquet("Portfolio Tuner/App/data/prices.parquet")

data = load_data()
available_assets = data.columns.tolist()

st.title("🚀 Get Started with Portfolio Tuner")
st.markdown("Welcome! In just a few steps, you'll go from entering your portfolio to optimizing it for better performance.")

# --- Step 1: Portfolio Input ---
st.header("📥 Step 1: Input Your Portfolio")
portfolio_df = edit_portfolio(available_assets, data, persistent=authentication_status)

if portfolio_df.empty or "Asset" not in portfolio_df.columns:
    st.warning("Please enter at least one asset to continue.")
    st.stop()

selected_assets = portfolio_df["Asset"].dropna().unique().tolist()

# --- Step 2: Date Range ---
st.header("📆 Step 2: Select Your Date Range")

max_date = data.index.max()
start_date = max_date - pd.Timedelta(days=90)
end_date = max_date
simulation_data = data[selected_assets].loc[start_date:end_date].copy()
simulation_data = simulation_data.replace(0, np.nan).ffill().dropna(how="any")

returns = simulation_data.pct_change().dropna()

# --- Step 3: Portfolio Metrics ---
st.header("📊 Step 3: Portfolio Overview (Past 90 Days)")

latest_prices = simulation_data.iloc[-1]
portfolio_df = portfolio_df.dropna(subset=["Asset", "Amount"])
values = portfolio_df.apply(lambda row: row["Amount"] * latest_prices.get(row["Asset"], 0), axis=1)
total_value = values.sum()

user_weights = {
    row["Asset"]: (row["Amount"] * latest_prices.get(row["Asset"], 0)) / total_value
    for _, row in portfolio_df.iterrows()
    if latest_prices.get(row["Asset"], 0) > 0
}

portfolio_returns = returns.dot(pd.Series(user_weights))
cumulative_returns = (1 + portfolio_returns).cumprod()
cumulative_return = cumulative_returns.iloc[-1] - 1
volatility = portfolio_returns.std() * np.sqrt(365)
sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(365)

col1, col2, col3 = st.columns(3)
with col1:
    fig = plot_single_gauge("Cumulative Return", cumulative_return * 100, metric_name="cumulative")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = plot_single_gauge("Annualized Volatility", volatility * 100, metric_name="volatility")
    st.plotly_chart(fig, use_container_width=True)

with col3:
    fig = plot_single_gauge("Sharpe Ratio", sharpe_ratio, metric_name="sharpe")
    st.plotly_chart(fig, use_container_width=True)

# --- Step 4: Strategy Backtest (Compact) ---
st.header("📈 Step 4: Backtest Strategies")

lookback_days = 90
rebalance_days = 30
nonnegative = True

# Add user portfolio to optimizer
lookback_df = data[selected_assets].loc[start_date - timedelta(days=lookback_days):start_date]
initial_allocs = run_optimizers(lookback_df, nonnegative_mvo=nonnegative)
initial_allocs["Your Portfolio"] = user_weights
strategies = list(initial_allocs.keys())

results = {}
for method in strategies:
    if method == "Your Portfolio":
        user_shares = {row["Asset"]: row["Amount"] for _, row in portfolio_df.iterrows()}
        res = dynamic_backtest_portfolio_user_fixed_shares(simulation_data, asset_amounts=user_shares)
    else:
        res = dynamic_backtest_portfolio(simulation_data, method, lookback_days, rebalance_days, nonnegative)
    results[method] = res

downsampled = downsample_results_dict(results, start_date, end_date)

st.altair_chart(add_interactivity(plot_cumulative_returns(downsampled), "date", "cumulative"), use_container_width=True)

# --- Step 5: Optimization Snapshot ---
st.header("🎯 Step 5: Strategy Suggestions (Today)")

st.markdown("See what each strategy recommends for your current portfolio mix:")

pie_charts = [pie_chart_allocation(pd.Series(initial_allocs[method]), method) for method in strategies]
st.altair_chart(alt.hconcat(*pie_charts), use_container_width=True)

# --- Step 6: Monte Carlo Simulation ---
st.header("🔮 Step 6: Future Simulation (Monte Carlo)")

if st.button("🚀 Run Future Simulation"):
    mc_result = run_smart_monte_carlo_simulation(
        user_weights, simulation_data,
        horizon_days=180, n_sims=100,
        correlation_strategy="shrinkage"
    )
    st.plotly_chart(mc_result["chart"], use_container_width=True)
    st.markdown(f"""
    **Forecast Summary:**
    - Median 6-Month Return: `{(mc_result['ci_high'] + mc_result['ci_low']) / 2:.1%}`
    - Best Case: `{mc_result['max']:.1%}`
    - Worst Case: `{mc_result['min']:.1%}`
    """)

# --- Footer: Next Steps ---
st.markdown("---")
st.success("🎉 You're done! Now dive deeper with the full tools below:")

links = {
    "📊 Dashboard": "pages/2_Portfolio_Dashboard.py",
    "🎯 Optimizer": "pages/3_Portfolio_Optimizer.py",
    "⏳ Backtest Lab": "pages/4_Backtest_Lab.py",
    "🎮 Playground": "pages/5_Playground.py"
}
cols = st.columns(len(links))
for i, (label, page_path) in enumerate(links.items()):
    with cols[i]:
        if st.button(f"Go to {label}"):
            st.switch_page(page_path)
