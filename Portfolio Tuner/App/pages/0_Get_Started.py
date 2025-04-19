import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Simulated imports for Portfolio Tuner components (assumed functionality based on provided code)
from auth import login_and_get_status
from components.portfolio_input import edit_portfolio
from utils.utils import ensure_utc, downsample_results_dict
from utils.plots import (
    plot_cumulative_returns, plot_single_gauge, pie_chart_allocation,
    generate_styled_summary_table, add_interactivity, plotly_pie_allocation,
    plotly_bar_allocation, plot_rolling_sharpe, plot_drawdowns, plot_allocations_per_method
)
from utils.simulation import run_smart_monte_carlo_simulation
from optimizer import run_optimizers
from utils.backtest import dynamic_backtest_portfolio, dynamic_backtest_portfolio_user_fixed_shares

# --- Config ---
st.set_page_config(page_title="Portfolio Tuner: Getting Started Guide", layout="wide")
authenticator, authentication_status, username = login_and_get_status()

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
    # Simulated data loading (assumed prices.parquet contains historical crypto prices)
    return pd.read_parquet("Portfolio Tuner/App/data/prices.parquet")

data = load_data()
available_assets = data.columns.tolist()
available_dates = data.index.sort_values()

# --- Predefined Example Portfolio ---
def create_example_portfolio():
    return pd.DataFrame({
        "Asset": ["BTC", "ETH", "SOL", "BNB"],
        "Amount": [0.5, 2.0, 50.0, 1.0]  # Example holdings: 0.5 BTC, 2 ETH, 50 SOL, 1 BNB
    })

# --- Getting Started Guide ---
st.title("🚀 Getting Started with Portfolio Tuner: A Step-by-Step Guide")
st.markdown("""
Welcome to **Portfolio Tuner**, the ultimate tool for crypto investors! Whether you're a beginner or a seasoned trader, this guide will walk you through analyzing, backtesting, optimizing, and forecasting your cryptocurrency portfolio using a predefined example portfolio. Let's dive in!
""")

# --- Step 1: Portfolio Setup ---
st.header("📥 Step 1: Set Up Your Portfolio")
narrative("Portfolio Tuner lets you input your crypto holdings to analyze their performance. For this guide, we'll use a predefined portfolio with Bitcoin (BTC), Ethereum (ETH), Solana (SOL), and Binance Coin (BNB).", color="#1F77B4")

portfolio_df = create_example_portfolio()
st.markdown("### Example Portfolio")
st.dataframe(portfolio_df, use_container_width=True)
st.markdown("""
This portfolio includes:
- 0.5 BTC
- 2 ETH
- 50 SOL
- 1 BNB

You can edit your own portfolio in the Portfolio Editor later, but for now, let's analyze this one.
""")

selected_assets = portfolio_df["Asset"].dropna().unique().tolist()

# --- Step 2: Date Range ---
st.header("📆 Step 2: Select Your Date Range")
narrative("Choose a time period to analyze your portfolio’s past performance. We’ll use the last 90 days to see how this portfolio performed.", color="#1F77B4")

max_date = data.index.max()
start_date = max_date - pd.Timedelta(days=90)
end_date = max_date
simulation_data = data[selected_assets].loc[start_date:end_date].copy()
simulation_data = simulation_data.replace(0, np.nan).ffill().dropna(how="any")

returns = simulation_data.pct_change().dropna()
st.markdown(f"**Selected Period**: {start_date.date()} to {end_date.date()}")

# --- Step 3: Portfolio Metrics ---
st.header("📊 Step 3: Portfolio Overview (Past 90 Days)")
narrative("Let’s see how your portfolio performed over the past 90 days. We’ll look at key metrics like cumulative return, volatility, and Sharpe Ratio to understand its risk and reward.", color="#1F77B4")

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
    st.markdown("*Cumulative Return*: The total percentage gain or loss over the period.")

with col2:
    fig = plot_single_gauge("Annualized Volatility", volatility * 100, metric_name="volatility")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("*Volatility*: Measures the price swings of your portfolio (higher means more risk).")

with col3:
    fig = plot_single_gauge("Sharpe Ratio", sharpe_ratio, metric_name="sharpe")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("*Sharpe Ratio*: A measure of risk-adjusted return (higher is better).")

st.markdown("""
These metrics give you a quick snapshot of your portfolio’s performance. A positive cumulative return means growth, while volatility and Sharpe Ratio help you assess risk. Portfolio Tuner’s intuitive gauges make it easy to understand these complex concepts!
""")

# --- Step 4: Strategy Backtest ---
st.header("📈 Step 4: Backtest Strategies")
narrative("Backtesting lets you see how your portfolio would have performed using different strategies. We’ll compare your portfolio against optimized strategies like Equal Weight, Mean-Variance Optimization (MVO), and Hierarchical Risk Parity (HRP).", color="#1F77B4")

lookback_days = 90
rebalance_days = 30
nonnegative = True

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

st.markdown("### Cumulative Returns Over Time")
st.altair_chart(add_interactivity(plot_cumulative_returns(downsampled), "date", "cumulative"), use_container_width=True)
st.markdown("""
This chart shows how each strategy performed over the past 90 days:
- **Equal Weight**: Allocates equally to each asset.
- **Mean-Variance Optimization (MVO)**: Optimizes for the best risk-return balance.
- **Hierarchical Risk Parity (HRP)**: Balances risk across assets based on their correlations.
- **Your Portfolio**: Your current holdings.

Backtesting helps you see which strategies might improve your portfolio’s performance. Portfolio Tuner makes this complex analysis simple!
""")

# --- Step 5: Optimization Snapshot ---
st.header("🎯 Step 5: Optimize Your Portfolio (Today’s Recommendations)")
narrative("Now let’s optimize your portfolio based on recent price trends. Portfolio Tuner suggests allocations to maximize returns while managing risk.", color="#4CAF50")

lookback = 90
lookback_df = data[user_weights.keys()].tail(lookback)
all_allocations = run_optimizers(lookback_df, nonnegative_mvo=True)
all_allocations["Your Portfolio"] = pd.Series(user_weights)

st.markdown("### Portfolio Allocations by Strategy")
pie_charts = [pie_chart_allocation(pd.Series(all_allocations[method]), method) for method in strategies]
st.altair_chart(alt.hconcat(*pie_charts), use_container_width=True)

st.markdown("""
These pie charts show how each strategy would allocate your portfolio today:
- **Your Portfolio**: Your current mix.
- **Equal Weight**, **MVO**, and **HRP**: Optimized suggestions.

Optimizing your portfolio can help you achieve better returns with lower risk. Portfolio Tuner’s advanced algorithms do the heavy lifting for you!
""")

# --- Step 6: Monte Carlo Simulation ---
st.header("🔮 Step 6: Forecast the Future (Monte Carlo Simulation)")
narrative("What might your portfolio look like in the future? Monte Carlo simulations run thousands of scenarios to predict potential outcomes, helping you plan with confidence.", color="#9C27B0")

horizon_days = 180
n_sims = 500
mc_result = run_smart_monte_carlo_simulation(
    user_weights, simulation_data,
    horizon_days=horizon_days, n_sims=n_sims,
    correlation_strategy="shrinkage"
)

st.plotly_chart(mc_result["chart"], use_container_width=True)
st.markdown(f"""
### Forecast Summary (Next 6 Months)
- **Median 6-Month Return**: `{((mc_result['ci_high'] + mc_result['ci_low']) / 2):.1%}`
- **Best Case**: `{mc_result['max']:.1%}`
- **Worst Case**: `{mc_result['min']:.1%}`
""")
st.markdown("""
This forecast shows the range of possible outcomes for your portfolio over the next 6 months. The median return is the expected midpoint, while the best and worst cases show the potential highs and lows. Portfolio Tuner’s simulations help you prepare for the future!
""")

# --- Footer: Next Steps ---
st.markdown("---")
st.success("🎉 You’ve Completed the Guide! Now Explore More with Portfolio Tuner")
st.markdown("""
You’ve just analyzed, backtested, optimized, and forecasted a crypto portfolio! Portfolio Tuner empowers you with advanced tools to make smarter investment decisions. Ready to dive deeper? Check out these features:
""")

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

st.markdown("""
Portfolio Tuner is here to help you navigate the volatile world of crypto investing with confidence. Start exploring your own portfolio today!
""")