import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Simulated imports for Portfolio Tuner components
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

# --- Custom CSS to Increase Font Size for Normal Text ---
st.markdown("""
<style>
/* Bump up normal text font size */
.stMarkdown p, .stMarkdown div, p {
    font-size: 18px !important;
    line-height: 1.5;
}

/* Restore proper sizes for headers */
h1 {
    font-size: 36px !important;
    font-weight: 700;
}
h2 {
    font-size: 28px !important;
    font-weight: 600;
}
h3 {
    font-size: 22px !important;
    font-weight: 600;
}

/* Preserve table/chart styling */
.stDataFrame, .stTable, .stPlotlyChart, .stAltairChart {
    font-size: inherit !important;
}

/* Optional: Larger narrative box text */
.narrative-box {
    font-size: 20px !important;
}
</style>
""", unsafe_allow_html=True)


# --- Style Helper ---
def narrative(text, color="#1F77B4"):
    st.markdown(
        f"""<div class="narrative-box" style="background-color: rgba(31, 119, 180, 0.1); padding: 10px; border-left: 4px solid {color}; margin-bottom: 10px;">
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
        "Amount": [0.05, 2.0, 10.0, 2.0]  # Example holdings: 0.05 BTC, 2 ETH, 10 SOL, 2 BNB
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

# Calculate portfolio values and allocations based on latest prices
max_date = data.index.max()
latest_prices = data.loc[max_date]
values = portfolio_df.apply(lambda row: row["Amount"] * latest_prices.get(row["Asset"], 0), axis=1)
total_value = values.sum()
portfolio_df["Value ($)"] = values
portfolio_df["Allocation (%)"] = (values / total_value * 100).round(2)

st.markdown("### Example Portfolio")
col1, col2 = st.columns([3, 2])  # Split layout for table and pie chart
with col1:
    st.dataframe(portfolio_df, use_container_width=True)
with col2:
    # Visualize allocations using plotly_pie_allocation
    weights = pd.Series(portfolio_df["Allocation (%)"].values, index=portfolio_df["Asset"])
    fig = plotly_pie_allocation(weights, title="📊 Portfolio Allocation", show_legend=True)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
This portfolio includes:
- 0.05 BTC
- 2 ETH
- 10 SOL
- 2 BNB

The table now shows the value of each holding in USD and its allocation as a percentage of the total portfolio. The pie chart visualizes these allocations for a clearer understanding. You can edit your own portfolio in the Portfolio Editor later, but for now, let's analyze this one.
""")

selected_assets = portfolio_df["Asset"].dropna().unique().tolist()

# --- Step 2: Date Range ---
st.header("📆 Step 2: Select Your Date Range")
narrative("Choose a time period to analyze your portfolio’s past performance. We’ll default to the last 90 days, but you can adjust the dates as needed.", color="#1F77B4")

# Ensure UTC-awareness for dates
max_date = ensure_utc(data.index.max())
default_start_date = max_date - pd.Timedelta(days=90)
default_end_date = max_date

col1, col2 = st.columns(2)
with col1:
    start_date_input = st.date_input("Start Date", value=default_start_date.date(), min_value=available_dates[0].date(), max_value=max_date.date())
with col2:
    end_date_input = st.date_input("End Date", value=default_end_date.date(), min_value=available_dates[0].date(), max_value=max_date.date())

# Convert to UTC-aware timestamps
start_date = ensure_utc(pd.Timestamp(datetime.combine(start_date_input, datetime.min.time())))
end_date = ensure_utc(pd.Timestamp(datetime.combine(end_date_input, datetime.min.time())))

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

simulation_data = data[selected_assets].loc[start_date:end_date].copy()
simulation_data = simulation_data.replace(0, np.nan).ffill().dropna(how="any")

returns = simulation_data.pct_change().dropna()
st.markdown(f"**Selected Period**: {start_date.date()} to {end_date.date()}")

# --- Step 3: Portfolio Metrics ---
st.header("📊 Step 3: Portfolio Overview (Selected Period)")
narrative(f"Let’s see how your portfolio performed from {start_date.date()} to {end_date.date()}. We’ll look at key metrics like cumulative return, volatility, and Sharpe Ratio to understand its risk and reward.", color="#1F77B4")

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
This chart shows how each strategy performed over the selected period:
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
st.success("🎉 Congratulations! You’ve Mastered the Basics of Portfolio Tuner!")

st.markdown("""
You’ve just explored the foundations of building, analyzing, and optimizing a crypto portfolio—now it's time to take full control of your strategy. Portfolio Tuner offers powerful tools to deepen your understanding and refine your investments. Here’s where to go next:
""")

links = {
    "🛠️ Portfolio Builder": {
        "path": "pages/1_Portfolio_Builder.py",
        "benefit": "Construct your portfolio by selecting assets and adjusting weights or amounts."
    },
    "📊 Tunerboard": {
        "path": "pages/2_Tunerboard.py",
        "benefit": "Monitor your portfolio’s performance in real-time with visuals and key metrics."
    },
    "🎯 Portfolio Tuner": {
        "path": "pages/3_Portfolio_Tuner.py",
        "benefit": "Apply optimization strategies like HRP and MVO to fine-tune your asset allocations."
    },
    "🧪 Strategy Sandbox": {
        "path": "pages/5_Strategy_Sandbox.py",
        "benefit": "Test allocation strategies side-by-side in a flexible experimentation space."
    },
    "📚 Tuner Glossary": {
        "path": "pages/6_Tuner_Glossary.py",
        "benefit": "Look up key crypto investing terms and portfolio theory concepts, right when you need them."
    }
}

cols = st.columns(len(links))
for i, (label, info) in enumerate(links.items()):
    with cols[i]:
        st.markdown(f"**{label}**: {info['benefit']}")
        if st.button(f"Go to {label} Now"):
            st.switch_page(info['path'])

st.markdown("""
Don't stop here—Portfolio Tuner is your companion for smarter, data-driven crypto investing. Jump into any tool above and start tuning your strategy today!
""")
