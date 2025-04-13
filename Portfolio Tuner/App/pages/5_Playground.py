import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from utils.plots import plot_cumulative_returns, add_interactivity, plot_single_gauge
from optimizer import run_optimizers
from utils.simulation import run_smart_monte_carlo_simulation

st.set_page_config(page_title="Playground", layout="wide")
st.title("🎮 Portfolio Playground")
st.markdown("Tinker with weights, simulate outcomes, and see how your ideas play out!")

# --- Load data ---
@st.cache_data
def load_data():
    return pd.read_parquet("Portfolio Tuner/App/data/prices.parquet")

data = load_data()
available_assets = data.columns.tolist()

# --- Asset selection ---
st.markdown("### 📂 Select Assets to Include")
selected_assets = st.multiselect(
    "Choose up to 10 assets to simulate:", 
    options=available_assets, 
    default=available_assets[:5],
    max_selections=10
)

if not selected_assets:
    st.warning("Please select at least one asset to proceed.")
    st.stop()

playground_assets = selected_assets
latest_prices = data.iloc[-1]

# --- Weight sliders and chart side-by-side ---
st.markdown("## 🪰 Adjust Your Hypothetical Portfolio")
slider_col, chart_col = st.columns([1, 2])

with slider_col:
    weights = {}
    total_weight = 0
    for asset in playground_assets:
        weight = st.slider(f"{asset} weight", 0.0, 1.0, 0.05, 0.005, key=asset)
        weights[asset] = weight
        total_weight += weight

weights = {k: v / total_weight for k, v in weights.items() if total_weight > 0}

risk_score = sum((v * np.std(data[k].pct_change())) for k, v in weights.items())

lookback_days = 365
simulation_data = data[playground_assets].tail(lookback_days)
pct_returns = simulation_data.pct_change().dropna()

portfolio_returns = pct_returns.dot(pd.Series(weights))
cumulative_returns = (1 + portfolio_returns).cumprod()
cumulative_df = pd.DataFrame({
    "date": simulation_data.index[-len(cumulative_returns):],
    "cumulative": cumulative_returns
})

mean_daily_return = portfolio_returns.mean()
volatility = portfolio_returns.std()
sharpe_ratio = (mean_daily_return / volatility)* np.sqrt(365.0) if volatility > 0 else 0

cumulative_return = cumulative_returns.iloc[-1] - 1
annualized_volatility = volatility * np.sqrt(365.0)

def styled_percent(value, red_if_high=False, threshold=0.2):
    pct = value * 100
    if red_if_high:
        color = "red" if value > threshold else "green"
    else:
        color = "green" if pct >= 0 else "red"
    return f"<span style='color:{color}'>{pct:.2f}%</span>"

def styled_number(value):
    color = "green" if value >= 0 else "red"
    return f"<span style='color:{color}'>{value:.2f}</span>"

with chart_col:
    #st.subheader("📊 Cumulative Returns")
    chart = plot_cumulative_returns({
        "Playground Portfolio": {
            "cumulative": cumulative_returns
        }
    }, show_legend = False)
    st.altair_chart(add_interactivity(chart, x_field="date", y_field="cumulative"), use_container_width=True)

    if risk_score > 0.05:
        st.markdown("**Portfolio Risk Level:** 🔥 High")
    elif risk_score > 0.03:
        st.markdown("**Portfolio Risk Level:** ⚠️ Medium")
    else:
        st.markdown("**Portfolio Risk Level:** 🧣 Low")

    #st.markdown("### 📊 Portfolio Stats (Past 365 Days)")
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = plot_single_gauge(
            title="Cumulative Return",
            value=cumulative_return ,  # convert to percent
            metric_name="cumulative"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = plot_single_gauge(
            title="Annualized Volatility",
            value=annualized_volatility,  # convert to percent
            metric_name="volatility"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = plot_single_gauge(
            title="Sharpe Ratio",
            value=sharpe_ratio,
            metric_name="sharpe"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Optional Monte Carlo Simulation ---
st.markdown("## 🔮 Monte Carlo Future Simulator")

if st.button("🔮 Run Smart Monte Carlo Simulation"):
    result = run_smart_monte_carlo_simulation(weights, data[playground_assets])
    st.plotly_chart(result["chart"], use_container_width=True)

    st.markdown("### 📈 Distribution Used per Asset")
    for asset, dist in result["distribution_used_per_asset"].items():
        st.markdown(f"- **{asset}**: `{dist}`")

    st.markdown(f"""
        **50% CI:** {result['ci_low']:.1%} to {result['ci_high']:.1%}  
        **Best Path:** {result['max']:.1%}  
        **Worst Path:** {result['min']:.1%}
    """)

st.markdown("---")
st.info("Try tweaking weights to see how your portfolio changes! Future simulations are based on historical volatility.")
