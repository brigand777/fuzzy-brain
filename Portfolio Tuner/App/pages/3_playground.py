import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from utils.plots import plot_cumulative_returns, add_interactivity
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
st.markdown("### 🗂️ Select Assets to Include")
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

# --- Sliders for asset weights ---
st.markdown("## 🧰 Adjust Your Hypothetical Portfolio")
weights = {}
total_weight = 0
for asset in playground_assets:
    weight = st.slider(f"{asset} weight", 0.0, 1.0, 0.05, 0.005)
    weights[asset] = weight
    total_weight += weight

# --- Normalize weights ---
weights = {k: v / total_weight for k, v in weights.items() if total_weight > 0}

# --- Show emoji tag ---
risk_score = sum((v * np.std(data[k].pct_change())) for k, v in weights.items())
if risk_score > 0.05:
    st.markdown("**Portfolio Risk Level:** 🔥 High")
elif risk_score > 0.03:
    st.markdown("**Portfolio Risk Level:** ⚠️ Medium")
else:
    st.markdown("**Portfolio Risk Level:** 🧊 Low")


lookback_days = 365
simulation_data = data[playground_assets].tail(lookback_days)
pct_returns = simulation_data.pct_change().dropna()

# Backtest simulated portfolio
portfolio_returns = pct_returns.dot(pd.Series(weights))
cumulative_returns = (1 + portfolio_returns).cumprod()
cumulative_df = pd.DataFrame({
    "date": simulation_data.index[-len(cumulative_returns):],
    "cumulative": cumulative_returns
})

# --- Portfolio Performance Metrics ---
st.markdown("### 📊 Portfolio Stats (Past 365 Days)")

mean_daily_return = portfolio_returns.mean()
volatility = portfolio_returns.std()
sharpe_ratio = (mean_daily_return / volatility)* np.sqrt(365.0) if volatility > 0 else 0

cumulative_return = cumulative_returns.iloc[-1] - 1
annualized_volatility = volatility * np.sqrt(365.0)

# Format colored value for HTML
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

# Display in columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"**Cumulative Return**<br>{styled_percent(cumulative_return)}", unsafe_allow_html=True)

with col2:
    st.markdown(f"**Annualized Volatility**<br>{styled_percent(annualized_volatility, red_if_high=True)}", unsafe_allow_html=True)

with col3:
    st.markdown(f"**Sharpe Ratio**<br>{styled_number(sharpe_ratio)}", unsafe_allow_html=True)


chart = plot_cumulative_returns({
    "Playground Portfolio": {
        "cumulative": cumulative_returns
    }
})

# --- Portfolio preview ---
st.subheader("📊 Cumulative Returns")
st.altair_chart(add_interactivity(chart, x_field="date", y_field="cumulative"), use_container_width=True)

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
