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
playground_assets = available_assets[:6]  # Limit to 5–6 assets to keep UI simple
latest_prices = data.iloc[-1]

st.markdown("## 🧰 Adjust Your Hypothetical Portfolio")

# --- Sliders for asset weights ---
weights = {}
total_weight = 0
for asset in playground_assets:
    weight = st.slider(f"{asset}", 0.0, 1.0, 0.2, 0.05)
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

# --- Portfolio preview ---
st.subheader("📊 Cumulative Returns")
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

# Pass to plot_cumulative_returns
chart = plot_cumulative_returns({
    "Playground Portfolio": {
        "cumulative": cumulative_returns
    }
})
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
