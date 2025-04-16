import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

from utils.plots import plot_cumulative_returns, add_interactivity, plot_single_gauge
from optimizer import run_optimizers
from utils.simulation import run_smart_monte_carlo_simulation

# --- Page Setup ---
st.set_page_config(page_title="Playground", layout="wide")
st.title("🎮 Portfolio Playground")

st.markdown(
    """
    <style>
    .streamlit-expanderHeader {
        font-size: 20px !important;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Narrative Helper ---
def narrative(text):
    st.markdown(
        f"""<div style="background-color: rgba(33, 150, 243, 0.1); padding: 10px; border-left: 4px solid #2196F3; font-size: 18px; margin-bottom: 10px;">
        {text}
        </div>""",
        unsafe_allow_html=True
    )

narrative("This is your space to explore! Adjust your hypothetical portfolio, simulate outcomes, and learn how different allocations affect your risk and return.")

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_parquet("Portfolio Tuner/App/data/prices.parquet")

data = load_data()
available_assets = data.columns.tolist()

# --- Step 1: Choose Assets ---
st.markdown("## Step 1: 🧠 Choose Your Assets")

with st.expander("💼 Try a Sample Crypto Portfolio"):
    preset = st.radio(
        "Choose a sample mix to get started:",
        ["None", "Big 3 (BTC/ETH/SOL)", "DeFi Focus", "Diversified Altcoins"]
    )

    if preset == "Big 3 (BTC/ETH/SOL)":
        selected_assets = ["BTC", "ETH", "SOL"]
    elif preset == "DeFi Focus":
        selected_assets = ["UNI", "AAVE", "COMP", "CRV"]
    elif preset == "Diversified Altcoins":
        selected_assets = ["MATIC", "DOT", "AVAX", "ADA", "LINK"]
    else:
        selected_assets = available_assets[:5]  # default fallback

selected_assets = st.multiselect(
    "Manually choose up to 10 assets to simulate:",
    options=available_assets,
    default=selected_assets,
    max_selections=10
)

if not selected_assets:
    st.warning("Please select at least one asset to proceed.")
    st.stop()

playground_assets = selected_assets
latest_prices = data.iloc[-1]

# --- Step 2: 🎚️ Adjust Your Hypothetical Portfolio ---
st.markdown("## Step 2: 🎚️ Adjust Your Hypothetical Portfolio")

slider_col, chart_col = st.columns([1, 2], gap="medium")

# --- Weight Sliders and Pie Chart ---
with slider_col:
    st.markdown("#### Set your asset weights (%):")

    # Reset button (reruns app and resets values)
    if st.button("🔁 Reset Weights to Even Split"):
        for asset in playground_assets:
            st.session_state[f"weight_{asset}"] = round(100.0 / len(playground_assets), 2)
        st.rerun()

    auto_normalize = st.checkbox("🔄 Automatically normalize to 100%", value=True)

    weights = {}
    total_weight = 0

    for asset in playground_assets:
        default_value = float(st.session_state.get(f"weight_{asset}", 1.0 / len(playground_assets) if playground_assets else 0.05))
        weight = st.slider(
            f"{asset} (%)",
            min_value=0.0,
            max_value=100.0,
            value=default_value,
            step=0.5,
            key=f"weight_{asset}"
        )
        weights[asset] = weight
        total_weight += weight

    st.markdown(f"**Total Allocation: {total_weight:.2f}%**")

    if auto_normalize and total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    else:
        weights = {k: v / 100.0 for k, v in weights.items()}  # convert % → fraction
        if abs(total_weight - 100) > 1:
            st.warning(f"⚠️ Your weights add up to {total_weight:.2f}%. Consider normalizing.")

    # Show pie chart of final weights
    pie_df = pd.DataFrame({"Asset": list(weights.keys()), "Weight": list(weights.values())})
    pie_fig = px.pie(pie_df, names="Asset", values="Weight", title="📊 Token Allocation")
    pie_fig.update_traces(textinfo="label+percent", hovertemplate="%{label}: %{percent}")
    st.plotly_chart(pie_fig, use_container_width=True)


# --- Portfolio Metrics Calculation ---
lookback_days = 365
# Replace this:
# simulation_data = data[playground_assets].tail(lookback_days)

# --- Clean simulation data ---
cleaned_data = data[playground_assets].copy()
cleaned_data = cleaned_data.replace(0, np.nan).ffill().dropna(how="any")

simulation_data = cleaned_data.tail(lookback_days)
returns = simulation_data.pct_change().dropna()

# --- Calculate weights using current price and slider values ---
latest_prices = simulation_data.iloc[-1]
portfolio_df = pd.DataFrame({
    "Asset": list(weights.keys()),
    "SliderWeight": list(weights.values()),
    "Price": [latest_prices.get(a, np.nan) for a in weights.keys()]
})
#portfolio_df["Value"] = portfolio_df["SliderWeight"] * portfolio_df["Price"]
#total_value = portfolio_df["Value"].sum()
#portfolio_df["NormalizedWeight"] = portfolio_df["Value"] / total_value

aligned_weights = pd.Series(weights).reindex(simulation_data.columns).fillna(0)
portfolio_returns = returns.dot(aligned_weights)
# --- Portfolio returns ---
cumulative_returns = (1 + portfolio_returns).cumprod()

mean_daily_return = portfolio_returns.mean()
volatility = portfolio_returns.std()
sharpe_ratio = (mean_daily_return / volatility) * np.sqrt(365.0) if volatility > 0 else 0

cumulative_return = cumulative_returns.iloc[-1] - 1
annualized_volatility = volatility * np.sqrt(365.0)
risk_score = sum((v * np.std(data[k].pct_change())) for k, v in weights.items())

# --- Chart Column ---
with chart_col:
    st.markdown("#### 📈 1-Year Cumulative Return")

    chart = plot_cumulative_returns({
        "Playground Portfolio": {
            "cumulative": cumulative_returns
        }
    }, show_legend=False)
    st.altair_chart(add_interactivity(chart, x_field="date", y_field="cumulative"), use_container_width=True)

    # --- Portfolio Gauges ---
    col1, col2, col3 = st.columns(3)

    with col1:
        try:
            fig = plot_single_gauge("Cumulative Return", cumulative_return * 100, metric_name="cumulative")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("📈 % growth over the past year")
        except Exception as e:
            st.warning("Could not render Cumulative Return gauge.")
            st.exception(e)

    with col2:
        try:
            fig = plot_single_gauge("Annualized Volatility", annualized_volatility * 100, metric_name="volatility")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("💡 Measures portfolio bumpiness")
        except Exception as e:
            st.warning("Could not render Volatility gauge.")
            st.exception(e)

    with col3:
        try:
            fig = plot_single_gauge("Sharpe Ratio", sharpe_ratio, metric_name="sharpe")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("📊 Return per unit of risk (above 1 is strong)")
        except Exception as e:
            st.warning("Could not render Sharpe Ratio gauge.")
            st.exception(e)

    # --- Risk Score ---
    if risk_score > 0.05:
        st.markdown("**Portfolio Risk Level:** 🔥 High")
    elif risk_score > 0.03:
        st.markdown("**Portfolio Risk Level:** ⚠️ Medium")
    else:
        st.markdown("**Portfolio Risk Level:** 🧣 Low")


# --- Step 4: Monte Carlo Simulator ---
st.markdown("## Step 4: 🔮 Forecast the Future")

with st.expander("🔮 Monte Carlo Future Simulator", expanded=False):
    narrative("We run 100 randomized future paths using fitted distributions and correlations. Useful for stress testing your allocation.")

    correlation_strategy = st.selectbox(
        "Choose correlation method:",
        ["shrinkage (default)", "historical", "independent"],
        help="Shrinkage = stable estimate; historical = recent correlation; independent = no correlation"
    )

    horizon_days = st.slider("⏳ Forecast horizon (days)", 30, 365, 180, step=30)
    n_sims = st.slider("🎲 Number of simulations", 50, 500, 100, step=50)

    if st.button("🚀 Run Monte Carlo Simulation"):
        if len(weights) < 2:
            st.warning("Add more assets for meaningful simulation.")
        else:
            result = run_smart_monte_carlo_simulation(
                weights,
                data[playground_assets],
                horizon_days=horizon_days,
                n_sims=n_sims,
                correlation_strategy=correlation_strategy.split(" ")[0]
            )
            st.plotly_chart(result["chart"], use_container_width=True)

            st.markdown("### 📊 Forecast Summary")
            st.markdown(f"""
            - **Median 6-Month Return:** {(result['ci_high'] + result['ci_low']) / 2:.1%}  
            - **Best Case Path:** {result['max']:.1%}  
            - **Worst Case Path:** {result['min']:.1%}  
            - **Distributions Used:**
            """)

            for asset, dist in result["distribution_used_per_asset"].items():
                st.markdown(f"`{asset}` → **{dist}**")

            st.info(f"Simulated using `{result['correlation_strategy']}` correlation strategy.")

# --- Footer ---
st.markdown("---")
st.info("Try tweaking weights or asset selections to explore different outcomes. This is your sandbox!")
