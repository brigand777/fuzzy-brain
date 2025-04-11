import streamlit as st
import pandas as pd
import altair as alt
import os

from auth import login_and_get_status
from utils.api_client import call_fastapi_optimizer
from optimizer import run_optimizers
from utils.plots import pie_chart_allocation
from components.portfolio_input import edit_portfolio
from user_input import get_optimization_methods

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# --- Authentication ---
authenticator, authentication_status, username = login_and_get_status()
st.title("🎯 Portfolio Optimizer")

# --- Helper ---
def narrative(text):
    st.markdown(
        f"""<div style="background-color: rgba(31, 119, 180, 0.2); padding: 10px; border-left: 4px solid #1F77B4; font-size: 18px; margin-bottom: 10px;">
        {text}
        </div>""",
        unsafe_allow_html=True
    )

# --- Load data ---
@st.cache_data
def load_data():
    return pd.read_parquet("Portfolio Tuner/App/data/prices.parquet")

data = load_data()
available_assets = data.columns.tolist()
st.success("Loaded historical price data.")

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
    portfolio_df = edit_portfolio(available_assets, data, persistent=False)

if portfolio_df.empty or "Asset" not in portfolio_df.columns:
    st.warning("Your portfolio is empty. Please add assets.")
    st.stop()

# --- Optimizer Trigger ---
st.markdown("## 📌 Compare Optimization Methods")
narrative("Run optimizations and compare different allocation strategies to your own.")

optimize_button = st.button("Optimize Portfolio")

if optimize_button:
    try:
        # Normalize portfolio weights
        latest_prices = data.iloc[-1]
        values = portfolio_df.apply(lambda row: row["Amount"] * latest_prices.get(row["Asset"], 0), axis=1)
        total_value = values.sum()
        user_weights = {
            row["Asset"]: (row["Amount"] * latest_prices.get(row["Asset"], 0)) / total_value
            for _, row in portfolio_df.iterrows()
            if latest_prices.get(row["Asset"], 0) > 0
        }

        # Optimizer input
        lookback = 90
        lookback_df = data[user_weights.keys()].tail(lookback)
        all_allocations = run_optimizers(lookback_df, nonnegative_mvo=True)

        # Add user's portfolio
        all_allocations["User Portfolio"] = pd.Series(user_weights)
        selected_methods = get_optimization_methods(all_allocations)

        st.markdown("### 📊 Initial Allocations")
        pie_charts = [
            pie_chart_allocation(pd.Series(all_allocations[m]).round(4), m)
            for m in selected_methods
        ]
        st.altair_chart(alt.hconcat(*pie_charts), use_container_width=True)

    except Exception as e:
        st.error("An error occurred during optimization.")
        st.error(f"Details: {e}")
else:
    st.info("Click the 'Optimize Portfolio' button to see initial allocations.")