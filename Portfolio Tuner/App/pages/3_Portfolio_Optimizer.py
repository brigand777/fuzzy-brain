import streamlit as st
import pandas as pd
import altair as alt
import os

from auth import login_and_get_status
from utils.api_client import call_fastapi_optimizer
from optimizer import run_optimizers
from utils.plots import pie_chart_allocation, bar_chart_allocation
from components.portfolio_input import edit_portfolio
from user_input import get_optimization_methods

# --- Page Setup ---
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
authenticator, authentication_status, username = login_and_get_status()
st.title("🎯 Portfolio Optimizer")

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

# --- Style Helper ---
def narrative(text):
    st.markdown(
        f"""<div style="background-color: rgba(31, 119, 180, 0.1); padding: 10px; border-left: 4px solid #1F77B4; font-size: 18px; margin-bottom: 10px;">
        {text}
        </div>""",
        unsafe_allow_html=True
    )

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_parquet("Portfolio Tuner/App/data/prices.parquet")

data = load_data()
available_assets = data.columns.tolist()
st.success("✅ Historical price data loaded.")

# --- Step 1: Portfolio Setup ---
st.markdown("## Step 1: 📁 Select Your Portfolio")

input_mode = st.radio("Where is your portfolio?", ["Use My Saved Portfolio", "Build Portfolio Here"])
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
            st.warning("⚠️ No saved portfolio found. Please switch to 'Build Portfolio Here'.")
            st.stop()
    else:
        st.warning("🔒 Please log in to use your saved portfolio.")
        st.stop()
else:
    portfolio_df = edit_portfolio(available_assets, data, persistent=False)

if portfolio_df.empty or "Asset" not in portfolio_df.columns:
    st.warning("🚫 Your portfolio is empty. Please add assets.")
    st.stop()

st.dataframe(portfolio_df, use_container_width=True)

# --- Step 2: Strategy Selection ---
st.markdown("## Step 2: 🧠 Choose Optimization Strategies")
narrative("These strategies help you decide how to split your money across different assets for better balance and lower risk.")

with st.expander("📖 What do these mean?"):
    st.markdown("""
    - **Equal Weight**: Every asset gets the same amount of money.
    - **Smart Spread (Mean Variance / MVO)**: Uses past price data to find what *would have worked best* historically.
    - **Group & Balance (HRB)**: Groups similar investments together and spreads risk across those groups.
    """)

default_methods = ["Equal Weight", "Mean Variance", "HRB", "User Portfolio"]
selected_methods = st.multiselect(
    "Which strategies do you want to compare?",
    options=default_methods,
    default=default_methods,
    help="Select one or more optimization methods to visualize"
)

# --- Step 3: Settings (Optional) ---
st.markdown("## Step 3: ⚙️ Advanced Settings (Optional)")

lookback = 90  # sensible default
with st.expander("Adjust analysis period?"):
    lookback = st.selectbox(
        "📅 How many days of data should we use?",
        options=[30, 60, 90, 180, 365],
        index=2
    )


# --- Step 4: Optimization Trigger ---
st.markdown("## Step 3: 🚀 Run the Optimization")
optimize_button = st.button("Optimize My Portfolio")

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
        lookback_df = data[user_weights.keys()].tail(lookback)
        all_allocations = run_optimizers(lookback_df, nonnegative_mvo=True)

        # Add user’s original weights
        all_allocations["User Portfolio"] = pd.Series(user_weights)

        # Filter selected
        filtered_allocs = {m: all_allocations[m] for m in selected_methods if m in all_allocations}

        # --- Step 5: Results ---
        st.markdown("## Step 4: 📊 Review Allocation Results")
        narrative("Here’s how each strategy suggests splitting your money across the assets in your portfolio.")

        pie_dfs = []
        bar_dfs = []

        for method in selected_methods:
            weights = pd.Series(all_allocations[method]).round(4)
            df = pd.DataFrame({'Asset': weights.index, 'Weight': weights.values})
            df["Method"] = method
            pie_dfs.append(df)
            bar_dfs.append(df)

        # 🥧 Pie Charts
        st.markdown("### 🥧 Pie Charts (Investment Mix)")

        pie_charts = [pie_chart_allocation(pd.Series(all_allocations[method]), method) for method in selected_methods]
        st.altair_chart(alt.hconcat(*pie_charts), use_container_width=True)

        # 📊 Bar Charts
        st.markdown("### 📈 Bar Charts (Compare Strategies)")

        max_weight = max(df["Weight"].max() for df in bar_dfs)
        cols = st.columns(2)

        for i, df in enumerate(bar_dfs):
            show_legend = (i == 1) or (len(bar_dfs) == 1 and i == 0)
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("Asset:N", sort="-y"),
                y=alt.Y("Weight:Q", title="Weight", scale=alt.Scale(domain=[0, max_weight])),
                color=alt.Color("Asset:N", title="Asset", legend=alt.Legend(title="Asset") if show_legend else None),
                tooltip=["Asset:N", alt.Tooltip("Weight:Q", format=".2%")]
            ).properties(
                title=df["Method"].iloc[0],
                width=250,
                height=350
            )
            cols[i % 2].altair_chart(chart, use_container_width=True)

        # 📘 Summary
        st.markdown("### 📘 Summary")
        st.markdown(f"✅ You compared **{len(selected_methods)}** strategies over the last **{lookback} days**.\n\nScroll up to review which mix feels best for your goals.")

    except Exception as e:
        st.error("❌ An error occurred during optimization.")
        st.error(f"Details: {e}")
else:
    st.info("👆 Click the 'Optimize My Portfolio' button to generate strategy comparisons.")
