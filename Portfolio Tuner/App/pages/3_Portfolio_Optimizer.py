import streamlit as st
import pandas as pd
import altair as alt
import os

from auth import login_and_get_status
from utils.api_client import call_fastapi_optimizer
from optimizer import run_optimizers
from utils.plots import pie_chart_allocation, bar_chart_allocation  # bar chart defined below
from components.portfolio_input import edit_portfolio
from user_input import get_optimization_methods
from utils.glossary import vintage_dropdown

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

# --- Optimization Settings ---
st.markdown("## 📌 Compare Optimization Methods")
narrative("Run optimizations and compare different allocation strategies to your own.")
vintage_dropdown(
    "📜 What are we Optimizing?",
    """We want to mitigate the risk the portfolio experiences by spreading our investements apart, but how much do we put in each basket?<br>
    Here we explore 3 different methods for that: Equal Weight, Mean  Variance (MVO), and Hierarchical Risk Budgeting (HRB): <br>
    MVO--is a traditioal financial technique that uses the past to give the best results had we known the future <br>
    HRB--lumps similar invetments into baskets and gives each basket a fixed risk budget to spread around""",
)

# --- User Controls BEFORE optimization ---
with st.expander("⚙️ Optimization Settings"):
    lookback = st.selectbox(
        "🔁 Select backtest window (days)", 
        options=[30, 60, 90, 180, 365],
        index=2,
        help="Use the dropdown to choose a lookback period"
    )

    default_methods = ["Equal Weight", "Mean Variance", "HRB", "User Portfolio"]
    selected_methods = st.multiselect(
        "🧠 Select optimization methods to display",
        options=default_methods,
        default=default_methods,
        help="Search and select multiple methods to visualize"
    )

# --- Optimizer Trigger ---
optimize_button = st.button("🚀 Optimize Portfolio")

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

        # Add user's portfolio
        all_allocations["User Portfolio"] = pd.Series(user_weights)

        # Filter to selected methods only (in case user unchecked some)
        filtered_allocs = {m: all_allocations[m] for m in selected_methods if m in all_allocations}

  
        # --- Allocation Charts ---
        st.markdown("### 🧩 Allocation Comparison Charts")

        # Combine pie/bar data
        pie_dfs = []
        bar_dfs = []

        for method in selected_methods:
            weights = pd.Series(all_allocations[method]).round(4)
            df = pd.DataFrame({'Asset': weights.index, 'Weight': weights.values})
            df["Method"] = method
            pie_dfs.append(df)
            bar_dfs.append(df)

        st.markdown("### 🥧 Allocation Pie Charts")

        # Generate all pie charts using your function, suppress legend on all but last
        pie_charts = []
        for i, method in enumerate(selected_methods):
            weights = pd.Series(all_allocations[method]).round(4)

            chart = pie_chart_allocation(weights, method)

            # Remove legend from all but last chart
            if i != len(selected_methods) - 1:
                chart = chart.encode(
                    color=alt.Color("Asset:N", legend=None)
                )

            pie_charts.append(chart)

        # Combine into one row
        st.altair_chart(
            alt.hconcat(*pie_charts).resolve_scale(color="shared"),
            use_container_width=True
        )




        # 📊 BAR CHARTS: 2-column layout, legend only on last chart in top row
        st.markdown("### 📊 Allocation Bar Charts")

        max_weight = max(df["Weight"].max() for df in bar_dfs)

        cols = st.columns(2)
        for i, df in enumerate(bar_dfs):
            # Show legend only on rightmost top-row chart (index 1), or chart 0 if only one
            show_legend = (i == 1) or (len(bar_dfs) == 1 and i == 0)
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("Asset:N", sort="-y"),
                y=alt.Y("Weight:Q", title="Weight", scale=alt.Scale(domain=[0, max_weight])),
                color=alt.Color("Asset:N", title="Asset", legend=alt.Legend(title="Asset") if show_legend else None),
                tooltip=["Asset:N", alt.Tooltip("Weight:Q", format=".2%")]
            ).properties(
                title=df["Method"].iloc[0],
                width=250,
                height=250
            )
            cols[i % 2].altair_chart(chart, use_container_width=True)





    except Exception as e:
        st.error("An error occurred during optimization.")
        st.error(f"Details: {e}")
else:
    st.info("Click the 'Optimize Portfolio' button to see initial allocations.")
