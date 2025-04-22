import streamlit as st
import pandas as pd
from utils.plots import plotly_pie_allocation

def render_lite_intro():
    st.subheader("🧊 Lite Mode Activated")
    st.info("You're in Lite Mode — we've simplified the experience for quicker insights.")

def render_lite_portfolio_sample(data):
    st.markdown("### 💼 Sample Portfolio (BTC/ETH/SOL)")
    assets = ["BTC", "ETH", "SOL"]
    amounts = [0.5, 2, 10]
    portfolio_df = pd.DataFrame({"Asset": assets, "Amount": amounts})

    # Calculate values
    latest_prices = data.iloc[-1]
    values = portfolio_df.apply(lambda row: row["Amount"] * latest_prices.get(row["Asset"], 0), axis=1)
    portfolio_df["Value ($)"] = values
    portfolio_df["Allocation (%)"] = (values / values.sum() * 100).round(2)

    st.dataframe(portfolio_df, use_container_width=True)

    st.markdown("#### 📊 Portfolio Allocation")
    weights = pd.Series(portfolio_df["Allocation (%)"].values, index=portfolio_df["Asset"])
    fig = plotly_pie_allocation(weights, title="Your Crypto Mix", show_legend=True)
    st.plotly_chart(fig, use_container_width=True)

    return portfolio_df
