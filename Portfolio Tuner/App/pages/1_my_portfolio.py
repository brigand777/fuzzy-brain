import streamlit as st
import pandas as pd
import os

from auth import login_and_get_status
from components.portfolio_input import edit_portfolio
from plots import (
    plot_asset_returns,
    plot_asset_prices,
    add_interactivity,
    plot_portfolio_allocation_3d,
    plot_historical_assets
)

st.set_page_config(page_title="My Portfolio", layout="wide")
authenticator, authentication_status, username = login_and_get_status()

st.title("My Portfolio")

def show_my_portfolio():
    # --- Load available assets from price data ---
    @st.cache_data
    def load_data():
        return pd.read_parquet("Portfolio Tuner/App/data/prices.parquet")

    data = load_data()
    available_assets = data.columns.tolist()

    # --- Load saved portfolio or initialize empty ---
    if authentication_status:
        portfolio_path = f"portfolios/{username}_portfolio.csv"
        if os.path.exists(portfolio_path):
            st.success("Loaded saved portfolio.")
            st.session_state.editable_portfolio = pd.read_csv(portfolio_path)
        else:
            os.makedirs("portfolios", exist_ok=True)
    else:
        st.info("Using temporary portfolio (not saved).")

    # --- Portfolio input section ---
    portfolio_df = edit_portfolio(available_assets)

    # --- Save updated portfolio if logged in ---
    if authentication_status and "editable_portfolio" in st.session_state:
        portfolio_path = f"portfolios/{username}_portfolio.csv"
        st.session_state.editable_portfolio.to_csv(portfolio_path, index=False)

    # --- Plotting toggle ---
    if "show_plot" not in st.session_state:
        st.session_state.show_plot = False

    if st.button("📊 Plot Historical Assets"):
        st.session_state.show_plot = not st.session_state.show_plot

    if st.session_state.show_plot:
        # Extract selected assets from the portfolio
        selected_assets = portfolio_df["Asset"].dropna().unique().tolist()
        if selected_assets:
            plot_historical_assets(data, selected_assets, portfolio_df=portfolio_df)
        else:
            st.warning("No assets found in your portfolio to plot.")

# Run the page
show_my_portfolio()
