import streamlit as st
import pandas as pd
import os

from auth import login_and_get_status
from components.portfolio_input import edit_portfolio

st.set_page_config(page_title="My Portfolio", layout="wide")
authenticator, authentication_status, username = login_and_get_status()

st.title("📝 Portfolio Editor")

# --- Load asset data ---
@st.cache_data
def load_data():
    return pd.read_parquet("Portfolio Tuner/App/data/prices.parquet")

data = load_data()
available_assets = data.columns.tolist()

# --- Load or initialize portfolio ---
if authentication_status:
    portfolio_path = f"Portfolio Tuner/App/portfolios/{username}_portfolio.csv"
    if os.path.exists(portfolio_path):
        st.success("Loaded saved portfolio.")
        st.session_state.editable_portfolio = pd.read_csv(portfolio_path)
    else:
        os.makedirs("Portfolio Tuner/App/portfolios", exist_ok=True)
else:
    st.info("Using temporary portfolio (not saved).")

# --- Portfolio input section ---
portfolio_df = edit_portfolio(available_assets, data, persistent=authentication_status)

# --- Navigation ---
st.markdown("---")
st.markdown("[📈 Go to Portfolio Dashboard](2_Portfolio_Dashboard.py)")