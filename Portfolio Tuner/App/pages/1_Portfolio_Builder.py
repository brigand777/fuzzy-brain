import streamlit as st
import pandas as pd
import os

from auth import login_and_get_status
from components.portfolio_input import edit_portfolio
from utils.glossary import chart_with_tooltip, add_info_icon, section_heading, inject_tooltip_css,set_global_font_style

# --- Page Setup ---
st.set_page_config(page_title="My Portfolio", layout="wide")
inject_tooltip_css()
set_global_font_style()
authenticator, authentication_status, username = login_and_get_status()

st.title("📝 Portfolio Builder")

# --- Load asset data ---
@st.cache_data
def load_data():
    return pd.read_parquet("Portfolio Tuner/App/data/prices.parquet")

data = load_data()
available_assets = data.columns.tolist()

# --- Ensure portfolios directory exists ---
os.makedirs("Portfolio Tuner/App/portfolios", exist_ok=True)

# --- Load or initialize user portfolio ---
if authentication_status:
    portfolio_path = f"Portfolio Tuner/App/portfolios/{username}_portfolio.csv"
    if os.path.exists(portfolio_path):
        st.success("✅ Loaded your saved portfolio.")
        st.session_state.editable_portfolio = pd.read_csv(portfolio_path)
    else:
        st.info("👤 No saved portfolio found. Start building one below.")
else:
    st.warning("⚠️ You are not logged in. Your portfolio changes will not be saved.")

# --- Portfolio Input UI ---
st.markdown("### 📌 Add or Adjust Assets")
portfolio_df = edit_portfolio(available_assets, data, persistent=authentication_status)

# --- Navigation Link to Dashboard ---
st.markdown("---")
if st.button("🔙 Go to Tunerboard"):
        st.switch_page("pages/2_Tunerboard.py")
