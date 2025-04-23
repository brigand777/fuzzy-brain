import streamlit as st
import pandas as pd
import os

from auth import login_and_get_status
from components.portfolio_input import edit_portfolio
from utils.glossary import chart_with_tooltip, add_info_icon, section_heading, inject_tooltip_css,set_global_font_style

# --- Page Setup ---
st.set_page_config(page_title="My Portfolio", layout="wide")

# --- Lite Mode Toggle ---
st.sidebar.markdown("### 🧪 Experimental")
st.session_state.lite_mode = st.sidebar.toggle("Lite Mode", value=st.session_state.get("lite_mode", False))

inject_tooltip_css()
set_global_font_style()

st.markdown("""
<style>
.stMarkdown p, .stMarkdown div, p {
    font-size: 18px !important;
    line-height: 1.5;
}
.stDataFrame, .stTable, .stPlotlyChart, .stAltairChart {
    font-size: inherit !important;
}
.narrative-box {
    font-size: 18px !important;
}
</style>
""", unsafe_allow_html=True)

authenticator, authentication_status, username = login_and_get_status()

st.title("📝 Portfolio Builder")

# --- Mascot Introduction Section ---
left, right = st.columns([1, 2])  # 1:2 ratio for mascot + message

with left:
    st.image("Portfolio Tuner/App/assets/Tuner_boy_tech2.png", width=250)

with right:
    st.markdown(f"""
        <div style="
            background-color: #FAF3D3;
            border: 1px solid #D6C899;
            padding: 20px 24px;
            border-radius: 12px;
            box-shadow: 2px 4px 8px rgba(0, 0, 0, 0.1);
            font-size: 1rem;
            line-height: 1.6;
            font-family: 'Merriweather', serif;
            color: #1A1A1A;
        ">
            Here's where you get to building! You can use the edit features to add or remove a crypto, but I really just like watching the charts move.<br><br>
            If it all looks a bit overwhelming, switch to lite mode :)
        </div>
        """, unsafe_allow_html=True)

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
