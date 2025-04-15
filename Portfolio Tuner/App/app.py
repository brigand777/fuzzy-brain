import streamlit as st
import requests
import pandas as pd
from auth import register_user, get_authenticator
from video_utils import display_video

# --- Streamlit App Config ---
st.set_page_config(page_title="Crypto Portfolio Optimizer", layout="wide")
st.sidebar.title("Crypto Portfolio Optimizer")

# --- Load Custom Font ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .page-section {
        margin-bottom: 2rem;
    }
    .page-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        color: #1F77B4;
    }
    .page-desc {
        font-size: 1rem;
        color: #444;
        margin-bottom: 0.5rem;
    }
    .page-button {
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title + Tagline ---
st.title("Portfolio Tuner")
st.markdown("<h3 style='font-size:20px; font-style:italic; color:#A9A9B3;'>Optimize Your Crypto, Maximize Your Gains.</h3>", unsafe_allow_html=True)

# --- Authenticator Setup ---
authenticator = get_authenticator()
name, authentication_status, username = authenticator.login("Login", "sidebar")

if authentication_status:
    st.session_state.auth_status = True
    st.session_state.username = username
    if username:
        name = authenticator.credentials["usernames"][username]["name"]
        st.sidebar.success(f"Logged in as {name}")
    if authenticator.logout("Logout", "sidebar"):
        for key in ["auth_status", "username"]:
            st.session_state.pop(key, None)
        st.experimental_rerun()
elif authentication_status is False:
    st.sidebar.error("Incorrect username or password.")
else:
    st.sidebar.info("Please log in.")

# --- FastAPI Optimizer Integration ---
def call_fastapi_optimizer(price_df, asset_weights, lookback_days, nonnegative):
    payload = {
        "assets": asset_weights,
        "price_data": price_df.to_dict(orient="list"),
        "lookback_days": lookback_days,
        "nonnegative": nonnegative
    }
    try:
        response = requests.post("http://localhost:8000/optimize", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Failed to fetch optimizations from backend.")
            return {}
    except Exception as e:
        st.error(f"Error contacting optimization API: {e}")
        return {}

# --- HOME PAGE ---

try:
    display_video("Portfolio Tuner/App/assets/homepage_video.mp4", height=600)
except:
    st.warning("⚠️ Unable to load homepage video. Please ensure the file exists and is accessible.")

st.markdown("## Welcome to Portfolio Tuner")

st.write("""
Portfolio Tuner is your personal toolkit for building, analyzing, and optimizing a crypto portfolio. Whether you're a casual holder or a serious allocator, our tools help you manage risk and make data-driven investment decisions.
""")

st.markdown("---")

pages = [
    ("📝 My Portfolio", "Create and edit your crypto portfolio. This is your starting point—define how much you hold of each asset.", "pages/1_My_Portfolio.py"),
    ("📊 Portfolio Dashboard", "Track performance, visualize metrics, and understand how your portfolio is evolving over time.", "pages/2_Portfolio_Dashboard.py"),
    ("🎯 Optimizer", "Compare different allocation strategies using HRP, MVO, and other methods to find the most effective distribution.", "pages/3_Portfolio_Optimizer.py"),
    ("⏳ Backtest Lab", "See how your strategies would have performed historically using flexible backtesting tools.", "pages/4_Backtest_Lab.py"),
    ("🎮 Playground", "Tweak allocations freely and simulate potential outcomes to understand risk and return tradeoffs.", "pages/5_Playground.py"),
    ("📖 Glossary", "Browse definitions and explanations of financial and crypto terms used throughout the app.", "pages/6_Glossary.py")
]

for title, desc, link in pages:
    with st.container():
        st.markdown(f"<div class='page-section'>", unsafe_allow_html=True)
        st.markdown(f"<div class='page-header'>{title}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='page-desc'>{desc}</div>", unsafe_allow_html=True)
        if st.button(f"Go to {title}", key=title):
            st.switch_page(link)
        st.markdown(f"</div>", unsafe_allow_html=True)

st.markdown("---")

# --- Registration ---
if authentication_status is not True:
    with st.sidebar.expander("Register New User"):
        new_name = st.text_input("Full Name", key='new_name')
        new_username = st.text_input("Username", key='new_username')
        new_password = st.text_input("Password", type="password", key='new_password')

        if st.button("Register"):
            if new_username.strip() == "" or new_password.strip() == "" or new_name.strip() == "":
                st.sidebar.error("Please fill all fields.")
            else:
                success = register_user(new_username, new_name, new_password)
                if success:
                    st.sidebar.success("Registration successful! You can now log in.")
                else:
                    st.sidebar.error("Username already exists.")