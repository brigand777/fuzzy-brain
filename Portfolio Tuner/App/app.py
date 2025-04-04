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
    """
    This function sends user-selected price data and weights to the FastAPI backend for optimization.
    Make sure both `price_df` and `asset_weights` come directly from user input or an uploaded portfolio.
    """
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

display_video("assets/homepage_video.mp4", height=600)
st.info("Select a page from the sidebar to begin exploring the optimization tools.")

st.markdown("## Welcome to the Crypto Portfolio Optimizer")
st.write("""
    This platform is designed to help retail crypto investors manage risk 
    through advanced portfolio optimization techniques and interactive visualizations.
""")

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
