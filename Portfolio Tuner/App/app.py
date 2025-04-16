import streamlit as st
import requests
import pandas as pd
from auth import register_user, get_authenticator
from video_utils import display_video

# --- App Config ---
st.set_page_config(page_title="Crypto Portfolio Optimizer", layout="wide")
st.sidebar.title("Crypto Portfolio Optimizer")

# --- Custom Font ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# --- Authenticator ---
authenticator = get_authenticator()
name, authentication_status, username = authenticator.login("Login", "sidebar")

# --- Welcome Header ---
st.title("📈 Portfolio Tuner")
st.markdown("<h3 style='color:#A9A9B3; font-style:italic;'>Your simple toolkit for smarter crypto investing.</h3>", unsafe_allow_html=True)

# --- Login Feedback ---
if authentication_status:
    st.session_state.auth_status = True
    st.session_state.username = username

    user_display_name = authenticator.credentials["usernames"][username]["name"]
    st.sidebar.success(f"✅ Logged in as {user_display_name}")

    if authenticator.logout("Logout", "sidebar"):
        for key in ["auth_status", "username"]:
            st.session_state.pop(key, None)
        st.experimental_rerun()

elif authentication_status is False:
    st.sidebar.error("Incorrect username or password.")
else:
    st.sidebar.info("Please log in to access your portfolio.")

# --- Homepage Video ---
try:
    display_video("Portfolio Tuner/App/assets/homepage_video.mp4", height=600)
except:
    st.warning("⚠️ Unable to load the intro video. Make sure the file exists.")

# --- Overview ---
st.markdown("## 👋 Welcome to Portfolio Tuner")

st.markdown("""
Whether you're new to crypto or just want to get a better handle on your holdings, **Portfolio Tuner** helps you:
- ✅ Create and manage a custom portfolio
- 📊 See how your assets are performing
- 🎯 Explore strategies for better diversification
- ⏳ Simulate your returns over time

No spreadsheets. No stress.
""")

# --- Navigation Cards ---
st.markdown("## 🔧 What would you like to do today?")

pages = [
    ("📝 My Portfolio", "Start here! Create or edit your crypto portfolio.",  "pages/1_My_Portfolio.py"),
    ("📊 Dashboard", "See how your portfolio is performing with visual insights.", "pages/2_Portfolio_Dashboard.py"),
    ("🎯 Optimizer", "Compare strategies and rebalance based on your risk comfort.", "pages/3_Portfolio_Optimizer.py"),
    ("⏳ Backtest Lab", "Look into the past to see how strategies would’ve performed.", "pages/4_Backtest_Lab.py"),
    ("🎮 Playground", "Tweak allocations and experiment freely — no pressure.", "pages/5_Playground.py"),
    ("📖 Glossary", "New to some of the terms? This glossary has you covered.", "pages/6_Glossary.py")
]

cols = st.columns(3)
for idx, (title, desc, link) in enumerate(pages):
    with cols[idx % 3]:
        st.markdown(f"### {title}")
        st.markdown(f"<span style='color:#dddddd'>{desc}</span>", unsafe_allow_html=True)
        if st.button(f"Go to {title}", key=title):
            st.switch_page(link)
st.markdown("---")

# --- Registration Section ---
if authentication_status is not True:
    with st.sidebar.expander("🔐 New Here? Create an Account"):
        new_name = st.text_input("Full Name", key='new_name')
        new_username = st.text_input("Choose a Username", key='new_username')
        new_password = st.text_input("Password", type="password", key='new_password')

        if st.button("Register"):
            if not new_name.strip() or not new_username.strip() or not new_password.strip():
                st.sidebar.error("Please fill in all fields.")
            else:
                success = register_user(new_username, new_name, new_password)
                if success:
                    st.sidebar.success("🎉 Registration successful! You can now log in.")
                else:
                    st.sidebar.error("Username already exists.")
