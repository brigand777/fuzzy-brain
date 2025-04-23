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
st.markdown("<h3 style='color:#A9A9B3; font-style:italic;'>Optimize your crypto, maximize your gains.</h3>", unsafe_allow_html=True)

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
'''
# --- Why This Exists ---
st.markdown("## 🤔 Why Portfolio Tuner?")
st.markdown("""
**Existing portfolio management tools are built for traditional investments.**  
But crypto portfolios move faster, trade 24/7, and behave differently.  
**That’s where we come in.**

Portfolio Tuner gives you:

- 🔗 **Native crypto asset support** — no manual uploads or workarounds
- 📡 **Real-time simulation** with on-chain or API-based data
- 🧠 **Risk and return metrics tailored to crypto volatility**
- 🎓 **Education-first UX** to guide your investing decisions
""")
'''
# --- Mascot Introduction Section ---
left, right = st.columns([1, 2])  # 1:2 ratio for mascot + message

with left:
    st.image("Portfolio Tuner/App/assets/Tuner_boy2.png", width=250)

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
        <h4 style="margin-top: 0;">Hey I'm Tuner Boy!</h4>
        I'm here to be your support buddy as you naviagte <strong>Portfolio Tuner</strong> – your personalized crypto strategy assistant.<br><br>
        I’m here to help you explore, experiment, and optimize your investments without stress.<br><br>
        I've been arond the block a few times so I'll try my best to guida ya through the nerd talk. Ready to tune things up?! Meet me in the Tuner Tour section, I'll see ya there!
    </div>
    """, unsafe_allow_html=True)


# --- Overview ---
st.markdown("## 👋 What You Can Do")

st.markdown("""
Whether you're a curious beginner or a seasoned investor, Portfolio Tuner helps you:
- ✅ Build and adjust a custom crypto portfolio
- 📊 Analyze historical and real-time performance
- 🎯 Compare with automated allocation strategies (HRB, MVO, Equal Weight)
- ⏳ Run Monte Carlo simulations to model future outcomes
- 🔍 Backtest your strategy against market history

No spreadsheets. No subscriptions. Just smarter investing.
""")

# --- Navigation Cards ---
st.markdown("## 🔧 What would you like to do today?")

pages = [
    ("🚀 Tuner Tour", "Start here! Get a quick guided overview of how to use the app.", "pages/0_Tuner_Tour.py"),
    ("🛠️ Portfolio Builder", "Build and customize your crypto portfolio with asset weights or quantities.", "pages/1_Portfolio_Builder.py"),
    ("📊 Tunerboard", "Visualize performance, risk, and correlations of your portfolio.", "pages/2_Tunerboard.py"),
    ("🎯 Portfolio Tuner", "Run optimizations like HRP, MVO, and Equal Weight to find better allocations.", "pages/3_Portfolio_Tuner.py"),
    ("🧪 Strategy Sandbox", "Experiment with strategy simulations and compare allocation outcomes.", "pages/5_Strategy_Sandbox.py"),
    ("📚 Tuner Glossary", "Look up key crypto investing terms and portfolio theory concepts.", "pages/6_Tuner_Glossary.py")
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

