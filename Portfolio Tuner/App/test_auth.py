import streamlit as st
import streamlit_authenticator as stauth

# Minimal credentials
credentials = {
    "usernames": {
        "demo_user": {
            "name": "Demo User",
            "password": "$2b$12$uFZ3sTO3BjE.5v9oE1zq5uPQ0Hrp7rRz6HBRPgR9guIuR0GPPZ6C2"  # password123
        }
    }
}

# Setup authenticator for version 0.2.2
authenticator = stauth.Authenticate(
    credentials,
    cookie_name="test_cookie",
    key="some_key",
    cookie_expiry_days=1
)

# ✅ Sidebar login
auth_status = authenticator.login("sidebar")

if auth_status:
    st.sidebar.success("Login successful")
    if authenticator.logout("Logout", "sidebar"):
        st.experimental_rerun()
elif auth_status is False:
    st.sidebar.error("Login failed")
else:
    st.sidebar.info("Please log in")
