import pandas as pd
import streamlit_authenticator as stauth
import bcrypt
import os
import streamlit as st

# File to store users
USERS_FILE = "Portfolio Tuner/App/users.csv"

def load_users():
    if os.path.exists(USERS_FILE):
        return pd.read_csv(USERS_FILE)
    return pd.DataFrame(columns=["username", "name", "password"])

def save_users(users_df):
    users_df.to_csv(USERS_FILE, index=False)

def register_user(username, name, password):
    users_df = load_users()
    if username in users_df["username"].values:
        return False  # username taken
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    new_row = pd.DataFrame([[username, name, hashed_pw]], columns=["username", "name", "password"])
    updated = pd.concat([users_df, new_row], ignore_index=True)
    save_users(updated)
    return True

def get_authenticator():
    users_df = load_users()
    credentials = {"usernames": {}}
    for _, row in users_df.iterrows():
        credentials["usernames"][row["username"]] = {
            "name": row["name"],
            "password": row["password"]  # <- must be bcrypt hash
        }

    return stauth.Authenticate(
        credentials,
        cookie_name="portfolio_app",
        key="portfolio_key",
        cookie_expiry_days=30
    )


def login_and_get_status():
    authenticator = get_authenticator()

    # Only show login if not already authenticated
    auth_status = st.session_state.get("auth_status")
    username = st.session_state.get("username")

    if auth_status is not True:
        # ✅ LOGIN FORM IN SIDEBAR
        auth_status = authenticator.login("sidebar")
        username = st.session_state.get("username")

        if auth_status:
            st.session_state.auth_status = auth_status
            st.session_state.username = username

    if auth_status:
        if username:
            name = authenticator.credentials["usernames"][username]["name"]
            st.sidebar.success(f"Logged in as {name}")
        if authenticator.logout("Logout", "sidebar"):
            for key in ["auth_status", "username"]:
                st.session_state.pop(key, None)
            st.experimental_rerun()
    elif auth_status is False:
        st.sidebar.error("Incorrect username/password.")
    else:
        st.sidebar.info("Please log in.")

    return authenticator, auth_status, username



