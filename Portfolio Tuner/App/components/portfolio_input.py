# components/portfolio_input.py

import streamlit as st
import pandas as pd
import os
from utils.plots import plot_portfolio_allocation_3d

def edit_portfolio(available_assets, prices: pd.DataFrame, persistent=True):
    if "editable_portfolio" not in st.session_state:
        st.session_state.editable_portfolio = pd.DataFrame(columns=["Asset", "Amount"])

    df = st.session_state.editable_portfolio.copy()

    if "show_edit" not in st.session_state:
        st.session_state.show_edit = False

    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "Absolute"

    # --- Get latest prices per asset ---
    latest_prices = prices.iloc[-1]

    # --- Compute derived value columns ---
    df["Price"] = df["Asset"].map(latest_prices)
    df["Value"] = df["Amount"] * df["Price"]
    total_value = df["Value"].sum()
    df["Percent"] = df["Value"] / total_value * 100 if total_value > 0 else 0

    # --- Display table and chart ---
    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.markdown("### Your Portfolio")
        st.dataframe(df[["Asset", "Amount", "Price", "Value", "Percent"]].style.format({
            "Amount": "{:.4f}", "Price": "${:.2f}", "Value": "${:.2f}", "Percent": "{:.2f}%"
        }), use_container_width=True, height=300)
    with col2:
        plot_portfolio_allocation_3d(df)

    # --- Toggle edit ---
    if st.button("✏️ Edit My Holdings"):
        st.session_state.show_edit = not st.session_state.show_edit

    if st.session_state.show_edit:
        st.markdown("### Manage Your Portfolio")
        
        st.radio("Select Input Mode:", ["Absolute", "Percentage"], key="input_mode", horizontal=True)

        with st.form("add_asset_form"):
            col1, col2 = st.columns([2, 1])
            with col1:
                asset = st.selectbox("Select Asset", options=sorted(available_assets))
            with col2:
                label = "Amount" if st.session_state.input_mode == "Absolute" else "Portfolio %"
                user_input = st.number_input(label, min_value=0.0, step=0.01, format="%.4f")
            submitted = st.form_submit_button("Add / Update Asset")
            if submitted:
                price = latest_prices.get(asset, 0)
                if st.session_state.input_mode == "Percentage":
                    amount = (user_input / 100) * total_value / price if price > 0 else 0
                else:
                    amount = user_input

                if asset in df["Asset"].values:
                    df.loc[df["Asset"] == asset, "Amount"] = amount
                else:
                    df = pd.concat([df, pd.DataFrame([[asset, amount]], columns=["Asset", "Amount"])], ignore_index=True)

                df = df.drop_duplicates(subset="Asset", keep="last").reset_index(drop=True)
                st.session_state.editable_portfolio = df[["Asset", "Amount"]]

                # Save only if allowed
                if persistent and st.session_state.get("auth_status") and st.session_state.get("username"):
                    username = st.session_state["username"]
                    os.makedirs("Portfolio Tuner/App/portfolios", exist_ok=True)
                    df[["Asset", "Amount"]].to_csv(f"Portfolio Tuner/App/portfolios/{username}_portfolio.csv", index=False)
                st.rerun()

        if not df.empty:
            selected_to_delete = st.multiselect("Select rows to delete", df["Asset"].tolist(), key="delete_selection")
            if st.button("Delete Selected"):
                df = df[~df["Asset"].isin(selected_to_delete)].reset_index(drop=True)
                st.session_state.editable_portfolio = df[["Asset", "Amount"]]

                if persistent and st.session_state.get("auth_status") and st.session_state.get("username"):
                    username = st.session_state["username"]
                    df[["Asset", "Amount"]].to_csv(f"Portfolio Tuner/App/portfolios/{username}_portfolio.csv", index=False)
                st.rerun()
        else:
            st.info("No assets in your portfolio yet.")

    return st.session_state.editable_portfolio
