# components/portfolio_input.py

import streamlit as st
import pandas as pd
import os
from utils.plots import plot_portfolio_allocation_3d

def edit_portfolio(available_assets, persistent=True):
    if "editable_portfolio" not in st.session_state:
        st.session_state.editable_portfolio = pd.DataFrame(columns=["Asset", "Amount"])

    df = st.session_state.editable_portfolio.copy()

    if "show_edit" not in st.session_state:
        st.session_state.show_edit = False

    # --- Display table and chart ---
    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.markdown("### Your Portfolio")
        st.dataframe(df.style.format(precision=4), use_container_width=True, height=300)
    with col2:
        plot_portfolio_allocation_3d(df)

    # --- Toggle edit ---
    if st.button("✏️ Edit My Holdings"):
        st.session_state.show_edit = not st.session_state.show_edit

    if st.session_state.show_edit:
        st.markdown("### Manage Your Portfolio")

        with st.form("add_asset_form"):
            col1, col2 = st.columns([2, 1])
            with col1:
                asset = st.selectbox("Select Asset", options=sorted(available_assets))
            with col2:
                amount = st.number_input("Amount", min_value=0.0, step=0.01, format="%.4f")
            submitted = st.form_submit_button("Add / Update Asset")
            if submitted:
                if asset in df["Asset"].values:
                    df.loc[df["Asset"] == asset, "Amount"] = amount
                else:
                    df = pd.concat([df, pd.DataFrame([[asset, amount]], columns=["Asset", "Amount"])], ignore_index=True)
                df = df.drop_duplicates(subset="Asset", keep="last").reset_index(drop=True)
                st.session_state.editable_portfolio = df

                # Save only if allowed
                if persistent and st.session_state.get("auth_status") and st.session_state.get("username"):
                    username = st.session_state["username"]
                    os.makedirs("portfolios", exist_ok=True)
                    pd.DataFrame(df).to_csv(f"Portfolio Tuner/App/portfolios/{username}_portfolio.csv", index=False)
                st.rerun()

        if not df.empty:
            selected_to_delete = st.multiselect("Select rows to delete", df["Asset"].tolist(), key="delete_selection")
            if st.button("Delete Selected"):
                df = df[~df["Asset"].isin(selected_to_delete)].reset_index(drop=True)
                st.session_state.editable_portfolio = df

                if persistent and st.session_state.get("auth_status") and st.session_state.get("username"):
                    username = st.session_state["username"]
                    pd.DataFrame(df).to_csv(f"Portfolio Tuner/App/portfolios/{username}_portfolio.csv", index=False)
                st.rerun()
        else:
            st.info("No assets in your portfolio yet.")

    return st.session_state.editable_portfolio
