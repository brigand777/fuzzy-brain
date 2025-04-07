import streamlit as st
import pandas as pd
import os
from utils.plots import plot_portfolio_allocation_3d

def edit_portfolio(available_assets, prices: pd.DataFrame, persistent=True):
    if "editable_portfolio" not in st.session_state:
        st.session_state.editable_portfolio = pd.DataFrame(columns=["Asset", "Amount"])

    if st.session_state.get("portfolio_saved"):
        st.success("✅ Portfolio saved successfully!")
        del st.session_state["portfolio_saved"]

    df = st.session_state.editable_portfolio.copy()

    if "show_edit" not in st.session_state:
        st.session_state.show_edit = False

    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "Absolute"

    # --- DEBUG LOG ---
    st.write("🔍 persistent:", persistent)
    st.write("🔍 auth_status:", st.session_state.get("auth_status"))
    st.write("🔍 username:", st.session_state.get("username"))

    # --- Get latest prices per asset ---
    latest_prices = prices.iloc[-1]

    # --- Compute derived value columns ---
    df["Price"] = df["Asset"].map(latest_prices)
    df["Value"] = df["Amount"] * df["Price"]
    total_value = df["Value"].sum()
    df["Percent"] = df["Value"] / total_value * 100 if total_value > 0 else 0

    # --- Display value summary and table/chart ---
    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.markdown("### Your Portfolio")
        st.markdown(f"**💰 Total Portfolio Value:** `${total_value:,.2f}`")
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
                    if price > 0 and total_value > 0:
                        pct = user_input / 100
                        value_x = pct * total_value
                        amount_x = value_x / price

                        # Scale down existing amounts
                        df["Amount"] *= (1 - pct)

                        # Add or update the new asset
                        if asset in df["Asset"].values:
                            df.loc[df["Asset"] == asset, "Amount"] += amount_x
                        else:
                            df = pd.concat([df, pd.DataFrame([[asset, amount_x]], columns=["Asset", "Amount"])], ignore_index=True)
                    else:
                        st.warning("Invalid price or portfolio value. Cannot add by percentage.")
                        return
                else:
                    amount = user_input
                    if asset in df["Asset"].values:
                        df.loc[df["Asset"] == asset, "Amount"] = amount
                    else:
                        df = pd.concat([df, pd.DataFrame([[asset, amount]], columns=["Asset", "Amount"])], ignore_index=True)

                df = df.drop_duplicates(subset="Asset", keep="last").reset_index(drop=True)
                st.session_state.editable_portfolio = df[["Asset", "Amount"]]

                if persistent and st.session_state.get("auth_status") and st.session_state.get("username"):
                    username = st.session_state["username"]
                    file_path = f"Portfolio Tuner/App/portfolios/{username}_portfolio.csv"
                    os.makedirs("Portfolio Tuner/App/portfolios", exist_ok=True)
                    try:
                        df[["Asset", "Amount"]].to_csv(file_path, index=False)
                        st.session_state["portfolio_saved"] = True
                        st.write("📁 File saved to:", file_path)
                        st.write("🕒 File last modified:", os.path.getmtime(file_path))
                        with open(file_path, "r") as f:
                            content = f.read()
                        st.code(content, language="csv")
                    except Exception as e:
                        st.error(f"❌ Failed to save portfolio: {e}")
                else:
                    st.toast("⚠️ Changes saved for session only (not persistent).")
                st.rerun()

    return st.session_state.editable_portfolio