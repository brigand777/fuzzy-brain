import streamlit as st
import pandas as pd
import os
from utils.plots import plot_portfolio_allocation_3d
from io import StringIO

def edit_portfolio(available_assets, prices: pd.DataFrame, persistent=True):
    if "editable_portfolio" not in st.session_state:
        st.session_state.editable_portfolio = pd.DataFrame(columns=["Asset", "Amount"])

    df = st.session_state.editable_portfolio.copy()

    if "show_edit" not in st.session_state:
        st.session_state.show_edit = False

    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "Absolute"

    # --- Upload portfolio ---
    st.markdown("### 📤 Upload Portfolio CSV")
    uploaded_file = st.file_uploader("Upload a CSV file with columns: Asset,Amount", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            if set(["Asset", "Amount"]).issubset(uploaded_df.columns):
                st.session_state.editable_portfolio = uploaded_df[["Asset", "Amount"]].drop_duplicates().reset_index(drop=True)
                if persistent and st.session_state.get("auth_status") and st.session_state.get("username"):
                    username = st.session_state["username"]
                    file_path = f"Portfolio Tuner/App/portfolios/{username}_portfolio.csv"
                    os.makedirs("Portfolio Tuner/App/portfolios", exist_ok=True)
                    uploaded_df[["Asset", "Amount"]].to_csv(file_path, index=False)
                    st.success("✅ Portfolio uploaded and saved.")
                    st.rerun()
                else:
                    st.success("✅ Portfolio uploaded for this session.")
                    st.rerun()
            else:
                st.error("CSV must contain 'Asset' and 'Amount' columns.")
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")

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

        # --- Download portfolio button ---
        csv = df[["Asset", "Amount"]].to_csv(index=False)
        st.download_button("📥 Download Portfolio CSV", csv, "portfolio.csv", "text/csv")

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
                        df["Amount"] *= (1 - pct)
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
                    except Exception as e:
                        st.error(f"❌ Failed to save portfolio: {e}")
                else:
                    st.toast("⚠️ Changes saved for session only (not persistent).")
                st.rerun()

    return st.session_state.editable_portfolio
