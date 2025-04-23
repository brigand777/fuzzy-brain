import streamlit as st
import pandas as pd
import os
from utils.plots import plot_portfolio_allocation_3d
from io import StringIO
import plotly.express as px
from utils.glossary import chart_with_tooltip, add_info_icon, section_heading, inject_tooltip_css
def edit_portfolio(available_assets, prices: pd.DataFrame, persistent=True):
    import os
    import pandas as pd
    import streamlit as st
    import plotly.express as px

    if "editable_portfolio" not in st.session_state:
        st.session_state.editable_portfolio = pd.DataFrame(columns=["Asset", "Amount"])

    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "Absolute"

    if "portfolio_base_value" not in st.session_state:
        st.session_state.portfolio_base_value = 10000.0

    df = st.session_state.editable_portfolio.copy()
    latest_prices = prices.iloc[-1]

    # --- Upload Section ---
    with st.expander("📤 Upload Portfolio CSV"):
        uploaded_file = st.file_uploader("Upload CSV with columns: `Asset`, `Amount`", type=["csv"])
        if uploaded_file:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                if {"Asset", "Amount"}.issubset(uploaded_df.columns):
                    df = uploaded_df[["Asset", "Amount"]].dropna().drop_duplicates()
                    st.session_state.editable_portfolio = df
                    if persistent and st.session_state.get("auth_status"):
                        username = st.session_state["username"]
                        path = f"Portfolio Tuner/App/portfolios/{username}_portfolio.csv"
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        df.to_csv(path, index=False)
                        st.success("✅ Uploaded and saved!")
                    else:
                        st.success("✅ Uploaded (session only)")
                    st.rerun()
                else:
                    st.error("CSV must contain both 'Asset' and 'Amount' columns.")
            except Exception as e:
                st.error(f"Upload error: {e}")

    # --- Display Portfolio ---
    df["Price"] = df["Asset"].map(latest_prices)
    df["Value"] = df["Amount"] * df["Price"]
    total_value = df["Value"].sum()
    df["Percent"] = df["Value"] / total_value * 100 if total_value > 0 else 0

    section_heading("📊 Your Portfolio Overview", short_description="The left chart shows us how much of each coin we have, and the right shows us the exact percentage of the total value of our portfolio is in each coin. Easy as pie!", level=3)
 
    col_table, col_chart = st.columns([1, 1.2])

    with col_table:
        st.markdown(f"**💼 Total Value:** `${total_value:,.2f}`")
        st.dataframe(
            df[["Asset", "Amount", "Price", "Value", "Percent"]].style.format({
                "Amount": "{:.4f}", "Price": "${:.2f}", "Value": "${:,.2f}", "Percent": "{:.2f}%"
            }),
            use_container_width=True
        )

        csv_download = df[["Asset", "Amount"]].to_csv(index=False)
        st.download_button("📥 Download CSV", csv_download, "portfolio.csv", "text/csv")

    with col_chart:
        df_sorted = df.sort_values("Percent", ascending=False)
        max_percent = df_sorted["Percent"].max()
        y_axis_max = max_percent * 1.15  # Add 15% padding to the tallest bar

        fig = px.bar(
            df_sorted,
            x="Asset",
            y="Percent",
            title="📊 Allocation Breakdown",
            text="Percent",
            labels={"Percent": "Allocation (%)", "Asset": "Token"},
            color="Asset",
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            xaxis_title="Asset",
            yaxis_title="Portfolio Allocation (%)",
            yaxis=dict(range=[0, y_axis_max]),
            uniformtext_minsize=8,
            uniformtext_mode="hide",
            height=350,
            margin=dict(t=40, b=30, l=10, r=10)
        )

        st.plotly_chart(fig, use_container_width=True)

    # --- Edit Portfolio ---
    st.markdown("### ✏️ Modify Portfolio Holdings")
    col1, col2 = st.columns([3, 1])

    with col1:
        asset = st.selectbox("Select Asset to Add/Update", options=sorted(available_assets), placeholder="Search or select asset...")


    with col2:
        input_mode = st.radio("Input Mode", ["Absolute", "Percentage"], key="portfolio_input_mode", horizontal=True)


    label = "Amount" if input_mode == "Absolute" else "% of Portfolio"
    user_input = st.number_input(label, min_value=0.0, step=0.01, format="%.4f")

    if st.button("➕ Add / Update Asset"):
        price = latest_prices.get(asset, 0)
        base_value = st.session_state.portfolio_base_value

        if price <= 0 or base_value <= 0:
            st.warning("❌ Invalid price or base value.")
        else:
            existing = df[df["Asset"] == asset]
            remaining_df = df[df["Asset"] != asset]

            if input_mode == "Percentage":
                new_value = (user_input / 100) * base_value
                new_amount = new_value / price

                if not remaining_df.empty:
                    remaining_total = remaining_df["Value"].sum()
                    if remaining_total > 0:
                        scale = (base_value - new_value) / remaining_total
                        remaining_df["Amount"] *= scale

                updated_df = pd.concat([
                    remaining_df,
                    pd.DataFrame([[asset, new_amount]], columns=["Asset", "Amount"])
                ])
            else:  # Absolute mode
                new_amount = user_input
                updated_df = pd.concat([
                    remaining_df,
                    pd.DataFrame([[asset, new_amount]], columns=["Asset", "Amount"])
                ])

            st.session_state.editable_portfolio = updated_df.drop_duplicates(subset="Asset").reset_index(drop=True)

            if persistent and st.session_state.get("auth_status"):
                username = st.session_state["username"]
                path = f"Portfolio Tuner/App/portfolios/{username}_portfolio.csv"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                st.session_state.editable_portfolio.to_csv(path, index=False)
                st.success("✅ Saved changes.")
            else:
                st.toast("✅ Updated (not saved).")
            st.rerun()

    # --- Delete Asset ---
    st.markdown("### 🗑️ Delete Asset from Portfolio")
    delete_asset = st.selectbox("Select Asset to Delete", options=df["Asset"].tolist(), key="delete_asset")
    if st.button("❌ Delete Selected Asset"):
        df = df[df["Asset"] != delete_asset]
        st.session_state.editable_portfolio = df.reset_index(drop=True)

        if persistent and st.session_state.get("auth_status"):
            username = st.session_state["username"]
            path = f"Portfolio Tuner/App/portfolios/{username}_portfolio.csv"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            st.success("🗑️ Deleted and saved.")
        else:
            st.toast("🗑️ Deleted (not saved).")
        st.rerun()

    # --- Rescale Section ---
    with st.expander("🧮 Rescale Portfolio"):
        suggested = round(total_value, 2)
        rescale_val = st.number_input(
            "Target Total Value ($)",
            min_value=0.0,
            value=float(suggested),
            step=0.01
        )

        if st.button("Rescale"):
            if total_value > 0:
                factor = rescale_val / total_value
                df["Amount"] *= factor
                st.session_state.portfolio_base_value = rescale_val
                st.session_state.editable_portfolio = df[["Asset", "Amount"]]
                if persistent and st.session_state.get("auth_status"):
                    path = f"Portfolio Tuner/App/portfolios/{username}_portfolio.csv"
                    df[["Asset", "Amount"]].to_csv(path, index=False)
                st.success("📏 Rescaled portfolio.")
                st.rerun()
            else:
                st.warning("⚠️ Total value must be greater than 0.")

    return st.session_state.editable_portfolio
