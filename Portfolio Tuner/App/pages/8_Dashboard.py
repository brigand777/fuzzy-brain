import streamlit as st
import pandas as pd
import os
import plotly.express as px
from plotly.colors import qualitative

from auth import login_and_get_status
from utils.plots import (
    plot_portfolio_dashboard,
    plot_historical_assets,
    plot_asset_cumulative_returns,
    plot_gauge_charts,
    plot_portfolio_absolute_value
)
from utils.gloscience import chart_with_tooltip, add_info_icon, section_heading, inject_tooltip_css

# --- Page Setup ---
st.set_page_config(page_title="Crypto Portfolio Dashboard", layout="wide")
inject_tooltip_css()
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,400;1,400&display=swap" rel="stylesheet">
<style>
    .metric-card {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

authenticator, authentication_status, username = login_and_get_status()
st.title("📈 Crypto Portfolio Dashboard")

# --- Load Asset Data ---
@st.cache_resource
def load_data():
    return pd.read_parquet("Portfolio Tuner/App/data/prices.parquet")

def ensure_utc(dt):
    dt = pd.to_datetime(dt)
    return dt if dt.tzinfo else dt.tz_localize("UTC")

data = load_data()
available_assets = data.columns.tolist()

# --- Welcome Banner ---
if "welcome_seen" not in st.session_state:
    with st.expander("Welcome to Your Crypto Dashboard! 🚀", expanded=True):
        st.markdown("View your portfolio's performance, compare with benchmarks, and explore key metrics.")
        if st.button("Get Started", key="welcome"):
            st.session_state.welcome_seen = True

# --- Load Saved Portfolio ---
if authentication_status:
    portfolio_path = f"Portfolio Tuner/App/portfolios/{username}_portfolio.csv"
    if os.path.exists(portfolio_path):
inject_tooltip_css()        portfolio_df = pd.read_csv(portfolio_path)
    else:
        st.warning("No saved portfolio found. Create one in the Portfolio Editor.")
        st.stop()

    selected_assets = portfolio_df["Asset"].dropna().unique().tolist()
    if not selected_assets:
        st.warning("No valid cryptos found in your portfolio.")
        st.stop()

    # --- Portfolio Health Snapshot ---
    latest_prices = data.loc[data.index.max()]
    values = portfolio_df.apply(lambda row: row["Amount"] * latest_prices.get(row["Asset"], 0), axis=1)
    total_value = values.sum()
    portfolio_df["Value ($)"] = values
    portfolio_df["Allocation (%)"] = (values / total_value * 100).round(2)
    
    st.markdown("### 💰 Portfolio Health")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Value", f"${total_value:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Top Crypto", portfolio_df["Asset"].iloc[0])
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Diversification", f"{len(portfolio_df)} Coins")
        st.markdown('</div>', unsafe_allow_html=True)
    if len(portfolio_df) < 3:
        st.warning("⚠️ Add more coins for better diversification.")

    # --- Date Range & Benchmark ---
    st.markdown("### 📅 Date Range & Benchmark")
    col1, col2 = st.columns([1, 1])
    with col1:
        preset_range = st.selectbox(
            "Quick Date Range",
            ["Last 30 days", "Last 90 days", "Last 1 year", "Custom"],
            index=1,
            key="preset_range"
        )
    with col2:
        if "selected_benchmark" not in st.session_state:
            st.session_state.selected_benchmark = "BTC" if "BTC" in available_assets else "None"
        st.selectbox(
            "Benchmark",
            options=["None"] + available_assets,
            index=(available_assets.index(st.session_state.selected_benchmark) + 1)
            if st.session_state.selected_benchmark in available_assets else 0,
            key="selected_benchmark"
        )

    max_date = data.index.max()
    min_date = data.index.min()
    if preset_range == "Custom":
        default_start = max_date - pd.Timedelta(days=90)
        date_range = st.date_input(
            "Custom Range",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
            key="custom_range"
        )
        if len(date_range) == 2:
            start_date, end_date = map(ensure_utc, date_range)
        else:
            st.warning("Please select a valid date range.")
            st.stop()
    else:
        days = {"Last 30 days": 30, "Last 90 days": 90, "Last 1 year": 365}[preset_range]
        end_date = max_date
        start_date = max_date - pd.Timedelta(days=days)

    # Validate dates as UTC timestamps
    start_date = pd.Timestamp(start_date, tz="UTC")
    end_date = pd.Timestamp(end_date, tz="UTC")
    if start_date >= end_date:
        st.error("🚫 Start date must be before end date.")
        st.stop()
    if start_date < min_date or end_date > max_date:
        st.error("🚫 Selected dates are outside available data range.")
        st.stop()

    benchmark = st.session_state.selected_benchmark if st.session_state.selected_benchmark != "None" else None

    # --- Interactive Dashboard ---
    try:
        metrics_fig, heatmap_fig = plot_portfolio_dashboard(
            data, selected_assets,
            portfolio_df=portfolio_df,
            date_range=(start_date, end_date),
            benchmark=benchmark
        )
    except Exception as e:
        st.error(f"⚠️ Error in dashboard: {e}")
        st.stop()

    st.markdown("### 📊 Portfolio Performance")
    # Combine value and returns into one interactive chart
    try:
        value_fig = plot_portfolio_absolute_value(data, selected_assets, start_date, end_date, portfolio_df)
        returns_fig = plot_asset_cumulative_returns(data, selected_assets, benchmark, start_date, end_date, portfolio_df)
        fig = px.line(title="Portfolio Value & Returns")
        for trace in value_fig.data:
            fig.add_trace(trace)
        for trace in returns_fig.data:
            fig.add_trace(trace)
        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {"label": "Value", "method": "update", "args": [{"visible": [True] + [False] * len(returns_fig.data)}]},
                    {"label": "Returns", "method": "update", "args": [{"visible": [False] * len(value_fig.data) + [True] * len(returns_fig.data)}]}
                ],
                "direction": "down",
                "showactive": True
            }],
            showlegend=True
        )
        chart_with_tooltip(
            title="Portfolio Value & Returns",
            short_desc="Track your portfolio's total value and compare returns against a benchmark like BTC.",
            glossary_url="6_Glossary.py#cumulative-return",
            chart_func=lambda: fig
        )
    except Exception as e:
        st.error(f"⚠️ Error in performance charts: {e}")

    # --- Metrics Section ---
    section_heading(
        "🧭 Key Metrics",
        term="Understand Your Portfolio",
        short_description="Volatility shows risk, returns show gains, and Sharpe measures reward per risk.",
        glossary_url="/Glossary#sharpe-ratio",
        level=3
    )
    st.info("ℹ️ Sharpe Ratio: Higher means better returns for the risk taken.")

    if metrics_fig and len(metrics_fig) >= 3:
        row = st.columns(3)
        for col, fig in zip(row, metrics_fig[:3]):
            with col:
                st.plotly_chart(fig, use_container_width=True)
        with st.expander("📐 Advanced Metrics"):
            row2 = st.columns(3)
            for col, fig in zip(row2, metrics_fig[3:]):
                with col:
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Incomplete metrics data.")

    # --- Comparison Charts ---
    st.markdown("### 🔍 Portfolio Insights")
    col1, col2 = st.columns([1, 1])
    with col1:
        chart_with_tooltip(
            title="Correlation Heatmap",
            term="Diversification Check",
            short_desc="High correlations mean assets move together; low or negative means they're more diverse.",
            glossary_url="/Glossary#correlation-heatmap",
            chart_func=lambda: heatmap_fig
        )
    with col2:
        pie_data = portfolio_df.set_index("Asset")["Value ($)"].reindex(selected_assets).fillna(0)
        fig = px.pie(
            values=pie_data.values,
            names=pie_data.index,
            title="Portfolio Allocation",
            color_discrete_sequence=qualitative.Safe
        )
        chart_with_tooltip(
            title="Portfolio Allocation",
            term="Your Crypto Mix",
            short_desc="See how your investments are split across different cryptocurrencies.",
            glossary_url="/Glossary#allocation",
            chart_func=lambda: fig
        )

    # --- Historical Performance ---
    with st.expander("📊 Historical Crypto Performance"):
        try:
            plot_historical_assets(
                data,
                selected_assets,
                portfolio_df=portfolio_df,
                date_range_default=(start_date, end_date)
            )
        except Exception as e:
            st.error(f"⚠️ Error in historical charts: {e}. Please check your date range or assets.")
            st.info("Try selecting a shorter date range or different assets.")

    # --- Share Feature ---
    if st.button("Share Insights", key="share"):
        st.write(f"My crypto portfolio is worth ${total_value:,.2f} with {len(portfolio_df)} coins! #CryptoInvesting")

    # --- Navigation ---
    st.markdown("---")
    if st.button("🔙 Portfolio Editor"):
        st.switch_page("pages/1_Portfolio_Editor.py")

else:
    st.warning("🔐 Please log in to view your portfolio.")