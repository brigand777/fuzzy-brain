import streamlit as st
import pandas as pd
import os

from auth import login_and_get_status
from utils.plots import (
    plot_portfolio_dashboard,
    plot_historical_assets,
    plot_asset_cumulative_returns,
    plot_gauge_charts,
    plot_portfolio_absolute_value
)
from utils.glossary import chart_with_tooltip, add_info_icon, section_heading, inject_tooltip_css

# --- Page Setup ---
st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
inject_tooltip_css()
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,400;1,400&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

authenticator, authentication_status, username = login_and_get_status()

st.title("📊 Portfolio Dashboard")

# --- Load asset data ---
@st.cache_data
def load_data():
    return pd.read_parquet("Portfolio Tuner/App/data/prices.parquet")

def ensure_utc(dt):
    dt = pd.to_datetime(dt)
    return dt if dt.tzinfo else dt.tz_localize("UTC")

data = load_data()
available_assets = data.columns.tolist()

# --- Load saved portfolio ---
if authentication_status:
    portfolio_path = f"Portfolio Tuner/App/portfolios/{username}_portfolio.csv"
    if os.path.exists(portfolio_path):
        portfolio_df = pd.read_csv(portfolio_path)
    else:
        st.warning("No saved portfolio found. Please create one in the Portfolio Editor.")
        st.stop()

    selected_assets = portfolio_df["Asset"].dropna().unique().tolist()

    if not selected_assets:
        st.warning("No valid assets found in your portfolio.")
        st.stop()

    if "selected_benchmark" not in st.session_state:
        st.session_state.selected_benchmark = "BTC" if "BTC" in available_assets else "None"

    # --- Quick Date Range Presets ---
    st.markdown("## 📅 Date Range & Benchmark")

    preset_range = st.selectbox("Quick Date Range:", ["Last 30 days", "Last 90 days", "Last 1 year", "Custom"])

    max_date = data.index.max()
    min_date = data.index.min()

    if preset_range == "Custom":
        default_start = max_date - pd.Timedelta(days=100)
        date_range = st.date_input(
            "Select custom date range:",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date
        )
        start_date, end_date = map(ensure_utc, date_range)
    else:
        days = {"Last 30 days": 30, "Last 90 days": 90, "Last 1 year": 365}[preset_range]
        end_date = max_date
        start_date = max_date - pd.Timedelta(days=days)

    # --- Benchmark selector (moved up here) ---
    st.selectbox(
        "📌 Benchmark for comparison:",
        options=["None"] + available_assets,
        index=(available_assets.index(st.session_state.selected_benchmark) + 1)
        if st.session_state.selected_benchmark in available_assets else 0,
        key="selected_benchmark"
    )

    benchmark = st.session_state.selected_benchmark
    benchmark = benchmark if benchmark != "None" else None

    # --- Dashboard Charts ---
    try:
        metrics_fig, heatmap_fig = plot_portfolio_dashboard(
            data, selected_assets,
            portfolio_df=portfolio_df,
            date_range=(start_date, end_date),
            benchmark=benchmark
        )
    except Exception as e:
        st.error(f"⚠️ Error in plot_portfolio_dashboard: {type(e).__name__} — {e}")
        st.stop()

    # --- Portfolio Value Chart ---
    col1, col2, col3 = st.columns([0.2, 8, 1])
    with col2:
        chart_with_tooltip(
            title="Portfolio Value Over Time",
            short_desc="This is it folks!\n How much is your investment worth?\n No fancy stuff — just the money $".replace("\n", "<br>"),
            glossary_url="6_Glossary.py#cumulative-return",
            chart_func=plot_portfolio_absolute_value,
            data=data,
            selected_assets=selected_assets,
            start=start_date,
            end=end_date,
            portfolio_df=portfolio_df
        )

    # --- Portfolio Metrics Section ---
    section_heading(
        "🧭 Portfolio Metrics",
        term="Here's where we get technical!",
        short_description="We want to know how much risk we are taking (volatility), what's our reward (returns), and what's the bang for buck (Sharpe).",
        glossary_url="/Glossary#sharpe-ratio",
        level=3
    )

    if metrics_fig and len(metrics_fig) == 6:
        row = st.columns(3)
        for col, fig in zip(row, metrics_fig[:3]):
            with col:
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("📐 Show Advanced Metrics"):
            row2 = st.columns(3)
            for col, fig in zip(row2, metrics_fig[3:]):
                with col:
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Expected 6 metrics for layout, but received a different number.")

    # --- Comparison Charts ---
    st.markdown("### 🔍 Portfolio Comparison")
    col1, col2 = st.columns(2)

    with col1:
        chart_with_tooltip(
            title="Correlation Heatmap",
            term="'Don't put all your eggs in one basket'",
            short_desc="""— well, this is the basket! Higher correlations
            mean the baskets are more similar, negative correlations mean they move oppositely, and close to 0 means they're truly distinct!""",
            glossary_url="/Glossary#correlation-heatmap",
            chart_func=lambda: heatmap_fig
        )

    with col2:
        chart_with_tooltip(
            title="Cumulative Return vs. Benchmark",
            term="Benchmark Comparison",
            short_desc="Mom always said comparison is the thief of joy, but in finance it's how we contextualize our investments. Check it out!",
            glossary_url="/Glossary#benchmark",
            chart_func=plot_asset_cumulative_returns,
            price_data=data,
            selected_assets=selected_assets,
            benchmark=benchmark,
            start=start_date,
            end=end_date,
            portfolio_df=portfolio_df
        )

    # --- Optional historical charts toggle ---
    if "show_plot" not in st.session_state:
        st.session_state.show_plot = False

    if st.button("📊 Show Historical Asset Performance"):
        st.session_state.show_plot = not st.session_state.show_plot

    if st.session_state.show_plot:
        try:
            plot_historical_assets(
                data,
                selected_assets,
                portfolio_df=portfolio_df,
                date_range_default=(start_date, end_date)
            )
        except Exception as e:
            import traceback
            st.error("⚠️ An error occurred in `plot_historical_assets()`")
            st.code(traceback.format_exc())
            st.stop()

    # --- Navigation ---
    st.markdown("---")
    from streamlit.source_util import get_pages
    st.write("All known pages:", [page["page_name"] for page in get_pages("").values()])

    if st.button("🔙 Go to Portfolio Editor"):
        st.switch_page("1_Portfolio_Editor")
        

else:
    st.warning("Please log in to view saved portfolio data.")
