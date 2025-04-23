import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative

from auth import login_and_get_status
from utils.plots import (
    plot_portfolio_dashboard,
    plot_gauge_charts
)
from utils.glossary import chart_with_tooltip, add_info_icon, section_heading, inject_tooltip_css

# --- Plot Functions ---
def plot_asset_cumulative_returns(price_data: pd.DataFrame,
                                  selected_assets: list,
                                  portfolio_df: pd.DataFrame,
                                  benchmark: str = None,
                                  start=None,
                                  end=None):
    if start and end:
        price_data = price_data.loc[start:end]

    latest_prices = price_data.iloc[-1]
    values = portfolio_df.apply(lambda row: row["Amount"] * latest_prices.get(row["Asset"], 0), axis=1)
    total_value = values.sum()
    weights = {
        row["Asset"]: (row["Amount"] * latest_prices.get(row["Asset"], 0)) / total_value
        for _, row in portfolio_df.iterrows()
        if row["Asset"] in price_data.columns
    }

    returns = price_data.pct_change().fillna(0)
    portfolio_returns = returns[list(weights.keys())].dot(pd.Series(weights))
    cumulative_df = pd.DataFrame({
        "date": price_data.index,
        "Portfolio": (1 + portfolio_returns).cumprod()
    })

    if benchmark and benchmark in price_data.columns:
        benchmark_returns = returns[benchmark]
        cumulative_df[benchmark] = (1 + benchmark_returns).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cumulative_df["date"], y=cumulative_df["Portfolio"], mode="lines", name="Portfolio"))
    if benchmark and benchmark in cumulative_df.columns:
        fig.add_trace(go.Scatter(x=cumulative_df["date"], y=cumulative_df[benchmark], mode="lines", name=benchmark))
    fig.update_layout(title="Cumulative Return", xaxis_title="Date", yaxis_title="Cumulative Return", hovermode="x unified")
    return fig


def plot_portfolio_absolute_value(data, selected_assets, start, end, portfolio_df):
    filtered_data = data[selected_assets].loc[start:end]
    amounts = portfolio_df.set_index("Asset").loc[selected_assets]["Amount"]
    dollar_values = filtered_data.multiply(amounts, axis=1)
    portfolio_value = dollar_values.sum(axis=1)
    df = portfolio_value.reset_index()
    df.columns = ["Date", "Portfolio Value"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Portfolio Value"], mode="lines", name="Portfolio Value", fill="tozeroy"))
    fig.update_layout(title="Portfolio Value Over Time", xaxis_title="Date", yaxis_title="Value ($)", hovermode="x unified")
    return fig

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

@st.cache_resource
def load_data():
    return pd.read_parquet("Portfolio Tuner/App/data/prices.parquet")

def ensure_utc(dt):
    dt = pd.to_datetime(dt)
    return dt if dt.tzinfo else dt.tz_localize("UTC")

data = load_data()
available_assets = data.columns.tolist()

if "dashboard_welcome_seen" not in st.session_state:
    st.session_state.dashboard_welcome_seen = False
if not st.session_state.dashboard_welcome_seen:
    with st.expander("Welcome to Your Crypto Dashboard! 🚀", expanded=True):
        st.markdown("View your portfolio's performance, compare with benchmarks, and explore key metrics.")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Dismiss", key="welcome_dismiss"):
                st.session_state.dashboard_welcome_seen = True
                st.rerun()
        with col2:
            if st.button("Learn More", key="welcome_learn"):
                st.markdown("[Explore the Glossary](/Glossary) or [Portfolio Editor](/pages/1_Portfolio_Editor.py)")
                st.session_state.dashboard_welcome_seen = True
                st.rerun()

if authentication_status:
    portfolio_path = f"Portfolio Tuner/App/portfolios/{username}_portfolio.csv"
    if os.path.exists(portfolio_path):
        portfolio_df = pd.read_csv(portfolio_path)
    else:
        st.warning("No saved portfolio found. Create one in the Portfolio Editor.")
        st.stop()

    selected_assets = portfolio_df["Asset"].dropna().unique().tolist()
    if not selected_assets:
        st.warning("No valid cryptos found in your portfolio.")
        st.stop()

    latest_prices = data.loc[data.index.max()]
    values = portfolio_df.apply(lambda row: row["Amount"] * latest_prices.get(row["Asset"], 0), axis=1)
    total_value = values.sum()
    portfolio_df["Value ($)"] = values
    portfolio_df["Allocation (%)"] = (values / total_value * 100).round(2)

    st.markdown("### 💰 Portfolio Health")
    col1, col2, col3 = st.columns(3)
    with col1:
        #st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Value", f"${total_value:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        #st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Top Crypto", portfolio_df["Asset"].iloc[0])
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        #st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Diversification", f"{len(portfolio_df)} Coins")
        st.markdown('</div>', unsafe_allow_html=True)
    if len(portfolio_df) < 3:
        st.warning("⚠️ Add more coins for better diversification.")

    #st.markdown("### 📅 Date Range & Benchmark")
    section_heading("📅 Date Range & Benchmark", short_description="Mother always said comparison is the thief of joy, but in finance benchmarks help us contextualize our investments! Here we can pick ours", level=3)
 
    col1, col2 = st.columns([1, 1])
    with col1:
        preset_range = st.selectbox("Quick Date Range", ["Last 30 days", "Last 90 days", "Last 1 year", "Custom"], index=1, key="preset_range")
    with col2:
        if "selected_benchmark" not in st.session_state:
            st.session_state.selected_benchmark = "BTC" if "BTC" in available_assets else "None"
        st.selectbox("Benchmark", options=["None"] + available_assets,
                     index=(available_assets.index(st.session_state.selected_benchmark) + 1)
                     if st.session_state.selected_benchmark in available_assets else 0,
                     key="selected_benchmark")

    max_date = data.index.max()
    min_date = data.index.min()
    try:
        if preset_range == "Custom":
            default_start = max_date - pd.Timedelta(days=90)
            date_range = st.date_input("Custom Range", value=(default_start, max_date), min_value=min_date, max_value=max_date, key="custom_range")
            if len(date_range) == 2:
                start_date, end_date = map(ensure_utc, date_range)
            else:
                st.warning("Please select a valid date range.")
                st.stop()
        else:
            days = {"Last 30 days": 30, "Last 90 days": 90, "Last 1 year": 365}[preset_range]
            end_date = max_date
            start_date = max_date - pd.Timedelta(days=days)
    except Exception as e:
        st.error(f"⚠️ Error processing dates: {e}")
        st.stop()

    try:
        start_date = start_date.tz_convert("UTC") if isinstance(start_date, pd.Timestamp) and start_date.tzinfo else pd.Timestamp(start_date, tz="UTC")
        end_date = end_date.tz_convert("UTC") if isinstance(end_date, pd.Timestamp) and end_date.tzinfo else pd.Timestamp(end_date, tz="UTC")
    except Exception as e:
        st.error(f"⚠️ Invalid date format: {e}")
        st.stop()

    if start_date >= end_date:
        st.error("🚫 Start date must be before end date.")
        st.stop()
    if start_date < min_date or end_date > max_date:
        st.error("🚫 Selected dates are outside available data range.")
        st.stop()

    benchmark = st.session_state.selected_benchmark if st.session_state.selected_benchmark != "None" else None

    try:
        metrics_fig, heatmap_fig = plot_portfolio_dashboard(data, selected_assets, portfolio_df=portfolio_df, date_range=(start_date, end_date), benchmark=benchmark)
    except Exception as e:
        st.error(f"⚠️ Error in dashboard: {e}")
        st.stop()

    #st.markdown("### 📊 Portfolio Performance")
    section_heading("📊 Portfolio Performance", short_description="This is it! How our portfolio has been doing this past while", level=3)
 
    try:
        value_fig = plot_portfolio_absolute_value(data, selected_assets, start_date, end_date, portfolio_df)
        returns_fig = plot_asset_cumulative_returns(data, selected_assets, portfolio_df, benchmark, start_date, end_date)
        st.plotly_chart(value_fig, use_container_width=True)
        st.plotly_chart(returns_fig, use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ Error in performance charts: {e}")
        st.info("Try selecting a shorter date range or verifying your assets.")

    section_heading("🧫 Key Metrics", term="Understand Your Portfolio", short_description="We want to know how much risk we are taking (volatility), what's our reward (returns), and what's the bang for buck (Sharpe).", glossary_url="/Glossary#sharpe-ratio", level=3)
    #st.info("ℹ️ Sharpe Ratio: Higher means better returns for the risk taken.")

    if metrics_fig and len(metrics_fig) >= 3:
        row = st.columns(3)
        for col, fig in zip(row, metrics_fig[:3]):
            with col:
                st.plotly_chart(fig, use_container_width=True)
        with st.expander("📊 Advanced Metrics"):
            row2 = st.columns(3)
            for col, fig in zip(row2, metrics_fig[3:]):
                with col:
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Incomplete metrics data.")


    st.markdown("### 🔍 Portfolio Insights")
    col1, col2 = st.columns([1, 1])
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
        pie_data = portfolio_df.set_index("Asset")["Value ($)"].reindex(selected_assets).fillna(0)
        fig = px.pie(values=pie_data.values, names=pie_data.index, title="Portfolio Allocation", color_discrete_sequence=qualitative.Safe)
        chart_with_tooltip(
            title="Portfolio Allocation",
            term="Your Crypto Mix",
            short_desc="This is where the eggs go into the baskets!",
            glossary_url="/Glossary#allocation",
            chart_func=lambda: fig
        )

    # --- Full Historical Data Plots in 2-column format ---
    st.markdown("### 📊 Historical Trends")
    try:
        combined_data = data[selected_assets].loc[start_date:end_date]
        portfolio_data = None
        if "Amount" in portfolio_df.columns:
            valid_assets = [a for a in portfolio_df["Asset"] if a in combined_data.columns]
            amounts = portfolio_df.set_index("Asset").loc[valid_assets, "Amount"].fillna(0)
            weights = amounts / amounts.sum()
            portfolio_series = (combined_data[valid_assets] * weights).sum(axis=1)
            portfolio_series.name = "Portfolio"
            combined_data = combined_data.join(portfolio_series)

        returns = combined_data.pct_change().fillna(0)
        cumulative = (1 + returns).cumprod()
        downsample_interval = max(1, len(cumulative) // 365)

        col1, col2 = st.columns(2)

        with col1:
            fig_cum = go.Figure()
            for col in cumulative.columns:
                fig_cum.add_trace(go.Scatter(x=cumulative.index[::downsample_interval], y=cumulative[col].iloc[::downsample_interval], mode='lines', name=col, line=dict(dash='dash') if col == "Portfolio" else dict()))
            fig_cum.update_layout(title="Cumulative Returns", xaxis_title="Date", yaxis_title="Return", hovermode="x unified")
            #st.plotly_chart(fig_cum, use_container_width=True)
            chart_with_tooltip(
                title="Cumulative Returns",
                short_desc="What if you invested $1 at the beginning of selected period into each coin? This is where we see who's pulling their own weight!",
                chart_func=lambda: fig_cum
            )

        with col2:
            fig_ret = go.Figure()
            for col in returns.columns:
                fig_ret.add_trace(go.Scatter(x=returns.index[::downsample_interval], y=returns[col].iloc[::downsample_interval]*100, mode='lines', name=col))
            fig_ret.update_layout(title="Daily Returns", xaxis_title="Date", yaxis_title="Return (%)", hovermode="x unified")
            #st.plotly_chart(fig_ret, use_container_width=True)
            chart_with_tooltip(
                title="Daily Returns",
                short_desc="Here we see the daily action of each coin!",
                chart_func=lambda: fig_ret
            )

        col3, col4 = st.columns(2)
        with col3:
            fig_price = go.Figure()
            for col in combined_data.columns:
                fig_price.add_trace(go.Scatter(x=combined_data.index[::downsample_interval], y=combined_data[col].iloc[::downsample_interval], mode='lines', name=col))
            fig_price.update_layout(title="Raw Prices", xaxis_title="Date", yaxis_title="Price", hovermode="x unified")
            #st.plotly_chart(fig_price, use_container_width=True)
            chart_with_tooltip(
                title="Raw Prices",
                short_desc="This is the price of each coin -- in full glory!",
                chart_func=lambda: fig_price
            )
        with st.expander("📊 Individual Crypto Performance"):
            try:
                rows = st.columns(2)
                for i, asset in enumerate(selected_assets):
                    col = rows[i % 2]
                    with col:
                        fig = px.line(data.loc[start_date:end_date], y=asset, title=f"{asset} Price History")
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ Error in historical charts: {e}")
                st.info("Try selecting a shorter date range or different assets.")
    except Exception as e:
        st.error(f"⚠️ Error in historical plots: {e}")

    st.markdown("---")
    if st.button("🔙 Portfolio Editor"):
        st.switch_page("pages/1_Portfolio_Editor.py")

else:
    st.warning("🔐 Please log in to view your portfolio.")
