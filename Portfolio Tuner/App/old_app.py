import streamlit as st
import pandas as pd
import altair as alt
from datetime import timedelta
from auth import register_user, get_authenticator
# Import functions from our modules.
from optimizer import run_optimizers  
from utils import dynamic_backtest_portfolio  
from user_input import get_backtest_settings, get_asset_selection, get_optimization_methods
from plots import (
    plot_cumulative_returns, 
    plot_rolling_sharpe, 
    plot_drawdowns, 
    plot_allocations_per_method,
    plot_asset_returns, 
    plot_asset_prices,
    pie_chart_allocation,
    add_interactivity
)
from video_utils import display_video
st.set_page_config(page_title="Crypto Portfolio Optimizer", layout="wide")
# --- Load Custom Font ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)


st.title("Portfolio Tuner")
st.markdown("<h3 style='font-size:20px; font-style:italic; color:#A9A9B3;'>Optimize Your Crypto, Maximize Your Gains.</h3>", unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Home", "Dashboard"])

if page == "Home":
    
    # Display the video from your assets folder. Adjust the height as desired.
    display_video("assets/homepage_video.mp4", height=600)
    st.info("Select 'Dashboard' from the sidebar to begin exploring the optimization tools.")
    
    st.markdown("## Welcome to the Crypto Portfolio Optimizer")
    st.write("""
        This platform is designed to help retail crypto investors manage risk 
        through advanced portfolio optimization techniques and interactive visualizations.
    """)
    
elif page == "Dashboard":
    # Load merged price data from Parquet.
    @st.cache_data
    def load_data():
        return pd.read_parquet("Data/prices.parquet")  # Ensure path matches your repo

    data = load_data()
    available_dates = data.index.sort_values()

    # Get user inputs from the sidebar.
    start_date, end_date, lookback_days, rebalance_days, nonnegative_toggle = get_backtest_settings(available_dates)
    selected_coins = get_asset_selection(data)
    data = data[selected_coins]

    # Define simulation data based on the selected dates.
    simulation_data = data.loc[start_date:end_date]
    if simulation_data.empty:
        st.error("No data available for the selected backtest period.")
        st.stop()

    # ----- Underlying Asset Plots: Plot Assets Button -----
    st.markdown("## Underlying Asset Data")
    plot_assets_button = st.button("Plot Assets")
    price_scale_option = st.sidebar.radio("Select Price Scale", ("Linear", "Log"))

    if plot_assets_button:
        st.markdown("### Daily Returns by Asset")
        base_returns_chart = plot_asset_returns(simulation_data, selected_coins)
        interactive_returns_chart = add_interactivity(base_returns_chart, x_field="date", y_field="Daily Return (%)")
        st.altair_chart(interactive_returns_chart, use_container_width=True)
        
        st.markdown("### Asset Prices")
        base_prices_chart = plot_asset_prices(simulation_data, selected_coins, log_scale=(price_scale_option=="Log"))
        interactive_prices_chart = add_interactivity(base_prices_chart, x_field="date", y_field="Price")
        st.altair_chart(interactive_prices_chart, use_container_width=True)

    # ----- Dynamic Backtesting -----
    st.markdown("## Dynamic Backtest Results")
    optimize_button = st.button("Optimize Portfolio")

    if optimize_button:
        try:
            # Compute initial allocations using the lookback window before the start date.
            lookback_window = data.loc[pd.to_datetime(start_date) - pd.Timedelta(days=lookback_days):start_date]
            initial_allocations = run_optimizers(lookback_window, nonnegative_mvo=nonnegative_toggle)

            # Let the user select which optimization methods to include.
            selected_methods = get_optimization_methods(initial_allocations)

            st.markdown("### Initial Allocations (Pie Charts)")
            pie_charts = []
            for method in selected_methods:
                chart = pie_chart_allocation(initial_allocations[method].round(4), method)
                pie_charts.append(chart)
            st.altair_chart(alt.hconcat(*pie_charts), use_container_width=True)

            st.write(f"Backtest period: {pd.to_datetime(start_date).date()} to {pd.to_datetime(end_date).date()}")
            st.write(f"Rebalance Frequency: Every {rebalance_days} days")
            st.write(f"Dynamic reoptimization uses the past {lookback_days} days of data with exponential weighting.")

            # Run dynamic backtest for each selected optimization method.
            results_dict = {}
            for method in selected_methods:
                res = dynamic_backtest_portfolio(simulation_data, method, lookback_days, rebalance_days, nonnegative_toggle)
                results_dict[method] = res

            st.markdown("### Cumulative Returns (starting at 0)")
            base_cumul_chart = plot_cumulative_returns(results_dict)
            interactive_cumul_chart = add_interactivity(base_cumul_chart, x_field="date", y_field="cumulative")
            st.altair_chart(interactive_cumul_chart, use_container_width=True)

            st.markdown("### Rolling Annualized Sharpe Ratio")
            base_sharpe_chart = plot_rolling_sharpe(results_dict)
            interactive_sharpe_chart = add_interactivity(base_sharpe_chart, x_field="date", y_field="rolling_sharpe")
            st.altair_chart(interactive_sharpe_chart, use_container_width=True)

            st.markdown("### Rolling Maximum Drawdown")
            base_drawdown_chart = plot_drawdowns(results_dict)
            interactive_drawdown_chart = add_interactivity(base_drawdown_chart, x_field="date", y_field="drawdown")
            st.altair_chart(interactive_drawdown_chart, use_container_width=True)

            st.markdown("### Dynamic Asset Allocations Per Method")
            for method in selected_methods:
                st.altair_chart(plot_allocations_per_method(results_dict[method]["allocations"], method), use_container_width=True)

            st.markdown("### Summary Metrics by Method")
            for method, res in results_dict.items():
                st.subheader(method)
                st.write(f"**Final Annualized Sharpe Ratio:** {res['sharpe']:.2f}")
                st.write(f"**Maximum Drawdown:** {res['drawdown']:.2%}")

        except Exception as e:
            st.error("An error occurred during dynamic backtesting. Underlying asset plots are still displayed.")
            st.error(f"Error details: {e}")
    else:
        st.info("Click the 'Optimize Portfolio' button to run the dynamic backtest and view optimization results.")
