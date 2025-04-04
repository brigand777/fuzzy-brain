import streamlit as st
import pandas as pd

def get_backtest_settings(available_dates):
    st.sidebar.header("Backtest Settings")
    # Determine the min and max dates from the available_dates index
    min_date = available_dates.min().date()
    max_date = available_dates.max().date()
    
    # Use the date_input widget to let the user select a date range.
    date_range = st.sidebar.date_input(
        "Select Backtest Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Validate the selected range.
    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.error("Please select a valid date range.")
        st.stop()
    
    # Convert the Python date objects to pandas Timestamps.
    start_date, end_date = map(pd.to_datetime, date_range)
    
    # Convert to the timezone of the available_dates if needed.
    # available_dates.tz returns None if tz-naive, otherwise the tz info.
    tz = available_dates.tz
    if tz is not None:
        # If the start_date and end_date are tz-naive, localize them.
        if start_date.tzinfo is None:
            start_date = start_date.tz_localize(tz)
        else:
            start_date = start_date.astimezone(tz)
        if end_date.tzinfo is None:
            end_date = end_date.tz_localize(tz)
        else:
            end_date = end_date.astimezone(tz)
    
    if start_date >= end_date:
        st.error("The start date must be before the end date.")
        st.stop()
        
    lookback_days = st.sidebar.number_input("Lookback Period (days) for Reoptimization", min_value=30, value=30, step=1)
    rebalance_days = st.sidebar.number_input("Rebalance Every N Days", min_value=1, value=30, step=1)
    nonnegative_toggle = st.sidebar.checkbox("Constrain MVO to Nonnegative Holdings", value=True)
    
    return start_date, end_date, lookback_days, rebalance_days, nonnegative_toggle

def get_asset_selection(data):
    coins = data.columns.tolist()
    selected_coins = st.sidebar.multiselect("Select Assets", coins, default=coins[:3])
    if not selected_coins:
        st.error("Please select at least one asset.")
        st.stop()
    return selected_coins

def get_optimization_methods(initial_allocations):
    available_methods = list(initial_allocations.keys())
    selected_methods = st.sidebar.multiselect("Select Optimization Methods", available_methods, default=available_methods)
    if not selected_methods:
        st.error("Please select at least one optimization method.")
        st.stop()
    return selected_methods
