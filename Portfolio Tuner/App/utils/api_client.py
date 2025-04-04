import requests
import streamlit as st

def call_fastapi_optimizer(price_df, asset_weights, lookback_days, nonnegative):
    payload = {
        "assets": asset_weights,
        "price_data": price_df.to_dict(orient="list"),
        "lookback_days": lookback_days,
        "nonnegative": nonnegative
    }
    try:
        response = requests.post("http://localhost:8000/optimize", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Failed to fetch optimizations from backend.")
            return {}
    except Exception as e:
        st.error(f"Error contacting optimization API: {e}")
        return {}
