import numpy as np
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fitter import Fitter
from scipy.stats import norm, t, johnsonsu

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fitter import Fitter
from scipy.stats import norm, t, johnsonsu

def run_smart_monte_carlo_simulation(weights, price_data, horizon_days=180, n_sims=1000):
    # --- Compute portfolio returns ---
    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    weights_array = np.array([weights.get(asset, 0) for asset in price_data.columns])
    portfolio_returns = log_returns.dot(weights_array)

    # --- Fit best distribution using fitter ---
    f = Fitter(portfolio_returns.values, distributions=['norm', 't', 'johnsonsu'], timeout=5)
    f.fit()
    best_dist_name = list(f.get_best().keys())[0]
    best_params = f.fitted_param[best_dist_name]

    # --- Simulate future returns ---
    dist_map = {'norm': norm, 't': t, 'johnsonsu': johnsonsu}
    dist = dist_map[best_dist_name]
    sim_returns = dist.rvs(*best_params, size=(horizon_days, n_sims))
    simulated_paths = np.cumprod(1 + sim_returns, axis=0)

    # --- Prepare DataFrame ---
    df = pd.DataFrame(simulated_paths)
    df.index.name = "Day"
    df["mean"] = df.mean(axis=1)
    df["ci_high"] = df.quantile(0.90, axis=1)
    df["ci_low"] = df.quantile(0.10, axis=1)

    # --- Plotly Interactive Chart ---
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["ci_high"], name="90% Confidence Upper",
        line=dict(color="lightblue", dash="dot"),
        hovertemplate="Day %{x}<br>90% Upper: %{y:.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df["ci_low"], name="10% Confidence Lower",
        line=dict(color="lightblue", dash="dot"),
        fill="tonexty", fillcolor="rgba(173,216,230,0.2)",
        hovertemplate="Day %{x}<br>10% Lower: %{y:.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df["mean"], name="Mean Path",
        line=dict(color="blue"),
        hovertemplate="Day %{x}<br>Mean: %{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title="Monte Carlo Portfolio Forecast",
        xaxis_title="Day",
        yaxis_title="Portfolio Value (Indexed)",
        hovermode="x unified",  # <- this mimics Altair-style vertical hover rule
        template="plotly_white"
    )

    return {
        "chart": fig,
        "ci_low": df["ci_low"].iloc[-1] - 1,
        "ci_high": df["ci_high"].iloc[-1] - 1,
        "min": df.drop(columns=["mean", "ci_high", "ci_low"]).iloc[-1].min() - 1,
        "max": df.drop(columns=["mean", "ci_high", "ci_low"]).iloc[-1].max() - 1,
        "distribution_used": best_dist_name
    }
