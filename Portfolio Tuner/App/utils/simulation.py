import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    f = Fitter(portfolio_returns.values, 
               distributions=['norm', 't', 'johnsonsu'],
               timeout=5)
    f.fit()
    best_dist_name = list(f.get_best().keys())[0]
    best_params = f.fitted_param[best_dist_name]

    # --- Simulate future returns based on best fit ---
    if best_dist_name == 'norm':
        sim_returns = norm.rvs(*best_params, size=(horizon_days, n_sims))
    elif best_dist_name == 't':
        sim_returns = t.rvs(*best_params, size=(horizon_days, n_sims))
    elif best_dist_name == 'johnsonsu':
        sim_returns = johnsonsu.rvs(*best_params, size=(horizon_days, n_sims))
    else:
        raise ValueError("Unsupported distribution selected")

    # --- Convert returns to price paths ---
    simulated_paths = np.cumprod(1 + sim_returns, axis=0)
    df = pd.DataFrame(simulated_paths)
    df.index.name = "Day"
    df["mean"] = df.mean(axis=1)
    df["ci_high"] = df.quantile(0.90, axis=1)
    df["ci_low"] = df.quantile(0.10, axis=1)

    # --- Plotly fan chart ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df["ci_high"], name="90% CI", line=dict(color="lightblue")))
    fig.add_trace(go.Scatter(y=df["ci_low"], name="10% CI", fill="tonexty",
                             fillcolor="rgba(173,216,230,0.2)", line=dict(color="lightblue")))
    fig.add_trace(go.Scatter(y=df["mean"], name="Mean Path", line=dict(color="blue")))

    return {
        "chart": fig,
        "ci_low": df["ci_low"].iloc[-1] - 1,
        "ci_high": df["ci_high"].iloc[-1] - 1,
        "min": df.drop(columns=["mean", "ci_high", "ci_low"]).iloc[-1].min() - 1,
        "max": df.drop(columns=["mean", "ci_high", "ci_low"]).iloc[-1].max() - 1,
        "distribution_used": best_dist_name
    }

    }
