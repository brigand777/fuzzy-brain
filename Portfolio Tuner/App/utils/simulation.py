import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fitter import Fitter
from scipy.stats import norm, t, johnsonsu
from sklearn.covariance import LedoitWolf

def run_smart_monte_carlo_simulation(weights, price_data, horizon_days=180, n_sims=1000, 
                                      corr_matrix=None, correlation_strategy="shrinkage"):
    """
    Smart Monte Carlo simulation with per-asset distribution fitting and flexible correlation structure.
    
    Parameters:
        weights (dict): Asset weights (by column name in price_data)
        price_data (DataFrame): Historical price data (columns = assets)
        horizon_days (int): Number of trading days to simulate
        n_sims (int): Number of simulations
        corr_matrix (ndarray, optional): User-supplied correlation matrix
        correlation_strategy (str): "historical", "shrinkage" (default), or "independent"
    """

    # --- Step 1: Calculate Log Returns ---
    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    assets = price_data.columns.tolist()
    weights_array = np.array([weights.get(asset, 0) for asset in assets])

    # --- Step 2: Fit Best Distribution Per Asset ---
    dist_map = {'norm': norm, 't': t, 'johnsonsu': johnsonsu}
    asset_distributions = {}

    for asset in assets:
        f = Fitter(log_returns[asset].values, distributions=['norm', 't', 'johnsonsu'], timeout=5)
        f.fit()
        best_name = list(f.get_best().keys())[0]
        best_params = f.fitted_param[best_name]
        asset_distributions[asset] = (dist_map[best_name], best_params)

    # --- Step 3: Determine Correlation Structure ---
    if corr_matrix is None:
        if correlation_strategy == "independent":
            corr_matrix = np.eye(len(assets))
        elif correlation_strategy == "historical":
            corr_matrix = log_returns.corr().values
        else:  # default: shrinkage
            lw = LedoitWolf().fit(log_returns.values)
            corr_matrix = lw.covariance_
            # Convert to correlation matrix
            d = np.sqrt(np.diag(corr_matrix))
            corr_matrix = corr_matrix / np.outer(d, d)
    
    # --- Step 4: Generate Correlated Normal Shocks ---
    # These are base shocks we will map to asset-specific distributions
    mvn_shocks = np.random.multivariate_normal(
        mean=np.zeros(len(assets)),
        cov=corr_matrix,
        size=(horizon_days * n_sims)
    ).reshape(horizon_days, n_sims, len(assets))

    # --- Step 5: Apply Marginal Distributions ---
    sim_returns = np.zeros_like(mvn_shocks)

    for i, asset in enumerate(assets):
        dist, params = asset_distributions[asset]
        # Transform standard normal to fitted marginal using inverse CDF (PPF)
        sim_returns[:, :, i] = dist.ppf(norm.cdf(mvn_shocks[:, :, i]), *params)

    # --- Step 6: Simulate Portfolio Value Paths ---
    asset_price_paths = np.cumprod(1 + sim_returns, axis=0)
    initial_prices = price_data.iloc[-1].values
    position_values = weights_array * initial_prices
    portfolio_paths = np.sum(asset_price_paths * position_values[np.newaxis, np.newaxis, :], axis=2)

    # --- Step 7: Create DataFrame with Statistics ---
    df = pd.DataFrame(portfolio_paths)
    df.index.name = "Day"
    df["mean"] = df.mean(axis=1)
    df["ci_high"] = df.quantile(0.90, axis=1)
    df["ci_low"] = df.quantile(0.10, axis=1)

    # --- Step 8: Plotly Interactive Chart ---
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df["ci_high"], name="90% Confidence Upper",
                             line=dict(color="lightblue", dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["ci_low"], name="10% Confidence Lower",
                             line=dict(color="lightblue", dash="dot"),
                             fill="tonexty", fillcolor="rgba(173,216,230,0.2)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["mean"], name="Mean Path", line=dict(color="blue")))

    fig.update_layout(
        title="Monte Carlo Portfolio Forecast (Correlated, Distribution-Fitted)",
        xaxis_title="Day",
        yaxis_title="Portfolio Value (Relative)",
        hovermode="x unified",
        template="plotly_white"
    )

    return {
        "chart": fig,
        "ci_low": df["ci_low"].iloc[-1] - 1,
        "ci_high": df["ci_high"].iloc[-1] - 1,
        "min": df.drop(columns=["mean", "ci_high", "ci_low"]).iloc[-1].min() - 1,
        "max": df.drop(columns=["mean", "ci_high", "ci_low"]).iloc[-1].max() - 1,
        "distribution_used_per_asset": {
            asset: list(Fitter(log_returns[asset].values, distributions=['norm', 't', 'johnsonsu'], timeout=5).get_best().keys())[0]
            for asset in assets
        },
        "correlation_strategy": correlation_strategy
    }
