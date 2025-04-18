import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fitter import Fitter
from scipy.stats import norm, t, johnsonsu
from sklearn.covariance import LedoitWolf

import numpy as np
import pandas as pd
from fitter import Fitter
from scipy.stats import norm, t, johnsonsu
from sklearn.covariance import LedoitWolf
import plotly.graph_objects as go

from optimizer import run_optimizers  # Make sure this points to your real optimizer logic

def run_monte_carlo_with_rebalancing(
    strategies: dict,
    price_data: pd.DataFrame,
    horizon_days: int = 180,
    n_sims: int = 100,
    rebalance_interval: int = 30,
    correlation_strategy: str = "shrinkage",
    percentiles: tuple = (25, 75)
    ):
    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    assets = price_data.columns.tolist()

    # --- Fit Distributions per Asset
    dist_map = {'norm': norm, 't': t, 'johnsonsu': johnsonsu}
    asset_distributions = {}
    distribution_used_per_asset = {}

    for asset in assets:
        try:
            f = Fitter(log_returns[asset].values, distributions=list(dist_map.keys()), timeout=5, verbose=False)
            f.fit()
            best_name = next(iter(f.get_best()))
            best_params = f.fitted_param[best_name]
        except Exception:
            best_name = "norm"
            best_params = norm.fit(log_returns[asset].values)

        asset_distributions[asset] = (dist_map[best_name], best_params)
        distribution_used_per_asset[asset] = best_name

    # --- Correlation Matrix
    if correlation_strategy == "independent":
        corr_matrix = np.eye(len(assets))
    elif correlation_strategy == "historical":
        corr_matrix = log_returns.corr().values
    else:
        lw = LedoitWolf().fit(log_returns.values)
        cov = lw.covariance_
        d = np.sqrt(np.diag(cov))
        corr_matrix = cov / np.outer(d, d)

    # --- Simulate Returns
    mvn_shocks = np.random.multivariate_normal(
        mean=np.zeros(len(assets)),
        cov=corr_matrix,
        size=horizon_days * n_sims
    ).reshape(horizon_days, n_sims, len(assets))

    sim_returns = np.empty_like(mvn_shocks)
    for i, asset in enumerate(assets):
        dist, params = asset_distributions[asset]
        sim_returns[:, :, i] = dist.ppf(norm.cdf(mvn_shocks[:, :, i]), *params)

    strategy_paths = {}
    summary_stats = {}

    for name in strategies:
        portfolio_paths = np.ones((horizon_days, n_sims))

        # Initial weights from strategy
        weights = strategies[name]
        weights_array = np.array([weights.get(asset, 0) for asset in assets])

        for day in range(horizon_days):
            if day % rebalance_interval == 0 and name != "User Portfolio":
                # Simulate lookback window from simulated returns
                sim_slice = sim_returns[:day + 1, :, :].mean(axis=1)  # approx price path
                sim_prices = pd.DataFrame(
                    np.cumprod(1 + sim_slice, axis=0),
                    columns=assets
                )
                try:
                    rebalanced_allocs = run_optimizers(sim_prices, nonnegative_mvo=True)
                    new_weights = rebalanced_allocs.get(name, weights)
                    weights_array = np.array([new_weights.get(asset, 0) for asset in assets])
                except:
                    pass  # fallback to last weights if error

            day_returns = sim_returns[day, :, :]
            if day == 0:
                portfolio_paths[day, :] = np.sum(1 + day_returns * weights_array[np.newaxis, :], axis=1)
            else:
                portfolio_paths[day, :] = portfolio_paths[day - 1, :] * np.sum(1 + day_returns * weights_array[np.newaxis, :], axis=1)

        median = np.median(portfolio_paths, axis=1)
        ci_low = np.percentile(portfolio_paths, percentiles[0], axis=1)
        ci_high = np.percentile(portfolio_paths, percentiles[1], axis=1)

        strategy_paths[name] = {
            "paths": portfolio_paths,
            "median": median,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

        summary_stats[name] = {
            "ci_low": ci_low[-1] - 1,
            "ci_high": ci_high[-1] - 1,
            "min": portfolio_paths[-1].min() - 1,
            "max": portfolio_paths[-1].max() - 1
        }

    # --- Plot
    fig = go.Figure()
    for name, result in strategy_paths.items():
        fig.add_trace(go.Scatter(
            x=np.arange(horizon_days),
            y=result["ci_high"],
            name=f"{name} CI High",
            line=dict(width=0),
            mode='lines',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=np.arange(horizon_days),
            y=result["ci_low"],
            name=f"{name} CI Low",
            line=dict(width=0),
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(173,216,230,0.15)',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=np.arange(horizon_days),
            y=result["median"],
            name=f"{name} Median",
            mode='lines'
        ))

    fig.update_layout(
        title=\"Monte Carlo Forecast — Rebalancing Strategies\",
        xaxis_title=\"Day\",
        yaxis_title=\"Portfolio Value\",
        hovermode=\"x unified\",
        template=\"plotly_white\"
    )

    return {
        \"chart\": fig,
        \"paths\": {name: result[\"paths\"] for name, result in strategy_paths.items()},
        \"summary\": summary_stats,
        \"distribution_used_per_asset\": distribution_used_per_asset,
        \"correlation_strategy\": correlation_strategy
    }


def run_monte_carlo_multi_strategy(strategies: dict, price_data: pd.DataFrame, horizon_days=180, n_sims=100,
                                    corr_matrix=None, correlation_strategy="shrinkage", percentiles=(25, 75)):
    """
    Run Monte Carlo simulation for multiple strategies over shared simulated market conditions.

    Parameters:
        strategies (dict): {strategy_name: {asset: weight}} mapping.
        price_data (DataFrame): Historical price data (assets as columns).
        horizon_days (int): Days to simulate.
        n_sims (int): Number of simulations.
        corr_matrix (np.ndarray): Optional correlation matrix override.
        correlation_strategy (str): "historical", "shrinkage", or "independent".
        percentiles (tuple): Confidence interval percentiles, e.g., (25, 75).
    
    Returns:
        dict with:
            "chart": Plotly figure,
            "paths": {strategy: portfolio paths array},
            "summary": {strategy: {ci_low, ci_high, min, max}},
            "distribution_used_per_asset": {asset: dist_name}
    """
    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    assets = price_data.columns.tolist()

    dist_map = {'norm': norm, 't': t, 'johnsonsu': johnsonsu}
    distribution_used_per_asset = {}
    asset_distributions = {}

    for asset in assets:
        try:
            f = Fitter(log_returns[asset].values, distributions=list(dist_map.keys()), timeout=5, verbose=False)
            f.fit()
            best_name = next(iter(f.get_best()))
            best_params = f.fitted_param[best_name]
        except Exception:
            best_name = "norm"
            best_params = norm.fit(log_returns[asset].values)

        asset_distributions[asset] = (dist_map[best_name], best_params)
        distribution_used_per_asset[asset] = best_name

    if corr_matrix is None:
        if correlation_strategy == "independent":
            corr_matrix = np.eye(len(assets))
        elif correlation_strategy == "historical":
            corr_matrix = log_returns.corr().values
        else:
            lw = LedoitWolf().fit(log_returns.values)
            cov = lw.covariance_
            d = np.sqrt(np.diag(cov))
            corr_matrix = cov / np.outer(d, d)

    mvn_shocks = np.random.multivariate_normal(
        mean=np.zeros(len(assets)),
        cov=corr_matrix,
        size=horizon_days * n_sims
    ).reshape(horizon_days, n_sims, len(assets))

    sim_returns = np.empty_like(mvn_shocks)
    for i, asset in enumerate(assets):
        dist, params = asset_distributions[asset]
        sim_returns[:, :, i] = dist.ppf(norm.cdf(mvn_shocks[:, :, i]), *params)

    strategy_paths = {}
    summary_stats = {}

    for name, weight_dict in strategies.items():
        weights_array = np.array([weight_dict.get(asset, 0) for asset in assets])
        asset_price_paths = np.cumprod(1 + sim_returns, axis=0)
        portfolio_paths = np.sum(asset_price_paths * weights_array[np.newaxis, np.newaxis, :], axis=2)
        portfolio_paths = portfolio_paths / portfolio_paths[0, :]

        median = np.median(portfolio_paths, axis=1)
        ci_low = np.percentile(portfolio_paths, percentiles[0], axis=1)
        ci_high = np.percentile(portfolio_paths, percentiles[1], axis=1)

        strategy_paths[name] = {
            "paths": portfolio_paths,
            "median": median,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

        summary_stats[name] = {
            "ci_low": ci_low[-1] - 1,
            "ci_high": ci_high[-1] - 1,
            "min": portfolio_paths[-1].min() - 1,
            "max": portfolio_paths[-1].max() - 1
        }

    fig = go.Figure()
    for name, result in strategy_paths.items():
        fig.add_trace(go.Scatter(
            x=np.arange(horizon_days),
            y=result["ci_high"],
            name=f"{name} CI High",
            line=dict(width=0),
            mode='lines',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=np.arange(horizon_days),
            y=result["ci_low"],
            name=f"{name} CI Low",
            line=dict(width=0),
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(173,216,230,0.15)',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=np.arange(horizon_days),
            y=result["median"],
            name=f"{name} Median",
            mode='lines'
        ))

    fig.update_layout(
        title="Monte Carlo Forecast — Multiple Strategies",
        xaxis_title="Day",
        yaxis_title="Portfolio Value",
        hovermode="x unified",
        template="plotly_white"
    )

    return {
        "chart": fig,
        "paths": {name: result["paths"] for name, result in strategy_paths.items()},
        "summary": summary_stats,
        "distribution_used_per_asset": distribution_used_per_asset,
        "correlation_strategy": correlation_strategy
    }

def run_smart_monte_carlo_simulation(weights, price_data, horizon_days=180, n_sims=100, 
                                      corr_matrix=None, correlation_strategy="shrinkage"):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from scipy.stats import norm, t, johnsonsu
    from sklearn.covariance import LedoitWolf
    from fitter import Fitter

    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    assets = price_data.columns.tolist()
    weights_array = np.array([weights.get(asset, 0) for asset in assets])

    dist_map = {'norm': norm, 't': t, 'johnsonsu': johnsonsu}
    distribution_used_per_asset = {}
    asset_distributions = {}

    # --- Faster Fitter Loop ---
    for asset in assets:
        try:
            f = Fitter(log_returns[asset].values, distributions=['norm', 't', 'johnsonsu'], timeout=5, verbose=False)
            f.fit()
            best_name = next(iter(f.get_best()))
            best_params = f.fitted_param[best_name]
        except Exception:
            # fallback to normal
            best_name = "norm"
            best_params = norm.fit(log_returns[asset].values)

        asset_distributions[asset] = (dist_map[best_name], best_params)
        distribution_used_per_asset[asset] = best_name

    # --- Correlation ---
    if corr_matrix is None:
        if correlation_strategy == "independent":
            corr_matrix = np.eye(len(assets))
        elif correlation_strategy == "historical":
            corr_matrix = log_returns.corr().values
        else:  # shrinkage
            lw = LedoitWolf().fit(log_returns.values)
            cov = lw.covariance_
            d = np.sqrt(np.diag(cov))
            corr_matrix = cov / np.outer(d, d)

    # --- Correlated base shocks ---
    mvn_shocks = np.random.multivariate_normal(
        mean=np.zeros(len(assets)),
        cov=corr_matrix,
        size=horizon_days * n_sims
    ).reshape(horizon_days, n_sims, len(assets))

    # --- Map to asset-specific distributions ---
    sim_returns = np.empty_like(mvn_shocks)
    for i, asset in enumerate(assets):
        dist, params = asset_distributions[asset]
        sim_returns[:, :, i] = dist.ppf(norm.cdf(mvn_shocks[:, :, i]), *params)

    # --- Simulate portfolio value paths ---
    asset_price_paths = np.cumprod(1 + sim_returns, axis=0)
    portfolio_paths = np.sum(asset_price_paths * weights_array[np.newaxis, np.newaxis, :], axis=2)
    portfolio_paths = portfolio_paths / portfolio_paths[0, :]

    # --- Statistics ---
    median = np.median(portfolio_paths, axis=1)
    ci_high = np.percentile(portfolio_paths, 75, axis=1)
    ci_low = np.percentile(portfolio_paths, 25, axis=1)

    # --- Downsample for Plotly (if large)
    step = max(1, len(median) // 250)
    idx = np.arange(0, len(median), step)
    df_plot = pd.DataFrame({
        "Day": idx,
        "median": median[idx],
        "ci_high": ci_high[idx],
        "ci_low": ci_low[idx]
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot["Day"], y=df_plot["ci_high"], name="75% Confidence Upper",
                             line=dict(color="lightblue", dash="dot")))
    fig.add_trace(go.Scatter(x=df_plot["Day"], y=df_plot["ci_low"], name="25% Confidence Lower",
                             line=dict(color="lightblue", dash="dot"),
                             fill="tonexty", fillcolor="rgba(173,216,230,0.2)"))
    fig.add_trace(go.Scatter(x=df_plot["Day"], y=df_plot["median"], name="Median Path", line=dict(color="blue")))

    fig.update_layout(
        title="Monte Carlo Portfolio Forecast (Correlated, Distribution-Fitted)",
        xaxis_title="Day",
        yaxis_title="Portfolio Value",
        hovermode="x unified",
        template="plotly_white"
    )

    return {
        "chart": fig,
        "ci_low": ci_low[-1] - 1,
        "ci_high": ci_high[-1] - 1,
        "min": portfolio_paths[-1].min() - 1,
        "max": portfolio_paths[-1].max() - 1,
        "distribution_used_per_asset": distribution_used_per_asset,
        "correlation_strategy": correlation_strategy
    }


def run_smart_monte_carlo_simulations(weights, price_data, horizon_days=180, n_sims=100, 
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
    distribution_used_per_asset = {}
    asset_distributions = {}

    for asset in assets:
        f = Fitter(log_returns[asset].values, distributions=['norm', 't', 'johnsonsu'], timeout=5)
        f.fit()
        best_name = list(f.get_best().keys())[0]
        best_params = f.fitted_param[best_name]
        asset_distributions[asset] = (dist_map[best_name], best_params)
        distribution_used_per_asset[asset] = best_name


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
    portfolio_paths = np.sum(asset_price_paths * weights_array[np.newaxis, np.newaxis, :], axis=2)
    portfolio_paths = portfolio_paths / portfolio_paths[0, :]

    # --- Step 7: Create DataFrame with Statistics ---
    df = pd.DataFrame(portfolio_paths)
    df.index.name = "Day"
    df["median"] = df.median(axis=1)
    df["ci_high"] = df.quantile(0.75, axis=1)
    df["ci_low"] = df.quantile(0.25, axis=1)

    # --- Step 8: Plotly Interactive Chart ---
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df["ci_high"], name="75% Confidence Upper",
                             line=dict(color="lightblue", dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["ci_low"], name="25% Confidence Lower",
                             line=dict(color="lightblue", dash="dot"),
                             fill="tonexty", fillcolor="rgba(173,216,230,0.2)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["median"], name="Median Path", line=dict(color="blue")))

    fig.update_layout(
        title="Monte Carlo Portfolio Forecast (Correlated, Distribution-Fitted)",
        xaxis_title="Day",
        yaxis_title="Portfolio Value",
        hovermode="x unified",
        template="plotly_white"
    )

    return {
    "chart": fig,
    "ci_low": df["ci_low"].iloc[-1] - 1,
    "ci_high": df["ci_high"].iloc[-1] - 1,
    "min": df.drop(columns=["median", "ci_high", "ci_low"]).iloc[-1].min() - 1,
    "max": df.drop(columns=["median", "ci_high", "ci_low"]).iloc[-1].max() - 1,
    "distribution_used_per_asset": distribution_used_per_asset,
    "correlation_strategy": correlation_strategy
}

