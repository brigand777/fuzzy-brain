# utils.py
import numpy as np
import pandas as pd
from core.optimizer import run_optimizers

# === Dynamic Backtest Function ===
def dynamic_backtest_portfolio(prices, method, lookback_days, rebalance_days, nonnegative_flag):
    """
    Perform a dynamic backtest with periodic reoptimization.
    For each rebalance date, only assets with a positive return standard deviation
    over the lookback window are included in the optimization. The optimizer then assigns
    weights (using equal weight, MVO, or HRP) to the valid assets. Assets with zero std are set to 0.
    
    Parameters:
      prices (DataFrame): Historical price data with datetime index.
      method (str): The optimization method to use (e.g., "HRB", "Mean Variance", "Equal Weight").
      lookback_days (int): Number of days to look back for reoptimization.
      rebalance_days (int): Frequency (in days) at which to rebalance the portfolio.
      nonnegative_flag (bool): Whether to enforce nonnegative weights in MVO.
    
    Returns:
      dict: Contains cumulative returns, rolling Sharpe, drawdowns, allocation history,
            final annualized Sharpe, and maximum drawdown.
    """

    # Calculate daily returns and get the dates from the returns index.
    returns = prices.pct_change().dropna()
    dates = returns.index
    weight_df = pd.DataFrame(index=dates, columns=prices.columns)

    for i in range(0, len(dates), rebalance_days):
        rebal_date = dates[i]
        lookback_start = rebal_date - pd.Timedelta(days=lookback_days)
        lookback_data = prices.loc[lookback_start:rebal_date]

        # If the lookback window is empty, fallback to previous weights or equal weights.
        if lookback_data.empty:
            end_idx = min(i + rebalance_days, len(dates))
            if i > 0:
                weight_df.iloc[i:end_idx] = weight_df.iloc[i - 1].values
            else:
                weight_df.iloc[i:end_idx] = pd.Series(1 / len(prices.columns), index=prices.columns).values
            continue

        # Compute lookback returns and calculate standard deviation per asset.
        lookback_returns = lookback_data.pct_change().dropna()
        asset_stds = lookback_returns.std()
        # Only include assets whose return std > 0.
        valid_assets = asset_stds[asset_stds > 0].index.tolist()

        # If no assets are valid, fallback to previous weights or equal weights.
        if len(valid_assets) == 0:
            end_idx = min(i + rebalance_days, len(dates))
            if i > 0:
                weight_df.iloc[i:end_idx] = weight_df.iloc[i - 1].values
            else:
                weight_df.iloc[i:end_idx] = pd.Series(1 / len(prices.columns), index=prices.columns).values
            continue

        # Filter the lookback data to only include valid assets.
        filtered_lookback_data = lookback_data[valid_assets]

        # Run the optimizer on the filtered data.
        dynamic_allocations = run_optimizers(filtered_lookback_data, nonnegative_mvo=nonnegative_flag)
        new_weights = dynamic_allocations[method]
        # Reindex new_weights to the full set of assets (assets not in valid_assets get weight 0).
        new_weights = new_weights.reindex(prices.columns).fillna(0)

        # Normalize weights if the sum is > 0.
        if new_weights.sum() > 0:
            new_weights /= new_weights.sum()
        else:
            # Fallback to previous weights, or equal weights if not available
            if i > 0:
                new_weights = weight_df.iloc[i - 1]
            else:
                new_weights = pd.Series(1 / len(prices.columns), index=prices.columns)


        # Apply these new weights for the period until the next rebalance.
        end_idx = min(i + rebalance_days, len(dates))
        weight_df.iloc[i:end_idx] = new_weights.values

    # Forward-fill any missing weights.
    weight_df = weight_df.ffill().fillna(0)
    portfolio_returns = (returns * weight_df).sum(axis=1)
    cumulative = (1 + portfolio_returns).cumprod()

    # Compute rolling annualized Sharpe (30-day window).
    rolling_sharpe = portfolio_returns.rolling(30).mean() / portfolio_returns.rolling(30).std()
    rolling_sharpe = np.sqrt(365) * rolling_sharpe

    # Compute rolling maximum drawdown.
    rolling_max = cumulative.cummax()
    drawdowns = cumulative / rolling_max - 1
    max_drawdown = drawdowns.min()

    # Calculate overall annualized Sharpe.
    daily_mean = portfolio_returns.mean()
    daily_std = portfolio_returns.std()
    total_sharpe = np.sqrt(365) * (daily_mean / daily_std) if daily_std > 0 else np.nan

    return {
        "cumulative": cumulative,
        "rolling_sharpe": rolling_sharpe,
        "drawdowns": drawdowns,
        "allocations": weight_df,
        "sharpe": total_sharpe,
        "drawdown": max_drawdown
    }

