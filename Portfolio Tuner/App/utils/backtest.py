# utils.py
import numpy as np
import pandas as pd
from optimizer import run_optimizers  # Make sure this function incorporates exponential weighting if desired

def dynamic_backtest_portfolio_user_fixed_shares(simulation_data, asset_amounts):
    """
    Backtest a portfolio with fixed share counts (not percent weights).

    Parameters:
        simulation_data (pd.DataFrame): Historical price data for each asset.
        asset_amounts (dict): Fixed number of shares per asset (e.g., {"BTC": 1.0, "ETH": 5.0}).
        rebalance_days (int): Frequency to reset to original share counts (set high to disable).

    Returns:
        dict: {
            "cumulative": Series of cumulative portfolio value,
            "drawdowns": Series of daily drawdowns,
            "drawdown": Float of max drawdown,
            "rolling_sharpe": Series of rolling Sharpe ratio,
            "allocations": DataFrame of implied percent weights over time
        }
    """
    # Create portfolio value over time using fixed number of shares
    prices = simulation_data.copy()
    assets = list(asset_amounts.keys())
    prices = prices[assets]

    # Calculate position values per asset (price * fixed share count)
    position_values = prices.multiply([asset_amounts[a] for a in assets], axis=1)
    portfolio_value = position_values.sum(axis=1)

    # Implied dynamic percent weights
    allocations = position_values.divide(portfolio_value, axis=0)

    # Calculate daily returns
    returns = portfolio_value.pct_change().fillna(0)

    # Cumulative returns
    cumulative = (1 + returns).cumprod()

    # Rolling Sharpe (30-day window)
    rolling_sharpe = returns.rolling(30).apply(
        lambda r: (r.mean() / r.std()) * np.sqrt(365.0) if r.std() != 0 else 0,
        raw=True
    )

    # Drawdowns
    rolling_max = portfolio_value.cummax()
    drawdowns = (portfolio_value - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    # Annualized Sharpe Ratio
    daily_std = returns.std()
    daily_mean = returns.mean()
    sharpe_ratio = np.sqrt(365.0) * (daily_mean / daily_std) if daily_std > 0 else 0

    return {
        "cumulative": cumulative,
        "rolling_sharpe": rolling_sharpe,
        "drawdowns": drawdowns,
        "drawdown": max_drawdown,
        "allocations": allocations,
        "sharpe": sharpe_ratio  
    }


def dynamic_backtest_portfolio_user(simulation_data, user_weights, lookback_days, rebalance_days, nonnegative_toggle):
    """
    Perform a dynamic backtest for the user’s portfolio using fixed, normalized weights.

    Parameters:
        simulation_data (pd.DataFrame): Price data for selected assets over the backtest period.
        user_weights (dict): A dictionary with asset names as keys and user-provided weights as values.
        lookback_days (int): Lookback period in days (unused here, but kept for consistent signature).
        rebalance_days (int): Frequency in days at which to rebalance the portfolio.
        nonnegative_toggle (bool): Whether to enforce nonnegative weights (unused here since user weights are assumed valid).

    Returns:
        dict: A dictionary containing:
            - "allocations": DataFrame recording the allocation on each rebalancing date (with each asset as a column).
            - "cumulative": Series of cumulative portfolio returns over the backtest period.
            - "sharpe": Final annualized Sharpe ratio (scalar).
            - "drawdown": Maximum drawdown (scalar).
            - "drawdowns": Daily drawdown series (time series).
            - "rolling_sharpe": Rolling annualized Sharpe ratio series (time series).
    """
    import numpy as np
    import pandas as pd

    # Normalize the user weights so they sum to 1
    total_weight = sum(user_weights.values())
    if total_weight == 0:
        raise ValueError("Total weight is zero in user portfolio backtest.")
    normalized_weights = {asset: weight / total_weight for asset, weight in user_weights.items() if asset in simulation_data.columns}

    # Initialize portfolio values
    initial_value = 1.0
    portfolio_values = pd.Series(index=simulation_data.index, dtype=float)
    portfolio_value = initial_value

    # Record allocations at each rebalance date as tuples (date, allocation_dict)
    allocation_records = []
    weight_series = pd.Series(normalized_weights)
    dates = simulation_data.index
    last_rebalance_date = dates[0]

    # Loop over each trading day
    for i, current_date in enumerate(dates):
        # Rebalance on the first day or when the rebalance period has passed
        if (current_date - last_rebalance_date).days >= rebalance_days or i == 0:
            last_rebalance_date = current_date
            allocation_records.append((current_date, normalized_weights.copy()))

        # Compute daily return
        if i == 0:
            daily_return = 1.0
        else:
            prev_date = dates[i - 1]
            asset_returns = simulation_data.loc[current_date] / simulation_data.loc[prev_date]
            daily_return = np.dot(weight_series.values, asset_returns.values)

        # Update portfolio value
        portfolio_value *= daily_return
        portfolio_values[current_date] = portfolio_value

    # Cumulative returns are represented by the portfolio_values (already cumulative)
    cumulative = portfolio_values.copy()

    # Compute daily returns for performance metrics
    returns = portfolio_values.pct_change().fillna(0)

    # Calculate overall annualized Sharpe ratio (assuming 252 trading days and zero risk-free rate)
    if returns.std() != 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # Calculate rolling Sharpe ratio using a 60-day rolling window (adjust as needed)
    rolling_window = 60
    rolling_sharpe = returns.rolling(window=rolling_window).apply(
        lambda r: (r.mean() / r.std()) * np.sqrt(252) if r.std() != 0 else 0,
        raw=True
    )

    # Calculate daily drawdown series: the percentage drop from the running maximum
    running_max = portfolio_values.cummax()
    drawdown_series = (portfolio_values - running_max) / running_max

    # Maximum drawdown is the minimum of the daily drawdown series
    max_drawdown = drawdown_series.min()

    # Transform allocation_records into a DataFrame with separate columns per asset.
    # Each row will have the allocation values for each asset at the rebalance date.
    allocation_list = []
    for date, alloc_dict in allocation_records:
        row = {"date": date}
        row.update(alloc_dict)
        allocation_list.append(row)
    allocations_df = pd.DataFrame(allocation_list).set_index("date")

    return {
        "allocations": allocations_df,
        "cumulative": cumulative,
        "sharpe": sharpe_ratio,
        "drawdown": max_drawdown,
        "drawdowns": drawdown_series,
        "rolling_sharpe": rolling_sharpe,
    }




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
    import numpy as np
    import pandas as pd

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

