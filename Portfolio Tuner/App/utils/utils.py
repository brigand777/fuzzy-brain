from datetime import timedelta
import pandas as pd

def downsample_results_dict(results_dict, start_date, end_date, target_points=365):
    """
    Downsamples time-series results to a target number of points per method.
    Keeps Sharpe ratio, drawdown, and allocations unchanged.
    """
    downsampled = {}

    for method, res in results_dict.items():
        days = max((end_date - start_date).days, 1)
        step = max(1, days // target_points)

        downsampled[method] = {
            "cumulative": res["cumulative"].iloc[::step],
            "rolling_sharpe": res["rolling_sharpe"].iloc[::step],
            "drawdowns": res["drawdowns"].iloc[::step],
            "allocations": res["allocations"].iloc[::step],
            "sharpe": res["sharpe"],
            "drawdown": res["drawdown"]
        }

    return downsampled
