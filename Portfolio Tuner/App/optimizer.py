import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage, leaves_list
import cvxpy as cp

def equal_weight(prices):
    n = len(prices.columns)
    return pd.Series([1/n]*n, index=prices.columns)

# === Robust HRP Function ===
def hrb(prices):
    """
    Hierarchical Risk Budgeting (HRB) implementation following López de Prado's methodology.

    Steps:
    1. Data Preprocessing: Calculate asset returns and remove assets with NaN or zero variance.
    2. Distance Matrix: Convert the correlation matrix to a distance matrix using:
         d = sqrt(0.5 * (1 - corr))
    3. Hierarchical Clustering: Sort assets based on a hierarchical clustering algorithm.
    4. Recursive Risk Allocation: 
         - For a given cluster split into two sub-clusters A and B, compute:
             sigma_A^2 = variance of an equal-weighted portfolio of cluster A
             sigma_B^2 = variance of an equal-weighted portfolio of cluster B
         - Allocation factor:
             alpha = 1 - (sigma_A^2 / (sigma_A^2 + sigma_B^2))
         - Recursively assign weights:
             w_i = alpha * (weight from cluster A) for assets in A,
             w_j = (1 - alpha) * (weight from cluster B) for assets in B.
    """
    import numpy as np
    import pandas as pd
    from scipy.cluster.hierarchy import linkage, leaves_list

    # Step 1: Compute asset returns and clean data
    returns = prices.pct_change().dropna()
    returns = returns.dropna(axis=1, how='any')
    returns = returns.loc[:, returns.std() > 0]

    if returns.shape[1] < 2:
        n = len(prices.columns)
        return pd.Series([1 / n] * n, index=prices.columns)

    # Step 2: Compute correlation matrix and convert it to a distance matrix
    corr = returns.corr()
    dist = np.sqrt(0.5 * (1 - corr))
    condensed_dist = dist.values[np.triu_indices_from(dist, k=1)]

    if np.isnan(condensed_dist).any() or np.isinf(condensed_dist).any():
        n = len(prices.columns)
        return pd.Series([1 / n] * n, index=prices.columns)

    # Step 3: Hierarchical clustering to determine the asset ordering
    linkage_matrix = linkage(condensed_dist, method='single')
    sort_ix = leaves_list(linkage_matrix)
    sorted_assets = returns.columns[sort_ix]

    # Helper function to compute the variance (sigma^2) of a cluster
    # using an equal-weighted portfolio, i.e.:
    # sigma_cluster^2 = (1/|cluster|^2) * 1^T * Sigma_cluster * 1
    def get_cluster_var(cov, assets):
        sub_cov = cov.loc[assets, assets]
        weights = np.ones(len(assets)) / len(assets)
        return weights @ sub_cov @ weights

    # Step 4: Recursive risk allocation
    def recursive_weights(cov, assets):
        # Base case: single asset gets full weight in its cluster
        if len(assets) == 1:
            return pd.Series([1], index=assets)
        # Split the assets into two sub-clusters
        split = len(assets) // 2
        left = assets[:split]
        right = assets[split:]
        # Compute the cluster variances for the left and right sub-clusters
        left_var = get_cluster_var(cov, left)
        right_var = get_cluster_var(cov, right)
        # Compute the allocation factor based on the risk budgeting equation:
        # alpha = 1 - (sigma_left^2 / (sigma_left^2 + sigma_right^2))
        alpha = 1 - left_var / (left_var + right_var)
        # Recursively compute weights for each sub-cluster and allocate risk accordingly
        left_weights = recursive_weights(cov, left) * alpha
        right_weights = recursive_weights(cov, right) * (1 - alpha)
        return pd.concat([left_weights, right_weights])

    cov = returns.cov()
    weights = recursive_weights(cov, sorted_assets)
    return weights.reindex(prices.columns).fillna(0)

'''
def hrp(prices):
    returns = prices.pct_change().dropna()

    # Drop assets with any NaNs or constant prices
    returns = returns.dropna(axis=1, how='any')
    returns = returns.loc[:, returns.std() > 0]

    if returns.shape[1] < 2:
        # Not enough valid assets to proceed
        n = len(prices.columns)
        return pd.Series([1/n]*n, index=prices.columns)

    # Distance matrix from correlation
    corr = returns.corr()
    dist = np.sqrt(0.5 * (1 - corr))
    condensed_dist = dist.values[np.triu_indices_from(dist, k=1)]

    if np.isnan(condensed_dist).any() or np.isinf(condensed_dist).any():
        n = len(prices.columns)
        return pd.Series([1/n]*n, index=prices.columns)

    # Hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='single')
    sort_ix = leaves_list(linkage_matrix)
    sorted_assets = returns.columns[sort_ix]

    def get_cluster_var(cov, assets):
        sub_cov = cov.loc[assets, assets]
        weights = np.ones(len(assets)) / len(assets)
        return weights @ sub_cov @ weights

    def recursive_weights(cov, assets):
        if len(assets) == 1:
            return pd.Series([1], index=assets)
        split = len(assets) // 2
        left = assets[:split]
        right = assets[split:]
        left_var = get_cluster_var(cov, left)
        right_var = get_cluster_var(cov, right)
        alloc = 1 - left_var / (left_var + right_var)
        return pd.concat([
            recursive_weights(cov, left) * alloc,
            recursive_weights(cov, right) * (1 - alloc)
        ])

    cov = returns.cov()
    weights = recursive_weights(cov, sorted_assets)
    return weights.reindex(prices.columns).fillna(0)
'''

# === Optimizer Wrapper ===
def run_optimizers(prices, nonnegative_mvo=True):
    equal = equal_weight(prices)
    mvo = mean_variance_opt(prices, nonnegative=nonnegative_mvo)
    hrp = hrb(prices)
    return {"Equal Weight": equal, "Mean Variance": mvo, "HRB": hrp}
def mean_variance_opt(prices, nonnegative=True):

    # Compute returns and drop any rows with missing values
    returns = prices.pct_change().dropna()
    n_assets = len(prices.columns)
    
    # If no returns are available, fallback to equal weights
    if returns.empty or len(returns) < 2:
        return pd.Series(np.ones(n_assets) / n_assets, index=prices.columns)
    
    # Use a robust covariance estimator (optional, but often helps in noisy data)
    try:
        lw = LedoitWolf()
        lw.fit(returns)
        cov = lw.covariance_
    except Exception:
        # Fallback to the regular covariance if LedoitWolf fails
        cov = returns.cov().values

    # Force symmetry to mitigate numerical precision issues.
    cov = (cov + cov.T) / 2

    # Check for any NaNs or infinite values in covariance; if found, fallback to equal weights.
    if np.isnan(cov).any() or np.isinf(cov).any():
        return pd.Series(np.ones(n_assets) / n_assets, index=prices.columns)

    # Set up and solve the quadratic optimization
    w = cp.Variable(n_assets)
    objective = cp.Minimize(cp.quad_form(w, cov))
    constraints = [cp.sum(w) == 1]
    if nonnegative:
        constraints.append(w >= 0)
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)  # Using SCS as a robust fallback solver

    # If the solver fails, return equal weights.
    weights = w.value
    if weights is None:
        return pd.Series(np.ones(n_assets) / n_assets, index=prices.columns)
    
    return pd.Series(weights, index=prices.columns)


