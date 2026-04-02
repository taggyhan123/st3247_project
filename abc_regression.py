"""
Regression-adjusted ABC  (Beaumont, Zhang & Balding, 2002).

After rejection ABC, fit a local weighted linear regression within the
accepted region to correct for the residual gap between simulated and
observed summaries.  This sharpens the posterior without extra simulations.

Algorithm:
  1. Start with rejection ABC accepted samples {θ_i, S_i, d_i}
  2. Assign Epanechnikov kernel weights: w_i = 1 - (d_i / d_max)^2
  3. For each parameter dimension k, fit weighted linear regression:
       θ_k = α_k + B_k^T · S + ε_k      (weighted by w_i)
  4. Adjust:  θ*_k = θ_k - B_k^T · (S_i - S_obs)

Reference:
  Beaumont, Zhang & Balding (2002). "Approximate Bayesian Computation
  in Population Genetics." Genetics 162(4), 2025–2035.
"""

import numpy as np


def epanechnikov_weights(distances):
    """Epanechnikov kernel weights based on normalised distances.

    w_i = 1 - (d_i / max(d))^2,  clipped to [0, 1].
    """
    d_max = np.max(distances)
    if d_max == 0:
        return np.ones(len(distances))
    u = distances / d_max
    w = 1.0 - u ** 2
    return np.clip(w, 0, None)


def weighted_least_squares(X, y, w):
    """Weighted least squares: solve (X^T W X) beta = X^T W y.

    Parameters
    ----------
    X : ndarray (n, p)  — design matrix (with intercept column)
    y : ndarray (n,)    — response
    w : ndarray (n,)    — weights

    Returns
    -------
    beta : ndarray (p,)
    """
    W = np.diag(w)
    XtW = X.T @ W
    beta = np.linalg.lstsq(XtW @ X, XtW @ y, rcond=None)[0]
    return beta


def regression_adjust(thetas, summaries, distances, s_obs):
    """Apply Beaumont et al. (2002) regression adjustment.

    Parameters
    ----------
    thetas : ndarray (n, 3)       — accepted parameter samples
    summaries : ndarray (n, d)    — corresponding summary statistics
    distances : ndarray (n,)      — distances to observed summaries
    s_obs : ndarray (d,)          — observed summary statistics

    Returns
    -------
    adjusted_thetas : ndarray (n, 3)
    """
    n, n_params = thetas.shape
    n_stats = summaries.shape[1]

    weights = epanechnikov_weights(distances)

    # Design matrix: intercept + summaries
    X = np.column_stack([np.ones(n), summaries])

    adjusted = np.copy(thetas)

    for k in range(n_params):
        beta = weighted_least_squares(X, thetas[:, k], weights)
        # beta[0] is intercept, beta[1:] are coefficients on summaries
        B = beta[1:]
        # Adjustment: shift each θ_k by -B^T (S_i - S_obs)
        residuals = summaries - s_obs  # (n, d)
        adjusted[:, k] = thetas[:, k] - residuals @ B

    return adjusted
