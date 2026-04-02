"""
Synthetic Likelihood  (Wood, 2010).

Instead of comparing summary statistics through a distance function (as in ABC),
this method assumes the summary statistics follow a multivariate normal
distribution conditional on the parameters:

    s | theta  ~  N(mu_theta, Sigma_theta)

The mean mu_theta and covariance Sigma_theta are estimated from repeated
simulations at each proposed theta, giving a "synthetic likelihood":

    l_s(theta) = -1/2 (s - mu_hat)^T Sigma_hat^{-1} (s - mu_hat)
                 -1/2 log|Sigma_hat|

This is explored via Metropolis-Hastings MCMC.

Uses Campbell's (1980) robust covariance estimation following Wood's
reference R implementation (sl package).

Reference:
  Wood (2010). "Statistical inference for noisy nonlinear ecological
  dynamic systems." Nature 466, 1102-1104.
"""

import numpy as np
from scipy.linalg import solve_triangular
from summary_stats import compute_summaries


def robust_vcov(sY, alpha=2.0, beta=1.25):
    """Robust covariance estimation using Campbell's method with QR preconditioning.

    Follows Wood's R implementation (sl package):
      1. Initial estimate via QR of centred, preconditioned statistics
      2. Mahalanobis distances to identify tail points
      3. Campbell down-weighting of extreme observations
      4. Recompute mean and covariance with weights

    Parameters
    ----------
    sY : ndarray (n_stats, n_reps)
        Each column is a replicate summary statistics vector.
    alpha, beta : float
        Campbell weighting parameters.

    Returns
    -------
    dict with keys:
        E : ndarray (n_stats, n_stats) — such that Sigma^{-1} = E^T E
        half_ldet_V : float            — (1/2) log|Sigma_hat|
        mY : ndarray (n_stats,)        — robust mean estimate
    """
    n_stats, n_reps = sY.shape

    # Step 1: initial mean and preconditioning
    mY = np.mean(sY, axis=1)
    sY1 = sY - mY[:, None]

    # Preconditioner D = diag of RMS of each row
    D = np.sqrt(np.mean(sY1 ** 2, axis=1))
    D[D == 0] = 1.0
    Di = 1.0 / D

    # Preconditioned residuals
    sY1_scaled = Di[:, None] * sY1  # (n_stats, n_reps)

    # QR decomposition of (sY1_scaled^T / sqrt(n_reps-1)) which is (n_reps, n_stats)
    # R is upper triangular (n_stats, n_stats)
    Q, R = np.linalg.qr(sY1_scaled.T / np.sqrt(n_reps - 1), mode='reduced')

    # Add small regularisation to R diagonal for numerical stability
    reg = 1e-8 * np.max(np.abs(np.diag(R)))
    R += reg * np.eye(n_stats)

    # Mahalanobis distances: solve R^T z = sY1_scaled (R^T is lower triangular)
    zz = solve_triangular(R.T, sY1_scaled, lower=True)  # (n_stats, n_reps)
    d = np.sqrt(np.sum(zz ** 2, axis=0))

    # Step 2: Campbell weights
    d0 = np.sqrt(n_stats) + alpha / np.sqrt(2)
    w = np.ones(n_reps)
    ind = d > d0
    w[ind] = np.exp(-0.5 * (d[ind] - d0) ** 2 / beta) * d0 / d[ind]

    # Step 3: Recompute weighted mean
    mY = np.sum(w[None, :] * sY, axis=1) / np.sum(w)
    sY1 = sY - mY[:, None]

    # Recompute preconditioner
    D = np.sqrt(np.mean(sY1 ** 2, axis=1))
    D[D == 0] = 1.0
    Di = 1.0 / D
    sY1_scaled = Di[:, None] * sY1

    # Weighted QR
    w_scale = np.sqrt(np.sum(w ** 2) - 1)
    if w_scale <= 0:
        w_scale = 1.0
    Q2, R2 = np.linalg.qr((w[:, None] * sY1_scaled.T) / w_scale, mode='reduced')

    # Regularise
    reg2 = 1e-8 * np.max(np.abs(np.diag(R2)))
    R2 += reg2 * np.eye(n_stats)

    # E such that Sigma^{-1} = E^T E
    # From Wood's R code: E = t(Di * backsolve(R, diag(n_stats)))
    # i.e. E = (Di * R^{-1})^T
    R2_inv = solve_triangular(R2, np.eye(n_stats), lower=False)  # upper triangular solve
    E = (Di[:, None] * R2_inv).T  # (n_stats, n_stats)

    # half log det: sum(log|R_ii|) + sum(log(D_i))
    half_ldet_V = np.sum(np.log(np.abs(np.diag(R2)))) + np.sum(np.log(D))

    return {"E": E, "half_ldet_V": half_ldet_V, "mY": mY}


def synthetic_log_likelihood(s_obs, sY):
    """Evaluate the log synthetic likelihood.

    Parameters
    ----------
    s_obs : ndarray (n_stats,)
        Observed summary statistics.
    sY : ndarray (n_stats, n_reps)
        Simulated summary statistics (each column = one replicate).

    Returns
    -------
    ll : float
        Log synthetic likelihood value.
    """
    # Remove replicates with non-finite values
    finite_mask = np.all(np.isfinite(sY), axis=0)
    sY = sY[:, finite_mask]

    if sY.shape[1] < sY.shape[0] + 2:
        return -np.inf  # not enough replicates for covariance estimation

    er = robust_vcov(sY)

    residual = er["E"] @ (s_obs - er["mY"])
    rss = np.sum(residual ** 2)
    ll = -rss / 2.0 - er["half_ldet_V"]

    return ll


def run_synthetic_likelihood_mcmc(
    s_obs,
    n_iter,
    n_sim_per_eval,
    simulator_fn,
    theta_init,
    proposal_cov,
    summary_fn=None,
    prior_bounds=None,
    seed=0,
    verbose=True,
):
    """Run MCMC with synthetic likelihood.

    At each MCMC step, the synthetic likelihood is evaluated by:
      1. Simulating n_sim_per_eval replicates at the proposed theta
      2. Computing summary statistics for each replicate
      3. Estimating mu and Sigma from these replicates
      4. Evaluating the MVN log-likelihood of s_obs

    Parameters
    ----------
    s_obs : ndarray (d,)
    n_iter : int                    — MCMC iterations
    n_sim_per_eval : int            — simulations per likelihood evaluation (e.g. 200-500)
    simulator_fn : callable         — f(beta, gamma, rho, seed=int) -> (inf, rew, deg)
    theta_init : ndarray (3,)
    proposal_cov : ndarray (3, 3)   — Gaussian random walk covariance
    summary_fn : callable or None
    prior_bounds : list of (lo, hi) — parameter bounds; None = default SIR priors
    seed : int
    verbose : bool

    Returns
    -------
    chain : ndarray (n_iter, 3)
    ll_chain : ndarray (n_iter,)    — log synthetic likelihood at each step
    acceptance_rate : float
    """
    if summary_fn is None:
        summary_fn = compute_summaries

    if prior_bounds is None:
        prior_bounds = [(0.05, 0.50), (0.02, 0.20), (0.00, 0.80)]

    rng = np.random.default_rng(seed)
    n_stats = len(s_obs)

    chain = np.zeros((n_iter, 3))
    ll_chain = np.zeros(n_iter)

    theta_current = theta_init.copy()

    # Evaluate synthetic likelihood at initial point
    def eval_sl(theta):
        sY = np.zeros((n_stats, n_sim_per_eval))
        for j in range(n_sim_per_eval):
            inf, rew, deg = simulator_fn(*theta, seed=int(rng.integers(0, 2**31)))
            sY[:, j] = summary_fn(inf, rew, deg)
        return synthetic_log_likelihood(s_obs, sY)

    ll_current = eval_sl(theta_current)
    n_accept = 0

    for i in range(n_iter):
        if verbose and (i + 1) % 500 == 0:
            rate = n_accept / (i + 1)
            print(f"  Iter {i+1}/{n_iter}, accept rate: {rate:.3f}, ll: {ll_current:.2f}")

        # Propose
        theta_prop = rng.multivariate_normal(theta_current, proposal_cov)

        # Prior check
        in_prior = True
        for val, (lo, hi) in zip(theta_prop, prior_bounds):
            if val < lo or val > hi:
                in_prior = False
                break

        if not in_prior:
            chain[i] = theta_current
            ll_chain[i] = ll_current
            continue

        # Evaluate synthetic likelihood at proposed theta
        ll_prop = eval_sl(theta_prop)

        # Metropolis-Hastings acceptance
        log_alpha = ll_prop - ll_current
        if np.log(rng.random()) < log_alpha:
            theta_current = theta_prop
            ll_current = ll_prop
            n_accept += 1

        chain[i] = theta_current
        ll_chain[i] = ll_current

    acceptance_rate = n_accept / n_iter
    if verbose:
        print(f"  Final acceptance rate: {acceptance_rate:.4f}")

    return chain, ll_chain, acceptance_rate
