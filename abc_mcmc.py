"""
ABC-MCMC  (Marjoram, Molitor, Plagnol & Tavaré, 2003).

A Markov chain Monte Carlo sampler within the ABC framework.  Instead of
drawing blindly from the prior, the chain proposes parameters informed by
the current state, which is more efficient for reaching small tolerances.

Algorithm:
  1. Initialise θ_0 (e.g. from a rejection ABC accepted sample)
  2. At each iteration:
     a. Propose θ' ~ q(· | θ_current)   [Gaussian random walk]
     b. If θ' outside prior support, reject immediately
     c. Simulate y' ~ model(θ'), compute S(y')
     d. If d(S(y'), S_obs) < ε:
          accept θ' with probability min(1, π(θ')/π(θ) · q(θ|θ')/q(θ'|θ))
          (for uniform prior + symmetric proposal: always accept)
        Else: stay at θ_current

Reference:
  Marjoram et al. (2003). "Markov chain Monte Carlo without likelihoods."
  PNAS 100(26), 15324–15328.
"""

import numpy as np
from summary_stats import compute_summaries
from abc_rejection import PRIOR_BOUNDS, normalised_distance


def _in_prior(theta):
    """Check if theta is within prior bounds."""
    bounds = [PRIOR_BOUNDS["beta"], PRIOR_BOUNDS["gamma"], PRIOR_BOUNDS["rho"]]
    for val, (lo, hi) in zip(theta, bounds):
        if val < lo or val > hi:
            return False
    return True


def run_abc_mcmc(
    s_obs,
    mads,
    epsilon,
    n_iter,
    simulator_fn,
    theta_init,
    proposal_cov,
    summary_fn=None,
    indices=None,
    seed=0,
    verbose=True,
):
    """Run ABC-MCMC.

    Parameters
    ----------
    s_obs : ndarray (d,)        — observed summary statistics
    mads : ndarray (d,)         — MAD normalisation constants
    epsilon : float             — acceptance threshold on normalised distance
    n_iter : int                — number of MCMC iterations
    simulator_fn : callable     — f(beta, gamma, rho, seed=int)
    theta_init : ndarray (3,)   — starting parameter values
    proposal_cov : ndarray (3,3) — covariance matrix for Gaussian random walk
    summary_fn : callable or None
    indices : list[int] or None — subset of summary indices
    seed : int
    verbose : bool

    Returns
    -------
    chain : ndarray (n_iter, 3)     — parameter chain (including repeats)
    distances : ndarray (n_iter,)   — distances at each step
    acceptance_rate : float
    """
    if summary_fn is None:
        summary_fn = compute_summaries

    rng = np.random.default_rng(seed)

    chain = np.zeros((n_iter, 3))
    dist_chain = np.zeros(n_iter)

    theta_current = theta_init.copy()

    # Compute distance at initial point
    inf, rew, deg = simulator_fn(*theta_current, seed=int(rng.integers(0, 2**31)))
    s_current = summary_fn(inf, rew, deg)
    d_current = normalised_distance(s_current, s_obs, mads, indices)

    n_accept = 0

    for i in range(n_iter):
        if verbose and (i + 1) % 5000 == 0:
            rate = n_accept / (i + 1)
            print(f"  Iter {i+1}/{n_iter}, accept rate: {rate:.3f}")

        # Propose
        theta_prop = rng.multivariate_normal(theta_current, proposal_cov)

        # Prior check
        if not _in_prior(theta_prop):
            chain[i] = theta_current
            dist_chain[i] = d_current
            continue

        # Simulate
        inf, rew, deg = simulator_fn(*theta_prop, seed=int(rng.integers(0, 2**31)))
        s_prop = summary_fn(inf, rew, deg)
        d_prop = normalised_distance(s_prop, s_obs, mads, indices)

        # Accept / reject
        if d_prop < epsilon:
            theta_current = theta_prop
            d_current = d_prop
            n_accept += 1

        chain[i] = theta_current
        dist_chain[i] = d_current

    acceptance_rate = n_accept / n_iter
    if verbose:
        print(f"  Final acceptance rate: {acceptance_rate:.4f}")

    return chain, dist_chain, acceptance_rate


def effective_sample_size(chain):
    """Estimate ESS from autocorrelation for each parameter dimension.

    Returns
    -------
    ess : ndarray (n_params,)
    """
    n, d = chain.shape
    ess = np.zeros(d)
    for k in range(d):
        x = chain[:, k]
        x = x - np.mean(x)
        var = np.var(x)
        if var == 0:
            ess[k] = 1.0
            continue
        # Compute autocorrelation until it drops below 0.05
        rho_sum = 0.0
        for lag in range(1, n):
            c = np.mean(x[: n - lag] * x[lag:]) / var
            if c < 0.05:
                break
            rho_sum += c
        ess[k] = n / (1 + 2 * rho_sum)
    return ess
