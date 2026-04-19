"""Synthetic Likelihood MCMC algorithm.

This module implements the Synthetic Likelihood method for simulation-based
inference using a robust covariance estimator.
"""

import numpy as np
from scipy.linalg import solve_triangular

from summary_statistic import SummaryStatistic, SummarySubset
from abc_utils import PriorSampler
from simulator import simulate

class SyntheticLikelihoodMCMC:
    """Synthetic Likelihood (Wood, 2010) Markov Chain Monte Carlo.

    This class implements the Synthetic Likelihood approach, which replaces the
    standard ABC distance check with an evaluation of an explicit Gaussian
    likelihood approximated from multiple forward simulations at each step.
    """
    def __init__(self,
                 rng: np.random.Generator,
                 prior_sampler: PriorSampler,
                 verbose: bool = False):
        """Initializes the Synthetic Likelihood MCMC runner.

        Args:
            rng: A NumPy random number generator instance.
            prior_sampler: An object to sample from and evaluate the prior.
            verbose: If True, prints progress information during the run.
        """
        self.rng = rng
        self.prior_sampler = prior_sampler
        self.verbose = verbose

    @staticmethod
    def robust_vcov(sY: np.ndarray, alpha: float = 2.0, beta: float = 1.25) -> dict:
        """Computes a robust covariance estimation using Campbell's method with QR preconditioning.

        Args:
            sY: The simulated summary statistics matrix of shape (n_stats, n_reps).
            alpha: The tuning parameter for Campbell's weights distance threshold.
            beta: The decay parameter for Campbell's weights.

        Returns:
            dict: A dictionary containing the preconditioning matrix 'E', the half
                log determinant 'half_ldet_V', and the weighted mean 'mY'.
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
        # i.e. E = (Di * R^{-1})^T
        R2_inv = solve_triangular(R2, np.eye(n_stats), lower=False)
        E = (Di[:, None] * R2_inv).T

        # half log det: sum(log|R_ii|) + sum(log(D_i))
        half_ldet_V = np.sum(np.log(np.abs(np.diag(R2)))) + np.sum(np.log(D))

        return {"E": E, "half_ldet_V": half_ldet_V, "mY": mY}

    @classmethod
    def synthetic_log_likelihood(cls, s_obs: np.ndarray, sY: np.ndarray) -> float:
        """Evaluates the log synthetic likelihood.

        Args:
            s_obs: The observed summary statistics array of shape (n_stats,).
            sY: The simulated summary statistics matrix of shape (n_stats, n_reps).

        Returns:
            float: The evaluated log synthetic likelihood, or -np.inf if not
                enough replicates are provided.
        """
        # Remove replicates with non-finite values
        finite_mask = np.all(np.isfinite(sY), axis=0)
        sY = sY[:, finite_mask]

        if sY.shape[1] < sY.shape[0] + 2:
            return -np.inf  # not enough replicates for covariance estimation

        er = cls.robust_vcov(sY)

        # Log N(s_obs | μ, Σ) = -0.5*(s-μ)^T Σ^{-1} (s-μ) - 0.5*log|Σ| + const.
        # The constant -d/2*log(2π) is omitted as it cancels in the MH ratio.
        residual = er["E"] @ (s_obs - er["mY"])
        rss = np.sum(residual ** 2)
        ll = -rss / 2.0 - er["half_ldet_V"]

        return float(ll)

    def _evaluate_theta_likelihood(self, 
                                   theta: np.ndarray, 
                                   s_obs_array: np.ndarray, 
                                   n_sim_per_eval: int, 
                                   subset: SummarySubset) -> float:
        """Simulates replicates at theta and evaluates the synthetic likelihood against observations.

        Args:
            theta: The parameter values to evaluate.
            s_obs_array: The observed summary statistics array.
            n_sim_per_eval: The number of simulation replicates to draw for covariance estimation.
            subset: The subset of summary statistics to use.

        Returns:
            float: The estimated log likelihood at theta.
        """
        rep_stats = [SummaryStatistic(*simulate(*theta, rng=self.rng)) 
                     for _ in range(n_sim_per_eval)]
        # Summaries are (n_reps, n_stats), transpose to (n_stats, n_reps)
        sY = SummaryStatistic.convert_list_to_ndarray(rep_stats, subset).T
        return self.synthetic_log_likelihood(s_obs_array, sY)

    def run(self,
            s_obs: SummaryStatistic,
            n_iter: int,
            n_sim_per_eval: int,
            theta_init: np.ndarray,
            proposal_cov: np.ndarray,
            subset: SummarySubset = SummarySubset.ALL):
        """Runs the Synthetic Likelihood MCMC algorithm.

        Args:
            s_obs: The observed summary statistics.
            n_iter: The number of MCMC iterations to run.
            n_sim_per_eval: The number of simulations per parameter evaluation.
            theta_init: The initial parameter values.
            proposal_cov: The covariance matrix of the Gaussian proposal distribution.
            subset: The subset of summary statistics to use. Defaults to ALL.

        Returns:
            tuple: A tuple containing:
                - chain (np.ndarray): The sequence of accepted parameter samples.
                - ll_chain (np.ndarray): The sequence of log-likelihood values.
                - acceptance_rate (float): The overall acceptance rate.
        """
        
        s_obs_array = s_obs.get_summaries(subset)
        
        chain = np.zeros((n_iter, 3))
        ll_chain = np.zeros(n_iter)

        theta_current = theta_init.copy()

        ll_current = self._evaluate_theta_likelihood(theta_current, s_obs_array, n_sim_per_eval, subset)
        n_accept = 0

        for i in range(n_iter):
            if self.verbose and (i + 1) % 500 == 0:
                rate = n_accept / (i + 1)
                print(f"  Iter {i+1}/{n_iter}, accept rate: {rate:.3f}, ll: {ll_current:.2f}")

            # Propose
            theta_prop = self.rng.multivariate_normal(theta_current, proposal_cov)

            # Prior check
            if not PriorSampler.in_prior(theta_prop):
                chain[i] = theta_current
                ll_chain[i] = ll_current
                continue

            # Evaluate synthetic likelihood at proposed theta
            ll_prop = self._evaluate_theta_likelihood(theta_prop, s_obs_array, n_sim_per_eval, subset)

            # Metropolis-Hastings acceptance
            log_alpha = ll_prop - ll_current
            if np.log(self.rng.random()) < log_alpha:
                theta_current = theta_prop
                ll_current = ll_prop
                n_accept += 1

            chain[i] = theta_current
            ll_chain[i] = ll_current

        acceptance_rate = n_accept / n_iter
        if self.verbose:
            print(f"  Final acceptance rate: {acceptance_rate:.4f}")

        return chain, ll_chain, acceptance_rate
