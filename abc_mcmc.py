"""MCMC Approximate Bayesian Computation (ABC-MCMC).

This module implements the ABC-MCMC algorithm (Marjoram et al. 2003) for
sampling from the approximate posterior distribution.
"""

import numpy as np
from summary_statistic import SummaryStatistic, SummarySubset
from abc_utils import PriorSampler, SummaryStatisticNormalizer
from simulator import simulate

class MCMCABC:
    """Markov Chain Monte Carlo Approximate Bayesian Computation (ABC-MCMC).

    This class implements the ABC-MCMC algorithm for sampling from the approximate 
    posterior distribution of parameters given observed summary statistics.
    """

    def __init__(self,
                 rng: np.random.Generator,
                 normalizer: SummaryStatisticNormalizer,
                 prior_sampler: PriorSampler,
                 verbose: bool = False):
        """Initializes the MCMCABC runner.

        Args:
            rng: A NumPy random number generator instance.
            normalizer: A normalizer for computing distances between summary
                statistics.
            prior_sampler: An object to sample from the prior distribution and
                evaluate prior bounds.
            verbose: If True, prints progress information during the run.
        """
        self.rng = rng
        self.normalizer = normalizer
        self.prior_sampler = prior_sampler
        self.verbose = verbose

    def run(self,
            s_obs: SummaryStatistic,
            n_iter: int,
            epsilon: float,
            theta_init: np.ndarray,
            proposal_cov: np.ndarray,
            subset: SummarySubset = SummarySubset.ALL,
            n_reps_per_sim: int = 1):
        """Runs the ABC-MCMC algorithm.

        Args:
            s_obs: The observed summary statistics.
            n_iter: The number of MCMC iterations to run.
            epsilon: The distance threshold for accepting proposed parameters.
            theta_init: The initial parameter values to start the Markov chain.
            proposal_cov: The covariance matrix of the Gaussian random walk
                proposal distribution.
            subset: The subset of summary statistics to use for distance
                computation. Defaults to SummarySubset.ALL.
            n_reps_per_sim: The number of simulation replicates to run for each
                parameter evaluation to reduce simulator noise. Defaults to 1.

        Returns:
            tuple: A tuple containing the Markov chain of parameter samples,
                the corresponding distances, and the overall acceptance rate.
        """
        
        chain = np.zeros((n_iter, 3))
        dist_chain = np.zeros(n_iter)

        theta_current = theta_init.copy()

        # Compute distance at initial point
        rep_stats = [SummaryStatistic(*simulate(*theta_current, rng=self.rng)) 
                     for _ in range(n_reps_per_sim)]
        stat_current = SummaryStatistic.aggregate_summary_statistics(rep_stats, agg="mean")
        d_current = self.normalizer.get_normalized_distance(stat_current, s_obs, subset)

        n_accept = 0

        for i in range(n_iter):
            if self.verbose and (i + 1) % 5000 == 0:
                rate = n_accept / (i + 1)
                print(f"  Iter {i+1}/{n_iter}, accept rate: {rate:.3f}")

            # Propose
            theta_prop = self.rng.multivariate_normal(theta_current, proposal_cov)

            # Prior check
            if not PriorSampler.in_prior(theta_prop):
                chain[i] = theta_current
                dist_chain[i] = d_current
                continue

            # Simulate
            rep_stats_prop = [SummaryStatistic(*simulate(*theta_prop, rng=self.rng)) 
                              for _ in range(n_reps_per_sim)]
            stat_prop = SummaryStatistic.aggregate_summary_statistics(rep_stats_prop, agg="mean")
            d_prop = self.normalizer.get_normalized_distance(stat_prop, s_obs, subset)

            # Accept / reject
            if d_prop < epsilon:
                theta_current = theta_prop
                d_current = d_prop
                n_accept += 1

            chain[i] = theta_current
            dist_chain[i] = d_current

        acceptance_rate = n_accept / n_iter
        if self.verbose:
            print(f"  Final acceptance rate: {acceptance_rate:.4f}")

        return chain, dist_chain, acceptance_rate

    @staticmethod
    def effective_sample_size(chain: np.ndarray) -> np.ndarray:
        """Estimates the effective sample size (ESS) for each parameter dimension.

        The ESS is estimated using the autocorrelation function of the chain,
        summing the autocorrelations until they drop below 0.05.

        Args:
            chain: The MCMC chain of samples, typically of shape (n_samples, n_dimensions).

        Returns:
            np.ndarray: An array of estimated effective sample sizes for each dimension.
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