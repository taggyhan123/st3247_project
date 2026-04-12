import numpy as np
from summary_statistic import SummaryStatistic, SummarySubset
from abc_utils import PriorSampler, SummaryStatisticNormalizer
from simulator import simulate

class MCMCABC:
    def __init__(self,
                 rng: np.random.Generator,
                 normalizer: SummaryStatisticNormalizer,
                 prior_sampler: PriorSampler,
                 verbose: bool = False):
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
        """Estimate ESS from autocorrelation for each parameter dimension."""
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