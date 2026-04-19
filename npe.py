"""Neural Posterior Estimation (NPE).

This module wraps the Neural Posterior Estimation functionality from the `sbi`
package to learn an amortised posterior distribution.
"""

import numpy as np
import torch

from sbi.inference import SNPE

from summary_statistic import SummaryStatistic, SummarySubset
from abc_utils import PriorSampler

class NeuralPosteriorEstimation:
    """Neural Posterior Estimation (NPE) using the `sbi` package.

    This class trains a conditional density estimator (e.g., a Masked Autoregressive Flow)
    to approximate the posterior distribution given simulated parameter-summary pairs.
    """
    def __init__(self,
                 rng: np.random.Generator,
                 prior_sampler: PriorSampler,
                 verbose: bool = False):
        """Initializes the Neural Posterior Estimation runner.

        Args:
            rng: A NumPy random number generator instance.
            prior_sampler: An object to provide the prior distribution bounds.
            verbose: If True, prints training and sampling progress information.
        """
        self.rng = rng
        self.prior_sampler = prior_sampler
        self.verbose = verbose

    def run(self,
            thetas: np.ndarray,
            summaries: np.ndarray,
            s_obs: SummaryStatistic,
            n_posterior_samples: int = 10_000,
            density_estimator: str = "maf",
            subset: SummarySubset = SummarySubset.ALL):
        """Trains the NPE model and samples from the approximated posterior.

        Args:
            thetas: An array of simulated parameter samples of shape (n_sims, n_params).
            summaries: An array of corresponding summary statistics of shape (n_sims, n_stats).
            s_obs: The observed summary statistics.
            n_posterior_samples: The number of samples to draw from the trained posterior.
            density_estimator: The type of neural density estimator to use (e.g., "maf").
            subset: The subset of summary statistics to use. Defaults to ALL.

        Returns:
            tuple: A tuple containing:
                - samples (np.ndarray): Draws from the approximate posterior.
                - posterior (sbi.inference.posteriors.DirectPosterior): The trained posterior object.
        """
        
        seed = int(self.rng.integers(0, 2**31))
        torch.manual_seed(seed)
        
        prior = self.prior_sampler.get_torch_prior()

        theta_t = torch.as_tensor(thetas, dtype=torch.float32)
        x_t = torch.as_tensor(summaries[:, subset.value], dtype=torch.float32)
        
        s_obs_array = s_obs.get_summaries(subset)
        s_obs_t = torch.as_tensor(s_obs_array, dtype=torch.float32)

        if self.verbose:
            print(f"Training NPE on {len(thetas)} simulations "
                  f"with {x_t.shape[1]}-dim summaries (estimator={density_estimator})")

        inference = SNPE(prior=prior, density_estimator=density_estimator)
        inference.append_simulations(theta_t, x_t)
        density_est = inference.train(show_train_summary=self.verbose)
        posterior = inference.build_posterior(density_est)

        if self.verbose:
            print(f"Sampling {n_posterior_samples} from posterior at s_obs ...")

        samples_t = posterior.sample((n_posterior_samples,), x=s_obs_t, show_progress_bars=False)
        samples = samples_t.detach().cpu().numpy()

        return samples, posterior
