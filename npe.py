"""
Neural Posterior Estimation (NPE)  using the `sbi` package.

NPE trains a conditional density estimator (normalizing flow) qϕ(θ | s) on
simulated (θ, s) pairs to directly approximate the posterior:

    qϕ(θ | s_obs) ≈ p(θ | s_obs)

Once trained, the network can be sampled from in milliseconds — fully amortized
inference. Reuses the simulations already produced by rejection ABC, so no
extra simulation budget is required.

References:
  Papamakarios & Murray (2016). "Fast ε-free Inference of Simulation Models
    with Bayesian Conditional Density Estimation."
  Greenberg, Nonnenmacher & Macke (2019). "Automatic Posterior Transformation
    for Likelihood-Free Inference."
  Tejero-Cantero et al. (2020). "sbi: A toolkit for simulation-based inference."
"""

import numpy as np
import torch

from sbi.inference import SNPE
from sbi.utils import BoxUniform

from abc_rejection import PRIOR_BOUNDS


def build_prior():
    """Box-uniform prior over (β, γ, ρ) matching PRIOR_BOUNDS."""
    low = torch.tensor([
        PRIOR_BOUNDS["beta"][0],
        PRIOR_BOUNDS["gamma"][0],
        PRIOR_BOUNDS["rho"][0],
    ], dtype=torch.float32)
    high = torch.tensor([
        PRIOR_BOUNDS["beta"][1],
        PRIOR_BOUNDS["gamma"][1],
        PRIOR_BOUNDS["rho"][1],
    ], dtype=torch.float32)
    return BoxUniform(low=low, high=high)


def run_npe(
    thetas,
    summaries,
    s_obs,
    n_posterior_samples=10_000,
    density_estimator="maf",
    seed=0,
    verbose=True,
):
    """Train a Neural Posterior Estimator and sample from it.

    Parameters
    ----------
    thetas : ndarray (n, 3)
        Simulated parameter draws.
    summaries : ndarray (n, d)
        Corresponding summary statistics.
    s_obs : ndarray (d,)
        Observed summary statistics.
    n_posterior_samples : int
        Number of samples to draw from the trained posterior.
    density_estimator : str
        Type of normalizing flow ("maf", "nsf", "made", ...).
    seed : int
    verbose : bool

    Returns
    -------
    samples : ndarray (n_posterior_samples, 3)
    posterior : sbi posterior object (queryable for any new s_obs)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    prior = build_prior()

    theta_t = torch.as_tensor(thetas, dtype=torch.float32)
    x_t = torch.as_tensor(summaries, dtype=torch.float32)
    s_obs_t = torch.as_tensor(s_obs, dtype=torch.float32)

    if verbose:
        print(f"Training NPE on {len(thetas)} simulations "
              f"with {summaries.shape[1]}-dim summaries (estimator={density_estimator})")

    inference = SNPE(prior=prior, density_estimator=density_estimator)
    inference.append_simulations(theta_t, x_t)
    density_est = inference.train(show_train_summary=verbose)
    posterior = inference.build_posterior(density_est)

    if verbose:
        print(f"Sampling {n_posterior_samples} from posterior at s_obs ...")

    samples_t = posterior.sample((n_posterior_samples,), x=s_obs_t, show_progress_bars=False)
    samples = samples_t.detach().cpu().numpy()

    return samples, posterior
