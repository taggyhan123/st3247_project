"""Utility classes for Approximate Bayesian Computation.

Contains the normalizer for standardizing summary statistics and the prior 
sampler for uniform prior definitions.
"""
from __future__ import annotations
import numpy as np
import numpy.typing as npt
import torch

from sbi.utils import BoxUniform

from simulator import simulate
from summary_statistic import SummaryStatistic, SummarySubset

class SummaryStatisticNormalizer:
    """Computes distances between summary statistics by normalizing via MAD.

    Uses the Median Absolute Deviation (MAD) of statistics generated from prior 
    simulations to ensure all components of the distance metric are scale-free.
    """
    def __init__(self, prior_samples: list[SummaryStatistic]):
        """Initializes the normalizer by calculating the MADs from prior samples.
        
        Args:
            prior_samples: A list of SummaryStatistic instances from pilot runs.
        """
        self.mads = self._compute_mads(prior_samples)

    @classmethod
    def from_mads(cls, mads: np.ndarray) -> SummaryStatisticNormalizer:
        """Reconstructs a normalizer from pre-computed MAD values.

        Args:
            mads: A NumPy array of median absolute deviations.

        Returns:
            SummaryStatisticNormalizer: A normalizer instance with the given MADs.
        """
        instance = cls.__new__(cls)
        instance.mads = mads
        return instance

    def _compute_mads(self, prior_samples: list[SummaryStatistic]) -> npt.NDArray:
        """Internal method to compute median absolute deviations."""
        if not prior_samples:
            return np.array([])
        summary_matrix = SummaryStatistic.convert_list_to_ndarray(prior_samples)
        medians = np.median(summary_matrix, axis=0)
        mads = np.median(np.abs(summary_matrix - medians), axis=0)
        mads[mads == 0] = 1.0
        return mads
    
    def get_normalized_distance(
        self, 
        s1: SummaryStatistic, 
        s2: SummaryStatistic,
        subset: SummarySubset = SummarySubset.ALL
    ) -> float:
        """Computes the MAD-normalized Euclidean distance between two summary sets.

        Args:
            s1: The first summary statistic object.
            s2: The second summary statistic object.
            subset: The subset of statistics to include in the distance computation.

        Returns:
            float: The normalized Euclidean distance.
        """
        s1_array = s1.get_summaries(subset)
        s2_array = s2.get_summaries(subset)
        mads = self.mads[subset.value]
        diff = (s1_array - s2_array) / mads
        return np.sqrt(np.sum(diff ** 2))

class PriorSampler:
    """Handles sampling from the uniform prior distribution of the parameters.

    Defines bounds for transmission (beta), recovery (gamma), and rewiring (rho).
    """
    PRIOR_BOUNDS = {
        "beta": (0.05, 0.50),
        "gamma": (0.02, 0.20),
        "rho": (0.00, 0.80),
    }

    def __init__(self, rng: np.random.Generator):
        """Initializes the prior sampler.
        
        Args:
            rng: A NumPy random number generator instance.
        """
        self.rng = rng
    
    def sample(self, n_samples: int) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Draws independent samples from the uniform prior distribution.

        Args:
            n_samples: The number of parameter sets to draw.

        Returns:
            tuple: Three NumPy arrays containing betas, gammas, and rhos.
        """
        betas = self.rng.uniform(*PriorSampler.PRIOR_BOUNDS["beta"], size=n_samples)
        gammas = self.rng.uniform(*PriorSampler.PRIOR_BOUNDS["gamma"], size=n_samples)
        rhos = self.rng.uniform(*PriorSampler.PRIOR_BOUNDS["rho"], size=n_samples)
        return betas, gammas, rhos
    
    @staticmethod
    def in_prior(theta: np.ndarray) -> bool:
        """Checks if a parameter vector theta falls within the prior bounds.

        Args:
            theta: A sequence or array containing [beta, gamma, rho].

        Returns:
            bool: True if the parameters fall strictly within the bounds, False otherwise.
        """
        bounds = [
            PriorSampler.PRIOR_BOUNDS["beta"],
            PriorSampler.PRIOR_BOUNDS["gamma"],
            PriorSampler.PRIOR_BOUNDS["rho"]
        ]
        for val, (lo, hi) in zip(theta, bounds):
            if val < lo or val > hi:
                return False
        return True

    @staticmethod
    def clip_to_prior(samples: np.ndarray) -> np.ndarray:
        """Clips parameter samples to lie within the prior bounds (in-place).

        Args:
            samples: An (N, 3) array of [beta, gamma, rho] parameter samples.

        Returns:
            np.ndarray: The same array, clipped in-place.
        """
        samples[:, 0] = np.clip(samples[:, 0], *PriorSampler.PRIOR_BOUNDS['beta'])
        samples[:, 1] = np.clip(samples[:, 1], *PriorSampler.PRIOR_BOUNDS['gamma'])
        samples[:, 2] = np.clip(samples[:, 2], *PriorSampler.PRIOR_BOUNDS['rho'])
        return samples

    @staticmethod
    def get_torch_prior():
        """Constructs a uniform prior distribution wrapped as a PyTorch distribution.

        Returns:
            sbi.utils.BoxUniform: The PyTorch-compatible prior bounds required for NPE.
        """
        low = torch.tensor([
            PriorSampler.PRIOR_BOUNDS["beta"][0],
            PriorSampler.PRIOR_BOUNDS["gamma"][0],
            PriorSampler.PRIOR_BOUNDS["rho"][0],
        ], dtype=torch.float32)
        high = torch.tensor([
            PriorSampler.PRIOR_BOUNDS["beta"][1],
            PriorSampler.PRIOR_BOUNDS["gamma"][1],
            PriorSampler.PRIOR_BOUNDS["rho"][1],
        ], dtype=torch.float32)
        return BoxUniform(low=low, high=high)


def run_pilot(
    prior_sampler: PriorSampler,
    rng: np.random.Generator,
    n_pilot: int = 2000,
) -> SummaryStatisticNormalizer:
    """Runs pilot simulations to fit a MAD normalizer for summary statistics.

    Args:
        prior_sampler: An object to sample from the prior distribution.
        rng: A NumPy random number generator instance.
        n_pilot: The number of pilot simulations to run. Defaults to 2000.

    Returns:
        SummaryStatisticNormalizer: A fitted normalizer for computing
            distances between summary statistics.
    """
    p_betas, p_gammas, p_rhos = prior_sampler.sample(n_pilot)
    pilot_summaries = []
    for i in range(n_pilot):
        inf, rew, deg = simulate(p_betas[i], p_gammas[i], p_rhos[i], rng=rng)
        pilot_summaries.append(SummaryStatistic(inf, rew, deg))
    return SummaryStatisticNormalizer(pilot_summaries)