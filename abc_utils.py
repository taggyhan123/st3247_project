from __future__ import annotations
import numpy as np
import numpy.typing as npt
import torch

from sbi.utils import BoxUniform

from summary_statistic import SummaryStatistic, SummarySubset

class SummaryStatisticNormalizer:
    def __init__(self, prior_samples: list[SummaryStatistic]):
        self.mads = self._compute_mads(prior_samples)

    def _compute_mads(self, prior_samples: list[SummaryStatistic]) -> npt.NDArray:
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
        s1_array = s1.get_summaries(subset)
        s2_array = s2.get_summaries(subset)
        mads = self.mads[subset.value]
        diff = (s1_array - s2_array) / mads
        return np.sqrt(np.sum(diff ** 2))

class PriorSampler:
    PRIOR_BOUNDS = {
        "beta": (0.05, 0.50),
        "gamma": (0.02, 0.20),
        "rho": (0.00, 0.80),
    }

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
    
    def sample(self, n_samples: int) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        betas = self.rng.uniform(*PriorSampler.PRIOR_BOUNDS["beta"], size=n_samples)
        gammas = self.rng.uniform(*PriorSampler.PRIOR_BOUNDS["gamma"], size=n_samples)
        rhos = self.rng.uniform(*PriorSampler.PRIOR_BOUNDS["rho"], size=n_samples)
        return betas, gammas, rhos
    
    @staticmethod
    def in_prior(theta: np.ndarray) -> bool:
        """Check if theta is within prior bounds."""
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
    def get_torch_prior():
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