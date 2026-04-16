"""
Run summary statistics subset comparison using cached rejection ABC simulations.
Recomputes distances under different summary subsets and reports posterior widths.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abc_rejection import normalised_distance, accept_quantile
from summary_stats import IDX_INFECTED, IDX_REWIRING, IDX_DEGREE, IDX_ALL

# Load cached rejection ABC simulations
data = np.load('results/rejection_abc_50k.npz')
thetas = data['thetas']
summaries = data['summaries']
mads = data['mads']
s_obs = data['s_obs']

print(f"Loaded {len(thetas)} simulations")

summary_subsets = {
    'Infected only (s1-s5)':   IDX_INFECTED,
    'Rewiring only (s6-s9)':   IDX_REWIRING,
    'Degree only (s10-s12)':   IDX_DEGREE,
    'Infected + Rewiring':     IDX_INFECTED + IDX_REWIRING,
    'Infected + Degree':       IDX_INFECTED + IDX_DEGREE,
    'All 12':                  IDX_ALL,
}

q = 0.01
results = {}
for name, indices in summary_subsets.items():
    dists = np.array([
        normalised_distance(summaries[i], s_obs, mads, indices)
        for i in range(len(summaries))
    ])
    acc_t, acc_d, acc_s, thr = accept_quantile(thetas, dists, summaries, quantile=q)
    results[name] = acc_t

print(f"\n{'Subset':>30s}  {'CI_beta':>9s} {'CI_gamma':>10s} {'CI_rho':>8s}  "
      f"{'r(b,r)':>8s} {'r(b,g)':>8s} {'r(g,r)':>8s}")
print('-' * 90)

for name, t in results.items():
    widths = []
    for k in range(3):
        lo, hi = np.percentile(t[:, k], [2.5, 97.5])
        widths.append(hi - lo)
    r_br = np.corrcoef(t[:, 0], t[:, 2])[0, 1]
    r_bg = np.corrcoef(t[:, 0], t[:, 1])[0, 1]
    r_gr = np.corrcoef(t[:, 1], t[:, 2])[0, 1]
    print(f'{name:>30s}  {widths[0]:>9.4f} {widths[1]:>10.4f} {widths[2]:>8.4f}  '
          f'{r_br:>8.4f} {r_bg:>8.4f} {r_gr:>8.4f}')

# Save
np.savez('results/summary_subsets.npz',
         **{name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus'): v
            for name, v in results.items()})
print("\nSaved to results/summary_subsets.npz")
