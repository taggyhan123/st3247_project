"""
Train Neural Posterior Estimation on the cached rejection ABC simulations
and compare against all other methods.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from npe import run_npe

# ---- Load cached simulations from rejection ABC ----
print("Loading cached rejection ABC simulations...")
data = np.load('results/rejection_abc_50k.npz')
thetas = data['thetas']
summaries = data['summaries']
s_obs = data['s_obs']
print(f"  {len(thetas)} simulations, {summaries.shape[1]} summaries")
print(f"  s_obs = {np.round(s_obs, 3)}")

# ---- Train NPE ----
print("\n" + "=" * 70)
print("METHOD 6: Neural Posterior Estimation (sbi, MAF density estimator)")
print("=" * 70)
t0 = time.time()
npe_samples, posterior = run_npe(
    thetas=thetas,
    summaries=summaries,
    s_obs=s_obs,
    n_posterior_samples=10_000,
    density_estimator="maf",
    seed=2026,
    verbose=True,
)
t_npe = time.time() - t0
print(f"NPE total time: {t_npe:.1f}s")

# Clip to prior bounds (NPE can sample slightly outside if density mass leaks)
from abc_rejection import PRIOR_BOUNDS
npe_samples[:, 0] = np.clip(npe_samples[:, 0], *PRIOR_BOUNDS['beta'])
npe_samples[:, 1] = np.clip(npe_samples[:, 1], *PRIOR_BOUNDS['gamma'])
npe_samples[:, 2] = np.clip(npe_samples[:, 2], *PRIOR_BOUNDS['rho'])

print(f"\nNPE posterior medians: beta={np.median(npe_samples[:,0]):.4f}, "
      f"gamma={np.median(npe_samples[:,1]):.4f}, rho={np.median(npe_samples[:,2]):.4f}")

# Save
np.savez('results/npe.npz', samples=npe_samples)

# ---- Load other methods' results for comparison ----
print("\n" + "=" * 70)
print("COMPARISON: ALL 6 METHODS")
print("=" * 70)

from abc_rejection import accept_quantile

# Rejection ABC q=1%
distances = data['distances']
acc_rej, _, _, _ = accept_quantile(thetas, distances, summaries, quantile=0.01)

# Other methods
reg_data = np.load('results/regression_abc.npz')
mcmc_data = np.load('results/abc_mcmc.npz')
smc_data = np.load('results/smc_abc.npz')
sl_data = np.load('results/synthetic_likelihood.npz')

methods = {
    'Rejection ABC (q=1%)': {'samples': acc_rej, 'weights': None},
    'Reg. Adjusted ABC':    {'samples': reg_data['adjusted_thetas'], 'weights': None},
    'ABC-MCMC':             {'samples': mcmc_data['chain_post'], 'weights': None},
    'SMC-ABC':              {'samples': smc_data['particles'], 'weights': smc_data['weights']},
    'Synthetic Likelihood': {'samples': sl_data['chain_post'], 'weights': None},
    'NPE (sbi, MAF)':       {'samples': npe_samples, 'weights': None},
}

print(f"\n{'Method':>25s}  {'beta_med':>9s} {'gamma_med':>10s} {'rho_med':>8s}  "
      f"{'beta_95w':>9s} {'gamma_95w':>10s} {'rho_95w':>8s}")
print('-' * 95)

for name, info in methods.items():
    s = info['samples']
    meds = [np.median(s[:, k]) for k in range(3)]
    widths = []
    for k in range(3):
        lo, hi = np.percentile(s[:, k], [2.5, 97.5])
        widths.append(hi - lo)
    print(f'{name:>25s}  {meds[0]:>9.4f} {meds[1]:>10.4f} {meds[2]:>8.4f}  '
          f'{widths[0]:>9.4f} {widths[1]:>10.4f} {widths[2]:>8.4f}')

print(f"\n{'Method':>25s}  {'beta 95% CI':>20s}  {'gamma 95% CI':>20s}  {'rho 95% CI':>20s}")
print('-' * 95)
for name, info in methods.items():
    s = info['samples']
    cis = []
    for k in range(3):
        lo, hi = np.percentile(s[:, k], [2.5, 97.5])
        cis.append(f'[{lo:.4f}, {hi:.4f}]')
    print(f'{name:>25s}  {cis[0]:>20s}  {cis[1]:>20s}  {cis[2]:>20s}')

print("\nDone! NPE results saved to results/npe.npz")
