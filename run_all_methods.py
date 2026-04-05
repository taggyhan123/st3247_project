"""
Run all 5 SBI methods and compare results.
Saves results to results/ and prints a comparison table.
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_all
from summary_stats import compute_observed_summaries, compute_summaries, IDX_ALL
from abc_rejection import run_rejection_abc, accept_quantile, PRIOR_BOUNDS
from abc_regression import regression_adjust
from abc_mcmc import run_abc_mcmc, effective_sample_size
from smc_abc import run_smc_abc
from synthetic_likelihood import run_synthetic_likelihood_mcmc
from simulator import simulate_fast

os.makedirs('results', exist_ok=True)

# ---- Load data ----
print("=" * 70)
print("Loading data and warming up Numba...")
inf_ts, rew_ts, deg_hist = load_all()
s_obs, s_per_rep = compute_observed_summaries(inf_ts, rew_ts, deg_hist)
_ = simulate_fast(0.2, 0.1, 0.3, seed=0)
print(f"Observed summaries ({len(s_obs)} stats): {np.round(s_obs, 3)}")

# ---- 1. Rejection ABC ----
print("\n" + "=" * 70)
print("METHOD 1: Rejection ABC (50,000 simulations)")
print("=" * 70)
t0 = time.time()
thetas, distances, summaries, mads = run_rejection_abc(
    s_obs=s_obs, n_sim=50_000, simulator_fn=simulate_fast,
    indices=IDX_ALL, seed=42, verbose=True,
)
t_rej = time.time() - t0
print(f"Completed in {t_rej:.1f}s")

acc_rej, acc_d_rej, acc_s_rej, thr_rej = accept_quantile(thetas, distances, summaries, quantile=0.01)
print(f"Accepted {len(acc_rej)} samples (q=1%), threshold={thr_rej:.3f}")

np.savez('results/rejection_abc_50k.npz',
         thetas=thetas, distances=distances, summaries=summaries, mads=mads, s_obs=s_obs)

# ---- 2. Regression-adjusted ABC ----
print("\n" + "=" * 70)
print("METHOD 2: Regression-Adjusted ABC (Beaumont et al. 2002)")
print("=" * 70)
acc_reg, acc_d_reg, acc_s_reg, thr_reg = accept_quantile(thetas, distances, summaries, quantile=0.05)
print(f"Using {len(acc_reg)} samples (q=5%) for regression adjustment")

t0 = time.time()
adjusted_thetas = regression_adjust(acc_reg, acc_s_reg, acc_d_reg, s_obs)
t_regadj = time.time() - t0

adjusted_thetas[:, 0] = np.clip(adjusted_thetas[:, 0], *PRIOR_BOUNDS['beta'])
adjusted_thetas[:, 1] = np.clip(adjusted_thetas[:, 1], *PRIOR_BOUNDS['gamma'])
adjusted_thetas[:, 2] = np.clip(adjusted_thetas[:, 2], *PRIOR_BOUNDS['rho'])
print(f"Regression adjustment took {t_regadj:.3f}s")

np.savez('results/regression_abc.npz', adjusted_thetas=adjusted_thetas)

# ---- 3. ABC-MCMC ----
print("\n" + "=" * 70)
print("METHOD 3: ABC-MCMC (Marjoram et al. 2003)")
print("=" * 70)
proposal_cov = np.cov(acc_rej.T) * 0.5
theta_init = np.median(acc_rej, axis=0)
epsilon = thr_rej * 1.2

print(f"  epsilon = {epsilon:.4f}, theta_init = {theta_init}")

t0 = time.time()
chain, dist_chain, acc_rate = run_abc_mcmc(
    s_obs=s_obs, mads=mads, epsilon=epsilon,
    n_iter=30_000, simulator_fn=simulate_fast,
    theta_init=theta_init, proposal_cov=proposal_cov,
    indices=IDX_ALL, seed=123, verbose=True,
)
t_mcmc = time.time() - t0
print(f"Completed in {t_mcmc:.1f}s, acceptance rate: {acc_rate:.4f}")

burn_in = 5000
chain_post = chain[burn_in:]
ess = effective_sample_size(chain_post)
print(f"ESS: beta={ess[0]:.0f}, gamma={ess[1]:.0f}, rho={ess[2]:.0f}")

np.savez('results/abc_mcmc.npz', chain=chain, chain_post=chain_post, acc_rate=acc_rate)

# ---- 4. SMC-ABC ----
print("\n" + "=" * 70)
print("METHOD 4: SMC-ABC (Beaumont et al. 2009)")
print("=" * 70)
t0 = time.time()
smc_particles, smc_weights, smc_epsilons, smc_all_particles = run_smc_abc(
    s_obs=s_obs, mads=mads,
    n_particles=1000, n_generations=10,
    simulator_fn=simulate_fast,
    indices=IDX_ALL, alpha=0.5, min_epsilon=0.5,
    seed=456, verbose=True,
)
t_smc = time.time() - t0
print(f"Completed in {t_smc:.1f}s, {len(smc_epsilons)} generations")
print(f"Final epsilon: {smc_epsilons[-1]:.4f}")

np.savez('results/smc_abc.npz',
         particles=smc_particles, weights=smc_weights, epsilons=np.array(smc_epsilons))

# ---- 5. Synthetic Likelihood ----
print("\n" + "=" * 70)
print("METHOD 5: Synthetic Likelihood (Wood, 2010)")
print("=" * 70)
sl_proposal_cov = np.cov(acc_rej.T) * 0.3
sl_theta_init = np.median(acc_rej, axis=0)

print(f"  n_iter=5000, n_sim_per_eval=200, theta_init={sl_theta_init}")

t0 = time.time()
sl_chain, sl_ll_chain, sl_acc_rate = run_synthetic_likelihood_mcmc(
    s_obs=s_obs, n_iter=5_000, n_sim_per_eval=200,
    simulator_fn=simulate_fast,
    theta_init=sl_theta_init,
    proposal_cov=sl_proposal_cov,
    seed=789, verbose=True,
)
t_sl = time.time() - t0
print(f"Completed in {t_sl:.1f}s, acceptance rate: {sl_acc_rate:.4f}")

sl_burn_in = 1000
sl_chain_post = sl_chain[sl_burn_in:]

np.savez('results/synthetic_likelihood.npz',
         chain=sl_chain, ll_chain=sl_ll_chain, chain_post=sl_chain_post, acc_rate=sl_acc_rate)

# ---- COMPARISON TABLE ----
print("\n" + "=" * 70)
print("COMPARISON OF ALL METHODS")
print("=" * 70)

methods = {
    'Rejection ABC (q=1%)': {'samples': acc_rej, 'weights': None, 'time': t_rej, 'n_sims': 50000 + 2000},
    'Reg. Adjusted ABC':    {'samples': adjusted_thetas, 'weights': None, 'time': t_rej + t_regadj, 'n_sims': 50000 + 2000},
    'ABC-MCMC':             {'samples': chain_post, 'weights': None, 'time': t_mcmc, 'n_sims': 30000},
    'SMC-ABC':              {'samples': smc_particles, 'weights': smc_weights, 'time': t_smc, 'n_sims': None},
    'Synthetic Likelihood': {'samples': sl_chain_post, 'weights': None, 'time': t_sl, 'n_sims': 5000 * 200},
}

print(f"\n{'Method':>25s}  {'beta_med':>9s} {'gamma_med':>10s} {'rho_med':>8s}  "
      f"{'beta_95w':>9s} {'gamma_95w':>10s} {'rho_95w':>8s}  {'Time(s)':>8s}")
print('-' * 105)

for name, info in methods.items():
    s = info['samples']
    meds = [np.median(s[:, k]) for k in range(3)]
    widths = []
    for k in range(3):
        lo, hi = np.percentile(s[:, k], [2.5, 97.5])
        widths.append(hi - lo)
    print(f'{name:>25s}  {meds[0]:>9.4f} {meds[1]:>10.4f} {meds[2]:>8.4f}  '
          f'{widths[0]:>9.4f} {widths[1]:>10.4f} {widths[2]:>8.4f}  {info["time"]:>8.1f}')

# ---- 95% CI table ----
print(f"\n{'Method':>25s}  {'beta 95% CI':>20s}  {'gamma 95% CI':>20s}  {'rho 95% CI':>20s}")
print('-' * 95)
for name, info in methods.items():
    s = info['samples']
    cis = []
    for k in range(3):
        lo, hi = np.percentile(s[:, k], [2.5, 97.5])
        cis.append(f'[{lo:.4f}, {hi:.4f}]')
    print(f'{name:>25s}  {cis[0]:>20s}  {cis[1]:>20s}  {cis[2]:>20s}')

print("\nDone! Results saved to results/")
