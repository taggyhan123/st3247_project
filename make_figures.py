"""
Generate all figures for the report from saved .npz results.
Outputs to figures/.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import corner

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abc_rejection import accept_quantile, normalised_distance
from summary_stats import IDX_INFECTED, IDX_REWIRING, IDX_DEGREE, IDX_ALL

os.makedirs('figures', exist_ok=True)

plt.rcParams['figure.dpi'] = 120

# Load all results
data = np.load('results/rejection_abc_50k.npz')
thetas = data['thetas']
distances = data['distances']
summaries = data['summaries']
mads = data['mads']
s_obs = data['s_obs']

reg_data = np.load('results/regression_abc.npz')
mcmc_data = np.load('results/abc_mcmc.npz')
smc_data = np.load('results/smc_abc.npz')
sl_data = np.load('results/synthetic_likelihood.npz')
npe_data = np.load('results/npe.npz')

acc_rej, _, _, _ = accept_quantile(thetas, distances, summaries, quantile=0.01)

param_labels = [r'$\beta$', r'$\gamma$', r'$\rho$']

# ======================================================================
# Figure: Rejection ABC corner plot
# ======================================================================
fig = corner.corner(
    acc_rej,
    labels=param_labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={'fontsize': 12},
    color='steelblue',
)
plt.suptitle('Rejection ABC Posterior (q=1%, all 12 summaries)', fontsize=14, y=1.02)
plt.savefig('figures/rejection_abc_corner.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/rejection_abc_corner.png")

# ======================================================================
# Figure: Summary statistics comparison
# ======================================================================
summary_subsets = {
    'Infected only':    IDX_INFECTED,
    'Rewiring only':    IDX_REWIRING,
    'Degree only':      IDX_DEGREE,
    'Inf + Rewiring':   IDX_INFECTED + IDX_REWIRING,
    'Inf + Degree':     IDX_INFECTED + IDX_DEGREE,
    'All 12':           IDX_ALL,
}
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

results = {}
for name, indices in summary_subsets.items():
    dists = np.array([
        normalised_distance(summaries[i], s_obs, mads, indices)
        for i in range(len(summaries))
    ])
    acc_t, _, _, _ = accept_quantile(thetas, dists, summaries, quantile=0.01)
    results[name] = acc_t

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for col in range(3):
    ax = axes[0, col]
    for i, (name, t) in enumerate(results.items()):
        ax.hist(t[:, col], bins=25, density=True, alpha=0.4,
                color=colors[i], label=name, histtype='stepfilled', linewidth=1.5)
    ax.set_xlabel(param_labels[col], fontsize=12)
    ax.set_ylabel('Density')
    ax.set_title(f'Marginal posterior: {param_labels[col]}')
    if col == 2:
        ax.legend(fontsize=7, loc='upper right')

# CI widths bar chart
ci_widths = {name: [] for name in results}
for name, t in results.items():
    for col in range(3):
        lo, hi = np.percentile(t[:, col], [2.5, 97.5])
        ci_widths[name].append(hi - lo)

x = np.arange(3)
width = 0.13
for i, (name, widths) in enumerate(ci_widths.items()):
    axes[1, 0].bar(x + i * width, widths, width, color=colors[i], alpha=0.7, label=name)
axes[1, 0].set_xticks(x + 2.5 * width)
axes[1, 0].set_xticklabels(param_labels)
axes[1, 0].set_ylabel('95% CI Width')
axes[1, 0].set_title('Posterior Width Comparison')
axes[1, 0].legend(fontsize=6)

# beta-rho scatter
for ax_idx, (name, color) in enumerate([('Infected only', '#e41a1c'), ('All 12', '#a65628')]):
    ax = axes[1, 1 + ax_idx]
    t = results[name]
    ax.scatter(t[:, 0], t[:, 2], alpha=0.3, s=8, color=color)
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\rho$')
    ax.set_title(name)
    corr = np.corrcoef(t[:, 0], t[:, 2])[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Summary Statistics Comparison (q=1%)', fontsize=14)
plt.tight_layout()
plt.savefig('figures/summary_stats_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/summary_stats_comparison.png")

# ======================================================================
# Figure: All methods comparison
# ======================================================================
methods = {
    'Rejection ABC': {'samples': acc_rej, 'weights': None},
    'Reg. Adjusted': {'samples': reg_data['adjusted_thetas'], 'weights': None},
    'ABC-MCMC':      {'samples': mcmc_data['chain_post'], 'weights': None},
    'SMC-ABC':       {'samples': smc_data['particles'], 'weights': smc_data['weights']},
    'Synth. Likelihood': {'samples': sl_data['chain_post'], 'weights': None},
    'NPE (MAF)':     {'samples': npe_data['samples'], 'weights': None},
}
method_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for col in range(3):
    ax = axes[col]
    for i, (name, info) in enumerate(methods.items()):
        s = info['samples']
        w = info['weights']
        ax.hist(s[:, col], bins=30, density=True, alpha=0.35,
                color=method_colors[i], label=name, weights=w,
                histtype='stepfilled')
    ax.set_xlabel(param_labels[col], fontsize=13)
    ax.set_ylabel('Density')
    ax.set_title(f'Posterior: {param_labels[col]}')
    if col == 2:
        ax.legend(fontsize=8)

plt.suptitle('All Six Methods: Posterior Marginals', fontsize=14)
plt.tight_layout()
plt.savefig('figures/all_methods_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/all_methods_comparison.png")

print("\nAll figures generated.")
