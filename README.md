# ST3247 Project: Parameter Inference for Epidemic Models

This repository contains our codebase for parameter inference on network epidemic models for the ST3247 Project. It compares various Approximate Bayesian Computation (ABC) methods, Synthetic Likelihood, and Neural Posterior Estimation (NPE) to infer the spreading parameters ($\beta$, $\gamma$, $\rho$).

## How to Use

### Prerequisites
Ensure you have Python 3.9+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```
`torch` and `sbi` can run on CPU; a GPU is optional and not required for this project.

### Running the Experiments
To run the entire suite of experiments, execute the main orchestrator script:
```bash
python main.py
```
This script will sequentially:
1. Run all 6 core inference algorithms and cache their results in the `results/` directory.
2. Run the budget-matched comparison.
3. Run the synthetic truth recovery experiment.
4. Generate all plots and save them in the `figures/` directory.

After a successful first run, you should see cached `.npz` outputs in `results/` and the comparison plots in `figures/`.

### Caching
To avoid redundant, expensive simulation calls, all intermediate results are automatically cached as `.npz` files in the `results/` directory. 
* If a script requires data (e.g., `make_figures.py`), it will load the cached files instantly.
* If you want to force a clean re-run of any experiment from scratch, simply delete the corresponding `.npz` file in the `results/` folder.

## What Each Module Does

### Orchestration & Experiments
* **`main.py`**: The central orchestrator. Runs core models, aggregates metrics into a comparison table, and triggers the other experiment scripts.
* **`budget_matched.py`**: Runs an isolated experiment restricting specific algorithms (Rejection ABC, SMC-ABC, NPE) to a strict budget of 50,000 simulations.
* **`synthetic_truth.py`**: Simulates synthetic data using known parameters and checks if the 95% Credible Intervals of the algorithms successfully cover the true values.
* **`run_summary_subsets.py`**: Analyzes how using different subsets of summary statistics impacts the width of the resulting posteriors.
* **`posterior_predictive.py`**: Uses the NPE posterior to simulate outcome envelopes and overlays them against the actual observed data.
* **`robustness.py`**: Verifies that conclusions are stable across random seeds (42, 123, 9999) and NPE training-set sizes (10k, 25k, 50k). Saves results to `results/robustness.npz`.
* **`make_figures.py`**: A plotting engine that loads cached `.npz` files to build the comparative visualizations without re-running any simulations.

### Core Implementations
* **`simulator.py`**: Contains the Numba-optimized, JIT-compiled forward network simulators.
* **`summary_statistic.py`**: Defines classes for computing, aggregating, and extracting specific subsets of summary statistics from time-series data.
* **`abc_rejection.py`**: Implementation of the Basic Rejection ABC algorithm.
* **`abc_regression.py`**: Implementation of Regression-Adjusted ABC.
* **`abc_mcmc.py`**: Implementation of the ABC-MCMC algorithm.
* **`smc_abc.py`**: Implementation of the Sequential Monte Carlo ABC (SMC-ABC) algorithm.
* **`synthetic_likelihood.py`**: Implementation of the Synthetic Likelihood MCMC algorithm.
* **`npe.py`**: Wraps the `sbi` package to perform Neural Posterior Estimation (NPE).
* **`abc_utils.py`**: Utility classes for sampling from uniform priors and normalizing summary distances via Median Absolute Deviation (MAD).
* **`data_loader.py`**: Utility script to load the observed network data (infected, rewiring, and degree histories).
