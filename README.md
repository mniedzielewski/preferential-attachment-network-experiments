# Linear Preferential Attachment in Directed Networks


## 📖 Project Overview

This project implements the computational and statistical aspects of the **Directed Linear Preferential Attachment Model** (introduced by Bollobás, Borgs, Chayes, and Riordan, 2003), with parameter estimation by maximum likelihood (developed by Wan, Wang, Davis, and Resnick, 2017). 

The repository provides a complete pipeline to:
1. **Simulate** directed scale-free networks in highly optimized linear time.
2. **Estimate** model parameters ($lpha, eta, \delta_{in}, \delta_{out}$) using Maximum Likelihood Estimation via numerical optimization (Brent's method).
3. **Evaluate** goodness-of-fit using Hill estimators, temporal trajectory tracking, and Kolmogorov-Smirnov statistics.
4. **Forecast** degree distributions on synthetic and real-world temporal networks.

The full theoretical background and numerical experiment results can be found in `report.pdf`.

---

## 🗂️ Repository Structure

The project is modularized into several Python scripts separating simulation, statistical theory, estimation, and experiments:

```text
bachelor-fragment/
├── data/                      # Directory for real-world temporal network datasets (SNAP)
├── out/                       # Output directory for generated figures (.pdf/.png) and tables (.csv/.tex)
├── pa_simulator.py            # Highly optimized, O(n) network generator using Numba
├── pa_mle.py                  # Maximum Likelihood Estimator implementation (profile likelihood & Brent's method)
├── pa_theory.py               # Exact theoretical constants (degree asymptotics, Fisher information bounds)
├── pa_stats.py                # Goodness-of-fit diagnostics, temporal trajectories, and Hill tail index estimator
├── pa_experiments.py          # Synthetic network experiments (Monte Carlo, efficiency, convergence rates)
├── pa_realdata.py             # Parsers for temporal edgelists and SNAP datasets
├── pa_real_experiments.py     # Out-of-sample predictive validation on real-world networks
├── README.md                  # This file
├── .gitignore                 # Git ignore file
└── report.pdf                 # Final numerical experiments report (LaTeX compiled)
```

---

## ⚙️ Installation & Requirements

### 1. Python Dependencies
The project requires **Python 3.8+** and standard scientific computing libraries. Install the dependencies using:
```bash
pip install numpy scipy matplotlib numba
```
*Note: `Numba` is required for the JIT compilation of the core simulation loop (`pa_simulator.py`), drastically reducing execution time for networks with millions of edges.*

### 2. Required Data Installation (Real Network Experiments)
To run the real-world network experiments (`pa_real_experiments.py`), you need to download four temporal network datasets from the [Stanford Large Network Dataset Collection (SNAP)](https://snap.stanford.edu/data/). 

Please download the following datasets and place them directly into the `data/` folder in the project root (alternatively, you can set the `PA_DATA_DIR` environment variable to a custom path):

1. **email-Eu-core temporal network**: [`email-Eu-core-temporal.txt`](https://snap.stanford.edu/data/email-Eu-core-temporal.html)
2. **CollegeMsg**: [`CollegeMsg.txt`](https://snap.stanford.edu/data/CollegeMsg.html)
3. **MathOverflow**: [`sx-mathoverflow-a2q.txt`](https://snap.stanford.edu/data/sx-mathoverflow.html)
4. **Reddit Hyperlinks**: [`soc-redditHyperlinks-body.tsv`](https://snap.stanford.edu/data/soc-RedditHyperlinks.html)

**Expected Data Directory Structure:**
```text
data/
├── email-Eu-core-temporal.txt
├── CollegeMsg.txt
├── sx-mathoverflow-a2q.txt
└── soc-redditHyperlinks-body.tsv
```

---

## 🚀 Usage 

### Running Synthetic Network Experiments
To reproduce the Monte Carlo simulations, limit degree frequency tests, MLE convergence diagnostics, and tail exponent evaluations (Experiments 1a, 1b, 1c, C, C-onset):

```bash
python pa_experiments.py
```
*Outputs (figures and CSVs) will be automatically saved to the `out/` directory.*

### Running Real-World Data Validation
Once the datasets are placed in the `data/` directory, you can run the out-of-sample predictive validation and parameter stability checks:

```bash
python pa_real_experiments.py
```
*This will parse the temporal graphs, fit the model on a $70\%$ training split, simulate the network's future growth, and evaluate the fit against the actual out-of-sample degree distributions.*

---

## 📚 References
* **[BBCR03]** Bollobás, B., Borgs, C., Chayes, J. T., Riordan, O. (2003). *Directed scale-free graphs*. Proceedings of the 14th Annual ACM–SIAM SODA.
* **[WWDR17]** Wan, P., Wang, T., Davis, R. A., Resnick, S. I. (2017). *Fitting the linear preferential attachment model*. Electronic Journal of Statistics 11(2).
