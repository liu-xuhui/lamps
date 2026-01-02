
# Environment Setup Instructions

This document describes how to create a reproducible Python environment for running the AdaMP simulations and generating the linear and nonlinear plots. All required Python dependencies are listed in `env/requirements.txt`, and the environment can be created automatically using the provided script `env/setup_venv.sh`.

A copy of the required third-party package `stability_selection` is included inside the repository under  
`src/python/adamp/third_party/stability_selection/`.  
It is imported locally and does not require installation through pip.

No specific Python version is strictly required, but compatibility has been tested on Python versions 3.10 through 3.13.

---

## Environment Setup (Windows Git Bash and macOS/Linux)

The setup procedure is identical for Windows (Git Bash) and macOS/Linux. Users only need to ensure they are running commands either in **Git Bash** (Windows) or a **standard terminal** (macOS/Linux).

### Step 1. Clone the repository

```

git clone <repository_url>
cd <repository_directory>

```

### Step 2. Create a virtual environment

```

python -m venv .venv

```

On macOS/Linux, if needed:

```

python3 -m venv .venv

```

### Step 3. Run the setup script

```

./env/setup_venv.sh

```

This script initializes the virtual environment, upgrades pip, and installs all required packages listed in `env/requirements.txt`.

### Step 4. Activate the environment in future sessions

Windows (Git Bash):

```

source .venv/Scripts/activate

```

macOS/Linux:

```

source .venv/bin/activate

```

---

## Running Experiments and Generating Plots

This repository contains code for running **linear** and **nonlinear** feature-selection experiments and reproducing all figures in the paper.

All experiment scripts write **temporary simulation outputs** to the `temp_results/` directory. These are then aggregated into clean result files inside the `results/` directory, which are used for plotting.

---

### 🔹 Nonlinear Experiments (Additive & Non-Additive)

All nonlinear experiment scripts are located under:

```
experiments/nonlinear/additive/
experiments/nonlinear/nonadditive/
```

Each folder contains scripts for all competing methods (AdaMP, HSIC-Lasso, Knockoff, MARS, SPAM).

#### Step 1 — Run the Simulation Scripts

From either directory, run the desired experiment scripts. For example:

```
python experiments/nonlinear/additive/run_knockoff.py
python experiments/nonlinear/additive/run_hsic.py
Rscript experiments/nonlinear/additive/run_adamp.R
...
```

Temporary results will be written to:

```
temp_results/
```

#### Step 2 — Summarize Results

After simulations finish, summarize the temporary files:

```
python experiments/nonlinear/additive/summarize_result.py
```

or

```
python experiments/nonlinear/nonadditive/summarize_result.py
```

This will generate aggregated result files under:

```
results/nonlinear/
```

#### Step 3 — Generate Figures

Once the results are available, the following plotting scripts reproduce the **paper-ready figures exactly**:

```
python src/python_plotting/make_nonlinear_additive_plots.py
python src/python_plotting/make_nonlinear_nonadditive_plots.py
python src/python_plotting/make_interaction_rate_plots.py
```

Figures will be saved automatically under the `results/` directory hierarchy.

---

### 🔹 Linear Experiments

Linear simulation scripts are located under:

```
experiments/linear/
```

After running the linear simulations and summarizing results, publication-ready figures are generated via:

```
python src/python_plotting/make_linear_plots.py
```

Output is written to:

```
results/linear/
```

---

### 🔹 Theory-Validation Experiments (Single-Seed Runs)

The following scripts in `experiments/other/` run **one fixed random seed** to produce the datasets and outputs used in the theory-validation figures:

```
experiments/other/run_locomp_linear.py
experiments/other/run_locosplit_linear.py
experiments/other/run_one_linear.py
experiments/other/run_one_nonlinear_additive.R
```

---

## Plot Generation (Summary)

Four plotting utilities are provided under:

```
src/python_plotting/
```

These scripts read the processed results in `results/` and generate the figures used in the paper:

* `make_linear_plots.py`
* `make_nonlinear_additive_plots.py`
* `make_nonlinear_nonadditive_plots.py`
* `make_interaction_rate_plots.py`

Before running the nonlinear plotting scripts, please set the following parameters inside the script:

```python
eval_metd    = "f1_score"     # "f1_score", "precision", or "recall"
oracle       = 1              # 1 = oracle setting, 0 = non-oracle setting
permute      = 1              # 1 = permuted correlation matrix, 0 = non-permuted
basemodel    = "both"         # "marsbase", "spambase", or "both"
```

For the figures reported in the paper, we **always use**
>
> ```
> basemodel = "both"
> ```
>
so that results from both base models are combined.

After setting these options, run the desired plotting script, e.g.:

```
python src/python_plotting/make_nonlinear_additive_plots.py
```

Figures will be written automatically into the appropriate sub-directory under:

```
results/
```
                                    

---

## Notes on Reproducibility

* All required Python code, including third-party components, is stored within the repository under `src/python/`.
* The `.venv/` directory is intentionally excluded from version control. Each user must create their own environment.
* Raw data under `data/` and generated results under `results/` are excluded from version control (except documentation placeholders).
* Plotting scripts require that experiment output files already exist in the appropriate directory structure.

