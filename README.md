
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

## Plot Generation

Two Python plotting utilities are provided under `src/python_plotting/`:

- `make_linear_plots.py`
- `make_nonlinear_plots.py`

These scripts read processed experiment results saved under the `results/` directory and generate publication-ready figures.

### Linear Plots

Linear plots can be generated directly:

```

python src/python_plotting/make_linear_plots.py

```

Output figures will be saved under:

```

results/linear/

```

### Nonlinear Plots

Nonlinear plots require the user to specify four parameters inside `make_nonlinear_plots.py`:

```

eval_metd   = "f1_score"      
# one of: "f1_score", "precision", "recall"
oracle      = False           
# currently MUST be False (oracle=True data not yet provided)
permute     = False           # True or False
basemodel   = "marsbase"      # "marsbase" or "spambase"

```

After setting these parameters, run:

```

python src/python_plotting/make_nonlinear_plots.py

```

Plots will be written to:

```

results/nonlinear/additive/

```

Non-additive plots will be supported in a future update.

---

## Notes on Reproducibility

* All required Python code, including third-party components, is stored within the repository under `src/python/`.
* The `.venv/` directory is intentionally excluded from version control. Each user must create their own environment.
* Raw data under `data/` and generated results under `results/` are excluded from version control (except documentation placeholders).
* Plotting scripts require that experiment output files already exist in the appropriate directory structure.

