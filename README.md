# Environment Setup Instructions

This document describes how to create a reproducible Python environment for running the AdaMP simulations and experiments on Windows (Git Bash) and macOS/Linux. All required Python dependencies are listed in `env/requirements.txt`, and the environment can be created automatically using the provided script `env/setup_venv.sh`.

A copy of the required third-party package `stability_selection` is included inside the repository under
`src/python/adamp/third_party/stability_selection/`.
It is imported locally and does not require installation through pip.

No specific Python version is strictly required, but compatibility has been tested on Python versions 3.10 through 3.12.

---

## 1. Windows (Git Bash) Setup

**Important:** All commands must be executed in **Git Bash**, not PowerShell or CMD.

### Step 1. Clone the repository

```bash
git clone <repository_url>
cd <repository_directory>
```

### Step 2. Create a virtual environment



```bash
python -m venv .venv
```


### Step 3. Run the setup script

```bash
./env/setup_venv.sh
```

This script initializes the virtual environment, upgrades pip, and installs all required packages listed in `requirements.txt`.

### Step 4. Activate the environment in future sessions

```bash
source .venv/Scripts/activate
```

---

## 2. macOS / Linux Setup

### Step 1. Clone the repository

```bash
git clone <repository_url>
cd <repository_directory>
```

### Step 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3. Install dependencies

You may use the automated script:

```bash
bash env/setup_venv.sh
```

or manually install:

```bash
pip install --upgrade pip
pip install -r env/requirements.txt
```

The included third-party code is imported without additional installation steps.

---

## 3. Running Experiments

After activating the virtual environment, individual experiment scripts can be executed directly. For example:

```bash
python experiments/linear/run_linear_oracle_perm.py
python experiments/nonlinear/additive/oracle_perm_mars/run_hsic_knockoff.py
python experiments/riboflavin/run_riboflavin_experiment.py
```

All generated results will be written to the `results/` directory following the repository’s directory structure.

---

## 4. Notes on Reproducibility

* The repository includes all required code, including external components placed under `src/python/adamp/third_party/`.
* The `.venv/` directory is intentionally excluded from version control. Users should create the environment following the instructions above.
* Data directories (`data/`) and simulation outputs (`results/`) are also excluded from version control, except for documentation files.

---


