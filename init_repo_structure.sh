#!/usr/bin/env bash
set -e

# Root-level files
touch README.md
touch LICENSE
touch .gitignore

# Environment configuration
mkdir -p env
touch env/requirements.txt
touch env/setup_venv.sh
touch env/R_packages.R

# Data directories
mkdir -p data/simulation
mkdir -p data/riboflavin

# Results directories
mkdir -p results/linear
mkdir -p results/nonlinear/additive
mkdir -p results/nonlinear/nonadditive
mkdir -p results/riboflavin
touch results/README.md

# Notebooks
mkdir -p notebooks
touch notebooks/adamp_python_main.ipynb

# Python AdaMP (library)
mkdir -p src/python/adamp
touch src/python/adamp/__init__.py
touch src/python/adamp/adamp_main.py
touch src/python/adamp/fit_functions.py
touch src/python/adamp/simulation_functions.py
touch src/python/adamp/validation_methods.py
touch src/python/adamp/sensitivity_analysis.py
touch src/python/adamp/plot_utils.py

# R AdaMP (library)
mkdir -p src/R/adamp
touch src/R/adamp/adamp_main.R
touch src/R/adamp/fit_functions.R
touch src/R/adamp/simulation_functions.R
touch src/R/adamp/validation_methods.R
touch src/R/adamp/sensitivity_analysis.R
touch src/R/adamp/mars_spam_wrappers.R

# Python plotting scripts
mkdir -p src/python_plotting
touch src/python_plotting/make_linear_plots.py
touch src/python_plotting/make_nonlinear_plots.py
touch src/python_plotting/make_riboflavin_plots.py

# Experiments: linear
mkdir -p experiments/linear
touch experiments/linear/run_linear_oracle_perm.py
touch experiments/linear/run_linear_oracle_noperm.py
touch experiments/linear/run_linear_nonoracle_perm.py
touch experiments/linear/run_linear_nonoracle_noperm.py

# Experiments: nonlinear
mkdir -p experiments/nonlinear/additive
mkdir -p experiments/nonlinear/nonadditive

# Helper to create the 8 nonlinear subfolders with 4 files each
create_nonlinear_condition () {
  base_dir="$1"   # e.g. experiments/nonlinear/additive or nonadditive
  cond="$2"       # e.g. oracle_perm_mars

  mkdir -p "$base_dir/$cond"
  touch "$base_dir/$cond/run_hsic_knockoff.py"
  touch "$base_dir/$cond/run_adamp.R"
  touch "$base_dir/$cond/run_spam.R"
  touch "$base_dir/$cond/run_mars.R"
}

NL_ADD_DIR="experiments/nonlinear/additive"
NL_NONADD_DIR="experiments/nonlinear/nonadditive"

conditions=(
  "oracle_perm_mars"
  "oracle_perm_spam"
  "oracle_noperm_mars"
  "oracle_noperm_spam"
  "nonoracle_perm_mars"
  "nonoracle_perm_spam"
  "nonoracle_noperm_mars"
  "nonoracle_noperm_spam"
)

# Additive conditions
for cond in "${conditions[@]}"; do
  create_nonlinear_condition "$NL_ADD_DIR" "$cond"
done

# Nonadditive conditions
for cond in "${conditions[@]}"; do
  create_nonlinear_condition "$NL_NONADD_DIR" "$cond"
done

# Experiments: riboflavin (Python only)
mkdir -p experiments/riboflavin
touch experiments/riboflavin/run_riboflavin_experiment.py

echo "AdaMP repo skeleton created."
