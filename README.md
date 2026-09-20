# LAMPS

This repository contains the LAMPS feature-selection implementation and the
experiment scripts used to reproduce the paper figures.

The installable Python package lives in `lamps_package/lamps` and is configured
by `pyproject.toml`. The paper-reproduction code remains under `src/`,
`experiments/`, `data/`, and `src/python_plotting/`.

## Installing LAMPS

The PyPI distribution name is **lamps-fs**; the Python import name is **lamps**.
After the first PyPI release, users can install without cloning this repository:

```bash
python -m pip install lamps-fs
```

Requires Python 3.10 or newer and NumPy 1.25 or newer. Until that release,
use the local installation instructions below. See
[the package guide](lamps_package/README.md) for a runnable example.

## Environment Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Install the Python package itself:

```bash
pip install .
```

This installs only the lightweight public LAMPS package and its required
dependency, `numpy`.

For full paper reproduction, install the experiment dependencies:

```bash
pip install -r env/requirements.txt
```

Some experiments also require R packages. Install them with:

```bash
Rscript env/R_packages.R
```

The vendored `stability_selection` code remains under
`src/python/adamp/third_party/stability_selection/` and does not need a separate
pip install. It is a copy of
[scikit-learn-contrib/stability-selection](https://github.com/scikit-learn-contrib/stability-selection),
which is not published on PyPI; see
[its README](src/python/adamp/third_party/stability_selection/README.md) for the
retrieval date and the two scikit-learn compatibility patches applied to it.

## Public Python Package

After `pip install .`, users can call LAMPS directly:

```python
from lamps import lamps_select

selected = lamps_select(X, y)
```

Hyperparameter tuning is available by passing a Python list for any of
`n_ratio`, `m_ratio`, `fit_func`, or `delta`. LAMPS will grid-search all
combinations, choose the run with the lowest final-epoch leave-one-out error,
show tuning progress, and print the best hyperparameters.

```python
selected = lamps_select(
    X,
    y,
    n_ratio=[0.3, 0.4, 0.5],
    m_ratio=[0.1, 0.15, 0.2],
    delta=[0.7, 0.8],
)
```

The same list interface works for model functions. Each function should use
the signature `fit_func(X_train, y_train, X_predict)`:

```python
selected = lamps_select(
    X,
    y,
    fit_func=[linear_fit, ridge_fit],
)
```

To inspect the result dictionary from the best hyperparameter setting, set
`return_complete_info=True`:

```python
selected, res = lamps_select(
    X,
    y,
    n_ratio=[0.4, 0.5],
    m_ratio=[0.1, 0.2],
    return_complete_info=True,
)
```

The public package is intentionally minimal. It is separate from the heavier
paper-reproduction scripts.

## Reproducing Paper Results

Before running the simulation experiments, run the Python scripts under
`data/simulation/` to generate the required simulation data.

Run commands from the repository root after activating the environment. The
shell scripts below generate data, run methods, summarize temporary outputs, and
create the corresponding publication figures.

Temporary method outputs are written to `temp_results/`. Summarized result files
and figures are written under `results/`.

### Linear Simulation

Main linear experiment and figure:

```bash
./experiments/linear/run_linear_experiment.sh
```

Linear hyperparameter-tuning experiment and figure:

```bash
./experiments/linear/run_linear_hyperparameter_tuning.sh
```

### Nonlinear Additive Simulation

Main nonlinear additive experiment and figure:

```bash
./experiments/nonlinear/additive/run_nonlinear_additive.sh
```

Nonlinear additive hyperparameter-tuning experiment and figure:

```bash
./experiments/nonlinear/additive/run_nonlinear_additive_hyperparameter_tuning.sh
```

### Nonlinear Nonadditive Simulation

Main nonlinear nonadditive interaction-rate experiment and figure:

```bash
./experiments/nonlinear/nonadditive/run_nonlinear_nonadditive.sh
```

Nonlinear nonadditive hyperparameter-tuning interaction-rate experiment and
figure:

```bash
./experiments/nonlinear/nonadditive/run_nonlinear_nonadditive_hyperparameter_tuning.sh
```

The nonadditive figure scripts use:

```text
src/python_plotting/make_interaction_rate_plots.py
src/python_plotting/make_interaction_rate_hyperparam_tuning_plots.py
```

`src/python_plotting/make_nonlinear_nonadditive_plots.py` is not part of the
current reproduction workflow.

### Theory-Validation Figures

LOCO split and LOCOMP:

```bash
./experiments/other/run_locosplit_locomp.sh
```

Correlated theory validation probability/delta epoch plots:

```bash
./experiments/correlated_theory/run_correlated_experiment_and_plot.py
```

Single linear run and probability/delta epoch plots:

```bash
./experiments/other/run_one_linear.sh
```

To generate only one plot kind:

```bash
./experiments/other/run_one_linear.sh prob
./experiments/other/run_one_linear.sh delta
```

### ROSMAP

ROSMAP preprocessing, methods, plots, and frequency table:

```bash
./experiments/rosmap/run_rosmap_experiment.sh
```

### Riboflavin

No shell wrapper is needed:

```bash
python experiments/riboflavin/run_riboflavin_experiment.py
```

## Script Parameters

The reproducibility shell scripts set script parameters through environment
variables:

```text
ADAMP_PERMUTE   0 or 1
ADAMP_ORACLE    0 or 1
ADAMP_PLOT_KIND prob or delta
```

Directly running individual scripts without these variables preserves their
default values.

## Notes

- The LAMPS package is licensed under the [MIT license](LICENSE).
- The vendored `stability_selection` code under
  `src/python/adamp/third_party/stability_selection/` is licensed under
  BSD-3-Clause, Copyright (c) 2018 Thomas Huijskens. Its license text is
  retained alongside the code in
  [that directory](src/python/adamp/third_party/stability_selection/LICENSE),
  and it is not included in the `lamps-fs` distribution.

- The shell scripts are intentionally long-running; many run full simulation
  grids across multiple methods.
- Python experiment dependencies are managed by `env/requirements.txt`.
- R experiment dependencies are managed separately by `env/R_packages.R`.
- The public pip package and the paper-reproduction code are intentionally kept
  separate.
