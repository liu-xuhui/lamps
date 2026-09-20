# LAMPS

This repository contains the LAMPS feature-selection implementation and the
experiment scripts used to reproduce the paper figures.

The installable Python package lives in `lamps_package/lamps` and is configured
by `pyproject.toml`. The paper-reproduction code remains under `src/`,
`experiments/`, `data/`, and `src/python_plotting/`.

## Installing LAMPS

The PyPI distribution name is **lamps-fs**; the Python import name is **lamps**.
Users can install the package without cloning this repository:

```bash
python -m pip install lamps-fs
```

Requires Python 3.10 or newer and NumPy 1.25 or newer. See
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

## Data Availability

The `data/` directory is not tracked by Git. Simulation data is generated
locally by the scripts in this repository, while the two case-study data sets
must be obtained separately as described below. All paths are relative to the
repository root.

### Simulation data

No download is needed. The generators under `data/simulation/` write the
required CSV files into `data/simulation/`:

```bash
python data/simulation/generate_linear_data.py
python data/simulation/generate_nonlinear_additive_data.py
python data/simulation/generate_nonlinear_nonadditive_data.py
```


### Riboflavin data

The riboflavin data of Dezeure et al. (2015) (`N = 71` observations,
`M = 4,088` covariates) is distributed with the
[`hdi`](https://cran.r-project.org/package=hdi) R package. `hdi` is installed by
`Rscript env/R_packages.R`.

`experiments/riboflavin/run_riboflavin_experiment.py` expects two CSV files,
each with the observation identifiers in the first column:

```text
data/riboflavin/riboflavin_X.csv    71 rows, 1 index column + 4,088 covariates
data/riboflavin/riboflavin_y.csv    71 rows, 1 index column + 1 response column
```

Export them from R, running from the repository root:

```r
library(hdi)
data(riboflavin)

dir.create(file.path("data", "riboflavin"), recursive = TRUE, showWarnings = FALSE)
write.csv(as.matrix(riboflavin$x), file.path("data", "riboflavin", "riboflavin_X.csv"))
write.csv(data.frame(y = riboflavin$y), file.path("data", "riboflavin", "riboflavin_y.csv"))
```

The default `row.names = TRUE` of `write.csv` produces the leading index column
that the Python script reads with `index_col=0`.

### ROSMAP data

The ROSMAP data used in the case study are **not** redistributed with this
repository. They are available from the Rush Alzheimer's Disease Center (RADC,
<https://www.radc.rush.edu/>). Access is subject to RADC's data access and Data
Use Agreement requirements. Obtain the data directly from RADC before running
the ROSMAP experiment.

The analysis uses a preprocessed matrix of `N = 507` observations and `M = 200`
gene-expression covariates, retained by high-variance screening, with the global
cognition score as the response. Place it at:

```text
data/rosmap/rosmap_200.csv
```

`experiments/rosmap/generate_train_test_data.R` reads this file by column
position, so the column order matters:

| Column | Contents |
|---|---|
| 1 | observation index (dropped) |
| 2 … 201 | the 200 gene-expression covariates |
| 202 | unused column (dropped) |
| 203 | response: global cognition score |

That is 203 columns in total. Column headers are preserved as read
(`check.names = FALSE`); the covariate headers in columns 2–201 are used as the
gene names in the selection-frequency table produced by
`experiments/rosmap/create_frequency_table.py`, so keep the original gene
identifiers there.


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

Requires `data/rosmap/rosmap_200.csv`, obtained from RADC under their Data Use
Agreement; see [ROSMAP data](#rosmap-data) for the access route and the expected
file layout.

ROSMAP preprocessing, methods, plots, and frequency table:

```bash
./experiments/rosmap/run_rosmap_experiment.sh
```

### Riboflavin

Requires `data/riboflavin/riboflavin_X.csv` and `riboflavin_y.csv`, exported
from the `hdi` R package; see [Riboflavin data](#riboflavin-data) for the export
snippet.

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
