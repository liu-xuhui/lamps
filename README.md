# AdaMP

This repository contains the AdaMP feature-selection implementation and the
experiment scripts used to reproduce the paper figures.

The installable Python package lives in `adamp_package/adamp` and is configured
by `pyproject.toml`. The paper-reproduction code remains under `src/`,
`experiments/`, `data/`, and `src/python_plotting/`.

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

This installs only the lightweight public AdaMP package and its required
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
pip install.

## Public Python Package

After `pip install .`, users can call AdaMP directly:

```python
from adamp import adamp_select

selected = adamp_select(X, y)
```

The public package is intentionally minimal. It is separate from the heavier
paper-reproduction scripts.

## Reproducing Paper Results

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

- The shell scripts are intentionally long-running; many run full simulation
  grids across multiple methods.
- Python experiment dependencies are managed by `env/requirements.txt`.
- R experiment dependencies are managed separately by `env/R_packages.R`.
- The public pip package and the paper-reproduction code are intentionally kept
  separate.
