#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Create the output directories this pipeline writes into.
mkdir -p data/simulation temp_results/hyperparam_tuning results/hyperparam_tuning

ADAMP_PERMUTE=0 python experiments/linear/run_hyper_tuning.py
ADAMP_PERMUTE=0 python experiments/linear/summarize_tuning_results.py
ADAMP_PERMUTE=0 python src/python_plotting/make_linear_hypertuning_plots.py
