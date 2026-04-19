#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

ADAMP_PERMUTE=0 Rscript experiments/nonlinear/additive/run_hyper_tuning.R
ADAMP_PERMUTE=0 python experiments/nonlinear/additive/summarize_tuning_results.py
ADAMP_PERMUTE=0 ADAMP_ORACLE=0 python src/python_plotting/make_nonlinear_additive_hypertuning_plots.py
