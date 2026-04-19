#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

python data/simulation/generate_nonlinear_additive_data.py

for permute in 0 1; do
  ADAMP_PERMUTE="$permute" Rscript experiments/nonlinear/additive/run_adamp.R
  ADAMP_PERMUTE="$permute" Rscript experiments/nonlinear/additive/run_mars.R
  for oracle in 0 1; do
    ADAMP_PERMUTE="$permute" ADAMP_ORACLE="$oracle" Rscript experiments/nonlinear/additive/run_spam.R
  done
  ADAMP_PERMUTE="$permute" python experiments/nonlinear/additive/run_hsic.py
  ADAMP_PERMUTE="$permute" python experiments/nonlinear/additive/run_knockoff.py
done

python experiments/nonlinear/additive/summarize_result.py

for permute in 0 1; do
  for oracle in 0 1; do
    ADAMP_PERMUTE="$permute" ADAMP_ORACLE="$oracle" python src/python_plotting/make_nonlinear_additive_plots.py
  done
done
