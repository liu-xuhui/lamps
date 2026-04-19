#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

python data/simulation/generate_nonlinear_nonadditive_data.py

for permute in 0 1; do
  ADAMP_PERMUTE="$permute" Rscript experiments/nonlinear/nonadditive/run_adamp.R
  ADAMP_PERMUTE="$permute" Rscript experiments/nonlinear/nonadditive/run_mars.R
  for oracle in 0 1; do
    ADAMP_PERMUTE="$permute" ADAMP_ORACLE="$oracle" Rscript experiments/nonlinear/nonadditive/run_spam.R
  done
  ADAMP_PERMUTE="$permute" python experiments/nonlinear/nonadditive/run_hsic.py
  ADAMP_PERMUTE="$permute" python experiments/nonlinear/nonadditive/run_knockoff.py
done

python experiments/nonlinear/nonadditive/summarize_result.py

for permute in 0 1; do
  ADAMP_PERMUTE="$permute" ADAMP_ORACLE=0 python src/python_plotting/make_interaction_rate_plots.py
done
