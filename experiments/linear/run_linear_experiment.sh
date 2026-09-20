#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Create the output directories this pipeline writes into.
mkdir -p data/simulation temp_results/linear results/linear

python data/simulation/generate_linear_data.py

for permute in 0 1; do
  ADAMP_PERMUTE="$permute" python experiments/linear/run_adamp.py
  ADAMP_PERMUTE="$permute" python experiments/linear/run_validation.py
done

python experiments/linear/summarize_results.py

for permute in 0 1; do
  ADAMP_PERMUTE="$permute" python src/python_plotting/make_linear_plots.py
done
