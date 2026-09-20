#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Create the output directories this pipeline writes into.
mkdir -p data/rosmap results/rosmap

Rscript experiments/rosmap/generate_train_test_data.R
Rscript experiments/rosmap/run_adamp.R
Rscript experiments/rosmap/run_spam.R
python experiments/rosmap/run_hsic.py
python experiments/rosmap/run_lasso.py
python experiments/rosmap/run_rf_comparison.py
python src/python_plotting/make_rosmap_plots.py
python experiments/rosmap/create_frequency_table.py
