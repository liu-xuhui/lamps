#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

Rscript experiments/rosmap/generate_train_test_data.R
Rscript experiments/rosmap/run_adamp.R
python experiments/rosmap/run_hsic.py
python experiments/rosmap/run_rf_comparison.py
python src/python_plotting/make_rosmap_plots.py
python experiments/rosmap/create_frequency_table.py
