#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Create the output directories this pipeline writes into.
mkdir -p data/simulation results/other

python experiments/other/run_locosplit_linear.py
python src/python_plotting/make_locosplit_plots.py

python experiments/other/run_locomp_linear.py
python src/python_plotting/make_locomp_plots.py
