#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python experiments/other/run_one_linear.py

if [ "$#" -eq 0 ]; then
  plot_kinds=(prob delta)
else
  plot_kinds=("$@")
fi

for plot_kind in "${plot_kinds[@]}"; do
  ADAMP_PLOT_KIND="$plot_kind" python src/python_plotting/make_prob_vs_epoch_plots.py
done
