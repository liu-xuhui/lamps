import numpy as np
import sys
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add src/python to PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from adamp.fit_functions import linear_reg
from adamp.adamp_main import *
from adamp.simulation_functions import SimuLinear
from adamp.simulation_functions import SimuLinear_sparsity
from adamp.validation_methods import *
import pandas as pd

import matplotlib.pyplot as plt


N = 200; M = 500; N1 = 50;

"""
MP setting
"""
max_iter = 5
func_name = 'linear'
fit_funcs = {
        'linear':linear_reg
            }
fit_func=fit_funcs[func_name]

n_ratio= 0.4
m_ratio= 0.12
delta = 0.8
K_i = int(max(50/((1-n_ratio)*m_ratio*m_ratio), 200/((1-n_ratio)*m_ratio)))
K = [K_i for i in range(max_iter)]
snr = 5
sparsity_list = [10, 20, 40, 80, 160, 320, 480]
corr = 0
num_simus = 10
permute = 0



results_rows = []
selection_cols = {}
total_iters = len(sparsity_list) * num_simus
with tqdm(total=total_iters, desc="Total simulations") as pbar:
    for s in sparsity_list:
        for i in range(num_simus):
            X, Y, X1, Y1 = SimuLinear_sparsity(N, M, N1, k = s, snr = snr, seed = 110+i, corr = corr, permute = False)

            np.random.seed(110+i)

            lassocv_select, _ = compute_lassocv_select_features(X, Y)

            elastic_select, _ = Elastic(X, Y)

            scaler = StandardScaler()
            Y = scaler.fit_transform(Y.reshape(-1, 1))

            res = indept_weight_sample_epochtuned(X, Y, X1, Y1, n_ratio, m_ratio, K, fit_func, delta, max_iter, plot=False, seed = 110+i)

            last_key = next(reversed(res)) - 1
            if last_key < 0:
                last_key = 0
            adamp_select = np.where(res[last_key]["prob_F"] >= delta * 0.5)[0]

            # ---- save AdaMP selection results for this sparsity/seed ----
            seed = 110 + i
            prob = res[last_key]["prob_F"]

            # non-oracle selection: variable length, pad with NaN to length M
            adamp_select_padded = np.full(M, np.nan)
            adamp_select_padded[:len(adamp_select)] = adamp_select

            # oracle ranking: always length M
            adamp_select_oracle = np.argsort(prob)[::-1]
            adamp_select_prob = np.sort(prob)[::-1]

            selection_cols[f"s{s}_seed{seed}_nonoracle"] = adamp_select_padded
            selection_cols[f"s{s}_seed{seed}_oracle_rank"] = adamp_select_oracle
            selection_cols[f"s{s}_seed{seed}_oracle_prob"] = adamp_select_prob


            signal_features = list(range(s))

            adamp_f1 = compute_f1_score(
                adamp_select, total_features=M, signal_features=signal_features
            )["f1_score"]

            lassocv_f1 = compute_f1_score(
                lassocv_select, total_features=M, signal_features=signal_features
            )["f1_score"]

            elastic_f1 = compute_f1_score(
                elastic_select, total_features=M, signal_features=signal_features
            )["f1_score"]

            results_rows.extend([
                {"sparsity": s, "simu": i, "method": "AdaMP", "f1_score": adamp_f1},
                {"sparsity": s, "simu": i, "method": "LassoCV", "f1_score": lassocv_f1},
                {"sparsity": s, "simu": i, "method": "ElasticNet", "f1_score": elastic_f1},
            ])
            pbar.update(1)



# store per-simulation results
results_df = pd.DataFrame(results_rows)

# average F1 across the 10 simulations for each sparsity and method
mean_df = (
    results_df.groupby(["sparsity", "method"], as_index=False)["f1_score"]
    .mean()
    .rename(columns={"f1_score": "mean_f1"})
)

# optional: standard error, useful for plotting
se_df = (
    results_df.groupby(["sparsity", "method"], as_index=False)["f1_score"]
    .sem()
    .rename(columns={"f1_score": "se_f1"})
)

summary_df = mean_df.merge(se_df, on=["sparsity", "method"], how="left")

# save results
out_dir = ROOT / "temp_results" / "sparsity_result"
out_dir.mkdir(parents=True, exist_ok=True)

selection_df = pd.DataFrame(selection_cols)
selection_df.to_csv(out_dir / "adamp_selection_by_seed.csv", index=False)

results_df.to_csv(out_dir / "f1_all_runs.csv", index=False)
summary_df.to_csv(out_dir / "f1_mean_by_sparsity.csv", index=False)

with open(out_dir / "f1_mean_by_sparsity.pkl", "wb") as f:
    pickle.dump(summary_df, f)

# plot F1 vs sparsity
plt.figure(figsize=(8, 5))

for method in ["AdaMP", "LassoCV", "ElasticNet"]:
    sub = summary_df[summary_df["method"] == method].sort_values("sparsity")
    plt.plot(sub["sparsity"], sub["mean_f1"], marker="o", label=method)
    plt.fill_between(
        sub["sparsity"],
        sub["mean_f1"] - sub["se_f1"],
        sub["mean_f1"] + sub["se_f1"],
        alpha=0.15
    )

plt.xlabel("Sparsity")
plt.ylabel("F1")
plt.title(f"F1 vs Sparsity (N={N}, M={M}, SNR={snr})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "f1_vs_sparsity.png", dpi=300, bbox_inches="tight")
plt.close()