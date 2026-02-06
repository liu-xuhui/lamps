import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---------------------------------------------------------------------
# Path setup: add src/python to PYTHONPATH
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from adamp.fit_functions import linear_reg
from adamp.adamp_main import indept_weight_sample_epochtuned
from adamp.simulation_functions import SimuLinear

# ---------------------------------------------------------------------
# Fixed experiment settings (LINEAR)
# ---------------------------------------------------------------------
N, M, N1 = 200, 500, 50

max_iter = 5
delta = 0.8

# Fixed as requested
permute = 0
corr = 0.5

# Only linear base model
fit_func = linear_reg
func_name = "linear"

# Only non-oracle selection (oracle=0)
oracle = 0

# True signal set for F1: features 1..10 in R => indices 0..9 in Python
signal_set = set(range(10))

# Experiment loops (from your original script)
snr_list = [5, 10, 20, 30]
num_simus = 10
rep_list = list(range(num_simus))

# Hyperparameter tuning grid (9 combos)
n_ratio_list = [0.3, 0.4, 0.5]
m_ratio_list = [0.1, 0.15, 0.2]
grid = [(nr, mr) for nr in n_ratio_list for mr in m_ratio_list]

# ---------------------------------------------------------------------
# Output dirs (as required)
# ---------------------------------------------------------------------
base_out_dir = ROOT / "temp_results" / "hyperparam_tuning" / "linear_corr0.5_permute0"
best_out_dir = base_out_dir / "best"
grid_out_dir = base_out_dir / "grid"

base_out_dir.mkdir(parents=True, exist_ok=True)
best_out_dir.mkdir(parents=True, exist_ok=True)
grid_out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def calc_K_list(n_ratio: float, m_ratio: float, max_iter: int) -> list[int]:
    denom1 = (1 - n_ratio) * (m_ratio ** 2)
    denom2 = (1 - n_ratio) * m_ratio
    if denom1 <= 0 or denom2 <= 0:
        raise ValueError(f"Invalid (n_ratio, m_ratio)=({n_ratio}, {m_ratio})")
    K_i = int(np.ceil(max(50 / denom1, 200 / denom2)))
    return [K_i for _ in range(max_iter)]

def last_epoch_key(res: dict) -> int:
    # res keys are ints in your code (you do next(reversed(res)))
    # Choose the last inserted key (Python 3.7+ preserves insertion order)
    return next(reversed(res))

def extract_final_loo(res: dict) -> float:
    # Always use last available loo
    k = last_epoch_key(res)
    loo = res[k].get("loo", None)
    if loo is None:
        raise RuntimeError("Could not extract final LOO from res[epoch]['loo']")
    return float(loo)

def extract_selected_features_nonoracle(res: dict, delta: float) -> np.ndarray:
    # Match your existing selection rule:
    # last_key = last_epoch - 1; if <0 then 0
    last_k = last_epoch_key(res) - 1
    if last_k < 0:
        last_k = 0
    prob_F = res[last_k].get("prob_F", None)
    if prob_F is None:
        raise RuntimeError("Selected epoch has no prob_F stored.")
    return np.where(np.asarray(prob_F) >= delta * 0.5)[0]

def f1_from_selected(selected_idx: np.ndarray, signal_set: set[int]) -> float:
    selected_set = set(map(int, np.unique(selected_idx)))
    tp = len(selected_set & signal_set)
    fp = len(selected_set - signal_set)
    fn = len(signal_set - selected_set)
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom

def fmt_ratio(x: float) -> str:
    # filenames like n0.3_m0.15 (clean + stable)
    s = f"{x}".rstrip("0").rstrip(".")
    return s

# ---------------------------------------------------------------------
# Main: for each (snr, rep), run 9 combos; pick best by final LOO
# Also: for each snr, write grid summary (avg loo + avg f1 across reps)
# ---------------------------------------------------------------------
total_runs = len(snr_list) * len(rep_list) * len(grid)
pbar = tqdm(total=total_runs, desc="Total grid runs")

for snr in snr_list:
    # store metrics: one row per grid point per rep
    loo_mat = np.full((len(grid), len(rep_list)), np.nan, dtype=float)
    f1_mat  = np.full((len(grid), len(rep_list)), np.nan, dtype=float)

    for rr, rep in enumerate(rep_list):
        # load data once per (snr, rep)
        data_path = ROOT / "data" / "simulation" / f"linear_corr{corr}_snr{snr}_permute{permute}_rep{rep}.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path)
        X = df.iloc[:, :M].to_numpy()
        Y = df["Y"].to_numpy()

        # placeholder simulation (consistent with your script)
        _, _, X1, Y1 = SimuLinear(N, M, N1, snr=snr, seed=113, corr=corr, permute=False)

        # standardize Y
        Y = StandardScaler().fit_transform(Y.reshape(-1, 1))

        # track best for this rep
        best_loo = np.inf
        best_selected = None
        best_n = None
        best_m = None

        for gg, (n_ratio, m_ratio) in enumerate(grid):
            K = calc_K_list(n_ratio, m_ratio, max_iter)

            # keep your seed pattern
            seed = 110 + rep

            res = indept_weight_sample_epochtuned(
                X, Y, X1, Y1,
                n_ratio, m_ratio, K,
                fit_func, delta, max_iter,
                plot=False, seed=seed
            )

            loo_final = extract_final_loo(res)
            selected = extract_selected_features_nonoracle(res, delta)
            f1 = f1_from_selected(selected, signal_set)

            loo_mat[gg, rr] = loo_final
            f1_mat[gg, rr]  = f1

            # save per-seed, per-grid selected features
            out_path_grid = base_out_dir / (
                f"adamlinear_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}"
                f"_n{fmt_ratio(n_ratio)}_m{fmt_ratio(m_ratio)}.csv"
            )
            pd.DataFrame({"var": selected}).to_csv(out_path_grid, index=False)

            # best-by-LOO
            if np.isfinite(loo_final) and loo_final < best_loo:
                best_loo = loo_final
                best_selected = selected
                best_n, best_m = n_ratio, m_ratio

            pbar.update(1)

        # save best per-seed selection
        best_out_path = best_out_dir / (
            f"adamlinear_selected_corr{corr}_snr{snr}_permute{permute}_oracle{oracle}_rep{rep}.csv"
        )
        pd.DataFrame({"var": best_selected}).to_csv(best_out_path, index=False)

        print(
            f"\nSNR={snr} rep={rep} best (n_ratio={best_n:.2f}, m_ratio={best_m:.2f}): "
            f"LOO={best_loo:.6f} | selected={len(best_selected)} | saved={best_out_path}"
        )

    # per-SNR grid summary (avg over reps)
    grid_rows = []
    for gg, (n_ratio, m_ratio) in enumerate(grid):
        avg_loo = float(np.nanmean(loo_mat[gg, :]))
        avg_f1  = float(np.nanmean(f1_mat[gg, :]))
        grid_rows.append(
            {"n_ratio": n_ratio, "m_ratio": m_ratio, "avg_loo_final": avg_loo, "avg_f1": avg_f1}
        )

    grid_df = pd.DataFrame(grid_rows)
    grid_csv_path = grid_out_dir / f"grid_search_snr{snr}_avg_loo_and_f1.csv"
    grid_df.to_csv(grid_csv_path, index=False)
    print(f"\nSaved per-SNR grid summary: {grid_csv_path}")

pbar.close()
print("\nDONE.\n")
