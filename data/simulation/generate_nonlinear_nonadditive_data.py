import pandas as pd
import sys
from pathlib import Path

# go from data/simulation/generate_nonlinear_data.py to repo root
project_root = Path(__file__).resolve().parents[2]
src_python = project_root / "src" / "python"
sys.path.insert(0, str(src_python))

from adamp.simulation_functions import SimuFriedmanNonAdditive


N = 200; M = 50; N1 = 50;

max_iter = 5

number_signals = 10
num_simus = 30
interaction_snr_list = [4, 6, 8, 10]
corr_list = [0, 0.5, 0.9]
corr_list = [0, 0.5, 0.9]
permute_list = [1, 0]

for ins in interaction_snr_list:
    for rep in range(num_simus):
        for corr in corr_list:
            for permute in permute_list:
                seed = 110 + rep
                X, Y, X1, Y1 = SimuFriedmanNonAdditive(N, M, N1, snr=1, seed=seed, corr = corr, interaction_snr = ins, permute = permute)
                df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])
                df["Y"] = Y
                df.to_csv(f"data/simulation/nonlinear_nonadditive_corr{corr}_snr{ins}_permute{permute}_rep{rep}.csv", index=False)