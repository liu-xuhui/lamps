
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
from adamp.simulation_functions import SimuLinear_old
import pandas as pd


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
K = [5787 for i in range(max_iter)]
snr_list = [2,4,6,8,10]
number_signals = 10
corr_list = [0, 0.5, 0.9]
num_simus = 10
permute = 1

total_iters = len(corr_list) * len(snr_list) * num_simus
with tqdm(total=total_iters, desc="Total simulations") as pbar:
    for corr in corr_list:
        for snr in snr_list:
            for i in range(num_simus):
                df = pd.read_csv(f"data/simulation/linear_corr{corr}_snr{snr}_permute{permute}_rep{i}.csv")

                X = df.iloc[:,:M].to_numpy()
                Y = df["Y"].values

                _, _, X1, Y1 = SimuLinear(N, M, N1, snr = snr, seed = 113, corr = corr, permute = False)

                scaler = StandardScaler()
                Y = scaler.fit_transform(Y.reshape(-1, 1))

                res = indept_weight_sample_epochtuned(X, Y, X1, Y1, n_ratio, m_ratio, K, fit_func, delta, max_iter, plot=False)

                last_key = next(reversed(res)) - 1
                if last_key < 0:
                    last_key = 0
                elif last_key == max_iter - 2:
                    last_key += 1
                linear_select = np.where(res[last_key]["prob_F"] >= delta * 0.5)[0]

                temp_df = pd.DataFrame()
                temp_df["var"] = linear_select

                temp_df.to_csv(f"temp_results/linear/adamlinear_selected_corr{corr}_snr{snr}_permute{permute}_oracle{0}_rep{i}.csv")
                pbar.update(1)

