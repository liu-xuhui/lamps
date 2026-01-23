
import numpy as np
import sys
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler

# Add src/python to PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from adamp.fit_functions import linear_reg
from adamp.adamp_main import *
from adamp.simulation_functions import SimuLinear
from adamp.simulation_functions import SimuLinear_old
import pandas as pd
from adamp.validation_methods import *


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
snr = 8
number_signals = 10
corr = 0.9

# df = pd.read_csv(f"data/simulation/linear_corr{corr}_snr{snr}_permute{0}_rep{3}.csv")

# X = df.iloc[:,:500].to_numpy()
# Y = df["Y"].values

# _, _, X1, Y1 = SimuLinear(N, M, N1, snr = snr, seed = 113, corr = corr, permute = False)
#X, Y, X1, Y1 = SimuLinear_old(N, M, N1, k=10, rho=0.9, snr=snr, seed=111)
X, Y, X1, Y1 = SimuLinear(N, M, N1, snr = 30, seed = 111, corr = corr, permute = True)
# X, Y, X1, Y1 = SimuLinear_old(N, M, N1, k=10, rho=corr, snr=2, seed=111)
scaler = StandardScaler()
Y = scaler.fit_transform(Y.reshape(-1, 1))

res = indept_weight_sample_epochtuned(X, Y, X1, Y1, n_ratio, m_ratio, K, fit_func, delta, max_iter, plot=False)

last_key = next(reversed(res)) - 1
if last_key < 0:
    last_key = 0
# elif last_key == max_iter - 2:
#     last_key += 1
nonlinear_select1 = np.where(res[last_key]["prob_F"] >= delta * 0.5)[0]
print(nonlinear_select1)

# selected, _ = Elastic_oracle(X, Y)
# print(selected)

# with open(f"results/other/one_linear_corr{corr}_experiment_res.pkl", "wb") as f:
#     pickle.dump(res, f)
