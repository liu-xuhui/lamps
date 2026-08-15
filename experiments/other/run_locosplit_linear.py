import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add src/python to PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from adamp.fit_functions import lasso_reg_cv
from adamp.LOCO_Functions import *
from adamp.simulation_functions import SimuLinear


"""
MP setting
"""
N = 200; M = 500; N1 = 50;

max_iter = 5
func_name = 'cv'
fit_funcs = {
        'cv':lasso_reg_cv
            }
fit_func=fit_funcs[func_name]

n_ratio= 0.4
m_ratio= 0.12
delta = 0.8
K = [5787 for i in range(max_iter)]
number_signals = 10
n = int(N*n_ratio)
m = int(M*m_ratio)

snr = 50
corr = 0.9
rep = 0
permute = 0

df = pd.read_csv(f"data/simulation/linear_corr{corr}_snr{snr}_permute{permute}_rep{rep}.csv")

X = df.iloc[:,:500].to_numpy()
Y = df["Y"].values

_, _, X1, Y1 = SimuLinear(N, M, N1, snr = snr, seed = 111, corr = corr, permute = False)

scaler = StandardScaler()
Y = scaler.fit_transform(Y.reshape(-1, 1))

np.random.seed(110+rep)

res = LOCOSplitReg(X,Y,X1, Y1, fit_funct = fit_func, ratio = n_ratio, selected_features=[],n_features = M,alpha=0.1,bonf=False)


with open(f"results/other/one_linear_corr{corr}_locosplitreg_res.pkl", "wb") as f:
    pickle.dump(res, f)