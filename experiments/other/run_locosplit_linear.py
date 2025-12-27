import sys
from pathlib import Path
import pickle

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

corr = 0.9
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
snr = 1
number_signals = 10
n = int(N*n_ratio)
m = int(M*m_ratio)

X, Y, X1, Y1 = SimuLinear(N, M, N1, k=10, rho=corr, snr=1, seed=110)

res = LOCOSplitReg(X,Y,X1, Y1, fit_funct = fit_func, ratio = n_ratio, selected_features=[],n_features = M,alpha=0.1,bonf=False)


with open(f"results/other/one_linear_corr{corr}_locosplitreg_res.pkl", "wb") as f:
    pickle.dump(res, f)