

import sys
from pathlib import Path
import pickle

# Add src/python to PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from adamp.fit_functions import linear_reg
from adamp.adamp_main import *
from adamp.simulation_functions import SimuLinear
from adamp.simulation_functions import SimuFriedmanAdditive


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
snr = 1
number_signals = 10
corr = 0

#X, Y, X1, Y1 = SimuLinear(N, M, N1, snr=1, seed=110, corr = 0)
X, Y, X1, Y1 = SimuFriedmanAdditive(N, M, N1, snr=1, seed=110, corr = 0)
# res = indept_weight_sample_epochtuned(X, Y, X1, Y1, n_ratio, m_ratio, K, fit_func, delta, max_iter, plot=False)

# with open(f"results/other/one_linear_corr{corr}_experiment_res.pkl", "wb") as f:
#     pickle.dump(res, f)
