import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Add src/python to PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from adamp.fit_functions import lasso_reg
from adamp.adamp_main import *
from adamp.validation_methods import CPSS
from adamp.validation_methods import compute_lassocv_select_features
from adamp.validation_methods import Elastic
from adamp.validation_methods import LassoStabilitySelection
from adamp.validation_methods import compute_eBIC_select_features


# This script has no shell wrapper; create its output directory here.
Path("results/riboflavin").mkdir(parents=True, exist_ok=True)

ribo_df = pd.read_csv("data/riboflavin/riboflavin_X.csv", index_col=0)
X = pd.read_csv("data/riboflavin/riboflavin_X.csv", index_col=0).to_numpy()
y = pd.read_csv("data/riboflavin/riboflavin_y.csv", index_col=0).iloc[:,0].values


X1 = X.copy()[:30, :]
Y1 = y.copy()[:30]


N = X.shape[0]; M = X.shape[1]; N1 = 30;

max_iter = 5

func_name = 'las'
fit_funcs = {
        'las':lasso_reg
            }
fit_func=fit_funcs[func_name]

n_ratio= 0.4
m_ratio= 0.12
delta = 0.8
K = [5787 for i in range(max_iter)]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaler = StandardScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

res = indept_weight_sample_epochtuned(X_scaled, y_scaled, X1, Y1, n_ratio, m_ratio, K, fit_func, delta, max_iter, plot=False, seed = 118)
last_key = next(reversed(res)) - 1
if last_key < 0:
    last_key = 0
linear_select = np.where(res[last_key]["prob_F"] >= delta * 0.5)[0]

np.random.seed(118)

df_adamp = pd.DataFrame()
df_adamp["var"] = linear_select
df_adamp["var_name"] = ribo_df.iloc[:, linear_select].columns
df_adamp.to_csv("results/riboflavin/adamp_linear.csv")

stab_selected, stab_score = LassoStabilitySelection(X, y)
df_stability = pd.DataFrame()
df_stability["var"] = stab_selected
df_stability["var_name"] = ribo_df.iloc[:, stab_selected].columns
df_stability.to_csv("results/riboflavin/stability.csv")

lassocv_selected, _ = compute_lassocv_select_features(X, y)
df_lassocv = pd.DataFrame()
df_lassocv["var"] = lassocv_selected
df_lassocv["var_name"] = ribo_df.iloc[:, lassocv_selected].columns
df_lassocv.to_csv("results/riboflavin/lassocv.csv")

lassoebic_selected, _ = compute_eBIC_select_features(X, y)
df_lassoebic = pd.DataFrame()
df_lassoebic["var"] = lassoebic_selected
df_lassoebic["var_name"] = ribo_df.iloc[:, lassoebic_selected].columns
df_lassoebic.to_csv("results/riboflavin/lassoebic.csv")

cpss_select, _ = CPSS(X, y)
df_cpss = pd.DataFrame()
df_cpss["var"] = cpss_select
df_cpss["var_name"] = ribo_df.iloc[:, cpss_select].columns
df_cpss.to_csv("results/riboflavin/cpss.csv")

elastic_select, _ = Elastic(X, y)
df_elastic = pd.DataFrame()
df_elastic["var"] = elastic_select
df_elastic["var_name"] = ribo_df.iloc[:, elastic_select].columns
df_elastic.to_csv("results/riboflavin/elastic.csv")


