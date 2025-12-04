import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from math import comb

from pyHSICLasso import HSICLasso
from knockpy import knockoff_filter

import sys
from pathlib import Path

# go to repo root
project_root = Path(__file__).resolve().parent.parent.parent.parent
third_party_path = project_root / "src" / "python" / "adamp" / "third_party"
sys.path.insert(0, str(third_party_path))

from stability_selection.stability_selection import StabilitySelection




def LassoStabilitySelection(X, y, verbose = 1, lambda_grid = None, threshold=0.6, plot = False):
  base_estimator = Lasso(max_iter=1000)

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  lambda_max = (np.max(np.abs(X_scaled.T @ y)) / X.shape[0])

  if lambda_grid is None:
    lambda_grid = np.logspace(np.log10(0.01 * lambda_max), np.log10(lambda_max), 25)

  selector = StabilitySelection(base_estimator=base_estimator, lambda_name='alpha',
                                lambda_grid = lambda_grid, verbose = verbose).fit(X_scaled, y)

  selected_scores = selector.stability_scores_.max(axis=1)

  if plot:
    plt.bar(range(len(selected_scores)), selected_scores)

  return selector.get_support(indices=True), selected_scores


def compute_eBIC_select_features(X, y, lambda_grid=None, gamma=0.5):
    n, p = X.shape
    eBIC_scores = []
    selected_sets = []

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = y.ravel()

    lambda_max = (np.max(np.abs(X_scaled.T @ y)) / X.shape[0])
    if lambda_grid is None:
        lambda_grid = np.logspace(np.log10(0.01 * lambda_max), np.log10(lambda_max), 25)
    

    for alpha in lambda_grid:
        model = Lasso(alpha=alpha, max_iter=1000)
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        RSS = np.sum((y - y_pred) ** 2)
        selected = np.where(model.coef_ != 0)[0]
        S = len(selected)

        if S == 0:
            eBIC = np.inf
        else:
            bic = n * np.log(RSS / n) + S * np.log(n)
            penalty = 2 * gamma * np.log(float(comb(p, S))) if S > 0 and S < p else 0
            eBIC = bic + penalty

        eBIC_scores.append(eBIC)
        selected_sets.append(selected)

    best_idx = np.argmin(eBIC_scores)
    best_lambda = lambda_grid[best_idx]
    selected_features = selected_sets[best_idx]

    coef_abs = np.abs(model.coef_)

    return selected_features.tolist(), coef_abs


def compute_lassocv_select_features(X, y, lambda_grid=None, fold=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lambda_max = (np.max(np.abs(X_scaled.T @ y)) / X.shape[0])
    if lambda_grid is None:
        lambda_grid = np.logspace(np.log10(0.01 * lambda_max), np.log10(lambda_max), 25)

    # LASSO with 5-fold CV
    lasso_cv = LassoCV(cv=fold, alphas=lambda_grid, max_iter=1000)
    lasso_cv.fit(X_scaled, y)

    # Results
    best_alpha = lasso_cv.alpha_
    selected_features = np.where(lasso_cv.coef_ != 0)[0]

    coef_abs = np.abs(lasso_cv.coef_)

    return selected_features.tolist(), coef_abs


def CPSS(X, y, alpha=0.01, B=50, tau=0.6, random_state=None):
    n, p = X.shape
    selection_counts = np.zeros(p)
    rng = np.random.RandomState(random_state)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Generate B complementary pairs of size floor(n/2)
    indices = np.arange(n)
    for b in range(B):
        perm = rng.permutation(indices)
        half = n // 2
        A1 = perm[:half]
        A2 = perm[half:half*2]  # ensure disjoint

        for A in [A1, A2]:
            model = Lasso(alpha=alpha, max_iter=10000)
            model.fit(X_scaled[A], y[A])
            selected = np.where(model.coef_ != 0)[0]
            selection_counts[selected] += 1

    # Selection frequency over 2B subsamples
    selection_freq = selection_counts / (2 * B)

    # Select features with frequency ≥ tau
    selected_features = np.where(selection_freq >= tau)[0].tolist()

    return selected_features, selection_freq


def HSIC_select(X, y, topK):
    hsic_lasso = HSICLasso()

    hsic_lasso.input(X, y)

    hsic_lasso.regression(topK)

    hsic_result = hsic_lasso.get_index()

    return hsic_result

def Konckoff_select(X, y, fdrate):
    kf1 = knockoff_filter.KnockoffFilter(fstat='randomforest')
    kf_selected = np.where(kf1.forward(X, y, fdr = fdrate) != 0)[0]
    return kf_selected