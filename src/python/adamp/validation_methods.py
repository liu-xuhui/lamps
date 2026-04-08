import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from math import comb
from scipy.special  import gammaln
from sklearn.linear_model import ElasticNet

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
            if S > 0 and S < p:
                log_binom = gammaln(p + 1) - gammaln(S + 1) - gammaln(p - S + 1)
                penalty = 2 * gamma * log_binom
            else:
                penalty = 0

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

def compute_lasso_oracle_select_features(X, y, s_true=10, lambda_grid=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create lambda grid (alpha values)
    lambda_max = (np.max(np.abs(X_scaled.T @ y)) / X.shape[0])
    if lambda_grid is None:
        lambda_grid = np.logspace(np.log10(0.01 * lambda_max), np.log10(lambda_max), 50)

    best_model = None
    closest_diff = float('inf')

    for alpha in lambda_grid:
        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=1000)
        model.fit(X_scaled, y)
        nonzero_coef = np.sum(model.coef_ != 0)

        if nonzero_coef == s_true:
            selected_features = np.where(model.coef_ != 0)[0]
            coef_abs = np.abs(model.coef_)
            return selected_features.tolist(), coef_abs

        # Track best match so far if exact match not found
        diff = np.abs(nonzero_coef - s_true)
        if diff < closest_diff:
            closest_diff = diff
            best_model = model

    selected_features = np.where(best_model.coef_ != 0)[0]
    coef_abs = np.abs(best_model.coef_)
    return selected_features.tolist(), coef_abs

def compute_lasso_oracle_range_select_features(X, y, s_true=20, lambda_grid=None, max_extra=10):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lambda_max = np.max(np.abs(X_scaled.T @ y)) / X.shape[0]
    if lambda_grid is None:
        lambda_grid = np.logspace(np.log10(0.01 * lambda_max), np.log10(lambda_max), 50)

    best_model_in_range = None
    best_nonzero_in_range = None

    best_model_fallback = None
    closest_diff = float("inf")

    for alpha in lambda_grid:
        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=1000)
        model.fit(X_scaled, y)
        nonzero_coef = np.sum(model.coef_ != 0)

        # First priority: number of nonzero coefficients in [s_true, s_true + max_extra]
        if s_true <= nonzero_coef <= s_true + max_extra:
            # Prefer the smallest model within the valid range
            if (best_model_in_range is None) or (nonzero_coef < best_nonzero_in_range):
                best_model_in_range = model
                best_nonzero_in_range = nonzero_coef

                if nonzero_coef == s_true:
                    selected_features = np.where(model.coef_ != 0)[0]
                    coef_abs = np.abs(model.coef_)
                    return selected_features.tolist(), coef_abs

        # Fallback: closest to s_true if no valid model is found
        diff = abs(nonzero_coef - s_true)
        if diff < closest_diff:
            closest_diff = diff
            best_model_fallback = model

    # Use best model in desired range if available
    if best_model_in_range is not None:
        selected_features = np.where(best_model_in_range.coef_ != 0)[0]
        coef_abs = np.abs(best_model_in_range.coef_)
        return selected_features.tolist(), coef_abs

    # Otherwise fall back to closest model
    selected_features = np.where(best_model_fallback.coef_ != 0)[0]
    coef_abs = np.abs(best_model_fallback.coef_)
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
            model = Lasso(alpha=alpha, max_iter=1000)
            model.fit(X_scaled[A], y[A])
            selected = np.where(model.coef_ != 0)[0]
            selection_counts[selected] += 1

    # Selection frequency over 2B subsamples
    selection_freq = selection_counts / (2 * B)

    # Select features with frequency ≥ tau
    selected_features = np.where(selection_freq >= tau)[0].tolist()

    return selected_features, selection_freq

def Elastic(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lambda_max = (np.max(np.abs(X_scaled.T @ y)) / X.shape[0])
    
    lambda_grid = np.logspace(np.log10(0.01 * lambda_max), np.log10(lambda_max), 25)

    # LASSO with 5-fold CV
    lasso_cv = LassoCV(cv=5, alphas=lambda_grid, max_iter=1000)
    lasso_cv.fit(X_scaled, y)

    # Results
    best_alpha = lasso_cv.alpha_

    elastic_net = ElasticNet(alpha=best_alpha, l1_ratio=0.5, max_iter=1000)
    elastic_net.fit(X_scaled, y)

    selected_features = np.where(elastic_net.coef_ != 0)[0]

    coef_abs = np.abs(elastic_net.coef_)

    return selected_features.tolist(), coef_abs

def Elastic_oracle(X, y, s_true=10):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define lambda (alpha) grid and l1_ratio grid
    lambda_max = (np.max(np.abs(X_scaled.T @ y)) / X.shape[0])
    lambda_grid = np.logspace(np.log10(0.01 * lambda_max), np.log10(lambda_max), 25)
    l1_ratio_grid = np.linspace(0.1, 1.0, 10)

    best_model = None
    closest_diff = float('inf')

    for l1_ratio in l1_ratio_grid:
        for alpha in lambda_grid:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=1000)
            model.fit(X_scaled, y)
            nonzero_coef = np.sum(model.coef_ != 0)

            # Return immediately if match found
            if nonzero_coef == s_true:
                selected_features = np.where(model.coef_ != 0)[0]
                coef_abs = np.abs(model.coef_)
                return selected_features.tolist(), coef_abs

            # Otherwise, store best match so far
            diff = np.abs(nonzero_coef - s_true)
            if diff < closest_diff:
                closest_diff = diff
                best_model = model

    selected_features = np.where(best_model.coef_ != 0)[0]
    coef_abs = np.abs(best_model.coef_)
    return selected_features.tolist(), coef_abs

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


def compute_f1_score(selected_features, total_features=500, signal_features=list(range(10))):
    selected_features = set(selected_features)
    signal_features   = set(signal_features)
    tp = len(selected_features & signal_features)
    fp = len(selected_features - signal_features)
    fn = len(signal_features - selected_features)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        'precision': precision, 'recall': recall, 'f1_score': f1,
        'true_positives': tp, 'false_positives': fp, 'false_negatives': fn
    }