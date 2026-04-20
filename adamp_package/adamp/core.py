"""Core AdaMP feature-selection implementation."""

from __future__ import annotations

import itertools
import sys

import numpy as np


NORMAL_975_QUANTILE = 1.959963984540054


def indept_sample_array(probability, rng):
    sampled_indices = []
    while len(sampled_indices) == 0:
        sampled_indices = np.where(rng.random(len(probability)) < probability)[0]
    return sampled_indices


def linear_regression_fit(X_train, y_train, X_pred):
    """Default dependency-free linear regression fit function."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    X_pred = np.asarray(X_pred, dtype=float)
    coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    return X_pred @ coef


def calc_K_list(n_ratio=0.4, m_ratio=0.12, max_iter=5):
    denom1 = (1 - n_ratio) * (m_ratio**2)
    denom2 = (1 - n_ratio) * m_ratio
    if denom1 <= 0 or denom2 <= 0:
        raise ValueError(f"Invalid (n_ratio, m_ratio)=({n_ratio}, {m_ratio})")
    K_i = int(np.ceil(max(50 / denom1, 200 / denom2)))
    return [K_i for _ in range(max_iter)]


def buildMP_indept(X, Y, n_ratio, m_ratio, prob_I=None, prob_F=None, delta=1, rng=None):
    N = len(X)
    M = len(X[0])
    n = int(n_ratio * N)

    if prob_I is None:
        idx_I = np.sort(rng.choice(N, size=n, replace=False))
    else:
        idx_I = np.sort(rng.choice(N, size=n, replace=False, p=prob_I))

    if prob_F is None:
        idx_F = indept_sample_array(np.ones(M) * m_ratio, rng)
    else:
        idx_F = indept_sample_array(prob_F, rng)

    x_mp = X[np.ix_(idx_I, idx_F)]
    y_mp = Y[np.ix_(idx_I)]
    return [idx_I, idx_F, x_mp, y_mp]


def _progress_bar(epoch, max_iter, completed, total, width=10, stream=None):
    if stream is None:
        stream = sys.stderr
    filled = int(np.floor(width * completed / total)) if total else width
    filled = min(width, max(0, filled))
    bar = "#" * filled + "-" * (width - filled)
    stream.write(f"\rEpoch {epoch + 1}/{max_iter} K [{bar}] {completed}/{total}")
    stream.flush()
    if completed >= total:
        stream.write("\n")
        stream.flush()


def _hyperparameter_progress_bar(completed, total, width=20, stream=None):
    if stream is None:
        stream = sys.stderr
    filled = int(np.floor(width * completed / total)) if total else width
    filled = min(width, max(0, filled))
    bar = "#" * filled + "-" * (width - filled)
    stream.write(f"\rHyperparameter tuning [{bar}] {completed}/{total} sets")
    stream.flush()
    if completed >= total:
        stream.write("\n")
        stream.flush()


def predictMP_indept(
    X,
    Y,
    X_pred,
    n_ratio,
    m_ratio,
    B,
    fit_func,
    prob_I=None,
    prob_F=None,
    delta=1,
    rng=None,
    progress_callback=None,
):
    N = len(X)
    M = len(X[0])
    in_mp_obs, in_mp_feature = np.zeros((B, N), dtype=bool), np.zeros((B, M), dtype=bool)
    predictions = []
    rngs = rng.spawn(B)
    update_every = max(1, int(np.ceil(B / 10)))

    for b in range(B):
        idx_I, idx_F, x_mp, y_mp = buildMP_indept(
            X, Y, n_ratio, m_ratio, prob_I, prob_F, delta, rng=rngs[b]
        )
        predictions.append(fit_func(x_mp, y_mp, X_pred[:, idx_F]))
        in_mp_obs[b, idx_I] = True
        in_mp_feature[b, idx_F] = True
        completed = b + 1
        if progress_callback is not None and (
            completed == B or completed % update_every == 0
        ):
            progress_callback(completed, B)

    return [np.array(predictions), in_mp_obs, in_mp_feature]


def MPRegFeatureScore_indept(
    X,
    Y,
    n_ratio,
    m_ratio,
    K,
    fit_func,
    prob_I,
    prob_F,
    delta,
    rng,
    progress_callback=None,
    X1=None,
    Y1=None,
):
    N = len(X)
    M = len(X[0])
    if X1 is None:
        X_pred = X
    else:
        X_pred = np.vstack((X, X1))

    predictions, in_mp_obs, in_mp_feature = predictMP_indept(
        X,
        Y,
        X_pred,
        n_ratio,
        m_ratio,
        K,
        fit_func,
        prob_I=prob_I,
        prob_F=prob_F,
        delta=delta,
        rng=rng,
        progress_callback=progress_callback,
    )

    predictions_train = predictions[:, :N]
    predictions_test = predictions[:, N:]
    mse_train = np.mean((Y - predictions_train.mean(0)) ** 2)
    if Y1 is None:
        mse_test = np.nan
    else:
        mse_test = np.mean((Y1 - predictions_test.mean(0)) ** 2)

    Y = np.asarray(Y)
    if Y.ndim > 1:
        Y = Y.reshape(-1)

    P = np.asarray(predictions_train)
    if P.ndim == 3 and P.shape[-1] == 1:
        P = P[..., 0]

    A = ~in_mp_obs
    F = ~in_mp_feature

    A_f = A.astype(np.float64)
    F_f = F.astype(np.float64)

    count_loo = A_f.sum(axis=0)
    sum_loo = (P * A_f).sum(axis=0)
    loo_pred = np.divide(
        sum_loo,
        count_loo,
        out=np.full_like(sum_loo, np.nan, dtype=np.float64),
        where=count_loo > 0,
    )
    resids_LOO = (Y - loo_pred) ** 2

    sum2_loo = ((P * P) * A_f).sum(axis=0)
    var_loo = np.divide(
        sum2_loo,
        count_loo,
        out=np.full_like(sum2_loo, np.nan, dtype=np.float64),
        where=count_loo > 0,
    ) - loo_pred**2
    var_loo = np.maximum(var_loo, 0.0)
    std_loo = np.sqrt(var_loo)

    LOO_sd = np.nanmean(std_loo)
    LOO_mean = np.nanmean(loo_pred)

    den = A_f.T @ F_f
    num = (P * A_f).T @ F_f
    loco_pred = np.divide(
        num,
        den,
        out=np.full_like(num, np.nan, dtype=np.float64),
        where=den > 0,
    )
    resid_loco = (Y[:, None] - loco_pred) ** 2
    zz = resid_loco - resids_LOO[:, None]
    Delta = np.nanmean(zz, axis=0)

    return {
        "Delta": Delta,
        "mse_train": mse_train,
        "mse_test": mse_test,
        "resids_LOO": resids_LOO,
        "loo": np.array(resids_LOO).mean(),
        "loo_std": LOO_sd,
        "loo_mean": LOO_mean,
        "m": sum(in_mp_feature[0] == 1),
        "last_pred": predictions[:, -1],
    }


def last_epoch_key(res):
    return next(reversed(res))


def extract_final_loo(res):
    last_k = last_epoch_key(res)
    loo = res[last_k].get("loo", None)
    if loo is None:
        raise RuntimeError("Could not extract final LOO from res[epoch]['loo'].")
    return float(loo)


def extract_selected_features_nonoracle(res, delta=0.8):
    last_k = last_epoch_key(res) - 1
    if last_k < 0:
        last_k = 0
    prob_F = res[last_k].get("prob_F", None)
    if prob_F is None:
        raise RuntimeError("Selected epoch has no prob_F stored.")
    return np.where(np.asarray(prob_F) >= delta * 0.5)[0]


def _coerce_K_list(K, n_ratio, m_ratio, max_iter):
    if K is None:
        return calc_K_list(n_ratio=n_ratio, m_ratio=m_ratio, max_iter=max_iter)
    if np.isscalar(K):
        return [int(K) for _ in range(max_iter)]
    if len(K) == 1:
        return [int(K[0]) for _ in range(max_iter)]
    if len(K) < max_iter:
        raise ValueError("K must be None, a scalar, length 1, or length >= max_iter.")
    return [int(k) for k in K[:max_iter]]


def _is_tuning_list(value):
    return isinstance(value, list)


def _coerce_grid_values(name, value, default=None):
    is_grid = _is_tuning_list(value)
    values = value if is_grid else [value]
    if len(values) == 0:
        raise ValueError(f"{name} list must contain at least one value.")
    if name == "fit_func":
        values = [default if fit_func is None else fit_func for fit_func in values]
        bad_values = [fit_func for fit_func in values if not callable(fit_func)]
        if bad_values:
            raise TypeError("fit_func values must be callable or None.")
    return values, is_grid


def _fit_func_label(fit_func):
    name = getattr(fit_func, "__name__", None)
    if name:
        return name
    wrapped_func = getattr(fit_func, "func", None)
    wrapped_name = getattr(wrapped_func, "__name__", None)
    if wrapped_name:
        return wrapped_name
    return fit_func.__class__.__name__


def _run_adamp(
    X,
    Y,
    n_ratio,
    m_ratio,
    K,
    fit_func,
    delta,
    max_iter,
    early_stop,
    seed,
    show_progress,
    X1=None,
    Y1=None,
):
    rng = np.random.default_rng(seed)
    N = len(X)
    M = len(X[0])
    kk = 0
    prob_I = None
    prob_F = None
    res = {}
    K_list = _coerce_K_list(K, n_ratio, m_ratio, max_iter)

    while kk < max_iter:
        progress_callback = None
        if show_progress:
            progress_callback = lambda completed, total, epoch=kk: _progress_bar(
                epoch, max_iter, completed, total
            )

        res[kk] = MPRegFeatureScore_indept(
            X,
            Y,
            n_ratio,
            m_ratio,
            K_list[kk],
            fit_func,
            prob_I,
            prob_F,
            delta,
            rng=rng,
            progress_callback=progress_callback,
            X1=X1,
            Y1=Y1,
        )

        if early_stop and kk > 0:
            cur = res[kk]
            prev = res[kk - 1]
            loo_change = np.array(cur["resids_LOO"]) - np.array(prev["resids_LOO"])
            loo_change = loo_change[~np.isnan(loo_change)]
            temp_z = (np.mean(loo_change) / np.std(loo_change, ddof=1)) * np.sqrt(N)
            threshold = -NORMAL_975_QUANTILE
            if temp_z >= threshold:
                res.popitem()
                break

        weight_tiuta = res[kk]["Delta"] - np.min(res[kk]["Delta"]) + 0.01 / M
        weight_sort = np.sort(weight_tiuta)[::-1]

        total_sum = sum(weight_sort)
        running_sum = 0
        target_index = 0
        m = M * m_ratio
        for i in range(M):
            running_sum += weight_sort[i]
            if i + 1 >= M:
                target_index = i
                break
            if total_sum - running_sum > (m / delta - i - 1) * weight_sort[i + 1]:
                target_index = i
                break
        delta_bar = sum(weight_sort[target_index + 1 :]) / (m / delta - (target_index + 1))

        weight = np.array([min(delta_tiuta, delta_bar) for delta_tiuta in weight_tiuta])
        sum_weight = sum(weight)
        prob_F = m * weight / sum_weight

        res[kk]["prob_F"] = prob_F
        res[kk]["weight"] = weight

        kk = kk + 1

    return res


def adamp_select(
    X,
    Y,
    n_ratio=0.4,
    m_ratio=0.12,
    K=None,
    fit_func=None,
    delta=0.8,
    max_iter=5,
    early_stop=True,
    return_complete_info=False,
    show_progress=True,
    seed=123,
):
    """Run AdaMP and return selected feature indices by default.

    Parameters
    ----------
    X, Y
        Training features and response.
    fit_func
        Callable with signature ``fit_func(X_train, y_train, X_predict)``. If
        omitted, a dependency-free least-squares linear regression is used.
        May also be a list of callables for hyperparameter tuning.
    n_ratio, m_ratio, delta
        Scalars for a single AdaMP run, or lists for grid-search tuning.
    return_complete_info
        If False, return selected feature indices. If True, return
        ``(selected_features, res)``. When any of ``n_ratio``, ``m_ratio``,
        ``fit_func``, or ``delta`` is a list, tuning is run before applying
        the same return rule.
    """
    if fit_func is None:
        fit_func = linear_regression_fit

    n_ratio_values, n_ratio_is_grid = _coerce_grid_values("n_ratio", n_ratio)
    m_ratio_values, m_ratio_is_grid = _coerce_grid_values("m_ratio", m_ratio)
    fit_func_values, fit_func_is_grid = _coerce_grid_values(
        "fit_func", fit_func, default=linear_regression_fit
    )
    delta_values, delta_is_grid = _coerce_grid_values("delta", delta)
    tune_hyperparameters = any(
        [n_ratio_is_grid, m_ratio_is_grid, fit_func_is_grid, delta_is_grid]
    )

    X = np.asarray(X)
    Y = np.asarray(Y)

    if tune_hyperparameters:
        grid = list(
            itertools.product(
                n_ratio_values, m_ratio_values, fit_func_values, delta_values
            )
        )
        best_loo = np.inf
        best_selected = None
        best_res = None
        best_params = None

        for completed, (cur_n_ratio, cur_m_ratio, cur_fit_func, cur_delta) in enumerate(
            grid, start=1
        ):
            res = _run_adamp(
                X=X,
                Y=Y,
                n_ratio=cur_n_ratio,
                m_ratio=cur_m_ratio,
                K=K,
                fit_func=cur_fit_func,
                delta=cur_delta,
                max_iter=max_iter,
                early_stop=early_stop,
                seed=seed,
                show_progress=show_progress,
            )
            loo_final = extract_final_loo(res)
            selected = extract_selected_features_nonoracle(res, delta=cur_delta)

            if np.isfinite(loo_final) and loo_final < best_loo:
                best_loo = loo_final
                best_selected = selected
                best_res = res
                best_params = (cur_n_ratio, cur_m_ratio, cur_fit_func, cur_delta)

            if show_progress:
                _hyperparameter_progress_bar(completed, len(grid))

        if best_res is None:
            raise RuntimeError("No finite final LOO value was found during tuning.")

        best_n_ratio, best_m_ratio, best_fit_func, best_delta = best_params
        print(
            "Best hyperparameters: "
            f"n_ratio={best_n_ratio}, "
            f"m_ratio={best_m_ratio}, "
            f"fit_func={_fit_func_label(best_fit_func)}, "
            f"delta={best_delta}, "
            f"last_epoch_LOO={best_loo}"
        )
        if return_complete_info:
            return best_selected, best_res
        return best_selected

    res = _run_adamp(
        X=X,
        Y=Y,
        n_ratio=n_ratio,
        m_ratio=m_ratio,
        K=K,
        fit_func=fit_func,
        delta=delta,
        max_iter=max_iter,
        early_stop=early_stop,
        seed=seed,
        show_progress=show_progress,
    )
    selected = extract_selected_features_nonoracle(res, delta=delta)
    if return_complete_info:
        return selected, res
    return selected


def indept_weight_sample_epochtuned_legacy(
    X,
    Y,
    X1,
    Y1,
    n_ratio,
    m_ratio,
    K,
    fit_func,
    delta,
    max_iter,
    plot=True,
    seed=123,
):
    """Compatibility wrapper for the pre-publication AdaMP call signature."""
    del plot
    return _run_adamp(
        X=np.asarray(X),
        Y=np.asarray(Y),
        n_ratio=n_ratio,
        m_ratio=m_ratio,
        K=K,
        fit_func=fit_func,
        delta=delta,
        max_iter=max_iter,
        early_stop=True,
        seed=seed,
        show_progress=False,
        X1=np.asarray(X1),
        Y1=np.asarray(Y1),
    )


def get_loco(i, j, in_mp_feature, in_mp_obs, predictions_train):
    b_keep_f = list(
        set(np.argwhere(~(in_mp_feature[:, j])).reshape(-1))
        & set(np.argwhere(~(in_mp_obs[:, i])).reshape(-1))
    )
    return predictions_train[b_keep_f, i].mean()
