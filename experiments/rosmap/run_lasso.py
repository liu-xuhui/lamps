"""Save ROSMAP Lasso feature sets for target sizes from 1 to 20.

For each train split and each k, this script looks for a Lasso penalty whose
model has exactly k nonzero coefficients.  A coarse logarithmic penalty path is
searched first, and intervals that skip k are refined by geometric bisection.
If an exact support size still cannot be found, the closest model is used and a
warning is printed.

Each split/k model is saved independently, so the output does not assume that
the Lasso supports are nested.  The output is intentionally written to the
same file as the original Lasso experiment: results/rosmap/rosmap_lasso.csv.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SEED = 118
N_SPLITS = 10
MAX_FEATURES = 20
SEARCH_SUPPORT_LIMIT = MAX_FEATURES + 10

# The coarse path is supplemented by bisection, so it does not need to be
# extremely dense.  The minimum ratio is expanded further if needed.
N_LAMBDAS = 100
MIN_LAMBDA_RATIO = 1e-3
MIN_ALLOWED_LAMBDA_RATIO = 1e-10
MAX_REFINEMENT_STEPS = 60

MAX_ITER = 20_000
TOL = 1e-7

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "rosmap"
OUT_DIR = PROJECT_ROOT / "results" / "rosmap"
OUT_FILE = OUT_DIR / "rosmap_lasso.csv"


@dataclass(frozen=True)
class LassoCandidate:
    """One fitted point on the Lasso regularization path."""

    alpha: float
    coef: np.ndarray
    n_selected: int


def fit_candidate(X_scaled: np.ndarray, y_centered: np.ndarray, alpha: float) -> LassoCandidate:
    """Fit one deterministic Lasso model and retain its support information."""

    model = Lasso(
        alpha=alpha,
        fit_intercept=False,
        max_iter=MAX_ITER,
        tol=TOL,
        selection="cyclic",
        random_state=SEED,
    )
    model.fit(X_scaled, y_centered)
    coef = model.coef_.copy()
    return LassoCandidate(
        alpha=float(alpha),
        coef=coef,
        n_selected=int(np.count_nonzero(coef)),
    )


def initial_path(X_scaled: np.ndarray, y_centered: np.ndarray) -> list[LassoCandidate]:
    """Fit a coarse path beyond the largest requested support size."""

    n_samples = X_scaled.shape[0]
    lambda_max = float(np.max(np.abs(X_scaled.T @ y_centered)) / n_samples)
    if not np.isfinite(lambda_max) or lambda_max <= 0:
        raise ValueError("Cannot construct a Lasso path because lambda_max is not positive.")

    ratio = MIN_LAMBDA_RATIO
    candidates: list[LassoCandidate] = []

    while True:
        alphas = np.geomspace(lambda_max, lambda_max * ratio, N_LAMBDAS)
        # Avoid refitting lambda_max when the path must be extended.
        if candidates:
            previous_min = candidates[-1].alpha
            alphas = alphas[alphas < previous_min]

        for alpha in alphas:
            candidates.append(fit_candidate(X_scaled, y_centered, alpha))
            # Go somewhat beyond k=20 so a support size skipped at an earlier
            # knot can still be found after a variable drops out of the path.
            # There is no need to approach an almost-unregularized solution.
            if candidates[-1].n_selected >= SEARCH_SUPPORT_LIMIT:
                return candidates
        if ratio <= MIN_ALLOWED_LAMBDA_RATIO:
            return candidates

        ratio = max(ratio * 0.01, MIN_ALLOWED_LAMBDA_RATIO)


def refine_for_k(
    X_scaled: np.ndarray,
    y_centered: np.ndarray,
    k: int,
    candidates: list[LassoCandidate],
) -> None:
    """Refine every coarse interval whose support counts straddle k."""

    ordered = sorted(candidates, key=lambda candidate: candidate.alpha, reverse=True)
    brackets: list[tuple[LassoCandidate, LassoCandidate]] = []

    for high_alpha_model, low_alpha_model in zip(ordered, ordered[1:]):
        high_delta = high_alpha_model.n_selected - k
        low_delta = low_alpha_model.n_selected - k
        if high_delta * low_delta < 0:
            brackets.append((high_alpha_model, low_alpha_model))

    for high_alpha_model, low_alpha_model in brackets:
        high_alpha = high_alpha_model.alpha
        high_count = high_alpha_model.n_selected
        low_alpha = low_alpha_model.alpha

        for _ in range(MAX_REFINEMENT_STEPS):
            # Lasso paths are naturally spaced on the log(lambda) scale.
            alpha = float(np.sqrt(high_alpha * low_alpha))
            if alpha == high_alpha or alpha == low_alpha:
                break

            candidate = fit_candidate(X_scaled, y_centered, alpha)
            candidates.append(candidate)

            if candidate.n_selected == k:
                return
            # Retain the half whose endpoint counts remain on opposite sides
            # of k.  This also handles local support-count decreases caused by
            # variables dropping out of a non-monotone Lasso path.
            if (high_count - k) * (candidate.n_selected - k) < 0:
                low_alpha = alpha
            else:
                high_alpha = alpha
                high_count = candidate.n_selected


def choose_candidate(candidates: list[LassoCandidate], k: int) -> tuple[LassoCandidate, bool]:
    """Choose an exact-k model, or deterministically choose the closest model."""

    exact = [candidate for candidate in candidates if candidate.n_selected == k]
    if exact:
        # If an interval of lambda values has k variables, use the strongest
        # regularization in that interval.
        return max(exact, key=lambda candidate: candidate.alpha), True

    # Tie-breaking is deterministic: prefer a model with at least k variables,
    # then prefer the larger lambda.
    closest = min(
        candidates,
        key=lambda candidate: (
            abs(candidate.n_selected - k),
            candidate.n_selected < k,
            -candidate.alpha,
        ),
    )
    return closest, False


def choose_models_for_all_k(X: np.ndarray, y: np.ndarray) -> list[LassoCandidate]:
    """Return the selected Lasso model for each target k=1,...,MAX_FEATURES."""

    X_scaled = StandardScaler().fit_transform(X)
    y_centered = np.asarray(y, dtype=float).ravel()
    y_centered = y_centered - y_centered.mean()

    candidates = initial_path(X_scaled, y_centered)
    selected_models: list[LassoCandidate] = []

    for k in range(1, MAX_FEATURES + 1):
        if not any(candidate.n_selected == k for candidate in candidates):
            refine_for_k(X_scaled, y_centered, k, candidates)

        selected, exact = choose_candidate(candidates, k)
        selected_models.append(selected)

        if not exact:
            print(
                "WARNING: "
                f"no lambda produced exactly {k} selected features; "
                f"using lambda={selected.alpha:.12g}, which selected "
                f"{selected.n_selected} features."
            )

    return selected_models


def load_split(split_id: int) -> tuple[np.ndarray, np.ndarray]:
    train_file = DATA_DIR / f"rosmap_train_split_{split_id:02d}.csv"
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    data = pd.read_csv(train_file)
    if "Y" not in data.columns:
        raise ValueError(f"'Y' column not found in {train_file}")

    X = data.drop(columns=["Y"]).to_numpy(dtype=float)
    y = data["Y"].to_numpy(dtype=float)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"Non-finite data found in {train_file}")
    return X, y


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    selected_sets: list[dict[str, object]] = []

    for split_id in range(1, N_SPLITS + 1):
        X, y = load_split(split_id)
        models = choose_models_for_all_k(X, y)

        for k, model in enumerate(models, start=1):
            feature_set = np.flatnonzero(model.coef != 0).astype(int).tolist()
            selected_sets.append(
                {
                    "split": f"split_{split_id:02d}",
                    "K": k,
                    "lambda": model.alpha,
                    "n_selected": model.n_selected,
                    "selected_features": json.dumps(feature_set),
                }
            )

        summary = ", ".join(
            f"k={k}:lambda={model.alpha:.6g}/n={model.n_selected}"
            for k, model in enumerate(models, start=1)
        )
        print(f"split {split_id:02d}: {summary}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(selected_sets)
    output.to_csv(OUT_FILE, index=False)

    print("Lasso feature-selection results saved to:")
    print(OUT_FILE)


if __name__ == "__main__":
    main()
