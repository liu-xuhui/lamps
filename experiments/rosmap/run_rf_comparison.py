import json
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
N_SPLITS = 10
K_LIST = list(range(1, 21, 1)) 

# Random Forest settings (fixed for fair comparison)
RF_PARAMS = dict(
    n_estimators=500,
    random_state=123,
    n_jobs=-1,
)

# Working dir is: C:\Users\95815\Desktop\research\adamp
DATA_DIR = os.path.join("data", "rosmap")
OUT_DIR = os.path.join("results", "rosmap")
os.makedirs(OUT_DIR, exist_ok=True)

# Precomputed feature-selection files. Most methods store rankings; Lasso and
# SPAM store an independent selected set for every split and K.
METHOD_FILES = {
    # "adamp_dr": os.path.join(OUT_DIR, "als_adamp_dr.csv"),
    "adamp_mr": os.path.join(OUT_DIR, "rosmap_adamp_mr.csv"),
    "lasso": os.path.join(OUT_DIR, "rosmap_lasso.csv"),
    # "adamp_sp": os.path.join(OUT_DIR, "als_adamp_sp.csv"),
    "hsic": os.path.join(OUT_DIR, "rosmap_hsic.csv"),
    "spam": os.path.join(OUT_DIR, "rosmap_spam.csv"),
}

# Output file names
OUT_FILES = {
    # "adamp_dr": os.path.join(OUT_DIR, "test_mse_adamp_dr.csv"),
    "adamp_mr": os.path.join(OUT_DIR, "test_mse_adamp_mr.csv"),
    "lasso": os.path.join(OUT_DIR, "test_mse_lasso.csv"),
    # "adamp_sp": os.path.join(OUT_DIR, "test_mse_adamp_sp.csv"),
    "hsic": os.path.join(OUT_DIR, "test_mse_hsic.csv"),
    "spam": os.path.join(OUT_DIR, "test_mse_spam.csv"),
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_split_data(split_id: int):
    """Load train/test split CSVs (generated from R). Returns X_train, y_train, X_test, y_test."""
    train_path = os.path.join(DATA_DIR, f"rosmap_train_split_{split_id:02d}.csv")
    test_path  = os.path.join(DATA_DIR, f"rosmap_test_split_{split_id:02d}.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train file: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test file: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    if "Y" not in train_df.columns or "Y" not in test_df.columns:
        raise ValueError("Both train and test CSVs must contain a 'Y' column.")

    y_train = train_df["Y"].to_numpy()
    X_train = train_df.drop(columns=["Y"]).to_numpy()

    y_test  = test_df["Y"].to_numpy()
    X_test  = test_df.drop(columns=["Y"]).to_numpy()

    return X_train, y_train, X_test, y_test


def load_rankings(path: str, n_splits: int):
    """
    Load the feature ranking matrix saved from R/Python.
    Expected: rows are ranked features (top first),
    columns correspond to split_01..split_10 (or similar).
    Values must be 0-based feature indices.
    Returns np.ndarray shape (n_ranks, n_splits).
    """
    df = pd.read_csv(path, index_col=0)  # keep rownames like rank_1 etc.
    arr = df.to_numpy()

    if arr.shape[1] != n_splits:
        raise ValueError(f"Expected {n_splits} split columns in {path}, got {arr.shape[1]}")

    # Ensure int
    arr = arr.astype(int)


    return arr


def load_independent_feature_sets(path: str, n_splits: int, method_name: str):
    """Load the independent feature set saved for every split and K."""
    method_label = method_name.upper()
    df = pd.read_csv(path)
    required_columns = {"split", "K", "lambda", "n_selected", "selected_features"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required {method_label} columns in {path}: "
            f"{sorted(missing_columns)}"
        )

    feature_sets = {}
    for row in df.itertuples(index=False):
        split_name = str(row.split)
        if not split_name.startswith("split_"):
            raise ValueError(f"Invalid split label in {path}: {split_name}")

        split_idx = int(split_name.removeprefix("split_"))
        K = int(row.K)
        key = (K, split_idx)
        if key in feature_sets:
            raise ValueError(
                f"Duplicate {method_label} feature set for K={K}, {split_name}"
            )

        try:
            selected = json.loads(row.selected_features)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid selected_features for K={K}, {split_name}: "
                f"{row.selected_features}"
            ) from error

        if not isinstance(selected, list) or not all(
            isinstance(index, int) and not isinstance(index, bool) for index in selected
        ):
            raise ValueError(
                f"selected_features must be a JSON list of integers for K={K}, "
                f"{split_name}"
            )
        if len(selected) != int(row.n_selected):
            raise ValueError(
                f"n_selected does not match selected_features for K={K}, {split_name}"
            )
        if len(selected) != len(set(selected)):
            raise ValueError(
                f"Duplicate {method_label} feature index for K={K}, {split_name}"
            )

        feature_sets[key] = np.asarray(selected, dtype=int)

    expected_keys = {
        (K, split_idx)
        for K in K_LIST
        for split_idx in range(1, n_splits + 1)
    }
    missing_keys = expected_keys - set(feature_sets)
    extra_keys = set(feature_sets) - expected_keys
    if missing_keys or extra_keys:
        raise ValueError(
            f"Unexpected {method_label} split/K entries in {path}; "
            f"missing={sorted(missing_keys)}, extra={sorted(extra_keys)}"
        )

    return feature_sets


def evaluate_method(method_name: str, ranking_path: str, out_path: str):
    if method_name in {"lasso", "spam"}:
        independent_feature_sets = load_independent_feature_sets(
            ranking_path,
            N_SPLITS,
            method_name,
        )
        rankings = None
    else:
        independent_feature_sets = None
        rankings = load_rankings(ranking_path, N_SPLITS)

    # We'll store MSE per split for each K
    rows = []
    for K in K_LIST:
        mse_per_split = []

        for split_idx in range(1, N_SPLITS + 1):
            X_train, y_train, X_test, y_test = load_split_data(split_idx)

            if method_name in {"lasso", "spam"}:
                # Lasso and SPAM store an independent selected set for every K;
                # neither file is interpreted as a nested feature ranking.
                feat_idx = independent_feature_sets[(K, split_idx)]
                if np.any(feat_idx < 0) or np.any(feat_idx >= X_train.shape[1]):
                    raise ValueError(
                        f"{method_name.upper()} feature index out of bounds for K={K}, "
                        f"split_{split_idx:02d}"
                    )
            else:
                # Get top-K features for this split (column split_idx-1)
                feat_idx = rankings[:K, split_idx - 1]
                # Ensure unique (some methods might accidentally repeat indices)
                feat_idx = np.unique(feat_idx)

            # Subset data
            Xtr = X_train[:, feat_idx]
            Xte = X_test[:, feat_idx]

            # Fit RF on TRAIN ONLY
            model = RandomForestRegressor(**RF_PARAMS)
            model.fit(Xtr, y_train)

            # Evaluate on TEST
            pred = model.predict(Xte)
            mse = mean_squared_error(y_test, pred)

            mse_per_split.append(mse)
            
        print(K)

        mse_per_split = np.array(mse_per_split, dtype=float)

        row = {"K": K, "mean_mse": mse_per_split.mean(), "std_mse": mse_per_split.std(ddof=1)}
        # Add split columns
        for j in range(N_SPLITS):
            row[f"split_{j+1:02d}"] = mse_per_split[j]
        rows.append(row)

    out_df = pd.DataFrame(rows)
    # Order columns nicely
    cols = ["K", "mean_mse", "std_mse"] + [f"split_{i:02d}" for i in range(1, N_SPLITS + 1)]
    out_df = out_df[cols]

    out_df.to_csv(out_path, index=False)
    print(f"[{method_name}] saved MSE table to: {out_path}")


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
def main():
    for method, path in METHOD_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ranking file not found for {method}: {path}")
        evaluate_method(method, path, OUT_FILES[method])


if __name__ == "__main__":
    main()
