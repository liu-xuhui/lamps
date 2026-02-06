import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
N_SPLITS = 10
K_LIST = list(range(1, 20, 1))  # 2,4,...,30

# Random Forest settings (fixed for fair comparison)
RF_PARAMS = dict(
    n_estimators=500,
    random_state=123,
    n_jobs=-1,
)

# Working dir is: C:\Users\95815\Desktop\research\adamp
DATA_DIR = os.path.join("data", "rosmap_methy")
OUT_DIR = os.path.join("results", "rosmap_methy")
os.makedirs(OUT_DIR, exist_ok=True)

# Precomputed feature ranking files (each column is a split; rows are ranked features)
METHOD_FILES = {
    # "adamp_dr": os.path.join(OUT_DIR, "als_adamp_dr.csv"),
    "adamp_mr": os.path.join(OUT_DIR, "rosmap_adamp_mr.csv"),
    "lasso": os.path.join(OUT_DIR, "rosmap_lasso.csv"),
    # "adamp_sp": os.path.join(OUT_DIR, "als_adamp_sp.csv"),
    "hsic": os.path.join(OUT_DIR, "rosmap_hsic.csv"),
}

# Output file names
OUT_FILES = {
    # "adamp_dr": os.path.join(OUT_DIR, "test_mse_adamp_dr.csv"),
    "adamp_mr": os.path.join(OUT_DIR, "test_mse_adamp_mr.csv"),
    "lasso": os.path.join(OUT_DIR, "test_mse_lasso.csv"),
    # "adamp_sp": os.path.join(OUT_DIR, "test_mse_adamp_sp.csv"),
    "hsic": os.path.join(OUT_DIR, "test_mse_hsic.csv"),
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


def evaluate_method(method_name: str, ranking_path: str, out_path: str):
    rankings = load_rankings(ranking_path, N_SPLITS)

    # We'll store MSE per split for each K
    rows = []
    for K in K_LIST:
        mse_per_split = []

        for split_idx in range(1, N_SPLITS + 1):
            X_train, y_train, X_test, y_test = load_split_data(split_idx)

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
