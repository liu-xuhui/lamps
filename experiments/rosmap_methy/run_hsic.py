import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from pyHSICLasso import HSICLasso


project_root = Path(__file__).resolve().parents[2]
src_python = project_root / "src" / "python"
sys.path.insert(0, str(src_python))

from adamp.validation_methods import HSIC_select

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
np.random.seed(118)

n_splits = 10
features_to_choose = 20

# Working directory:
# C:\Users\95815\Desktop\research\adamp
data_dir = os.path.join("data", "rosmap_methy")
out_dir  = os.path.join("results", "rosmap_methy")

os.makedirs(out_dir, exist_ok=True)

# --------------------------------------------------
# Storage: 50 x 10 (rows = rank, cols = split)
# --------------------------------------------------
selected_mat = np.zeros((features_to_choose, n_splits), dtype=int)

# --------------------------------------------------
# Loop over splits
# --------------------------------------------------
for i in range(1, n_splits + 1):

    train_file = os.path.join(
        data_dir, f"rosmap_train_split_{i:02d}.csv"
    )

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")

    df = pd.read_csv(train_file)

    if "Y" not in df.columns:
        raise ValueError(f"'Y' column not found in {train_file}")

    Y = df["Y"].values
    X = df.drop(columns=["Y"]).values  # X is already numeric


    # hsic_lasso = HSICLasso()

    # hsic_lasso.input(X, Y)

    # hsic_lasso.regression(10)
    # hsic_lasso.dump()

    # break

    # --------------------------------------------------
    # HSIC feature selection (TRAIN ONLY)
    # --------------------------------------------------
    # HSIC_select returns features sorted by importance
    selected = HSIC_select(X, Y, features_to_choose)

    selected = np.asarray(selected, dtype=int)

    if selected.shape[0] != features_to_choose:
        print("rep: " + str(i))
        print(selected.shape[0])
        # raise ValueError(
        #     f"HSIC_select returned {selected.shape[0]} features, "
        #     f"expected {features_to_choose}"
        # )

    # Store as column i-1
    # selected_mat[:, i - 1] = selected
    selected_mat[:len(selected), i - 1] = selected

# --------------------------------------------------
# Save result
# --------------------------------------------------


col_names = [f"split_{i:02d}" for i in range(1, n_splits + 1)]
row_names = [f"rank_{i+1}" for i in range(features_to_choose)]

df_out = pd.DataFrame(
    selected_mat,
    index=row_names,
    columns=col_names
)

out_file = os.path.join(
    out_dir,
    f"rosmap_hsic.csv"
)

df_out.to_csv(out_file)

print("HSIC feature-selection results saved to:")
print(out_file)


