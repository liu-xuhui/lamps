import os
import pandas as pd
import matplotlib.pyplot as plt

def _read_mse_csv(path: str, method_name: str) -> pd.DataFrame:
    """
    Expected columns:
      - K (number of selected features)
      - mean_mse (or mse)
      - std_mse / se_mse / std_error (anything like that)
    We only use the first 3 columns to be robust to extra split_* columns.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    # Robust: use first 3 columns regardless of headers (handles split_01,... extras)
    if df.shape[1] < 3:
        raise ValueError(f"{method_name}: expected >=3 columns (K, mean_mse, se), got {df.shape[1]}")

    df = df.iloc[:, :3].copy()
    df.columns = ["K", "mean_mse", "se_mse"]

    # Coerce numeric
    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df["mean_mse"] = pd.to_numeric(df["mean_mse"], errors="coerce")
    df["se_mse"] = pd.to_numeric(df["se_mse"], errors="coerce")

    df = df.dropna(subset=["K", "mean_mse"]).sort_values("K")
    df["method"] = method_name
    return df

def main():
    # Working directory (repo root): C:\Users\95815\Desktop\research\adamp
    repo_root = r"C:\Users\95815\Desktop\research\adamp"

    # Input CSVs
    adamp_path = os.path.join(repo_root, "results", "rosmap", "test_mse_adamp_mr.csv")
    hsic_path  = os.path.join(repo_root, "results", "rosmap", "test_mse_hsic.csv")
    lasso_path = os.path.join(repo_root, "results", "rosmap", "test_mse_lasso.csv")

    # Output
    outdir = os.path.join(repo_root, "results", "rosmap")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "rosmap_mse_vs_K.png")

    # Read
    df_adamp = _read_mse_csv(adamp_path, "AdaMP (marsbase)")
    df_hsic  = _read_mse_csv(hsic_path,  "HSIC")
    df_lasso = _read_mse_csv(lasso_path, "Lasso")

    # -------- Plot --------
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=False)

    # Styling (kept consistent with your other plots)
    styles = {
        "AdaMP (marsbase)": dict(marker="o", lw=2.2, zorder=4),
        "HSIC":             dict(marker="s", lw=1.8, zorder=3),
        "Lasso":            dict(marker="^", lw=1.8, zorder=2),
    }

    for df in (df_adamp, df_hsic, df_lasso):
        name = df["method"].iloc[0]
        st = styles.get(name, {})
        ax.errorbar(
            df["K"].values,
            df["mean_mse"].values,
            yerr=df["se_mse"].values,
            fmt=st.get("marker", "o") + "-",
            linewidth=st.get("lw", 1.8),
            markersize=6,
            capsize=3,
            elinewidth=1.0,
            label=name,
            zorder=st.get("zorder", 3),
        )

    ax.set_xlabel("Number of selected features (K)")
    ax.set_ylabel("Mean test MSE (10-fold CV)")

    ax.set_xticks(df_adamp["K"].values)
    
    ax.grid(True, alpha=0.25, linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.set_title("ROSMAP — Test MSE vs Number of Selected Features", pad=10)

    # Put legend below, avoid overlap
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
    )

    fig.subplots_adjust(bottom=0.28, top=0.88)

    fig.savefig(outpath, bbox_inches="tight", bbox_extra_artists=(legend,))
    plt.show()

    print(f"Saved figure to: {outpath}")

if __name__ == "__main__":
    main()
