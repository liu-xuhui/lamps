import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# User input
# -----------------------------
corr = 0.9
csv_fname = "prob_F_epochs_0314_corr0.9_snr5_permute0_rep2_mr.csv"
csv_path  = os.path.join("results", "other", csv_fname)

M = 500
num_signals = 10

# -----------------------------
# Load the last column (latest epoch prob_F) from CSV
# -----------------------------
df = pd.read_csv(csv_path)

# Option A (recommended): take the last non-"feature" column
prob_cols = [c for c in df.columns if c.lower() != "feature"]
if len(prob_cols) == 0:
    raise ValueError("No probability columns found in the CSV (expected columns like epoch_1, epoch_2, ...).")

last_col = prob_cols[-1]
arr = df[last_col].to_numpy(dtype=float)

# (Optional sanity check)
if arr.shape[0] != M:
    print(f"Warning: expected M={M} rows, but got {arr.shape[0]} rows from {csv_fname}")

arr = arr[np.isfinite(arr)]

# Split into signal vs noise (assumes signals are the first num_signals entries)
signal_vals = arr[:num_signals]
noise_vals  = arr[num_signals:]

# -----------------------------
# Plot (single probability histogram, dual y-axis)
# -----------------------------
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

# bins computed from the combined values (for comparability signal/noise in same plot)
all_vals = np.concatenate([signal_vals, noise_vals])
all_vals = all_vals[np.isfinite(all_vals)]
bins = np.histogram_bin_edges(all_vals, bins="auto")

signal_color = "#D7191C"  # red
noise_color  = "#2C7BB6"  # blue

fig, ax_noise = plt.subplots(1, 1, figsize=(4.8, 3.8))
ax_signal = ax_noise.twinx()

# noise (left y-axis)
ax_noise.hist(
    noise_vals,
    bins=bins,
    density=False,
    alpha=0.35,
    color=noise_color,
    edgecolor="black",
    linewidth=0.6,
    label="Noise",
)

# signal (right y-axis)
ax_signal.hist(
    signal_vals,
    bins=bins,
    density=False,
    alpha=0.80,
    color=signal_color,
    edgecolor="black",
    linewidth=0.8,
    label="Signal",
)

x_label = r"Sampling probability $q_j$"
ax_noise.set_title(f"Sampling probability n_ratio = 0.3 m_ratio = 0.14")
ax_noise.set_xlabel(x_label)
ax_noise.set_ylabel("Noise frequency")
ax_signal.set_ylabel("Signal frequency")

ax_noise.grid(axis="y", alpha=0.25, linewidth=0.7)
for spine in ["top", "right"]:
    ax_noise.spines[spine].set_visible(False)
ax_signal.spines["top"].set_visible(False)

# combined legend
h1, l1 = ax_noise.get_legend_handles_labels()
h2, l2 = ax_signal.get_legend_handles_labels()
ax_noise.legend(h2 + h1, l2 + l1, frameon=False, loc="upper right")

out_fname = os.path.join("results", "other", f"sampling_probability_hist_corr{corr}_0314_rep2_last_epoch.png")
fig.tight_layout()
fig.savefig(out_fname, bbox_inches="tight")
plt.show()
