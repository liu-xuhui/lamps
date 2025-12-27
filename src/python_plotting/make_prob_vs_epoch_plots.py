import pickle
import numpy as np
import matplotlib.pyplot as plt

# TODO: Add corr as user input setting

# User input
corr = 0
plot_kind = "prob"   # "delta" or "prob"

with open(f"results/other/one_linear_corr{corr}_experiment_res.pkl", "rb") as f:
    res = pickle.load(f)

M = 500
num_signals = 10

arr_Deltas_list = [np.asarray(res[i]["Delta"], dtype=float).ravel() for i in range(3)]
arr_probs_list  = [np.asarray(res[i]["prob_F"], dtype=float).ravel() for i in range(3)]


if plot_kind == "delta":
    arr_list   = arr_Deltas_list
    x_label    = r"Feature importance $\Delta_j$"
    out_fname  = f"results/other/feature_importance_hist{corr}_3epochs.png"
elif plot_kind == "prob":
    arr_list   = arr_probs_list
    x_label    = r"Sampling probability $q_j$"
    out_fname  = f"results/other/sampling_probability_hist{corr}_3epochs.png"
else:
    raise ValueError("plot_kind must be 'delta' or 'prob'.")

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

# Sanity check: expect exactly 3 epochs
if len(arr_list) != 3:
    raise ValueError("arr_list must contain exactly 3 arrays (one per epoch).")

# Compute global bin edges across all epochs for comparability
all_vals = np.concatenate(arr_list)
all_vals = all_vals[np.isfinite(all_vals)]
bins = np.histogram_bin_edges(all_vals, bins="auto")

# Colors
signal_color = "#D7191C"   # red
noise_color  = "#2C7BB6"   # blue

epoch_titles = ["Epoch 1", "Epoch 2", "Epoch 3"]

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharex=True)

for idx, arr in enumerate(arr_list):
    # Ensure numeric + finite
    arr = np.asarray(arr, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]

    # Split into signal vs noise
    signal_vals = arr[:num_signals]
    noise_vals  = arr[num_signals:]

    ax_noise = axes[idx]          # left y-axis
    ax_signal = ax_noise.twinx()  # right y-axis

    # --- Plot noise (left axis) ---
    ax_noise.hist(
        noise_vals,
        bins=bins,
        density=False,          # frequency (counts)
        alpha=0.35,
        color=noise_color,
        edgecolor="black",
        linewidth=0.6,
        label="Noise",
    )

    # --- Plot signal (right axis) ---
    ax_signal.hist(
        signal_vals,
        bins=bins,
        density=False,          # frequency (counts)
        alpha=0.80,
        color=signal_color,
        edgecolor="black",
        linewidth=0.8,
        label="Signal",
    )

    # Titles & labels
    ax_noise.set_title(epoch_titles[idx])
    ax_noise.set_xlabel(x_label)
    if idx == 0:
        ax_noise.set_ylabel("Noise frequency")
        ax_signal.set_ylabel("Signal frequency")

    # Grid & spines (keep it clean/publication style)
    ax_noise.grid(axis="y", alpha=0.25, linewidth=0.7)
    for spine in ["top", "right"]:
        ax_noise.spines[spine].set_visible(False)
    ax_signal.spines["top"].set_visible(False)

    # Legend only once (combine handles from both axes)
    if idx == 0:
        h1, l1 = ax_noise.get_legend_handles_labels()
        h2, l2 = ax_signal.get_legend_handles_labels()
        ax_noise.legend(h2 + h1, l2 + l1, frameon=False, loc="upper right")

fig.tight_layout()
fig.savefig(out_fname, bbox_inches="tight")
plt.show()

