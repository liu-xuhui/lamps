import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("results/other/one_linear_corr09_experiment_res.pkl", "rb") as f:
  res = pickle.load(f)

M = 500
num_signals = 10


arr_Deltas_list = [res[i]["Delta"] for i in range(3)]
arr_probs_list = [res[i]["prob_F"] for i in range(3)]


plot_kind = "delta"   # or "prob"

if plot_kind == "delta":
    arr_list   = arr_Deltas_list
    x_label    = r"Feature importance $\Delta_j$"
    out_fname  = "results/other/feature_importance_hist_3epochs.png"
elif plot_kind == "prob":
    arr_list   = arr_probs_list
    x_label    = r"Sampling probability $q_j$"
    out_fname  = "results/other/sampling_probability_hist_3epochs.png"
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
bins = np.histogram_bin_edges(all_vals, bins="auto")

# Colors (same as before)
signal_color = "#D7191C"   # red
noise_color  = "#2C7BB6"   # blue

epoch_titles = ["Epoch 1", "Epoch 2", "Epoch 3"]

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)

for idx, arr in enumerate(arr_list):
    ax = axes[idx]

    # Split into signal vs noise
    signal_vals = arr[:10]
    noise_vals  = arr[10:]

    # Signal histogram
    ax.hist(
        signal_vals,
        bins=bins,
        density=True,
        alpha=0.75,
        color=signal_color,
        edgecolor="black",
        linewidth=0.6,
        label=r"Signal",
    )

    # Noise histogram
    ax.hist(
        noise_vals,
        bins=bins,
        density=True,
        alpha=0.45,
        color=noise_color,
        edgecolor="black",
        linewidth=0.6,
        label=r"Noise",
    )

    # Titles & labels
    ax.set_title(epoch_titles[idx])
    ax.set_xlabel(x_label)
    if idx == 0:
        ax.set_ylabel("Density")

    # Grid & spines
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Legend only once (left panel)
    if idx == 0:
        ax.legend(frameon=False, loc="best")

fig.tight_layout()
fig.savefig(out_fname, bbox_inches="tight")
plt.show()
