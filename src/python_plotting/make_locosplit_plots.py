import pickle
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Load data
# -------------------------------------------------
with open("results/other/one_linear_corr09_locosplitreg_res.pkl", "rb") as f:
    res = pickle.load(f)

M = 500
num_signals = 10

arr_Deltas_list = [res["Delta"]]
arr_probs_list  = []

# -------------------------------------------------
# USER OPTIONS
# -------------------------------------------------
plot_kind = "delta"    # "delta" only
epoch_idx = 0         # choose which one: 0, 1, or 2

if plot_kind == "delta":
    arr = arr_Deltas_list[epoch_idx]
    x_label   = r"Feature importance $\Delta_j$"
    out_fname = f"results/other/feature_importance_hist_locosplit.png"
elif plot_kind == "prob":
    arr = arr_probs_list[epoch_idx]
    x_label   = r"Sampling probability $q_j$"
    out_fname = f"results/other/sampling_probability_hist_locosplit.png"
else:
    raise ValueError("plot_kind must be 'delta' or 'prob'.")

# -------------------------------------------------
# Plot styling (publication standard)
# -------------------------------------------------
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

# -------------------------------------------------
# Histogram bins (based on this single epoch)
# -------------------------------------------------
bins = np.histogram_bin_edges(arr, bins="auto")

# Colors
signal_color = "#D7191C"   # red
noise_color  = "#2C7BB6"   # blue

# -------------------------------------------------
# Create figure
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(5.8, 4.0))

# Split into signal vs noise
signal_vals = arr[:num_signals]
noise_vals  = arr[num_signals:]

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

# Labels & title
ax.set_xlabel(x_label)
ax.set_ylabel("Frequency")
ax.set_title("LOCO-Split Delta")

# Grid & spines
ax.grid(axis="y", alpha=0.25, linewidth=0.7)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# Legend
ax.legend(frameon=False, loc="best")

# Save & show
fig.tight_layout()
fig.savefig(out_fname, bbox_inches="tight")
plt.show()
