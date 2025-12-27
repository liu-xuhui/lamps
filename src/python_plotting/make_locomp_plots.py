import pickle
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# USER OPTIONS
# -------------------------------------------------
plot_kind = "delta"    # "delta" only
epoch_idx = 0
corr = 0.9

# -------------------------------------------------
# Load data
# -------------------------------------------------
with open(f"results/other/one_linear_corr{corr}_locompreg_res.pkl", "rb") as f:
    res = pickle.load(f)

M = 500
num_signals = 10

arr_Deltas_list = [res["Delta"]]
arr_probs_list  = []


if plot_kind == "delta":
    arr = arr_Deltas_list[epoch_idx]
    x_label   = r"Feature importance $\Delta_j$"
    out_fname = f"results/other/feature_importance_hist_corr{corr}_locomp.png"
else:
    raise ValueError("plot_kind must be 'delta'.")
# elif plot_kind == "prob":
#     arr = arr_probs_list[epoch_idx]
#     x_label   = r"Sampling probability $q_j$"
#     out_fname = f"results/other/sampling_probability_hist_corr{corr}_locomp.png"


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
# Histogram bins
# -------------------------------------------------
bins = np.histogram_bin_edges(arr, bins="auto")

# Colors
signal_color = "#D7191C"   # red
noise_color  = "#2C7BB6"   # blue

# -------------------------------------------------
# Create figure (TWO AXES)
# -------------------------------------------------
fig, ax_noise = plt.subplots(figsize=(5.8, 4.0))
ax_signal = ax_noise.twinx()

# Split into signal vs noise
signal_vals = arr[:num_signals]
noise_vals  = arr[num_signals:]

# -------------------------------------------------
# Noise histogram (LEFT axis)
# -------------------------------------------------
ax_noise.hist(
    noise_vals,
    bins=bins,
    density=False,
    alpha=0.45,
    color=noise_color,
    edgecolor="black",
    linewidth=0.6,
    label="Noise",
)

# -------------------------------------------------
# Signal histogram (RIGHT axis)
# -------------------------------------------------
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

# -------------------------------------------------
# Labels & title
# -------------------------------------------------
ax_noise.set_xlabel(x_label)
ax_noise.set_ylabel("Noise frequency")
ax_signal.set_ylabel("Signal frequency")
ax_noise.set_title("LOCO-MP Delta")

# -------------------------------------------------
# Grid & spines
# -------------------------------------------------
ax_noise.grid(axis="y", alpha=0.25, linewidth=0.7)
for spine in ["top", "right"]:
    ax_noise.spines[spine].set_visible(False)
ax_signal.spines["top"].set_visible(False)

# -------------------------------------------------
# Legend (combined)
# -------------------------------------------------
h1, l1 = ax_noise.get_legend_handles_labels()
h2, l2 = ax_signal.get_legend_handles_labels()
ax_noise.legend(h2 + h1, l2 + l1, frameon=False, loc="upper right")

# -------------------------------------------------
# Save & show
# -------------------------------------------------
fig.tight_layout()
fig.savefig(out_fname, bbox_inches="tight")
plt.show()

