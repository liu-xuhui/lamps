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
with open(f"results/other/one_linear_corr{corr}_locosplitreg_res.pkl", "rb") as f:
    res = pickle.load(f)

M = 500
num_signals = 10

arr_Deltas_list = [res["Delta"]]
arr_probs_list  = []


if plot_kind == "delta":
    arr = arr_Deltas_list[epoch_idx]
    x_label   = r"Feature importance $\Delta_j$"
    out_fname = f"results/other/feature_importance_hist_corr{corr}_locosplit.png"
else:
    raise ValueError("plot_kind must be 'delta'")
# elif plot_kind == "prob":
#     arr = arr_probs_list[epoch_idx]
#     x_label   = r"Sampling probability $q_j$"
#     out_fname = f"results/other/sampling_probability_hist_corr{corr}_locosplit.png"


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
bin_width = bins[1] - bins[0]

# Colors
signal_color = "#D7191C"   # red
noise_color  = "#2C7BB6"   # blue

# -------------------------------------------------
# Create figure (two y-axes)
# -------------------------------------------------
fig, ax_noise = plt.subplots(figsize=(5.8, 4.0))
ax_signal = ax_noise.twinx()

# Split into signal vs noise
signal_vals = arr[:num_signals]
noise_vals  = arr[num_signals:]

# -------------------------------------------------
# Offset bar histograms (same bins, shifted centers)
# Keep original transparency: signal alpha=0.75, noise alpha=0.45
# -------------------------------------------------
sig_counts, _ = np.histogram(signal_vals, bins=bins)
noi_counts, _ = np.histogram(noise_vals,  bins=bins)

centers = bins[:-1] + 0.5 * bin_width

# small symmetric offset so bars near 0 don't overlap
offset = 0.18 * bin_width
bar_w  = 0.36 * bin_width

# Signal bars (RIGHT axis) shifted right
ax_signal.bar(
    centers + offset,
    sig_counts,
    width=bar_w,
    color=signal_color,
    edgecolor="black",
    linewidth=0.6,
    alpha=0.75,
    label="Signal",
    zorder=2,
)

# Noise bars (LEFT axis) shifted left
ax_noise.bar(
    centers - offset,
    noi_counts,
    width=bar_w,
    color=noise_color,
    edgecolor="black",
    linewidth=0.6,
    alpha=0.45,
    label="Noise",
    zorder=3,
)

# Labels & title
ax_noise.set_xlabel(x_label)
ax_noise.set_ylabel("Noise frequency")
ax_signal.set_ylabel("Signal frequency")
ax_noise.set_title("LOCO-Split Delta")

# Grid & spines
ax_noise.grid(axis="y", alpha=0.25, linewidth=0.7)
for spine in ["top", "right"]:
    ax_noise.spines[spine].set_visible(False)
ax_signal.spines["top"].set_visible(False)

# Legend (combined)
h1, l1 = ax_signal.get_legend_handles_labels()
h2, l2 = ax_noise.get_legend_handles_labels()
ax_noise.legend(h1 + h2, l1 + l2, frameon=False, loc="best")

# Save & show
fig.tight_layout()
fig.savefig(out_fname, bbox_inches="tight")
plt.show()

