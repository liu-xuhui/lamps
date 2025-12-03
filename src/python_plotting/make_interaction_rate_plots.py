import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


oracle = False # oracle must be False
permute = False
basemodel = "marsbase" # "marsbase" or "spambase"

corr_list    = [0, 0.5, 0.9]

metric_name_map = {
    "f1_score": "F1 Score",
    "precision": "Precision",
    "recall": "Recall"
}


y_label = "Both-Interaction Selection (%)"
snr_list        = [4, 6, 8, 10]

oracle_str  = "oracle"    if oracle  else "nonoracle"
permute_str = "permute"   if permute else "nonpermute"

title_label = f"Nonlinear Nonadditive — Rate of Selecting Both Interaction Features vs. Interaction SNR {permute_str}"

snr_list        = [4, 6, 8, 10]
num_simus       = 30
max_iter        = 5
delta           = 0.8
number_signals  = 10
M               = 50

with open(f"results/nonlinear/nonadditive/nonlinear_nonadditive_{oracle_str}_{permute_str}_result_dict.pkl", "rb") as f:
    result_dict = pickle.load(f)


# ----------------------------------------------------------

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 120,
    "savefig.dpi": 300
})

# Colors/markers (AdaMP highlighted)
if basemodel == "marsbase":
    method_styles = {
        "AdaMP (marsbase)":             {"color": "red",       "marker": "o", "lw": 2.2, "zorder": 4},
        "Knockoffs_fdr01":   {"color": "#7AA6DC",   "marker": "x", "lw": 1.6},
        "Knockoffs_fdr02":   {"color": "#9CCB86",   "marker": "<", "lw": 1.6},
        "Knockoffs_fdr03":   {"color": "#DBA159",   "marker": ">", "lw": 1.6},
        "spAM":              {"color": "#B57BA6",   "marker": "v", "lw": 1.6},
        "MARS":              {"color": "#888888",   "marker": "*", "lw": 1.6},
    }
else:
    method_styles = {
        "AdaMP (spambase)":             {"color": "red",       "marker": "o", "lw": 2.2, "zorder": 4},
        "Knockoffs_fdr01":   {"color": "#7AA6DC",   "marker": "x", "lw": 1.6},
        "Knockoffs_fdr02":   {"color": "#9CCB86",   "marker": "<", "lw": 1.6},
        "Knockoffs_fdr03":   {"color": "#DBA159",   "marker": ">", "lw": 1.6},
        "spAM":              {"color": "#B57BA6",   "marker": "v", "lw": 1.6},
        "MARS":              {"color": "#888888",   "marker": "*", "lw": 1.6},
    }

# Create grid: 1 row x 3 columns, share Y axis
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True, constrained_layout=True)

# Collect legend handles only once
legend_handles = None
legend_labels  = None

for j, rho in enumerate(corr_list):
    save_dir = "results/nonlinear/nonadditive"

    com_method_result_dict = result_dict[rho]

    # containers for per-SNR replicate F1
    rate_AdaMP = {snr: [] for snr in snr_list}
    rate_KF01  = {snr: [] for snr in snr_list}
    rate_KF02  = {snr: [] for snr in snr_list}
    rate_KF03  = {snr: [] for snr in snr_list}
    rate_SPAM  = {snr: [] for snr in snr_list}
    rate_MARS  = {snr: [] for snr in snr_list}

    # loop through SNRs and seeds
    for snr in snr_list:
        for i in range(num_simus):                
            if basemodel == "marsbase":
                adam_select = set(com_method_result_dict["adammars"][snr][i])
                rate_AdaMP[snr].append(int((1 in adam_select) and (2 in adam_select)))
            else:
                adam_select = set(com_method_result_dict["adamspam"][snr][i])
                rate_AdaMP[snr].append(int((1 in adam_select) and (2 in adam_select)))

            kf1_select = set(com_method_result_dict["kf1"][snr][i])
            rate_KF01[snr].append(int((0 in kf1_select) and (1 in kf1_select)))

            kf2_select = set(com_method_result_dict["kf2"][snr][i])
            rate_KF02[snr].append(int((0 in kf2_select) and (1 in kf2_select)))

            kf3_select = set(com_method_result_dict["kf3"][snr][i])
            rate_KF03[snr].append(int((0 in kf3_select) and (1 in kf3_select)))

            # spAM / MARS are 1-based (from R) → check {1,2}
            spam_select = set(com_method_result_dict["spam"][snr][i])
            rate_SPAM[snr].append(int((1 in spam_select) and (2 in spam_select)))

            mars_select = set(com_method_result_dict["mars"][snr][i])
            rate_MARS[snr].append(int((1 in mars_select) and (2 in mars_select)))

    # helper to get mean and SE in SNR order
    def mean_se(dct):
        means = [np.mean(dct[s]) if len(dct[s])>0 else np.nan for s in snr_list]
        ses   = [np.std(dct[s], ddof=1)/np.sqrt(len(dct[s])) if len(dct[s])>1 else 0.0 for s in snr_list]
        return np.array(means), np.array(ses)

    if basemodel == "marsbase":
        method_to_data = {
            "AdaMP (marsbase)":           mean_se(rate_AdaMP),
            "Knockoffs_fdr01": mean_se(rate_KF01),
            "Knockoffs_fdr02": mean_se(rate_KF02),
            "Knockoffs_fdr03": mean_se(rate_KF03),
            "spAM":            mean_se(rate_SPAM),
            "MARS":            mean_se(rate_MARS),
        }
    else:
        method_to_data = {
            "AdaMP (spambase)":           mean_se(rate_AdaMP),
            "Knockoffs_fdr01": mean_se(rate_KF01),
            "Knockoffs_fdr02": mean_se(rate_KF02),
            "Knockoffs_fdr03": mean_se(rate_KF03),
            "spAM":            mean_se(rate_SPAM),
            "MARS":            mean_se(rate_MARS),
        }

    ax = axes[j]
    handles = []
    labels  = []

    # plot each method with error bars (points+lines)
    for name, (means, ses) in method_to_data.items():
        style = method_styles[name]
        h = ax.errorbar(
            snr_list, means, yerr=ses, fmt=style["marker"]+'-',  # point + line
            linewidth=style.get("lw", 1.6), markersize=6,
            color=style["color"], capsize=3, elinewidth=1.0, zorder=style.get("zorder", 3),
            label=name
        )
        # collect handles/labels from first subplot for shared legend
        if j == 0:
            handles.append(h)
            labels.append(name)

    # axes cosmetics
    ax.set_xlabel("Interaction SNR")
    if j == 0:
        ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # panel annotation for correlation (ρ)
    ax.text(0.02, 0.95, rf"$\rho = {rho}$", transform=ax.transAxes,
            ha="left", va="top", fontsize=11, bbox=dict(boxstyle="round,pad=0.2",
            facecolor="white", edgecolor="#DDDDDD", alpha=0.9))

    # Save legend info once
    if j == 0:
        legend_handles, legend_labels = handles, labels

# Shared suptitle & legend
fig.suptitle(title_label, y=1.03)
if legend_handles is not None:
    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=4, frameon=False)

# Add a bit of bottom space for the shared legend
fig.subplots_adjust(bottom=0.18, top=0.85, wspace=0.1)

# Save the combined figure once; you can also save per-setting inside the loop if desired
outdir = "results/nonlinear/nonadditive"
figpath = os.path.join(outdir, f"both_interaction_rate_{permute_str}_{basemodel}.png")
fig.savefig(figpath, bbox_inches="tight")
plt.show()



