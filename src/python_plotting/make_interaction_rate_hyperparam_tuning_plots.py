import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ---------------- user-configurable inputs ----------------
oracle   = 0  # oracle must be 0
permute  = 0  # 1 = permute, 0 = nonpermute
basemodel = "both"  # "marsbase", "spambase", or "both"

corr_list = [0.5]       # keep ONE rho if you want ONE plot
snr_list  = [4, 6, 8, 10]

num_simus      = 30
max_iter       = 5
delta          = 0.8
number_signals = 10
M              = 50
# ----------------------------------------------------------

y_label = "Both-Interaction Selection (%)"

oracle_str  = "oracle"    if oracle == 1  else "nonoracle"
permute_str = "permute"   if permute == 1 else "nonpermute"

title_label = (
    "Nonlinear Nonadditive — Rate of Selecting Both Interaction Features Hyperparam Tuning "
    f"vs. Interaction SNR {permute_str}"
)

# Load result dict
with open(
    f"results/hyperparam_tuning/nonlinear_nonadditive_oracle{oracle}_permute{permute}_result_dict.pkl",
    "rb"
) as f:
    result_dict = pickle.load(f)

# ----------------------------------------------------------
# Matplotlib settings
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

# ----------------------------------------------------------
# Method styles (AdaMP highlighted)
# ----------------------------------------------------------
if basemodel == "marsbase":
    method_styles = {
        "AdaMP (marsbase)": {"color": "red",       "marker": "o", "lw": 2.2, "zorder": 4},
        "Knockoffs_fdr01":  {"color": "#7AA6DC",   "marker": "x", "lw": 1.6},
        "Knockoffs_fdr02":  {"color": "#9CCB86",   "marker": "<", "lw": 1.6},
        "Knockoffs_fdr03":  {"color": "#DBA159",   "marker": ">", "lw": 1.6},
        "spAM":             {"color": "#B57BA6",   "marker": "v", "lw": 1.6},
        "MARS":             {"color": "#888888",   "marker": "*", "lw": 1.6},
    }
elif basemodel == "spambase":
    method_styles = {
        "AdaMP (spambase)": {"color": "red",       "marker": "o", "lw": 2.2, "zorder": 4},
        "Knockoffs_fdr01":  {"color": "#7AA6DC",   "marker": "x", "lw": 1.6},
        "Knockoffs_fdr02":  {"color": "#9CCB86",   "marker": "<", "lw": 1.6},
        "Knockoffs_fdr03":  {"color": "#DBA159",   "marker": ">", "lw": 1.6},
        "spAM":             {"color": "#B57BA6",   "marker": "v", "lw": 1.6},
        "MARS":             {"color": "#888888",   "marker": "*", "lw": 1.6},
    }
else:  # basemodel == "both"
    method_styles = {
        "AdaMP (marsbase)": {"color": "#D7191C",   "marker": "o", "lw": 2.4, "zorder": 5},
        "AdaMP (spambase)": {"color": "#2C7BB6",   "marker": "D", "lw": 2.4, "zorder": 4},
        "Knockoffs_fdr01":  {"color": "#7AA6DC",   "marker": "x", "lw": 1.6},
        "Knockoffs_fdr02":  {"color": "#9CCB86",   "marker": "<", "lw": 1.6},
        "Knockoffs_fdr03":  {"color": "#DBA159",   "marker": ">", "lw": 1.6},
        "spAM":             {"color": "#B57BA6",   "marker": "v", "lw": 1.6},
        "MARS":             {"color": "#888888",   "marker": "*", "lw": 1.6},
    }

# ------------------- ONE PLOT ONLY -------------------
if len(corr_list) != 1:
    raise ValueError("For a single plot, set corr_list to exactly one value, e.g., corr_list = [0.5].")

rho = corr_list[0]
com_method_result_dict = result_dict[rho]

# containers for per-SNR replicate rates
if basemodel == "both":
    rate_AdaMP_mars = {snr: [] for snr in snr_list}
    rate_AdaMP_spam = {snr: [] for snr in snr_list}
else:
    rate_AdaMP      = {snr: [] for snr in snr_list}

rate_KF01 = {snr: [] for snr in snr_list}
rate_KF02 = {snr: [] for snr in snr_list}
rate_KF03 = {snr: [] for snr in snr_list}
rate_SPAM = {snr: [] for snr in snr_list}
rate_MARS = {snr: [] for snr in snr_list}

# loop through SNRs and seeds
for snr in snr_list:
    for i in range(num_simus):

        # AdaMP (marsbase)
        if basemodel in ("marsbase", "both"):
            adam_mars_select = set(com_method_result_dict["adammars"][snr][i])
            val_mars = int((1 in adam_mars_select) and (2 in adam_mars_select))
            if basemodel == "both":
                rate_AdaMP_mars[snr].append(val_mars)
            else:
                rate_AdaMP[snr].append(val_mars)

        # AdaMP (spambase)
        if basemodel in ("spambase", "both"):
            adam_spam_select = set(com_method_result_dict["adamspam"][snr][i])
            val_spam = int((1 in adam_spam_select) and (2 in adam_spam_select))
            if basemodel == "both":
                rate_AdaMP_spam[snr].append(val_spam)
            else:
                rate_AdaMP[snr].append(val_spam)

        # Knockoffs: check {0,1}
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
    means = [np.mean(dct[s]) if len(dct[s]) > 0 else np.nan for s in snr_list]
    ses   = [
        np.std(dct[s], ddof=1) / np.sqrt(len(dct[s])) if len(dct[s]) > 1 else 0.0
        for s in snr_list
    ]
    return np.array(means), np.array(ses)

# Build method_to_data depending on basemodel
if basemodel == "marsbase":
    method_to_data = {
        "AdaMP (marsbase)": mean_se(rate_AdaMP),
        "Knockoffs_fdr01":  mean_se(rate_KF01),
        "Knockoffs_fdr02":  mean_se(rate_KF02),
        "Knockoffs_fdr03":  mean_se(rate_KF03),
        "spAM":             mean_se(rate_SPAM),
        "MARS":             mean_se(rate_MARS),
    }
elif basemodel == "spambase":
    method_to_data = {
        "AdaMP (spambase)": mean_se(rate_AdaMP),
        "Knockoffs_fdr01":  mean_se(rate_KF01),
        "Knockoffs_fdr02":  mean_se(rate_KF02),
        "Knockoffs_fdr03":  mean_se(rate_KF03),
        "spAM":             mean_se(rate_SPAM),
        "MARS":             mean_se(rate_MARS),
    }
else:  # both
    method_to_data = {
        "AdaMP (marsbase)": mean_se(rate_AdaMP_mars),
        "AdaMP (spambase)": mean_se(rate_AdaMP_spam),
        "Knockoffs_fdr01":  mean_se(rate_KF01),
        "Knockoffs_fdr02":  mean_se(rate_KF02),
        "Knockoffs_fdr03":  mean_se(rate_KF03),
        "spAM":             mean_se(rate_SPAM),
        "MARS":             mean_se(rate_MARS),
    }

# ------------------- Plot (single panel) -------------------
fig, ax = plt.subplots(figsize=(7.8, 4.6), constrained_layout=False)

for name, (means, ses) in method_to_data.items():
    style = method_styles[name]
    ax.errorbar(
        snr_list, means, yerr=ses, fmt=style["marker"] + "-",
        linewidth=style.get("lw", 1.6),
        markersize=6,
        color=style["color"],
        capsize=3,
        elinewidth=1.0,
        zorder=style.get("zorder", 3),
        label=name
    )

ax.set_xlabel("Interaction SNR")
ax.set_ylabel(y_label)

ax.grid(True, alpha=0.25, linewidth=0.8)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.text(
    0.02, 0.95, rf"$\rho = {rho}$", transform=ax.transAxes,
    ha="left", va="top", fontsize=11,
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#DDDDDD", alpha=0.9),
)

ax.set_title(title_label, pad=10)

# Legend below plot (prevents overlap/squeezing)
legend = ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=4,
    frameon=False
)

# Reserve space for the legend
fig.subplots_adjust(bottom=0.28, top=0.88)

# Save
outdir = "results/hyperparam_tuning"
os.makedirs(outdir, exist_ok=True)
figpath = os.path.join(outdir, f"both_interaction_rate_{permute_str}_{basemodel}_rho{rho}.png")

fig.savefig(figpath, dpi=300, bbox_inches="tight", bbox_extra_artists=(legend,))
plt.show()
