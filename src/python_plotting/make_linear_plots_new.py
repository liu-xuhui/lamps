import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ---------------- user-configurable inputs ----------------
eval_metd    = "f1_score"   # "f1_score" | "precision" | "recall"
oracle       = 1            # must be 1 (placeholder in filenames)
permute      = 1            # 1 = permute, 0 = nonpermute
basemodel    = "linear"     # only option

corr_list    = [0, 0.5, 0.9]
snr_list     = [5, 10, 20, 30, 40, 50, 60]
num_simus    = 10
max_iter     = 5
delta        = 0.8
number_signals = 10
M            = 500
# ----------------------------------------------------------

# Load the per-correlation results dict
with open(f"results/linear/linear_oracle{oracle}_permute{permute}_result_dict.pkl", "rb") as f:
    result_dict = pickle.load(f)

# Metric label mapping
metric_name_map = {"f1_score": "F1 Score", "precision": "Precision", "recall": "Recall"}
y_label = f"Feature-Selection {metric_name_map.get(eval_metd, eval_metd.title())}"

oracle_str  = "oracle" if oracle else "nonoracle"   # placeholder for title/filename
permute_str = "permute" if permute else "nonpermute"
title_label = (
    f"Linear — Feature-Selection {metric_name_map.get(eval_metd, eval_metd.title())} "
    f"vs. SNR {oracle_str} {permute_str}"
)

def compute_f1_score(selected_features, total_features=500, signal_features=list(range(10))):
    selected_features = set(selected_features)
    signal_features   = set(signal_features)
    tp = len(selected_features & signal_features)
    fp = len(selected_features - signal_features)
    fn = len(signal_features - selected_features)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        'precision': precision, 'recall': recall, 'f1_score': f1,
        'true_positives': tp, 'false_positives': fp, 'false_negatives': fn
    }

# Matplotlib publication style
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 120, "savefig.dpi": 300
})

# Colors/markers (AdaMP highlighted)
method_styles = {
    "AdaMP (linear)": {"color": "red",      "marker": "o", "lw": 2.2, "zorder": 5},
    "Stability":      {"color": "#596780",  "marker": "s", "lw": 1.6},
    "eBIC":           {"color": "#7AA6DC",  "marker": "x", "lw": 1.6},
    "Lasso-CV":       {"color": "#9CCB86",  "marker": "<", "lw": 1.6},
    "Lasso-OR":       {"color": "#DBA159",  "marker": ">", "lw": 1.6},
    "CPSS":           {"color": "#B57BA6",  "marker": "v", "lw": 1.6},
    "ElasticNet":     {"color": "#888888",  "marker": "*", "lw": 1.6},
    "ElasticNet-OR":  {"color": "#AA4499",  "marker": "D", "lw": 1.6},
}

# 1×3 grid over correlations; shared Y, one legend
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True, constrained_layout=False)
legend_handles = None
legend_labels  = None

for j, rho in enumerate(corr_list):
    # per-correlation method dict
    com_method_result_dict = result_dict[rho]

    # containers for per-SNR replicate metric
    met_AdaMP  = {snr: [] for snr in snr_list}
    met_Stab   = {snr: [] for snr in snr_list}
    met_eBIC   = {snr: [] for snr in snr_list}
    met_LCV    = {snr: [] for snr in snr_list}
    met_LOR    = {snr: [] for snr in snr_list}
    met_CPSS   = {snr: [] for snr in snr_list}
    met_EN     = {snr: [] for snr in snr_list}
    met_ENOR   = {snr: [] for snr in snr_list}

    # loop through SNRs and seeds (all linear methods assumed 0-based indexing for signals)
    for snr in snr_list:
        for i in range(num_simus):
            # AdaMP (linear)
            adam_select = com_method_result_dict["adamlinear"][snr][i]
            met_AdaMP[snr].append(
                compute_f1_score(adam_select, M, list(range(number_signals)))[eval_metd]
            )
            # Stability selection
            stab_select = com_method_result_dict["stability"][snr][i]
            met_Stab[snr].append(
                compute_f1_score(stab_select, M, list(range(number_signals)))[eval_metd]
            )
            # eBIC
            ebic_select = com_method_result_dict["ebic"][snr][i]
            met_eBIC[snr].append(
                compute_f1_score(ebic_select, M, list(range(number_signals)))[eval_metd]
            )
            # Lasso-CV
            lcv_select = com_method_result_dict["lassocv"][snr][i]
            met_LCV[snr].append(
                compute_f1_score(lcv_select, M, list(range(number_signals)))[eval_metd]
            )
            # Lasso-OR
            lor_select = com_method_result_dict["lassoor"][snr][i]
            met_LOR[snr].append(
                compute_f1_score(lor_select, M, list(range(number_signals)))[eval_metd]
            )
            # CPSS
            cpss_select = com_method_result_dict["cpss"][snr][i]
            met_CPSS[snr].append(
                compute_f1_score(cpss_select, M, list(range(number_signals)))[eval_metd]
            )
            # ElasticNet (CV)
            en_select = com_method_result_dict["elastic"][snr][i]
            met_EN[snr].append(
                compute_f1_score(en_select, M, list(range(number_signals)))[eval_metd]
            )
            # ElasticNet-OR
            enor_select = com_method_result_dict["elasticor"][snr][i]
            met_ENOR[snr].append(
                compute_f1_score(enor_select, M, list(range(number_signals)))[eval_metd]
            )

    # helper: mean & SE arrays in SNR order
    def mean_se(dct):
        means = [np.mean(dct[s]) if len(dct[s])>0 else np.nan for s in snr_list]
        ses   = [np.std(dct[s], ddof=1)/np.sqrt(len(dct[s])) if len(dct[s])>1 else 0.0 for s in snr_list]
        return np.array(means), np.array(ses)

    method_to_data = {
        "AdaMP (linear)": mean_se(met_AdaMP),
        "Stability":      mean_se(met_Stab),
        "eBIC":           mean_se(met_eBIC),
        "Lasso-CV":       mean_se(met_LCV),
        "Lasso-OR":       mean_se(met_LOR),
        "CPSS":           mean_se(met_CPSS),
        "ElasticNet":     mean_se(met_EN),
        "ElasticNet-OR":  mean_se(met_ENOR),
    }

    ax = axes[j]
    handles, labels = [], []

    # plot each method with error bars
    for name, (means, ses) in method_to_data.items():
        style = method_styles[name]
        h = ax.errorbar(
            snr_list, means, yerr=ses, fmt=style["marker"]+'-',
            linewidth=style.get("lw", 1.6), markersize=6,
            color=style["color"], capsize=3, elinewidth=1.0,
            zorder=style.get("zorder", 3), label=name
        )
        if j == 0:
            handles.append(h); labels.append(name)

    # cosmetics
    ax.set_xlabel("SNR")
    if j == 0:
        ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # annotate rho
    ax.text(0.02, 0.95, rf"$\rho = {rho}$", transform=ax.transAxes,
            ha="left", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#DDDDDD", alpha=0.9))

    if j == 0:
        legend_handles, legend_labels = handles, labels

# Shared title & legend
fig.suptitle(title_label, y=1.03)
if legend_handles is not None:
    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=4, frameon=False)

fig.subplots_adjust(bottom=0.25, top=0.9, wspace=0.1)

# Save
outdir = "results/linear"
os.makedirs(outdir, exist_ok=True)
figpath = os.path.join(outdir, f"{eval_metd}_{oracle_str}_{permute_str}_{basemodel}.png")
fig.savefig(figpath, bbox_inches="tight")
plt.show()
