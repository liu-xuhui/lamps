import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

corr_list    = [0, 0.5, 0.9]
eval_metd    = "recall"
oracle = True
permute = True
basemodel = "marsbase" # or "spambase"

metric_name_map = {
    "f1_score": "F1 Score",
    "precision": "Precision",
    "recall": "Recall"
}

y_label = f"Feature-Selection {metric_name_map.get(eval_metd, eval_metd.title())}"

oracle_str  = "Oracle"    if oracle  else "nonoracle"
permute_str = "Permute"   if permute else "nonpermute"

title_label = (
    f"Nonlinear NonAdditive — Feature-Selection "
    f"{metric_name_map.get(eval_metd, eval_metd.title())} vs. SNR "
    f"{oracle_str} {permute_str}"
)


snr_list        = [4, 6, 8, 10]
num_simus       = 30
max_iter        = 5
delta           = 0.8
number_signals  = 10
M               = 50

with open(f"result/nonlinear/nonadditive/nonlinear_nonadditive_{oracle_str}_{permute_str}_result_dict.pkl", "rb") as f:
    result_dict = pickle.load(f)

def compute_f1_score(selected_features, total_features=50, signal_features=list(range(10))):
    selected_features = set(selected_features)
    signal_features = set(signal_features)

    # True Positives (TP): selected & signal
    tp = len(selected_features & signal_features)

    # False Positives (FP): selected & not signal
    fp = len(selected_features - signal_features)

    # False Negatives (FN): signal & not selected
    fn = len(signal_features - selected_features)

    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 Score: 2 * (precision * recall) / (precision + recall)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }

# ----------------------------------------------------------
if oracle:
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
            "HSIC Lasso":        {"color": "#596780",   "marker": "s", "lw": 1.6},
            "spAM":              {"color": "#B57BA6",   "marker": "v", "lw": 1.6},
        }
    else:
        method_styles = {
            "AdaMP (spambase)":             {"color": "red",       "marker": "o", "lw": 2.2, "zorder": 4},
            "HSIC Lasso":        {"color": "#596780",   "marker": "s", "lw": 1.6},
            "spAM":              {"color": "#B57BA6",   "marker": "v", "lw": 1.6},
        }

    # Create grid: 1 row x 3 columns, share Y axis
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True, constrained_layout=True)

    # Collect legend handles only once
    legend_handles = None
    legend_labels  = None

    for j, rho in enumerate(corr_list):
        save_dir = "result/nonlinear/nonadditive"

        com_method_result_dict = result_dict[rho]

        # containers for per-SNR replicate F1
        f1_AdaMP = {snr: [] for snr in snr_list}
        f1_HSIC  = {snr: [] for snr in snr_list}
        f1_SPAM  = {snr: [] for snr in snr_list}

        # loop through SNRs and seeds
        for snr in snr_list:
            for i in range(num_simus):
                
                # Other methods selection → F1 (from the loaded comparison dict)
                if basemodel == "marsbase":
                    adam_select = com_method_result_dict["adammars"][snr][i]
                    f1_AdaMP[snr].append(compute_f1_score(adam_select, M, list(range(1,number_signals+1)))[eval_metd])
                else:
                    adam_select = com_method_result_dict["adamspam"][snr][i]
                    f1_AdaMP[snr].append(compute_f1_score(adam_select, M, list(range(1,number_signals+1)))[eval_metd])

                hsic_select = com_method_result_dict["hsic"][snr][i]
                f1_HSIC[snr].append(compute_f1_score(hsic_select, M, list(range(number_signals)))[eval_metd])

                spam_select = com_method_result_dict["spam"][snr][i]
                f1_SPAM[snr].append(compute_f1_score(spam_select, M, list(range(1,number_signals+1)))[eval_metd])


        # helper to get mean and SE in SNR order
        def mean_se(dct):
            means = [np.mean(dct[s]) if len(dct[s])>0 else np.nan for s in snr_list]
            ses   = [np.std(dct[s], ddof=1)/np.sqrt(len(dct[s])) if len(dct[s])>1 else 0.0 for s in snr_list]
            return np.array(means), np.array(ses)

        method_to_data = {
            "AdaMP":           mean_se(f1_AdaMP),
            "HSIC Lasso":      mean_se(f1_HSIC),
            "spAM":            mean_se(f1_SPAM),
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
    outdir = "result/nonlinear/nonadditive"
    figpath = os.path.join(outdir, f"{eval_metd}_nonadditive_{oracle_str}_{permute_str}.png")
    fig.savefig(figpath, bbox_inches="tight")
    plt.show()
else:
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
        save_dir = "result/nonlinear/nonadditive"

        com_method_result_dict = result_dict[rho]

        # containers for per-SNR replicate F1
        f1_AdaMP = {snr: [] for snr in snr_list}
        f1_KF01  = {snr: [] for snr in snr_list}
        f1_KF02  = {snr: [] for snr in snr_list}
        f1_KF03  = {snr: [] for snr in snr_list}
        f1_SPAM  = {snr: [] for snr in snr_list}
        f1_MARS  = {snr: [] for snr in snr_list}

        # loop through SNRs and seeds
        for snr in snr_list:
            for i in range(num_simus):                
                # Other methods selection → F1 (from the loaded comparison dict)
                if basemodel == "marsbase":
                    adam_select = com_method_result_dict["adammars"][snr][i]
                    f1_AdaMP[snr].append(compute_f1_score(adam_select, M, list(range(1,number_signals+1)))[eval_metd])
                else:
                    adam_select = com_method_result_dict["adamspam"][snr][i]
                    f1_AdaMP[snr].append(compute_f1_score(adam_select, M, list(range(1,number_signals+1)))[eval_metd])

                mars_select = com_method_result_dict["mars"][snr][i]
                f1_MARS[snr].append(compute_f1_score(mars_select, M, list(range(1,number_signals+1)))[eval_metd])

                spam_select = com_method_result_dict["spam"][snr][i]
                f1_SPAM[snr].append(compute_f1_score(spam_select, M, list(range(1,number_signals+1)))[eval_metd])

                kf_selected1 = com_method_result_dict["kf1"][snr][i]
                f1_KF01[snr].append(compute_f1_score(kf_selected1, M, list(range(number_signals)))[eval_metd])

                kf_selected2 = com_method_result_dict["kf2"][snr][i]
                f1_KF02[snr].append(compute_f1_score(kf_selected2, M, list(range(number_signals)))[eval_metd])

                kf_selected3 = com_method_result_dict["kf3"][snr][i]
                f1_KF03[snr].append(compute_f1_score(kf_selected3, M, list(range(number_signals)))[eval_metd])

        # helper to get mean and SE in SNR order
        def mean_se(dct):
            means = [np.mean(dct[s]) if len(dct[s])>0 else np.nan for s in snr_list]
            ses   = [np.std(dct[s], ddof=1)/np.sqrt(len(dct[s])) if len(dct[s])>1 else 0.0 for s in snr_list]
            return np.array(means), np.array(ses)

        method_to_data = {
            "AdaMP":           mean_se(f1_AdaMP),
            "Knockoffs_fdr01": mean_se(f1_KF01),
            "Knockoffs_fdr02": mean_se(f1_KF02),
            "Knockoffs_fdr03": mean_se(f1_KF03),
            "spAM":            mean_se(f1_SPAM),
            "MARS":            mean_se(f1_MARS),
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
    outdir = "result/nonlinear/nonadditive"
    figpath = os.path.join(outdir, f"{eval_metd}_nonadditive_{oracle_str}_{permute_str}.png")
    fig.savefig(figpath, bbox_inches="tight")
    plt.show()



