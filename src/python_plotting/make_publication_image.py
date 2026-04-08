import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# User-configurable inputs
# ==========================================================
eval_metd = "f1_score"      # "f1_score" | "precision" | "recall"
oracle = 0                   # 0 or 1
basemodel = "nonlinear"         # "linear" or "nonlinear"

corr_list = [0, 0.5, 0.9]
M = 500

metric_name_map = {
    "f1_score": "F1 Score",
    "precision": "Precision",
    "recall": "Recall"
}
y_label = f"Feature-Selection {metric_name_map.get(eval_metd, eval_metd.title())}"
oracle_str = "oracle" if oracle else "nonoracle"

# ==========================================================
# Common helpers
# ==========================================================
def compute_metrics(selected_features, signal_features):
    selected_features = set(selected_features)
    signal_features = set(signal_features)

    tp = len(selected_features & signal_features)
    fp = len(selected_features - signal_features)
    fn = len(signal_features - selected_features)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }

def mean_se(dct, x_list):
    means = [np.mean(dct[x]) if len(dct[x]) > 0 else np.nan for x in x_list]
    ses = [
        np.std(dct[x], ddof=1) / np.sqrt(len(dct[x])) if len(dct[x]) > 1 else 0.0
        for x in x_list
    ]
    return np.array(means), np.array(ses)

def get_signal_features(index_mode, number_signals):
    if index_mode == "zero_based":
        return list(range(number_signals))
    elif index_mode == "one_based":
        return list(range(1, number_signals + 1))
    else:
        raise ValueError(f"Unknown index_mode: {index_mode}")

def get_method_curves(com_method_result_dict, method_specs, x_list, num_simus, number_signals):
    out = {}

    for label, method_key, index_mode in method_specs:
        metric_by_x = {x: [] for x in x_list}
        signal_features = get_signal_features(index_mode, number_signals)

        for x in x_list:
            for i in range(num_simus):
                selected = com_method_result_dict[method_key][x][i]
                metric_by_x[x].append(
                    compute_metrics(selected, signal_features)[eval_metd]
                )

        out[label] = mean_se(metric_by_x, x_list)

    return out

# ==========================================================
# Plot style
# ==========================================================
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

# ==========================================================
# Configuration by model family
# ==========================================================
if basemodel == "linear":
    x_list = [5, 10, 20, 30, 40, 50, 60]
    num_simus = 10
    number_signals = 10

    with open(f"results/linear/linear_oracle{oracle}_permute0_result_dict.pkl", "rb") as f:
        result_dict_setting1 = pickle.load(f)

    with open(f"results/linear/linear_oracle{oracle}_permute1_result_dict.pkl", "rb") as f:
        result_dict_setting2 = pickle.load(f)

    row_titles = ["Linear Setting 1", "Linear Setting 2"]
    suptitle = f"Linear — Feature-Selection {metric_name_map.get(eval_metd, eval_metd.title())} vs. SNR ({oracle_str})"
    outdir = "temp_results/image"
    outfile = os.path.join(outdir, f"linear_{eval_metd}_{oracle_str}_combined.png")

    method_styles = {
        "AdaMP (OLS)":          {"color": "red",      "marker": "o", "lw": 2.2, "zorder": 5},
        "Stability Selection":  {"color": "#596780",  "marker": "s", "lw": 1.6},
        "Lasso-eBIC":           {"color": "#7AA6DC",  "marker": "x", "lw": 1.6},
        "Lasso-CV":             {"color": "#9CCB86",  "marker": "<", "lw": 1.6},
        "Lasso-OR":             {"color": "#DBA159",  "marker": ">", "lw": 1.6},
        "CPSS":                 {"color": "#B57BA6",  "marker": "v", "lw": 1.6},
        "ElasticNet-CV":        {"color": "#888888",  "marker": "*", "lw": 1.6},
        "ElasticNet-OR":        {"color": "#AA4499",  "marker": "D", "lw": 1.6},
    }

    method_specs = [
        ("AdaMP (OLS)",         "adamlinear", "zero_based"),
        ("Stability Selection", "stability",  "zero_based"),
        ("Lasso-eBIC)",         "ebic",       "zero_based"),  # temporary placeholder, fixed below
        ("Lasso-CV",            "lassocv",    "zero_based"),
        ("Lasso-OR",            "lassoor",    "zero_based"),
        ("CPSS",                "cpss",       "zero_based"),
        ("ElasticNet-CV",       "elastic",    "zero_based"),
        ("ElasticNet-OR",       "elasticor",  "zero_based"),
    ]
    method_specs[2] = ("Lasso-eBIC", "ebic", "zero_based")

elif basemodel == "nonlinear":
    x_list = [0.5, 1, 2, 5]
    num_simus = 10
    number_signals = 10

    with open(f"results/nonlinear/additive/nonlinear_additive_oracle{oracle}_permute0_result_dict.pkl", "rb") as f:
        result_dict_setting1 = pickle.load(f)

    with open(f"results/nonlinear/additive/nonlinear_additive_oracle{oracle}_permute1_result_dict.pkl", "rb") as f:
        result_dict_setting2 = pickle.load(f)

    row_titles = ["Nonlinear Additive Setting 1", "Nonlinear Additive Setting 2"]
    suptitle = f"Nonlinear Additive — Feature-Selection {metric_name_map.get(eval_metd, eval_metd.title())} vs. SNR ({oracle_str})"
    outdir = "temp_results/image"
    outfile = os.path.join(outdir, f"nonlinear_additive_{eval_metd}_{oracle_str}_combined.png")

    if oracle:
        method_styles = {
            "AdaMP (marsbase)":  {"color": "#D7191C", "marker": "o", "lw": 2.4, "zorder": 5},
            "AdaMP (spambase)":  {"color": "#2C7BB6", "marker": "D", "lw": 2.4, "zorder": 4},
            "HSIC Lasso":        {"color": "#596780", "marker": "s", "lw": 1.6},
            "spAM":              {"color": "#B57BA6", "marker": "v", "lw": 1.6},
        }

        method_specs = [
            ("AdaMP (marsbase)", "adammars", "one_based"),
            ("AdaMP (spambase)", "adamspam", "one_based"),
            ("HSIC Lasso",       "hsic",     "zero_based"),
            ("spAM",             "spam",     "one_based"),
        ]
    else:
        method_styles = {
            "AdaMP (marsbase)":  {"color": "#D7191C", "marker": "o", "lw": 2.4, "zorder": 5},
            "AdaMP (spambase)":  {"color": "#2C7BB6", "marker": "D", "lw": 2.4, "zorder": 4},
            "Knockoffs_fdr01":   {"color": "#7AA6DC", "marker": "x", "lw": 1.6},
            "Knockoffs_fdr02":   {"color": "#9CCB86", "marker": "<", "lw": 1.6},
            "Knockoffs_fdr03":   {"color": "#DBA159", "marker": ">", "lw": 1.6},
            "spAM":              {"color": "#B57BA6", "marker": "v", "lw": 1.6},
            "MARS":              {"color": "#888888", "marker": "*", "lw": 1.6},
        }

        method_specs = [
            ("AdaMP (marsbase)", "adammars", "one_based"),
            ("AdaMP (spambase)", "adamspam", "one_based"),
            ("Knockoffs_fdr01",  "kf1",      "zero_based"),
            ("Knockoffs_fdr02",  "kf2",      "zero_based"),
            ("Knockoffs_fdr03",  "kf3",      "zero_based"),
            ("spAM",             "spam",     "one_based"),
            ("MARS",             "mars",     "one_based"),
        ]
else:
    raise ValueError("basemodel must be 'linear' or 'nonlinear'.")

# ==========================================================
# Combined 2x3 plot
# ==========================================================
fig, axes = plt.subplots(
    2, 3,
    figsize=(13.8, 9.2),
    sharex=True,
    sharey=True,
    constrained_layout=False
)

all_result_dicts = [result_dict_setting1, result_dict_setting2]

legend_handles = None
legend_labels = None

for row_idx, result_dict in enumerate(all_result_dicts):
    for col_idx, rho in enumerate(corr_list):
        ax = axes[row_idx, col_idx]
        com_method_result_dict = result_dict[rho]
        method_to_data = get_method_curves(
            com_method_result_dict=com_method_result_dict,
            method_specs=method_specs,
            x_list=x_list,
            num_simus=num_simus,
            number_signals=number_signals
        )

        handles = []
        labels = []

        for name, (means, ses) in method_to_data.items():
            style = method_styles[name]
            h = ax.errorbar(
                x_list,
                means,
                yerr=ses,
                fmt=style["marker"] + "-",
                linewidth=style.get("lw", 1.6),
                markersize=6,
                color=style["color"],
                capsize=3,
                elinewidth=1.0,
                zorder=style.get("zorder", 3),
                label=name
            )
            if row_idx == 0 and col_idx == 0:
                handles.append(h)
                labels.append(name)

        ax.grid(True, alpha=0.25, linewidth=0.8)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        ax.text(
            0.02, 0.95, rf"$\rho = {rho}$",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=11,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="#DDDDDD",
                alpha=0.9
            )
        )

        if col_idx == 0:
            ax.set_ylabel(y_label)

        if row_idx == 1:
            ax.set_xlabel("SNR")

        if row_idx == 0 and col_idx == 0:
            legend_handles, legend_labels = handles, labels

# Row titles
fig.text(0.5, 0.965, row_titles[0], ha="center", va="top", fontsize=14)
fig.text(0.5, 0.525, row_titles[1], ha="center", va="top", fontsize=14)

# Shared legend
if legend_handles is not None:
    ncol = 4 if len(legend_labels) >= 4 else len(legend_labels)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, 0.03)
    )

# fig.suptitle(suptitle, y=0.995, fontsize=16)

fig.subplots_adjust(
    left=0.08,
    right=0.99,
    top=0.95,
    bottom=0.16,
    wspace=0.10,
    hspace=0.30
)

os.makedirs(outdir, exist_ok=True)
fig.savefig(outfile, bbox_inches="tight")
plt.show()