import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# This file is just temporary code for plot linear experiment result.
# TODO: change linear experiment setting to align with the nonlinear setting
# TODO: run linear correlated 1 and correlated 2 setting and save oracle and non-oracle result
# TODO: save selections but not just F1 score so that Precision and recall can be plot

folder_path = "results/linear"

# Dictionary to store results
results_dict = {}

# Loop through all pkl files
for filename in os.listdir(folder_path):
    if filename.endswith("_result_dict.pkl"):
        short_key = filename.replace("_result_dict.pkl", "")
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "rb") as f:
            results_dict[short_key] = pickle.load(f)

with open("results/linear/result_dict_lowsnr.pkl", "rb") as f:
    lowsnr_dict = pickle.load(f)


results_dict["highdim_00"][0.05] = lowsnr_dict["highdim0"]
results_dict["highdim_05"][0.05] = lowsnr_dict["highdim5"]
results_dict["highdim_09"][0.05] = lowsnr_dict["highdim9"]
results_dict["middim_00"][0.05] = lowsnr_dict["middim0"]
results_dict["middim_05"][0.05] = lowsnr_dict["middim5"]
results_dict["middim_09"][0.05] = lowsnr_dict["middim9"]
results_dict["lowdim_00"][0.05] = lowsnr_dict["lowdim0"]
results_dict["lowdim_05"][0.05] = lowsnr_dict["lowdim5"]
results_dict["lowdim_09"][0.05] = lowsnr_dict["lowdim9"]


palette = sns.color_palette("colorblind", 7)

# Assign colors explicitly (AdaMP = red)
colors = {
    'AdaMP': 'red',
    'Stability Selection': palette[0],
    'Lasso (eBIC)': palette[1],
    'Lasso (CV)': palette[3],
    'Lasso (Oracle)': palette[4],
    'CPSS': palette[5],
    'Elastic Net (Oracle)': palette[6]
}

markers = {
    'AdaMP': 'o',
    'Stability Selection': 's',
    'Lasso (eBIC)': 'D',
    'Lasso (CV)': '^',
    'Lasso (Oracle)': 'p',
    'CPSS': '*',
    'Elastic Net (Oracle)': 'x'
}

linestyles = {
    'AdaMP': '-',
    'Stability Selection': '--',
    'Lasso (eBIC)': '-.',
    'Lasso (CV)': ':',
    'Lasso (Oracle)': '--',
    'CPSS': '-.',
    'Elastic Net (Oracle)': ':'
}

resdictkeys = [
    'highdim_00', 'highdim_05', 'highdim_09',
    'middim_00', 'middim_05', 'middim_09',
    'lowdim_00', 'lowdim_05', 'lowdim_09'
]

title_map = {
    '00': "Independent (ρ=0)",
    '05': "ρ=0.5",
    '09': "ρ=0.9"
}

row_map = {
    'highdim': "High-dim",
    'middim': "Mid-dim",
    'lowdim': "Low-dim"
}

fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)

for idx, key in enumerate(resdictkeys):
    ax = axes[idx // 3, idx % 3]

    result_dict = results_dict[key]
    # sorted_snrs = sorted(result_dict.keys())
    sorted_snrs = [0.05, 0.1, 0.3, 1, 3]

    f1_data = {
        'AdaMP': [result_dict[snr]['adam_f1'] for snr in sorted_snrs],
        'Stability Selection': [result_dict[snr]['las_f1'] for snr in sorted_snrs],
        'Lasso (eBIC)': [result_dict[snr]['ebic_f1'] for snr in sorted_snrs],
        'Lasso (CV)': [result_dict[snr]['lassocv_f1'] for snr in sorted_snrs],
        'Lasso (Oracle)': [result_dict[snr]['lassoor_f1'] for snr in sorted_snrs],
        'CPSS': [result_dict[snr]['cpss_f1'] for snr in sorted_snrs],
        'Elastic Net (Oracle)': [result_dict[snr]['elastic_f1'] for snr in sorted_snrs]
    }

    for method, values in f1_data.items():
        means = np.array([np.mean(v) for v in values])
        stds = np.array([np.std(v) for v in values])
        se = stds / np.sqrt([len(v) for v in values])

        ax.errorbar(sorted_snrs, means, yerr=se,
            marker=markers[method],
            color=colors[method],
            linestyle=linestyles[method],
            linewidth=3 if method == 'AdaMP' else 1.8,
            markersize=6 if method == 'AdaMP' else 5,
            capsize=3,   # small horizontal bar at ends
            elinewidth=1.2,
            label=method)

    dim, corr = key.split("_")
    ax.set_title(f"{row_map[dim]} - {title_map[corr]}", fontsize=13)
    ax.set_xscale("log")

    # Adjusted y-axis
    ax.set_ylim(0.1, 1.05)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

# Shared labels
fig.text(0.5, 0.04, 'SNR', ha='center', fontsize=15)
fig.text(0.04, 0.5, 'F1 Score', va='center', rotation='vertical', fontsize=15)

# Legend
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=12, frameon=False)

# Global title
fig.suptitle("Feature Selection F1 Scores Across Dimensionality and Correlation Settings",
             fontsize=18, y=0.995)

plt.tight_layout(rect=[0, 0.08, 1, 0.96])
plt.savefig("results/linear/temp_f1_comparison.pdf", bbox_inches="tight")
plt.show()

