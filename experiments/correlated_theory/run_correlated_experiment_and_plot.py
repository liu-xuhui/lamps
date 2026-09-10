"""Run correlated-feature LAMPS experiments and create plots.

Examples
--------
Run the default nonlinear experiment and plot probabilities::

    python run_correlated_experiment_and_plot.py

Run the linear experiment and create both probability and Delta outputs::

    python run_correlated_experiment_and_plot.py \
        --model linear --metric both

Recreate a plot from an existing CSV without rerunning ADAM-P::

    python run_correlated_experiment_and_plot.py \
        --model nonlinear --metric probability --from-csv
"""

import argparse
import csv
import os
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/tmp/correlated_experiment_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from adamp import adamp_select
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


DEFAULT_N = 500
DEFAULT_M = 100
DEFAULT_S = 1
DEFAULT_SNR = 1.0
DEFAULT_MAX_ITERS = 10
DEFAULT_CORR_VALUES = (0.5, 0.7, 0.9)
DEFAULT_DATA_SEED = 110
DEFAULT_ADAMP_SEED = 110
DEFAULT_M_RATIO = 0.12
DEFAULT_OUTPUT_DIR = "/results/other/correlated_theory"

METRIC_COLUMNS = {
    "probability": ("prob_signal_X1", "prob_correlated_noise_X2"),
    "delta": ("Delta_signal_X1", "Delta_correlated_noise_X2"),
}


def tree_fit(X, Y, X_predict):
    """Fit the nonlinear base model used by the original experiment."""
    model = DecisionTreeRegressor(random_state=DEFAULT_ADAMP_SEED)
    model.fit(X, Y)
    return model.predict(X_predict)


def linear_fit(X, Y, X_predict):
    """Fit the requested sklearn linear-regression base model."""
    model = LinearRegression()
    model.fit(X, Y)
    return model.predict(X_predict)


def _generate_correlated_features(N, M, seed, corr):
    if M < 2:
        raise ValueError("M must be at least 2 to define correlated X1 and X2.")
    if not -1.0 <= corr <= 1.0:
        raise ValueError("corr must be between -1 and 1.")

    rng = np.random.default_rng(seed)
    covariance = np.eye(M)
    covariance[0, 1] = corr
    covariance[1, 0] = corr
    X = rng.multivariate_normal(np.zeros(M), covariance, size=N)
    noise = rng.normal(scale=1, size=N)
    return X, noise


def _standardize_response(Y, standardize_y):
    if not standardize_y:
        return Y

    scaler = StandardScaler()
    # Flatten after the requested sklearn transformation because ADAM-P uses
    # a one-dimensional response when calculating residuals.
    return scaler.fit_transform(Y.reshape(-1, 1)).ravel()


def simu_simple_nonlinear(
    N,
    M,
    s,
    snr=2,
    seed=110,
    corr=0,
    standardize_y=True,
):
    """Generate the original squared-signal response."""
    if s not in (1, 2):
        raise ValueError("s must be 1 or 2")

    X, noise = _generate_correlated_features(N, M, seed, corr)
    Y = snr * X[:, 0] ** 2 + noise
    if s == 2:
        Y = Y + snr * X[:, 1] ** 2

    return X, _standardize_response(Y, standardize_y)


def simu_simple_linear(
    N,
    M,
    s,
    snr=2,
    seed=110,
    corr=0,
    standardize_y=True,
):
    """Generate the same response as above, with squared terms made linear."""
    if s not in (1, 2):
        raise ValueError("s must be 1 or 2")

    X, noise = _generate_correlated_features(N, M, seed, corr)
    Y = snr * X[:, 0] + noise
    if s == 2:
        Y = Y + snr * X[:, 1]

    return X, _standardize_response(Y, standardize_y)


def run_experiments(args):
    """Run all correlations once and retain probability and Delta paths."""
    rows = []
    simulate = (
        simu_simple_linear if args.model == "linear" else simu_simple_nonlinear
    )
    fit_func = linear_fit if args.model == "linear" else tree_fit

    for corr in args.corr_values:
        print(f"Running {args.model} experiment for corr={corr:g} ...")
        X, Y = simulate(
            N=args.N,
            M=args.M,
            s=args.s,
            snr=args.snr,
            seed=args.data_seed,
            corr=corr,
            standardize_y=args.standardize_y,
        )
        selected, results = adamp_select(
            X,
            Y,
            fit_func=fit_func,
            m_ratio=args.m_ratio,
            max_iter=args.max_iters,
            early_stop=False,
            return_complete_info=True,
            show_progress=not args.no_progress,
            seed=args.adamp_seed,
        )

        selected_text = " ".join(map(str, selected))
        for epoch in range(1, args.max_iters + 1):
            epoch_result = results[epoch - 1]
            rows.append(
                {
                    "model": args.model,
                    "standardize_y": args.standardize_y,
                    "N": args.N,
                    "M": args.M,
                    "s": args.s,
                    "snr": args.snr,
                    "m_ratio": args.m_ratio,
                    "corr": corr,
                    "epoch": epoch,
                    "prob_signal_X1": epoch_result["prob_F"][0],
                    "prob_correlated_noise_X2": epoch_result["prob_F"][1],
                    "Delta_signal_X1": epoch_result["Delta"][0],
                    "Delta_correlated_noise_X2": epoch_result["Delta"][1],
                    "selected_features": selected_text,
                }
            )

    return rows


def output_paths(output_dir, model, M, metric):
    stem = f"{model}_M{M}_{metric}_vs_epoch"
    return output_dir / f"{stem}.csv", output_dir / f"{stem}.png"


def save_rows(rows, csv_path, metric):
    """Save one metric per CSV so its filename and contents agree."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    metric_columns = METRIC_COLUMNS[metric]
    fieldnames = [
        "model",
        "standardize_y",
        "N",
        "M",
        "s",
        "snr",
        "m_ratio",
        "corr",
        "epoch",
        *metric_columns,
        "selected_features",
    ]

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_bool(value):
    return value.strip().lower() in {"true", "1", "yes"}


def load_rows(csv_path, metric):
    rows = []
    required_metric_columns = METRIC_COLUMNS[metric]

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = set(required_metric_columns) - set(reader.fieldnames or [])
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{csv_path} is missing columns: {missing_text}")

        for row in reader:
            parsed = {
                **row,
                "standardize_y": _parse_bool(row["standardize_y"]),
                "N": int(row["N"]),
                "M": int(row["M"]),
                "s": int(row["s"]),
                "snr": float(row["snr"]),
                "m_ratio": float(row["m_ratio"]),
                "corr": float(row["corr"]),
                "epoch": int(row["epoch"]),
            }
            for column in required_metric_columns:
                parsed[column] = float(row[column])
            rows.append(parsed)

    return rows


def _delta_limits(rows, signal_column, noise_column):
    values = np.asarray(
        [
            0.0,
            *(row[signal_column] for row in rows),
            *(row[noise_column] for row in rows),
        ],
        dtype=float,
    )
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return -0.05, 0.05

    lower = min(0.0, float(finite_values.min()))
    upper = max(0.0, float(finite_values.max()))
    span = upper - lower
    padding = 0.08 * span if span > 0 else max(abs(upper), 1.0) * 0.08
    return lower - padding, upper + padding


def plot_rows(rows, figure_path, metric, m_ratio, show=False):
    """Create the compact publication-style plot from the first script."""
    if not rows:
        raise ValueError("No rows are available to plot.")

    corr_values = sorted({row["corr"] for row in rows})
    signal_column, noise_column = METRIC_COLUMNS[metric]
    y_label = "Probability" if metric == "probability" else r"$\Delta$"
    initial_value = m_ratio if metric == "probability" else 0.0
    max_epoch = max(row["epoch"] for row in rows)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        1,
        len(corr_values),
        figsize=(2.5 * len(corr_values) + 0.9, 2.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, corr in zip(axes, corr_values):
        corr_rows = sorted(
            (row for row in rows if row["corr"] == corr),
            key=lambda row: row["epoch"],
        )
        epochs = [0, *(row["epoch"] for row in corr_rows)]
        signal_values = [
            initial_value,
            *(row[signal_column] for row in corr_rows),
        ]
        noise_values = [
            initial_value,
            *(row[noise_column] for row in corr_rows),
        ]

        ax.plot(
            epochs,
            signal_values,
            color="red",
            marker="o",
            linewidth=1.6,
            markersize=4.5,
            label=r"$X_1$ signal",
        )
        ax.plot(
            epochs,
            noise_values,
            color="blue",
            marker="o",
            linewidth=1.6,
            markersize=4.5,
            label=r"$X_2$ correlated noise",
        )
        ax.set_title(rf"$\rho = {corr:g}$", pad=4)
        ax.set_xlim(-0.3, max_epoch + 0.3)
        ax.set_xticks(range(max_epoch + 1))
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.tick_params(direction="out", length=4, width=0.8)

    if metric == "probability":
        axes[0].set_ylim(0.0, 0.85)
        axes[0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    else:
        axes[0].set_ylim(*_delta_limits(rows, signal_column, noise_column))

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.79, 0.5),
        frameon=True,
        borderpad=0.35,
        handlelength=2.2,
    )
    fig.text(0.43, 0.03, "Iterations", ha="center", va="center", fontsize=11)
    fig.text(
        0.02,
        0.5,
        y_label,
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=11,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.78,
        bottom=0.22,
        top=0.87,
        wspace=0.18,
    )

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=400, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run a correlated linear/nonlinear ADAM-P experiment, save its "
            "CSV data, and create a publication-style probability or Delta plot."
        )
    )
    parser.add_argument(
        "--model",
        choices=("nonlinear", "linear"),
        default="nonlinear",
        help="Data-generating process and base model (default: nonlinear).",
    )
    parser.add_argument(
        "--metric",
        "--plot",
        dest="metric",
        type=str.lower,
        choices=("probability", "delta", "both"),
        default="probability",
        help="Output metric to save and plot (default: probability).",
    )
    parser.add_argument("--N", type=int, default=DEFAULT_N, help="Sample size.")
    parser.add_argument("--M", type=int, default=DEFAULT_M, help="Feature count.")
    parser.add_argument(
        "--s",
        type=int,
        choices=(1, 2),
        default=DEFAULT_S,
        help="Number of signal variables (default: 1).",
    )
    parser.add_argument("--snr", type=float, default=DEFAULT_SNR)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument(
        "--corr-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_CORR_VALUES),
        metavar="RHO",
    )
    parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument("--adamp-seed", type=int, default=DEFAULT_ADAMP_SEED)
    parser.add_argument(
        "--m-ratio",
        type=float,
        default=DEFAULT_M_RATIO,
        help=(
            "ADAM-P feature-sampling ratio and probability value at epoch 0 "
            "(default: 0.12)."
        ),
    )
    parser.add_argument(
        "--standardize-y",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize Y inside the simulation function (default: enabled).",
    )
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Recreate output plot(s) from the named CSV file(s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated CSV and PNG files.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each figure after saving it.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Hide ADAM-P progress bars.",
    )
    args = parser.parse_args(argv)

    if args.N < 2:
        parser.error("--N must be at least 2")
    if args.M < 2:
        parser.error("--M must be at least 2")
    if args.max_iters < 1:
        parser.error("--max-iters must be at least 1")
    if not 0 < args.m_ratio <= 1:
        parser.error("--m-ratio must be in (0, 1]")
    if any(not -1 <= corr <= 1 for corr in args.corr_values):
        parser.error("all --corr-values must be between -1 and 1")

    return args


def main(argv=None):
    args = parse_args(argv)
    metrics = (
        ("probability", "delta") if args.metric == "both" else (args.metric,)
    )

    rows = None
    if not args.from_csv:
        rows = run_experiments(args)

    for metric in metrics:
        csv_path, figure_path = output_paths(
            args.output_dir, args.model, args.M, metric
        )
        if args.from_csv:
            metric_rows = load_rows(csv_path, metric)
        else:
            metric_rows = rows
            save_rows(metric_rows, csv_path, metric)
            print(f"Saved {metric} data to: {csv_path}")

        plot_rows(
            metric_rows,
            figure_path,
            metric=metric,
            m_ratio=float(metric_rows[0]["m_ratio"]),
            show=args.show,
        )
        print(f"Saved {metric} figure to: {figure_path}")


if __name__ == "__main__":
    main()
