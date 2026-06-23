"""Generate all experimental figures referenced by the manuscript.

The numerical values below mirror the corresponding tables in
paper/manuscript/paper.tex. Matplotlib's PDF backend keeps lines, markers,
text, and bars as vector objects so the figures remain sharp when enlarged.
"""

import os
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = PAPER_DIR / "figures"
MPL_CACHE_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "ikgc-matplotlib-cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

DATASET_COLORS = ["#0072B2", "#D55E00", "#009E73"]
VARIANT_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
MARKERS = ["o", "s", "^"]
MODEL_COLORS = ["#999999", "#56B4E9", "#E69F00", "#D55E00", "#0072B2"]
MODEL_MARKERS = ["o", "s", "^", "D", "P"]


def configure_style() -> None:
    """Apply a compact, color-blind-friendly style suitable for print."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.8,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def finish_axis(ax: plt.Axes) -> None:
    """Use restrained axes and horizontal guides."""
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.19,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        va="top",
    )


def save_pdf(fig: plt.Figure, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES_DIR / filename,
        format="pdf",
        metadata={
            "Title": filename.removesuffix(".pdf").replace("_", " "),
            "Creator": "Matplotlib",
        },
    )
    plt.close(fig)


def plot_ablation() -> None:
    """Plot average ablation results from Tables 3 and 4."""
    datasets = ["WN18RR-ind", "FB15k237-ind", "NELL995-ind"]
    variants = ["NSIK (full)", "w/o gating", "w/o diffusion", "w/o source enh."]

    hits = np.array(
        [
            [94.86, 95.97, 95.18],
            [88.75, 90.28, 90.25],
            [87.50, 88.88, 88.68],
            [89.40, 91.33, 91.63],
        ]
    )
    auc_pr = np.array(
        [
            [98.45, 97.51, 97.23],
            [95.15, 94.05, 93.73],
            [94.50, 93.33, 93.03],
            [94.83, 93.70, 93.48],
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75))
    x = np.arange(len(datasets))
    width = 0.18
    offsets = (np.arange(len(variants)) - 1.5) * width

    panels = [
        (axes[0], hits, "Hits@10", (86.5, 97.0), np.arange(88, 98, 2)),
        (axes[1], auc_pr, "AUC-PR", (92.0, 99.2), np.arange(93, 100, 1)),
    ]

    for panel_index, (ax, values, metric, limits, ticks) in enumerate(panels):
        for variant_index, (variant, color) in enumerate(
            zip(variants, VARIANT_COLORS, strict=True)
        ):
            ax.bar(
                x + offsets[variant_index],
                values[variant_index],
                width=width,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                label=variant,
                zorder=3,
            )

        ax.set_xticks(x, datasets)
        ax.set_ylabel(f"{metric} (%)")
        ax.set_ylim(*limits)
        ax.set_yticks(ticks)
        ax.set_title(f"Average {metric} across variants")
        finish_axis(ax)
        add_panel_label(ax, f"({chr(ord('a') + panel_index)})")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=4,
        columnspacing=1.2,
        handlelength=1.5,
    )
    fig.subplots_adjust(left=0.08, right=0.995, top=0.84, bottom=0.25, wspace=0.28)
    save_pdf(fig, "experiment_ablation.pdf")


def plot_role_sensitivity() -> None:
    """Plot structural-role sensitivity results from Table 5."""
    roles = np.array([4, 8, 12, 16])
    datasets = ["WN18RR-ind", "FB15k237-ind", "NELL995-ind"]

    hits = np.array(
        [
            [91.20, 94.86, 92.55, 92.10],
            [93.05, 95.97, 94.30, 93.95],
            [93.40, 95.18, 94.90, 94.35],
        ]
    )
    auc_pr = np.array(
        [
            [96.20, 98.45, 97.40, 96.90],
            [94.90, 97.51, 96.10, 95.70],
            [95.10, 97.23, 96.00, 95.60],
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75))
    panels = [
        (axes[0], hits, "Hits@10", (90.5, 96.5), np.arange(91, 97, 1)),
        (axes[1], auc_pr, "AUC-PR", (94.3, 99.0), np.arange(95, 100, 1)),
    ]

    for panel_index, (ax, values, metric, limits, ticks) in enumerate(panels):
        for dataset, color, marker, series in zip(
            datasets, DATASET_COLORS, MARKERS, values, strict=True
        ):
            ax.plot(
                roles,
                series,
                color=color,
                marker=marker,
                markerfacecolor="white",
                markeredgewidth=1.1,
                label=dataset,
                zorder=3,
            )

        ax.axvline(8, color="#666666", linewidth=0.8, linestyle=":", zorder=1)
        ax.set_xticks(roles)
        ax.set_xlabel("Number of structural roles")
        ax.set_ylabel(f"{metric} (%)")
        ax.set_ylim(*limits)
        ax.set_yticks(ticks)
        ax.set_title(f"{metric} sensitivity")
        finish_axis(ax)
        add_panel_label(ax, f"({chr(ord('a') + panel_index)})")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=3,
        columnspacing=1.5,
        handlelength=2.1,
    )
    fig.subplots_adjust(left=0.08, right=0.995, top=0.84, bottom=0.25, wspace=0.28)
    save_pdf(fig, "experiment_role_sensitivity.pdf")


def plot_additional_hyperparameters() -> None:
    """Plot diffusion-step and Top-K source sensitivity."""
    steps = np.array([1, 3, 5, 7, 9])
    step_hits = np.array([93.84, 94.76, 95.34, 94.91, 94.37])
    step_auc = np.array([96.92, 97.41, 97.73, 97.48, 97.12])

    sources = np.array([50, 100, 150, 200])
    source_hits = np.array([94.62, 95.34, 94.88, 94.31])
    source_auc = np.array([97.29, 97.73, 97.46, 97.08])

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75))
    panels = [
        (
            axes[0],
            steps,
            step_hits,
            step_auc,
            "Number of diffusion steps",
            "Diffusion-step sensitivity",
            5,
        ),
        (
            axes[1],
            sources,
            source_hits,
            source_auc,
            "Number of diffusion sources",
            "Top-K source sensitivity",
            100,
        ),
    ]

    for panel_index, (ax, x, hits, auc_pr, xlabel, title, optimum) in enumerate(
        panels
    ):
        ax.plot(
            x,
            hits,
            color=DATASET_COLORS[0],
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.1,
            label="Hits@10",
            zorder=3,
        )
        ax.plot(
            x,
            auc_pr,
            color=DATASET_COLORS[1],
            marker="s",
            markerfacecolor="white",
            markeredgewidth=1.1,
            label="AUC-PR",
            zorder=3,
        )
        ax.axvline(optimum, color="#666666", linewidth=0.8, linestyle=":", zorder=1)
        ax.set_xticks(x)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Average performance (%)")
        ax.set_ylim(93.4, 98.1)
        ax.set_yticks(np.arange(94, 99, 1))
        ax.set_title(title)
        finish_axis(ax)
        add_panel_label(ax, f"({chr(ord('a') + panel_index)})")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=2,
        columnspacing=1.8,
        handlelength=2.1,
    )
    fig.subplots_adjust(left=0.08, right=0.995, top=0.84, bottom=0.25, wspace=0.28)
    save_pdf(fig, "experiment_hyperparameters.pdf")


def plot_noise_robustness() -> None:
    """Plot noise-injection robustness results for all compared models."""
    noise = np.array([0, 5, 10, 20])
    models = ["GraIL", "CoMPILE", "QAAR", "GLAR", "NSIK"]
    hits = np.array(
        [
            [73.24, 70.10, 66.85, 61.40],
            [74.90, 72.30, 69.10, 63.80],
            [88.25, 85.40, 82.60, 77.20],
            [93.55, 91.20, 88.75, 83.10],
            [94.86, 91.35, 89.90, 85.70],
        ]
    )
    auc_pr = np.array(
        [
            [91.75, 89.60, 86.30, 81.20],
            [97.79, 95.20, 92.40, 88.10],
            [95.66, 93.10, 90.80, 86.50],
            [98.42, 96.85, 94.70, 90.10],
            [98.45, 96.40, 95.10, 91.80],
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.9))
    panels = [
        (axes[0], hits, "Hits@10", (59, 97), np.arange(60, 101, 10)),
        (axes[1], auc_pr, "AUC-PR", (79, 100), np.arange(80, 101, 5)),
    ]

    for panel_index, (ax, values, metric, limits, ticks) in enumerate(panels):
        for model, color, marker, series in zip(
            models, MODEL_COLORS, MODEL_MARKERS, values, strict=True
        ):
            is_nsik = model == "NSIK"
            ax.plot(
                noise,
                series,
                color=color,
                marker=marker,
                markerfacecolor=color if is_nsik else "white",
                markeredgewidth=1.1,
                linewidth=2.1 if is_nsik else 1.35,
                label=model,
                zorder=4 if is_nsik else 3,
            )
        ax.set_xticks(noise)
        ax.set_xlabel("Injected-noise ratio (%)")
        ax.set_ylabel(f"{metric} (%)")
        ax.set_ylim(*limits)
        ax.set_yticks(ticks)
        ax.set_title(f"{metric} under structural noise")
        finish_axis(ax)
        add_panel_label(ax, f"({chr(ord('a') + panel_index)})")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=5,
        columnspacing=1.05,
        handlelength=1.8,
    )
    fig.subplots_adjust(left=0.08, right=0.995, top=0.84, bottom=0.25, wspace=0.28)
    save_pdf(fig, "experiment_noise_robustness.pdf")


def main() -> None:
    configure_style()
    plot_ablation()
    plot_role_sensitivity()
    plot_additional_hyperparameters()
    plot_noise_robustness()
    print(f"Generated experimental figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
