# prism/analysis/plot_defense_bar.py
"""Grouped bar chart for L0/L1/L2 defense comparison (replaces Table 2)."""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prism.analysis.loader import (
    DPI,
    FULL_WIDTH,
    aggregate_with_ci,
    load_results_df,
    setup_style,
)

_MODES = [
    ("unconstrained", "L0: Unconstrained"),
    ("explicit", "L1: Explicit"),
    ("full_defense", "L2: Full Defense"),
]

_DYADIC = {"cross_domain", "mediated_comm", "cross_user"}
_MP = {"multi_party_group", "hub_and_spoke", "competitive", "affinity_modulated"}

METRICS = ["leakage_rate", "ias", "tcq", "task_completed"]
METRIC_LABELS = [
    r"Leakage$\downarrow$",
    r"IAS$\uparrow$",
    r"TCQ$\uparrow$",
    r"Task Compl.%$\uparrow$",
]

# Colours matching the reference image style (PowerPoint palette)
COLORS = ["#5B9BD5", "#ED7D31", "#70AD47"]


def _fmt(metric: str, val: float) -> str:
    if metric == "task_completed":
        return f"{val * 100:.1f}"
    return f"{val:.2f}"


def _draw_panel(ax: plt.Axes, sub_df: pd.DataFrame, title: str) -> list:
    """Draw one Dyadic / Multi-Party panel. Returns legend handles."""
    agg = aggregate_with_ci(sub_df, "privacy_mode", METRICS)

    n = len(METRICS)
    x = np.arange(n)
    width = 0.22
    gap = 0.03
    offsets = [-(width + gap), 0, width + gap]

    # Stagger annotation heights: left/right at base, middle raised
    annot_y_offsets = [0.015, 0.05, 0.015]

    # Collect all values first for trend lines
    all_vals: list[np.ndarray] = []

    handles = []
    for mode_idx, ((mode_code, mode_label), off, color) in enumerate(
        zip(_MODES, offsets, COLORS)
    ):
        vals = []
        for m in METRICS:
            v = agg.loc[mode_code, f"{m}_mean"] if mode_code in agg.index else 0.0
            vals.append(v)
        vals = np.array(vals)
        all_vals.append(vals)

        bars = ax.bar(
            x + off, vals, width,
            label=mode_label, color=color,
            edgecolor="white", linewidth=0.5, zorder=3,
        )
        handles.append(bars)

        # Dot markers on top of bars (reference style)
        ax.scatter(
            x + off, vals,
            color=color, s=18, zorder=5,
            edgecolors="white", linewidths=0.4,
        )

        # Semi-transparent trend line (reference style)
        ax.plot(x + off, vals, color=color, linewidth=1.5, alpha=0.25, zorder=2)

        # Value annotations with staggered heights to avoid overlap
        y_off = annot_y_offsets[mode_idx]
        for j, (bar, v) in enumerate(zip(bars, vals)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + y_off,
                _fmt(METRICS[j], v),
                ha="center", va="bottom",
                fontsize=5.5, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=7.5)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    return handles


def generate(df: pd.DataFrame, output_path: str) -> None:
    setup_style()

    dyadic_df = df[df["category"].isin(_DYADIC)]
    mp_df = df[df["category"].isin(_MP)]

    fig, (ax_d, ax_m) = plt.subplots(
        1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.33),
    )

    _draw_panel(ax_d, dyadic_df, "Dyadic (CD, MC, CU)")
    _draw_panel(ax_m, mp_df, "Multi-Party (GC, HS, CM, AM)")

    # Shared legend below panels
    legend_labels = [label for _, label in _MODES]
    fig.legend(
        labels=legend_labels,
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="white")
            for c in COLORS
        ],
        loc="lower center",
        ncol=3,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate defense comparison bar chart (replaces Table 2)."
    )
    parser.add_argument(
        "--results-dir", default="prism/analysis/results",
        help="Directory containing eval_*.json files",
    )
    parser.add_argument(
        "--output", "-o", default="latex/prism/figures/defense_bar.pdf",
        help="Output PDF path",
    )
    args = parser.parse_args()
    df = load_results_df(args.results_dir)
    generate(df, args.output)
    print(f"Wrote {args.output}")
