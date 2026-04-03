# prism/analysis/plot_severity.py
"""Stacked bar chart of leakage severity (none / partial / full) (Figure 4)."""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prism.analysis.loader import (
    CATEGORY_DISPLAY,
    DPI,
    FULL_WIDTH,
    MODEL_ORDER,
    display_name,
    load_results_df,
    setup_style,
)

_SEVERITY_COLORS = {
    "none": "#2ecc71",
    "partial": "#f39c12",
    "full": "#e74c3c",
}

_PANEL_CATEGORIES = ["cross_domain", "cross_user", "mediated_comm"]


def generate(df: pd.DataFrame, output_path: str) -> None:
    """Generate stacked severity bar chart and save to *output_path*."""
    setup_style()

    # Determine model order
    all_models_in_data = set(df["model"].unique())
    ordered_models = [m for m in MODEL_ORDER if m in all_models_in_data]
    ordered_models += sorted(m for m in all_models_in_data if m not in MODEL_ORDER)

    # Categories actually present in data
    categories_present = [c for c in _PANEL_CATEGORIES if c in df["category"].unique()]
    if not categories_present:
        categories_present = sorted(df["category"].unique())
    n_panels = len(categories_present)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(FULL_WIDTH, FULL_WIDTH * 0.35),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    for col, category in enumerate(categories_present):
        ax = axes[col]
        cat_df = df[df["category"] == category]
        cat_display = CATEGORY_DISPLAY.get(category, category)

        # Filter models present in this category
        models_here = [m for m in ordered_models if m in cat_df["model"].unique()]
        if not models_here:
            ax.set_title(cat_display, fontsize=9)
            ax.set_visible(False)
            continue

        x = np.arange(len(models_here))
        bar_labels = [display_name(m) for m in models_here]

        none_pcts: list[float] = []
        partial_pcts: list[float] = []
        full_pcts: list[float] = []

        for model in models_here:
            mdf = cat_df[cat_df["model"] == model]
            total = mdf["n_items"].sum()
            if total == 0:
                none_pcts.append(0.0)
                partial_pcts.append(0.0)
                full_pcts.append(0.0)
            else:
                none_pcts.append(mdf["n_none"].sum() / total * 100.0)
                partial_pcts.append(mdf["n_partial"].sum() / total * 100.0)
                full_pcts.append(mdf["n_full"].sum() / total * 100.0)

        none_arr = np.array(none_pcts)
        partial_arr = np.array(partial_pcts)
        full_arr = np.array(full_pcts)

        bar_width = 0.6

        b_none = ax.bar(x, none_arr, bar_width, label="None" if col == 0 else None,
                        color=_SEVERITY_COLORS["none"])
        b_partial = ax.bar(x, partial_arr, bar_width, bottom=none_arr,
                           label="Partial" if col == 0 else None,
                           color=_SEVERITY_COLORS["partial"])
        b_full = ax.bar(x, full_arr, bar_width, bottom=none_arr + partial_arr,
                        label="Full" if col == 0 else None,
                        color=_SEVERITY_COLORS["full"])

        # Value annotations on each segment
        for i, (n_pct, p_pct, f_pct) in enumerate(zip(none_arr, partial_arr, full_arr)):
            if n_pct >= 5:
                ax.text(i, n_pct / 2, f"{n_pct:.0f}", ha="center", va="center",
                        fontsize=5, color="white", fontweight="bold")
            if p_pct >= 5:
                ax.text(i, n_pct + p_pct / 2, f"{p_pct:.0f}", ha="center", va="center",
                        fontsize=5, color="white", fontweight="bold")
            if f_pct >= 5:
                ax.text(i, n_pct + p_pct + f_pct / 2, f"{f_pct:.0f}", ha="center",
                        va="center", fontsize=5, color="white", fontweight="bold")

        ax.set_title(cat_display, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, rotation=30, ha="right", fontsize=6)
        ax.set_ylim(0, 105)

        if col == 0:
            ax.set_ylabel("Leakage Severity (%)", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate severity stacked bar chart (Fig. 4).")
    parser.add_argument("--results-dir", default="prism/analysis/results",
                        help="Directory containing eval_*.json files")
    parser.add_argument("--output", "-o", default="prism/analysis/plots/severity.pdf",
                        help="Output image file path")
    args = parser.parse_args()

    df = load_results_df(args.results_dir)
    generate(df, args.output)
    print(f"Wrote {args.output}")
