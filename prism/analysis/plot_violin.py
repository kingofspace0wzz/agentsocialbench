# prism/analysis/plot_violin.py
"""Violin plots of leakage distribution per model, faceted by category (Appendix)."""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from prism.analysis.loader import (
    CATEGORY_DISPLAY,
    DPI,
    FULL_WIDTH,
    MODEL_ORDER,
    display_name,
    model_color,
    load_results_df,
    setup_style,
)

_PANEL_CATEGORIES = ["cross_domain", "cross_user", "mediated_comm"]


def generate(df: pd.DataFrame, output_path: str) -> None:
    """Generate violin leakage-distribution plots and save to *output_path*."""
    setup_style()

    # Determine model order
    all_models_in_data = set(df["model"].unique())
    ordered_models = [m for m in MODEL_ORDER if m in all_models_in_data]
    ordered_models += sorted(m for m in all_models_in_data if m not in MODEL_ORDER)

    categories_present = [c for c in _PANEL_CATEGORIES if c in df["category"].unique()]
    if not categories_present:
        categories_present = sorted(df["category"].unique())
    n_panels = len(categories_present)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(FULL_WIDTH, FULL_WIDTH * 0.4),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    # Prepare a display-name column for nicer x labels
    df_plot = df.copy()
    df_plot["model_label"] = df_plot["model"].map(display_name)

    # Build ordered label list
    ordered_labels = [display_name(m) for m in ordered_models
                      if display_name(m) in df_plot["model_label"].unique()]

    for col, category in enumerate(categories_present):
        ax = axes[col]
        cat_df = df_plot[df_plot["category"] == category]
        cat_display = CATEGORY_DISPLAY.get(category, category)

        if cat_df.empty or cat_df["leakage_rate"].isna().all():
            ax.set_title(cat_display, fontsize=9)
            ax.set_visible(False)
            continue

        palette = {display_name(m): model_color(m) for m in ordered_models}
        sns.violinplot(
            data=cat_df,
            x="model_label",
            y="leakage_rate",
            order=[lbl for lbl in ordered_labels if lbl in cat_df["model_label"].unique()],
            hue="model_label",
            palette=palette,
            inner="box",
            cut=0,
            alpha=0.5,
            ax=ax,
            linewidth=0.8,
            legend=False,
        )

        ax.set_title(cat_display, fontsize=9)
        if col == 0:
            ax.set_ylabel("Leakage Rate", fontsize=8)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30, labelsize=6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate violin leakage-distribution plots.")
    parser.add_argument("--results-dir", default="prism/analysis/results",
                        help="Directory containing eval_*.json files")
    parser.add_argument("--output", "-o", default="prism/analysis/plots/violin.pdf",
                        help="Output image file path")
    args = parser.parse_args()

    df = load_results_df(args.results_dir)
    generate(df, args.output)
    print(f"Wrote {args.output}")
