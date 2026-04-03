# prism/analysis/plot_domain_heatmap.py
"""Domain-pair heatmap — cross-domain leakage per (domain_pair × model) (Figure 2)."""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from prism.analysis.loader import (
    DPI,
    MODEL_ORDER,
    SINGLE_COL_WIDTH,
    display_name,
    load_results_df,
    setup_style,
)


def generate(df: pd.DataFrame, output_path: str) -> None:
    """Generate domain-pair leakage heatmap and save to *output_path*."""
    setup_style()

    # Filter to cross_domain with known domain pairs
    cd_df = df[(df["category"] == "cross_domain") & df["domain_pair"].notna()].copy()

    if cd_df.empty:
        fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * 1.4))
        ax.text(0.5, 0.5, "No cross-domain data", ha="center", va="center",
                transform=ax.transAxes)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        return

    # Build pivot: rows = domain_pair, columns = model display name, values = mean leakage * 100
    models_in_data = set(cd_df["model"].unique())
    ordered_models = [m for m in MODEL_ORDER if m in models_in_data]
    ordered_models += sorted(m for m in models_in_data if m not in MODEL_ORDER)

    # Map model IDs to display names for pivot columns
    cd_df = cd_df.copy()
    cd_df["model_label"] = cd_df["model"].map(display_name)

    pivot = cd_df.groupby(["domain_pair", "model_label"])["leakage_rate"].mean().unstack()

    # Ordered columns by MODEL_ORDER
    ordered_col_labels = [display_name(m) for m in ordered_models
                          if display_name(m) in pivot.columns]
    remaining_cols = [c for c in pivot.columns if c not in ordered_col_labels]
    pivot = pivot[ordered_col_labels + remaining_cols]

    # Multiply by 100 for percentage display
    pivot_pct = pivot * 100.0

    # Sort rows by row mean (most vulnerable at top)
    row_means = pivot_pct.mean(axis=1)
    pivot_pct = pivot_pct.loc[row_means.sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * 1.4))

    sns.heatmap(
        pivot_pct,
        annot=True,
        fmt=".0f",
        cmap="Reds",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Leakage %", "shrink": 0.8},
    )

    ax.set_title("Cross-Domain Leakage (%)", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Domain Pair")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cross-domain heatmap (Fig. 2).")
    parser.add_argument("--results-dir", default="prism/analysis/results",
                        help="Directory containing eval_*.json files")
    parser.add_argument("--output", "-o", default="prism/analysis/plots/domain_heatmap.pdf",
                        help="Output image file path")
    args = parser.parse_args()

    df = load_results_df(args.results_dir)
    generate(df, args.output)
    print(f"Wrote {args.output}")
