# prism/analysis/plot_category.py
"""Category comparison: split into Dyadic and Multi-Party subfigures."""
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
    MIN_SAMPLE_THRESHOLD,
    MODEL_ORDER,
    compute_ci,
    display_name,
    model_color,
    load_results_df,
    setup_style,
)

# (metric, y-label, ylim_lo, ylim_hi)
_METRICS = [
    ("leakage_rate", "Leakage Rate", 0.0, 0.6),
    ("ias", "Info. Abstraction Score", 0.4, 1.0),
    ("tcq", "Task Completion Quality", 0.4, 1.0),
]

_DYADIC_CATEGORIES = ["cross_domain", "cross_user", "mediated_comm"]
_MP_CATEGORIES_LIST = [
    "multi_party_group", "hub_and_spoke", "competitive", "affinity_modulated",
]
_MP_CATEGORIES_SET = set(_MP_CATEGORIES_LIST)


def _draw_figure(
    df: pd.DataFrame,
    output_path: str,
    categories: list[str],
    title: str,
) -> None:
    """Generate a 3-row (metric) × N-category grouped bar chart."""
    setup_style()

    all_models_in_data = set(df["model"].unique())
    ordered_models = [m for m in MODEL_ORDER if m in all_models_in_data]
    ordered_models += sorted(m for m in all_models_in_data if m not in MODEL_ORDER)

    categories_present = [c for c in categories if c in df["category"].unique()]
    if not categories_present:
        return

    n_cats = len(categories_present)
    x = np.arange(n_cats)
    group_labels = [CATEGORY_DISPLAY.get(c, c) for c in categories_present]

    fig, axes = plt.subplots(
        len(_METRICS), 1,
        figsize=(FULL_WIDTH, FULL_WIDTH * 0.72),
        sharex=True,
    )

    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

    for ax_idx, (metric, ylabel, ylim_lo, ylim_hi) in enumerate(_METRICS):
        ax = axes[ax_idx]

        for cat_idx, category in enumerate(categories_present):
            cat_df = df[df["category"] == category]

            if category in _MP_CATEGORIES_SET:
                cat_models = [
                    m for m in ordered_models
                    if len(cat_df[cat_df["model"] == m]) >= MIN_SAMPLE_THRESHOLD
                ]
            else:
                cat_models = [
                    m for m in ordered_models
                    if not cat_df[cat_df["model"] == m].empty
                ]

            n_cat_models = len(cat_models)
            cat_bar_width = 0.8 / max(n_cat_models, 1)

            for m_idx, model in enumerate(cat_models):
                color = model_color(model)
                series = cat_df[cat_df["model"] == model][metric]
                m_val, lo, hi = compute_ci(series)
                err_lo = m_val - lo
                err_hi = hi - m_val

                means_arr = np.array([m_val], dtype=float)
                errs_arr = np.nan_to_num(
                    np.array([[err_lo], [err_hi]], dtype=float), nan=0.0
                )
                offset = (m_idx - (n_cat_models - 1) / 2.0) * cat_bar_width

                label = (
                    display_name(model)
                    if ax_idx == 0 and cat_idx == 0
                    else None
                )

                bars = ax.bar(
                    np.array([x[cat_idx]]) + offset, means_arr, cat_bar_width,
                    yerr=errs_arr, capsize=2,
                    color=color,
                    label=label,
                    zorder=3,
                )

                for bar, mean_val in zip(bars, means_arr):
                    if not np.isnan(mean_val):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            bar.get_height() + 0.008,
                            f"{mean_val:.2f}",
                            ha="center", va="bottom", fontsize=5.5, rotation=90,
                        )

        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_ylim(ylim_lo, ylim_hi + 0.08)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

        if ax_idx == 0:
            ax.legend(fontsize=6.5, ncol=4, loc="upper right", framealpha=0.9)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(group_labels, fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def generate(df: pd.DataFrame, output_path: str) -> None:
    """Generate split Dyadic and Multi-Party category comparison figures."""
    df = df[df["privacy_mode"] == "unconstrained"].copy()

    # Dyadic figure
    dyadic_path = output_path.replace(".pdf", "_dyadic.pdf")
    _draw_figure(df, dyadic_path, _DYADIC_CATEGORIES, "Dyadic Categories")
    print(f"Wrote {dyadic_path}")

    # Multi-party figure
    mp_path = output_path.replace(".pdf", "_mp.pdf")
    _draw_figure(df, mp_path, _MP_CATEGORIES_LIST, "Multi-Party Categories")
    print(f"Wrote {mp_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate category comparison plots.")
    parser.add_argument("--results-dir", default="prism/analysis/results",
                        help="Directory containing eval_*.json files")
    parser.add_argument("--output", "-o", default="latex/prism/figures/category_comparison.pdf",
                        help="Output image file path (base name)")
    args = parser.parse_args()

    df = load_results_df(args.results_dir)
    generate(df, args.output)
