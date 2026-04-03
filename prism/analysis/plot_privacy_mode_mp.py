# prism/analysis/plot_privacy_mode_mp.py
"""Multi-party privacy mode effect figure: per-category line plots across L0→L2→L4.

Shows 3 metrics (rows) × 3 multi-party categories (columns) = 9 panels.
Each panel has one line per model connecting privacy levels with CI error bars.
Intended as an appendix figure mirroring plot_privacy_mode.py for GC/CM/AM.
"""
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
    MIN_SAMPLE_THRESHOLD,
    MODEL_ORDER,
    compute_ci,
    display_name,
    load_results_df,
    model_color,
    model_marker,
    setup_style,
)

# Metrics: (column, display_label, lower_is_better)
_METRICS = [
    ("leakage_rate", "Leakage Rate", True),
    ("ias", "IAS", False),
    ("task_completed", "Task Compl. %", False),
]

# Multi-party categories shown as columns
_CATEGORIES = [
    ("multi_party_group", "Group Chat"),
    ("hub_and_spoke", "Hub-and-Spoke"),
    ("competitive", "Competitive"),
    ("affinity_modulated", "Affinity-Mod."),
]

# Privacy modes in order (L1/implicit and L3/enhanced omitted for clarity)
_TRAJECTORY_MODES = ["unconstrained", "explicit", "full_defense"]

_MODE_LABELS: dict[str, str] = {
    "unconstrained": "L0\nUnconstr.",
    "explicit": "L1\nExplicit",
    "full_defense": "L2\nFull Def.",
}


def generate(df: pd.DataFrame, output_path: str) -> None:
    """Generate 3×3 panel figure for multi-party categories and save to *output_path*."""
    setup_style()

    n_rows = len(_METRICS)
    n_cols = len(_CATEGORIES)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FULL_WIDTH, FULL_WIDTH * 0.55),
        sharey=False,
    )

    modes_in_data = [m for m in _TRAJECTORY_MODES if m in df["privacy_mode"].unique()]
    mode_label_list = [_MODE_LABELS.get(m, m) for m in modes_in_data]
    x_positions = list(range(len(modes_in_data)))

    # Determine model order; filter by MIN_SAMPLE_THRESHOLD for multi-party data
    all_models_in_data = set(df["model"].unique())
    ordered_models = [m for m in MODEL_ORDER if m in all_models_in_data]
    ordered_models += sorted(m for m in all_models_in_data if m not in MODEL_ORDER)

    # Keep only models that meet the threshold in at least one multi-party category
    mp_categories = {cat for cat, _ in _CATEGORIES}
    qualifying_models: list[str] = []
    for model in ordered_models:
        model_df = df[(df["model"] == model) & (df["category"].isin(mp_categories))]
        if len(model_df) >= MIN_SAMPLE_THRESHOLD:
            qualifying_models.append(model)

    # Fall back to all models if none qualify (e.g. in tests with sparse data)
    if not qualifying_models:
        qualifying_models = ordered_models

    legend_handles: list = []
    legend_labels: list[str] = []
    seen_labels: set[str] = set()

    for col_idx, (category, cat_label) in enumerate(_CATEGORIES):
        cat_df = df[df["category"] == category]

        for row_idx, (metric, metric_label, _lower_is_better) in enumerate(_METRICS):
            ax = axes[row_idx, col_idx]

            data_mins: list[float] = []
            data_maxs: list[float] = []

            for model in qualifying_models:
                model_df = cat_df[cat_df["model"] == model]
                if model_df.empty:
                    continue

                means: list[float] = []
                ci_los: list[float] = []
                ci_his: list[float] = []

                for mode in modes_in_data:
                    mode_df = model_df[model_df["privacy_mode"] == mode]
                    if mode_df.empty:
                        means.append(np.nan)
                        ci_los.append(0.0)
                        ci_his.append(0.0)
                        continue

                    if metric == "task_completed":
                        series = mode_df[metric].astype(float) * 100.0
                    else:
                        series = mode_df[metric].astype(float)

                    mean, lo, hi = compute_ci(series)
                    means.append(mean)
                    ci_los.append(mean - lo if not np.isnan(mean) else 0.0)
                    ci_his.append(hi - mean if not np.isnan(mean) else 0.0)
                    if not np.isnan(mean):
                        data_mins.append(mean)
                        data_maxs.append(mean)

                if all(np.isnan(m) for m in means):
                    continue

                color = model_color(model)
                marker = model_marker(model)
                label = display_name(model)

                line, = ax.plot(
                    x_positions, means,
                    color=color, marker=marker,
                    linewidth=1.2, markersize=4,
                    zorder=3, alpha=0.9,
                )
                ax.errorbar(
                    x_positions, means,
                    yerr=[ci_los, ci_his],
                    fmt="none", color=color,
                    capsize=2, linewidth=0.8,
                    zorder=2, alpha=0.6,
                )

                if label not in seen_labels:
                    legend_handles.append(line)
                    legend_labels.append(label)
                    seen_labels.add(label)

            # Axes formatting
            ax.set_xticks(x_positions)
            ax.set_xticklabels(mode_label_list, fontsize=5.5)
            ax.grid(True, alpha=0.3)

            # Y-axis zoom with sensible bounds
            if data_mins and data_maxs:
                d_min, d_max = min(data_mins), max(data_maxs)
                pad = max((d_max - d_min) * 0.2, 0.02)
                y_lo = d_min - pad
                y_hi = d_max + pad
                if metric == "task_completed":
                    y_lo = max(0.0, y_lo)
                    y_hi = min(100.0, y_hi)
                else:
                    y_lo = max(0.0, y_lo)
                    y_hi = min(1.0, y_hi)
                ax.set_ylim(y_lo, y_hi)

            # Row labels (left column only)
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=7)

            # Column titles (top row only)
            if row_idx == 0:
                ax.set_title(cat_label, fontsize=9, fontweight="bold")

            # X-axis label (bottom row only)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Privacy Level", fontsize=7)

    # Shared legend below all panels
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="lower center",
            ncol=min(len(legend_handles), 6),
            fontsize=6,
            bbox_to_anchor=(0.5, -0.04),
            frameon=True,
        )

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate multi-party privacy mode effect figure (appendix)."
    )
    parser.add_argument("--results-dir", default="prism/analysis/results",
                        help="Directory containing eval_*.json files")
    parser.add_argument("--output", "-o",
                        default="latex/prism/figures/privacy_mode_effect_mp.pdf",
                        help="Output image file path")
    args = parser.parse_args()

    df = load_results_df(args.results_dir)
    generate(df, args.output)
