# prism/analysis/plot_privacy_mode_grouped.py
"""Privacy mode effect: Dyadic vs Multi-Party grouped averages.

2 rows (Dyadic, Multi-Party) × 3 columns (Leakage, IAS, Task Compl. %).
Each panel has one line per model connecting privacy levels with CI error bars.
Metrics are averaged across all categories within each group.
"""
from __future__ import annotations

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
    model_color,
    model_marker,
    setup_style,
)

_METRICS = [
    ("leakage_rate", "Leakage Rate", True),
    ("ias", "IAS", False),
    ("task_completed", "Task Compl. %", False),
]

_GROUPS = [
    ("Dyadic", ["cross_domain", "mediated_comm", "cross_user"]),
    ("Multi-Party", ["multi_party_group", "competitive", "affinity_modulated"]),
]

_TRAJECTORY_MODES = ["unconstrained", "explicit", "full_defense"]
_MODE_SHORT = {
    "unconstrained": "L0\nUnconstrained",
    "explicit": "L1\nExplicit",
    "full_defense": "L2\nFull Defense",
}

_MP_CATEGORIES = {"multi_party_group", "competitive", "affinity_modulated"}


def generate(df: pd.DataFrame, output_path: str) -> None:
    """Generate 2×3 grouped privacy mode effect figure."""
    setup_style()

    n_rows = len(_GROUPS)
    n_cols = len(_METRICS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FULL_WIDTH, FULL_WIDTH * 0.5),
        sharey=False,
    )

    modes_in_data = [m for m in _TRAJECTORY_MODES if m in df["privacy_mode"].unique()]
    mode_labels = [_MODE_SHORT.get(m, m) for m in modes_in_data]
    x_positions = list(range(len(modes_in_data)))

    all_models_in_data = set(df["model"].unique())
    ordered_models = [m for m in MODEL_ORDER if m in all_models_in_data]

    legend_handles: list = []
    legend_labels: list[str] = []
    seen_labels: set[str] = set()

    for row_idx, (group_label, categories) in enumerate(_GROUPS):
        # Filter to categories in this group
        group_df = df[df["category"].isin(categories)]

        # For multi-party, only include models with sufficient data
        if any(c in _MP_CATEGORIES for c in categories):
            group_models = [
                m for m in ordered_models
                if any(
                    len(group_df[(group_df["model"] == m) & (group_df["category"] == c)]) >= MIN_SAMPLE_THRESHOLD
                    for c in categories
                )
            ]
        else:
            group_models = [m for m in ordered_models if not group_df[group_df["model"] == m].empty]

        for col_idx, (metric, metric_label, lower_is_better) in enumerate(_METRICS):
            ax = axes[row_idx, col_idx]

            data_mins: list[float] = []
            data_maxs: list[float] = []

            for model in group_models:
                model_df = group_df[group_df["model"] == model]
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
            ax.set_xticklabels(mode_labels, fontsize=5.5)
            ax.grid(True, alpha=0.3)

            # Y-axis zoom
            if data_mins and data_maxs:
                d_min, d_max = min(data_mins), max(data_maxs)
                pad = max((d_max - d_min) * 0.2, 0.02)
                y_lo = d_min - pad
                y_hi = d_max + pad
                if metric == "task_completed":
                    y_lo = max(0, y_lo)
                    y_hi = min(100, y_hi)
                else:
                    y_lo = max(0, y_lo)
                    y_hi = min(1.0, y_hi)
                ax.set_ylim(y_lo, y_hi)

            # Row labels (left column only)
            if col_idx == 0:
                ax.set_ylabel(group_label, fontsize=8, fontweight="bold")

            # Column titles (top row only)
            if row_idx == 0:
                ax.set_title(metric_label, fontsize=9, fontweight="bold")

            # X-axis label (bottom row only)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Privacy Level", fontsize=7)

    # Shared legend below
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
