"""L0 vs ZDD comparison across multi-party categories.

Line-plot style matching plot_privacy_mode.py / plot_privacy_mode_mp.py.
Shows 3 metrics (rows) x 4 multi-party categories (columns) = 12 panels.
Each panel has one line per model connecting L0 and ZDD with CI error bars.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prism.analysis.loader import (
    FULL_WIDTH,
    MIN_SAMPLE_THRESHOLD,
    MODEL_ORDER,
    compute_ci,
    display_name,
    model_color,
    model_marker,
    setup_style,
)

_CATEGORIES = [
    ("multi_party_group", "Group Chat"),
    ("hub_and_spoke", "Hub-and-Spoke"),
    ("competitive", "Competitive"),
    ("affinity_modulated", "Affinity-Mod."),
]

_METRICS = [
    ("leakage_rate", "Leakage Rate", True),
    ("ias", "IAS", False),
    ("task_completed", "Task Compl. %", False),
]

_MODES = ["unconstrained", "zdd"]
_MODE_LABELS = {"unconstrained": "L0\nUnconstrained", "zdd": "ZDD"}


def generate(df: pd.DataFrame, output_path: str) -> None:
    """Generate L0 vs ZDD comparison figure in line-plot style."""
    setup_style()

    zdd_df = df[df["privacy_mode"] == "zdd"]
    if zdd_df.empty:
        fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3))
        ax.text(0.5, 0.5, "No ZDD data", transform=ax.transAxes, ha="center")
        fig.savefig(output_path)
        plt.close(fig)
        return

    n_rows = len(_METRICS)
    n_cols = len(_CATEGORIES)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(FULL_WIDTH, FULL_WIDTH * 0.55))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Find models with ZDD data in at least one MP category
    mp_cats = {c for c, _ in _CATEGORIES}
    all_models_in_data = [m for m in MODEL_ORDER if m in df["model"].unique()]
    extra = sorted(m for m in df["model"].unique() if m not in MODEL_ORDER)
    all_models_in_data += extra

    models = [m for m in all_models_in_data
              if any(len(zdd_df[(zdd_df["model"] == m) & (zdd_df["category"] == c)])
                     >= MIN_SAMPLE_THRESHOLD for c in mp_cats)]

    modes_in_data = [m for m in _MODES if m in df["privacy_mode"].unique()]

    legend_handles = []
    legend_labels = []
    seen = set()

    for col_idx, (category, cat_label) in enumerate(_CATEGORIES):
        cat_df = df[df["category"] == category]

        for row_idx, (metric, metric_label, _lower) in enumerate(_METRICS):
            ax = axes[row_idx, col_idx]

            for model in models:
                model_df = cat_df[cat_df["model"] == model]
                if len(model_df[model_df["privacy_mode"].isin(mp_cats | set(_MODES))]) == 0:
                    continue

                xs, ys, errs = [], [], []
                for mode_idx, mode in enumerate(modes_in_data):
                    mode_df = model_df[model_df["privacy_mode"] == mode]
                    if mode_df.empty:
                        continue
                    vals = mode_df[metric].dropna()
                    if metric == "task_completed":
                        vals = vals * 100
                    if len(vals) == 0:
                        continue
                    mean, lo, hi = compute_ci(vals)
                    xs.append(mode_idx)
                    ys.append(mean)
                    errs.append((mean - lo, hi - mean))

                if xs:
                    errs_lo, errs_hi = zip(*errs)
                    color = model_color(model)
                    marker = model_marker(model)
                    name = display_name(model)
                    line = ax.errorbar(
                        xs, ys, yerr=[errs_lo, errs_hi],
                        color=color, marker=marker,
                        capsize=3, linewidth=1.2, markersize=5,
                        label=name if name not in seen else None,
                    )
                    if name not in seen:
                        legend_handles.append(line)
                        legend_labels.append(name)
                        seen.add(name)

            ax.set_xticks(range(len(modes_in_data)))
            ax.set_xticklabels([_MODE_LABELS.get(m, m) for m in modes_in_data],
                               fontsize=7)
            if row_idx == 0:
                ax.set_title(cat_label, fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=8)

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="lower center",
                   ncol=min(len(legend_labels), 6), framealpha=0.9, fontsize=7)

    fig.subplots_adjust(bottom=0.13, hspace=0.35, wspace=0.3, top=0.95)
    fig.savefig(output_path)
    plt.close(fig)
