# prism/analysis/plot_pareto.py
"""Trajectory lines: privacy-utility trade-off across L0→L1→L2→L4 (Figure 3).

2×3 grid of panels: top row CD/CU/MC (dyadic), bottom row GC/CM/AM (multi-party).
Lines connect privacy modes in order, colored per model.
Privacy level is encoded by BOTH color (model) AND marker shape (level), making
both dimensions immediately readable.

Redesign details:
- Marker shapes encode privacy level: L0=circle, L1=square, L2=triangle, L4=star
- Model colors distinguish models (from model_color())
- Points are larger (s=60) so shapes are visible
- Lines are thicker (linewidth=1.8) and slightly transparent (alpha=0.6)
- L0/L4 text annotations removed (redundant with marker shapes)
- Shared marker-shape legend below figure shows L0/L1/L2/L4
- Multi-party panels (GC/CM/AM) only show models with ≥MIN_SAMPLE_THRESHOLD scenarios
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

from prism.analysis.loader import (
    CATEGORY_DISPLAY,
    DPI,
    FULL_WIDTH,
    MIN_SAMPLE_THRESHOLD,
    MODEL_ORDER,
    display_name,
    model_color,
    load_results_df,
    setup_style,
)

# Category display order for the 2×3 grid of panels
# (top row: CD/CU/MC, bottom row: GC/CM/AM)
_PANEL_CATEGORIES = [
    ("cross_domain", "CDLR"),
    ("cross_user", "CULR"),
    ("mediated_comm", "MLR"),
    ("multi_party_group", "MPLR"),
    ("hub_and_spoke", "HALR"),
    ("competitive", "CSLR"),
    ("affinity_modulated", "1−ACS"),
]

# Multi-party categories require MIN_SAMPLE_THRESHOLD for model inclusion
_MP_CATEGORIES = {"multi_party_group", "hub_and_spoke", "competitive", "affinity_modulated"}

# Privacy level → short label and distinct marker shape
# Paper convention: L0 (unconstrained), L1 (explicit), L2 (full defense)
# Implicit is shown but not numbered in main text
_LEVEL_MARKERS = {
    "unconstrained": ("L0", "o"),    # circle
    "implicit":      ("Imp", "s"),   # square
    "explicit":      ("L1", "^"),    # triangle-up
    "full_defense":  ("L2", "*"),    # star
}

# Trajectory order: L0 → Implicit → L1 → L2
_TRAJECTORY_MODES = ["unconstrained", "implicit", "explicit", "full_defense"]


def generate(df: pd.DataFrame, output_path: str) -> None:
    """Generate dynamic-grid trajectory scatter/line plot and save to *output_path*."""
    setup_style()

    # Filter to categories that have data
    categories_present = [
        (cat, metric) for cat, metric in _PANEL_CATEGORIES
        if not df[df["category"] == cat].empty
    ]
    n_panels = len(categories_present)
    n_cols = 4
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FULL_WIDTH, FULL_WIDTH * 0.35 * n_rows),
        sharey=False,
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    # Hide unused axes
    for i in range(n_panels, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Determine models to plot
    all_models_in_data = set(df["model"].unique())
    ordered_models = [m for m in MODEL_ORDER if m in all_models_in_data]
    ordered_models += sorted(m for m in all_models_in_data if m not in MODEL_ORDER)

    # Model legend handles (line + scatter combined proxy)
    model_legend_handles: list = []
    model_legend_labels: list[str] = []
    seen_model_labels: set[str] = set()

    for idx, (category, metric_name) in enumerate(categories_present):
        ax = axes_flat[idx]
        cat_df = df[df["category"] == category]
        cat_display = CATEGORY_DISPLAY.get(category, category)

        # Filter models per panel: multi-party categories require MIN_SAMPLE_THRESHOLD
        if category in _MP_CATEGORIES:
            panel_models = [
                m for m in ordered_models
                if len(cat_df[cat_df["model"] == m]) >= MIN_SAMPLE_THRESHOLD
            ]
        else:
            panel_models = [
                m for m in ordered_models
                if not cat_df[cat_df["model"] == m].empty
            ]

        all_xs: list[float] = []
        model_points: dict[str, list[tuple[float, float]]] = {}

        for model in panel_models:
            model_df = cat_df[cat_df["model"] == model]
            if model_df.empty:
                continue

            color = model_color(model)
            label = display_name(model)

            xs: list[float] = []
            ys: list[float] = []
            markers_used: list[str] = []

            for mode in _TRAJECTORY_MODES:
                mode_df = model_df[model_df["privacy_mode"] == mode]
                if mode_df.empty:
                    continue
                leak = (
                    float(mode_df["leakage_rate"].mean())
                    if mode_df["leakage_rate"].notna().any()
                    else np.nan
                )
                tcq = (
                    float(mode_df["tcq"].mean())
                    if mode_df["tcq"].notna().any()
                    else np.nan
                )
                if not np.isnan(leak) and not np.isnan(tcq):
                    xs.append(leak)
                    ys.append(tcq)
                    _, mkr = _LEVEL_MARKERS.get(mode, (mode, "o"))
                    markers_used.append(mkr)

            if not xs:
                continue

            all_xs.extend(xs)
            model_points[model] = list(zip(xs, ys))

            # Draw trajectory line (thicker, slightly transparent so markers stand out)
            (line,) = ax.plot(
                xs, ys,
                color=color,
                linewidth=1.8,
                alpha=0.6,
                zorder=3,
            )

            # Draw each point with its privacy-level-specific marker shape (larger)
            for x_pt, y_pt, mkr in zip(xs, ys, markers_used):
                ax.scatter(
                    [x_pt], [y_pt],
                    color=color,
                    marker=mkr,
                    s=60,
                    zorder=5,
                    edgecolors="white",
                    linewidths=0.4,
                )

            if label not in seen_model_labels:
                # Proxy artist: colored line for model legend
                proxy = mlines.Line2D(
                    [], [],
                    color=color,
                    linewidth=1.8,
                    marker="o",
                    markersize=5,
                    label=label,
                )
                model_legend_handles.append(proxy)
                model_legend_labels.append(label)
                seen_model_labels.add(label)

        ax.set_title(cat_display, fontsize=9)
        ax.set_xlabel(f"{metric_name} (lower = better)", fontsize=7)

        # Dynamic x-limits with 10% padding
        if all_xs:
            x_min, x_max = min(all_xs), max(all_xs)
            x_range = x_max - x_min if x_max > x_min else 0.1
            ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
        else:
            ax.set_xlim(-0.05, 1.05)

        # Dynamic y-limits: zoom into actual TCQ range with 5% padding
        all_y = [y for points in model_points.values() for _, y in points]
        if all_y:
            y_min = min(all_y) - 0.05
            y_max = max(all_y) + 0.05
            ax.set_ylim(max(0, y_min), min(1.05, y_max))
        else:
            ax.set_ylim(0.0, 1.05)

        if idx % n_cols == 0:
            ax.set_ylabel("TCQ (higher = better)", fontsize=7)

        ax.grid(True, alpha=0.3)

    # Build privacy-level (marker shape) legend entries
    level_legend_handles: list = []
    for mode in _TRAJECTORY_MODES:
        lbl, mkr = _LEVEL_MARKERS[mode]
        proxy = mlines.Line2D(
            [], [],
            color="black",
            marker=mkr,
            markersize=6,
            linestyle="None",
            label=lbl,
        )
        level_legend_handles.append(proxy)

    # Combined legend: model colors first, then a separator-ish gap, then level shapes
    all_handles = model_legend_handles + level_legend_handles
    all_labels = model_legend_labels + [_LEVEL_MARKERS[m][0] for m in _TRAJECTORY_MODES]

    if all_handles:
        # Two-row legend below figure: models on top row, levels on bottom row
        # We use two separate legends positioned side-by-side
        n_models = len(model_legend_handles)
        n_levels = len(level_legend_handles)

        legend_model = fig.legend(
            model_legend_handles,
            model_legend_labels,
            loc="lower left",
            ncol=n_models,
            fontsize=6,
            bbox_to_anchor=(0.02, -0.14),
            frameon=True,
            title="Model",
            title_fontsize=6,
        )
        legend_level = fig.legend(
            level_legend_handles,
            [_LEVEL_MARKERS[m][0] for m in _TRAJECTORY_MODES],
            loc="lower right",
            ncol=n_levels,
            fontsize=6,
            bbox_to_anchor=(0.98, -0.14),
            frameon=True,
            title="Privacy Level",
            title_fontsize=6,
        )
        fig.add_artist(legend_model)

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate trajectory pareto plot (Fig. 3)."
    )
    parser.add_argument(
        "--results-dir",
        default="prism/analysis/results",
        help="Directory containing eval_*.json files",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="prism/analysis/plots/pareto_frontier.pdf",
        help="Output image file path",
    )
    args = parser.parse_args()

    df = load_results_df(args.results_dir)
    generate(df, args.output)
    print(f"Wrote {args.output}")
