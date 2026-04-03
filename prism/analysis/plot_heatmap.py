# prism/analysis/plot_heatmap.py
"""Paired behavioral heatmap — L0 vs L2, generic behaviors (Figure 5)."""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.gridspec import GridSpec

from prism.analysis.loader import (
    DPI,
    FULL_WIDTH,
    GENERIC_NEGATIVE,
    GENERIC_POSITIVE,
    MODEL_ORDER,
    display_name,
    load_results_df,
    setup_style,
)
from prism.analysis.table_behavioral import BEHAVIOR_DISPLAY

# The 8 generic behaviors in desired column order: 4 neg + 4 pos
_ALL_GENERIC = GENERIC_NEGATIVE + GENERIC_POSITIVE

# Short column labels
_COL_LABELS = [BEHAVIOR_DISPLAY.get(b, b) for b in _ALL_GENERIC]

# (matplotlib marker, face colour) for each of the 8 generic behaviors
_MARKER_DEFS = [
    ("o", "#C0392B"),  # Oversharing
    ("s", "#E67E22"),  # Cross-referencing
    ("^", "#8E44AD"),  # Implicit Disclosure
    ("D", "#D4880F"),  # Unnecessary Context
    ("v", "#2471A3"),  # Appropriate Abstraction
    ("P", "#1ABC9C"),  # Minimal Disclosure
    ("*", "#1B4F72"),  # Deflection
    ("X", "#27AE60"),  # Boundary Maintenance
]


def _build_matrix(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Compute observation % per (model, behavior) for all generic behaviors.

    Rows = models (display names), columns = behaviors (display labels).
    Values in [0, 100].
    """
    records: dict[str, dict[str, float]] = {}
    for model in models:
        mdf = df[df["model"] == model]
        row: dict[str, float] = {}
        for beh, col_label in zip(_ALL_GENERIC, _COL_LABELS):
            if beh not in mdf.columns:
                row[col_label] = np.nan
            else:
                valid = mdf[beh].dropna()
                row[col_label] = float(valid.astype(float).mean() * 100.0) if len(valid) > 0 else np.nan
        records[display_name(model)] = row
    return pd.DataFrame(records).T  # rows=models, cols=behaviors


def generate(df: pd.DataFrame, output_path: str) -> None:
    """Generate condensed paired L0/L2 behavioral heatmap with symbol legend.

    Layout: two heatmap rows (L0, L2) on the left with coloured-symbol
    column identifiers, and a legend column on the right mapping each
    symbol to its behaviour name.
    """
    setup_style()

    # Determine model order
    all_models_in_data = set(df["model"].unique())
    ordered_models = [m for m in MODEL_ORDER if m in all_models_in_data]
    ordered_models += sorted(m for m in all_models_in_data if m not in MODEL_ORDER)

    # Separate into L0 and L2 subsets
    df_l0 = df[df["privacy_mode"] == "unconstrained"]
    df_l2 = df[df["privacy_mode"] == "full_defense"]

    # Build matrices (fall back to all data if mode is absent)
    if df_l0.empty:
        df_l0 = df
    if df_l2.empty:
        df_l2 = df

    mat_l0 = _build_matrix(df_l0, ordered_models)
    mat_l2 = _build_matrix(df_l2, ordered_models)

    neg_labels = [BEHAVIOR_DISPLAY.get(b, b) for b in GENERIC_NEGATIVE]
    pos_labels = [BEHAVIOR_DISPLAY.get(b, b) for b in GENERIC_POSITIVE]
    all_labels = neg_labels + pos_labels
    n_neg = len(neg_labels)

    # ── Figure layout: heatmaps (left) + legend column (right) ───
    fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.30))
    gs = GridSpec(
        2, 2,
        width_ratios=[3.5, 1.2],
        hspace=0.35,
        wspace=0.05,
        figure=fig,
    )

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_leg = fig.add_subplot(gs[:, 1])
    ax_leg.axis("off")

    panels = [
        (ax_top, mat_l0[all_labels], "L0: Unconstrained"),
        (ax_bot, mat_l2[all_labels], "L2: Full Defense"),
    ]

    cmap_neg = plt.cm.Reds
    cmap_pos = plt.cm.Blues

    for panel_idx, (ax, data, title) in enumerate(panels):
        plot_data = data.fillna(0.0)
        ax.set_facecolor("white")

        n_rows, n_cols = plot_data.shape
        for i in range(n_rows):
            for j in range(n_cols):
                val = plot_data.iloc[i, j]
                normed = val / 100.0
                if j < n_neg:
                    color = cmap_neg(0.15 + normed * 0.65)
                else:
                    color = cmap_pos(0.15 + normed * 0.65)
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1,
                    facecolor=color, linewidth=0.5, edgecolor="white",
                ))
                ax.text(
                    j + 0.5, i + 0.5, f"{val:.0f}",
                    ha="center", va="center",
                    fontsize=6.5, fontweight="bold",
                    color="white" if normed > 0.55 else "black",
                )

        ax.set_xlim(0, n_cols)
        ax.set_ylim(n_rows, 0)
        ax.set_yticks([i + 0.5 for i in range(n_rows)])
        ax.set_yticklabels(plot_data.index, rotation=0, fontsize=6)
        ax.set_title(title, fontsize=8, pad=4)

        # Vertical separator between negative and positive columns
        ax.axvline(x=n_neg, color="black", linewidth=1.5)

        # Group headers
        ax.text(
            n_neg / 2, -0.3, "Negative",
            ha="center", va="bottom", fontsize=6.5,
            fontstyle="italic", color="#c0392b",
        )
        ax.text(
            n_neg + len(pos_labels) / 2, -0.3, "Positive",
            ha="center", va="bottom", fontsize=6.5,
            fontstyle="italic", color="#2471a3",
        )

        # Coloured symbol markers replace text x-tick labels
        ax.set_xticks([j + 0.5 for j in range(n_cols)])
        ax.set_xticklabels([""] * n_cols)

        if panel_idx == 1:  # bottom panel: draw markers below cells
            for j, (marker, mcolor) in enumerate(_MARKER_DEFS):
                ax.plot(
                    j + 0.5, n_rows + 0.4,
                    marker=marker, color=mcolor,
                    markersize=5, linestyle="none", clip_on=False,
                )
        else:
            ax.tick_params(axis="x", length=0)

    # ── Legend panel ──────────────────────────────────────────────
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)

    y = 0.94
    step = 0.075

    ax_leg.text(0.0, y, "Negative", fontsize=7, fontweight="bold", color="#c0392b")
    y -= step * 0.7
    for i in range(n_neg):
        marker, mcolor = _MARKER_DEFS[i]
        ax_leg.plot(0.08, y, marker=marker, color=mcolor,
                    markersize=5, linestyle="none")
        ax_leg.text(0.18, y, all_labels[i], fontsize=6, va="center")
        y -= step

    y -= step * 0.4
    ax_leg.text(0.0, y, "Positive", fontsize=7, fontweight="bold", color="#2471a3")
    y -= step * 0.7
    for i in range(len(pos_labels)):
        marker, mcolor = _MARKER_DEFS[n_neg + i]
        ax_leg.plot(0.08, y, marker=marker, color=mcolor,
                    markersize=5, linestyle="none")
        ax_leg.text(0.18, y, all_labels[n_neg + i], fontsize=6, va="center")
        y -= step

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paired behavioral heatmap (Fig. 5).")
    parser.add_argument("--results-dir", default="prism/analysis/results",
                        help="Directory containing eval_*.json files")
    parser.add_argument("--output", "-o", default="prism/analysis/plots/behavioral_heatmap.pdf",
                        help="Output image file path")
    args = parser.parse_args()

    df = load_results_df(args.results_dir)
    generate(df, args.output)
    print(f"Wrote {args.output}")
