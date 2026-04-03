# prism/analysis/plot_cu_asymmetry.py
"""Cross-user asymmetry scatter: user_a_leakage vs user_b_leakage (Appendix)."""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prism.analysis.loader import (
    CATEGORY_MARKERS,
    DPI,
    FULL_WIDTH,
    MODEL_ORDER,
    PRIVACY_MODE_ORDER,
    display_name,
    model_color,
    load_results_df,
    setup_style,
)

# Simple marker mapping for privacy modes
_MODE_MARKERS = {
    "unconstrained": "o",
    "implicit": "s",
    "explicit": "^",
    "enhanced": "D",
    "full_defense": "P",
}


def generate(df: pd.DataFrame, output_path: str) -> None:
    """Generate CU asymmetry scatter and save to *output_path*."""
    setup_style()

    cu_df = df[df["category"] == "cross_user"].copy()

    if cu_df.empty or cu_df["user_a_leakage"].isna().all():
        fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.5, FULL_WIDTH * 0.5))
        ax.text(0.5, 0.5, "No cross-user data", ha="center", va="center",
                transform=ax.transAxes)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.5, FULL_WIDTH * 0.5))

    # Reference diagonal y = x
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="y = x")

    # Determine model order
    all_models_in_data = set(cu_df["model"].unique())
    ordered_models = [m for m in MODEL_ORDER if m in all_models_in_data]
    ordered_models += sorted(m for m in all_models_in_data if m not in MODEL_ORDER)

    modes_present = [m for m in PRIVACY_MODE_ORDER if m in cu_df["privacy_mode"].unique()]

    legend_model_handles: dict[str, object] = {}
    legend_mode_handles: dict[str, object] = {}

    for model in ordered_models:
        color = model_color(model)
        model_df = cu_df[cu_df["model"] == model]

        for mode in modes_present:
            mode_df = model_df[model_df["privacy_mode"] == mode]
            if mode_df.empty:
                continue

            xa = mode_df["user_a_leakage"].dropna()
            xb = mode_df["user_b_leakage"].dropna()
            # Align indices
            common = xa.index.intersection(xb.index)
            if len(common) == 0:
                continue

            marker = _MODE_MARKERS.get(mode, "o")
            sc = ax.scatter(
                xa.loc[common],
                xb.loc[common],
                color=color,
                marker=marker,
                alpha=0.7,
                s=20,
                edgecolors="none",
                zorder=3,
            )

            label_model = display_name(model)
            if label_model not in legend_model_handles:
                legend_model_handles[label_model] = sc

            label_mode = f"L{PRIVACY_MODE_ORDER.index(mode)}: {mode.replace('_', ' ').title()}"
            if label_mode not in legend_mode_handles:
                legend_mode_handles[label_mode] = ax.scatter(
                    [], [], color="grey", marker=marker, s=20
                )

    ax.set_xlabel("User A Leakage Rate", fontsize=8)
    ax.set_ylabel("User B Leakage Rate", fontsize=8)
    ax.set_title("Cross-User Asymmetry", fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")

    # Two separate legends: one for models, one for privacy modes
    if legend_model_handles:
        leg1 = ax.legend(
            list(legend_model_handles.values()),
            list(legend_model_handles.keys()),
            title="Model", fontsize=6, title_fontsize=6,
            loc="upper left", frameon=True,
        )
        ax.add_artist(leg1)

    if legend_mode_handles:
        ax.legend(
            list(legend_mode_handles.values()),
            list(legend_mode_handles.keys()),
            title="Privacy Mode", fontsize=6, title_fontsize=6,
            loc="lower right", frameon=True,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cross-user asymmetry scatter.")
    parser.add_argument("--results-dir", default="prism/analysis/results",
                        help="Directory containing eval_*.json files")
    parser.add_argument("--output", "-o", default="prism/analysis/plots/cu_asymmetry.pdf",
                        help="Output image file path")
    args = parser.parse_args()

    df = load_results_df(args.results_dir)
    generate(df, args.output)
    print(f"Wrote {args.output}")
