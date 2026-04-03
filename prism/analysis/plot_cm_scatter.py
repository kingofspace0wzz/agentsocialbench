"""CM Competitive Privacy: Self-Leakage vs. Extraction scatter plot."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from prism.analysis.loader import (
    MIN_SAMPLE_THRESHOLD,
    MODEL_ORDER,
    SINGLE_COL_WIDTH,
    display_name,
    model_color,
    setup_style,
)

_MODE_MARKERS = {
    "unconstrained": ("L0", "o"),
    "implicit": ("L1", "s"),
    "explicit": ("L2", "^"),
    "full_defense": ("L4", "*"),
}


def generate(df: pd.DataFrame, output_path: str) -> None:
    """Generate CM self-leakage vs extraction scatter."""
    setup_style()

    cm = df[df["category"] == "competitive"].copy() if not df.empty and "category" in df.columns else pd.DataFrame()
    if cm.empty or "cslr" not in cm.columns or "cer" not in cm.columns:
        fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * 0.9))
        ax.text(0.5, 0.5, "No CM data", transform=ax.transAxes, ha="center")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * 0.9))

    models = [m for m in MODEL_ORDER
              if len(cm[cm["model"] == m]) >= MIN_SAMPLE_THRESHOLD]

    for model in models:
        model_df = cm[cm["model"] == model]
        color = model_color(model)
        name = display_name(model)

        # Plot mean per privacy mode
        for mode, (label, marker) in _MODE_MARKERS.items():
            mode_df = model_df[model_df["privacy_mode"] == mode]
            if mode_df.empty:
                continue
            mean_cslr = mode_df["cslr"].dropna().mean()
            mean_cer = mode_df["cer"].dropna().mean()
            ax.scatter(mean_cslr, mean_cer, color=color, marker=marker,
                       s=60, zorder=3)

        # Connect same-model points with faint trajectory line
        mode_points = []
        for mode in ["unconstrained", "implicit", "explicit", "full_defense"]:
            mode_df = model_df[model_df["privacy_mode"] == mode]
            if not mode_df.empty:
                mode_points.append((mode_df["cslr"].dropna().mean(),
                                    mode_df["cer"].dropna().mean()))
        if len(mode_points) > 1:
            xs, ys = zip(*mode_points)
            ax.plot(xs, ys, color=color, alpha=0.3, linewidth=1, zorder=2)

    ax.set_xlabel("CSLR (Self-Leakage, lower = better)")
    ax.set_ylabel("CER (Extraction, lower = better)")
    ax.set_title("Competitive Privacy Dynamics")

    # Legend: one entry per model
    handles = [plt.Line2D([0], [0], marker="o", color=model_color(m),
               linestyle="", markersize=6, label=display_name(m))
               for m in models]
    if handles:
        ax.legend(handles=handles, loc="best", fontsize=7)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
