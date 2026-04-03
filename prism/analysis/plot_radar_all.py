# prism/analysis/plot_radar_all.py
"""Single radar with all 6 models overlaid — full-width figure (Appendix)."""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prism.analysis.loader import (
    FULL_WIDTH,
    DPI,
    MODEL_ORDER,
    display_name,
    model_color,
    model_marker,
    load_results_df,
    setup_style,
)

# Same 6 axes as Task 8
_AXIS_LABELS = [
    "Privacy\n(1\u2212CDLR)",
    "Mediation\n(1\u2212MLR)",
    "Cross-User\n(1\u2212CULR)",
    "IAS",
    "TCQ",
    "Safety\n(4\u2212NegBeh)/4",
]


def _compute_radar_values(df: pd.DataFrame, model: str) -> list[float] | None:
    """Compute 6 radar axis values for a single model."""
    mdf = df[df["model"] == model]
    if mdf.empty:
        return None

    cd_df = mdf[mdf["category"] == "cross_domain"]
    privacy_val = (1.0 - float(cd_df["leakage_rate"].mean())
                   if not cd_df.empty and cd_df["leakage_rate"].notna().any() else 0.5)

    mc_df = mdf[mdf["category"] == "mediated_comm"]
    mediation_val = (1.0 - float(mc_df["leakage_rate"].mean())
                     if not mc_df.empty and mc_df["leakage_rate"].notna().any() else np.nan)

    cu_df = mdf[mdf["category"] == "cross_user"]
    cu_val = (1.0 - float(cu_df["leakage_rate"].mean())
              if not cu_df.empty and cu_df["leakage_rate"].notna().any() else 0.5)

    ias_val = float(mdf["ias"].mean()) if mdf["ias"].notna().any() else 0.5
    tcq_val = float(mdf["tcq"].mean()) if mdf["tcq"].notna().any() else 0.5

    if mdf["negative_count"].notna().any():
        safety_val = (4.0 - float(mdf["negative_count"].mean())) / 4.0
    else:
        safety_val = 0.5

    values = [privacy_val, mediation_val, cu_val, ias_val, tcq_val, safety_val]
    # Preserve NaN (missing category) rather than clamping; caller handles display
    return [np.nan if np.isnan(v) else max(0.0, min(1.0, v)) for v in values]


def generate(df: pd.DataFrame, output_path: str) -> None:
    """Generate single radar with all models and save to *output_path*."""
    setup_style()

    all_models_in_data = set(df["model"].unique())
    ordered_models = [m for m in MODEL_ORDER if m in all_models_in_data]
    ordered_models += sorted(m for m in all_models_in_data if m not in MODEL_ORDER)

    n_axes = len(_AXIS_LABELS)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(
        figsize=(FULL_WIDTH, FULL_WIDTH * 0.55),
        subplot_kw=dict(polar=True),
    )

    # Grid rings
    for level in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles_closed, [level] * (n_axes + 1), "grey",
                linewidth=0.4, linestyle="--", alpha=0.4, zorder=0)

    handles = []
    missing_data_models: list[str] = []
    for model in ordered_models:
        vals = _compute_radar_values(df, model)
        if vals is None:
            continue
        color = model_color(model)
        marker = model_marker(model)
        label = display_name(model)

        has_nan = any(np.isnan(v) for v in vals)
        if has_nan:
            missing_data_models.append(label)

        # Replace NaN with 0 so the polygon closes; skip markers at NaN vertices
        vals_for_poly = [0.0 if np.isnan(v) else v for v in vals]
        vals_closed = vals_for_poly + vals_for_poly[:1]

        line, = ax.plot(angles_closed, vals_closed, linewidth=1.5, color=color,
                        marker=None, zorder=3)
        ax.fill(angles_closed, vals_closed, alpha=0.15, color=color, zorder=2)

        # Draw markers only at non-NaN vertices
        for i, (angle, val) in enumerate(zip(angles, vals)):
            if not np.isnan(val):
                ax.plot([angle], [val], color=color, marker=marker, markersize=4, zorder=4)

        handles.append((line, label))

    ax.set_thetagrids(np.degrees(angles), _AXIS_LABELS, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=5, color="grey")
    ax.grid(True, alpha=0.3)

    if handles:
        ax.legend(
            [h for h, _ in handles],
            [lbl for _, lbl in handles],
            loc="upper right",
            bbox_to_anchor=(1.35, 1.1),
            fontsize=7,
            frameon=True,
        )

    # Add footnote for models with missing category data
    if missing_data_models:
        missing_str = ", ".join(sorted(set(missing_data_models)))
        fig.text(
            0.5, -0.02,
            f"* {missing_str}: no Mediated Comm. data (axis set to 0)",
            ha="center", fontsize=6, color="grey", style="italic",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all-model radar chart (Appendix).")
    parser.add_argument("--results-dir", default="prism/analysis/results",
                        help="Directory containing eval_*.json files")
    parser.add_argument("--output", "-o", default="prism/analysis/plots/radar_all.pdf",
                        help="Output image file path")
    args = parser.parse_args()

    df = load_results_df(args.results_dir)
    generate(df, args.output)
    print(f"Wrote {args.output}")
