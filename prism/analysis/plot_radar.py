# prism/analysis/plot_radar.py
"""Two-panel radar chart — (a) Dyadic and (b) Multi-Party (Figure 1)."""
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
    MIN_SAMPLE_THRESHOLD,
    MODEL_GROUP,
    MODEL_ORDER,
    display_name,
    model_color,
    model_marker,
    load_results_df,
    setup_style,
)

# Axis definitions: (label, category_filter, metric, invert)
# category_filter=None means compute across all categories in the panel.
# invert=True means the raw value is "lower is better" so we flip it (1 - val).

_DYADIC_AXES = [
    ("Cross-Domain\nPrivacy",    "cross_domain",       "leakage_rate", True),
    ("Mediated Comm.\nPrivacy",  "mediated_comm",      "leakage_rate", True),
    ("Cross-User\nPrivacy",      "cross_user",         "leakage_rate", True),
    ("Information\nAbstraction", None,                  "ias",          False),
    ("Task\nCompletion",         None,                  "tcq",          False),
    ("Behavioral\nSafety",       None,                  "safety",       False),
]

_MULTIPARTY_AXES = [
    ("Group Chat\nPrivacy",        "multi_party_group",   "leakage_rate", True),
    ("Hub-and-Spoke\nPrivacy",     "hub_and_spoke",       "leakage_rate", True),
    ("Competitive\nPrivacy",       "competitive",         "leakage_rate", True),
    ("Affinity\nCompliance",       "affinity_modulated",  "acs",          False),
    ("Information\nAbstraction",   None,                  "ias",          False),
    ("Task\nCompletion",           None,                  "tcq",          False),
]


def _compute_radar_values(
    df: pd.DataFrame,
    model: str,
    axes_config: list[tuple[str, str | None, str, bool]],
    categories: list[str],
) -> list[float]:
    """Compute radar values for a model using axis definitions.

    For axes with category_filter: filter to that category, compute metric.
    For axes with category_filter=None: compute across all listed categories.
    Special cases: 'acs' = 1 - leakage_rate; 'safety' = (4 - neg_count) / 4
    """
    values: list[float] = []
    model_df = df[df["model"] == model]
    cats_df = model_df[model_df["category"].isin(categories)]

    for _label, cat_filter, metric, invert in axes_config:
        if cat_filter is not None:
            sub = model_df[model_df["category"] == cat_filter]
        else:
            sub = cats_df

        if sub.empty:
            values.append(float("nan"))
            continue

        if metric == "safety":
            neg = sub["negative_count"].dropna()
            val = (4.0 - neg.mean()) / 4.0 if len(neg) > 0 else float("nan")
        elif metric == "acs":
            lr = sub["leakage_rate"].dropna()
            val = 1.0 - lr.mean() if len(lr) > 0 else float("nan")
        else:
            col = sub[metric].dropna()
            val = float(col.mean()) if len(col) > 0 else float("nan")

        if invert:
            val = 1.0 - val if not pd.isna(val) else float("nan")

        # Clamp to [0, 1] for display (preserve NaN)
        if not pd.isna(val):
            val = max(0.0, min(1.0, val))

        values.append(val)

    return values


def _draw_radar_panel(
    ax,
    df: pd.DataFrame,
    axes_config: list[tuple[str, str | None, str, bool]],
    categories: list[str],
    models: list[str],
    title: str,
) -> list[tuple]:
    """Draw radar polygons for a list of models on the given polar axes.

    Returns a list of (handle, label) pairs for legend construction.
    """
    labels = [a[0] for a in axes_config]
    n_axes = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    # Draw grid rings
    for level in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles_closed, [level] * (n_axes + 1), "grey",
                linewidth=0.4, linestyle="--", alpha=0.4, zorder=0)

    handles: list[tuple] = []
    for model in models:
        vals = _compute_radar_values(df, model, axes_config, categories)
        if all(pd.isna(v) for v in vals):
            continue

        color = model_color(model)
        marker = model_marker(model)
        label = display_name(model)

        # Replace NaN with 0 so the polygon closes; skip markers at NaN vertices
        vals_for_poly = [0.0 if pd.isna(v) else v for v in vals]
        vals_closed = vals_for_poly + vals_for_poly[:1]

        line, = ax.plot(angles_closed, vals_closed, linewidth=1.5, color=color,
                        marker=None, zorder=3)
        ax.fill(angles_closed, vals_closed, alpha=0.15, color=color, zorder=2)

        # Draw markers only at non-NaN vertices
        for i, (angle, val) in enumerate(zip(angles, vals)):
            if not pd.isna(val):
                ax.plot([angle], [val], color=color, marker=marker,
                        markersize=4, zorder=4)

        handles.append((line, label))

    # Configure axes
    ax.set_thetagrids(np.degrees(angles), labels, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=5, color="grey")
    ax.grid(True, alpha=0.3)
    ax.set_title(title, size=9, pad=15)

    return handles


def generate(df: pd.DataFrame, output_path: str) -> None:
    """Generate a two-panel radar chart and save to *output_path*."""
    df = df[df["privacy_mode"] == "unconstrained"].copy()
    setup_style()

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        subplot_kw=dict(polar=True),
        figsize=(FULL_WIDTH, FULL_WIDTH * 0.45),
    )

    dyadic_cats = ["cross_domain", "mediated_comm", "cross_user"]
    mp_cats = ["multi_party_group", "hub_and_spoke", "competitive", "affinity_modulated"]

    # Panel (a): Dyadic -- all models in MODEL_ORDER present in data
    all_models_in_data = set(df["model"].unique())
    dyadic_models = [m for m in MODEL_ORDER if m in all_models_in_data]
    dyadic_models += sorted(m for m in all_models_in_data if m not in MODEL_ORDER)

    _draw_radar_panel(ax1, df, _DYADIC_AXES, dyadic_cats,
                      models=dyadic_models, title="(a) Dyadic Categories")

    # Panel (b): Multi-party -- only models with sufficient data
    mp_models = [
        m for m in MODEL_ORDER
        if any(
            len(df[(df["model"] == m) & (df["category"] == c)]) >= MIN_SAMPLE_THRESHOLD
            for c in mp_cats
        )
    ]
    # Also include non-canonical models that meet the threshold
    mp_models += sorted(
        m for m in all_models_in_data
        if m not in MODEL_ORDER
        and any(
            len(df[(df["model"] == m) & (df["category"] == c)]) >= MIN_SAMPLE_THRESHOLD
            for c in mp_cats
        )
    )

    if mp_models:
        handles_b = _draw_radar_panel(ax2, df, _MULTIPARTY_AXES, mp_cats,
                                      models=mp_models,
                                      title="(b) Multi-Party Categories")
    else:
        ax2.set_title("(b) Multi-Party Categories", size=9, pad=15)
        ax2.text(0, 0, "No data", ha="center", va="center", fontsize=9)

    # Shared legend below -- merge handles from both panels, deduplicate
    handles_a = ax1.get_legend_handles_labels()
    handles_b_raw = ax2.get_legend_handles_labels()

    seen_labels: set[str] = set()
    merged_handles = []
    merged_labels = []
    for h_list, l_list in [(handles_a[0], handles_a[1]),
                           (handles_b_raw[0], handles_b_raw[1])]:
        for h, lbl in zip(h_list, l_list):
            if lbl not in seen_labels:
                seen_labels.add(lbl)
                merged_handles.append(h)
                merged_labels.append(lbl)

    if merged_handles:
        fig.legend(merged_handles, merged_labels, loc="lower center",
                   ncol=min(len(merged_labels), 6), framealpha=0.9, fontsize=7)

    fig.subplots_adjust(bottom=0.18, wspace=0.35)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate two-panel radar chart (Fig. 1).")
    parser.add_argument("--results-dir", default="prism/analysis/results",
                        help="Directory containing eval_*.json files")
    parser.add_argument("--output", "-o", default="prism/analysis/plots/radar_comparison.pdf",
                        help="Output image file path")
    args = parser.parse_args()

    df = load_results_df(args.results_dir)
    generate(df, args.output)
    print(f"Wrote {args.output}")
