# prism/analysis/table_main.py
"""Generate the main results LaTeX table — one row per model, all core metrics with 95% CI.

Columns: Model | CDLR↓ | MLR↓ | CULR↓ | MPLR↓ | HALR↓ | CSLR↓ | ACS↑ | IAS↑ | TCQ↑ | Task%↑
Rows are grouped open-source first, then a midrule, then closed-source.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from prism.analysis.loader import (
    MIN_SAMPLE_THRESHOLD,
    MODEL_GROUP,
    MODEL_ORDER,
    aggregate_with_ci,
    compute_ci,
    display_name,
    format_ci,
    load_results_df,
    to_latex_table,
)

# GPT-5 Mini identifier — no mediated_comm (MC) data for this model.
_GPT5_MINI = "gpt-5-mini"

# Global metrics computed across all categories.
_GLOBAL_METRICS = ["ias", "tcq", "task_completed"]

# Multi-party categories — MIN_SAMPLE_THRESHOLD applies only to these.
_MP_CATEGORIES = {"multi_party_group", "hub_and_spoke", "competitive", "affinity_modulated"}

# Column headers with arrows.
HEADERS: list[str] = [
    r"CDLR$\downarrow$",
    r"MLR$\downarrow$",
    r"CULR$\downarrow$",
    r"MPLR$\downarrow$",
    r"HALR$\downarrow$",
    r"CSLR$\downarrow$",
    r"ACS$\uparrow$",
    r"IAS$\uparrow$",
    r"TCQ$\uparrow$",
    r"Task\%$\uparrow$",
]

# Logical column order matching headers.
# Each entry is (column_key, lower_is_better, decimals, pct).
_COLUMNS: list[tuple[str, bool, int, bool]] = [
    ("cdlr",           True,  2, False),
    ("mlr",            True,  2, False),
    ("culr",           True,  2, False),
    ("mplr",           True,  2, False),
    ("halr",           True,  2, False),
    ("cslr",           True,  2, False),
    ("acs",            False, 2, False),   # ACS is higher-is-better
    ("ias",            False, 2, False),
    ("tcq",            False, 2, False),
    ("task_completed", False, 1, True),    # must match DataFrame column name
]


def _fmt(mean: float, lo: float, hi: float, decimals: int, pct: bool) -> str:
    """Format a cell; return '--' when mean is NaN."""
    return format_ci(mean, lo, hi, decimals=decimals, pct=pct)


def _compute_category_leakage(
    df: pd.DataFrame, model: str, category: str
) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) for leakage_rate in a given category for one model.

    For multi-party categories, returns NaN when sample count is below MIN_SAMPLE_THRESHOLD.
    """
    sub = df[(df["model"] == model) & (df["category"] == category)]
    if category in _MP_CATEGORIES and len(sub) < MIN_SAMPLE_THRESHOLD:
        return (float("nan"), float("nan"), float("nan"))
    if sub.empty:
        return (float("nan"), float("nan"), float("nan"))
    return compute_ci(sub["leakage_rate"].dropna())


def _compute_acs(df: pd.DataFrame, model: str) -> tuple[float, float, float]:
    """Compute ACS = 1 - leakage_rate for affinity_modulated scenarios."""
    sub = df[(df["model"] == model) & (df["category"] == "affinity_modulated")]
    if len(sub) < MIN_SAMPLE_THRESHOLD:
        return (float("nan"), float("nan"), float("nan"))
    vals = 1.0 - sub["leakage_rate"].dropna()
    return compute_ci(vals)


def generate(df: pd.DataFrame, output_path: str) -> None:
    """Generate the main results table and write it to *output_path*."""
    # Main results table shows baseline (unconstrained) behavior only.
    df = df[df["privacy_mode"] == "unconstrained"].copy()

    # Filter to only models that appear in both MODEL_ORDER and the data.
    models_in_data = set(df["model"].unique())
    ordered_models = [m for m in MODEL_ORDER if m in models_in_data]

    # Also include any models not in MODEL_ORDER (appended at end, open group).
    extra_models = sorted(m for m in models_in_data if m not in MODEL_ORDER)

    open_models = [m for m in ordered_models if MODEL_GROUP.get(m, "open") == "open"]
    closed_models = [m for m in ordered_models if MODEL_GROUP.get(m, "open") == "closed"]

    # Append extra models to open group (unknown models treated as open).
    open_models = open_models + extra_models

    all_models = open_models + closed_models

    # ------------------------------------------------------------------
    # Compute global metrics per model via aggregate_with_ci.
    # ------------------------------------------------------------------
    global_agg = aggregate_with_ci(df, "model", _GLOBAL_METRICS)

    # ------------------------------------------------------------------
    # Compute per-category leakage per model.
    # ------------------------------------------------------------------
    cat_leakage: dict[str, dict[str, tuple[float, float, float]]] = {}
    for model in all_models:
        cat_leakage[model] = {
            "cdlr": _compute_category_leakage(df, model, "cross_domain"),
            "mlr":  _compute_category_leakage(df, model, "mediated_comm"),
            "culr": _compute_category_leakage(df, model, "cross_user"),
            "mplr": _compute_category_leakage(df, model, "multi_party_group"),
            "halr": _compute_category_leakage(df, model, "hub_and_spoke"),
            "cslr": _compute_category_leakage(df, model, "competitive"),
            "acs":  _compute_acs(df, model),
        }

    # Categorical column keys (sourced from cat_leakage, not global_agg).
    _CAT_COLS = {"cdlr", "mlr", "culr", "mplr", "halr", "cslr", "acs"}

    # ------------------------------------------------------------------
    # Gather all numeric values per column to find the best.
    # ------------------------------------------------------------------
    col_means: dict[str, list[float]] = {col: [] for col, *_ in _COLUMNS}
    for model in all_models:
        for col, lower_is_better, decimals, pct in _COLUMNS:
            if col in _CAT_COLS:
                # GPT-5 Mini has no MLR data → force NaN so it isn't "best"
                if col == "mlr" and model == _GPT5_MINI:
                    col_means[col].append(float("nan"))
                else:
                    mean, *_ = cat_leakage[model][col]
                    col_means[col].append(mean)
            else:
                if model in global_agg.index:
                    mean = global_agg.loc[model, f"{col}_mean"]
                else:
                    mean = float("nan")
                col_means[col].append(mean)

    def _best_idx(col: str, lower: bool) -> int:
        vals = col_means[col]
        if lower:
            return min(range(len(vals)),
                       key=lambda i: vals[i] if not math.isnan(vals[i]) else float("inf"))
        else:
            return max(range(len(vals)),
                       key=lambda i: vals[i] if not math.isnan(vals[i]) else float("-inf"))

    best: dict[str, int] = {}
    for col, lower_is_better, *_ in _COLUMNS:
        best[col] = _best_idx(col, lower_is_better)

    # ------------------------------------------------------------------
    # Build table rows.
    # ------------------------------------------------------------------
    header = "Model & " + " & ".join(HEADERS) + r" \\"
    rows: list[str] = [header, r"\midrule"]

    for group_models in (open_models, closed_models):
        if not group_models:
            continue
        for model in group_models:
            # Find absolute position in all_models for best detection.
            abs_pos = all_models.index(model)

            cells: list[str] = [display_name(model)]

            for col, lower_is_better, decimals, pct in _COLUMNS:
                # Retrieve value.
                if col in _CAT_COLS:
                    if col == "mlr" and model == _GPT5_MINI:
                        cell = "--"
                    else:
                        mean, lo, hi = cat_leakage[model][col]
                        cell = _fmt(mean, lo, hi, decimals, pct)
                else:
                    if model in global_agg.index:
                        mean = global_agg.loc[model, f"{col}_mean"]
                        lo = global_agg.loc[model, f"{col}_ci_lo"]
                        hi = global_agg.loc[model, f"{col}_ci_hi"]
                    else:
                        mean, lo, hi = float("nan"), float("nan"), float("nan")
                    cell = _fmt(mean, lo, hi, decimals, pct)

                # Bold if best and cell is not "--".
                if cell != "--" and best[col] == abs_pos:
                    cell = r"\textbf{" + cell + "}"

                cells.append(cell)

            rows.append(" & ".join(cells) + r" \\")

        # Midrule between open and closed groups (only if both exist).
        if group_models is open_models and closed_models:
            rows.append(r"\midrule")

    body = "\n".join(rows)
    n_cols = 1 + len(_COLUMNS)
    col_spec = "@{}l" + "c" * len(_COLUMNS) + "@{}"

    caption = (
        "Main results under L0 (unconstrained) with 95\\% confidence intervals. "
        "Privacy metrics (CDLR, MLR, CULR, MPLR, HALR, CSLR): lower is better. "
        "ACS (Affinity Compliance Score): higher is better. "
        "Utility metrics (IAS, TCQ, Task\\%): higher is better. "
        "\\texttt{--} indicates insufficient data ($n < 10$)."
    )

    table = to_latex_table(
        body,
        caption=caption,
        label="tab:main_results",
        col_spec=col_spec,
        resize=True,
    )

    with open(output_path, "w") as f:
        f.write(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate main results LaTeX table.")
    parser.add_argument("--results-dir", default="prism/analysis/results",
                        help="Directory containing eval_*.json files")
    parser.add_argument("--output", "-o", default="prism/analysis/output/table_main.tex",
                        help="Output .tex file path")
    args = parser.parse_args()

    df = load_results_df(args.results_dir)
    generate(df, args.output)
    print(f"Wrote {args.output}")
