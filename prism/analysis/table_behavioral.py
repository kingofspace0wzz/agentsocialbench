# prism/analysis/table_behavioral.py
"""Generate the behavioral frequency LaTeX table — observation rates per model."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from prism.analysis.loader import (
    CU_NEGATIVE,
    CU_POSITIVE,
    GENERIC_NEGATIVE,
    GENERIC_POSITIVE,
    MODEL_ORDER,
    display_name,
    load_results_df,
    to_latex_table,
)

# Display names for generic behaviors
BEHAVIOR_DISPLAY: dict[str, str] = {
    # Generic negative
    "oversharing": "Oversharing",
    "cross_referencing": "Cross-referencing",
    "implicit_disclosure": "Implicit Disclosure",
    "unnecessary_context": "Unnecessary Context",
    # Generic positive
    "appropriate_abstraction": "Appropriate Abstraction",
    "minimal_disclosure": "Minimal Disclosure",
    "deflection": "Deflection",
    "boundary_maintenance": "Boundary Maintenance",
    # CU-specific negative
    "reciprocal_probing": "Reciprocal Probing",
    "constraint_over_specification": "Constraint Over-specification",
    "implicit_quid_pro_quo": "Implicit Quid Pro Quo",
    # CU-specific positive
    "symmetric_restraint": "Symmetric Restraint",
    "graceful_non_answer": "Graceful Non-answer",
    "boundary_negotiation": "Boundary Negotiation",
}


def _pct_cell(series: pd.Series) -> str:
    """Format the observation frequency as an integer percentage, or '--' if all NaN."""
    valid = series.dropna()
    if len(valid) == 0:
        return "--"
    # Behaviors are boolean (True/False); mean gives the fraction
    rate = valid.astype(float).mean()
    return f"{round(rate * 100)}\\%"


def _pct_rate(series: pd.Series) -> float | None:
    """Return the numeric rate (0–1) for a behavior series, or None if all NaN."""
    valid = series.dropna()
    if len(valid) == 0:
        return None
    return float(valid.astype(float).mean())


def _make_behavioral_row(
    beh: str,
    display: str,
    models: list[str],
    df: pd.DataFrame,
    lower_is_better: bool,
) -> str:
    """Build a LaTeX table row with bold-best highlighting for *beh*."""
    # Compute rates for all models
    rates: list[float | None] = []
    for model in models:
        model_df = df[df["model"] == model]
        col = model_df[beh] if beh in model_df.columns else pd.Series(dtype=float)
        rates.append(_pct_rate(col))

    # Determine the best numeric rate among non-None entries
    numeric_rates = [r for r in rates if r is not None]
    if numeric_rates:
        best_rate = min(numeric_rates) if lower_is_better else max(numeric_rates)
    else:
        best_rate = None

    cells: list[str] = [f"\\quad {display}"]
    for rate in rates:
        if rate is None:
            cells.append("--")
        else:
            cell_str = f"{round(rate * 100)}\\%"
            if best_rate is not None and abs(rate - best_rate) < 1e-9:
                cell_str = r"\textbf{" + cell_str + r"}"
            cells.append(cell_str)
    return " & ".join(cells) + r" \\"


def generate(df: pd.DataFrame, output_path: str) -> None:
    """Generate the behavioral frequency table and write it to *output_path*."""
    # Order models: MODEL_ORDER first, then any extras sorted alphabetically.
    models_in_data = set(df["model"].unique())
    models = [m for m in MODEL_ORDER if m in models_in_data]
    models += sorted(m for m in models_in_data if m not in MODEL_ORDER)

    n_models = len(models)
    total_cols = 1 + n_models  # behavior name + one column per model

    # Check whether CU-specific data exists (any non-NaN CU behavior)
    has_cu = False
    for beh in CU_NEGATIVE + CU_POSITIVE:
        if beh in df.columns and df[beh].notna().any():
            has_cu = True
            break

    # Build header using display names.
    header = "Behavior & " + " & ".join(display_name(m) for m in models) + r" \\"

    rows: list[str] = [header, r"\midrule"]

    # --- Generic Negative Behaviors ---
    rows.append(
        f"\\multicolumn{{{total_cols}}}{{l}}{{\\textit{{Negative Behaviors (Generic)}}}} \\\\"
    )
    for beh in GENERIC_NEGATIVE:
        display = BEHAVIOR_DISPLAY.get(beh, beh)
        rows.append(_make_behavioral_row(beh, display, models, df, lower_is_better=True))

    # --- Generic Positive Behaviors ---
    rows.append(r"\midrule")
    rows.append(
        f"\\multicolumn{{{total_cols}}}{{l}}{{\\textit{{Positive Behaviors (Generic)}}}} \\\\"
    )
    for beh in GENERIC_POSITIVE:
        display = BEHAVIOR_DISPLAY.get(beh, beh)
        rows.append(_make_behavioral_row(beh, display, models, df, lower_is_better=False))

    # --- CU-specific sections (only if CU data exists) ---
    if has_cu:
        rows.append(r"\midrule")
        rows.append(
            f"\\multicolumn{{{total_cols}}}{{l}}{{\\textit{{Negative Behaviors (Cross-User)}}}} \\\\"
        )
        for beh in CU_NEGATIVE:
            display = BEHAVIOR_DISPLAY.get(beh, beh)
            rows.append(_make_behavioral_row(beh, display, models, df, lower_is_better=True))

        rows.append(r"\midrule")
        rows.append(
            f"\\multicolumn{{{total_cols}}}{{l}}{{\\textit{{Positive Behaviors (Cross-User)}}}} \\\\"
        )
        for beh in CU_POSITIVE:
            display = BEHAVIOR_DISPLAY.get(beh, beh)
            rows.append(_make_behavioral_row(beh, display, models, df, lower_is_better=False))

    body = "\n".join(rows)
    col_spec = "@{}l" + "c" * n_models + "@{}"

    table = to_latex_table(
        body,
        caption="Behavioral observation frequencies per model (\\% of scenarios).",
        label="tab:behavioral_freq",
        col_spec=col_spec,
    )

    with open(output_path, "w") as f:
        f.write(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate behavioral frequency LaTeX table.")
    parser.add_argument("--results-dir", default="prism/analysis/results",
                        help="Directory containing eval_*.json files")
    parser.add_argument("--output", "-o", default="prism/analysis/output/table_behavioral.tex",
                        help="Output .tex file path")
    args = parser.parse_args()

    df = load_results_df(args.results_dir)
    generate(df, args.output)
    print(f"Wrote {args.output}")
