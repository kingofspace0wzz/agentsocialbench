# prism/analysis/table_defense.py
"""Generate the defense (privacy-mode) comparison LaTeX table.

Shows L0 (Unconstrained), L1 (Explicit), L2 (Full Defense) — the three
levels used in the main paper. Rows are split into Dyadic and Multi-Party
sections. No CI shown — just mean values.

Usage::

    python3 -m prism.analysis.table_defense
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from prism.analysis.loader import (
    MIN_SAMPLE_THRESHOLD,
    aggregate_with_ci,
    load_results_df,
    to_latex_table,
)

# The three levels shown in the paper (mapping code name → display label)
_MODES = [
    ("unconstrained", "L0: Unconstrained"),
    ("explicit", "L1: Explicit"),
    ("full_defense", "L2: Full Defense"),
]

_DYADIC_CATS = {"cross_domain", "mediated_comm", "cross_user"}
_MP_CATS = {"multi_party_group", "hub_and_spoke", "competitive", "affinity_modulated"}

METRICS: list[str] = ["leakage_rate", "ias", "tcq", "task_completed"]

HEADERS: list[str] = [
    r"Leakage$\downarrow$",
    r"IAS$\uparrow$",
    r"TCQ$\uparrow$",
    r"Task\%$\uparrow$",
]

FMT: list[tuple[int, bool]] = [
    (2, False),
    (2, False),
    (2, False),
    (1, True),
]

_LOWER_IS_BETTER = {"leakage_rate"}


def _format_val(val: float, decimals: int = 2, pct: bool = False) -> str:
    if math.isnan(val):
        return "--"
    scale = 100.0 if pct else 1.0
    return f"{val * scale:.{decimals}f}"


def _format_delta(val: float, decimals: int = 2, pct: bool = False) -> str:
    if math.isnan(val):
        return "--"
    scale = 100.0 if pct else 1.0
    if abs(val) < 0.005:
        val = 0.0
    return f"{val * scale:+.{decimals}f}"


def _build_section_rows(df: pd.DataFrame, modes_present: list[str]) -> list[str]:
    """Build rows for a group of categories (dyadic or multi-party)."""
    agg = aggregate_with_ci(df, "privacy_mode", METRICS)

    # Best per column for bolding
    best: dict[str, float] = {}
    for metric in METRICS:
        col = f"{metric}_mean"
        vals = [agg.loc[m, col] for m in modes_present
                if m in agg.index and not math.isnan(agg.loc[m, col])]
        if vals:
            best[metric] = min(vals) if metric in _LOWER_IS_BETTER else max(vals)

    rows: list[str] = []
    for mode_code, mode_label in _MODES:
        if mode_code not in modes_present:
            continue
        cells: list[str] = [mode_label]
        for metric, (decimals, pct) in zip(METRICS, FMT):
            if mode_code in agg.index:
                mean = agg.loc[mode_code, f"{metric}_mean"]
                cell = _format_val(mean, decimals, pct)
                if (metric in best and not math.isnan(mean)
                        and abs(mean - best[metric]) < 1e-9):
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)
            else:
                cells.append("--")
        rows.append(" & ".join(cells) + r" \\")

    # Delta row
    first = _MODES[0][0]
    last = _MODES[-1][0]
    if first in modes_present and last in modes_present and first in agg.index and last in agg.index:
        rows.append(r"\midrule")
        cells = [rf"$\Delta$ L0$\to$L2"]
        for metric, (decimals, pct) in zip(METRICS, FMT):
            fm = agg.loc[first, f"{metric}_mean"]
            lm = agg.loc[last, f"{metric}_mean"]
            if math.isnan(fm) or math.isnan(lm):
                cells.append("--")
            else:
                cells.append(_format_delta(lm - fm, decimals, pct))
        rows.append(" & ".join(cells) + r" \\")

    return rows


def _build_side_by_side_rows(
    dyadic_df: pd.DataFrame, mp_df: pd.DataFrame, modes_present: list[str],
) -> list[str]:
    """Build rows with Dyadic and Multi-Party as side-by-side column groups."""
    d_agg = aggregate_with_ci(dyadic_df, "privacy_mode", METRICS)
    m_agg = aggregate_with_ci(mp_df, "privacy_mode", METRICS)

    # Best per column for bolding (dyadic)
    d_best: dict[str, float] = {}
    for metric in METRICS:
        col = f"{metric}_mean"
        vals = [d_agg.loc[m, col] for m in modes_present
                if m in d_agg.index and not math.isnan(d_agg.loc[m, col])]
        if vals:
            d_best[metric] = min(vals) if metric in _LOWER_IS_BETTER else max(vals)
    # Best per column for bolding (multi-party)
    m_best: dict[str, float] = {}
    for metric in METRICS:
        col = f"{metric}_mean"
        vals = [m_agg.loc[m, col] for m in modes_present
                if m in m_agg.index and not math.isnan(m_agg.loc[m, col])]
        if vals:
            m_best[metric] = min(vals) if metric in _LOWER_IS_BETTER else max(vals)

    rows: list[str] = []
    for mode_code, mode_label in _MODES:
        if mode_code not in modes_present:
            continue
        cells: list[str] = [mode_label]
        # Dyadic columns
        for metric, (decimals, pct) in zip(METRICS, FMT):
            if mode_code in d_agg.index:
                mean = d_agg.loc[mode_code, f"{metric}_mean"]
                cell = _format_val(mean, decimals, pct)
                if (metric in d_best and not math.isnan(mean)
                        and abs(mean - d_best[metric]) < 1e-9):
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)
            else:
                cells.append("--")
        # Multi-party columns
        for metric, (decimals, pct) in zip(METRICS, FMT):
            if mode_code in m_agg.index:
                mean = m_agg.loc[mode_code, f"{metric}_mean"]
                cell = _format_val(mean, decimals, pct)
                if (metric in m_best and not math.isnan(mean)
                        and abs(mean - m_best[metric]) < 1e-9):
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)
            else:
                cells.append("--")
        rows.append(" & ".join(cells) + r" \\")

    # Delta row
    first = _MODES[0][0]
    last = _MODES[-1][0]
    if first in modes_present and last in modes_present:
        rows.append(r"\midrule")
        cells = [rf"$\Delta$ L0$\to$L2"]
        for agg in (d_agg, m_agg):
            for metric, (decimals, pct) in zip(METRICS, FMT):
                if first in agg.index and last in agg.index:
                    fm = agg.loc[first, f"{metric}_mean"]
                    lm = agg.loc[last, f"{metric}_mean"]
                    if math.isnan(fm) or math.isnan(lm):
                        cells.append("--")
                    else:
                        cells.append(_format_delta(lm - fm, decimals, pct))
                else:
                    cells.append("--")
        rows.append(" & ".join(cells) + r" \\")

    return rows


def generate(df: pd.DataFrame, output_path: str) -> None:
    """Generate the defense comparison table with side-by-side dyadic/multi-party."""
    modes_in_data = [m for m, _ in _MODES if m in df["privacy_mode"].unique()]

    dyadic_df = df[df["category"].isin(_DYADIC_CATS)]
    mp_df = df[df["category"].isin(_MP_CATS)]

    # Short headers for compact side-by-side layout
    short_headers = [r"Leak.$\downarrow$", r"IAS$\uparrow$",
                     r"TCQ$\uparrow$", r"Task\%$\uparrow$"]

    header_parts = [
        r" & \multicolumn{4}{c}{\textit{Dyadic (CD, MC, CU)}}"
        r" & \multicolumn{4}{c}{\textit{Multi-Party (GC, HS, CM, AM)}} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        "Privacy Level & " + " & ".join(short_headers) + " & "
        + " & ".join(short_headers) + r" \\",
    ]
    all_rows: list[str] = header_parts + [r"\midrule"]
    all_rows.extend(_build_side_by_side_rows(dyadic_df, mp_df, modes_in_data))

    body = "\n".join(all_rows)
    n_cols = 1 + 2 * len(METRICS)  # 9
    col_spec = "@{}l" + "c" * (n_cols - 1) + "@{}"

    caption = (
        r"Privacy instruction level comparison. "
        r"Leakage: aggregate rate across all categories (lower is better). "
        r"IAS, TCQ, Task\%: higher is better. "
        r"Multi-party evaluated on 6 of 8 models due to data availability."
    )

    table = to_latex_table(
        body,
        caption=caption,
        label="tab:defense_results",
        col_spec=col_spec,
    )

    with open(output_path, "w") as f:
        f.write(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate defense comparison LaTeX table.")
    parser.add_argument("--results-dir", default="prism/analysis/results",
                        help="Directory containing eval_*.json files")
    parser.add_argument("--output", "-o", default="prism/analysis/output/table_defense.tex",
                        help="Output .tex file path")
    args = parser.parse_args()

    df = load_results_df(args.results_dir)
    generate(df, args.output)
    print(f"Wrote {args.output}")
