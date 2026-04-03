# prism/analysis/loader.py
"""Shared data loader, bootstrap CI, and formatting utilities for PRISM analysis.

This module provides the foundation for all analysis scripts:
- Loading eval JSON results into a pandas DataFrame
- Bootstrap confidence interval computation
- Aggregation with CI
- LaTeX formatting helpers
- Matplotlib/seaborn style configuration
"""
from __future__ import annotations

import glob
import json
import math
import os
from typing import Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GENERIC_NEGATIVE = [
    "oversharing",
    "cross_referencing",
    "implicit_disclosure",
    "unnecessary_context",
]

GENERIC_POSITIVE = [
    "appropriate_abstraction",
    "minimal_disclosure",
    "deflection",
    "boundary_maintenance",
]

CU_NEGATIVE = [
    "reciprocal_probing",
    "constraint_over_specification",
    "implicit_quid_pro_quo",
]

CU_POSITIVE = [
    "symmetric_restraint",
    "graceful_non_answer",
    "boundary_negotiation",
]

METRIC_DISPLAY = {
    "leakage_rate": "Leakage Rate",
    "full_leakage_rate": "Full Leakage Rate",
    "ias": "Info. Abstraction Score",
    "tcq": "Task Completion Quality",
    "task_completed": "Task Completed",
    "negative_count": "Neg. Behaviors",
    "positive_count": "Pos. Behaviors",
    "max_leakage": "Max Leakage",
    "user_a_leakage": "User A Leakage",
    "user_b_leakage": "User B Leakage",
    "cu_negative_count": "CU Neg. Behaviors",
    "cu_positive_count": "CU Pos. Behaviors",
    "rtc": "Rounds to Completion",
    "category_metric": "Category Metric",
}

MODEL_DISPLAY = {
    "deepseek.v3.2": "DeepSeek V3.2",
    "gpt-5-mini": "GPT-5 Mini",
    "qwen.qwen3-235b-a22b-2507-v1:0": "Qwen3-235B",
    "moonshotai.kimi-k2.5": "Kimi K2.5",
    "minimax.minimax-m2.1": "MiniMax M2.1",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": "Claude Haiku 4.5",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": "Claude Sonnet 4.5",
    "us.anthropic.claude-sonnet-4-6": "Claude Sonnet 4.6",
}

MODEL_COLORS = {
    "deepseek.v3.2": "#1f77b4",
    "gpt-5-mini": "#ff7f0e",
    "qwen.qwen3-235b-a22b-2507-v1:0": "#2ca02c",
    "moonshotai.kimi-k2.5": "#17becf",
    "minimax.minimax-m2.1": "#bcbd22",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": "#d62728",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": "#9467bd",
    "us.anthropic.claude-sonnet-4-6": "#8c564b",
}

MODEL_MARKERS = {
    "deepseek.v3.2": "o",
    "gpt-5-mini": "s",
    "qwen.qwen3-235b-a22b-2507-v1:0": "^",
    "moonshotai.kimi-k2.5": "v",
    "minimax.minimax-m2.1": "h",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": "D",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": "P",
    "us.anthropic.claude-sonnet-4-6": "X",
}

MODEL_ORDER = [
    "deepseek.v3.2",
    "qwen.qwen3-235b-a22b-2507-v1:0",
    "moonshotai.kimi-k2.5",
    "minimax.minimax-m2.1",
    "gpt-5-mini",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-sonnet-4-6",
]

MODEL_GROUP = {
    "deepseek.v3.2": "open",
    "gpt-5-mini": "closed",
    "qwen.qwen3-235b-a22b-2507-v1:0": "open",
    "moonshotai.kimi-k2.5": "closed",
    "minimax.minimax-m2.1": "closed",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": "closed",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": "closed",
    "us.anthropic.claude-sonnet-4-6": "closed",
}

PRIVACY_MODE_DISPLAY = {
    "unconstrained": "L0: Unconstrained",
    "implicit": "L1: Implicit",
    "explicit": "L2: Explicit",
    "enhanced": "L3: Enhanced",
    "full_defense": "L4: Full Defense",
    "zdd": "ZDD: Zero-Disclosure",
}

PRIVACY_MODE_ORDER = [
    "unconstrained", "implicit", "explicit", "enhanced", "full_defense",
    "zdd",  # after "full_defense"
]

CATEGORY_MARKERS = {
    "cross_domain": "o",
    "cross_user": "^",
    "mediated_comm": "s",
    "multi_party_group": "D",
    "hub_and_spoke": "P",
    "competitive": "X",
    "affinity_modulated": "v",
}

CATEGORY_DISPLAY = {
    "cross_domain": "Cross-Domain",
    "cross_user": "Cross-User",
    "mediated_comm": "Mediated Comm.",
    "multi_party_group": "Group Chat",
    "hub_and_spoke": "Hub-and-Spoke",
    "competitive": "Competitive",
    "affinity_modulated": "Affinity Modulated",
}

MIN_SAMPLE_THRESHOLD = 10  # Minimum eval files per model-category to include

SINGLE_COL_WIDTH = 3.25
FULL_WIDTH = 6.75
DPI = 300


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _safe_get(d: dict, *keys, default=None):
    """Navigate nested dict keys safely, returning *default* on any miss."""
    val = d
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
        if val is None:
            return default
    return val


def _parse_eval(result: dict) -> dict:
    """Extract a flat row dict from a single eval JSON object."""
    row: dict = {}

    # Top-level identifiers
    row["scenario_id"] = result.get("scenario_id", "")
    row["model"] = result.get("model", "unknown")
    row["category"] = result.get("category", "unknown")
    row["privacy_mode"] = result.get("privacy_mode", "unknown")

    # Privacy metrics
    priv = result.get("privacy", {})
    row["leakage_rate"] = priv.get("aggregate_rate", np.nan)
    row["ias"] = _safe_get(result, "abstraction", "mean_ias", default=np.nan)
    row["tcq"] = _safe_get(result, "task_completion", "tcq_score", default=np.nan)
    row["task_completed"] = _safe_get(result, "task_completion", "task_completed", default=False)

    # full_leakage_rate: present at top-level for CD, computed from users for CU
    if "full_leakage_rate" in priv:
        row["full_leakage_rate"] = priv["full_leakage_rate"]
    else:
        # CU: average of user-level full_leakage_rate
        ua_full = _safe_get(priv, "user_a", "full_leakage_rate", default=None)
        ub_full = _safe_get(priv, "user_b", "full_leakage_rate", default=None)
        if ua_full is not None and ub_full is not None:
            row["full_leakage_rate"] = (ua_full + ub_full) / 2.0
        else:
            row["full_leakage_rate"] = np.nan

    # Behavioral: negative/positive counts
    beh = result.get("behavioral", {})
    row["negative_count"] = beh.get("negative_count", 0)
    row["positive_count"] = beh.get("positive_count", 0)

    # Per-behavior boolean columns (generic)
    for bname in GENERIC_NEGATIVE:
        row[bname] = _safe_get(beh, "negative", bname, "observed", default=False)
    for bname in GENERIC_POSITIVE:
        row[bname] = _safe_get(beh, "positive", bname, "observed", default=False)

    # CU-specific columns
    is_cu = "user_a" in priv and "user_b" in priv
    if is_cu:
        row["user_a_leakage"] = _safe_get(priv, "user_a", "leakage_rate", default=np.nan)
        row["user_b_leakage"] = _safe_get(priv, "user_b", "leakage_rate", default=np.nan)
        ua = row["user_a_leakage"]
        ub = row["user_b_leakage"]
        if not (np.isnan(ua) if isinstance(ua, float) else False) and \
           not (np.isnan(ub) if isinstance(ub, float) else False):
            row["max_leakage"] = max(ua, ub)
        else:
            row["max_leakage"] = np.nan

        cu_spec = _safe_get(beh, "cu_specific", default={})
        if isinstance(cu_spec, dict) and cu_spec.get("enabled", False):
            for bname in CU_NEGATIVE:
                row[bname] = _safe_get(cu_spec, "negative", bname, "observed", default=False)
            for bname in CU_POSITIVE:
                row[bname] = _safe_get(cu_spec, "positive", bname, "observed", default=False)
            row["cu_negative_count"] = cu_spec.get("cu_negative_count", 0)
            row["cu_positive_count"] = cu_spec.get("cu_positive_count", 0)
        else:
            for bname in CU_NEGATIVE + CU_POSITIVE:
                row[bname] = np.nan
            row["cu_negative_count"] = np.nan
            row["cu_positive_count"] = np.nan
    else:
        row["user_a_leakage"] = np.nan
        row["user_b_leakage"] = np.nan
        row["max_leakage"] = np.nan
        for bname in CU_NEGATIVE + CU_POSITIVE:
            row[bname] = np.nan
        row["cu_negative_count"] = np.nan
        row["cu_positive_count"] = np.nan

    # Efficiency metrics
    eff = result.get("efficiency") or {}
    row["rtc"] = eff.get("rounds_to_completion", np.nan)
    row["max_rounds"] = eff.get("max_rounds", np.nan)
    row["completed_early"] = eff.get("completed_early", np.nan)

    # Category-specific metric (GC/HS/CM/AM): present only when evaluator emits it
    row["category_metric"] = priv.get("category_metric", np.nan)

    # Domain pair extraction for CD scenarios
    if row["category"] == "cross_domain":
        parts = row["scenario_id"].split("_")
        if len(parts) >= 4:
            row["source_domain"] = parts[1]
            row["target_domain"] = parts[2]
            row["domain_pair"] = f"{parts[1]}\u2192{parts[2]}"
        else:
            row["source_domain"] = np.nan
            row["target_domain"] = np.nan
            row["domain_pair"] = np.nan
    else:
        row["source_domain"] = np.nan
        row["target_domain"] = np.nan
        row["domain_pair"] = np.nan

    # Per-item leakage level counts
    items = priv.get("items", [])
    if not items and "user_a" in priv:
        items = (priv.get("user_a", {}).get("items", [])
                 + priv.get("user_b", {}).get("items", []))
    levels = [item.get("level", "none") for item in items]
    row["n_none"] = levels.count("none")
    row["n_partial"] = levels.count("partial")
    row["n_full"] = levels.count("full")
    row["n_items"] = len(levels)

    # CM-specific: extract separate CSLR and CER from details
    if row["category"] == "competitive":
        details = priv.get("details", {})
        row["cslr"] = details.get("cslr", float("nan"))
        row["cer"] = details.get("cer", float("nan"))
    else:
        row["cslr"] = float("nan")
        row["cer"] = float("nan")

    # AM-specific: extract per-tier ACS scores
    _AM_TIERS = ["close", "friend", "acquaintance", "stranger"]
    if row["category"] == "affinity_modulated":
        details = priv.get("details", {})
        tier_scores = details.get("tier_scores", {})

        if not tier_scores and items:
            # Fallback: compute from items for eval files that lack tier_scores
            from collections import defaultdict
            tier_items = defaultdict(list)
            for item in items:
                tier = item.get("tier", "unknown")
                tier = tier.lower().strip() if isinstance(tier, str) else "unknown"
                tier_items[tier].append(item)
            for tier, t_items in tier_items.items():
                t_total = len(t_items)
                t_viol = sum(1 for i in t_items if i.get("level") in ("partial", "full"))
                tier_scores[tier] = 1.0 - (t_viol / t_total) if t_total > 0 else 1.0

        for tier in _AM_TIERS:
            row[f"acs_{tier}"] = tier_scores.get(tier, float("nan"))
    else:
        for tier in _AM_TIERS:
            row[f"acs_{tier}"] = float("nan")

    return row


def load_results_df(results_dir: str = "prism/analysis/results") -> pd.DataFrame:
    """Load all ``eval_*.json`` files from *results_dir* into a DataFrame.

    Each row contains flat columns for the scenario's metrics, per-behavior
    booleans, and CU-specific fields (``NaN`` when not applicable).
    """
    rows: list[dict] = []
    pattern = os.path.join(results_dir, "**", "eval_*.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        try:
            with open(path) as f:
                data = json.load(f)
            rows.append(_parse_eval(data))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: skipping malformed result file {path}: {exc}")

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def compute_ci(
    series: pd.Series,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean of *series*.

    Returns ``(mean, ci_lo, ci_hi)``.

    Edge cases:
    - Empty series -> ``(NaN, NaN, NaN)``
    - Single value -> ``(val, val, val)``
    - Constant series -> ``(val, val, val)``
    """
    values = series.dropna().values
    n = len(values)

    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    if n == 1:
        v = float(values[0])
        return (v, v, v)

    mean = float(np.mean(values))

    # Constant series shortcut
    if np.all(values == values[0]):
        return (mean, mean, mean)

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - confidence
    ci_lo = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (mean, ci_lo, ci_hi)


def paired_bootstrap_test(
    series_a: pd.Series,
    series_b: pd.Series,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> float:
    """Two-sided paired bootstrap test. Returns p-value."""
    a = series_a.dropna().values
    b = series_b.dropna().values
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    a, b = a[:n], b[:n]
    observed_diff = float(np.mean(a) - np.mean(b))
    diffs = a - b
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=n, replace=True)
        if abs(np.mean(sample)) >= abs(observed_diff):
            count += 1
    return count / n_bootstrap


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_with_ci(
    df: pd.DataFrame,
    group_col: str,
    metric_cols: Sequence[str],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Group *df* by *group_col* and compute mean +/- CI for each metric.

    Returns a DataFrame indexed by the group values with columns:
    ``{metric}_mean``, ``{metric}_ci_lo``, ``{metric}_ci_hi``, and ``count``.
    """
    records: list[dict] = []
    for group_val, grp in df.groupby(group_col):
        rec: dict = {}
        for metric in metric_cols:
            m, lo, hi = compute_ci(grp[metric], confidence=confidence,
                                   n_bootstrap=n_bootstrap, seed=seed)
            rec[f"{metric}_mean"] = m
            rec[f"{metric}_ci_lo"] = lo
            rec[f"{metric}_ci_hi"] = hi
        rec["count"] = len(grp)
        rec[group_col] = group_val
        records.append(rec)

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.set_index(group_col)
    return out


# ---------------------------------------------------------------------------
# LaTeX formatting
# ---------------------------------------------------------------------------


def format_ci(
    mean: float,
    ci_lo: float,
    ci_hi: float,
    decimals: int = 2,
    pct: bool = False,
) -> str:
    r"""Format a value with CI for LaTeX tables.

    Output looks like: ``0.18 {\scriptsize $\pm$0.03}``

    If *pct* is True, values are multiplied by 100 first.
    """
    if math.isnan(mean):
        return "--"
    scale = 100.0 if pct else 1.0
    m = mean * scale
    half = (ci_hi - ci_lo) / 2.0 * scale
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(m)} {{\\scriptsize $\\pm${fmt.format(half)}}}"


def to_latex_table(
    body: str,
    caption: str,
    label: str,
    col_spec: str,
    resize: bool = False,
) -> str:
    r"""Wrap *body* rows in a full LaTeX table environment with booktabs.

    If *resize* is True, the tabular is wrapped in
    ``\resizebox{\columnwidth}{!}{...}`` to fit column width.
    """
    tabular = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        body.rstrip("\n"),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    tabular_str = "\n".join(tabular)

    if resize:
        inner = f"\\resizebox{{\\columnwidth}}{{!}}{{\n{tabular_str}\n}}"
    else:
        inner = tabular_str

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        inner,
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Matplotlib / Seaborn style
# ---------------------------------------------------------------------------


def setup_style() -> None:
    """Configure matplotlib and seaborn for publication-quality paper plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=0.9)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.framealpha": 0.9,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def display_name(model: str) -> str:
    """Return human-readable display name for a model."""
    return MODEL_DISPLAY.get(model, model)


def model_color(model: str) -> str:
    """Return consistent color for a model."""
    return MODEL_COLORS.get(model, "#7f7f7f")


def model_marker(model: str) -> str:
    """Return consistent marker for a model."""
    return MODEL_MARKERS.get(model, "o")


def privacy_mode_label(mode: str) -> str:
    """Return display label for a privacy mode."""
    return PRIVACY_MODE_DISPLAY.get(mode, mode)
