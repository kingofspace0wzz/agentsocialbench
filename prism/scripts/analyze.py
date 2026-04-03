# prism/scripts/analyze.py
"""CLI entry point for PRISM analysis: print tables, save CSV/JSON, and plot results.

Usage:
    # Print summary tables to terminal
    python -m prism.scripts.analyze --results-dir prism/analysis/results

    # Save tables to CSV
    python -m prism.scripts.analyze --results-dir prism/analysis/results --save-csv prism/analysis/output

    # Save summary to JSON
    python -m prism.scripts.analyze --results-dir prism/analysis/results --save-json prism/analysis/output/summary.json

    # Generate plots
    python -m prism.scripts.analyze --results-dir prism/analysis/results --plot prism/analysis/plots

    # Everything at once
    python -m prism.scripts.analyze --results-dir prism/analysis/results --save-csv prism/analysis/output --save-json prism/analysis/output/summary.json --plot prism/analysis/plots
"""
import argparse
import csv
import json
import os

from prism.analysis.aggregate import (
    load_results,
    aggregate_by_model,
    aggregate_by_category,
    aggregate_by_privacy_mode,
    generate_summary,
)
from prism.analysis.pareto import compute_pareto_frontier


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_table(title: str, headers: list[str], rows: list[list], col_widths: list[int] | None = None):
    """Print a formatted ASCII table."""
    if not rows:
        print(f"\n{title}: (no data)\n")
        return

    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            w = len(h)
            for row in rows:
                w = max(w, len(str(row[i])))
            col_widths.append(w + 2)

    header_line = "".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
    sep = "".join("-" * w for w in col_widths)

    print(f"\n{title}")
    print(sep)
    print(header_line)
    print(sep)
    for row in rows:
        print("".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row)))
    print(sep)
    print()


def _fmt(val, decimals=3):
    """Format a numeric value."""
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


# ---------------------------------------------------------------------------
# Print functions
# ---------------------------------------------------------------------------

def print_by_model(by_model: dict):
    headers = ["Model", "N", "Leakage", "Full Leak", "IAS", "TCQ", "Task%", "Neg Beh", "Pos Beh"]
    rows = []
    for model, m in sorted(by_model.items()):
        rows.append([
            model, m["count"],
            _fmt(m["avg_leakage_rate"]), _fmt(m["avg_full_leakage_rate"]),
            _fmt(m["avg_ias"]), _fmt(m["avg_tcq"]),
            _fmt(m["task_completion_rate"] * 100, 1) + "%",
            _fmt(m["avg_negative_behaviors"], 2), _fmt(m["avg_positive_behaviors"], 2),
        ])
    _print_table("Results by Model", headers, rows)


def print_by_category(by_category: dict):
    headers = ["Category", "N", "Leakage", "IAS", "TCQ"]
    rows = []
    for cat, c in sorted(by_category.items()):
        rows.append([cat, c["count"], _fmt(c["avg_leakage_rate"]), _fmt(c["avg_ias"]), _fmt(c["avg_tcq"])])
    _print_table("Results by Category", headers, rows)


def print_by_privacy_mode(by_mode: dict):
    headers = ["Privacy Mode", "N", "Leakage", "IAS", "TCQ"]
    rows = []
    for mode, m in sorted(by_mode.items()):
        rows.append([mode, m["count"], _fmt(m["avg_leakage_rate"]), _fmt(m["avg_ias"]), _fmt(m["avg_tcq"])])
    _print_table("Results by Privacy Mode", headers, rows)


def print_per_scenario(results: list[dict]):
    headers = ["Scenario", "Model", "Mode", "Leakage", "Full Leak", "IAS", "TCQ", "Neg", "Pos"]
    rows = []
    for r in sorted(results, key=lambda x: x.get("scenario_id", "")):
        priv = r.get("privacy", {})
        abst = r.get("abstraction", {})
        tc = r.get("task_completion", {})
        beh = r.get("behavioral", {})
        rows.append([
            r.get("scenario_id", "?"),
            r.get("model", "?"),
            r.get("privacy_mode", "?"),
            _fmt(priv.get("aggregate_rate", 0)),
            _fmt(priv.get("full_leakage_rate", 0)),
            _fmt(abst.get("mean_ias", 0)),
            _fmt(tc.get("tcq_score", 0)),
            beh.get("negative_count", 0),
            beh.get("positive_count", 0),
        ])
    _print_table("Per-Scenario Results", headers, rows)


# ---------------------------------------------------------------------------
# Save functions
# ---------------------------------------------------------------------------

def _write_csv(path: str, headers: list[str], rows: list[list]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  Saved: {path}")


def save_csvs(results: list[dict], by_model: dict, by_category: dict, by_mode: dict, output_dir: str):
    """Save all tables as CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # Per-scenario
    headers = ["scenario_id", "model", "privacy_mode", "category", "leakage_rate", "full_leakage_rate",
               "ias", "tcq", "task_completed", "negative_behaviors", "positive_behaviors"]
    rows = []
    for r in sorted(results, key=lambda x: x.get("scenario_id", "")):
        priv = r.get("privacy", {})
        abst = r.get("abstraction", {})
        tc = r.get("task_completion", {})
        beh = r.get("behavioral", {})
        rows.append([
            r.get("scenario_id", ""), r.get("model", ""), r.get("privacy_mode", ""),
            r.get("category", ""),
            priv.get("aggregate_rate", 0), priv.get("full_leakage_rate", 0),
            abst.get("mean_ias", 0), tc.get("tcq_score", 0),
            tc.get("task_completed", False),
            beh.get("negative_count", 0), beh.get("positive_count", 0),
        ])
    _write_csv(os.path.join(output_dir, "per_scenario.csv"), headers, rows)

    # By model
    headers = ["model", "count", "avg_leakage_rate", "avg_full_leakage_rate", "avg_ias", "avg_tcq",
               "task_completion_rate", "avg_negative_behaviors", "avg_positive_behaviors"]
    rows = [[model] + [m[h] for h in headers[1:]] for model, m in sorted(by_model.items())]
    _write_csv(os.path.join(output_dir, "by_model.csv"), headers, rows)

    # By category
    headers = ["category", "count", "avg_leakage_rate", "avg_ias", "avg_tcq"]
    rows = [[cat] + [c[h] for h in headers[1:]] for cat, c in sorted(by_category.items())]
    _write_csv(os.path.join(output_dir, "by_category.csv"), headers, rows)

    # By privacy mode
    headers = ["privacy_mode", "count", "avg_leakage_rate", "avg_ias", "avg_tcq"]
    rows = [[mode] + [m[h] for h in headers[1:]] for mode, m in sorted(by_mode.items())]
    _write_csv(os.path.join(output_dir, "by_privacy_mode.csv"), headers, rows)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def save_plots(results: list[dict], by_model: dict, plot_dir: str):
    """Generate and save analysis plots. Requires matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping plots. Install with: pip install matplotlib")
        return

    os.makedirs(plot_dir, exist_ok=True)

    # 1. Bar chart: leakage rate by model
    models = sorted(by_model.keys())
    if models:
        leakage = [by_model[m]["avg_leakage_rate"] for m in models]
        tcq = [by_model[m]["avg_tcq"] for m in models]

        fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))
        x = range(len(models))
        w = 0.35
        ax.bar([i - w/2 for i in x], leakage, w, label="Avg Leakage Rate", color="#e74c3c")
        ax.bar([i + w/2 for i in x], tcq, w, label="Avg TCQ", color="#2ecc71")
        ax.set_xticks(list(x))
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Privacy Leakage vs Task Completion by Model")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        path = os.path.join(plot_dir, "model_leakage_vs_tcq.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    # 2. Scatter: Pareto frontier (leakage vs TCQ per scenario)
    xs = [r.get("privacy", {}).get("aggregate_rate", 0) for r in results]
    ys = [r.get("task_completion", {}).get("tcq_score", 0) for r in results]
    if xs and ys:
        points = [{"leakage": x, "tcq": y} for x, y in zip(xs, ys)]
        frontier = compute_pareto_frontier(points, x_key="leakage", y_key="tcq")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(xs, ys, alpha=0.5, label="Scenarios", color="#3498db")
        if frontier:
            fx = [p["leakage"] for p in frontier]
            fy = [p["tcq"] for p in frontier]
            ax.plot(fx, fy, "r-o", label="Pareto Frontier", markersize=6)
        ax.set_xlabel("Leakage Rate (lower is better)")
        ax.set_ylabel("TCQ Score (higher is better)")
        ax.set_title("Privacy-Utility Pareto Frontier")
        ax.legend()
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        path = os.path.join(plot_dir, "pareto_frontier.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    # 3. Behavioral analysis stacked bar
    if models:
        neg = [by_model[m]["avg_negative_behaviors"] for m in models]
        pos = [by_model[m]["avg_positive_behaviors"] for m in models]

        fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))
        x = range(len(models))
        ax.bar(x, neg, label="Avg Negative Behaviors", color="#e74c3c")
        ax.bar(x, pos, bottom=neg, label="Avg Positive Behaviors", color="#2ecc71")
        ax.set_xticks(list(x))
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylabel("Count")
        ax.set_title("Behavioral Analysis by Model")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(plot_dir, "behavioral_by_model.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    # 4. Privacy mode comparison
    from prism.analysis.aggregate import aggregate_by_privacy_mode
    by_mode = aggregate_by_privacy_mode(results)
    modes = sorted(by_mode.keys())
    if len(modes) >= 2:
        metrics = ["avg_leakage_rate", "avg_ias", "avg_tcq"]
        labels = ["Leakage", "IAS", "TCQ"]
        x = range(len(metrics))
        w = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, mode in enumerate(modes):
            vals = [by_mode[mode][m] for m in metrics]
            offset = (i - (len(modes) - 1) / 2) * w
            ax.bar([xi + offset for xi in x], vals, w, label=mode)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Score")
        ax.set_title("Implicit vs Explicit Privacy Mode")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        path = os.path.join(plot_dir, "privacy_mode_comparison.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PRISM Analysis — summarize, export, and plot evaluation results")
    parser.add_argument("--results-dir", default="prism/analysis/results",
                       help="Directory containing eval_*.json files (default: prism/analysis/results)")
    parser.add_argument("--save-csv", metavar="DIR",
                       help="Save summary tables as CSV files to this directory")
    parser.add_argument("--save-json", metavar="PATH",
                       help="Save full summary as a JSON file")
    parser.add_argument("--plot", metavar="DIR",
                       help="Generate and save plots to this directory (requires matplotlib)")
    parser.add_argument("--per-scenario", action="store_true",
                       help="Also print per-scenario breakdown")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress terminal table output (only save files)")

    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No evaluation results found in {args.results_dir}")
        return

    by_model = aggregate_by_model(results)
    by_category = aggregate_by_category(results)
    by_mode = aggregate_by_privacy_mode(results)

    print(f"Loaded {len(results)} evaluation results from {args.results_dir}")

    # Print tables
    if not args.quiet:
        print_by_model(by_model)
        print_by_category(by_category)
        print_by_privacy_mode(by_mode)
        if args.per_scenario:
            print_per_scenario(results)

    # Save CSV
    if args.save_csv:
        print(f"\nSaving CSVs to {args.save_csv}/")
        save_csvs(results, by_model, by_category, by_mode, args.save_csv)

    # Save JSON
    if args.save_json:
        summary = generate_summary(args.results_dir)
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {args.save_json}")

    # Plots
    if args.plot:
        print(f"\nGenerating plots in {args.plot}/")
        save_plots(results, by_model, args.plot)


if __name__ == "__main__":
    main()
