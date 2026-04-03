# prism/analysis/generate_all.py
"""CLI orchestrator that runs all table and plot generators for PRISM paper outputs.

Usage examples::

    python3 -m prism.analysis.generate_all
    python3 -m prism.analysis.generate_all --only tables
    python3 -m prism.analysis.generate_all --only plots
    python3 -m prism.analysis.generate_all --models gpt-5,claude
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate all PRISM analysis tables and plots.",
    )
    parser.add_argument(
        "--results-dir",
        default="prism/analysis/results",
        help="Directory containing eval_*.json result files (default: prism/analysis/results)",
    )
    parser.add_argument(
        "--tables-dir",
        default="latex/prism/tables",
        help="Output directory for LaTeX tables (default: latex/prism/tables)",
    )
    parser.add_argument(
        "--plots-dir",
        default="latex/prism/figures",
        help="Output directory for PDF plots (default: latex/prism/figures)",
    )
    parser.add_argument(
        "--only",
        choices=["tables", "plots"],
        default=None,
        help="Generate only tables or only plots (default: both)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of models to include (default: all)",
    )

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    from prism.analysis.loader import load_results_df

    df = load_results_df(args.results_dir)
    if df.empty:
        print(f"ERROR: No results found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Filter models if requested
    # ------------------------------------------------------------------
    if args.models:
        selected = [m.strip() for m in args.models.split(",")]
        df = df[df["model"].isin(selected)]
        if df.empty:
            print(
                f"ERROR: No results match --models {args.models}",
                file=sys.stderr,
            )
            sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Print summary
    # ------------------------------------------------------------------
    models = sorted(df["model"].unique())
    categories = sorted(df["category"].unique())
    print(f"Loaded {len(df)} evaluations")
    print(f"  Models:     {', '.join(models)}")
    print(f"  Categories: {', '.join(categories)}")

    # ------------------------------------------------------------------
    # 4. Create output directories
    # ------------------------------------------------------------------
    generate_tables = args.only in (None, "tables")
    generate_plots = args.only in (None, "plots")

    if generate_tables:
        os.makedirs(args.tables_dir, exist_ok=True)
    if generate_plots:
        os.makedirs(args.plots_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 5. Generate tables (lazy import to avoid matplotlib dep)
    # ------------------------------------------------------------------
    if generate_tables:
        from prism.analysis import table_main, table_defense, table_behavioral

        print("\nGenerating tables ...")

        path = os.path.join(args.tables_dir, "main_results.tex")
        table_main.generate(df, path)
        print(f"  -> {path}")

        path = os.path.join(args.tables_dir, "defense_ladder.tex")
        table_defense.generate(df, path)
        print(f"  -> {path}")

        path = os.path.join(args.tables_dir, "behavioral_freq.tex")
        table_behavioral.generate(df, path)
        print(f"  -> {path}")

    # ------------------------------------------------------------------
    # 6. Generate plots (lazy import so --only tables skips matplotlib)
    # ------------------------------------------------------------------
    if generate_plots:
        from prism.analysis import plot_radar, plot_domain_heatmap, plot_pareto
        from prism.analysis import plot_severity, plot_heatmap, plot_category
        from prism.analysis import plot_violin, plot_cu_asymmetry, plot_radar_all
        from prism.analysis import plot_privacy_mode
        from prism.analysis import plot_cm_scatter
        from prism.analysis import plot_privacy_mode_mp
        from prism.analysis import plot_zdd

        print("\nGenerating plots ...")

        path = os.path.join(args.plots_dir, "privacy_mode_effect.pdf")
        plot_privacy_mode.generate(df, path)
        print(f"  -> {path}")

        path = os.path.join(args.plots_dir, "radar_comparison.pdf")
        plot_radar.generate(df, path)
        print(f"  -> {path}")

        path = os.path.join(args.plots_dir, "domain_pair_heatmap.pdf")
        plot_domain_heatmap.generate(df, path)
        print(f"  -> {path}")

        path = os.path.join(args.plots_dir, "pareto_frontier.pdf")
        plot_pareto.generate(df, path)
        print(f"  -> {path}")

        path = os.path.join(args.plots_dir, "leakage_severity.pdf")
        plot_severity.generate(df, path)
        print(f"  -> {path}")

        path = os.path.join(args.plots_dir, "behavioral_heatmap.pdf")
        plot_heatmap.generate(df, path)
        print(f"  -> {path}")

        path = os.path.join(args.plots_dir, "category_comparison.pdf")
        plot_category.generate(df, path)
        print(f"  -> {path}")

        path = os.path.join(args.plots_dir, "violin_distribution.pdf")
        plot_violin.generate(df, path)
        print(f"  -> {path}")

        path = os.path.join(args.plots_dir, "cu_asymmetry.pdf")
        plot_cu_asymmetry.generate(df, path)
        print(f"  -> {path}")

        path = os.path.join(args.plots_dir, "radar_all.pdf")
        plot_radar_all.generate(df, path)
        print(f"  -> {path}")

        # CM competitive scatter
        path = os.path.join(args.plots_dir, "cm_competitive.pdf")
        print(f"  plot_cm_scatter -> {path}")
        plot_cm_scatter.generate(df, path)

        # Multi-party privacy mode effect
        path = os.path.join(args.plots_dir, "privacy_mode_mp.pdf")
        print(f"  plot_privacy_mode_mp -> {path}")
        plot_privacy_mode_mp.generate(df, path)

        # L0 vs ZDD comparison
        path = os.path.join(args.plots_dir, "zdd_comparison.pdf")
        print(f"  plot_zdd -> {path}")
        plot_zdd.generate(df, path)

    # ------------------------------------------------------------------
    # 7. Done
    # ------------------------------------------------------------------
    print("\nDone!")


if __name__ == "__main__":
    main()
