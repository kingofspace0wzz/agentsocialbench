# prism/scripts/simulate.py
"""CLI entry point for PRISM simulations.

Usage:
    # Single scenario
    python -m prism.scripts.simulate --scenario prism/data/samples/cd_sample_01.json --llm gemini --privacy-mode implicit

    # Multiple scenarios
    python -m prism.scripts.simulate --scenario file1.json file2.json file3.json --llm openai

    # All scenarios in a directory
    python -m prism.scripts.simulate --batch-dir prism/data/scenarios/cross_domain --llm gemini --privacy-mode explicit

    # Filter scenarios in a directory by glob pattern
    python -m prism.scripts.simulate --batch-dir prism/data/scenarios/cross_domain --pattern "*health*" --llm openai

    # Combine multiple patterns
    python -m prism.scripts.simulate --batch-dir prism/data/scenarios/cross_domain --pattern "*health*" "*finance*" --llm gemini

    # Run only the first 10 scenarios in a directory
    python -m prism.scripts.simulate --batch-dir prism/data/scenarios/cross_domain --limit 10 --llm openai

    # Skip the first 10 and run the next 10 (scenarios 11-20)
    python -m prism.scripts.simulate --batch-dir prism/data/scenarios/cross_domain --offset 10 --limit 10 --llm openai
"""
import argparse
import glob
import json
import os
from datetime import datetime

from prism.core.llm import sanitize_model_name, resolve_model
from prism.simulation.simulation import PRISMSimulation
from prism.simulation.graph_simulation import GraphSimulation

NEW_CATEGORIES = {"multi_party_group", "hub_and_spoke", "competitive", "affinity_modulated"}


def run_single(scenario_path, llm_type, model, privacy_mode, max_rounds, output_dir):
    """Run a single simulation."""
    print(f"  Loading scenario: {scenario_path}")

    # Read category from scenario to determine engine
    with open(scenario_path) as f:
        scenario_data = json.load(f)
    category = scenario_data.get("category", "")

    if category in NEW_CATEGORIES:
        sim = GraphSimulation(
            llm_type=llm_type, model=model,
            privacy_mode=privacy_mode, max_rounds=max_rounds,
        )
        sim.load_scenario(scenario_path)
        print(f"  Category: {sim.category} (GraphSimulation)")
        print(f"  Agents: {list(sim.agents.keys())}")
        print(f"  Privacy mode: {privacy_mode}")
        print(f"  Running simulation...")
        sim.run()
        # Build output path
        scenario_id = sim.scenario.get("scenario_id", "unknown")
        model_folder = sanitize_model_name(sim.model or resolve_model(llm_type, model))
        out_subdir = os.path.join(output_dir, sim.category, llm_type, model_folder, privacy_mode)
        os.makedirs(out_subdir, exist_ok=True)
        out_path = sim.save(out_subdir)
        print(f"  Saved: {out_path}")
        return out_path
    else:
        sim = PRISMSimulation(
            llm_type=llm_type,
            model=model,
            privacy_mode=privacy_mode,
            max_rounds=max_rounds,
        )
        sim.load_scenario(scenario_path)
        sim.initialize_participants()

        print(f"  Category: {sim.category}")
        print(f"  Agents: {[a.name for a in sim.agents]}")
        print(f"  Humans: {[h.name for h in sim.humans]}")
        print(f"  Privacy mode: {privacy_mode}")
        print(f"  Running simulation...")

        result = sim.run()

        # Build output path
        scenario_id = result["scenario_id"]
        model_folder = sanitize_model_name(sim.model)
        out_subdir = os.path.join(output_dir, sim.category, llm_type, model_folder, privacy_mode)
        os.makedirs(out_subdir, exist_ok=True)
        out_path = os.path.join(out_subdir, f"sim_{scenario_id}.json")

        sim.save(out_path)
        print(f"  Saved: {out_path}")
        print(f"  Rounds: {result['num_rounds']}")
        return out_path


def run_batch(batch_dir, llm_type, model, privacy_mode, max_rounds, output_dir, patterns=None, offset=None, limit=None):
    """Run simulations for all scenarios in a directory, optionally filtered by glob patterns, offset, and/or limited."""
    if patterns:
        scenario_files = set()
        for pat in patterns:
            scenario_files.update(glob.glob(os.path.join(batch_dir, pat)))
        scenario_files = sorted(scenario_files)
        print(f"Found {len(scenario_files)} scenarios matching {patterns} in {batch_dir}")
    else:
        scenario_files = sorted(glob.glob(os.path.join(batch_dir, "*.json")))
        print(f"Found {len(scenario_files)} scenarios in {batch_dir}")

    if offset and offset < len(scenario_files):
        scenario_files = scenario_files[offset:]
        print(f"Skipped first {offset} scenarios")

    if limit and limit < len(scenario_files):
        scenario_files = scenario_files[:limit]
        print(f"Limited to {limit} scenarios")

    results = []
    for i, path in enumerate(scenario_files):
        print(f"\n[{i+1}/{len(scenario_files)}] {os.path.basename(path)}")
        try:
            out_path = run_single(path, llm_type, model, privacy_mode, max_rounds, output_dir)
            results.append({"scenario": path, "output": out_path, "status": "success"})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"scenario": path, "output": None, "status": f"error: {e}"})

    success = sum(1 for r in results if r["status"] == "success")
    print(f"\nBatch complete: {success}/{len(results)} succeeded")
    return results


def main():
    parser = argparse.ArgumentParser(description="PRISM Simulation Runner")
    parser.add_argument("--scenario", nargs="+", help="Path(s) to one or more scenario JSON files")
    parser.add_argument("--batch-dir", help="Directory of scenario JSON files to process")
    parser.add_argument("--pattern", nargs="+",
                       help="Glob pattern(s) to filter files in --batch-dir (e.g. '*health*' '*finance*')")
    parser.add_argument("--offset", type=int, default=None,
                       help="Skip first N sorted scenarios in --batch-dir before applying --limit")
    parser.add_argument("--limit", type=int, default=None,
                       help="Max number of scenarios to run from --batch-dir (after --offset)")
    parser.add_argument("--llm", default="gemini", choices=["gemini", "openai", "together", "bedrock"],
                       help="LLM provider (default: gemini)")
    parser.add_argument("--model", default=None, help="Specific model name")
    parser.add_argument("--privacy-mode", default="implicit",
                       choices=["unconstrained", "implicit", "explicit", "enhanced", "full_defense", "zdd"],
                       help="Privacy instruction level: unconstrained (L0), implicit (L1), explicit (L2), enhanced (L3), full_defense (L4), zdd (Zero-Data-Disclosure). Default: implicit")
    parser.add_argument("--max-rounds", type=int, default=10, help="Max simulation rounds (default: 10)")
    parser.add_argument("--output-dir", default="prism/simulations", help="Output directory")

    args = parser.parse_args()

    print(f"PRISM Simulation Runner")
    resolved = resolve_model(args.llm, args.model)
    print(f"  LLM: {args.llm} (model: {resolved})")
    print(f"  Privacy mode: {args.privacy_mode}")
    print(f"  Max rounds: {args.max_rounds}")
    print()

    if args.pattern and not args.batch_dir:
        parser.error("--pattern requires --batch-dir")
    if args.offset and not args.batch_dir:
        parser.error("--offset requires --batch-dir")

    if args.scenario:
        for scenario_path in args.scenario:
            run_single(scenario_path, args.llm, args.model, args.privacy_mode, args.max_rounds, args.output_dir)
    elif args.batch_dir:
        run_batch(args.batch_dir, args.llm, args.model, args.privacy_mode, args.max_rounds, args.output_dir,
                  patterns=args.pattern, offset=args.offset, limit=args.limit)
    else:
        parser.error("Either --scenario or --batch-dir is required")


if __name__ == "__main__":
    main()
