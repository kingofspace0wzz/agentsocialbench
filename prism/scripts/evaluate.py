# prism/scripts/evaluate.py
"""CLI entry point for PRISM evaluation.

Usage:
    python -m prism.scripts.evaluate --simulation path/to/sim.json --scenario path/to/scenario.json --llm gemini
    python -m prism.scripts.evaluate --batch-sim-dir prism/simulations/cross_domain/gemini/implicit --batch-scenario-dir prism/data/scenarios/cross_domain --llm openai
"""
import argparse
import glob
import json
import os

from prism.evaluation.evaluate import evaluate_simulation
from prism.core.json_utils import write_json


def run_single(simulation_path, scenario_path, llm_type, model, output_dir, cu_behavioral=False):
    """Run evaluation on a single simulation."""
    print(f"  Loading simulation: {simulation_path}")
    print(f"  Loading scenario: {scenario_path}")

    with open(simulation_path) as f:
        simulation = json.load(f)
    with open(scenario_path) as f:
        scenario = json.load(f)

    print(f"  Evaluating with {llm_type}...")
    result = evaluate_simulation(
        scenario, simulation, output_dir=output_dir,
        llm_type=llm_type, model=model, cu_behavioral=cu_behavioral,
    )

    # Print privacy results — per-user output for CULR, single aggregate for CDLR/MLR
    privacy = result["privacy"]
    if privacy["metric"] == "culr":
        print(f"  Privacy (CULR): aggregate={privacy['aggregate_rate']:.2f}, max={privacy['max_rate']:.2f}")
        for key in ("user_a", "user_b"):
            if key in privacy:
                u = privacy[key]
                print(f"    {u['user_name']}: leakage={u['leakage_rate']:.2f}, full={u['full_leakage_rate']:.2f}")
    else:
        print(f"  Privacy ({privacy['metric']}): aggregate={privacy['aggregate_rate']:.2f}, full={privacy['full_leakage_rate']:.2f}")
    print(f"  Abstraction (IAS): mean={result['abstraction']['mean_ias']:.2f}")
    print(f"  Task Completion (TCQ): score={result['task_completion']['tcq_score']:.2f}, completed={result['task_completion']['task_completed']}")
    print(f"  Behavioral: {result['behavioral']['negative_count']} negative, {result['behavioral']['positive_count']} positive")

    return result


def _eval_exists(sim_path, output_dir):
    """Check if evaluation result already exists for a simulation."""
    try:
        with open(sim_path) as f:
            sim = json.load(f)
        model_name = sim.get("model", "unknown")
        privacy_mode = sim.get("privacy_mode", "unknown")
        scenario_id = sim.get("scenario_id", "unknown")
        eval_path = os.path.join(output_dir, model_name, privacy_mode, f"eval_{scenario_id}.json")
        return os.path.exists(eval_path)
    except Exception:
        return False


def run_batch(sim_dir, scenario_dir, llm_type, model, output_dir, cu_behavioral=False):
    """Run evaluations for all simulations in a directory."""
    sim_files = sorted(glob.glob(os.path.join(sim_dir, "*.json")))
    print(f"Found {len(sim_files)} simulations in {sim_dir}")

    results = []
    skipped = 0
    for i, sim_path in enumerate(sim_files):
        sim_name = os.path.basename(sim_path)
        # Try to find matching scenario
        scenario_id = sim_name.replace("sim_", "").replace(".json", "")
        scenario_candidates = [
            os.path.join(scenario_dir, f"{scenario_id}.json"),
            os.path.join(scenario_dir, sim_name),
        ]

        scenario_path = None
        for candidate in scenario_candidates:
            if os.path.exists(candidate):
                scenario_path = candidate
                break

        if not scenario_path:
            print(f"\n[{i+1}/{len(sim_files)}] {sim_name} — SKIPPED (no matching scenario)")
            continue

        # Skip if evaluation result already exists
        if _eval_exists(sim_path, output_dir):
            skipped += 1
            print(f"\n[{i+1}/{len(sim_files)}] {sim_name} — SKIPPED (already evaluated)")
            continue

        print(f"\n[{i+1}/{len(sim_files)}] {sim_name}")
        try:
            result = run_single(sim_path, scenario_path, llm_type, model, output_dir, cu_behavioral=cu_behavioral)
            results.append({"simulation": sim_path, "status": "success"})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"simulation": sim_path, "status": f"error: {e}"})

    success = sum(1 for r in results if r["status"] == "success")
    print(f"\nBatch complete: {success}/{len(results)} succeeded, {skipped} skipped (already evaluated)")
    return results


def main():
    parser = argparse.ArgumentParser(description="PRISM Evaluation Runner")
    parser.add_argument("--simulation", help="Path to a single simulation JSON")
    parser.add_argument("--scenario", help="Path to the scenario JSON (for single mode)")
    parser.add_argument("--batch-sim-dir", help="Directory of simulation JSONs")
    parser.add_argument("--batch-scenario-dir", help="Directory of scenario JSONs")
    parser.add_argument("--llm", default="gemini", choices=["gemini", "openai", "together", "bedrock"],
                       help="LLM provider for evaluation judges (default: gemini)")
    parser.add_argument("--model", default=None, help="Specific model for evaluation")
    parser.add_argument("--output-dir", default="prism/analysis/results", help="Output directory for results")
    parser.add_argument("--cu-behavioral", action="store_true", default=False,
                       help="Enable CU-specific behavioral analysis for cross-user scenarios")

    args = parser.parse_args()

    print(f"PRISM Evaluation Runner")
    print(f"  Judge LLM: {args.llm} (model: {args.model or 'default'})")
    print()

    if args.simulation and args.scenario:
        run_single(args.simulation, args.scenario, args.llm, args.model, args.output_dir, cu_behavioral=args.cu_behavioral)
    elif args.batch_sim_dir and args.batch_scenario_dir:
        run_batch(args.batch_sim_dir, args.batch_scenario_dir, args.llm, args.model, args.output_dir, cu_behavioral=args.cu_behavioral)
    else:
        parser.error("Either (--simulation + --scenario) or (--batch-sim-dir + --batch-scenario-dir) required")


if __name__ == "__main__":
    main()
