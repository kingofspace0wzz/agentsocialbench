# prism/scripts/generate.py
"""CLI entry point for PRISM dataset generation.

Usage:
    python -m prism.scripts.generate --category cd --count 10 --llm gemini
    python -m prism.scripts.generate --category mc --count 10 --llm gemini
    python -m prism.scripts.generate --category cu --count 5 --llm gemini
    python -m prism.scripts.generate --profiles-only --count 20
"""
import argparse
import glob
import os
import time
from datetime import datetime

from prism.core.json_utils import write_json
from prism.generation.profiles import generate_profile, generate_group_profiles
from prism.generation.attribute_pools import sample_skeletons
from prism.generation.scenarios import (
    generate_cd_scenario, generate_mc_scenario, generate_cu_scenario, validate_scenario
)
from prism.generation.scenarios_extended import (
    generate_mg_scenario, generate_hs_scenario, generate_cm_scenario, generate_am_scenario,
    MG_GROUP_TYPES, HS_HUB_TYPES, CM_COMPETITION_TYPES, AM_SOCIAL_CONTEXTS,
    validate_extended_scenario,
)
from prism.generation.verify import verify_scenario

# Domain pairs for CD scenarios
CD_DOMAIN_PAIRS = [
    ("health", "social"), ("health", "finance"), ("health", "schedule"),
    ("finance", "social"), ("finance", "schedule"), ("finance", "professional"),
    ("social", "schedule"), ("social", "professional"),
    ("professional", "schedule"), ("lifestyle", "health"),
    ("lifestyle", "social"), ("lifestyle", "finance"),
]

# Mediation types for MC scenarios
MC_MEDIATION_TYPES = [
    "group_event_planning", "information_brokering", "conflict_mediation",
    "recommendation_sharing", "schedule_coordination", "social_introductions",
]

# Interaction types for CU scenarios
CU_INTERACTION_TYPES = [
    "shared_activity", "resource_sharing", "joint_planning",
    "information_exchange", "social_introduction",
]


def _next_profile_index(output_dir: str) -> int:
    """Find the next available profile index by scanning existing files."""
    existing = glob.glob(os.path.join(output_dir, "profile_*.json"))
    if not existing:
        return 1
    indices = []
    for path in existing:
        basename = os.path.basename(path)
        # Extract number from "profile_NNN.json"
        try:
            num = int(basename.replace("profile_", "").replace(".json", ""))
            indices.append(num)
        except ValueError:
            continue
    return max(indices) + 1 if indices else 1


def generate_profiles_batch(count, llm_type, model, output_dir, seed=None):
    """Generate a batch of user profiles with diverse pre-sampled skeletons.

    New profiles are appended after existing ones (no overwriting).
    Uses a time-based seed by default so each run produces different skeletons.
    """
    os.makedirs(output_dir, exist_ok=True)
    start_index = _next_profile_index(output_dir)

    if seed is None:
        seed = int(time.time() * 1000) % (2**31)
    print(f"  Skeleton seed: {seed}  (starting at profile_{start_index:03d})")

    skeletons = sample_skeletons(count, seed=seed)
    profiles = []
    for i, skeleton in enumerate(skeletons):
        file_index = start_index + i
        print(f"  Generating profile {i+1}/{count} ({skeleton['name']}, {skeleton['age']}, {skeleton['location']})...")
        try:
            profile = generate_profile(llm_type=llm_type, model=model, skeleton=skeleton)
            path = os.path.join(output_dir, f"profile_{file_index:03d}.json")
            write_json(path, profile)
            profiles.append(profile)
            print(f"    Saved to {path}")
        except Exception as e:
            print(f"    Failed: {e}")
    return profiles


def generate_scenarios_batch(category, count, llm_type, model, output_dir, verify_flag=True, seed=None,
                             group_type=None, hub_type=None, competition_type=None, social_context=None):
    """Generate and optionally verify a batch of scenarios."""
    category_dirs = {
        "cd": "cross_domain", "mc": "mediated_comm", "cu": "cross_user",
        "mg": "multi_party_group", "hs": "hub_and_spoke", "cm": "competitive", "am": "affinity_modulated",
    }
    subdir = os.path.join(output_dir, category_dirs[category])
    os.makedirs(subdir, exist_ok=True)

    # New categories use 3-5 profiles per scenario; existing use 1-2
    NEW_CATS = {"mg", "hs", "cm", "am"}
    profiles_per_scenario = 5 if category in NEW_CATS else 2

    if seed is None:
        seed = int(time.time() * 1000) % (2**31)
    print(f"  Skeleton seed: {seed}")

    # Pre-sample enough skeletons
    max_profiles = count * 3 * profiles_per_scenario
    skeletons = sample_skeletons(max_profiles, seed=seed)
    skeleton_idx = 0

    def _next_skeleton():
        nonlocal skeleton_idx
        skel = skeletons[skeleton_idx % len(skeletons)]
        skeleton_idx += 1
        return skel

    generated = 0
    attempts = 0
    max_attempts = count * 3  # allow 3x attempts for verification failures

    while generated < count and attempts < max_attempts:
        attempts += 1
        print(f"  Generating {category.upper()} scenario {generated+1}/{count} (attempt {attempts})...")

        try:
            is_new_cat = category in NEW_CATS
            if category == "cd":
                # Generate a profile and pick a domain pair
                profile = generate_profile(llm_type=llm_type, model=model, skeleton=_next_skeleton())
                pair = CD_DOMAIN_PAIRS[generated % len(CD_DOMAIN_PAIRS)]
                scenario = generate_cd_scenario(profile, pair[0], pair[1], llm_type=llm_type, model=model)
                cat_name = "cross_domain"
            elif category == "mc":
                profile_a = generate_profile(llm_type=llm_type, model=model, skeleton=_next_skeleton())
                profile_b = generate_profile(llm_type=llm_type, model=model, skeleton=_next_skeleton())
                med_type = MC_MEDIATION_TYPES[generated % len(MC_MEDIATION_TYPES)]
                scenario = generate_mc_scenario(profile_a, profile_b, med_type, llm_type=llm_type, model=model)
                cat_name = "mediated_comm"
            elif category == "cu":
                profile_a = generate_profile(llm_type=llm_type, model=model, skeleton=_next_skeleton())
                profile_b = generate_profile(llm_type=llm_type, model=model, skeleton=_next_skeleton())
                int_type = CU_INTERACTION_TYPES[generated % len(CU_INTERACTION_TYPES)]
                scenario = generate_cu_scenario(profile_a, profile_b, int_type, llm_type=llm_type, model=model)
                cat_name = "cross_user"
            elif category == "mg":
                # Stage 1: Generate coherent group profiles
                g_type = group_type or MG_GROUP_TYPES[generated % len(MG_GROUP_TYPES)]
                print(f"    Generating coherent group profiles ({g_type}, 4 members)...")
                profiles = generate_group_profiles(g_type, count=4, llm_type=llm_type, model=model)
                # Stage 2: Generate scenario structure around those profiles
                scenario = generate_mg_scenario(profiles, g_type, llm_type=llm_type, model=model)
                cat_name = "multi_party_group"
            elif category == "hs":
                # Stage 1: Generate coherent group profiles
                h_type = hub_type or HS_HUB_TYPES[generated % len(HS_HUB_TYPES)]
                print(f"    Generating coherent group profiles ({h_type}, 5 members)...")
                profiles = generate_group_profiles(h_type, count=5, llm_type=llm_type, model=model)
                # Stage 2: Generate scenario structure
                scenario = generate_hs_scenario(profiles, h_type, llm_type=llm_type, model=model)
                cat_name = "hub_and_spoke"
            elif category == "cm":
                # Stage 1: Generate coherent group profiles
                c_type = competition_type or CM_COMPETITION_TYPES[generated % len(CM_COMPETITION_TYPES)]
                print(f"    Generating coherent group profiles ({c_type}, 4 members)...")
                profiles = generate_group_profiles(c_type, count=4, llm_type=llm_type, model=model)
                # Stage 2: Generate scenario structure
                scenario = generate_cm_scenario(profiles, c_type, llm_type=llm_type, model=model)
                cat_name = "competitive"
            elif category == "am":
                # Stage 1: Generate coherent group profiles
                s_ctx = social_context or AM_SOCIAL_CONTEXTS[generated % len(AM_SOCIAL_CONTEXTS)]
                print(f"    Generating coherent group profiles ({s_ctx}, 4 members)...")
                profiles = generate_group_profiles(s_ctx, count=4, llm_type=llm_type, model=model)
                # Stage 2: Generate scenario structure
                scenario = generate_am_scenario(profiles, s_ctx, llm_type=llm_type, model=model)
                cat_name = "affinity_modulated"

            # Validate structure
            if is_new_cat:
                errors = validate_extended_scenario(scenario)
            else:
                errors = validate_scenario(scenario, cat_name)
            if errors:
                print(f"    Validation failed: {errors}")
                continue

            # Optionally verify with LLM judge (skip for new categories by default)
            if verify_flag and not is_new_cat:
                print(f"    Verifying scenario...")
                result = verify_scenario(scenario, cat_name, llm_type=llm_type, model=model)
                if not result["overall_pass"]:
                    print(f"    Verification failed: {result['issues']}")
                    continue

            # Save — deduplicate scenario IDs to prevent overwriting
            scenario_id = scenario.get("scenario_id", f"{category.upper()}_{generated+1:03d}")
            path = os.path.join(subdir, f"{scenario_id}.json")
            if os.path.exists(path):
                # Append a timestamp suffix to avoid collision
                suffix = datetime.now().strftime("%H%M%S")
                scenario_id = f"{scenario_id}_{suffix}"
                scenario["scenario_id"] = scenario_id
                path = os.path.join(subdir, f"{scenario_id}.json")
            write_json(path, scenario)
            generated += 1
            print(f"    Saved: {path}")

        except Exception as e:
            print(f"    Error: {e}")
            continue

    print(f"\nGenerated {generated}/{count} {category.upper()} scenarios in {attempts} attempts.")
    return generated


def main():
    parser = argparse.ArgumentParser(description="PRISM Dataset Generation")
    parser.add_argument("--category", choices=["cd", "mc", "cu", "mg", "hs", "cm", "am"],
                       help="Scenario category to generate")
    parser.add_argument("--group-type", default=None,
                       help=f"Group type for MG scenarios. Options: {MG_GROUP_TYPES}")
    parser.add_argument("--hub-type", default=None,
                       help=f"Hub type for HS scenarios. Options: {HS_HUB_TYPES}")
    parser.add_argument("--competition-type", default=None,
                       help=f"Competition type for CM scenarios. Options: {CM_COMPETITION_TYPES}")
    parser.add_argument("--social-context", default=None,
                       help=f"Social context for AM scenarios. Options: {AM_SOCIAL_CONTEXTS}")
    parser.add_argument("--count", type=int, default=10, help="Number of scenarios to generate")
    parser.add_argument("--llm", default="gemini", choices=["gemini", "openai", "together", "bedrock"],
                       help="LLM provider for generation")
    parser.add_argument("--model", default=None, help="Specific model name")
    parser.add_argument("--output-dir", default="prism/data/scenarios", help="Output directory")
    parser.add_argument("--profiles-only", action="store_true", help="Only generate profiles")
    parser.add_argument("--profiles-dir", default="prism/data/profiles", help="Profiles output directory")
    parser.add_argument("--no-verify", action="store_true", help="Skip LLM verification of scenarios")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for skeleton sampling (default: time-based for unique runs)")

    args = parser.parse_args()

    print(f"PRISM Dataset Generation")
    print(f"  LLM: {args.llm} (model: {args.model or 'default'})")
    print(f"  Count: {args.count}")
    print()

    if args.profiles_only:
        print("Generating profiles...")
        generate_profiles_batch(args.count, args.llm, args.model, args.profiles_dir, seed=args.seed)
    elif args.category:
        print(f"Generating {args.category.upper()} scenarios...")
        generate_scenarios_batch(
            args.category, args.count, args.llm, args.model,
            args.output_dir, verify_flag=not args.no_verify, seed=args.seed,
            group_type=args.group_type, hub_type=args.hub_type,
            competition_type=args.competition_type, social_context=args.social_context,
        )
    else:
        parser.error("Either --category or --profiles-only is required")


if __name__ == "__main__":
    main()
