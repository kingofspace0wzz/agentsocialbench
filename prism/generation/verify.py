# prism/generation/verify.py
"""5-criteria scenario verifier using LLM judge. Adapted from MAGPIE's verify_solvability."""
import os
import json
import yaml

from prism.core.llm import generate
from prism.core.json_utils import coerce_json

VERIFY_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "verify_prompt.yaml")

CRITERIA = [
    "has_privacy_tension",
    "boundary_justified",
    "is_solvable",
    "abstraction_exists",
    "is_non_trivial",
]


def parse_verification_result(llm_response: str) -> dict:
    """Parse LLM judge verification response into structured result."""
    parsed = coerce_json(llm_response)

    issues = []
    criteria = {}
    for criterion in CRITERIA:
        if criterion in parsed:
            criteria[criterion] = parsed[criterion]
            if not parsed[criterion].get("pass", False):
                issues.append(f"{criterion}: {parsed[criterion].get('explanation', 'Failed')}")
        else:
            criteria[criterion] = {"pass": False, "explanation": "Not evaluated"}
            issues.append(f"{criterion}: Not evaluated by judge")

    return {
        "overall_pass": len(issues) == 0,
        "criteria": criteria,
        "issues": issues,
    }


def verify_graph_consistency(scenario: dict) -> list:
    """Check that all participants referenced in edges exist."""
    participant_names = {p["user_name"].lower() for p in scenario.get("participants", [])}
    edges = scenario.get("social_graph", {}).get("edges", [])
    issues = []
    for e in edges:
        if e.get("from", "").lower() not in participant_names:
            issues.append(f"Edge references unknown participant: {e['from']}")
        if e.get("to", "").lower() not in participant_names:
            issues.append(f"Edge references unknown participant: {e['to']}")
    # Check for orphan nodes (participants with no edges)
    if edges:
        mentioned = set()
        for e in edges:
            mentioned.add(e.get("from", "").lower())
            mentioned.add(e.get("to", "").lower())
        orphans = participant_names - mentioned
        if orphans:
            issues.append(f"Participants with no edges (orphan nodes): {orphans}")
    return issues


def verify_affinity_monotonicity(scenario: dict) -> list:
    """Check that tiered must_not_cross is monotonically restrictive."""
    issues = []
    tier_order = ["close", "friend", "acquaintance", "stranger"]

    for p in scenario.get("participants", []):
        mnc = p.get("privacy_boundary", {}).get("must_not_cross", {})
        if not isinstance(mnc, dict):
            continue  # flat list, skip

        # Check tiers are valid
        for tier in mnc:
            if tier not in tier_order:
                issues.append(f"{p['user_name']}: unknown tier '{tier}'")

        # Check monotonicity: close ⊆ friend ⊆ acquaintance ⊆ stranger
        for i in range(len(tier_order) - 1):
            smaller_tier = tier_order[i]
            larger_tier = tier_order[i + 1]
            smaller_set = set(mnc.get(smaller_tier, []))
            larger_set = set(mnc.get(larger_tier, []))
            if not smaller_set.issubset(larger_set):
                diff = smaller_set - larger_set
                issues.append(
                    f"{p['user_name']}: tier '{smaller_tier}' has items {diff} "
                    f"not in '{larger_tier}' (violates monotonicity)"
                )
    return issues


def verify_coordinator_exists(scenario: dict) -> list:
    """Check that HS/CM scenarios have exactly one coordinator."""
    issues = []
    category = scenario.get("category", "")
    if category not in ("hub_and_spoke", "competitive"):
        return issues

    coordinators = [p for p in scenario.get("participants", []) if p.get("task_role") == "coordinator"]
    if len(coordinators) == 0:
        issues.append(f"{category} scenario has no coordinator (task_role='coordinator')")
    elif len(coordinators) > 1:
        names = [c["user_name"] for c in coordinators]
        issues.append(f"{category} scenario has multiple coordinators: {names}")
    return issues


EXTENDED_CATEGORIES = {"multi_party_group", "hub_and_spoke", "competitive", "affinity_modulated"}


def verify_scenario(
    scenario: dict,
    category: str,
    llm_type: str = "gemini",
    model: str = None,
) -> dict:
    """Verify a scenario against 5 quality criteria using LLM judge.

    For extended categories (multi_party_group, hub_and_spoke, competitive,
    affinity_modulated), additional structural checks are also run.

    Returns:
        {"overall_pass": bool, "criteria": {...}, "issues": [str]}
    """
    with open(VERIFY_PROMPT_PATH) as f:
        prompt_template = yaml.safe_load(f)["prompt"]

    scenario_json = json.dumps(scenario, indent=2)
    # Use str.replace() instead of str.format() to avoid KeyError when
    # scenario JSON values contain literal { or } characters.
    prompt = prompt_template.replace("{scenario_json}", scenario_json).replace("{category}", category)

    response = generate(prompt, llm_type=llm_type, model=model, temperature=0.1)
    result = parse_verification_result(response)

    # Run additional structural checks for extended categories
    scenario_category = scenario.get("category", category)
    if scenario_category in EXTENDED_CATEGORIES:
        extra_issues = []
        extra_issues.extend(verify_graph_consistency(scenario))
        extra_issues.extend(verify_affinity_monotonicity(scenario))
        extra_issues.extend(verify_coordinator_exists(scenario))
        if extra_issues:
            result["issues"].extend(extra_issues)
            result["overall_pass"] = False

    return result
