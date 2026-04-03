# prism/generation/scenarios.py
"""Category-specific scenario generators for PRISM."""
import json
import os
import yaml
from datetime import datetime

from prism.core.llm import generate, DEFAULT_MODELS
from prism.core.json_utils import coerce_json


def _resolve_model(llm_type: str, model: str | None) -> str:
    """Return the actual model name, resolving None to the provider default."""
    return model or DEFAULT_MODELS.get(llm_type, "unknown")


def _generate_with_parse_retry(
    prompt: str, llm_type: str, model: str | None, temperature: float, max_parse_retries: int = 2
) -> dict:
    """Call LLM and parse JSON response, retrying on parse failure.

    This avoids wasting an entire profile+scenario generation attempt when the
    LLM returns syntactically broken JSON on the first try.
    """
    last_error = None
    for attempt in range(max_parse_retries):
        response = generate(prompt, llm_type=llm_type, model=model, temperature=temperature)
        try:
            return coerce_json(response)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            if attempt < max_parse_retries - 1:
                continue
    raise last_error

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

REQUIRED_FIELDS = ["scenario_id", "category", "task", "privacy_boundary", "success_criteria", "agents"]
REQUIRED_PRIVACY_FIELDS = ["must_not_cross"]


def _normalize_cu_privacy_keys(scenario: dict, profile_a: dict, profile_b: dict) -> None:
    """Ensure CU privacy_boundary uses lowercase first names as keys.

    The CU prompt instructs the LLM to use '<first_name_lowercase>' as keys,
    but models sometimes use full names, different casing, or other variants.
    This normalizes the keys to match what the simulation expects (first name
    only, lowercased), preventing silent lookup failures at simulation time.
    """
    pb = scenario.get("privacy_boundary", {})
    if not isinstance(pb, dict):
        return

    expected_a = profile_a.get("name", "").split()[0].lower()
    expected_b = profile_b.get("name", "").split()[0].lower()
    if not expected_a or not expected_b:
        return

    # Find which existing keys are per-user dicts (not "minimum_info_needed" etc.)
    user_entries = {}
    other_entries = {}
    for key, val in pb.items():
        if isinstance(val, dict) and "must_not_cross" in val:
            user_entries[key] = val
        else:
            other_entries[key] = val

    if len(user_entries) != 2:
        return  # Can't reliably map; leave as-is for validation to catch

    # Map the two user entries to the correct first names by matching against
    # profile names (the LLM may have used full name, different casing, etc.)
    keys = list(user_entries.keys())
    name_a_full = profile_a.get("name", "").lower()
    name_b_full = profile_b.get("name", "").lower()

    # Heuristic: match each key to the profile whose name contains it
    mapping = {}
    for key in keys:
        key_lower = key.lower()
        if key_lower in name_a_full or expected_a in key_lower:
            mapping[key] = expected_a
        elif key_lower in name_b_full or expected_b in key_lower:
            mapping[key] = expected_b

    # Only rewrite if we matched both unambiguously
    if len(set(mapping.values())) == 2:
        new_pb = dict(other_entries)
        for old_key, new_key in mapping.items():
            new_pb[new_key] = user_entries[old_key]
        scenario["privacy_boundary"] = new_pb


def validate_scenario(scenario: dict, category: str) -> list[str]:
    """Validate a scenario has all required fields for its category.

    Args:
        scenario: The scenario dict to validate.
        category: One of "cross_domain", "mediated_comm", "cross_user".

    Returns:
        List of error strings. Empty list means valid.
    """
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in scenario:
            errors.append(f"Missing required field: {field}")

    if "privacy_boundary" in scenario:
        pb = scenario["privacy_boundary"]
        # For cross_user scenarios, privacy_boundary may be nested per-user
        if category == "cross_user" and not isinstance(pb.get("must_not_cross", None), list):
            # Check if there are per-user sub-keys that each have must_not_cross
            has_per_user = any(
                isinstance(v, dict) and "must_not_cross" in v
                for v in pb.values() if isinstance(v, dict)
            )
            if not has_per_user and "must_not_cross" not in pb:
                errors.append("Missing privacy_boundary.must_not_cross")
        elif category != "cross_user":
            for pf in REQUIRED_PRIVACY_FIELDS:
                if pf not in pb:
                    errors.append(f"Missing privacy_boundary.{pf}")

    if category == "mediated_comm" and "humans" not in scenario:
        errors.append("MC scenarios require 'humans' field")

    return errors


def generate_cd_scenario(
    profile: dict,
    source_domain: str,
    target_domain: str,
    llm_type: str = "gemini",
    model: str = None,
) -> dict:
    """Generate a cross-domain coordination scenario.

    Args:
        profile: User profile dict (from generate_profile).
        source_domain: The domain holding private info (e.g., "health").
        target_domain: The domain that needs abstracted info (e.g., "social").
        llm_type: LLM provider to use.
        model: Model name override.

    Returns:
        Scenario dict with all required fields.
    """
    with open(os.path.join(PROMPTS_DIR, "cd_prompt.yaml")) as f:
        prompt_template = yaml.safe_load(f)["prompt"]

    prompt = prompt_template.format(
        profile=profile, source_domain=source_domain, target_domain=target_domain
    )
    resolved_model = _resolve_model(llm_type, model)
    response = _generate_with_parse_retry(prompt, llm_type=llm_type, model=model, temperature=0.7)
    scenario = response
    scenario["category"] = "cross_domain"
    scenario["metadata"] = {
        "source_domain": source_domain,
        "target_domain": target_domain,
        "generated_by": resolved_model,
        "generation_timestamp": datetime.now().isoformat(),
    }
    return scenario


def generate_mc_scenario(
    profile_a: dict,
    profile_b: dict,
    mediation_type: str,
    llm_type: str = "gemini",
    model: str = None,
) -> dict:
    """Generate a mediated communication scenario.

    Args:
        profile_a: Profile of the user whose agent mediates.
        profile_b: Profile of the other human participant.
        mediation_type: One of group_event_planning, information_brokering,
            conflict_mediation, recommendation_sharing, schedule_coordination,
            social_introductions.
        llm_type: LLM provider to use.
        model: Model name override.

    Returns:
        Scenario dict with all required fields.
    """
    with open(os.path.join(PROMPTS_DIR, "mc_prompt.yaml")) as f:
        prompt_template = yaml.safe_load(f)["prompt"]

    prompt = prompt_template.format(
        profile_a=profile_a, profile_b=profile_b, mediation_type=mediation_type
    )
    resolved_model = _resolve_model(llm_type, model)
    scenario = _generate_with_parse_retry(prompt, llm_type=llm_type, model=model, temperature=0.7)
    scenario["category"] = "mediated_comm"
    scenario["metadata"] = {
        "mediation_type": mediation_type,
        "generated_by": resolved_model,
        "generation_timestamp": datetime.now().isoformat(),
    }
    return scenario


def generate_cu_scenario(
    profile_a: dict,
    profile_b: dict,
    interaction_type: str,
    llm_type: str = "gemini",
    model: str = None,
) -> dict:
    """Generate a cross-user agent interaction scenario.

    Args:
        profile_a: Profile of the first user.
        profile_b: Profile of the second user.
        interaction_type: One of shared_activity, resource_sharing,
            joint_planning, information_exchange, social_introduction.
        llm_type: LLM provider to use.
        model: Model name override.

    Returns:
        Scenario dict with all required fields.
    """
    with open(os.path.join(PROMPTS_DIR, "cu_prompt.yaml")) as f:
        prompt_template = yaml.safe_load(f)["prompt"]

    prompt = prompt_template.format(
        profile_a=profile_a, profile_b=profile_b, interaction_type=interaction_type
    )
    resolved_model = _resolve_model(llm_type, model)
    scenario = _generate_with_parse_retry(prompt, llm_type=llm_type, model=model, temperature=0.7)
    scenario["category"] = "cross_user"
    scenario["metadata"] = {
        "interaction_type": interaction_type,
        "generated_by": resolved_model,
        "generation_timestamp": datetime.now().isoformat(),
    }

    # Fix 3: Normalize CU privacy boundary keys to lowercase first names
    _normalize_cu_privacy_keys(scenario, profile_a, profile_b)

    return scenario
