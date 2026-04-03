# prism/generation/profiles.py
"""Multi-domain user profile generator for PRISM scenarios."""
import os
import yaml

from prism.core.llm import generate
from prism.core.json_utils import coerce_json

import json

REQUIRED_DOMAINS = ["health", "finance", "social", "schedule", "professional", "lifestyle"]
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "profile_prompt.yaml")
GROUP_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "group_profile_prompt.yaml")


def validate_profile(profile: dict) -> list[str]:
    """Validate a user profile has all required domains and sensitivity labels."""
    errors = []
    for domain in REQUIRED_DOMAINS:
        if domain not in profile:
            errors.append(f"Missing domain: {domain}")
        elif not isinstance(profile[domain], dict):
            errors.append(f"Domain {domain} must be a dict")
        elif "sensitivity" not in profile[domain]:
            errors.append(f"Domain {domain} missing sensitivity label")
    return errors


def generate_profile(
    llm_type: str = "gemini",
    model: str = None,
    constraints: dict = None,
    skeleton: dict = None,
) -> dict:
    """Generate a single multi-domain user profile.

    Args:
        llm_type: LLM provider to use (default: "gemini").
        model: Model name override; uses provider default if None.
        constraints: Optional dict to force certain attributes
                     (e.g., {"health": {"conditions": ["diabetes"]}}).
                     Ignored when skeleton is provided.
        skeleton: Pre-sampled demographic skeleton from attribute_pools.sample_skeleton().
                  When provided, uses the constrained prompt template to ensure diversity.
    Returns:
        Profile dict with 6 domains, each with sensitivity label (1-5).
    """
    with open(PROMPT_PATH) as f:
        prompts = yaml.safe_load(f)

    if skeleton:
        # Use constrained prompt with pre-sampled attributes
        prompt_template = prompts["constrained_prompt"]
        prompt = prompt_template.format(
            name=skeleton["name"],
            age=skeleton["age"],
            gender=skeleton["gender"],
            location=skeleton["location"],
            occupation=skeleton["occupation"],
            life_situation=skeleton["life_situation"],
            health_conditions=", ".join(skeleton["health_conditions"]),
            sensitivity_targets=skeleton["sensitivity_targets"],
        )
    else:
        # Free-form generation (original behavior)
        prompt_template = prompts["prompt"]
        constraint_text = ""
        if constraints:
            constraint_text = f"\n\nThe profile MUST include these specific attributes:\n{constraints}"
        prompt = prompt_template + constraint_text

    response = generate(prompt, llm_type=llm_type, model=model, temperature=0.8)
    profile = coerce_json(response)

    errors = validate_profile(profile)
    if errors:
        raise ValueError(f"Generated profile validation failed: {errors}")

    return profile


def generate_group_profiles(
    group_type: str,
    count: int = 4,
    llm_type: str = "gemini",
    model: str = None,
) -> list[dict]:
    """Generate a coherent group of user profiles that make sense together.

    Unlike generate_profile() which produces independent individuals, this
    generates a group where the members have plausible relationships — e.g.,
    a family with realistic ages, a workplace team with compatible roles,
    or a classroom with a teacher and students.

    Args:
        group_type: Social context — e.g., "family", "workplace_team", "classroom",
                    "friend_group", "neighborhood", "hr_recruiter", "sales_marketplace", etc.
        count: Number of profiles to generate (3-6).
        llm_type: LLM provider.
        model: Model name override.

    Returns:
        List of profile dicts, each with 6 domains and sensitivity labels.
    """
    from prism.core.json_utils import coerce_json as _coerce

    with open(GROUP_PROMPT_PATH) as f:
        prompt_template = yaml.safe_load(f)["prompt"]

    prompt = prompt_template.format(
        group_type=group_type,
        count=count,
    )

    response = generate(prompt, llm_type=llm_type, model=model, temperature=0.8)
    result = _coerce(response)

    # Result should be {"profiles": [...]}
    profiles = result.get("profiles", result if isinstance(result, list) else [])

    if not isinstance(profiles, list) or len(profiles) < 2:
        raise ValueError(f"Expected list of {count} profiles, got {type(profiles)} with {len(profiles) if isinstance(profiles, list) else 0} items")

    # Validate each profile
    validated = []
    for i, profile in enumerate(profiles):
        errors = validate_profile(profile)
        if errors:
            # Try to repair — add missing domains with minimal content
            for domain in REQUIRED_DOMAINS:
                if domain not in profile:
                    profile[domain] = {"sensitivity": 2}
                elif not isinstance(profile[domain], dict):
                    profile[domain] = {"sensitivity": 2}
                elif "sensitivity" not in profile[domain]:
                    profile[domain]["sensitivity"] = 2
        validated.append(profile)

    return validated
