# prism/generation/scenarios_extended.py
"""Scenario generators for multi-party categories (MG, HS, CM, AM).

Two-stage generation:
  Stage 1: Coherent group profiles are generated via profiles.generate_group_profiles()
  Stage 2: This module generates the scenario STRUCTURE (task, social_graph, privacy_boundaries)
           around those profiles. The LLM does NOT regenerate profiles — they are injected
           programmatically after LLM generation.
"""
import json
import os
import random
import yaml
from datetime import datetime

from prism.core.llm import generate, DEFAULT_MODELS
from prism.core.json_utils import coerce_json


def _resolve_model(llm_type, model):
    return model or DEFAULT_MODELS.get(llm_type, "unknown")


def _generate_with_parse_retry(prompt, llm_type, model, temperature, max_parse_retries=2):
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

# Valid subcategories for each new category
MG_GROUP_TYPES = ["family", "friend_group", "workplace_team", "classroom", "neighborhood"]
HS_HUB_TYPES = ["hr_recruiter", "teacher_advisor", "event_organizer", "property_manager", "medical_coordinator"]
CM_COMPETITION_TYPES = ["sales_marketplace", "job_market", "housing_rental", "dating_matching", "grant_allocation"]
AM_SOCIAL_CONTEXTS = ["mixed_closeness_friends", "workplace_personal_overlap", "extended_family", "new_connection", "asymmetric_relationship"]


def _add_metadata(scenario, llm_type, model, sub_type=None):
    """Add generation metadata to scenario."""
    scenario.setdefault("metadata", {})
    scenario["metadata"]["generated_by"] = _resolve_model(llm_type, model)
    scenario["metadata"]["timestamp"] = datetime.now().isoformat()
    if sub_type:
        scenario["metadata"]["sub_type"] = sub_type
    return scenario


def _inject_profiles(scenario, profiles):
    """Inject pre-generated user_profile into each participant by matching names.

    The LLM generates participants with user_name but no user_profile.
    This function matches each participant to a profile by name and injects
    the full profile. If no match is found, the profile is injected by index.
    """
    profile_by_name = {}
    for p in profiles:
        name = p.get("name", "")
        profile_by_name[name.lower()] = p
        # Also index by first name
        first = name.split()[0].lower() if name else ""
        if first:
            profile_by_name[first] = p

    participants = scenario.get("participants", [])
    for idx, participant in enumerate(participants):
        if "user_profile" in participant and participant["user_profile"]:
            continue  # already has profile (shouldn't happen, but safe)

        user_name = participant.get("user_name", "")
        # Try matching by full name, first name
        matched = (
            profile_by_name.get(user_name.lower())
            or profile_by_name.get(user_name.split()[0].lower() if user_name else "")
        )
        if matched:
            participant["user_profile"] = matched
        elif idx < len(profiles):
            # Fallback: match by index
            participant["user_profile"] = profiles[idx]
        else:
            participant["user_profile"] = {}


def _normalize_name(name):
    """Normalize a name for matching between participants and social_graph edges.

    Returns a set of possible normalized forms to handle LLM inconsistencies:
    - Full name vs first name only
    - Unicode accents (María vs Maria)
    - CJK name ordering (family name first)
    """
    import unicodedata
    name = name.strip().lower()
    # Strip Unicode accents: María → maria
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name


def _name_variants(name):
    """Return all plausible normalized forms of a name for fuzzy matching.

    "Alice Chen" → {"alice chen", "alice", "chen"}
    "Li" → {"li"}
    """
    norm = _normalize_name(name)
    parts = norm.split()
    variants = {norm}  # full normalized name
    for p in parts:
        variants.add(p)  # each part (first name, last name, etc.)
    return variants


def validate_extended_scenario(scenario):
    """Validate required fields for extended scenario format.

    Returns list of error strings. Empty list means valid.
    Mirrors the contract of validate_scenario() in scenarios.py.
    """
    errors = []
    category = scenario.get("category", "")
    required = ["scenario_id", "category", "task", "participants", "success_criteria"]
    missing = [f for f in required if f not in scenario]
    if missing:
        errors.extend([f"Missing required field: {f}" for f in missing])

    # Build participant name variant sets for fuzzy graph matching
    participant_variant_sets = []  # list of (display_name, set_of_variants)
    for idx, p in enumerate(scenario.get("participants", [])):
        # user_profile is injected programmatically, so don't require it from LLM
        p_required = ["user_name", "agent_name", "task_role", "system_prompt_context", "privacy_boundary"]
        p_missing = [f for f in p_required if f not in p]
        if p_missing:
            errors.append(f"Participant {idx} missing fields: {p_missing}")

        uname = p.get("user_name", "")
        if uname:
            participant_variant_sets.append((uname, _name_variants(uname)))

        # Validate privacy_boundary structure
        pb = p.get("privacy_boundary", {})
        if not isinstance(pb, dict):
            errors.append(f"Participant {idx} privacy_boundary must be a dict")
            continue

        mnc = pb.get("must_not_cross")
        if category == "affinity_modulated":
            # AM uses a tiered dict: {close: [...], friend: [...], ...}
            if not isinstance(mnc, dict):
                errors.append(f"Participant {idx} must_not_cross must be a tier dict for AM")
            elif not mnc.get("stranger") and not mnc.get("acquaintance"):
                errors.append(f"Participant {idx} must_not_cross tiers are empty")
        else:
            # All other categories use a flat list
            if not isinstance(mnc, list) or len(mnc) == 0:
                errors.append(f"Participant {idx} must_not_cross is missing or empty")

    # Validate social_graph consistency if present
    # Use fuzzy name matching: a participant matches an edge name if ANY of their
    # name variants (full name, first name, last name, accent-stripped) overlap.
    edges = scenario.get("social_graph", {}).get("edges", [])
    if edges:
        edge_name_variants = set()
        for e in edges:
            edge_name_variants.update(_name_variants(e.get("from", "")))
            edge_name_variants.update(_name_variants(e.get("to", "")))
        missing = []
        for display_name, variants in participant_variant_sets:
            if not variants & edge_name_variants:
                missing.append(display_name)
        if missing:
            errors.append(f"Participants with no edges in social_graph: {missing}")

    return errors


def sanitize_task_description(scenario, llm_type="gemini", model=None):
    """Rewrite the task description to remove any leaked private information.

    The LLM often puts private details (e.g., "pregnancy test", "gambling debt")
    directly into the task field. This function calls an LLM to rewrite the task
    using only acceptable abstractions, preserving the task's meaning.

    Modifies the scenario in-place and returns it.
    """
    # Collect all must_not_cross items across participants
    private_items = []
    abstractions = {}
    for p in scenario.get("participants", []):
        pb = p.get("privacy_boundary", {})
        mnc = pb.get("must_not_cross", [])
        if isinstance(mnc, dict):
            # AM tiered format — flatten all tiers
            for tier_items in mnc.values():
                if isinstance(tier_items, list):
                    private_items.extend(tier_items)
        elif isinstance(mnc, list):
            private_items.extend(mnc)
        abs_map = pb.get("acceptable_abstractions", {})
        if isinstance(abs_map, dict):
            abstractions.update(abs_map)

    if not private_items:
        return scenario

    # Check if the task actually contains any private keywords
    task_lower = scenario.get("task", "").lower()
    leaked = [item for item in private_items
              if any(w in task_lower for w in item.lower().split() if len(w) > 7)]
    if not leaked:
        return scenario  # no leakage detected, skip the LLM call

    sanitize_prompt = (
        "Rewrite the following task description to remove all private/sensitive "
        "information. Replace any private details with their acceptable abstractions.\n\n"
        f"TASK DESCRIPTION:\n{scenario['task']}\n\n"
        f"PRIVATE ITEMS TO REMOVE (replace with abstractions):\n"
    )
    for item in leaked:
        abstraction = abstractions.get(item, "a private matter")
        sanitize_prompt += f"- \"{item}\" → \"{abstraction}\"\n"
    sanitize_prompt += (
        "\nReturn ONLY the rewritten task description. Keep the same length and level "
        "of detail. Do not add explanations or markdown."
    )

    rewritten = generate(sanitize_prompt, llm_type=llm_type, model=model, temperature=0.3)
    # Clean up: strip quotes and whitespace
    rewritten = rewritten.strip().strip('"').strip("'").strip()
    if rewritten and len(rewritten) > 20:
        scenario["task"] = rewritten

    return scenario


def _generate_scenario(prompt_file, profiles, sub_type, category, llm_type, model, **format_kwargs):
    """Shared generation logic for all multi-party categories.

    1. Load prompt template
    2. Format with profiles summary + sub_type-specific args
    3. Call LLM to generate scenario structure (no user_profile)
    4. Inject profiles programmatically
    5. Sanitize task description (remove leaked private info)
    6. Set category, add metadata
    """
    with open(os.path.join(PROMPTS_DIR, prompt_file)) as f:
        prompt_template = yaml.safe_load(f)["prompt"]

    # Build a concise profile summary for the prompt (names + key details, not full JSON)
    profile_summaries = []
    for p in profiles:
        name = p.get("name", "Unknown")
        age = p.get("demographics", {}).get("age", "?")
        occ = p.get("demographics", {}).get("occupation", p.get("professional", {}).get("role", "?"))
        health = p.get("health", {}).get("conditions", [])
        health_str = ", ".join(health) if health else "none"
        finance_sens = p.get("finance", {}).get("sensitivity", 2)
        health_sens = p.get("health", {}).get("sensitivity", 2)
        social_notes = p.get("social", {}).get("private_conversations", [])
        social_str = ", ".join(social_notes[:2]) if social_notes else "none"
        profile_summaries.append(
            f"- {name} (age {age}, {occ}): health conditions: {health_str} "
            f"(sensitivity {health_sens}), finance sensitivity: {finance_sens}, "
            f"social notes: {social_str}"
        )
    profiles_text = "\n".join(profile_summaries)

    prompt = prompt_template.format(
        num_participants=len(profiles),
        profiles=profiles_text,
        **format_kwargs,
    )

    scenario = _generate_with_parse_retry(prompt, llm_type, model, temperature=0.8)
    scenario["category"] = category
    _inject_profiles(scenario, profiles)
    sanitize_task_description(scenario, llm_type=llm_type, model=model)
    _add_metadata(scenario, llm_type, model, sub_type=sub_type)
    # Validation is done by the caller (generate.py) — don't call here
    return scenario


def generate_mg_scenario(profiles, group_type=None, llm_type="gemini", model=None):
    """Generate a Multi-Party Group Chat scenario.

    Args:
        profiles: List of 3-6 coherent user profile dicts (from generate_group_profiles)
        group_type: One of MG_GROUP_TYPES, or None for random
    """
    if group_type is None:
        group_type = random.choice(MG_GROUP_TYPES)
    return _generate_scenario(
        "mg_prompt.yaml", profiles, group_type, "multi_party_group",
        llm_type, model, group_type=group_type,
    )


def generate_hs_scenario(profiles, hub_type=None, llm_type="gemini", model=None):
    """Generate a Hub-and-Spoke scenario."""
    if hub_type is None:
        hub_type = random.choice(HS_HUB_TYPES)
    return _generate_scenario(
        "hs_prompt.yaml", profiles, hub_type, "hub_and_spoke",
        llm_type, model, hub_type=hub_type,
    )


def generate_cm_scenario(profiles, competition_type=None, llm_type="gemini", model=None):
    """Generate a Competitive Multi-Agent scenario."""
    if competition_type is None:
        competition_type = random.choice(CM_COMPETITION_TYPES)
    return _generate_scenario(
        "cm_prompt.yaml", profiles, competition_type, "competitive",
        llm_type, model, competition_type=competition_type,
    )


def generate_am_scenario(profiles, social_context=None, llm_type="gemini", model=None):
    """Generate an Affinity-Modulated Privacy scenario."""
    if social_context is None:
        social_context = random.choice(AM_SOCIAL_CONTEXTS)
    return _generate_scenario(
        "am_prompt.yaml", profiles, social_context, "affinity_modulated",
        llm_type, model, social_context=social_context,
    )
