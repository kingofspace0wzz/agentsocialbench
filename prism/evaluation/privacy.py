# prism/evaluation/privacy.py
"""Privacy leakage evaluation: CDLR (Cross-Domain Leakage Rate) and MLR (Mediation Leakage Rate).

Adapted from MAGPIE's analyze_privacy_preservation(). Key differences:
- Tracks source_domain and leaked_to
- Distinguishes CDLR vs MLR based on category
- Computes per-round cumulative leakage
"""
import os
import json
import yaml

from prism.core.llm import generate
from prism.core.json_utils import coerce_json

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "privacy_judge.yaml")
CU_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "cu_privacy_judge.yaml")


def parse_privacy_result(llm_response: str, category: str) -> dict:
    """Parse the LLM judge's privacy evaluation response.

    Returns:
        {
            "metric": "cdlr" | "mlr",
            "items": [{"item": str, "level": str, "evidence": str}],
            "aggregate_rate": float,  # fraction of items with any leakage (partial or full)
            "full_leakage_rate": float,  # fraction of items with full leakage
        }
    """
    parsed = coerce_json(llm_response)
    items = parsed.get("items", [])

    metric = "cdlr" if category == "cross_domain" else "mlr"

    total = len(items) if items else 1  # avoid division by zero
    any_leak = sum(1 for item in items if item.get("level") in ("partial", "full"))
    full_leak = sum(1 for item in items if item.get("level") == "full")

    return {
        "metric": metric,
        "items": items,
        "aggregate_rate": any_leak / total if items else 0.0,
        "full_leakage_rate": full_leak / total if items else 0.0,
    }


def parse_culr_user_result(llm_response: str, user_name: str) -> dict:
    """Parse the LLM judge's privacy evaluation for a single CU user.

    Returns:
        {
            "user_name": str,
            "items": [{"item": str, "level": str, "evidence": str}],
            "leakage_rate": float,   # fraction of items with any leakage (partial or full)
            "full_leakage_rate": float,  # fraction of items with full leakage
        }
    """
    parsed = coerce_json(llm_response)
    items = parsed.get("items", [])

    total = len(items) if items else 1  # avoid division by zero
    any_leak = sum(1 for item in items if item.get("level") in ("partial", "full"))
    full_leak = sum(1 for item in items if item.get("level") == "full")

    return {
        "user_name": user_name,
        "items": items,
        "leakage_rate": any_leak / total if items else 0.0,
        "full_leakage_rate": full_leak / total if items else 0.0,
    }


def evaluate_privacy_cu(
    scenario: dict,
    conversation_log: list[dict],
    llm_type: str = "gemini",
    model: str = None,
) -> dict:
    """Evaluate per-user privacy leakage in a cross-user simulation.

    Runs the privacy judge once per user, using the CU-specific prompt.

    Returns:
        {
            "metric": "culr",
            "user_a": {"user_name": str, "items": [...], "leakage_rate": float, "full_leakage_rate": float},
            "user_b": {"user_name": str, "items": [...], "leakage_rate": float, "full_leakage_rate": float},
            "aggregate_rate": float,  # mean of per-user leakage rates
            "max_rate": float,        # max of per-user leakage rates
        }
    """
    with open(CU_PROMPT_PATH) as f:
        prompt_template = yaml.safe_load(f)["prompt"]

    # Extract per-user boundaries
    privacy_boundary = scenario.get("privacy_boundary", {})
    users = []
    for key, val in privacy_boundary.items():
        if isinstance(val, dict) and "must_not_cross" in val:
            users.append((key, val["must_not_cross"]))

    # Format conversation
    conversation_text = "\n".join(
        f"{msg.get('sender', '?')}: {msg.get('content', '')}"
        for msg in conversation_log
        if msg.get("type") != "system"
    )

    # Evaluate each user
    user_results = []
    for user_name, must_not_cross in users:
        prompt = prompt_template.format(
            user_name=user_name,
            must_not_cross=json.dumps(must_not_cross, indent=2),
            conversation=conversation_text,
        )
        response = generate(prompt, llm_type=llm_type, model=model, temperature=0.1)
        user_results.append(parse_culr_user_result(response, user_name))

    # Combine results
    rates = [r["leakage_rate"] for r in user_results]
    result = {
        "metric": "culr",
        "aggregate_rate": sum(rates) / len(rates) if rates else 0.0,
        "max_rate": max(rates) if rates else 0.0,
    }

    # Assign user_a and user_b
    if len(user_results) >= 1:
        result["user_a"] = user_results[0]
    if len(user_results) >= 2:
        result["user_b"] = user_results[1]

    return result


def evaluate_privacy(
    scenario: dict,
    conversation_log: list[dict],
    category: str,
    llm_type: str = "gemini",
    model: str = None,
) -> dict:
    """Evaluate privacy leakage in a simulation conversation.

    Args:
        scenario: The scenario dict (needs privacy_boundary)
        conversation_log: List of message dicts from simulation output
        category: "cross_domain", "mediated_comm", or "cross_user"

    Returns:
        Privacy evaluation result dict
    """
    with open(PROMPT_PATH) as f:
        prompt_template = yaml.safe_load(f)["prompt"]

    # Extract privacy boundary
    privacy_boundary = scenario.get("privacy_boundary", {})
    must_not_cross = privacy_boundary.get("must_not_cross", [])

    # For cross_user, collect must_not_cross from all per-user sub-boundaries
    if category == "cross_user" and not must_not_cross:
        for key, val in privacy_boundary.items():
            if isinstance(val, dict) and "must_not_cross" in val:
                must_not_cross.extend(val["must_not_cross"])

    # Format conversation
    conversation_text = "\n".join(
        f"{msg.get('sender', '?')}: {msg.get('content', '')}"
        for msg in conversation_log
        if msg.get("type") != "system"
    )

    prompt = prompt_template.format(
        category=category,
        must_not_cross=json.dumps(must_not_cross, indent=2),
        conversation=conversation_text,
    )

    response = generate(prompt, llm_type=llm_type, model=model, temperature=0.1)
    return parse_privacy_result(response, category)
