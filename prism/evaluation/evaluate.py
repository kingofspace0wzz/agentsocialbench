# prism/evaluation/evaluate.py
"""Evaluation orchestrator. Runs all 4 metric modules on a simulation."""
import json
import os
from datetime import datetime

from prism.core.json_utils import write_json
from prism.evaluation.privacy import evaluate_privacy, evaluate_privacy_cu
from prism.evaluation.abstraction import evaluate_abstraction
from prism.evaluation.task_completion import evaluate_task_completion
from prism.evaluation.behavioral import evaluate_behavioral, evaluate_cu_behavioral, detect_premature_user_contact
from prism.evaluation.privacy_extended import (
    evaluate_privacy_mg, evaluate_privacy_hs,
    evaluate_privacy_cm, evaluate_privacy_am,
    normalize_scenario_for_eval,
)
from prism.evaluation.task_completion_extended import (
    evaluate_task_completion_mg, evaluate_task_completion_hs,
    evaluate_task_completion_cm, evaluate_task_completion_am,
)


# Category dispatch for evaluation routing
# NOTE: lambdas reference module-level names so unittest.mock.patch works correctly.
PRIVACY_EVALUATORS = {
    "cross_domain": lambda s, c, **kw: evaluate_privacy(s, c, "cross_domain", **kw),
    "mediated_comm": lambda s, c, **kw: evaluate_privacy(s, c, "mediated_comm", **kw),
    "cross_user": lambda s, c, **kw: evaluate_privacy_cu(s, c, **kw),
    "multi_party_group": lambda s, c, **kw: evaluate_privacy_mg(s, c, **kw),
    "hub_and_spoke": lambda s, c, **kw: evaluate_privacy_hs(s, c, **kw),
    "competitive": lambda s, c, **kw: evaluate_privacy_cm(s, c, **kw),
    "affinity_modulated": lambda s, c, **kw: evaluate_privacy_am(s, c, **kw),
}

TCQ_EVALUATORS = {
    "cross_domain": lambda s, c, **kw: evaluate_task_completion(s, c, **kw),
    "mediated_comm": lambda s, c, **kw: evaluate_task_completion(s, c, **kw),
    "cross_user": lambda s, c, **kw: evaluate_task_completion(s, c, **kw),
    "multi_party_group": lambda s, c, **kw: evaluate_task_completion_mg(s, c, **kw),
    "hub_and_spoke": lambda s, c, **kw: evaluate_task_completion_hs(s, c, **kw),
    "competitive": lambda s, c, **kw: evaluate_task_completion_cm(s, c, **kw),
    "affinity_modulated": lambda s, c, **kw: evaluate_task_completion_am(s, c, **kw),
}

NEW_CATEGORIES = {"multi_party_group", "hub_and_spoke", "competitive", "affinity_modulated"}


def _evaluate_efficiency(sim_data: dict) -> dict:
    """Extract deterministic efficiency metrics from simulation output.

    No LLM call — pure field extraction.
    """
    category = sim_data.get("category", "")
    num_rounds = sim_data.get("num_rounds", 0)
    max_rounds = sim_data.get("max_rounds", 10)

    rtc = num_rounds  # default: total rounds used

    if category in ("cross_domain", "cross_user"):
        dp = sim_data.get("delivery_phase") or {}
        if dp.get("activated") and dp.get("trigger_round", 0) >= 1:
            rtc = dp["trigger_round"]

    return {
        "rounds_to_completion": rtc,
        "max_rounds": max_rounds,
        "completed_early": rtc < max_rounds,
    }


def evaluate_simulation(
    scenario: dict,
    simulation: dict,
    output_dir: str = None,
    llm_type: str = "gemini",
    model: str = None,
    cu_behavioral: bool = False,
) -> dict:
    """Run all evaluation metrics on a single simulation.

    Args:
        scenario: The scenario dict (with privacy_boundary, task, success_criteria)
        simulation: The simulation output dict (with conversation_log, metadata)
        output_dir: If provided, save result JSON to this directory
        llm_type: LLM provider for evaluation judges
        model: Specific model for evaluation

    Returns:
        Combined evaluation result dict
    """
    category = simulation.get("category", scenario.get("category", ""))
    conversation_log = simulation.get("conversation_log", [])

    # Run all 4 evaluations
    # Route privacy evaluation via dispatch
    privacy_evaluator = PRIVACY_EVALUATORS.get(category)
    if privacy_evaluator:
        privacy_result = privacy_evaluator(scenario, conversation_log, llm_type=llm_type, model=model)
    else:
        privacy_result = evaluate_privacy(scenario, conversation_log, category, llm_type=llm_type, model=model)

    # Route TCQ evaluation via dispatch
    tcq_evaluator = TCQ_EVALUATORS.get(category, evaluate_task_completion)
    tcq_result = tcq_evaluator(scenario, conversation_log, llm_type=llm_type, model=model)

    # Normalize scenario for existing evaluators if new category
    eval_scenario = normalize_scenario_for_eval(scenario) if category in NEW_CATEGORIES else scenario
    abstraction_result = evaluate_abstraction(eval_scenario, conversation_log, llm_type=llm_type, model=model)
    behavioral_result = evaluate_behavioral(eval_scenario, conversation_log, llm_type=llm_type, model=model)

    # CD-specific: detect premature user contact (deterministic, not LLM-judged)
    if category == "cross_domain":
        agent_names = [a.get("role", a.get("name", "")) for a in scenario.get("agents", [])]
        behavioral_result["premature_user_contact"] = detect_premature_user_contact(
            conversation_log, agent_names
        )

    # Optional: CU-specific behavioral analysis
    if cu_behavioral and category == "cross_user":
        cu_behavioral_result = evaluate_cu_behavioral(scenario, conversation_log, llm_type=llm_type, model=model)
        behavioral_result["cu_specific"] = cu_behavioral_result

    # Efficiency metrics (deterministic, no LLM call)
    efficiency_result = _evaluate_efficiency(simulation)

    # Build combined result
    result = {
        "scenario_id": simulation.get("scenario_id", scenario.get("scenario_id", "unknown")),
        "category": category,
        "model": simulation.get("model", "unknown"),
        "privacy_mode": simulation.get("privacy_mode", "unknown"),
        "evaluator": {
            "llm_type": llm_type,
            "model": model or "default",
        },
        "timestamp": datetime.now().isoformat(),
        "privacy": privacy_result,
        "abstraction": abstraction_result,
        "task_completion": tcq_result,
        "behavioral": behavioral_result,
        "efficiency": efficiency_result,
    }

    # Optionally save — organize by model/privacy_mode to avoid collisions
    if output_dir:
        model_name = result["model"]
        privacy_mode = result["privacy_mode"]
        sub_dir = os.path.join(output_dir, model_name, privacy_mode)
        os.makedirs(sub_dir, exist_ok=True)
        scenario_id = result["scenario_id"]
        path = os.path.join(sub_dir, f"eval_{scenario_id}.json")
        write_json(path, result)

    return result
