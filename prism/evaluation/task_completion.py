# prism/evaluation/task_completion.py
"""Task Completion Quality (TCQ) evaluation. Adapted from MAGPIE's analyze_task_completion."""
import os
import json
import yaml

from prism.core.llm import generate
from prism.core.json_utils import coerce_json

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "task_judge.yaml")


def parse_tcq_result(llm_response: str) -> dict:
    """Parse the LLM judge's task completion response."""
    parsed = coerce_json(llm_response)

    tcq_score = parsed.get("tcq_score", 0.0)
    tcq_score = max(0.0, min(1.0, float(tcq_score)))  # clamp to [0, 1]

    return {
        "task_completed": parsed.get("task_completed", False),
        "tcq_score": tcq_score,
        "justification": parsed.get("justification", "No justification provided"),
    }


def evaluate_task_completion(
    scenario: dict,
    conversation_log: list[dict],
    llm_type: str = "gemini",
    model: str = None,
) -> dict:
    """Evaluate task completion quality in a simulation.

    Args:
        scenario: The scenario dict (needs task and success_criteria)
        conversation_log: List of message dicts from simulation output
        llm_type: Provider — "gemini", "openai", "together", "bedrock"
        model: Model name. Defaults per provider if None.

    Returns:
        Task completion evaluation result dict with keys:
        - task_completed: bool
        - tcq_score: float in [0, 1]
        - justification: str
    """
    with open(PROMPT_PATH) as f:
        prompt_template = yaml.safe_load(f)["prompt"]

    task = scenario.get("task", scenario.get("coordination_task",
           scenario.get("mediation_task", scenario.get("interaction_task", ""))))
    success_criteria = scenario.get("success_criteria", {})

    conversation_text = "\n".join(
        f"{msg.get('sender', '?')}: {msg.get('content', '')}"
        for msg in conversation_log
        if msg.get("type") != "system"
    )

    prompt = prompt_template.format(
        task=task,
        success_criteria=json.dumps(success_criteria, indent=2),
        conversation=conversation_text,
    )

    response = generate(prompt, llm_type=llm_type, model=model, temperature=0.1)
    return parse_tcq_result(response)
