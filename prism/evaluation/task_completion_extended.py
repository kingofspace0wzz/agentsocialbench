# prism/evaluation/task_completion_extended.py
"""Category-specific TCQ evaluation for multi-party categories."""
import os
import json
import yaml

from prism.core.llm import generate
from prism.evaluation.task_completion import parse_tcq_result

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _evaluate_tcq_with_prompt(prompt_file, scenario, conversation_log, llm_type="gemini", model=None):
    """Shared helper: load category prompt, format, call LLM, parse result."""
    with open(os.path.join(PROMPTS_DIR, prompt_file)) as f:
        prompt_template = yaml.safe_load(f)["prompt"]

    task = scenario.get("task", "")
    success_criteria = scenario.get("success_criteria", {})
    participants = ", ".join(p["user_name"] for p in scenario.get("participants", []))

    conversation_text = "\n".join(
        f"{msg.get('sender', '?')}: {msg.get('content', '')}"
        for msg in conversation_log
        if msg.get("type") != "system"
    )

    prompt = prompt_template.format(
        task=task,
        success_criteria=json.dumps(success_criteria, indent=2),
        participants=participants,
        conversation=conversation_text,
    )

    response = generate(prompt, llm_type=llm_type, model=model, temperature=0.1)
    return parse_tcq_result(response)


def evaluate_task_completion_mg(scenario, conversation_log, llm_type="gemini", model=None):
    return _evaluate_tcq_with_prompt("mg_task_judge.yaml", scenario, conversation_log, llm_type, model)

def evaluate_task_completion_hs(scenario, conversation_log, llm_type="gemini", model=None):
    return _evaluate_tcq_with_prompt("hs_task_judge.yaml", scenario, conversation_log, llm_type, model)

def evaluate_task_completion_cm(scenario, conversation_log, llm_type="gemini", model=None):
    return _evaluate_tcq_with_prompt("cm_task_judge.yaml", scenario, conversation_log, llm_type, model)

def evaluate_task_completion_am(scenario, conversation_log, llm_type="gemini", model=None):
    return _evaluate_tcq_with_prompt("am_task_judge.yaml", scenario, conversation_log, llm_type, model)
