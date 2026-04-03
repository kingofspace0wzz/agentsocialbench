# prism/core/llm.py
"""Unified LLM interface supporting OpenAI, Gemini, Together AI, and AWS Bedrock.

Forked and unified from MAGPIE's simulate_agents.py and analysis.py.
"""
import os
import json
import time
import random
from typing import Optional

from prism.core.env import load_env_file, get_gemini_api_keys


# Load env on import
load_env_file()

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-pro",
    "openai": "gpt-4o",
    "together": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "bedrock": "us.anthropic.claude-sonnet-4-5-20250514",
}


def sanitize_model_name(model: str) -> str:
    """Replace '/' with '--' for filesystem-safe folder names."""
    return model.replace("/", "--")


def resolve_model(llm_type: str, model: Optional[str]) -> str:
    """Return the explicit model or the default for the provider."""
    return model if model is not None else DEFAULT_MODELS[llm_type]


def generate(
    prompt: str | list[dict],
    llm_type: str = "gemini",
    model: str = None,
    temperature: float = 0.7,
    response_format: str = None,
    max_retries: int = 20,
) -> str:
    """Unified LLM call. Returns raw text response.

    Args:
        prompt: Either a plain string or a list of chat messages [{"role": ..., "content": ...}]
        llm_type: Provider — "gemini", "openai", "together", "bedrock"
        model: Model name. Defaults per provider if None.
        temperature: Sampling temperature.
        response_format: Set to "json" for JSON mode (OpenAI/Together only).
        max_retries: Number of retry attempts on failure.
    """
    if model is None:
        model = DEFAULT_MODELS.get(llm_type)

    if llm_type == "gemini":
        return _call_gemini(prompt, model, temperature, max_retries)
    elif llm_type == "openai":
        return _call_openai(prompt, model, temperature, response_format, max_retries)
    elif llm_type == "together":
        return _call_together(prompt, model, temperature, response_format, max_retries)
    elif llm_type == "bedrock":
        return _call_bedrock(prompt, model, temperature, max_retries)
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_type}")


def _ensure_messages(prompt: str | list[dict]) -> list[dict]:
    """Convert string prompt to chat messages format if needed."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return prompt


def _call_openai(
    prompt: str | list[dict], model: str, temperature: float,
    response_format: str, max_retries: int
) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = _ensure_messages(prompt)

    kwargs = {"model": model, "messages": messages}
    # GPT-5+ and reasoning models (o3/o4) only support the default temperature (1).
    if not model.startswith(("gpt-5", "o3", "o4")):
        kwargs["temperature"] = temperature
    if response_format == "json":
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt * 5, 60))
            else:
                raise


def _call_gemini(
    prompt: str | list[dict], model: str, temperature: float, max_retries: int
) -> str:
    from google import genai
    from google.genai import types

    api_keys = get_gemini_api_keys()
    if not api_keys:
        raise RuntimeError("No Gemini API keys found. Set GEMINI_API_KEY or GEMINI_API_KEY_1-5.")

    # Separate system messages from user messages for Gemini
    system_instruction = None
    if isinstance(prompt, list):
        system_msgs = [m["content"] for m in prompt if m.get("role") == "system"]
        user_msgs = [m["content"] for m in prompt if m.get("role") != "system"]
        if system_msgs:
            system_instruction = "\n\n".join(system_msgs)
        prompt = "\n\n".join(user_msgs)
    else:
        prompt = prompt

    for attempt in range(max_retries):
        try:
            api_key = random.choice(api_keys)
            client = genai.Client(api_key=api_key)
            config = types.GenerateContentConfig(temperature=temperature)
            if system_instruction:
                config.system_instruction = system_instruction
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt * 5, 60))
            else:
                raise


def _call_together(
    prompt: str | list[dict], model: str, temperature: float,
    response_format: str, max_retries: int
) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("TOGETHER_API_KEY"),
        base_url="https://api.together.xyz/v1",
    )
    messages = _ensure_messages(prompt)

    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if response_format == "json":
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt * 5, 60))
            else:
                raise


def _call_bedrock(
    prompt: str | list[dict], model: str, temperature: float, max_retries: int
) -> str:
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "bedrock-runtime", region_name="us-west-2",
        config=Config(read_timeout=500),
    )

    system_text = ""
    messages = []
    if isinstance(prompt, list):
        for m in prompt:
            if m.get("role") == "system":
                system_text += ("\n\n" if system_text else "") + m.get("content", "")
            else:
                messages.append({"role": m["role"], "content": [{"text": m.get("content", "")}]})
    else:
        messages = [{"role": "user", "content": [{"text": prompt}]}]

    converse_kwargs = {
        "modelId": model,
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": 16384,
            "temperature": temperature,
        },
    }
    if system_text:
        converse_kwargs["system"] = [{"text": system_text}]

    for attempt in range(max_retries):
        try:
            print(f"Calling bedrock model {model}")
            response = client.converse(**converse_kwargs)
            result = response["output"]["message"]
            # Find the first text content block (skip reasoningContent blocks
            # returned by models like MiniMax)
            for block in result["content"]:
                if "text" in block:
                    return block["text"] or ""
            raise ValueError(f"No text block in Bedrock Converse response: {result['content']}")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt * 5, 60))
            else:
                raise
