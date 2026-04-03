# prism/core/json_utils.py
"""JSON parsing and file I/O utilities. Forked from MAGPIE create_datapoints.py."""
import json
import os
import re


def coerce_json(text: str) -> dict:
    """Robustly extract JSON from LLM responses.

    Handles: markdown fences, surrounding text, trailing commas.
    Raises ValueError for None/empty, json.JSONDecodeError for unparseable.
    """
    if not text:
        raise ValueError("Empty or None input")

    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)

    # Extract outermost {...} block using brace counting
    start = text.find('{')
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    depth = 0
    in_string = False
    escape_next = False
    end = -1
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        raise json.JSONDecodeError("No complete JSON object found", text, 0)
    text = text[start:end + 1]

    # Patch trailing commas: ,\n} or ,\n]
    text = re.sub(r',\s*(\}|\])', r'\1', text)

    return json.loads(text)


def write_json(path: str, obj, indent: int = 2):
    """Write object as pretty JSON. Creates parent directories."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def read_yaml_text(path: str) -> str:
    """Read a YAML/text file and return raw contents."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
