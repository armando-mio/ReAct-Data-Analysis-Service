"""Core utilities for string sanitation and code extraction."""

import re


def clean_code_snippet(raw_text: str) -> str:
    """Extract and sanitize raw executable Python code from markdown blocks or backticks.

    Handles fenced code blocks (```python ... ```), unfenced blocks, and mixed text.
    """
    text = (raw_text or "").strip()
    pattern = r"```(?:python)?\s*([\s\S]*?)\s*```"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
