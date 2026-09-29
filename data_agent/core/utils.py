"""Core utilities for string sanitation and code extraction."""

import re
from typing import List


def extract_python_code(raw_text: str) -> str:
    """Extract and sanitize raw executable Python code from various LLM output formats.

    Robustly handles:
    1. Standard fenced markdown blocks: ```python ... ```
    2. Generic fenced markdown blocks: ``` ... ```
    3. Truncated output with missing closing backticks: ```python ... (EOF)
    4. Raw Python code preceded/followed by conversational explanations without backticks.
    """
    text = (raw_text or "").strip()
    if not text:
        return ""

    # Case 1 & 2: Complete fenced code blocks (```python ... ``` or ``` ... ```)
    fenced_pattern = r"```(?:python|py)?\s*\n([\s\S]*?)\n```"
    matches = re.findall(fenced_pattern, text, flags=re.IGNORECASE)
    if matches:
        # Return the last or longest non-empty block
        candidates = [m.strip() for m in matches if m.strip()]
        if candidates:
            return candidates[-1]

    # Case 3: Missing closing backticks (truncated LLM output starting with ```python or ```)
    unclosed_pattern = r"```(?:python|py)?\s*\n([\s\S]*)$"
    unclosed_match = re.search(unclosed_pattern, text, flags=re.IGNORECASE)
    if unclosed_match:
        content = unclosed_match.group(1).strip()
        # In case there are trailing backticks at the very end without a preceding newline
        content = re.sub(r"```+$", "", content).strip()
        if content:
            return content

    # Case 4: No markdown backticks or conversational explanations surrounding code
    lines = text.splitlines()
    code_lines: List[str] = []
    in_code = False

    # Signatures that strongly indicate Python code lines
    code_starter_patterns = (
        r"^(?:import |from |def |class |with |for |while |if |try:|async def )",
        r"^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*",  # assignments like df = ...
        r"^(?:fig|df|plt|print)\b",
    )

    for line in lines:
        stripped = line.strip()

        # Check if line looks like start of python code
        if not in_code:
            if any(re.match(p, stripped) for p in code_starter_patterns):
                in_code = True
                code_lines.append(line)
        else:
            # Check for conversational sign-offs that signify end of code
            conversational_end = (
                stripped.lower().startswith("hope this helps")
                or stripped.lower().startswith("let me know")
                or stripped.lower().startswith("this code will")
                or stripped.lower().startswith("in this code,")
                or stripped.lower().startswith("explanation:")
                or stripped.lower().startswith("note:")
            )
            if conversational_end:
                break
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()

    # Fallback: strip any outer backticks and return
    cleaned = re.sub(r"^```(?:python)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    return cleaned


def clean_code_snippet(raw_text: str) -> str:
    """Alias for backwards compatibility with existing callers."""
    return extract_python_code(raw_text)
