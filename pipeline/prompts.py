"""
Role prompts live in prompts/<name>.txt so they can be read and edited without
touching code. Each run logs a short hash of every prompt it used, so the lab can
tell which prompt version produced which result.
"""

import hashlib
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> tuple[str, str]:
    """Return (prompt text, 12-character version hash)."""
    text = (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def render(template: str, **values) -> str:
    """Fill {name} placeholders. Plain replace, so braces in code or docs are left alone."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def build_request(concepts: str, dataset_description: str) -> str:
    """The user's request as one block of text; every mode starts from this."""
    parts = [f"Requested concepts: {concepts}"]
    if dataset_description:
        parts.append(f"Dataset: {dataset_description}")
    return "\n".join(parts)
