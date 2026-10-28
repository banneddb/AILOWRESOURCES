"""
Decompose: turn the request into a list of concepts, each saying what it needs
from the API library. Concepts that are plain Python (e.g. "for loops") need
nothing from the library and skip retrieval.
"""

from ..llm import call_claude_json
from ..prompts import load_prompt, render

SCHEMA = {
    "type": "object",
    "properties": {
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "api_need": {"type": "string"},   # "" = plain Python, nothing needed from the library
                },
                "required": ["name", "api_need"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["concepts"],
    "additionalProperties": False,
}


def decompose(cfg, log, *, corpus: str, request: str, feedback: str | None = None) -> list[dict]:
    template, version = load_prompt("decompose")
    log.add_prompt("decompose", version)
    system = render(template, corpus=corpus, max_concepts=cfg.max_concepts)
    user = request
    if feedback:
        user += f"\n\nA previous plan didn't work for this library:\n{feedback}\n\nMake a new plan."

    def validate(data):
        names = [c["name"].strip() for c in data["concepts"]]
        if not names:
            return "list at least one concept"
        if len(names) > cfg.max_concepts:
            return f"list at most {cfg.max_concepts} concepts, got {len(names)}"
        if "" in names or len(set(names)) != len(names):
            return "every concept needs a unique, non-empty name"
        return None

    data = call_claude_json(cfg, log, role="decompose", system=system, user=user,
                            schema=SCHEMA, validate=validate)
    return [{"name": c["name"].strip(), "api_need": c["api_need"].strip()} for c in data["concepts"]]
