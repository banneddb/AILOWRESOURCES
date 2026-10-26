"""
Agent 2 (Critique): do the retrieved passages actually provide what each concept
needs? Claude judges each concept, then a deterministic check keeps it honest:
any function it names must exist in the corpus's API index. A "fits" verdict
backed only by names that don't exist is turned into "retry".
"""

from ..api_index import ApiIndex
from ..llm import call_claude_json
from ..prompts import load_prompt, render
from ..retrieve import Chunk

SCHEMA = {
    "type": "object",
    "properties": {
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["fits", "retry", "unsupported"]},
                    "apis_found": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                    "suggestion": {"type": "string"},
                },
                "required": ["name", "verdict", "apis_found", "reason", "suggestion"],
                "additionalProperties": False,
            },
        },
        "action": {"type": "string", "enum": ["accept", "rewrite_queries", "redecompose"]},
        "reason": {"type": "string"},
    },
    "required": ["concepts", "action", "reason"],
    "additionalProperties": False,
}


def _bare_name(api: str) -> str:
    """'`sound.getSamples(s)`' -> 'getSamples'"""
    return api.strip().strip("`").split("(")[0].split(".")[-1].strip()


def _passages(concepts: list[dict], results: dict[str, list[Chunk]]) -> str:
    sections = []
    for c in concepts:
        chunks = results.get(c["name"], [])
        body = "\n\n".join(f"[{ch.id}]\n{ch.text}" for ch in chunks) or "(no passages found)"
        sections.append(f"## Concept: {c['name']}\nNeeds from the library: {c['api_need']}\n\n{body}")
    return "\n\n".join(sections)


def critique(cfg, log, *, corpus: str, concepts: list[dict], results: dict[str, list[Chunk]],
             index: ApiIndex) -> dict:
    template, version = load_prompt("critic")
    log.add_prompt("critic", version)
    system = render(template, corpus=corpus)
    wanted = {c["name"] for c in concepts}

    def validate(data):
        missing = sorted(wanted - {c["name"] for c in data["concepts"]})
        return f"no verdict for: {missing} (use the concept names exactly as given)" if missing else None

    data = call_claude_json(cfg, log, role="critic", system=system, user=_passages(concepts, results),
                            schema=SCHEMA, validate=validate)

    entries = []
    for c in data["concepts"]:
        if c["name"] not in wanted:
            continue
        apis = list(dict.fromkeys(n for n in map(_bare_name, c["apis_found"]) if n))
        verified = [a for a in apis if a in index]
        verdict, reason = c["verdict"], c["reason"]
        if verdict == "fits" and not verified:
            verdict = "retry"
            reason += " [Automatic check: none of the functions named exist in the corpus.]"
        entries.append({
            "name": c["name"],
            "verdict": verdict,
            "critic_verdict": c["verdict"],     # what Claude said, before the check
            "apis_verified": verified,
            "apis_not_in_corpus": [a for a in apis if a not in index],
            "reason": reason,
            "suggestion": c["suggestion"],
        })

    # Keep the action consistent with the (checked) verdicts.
    action = data["action"]
    any_retry = any(e["verdict"] == "retry" for e in entries)
    if action == "accept" and any_retry:
        action = "rewrite_queries"
    elif action == "rewrite_queries" and not any_retry:
        action = "accept"
    return {"action": action, "critic_action": data["action"], "reason": data["reason"], "concepts": entries}
