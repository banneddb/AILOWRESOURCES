"""
Agent 1 (Retrieve): Claude writes search queries for each concept, then plain
code runs them against ChromaDB. On later loop iterations Claude also sees why
earlier results were rejected, so it can try different wording.
"""

import json

from ..llm import call_claude_json
from ..prompts import load_prompt, render
from ..retrieve import Chunk, retrieve

SCHEMA = {
    "type": "object",
    "properties": {
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "queries": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "queries"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["concepts"],
    "additionalProperties": False,
}


def write_queries(cfg, log, *, corpus: str, concepts: list[dict], attempts: dict) -> dict[str, list[str]]:
    """Return {concept name: [search queries]}. `attempts` holds earlier queries + critic notes per concept."""
    template, version = load_prompt("retriever")
    log.add_prompt("retriever", version)
    system = render(template, corpus=corpus, max_queries=cfg.max_queries_per_concept)
    user = json.dumps([
        {"name": c["name"], "api_need": c["api_need"], "earlier_attempts": attempts.get(c["name"], [])}
        for c in concepts
    ], indent=2)
    wanted = {c["name"] for c in concepts}

    def validate(data):
        got = {c["name"]: [q for q in c["queries"] if q.strip()] for c in data["concepts"]}
        missing = sorted(n for n in wanted if not got.get(n))
        return f"no queries for: {missing} (use the concept names exactly as given)" if missing else None

    data = call_claude_json(cfg, log, role="retriever", system=system, user=user,
                            schema=SCHEMA, validate=validate)
    return {
        c["name"]: [q.strip() for q in c["queries"] if q.strip()][:cfg.max_queries_per_concept]
        for c in data["concepts"] if c["name"] in wanted
    }


def run_queries(cfg, corpus: str, queries: dict[str, list[str]], embedder) -> tuple[dict[str, list[Chunk]], list[dict]]:
    """
    Search for every query. Each concept keeps its top_k chunks across all of its
    queries (a chunk found twice keeps its best score). Also returns the log entries.
    """
    by_concept, log_entries = {}, []
    for name, concept_queries in queries.items():
        best: dict[str, Chunk] = {}
        for query in concept_queries:
            chunks = retrieve(cfg, corpus, query, embedder)
            log_entries.append({"concept": name, "query": query, "chunks": [c.log_entry() for c in chunks]})
            for chunk in chunks:
                if chunk.id not in best or chunk.score > best[chunk.id].score:
                    best[chunk.id] = chunk
        by_concept[name] = sorted(best.values(), key=lambda c: c.score, reverse=True)[:cfg.top_k]
    return by_concept, log_entries
