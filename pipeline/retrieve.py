"""Find the chunks most similar to a query."""

from dataclasses import asdict, dataclass, replace

from .config import Config
from .store import get_collection


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    score: float   # cosine similarity: 1.0 = same direction, ~0 = unrelated

    def log_entry(self) -> dict:
        """What goes in the run log (the chunk text is skipped; the id is enough to look it up)."""
        entry = asdict(self)
        del entry["text"]
        return entry


def retrieve(cfg: Config, corpus: str, query: str, embedder, k: int | None = None) -> list[Chunk]:
    collection = get_collection(cfg, corpus)
    results = collection.query(
        query_embeddings=embedder.embed([query]),
        n_results=k or cfg.top_k,
        include=["documents", "metadatas", "distances"],
    )
    # Chroma returns one list per query; we sent one query, so take [0].
    return [
        Chunk(id=chunk_id, text=doc, source=meta["source"], score=round(1 - distance, 4))
        for chunk_id, doc, meta, distance in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def format_context(chunks: list[Chunk]) -> str:
    """Join chunks into one block of text for the prompt, labelled by source."""
    return "\n\n---\n\n".join(f"[{c.id}]\n{c.text}" for c in chunks)


def apply_budget(chunks: list[Chunk], max_lines: int) -> tuple[list[Chunk], dict]:
    """
    Cap how much documentation the generator sees (context_budget_lines).

    Chunks are taken best score first until max_lines lines are used; the chunk
    that crosses the limit is cut short so the total is exactly max_lines.
    0 means no limit (every chunk, still best score first).
    Returns the chunks to send and a summary for the run log.
    """
    kept, used, truncated = [], 0, None
    for chunk in sorted(chunks, key=lambda c: c.score, reverse=True):
        lines = chunk.text.splitlines()
        if max_lines:
            room = max_lines - used
            if room <= 0:
                break
            if len(lines) > room:
                lines = lines[:room]
                chunk = replace(chunk, text="\n".join(lines))
                truncated = chunk.id
        kept.append(chunk)
        used += len(lines)
    summary = {
        "budget_lines": max_lines,                  # 0 = no limit
        "chunks_available": len(chunks),
        "chunks_given": [c.id for c in kept],
        "truncated_chunk": truncated,               # the chunk cut short to fit, if any
        "lines_given": used,
        "chars_given": sum(len(c.text) for c in kept),
    }
    return kept, summary
