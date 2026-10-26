"""Find the chunks most similar to a query."""

from dataclasses import asdict, dataclass

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
