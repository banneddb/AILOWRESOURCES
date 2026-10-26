"""
Where the vectors live (ChromaDB).

Each corpus gets its own collection per embedding model and chunk setting, e.g.
"bridges__all-MiniLM-L6-v2__c500_o100". Vectors from different embedding models
can't be compared with each other, and chunks of different sizes give different
retrieval results, so mixing them in one collection would silently corrupt an
experiment. Separate collections also let you switch settings without re-ingesting.
"""

import re

import chromadb

from .config import Config


def collection_name(cfg: Config, corpus: str) -> str:
    model_slug = re.sub(r"[^A-Za-z0-9._-]", "-", cfg.model)  # e.g. "qwen3-embedding:0.6b" -> "qwen3-embedding-0.6b"
    return f"{corpus}__{model_slug}__c{cfg.chunk_size}_o{cfg.overlap}"


def get_client(cfg: Config):
    return chromadb.PersistentClient(path=cfg.chroma_dir)


def get_collection(cfg: Config, corpus: str, create: bool = False):
    client = get_client(cfg)
    name = collection_name(cfg, corpus)
    if create:
        return client.get_or_create_collection(
            name=name,
            # Cosine distance compares the direction of vectors, the usual choice for text.
            configuration={"hnsw": {"space": "cosine"}},
            metadata={
                "corpus": corpus,
                "embedding_provider": cfg.provider,
                "embedding_model": cfg.model,
                "chunk_size": cfg.chunk_size,
                "overlap": cfg.overlap,
            },
        )
    if name not in [c.name for c in client.list_collections()]:
        raise LookupError(
            f"No collection {name!r}. Run:  python run.py ingest {corpus}\n"
            "(Changing the embedding model or chunk settings needs a fresh ingest.)"
        )
    return client.get_collection(name)


def delete_collection(cfg: Config, corpus: str) -> bool:
    client = get_client(cfg)
    name = collection_name(cfg, corpus)
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
        return True
    return False
