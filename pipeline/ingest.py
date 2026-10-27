"""
Ingest a corpus: read every doc file in data/<corpus>/, chunk it, embed the
chunks, and store them in that corpus's Chroma collection.
"""

import re
from html.parser import HTMLParser
from pathlib import Path

from .chunking import chunk_text
from .config import Config
from .embeddings import make_embedder
from .store import collection_name, delete_collection, get_collection


class _HTMLToText(HTMLParser):
    """Keep the visible text of an HTML page (e.g. generated API docs), drop tags/scripts."""
    SKIP = {"script", "style"}
    # Tags that start a new line on the rendered page, so they start one in the text too.
    BLOCK = {"p", "br", "div", "li", "tr", "pre", "table", "ul", "ol", "dl", "dt", "dd",
             "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "hr"}

    def __init__(self):
        super().__init__()
        self.parts, self._skipping = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skipping += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skipping:
            self._skipping -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skipping:
            self.parts.append(data)


def read_doc(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in (".html", ".htm"):
        parser = _HTMLToText()
        parser.feed(text)
        text = "".join(parser.parts)
        text = re.sub(r"[ \t]+\n", "\n", text)          # trailing spaces
        text = re.sub(r"\n{3,}", "\n\n", text).strip()  # at most one blank line in a row
    return text


def iter_doc_files(corpus_dir: Path, extensions: tuple, include: list | None = None) -> list[Path]:
    """The corpus's include patterns if it has them, otherwise every file with an allowed extension."""
    if include:
        matches = {p for pattern in include for p in corpus_dir.glob(pattern)}
    else:
        allowed = {e.lower() for e in extensions}
        matches = {p for p in corpus_dir.rglob("*") if p.suffix.lower() in allowed}
    return sorted(p for p in matches if p.is_file())


def ingest_corpus(cfg: Config, corpus: str, reset: bool = False) -> dict:
    corpus_dir = Path(cfg.data_dir) / corpus
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"No folder {corpus_dir}. Put the {corpus} docs there first.")

    include = cfg.corpus_include.get(corpus)
    files = iter_doc_files(corpus_dir, cfg.extensions, include)
    if not files:
        rule = f"include patterns {include}" if include else f"extensions {cfg.extensions}"
        raise FileNotFoundError(f"No files matching {rule} in {corpus_dir}")

    if reset:  # start clean, e.g. after deleting doc files
        delete_collection(cfg, corpus)
    collection = get_collection(cfg, corpus, create=True)
    embedder = make_embedder(cfg)

    total_chunks = 0
    for path in files:
        source = path.relative_to(corpus_dir).as_posix()
        chunks = chunk_text(read_doc(path), cfg.chunk_size, cfg.overlap)
        # Remove this file's old chunks first, so a file that got shorter
        # doesn't leave stale chunks behind.
        collection.delete(where={"source": source})
        if not chunks:
            continue
        collection.upsert(
            ids=[f"{source}::{i}" for i in range(len(chunks))],  # full path, so same-named files don't collide
            documents=chunks,
            embeddings=embedder.embed(chunks),
            metadatas=[{"source": source, "chunk_index": i} for i in range(len(chunks))],
        )
        total_chunks += len(chunks)
        print(f"  {source}: {len(chunks)} chunks")

    return {
        "collection": collection_name(cfg, corpus),
        "files": len(files),
        "chunks": total_chunks,
        "collection_size": collection.count(),
    }
