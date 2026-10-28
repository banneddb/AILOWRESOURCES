"""Split long documents into overlapping pieces small enough to embed and retrieve."""

import re

WORD = re.compile(r"\S+")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Word-based sliding window. With chunk_size=500, overlap=100 the chunks cover
    words 0-499, 400-899, 800-1299, ... The overlap means a sentence cut at one
    chunk's edge still appears whole in the next chunk.

    Each chunk is cut out of the original text (from its first word to its last),
    so line breaks and indentation survive. That keeps code excerpts readable and
    lets context_budget_lines count real lines.
    """
    if not 0 <= overlap < chunk_size:
        raise ValueError(f"need 0 <= overlap < chunk_size, got overlap={overlap}, chunk_size={chunk_size}")

    words = [(m.start(), m.end()) for m in WORD.finditer(text)]   # where each word sits in the text
    chunks = []
    start = 0
    while start < len(words):
        last = min(start + chunk_size, len(words)) - 1
        chunks.append(text[words[start][0]:words[last][1]])
        if start + chunk_size >= len(words):
            break  # this chunk already reached the end; don't emit a tail that is pure overlap
        start += chunk_size - overlap
    return chunks
