"""Split long documents into overlapping pieces small enough to embed and retrieve."""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Word-based sliding window. With chunk_size=500, overlap=100 the chunks cover
    words 0-499, 400-899, 800-1299, ... The overlap means a sentence cut at one
    chunk's edge still appears whole in the next chunk.
    """
    if not 0 <= overlap < chunk_size:
        raise ValueError(f"need 0 <= overlap < chunk_size, got overlap={overlap}, chunk_size={chunk_size}")

    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break  # this chunk already reached the end; don't emit a tail that is pure overlap
        start += chunk_size - overlap
    return chunks
