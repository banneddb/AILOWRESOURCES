"""
RAG-augmented chat script.

Pipeline:
  1. Load & chunk your source docs (once, or whenever docs change)
  2. Embed each chunk and store in ChromaDB
  3. At query time: embed the user's question, retrieve top-k similar chunks
  4. Inject retrieved chunks into the prompt sent to Claude

pip install anthropic chromadb sentence-transformers python-dotenv
"""

from dotenv import load_dotenv
load_dotenv()

import os
import anthropic
import chromadb
from chromadb.utils import embedding_functions

client = anthropic.Anthropic()

# --- 1. Vector DB setup ---------------------------------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# ChromaDB needs an embedding function. sentence-transformers is a solid
# free local default; swap this out if you're using a hosted embedding API.
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="docs",
    embedding_function=embed_fn,
)


# --- 2. Chunking + ingestion (run once to populate the DB) ----------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    """Simple word-based sliding-window chunker."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def ingest_document(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    ids = [f"{os.path.basename(filepath)}-{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=[{"source": filepath} for _ in chunks],
    )
    print(f"Ingested {len(chunks)} chunks from {filepath}")


# --- 3. Retrieval -----------------------------------------------------------
def retrieve_context(query: str, k: int = 5) -> str:
    results = collection.query(query_texts=[query], n_results=k)
    retrieved_chunks = results["documents"][0]
    return "\n\n---\n\n".join(retrieved_chunks)


# --- 4. RAG-augmented call to Claude ---------------------------------------
def ask_with_rag(user_input: str) -> str:
    context = retrieve_context(user_input)

    system_prompt = (
        "You are a helpful assistant. Use the retrieved context below to "
        "answer the user's question. If the context doesn't contain the "
        "answer, say so rather than guessing.\n\n"
        f"Retrieved context:\n{context}"
    )

    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_input}],
    )

    return "".join(block.text for block in message.content if block.type == "text")


if __name__ == "__main__":
    # One-time setup: point this at your API docs / source files
    # ingest_document("bridges_api_docs.txt")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break

        answer = ask_with_rag(user_input)
        print("Claude:", answer)