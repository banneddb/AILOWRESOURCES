"""
Turn text into embeddings: lists of numbers where similar meanings end up close
together. Retrieval works by embedding the question and finding the nearest chunks.

Two backends, chosen by `provider` in config.toml:
  - sentence-transformers: the model runs inside this Python process.
  - ollama: the model runs in the Ollama app; we send it text over HTTP.

We compute embeddings ourselves and hand Chroma plain vectors, so it is always
clear which model produced them.
"""

import json
import urllib.error
import urllib.request

from .config import Config


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        # Imported here because loading torch is slow; only pay for it when used.
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class OllamaEmbedder:
    BATCH_SIZE = 32

    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.url = base_url.rstrip("/") + "/api/embed"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            vectors.extend(self._post(texts[i:i + self.BATCH_SIZE]))
        return vectors

    def _post(self, batch: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model_name, "input": batch}).encode("utf-8")
        request = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.loads(response.read())["embeddings"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Ollama rejected the request ({e.code}): {e.read().decode(errors='replace')}. "
                f"Has the model been pulled?  ollama pull {self.model_name}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Can't reach Ollama at {self.url}. Is the Ollama app running?") from e


def make_embedder(cfg: Config):
    if cfg.provider == "sentence-transformers":
        return SentenceTransformerEmbedder(cfg.model)
    if cfg.provider == "ollama":
        return OllamaEmbedder(cfg.model, cfg.ollama_url)
    raise ValueError(f"Unknown embedding provider: {cfg.provider}")
