"""Turn text into vectors, for retrieving similar solved examples.

Two embedders with the same interface (a callable: list[str] -> list[list[float]]):

- ApiEmbedder: any OpenAI-compatible /embeddings endpoint, for example OpenAI
  (text-embedding-3-small), Mistral (mistral-embed) or a local Ollama model
  (nomic-embed-text). Configured with environment variables:

      EMBED_BASE_URL   defaults to LLM_BASE_URL
      EMBED_API_KEY    defaults to LLM_API_KEY
      EMBED_MODEL      required to use the API embedder

  Groq has no embeddings endpoint: with Groq for the LLM, point EMBED_BASE_URL to
  another provider or use the local embedder.

- HashingEmbedder: a local, dependency-free embedder based on hashed word and
  character n-grams. It needs no download and no API key, works in English and
  French, and keeps the tests fully offline. It captures shared vocabulary, not
  meaning, so an API embedding model retrieves better on paraphrases.
"""
import hashlib
import math
import os
import re
import unicodedata

import requests


class EmbeddingError(Exception):
    pass


class ApiEmbedder:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, timeout: int = 60):
        self.model = model or os.environ.get("EMBED_MODEL")
        if not self.model:
            raise EmbeddingError("Set EMBED_MODEL to use an embeddings API.")
        self.base_url = (base_url or os.environ.get("EMBED_BASE_URL")
                         or os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("EMBED_API_KEY") or os.environ.get("LLM_API_KEY", "ollama")
        self.timeout = timeout
        self.name = f"api:{self.base_url}:{self.model}"

    def __call__(self, texts: list[str]) -> list[list[float]]:
        r = requests.post(f"{self.base_url}/embeddings",
                          headers={"Authorization": f"Bearer {self.api_key}"},
                          json={"model": self.model, "input": texts}, timeout=self.timeout)
        if r.status_code != 200:
            raise EmbeddingError(f"Embeddings API returned {r.status_code}: {r.text[:300]}")
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))   # é -> e
    return re.sub(r"[^a-z0-9 ]+", " ", text)


class HashingEmbedder:
    """Hashed bag of words and character trigrams, L2-normalised."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing:{dim}"

    def _features(self, text: str) -> list[str]:
        words = _normalise(text).split()
        feats = [f"w:{w}" for w in words]
        feats += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"#{w}#"
            feats += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
        return feats

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for f in self._features(text):
            h = int.from_bytes(hashlib.md5(f.encode()).digest()[:8], "little")
            vec[h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


def default_embedder():
    """The API embedder when EMBED_MODEL is set, otherwise the local one."""
    return ApiEmbedder() if os.environ.get("EMBED_MODEL") else HashingEmbedder()
