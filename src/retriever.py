"""Retrieve the solved examples closest to a new question (retrieval-augmented generation).

The example bank (examples/examples.json) holds question -> SQL pairs written for this
warehouse. Their questions are embedded and stored in a Chroma vector database on
disk. For a new question, the k nearest examples by cosine similarity are added to
the prompt as few-shot examples, so the LLM sees how similar questions were solved.

The collection is rebuilt automatically when the examples or the embedder change.
None of the evaluation questions are in the example bank (a test checks it), so
the evaluation measures generalisation, not recall.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from embeddings import default_embedder

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_PATH = ROOT / "examples" / "examples.json"
STORE_PATH = ROOT / ".chroma"


@dataclass
class Example:
    question: str
    sql: str
    score: float = 0.0          # cosine similarity to the asked question


def load_examples(path: Path = EXAMPLES_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


class ExampleRetriever:
    def __init__(self, examples: list[dict] | None = None, embedder=None,
                 store_path: Path | None = STORE_PATH, collection: str = "sql_examples"):
        self.examples = examples if examples is not None else load_examples()
        self.embedder = embedder or default_embedder()
        settings = Settings(anonymized_telemetry=False)
        self.client = (chromadb.PersistentClient(path=str(store_path), settings=settings)
                       if store_path else chromadb.EphemeralClient(settings=settings))
        fingerprint = hashlib.sha256(json.dumps(
            [self.embedder.name, self.examples], sort_keys=True).encode()).hexdigest()[:16]
        name = f"{collection}_{fingerprint}"
        existing = {c.name if hasattr(c, "name") else c for c in self.client.list_collections()}
        if name in existing:
            self.collection = self.client.get_collection(name)
        else:
            for old in existing:                      # drop stale versions of the bank
                if str(old).startswith(collection + "_"):
                    self.client.delete_collection(str(old))
            self.collection = self.client.create_collection(
                name, metadata={"hnsw:space": "cosine"})
            self._index()

    def _index(self):
        questions = [e["question"] for e in self.examples]
        self.collection.add(
            ids=[str(i) for i in range(len(self.examples))],
            embeddings=self.embedder(questions),
            documents=questions,
            metadatas=[{"sql": e["sql"]} for e in self.examples])

    def search(self, question: str, k: int = 3, min_score: float = 0.0) -> list[Example]:
        if not self.examples or k <= 0:
            return []
        res = self.collection.query(query_embeddings=self.embedder([question]),
                                    n_results=min(k, len(self.examples)))
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            score = 1.0 - dist                      # cosine distance -> similarity
            if score >= min_score:
                out.append(Example(doc, meta["sql"], round(score, 4)))
        return out


def format_examples(examples: list[Example]) -> str:
    if not examples:
        return ""
    parts = [f"Question: {e.question}\nSQL:\n```sql\n{e.sql}\n```" for e in examples]
    return ("Solved examples on this warehouse, retrieved because they are similar to the "
            "question. Reuse their patterns when they apply:\n\n" + "\n\n".join(parts))
