"""Tests for the example bank, the embedder and the vector-database retrieval.
Offline: they use the local hashing embedder and an in-memory Chroma collection."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from assistant import Assistant  # noqa: E402
from embeddings import HashingEmbedder  # noqa: E402
from guard import connect_readonly, run_safe  # noqa: E402
from retriever import ExampleRetriever, format_examples, load_examples  # noqa: E402

DB = ROOT / "warehouse" / "retail.db"
EXAMPLES = load_examples()
EVAL = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def retriever():
    return ExampleRetriever(embedder=HashingEmbedder(), store_path=None)


def test_every_example_runs_through_the_guard():
    con = connect_readonly(DB)
    for e in EXAMPLES:
        res = run_safe(con, e["sql"])
        assert res.rows, e["question"]


def test_no_evaluation_question_leaks_into_the_example_bank():
    norm = lambda s: " ".join(s.lower().split())     # noqa: E731
    eval_q = {norm(q["question"]) for q in EVAL}
    eval_sql = {norm(q["sql"]) for q in EVAL}
    for e in EXAMPLES:
        assert norm(e["question"]) not in eval_q
        assert norm(e["sql"]) not in eval_sql


def test_embeddings_are_normalised_and_deterministic():
    emb = HashingEmbedder()
    a, b = emb(["Quelle est la marge totale ?", "Quelle est la marge totale ?"])
    assert a == b
    assert abs(sum(v * v for v in a) - 1.0) < 1e-9


@pytest.mark.parametrize("question, expected", [
    ("How many customers are based in Sousse?", "How many customers live in Sfax?"),
    ("Show the running total of revenue by month for 2025",
     "What is the cumulative revenue month by month in 2024?"),
    ("Combien de clients B2B avons-nous ?", "Combien de clients B2C avons-nous ?"),
])
def test_retrieves_the_most_similar_example(retriever, question, expected):
    top = retriever.search(question, k=3)
    assert expected in [e.question for e in top]
    assert top == sorted(top, key=lambda e: -e.score)


def test_retrieved_examples_are_added_to_the_prompt(retriever):
    seen = {}

    def fake_llm(messages):
        seen["system"] = messages[0]["content"]
        return "```sql\nSELECT COUNT(*) FROM dim_customer WHERE city = 'Sousse'\n```"

    answer = Assistant(DB, llm=fake_llm, retriever=retriever, k=2).ask(
        "How many customers are based in Sousse?")
    assert answer.result.rows[0][0] > 0
    assert len(answer.examples) == 2
    assert format_examples(answer.examples) in seen["system"]


def test_without_retriever_the_prompt_has_no_examples():
    seen = {}

    def fake_llm(messages):
        seen["system"] = messages[0]["content"]
        return "SELECT 1 FROM dim_store LIMIT 1"

    answer = Assistant(DB, llm=fake_llm).ask("Anything")
    assert answer.examples == [] and "Solved examples" not in seen["system"]


def test_persistent_store_is_reused_and_rebuilt_when_examples_change(tmp_path):
    emb = HashingEmbedder()
    first = ExampleRetriever(EXAMPLES[:5], emb, store_path=tmp_path)
    again = ExampleRetriever(EXAMPLES[:5], emb, store_path=tmp_path)
    assert again.collection.name == first.collection.name
    changed = ExampleRetriever(EXAMPLES[:6], emb, store_path=tmp_path)
    assert changed.collection.name != first.collection.name
    assert changed.collection.count() == 6


def test_api_embedder_sends_texts_and_keeps_their_order(monkeypatch):
    from embeddings import ApiEmbedder
    sent = {}

    class FakeResponse:
        status_code = 200

        def json(self):        # the API may return items out of order
            return {"data": [{"index": 1, "embedding": [0.0, 1.0]},
                             {"index": 0, "embedding": [1.0, 0.0]}]}

    def fake_post(url, headers, json, timeout):
        sent.update(url=url, body=json)
        return FakeResponse()

    monkeypatch.setattr("embeddings.requests.post", fake_post)
    emb = ApiEmbedder(model="mistral-embed", base_url="https://api.mistral.ai/v1", api_key="k")
    assert emb(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert sent["url"].endswith("/embeddings") and sent["body"]["input"] == ["a", "b"]
