# Text-to-SQL BI assistant

Ask a business question in plain English or French ("Quel magasin a la meilleure marge ?") and get the SQL query and the answer, computed on a retail data warehouse. Similar solved examples are retrieved from a vector database (RAG), an LLM writes the SQL, and a three-layer guard makes sure it can only **read** the warehouse.

## Why this project

Business users wait for analysts to write SQL or to add a dashboard page. This assistant lets them query a star-schema warehouse directly. It also deals with the two problems that make LLM-generated SQL risky in a company: queries that are **wrong** (measured with an evaluation set) and queries that are **dangerous** (blocked by the guard).

## How it works

```
question ─► embedding ─► Chroma vector database ─► 3 most similar solved examples
        ─► prompt (rules + warehouse schema + retrieved examples) ─► LLM ─► SQL
        ─► guard: static checks ─► read-only connection ─► SQLite authorizer ─► time & row limits
        ─► result table
           └─ on an SQL error, the error goes back to the LLM for one correction attempt
```

**The warehouse** (`warehouse/build_warehouse.py`) is a simulated parapharmacy retail star schema in SQLite: `fact_sales` (60,000 sales lines, 2024–2025) with `dim_date`, `dim_product` (26 products, 6 categories), `dim_store` (7 stores in Tunisia) and `dim_customer` (1,200 B2C and B2B customers). It includes seasonality (sunscreen in summer, vitamins in winter) and 2025 growth, so the questions have real answers to find.

**The guard** (`src/guard.py`):

1. Static checks: one statement only, and it must be a `SELECT` or `WITH … SELECT`.
2. The database file is opened in read-only mode.
3. An SQLite authorizer allows reads of the five warehouse tables and nothing else: no writes, no `PRAGMA`, no `ATTACH`, no system tables.
4. Queries are stopped after 5 seconds, and results are capped at 200 rows.

If the guard refuses a query, the assistant stops: it never asks the model to find a way around it.

**Retrieval of similar examples (RAG)** (`src/embeddings.py`, `src/retriever.py`): `examples/examples.json` holds 32 question → SQL pairs written for this warehouse, in English and French, covering the patterns the model needs: joins on the star schema, year and quarter filters, margin rates, year-over-year comparisons with `CASE WHEN`, rankings and running totals with window functions, `NOT EXISTS`. Their questions are embedded and stored in a persistent **Chroma** collection (cosine similarity). For each new question, the 3 nearest examples are added to the prompt as few-shot examples. The collection is rebuilt automatically when the examples or the embedding model change.

Two embedders share the same interface:

- any OpenAI-compatible **embeddings API**: OpenAI `text-embedding-3-small`, Mistral `mistral-embed`, or a local Ollama model such as `nomic-embed-text`, set with `EMBED_BASE_URL`, `EMBED_API_KEY` and `EMBED_MODEL`;
- a **local hashing embedder** (words and character trigrams), used when `EMBED_MODEL` is not set: no download, no key, works offline and in the tests. It matches shared vocabulary rather than meaning, so an embeddings API retrieves better on paraphrases.

None of the 20 evaluation questions is in the example bank (a test checks it), so the evaluation measures how well the model generalises, not whether it recalls an answer.

**The LLM** (`src/llm.py`) is any OpenAI-compatible API: Groq or Mistral (free tiers), OpenAI, or a local model through Ollama. You choose it with environment variables (see `.env.example`).

## Evaluation

`eval/questions.json` holds 20 business questions (4 easy, 9 medium, 7 hard; one in French) with reference SQL. Examples: revenue growth 2025 vs 2024, top 3 B2B customers, margin rate by store, share of revenue from the top 10% of customers (window functions).

`eval/run_eval.py` measures **execution accuracy**: a generated query is correct when it returns the same result as the reference query, whatever its column names.

```bash
python eval/run_eval.py --oracle   # checks the harness itself: must print 20/20
python eval/run_eval.py            # evaluates your LLM, writes eval/results.json
python eval/run_eval.py --rag      # same with retrieved examples, writes eval/results_rag.json
```

Running both shows what retrieval adds for a given model.

| Model | Retrieval | Easy | Medium | Hard | Total |
|---|---|---|---|---|---|
| *(run the evaluation and fill in your results)* | no | | | | |
| | RAG, k = 3 | | | | |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python warehouse/build_warehouse.py
python -m pytest -q tests                # 23 tests, no API key needed

export LLM_BASE_URL=https://api.groq.com/openai/v1   # or set the values from .env.example
export LLM_API_KEY=...
export LLM_MODEL=llama-3.3-70b-versatile
python src/assistant.py "What was the revenue per quarter in 2025?"
python src/assistant.py --no-rag "What was the revenue per quarter in 2025?"   # without examples

# optional: an embeddings API instead of the local embedder
export EMBED_BASE_URL=https://api.mistral.ai/v1 EMBED_API_KEY=... EMBED_MODEL=mistral-embed
```

Example output, with the local embedder:

```
Retrieved examples:
  0.81  What was the revenue per month in 2025?
  0.55  What was the margin of the Center region in 2025?
  0.50  Which quarter of 2024 had the lowest revenue?

SQL (1 attempt):
SELECT d.year, ROUND(SUM(f.revenue), 2) AS revenue FROM fact_sales f JOIN dim_date d ...

year | revenue
-----+-----------
2024 | 4383066.03
2025 | 4843894.38
```

## Project structure

```
warehouse/build_warehouse.py   star-schema warehouse with simulated data
src/guard.py                   SQL safety layers and limits
src/llm.py                     OpenAI-compatible chat client
src/embeddings.py              embeddings API client and local hashing embedder
src/retriever.py               Chroma vector store and similar-example retrieval
src/assistant.py               prompt, retrieved examples, SQL extraction, self-correction, CLI
examples/examples.json         32 solved question -> SQL examples for retrieval
eval/questions.json            20 questions with reference SQL
eval/run_eval.py               execution-accuracy evaluation, with or without retrieval
tests/                         23 tests with scripted fake LLMs, fully offline
```

## Limits and next steps

- A semicolon inside a text value is refused by the one-statement check. That is a deliberate trade-off for simplicity.
- The retrieved examples guide the model but do not bypass the guard: every query still goes through the same checks.
- Next: a small web UI (Streamlit) with an automatic chart, adding validated user questions to the example bank, and a PostgreSQL version of the warehouse.
