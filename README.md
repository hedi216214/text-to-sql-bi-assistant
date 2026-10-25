# Text-to-SQL BI assistant

Ask a business question in plain English or French ("Quel magasin a la meilleure marge ?") and get the SQL query and the answer, computed on a retail data warehouse. An LLM writes the SQL; a three-layer guard makes sure it can only **read** the warehouse.

## Why this project

Business users wait for analysts to write SQL or to add a dashboard page. This assistant lets them query a star-schema warehouse directly. It also deals with the two problems that make LLM-generated SQL risky in a company: queries that are **wrong** (measured with an evaluation set) and queries that are **dangerous** (blocked by the guard).

## How it works

```
question ─► prompt (rules + warehouse schema) ─► LLM ─► SQL
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

**The LLM** (`src/llm.py`) is any OpenAI-compatible API: Groq or Mistral (free tiers), OpenAI, or a local model through Ollama. You choose it with environment variables (see `.env.example`).

## Evaluation

`eval/questions.json` holds 20 business questions (4 easy, 9 medium, 7 hard; one in French) with reference SQL. Examples: revenue growth 2025 vs 2024, top 3 B2B customers, margin rate by store, share of revenue from the top 10% of customers (window functions).

`eval/run_eval.py` measures **execution accuracy**: a generated query is correct when it returns the same result as the reference query, whatever its column names.

```bash
python eval/run_eval.py --oracle   # checks the harness itself: must print 20/20
python eval/run_eval.py            # evaluates your LLM, writes eval/results.json
```

| Model | Easy | Medium | Hard | Total |
|---|---|---|---|---|
| *(run the evaluation and fill in your results)* | | | | |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python warehouse/build_warehouse.py
python -m pytest -q tests                # 13 tests, no API key needed

export LLM_BASE_URL=https://api.groq.com/openai/v1   # or set the values from .env.example
export LLM_API_KEY=...
export LLM_MODEL=llama-3.3-70b-versatile
python src/assistant.py "What was the revenue per quarter in 2025?"
```

Example output:

```
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
src/assistant.py               prompt, SQL extraction, self-correction, CLI
eval/questions.json            20 questions with reference SQL
eval/run_eval.py               execution-accuracy evaluation
tests/                         13 tests with scripted fake LLMs
```

## Limits and next steps

- A semicolon inside a text value is refused by the one-statement check. That is a deliberate trade-off for simplicity.
- Next: a small web UI (Streamlit) with an automatic chart, few-shot examples chosen by similarity to the question, and a PostgreSQL version of the warehouse.
