"""Ask business questions in plain English or French; get SQL and the answer.

Flow:
    question → the most similar solved examples are retrieved from a vector database (RAG)
    → prompt with the warehouse schema and those examples → LLM writes SQL
    → guard checks and runs it read-only → on error, the error goes back to the
      LLM for one correction attempt → table of results

Usage:
    python src/assistant.py "What was the revenue by category in 2025?"
    python src/assistant.py "Quel magasin a la meilleure marge ?"
    python src/assistant.py --no-rag "..."      # without retrieved examples
"""
import argparse
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from guard import QueryResult, UnsafeQuery, connect_readonly, run_safe
from llm import chat

DB_PATH = Path(__file__).resolve().parents[1] / "warehouse" / "retail.db"

RULES = """You are a senior BI analyst. Write ONE SQLite SELECT query that answers the question.
Rules:
- Use only the tables and columns in the schema. Join facts to dimensions on their *_key columns.
- Revenue = SUM(fact_sales.revenue). Margin = SUM(revenue) - SUM(cost).
  Margin rate = (SUM(revenue) - SUM(cost)) / SUM(revenue).
- Filter dates with dim_date (year, quarter, month), not with string functions on keys.
- Round money to 2 decimals. Give every computed column a short snake_case alias.
- Add ORDER BY when the question asks for a ranking, and LIMIT for "top N".
- Answer with the SQL only, inside a ```sql code block."""


def schema_text(db_path: Path = DB_PATH) -> str:
    """The CREATE TABLE statements, including column comments, read from the database."""
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    rows = con.execute("SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    con.close()
    return "\n\n".join(r[0] for r in rows)


def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.S | re.I)
    return (m.group(1) if m else text).strip()


@dataclass
class Answer:
    question: str
    sql: str
    result: QueryResult | None
    attempts: int
    error: str | None = None
    examples: list = field(default_factory=list)   # examples retrieved for the prompt


class Assistant:
    def __init__(self, db_path: Path = DB_PATH, llm: Callable[[list[dict]], str] = chat,
                 max_attempts: int = 2, retriever=None, k: int = 3):
        """retriever: an ExampleRetriever (see retriever.py), or None for no examples."""
        self.con = connect_readonly(db_path)
        self.llm = llm
        self.max_attempts = max_attempts
        self.retriever = retriever
        self.k = k
        self.system = f"{RULES}\n\nSchema:\n{schema_text(db_path)}"

    def system_prompt(self, question: str) -> tuple[str, list]:
        if self.retriever is None:
            return self.system, []
        from retriever import format_examples
        examples = self.retriever.search(question, k=self.k)
        return f"{self.system}\n\n{format_examples(examples)}", examples

    def ask(self, question: str) -> Answer:
        system, examples = self.system_prompt(question)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": question}]
        sql, error = "", None
        for attempt in range(1, self.max_attempts + 1):
            reply = self.llm(messages)
            sql = extract_sql(reply)
            try:
                return Answer(question, sql, run_safe(self.con, sql), attempt, examples=examples)
            except UnsafeQuery as e:
                # Never ask the model to work around the guard: stop here.
                return Answer(question, sql, None, attempt, error=str(e), examples=examples)
            except sqlite3.Error as e:
                error = str(e)
                messages += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": f"The query failed with: {error}\n"
                              "Fix it and answer with the corrected SQL only."}]
        return Answer(question, sql, None, self.max_attempts, error=error, examples=examples)


def format_table(res: QueryResult, max_rows: int = 20) -> str:
    rows = [[("" if v is None else str(v)) for v in r] for r in res.rows[:max_rows]]
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
              for i, c in enumerate(res.columns)]
    line = " | ".join(c.ljust(w) for c, w in zip(res.columns, widths))
    out = [line, "-+-".join("-" * w for w in widths)]
    out += [" | ".join(v.ljust(w) for v, w in zip(r, widths)) for r in rows]
    if len(res.rows) > max_rows or res.truncated:
        out.append(f"... ({len(res.rows)}{'+' if res.truncated else ''} rows)")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--no-rag", action="store_true", help="do not add retrieved examples")
    ap.add_argument("-k", type=int, default=3, help="number of examples to retrieve")
    args = ap.parse_args()
    retriever = None
    if not args.no_rag:
        from retriever import ExampleRetriever
        retriever = ExampleRetriever()
    a = Assistant(retriever=retriever, k=args.k).ask(args.question)
    if a.examples:
        print("Retrieved examples:")
        for e in a.examples:
            print(f"  {e.score:.2f}  {e.question}")
        print()
    print(f"SQL ({a.attempts} attempt{'s' if a.attempts > 1 else ''}):\n{a.sql}\n")
    print(format_table(a.result) if a.result else f"Error: {a.error}")
