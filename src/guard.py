"""Safe execution of LLM-generated SQL.

Three layers, so one mistake is not enough to do damage:
1. Static checks: one statement only, must be a SELECT (or WITH ... SELECT).
2. The database is opened read-only (SQLite URI mode=ro).
3. An SQLite authorizer allows only reads of whitelisted tables and blocks
   everything else (writes, PRAGMA, ATTACH, system tables).
A time limit and a row limit protect against runaway queries.
"""
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_TABLES = {"dim_date", "dim_product", "dim_store", "dim_customer", "fact_sales"}
MAX_ROWS = 200
TIMEOUT_S = 5.0


class UnsafeQuery(Exception):
    """The query was refused before or during execution."""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool = False
    elapsed_ms: float = 0.0
    sql: str = field(default="", repr=False)


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return re.sub(r"--[^\n]*", " ", sql)


def check_static(sql: str) -> str:
    """Return the cleaned SQL or raise UnsafeQuery."""
    cleaned = _strip_comments(sql).strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeQuery("Empty query.")
    if ";" in cleaned:
        raise UnsafeQuery("Only one SQL statement is allowed.")
    if not re.match(r"(?is)^\s*(select|with)\b", cleaned):
        raise UnsafeQuery("Only SELECT queries are allowed.")
    return cleaned


def make_authorizer(real_tables: set[str]):
    """Allow reads of whitelisted tables and of CTE names; deny everything else."""
    def authorizer(action, arg1, arg2, db_name, trigger):
        if action == sqlite3.SQLITE_READ:
            name = (arg1 or "").lower()
            if name in ALLOWED_TABLES:
                return sqlite3.SQLITE_OK
            # A name that is not a real table is a CTE defined inside the query.
            if name and name not in real_tables and not name.startswith("sqlite_"):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    return authorizer


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True,
                          check_same_thread=False)
    real = {r[0].lower() for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}
    con.set_authorizer(make_authorizer(real))
    return con


def run_safe(con: sqlite3.Connection, sql: str, max_rows: int = MAX_ROWS,
             timeout_s: float = TIMEOUT_S) -> QueryResult:
    cleaned = check_static(sql)
    start = time.perf_counter()
    con.set_progress_handler(lambda: int(time.perf_counter() - start > timeout_s), 10_000)
    try:
        cur = con.execute(cleaned)
        rows = cur.fetchmany(max_rows + 1)
    except sqlite3.DatabaseError as e:
        msg = str(e)
        if "not authorized" in msg or "prohibited" in msg:
            raise UnsafeQuery(f"Refused by the database guard: {msg}") from e
        if "interrupted" in msg:
            raise UnsafeQuery(f"Query took longer than {timeout_s:.0f} s and was stopped.") from e
        raise
    finally:
        con.set_progress_handler(None, 0)
    cols = [d[0] for d in cur.description] if cur.description else []
    return QueryResult(columns=cols, rows=rows[:max_rows], truncated=len(rows) > max_rows,
                       elapsed_ms=round((time.perf_counter() - start) * 1000, 1), sql=cleaned)
