"""Run with:  python -m pytest -q tests   (after python warehouse/build_warehouse.py)
No LLM or API key needed: the LLM is replaced by scripted fakes."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from assistant import Assistant, extract_sql  # noqa: E402
from guard import UnsafeQuery, connect_readonly, run_safe  # noqa: E402

DB = ROOT / "warehouse" / "retail.db"


@pytest.fixture(scope="module")
def con():
    return connect_readonly(DB)


@pytest.mark.parametrize("sql", [
    "DROP TABLE fact_sales",
    "DELETE FROM fact_sales",
    "SELECT 1; DELETE FROM fact_sales",
    "WITH a AS (SELECT 1) DELETE FROM fact_sales",
    "SELECT name FROM sqlite_master",
    "PRAGMA table_info(fact_sales)",
    "ATTACH DATABASE 'other.db' AS x",
])
def test_guard_blocks_dangerous_sql(con, sql):
    with pytest.raises(UnsafeQuery):
        run_safe(con, sql)


def test_guard_allows_ctes_and_window_functions(con):
    res = run_safe(con, "WITH r AS (SELECT customer_key, SUM(revenue) rev FROM fact_sales "
                        "GROUP BY 1) SELECT COUNT(*), MAX(rk) FROM (SELECT RANK() OVER "
                        "(ORDER BY rev DESC) rk FROM r)")
    assert res.rows[0][0] == 1200


def test_guard_stops_slow_queries(con):
    with pytest.raises(UnsafeQuery, match="longer than"):
        run_safe(con, "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) "
                      "SELECT COUNT(*) FROM n", timeout_s=0.5)


def test_row_limit(con):
    res = run_safe(con, "SELECT * FROM fact_sales", max_rows=50)
    assert len(res.rows) == 50 and res.truncated


def test_extract_sql_from_markdown():
    assert extract_sql("Here:\n```sql\nSELECT 1\n```") == "SELECT 1"


def test_self_correction_after_sql_error():
    replies = iter(["```sql\nSELECT SUM(amount) FROM fact_sales\n```",      # wrong column
                    "```sql\nSELECT ROUND(SUM(revenue), 2) FROM fact_sales\n```"])
    seen = []

    def fake_llm(messages):
        seen.append(messages[-1]["content"])
        return next(replies)

    answer = Assistant(DB, llm=fake_llm).ask("Total revenue?")
    assert answer.attempts == 2 and answer.result is not None
    assert "no such column" in seen[1]          # the error was sent back to the model


def test_unsafe_sql_is_not_retried():
    calls = []

    def fake_llm(messages):
        calls.append(1)
        return "DELETE FROM fact_sales"

    answer = Assistant(DB, llm=fake_llm).ask("Delete everything")
    assert answer.result is None and len(calls) == 1 and "SELECT" in answer.error
