"""Measure execution accuracy: does the generated SQL return the same result as the
reference SQL? Column names and (unless the question needs it) row order are ignored.

Usage:
    python eval/run_eval.py                 # uses the LLM set in LLM_* env vars
    python eval/run_eval.py --oracle        # sanity check: feeds the reference SQL back

Writes eval/results.json with one line per question.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from assistant import Assistant  # noqa: E402
from guard import connect_readonly, run_safe  # noqa: E402


def normalise(rows, ordered: bool, first_column_only: bool):
    def cell(v):
        return round(v, 2) if isinstance(v, float) else v
    out = [tuple(cell(v) for v in (r[:1] if first_column_only else r)) for r in rows]
    return out if ordered else sorted(out, key=repr)


def same(gold_rows, pred_rows, q) -> bool:
    ordered, first = q.get("ordered", False), q.get("first_column_only", False)
    g = normalise(gold_rows, ordered, first)
    p = normalise(pred_rows, ordered, first)
    if g == p:
        return True
    if len(g) != len(p) or not g or len(p[0]) <= len(g[0]):
        return False
    # The prediction has extra columns: accept it if each reference row is contained
    # in a different predicted row (in the same position when order matters).
    if ordered:
        return all(set(gr) <= set(pr) for gr, pr in zip(g, p))
    remaining = list(p)
    for gr in g:
        match = next((pr for pr in remaining if set(gr) <= set(pr)), None)
        if match is None:
            return False
        remaining.remove(match)
    return True


def main(oracle: bool):
    questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    gold_con = connect_readonly(ROOT / "warehouse" / "retail.db")
    by_text = {q["question"]: q["sql"] for q in questions}

    def oracle_llm(messages):
        return f"```sql\n{by_text[messages[1]['content']]}\n```"

    assistant = Assistant(llm=oracle_llm) if oracle else Assistant()
    results, score = [], Counter()
    for q in questions:
        gold = run_safe(gold_con, q["sql"]).rows
        a = assistant.ask(q["question"])
        ok = a.result is not None and same(gold, a.result.rows, q)
        score[q["level"]] += ok
        score["total"] += ok
        results.append({"id": q["id"], "level": q["level"], "correct": ok,
                        "attempts": a.attempts, "sql": a.sql, "error": a.error})
        print(f"[{'OK ' if ok else 'BAD'}] {q['id']:2d} {q['level']:6s} {q['question']}")

    levels = Counter(q["level"] for q in questions)
    summary = {lvl: f"{score[lvl]}/{n}" for lvl, n in levels.items()}
    summary["total"] = f"{score['total']}/{len(questions)}"
    (ROOT / "eval" / "results.json").write_text(json.dumps(
        {"summary": summary, "results": results}, indent=2, ensure_ascii=False))
    print("\nExecution accuracy:", summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true")
    main(ap.parse_args().oracle)
