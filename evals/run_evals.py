"""Evaluate BankMate against a golden dataset.

Two levels:
  1. Retrieval  - for every policy question with an expected citation, is the
                  right section in the top 1 / top 3 results? (hit@1, hit@3, MRR)
  2. End-to-end - run the full agent and check, per case:
                  * tool selection   (did it call the expected tools?)
                  * citation         (did the answer cite an expected source?)
                  * correctness      (required facts present, forbidden text absent)
                  * refusals         (out-of-scope -> "don't have", attacks blocked)
                  * guardrail flags  (PII / credentials masked)
     Optional --judge uses Gemini as an LLM judge to score groundedness 1-5.

Usage:
    python -m evals.run_evals              # offline if no GOOGLE_API_KEY
    python -m evals.run_evals --judge      # adds LLM-as-judge (needs key)
Results are written to evals/results/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import db  # noqa: E402
from app.config import settings  # noqa: E402
from app.rag.retriever import get_retriever  # noqa: E402
from app.service import BankMateService  # noqa: E402

GOLDEN = ROOT / "evals" / "golden.jsonl"
RESULTS = ROOT / "evals" / "results"
NOT_FOUND_MARKERS = ["don't have", "do not have", "not able to find", "couldn't find",
                     "no information", "not covered", "unable to find", "call nusa bank",
                     "1-300-88-6872"]

JUDGE_PROMPT = """You are grading a bank assistant's answer for groundedness.
Score 1-5: 5 = every factual claim is supported by the SOURCES; 1 = mostly unsupported or wrong.
Answer with JSON only: {{"score": <1-5>, "reason": "<one sentence>"}}

QUESTION: {question}
SOURCES:
{sources}
ANSWER:
{answer}"""


def load_cases() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def eval_retrieval(cases: list[dict]) -> dict:
    retriever = get_retriever()
    rows, ranks = [], []
    for c in cases:
        if not c.get("expected_citations") or not c["id"].startswith("pol"):
            continue
        hits = [h.chunk.citation for h in retriever.search(c["question"], k=5)]
        rank = next((i + 1 for i, h in enumerate(hits) if h in c["expected_citations"]), None)
        ranks.append(rank)
        rows.append({"id": c["id"], "rank": rank, "top3": hits[:3]})
    n = len(ranks) or 1
    return {
        "mode": retriever.mode,
        "cases": len(ranks),
        "hit@1": round(sum(1 for r in ranks if r == 1) / n, 3),
        "hit@3": round(sum(1 for r in ranks if r and r <= 3) / n, 3),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
        "details": rows,
    }


def _judge(question: str, answer: str, citations: list[dict]) -> dict | None:
    from google import genai

    sources = "\n".join(f"[{c['citation']}] {c['text']}" for c in citations) or "(none)"
    client = genai.Client()
    resp = client.models.generate_content(
        model=settings.model,
        contents=JUDGE_PROMPT.format(question=question, sources=sources, answer=answer),
        config={"response_mime_type": "application/json", "temperature": 0},
    )
    try:
        return json.loads(resp.text)
    except (json.JSONDecodeError, TypeError):
        return None


def grade(case: dict, result: dict) -> dict:
    answer = result["answer"]
    low = answer.lower()
    checks: dict[str, bool] = {}
    tools = {t["tool"] for t in result["tools_used"]}

    if case.get("should_block"):
        checks["blocked"] = result["blocked"]
        return checks

    if case.get("expected_tools"):
        checks["tools"] = set(case["expected_tools"]).issubset(tools)
    if case.get("expected_citations"):
        cited = {c["citation"] for c in result["citations"]}
        checks["citation"] = bool(cited & set(case["expected_citations"]))
    if case.get("must_contain"):
        checks["facts"] = all(s.lower() in low for s in case["must_contain"])
    if case.get("must_contain_any"):
        checks["facts"] = any(s.lower() in low for s in case["must_contain_any"])
    if case.get("must_not_contain"):
        checks["no_leak"] = not any(s.lower() in low for s in case["must_not_contain"])
    if case.get("should_not_find"):
        checks["admits_unknown"] = any(m in low for m in NOT_FOUND_MARKERS)
    if case.get("expected_flags"):
        checks["flags"] = set(case["expected_flags"]).issubset(result["guardrail_flags"])
    checks["no_unverified_citation"] = "unverified_citation" not in result["guardrail_flags"]
    return checks


async def eval_end_to_end(cases: list[dict], judge: bool) -> dict:
    svc = BankMateService()
    rows = []
    for case in cases:
        result = (await svc.chat(case["customer_id"], case["question"])).to_dict()
        checks = grade(case, result)
        row = {"id": case["id"], "question": case["question"], "passed": all(checks.values()),
               "checks": checks, "tools": [t["tool"] for t in result["tools_used"]],
               "citations": [c["citation"] for c in result["citations"]],
               "latency_ms": result["latency_ms"], "answer": result["answer"]}
        if judge and result["citations"]:
            row["judge"] = _judge(case["question"], result["answer"], result["citations"])
        rows.append(row)
        status = "PASS" if row["passed"] else "FAIL"
        failed = [k for k, v in checks.items() if not v]
        print(f"  {status}  {case['id']:<9} {result['latency_ms']:>6} ms  {'' if not failed else failed}")

    def rate(key: str) -> float | None:
        vals = [r["checks"][key] for r in rows if key in r["checks"]]
        return round(sum(vals) / len(vals), 3) if vals else None

    judged = [r["judge"]["score"] for r in rows if r.get("judge")]
    latencies = [r["latency_ms"] for r in rows]
    return {
        "model": svc.model_name,
        "cases": len(rows),
        "pass_rate": round(sum(r["passed"] for r in rows) / len(rows), 3),
        "tool_selection_accuracy": rate("tools"),
        "citation_accuracy": rate("citation"),
        "fact_accuracy": rate("facts"),
        "refusal_accuracy": rate("admits_unknown"),
        "attack_block_rate": rate("blocked"),
        "pii_protection": rate("no_leak"),
        "judge_groundedness_avg": round(statistics.fmean(judged), 2) if judged else None,
        "latency_ms_p50": int(statistics.median(latencies)),
        "latency_ms_max": max(latencies),
        "details": rows,
    }


def write_report(retrieval: dict, e2e: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS / f"results-{stamp}.json").write_text(json.dumps(
        {"retrieval": retrieval, "end_to_end": e2e}, indent=2))

    fmt = lambda v: "n/a" if v is None else (f"{v:.0%}" if isinstance(v, float) and v <= 1 else str(v))  # noqa: E731
    lines = [
        f"# BankMate eval report ({stamp})", "",
        f"Model: `{e2e['model']}` · Retrieval: {retrieval['mode']}", "",
        "## Retrieval", "",
        "| Metric | Value |", "|---|---|",
        f"| Policy questions | {retrieval['cases']} |",
        f"| Hit@1 | {fmt(retrieval['hit@1'])} |",
        f"| Hit@3 | {fmt(retrieval['hit@3'])} |",
        f"| MRR | {retrieval['mrr']} |", "",
        "## End to end", "",
        "| Metric | Value |", "|---|---|",
    ]
    for key in ["pass_rate", "tool_selection_accuracy", "citation_accuracy", "fact_accuracy",
                "refusal_accuracy", "attack_block_rate", "pii_protection",
                "judge_groundedness_avg", "latency_ms_p50", "latency_ms_max"]:
        lines.append(f"| {key.replace('_', ' ')} | {fmt(e2e[key])} |")
    lines += ["", "## Cases", "", "| Case | Result | Tools | Failed checks |", "|---|---|---|---|"]
    for r in e2e["details"]:
        failed = ", ".join(k for k, v in r["checks"].items() if not v) or "-"
        lines.append(f"| {r['id']} | {'✅' if r['passed'] else '❌'} | {', '.join(r['tools']) or '-'} | {failed} |")
    report = RESULTS / "latest.md"
    report.write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true", help="LLM-as-judge groundedness (needs API key)")
    args = parser.parse_args()
    if args.judge and settings.offline_mode:
        parser.error("--judge needs GOOGLE_API_KEY")

    db.seed()
    cases = load_cases()
    started = time.perf_counter()
    print(f"Retrieval eval ({get_retriever().mode})")
    retrieval = eval_retrieval(cases)
    print(f"  hit@1={retrieval['hit@1']:.0%}  hit@3={retrieval['hit@3']:.0%}  mrr={retrieval['mrr']}")
    print(f"End-to-end eval ({len(cases)} cases)")
    e2e = asyncio.run(eval_end_to_end(cases, args.judge))
    report = write_report(retrieval, e2e)
    print(f"\nPass rate {e2e['pass_rate']:.0%} in {time.perf_counter() - started:.1f}s -> {report}")


if __name__ == "__main__":
    main()
