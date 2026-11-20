"""Run the evaluation prompts and check the answers.

uv run python eval/run_eval.py           # run all prompts, write eval/results.json
uv run python eval/run_eval.py --report  # print the results table (incl. manual scores)
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pydantic_ai.exceptions import ModelHTTPError

from research_scout.agent import research
from research_scout.models import CitationConstraint, Paper, SearchConstraints, YearConstraint

EVAL_DIR = Path(__file__).parent
RESULTS = EVAL_DIR / "results.json"
MANUAL = EVAL_DIR / "manual_scores.json"
CITATION_TOLERANCE = 0.05  # OpenAlex counts change over time


async def verify_source(answer: dict) -> bool:
    """Re-fetch the OpenAlex record and compare it with the answer's metadata."""
    paper = answer["paper"]
    work_id = paper["citation_source_url"].rsplit("/", 1)[-1]
    params = {"api_key": os.getenv("OPENALEX_API_KEY")} if os.getenv("OPENALEX_API_KEY") else {}
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(f"https://api.openalex.org/works/{work_id}", params=params)
    if response.is_error:
        return False
    work = response.json()
    live = work.get("cited_by_count") or 0
    return (
        work.get("title") == paper["title"]
        and work.get("publication_year") == paper["year"]
        and abs(live - (paper["citation_count"] or 0)) <= CITATION_TOLERANCE * max(live, 1)
    )


def check(case: dict, answer: dict) -> dict:
    """Automatic checks against the constraints written in prompts.json."""
    expected = SearchConstraints(
        topic=case["prompt"],
        year=YearConstraint(**case["year"]) if case["year"] else None,
        citations=CitationConstraint(**case["citations"]) if case["citations"] else None,
    )
    checks = {"outcome": (answer["paper"] is not None) == case["expect_found"]}
    if case["year"] or case["citations"]:
        parsed = answer["constraints"]
        checks["parsed"] = (parsed["year"] == case["year"]) and (
            parsed["citations"] == case["citations"]
        )
    if answer["paper"]:
        p = answer["paper"]
        paper = Paper(
            openalex_id=p["citation_source_url"],
            title=p["title"],
            year=p["year"],
            citation_count=p["citation_count"],
        )
        problems = expected.violations(paper)
        if case["year"]:
            checks["year"] = not any("year" in problem for problem in problems)
        if case["citations"]:
            checks["citations"] = not any("citation" in problem for problem in problems)
    return checks


async def research_with_retry(prompt: str, attempts: int = 3) -> dict:
    """The free-tier model sometimes returns 429 (rate limit) or 503 (overloaded); retry."""
    for attempt in range(attempts):
        try:
            return (await research(prompt)).model_dump()
        except ModelHTTPError as exc:
            if exc.status_code not in (429, 503) or attempt == attempts - 1:
                raise
            print(f"    model returned {exc.status_code}, retrying in 60 s")
            await asyncio.sleep(60)


async def run() -> None:
    cases = json.loads((EVAL_DIR / "prompts.json").read_text(encoding="utf-8"))
    results = []
    for case in cases:
        print(f"[{case['id']}] {case['prompt']}")
        try:
            answer = await research_with_retry(case["prompt"])
        except Exception as exc:  # record failures instead of stopping the run
            results.append({"id": case["id"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"    error: {exc}")
            continue
        checks = check(case, answer)
        if answer["paper"]:
            checks["source"] = await verify_source(answer)
        results.append({"id": case["id"], "answer": answer, "checks": checks})
        print(f"    {checks}")
    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def report() -> None:
    cases = {c["id"]: c for c in json.loads((EVAL_DIR / "prompts.json").read_text("utf-8"))}
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    manual = json.loads(MANUAL.read_text(encoding="utf-8")) if MANUAL.exists() else {}
    columns = ["outcome", "parsed", "year", "citations", "source", "relevant", "explanation"]
    mark = {True: "✓", False: "✗", None: "–"}

    print("| # | Category | Paper | " + " | ".join(c.capitalize() for c in columns) + " | Tools |")
    print("|---" * (len(columns) + 4) + "|")
    totals = {c: [0, 0] for c in columns}
    for r in results:
        if "error" in r:
            print(f"| {r['id']} | {cases[r['id']]['category']} | error | " + "| " * len(columns))
            continue
        scores = {**r["checks"], **manual.get(str(r["id"]), {})}
        p = r["answer"]["paper"]
        paper = f"{p['title'][:45]} ({p['year']}, {p['citation_count']})" if p else "none"
        cells = []
        for c in columns:
            value = scores.get(c)
            if value is not None:
                totals[c][0] += value
                totals[c][1] += 1
            cells.append(mark[value])
        print(
            f"| {r['id']} | {cases[r['id']]['category']} | {paper} | "
            + " | ".join(cells)
            + f" | {r['answer']['tool_calls']} |"
        )
    print("\n" + ", ".join(f"{c}: {ok}/{n}" for c, (ok, n) in totals.items() if n))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="print the results table only")
    args = parser.parse_args()
    load_dotenv()
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console can't print ✓/✗ otherwise
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    report() if args.report else asyncio.run(run())
