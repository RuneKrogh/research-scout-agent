"""Command-line entry point: research-scout "<request>"."""

import argparse
import asyncio
import os
import textwrap

from dotenv import load_dotenv

from research_scout.models import ResearchAnswer


def format_answer(answer: ResearchAnswer) -> str:
    lines: list[str] = []
    if answer.paper:
        p = answer.paper
        authors = ", ".join(p.authors[:5]) + (" et al." if len(p.authors) > 5 else "")
        lines += [
            f"Title:     {p.title}",
            f"Authors:   {authors or 'unknown'}",
            f"Year:      {p.year or 'unknown'}",
            f"Citations: {p.citation_count if p.citation_count is not None else 'unknown'} "
            f"(source: {p.citation_source}, {p.citation_source_url})",
            f"Link:      {p.link}",
            "",
        ]
    else:
        lines += ["No paper found that satisfies the request.", ""]
    lines.append(textwrap.fill(answer.explanation, width=88))
    if answer.uncertainty:
        lines += ["", textwrap.fill(f"Uncertainty: {answer.uncertainty}", width=88)]
    c = answer.constraints
    parts = [f"topic='{c.topic}'"]
    if c.year:
        parts.append(f"year {c.year.operator} {c.year.year}")
    if c.citations:
        parts.append(f"citations {c.citations.operator.replace('_', ' ')} {c.citations.value}")
    lines += ["", f"Interpreted as: {', '.join(parts)}"]
    if c.assumptions:
        lines.append(f"Assumptions: {'; '.join(c.assumptions)}")
    lines.append(f"Search tool calls: {answer.tool_calls}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find a research paper matching a request.")
    parser.add_argument("request", help="e.g. 'a RAG paper before 2021 with >500 citations'")
    parser.add_argument("--json", action="store_true", help="print the structured answer as JSON")
    args = parser.parse_args()

    load_dotenv()
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    from research_scout.agent import research

    answer = asyncio.run(research(args.request))
    print(answer.model_dump_json(indent=2) if args.json else format_answer(answer))


if __name__ == "__main__":
    main()
