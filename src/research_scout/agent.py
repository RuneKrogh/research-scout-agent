"""The research agent: the LLM interprets and judges, code searches and verifies."""

import os
from dataclasses import dataclass, field
from datetime import date

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from research_scout.models import (
    AgentDecision,
    Paper,
    PaperResult,
    ResearchAnswer,
    SearchConstraints,
)
from research_scout.openalex import OpenAlexClient, OpenAlexError, SortOrder

# Flash-Lite: the free-tier Flash models were often overloaded during development.
DEFAULT_MODEL = "google:gemini-3.5-flash-lite"
MAX_SEARCHES = 4
CANDIDATES_PER_SEARCH = 10
ABSTRACT_PREVIEW_CHARS = 600


@dataclass
class ResearchDeps:
    """State shared between the tool and the output validator during one run."""

    client: OpenAlexClient
    today: date = field(default_factory=date.today)
    seen: dict[str, Paper] = field(default_factory=dict)
    searches: int = 0


INSTRUCTIONS = """\
You are a research scout. You find ONE research paper that satisfies the user's request,
using only papers returned by the `search_papers` tool.

Workflow:
1. Extract the constraints: a short topic query, an optional year constraint and an optional
   citation constraint. Record how you interpreted vague wording in `assumptions`.
2. Call `search_papers` with those constraints. The tool only returns papers that already
   satisfy the year and citation constraints.
3. If no candidate is relevant, you may search again with a broader or rephrased topic
   (synonyms, fewer words). Never loosen the user's year or citation constraints.
4. Pick the single most relevant candidate, judged from its title and abstract. Among
   similarly relevant candidates, prefer established work (more citations, a real venue)
   over obscure preprints. If the user gave no citation constraint, consider also
   searching with sort="citations" to find influential papers.
5. Return its `openalex_id` as `selected_paper_id` and explain why it matches. Follow any
   format the user asked for (e.g. a 5-7 sentence summary, or whether it is useful for X).
6. If nothing relevant was found, return `selected_paper_id: null` and explain why.

Rules:
- `constraints` in your final answer must describe the USER's request (their topic, year
  and citation constraints), not the query of your last search.
- Never state a title, author, year or citation count yourself; the system adds them from
  the tool results. Base your explanation only on the title and abstract you were given.
- Use `uncertainty` for anything you are unsure about, e.g. a weak topic match, a missing
  abstract, or an ambiguous request.
- Today's date is {today}. Interpret "recent" as published in the last three years
  (operator "from", year {recent}) unless the user says otherwise, and note that assumption.
"""


def _candidate_view(paper: Paper) -> dict:
    """What the LLM sees about a candidate."""
    abstract = paper.abstract or "(no abstract available)"
    if len(abstract) > ABSTRACT_PREVIEW_CHARS:
        abstract = abstract[:ABSTRACT_PREVIEW_CHARS].rsplit(" ", 1)[0] + " ..."
    return {
        "openalex_id": paper.openalex_id,
        "title": paper.title,
        "year": paper.year,
        "citation_count": paper.citation_count,
        "abstract": abstract,
    }


def build_agent(model: Model | str | None = None) -> Agent[ResearchDeps, AgentDecision]:
    agent = Agent(
        model or os.getenv("RESEARCH_SCOUT_MODEL") or DEFAULT_MODEL,
        deps_type=ResearchDeps,
        output_type=AgentDecision,
        retries=2,
    )

    @agent.instructions
    def instructions(ctx: RunContext[ResearchDeps]) -> str:
        today = ctx.deps.today
        return INSTRUCTIONS.format(today=today.isoformat(), recent=today.year - 3)

    @agent.tool
    async def search_papers(
        ctx: RunContext[ResearchDeps],
        constraints: SearchConstraints,
        sort: SortOrder = "relevance",
    ) -> dict:
        """Search OpenAlex for papers on a topic that satisfy the year and citation constraints.

        Args:
            constraints: Topic plus optional year and citation constraints.
            sort: "relevance" (default) or "citations" (most cited first).
        """
        if ctx.deps.searches >= MAX_SEARCHES:
            return {"error": f"Search limit of {MAX_SEARCHES} reached. Decide now."}
        ctx.deps.searches += 1
        try:
            papers = await ctx.deps.client.search_works(
                constraints.topic,
                sort=sort,
                limit=CANDIDATES_PER_SEARCH,
                **constraints.search_filters(),
            )
        except OpenAlexError as exc:
            return {"error": str(exc)}

        # Double-check the API's filtering in code; reject anything that slipped through.
        accepted = [p for p in papers if not constraints.violations(p)]
        for paper in accepted:
            ctx.deps.seen[paper.openalex_id] = paper
        return {
            "candidates": [_candidate_view(p) for p in accepted],
            "rejected_by_constraints": len(papers) - len(accepted),
        }

    @agent.output_validator
    def validate(ctx: RunContext[ResearchDeps], decision: AgentDecision) -> AgentDecision:
        if ctx.deps.searches == 0:
            raise ModelRetry("Call search_papers before answering.")
        if decision.selected_paper_id is None:
            return decision
        paper = ctx.deps.seen.get(decision.selected_paper_id)
        if paper is None:
            raise ModelRetry(
                f"{decision.selected_paper_id} was not returned by search_papers. "
                "Select an openalex_id from the search results, or null."
            )
        problems = decision.constraints.violations(paper)
        if problems:
            raise ModelRetry(
                f"The selected paper violates the constraints: {'; '.join(problems)}. "
                "Select another paper or null."
            )
        return decision

    return agent


def to_answer(decision: AgentDecision, deps: ResearchDeps) -> ResearchAnswer:
    """Build the final answer; all paper metadata comes from the tool results."""
    paper = deps.seen.get(decision.selected_paper_id) if decision.selected_paper_id else None
    return ResearchAnswer(
        paper=PaperResult.from_paper(paper) if paper else None,
        explanation=decision.explanation,
        uncertainty=decision.uncertainty,
        constraints=decision.constraints,
        tool_calls=deps.searches,
    )


async def research(query: str) -> ResearchAnswer:
    """Run the agent on one user request."""
    client = OpenAlexClient()
    deps = ResearchDeps(client=client)
    try:
        result = await build_agent().run(query, deps=deps)
    finally:
        await client.aclose()
    return to_answer(result.output, deps)
