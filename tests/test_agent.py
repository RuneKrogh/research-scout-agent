"""Agent tests with a scripted model (no LLM or network calls)."""

import json

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from research_scout.agent import ResearchDeps, build_agent, to_answer
from research_scout.models import Paper

CONSTRAINTS = {
    "topic": "LLM agents software engineering",
    "year": {"operator": "after", "year": 2022},
    "citations": {"operator": "at_least", "value": 100},
}
GOOD = Paper(openalex_id="W-good", title="Agents for SE", year=2023, citation_count=500)
TOO_OLD = Paper(openalex_id="W-old", title="Old agents", year=2021, citation_count=900)


class FakeClient:
    def __init__(self, papers: list[Paper]):
        self.papers = papers
        self.calls: list[dict] = []

    async def search_works(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return self.papers


def scripted(*decisions: str | None):
    """A model that searches once, then returns the given paper ids in turn."""
    remaining = list(decisions)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "search_papers"
        ]
        if not tool_returns:
            return ModelResponse(
                parts=[ToolCallPart("search_papers", {"constraints": CONSTRAINTS})]
            )
        decision = {
            "constraints": CONSTRAINTS,
            "selected_paper_id": remaining.pop(0),
            "explanation": "It matches.",
        }
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, decision)])

    return FunctionModel(respond)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_happy_path_uses_tool_metadata():
    client = FakeClient([GOOD])
    deps = ResearchDeps(client=client)
    result = await build_agent(scripted("W-good")).run("query", deps=deps)
    answer = to_answer(result.output, deps)

    assert client.calls[0]["year_after"] == 2022
    assert client.calls[0]["min_citations"] == 100
    assert answer.found
    assert answer.paper.citation_count == 500
    assert answer.paper.citation_source == "OpenAlex"
    assert answer.tool_calls == 1


@pytest.mark.anyio
async def test_tool_rejects_papers_that_violate_constraints():
    client = FakeClient([GOOD, TOO_OLD])
    deps = ResearchDeps(client=client)
    result = await build_agent(scripted("W-good")).run("query", deps=deps)

    tool_return = next(
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    )
    content = tool_return.content
    content = json.loads(content) if isinstance(content, str) else content
    assert [c["openalex_id"] for c in content["candidates"]] == ["W-good"]
    assert content["rejected_by_constraints"] == 1
    assert "W-old" not in deps.seen


@pytest.mark.anyio
async def test_invented_paper_id_is_retried():
    deps = ResearchDeps(client=FakeClient([GOOD]))
    result = await build_agent(scripted("W-made-up", "W-good")).run("query", deps=deps)
    assert result.output.selected_paper_id == "W-good"


@pytest.mark.anyio
async def test_no_match_returns_not_found():
    deps = ResearchDeps(client=FakeClient([]))
    result = await build_agent(scripted(None)).run("query", deps=deps)
    answer = to_answer(result.output, deps)
    assert not answer.found
    assert answer.paper is None
