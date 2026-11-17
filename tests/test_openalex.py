import httpx
import pytest
import respx

from research_scout.models import Paper
from research_scout.openalex import (
    BASE_URL,
    OpenAlexClient,
    OpenAlexError,
    build_filter,
    deduplicate,
    parse_work,
    reconstruct_abstract,
)

WORK = {
    "id": "https://openalex.org/W1",
    "doi": "https://doi.org/10.1/abc",
    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "publication_year": 2020,
    "cited_by_count": 19234,
    "authorships": [
        {"author": {"display_name": "Patrick Lewis"}},
        {"author": {"display_name": "Ethan Perez"}},
    ],
    "primary_location": {"landing_page_url": "http://arxiv.org/abs/2005.11401"},
    "abstract_inverted_index": {"Large": [0], "models": [2], "language": [1]},
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_build_filter_year_and_citations():
    assert build_filter(year_after=2022, min_citations=100) == (
        "publication_year:>2022,cited_by_count:>99"
    )
    assert build_filter(year_before=2021, max_citations=500) == (
        "publication_year:<2021,cited_by_count:<501"
    )
    assert build_filter(year=2023) == "publication_year:2023"
    assert build_filter() == ""


def test_reconstruct_abstract():
    assert reconstruct_abstract(WORK["abstract_inverted_index"]) == "Large language models"
    assert reconstruct_abstract(None) is None


def test_parse_work():
    paper = parse_work(WORK)
    assert paper.title.startswith("Retrieval-Augmented")
    assert paper.authors == ["Patrick Lewis", "Ethan Perez"]
    assert paper.year == 2020
    assert paper.citation_count == 19234
    assert paper.link == "https://doi.org/10.1/abc"


def test_parse_work_handles_missing_fields():
    paper = parse_work({"id": "https://openalex.org/W2"})
    assert paper.title == "(untitled)"
    assert paper.authors == []
    assert paper.citation_count is None
    assert paper.link == "https://openalex.org/W2"


def test_deduplicate_keeps_most_cited_version():
    a = Paper(openalex_id="W1", title="Same Paper", citation_count=10)
    b = Paper(openalex_id="W2", title="Other", citation_count=5)
    c = Paper(openalex_id="W3", title="same paper!", citation_count=99)
    assert [p.openalex_id for p in deduplicate([a, b, c])] == ["W3", "W2"]


@pytest.mark.anyio
@respx.mock
async def test_search_works_sends_filters_and_parses():
    route = respx.get(f"{BASE_URL}/works").mock(
        return_value=httpx.Response(200, json={"results": [WORK]})
    )
    client = OpenAlexClient(api_key="key")
    papers = await client.search_works(
        "retrieval augmented generation", year_before=2021, min_citations=500, sort="citations"
    )
    await client.aclose()

    params = route.calls.last.request.url.params
    assert params["search.title_and_abstract"] == "retrieval augmented generation"
    assert params["filter"] == "publication_year:<2021,cited_by_count:>499"
    assert params["sort"] == "cited_by_count:desc"
    assert params["api_key"] == "key"
    assert papers[0].citation_count == 19234


@pytest.mark.anyio
@respx.mock
async def test_search_works_raises_on_client_error():
    respx.get(f"{BASE_URL}/works").mock(return_value=httpx.Response(400, text="bad filter"))
    client = OpenAlexClient(api_key="")
    with pytest.raises(OpenAlexError, match="400"):
        await client.search_works("agents")
    await client.aclose()


@pytest.mark.anyio
@respx.mock
async def test_search_works_retries_on_rate_limit(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr("research_scout.openalex.asyncio.sleep", no_sleep)
    respx.get(f"{BASE_URL}/works").mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json={"results": []})]
    )
    client = OpenAlexClient(api_key="")
    assert await client.search_works("agents") == []
    await client.aclose()


@pytest.mark.anyio
async def test_search_works_rejects_empty_query():
    client = OpenAlexClient(api_key="")
    with pytest.raises(ValueError):
        await client.search_works("  ")
    await client.aclose()
