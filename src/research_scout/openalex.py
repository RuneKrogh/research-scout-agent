"""Client for the OpenAlex works API (https://help.openalex.org)."""

import asyncio
import os
import re
from typing import Literal

import httpx

from research_scout.models import Paper

BASE_URL = "https://api.openalex.org"
SELECT_FIELDS = (
    "id,doi,title,publication_year,cited_by_count,authorships,primary_location,"
    "abstract_inverted_index"
)
MAX_RETRIES = 2

SortOrder = Literal["relevance", "citations"]


class OpenAlexError(RuntimeError):
    """Raised when OpenAlex cannot be reached or returns an error."""


def build_filter(
    *,
    min_year: int | None = None,
    max_year: int | None = None,
    min_citations: int | None = None,
    max_citations: int | None = None,
) -> str:
    """Build an OpenAlex filter string from inclusive bounds."""
    parts = []
    if min_year is not None:
        parts.append(f"publication_year:>{min_year - 1}")
    if max_year is not None:
        parts.append(f"publication_year:<{max_year + 1}")
    if min_citations is not None:
        parts.append(f"cited_by_count:>{min_citations - 1}")
    if max_citations is not None:
        parts.append(f"cited_by_count:<{max_citations + 1}")
    return ",".join(parts)


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """Rebuild plain text from OpenAlex's inverted abstract index."""
    if not inverted_index:
        return None
    positions = {pos: word for word, idxs in inverted_index.items() for pos in idxs}
    return " ".join(positions[i] for i in sorted(positions))


def parse_work(work: dict) -> Paper:
    """Convert one OpenAlex work object into a Paper."""
    location = work.get("primary_location") or {}
    return Paper(
        openalex_id=work["id"],
        title=work.get("title") or "(untitled)",
        authors=[
            a["author"]["display_name"]
            for a in work.get("authorships") or []
            if a.get("author", {}).get("display_name")
        ],
        year=work.get("publication_year"),
        citation_count=work.get("cited_by_count"),
        doi=work.get("doi"),
        url=location.get("landing_page_url"),
        abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
    )


def deduplicate(papers: list[Paper]) -> list[Paper]:
    """Merge versions of the same paper (e.g. preprint and proceedings), keeping the most cited."""
    best: dict[str, Paper] = {}
    for paper in papers:
        key = re.sub(r"[^a-z0-9]", "", paper.title.lower())
        if key not in best or (paper.citation_count or 0) > (best[key].citation_count or 0):
            best[key] = paper
    return list(best.values())


class OpenAlexClient:
    """Async client for searching OpenAlex works."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("OPENALEX_API_KEY")
        self._http = httpx.AsyncClient(base_url=BASE_URL, timeout=20)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def search_works(
        self,
        query: str,
        *,
        min_year: int | None = None,
        max_year: int | None = None,
        min_citations: int | None = None,
        max_citations: int | None = None,
        sort: SortOrder = "relevance",
        limit: int = 10,
    ) -> list[Paper]:
        """Search works by topic in title and abstract, with optional inclusive bounds."""
        params: dict[str, str | int] = {
            "search.title_and_abstract": query,
            "select": SELECT_FIELDS,
            "per_page": limit,
        }
        filters = build_filter(
            min_year=min_year,
            max_year=max_year,
            min_citations=min_citations,
            max_citations=max_citations,
        )
        if filters:
            params["filter"] = filters
        if sort == "citations":
            params["sort"] = "cited_by_count:desc"
        if self.api_key:
            params["api_key"] = self.api_key

        data = await self._get("/works", params)
        return deduplicate([parse_work(w) for w in data.get("results", [])])

    async def _get(self, path: str, params: dict) -> dict:
        """GET with retries on network errors, rate limits (429) and server errors."""
        for attempt in range(MAX_RETRIES + 1):
            last = attempt == MAX_RETRIES
            try:
                response = await self._http.get(path, params=params)
            except httpx.HTTPError as exc:
                if last:
                    raise OpenAlexError(f"OpenAlex request failed: {exc}") from exc
            else:
                retryable = response.status_code == 429 or response.status_code >= 500
                if not response.is_error:
                    return response.json()
                if last or not retryable:
                    raise OpenAlexError(
                        f"OpenAlex error {response.status_code}: {response.text[:300]}"
                    )
            await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable")
