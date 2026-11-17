"""Client for the OpenAlex works API (https://help.openalex.org)."""

import asyncio
import os
import re
from typing import Literal

import httpx

from research_scout.models import Paper

BASE_URL = "https://api.openalex.org"
MAX_PER_PAGE = 100
ABSTRACT_MAX_CHARS = 1500
SELECT_FIELDS = (
    "id,doi,title,publication_year,cited_by_count,authorships,primary_location,"
    "abstract_inverted_index"
)

SortOrder = Literal["relevance", "citations"]


class OpenAlexError(RuntimeError):
    """Raised when OpenAlex cannot be reached or returns an error."""


def build_filter(
    *,
    year: int | None = None,
    year_after: int | None = None,
    year_before: int | None = None,
    min_citations: int | None = None,
    max_citations: int | None = None,
) -> str:
    """Build an OpenAlex filter string.

    Year bounds are exclusive (after 2022 means 2023 or later). Citation bounds are
    inclusive (at least 100 means 100 or more).
    """
    parts: list[str] = []
    if year is not None:
        parts.append(f"publication_year:{year}")
    if year_after is not None:
        parts.append(f"publication_year:>{year_after}")
    if year_before is not None:
        parts.append(f"publication_year:<{year_before}")
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
    text = " ".join(positions[i] for i in sorted(positions))
    if len(text) > ABSTRACT_MAX_CHARS:
        text = text[:ABSTRACT_MAX_CHARS].rsplit(" ", 1)[0] + " ..."
    return text


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


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


def deduplicate(papers: list[Paper]) -> list[Paper]:
    """Merge versions of the same paper (e.g. preprint and proceedings).

    Keeps the first occurrence's position and the highest citation count's record.
    """
    best: dict[str, Paper] = {}
    order: list[str] = []
    for paper in papers:
        key = _title_key(paper.title)
        if key not in best:
            order.append(key)
            best[key] = paper
        elif (paper.citation_count or 0) > (best[key].citation_count or 0):
            best[key] = paper
    return [best[k] for k in order]


class OpenAlexClient:
    """Async client for searching OpenAlex works."""

    def __init__(
        self,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 20.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("OPENALEX_API_KEY")
        self._http = http_client or httpx.AsyncClient(base_url=BASE_URL, timeout=timeout)
        self.max_retries = max_retries

    async def aclose(self) -> None:
        await self._http.aclose()

    async def search_works(
        self,
        query: str,
        *,
        year: int | None = None,
        year_after: int | None = None,
        year_before: int | None = None,
        min_citations: int | None = None,
        max_citations: int | None = None,
        sort: SortOrder = "relevance",
        limit: int = 10,
    ) -> list[Paper]:
        """Search works by topic in title and abstract, with optional filters."""
        if not query.strip():
            raise ValueError("query must not be empty")
        params: dict[str, str | int] = {
            "search.title_and_abstract": query,
            "select": SELECT_FIELDS,
            "per_page": max(1, min(limit, MAX_PER_PAGE)),
        }
        filters = build_filter(
            year=year,
            year_after=year_after,
            year_before=year_before,
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
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._http.get(path, params=params)
            except httpx.HTTPError as exc:
                if attempt == self.max_retries:
                    raise OpenAlexError(f"OpenAlex request failed: {exc}") from exc
                await asyncio.sleep(2**attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == self.max_retries:
                    break
                retry_after = response.headers.get("Retry-After", "")
                await asyncio.sleep(float(retry_after) if retry_after.isdigit() else 2**attempt)
                continue
            if response.is_error:
                raise OpenAlexError(f"OpenAlex error {response.status_code}: {response.text[:300]}")
            return response.json()
        raise OpenAlexError(f"OpenAlex error {response.status_code} after retries")
