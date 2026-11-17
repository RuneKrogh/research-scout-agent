import pytest

from research_scout.models import (
    CitationConstraint,
    Paper,
    PaperResult,
    SearchConstraints,
    YearConstraint,
)


def paper(year=2023, citations=150):
    return Paper(
        openalex_id="https://openalex.org/W1",
        title="A paper",
        authors=["A. Author"],
        year=year,
        citation_count=citations,
        doi="https://doi.org/10.1/x",
    )


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("before", (None, 2020)),
        ("after", (2022, None)),
        ("in", (2021, 2021)),
        ("from", (2021, None)),
        ("until", (None, 2021)),
    ],
)
def test_year_bounds(operator, expected):
    assert YearConstraint(operator=operator, year=2021).bounds() == expected


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("at_least", (100, None)),
        ("more_than", (101, None)),
        ("at_most", (None, 100)),
        ("less_than", (None, 99)),
        ("approximately", (80, 120)),
    ],
)
def test_citation_bounds(operator, expected):
    assert CitationConstraint(operator=operator, value=100).bounds() == expected


def test_search_filters_are_inclusive_bounds():
    c = SearchConstraints(
        topic="RAG",
        year=YearConstraint(operator="before", year=2021),
        citations=CitationConstraint(operator="more_than", value=500),
    )
    assert c.search_filters() == {"max_year": 2020, "min_citations": 501}

    exact = SearchConstraints(topic="RAG", year=YearConstraint(operator="in", year=2023))
    assert exact.search_filters() == {"min_year": 2023, "max_year": 2023}


def test_violations():
    c = SearchConstraints(
        topic="agents",
        year=YearConstraint(operator="after", year=2022),
        citations=CitationConstraint(operator="at_least", value=100),
    )
    assert c.violations(paper()) == []
    assert c.violations(paper(year=2022)) == ["year 2022 is not after 2022"]
    assert c.violations(paper(citations=99)) == ["99 citations is not at least 100"]
    assert c.violations(paper(year=None, citations=None)) == [
        "publication year unknown",
        "citation count unknown",
    ]


def test_no_constraints_accepts_anything():
    assert SearchConstraints(topic="agents").violations(paper(year=None)) == []


def test_paper_result_takes_metadata_from_tool():
    result = PaperResult.from_paper(paper())
    assert result.citation_count == 150
    assert result.citation_source == "OpenAlex"
    assert result.citation_source_url == "https://openalex.org/W1"
    assert result.link == "https://doi.org/10.1/x"
