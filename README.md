# Research Scout Agent

A PydanticAI agent that finds a research paper matching a topic, a year constraint and a citation constraint. It verifies the result against an external paper API.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/RuneKrogh/research-scout-agent.git
cd research-scout-agent
uv sync
cp .env.example .env   # then set GOOGLE_API_KEY
```

## Usage

```bash
uv run research-scout "Find a paper about retrieval-augmented generation published before 2021 with more than 500 citations. Summarize its contribution in 5-7 sentences."
```

Add `--json` for the structured answer, or `--model` to use another PydanticAI model.

## How it works

1. **Parse** (LLM): the request becomes `SearchConstraints`: a topic plus an optional year constraint (before / after / in / from / until) and citation constraint (at least / more than / at most / less than / approximately).
2. **Search** (code): the `search_papers` tool converts the constraints to exact bounds and queries the [OpenAlex](https://openalex.org) API.
3. **Filter** (code): every result is checked against the constraints again; violating papers are rejected before the LLM sees them.
4. **Select** (LLM): the agent picks the most relevant candidate from titles and abstracts, may search again with a rephrased topic (max 4 searches), and explains its choice and any uncertainty.
5. **Validate** (code): the chosen ID must be a paper the tool returned and must satisfy the constraints; otherwise the LLM is asked to retry.
6. **Answer** (code): title, authors, year, citation count and link are copied from the OpenAlex record. The LLM never writes these fields, so it cannot invent a citation count.

## LLM choice

Google Gemini via the [Google AI Studio](https://aistudio.google.com) free tier (`gemini-3.5-flash-lite`, falling back to `gemini-3.1-flash-lite`):

- Free without a credit card, from the course's list of free LLM APIs.
- Supported natively by PydanticAI, with reliable tool calling and structured output.
- Flash-Lite instead of Flash: during development the free-tier Flash models were often overloaded (HTTP 503 or requests hanging for minutes), while Flash-Lite answered in a few seconds.
- Limits are per project and shown in AI Studio; requests per day reset at midnight Pacific time. Note that free-tier prompts may be used by Google to improve its products.

## Evaluation

_TBD_

## Reflection

_TBD_

## Contributions

Solo project: Rune Krogh.
