# Research Scout Agent

A PydanticAI agent that finds a research paper matching a topic, a publication year and a citation count, using the OpenAlex API.

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
uv run research-scout "Find a paper about retrieval-augmented generation published before 2021 with more than 500 citations."
```

## How it works

1. The LLM extracts the topic, year constraint and citation constraint from the request.
2. The search tool queries OpenAlex with those constraints and drops any paper that doesn't meet them.
3. The LLM picks the most relevant paper from the titles and abstracts and explains why. It can search again with a different query if nothing fits.
4. Code checks that the chosen paper was actually returned by the tool and meets the constraints, otherwise the LLM has to choose again.
5. Title, authors, year, citation count and link are copied from OpenAlex, so the LLM never writes them itself.

## LLM choice

Gemini 3.5 Flash-Lite through Google AI Studio's free tier. It supports tool calling and structured output, which the agent depends on. I started with Gemini Flash, but it was frequently overloaded on the free tier, while Flash-Lite responded in a few seconds with good results. The free tier has per-minute and per-day request limits, but one request uses only 2-3 model calls, so even a full evaluation run stays well within them.

## Evaluation

_TBD_

## Reflection

_TBD_

## Contributions

Solo project: Rune Krogh.
