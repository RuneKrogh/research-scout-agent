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

The evaluation set in `eval/prompts.json` has 13 prompts. They cover broad and narrow topics, before/after/exact year constraints, minimum and maximum citation counts, two requests where no good paper exists, and three ambiguous requests. Each prompt also stores the constraints it should be parsed into and whether a paper is expected.

```bash
uv run python eval/run_eval.py           # run all prompts, saves eval/results.json
uv run python eval/run_eval.py --report  # print the results table
```

Some things about an answer are simply right or wrong, like whether the paper is from the right year. Code can check those. Other things, like whether the paper actually fits the topic, need someone to read it. I split the evaluation the same way.

This split is common in research on evaluating agents. τ-bench ([Yao et al., 2024](https://arxiv.org/abs/2406.12045)) scores an agent by comparing what it did with a known correct result. Another LLM could grade the rest, but [Zheng et al. (2023)](https://arxiv.org/abs/2306.05685) show that LLM graders have biases, and with only 13 prompts it was easy to grade them myself. When scoring the explanations, I only counted a claim if the paper's abstract supports it, which is the same idea RAGAS ([Es et al., 2023](https://arxiv.org/abs/2309.15217)) calls faithfulness.

The script checks that the agent found a paper when one exists and said "none" when it doesn't. It also checks that the agent read the year and citation limits correctly, and that the paper it picked meets them. Finally, it looks the paper up on OpenAlex again and compares the title, year and citation count, which catches made-up details. Citation counts may differ by up to 5 %, because they keep growing.

I scored relevance and explanations by hand in `eval/manual_scores.json`, by reading each answer next to the paper's abstract. Each score has a short note on why.

| # | Request | Paper | Outcome | Year | Citations | Source | Relevant | Explanation |
|---|---|---|---|---|---|---|---|---|
| 1 | broad topic, after year, min citations | Demystifying LLM-Based Software Engineering Agents (2025, 1058) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 2 | before year, min citations | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020, 19229) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 3 | exact year, min citations | Graph WaveNet for Deep Spatial-Temporal Graph Modeling (2019, 4548) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 4 | narrow topic, after year, min citations | Automated Program Repair in the Era of Large Pre-trained Language Models (2023, 887) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 5 | narrow topic, from year, min citations | Secure, privacy-preserving and federated machine learning in medical imaging (2020, 2082) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 6 | broad topic, before year, high citations | Playing Atari with Deep Reinforcement Learning (2013, 17230) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 7 | exact year, max citations | Forensic Analysis of Indirect Prompt Injection Attacks on LLM Agents (2024, 16) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 8 | approximate citations | Rethinking Spatial Dimensions of Vision Transformers (2021, 986) | ✓ | – | ✓ | ✓ | ✓ | ✓ |
| 9 | no good paper (impossible constraints) | none | ✓ | – | – | – | – | ✓ |
| 10 | no good paper (too specific) | none | ✓ | – | – | – | – | ✓ |
| 11 | ambiguous (recent) | Demystifying LLM-Based Software Engineering Agents (2025, 1058) | ✓ | ✓ | – | ✓ | ✗ | ✗ |
| 12 | ambiguous (vague topic) | Attention Is All You Need (2017, 151778) | ✓ | – | – | ✓ | ✓ | ✓ |
| 13 | ambiguous (older, influential) | Word Representations: A Simple and General Method for Semi-Supervised Learning (2010, 2392) | ✓ | – | – | ✓ | ✓ | ✓ |

Every automatic check passed. The agent always gave the right kind of answer, read every year and citation limit correctly, stayed within those limits, and never made up details about a paper. For the two impossible requests it searched three or four times, then said nothing matched and explained why.

The hand-scored part is weaker. 10 of 11 papers fit the topic, and 11 of 13 explanations were good. Both misses are the same paper, Agentless. Its point is that you don't need complex tool-using agents to fix software bugs. The agent described it as the opposite, a study of how such agents use tools. It repeated the start of the abstract and missed the conclusion. In #11, which asked about agents that use tools, that also makes it the wrong paper.

So the code can make sure the facts about a paper are correct, but not that the LLM understood the paper.

## Reflection

_TBD_

## Contributions

Solo project: Rune Krogh.
