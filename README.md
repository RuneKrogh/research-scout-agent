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

You can get a free Gemini key at [Google AI Studio](https://aistudio.google.com/apikey). An OpenAlex key is optional. Without one you share a small daily limit with everyone on your IP address, and a free key from [OpenAlex](https://openalex.org/settings/api) raises it.

## Usage

```bash
uv run research-scout "Find a paper about retrieval-augmented generation published before 2021 with more than 500 citations."
```

Add `--json` to get the full structured answer instead of the readable version.

## How it works

1. The LLM extracts the topic, year constraint and citation constraint from the request.
2. The search tool queries OpenAlex with those constraints and drops any paper that doesn't meet them.
3. The LLM picks the most relevant paper from the titles and the first part of each abstract, and explains why. It can search again with a different query if nothing fits. If something is unclear, like what "recent" means, it says how it read the request and what it is unsure about.
4. Code checks that the chosen paper was actually returned by the tool and meets the constraints, otherwise the LLM has to choose again.
5. Title, authors, year, citation count and link are copied from OpenAlex, so the LLM never writes them itself.

## LLM choice

Gemini 3.5 Flash-Lite through Google AI Studio's free tier. It supports tool calling and structured output, which the agent depends on. I started with Gemini Flash, but it was frequently overloaded on the free tier, while Flash-Lite responded in a few seconds with good results. The free tier allows 15 requests per minute and also has a daily limit. One request to the agent uses only 2-3 model calls, so normal use stays within it. Running all 13 evaluation prompts back to back did hit the per-minute limit once, so the evaluation script waits a minute and tries again.

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

The hand-scored part is weaker. 10 of 11 papers fit the topic, and 11 of 13 explanations were good. Both misses are the same paper, Agentless. Its point is that you don't need complex tool-using agents to fix software bugs. The agent described it as the opposite, a study of how such agents use tools. The reason is that the agent only sees the first 600 characters of each abstract, and for this paper the cut comes right before the authors say they skip agents. So the agent never saw the conclusion. In #11, which asked about agents that use tools, that also makes it the wrong paper.

So the code can make sure the facts about a paper are correct, but not that the LLM understood the paper.

## Reflection

The part that worked best was letting code handle everything that has a right answer. The LLM only reads the request and judges relevance. Searching, filtering and copying the paper details are done in code. Because of that, every year and citation check passed, and no answer had a wrong title or citation count. The agent was also honest when nothing fit. For the two impossible requests it searched a few times and then said there was no match, instead of returning something close.

The least reliable part was the free Gemini tier. Requests sometimes failed because the model was overloaded or the rate limit was hit, and the larger Flash models sometimes took over 100 seconds to answer. OpenAlex also needed some care. Its normal search includes the full text of papers and returned a lot of noise, so the tool only searches titles and abstracts. The same paper can also appear twice, for example as a preprint and as the published version, so duplicates are removed by title.

The agent usually needed only one search. That was the case for 9 of the 13 prompts. It searched more when it had a reason to. The two impossible requests took three and four searches, and the vague requests took two to four, because the agent tried other wordings before deciding.

The LLM never made up paper details, because it can't. It only returns the ID of a paper the tool found, and the code rejects any ID the tool didn't return. Its explanations are a different story. In the Agentless case the explanation said the opposite of what the paper argues. Every fact in the answer was correct, but the meaning was wrong, which is a quieter kind of hallucination that the code can't catch.

Wrong answers are prevented in a few layers. The search tool filters by year and citations, and the code checks the result again in case OpenAlex let something through. Before an answer is accepted, the code checks that the chosen paper came from the search and meets the constraints. If not, the LLM has to choose again. The paper details in the answer are copied from OpenAlex. The agent is also limited to four searches, so it can't keep searching forever.

With more time, I would first give the LLM more of each abstract, since that alone would probably have fixed the Agentless case. I would also make the evaluation set bigger, with several prompts for each kind of request, and use an LLM grader with a few spot checks by hand, so that relevance can be scored on more than 13 answers. Finally, a paid or more stable model would make the agent faster and remove most of the retries.

## Contributions

Solo project: Rune Krogh.
