# Arcee Trinity vs Qwen: Model Family Comparison

**Date:** 2026-04-29

This document compares the Arcee Trinity and Qwen model families using two
independent evaluation sources. Each source evaluates a **different Trinity
variant**, so the two sections should be read as complementary views rather than
direct cross-references.

| Section | Trinity Model Evaluated | Evaluation Method |
|---------|------------------------|-------------------|
| [Part 1: Chatbot Arena](#part-1-chatbot-arena-human-preference-ratings) | Trinity-Large-Preview | Human preference votes |
| [Part 2: Artificial Analysis](#part-2-artificial-analysis-automated-benchmarks) | Trinity Large Thinking | Automated benchmarks |

---

# Part 1: Chatbot Arena Human Preference Ratings

**Source:**
[lmarena-ai/leaderboard-dataset](https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset),
`text_style_control` config, `latest` split (CC-BY-4.0)

## About Chatbot Arena

[Chatbot Arena](https://lmarena.ai/) (formerly LMSYS) ranks models using
**head-to-head human preference votes**. Real users submit prompts to two
anonymous models side-by-side, then choose which response they prefer. These
pairwise outcomes are aggregated into Bradley-Terry ratings (similar to Elo in
chess) -- higher is better, with scores typically ranging from ~900 to ~1550.

The `text_style_control` variant adjusts for verbosity bias, so models don't get
rewarded simply for producing longer responses.

Ratings are broken down into **27 topic-based categories** (coding, math,
creative writing, legal, etc.) based on conversation content, giving a detailed
profile of where each model excels relative to the full field. Because these are
human judgments rather than automated test suites, they reflect what real users
find helpful -- but they also carry the biases of the voter population (skewed
toward AI-savvy early adopters) and the preference-vs-correctness gap (a
confident but wrong answer can still win votes).

**Trinity model evaluated:** `trinity-large-preview` (17,933 votes).
`trinity-large-thinking` has not been submitted to Arena.

**Qwen coverage:** 35 models across generations 1.5, 2, 2.5, 3, 3.5, and 3.6.

---

## Global Arena Rankings

Before comparing the two families to each other, here is where the top models
from each family sit in the **full Arena leaderboard** (351 models total).

### Top Qwen model: qwen3.5-max-preview (rank 21 of 351)

| Rank | Model | Rating | Votes |
|-----:|-------|-------:|------:|
| 1 | claude-opus-4-7-thinking | 1503.2 | 5,996 |
| 2 | claude-opus-4-6-thinking | 1501.5 | 20,917 |
| 3 | claude-opus-4-6 | 1496.0 | 22,200 |
| 4 | claude-opus-4-7 | 1493.3 | 6,746 |
| 5 | gemini-3.1-pro-preview | 1493.1 | 26,125 |
| 6 | muse-spark | 1489.0 | 7,931 |
| 7 | gpt-5.5-high | 1488.1 | 3,587 |
| 8 | gemini-3-pro | 1485.8 | 41,400 |
| 9 | grok-4.20-beta1 | 1480.5 | 15,438 |
| 10 | gpt-5.4-high | 1478.8 | 14,328 |
| 11 | grok-4.20-beta-0309-reasoning | 1478.0 | 14,557 |
| 12 | gpt-5.2-chat-latest-20260210 | 1476.7 | 20,668 |
| 13 | grok-4.20-multi-agent-beta-0309 | 1475.7 | 14,948 |
| 14 | gpt-5.5 | 1475.0 | 3,757 |
| 15 | gemini-3-flash | 1473.5 | 30,809 |
| 16 | claude-opus-4-5-thinking-32k | 1472.7 | 37,181 |
| 17 | glm-5.1 | 1469.9 | 9,670 |
| 18 | grok-4.1-thinking | 1468.7 | 52,248 |
| 19 | claude-opus-4-5 | 1468.5 | 52,042 |
| 20 | gpt-5.4 | 1467.1 | 14,942 |
| **21** | **qwen3.5-max-preview** | **1465.4** | **12,130** |
| 22 | mimo-v2.5-pro | 1465.4 | 3,375 |
| 23 | deepseek-v4-pro | 1463.1 | 4,175 |
| 24 | claude-sonnet-4-6 | 1463.0 | 14,205 |
| 25 | gemini-3-flash (thinking-minimal) | 1462.9 | 38,415 |
| 26 | deepseek-v4-pro-thinking | 1462.1 | 3,823 |

### Trinity-Large-Preview (rank 133 of 351)

| Rank | Model | Rating | Votes |
|-----:|-------|-------:|------:|
| 128 | hunyuan-turbos-20250416 | 1382.3 | 10,722 |
| 129 | gpt-4.1-mini-2025-04-14 | 1382.2 | 39,367 |
| 130 | gemini-2.5-flash-lite-no-thinking | 1379.6 | 47,300 |
| 131 | glm-4.6v | 1377.3 | 2,807 |
| 132 | qwen3-235b-a22b | 1374.4 | 26,287 |
| **133** | **trinity-large-preview** | **1374.4** | **17,933** |
| 134 | gemini-2.5-flash-lite-thinking | 1374.3 | 32,949 |
| 135 | qwen2.5-max | 1374.1 | 32,626 |
| 136 | glm-4.5-air | 1372.8 | 31,130 |
| 137 | claude-3-5-sonnet-20241022 | 1372.0 | 88,363 |
| 138 | claude-3-7-sonnet-20250219 | 1370.7 | 43,216 |

---

## Trinity vs Qwen Rankings

Within just the Trinity + Qwen subset (36 models), Trinity-Large-Preview sits at
rank 19, tied with `qwen3-235b-a22b` and `qwen2.5-max` at ~1374.

| Rank | Model | Votes | Overall |
|-----:|-------|------:|--------:|
| 1 | qwen3.5-max-preview | 12,130 | 1465.4 |
| 2 | qwen3.5-397b-a17b | 19,928 | 1446.7 |
| 3 | qwen3.6-plus | 6,143 | 1446.4 |
| 4 | qwen3-max-preview | 27,767 | 1434.6 |
| 5 | qwen3-max-2025-09-23 | 9,182 | 1423.9 |
| 6 | qwen3-235b-a22b-instruct-2507 | 85,971 | 1422.5 |
| 7 | qwen3.5-122b-a10b | 16,762 | 1417.9 |
| 8 | qwen3-vl-235b-a22b-instruct | 11,529 | 1415.5 |
| 9 | qwen3.5-27b | 16,330 | 1404.6 |
| 10 | qwen3-235b-a22b-no-thinking | 38,251 | 1402.9 |
| 11 | qwen3-next-80b-a3b-instruct | 22,912 | 1401.5 |
| 12 | qwen3-235b-a22b-thinking-2507 | 9,008 | 1399.1 |
| 13 | qwen3.5-flash | 17,064 | 1398.2 |
| 14 | qwen3.5-35b-a3b | 16,914 | 1396.2 |
| 15 | qwen3-vl-235b-a22b-thinking | 7,951 | 1395.7 |
| 16 | qwen3-coder-480b-a35b-instruct | 25,762 | 1387.4 |
| 17 | qwen3-30b-a3b-instruct-2507 | 23,771 | 1383.2 |
| **18** | **qwen3-235b-a22b** | **26,287** | **1374.4** |
| **19** | **trinity-large-preview** | **17,933** | **1374.4** |
| **20** | **qwen2.5-max** | **32,626** | **1374.1** |
| 21 | qwen3-next-80b-a3b-thinking | 13,713 | 1369.1 |
| 22 | qwen3-32b | 3,926 | 1347.0 |
| 23 | qwen-plus-0125 | 5,819 | 1345.9 |
| 24 | qwen3-30b-a3b | 26,506 | 1327.1 |
| 25 | qwen-max-0919 | 16,478 | 1317.6 |
| 26 | qwen2.5-plus-1127 | 10,187 | 1314.8 |
| 27 | qwen2.5-72b-instruct | 39,406 | 1302.4 |
| 28 | qwen2.5-coder-32b-instruct | 5,432 | 1270.1 |
| 29 | qwen2-72b-instruct | 37,325 | 1261.0 |
| 30 | qwen1.5-110b-chat | 26,195 | 1233.3 |
| 31 | qwen1.5-72b-chat | 39,302 | 1232.4 |
| 32 | qwen1.5-32b-chat | 21,741 | 1203.0 |
| 33 | qwen1.5-14b-chat | 17,839 | 1190.1 |
| 34 | qwen1.5-7b-chat | 4,737 | 1142.9 |
| 35 | qwen-14b-chat | 4,964 | 1137.7 |
| 36 | qwen1.5-4b-chat | 7,597 | 1089.3 |

---

## Full Category Ratings

Ratings across all 14 key categories for Trinity and the closest Qwen models.

### Part 1: General capabilities

| Model | overall | coding | math | creative | instruct | hard | expert |
|-------|--------:|-------:|-----:|---------:|---------:|-----:|-------:|
| qwen3-235b-a22b-instruct-2507 | 1422.5 | 1471.4 | 1419.9 | 1380.3 | 1415.5 | 1448.3 | 1448.3 |
| qwen3-235b-a22b-no-thinking | 1402.9 | 1445.4 | 1394.4 | 1368.2 | 1380.7 | 1421.1 | 1395.3 |
| qwen3-235b-a22b | 1374.4 | 1432.8 | 1393.3 | 1323.7 | 1356.4 | 1391.6 | 1384.6 |
| **trinity-large-preview** | **1374.4** | **1440.8** | **1358.7** | **1351.1** | **1370.1** | **1400.8** | **1409.9** |
| qwen2.5-max | 1374.1 | 1402.3 | 1364.5 | 1354.0 | 1356.5 | 1385.0 | 1367.0 |
| qwen3-32b | 1347.0 | 1407.1 | 1398.7 | 1304.9 | 1331.2 | 1367.5 | 1397.6 |
| qwen3-30b-a3b | 1327.1 | 1386.2 | 1352.3 | 1284.7 | 1311.1 | 1345.4 | 1343.9 |

### Part 2: Conversational and industry categories

| Model | multi_turn | long_query | sw/it | legal | science | ind_math | writing |
|-------|----------:|-----------:|------:|------:|--------:|---------:|--------:|
| qwen3-235b-a22b-instruct-2507 | 1437.3 | 1433.6 | 1461.9 | 1434.2 | 1443.7 | 1433.8 | 1396.8 |
| qwen3-235b-a22b-no-thinking | 1410.6 | 1411.1 | 1438.6 | 1405.1 | 1417.0 | 1403.1 | 1379.1 |
| qwen3-235b-a22b | 1372.0 | 1378.0 | 1416.2 | 1364.8 | 1382.5 | 1400.6 | 1345.1 |
| **trinity-large-preview** | **1375.2** | **1394.4** | **1421.8** | **1406.5** | **1401.7** | **1371.8** | **1355.4** |
| qwen2.5-max | 1372.6 | 1384.1 | 1397.5 | 1387.1 | 1389.7 | 1368.1 | 1362.3 |
| qwen3-32b | 1338.1 | 1354.9 | 1398.3 | 1326.3 | 1378.1 | 1424.9 | 1307.9 |
| qwen3-30b-a3b | 1320.8 | 1339.6 | 1371.2 | 1323.0 | 1333.0 | 1367.1 | 1304.6 |

---

## Head-to-Head: Trinity vs qwen3-235b-a22b

The closest Qwen model by overall rating. Trinity wins 11 of 14 categories.

| Category | Trinity | Qwen | Delta | Winner |
|----------|--------:|-----:|------:|--------|
| legal | 1406.5 | 1364.8 | **+41.7** | Trinity |
| creative | 1351.1 | 1323.7 | **+27.4** | Trinity |
| expert | 1409.9 | 1384.6 | **+25.3** | Trinity |
| science | 1401.7 | 1382.5 | +19.2 | Trinity |
| long_query | 1394.4 | 1378.0 | +16.3 | Trinity |
| instruct | 1370.1 | 1356.4 | +13.7 | Trinity |
| writing | 1355.4 | 1345.1 | +10.3 | Trinity |
| hard_prompts | 1400.8 | 1391.6 | +9.2 | Trinity |
| coding | 1440.8 | 1432.8 | +8.0 | Trinity |
| sw/it | 1421.8 | 1416.2 | +5.6 | Trinity |
| multi_turn | 1375.2 | 1372.0 | +3.2 | Trinity |
| overall | 1374.4 | 1374.4 | -0.0 | Tie |
| ind_math | 1371.8 | 1400.6 | **-28.8** | Qwen |
| math | 1358.7 | 1393.3 | **-34.6** | Qwen |

## Head-to-Head: Trinity vs qwen2.5-max

The second-closest Qwen model by overall rating. Trinity wins 11 of 14
categories.

| Category | Trinity | Qwen | Delta | Winner |
|----------|--------:|-----:|------:|--------|
| expert | 1409.9 | 1367.0 | **+42.9** | Trinity |
| coding | 1440.8 | 1402.3 | **+38.5** | Trinity |
| sw/it | 1421.8 | 1397.5 | +24.3 | Trinity |
| legal | 1406.5 | 1387.1 | +19.4 | Trinity |
| hard_prompts | 1400.8 | 1385.0 | +15.8 | Trinity |
| instruct | 1370.1 | 1356.5 | +13.6 | Trinity |
| science | 1401.7 | 1389.7 | +12.0 | Trinity |
| long_query | 1394.4 | 1384.1 | +10.3 | Trinity |
| ind_math | 1371.8 | 1368.1 | +3.7 | Trinity |
| multi_turn | 1375.2 | 1372.6 | +2.6 | Trinity |
| overall | 1374.4 | 1374.1 | +0.3 | Trinity |
| creative | 1351.1 | 1354.0 | -3.0 | Qwen |
| math | 1358.7 | 1364.5 | -5.8 | Qwen |
| writing | 1355.4 | 1362.3 | -6.9 | Qwen |

## Head-to-Head: Trinity vs qwen3-235b-a22b-no-thinking

One tier above Trinity overall (+28.5). Trinity wins only 2 of 14 categories.

| Category | Trinity | Qwen | Delta | Winner |
|----------|--------:|-----:|------:|--------|
| expert | 1409.9 | 1395.3 | +14.6 | Trinity |
| legal | 1406.5 | 1405.1 | +1.5 | Trinity |
| coding | 1440.8 | 1445.4 | -4.6 | Qwen |
| instruct | 1370.1 | 1380.7 | -10.5 | Qwen |
| science | 1401.7 | 1417.0 | -15.3 | Qwen |
| long_query | 1394.4 | 1411.1 | -16.8 | Qwen |
| sw/it | 1421.8 | 1438.6 | -16.8 | Qwen |
| creative | 1351.1 | 1368.2 | -17.2 | Qwen |
| hard_prompts | 1400.8 | 1421.1 | -20.3 | Qwen |
| writing | 1355.4 | 1379.1 | -23.7 | Qwen |
| overall | 1374.4 | 1402.9 | -28.5 | Qwen |
| ind_math | 1371.8 | 1403.1 | -31.3 | Qwen |
| math | 1358.7 | 1394.4 | -35.7 | Qwen |
| multi_turn | 1375.2 | 1410.6 | -35.5 | Qwen |

## Head-to-Head: Trinity vs qwen3-32b

Trinity is significantly ahead overall (+27.5). Trinity wins 12 of 14
categories.

| Category | Trinity | Qwen | Delta | Winner |
|----------|--------:|-----:|------:|--------|
| legal | 1406.5 | 1326.3 | **+80.2** | Trinity |
| writing | 1355.4 | 1307.9 | +47.5 | Trinity |
| creative | 1351.1 | 1304.9 | +46.1 | Trinity |
| long_query | 1394.4 | 1354.9 | +39.5 | Trinity |
| instruct | 1370.1 | 1331.2 | +39.0 | Trinity |
| multi_turn | 1375.2 | 1338.1 | +37.1 | Trinity |
| coding | 1440.8 | 1407.1 | +33.7 | Trinity |
| hard_prompts | 1400.8 | 1367.5 | +33.3 | Trinity |
| overall | 1374.4 | 1347.0 | +27.5 | Trinity |
| science | 1401.7 | 1378.1 | +23.6 | Trinity |
| sw/it | 1421.8 | 1398.3 | +23.5 | Trinity |
| expert | 1409.9 | 1397.6 | +12.3 | Trinity |
| math | 1358.7 | 1398.7 | **-39.9** | Qwen |
| ind_math | 1371.8 | 1424.9 | **-53.1** | Qwen |

---

## Win/Loss Summary

Category wins across all head-to-head matchups (14 categories each):

| vs Qwen Model | Trinity Wins | Qwen Wins | Overall Gap |
|----------------|:------------:|:---------:|:-----------:|
| qwen3-30b-a3b | 14 | 0 | +47.3 |
| qwen3-32b | 12 | 2 | +27.5 |
| qwen2.5-max | 11 | 3 | +0.3 |
| qwen3-235b-a22b | 11 | 3 | 0.0 |
| qwen3-235b-a22b-no-thinking | 2 | 12 | -28.5 |
| qwen3-235b-a22b-instruct-2507 | 0 | 14 | -48.1 |

---

## Arena Key Findings

1. **Overall positioning:** Trinity-Large-Preview matches `qwen3-235b-a22b` and
   `qwen2.5-max` in overall rating (~1374), placing it in the upper-middle tier
   of the Qwen family.

2. **Trinity's strengths** (relative to same-tier Qwen models):
   - **Legal/government** (+42 vs qwen3-235b-a22b) -- the largest advantage
   - **Expert reasoning** (+25 vs qwen3-235b-a22b, +43 vs qwen2.5-max)
   - **Creative writing** (+27 vs qwen3-235b-a22b)
   - **Science** (+19 vs qwen3-235b-a22b)
   - **Coding** (+8 vs qwen3-235b-a22b, +39 vs qwen2.5-max)

3. **Trinity's weakness:** Math is the consistent weak spot. Trinity loses the
   math category in every matchup, even against lower-ranked models like
   `qwen3-32b` (-40) and `qwen3-30b-a3b` (-6 margin, despite +47 overall).

4. **Against newer instruct-tuned Qwen models** (e.g.,
   `qwen3-235b-a22b-instruct-2507`), Trinity falls behind across all categories,
   with gaps of 30-62 points.

5. **Profile difference:** Trinity has a more balanced, humanities-leaning
   profile (strong in legal, creative, expert, science). Qwen models generally
   skew toward STEM (math, coding, industry_mathematical), with the gap
   narrowing or reversing in non-STEM categories.

---
---

# Part 2: Artificial Analysis Automated Benchmarks

**Source:** [Artificial Analysis](https://artificialanalysis.ai/) (accessed
2026-04-29)

## About Artificial Analysis

[Artificial Analysis](https://artificialanalysis.ai/) evaluates models using
**automated benchmark suites** -- standardized tests with known correct answers,
run programmatically. Their Intelligence Index (v4.0) aggregates scores from 10
evaluations:

- **GPQA Diamond** -- graduate-level science questions
- **Humanity's Last Exam (HLE)** -- extremely difficult cross-domain questions
- **SciCode** -- scientific coding problems
- **Terminal-Bench Hard** -- complex terminal/CLI tasks
- **IFBench** -- instruction following
- **AA-LCR** -- long-context retrieval
- **AA-Omniscience** -- broad knowledge assessment
- **GDPval-AA** -- GDP prediction (quantitative reasoning)
- **tau2-Bench Telecom** -- domain-specific agent tasks
- **CritPt** -- critical thinking

Unlike Arena's human preference votes, these benchmarks have **objectively
correct answers** -- a model either retrieves the right needle from a long
context or it doesn't, it either solves the math problem or it doesn't. This
makes AA scores more precise for measurable capabilities (coding, math, factual
recall) but less reflective of subjective qualities like writing style,
helpfulness, or conversational fluency.

AA also independently measures **speed** (output tokens/sec), **latency** (time
to first token), and **pricing** across API providers, providing a practical
deployment perspective that Arena does not cover.

**Trinity model evaluated:** `Trinity Large Thinking` (reasoning model, 399B
total / 13B active parameters). The non-thinking variants (Trinity Large,
Trinity Large Preview) are not on the site.

**Qwen coverage:** Extensive across generations 2, 2.5, 3, 3.5, and 3.6,
including both reasoning and non-reasoning modes.

---

## Global AA Rankings

Before the Trinity-vs-Qwen comparison, here is where each family's top models
sit in the **full AA Intelligence Index** (~81 models evaluated).

### Top Qwen model: Qwen3.6 Max Preview (score 52, ~rank 12)

| Rank | Model | AA Intelligence |
|-----:|-------|:---------------:|
| 1 | GPT-5.5 (xhigh) | 60 |
| 2 | GPT-5.5 (high) | 59 |
| 3 | Claude Opus 4.7 (max) | 57 |
| 4 | Gemini 3.1 Pro Preview | 57 |
| 5 | GPT-5.4 (xhigh) | 57 |
| 6 | GPT-5.5 (medium) | 57 |
| 7 | Kimi K2.6 | 54 |
| 8 | MiMo-V2.5-Pro | 54 |
| 9 | GPT-5.3 Codex (xhigh) | 54 |
| 10 | Muse Spark | 52 |
| 11 | Claude Opus 4.7 (non-reasoning) | 52 |
| **12** | **Qwen3.6 Max Preview** | **52** |
| 13 | Claude Sonnet 4.6 (max) | 52 |
| 14 | DeepSeek V4 Pro (Max) | 52 |
| 15 | GLM-5.1 | 51 |
| 16 | GPT-5.5 (low) | 51 |
| **17** | **Qwen3.6 Plus** | **50** |

### Trinity Large Thinking (score 32, ~rank 39)

| Rank | Model | AA Intelligence |
|-----:|-------|:---------------:|
| 33 | gpt-oss-120B (high) | 33 |
| 34 | Mercury 2 | 33 |
| 35 | Qwen3.5 9B | 32 |
| 36 | Gemma 4 31B (non-reasoning) | 32 |
| 37 | K-EXAONE | 32 |
| 38 | DeepSeek V3.2 | 32 |
| **39** | **Trinity Large Thinking** | **32** |
| 40 | Qwen3.6 35B A3B (non-reasoning) | 32 |
| 41 | Grok 3 mini Reasoning (high) | 32 |
| 42 | Gemma 4 26B A4B | 31 |
| 43 | Claude 4.5 Haiku | 31 |
| 44 | Qwen3 Max | 31 |
| 45 | Qwen3.5 35B A3B (non-reasoning) | 31 |

---

## Trinity vs Qwen: Intelligence, Speed, Price, and Latency

### Reasoning Models

| Model | Params (total/active) | AA Intelligence | Speed (t/s) | TTFT (s) | Price ($/1M blend) | Context |
|-------|----------------------:|:---------------:|------------:|---------:|-------------------:|--------:|
| **Trinity Large Thinking** | **399B / 13B** | **32** | **132.0** | **1.01** | **$0.40** | **512k** |
| Qwen3 235B A22B 2507 (R) | 235B / 22B | 30 | 52.4 | 2.94 | $2.63 | 256k |
| Qwen3 235B A22B (R) | 235B / 22B | 20 | 54.5 | 2.89 | $2.63 | 33k |
| Qwen3 32B (R) | 32.8B / 32.8B | 17 | 94.0 | -- | $1.23 | 33k |
| Qwen3 30B A3B (R) | 30.5B / 3.3B | 15 | 75.3 | 2.44 | $0.75 | 33k |

### Non-Reasoning Models

| Model | Params (total/active) | AA Intelligence | Speed (t/s) | TTFT (s) | Price ($/1M blend) | Context |
|-------|----------------------:|:---------------:|------------:|---------:|-------------------:|--------:|
| Qwen3 Max | proprietary | 31 | 32.6 | 3.54 | $2.40 | 262k |
| Qwen3 235B A22B 2507 | 235B / 22B | 25 | 54.7 | 2.55 | $1.23 | 256k |
| Qwen3 235B A22B | 235B / 22B | 17 | 55.4 | 2.84 | $1.23 | 33k |
| Qwen2.5 Max | proprietary | 16 | 45.6 | 3.19 | $2.80 | 32k |
| Qwen3 32B | 32.8B / 32.8B | 15 | 90.3 | 2.46 | $1.23 | 33k |
| Qwen2.5 72B Instruct | 72B / 72B | 16 | 28.1 | 3.49 | ~$0.19 | 131k |
| Qwen3 30B A3B | 30.5B / 3.3B | 13 | 72.4 | 2.41 | $0.35 | 33k |

### Latest Generation (from AA leaderboard, not all have detailed pages)

| Model | AA Intelligence | Speed (t/s) | TTFT (s) | Price ($/1M blend) | Context |
|-------|:---------------:|------------:|---------:|-------------------:|--------:|
| Qwen3.6 Max Preview | 52 | 33 | 3.67 | $2.92 | 256k |
| Qwen3.6 Plus | 50 | 52 | 3.16 | $1.13 | 1M |
| Qwen3.6 27B | 46 | 64 | 3.93 | $1.35 | 262k |
| Qwen3.5 397B A17B | 45 | 53 | 2.36 | $1.35 | 262k |
| Qwen3.6 35B A3B | 43 | 194 | 2.50 | $0.56 | 262k |
| Qwen3.5 122B A10B | 42 | 119 | 2.61 | $1.10 | 262k |
| **Trinity Large Thinking** | **32** | **132** | **1.01** | **$0.40** | **512k** |

---

## AA Key Findings

1. **Intelligence:** Trinity Large Thinking scores 32 on the AA Intelligence
   Index, placing it above the closest open-weight Qwen reasoning models: Qwen3
   235B 2507 (R) at 30, and Qwen3 235B (R) at 20. However, Qwen's latest
   generation (3.5/3.6) scores significantly higher (42-52), though these are
   mostly proprietary API-only models.

2. **Speed advantage:** Trinity is dramatically faster than comparable Qwen
   models: 132 t/s vs 52-55 t/s for the Qwen3 235B variants. This is likely due
   to Trinity's much smaller active parameter count (13B active vs 22B active)
   despite having more total parameters (399B vs 235B).

3. **Latency advantage:** Trinity's TTFT of 1.01s is roughly 3x faster than the
   Qwen3 235B models (2.5-2.9s) and Qwen Max (3.5s).

4. **Price advantage:** Trinity's blended price of $0.40/1M tokens is
   significantly cheaper than the Qwen3 235B reasoning variant ($2.63/1M) and
   competitive with the smallest Qwen MoE model, Qwen3 30B A3B ($0.75/1M
   reasoning, $0.35/1M non-reasoning).

5. **Context window:** Trinity offers the largest context at 512k tokens,
   compared to 256k for the latest Qwen models and only 33k for the original
   Qwen3 235B.

6. **Verbosity caveat:** Trinity generated 150M output tokens during AA's
   Intelligence Index evaluation, which is very verbose compared to the 43M
   median. This inflates total evaluation cost and may affect perceived quality
   in cost-per-useful-token terms.

---
---

# Overall Conclusions

The two evaluation sources measure different things (human preference vs
automated benchmarks), evaluate different Trinity variants (Preview vs
Thinking), and tell somewhat different stories. Here is what holds up across
both.

## 1. Trinity is a strong mid-tier model, not a frontier model

Both sources place Trinity solidly in the middle of the pack globally:

- **Arena:** rank 133 of 351 (Trinity-Large-Preview at 1374)
- **AA:** score 32, rank ~39 of 81 (Trinity Large Thinking)

For context, the models that beat Trinity include not just the expected frontier
leaders (Claude Opus 4.7, GPT-5.x, Gemini 3 Pro) but also mid-range offerings
like GPT-4.1-mini, Gemini 2.5 Flash Lite, and Claude 3.5 Sonnet. Trinity is
competitive but not in the top tier on either source.

## 2. Qwen has a much broader and deeper lineup

Qwen fields 35 models in Arena and extensive coverage on AA, spanning from tiny
(1.5-4B) to frontier (Qwen3.6 Max Preview at AA score 52, Arena rank 21).
Trinity has exactly one model on each platform. This means:

- Qwen can cover a wider range of deployment scenarios (edge to cloud, cost-
  sensitive to quality-maximizing)
- Qwen's best models (3.5/3.6 generation) significantly outclass Trinity on both
  sources
- But Trinity doesn't need to compete with the full Qwen lineup -- it targets a
  specific niche

## 3. Trinity's niche: speed and cost at reasonable quality

The most consistent finding across both sources is Trinity's **deployment
efficiency**. On AA, Trinity Large Thinking delivers:

- **2.5x faster output** than comparable Qwen reasoning models (132 vs 52 t/s)
- **3x lower latency** (1.0s vs 2.9s TTFT)
- **6.5x cheaper** than Qwen3 235B reasoning ($0.40 vs $2.63/1M tokens)
- **2x the context window** (512k vs 256k)

This is the MoE architecture paying off: 399B total parameters give breadth of
knowledge, but only 13B active parameters keep inference fast and cheap.

## 4. Quality profiles differ: Trinity leans humanities, Qwen leans STEM

Arena category breakdowns show a consistent pattern when comparing Trinity
against same-tier Qwen models:

- **Trinity excels at:** legal/government (+42), expert reasoning (+25),
  creative writing (+27), science (+19)
- **Qwen excels at:** math (+35), mathematical industry (+29)

This suggests Trinity may be a better fit for use cases like document analysis,
legal research, content generation, and conversational chatbots, while Qwen has
the edge for math-heavy workloads.

## 5. Evaluation coverage is a limitation

A significant caveat: the two sources evaluate **different Trinity models**.
Arena has Trinity-Large-Preview (the non-thinking chat model), while AA has
Trinity Large Thinking (the reasoning model). There is no single source that
benchmarks both variants, and neither source covers the full Trinity lineup.
This makes it difficult to build a complete picture of Trinity's capabilities
compared to Qwen, where dozens of variants are benchmarked on both platforms.

## 6. Summary for deployment planning

| Factor | Trinity | Qwen |
|--------|---------|------|
| Top-tier quality | No -- mid-tier on both sources | Yes -- Qwen 3.5/3.6 are frontier-class |
| Same-tier quality | Competitive with Qwen3-235b-a22b | Broader range of options |
| Speed | Strong advantage (132 t/s) | Slower at comparable quality (52 t/s) |
| Latency | Strong advantage (1.0s TTFT) | Higher (2.5-3.5s) |
| Price | Strong advantage ($0.40/1M) | Higher at comparable quality ($1.23-2.63/1M) |
| Context window | 512k | 33k-262k (varies by model) |
| Math/STEM tasks | Weak spot | Consistent strength |
| Humanities/legal/creative | Strength | Weaker at same tier |
| Model variety | 1 model per platform | 35 models on Arena, extensive on AA |
| Open weights | Yes (Apache 2.0) | Yes for most (Apache 2.0 or Qwen License) |

**Bottom line:** Trinity offers an appealing price/performance ratio for
non-math workloads -- faster, cheaper, and longer-context than same-quality Qwen
models. But Qwen's latest generation (3.5/3.6) operates at a quality level
Trinity doesn't currently reach, and Qwen's breadth of model sizes gives it more
flexibility for different deployment constraints. The choice depends on whether
the priority is raw quality (Qwen 3.5/3.6), deployment efficiency at good-enough
quality (Trinity), or a specific domain strength.
