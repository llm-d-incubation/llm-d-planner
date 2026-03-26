# Use-Case Specific Model Ranking Methodology

## Overview

This document explains how we create use-case specific CSV files that rank the 204 open-source models based on their performance for specific tasks.

## Methodology

### Data Source
- **Master CSV**: `opensource_all_benchmarks.csv` (204 open-source models)
- **Benchmark Source**: [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/methodology/intelligence-benchmarking)

### Weighting Strategy

Each use case has a custom weighting scheme that emphasizes relevant benchmarks based on the task requirements. The weights sum to 1.0 for each use case.

### Available Benchmarks

From the Artificial Analysis Intelligence Index Evaluation Suite:

1. **MMLU-Pro** - Multi-task language understanding (12,032 questions)
2. **HLE** - Humanity's Last Exam (2,684 questions)
3. **AA-LCR** - Long Context Reasoning (100 questions, 3 repeats)
4. **GPQA Diamond** - Scientific reasoning (198 questions, 5 repeats)
5. **AIME 2025** - Competition math (30 questions, 10 repeats)
6. **IFBench** - Instruction following (294 questions, 5 repeats)
7. **SciCode** - Scientific code generation (338 subproblems, 3 repeats)
8. **LiveCodeBench** - Code generation (315 questions, 3 repeats)
9. **Terminal-Bench Hard** - Agentic workflows (47 tasks, 3 repeats)
10. **τ²-Bench Telecom** - Agentic workflows (114 tasks, 3 repeats)
11. **Artificial Analysis Intelligence Index** - Composite score (all 10 benchmarks)
12. **Artificial Analysis Coding Index** - Composite score (LiveCodeBench, SciCode, Terminal-Bench Hard)
13. **Artificial Analysis Math Index** - Composite score (AIME 2025)

## Use Case Weightings

> **Updated 2024**: Weights adjusted based on research from SCORPIO, vLLM, Splitwise, SARATHI, Azure OpenAI, and Anthropic papers on LLM serving and workload patterns.

### 1. chatbot_conversational
**Description**: Real-time conversational chatbots (short prompts, short responses)

**Weights**:
- MMLU-Pro: 30% (General knowledge critical for conversations)
- IFBench: 30% (Instruction following **CRITICAL** for chat behavior) ↑10%
- HLE: 20% (Reasoning) ↓5%
- Intelligence Index: 15% (Overall intelligence)
- GPQA: 5% (Scientific reasoning less needed for chat) ↓5%

**Research Basis**: Azure OpenAI Chatbot Workload Study shows session-based interactions with 30-90 second think time. Instruction following is critical for proper conversational behavior. Scientific reasoning (GPQA) is less relevant for general chat.

---

### 2. code_completion
**Description**: Fast code completion/autocomplete (short prompts, short completions)

**Weights**:
- LiveCodeBench: 35% (Primary code benchmark) ↓5%
- SciCode: 30% (Scientific code understanding)
- Coding Index: 20% (Overall coding ability)
- Terminal-Bench Hard: 10% (Agentic workflows)
- IFBench: 5% (Follow code patterns/conventions) **NEW**

**Research Basis**: SCORPIO paper shows TTFT < 150ms is critical for code completion. GitHub Copilot research shows bursty workloads with pattern-following behavior. Added IFBench to ensure models follow existing code patterns and conventions.

---

### 3. code_generation_detailed
**Description**: Detailed code generation with explanations (medium prompts, long responses)

**Weights**:
- LiveCodeBench: 30% (Code generation)
- SciCode: 25% (Scientific code)
- IFBench: 20% (Instruction following for explanations)
- Coding Index: 15% (Overall coding)
- HLE: 10% (Reasoning for explanations)

**Research Basis**: Users wait for detailed output. Instruction following is important for generating explanations alongside code.

---

### 4. translation
**Description**: Document translation (medium prompts, medium responses)

**Weights**:
- IFBench: 35% (Instruction following critical for accurate translation)
- MMLU-Pro: 30% (Language understanding)
- HLE: 20% (Reasoning)
- Intelligence Index: 15% (Overall intelligence)

**Research Basis**: Google Cloud Translation patterns show instruction following is critical for accurate translation. Less need for coding or math capabilities.

---

### 5. content_generation
**Description**: Content creation, marketing copy (medium prompts, medium responses)

**Weights**:
- MMLU-Pro: 30% (General knowledge - facts to include)
- HLE: 25% (Reasoning)
- IFBench: 25% (Instruction following)
- Intelligence Index: 20% (Overall intelligence)

**Research Basis**: Adobe Research shows iterative generate→edit→regenerate pattern. Balanced approach with knowledge for factual content.

---

### 6. summarization_short
**Description**: Short document summarization (medium prompts, short summaries)

**Weights**:
- HLE: 30% (Reasoning **CRITICAL** for identifying key points) ↑5%
- MMLU-Pro: 25% (Understanding content) ↓5%
- IFBench: 25% (Instruction following)
- Intelligence Index: 20% (Overall intelligence)

**Research Basis**: AWS Document Processing Study shows reasoning is more critical than knowledge for summarization - the model must identify what's important, not just understand content. Differentiated from content_generation.

---

### 7. document_analysis_rag
**Description**: RAG-based document Q&A (long prompts with context, medium responses)

**Weights**:
- AA-LCR: 40% (Long context reasoning - **CRITICAL**) ↑10%
- MMLU-Pro: 20% (Knowledge retrieval) ↓5%
- HLE: 20% (Reasoning)
- IFBench: 10% (Instruction following) ↓5%
- τ²-Bench: 10% (Agentic workflows for complex queries)

**Research Basis**: vLLM and Splitwise papers show context handling dominates RAG performance. Anthropic RAG Production Analysis shows exploratory sessions with 5-20 questions. Long context is the bottleneck.

---

### 8. long_document_summarization
**Description**: Long document summarization (very long prompts, medium summaries)

**Weights**:
- AA-LCR: 45% (Long context reasoning - **CRITICAL**) ↑5%
- MMLU-Pro: 20% (Understanding) ↓5%
- HLE: 20% (Reasoning)
- IFBench: 15% (Instruction following)

**Research Basis**: SARATHI and Splitwise papers show TTFT is dominated by prefill time for long inputs (4K-128K tokens). Long context capability is the primary bottleneck.

---

### 9. research_legal_analysis
**Description**: Research/legal document analysis (very long prompts, detailed analysis)

**Weights**:
- AA-LCR: 40% (Long context reasoning - **CRITICAL**) ↑10%
- MMLU-Pro: 25% (Knowledge - **CRITICAL**)
- HLE: 15% (Reasoning) ↓5%
- GPQA: 10% (Scientific reasoning)
- IFBench: 5% (Instruction following) ↓5%
- τ²-Bench: 5% (Agentic workflows for complex analysis)

**Research Basis**: MLSys Enterprise LLM Workload study shows legal/research documents are 16K-128K tokens. Long context handling is the primary bottleneck. Batch-oriented, quality over speed.

---

## Score Calculation

For each model and use case:

1. Extract benchmark scores from the master CSV
2. Apply use-case specific weights to each available benchmark
3. Calculate weighted average: `score = Σ(benchmark_score × weight) / Σ(weight)`
4. Normalize by total weight (handles missing benchmarks gracefully)
5. Convert to percentage (0-100%)

**Missing Scores**: If a benchmark score is missing (N/A), it's excluded from the calculation. The final score is normalized by the sum of available weights.

## Output Files

Each use case generates a CSV file with:
- Model Name
- Provider
- Dataset
- Use Case Score (percentage)

Models are sorted by score (descending), with top performers listed first.

## Files Generated

1. `opensource_chatbot_conversational.csv`
2. `opensource_code_completion.csv`
3. `opensource_code_generation_detailed.csv`
4. `opensource_translation.csv`
5. `opensource_content_generation.csv`
6. `opensource_summarization_short.csv`
7. `opensource_document_analysis_rag.csv`
8. `opensource_long_document_summarization.csv`
9. `opensource_research_legal_analysis.csv`

## Usage

To regenerate all use-case CSV files:

```bash
python3 create_usecase_scores.py
```

## Notes

- Weights are based on the Artificial Analysis Intelligence Index methodology
- **Updated 2024**: Weights refined based on academic research papers:
  - SCORPIO (arXiv:2505.23022) - SLO-oriented LLM serving
  - vLLM (SOSP 2023) - Memory management and context handling
  - Splitwise (ISCA 2024) - Prefill vs decode optimization
  - SARATHI (ISCA 2024) - Chunked prefills for long context
  - Azure OpenAI Workload Study - Chatbot patterns
  - Anthropic RAG Production Analysis - Document Q&A patterns
  - MLSys Enterprise LLM Workload - Legal/research patterns
- All weights sum to 1.0 for consistency
- Missing benchmarks are handled gracefully (excluded from calculation)

## Change Log

| Use Case | Change | Reason |
|----------|--------|--------|
| chatbot_conversational | IFBench ↑10%, GPQA ↓5%, HLE ↓5% | Research shows instruction following critical for chat |
| code_completion | IFBench +5% (new), LiveCodeBench ↓5% | Pattern following for code conventions |
| summarization_short | HLE ↑5%, MMLU-Pro ↓5% | Reasoning needed to identify key points |
| document_analysis_rag | AA-LCR ↑10%, MMLU-Pro ↓5%, IFBench ↓5% | Long context dominates RAG performance |
| long_document_summarization | AA-LCR ↑5%, MMLU-Pro ↓5% | Prefill time is bottleneck |
| research_legal_analysis | AA-LCR ↑10%, HLE ↓5%, IFBench ↓5% | Legal docs are 16K-128K tokens |

