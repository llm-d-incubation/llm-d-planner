# Mapping LLM Use Cases to Traffic Profiles and Experience-Driven SLOs

## Overview

This document defines the framework for mapping **LLM use cases** to both their corresponding **traffic profiles** and **Service Level Objectives (SLOs)**.

The purpose is to guide capacity planning, hardware selection, and cost optimization by distinguishing between what is **technically feasible** and what is **necessary for good user experience**.

In practice, this framework supports a workflow like the following:

1. Determine the user's **use case**.  
2. Map that use case to its **traffic profile** (input/output token lengths).  
3. Apply the corresponding **SLO target ranges** for TTFT, ITL, and E2E latency.  
4. Generate a default SLO target within each range based on the user's **latency priority** (high/medium/low).
5. Combine with any user or system constraints (e.g., cost limits, available GPUs).  
6. Evaluate benchmark data to identify (GPU, model, configuration) combinations that meet those SLOs.  
7. If throughput exceeds a single GPU's capacity, scale horizontally with multiple instances.

The data backing this framework lives in `src/planner/data/configuration/usecase_slo_workload.json`.

---

## 1. Traffic Profile Definitions

Traffic profiles describe the shape of LLM workloads in terms of **prompt (input) tokens** and **completion (output) tokens**.  
Different applications naturally cluster around characteristic token ratios.

| **Prompt Tokens** | **Output Tokens** | **Pattern Description** | **Typical Use Cases** | **Notes** |
|-------------------:|------------------:|--------------------------|------------------------|------------|
| **512** | **256** | Medium input, short output | Chatbot, interactive Q&A, code completions, translation, content generation | Most common interactive workload; strongly latency-sensitive. |
| **1024** | **1024** | Long input, long output | Detailed code generation | Balanced workloads; typically less latency-sensitive but throughput-heavy. |
| **4096** | **512** | Very long input, short output | Summarization, document analysis, RAG Q&A over long context | Prefill-dominated workloads where TTFT is the primary bottleneck. |
| **10240** | **1536** | Extra-long input, medium output | Multi-document summarization, research or legal analysis | Edge workloads for long-context models; extremely memory- and bandwidth-intensive. |

### Rationale

- **Input length** drives **prefill cost** → directly impacts *Time to First Token (TTFT)*.  
- **Output length** drives **generation time** → affects *Inter-Token Latency (ITL)* and *End-to-End (E2E)* latency.  
- Use cases with the same traffic profiles may still differ in **SLO strictness** based on their **user experience expectations**.

---

## 2. Unified Mapping of Use Cases to Traffic Profiles and SLOs

The table below maps each use case to its traffic profile and SLO target ranges. The Planner generates a default target within each range based on the user's latency priority setting:

- **High** latency priority → 25th percentile of range (tighter targets)
- **Medium** latency priority → 50th percentile (balanced)
- **Low** latency priority → 75th percentile (more relaxed)

Users can review and adjust these defaults during the specification review step.

| **Use Case** | **Traffic Profile** <br>(Prompt → Output) | **TTFT p95 Range** | **ITL p95 Range** | **E2E p95 Range** | **Design Rationale** |
|---|---|---|---|---|---|
| **Chatbot / Conversational AI** | **512 → 256** | 100–500 ms | 15–50 ms | 3.9–13.3 s | Highly interactive; perceived responsiveness drives satisfaction. |
| **Code Completion (IDE autocomplete)** | **512 → 256** | 50–200 ms | 10–35 ms | 2.6–9.2 s | Sub-200 ms first-token latency essential for fluid typing experience. |
| **Code Generation (full implementations)** | **1024 → 1024** | 150–600 ms | 15–45 ms | 15.5–46.7 s | Users tolerate short delay for larger outputs; quality prioritized over immediacy. |
| **Translation** | **512 → 256** | 200–800 ms | 20–50 ms | 5.3–20 s | Non-interactive; a few-second delay acceptable. |
| **Content Generation** | **512 → 256** | 200–800 ms | 20–50 ms | 5.3–25 s | Emphasis on completeness and coherence over latency. |
| **Short Summarization (<10 pages)** | **4096 → 512** | 200–800 ms | 20–50 ms | 10.4–26.4 s | Prefill-heavy; user can wait briefly for summary. |
| **Document RAG / Q&A** | **4096 → 512** | 400–1200 ms | 25–60 ms | 13.2–40 s | Prefill cost dominates; responsiveness helps iterative Q&A. |
| **Long Document Summarization (10+ pages)** | **10240 → 1536** | 800–3000 ms | 30–70 ms | 46.9–110.5 s | User expects processing delay; prioritize throughput. |
| **Research / Legal Analysis** | **10240 → 1536** | 1500–5000 ms | 30–80 ms | 60–300 s | Asynchronous processing; cost and throughput optimized. |

---

## 3. Experience Classes (Design Rationale)

The experience classes below describe the UX reasoning behind the SLO ranges. They are a design framework, not a user-facing feature in the tool — the tool exposes the SLO ranges directly, and users adjust targets within those ranges.

| **Experience Class** | **User Expectation** | **Example Applications** | **Latency Tolerance** |
|----------------------|----------------------|---------------------------|------------------------|
| **Instant (UX-Critical)** | Feels real-time; user notices any delay | Code completion, inline assistants | TTFT ≤200 ms; ITL ≤35 ms |
| **Conversational** | Feels natural; output streams smoothly | Chatbots, Q&A, support bots | TTFT ≤500 ms; ITL ≤50 ms |
| **Interactive** | Some waiting acceptable | RAG workflows, analysis bots | TTFT ≤800 ms; ITL ≤50 ms |
| **Deferred** | User expects delay (spinner acceptable) | Translation, summarization | TTFT ≤1.2 s; ITL ≤60 ms |
| **Batch / Offline** | Fully asynchronous; throughput prioritized | Research, document processing | TTFT ≤5 s; ITL ≤80 ms |

---

## 4. Practical Implications

### Hardware and Cost Optimization
Two workloads can have the **same traffic profile** but **different latency requirements**:
- **Latency-sensitive use cases** (chat, code completion) justify **higher-end GPUs** (A100, H100) with lower batching and tighter scheduling.
- **Throughput-oriented use cases** (summarization, translation) can use **mid-range GPUs** (L40S, A10) with larger batches and lower cost.

### Deployment Strategy
| **Experience Class** | **Hardware Tier** | **Batching Strategy** | **Priority Goal** |
|----------------------|------------------|----------------------|-------------------|
| Instant / Conversational | Premium GPU (A100/H100) | Small batches (≤4) | Low TTFT & ITL |
| Interactive | Balanced GPU (L40S, A10G) | Medium batches (8–16) | Balance latency & throughput |
| Deferred / Batch | Cost GPU (A10, T4) | Large batches (≥16) | Maximize throughput / cost |

### Throughput Handling
If the required throughput exceeds the capacity of a single GPU:
- **Replicate instances** of the same type.
- Scale horizontally to achieve desired QPS (queries per second).
- Maintain per-instance SLO compliance before aggregation.

---

## 5. Summary

- Traffic profiles define computational **load shape** (prefill vs. generation ratio).  
- SLOs are defined as **ranges** — the tool picks a default target within each range based on the user's latency priority, and the user can adjust.  
- Identical traffic patterns may have **different SLO ranges** due to distinct UX requirements.  
- Separating "traffic profile" (what the workload *is*) from "experience class" (what the workload *needs*) enables precise capacity planning.  
- Hardware, batching, and scheduling strategies should be tuned to **meet the target SLOs** at the lowest feasible cost.  
