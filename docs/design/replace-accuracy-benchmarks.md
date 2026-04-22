# Replace Artificial Analysis Benchmarks with Chatbot Arena Leaderboard Data

## Context

Artificial Analysis (AA) data is licensed for personal use only, making it
unsuitable for an open-source project. All AA-sourced data (204 models, 17
benchmarks, 9 weighted-score CSVs) must be replaced.

After evaluating several alternatives (HF Open LLM Leaderboard v2, LiveBench,
SEAL, OpenCompass), the **Chatbot Arena leaderboard dataset**
([lmarena-ai/leaderboard-dataset](https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset))
was selected as the replacement:

- **License:** CC-BY-4.0 (confirmed by lmarena-ai team, compatible with Apache
  2.0)
- **342 models** with Arena ratings based on 1.5M+ human preference votes
- **27 category-level scores** including: overall, coding, math,
  creative_writing, instruction_following, expert, hard_prompts, multi_turn,
  longer_query, plus industry-specific and language-specific categories
- **Actively maintained** — updated daily with new battle results
- **Includes both open and closed-source models** — quality scores are available
  for all; model availability depends on user credentials and access

### Why Arena over HF Open LLM Leaderboard v2?

The HF Open LLM Leaderboard was **retired in March 2025** and is no longer
updated. It also had no license set on its results dataset. The Arena
leaderboard is actively maintained, has clear licensing (CC-BY-4.0), and its
category-level scores (coding, math, instruction_following, etc.) map naturally
to our use cases — eliminating the need for a custom benchmark-weighting scheme.

### Key design change

The old approach used individual benchmark scores (MMLU-Pro, GPQA, etc.)
weighted per use case to produce composite scores. The new approach uses **Arena
category ratings directly** — each use case maps to one or more Arena categories
via a weighted combination. This is simpler and based on real human preference
judgments rather than automated benchmarks.

### Important: Human preference data, not automated benchmarks

Arena scores are **not** from automated benchmark suites (MMLU, HumanEval,
etc.). They are Bradley-Terry ratings derived from **1.5M+ pairwise human
preference votes** — real users comparing two anonymous model outputs and
choosing the better one. Category-level scores (coding, math, etc.) are
breakdowns of these human judgments by conversation topic.

**Strengths over automated benchmarks:**

- Measures what users actually care about (helpful, accurate, correct responses)
- Harder to game than standardized tests
- Style-controlled ratings reduce bias toward verbose responses
- Strong statistical confidence (all models have 500+ votes; most have
  thousands)

**Shortcomings to be aware of:**

- **Coverage gaps** — smaller and newer models may not be in Arena (3 of our
  catalog models are missing)
- **Preference ≠ correctness** — voters favor "impressive" responses, which may
  not always mean factually accurate or logically sound (e.g., a confident but
  wrong legal analysis may win votes)
- **No domain-specific depth** — Arena categories like
  `industry_legal_and_government` are based on conversation topic, not verified
  domain expertise. Purpose-built benchmarks (LegalBench, IFEval, LongBench)
  would provide stronger signals for specific use cases
- **Voter population bias** — Arena users skew toward AI-savvy early adopters,
  not necessarily representative of enterprise users

These limitations are addressable — see the **Future Enhancements** section for
plans to augment Arena data with automated benchmarks and domain-specific
evaluations.

For the initial implementation, the methodology documentation and scoring system
should be framed around "human preference ratings" rather than "benchmark
scores." If automated benchmarks are incorporated later (see Future
Enhancements), the framing should shift to reflect the hybrid approach —
"quality scores derived from human preference ratings and automated benchmarks."

---

## Data Source Details

**Dataset:** `text_style_control` config, `latest` split (style-controlled
ratings reduce bias toward verbose responses)

**Schema:** `model_name, organization, license, rating, rating_lower,
rating_upper, variance, vote_count, rank, category, leaderboard_publish_date`

**Rating scale:** Bradley-Terry model scores, typically ~900–1550. Must be
normalized to 0–100 for compatibility with the existing scoring system.

**Relevant categories for our use cases (27 total):**

| Category | Models | Rating Range | Description |
|----------|--------|-------------|-------------|
| `overall` | 342 | 952–1504 | General quality |
| `coding` | 337 | 880–1571 | Code generation and debugging |
| `math` | 332 | 889–1518 | Mathematical reasoning |
| `creative_writing` | 340 | 930–1499 | Creative text generation |
| `instruction_following` | 342 | 906–1515 | Following specific instructions |
| `hard_prompts` | 342 | 915–1535 | Challenging technical queries |
| `expert` | 291 | 1079–1562 | Expert-level reasoning |
| `multi_turn` | 340 | 865–1521 | Multi-turn conversation |
| `longer_query` | 320 | 1039–1526 | Long-form queries |
| `industry_software_and_it_services` | 342 | 902–1555 | Software/IT domain |
| `industry_legal_and_government` | 315 | 997–1517 | Legal domain |
| `industry_life_and_physical_and_social_science` | 340 | 938–1525 | Science domain |
| `industry_mathematical` | 325 | 893–1523 | Mathematical domain |
| `industry_writing_and_literature_and_language` | 341 | 923–1504 | Writing/language domain |

**Vote counts:** All 342 models in the `latest` split have 500+ votes; most have
thousands. The dataset is pre-curated for statistical confidence — no minimum
vote threshold filter is needed.

**Model name format:** Arena uses short names like `llama-3.1-8b-instruct`, not
HuggingFace repo IDs. Our catalog uses HF IDs like
`meta-llama/llama-3.1-8b-instruct`. Matching requires stripping the org prefix
and quantization suffixes from catalog IDs.

**Catalog coverage:** Analysis shows strong overlap — **44 of 47** catalog
models match Arena data after name normalization:

- 17 direct exact matches (llama 3.1/3.3/4, mistral-small, mixtral, granite,
  phi-4, gemma, gpt-oss, etc.)
- 22 quantized variants (fp8, w4a16, w8a8) that inherit their base model's Arena
  score
- 5 additional matches via improved normalization (stripping `meta-` prefix,
  `-hf` suffix, Kimi K2 name variants)
- **3 truly unmatched base models:** `qwen2.5-7b-instruct`, `qwen3-8b`,
  `smollm3-3b` — these were never submitted to Arena as competitors

**Handling unmatched models:** The 3 unmatched base models are covered via
family-based interpolation (see Phase A4). Their scores are estimated from
larger siblings in the same model family using empirical scaling ratios derived
from Arena data. Interpolated entries are marked in the CSV for transparency.
This gives all 47 catalog models a quality score — 44 from direct Arena data and
3 from interpolation.

---

## Phase A: Data Acquisition

### A1. Create `scripts/fetch_arena_leaderboard.py`

- Download `lmarena-ai/leaderboard-dataset` `text_style_control` config,
  `latest` split (Parquet)
- Pivot the data: one row per model, one column per category rating
- Normalize ratings to 0–100 scale using min-max normalization within each
  category
- Output to `data/benchmarks/quality/arena_leaderboard.csv`
- New CSV format:

  ```
  model_name,overall,coding,math,creative_writing,instruction_following,hard_prompts,expert,multi_turn,longer_query
  llama-3.1-8b-instruct,45.2,38.1,42.7,...
  ```

- Print coverage report: which of our 47 catalog models have matches

### A2. Create `scripts/refresh_arena_scores.sh`

End-to-end refresh script that fetches the latest Arena data and regenerates all
weighted score CSVs:

```bash
#!/usr/bin/env bash
# Refresh Arena leaderboard data and recalculate weighted scores.
# Usage: ./scripts/refresh_arena_scores.sh
#   or:  make refresh-quality-scores
set -euo pipefail

echo "Fetching latest Arena leaderboard data..."
uv run python scripts/fetch_arena_leaderboard.py

echo "Recalculating weighted scores..."
uv run python scripts/recalculate_weighted_scores.py

echo "Done. Review changes with: git diff data/benchmarks/quality/"
```

### A3. Add Makefile target

Add `refresh-quality-scores` target under the `##@ Data` section:

```makefile
##@ Data

refresh-quality-scores: ## Refresh Arena leaderboard data and recalculate weighted scores
 @printf "$(BLUE)Refreshing Arena quality scores...$(NC)\n"
 @./scripts/refresh_arena_scores.sh
 @printf "$(GREEN)✓ Arena scores refreshed$(NC)\n"
 @printf "$(YELLOW)Review changes: git diff data/benchmarks/quality/$(NC)\n"
```

### A4. Interpolate scores for missing catalog models

Three catalog base models have no Arena data: `qwen2.5-7b-instruct`, `qwen3-8b`,
`smollm3-3b`. Rather than falling back to 0.0, we estimate their scores using
known scaling relationships from Arena families with multiple parameter sizes.

**Reference scaling data from Arena (overall category):**

| Family | Small | Rating | Large | Rating | Ratio |
|--------|-------|--------|-------|--------|-------|
| Llama 3.1 | 8B | 1211 | 70B | 1293 | 0.937 |
| Gemma 2 | 2B | 1199 | 9B | 1265 | 0.948 |
| Gemma 2 | 9B | 1265 | 27B | 1288 | 0.982 |
| Qwen3 | 32B | 1347 | 235B | 1375 | 0.980 |

**Interpolation approach:**

1. For each missing model, identify the closest Arena model in the same family
   (same generation, different size)
2. Compute parameter ratio and apply the empirical scaling pattern from the
   reference data above
3. Perform interpolation on the raw Bradley-Terry ratings (before 0-100
   normalization) since the scaling relationship is in rating space
4. Add interpolated scores to `arena_leaderboard.csv` with an `interpolated`
   column (`true`/`false`) for transparency

**Estimated interpolations:**

| Missing Model | Reference Model | Approach |
|--------------|----------------|----------|
| `qwen2.5-7b-instruct` | `qwen2.5-72b-instruct` (1303) | Scale down using Llama 3.1 8B/70B ratio (~0.937) |
| `qwen3-8b` | `qwen3-32b` (1347) | Scale down using Gemma 2 9B/27B and Llama 3.1 ratios |
| `smollm3-3b` | `smollm2-1.7b-instruct` (1113) | Less reliable (different generation); use as floor and scale up slightly for larger param count |

**Implementation:** Add interpolation logic to
`scripts/fetch_arena_leaderboard.py`. The script reads the model catalog,
identifies unmatched base models, and applies family-based interpolation.
Interpolated entries are preserved across automated refreshes unless the model
appears in a future Arena update (at which point the real score replaces the
estimate).

### A5. Handle quantized derivatives

Quantized variants (fp8, w4a16, w8a8) inherit their base model's Arena score
with a **quantization discount** applied. Quantization degrades model quality —
the discount reflects this tradeoff.

**Quantization discounts:**

| Quantization | Discount | Rationale |
|---|---|---|
| FP8 (`-fp8`, `-fp8-dynamic`) | 1% | Nearly lossless; minimal quality degradation |
| W8A8 (`-quantized.w8a8`) | 2% | Minor degradation from symmetric INT8 quantization |
| W4A16 (`-quantized.w4a16`) | 5% | Noticeable quality loss from aggressive 4-bit weight quantization |

**Example:** If `llama-3.3-70b-instruct` has an Arena quality score of 65.0:

- `llama-3.3-70b-instruct-fp8-dynamic` → 65.0 × 0.99 = 64.4
- `llama-3.3-70b-instruct-quantized.w8a8` → 65.0 × 0.98 = 63.7
- `llama-3.3-70b-instruct-quantized.w4a16` → 65.0 × 0.95 = 61.8

These are conservative estimates based on published research. The exact
degradation variesmark by model and task, but even rough discounts are more accurate
than treating all variants as identical.

**Configuration file:** `data/configuration/quantization_discounts.json`

```json
{
  "_description": "Quality score discounts for quantized model variants. The base model's quality score is multiplied by (1 - discount) to account for quantization-induced quality degradation.",
  "_rationale": {
    "fp8": "FP8 quantization is nearly lossless with minimal quality degradation.",
    "w8a8": "Symmetric INT8 quantization causes minor quality degradation.",
    "w4a16": "Aggressive 4-bit weight quantization causes noticeable quality loss."
  },
  "discounts": {
    "fp8": 0.01,
    "w8a8": 0.02,
    "w4a16": 0.05
  },
  "suffix_patterns": {
    "fp8": ["-fp8", "-fp8-dynamic"],
    "w8a8": ["-quantized.w8a8"],
    "w4a16": ["-quantized.w4a16"]
  }
}
```

Discounts can be tuned by editing this file without code changes, consistent
with how `priority_weights.json` and other configuration is managed.

**Name normalization** must handle these patterns:

- `redhatai/llama-3.3-70b-instruct-fp8-dynamic` → `llama-3.3-70b-instruct` + 1%
  discount
- `redhatai/qwen2.5-7b-instruct-quantized.w4a16` → `qwen2.5-7b-instruct` + 5%
  discount
- `redhatai/meta-llama-3.1-8b-instruct-fp8-dynamic` → `llama-3.1-8b-instruct` +
  1% discount (strip `meta-` prefix too)
- `nvidia/llama-3.1-nemotron-70b-instruct-hf` →
  `llama-3.1-nemotron-70b-instruct` (no discount — not quantized, just a naming
  variant)

**Implementation:** The discount is applied in `get_quality_score()` after
resolving the base model's score. The quantization type is detected from the
model name suffix during normalization.

This gives all 47 catalog models a quality score — 44 from direct Arena data, 3
from interpolation, with quantized variants receiving their base model's score
minus the appropriate discount.

### A6. Commit the generated CSV

---

## Phase B: Core Engine Update

### B1. Rewrite `scripts/recalculate_weighted_scores.py`

Use-case weights are externalized to
`data/configuration/usecase_quality_weights.json`. Weights use integer values
(0–10) expressing relative importance, consistent with `priority_weights.json`.
The code normalizes them to percentages at runtime (`weight / sum_of_weights`),
so they don't need to sum to any fixed value — adjusting one weight doesn't
require rebalancing the others.

```json
{
  "_description": "Mapping from use cases to Arena category weights for quality scoring. Each use case defines which Arena categories contribute to its quality score and their relative importance (0-10 scale). Weights are normalized to percentages at runtime.",
  "_note": "These weights are initial estimates based on category relevance. They should be validated by comparing resulting rankings against known model quality for each use case.",
  "weights": {
    "chatbot_conversational": {
      "overall": 6, "multi_turn": 5,
      "instruction_following": 5, "creative_writing": 4
    },
    "code_completion": {
      "coding": 10, "industry_software_and_it_services": 5,
      "hard_prompts": 3, "instruction_following": 2
    },
    "code_generation_detailed": {
      "coding": 9, "hard_prompts": 4,
      "industry_software_and_it_services": 4, "math": 3
    },
    "translation": {
      "overall": 6, "instruction_following": 6,
      "industry_writing_and_literature_and_language": 5,
      "longer_query": 3
    },
    "content_generation": {
      "creative_writing": 7, "instruction_following": 5,
      "overall": 4, "longer_query": 4
    },
    "summarization_short": {
      "instruction_following": 7, "overall": 5,
      "hard_prompts": 4, "longer_query": 4
    },
    "document_analysis_rag": {
      "hard_prompts": 5, "expert": 5,
      "longer_query": 5, "overall": 5
    },
    "long_document_summarization": {
      "longer_query": 7, "instruction_following": 5,
      "overall": 4, "expert": 4
    },
    "research_legal_analysis": {
      "expert": 6, "industry_legal_and_government": 5,
      "hard_prompts": 5, "math": 4
    }
  }
}
```

The script reads this config file at runtime — weights can be tuned without code
changes, consistent with `priority_weights.json` and other configuration. These
serve as **default values** — see "User-customizable quality weights" in the
Future Enhancements section for the longer-term vision.

- Read normalized scores as raw floats
- Output weighted_scores CSVs with format: `model_name,weighted_score` (float,
  0-100)

### B2. Regenerate weighted score CSVs

- Run updated script to produce 9 new CSVs in
  `data/benchmarks/quality/weighted_scores/`

### B3. Rewrite `src/planner/recommendation/quality/usecase_scorer.py`

**Remove:**

- `BENCHMARK_TO_AA_MAP` dict (40 entries) -- no longer needed
- Fuzzy partial word matching logic -- replaced by simpler name matching

**Simplify `_load_csv_scores()`:**

- Read `model_name` as primary key (Arena short names)
- Read `weighted_score` as raw float
- Index by both full name and normalized variants

**Simplify `get_quality_score()` lookup chain:**

1. Exact match on Arena model name (lowercased)
2. Match on base model name (strip org prefix + quantization suffixes from
   catalog ID to get Arena-style name)
3. Dynamic family interpolation (see below)
4. Catalog fallback (unchanged)
5. Return 0.0

**Dynamic family interpolation (step 3):** Users can enter any HuggingFace
model, not just catalog models. When a model has no direct Arena match, attempt
runtime interpolation:

1. Parse the model family and parameter count from the model ID (e.g.,
   `meta-llama/Llama-3.1-13B-Instruct` → family `llama-3.1`, params `13B`)
2. Find Arena models in the same family (e.g., `llama-3.1-8b-instruct` at 1211,
   `llama-3.1-70b-instruct` at 1293)
3. Interpolate using the parameter count and the empirical scaling ratios from
   Phase A4
4. Log that the score is an estimate so it's visible in debugging

This requires the full `arena_leaderboard.csv` (with raw ratings and model
names) to be loaded at startup, not just the weighted score CSVs. The scorer
should index Arena models by family for efficient lookup.

**Update docstrings:** Replace "Artificial Analysis" with "Chatbot Arena
Leaderboard"

### B4. Rename "accuracy" to "quality" throughout

Arena human preference ratings measure overall quality, not just factual
accuracy. Rename to reflect this.

**Schema changes (`src/planner/shared/schemas/recommendation.py`):**

- `accuracy_score` → `quality_score`

**API parameter changes:**

- `min_accuracy` → `min_quality` (query parameter in recommendation endpoints)
- `best_accuracy` → `best_quality` (ranked list key)

**Scoring weight key (`src/planner/recommendation/scorer.py`):**

- `DEFAULT_WEIGHTS["accuracy"]` → `DEFAULT_WEIGHTS["quality"]`

**UI — Technical Specification tab (`ui/components/slo.py`):**

The current UI has a hardcoded `TASK_DATASETS` dict (lines 19-65) that displays
AA benchmark names and weights for each use case, with the attribution "Weights
from Artificial Analysis Intelligence Index." This must be replaced:

- Remove the hardcoded `TASK_DATASETS` dict
- Add a backend API call (or extend the existing reference data endpoint) to
  serve the use-case quality weights from
  `data/configuration/usecase_quality_weights.json`
- Rename the section header from "Accuracy Benchmarks" to "Quality Scores"
- Display Arena category names with their derived percentages (computed from
  integer weights at render time: `weight / sum_of_weights × 100`)
- Update attribution to "Arena category weights (Chatbot Arena, CC-BY-4.0)"

This eliminates the duplication between UI and backend — both now read from the
same externalized config file, so weights can't drift apart.

**Directory rename:**

- `data/benchmarks/accuracy/` → `data/benchmarks/quality/`
- Update all path references in code, scripts, Makefile, and GitHub Action

**Files affected (~190 occurrences):**

| Location | Files | Occurrences |
|----------|-------|-------------|
| `src/planner/` | 12 files | ~111 |
| `ui/` | 8 files | ~66 |
| `tests/` | 5 files | ~13 |

**Approach:** Use project-wide find-and-replace with manual review. Key renames:

- `accuracy_score` → `quality_score`
- `min_accuracy` → `min_quality`
- `best_accuracy` → `best_quality`
- `"accuracy"` (as dict key) → `"quality"`
- `data/benchmarks/accuracy` → `data/benchmarks/quality`
- "Accuracy" in user-facing strings → "Quality"

### B5. Remove task bonuses from `src/planner/recommendation/analyzer.py`

The `TASK_BONUSES` dict and `get_task_bonus()` function apply hardcoded
keyword-based bonuses to the balanced score (e.g., `"coder": 20` for
`code_completion`). Use-case differentiation is already captured in the weighted
quality scores — each use case has different category weights, so
task-appropriate models naturally score higher. The task bonuses double-count
this effect and add fragile model-name keyword matching that must be manually
maintained.

**Remove:**

- `TASK_BONUSES` dict (lines 26-114)
- `get_task_bonus()` function (lines 117-139)

**Simplify `_recalculate_balanced_scores()`:**

- Remove task bonus logic
- Balanced score formula becomes: `Quality × 70% + (Latency + Cost) / 2 × 30%`
  (no `+ Task_Bonus`)

**Update tests:** Remove any tests that assert specific task bonus values

### B6. Update tests

- `tests/unit/test_usecase_scorer_fallback.py`: Update AA model names to Arena
  names
- Add `tests/unit/test_usecase_scorer_arena.py`:
  - Known Arena models get nonzero scores
  - Catalog model IDs (with org prefix) match Arena names
  - Quantized variants match base model scores
  - Unknown models return 0.0
  - Catalog fallback still works
- Update any tests asserting task bonus values
- Update all `accuracy_score` → `quality_score` references in tests

### B7. Verify

```bash
cd src && uv run pytest ../tests/unit/ -v
make lint && make format && make typecheck
```

---

## Phase C: Cleanup

### C1. Delete all Artificial Analysis data and related scripts

All AA-sourced data must be removed from the repository (old paths, before
directory rename):

- `data/benchmarks/accuracy/opensource_all_benchmarks.csv`
- `data/benchmarks/accuracy/opensource_all_benchmarks_interpolated.csv`
- `data/benchmarks/accuracy/weighted_scores/opensource_chatbot_conversational.csv`
- `data/benchmarks/accuracy/weighted_scores/opensource_code_completion.csv`
- `data/benchmarks/accuracy/weighted_scores/opensource_code_generation_detailed.csv`
- `data/benchmarks/accuracy/weighted_scores/opensource_content_generation.csv`
- `data/benchmarks/accuracy/weighted_scores/opensource_document_analysis_rag.csv`
- `data/benchmarks/accuracy/weighted_scores/opensource_long_document_summarization.csv`
- `data/benchmarks/accuracy/weighted_scores/opensource_research_legal_analysis.csv`
- `data/benchmarks/accuracy/weighted_scores/opensource_summarization_short.csv`
- `data/benchmarks/accuracy/weighted_scores/opensource_translation.csv`
- `scripts/interpolate_benchmark_scores_robust.py`
- `scripts/interpolate_benchmark_scores.py` (if exists)

### C2. Update all files referencing "Artificial Analysis"

The following files contain AA references and must be updated:

**Code (rewritten in Phase B):**

- `src/planner/recommendation/quality/usecase_scorer.py`
- `scripts/recalculate_weighted_scores.py`

**Code (docstring/comment updates only):**

- `src/planner/recommendation/scorer.py`
- `src/planner/recommendation/config_finder.py`
- `src/planner/api/routes/reference_data.py`
- `ui/components/slo.py`

**Docs (see Phase D):**

- `CLAUDE.md`
- `docs/ARCHITECTURE.md`
- `docs/USE_CASE_METHODOLOGY.md`
- `docs/Recommendation Flow.md`

**Archive (low priority, optional):**

- `data/archive/all_usecases_config.json`
- `data/archive/usecase_weights.json`
- `data/archive/slo_ranges.json`
- `data/archive/redhat_models_benchmarks.csv`
- `data/archive/interpolation_report.json`

---

## Phase D: Documentation and Attribution

### D1. CC-BY-4.0 attribution (required by license)

The Arena dataset is licensed CC-BY-4.0, which requires attribution. Add credit
in:

- **`NOTICE` or `LICENSE` file**: Add a section noting "Model quality scores
  from Chatbot Arena (lmarena-ai/leaderboard-dataset), licensed CC-BY-4.0"
- **`data/benchmarks/quality/README.md`**: New file documenting the data source,
  license, and how to refresh the data
- **`docs/USE_CASE_METHODOLOGY.md`**: Reference the data source with attribution

UI attribution is not legally required — repo-level attribution is sufficient.

### D2. Documentation updates

- `docs/USE_CASE_METHODOLOGY.md`: Full rewrite — new data source, new category
  mappings, new weights
- `CLAUDE.md`: Replace "Artificial Analysis" references throughout
- `docs/ARCHITECTURE.md`: Update data source references
- `docs/Recommendation Flow.md`: Update data source references

---

## Phase E: Automated Data Refresh

### E1. Create `.github/workflows/refresh-quality-scores.yml`

Periodic GitHub Action to fetch the latest Arena data and open a PR if scores
have changed:

```yaml
name: Refresh Arena Scores

on:
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday at 06:00 UTC
  workflow_dispatch:       # Allow manual trigger

jobs:
  refresh:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
    steps:
      - uses: actions/checkout@v6
      - uses: astral-sh/setup-uv@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - run: uv sync --extra dev

      - name: Refresh Arena data
        run: ./scripts/refresh_arena_scores.sh

      - name: Check for changes
        id: changes
        run: |
          if git diff --quiet data/benchmarks/quality/; then
            echo "changed=false" >> "$GITHUB_OUTPUT"
          else
            echo "changed=true" >> "$GITHUB_OUTPUT"
          fi

      - name: Create PR
        if: steps.changes.outputs.changed == 'true'
        uses: peter-evans/create-pull-request@v7
        with:
          branch: chore/refresh-quality-scores-scores
          title: "chore: refresh Arena leaderboard scores"
          body: |
            Automated weekly refresh of Chatbot Arena leaderboard data.

            Source: [lmarena-ai/leaderboard-dataset](https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset) (CC-BY-4.0)

            Review the diff to verify score changes are reasonable.
          commit-message: "chore: refresh Arena leaderboard scores"
          delete-branch: true
```

**Design decisions:**

- **Weekly cadence**: Arena updates daily, but weekly is sufficient for our use
  case and avoids PR noise
- **PR-based**: Changes go through review rather than auto-merging, so
  maintainers can verify score shifts are reasonable
- **Manual trigger**: `workflow_dispatch` allows on-demand refresh when new
  models are added to the catalog
- **Idempotent**: If scores haven't changed, no PR is created

---

## Critical Files

| File | Change |
|------|--------|
| `scripts/fetch_arena_leaderboard.py` | NEW |
| `scripts/refresh_arena_scores.sh` | NEW |
| `scripts/recalculate_weighted_scores.py` | REWRITE |
| `.github/workflows/refresh-quality-scores.yml` | NEW |
| `Makefile` | ADD `refresh-quality-scores` target |
| `data/configuration/usecase_quality_weights.json` | NEW |
| `data/configuration/quantization_discounts.json` | NEW |
| `data/benchmarks/quality/arena_leaderboard.csv` | NEW |
| `data/benchmarks/quality/weighted_scores/*.csv` | REGENERATE |
| `src/planner/recommendation/quality/usecase_scorer.py` | MAJOR REWRITE |
| `src/planner/recommendation/analyzer.py` | REMOVE task bonuses |
| `src/planner/recommendation/scorer.py` | DOCSTRING ONLY |
| `src/planner/recommendation/config_finder.py` | COMMENTS ONLY |
| `src/planner/api/routes/reference_data.py` | MODERATE |
| `ui/components/slo.py` | MODERATE (replace hardcoded benchmarks with API-driven quality weights) |
| `tests/unit/test_usecase_scorer_fallback.py` | UPDATE |
| `tests/unit/test_usecase_scorer_arena.py` | NEW |
| `docs/USE_CASE_METHODOLOGY.md` | FULL REWRITE |

## Verification

1. `cd src && uv run pytest ../tests/unit/ -v` -- all unit tests pass
2. `make lint && make format && make typecheck` -- clean
3. Manual check: run `make refresh-quality-scores` and verify top 5 models per
   use case look reasonable
4. Coverage check: all 47 catalog models get a nonzero quality score (44 direct
   - 3 interpolated)
5. Verify interpolated scores are marked in `arena_leaderboard.csv` and fall
   within reasonable ranges
6. Verify quantized variants resolve to their base model's score
7. Compare old vs new scores for a few known models to sanity-check rankings

## Alternatives Considered

| Source | Why Not |
|--------|---------|
| HF Open LLM Leaderboard v2 | Retired March 2025; no license on results dataset |
| LiveBench | ~50 models (too few); no license on HF datasets |
| SEAL (Scale) | Proprietary; no data export |
| OpenCompass | Framework, not pre-computed data; requires running evaluations |
| Artificial Analysis | Personal use license only (current problem) |

---

## Future Enhancements

The following ideas should be researched and evaluated once the Arena-based
system is operational. They are not in scope for the initial replacement.

### Multi-source quality scoring pipeline

Arena's human preference ratings provide a strong quality signal, but they have
limitations: coverage gaps for smaller/newer models, and preference votes may
not fully capture domain-specific accuracy (e.g., factual correctness in legal
analysis, mathematical proof validity). Several additional data sources and
approaches should be researched and evaluated:

**Additional data sources to evaluate:**

- **Every Eval Ever (EEE)**
  ([evaleval/EEE_datastore](https://huggingface.co/datasets/evaleval/EEE_datastore),
  MIT licensed) — a crowdsourced aggregation of 37 benchmarks including
  LiveCodeBench, BFCL, MMLU-lite, HELM variants, and HF Open LLM Leaderboard v2.
  Uses HuggingFace model IDs natively (no name matching needed). Since EEE
  aggregates multiple sources, we would need to pick and choose which
  constituent benchmarks make sense for our use cases and deduplicate any Arena
  data that may be included.
- **lm-evaluation-harness** (MIT licensed) — run standardized benchmarks
  ourselves for models not covered by other sources. Particularly useful for
  catalog models without Arena data.

**Domain-specific benchmarks to research:**

| Benchmark | Use Cases | Why |
|---|---|---|
| IFEval (instruction following) | chatbot, summarization, content_generation | Stronger signal than Arena's `instruction_following` category for strict adherence |
| LongBench (long context) | long_document_summarization, document_analysis_rag | Directly measures long-context capability vs Arena's `longer_query` proxy |
| LegalBench (legal reasoning) | research_legal_analysis | Purpose-built for legal domain vs Arena's generic `industry_legal_and_government` |
| LiveCodeBench (coding) | code_completion, code_generation | Direct coding quality measurement to complement Arena's `coding` preference votes |
| BFCL (function calling) | code_completion, document_analysis_rag | Measures structured output and tool use capability |
| MMLU-lite (general knowledge) | translation, research_legal_analysis | Domain knowledge coverage |

These are not all currently in EEE (IFEval, LongBench, LegalBench are missing)
but are all available via `lm-evaluation-harness`.

**Architecture considerations:**

Incorporating multiple sources would require a data curation pipeline beyond the
current single-source fetch script: fetching from multiple sources, normalizing
different scales (Bradley-Terry ratings vs benchmark percentages vs raw scores),
deduplicating overlapping data, merging into unified per-model quality profiles,
and applying use-case weights across the combined scores. This is a significant
piece of work that should be designed once we have operational experience with
the Arena-based system and can identify where preference-based scoring falls
short.

A hybrid scoring approach — combining Arena preference ratings with automated
benchmark scores, weighted by use case (e.g., preference-heavy for creative
writing, benchmark-heavy for math/code) — is the likely end state.

### User-customizable quality weights

The use-case quality weights in `usecase_quality_weights.json` are defaults.
Users often have domain-specific knowledge about what matters most for their
particular deployment — e.g., a chatbot that's heavily coding-focused might want
to add or increase the `coding` weight for `chatbot_conversational`.

Since weights use a 0–10 relative scale (normalized to percentages at runtime),
customization is straightforward — users adjust individual weights without
needing to rebalance the others. A future enhancement would allow users to
customize the Arena category weights during the specification phase, either by:

- Adjusting the defaults for their chosen use case (e.g., "increase coding
  importance" by bumping `coding` from 0 to 7)
- Selecting from the 27 Arena categories and assigning their own weights
- Defining an entirely custom use case with a custom weight profile

This would compute quality scores dynamically at recommendation time rather than
relying on pre-computed weighted score CSVs. The architecture already supports
this — the Arena category ratings are stored per-model in
`arena_leaderboard.csv`, so computing a custom weighted score is a simple dot
product at query time.

### Moving quality data to PostgreSQL

The initial implementation loads quality scores from CSV files into in-memory
dicts at startup. This is simple and fast for the current dataset size (~342
models × 27 categories). However, if quality data grows or we want tighter
integration with performance benchmarks, moving to PostgreSQL should be
considered:

- **Unified data layer** — all benchmark data (performance + quality) in one
  place with consistent access patterns
- **SQL joins** — enable queries that combine quality and performance data
  (e.g., "models where quality > X AND latency < Y on this GPU")
- **Runtime updates** — the refresh workflow could update the DB directly
  instead of committing CSV files
- **Scalability** — better suited if the number of models or categories grows
  significantly

The tradeoff is added complexity: PostgreSQL becomes a hard dependency for
quality scoring, unit tests would need DB mocking, and CSV version history
(git-tracked score changes over time) would be lost. For now, CSV-in-memory is
the right choice given the small dataset size and the value of keeping unit
tests DB-free.
