# NeuralNav Refactoring Plan: Align with ARCHITECTUREv2.md

## Summary

Refactor the backend codebase to align with the 5 major components defined in ARCHITECTUREv2.md. This is an incremental refactoring done in testable steps. Since the project is early-stage with few users, we do NOT need backward compatibility after completion - old modules will be deleted after migration.

---

## Current vs Target Structure

### Current Structure

```text
backend/src/
├── api/routes.py              # Monolithic 1191 lines
├── context_intent/            # MIXED: Intent + Specification
│   ├── extractor.py           # Intent extraction
│   ├── schema.py              # ALL Pydantic models (238 lines)
│   ├── traffic_profile.py     # Specification logic
│   └── gpu_normalizer.py
├── recommendation/            # Recommendation Service
│   ├── capacity_planner.py    # Config Finder
│   ├── solution_scorer.py     # Scorer
│   ├── ranking_service.py     # Analyzer
│   └── usecase_quality_scorer.py
├── deployment/                # MIXED: Configuration + K8s automation
│   ├── generator.py           # YAML generation
│   ├── cluster.py             # K8s deployment
│   └── templates/
├── knowledge_base/            # Data layer (correct)
├── llm/                       # LLM client (correct)
└── orchestration/workflow.py  # Workflow coordination
```

### Target Structure (aligned with ARCHITECTUREv2.md)

```text
backend/src/
├── api/                       # API Gateway
│   ├── app.py                 # FastAPI app factory
│   └── routes/                # Modular endpoints
│       ├── health.py
│       ├── intent.py          # /api/v1/extract
│       ├── specification.py   # SLO/workload endpoints
│       ├── recommendation.py  # /api/v1/recommend, /api/v1/ranked-recommend-from-spec
│       ├── configuration.py   # /api/v1/deploy, /api/v1/deployments/*
│       └── reference_data.py  # /api/v1/models, use-cases, etc.
├── intent_extraction/         # 1. Intent Extraction Service
│   ├── service.py             # Service facade
│   └── extractor.py
├── specification/             # 2. Specification Service
│   ├── service.py             # Service facade
│   ├── traffic_profile.py
│   └── slo_mapper.py
├── recommendation/            # 3. Recommendation Service (reorganized)
│   ├── service.py             # Service facade
│   ├── config_finder.py       # Was capacity_planner.py
│   ├── scorer.py              # Was solution_scorer.py
│   ├── analyzer.py            # Was ranking_service.py
│   └── quality/usecase_scorer.py
├── configuration/             # 4. Configuration Service
│   ├── service.py             # Service facade
│   ├── generator.py           # YAML generation
│   ├── validator.py
│   └── templates/             # Jinja2 templates
├── cluster/                   # K8s automation (NOT a core service)
│   └── manager.py
├── knowledge_base/            # Infrastructure: Data Layer (unchanged)
├── llm/                       # Infrastructure: LLM client (unchanged)
└── shared/                    # Shared schemas
    └── schemas/
        ├── intent.py          # DeploymentIntent, ConversationMessage
        ├── specification.py   # TrafficProfile, SLOTargets, DeploymentSpecification
        └── recommendation.py  # DeploymentRecommendation, GPUConfig, etc.
```

---

## Implementation Phases

Each step is independently testable. No backward compatibility needed after completion.

### Phase 1: Shared Schemas

**Step 1.1: Create shared schemas module**

- Create `backend/src/shared/schemas/` directory
- Split `context_intent/schema.py` into:
  - `shared/schemas/intent.py` - DeploymentIntent, ConversationMessage
  - `shared/schemas/specification.py` - TrafficProfile, SLOTargets, DeploymentSpecification
  - `shared/schemas/recommendation.py` - DeploymentRecommendation, GPUConfig, ConfigurationScores, RankedRecommendationsResponse
- Create `shared/schemas/__init__.py` with all exports
- Update all imports across codebase to use new location
- Delete `context_intent/schema.py`

**Test:** Run API server, verify all endpoints work

### Phase 2: Split API Routes

**Step 2.1: Create modular route files**

- Create `backend/src/api/routes/` directory
- Create route modules:
  - `health.py` - /health
  - `intent.py` - /api/v1/extract
  - `specification.py` - /api/v1/slo-defaults, /api/v1/workload-profile, /api/v1/expected-rps
  - `recommendation.py` - /api/v1/recommend, /api/v1/ranked-recommend-from-spec
  - `configuration.py` - /api/v1/deploy, /api/v1/deploy-to-cluster, /api/v1/deployments/*
  - `reference_data.py` - /api/v1/models, /api/v1/gpu-types, /api/v1/use-cases, etc.
- Create `api/app.py` as FastAPI app factory that assembles routers
- Delete old `routes.py`

**Test:** Run API server, test each endpoint category

### Phase 3: Intent Extraction Service

**Step 3.1: Create intent_extraction module**

- Create `backend/src/intent_extraction/` directory
- Move `context_intent/extractor.py` to `intent_extraction/extractor.py`
- Move `context_intent/gpu_normalizer.py` to `shared/utils/gpu_normalizer.py`
- Create `intent_extraction/service.py` with IntentExtractionService facade
- Update imports in routes and workflow
- Delete `context_intent/extractor.py`

**Test:** POST to /api/v1/extract, verify intent extraction works

### Phase 4: Specification Service

**Step 4.1: Create specification module**

- Create `backend/src/specification/` directory
- Move `context_intent/traffic_profile.py` to `specification/traffic_profile.py`
- Create `specification/service.py` with SpecificationService facade
- Update imports in routes and workflow
- Delete remaining `context_intent/` directory

**Test:** Verify specification generation in workflow

### Phase 5: Configuration Service

**Step 5.1: Create configuration module**

- Create `backend/src/configuration/` directory
- Move from `deployment/`:
  - `generator.py` -> `configuration/generator.py`
  - `validator.py` -> `configuration/validator.py`
  - `templates/` -> `configuration/templates/`
- Create `configuration/service.py` with ConfigurationService facade

**Step 5.2: Create cluster module**

- Create `backend/src/cluster/` directory
- Move `deployment/cluster.py` to `cluster/manager.py`
- Update imports
- Delete `deployment/` directory

**Test:** Generate YAML via /api/v1/deploy, verify cluster operations

### Phase 6: Recommendation Service Cleanup

**Step 6.1: Rename files to match ARCHITECTUREv2**

- Rename within `recommendation/`:
  - `capacity_planner.py` -> `config_finder.py` (rename class to ConfigFinder)
  - `solution_scorer.py` -> `scorer.py` (rename class to Scorer)
  - `ranking_service.py` -> `analyzer.py` (rename class to Analyzer)
- Move `usecase_quality_scorer.py` to `quality/usecase_scorer.py`
- Create `recommendation/service.py` with RecommendationService facade
- Delete `model_evaluator.py` (legacy, unused)

**Test:** Full recommendation flow via /api/v1/ranked-recommend-from-spec

### Phase 7: Finalize

**Step 7.1: Update workflow.py**

- Refactor `orchestration/workflow.py` to use new service facades
- Consider renaming to match API Gateway role

**Step 7.2: Update CLAUDE.md**

- Update repository structure documentation to reflect new layout

---

## Critical Files to Modify

| Current File | Action | New Location |
|--------------|--------|--------------|
| backend/src/api/routes.py | Split & delete | api/app.py + api/routes/*.py |
| backend/src/context_intent/schema.py | Split & delete | shared/schemas/*.py |
| backend/src/context_intent/extractor.py | Move & delete | intent_extraction/extractor.py |
| backend/src/context_intent/traffic_profile.py | Move & delete | specification/traffic_profile.py |
| backend/src/context_intent/gpu_normalizer.py | Move & delete | shared/utils/gpu_normalizer.py |
| backend/src/recommendation/capacity_planner.py | Rename | recommendation/config_finder.py |
| backend/src/recommendation/solution_scorer.py | Rename | recommendation/scorer.py |
| backend/src/recommendation/ranking_service.py | Rename | recommendation/analyzer.py |
| backend/src/recommendation/usecase_quality_scorer.py | Move | recommendation/quality/usecase_scorer.py |
| backend/src/recommendation/model_evaluator.py | Delete | (legacy, unused) |
| backend/src/deployment/generator.py | Move & delete | configuration/generator.py |
| backend/src/deployment/validator.py | Move & delete | configuration/validator.py |
| backend/src/deployment/templates/ | Move & delete | configuration/templates/ |
| backend/src/deployment/cluster.py | Move & delete | cluster/manager.py |
| backend/src/orchestration/workflow.py | Refactor | Use new service facades |

---

## Verification Strategy

After each phase:

1. Start API server:

   ```bash
   cd backend && uvicorn src.api.app:app --reload
   ```

2. Test endpoints:

   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/api/v1/models
   curl http://localhost:8000/api/v1/use-cases
   ```

3. Test Streamlit UI works end-to-end:

   ```bash
   cd ui && streamlit run app.py
   ```

4. Run any existing tests:

   ```bash
   cd backend && pytest
   ```

---

## Estimated Effort

- **Phase 1** (Schemas): Small - mechanical splitting
- **Phase 2** (Routes): Medium - 1191 lines to organize
- **Phase 3** (Intent): Small - simple move
- **Phase 4** (Specification): Small - simple move
- **Phase 5** (Configuration): Medium - move + cluster separation
- **Phase 6** (Recommendation): Medium - renames + class updates
- **Phase 7** (Finalize): Small - documentation updates
