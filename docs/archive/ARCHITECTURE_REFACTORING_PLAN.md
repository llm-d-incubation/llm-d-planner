# Architecture Refactoring Plan

**Date:** November 4, 2025
**Status:** ✅ Completed
**Objective:** Refactor architecture documentation and code to reflect refined 4-engine + UI Layer structure

---

## Context

The NeuralNav architecture is being reorganized from 8 components to a clearer 4-engine + UI Layer structure. This improves:
- **Clarity:** UI is horizontal (sits on top), engines are vertical domain services
- **Modularity:** Each engine has clear responsibilities and boundaries
- **Alignment:** Code organization matches conceptual architecture
- **Vision:** Shows full future capabilities, not just current phase

---

## Architecture Changes

### Before (8 Components)
1. Conversational Interface Layer
2. Context & Intent Engine
3. Recommendation Engine (with 3 sub-components)
4. Deployment Automation Engine
5. Knowledge Base & Benchmarking Data Store
6. LLM Backend
7. Orchestration & Workflow Engine
8. Inference Observability

### After (UI Layer + 4 Engines + Data Layer)

```
┌─────────────────────────────────────────────────┐
│           UI Layer (Presentation)               │
│   Streamlit → React (future)                    │
│   Chat | Spec Editor | Visualizer | Dashboard   │
└─────────────────────────────────────────────────┘
                 ↕ API Gateway
┌─────────────────────────────────────────────────┐
│              Core Engines                       │
├─────────────────────────────────────────────────┤
│ 1. Intent & Specification Engine                │
│    • Conversation & intent extraction           │
│    • Traffic profile determination              │
│    • Specification generation                   │
│    Code: context_intent/, llm/                  │
│                                                 │
│ 2. Recommendation Engine                        │
│    • Benchmark query & filtering                │
│    • SLO compliance checking                    │
│    • Ranking by priorities                      │
│    Code: recommendation/                        │
│                                                 │
│ 3. Deployment Engine                            │
│    • YAML generation                            │
│    • K8s lifecycle management                   │
│    Code: deployment/                            │
│                                                 │
│ 4. Observability Engine                         │
│    • Monitoring & testing                       │
│    • Metrics collection                         │
│    • Feedback loop (future)                     │
│    Code: observability/ (future module)         │
└─────────────────────────────────────────────────┘
                 ↕ Data Access
┌─────────────────────────────────────────────────┐
│         Knowledge Base (Data Layer)             │
│  PostgreSQL: Benchmarks, Outcomes               │
│  JSON: SLO Templates, Catalog, Hardware         │
│  Code: knowledge_base/                          │
└─────────────────────────────────────────────────┘
```

### Infrastructure (Not Numbered Components)
- **API Gateway**: FastAPI routes in `api/` - coordinates service calls
- **Orchestration**: Workflow state management (Streamlit session → persistent store future)

---

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| **Use "Engine" terminology** | Matches current docs, implies functional modules (not microservices) |
| **UI as separate horizontal layer** | UI touches all engines; better separation of concerns |
| **"Intent & Specification Engine"** | Clearer than "Context & Intent"; emphasizes transformation |
| **Traffic profile in Intent Engine** | Part of specification output, not recommendation logic |
| **Knowledge Base as data layer** | Shared infrastructure, not domain component |
| **Show full future vision** | No phase markers; document complete architecture |
| **Hybrid storage in Knowledge Base** | PostgreSQL for benchmarks (query performance), JSON for config (version control) |

---

## Tasks

### Phase 1: Plan Document ✅
- [x] Create `docs/ARCHITECTURE_REFACTORING_PLAN.md`

### Phase 2: Documentation Updates

#### Task 2.1: Update `docs/ARCHITECTURE.md`
**Changes:**
- Lines 65-127: Rewrite "Architecture Components" section
  - Remove 8-component numbered list
  - Add UI Layer description
  - Add 4 engine descriptions
  - Document API Gateway as infrastructure
  - Document Knowledge Base as data layer

- Throughout document:
  - Rename "Context & Intent Engine" → "Intent & Specification Engine"
  - Move traffic profile generation from Recommendation to Intent & Specification Engine
  - Update all component cross-references

- Lines 546-844: Update Knowledge Base section
  - Clarify hybrid storage: PostgreSQL for benchmarks, JSON for config
  - Document external integrations within relevant engines

- Lines 972-1050: Update data flow diagram
  - Show UI Layer at top
  - Show feedback loop: Observability → Knowledge Base → Intent & Specification

- Lines 1054-1066: Update technology choices table
  - Remove phase columns
  - Show future vision only
  - Use "Engine" terminology

**Status:** ✅ Completed

#### Task 2.2: Update `docs/architecture-diagram.md`
**Changes:**
- Main component diagram:
  - Show UI Layer (horizontal bar)
  - Show 4 engines (vertical columns)
  - Show Knowledge Base (data foundation)
  - Show API Gateway connections
  - Show feedback loop arrows

- Sequence diagrams:
  - Move traffic profile generation to Intent & Specification Engine step
  - Remove from Recommendation Engine

- Component breakdown sections:
  - Reorganize under 4 engines + UI Layer
  - Move Traffic Profile Generator description to Intent & Specification Engine

**Status:** ✅ Completed

#### Task 2.3: Update `docs/simplified-architecture-diagram.md`
**Changes:**
- Update all 4 diagram options:
  - Use "Intent & Specification Engine" naming
  - Show PostgreSQL + JSON hybrid storage
  - Include feedback loop (future vision)

**Status:** ✅ Completed

#### Task 2.4: Update `CLAUDE.md`
**Changes:**
- Project overview section:
  - Update component count (4 engines + UI Layer)
  - Update component names
  - Update key concepts to reflect new structure

**Status:** ✅ Completed

### Phase 3: Code Refactoring

#### Task 3.1: Move `traffic_profile.py` to Intent & Specification Engine
**Current location:** `backend/src/recommendation/traffic_profile.py`
**New location:** `backend/src/context_intent/traffic_profile.py`

**Rationale:** Traffic profile is part of the specification that Intent & Specification Engine produces. The Recommendation Engine should consume the spec (including traffic profile), not generate it.

**Files to move:**
```bash
mv backend/src/recommendation/traffic_profile.py \
   backend/src/context_intent/traffic_profile.py
```

**Status:** ✅ Completed

#### Task 3.2: Update imports in `backend/src/orchestration/workflow.py`
**Change:**
```python
# Before
from recommendation.traffic_profile import TrafficProfileGenerator

# After
from context_intent.traffic_profile import TrafficProfileGenerator
```

**Status:** ✅ Completed

#### Task 3.3: Update imports in `backend/src/api/routes.py`
**Change:**
```python
# Before
from recommendation.traffic_profile import TrafficProfileGenerator

# After
from context_intent.traffic_profile import TrafficProfileGenerator
```

**Status:** ✅ Completed (no test imports to update)

#### Task 3.4: Update test imports (if any)
**Search for:**
```bash
grep -r "from recommendation.traffic_profile" backend/tests/
```

**Update:** Any test files that import TrafficProfileGenerator

**Status:** Pending

#### Task 3.5: Run tests to verify refactoring
**Commands:**
```bash
cd backend
make test
```

**Expected:** All tests pass, no import errors

**Status:** ✅ Completed - All tests passing (24/24)

---

## Progress Tracking

### Completed ✅
- Plan document created
- All documentation updates completed
  - ARCHITECTURE.md updated with new 4-engine + UI Layer structure
  - architecture-diagram.md updated with new Mermaid diagrams
  - simplified-architecture-diagram.md updated with new component names
  - CLAUDE.md updated with architecture overview and terminology
- All code refactoring completed
  - traffic_profile.py moved from recommendation/ to context_intent/
  - Imports updated in workflow.py and routes.py
  - All tests passing (24/24)

---

## Rollback Plan

If issues arise:

1. **Documentation rollback:**
   ```bash
   git checkout HEAD -- docs/ARCHITECTURE.md
   git checkout HEAD -- docs/architecture-diagram.md
   git checkout HEAD -- docs/simplified-architecture-diagram.md
   git checkout HEAD -- CLAUDE.md
   ```

2. **Code rollback:**
   ```bash
   git checkout HEAD -- backend/src/context_intent/traffic_profile.py
   git checkout HEAD -- backend/src/recommendation/traffic_profile.py
   git checkout HEAD -- backend/src/orchestration/workflow.py
   git checkout HEAD -- backend/src/api/routes.py
   ```

---

## Success Criteria

- [x] All documentation reflects new 4-engine + UI Layer structure
- [x] All diagrams show correct component relationships
- [x] Traffic profile generation is in Intent & Specification Engine
- [x] All imports updated correctly
- [x] All tests pass
- [x] Code organization matches documented architecture
- [x] No breaking changes to functionality

---

## Notes

- **Terminology:** Using "Engine" (not "Service") throughout for consistency
- **Storage:** Knowledge Base uses hybrid approach (PostgreSQL + JSON), not all PostgreSQL
- **Vision:** Documentation shows complete future architecture, not just current phase
- **External APIs:** Documented within relevant engines (not separate component)

---

## Contact

For questions about this refactoring, see discussion in session dated November 4, 2025.
