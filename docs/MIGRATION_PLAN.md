# Plan: Rename NeuralNav → Planner & Integrate Config Explorer

## Context

The repository has moved from `redhat-et/neuralnav` to `llm-d-incubation/llm-d-planner`. We need to:
1. Rename all "NeuralNav"/"neuralnav" references to "Planner"/"planner"
2. Integrate the imported `config_explorer/` into the unified project structure
3. Add Config Explorer UI as separate pages accessible via a view toggle in the Streamlit UI

Config Explorer has 2 functional areas: **Capacity Planner** (memory/KV cache calculations) and **GPU Recommender** (BentoML roofline estimation + cost).

---

## 1. Proposed Directory Structure

Config Explorer modules are **flattened** as peers alongside existing modules (not nested under a `config_explorer/` sub-package). A shared `cli/` package holds all CLI entry points.

```
src/planner/                          # renamed from src/planner/
├── __init__.py
├── api/                              # (unchanged internal structure)
├── capacity_planner.py              # NEW: from config_explorer (memory/KV cache calculations)
├── cli/                             # NEW: CLI package (shared entry point)
│   ├── __init__.py
│   ├── __main__.py                  # python -m planner.cli
│   └── planner_cli.py               # plan + estimate subcommands (from config_explorer/cli.py)
├── cluster/
├── configuration/
├── gpu_recommender/                 # NEW: from config_explorer/recommender/
│   ├── __init__.py
│   ├── recommender.py              # GPURecommender (BentoML llm-optimizer)
│   └── cost_manager.py             # GPU cost management
├── intent_extraction/
├── knowledge_base/
├── llm/
├── orchestration/
├── recommendation/
├── shared/
└── specification/

ui/                                   # unified UI
├── app.py                            # updated: "Planner" branding, view toggle
├── api_client.py
├── state.py                          # merged session state
├── helpers.py
├── components/
│   ├── capacity_planner.py           # NEW: from config_explorer/Capacity_Planner.py
│   ├── gpu_recommender.py            # NEW: from config_explorer/pages/2_GPU_Recommender.py
│   ├── config_explorer_state.py      # NEW: from config_explorer/util.py (Scenario dataclass + session helpers)
│   ├── deployment.py
│   ├── deployment_management.py
│   ├── dialogs.py
│   ├── extraction.py
│   ├── recommendations.py
│   ├── settings.py
│   └── slo.py

data/
├── configuration/
│   ├── gpu_specs.json                # from config_explorer/db.json
│   ├── gpu_costs.json                # from config_explorer/gpu_costs.json
│   ├── model_catalog.json            # existing
│   ├── slo_templates.json            # existing
│   └── ...                           # other existing files
├── benchmarks/                       # existing (unchanged)
└── archive/                          # existing

tests/
├── test_capacity_planner.py          # NEW: from config_explorer/tests/capacity_planner_test.py
├── test_cli.py                       # NEW: from config_explorer/tests/test_cli.py
├── test_cost_integration.py          # NEW: from config_explorer/tests/test_cost_integration.py
├── test_cost_manager.py              # NEW: from config_explorer/tests/test_cost_manager.py
├── test_recommender_cost.py          # NEW: from config_explorer/tests/test_recommender_cost.py
├── conftest.py
├── test_postgresql_integration.py
├── test_recommendation_workflow.py
└── test_yaml_generation.py
```

---

## 2. Config Explorer Mapping Table

| Current Path | Purpose | New Path | Action |
|---|---|---|---|
| `config_explorer/src/config_explorer/capacity_planner.py` | Memory/KV cache calculations (984 lines) | `src/planner/capacity_planner.py` | Move as-is |
| `config_explorer/src/config_explorer/cli.py` | CLI interface (plan + estimate) | `src/planner/cli/planner_cli.py` | Move, update imports. Named generically to support adding NeuralNav CLI features later. |
| `config_explorer/src/config_explorer/__main__.py` | CLI entry point | `src/planner/cli/__main__.py` | Move, update imports |
| `config_explorer/src/config_explorer/__init__.py` | Package marker | Not needed (modules flattened) | Delete |
| `config_explorer/src/config_explorer/recommender/` | GPURecommender + CostManager | `src/planner/gpu_recommender/` | Move, update imports |
| `config_explorer/Capacity_Planner.py` | Streamlit Capacity Planner page (805 lines) | `ui/components/capacity_planner.py` | Refactor into tab render function |
| `config_explorer/pages/2_GPU_Recommender.py` | Streamlit GPU Recommender page (1272 lines) | `ui/components/gpu_recommender.py` | Refactor into tab render function |
| `config_explorer/util.py` | Scenario dataclass + session state helpers (163 lines) | `ui/components/config_explorer_state.py` | Move, update imports |
| `config_explorer/db.py` | GPU specs loader (9 lines) | Remove; inline into `config_explorer_state.py` | Merge (trivial) |
| `config_explorer/db.json` | GPU specs database (12 GPUs) | `data/configuration/gpu_specs.json` | Move |
| `config_explorer/gpu_costs.json` | GPU cost reference data | `data/configuration/gpu_costs.json` | Move |
| `config_explorer/tests/` | 5 test files | `tests/` (flattened alongside existing tests) | Move, update imports |
| `config_explorer/examples/` | Usage example script | `examples/config_explorer/` or remove | Move or remove |
| `config_explorer/pyproject.toml` | Separate project config | Remove; merge deps into root `pyproject.toml` | Merge + delete |
| `config_explorer/requirements.txt` | Core dependencies | Remove; merged into root `pyproject.toml` | Merge + delete |
| `config_explorer/requirements-streamlit.txt` | UI dependencies | Remove; merged into root `pyproject.toml` | Merge + delete |
| `config_explorer/pytest.ini` | Test config | Remove; use root `pyproject.toml` pytest config | Merge + delete |
| `config_explorer/README.md` | Documentation | Move to `docs/CONFIG_EXPLORER.md` | Move |
| `config_explorer/empirical-vllm-memory-results.md` | Memory profiling reference | Move to `docs/` | Move |
| `config_explorer/__init__.py` | Top-level package marker | Remove (no longer needed) | Delete |

---

## 3. Overlap & Consolidation Analysis

### Overlapping areas

| Area | NeuralNav | Config Explorer | Recommendation |
|---|---|---|---|
| **GPU specs/data** | `gpu_normalizer.py` with CANONICAL_GPUS set; `model_catalog.json` has GPU aliases | `db.json` with 12 GPU types (memory + prefix) | Keep separate during migration. Different purposes: NeuralNav normalizes benchmark GPU names; Config Explorer needs memory sizes for capacity planning. **A high priority after restructure**: unify into single GPU registry (see Follow-on Step 1). |
| **GPU cost data** | Scorer uses cost from benchmark DB (monthly cost per config) | `gpu_costs.json` with $/hr reference values per GPU type | Keep separate. Different cost models: NeuralNav costs are per-deployment; Config Explorer costs are per-GPU-hour for comparison. |
| **Recommendation concept** | SLO-driven: user intent → traffic profile → benchmark lookup → ranked configs | HW-driven: model params → memory calc → roofline perf estimate → GPU ranking | Keep separate. Complementary approaches. Future integration noted in requirements but NOT this step. |
| **Pydantic models** | `shared/schemas/` (DeploymentIntent, GPUConfig, etc.) | None (uses dataclasses: KVCacheDetail, Scenario) | Keep separate. No overlap in actual data models. |
| **Streamlit session state** | `ui/state.py` with `SESSION_DEFAULTS` dict | `util.py` with `Scenario` dataclass + helpers | Keep separate session state namespaces. Config Explorer uses `st.session_state['scenario']`; NeuralNav uses flat keys. No collision. |

### Conclusion

No code should be merged at this stage. The two systems are complementary, not duplicative. They coexist as peer modules within `src/planner/`.

---

## 4. UI Integration Plan

### Current state

- **NeuralNav UI** (`ui/app.py`): Single Streamlit app with 6 tabs using `st.tabs()`:
  1. Define Use Case, 2. Technical Specification, 3. Recommendations, 4. Deployment, 5. Deployment Management, 6. Configuration
- **Config Explorer UI**: Multi-page Streamlit app with 2 pages:
  1. `Capacity_Planner.py` (main page), 2. `pages/2_GPU_Recommender.py`

### Integration approach

Keep Config Explorer as a **separate view** rather than adding tabs to the existing NeuralNav workflow. The UI defaults to the NeuralNav view and provides a toggle (e.g., sidebar selector or button) to switch to Config Explorer. This preserves the existing NeuralNav workflow without disruption.

```python
# In ui/app.py — view selector in sidebar
view = st.sidebar.selectbox("View", ["Planner", "Config Explorer"])

if view == "Planner":
    # Existing 6-tab NeuralNav workflow (unchanged)
    tabs = st.tabs([
        "Define Use Case",
        "Technical Specification",
        "Recommendations",
        "Deployment",
        "Deployment Management",
        "Configuration",
    ])
    # ... existing tab rendering ...
elif view == "Config Explorer":
    # Config Explorer pages
    ce_tabs = st.tabs(["Capacity Planner", "GPU Recommender"])
    with ce_tabs[0]:
        render_capacity_planner()
    with ce_tabs[1]:
        render_gpu_recommender()
```

### Files involved

1. **`ui/components/config_explorer_state.py`** (new) - From `config_explorer/util.py`:
   - `Scenario` dataclass
   - Session state init/update helpers
   - GPU specs loading (absorb `db.py` logic, update path to `data/configuration/gpu_specs.json`)

2. **`ui/components/capacity_planner.py`** (new) - From `config_explorer/Capacity_Planner.py`:
   - Wrap entire page content in `def render_capacity_planner():` function
   - Replace `import db` → use `config_explorer_state` GPU specs loading
   - Replace `from src.config_explorer.capacity_planner import *` → `from planner.capacity_planner import ...`
   - Replace `import util` → `from components.config_explorer_state import ...`
   - Remove `st.set_page_config()` call (already set in app.py)

3. **`ui/components/gpu_recommender.py`** (new) - From `config_explorer/pages/2_GPU_Recommender.py`:
   - Wrap in `def render_gpu_recommender():` function
   - Fix `sys.path` hack → proper import `from planner.gpu_recommender.recommender import GPURecommender`
   - Fix `from config_explorer.recommender.cost_manager import CostManager` → `from planner.gpu_recommender.cost_manager import CostManager`

4. **`ui/app.py`** - Modifications:
   - Add view selector (sidebar toggle between "Planner" and "Config Explorer")
   - Add imports for new component render functions
   - Add init call for config_explorer session state
   - Render Config Explorer view when selected
   - Rename "NeuralNav" → "Planner" in page title, hero, branding

5. **`ui/state.py`** - Add config_explorer session state init (call `config_explorer_state.init_session_state()`)

---

## 5. Step-by-Step Migration Plan

### Step 1: Rename `src/planner/` → `src/planner/`

**Files to modify:**

- `mv src/planner src/planner`
- Update ALL `from neuralnav.` → `from planner.` imports in:
  - `src/planner/**/*.py` (17+ source files)
  - `tests/*.py` (conftest.py, 3 test files)
  - `scripts/*.py` (3 script files)
- `pyproject.toml`: name, packages, known-first-party
- `Makefile`: REGISTRY_ORG, BACKEND_IMAGE, backend startup command, pkill patterns, DB credentials (neuralnav → planner for container names, DB name/user)
- `Dockerfile`: COPY path, CMD
- `docker-compose.yml`: container names, network, DB name/user, environment vars
- `docker-compose.dev.yml`: CMD
- `.github/workflows/ci.yml`: --cov path
- `scripts/kind-cluster.sh`: CLUSTER_NAME
- `deploy/kubernetes/*.yaml`: namespace, labels, image refs, secrets, DB connection strings
- `ui/app.py`: page_title, hero text, logo references
- All docs: CLAUDE.md, README.md, CONTRIBUTING.md, docs/*.md

### Step 2: Move Config Explorer core logic into `src/planner/` (flattened)

```bash
# Capacity planner module (single file, peer to other packages)
cp config_explorer/src/config_explorer/capacity_planner.py src/planner/capacity_planner.py

# GPU recommender package
mkdir -p src/planner/gpu_recommender
cp config_explorer/src/config_explorer/recommender/__init__.py src/planner/gpu_recommender/
cp config_explorer/src/config_explorer/recommender/recommender.py src/planner/gpu_recommender/
cp config_explorer/src/config_explorer/recommender/cost_manager.py src/planner/gpu_recommender/

# CLI package
mkdir -p src/planner/cli
cp config_explorer/src/config_explorer/__main__.py src/planner/cli/__main__.py
cp config_explorer/src/config_explorer/cli.py src/planner/cli/planner_cli.py
```

- Create `src/planner/cli/__init__.py`
- Update imports in `planner_cli.py`: `from config_explorer.capacity_planner` → `from planner.capacity_planner`, `from config_explorer.recommender` → `from planner.gpu_recommender`
- Update imports in `cli/__main__.py`: `from config_explorer.cli` → `from planner.cli.config_explorer_cli`
- Update `gpu_recommender/__init__.py` imports if any
- Update `cost_manager.py` to load `gpu_costs.json` from `data/configuration/gpu_costs.json` (relative path resolution)

### Step 3: Move Config Explorer data files

```bash
cp config_explorer/db.json data/configuration/gpu_specs.json
cp config_explorer/gpu_costs.json data/configuration/gpu_costs.json
```

- Update `cost_manager.py` default path for `gpu_costs.json`
- The `db.py` loader will be absorbed into UI component

### Step 4: Move Config Explorer tests (flattened into tests/)

```bash
cp config_explorer/tests/capacity_planner_test.py tests/test_capacity_planner.py
cp config_explorer/tests/test_cli.py tests/test_planner_cli.py
cp config_explorer/tests/test_cost_integration.py tests/test_cost_integration.py
cp config_explorer/tests/test_cost_manager.py tests/test_cost_manager.py
cp config_explorer/tests/test_recommender_cost.py tests/test_recommender_cost.py
```

- Update imports: `from config_explorer.` → `from planner.` (e.g., `from planner.capacity_planner`, `from planner.gpu_recommender`)
- Rename `capacity_planner_test.py` → `test_capacity_planner.py` to match pytest naming convention
- Ensure test markers are compatible with root pytest config

### Step 5: Merge dependencies into root `pyproject.toml`

Config Explorer currently uses pip/setuptools with `requirements.txt` files (no uv). This step consolidates everything under the root `pyproject.toml` managed by uv, eliminating the separate Config Explorer packaging.

Add to `[project.dependencies]`:

```
"huggingface_hub>=0.34.4",
"numpy>=2.3.2",
"scipy>=1.16.1",
"transformers>=4.55.4",
"llm-optimizer @ git+https://github.com/bentoml/llm-optimizer.git",
```

Note: `matplotlib` can go in `[project.optional-dependencies] ui` since it's only used for plotting. `pandas` already exists. `pydantic` and `pyyaml` already exist.

Add CLI entry point:

```toml
[project.scripts]
planner = "planner.cli.planner_cli:main"
```

Update `plotly` version in ui extras if needed (Config Explorer needs `>=6.3.0`, current NeuralNav has `>=5.0.0`).

### Step 6: Integrate Config Explorer UI as a separate view

1. Create `ui/components/config_explorer_state.py` from `config_explorer/util.py`:
   - Update `from src.config_explorer.capacity_planner import *` → `from planner.capacity_planner import ...`
   - Absorb `db.py` logic: load GPU specs from `data/configuration/gpu_specs.json`
   - Fix path to be relative to project root

2. Create `ui/components/capacity_planner.py` from `config_explorer/Capacity_Planner.py`:
   - Wrap all page-level code in `def render_capacity_planner_tab():`
   - Update all imports (db, util, src.config_explorer → planner.capacity_planner)
   - Remove standalone `st.set_page_config()` if present

3. Create `ui/components/gpu_recommender.py` from `config_explorer/pages/2_GPU_Recommender.py`:
   - Wrap in `def render_gpu_recommender_tab():`
   - Remove `sys.path` manipulation
   - Update imports to use `planner.gpu_recommender`

4. Update `ui/app.py`:
   - Add imports for new components
   - Add 2 new tabs ("Capacity Planner", "GPU Recommender")
   - Initialize config_explorer session state in `init_session_state()`
   - Rename branding: "NeuralNav" → "Planner"

5. Update `ui/requirements.txt` (for Docker):
   - Add `plotly>=6.3.0` (bump from >=5.0.0)
   - Add `matplotlib>=3.10.5`

### Step 7: Move remaining Config Explorer files

```bash
cp config_explorer/README.md docs/CONFIG_EXPLORER.md
cp config_explorer/empirical-vllm-memory-results.md docs/
mkdir -p examples/config_explorer
cp config_explorer/examples/gpu_recommender_example.py examples/config_explorer/
```

### Step 8: Remove `config_explorer/` directory

After verifying everything works, delete the entire `config_explorer/` directory.

### Step 9: Update documentation

- `CLAUDE.md`: Full rewrite of references (neuralnav → planner, add config_explorer section)
- `README.md`: Update project name, description, paths
- `CONTRIBUTING.md`: Update repo URLs, paths
- `docs/*.md`: Search-and-replace neuralnav → planner
- Update `ui/static/` logo/branding if applicable

### Step 10: Run `uv sync` and verify

```bash
uv sync --extra ui --extra dev
uv run ruff check src/ ui/ tests/
uv run ruff format --check src/ ui/ tests/
uv run mypy src/
cd src && uv run pytest ../tests/ -v -m "not database and not integration"
```

---

## Verification

1. **Lint & typecheck**: `make lint && make typecheck`
2. **Unit tests**: `make test-unit` — all existing tests pass with renamed imports
3. **Config Explorer tests**: `cd src && uv run pytest ../tests/test_capacity_planner.py ../tests/test_planner_cli.py ../tests/test_cost_manager.py ../tests/test_cost_integration.py ../tests/test_recommender_cost.py -v`
4. **CLI**: `uv run planner plan --model Qwen/Qwen2.5-3B --gpu-memory 80 --max-model-len 16000`
5. **Backend**: `make start-backend && curl http://localhost:8000/health`
6. **UI**: `make start-ui` — verify view toggle works, existing Planner tabs render unchanged, and Config Explorer view renders Capacity Planner and GPU Recommender
7. **No residual references**: `grep -r "neuralnav" src/ ui/ tests/ scripts/ Makefile Dockerfile docker-compose*.yml .github/ deploy/` should return nothing

---

## Follow-on Steps (not part of this migration)

These are recommended next steps after the migration is complete:

### 1. Unify GPU data

Consolidate the separate GPU data sources into a single registry:

- `gpu_specs.json` (Config Explorer: GPU memory sizes)
- `gpu_normalizer.py` + `model_catalog.json` (NeuralNav: GPU name aliases and canonical names)
- `gpu_costs.json` (Config Explorer: reference $/hr costs)

A unified GPU registry would provide memory, cost, aliases, and canonical names in one place. Keeping a single source of truth for the GPU registry is critical.

### 2. Add FastAPI endpoints for Capacity Planner and GPU Recommender

Create API routes so the UI accesses all backend logic consistently via HTTP rather than direct library imports. This ensures the UI and backend can run in separate containers and keeps the architecture uniform.

- Add `src/planner/api/routes/capacity_planner.py` — thin route handlers calling `planner.capacity_planner`
- Add `src/planner/api/routes/gpu_recommender.py` — route handlers calling `planner.gpu_recommender`
- Add corresponding functions in `ui/api_client.py`
- Update the UI components to call the API instead of importing library code directly

### 3. Use Capacity Planner and GPU Recommender for estimated recommendations

Integrate Config Explorer's capabilities into the existing recommendation pipeline to provide estimated recommendations when benchmark data is unavailable for a model/GPU combination:

- Use `capacity_planner.py` to determine which GPU configurations can physically fit a model (memory feasibility filtering)
- Use `gpu_recommender` (BentoML roofline model) to generate synthetic performance estimates for configurations that lack benchmark data
- Feed these estimates into the existing scoring/ranking pipeline as a fallback when PostgreSQL benchmark data has no matching entries

### 4. Add NeuralNav functionality to the CLI

Extend `src/planner/cli/` with subcommands for existing NeuralNav functionality (e.g., `planner recommend`, `planner deploy`) alongside the existing Config Explorer commands (`planner plan`, `planner estimate`).

### 5. Consolidate data models

Evaluate whether Config Explorer's dataclasses (`KVCacheDetail`, `Scenario`) should migrate to Pydantic models in `shared/schemas/` for consistency with the rest of the codebase, especially if they start crossing module boundaries via API endpoints.
