# Design: Config Explorer Backend Migration (Issue #131)

## Context

Issue #131 implements Steps 2–5 of the migration plan (`docs/MIGRATION_PLAN.md`):
move Config Explorer's backend modules into the unified `src/neuralnav/` package.
Step 1 (rename neuralnav → planner) and Step 6 (UI integration) are separate PRs.

## Scope

**In this PR:**
- Move backend Python modules into `src/neuralnav/`
- Move data files into `data/configuration/`
- Move and update tests into `tests/`
- Merge Config Explorer deps into root `pyproject.toml`
- Add `neuralnav-plan` CLI entry point
- Delete backend-only parts of `config_explorer/`

**Not in this PR:**
- UI integration (Streamlit components, sidebar toggle)
- neuralnav → planner rename

---

## 1. File Moves (Backend)

| Source | Destination | Import changes |
|--------|-------------|----------------|
| `config_explorer/src/config_explorer/capacity_planner.py` | `src/neuralnav/capacity_planner.py` | None (no internal config_explorer imports) |
| `config_explorer/src/config_explorer/recommender/__init__.py` | `src/neuralnav/gpu_recommender/__init__.py` | None |
| `config_explorer/src/config_explorer/recommender/recommender.py` | `src/neuralnav/gpu_recommender/recommender.py` | `from config_explorer.capacity_planner` → `from neuralnav.capacity_planner`; `from config_explorer.recommender.cost_manager` → `from neuralnav.gpu_recommender.cost_manager` |
| `config_explorer/src/config_explorer/recommender/cost_manager.py` | `src/neuralnav/gpu_recommender/cost_manager.py` | Fix data file path (see §3) |
| `config_explorer/src/config_explorer/cli.py` | `src/neuralnav/cli/planner_cli.py` | All `from config_explorer.*` → `from neuralnav.*`; remove `start_ui()` subcommand |
| `config_explorer/src/config_explorer/__main__.py` | `src/neuralnav/cli/__main__.py` | `from config_explorer.cli import main` → `from neuralnav.cli.planner_cli import main` |
| _(new)_ | `src/neuralnav/cli/__init__.py` | Empty package marker |

---

## 2. Data Files

| Source | Destination |
|--------|-------------|
| `config_explorer/db.json` | `data/configuration/gpu_specs.json` |
| `config_explorer/gpu_costs.json` | `data/configuration/gpu_costs.json` |

---

## 3. Path Fix in `cost_manager.py`

The current loader navigates 4 levels up from the source file to find `gpu_costs.json` at the `config_explorer/` root:

```python
# current (broken after move)
cost_file = Path(__file__).parent.parent.parent.parent / "gpu_costs.json"
```

After moving to `src/neuralnav/gpu_recommender/cost_manager.py`, `parents[3]` resolves to the repo root:

```python
# updated
cost_file = Path(__file__).parents[3] / "data" / "configuration" / "gpu_costs.json"
```

---

## 4. Tests

Moved into `tests/unit/` or `tests/integration/` to match the existing repo structure, renamed to match pytest convention, imports updated:

| Source | Destination | Import change |
|--------|-------------|---------------|
| `config_explorer/tests/capacity_planner_test.py` | `tests/integration/test_capacity_planner.py` | `from src.config_explorer.capacity_planner import *` → explicit `from neuralnav.capacity_planner import ...` |
| `config_explorer/tests/test_cli.py` | `tests/integration/test_planner_cli.py` | `config_explorer.*` → `neuralnav.*` |
| `config_explorer/tests/test_cost_integration.py` | `tests/integration/test_cost_integration.py` | Same |
| `config_explorer/tests/test_cost_manager.py` | `tests/unit/test_cost_manager.py` | Same |
| `config_explorer/tests/test_recommender_cost.py` | `tests/unit/test_recommender_cost.py` | Same |

Tests that call HuggingFace APIs (e.g., `test_get_model_info_and_config_from_hf`) must be
marked `@pytest.mark.integration` so the unit test suite runs without network access.

---

## 5. Dependencies

Add to `[project.dependencies]` in root `pyproject.toml`:

```toml
"huggingface_hub>=0.34.4",
"transformers>=4.55.4",
"llm-optimizer @ git+https://github.com/bentoml/llm-optimizer.git",
```

Note: `numpy` and `scipy` are not directly imported by the migrated backend modules; they are
transitive dependencies of `llm-optimizer` and do not need to be added explicitly.

**Potential version conflicts**: `llm-optimizer` may require newer versions of packages already
pinned in root `pyproject.toml` (e.g., `pydantic==2.9.2`, `pandas==2.2.0`). Run `uv sync` after
adding the new deps; if resolution fails, bump the conflicting pins accordingly and re-run the
full test suite.

Add `matplotlib>=3.10.5` to `[project.optional-dependencies] ui` (only used for plotting in UI code).

Bump `plotly` in `ui` extras from `>=5.0.0` to `>=6.3.0`.

Add a new `[project.scripts]` section (root `pyproject.toml` does not currently have one):

```toml
[project.scripts]
neuralnav-plan = "neuralnav.cli.planner_cli:main"
```

Remove:
- `config_explorer/pyproject.toml`
- `config_explorer/requirements.txt`
- `config_explorer/requirements-streamlit.txt`
- `config_explorer/pytest.ini`

---

## 6. Cleanup

Delete after moving backend files:

```
config_explorer/src/           (entire directory)
config_explorer/tests/         (entire directory)
config_explorer/pyproject.toml
config_explorer/requirements.txt
config_explorer/requirements-streamlit.txt
config_explorer/pytest.ini
config_explorer/db.json
config_explorer/gpu_costs.json
config_explorer/db.py
config_explorer/__init__.py
```

Leave in place for the UI PR:

```
config_explorer/Capacity_Planner.py
config_explorer/pages/
config_explorer/util.py
config_explorer/examples/
config_explorer/README.md
config_explorer/empirical-vllm-memory-results.md
```

---

## 7. Verification

```bash
uv sync --extra ui --extra dev
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
cd src && uv run pytest ../tests/unit/test_cost_manager.py ../tests/unit/test_recommender_cost.py -v -m "not integration"
uv run neuralnav-plan plan --model Qwen/Qwen2.5-0.5B --gpu-memory 80
```
