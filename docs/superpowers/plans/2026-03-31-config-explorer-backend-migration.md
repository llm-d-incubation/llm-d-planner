# Config Explorer Backend Migration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Config Explorer's backend Python modules, data files, and tests into `src/neuralnav/`, merge dependencies into root `pyproject.toml`, and delete the vacated backend files from `config_explorer/`.

**Architecture:** Copy each source file to its new location under `src/neuralnav/`, update all `config_explorer.*` imports to `neuralnav.*`, fix the `gpu_costs.json` path, and remove the `start_ui()` CLI subcommand. `CostManager` and `GPURecommender` are merged into a single flat module `src/neuralnav/gpu_recommender.py` (no sub-package). Tests move to `tests/unit/` or `tests/integration/` matching the repo's existing structure.

**Tech Stack:** Python 3.11+, uv, hatchling, pytest, huggingface_hub, transformers, llm-optimizer (BentoML)

**Spec:** `docs/superpowers/specs/2026-03-31-config-explorer-backend-migration-design.md`

**Note on test_recommender_cost.py:** The spec places this in `tests/unit/`, but `GPURecommender.__init__` calls HuggingFace on every instantiation — all fixtures in that file require network. This plan corrects the destination to `tests/integration/test_recommender_cost.py` and marks it `@pytest.mark.integration`.

---

## File Map

### Created
| File | Source |
|------|--------|
| `src/neuralnav/capacity_planner.py` | `config_explorer/src/config_explorer/capacity_planner.py` (copy, no changes) |
| `src/neuralnav/gpu_recommender.py` | merge of `cost_manager.py` + `recommender.py`; fix data path + imports |
| `src/neuralnav/cli/__init__.py` | new empty file |
| `src/neuralnav/cli/__main__.py` | copy + fix import |
| `src/neuralnav/cli/planner_cli.py` | copy + fix imports + remove `start_ui()` |
| `data/configuration/gpu_specs.json` | `config_explorer/db.json` (copy) |
| `data/configuration/gpu_costs.json` | `config_explorer/gpu_costs.json` (copy) |
| `tests/unit/test_cost_manager.py` | copy + fix import |
| `tests/integration/test_capacity_planner.py` | copy + fix import + add markers |
| `tests/integration/test_planner_cli.py` | copy + fix CLI name + add markers |
| `tests/integration/test_cost_integration.py` | copy + remove sys.path hack + fix import + add markers |
| `tests/integration/test_recommender_cost.py` | copy + fix import + add markers |

### Modified
| File | Change |
|------|--------|
| `pyproject.toml` | Add 3 deps, add `[project.scripts]`, bump plotly, add matplotlib |

### Deleted
```
config_explorer/src/
config_explorer/tests/
config_explorer/pyproject.toml
config_explorer/requirements.txt
config_explorer/requirements-streamlit.txt
config_explorer/pytest.ini
config_explorer/db.json
config_explorer/gpu_costs.json
config_explorer/db.py
config_explorer/__init__.py
```

---

## Chunk 1: Dependencies and Data Files

### Task 1: Update `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add new dependencies**

In `pyproject.toml`, add to the `[project.dependencies]` list (after the existing `"requests==2.32.5"` line):

```toml
    "huggingface_hub>=0.34.4",
    "transformers>=4.55.4",
    "llm-optimizer @ git+https://github.com/bentoml/llm-optimizer.git",
```

- [ ] **Step 2: Bump plotly and add matplotlib in ui extras**

In `[project.optional-dependencies]`, change the `ui` list from:
```toml
ui = [
    "streamlit==1.39.0",
    "plotly>=5.0.0",
]
```
to:
```toml
ui = [
    "streamlit==1.39.0",
    "plotly>=6.3.0",
    "matplotlib>=3.10.5",
]
```

- [ ] **Step 3: Add `[project.scripts]` section**

Add a new section after `[project.optional-dependencies]` (before `[build-system]`):

```toml
[project.scripts]
neuralnav-plan = "neuralnav.cli.planner_cli:main"
```

- [ ] **Step 4: Run `uv sync` and check for conflicts**

```bash
uv sync --extra ui --extra dev
```

Expected: resolves without errors. If `uv sync` fails due to version conflicts (e.g., pydantic or pandas), bump those pins to meet the new requirements and re-run. Document any bumped pins in the commit message.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock && git commit -s -m "chore: add config-explorer deps and neuralnav-plan entry point

Add huggingface_hub, transformers, llm-optimizer dependencies.
Add neuralnav-plan CLI entry point.
Bump plotly to >=6.3.0, add matplotlib to ui extras.

Assisted-by: Claude <noreply@anthropic.com>"
```

---

### Task 2: Copy data files

**Files:**
- Create: `data/configuration/gpu_specs.json`
- Create: `data/configuration/gpu_costs.json`

- [ ] **Step 1: Copy data files**

```bash
cp config_explorer/db.json data/configuration/gpu_specs.json
cp config_explorer/gpu_costs.json data/configuration/gpu_costs.json
```

- [ ] **Step 2: Verify files exist and are valid JSON**

```bash
python3 -c "import json; json.load(open('data/configuration/gpu_specs.json')); print('gpu_specs.json OK')"
python3 -c "import json; json.load(open('data/configuration/gpu_costs.json')); print('gpu_costs.json OK')"
```

Expected: both print `OK`.

- [ ] **Step 3: Commit**

```bash
git add data/configuration/gpu_specs.json data/configuration/gpu_costs.json && git commit -s -m "chore: add gpu_specs.json and gpu_costs.json to data/configuration/

Move config_explorer data files to unified data directory.

Assisted-by: Claude <noreply@anthropic.com>"
```

---

## Chunk 2: Backend Modules

### Task 3: Create `src/neuralnav/capacity_planner.py`

**Files:**
- Create: `src/neuralnav/capacity_planner.py`

- [ ] **Step 1: Copy the file**

```bash
cp config_explorer/src/config_explorer/capacity_planner.py src/neuralnav/capacity_planner.py
```

No import changes are needed — `capacity_planner.py` has no internal `config_explorer` imports.

- [ ] **Step 2: Run ruff and mypy on the new file**

```bash
uv run ruff check src/neuralnav/capacity_planner.py
uv run mypy src/neuralnav/capacity_planner.py
```

Expected: no errors (or only pre-existing issues from the original file — do not fix unrelated issues).

- [ ] **Step 3: Commit**

```bash
git add src/neuralnav/capacity_planner.py && git commit -s -m "feat: add capacity_planner module to src/neuralnav/

Move Config Explorer capacity planning logic (memory/KV cache calculations)
into the unified neuralnav package.

Assisted-by: Claude <noreply@anthropic.com>"
```

---

### Task 4: Create `src/neuralnav/gpu_recommender.py`

`CostManager` and `GPURecommender` are merged into a single flat module. `CostManager` comes first (no import needed since they share scope), followed by `GPURecommender`.

**Files:**
- Create: `src/neuralnav/gpu_recommender.py`
- Test: `tests/unit/test_cost_manager.py`

- [ ] **Step 1: Write the failing unit test first**

Create `tests/unit/test_cost_manager.py` with the content below. The import will fail until the module exists — that's expected.

```python
"""Unit tests for CostManager"""
import pytest
import json
import tempfile
from pathlib import Path
from neuralnav.gpu_recommender import CostManager


class TestCostManager:
    """Test suite for CostManager class"""

    def test_init_default(self):
        """Test initialization with default costs"""
        manager = CostManager()
        assert manager.default_costs is not None
        assert isinstance(manager.default_costs, dict)
        assert manager.custom_costs == {}

    def test_init_with_custom_costs(self):
        """Test initialization with custom costs"""
        custom = {"H100": 30.0, "A100": 20.0}
        manager = CostManager(custom_costs=custom)
        assert manager.custom_costs == custom

    def test_get_cost_default(self):
        """Test getting cost from default database"""
        manager = CostManager()

        # Test known GPU
        cost = manager.get_cost("H100", num_gpus=1)
        assert cost is not None
        assert cost > 0

        # Test multi-GPU
        cost_multi = manager.get_cost("H100", num_gpus=2)
        assert cost_multi == cost * 2

    def test_get_cost_custom_override(self):
        """Test that custom costs override defaults"""
        custom = {"H100": 30.0}
        manager = CostManager(custom_costs=custom)

        cost = manager.get_cost("H100", num_gpus=1)
        assert cost == 30.0

    def test_get_cost_unknown_gpu(self):
        """Test getting cost for unknown GPU"""
        manager = CostManager()
        cost = manager.get_cost("UNKNOWN_GPU", num_gpus=1)
        assert cost is None

    def test_get_all_costs(self):
        """Test getting all costs"""
        manager = CostManager()
        all_costs = manager.get_all_costs()

        assert isinstance(all_costs, dict)
        assert len(all_costs) > 0
        assert "H100" in all_costs

    def test_get_all_costs_with_custom(self):
        """Test that get_all_costs includes custom overrides"""
        custom = {"H100": 30.0, "CUSTOM_GPU": 50.0}
        manager = CostManager(custom_costs=custom)

        all_costs = manager.get_all_costs()
        assert all_costs["H100"] == 30.0
        assert all_costs["CUSTOM_GPU"] == 50.0

    def test_has_cost(self):
        """Test checking if cost data exists"""
        manager = CostManager()

        assert manager.has_cost("H100") is True
        assert manager.has_cost("UNKNOWN_GPU") is False

    def test_has_cost_custom(self):
        """Test has_cost with custom costs"""
        custom = {"CUSTOM_GPU": 50.0}
        manager = CostManager(custom_costs=custom)

        assert manager.has_cost("CUSTOM_GPU") is True

    def test_multi_gpu_cost_calculation(self):
        """Test cost calculation for multiple GPUs"""
        manager = CostManager()

        single_cost = manager.get_cost("H100", num_gpus=1)
        double_cost = manager.get_cost("H100", num_gpus=2)
        quad_cost = manager.get_cost("H100", num_gpus=4)

        assert double_cost == single_cost * 2
        assert quad_cost == single_cost * 4

    def test_zero_gpus(self):
        """Test cost calculation with zero GPUs"""
        manager = CostManager()
        cost = manager.get_cost("H100", num_gpus=0)
        assert cost == 0

    def test_negative_gpus(self):
        """Test cost calculation with negative GPUs (edge case)"""
        manager = CostManager()
        cost = manager.get_cost("H100", num_gpus=-1)
        # Should still calculate (negative cost)
        assert cost is not None

    def test_default_costs_structure(self):
        """Test that default costs have expected structure"""
        manager = CostManager()

        for gpu_name, data in manager.default_costs.items():
            # Skip non-GPU entries like _disclaimer
            if not isinstance(data, dict):
                continue
            if "cost" not in data:
                continue

            assert "cost" in data
            assert "source" in data
            assert isinstance(data["cost"], (int, float))
            assert data["cost"] >= 0

    def test_custom_costs_override_all(self):
        """Test that custom costs can override multiple GPUs"""
        custom = {
            "H100": 30.0,
            "A100": 20.0,
            "L40": 25.0,
        }
        manager = CostManager(custom_costs=custom)

        assert manager.get_cost("H100") == 30.0
        assert manager.get_cost("A100") == 20.0
        assert manager.get_cost("L40") == 25.0

    def test_empty_custom_costs(self):
        """Test with empty custom costs dict"""
        manager = CostManager(custom_costs={})

        # Should fall back to defaults
        cost = manager.get_cost("H100")
        assert cost is not None
        assert cost > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: Run the test and confirm it fails with ImportError**

```bash
cd src && uv run pytest ../tests/unit/test_cost_manager.py -v 2>&1 | head -20
```

Expected: `ImportError: No module named 'neuralnav.gpu_recommender'`

- [ ] **Step 3: Create `src/neuralnav/gpu_recommender.py`**

Concatenate `cost_manager.py` and `recommender.py` into one file, then apply the two edits below:

```bash
cat config_explorer/src/config_explorer/recommender/cost_manager.py \
    config_explorer/src/config_explorer/recommender/recommender.py \
    > src/neuralnav/gpu_recommender.py
```

**Edit 1 — fix the data path** in the `_load_default_costs` method. Find:

```python
        cost_file = Path(__file__).parent.parent.parent.parent / "gpu_costs.json"
```

Replace with (2 levels up from `src/neuralnav/gpu_recommender.py` reaches `src/`, one more reaches repo root):

```python
        cost_file = Path(__file__).parent.parent.parent / "data" / "configuration" / "gpu_costs.json"
```

**Edit 2 — fix `GPURecommender` imports**. The two `config_explorer` import lines near the top of the `GPURecommender` section (originally from `recommender.py`) reference modules that no longer need importing since `CostManager` is now defined in the same file. Replace:

```python
from config_explorer.capacity_planner import get_model_config_from_hf, get_model_info_from_hf, get_text_config
from config_explorer.recommender.cost_manager import CostManager
```

with (remove the `CostManager` import entirely — it's defined above in the same file; keep only the capacity_planner import):

```python
from neuralnav.capacity_planner import get_model_config_from_hf, get_model_info_from_hf, get_text_config
```

Also remove the duplicate `from pathlib import Path` and `from typing import ...` imports that `cat` will have duplicated — keep only one copy of each at the top of the file.

- [ ] **Step 4: Run unit tests — they should pass now**

```bash
cd src && uv run pytest ../tests/unit/test_cost_manager.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Verify ruff and mypy on the new module**

```bash
uv run ruff check src/neuralnav/gpu_recommender.py
uv run mypy src/neuralnav/gpu_recommender.py
```

Expected: no new errors (pre-existing mypy issues from the original files are acceptable).

- [ ] **Step 6: Commit**

```bash
git add src/neuralnav/gpu_recommender.py tests/unit/test_cost_manager.py && git commit -s -m "feat: add gpu_recommender module to src/neuralnav/

Merge Config Explorer CostManager and GPURecommender into a single flat
module. Fix gpu_costs.json data path to new location.

Assisted-by: Claude <noreply@anthropic.com>"
```

---

### Task 5: Create `src/neuralnav/cli/` package

**Files:**
- Create: `src/neuralnav/cli/__init__.py`
- Create: `src/neuralnav/cli/__main__.py`
- Create: `src/neuralnav/cli/planner_cli.py`

- [ ] **Step 1: Copy `__main__.py` and fix import**

```bash
cp config_explorer/src/config_explorer/__main__.py src/neuralnav/cli/__main__.py
```

Edit `src/neuralnav/cli/__main__.py`. Replace:

```python
from config_explorer.cli import main
```

with:

```python
from neuralnav.cli.planner_cli import main
```

- [ ] **Step 2: Create empty `__init__.py`**

```bash
touch src/neuralnav/cli/__init__.py
```

- [ ] **Step 3: Copy `cli.py` as `planner_cli.py` and apply all changes**

```bash
cp config_explorer/src/config_explorer/cli.py src/neuralnav/cli/planner_cli.py
```

Edit `src/neuralnav/cli/planner_cli.py`:

**a) Fix imports** — two separate replacements. Leave all other imports (including `llm_optimizer`) unchanged:

```python
# Replace (multi-line import block starting with):
from config_explorer.capacity_planner import (
# With:
from neuralnav.capacity_planner import (
```

```python
# Replace:
from config_explorer.recommender.recommender import GPURecommender
# With:
from neuralnav.gpu_recommender import GPURecommender
```

**b) Remove the `start_ui()` function** — delete the entire function definition. It starts with `def start_ui():` and ends just before `def plan_capacity(args):`. The whole block looks like:

```python
def start_ui():
    """Start the Streamlit UI"""

    # Get the path to Capacity_Planner.py
    config_explorer_dir = Path(__file__).parent.parent.parent
    ui_file = config_explorer_dir / "Capacity_Planner.py"
    ...
```

Delete from `def start_ui():` through the last line of the function body.

**c) Remove the `start` subparser** — in `main()`, delete the block:

```python
    # Start command
    subparsers.add_parser(
        'start',
        help='Start the Streamlit UI'
    )
```

**d) Remove the `start` dispatch** — in `main()`, delete:

```python
    if args.command == 'start':
        start_ui()
    elif args.command == 'plan':
```

and replace the resulting `elif` with `if`:

```python
    if args.command == 'plan':
        plan_capacity(args)
    elif args.command == 'estimate':
        estimate_performance(args)
    else:
        parser.print_help()
        sys.exit(1)
```

- [ ] **Step 4: Smoke-test the CLI entry point**

After `uv sync` has already been run (Task 1), the entry point is installed. Run:

```bash
uv run neuralnav-plan --help
```

Expected output: shows `plan` and `estimate` subcommands, no `start` subcommand.

```bash
uv run neuralnav-plan plan --help
```

Expected: shows `--model`, `--gpu-memory`, etc.

- [ ] **Step 5: Run ruff on the cli package**

```bash
uv run ruff check src/neuralnav/cli/
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/neuralnav/cli/ && git commit -s -m "feat: add cli package to src/neuralnav/

Move Config Explorer CLI (plan + estimate subcommands) into the unified
neuralnav package as neuralnav.cli.planner_cli. Remove standalone start_ui
subcommand (UI integration is a separate PR).

Assisted-by: Claude <noreply@anthropic.com>"
```

---

## Chunk 3: Tests

### Task 6: Move integration tests

**Files:**
- Create: `tests/integration/test_capacity_planner.py`
- Create: `tests/integration/test_planner_cli.py`
- Create: `tests/integration/test_cost_integration.py`
- Create: `tests/integration/test_recommender_cost.py`

These tests all require HuggingFace network access or invoke the CLI binary, so they all go in `tests/integration/` and are marked `@pytest.mark.integration`.

We copy-and-update each file. We do not run them locally (they require network + installed models) — CI with the `integration` marker will cover them.

- [ ] **Step 1: Create `tests/integration/test_capacity_planner.py`**

Copy the file:

```bash
cp config_explorer/tests/capacity_planner_test.py tests/integration/test_capacity_planner.py
```

Edit `tests/integration/test_capacity_planner.py`:

Replace the import line:

```python
from src.config_explorer.capacity_planner import *
```

with explicit imports (replacing the wildcard is required — `ruff` will flag it):

```python
import pytest
from neuralnav.capacity_planner import (
    # Classes / enums
    AttentionType,
    KVCacheDetail,
    # Model info
    get_model_config_from_hf,
    get_model_info_from_hf,
    get_safetensors_metadata_from_hf,
    get_text_config,
    model_total_params,
    model_memory_req,
    max_context_len,
    # Dtype / quantization helpers
    inference_dtype,
    inference_dtype_byte,
    precision_to_byte,
    get_quant_method,
    get_quant_bytes,
    # Architecture helpers
    is_moe,
    is_multimodal,
    get_num_experts,
    # KV cache / memory
    kv_cache_req,
    allocatable_kv_cache_memory,
    total_kv_cache_blocks,
    max_concurrent_requests,
    estimate_vllm_activation_memory,
    estimate_vllm_cuda_graph_memory,
    estimate_vllm_non_torch_memory,
    # Parallelism / GPU sizing
    find_possible_tp,
    gpus_required,
    per_gpu_model_memory_required,
    auto_max_model_len,
    experts_per_ep_group,
    # Parameter / memory math
    model_params_by_dtype,
    parameter_memory_req,
    # Constants
    ACTIVATION_MEMORY_BASE_DENSE_GIB,
    ACTIVATION_MEMORY_BASE_MULTIMODAL_GIB,
    VALIDATED_ACTIVATION_PROFILES,
)
```

Add `@pytest.mark.integration` decorator above every test function definition in the file. Example:

```python
@pytest.mark.integration
def test_get_model_info_and_config_from_hf():
    ...
```

Remove the `import pytest` line at the top if it already exists (avoid duplicate).

- [ ] **Step 2: Create `tests/integration/test_planner_cli.py`**

Copy the file:

```bash
cp config_explorer/tests/test_cli.py tests/integration/test_planner_cli.py
```

Edit `tests/integration/test_planner_cli.py`:

**a) Fix the CLI command name** in `run_cli()`:

```python
# Replace:
    cmd = ["config-explorer"] + list(args)
# With:
    cmd = ["neuralnav-plan"] + list(args)
```

**b) Fix the string assertion in `test_help()`**:

```python
# Replace:
        assert "config-explorer" in result.stdout.lower() or "config explorer" in result.stdout.lower()
# With:
        assert "neuralnav-plan" in result.stdout.lower() or "neuralnav" in result.stdout.lower()
```

**c) Delete `test_start_help()`** — the `start` subcommand no longer exists. Remove the entire method:

```python
    def test_start_help(self):
        """Test start help"""
        result = run_cli("start", "--help")
        assert result.returncode == 0
```

**d) Add `@pytest.mark.integration` to every test class and standalone test function.**

For class-level marking:

```python
@pytest.mark.integration
class TestHelp:
    ...

@pytest.mark.integration
class TestPlanCommand:
    ...
```

(Apply to all classes in the file.)

- [ ] **Step 3: Create `tests/integration/test_cost_integration.py`**

Copy the file:

```bash
cp config_explorer/tests/test_cost_integration.py tests/integration/test_cost_integration.py
```

Edit `tests/integration/test_cost_integration.py`:

**a) Remove the `sys.path` hack** — delete these lines near the top:

```python
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
```

**b) Fix the import**:

```python
# Replace:
from config_explorer.recommender import GPURecommender, CostManager
# With:
from neuralnav.gpu_recommender import GPURecommender, CostManager
```

**c) Add `@pytest.mark.integration` above each test function** (`test_cost_manager`, `test_gpu_recommender`).

- [ ] **Step 4: Create `tests/integration/test_recommender_cost.py`**

Copy the file:

```bash
cp config_explorer/tests/test_recommender_cost.py tests/integration/test_recommender_cost.py
```

Edit `tests/integration/test_recommender_cost.py`:

**a) Fix the import**:

```python
# Replace:
from config_explorer.recommender import GPURecommender
# With:
from neuralnav.gpu_recommender import GPURecommender
```

**b) Add `@pytest.mark.integration` to the test class**:

```python
@pytest.mark.integration
class TestGPURecommenderCost:
    ...
```

- [ ] **Step 5: Verify the unit tests still pass**

```bash
cd src && uv run pytest ../tests/unit/ -v -m "not integration"
```

Expected: all existing unit tests pass, plus the new `test_cost_manager.py` tests pass.

- [ ] **Step 6: Verify the integration test files at least import cleanly** (no network needed to check syntax)

```bash
uv run python -c "import ast; ast.parse(open('tests/integration/test_capacity_planner.py').read()); print('OK')"
uv run python -c "import ast; ast.parse(open('tests/integration/test_planner_cli.py').read()); print('OK')"
uv run python -c "import ast; ast.parse(open('tests/integration/test_cost_integration.py').read()); print('OK')"
uv run python -c "import ast; ast.parse(open('tests/integration/test_recommender_cost.py').read()); print('OK')"
```

Expected: all print `OK`.

- [ ] **Step 7: Run ruff over the new test files**

```bash
uv run ruff check tests/integration/test_capacity_planner.py tests/integration/test_planner_cli.py tests/integration/test_cost_integration.py tests/integration/test_recommender_cost.py
```

Fix any issues ruff reports before committing.

- [ ] **Step 8: Commit**

```bash
git add tests/integration/test_capacity_planner.py tests/integration/test_planner_cli.py tests/integration/test_cost_integration.py tests/integration/test_recommender_cost.py && git commit -s -m "test: add config-explorer integration tests to tests/integration/

Move and update capacity planner, CLI, cost integration, and GPU recommender
cost tests. Update imports from config_explorer.* to neuralnav.*. Add
@pytest.mark.integration markers. Fix test_planner_cli.py to call
neuralnav-plan instead of config-explorer.

Assisted-by: Claude <noreply@anthropic.com>"
```

---

## Chunk 4: Cleanup and Final Verification

### Task 7: Delete config_explorer backend files

**Files:** various deletions under `config_explorer/`

- [ ] **Step 1: Delete backend source tree**

```bash
rm -rf config_explorer/src/
rm -rf config_explorer/tests/
```

- [ ] **Step 2: Delete packaging and config files**

```bash
rm config_explorer/pyproject.toml
rm config_explorer/requirements.txt
rm config_explorer/requirements-streamlit.txt
rm config_explorer/pytest.ini
```

- [ ] **Step 3: Delete data and utility files that have been moved**

```bash
rm config_explorer/db.json
rm config_explorer/gpu_costs.json
rm config_explorer/db.py
rm config_explorer/__init__.py
```

- [ ] **Step 4: Verify the remaining config_explorer/ contents are only UI files**

```bash
ls config_explorer/
```

Expected output (UI files kept for the UI PR):
```
Capacity_Planner.py
empirical-vllm-memory-results.md
examples/
pages/
README.md
util.py
```

If anything unexpected is listed, investigate before deleting.

- [ ] **Step 5: Commit**

```bash
git add -A config_explorer/ && git commit -s -m "chore: remove config_explorer backend files after migration

Delete config_explorer/src/, tests/, pyproject.toml, requirements files,
pytest.ini, db.json, gpu_costs.json, db.py, and __init__.py. These have
been migrated to src/neuralnav/ and data/configuration/. UI files remain
in config_explorer/ pending the UI integration PR.

Assisted-by: Claude <noreply@anthropic.com>"
```

---

### Task 8: Final verification

- [ ] **Step 1: Run full lint and format check**

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

Expected: no errors. If ruff reports issues, fix them and re-run before proceeding.

- [ ] **Step 2: Run mypy**

```bash
uv run mypy src/neuralnav/
```

Expected: no new errors beyond what existed before this PR. Pre-existing errors in unmodified files are acceptable.

- [ ] **Step 3: Run the full unit test suite**

```bash
cd src && uv run pytest ../tests/unit/ -v -m "not integration and not database"
```

Expected: all pass.

- [ ] **Step 4: Smoke-test the CLI end-to-end**

```bash
uv run neuralnav-plan plan --model Qwen/Qwen2.5-0.5B --gpu-memory 80 --max-model-len 4096
```

Expected: JSON output with `model_memory_gb`, `kv_cache_detail`, etc.

- [ ] **Step 5: Verify no residual `config_explorer` imports in src/**

```bash
grep -r "from config_explorer" src/ tests/
```

Expected: no output.

- [ ] **Step 6: Commit any lint fixes (if needed)**

If steps 1–2 required fixes:

```bash
git add src/ tests/ && git commit -s -m "fix: ruff/mypy cleanup after config-explorer migration

Assisted-by: Claude <noreply@anthropic.com>"
```

---

## Summary of Commits

1. `chore: add config-explorer deps and neuralnav-plan entry point`
2. `chore: add gpu_specs.json and gpu_costs.json to data/configuration/`
3. `feat: add capacity_planner module to src/neuralnav/`
4. `feat: add gpu_recommender module to src/neuralnav/`
5. `feat: add cli package to src/neuralnav/`
6. `test: add config-explorer integration tests to tests/integration/`
7. `chore: remove config_explorer backend files after migration`
8. `fix: ruff/mypy cleanup after config-explorer migration` _(if needed)_
