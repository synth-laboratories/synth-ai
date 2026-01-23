"""Validation tests for all demo notebooks.

These tests ensure each demo notebook:
1. Is valid JSON and can be parsed
2. Has required structure (markdown + code cells)
3. Code cells have valid Python syntax
4. Core imports are available
"""

import json
from pathlib import Path

import pytest

# Base path for demos
DEMOS_PATH = Path(__file__).parent.parent.parent / "demos"

# All primary demo notebooks (excluding .executed versions which are just saved outputs)
DEMO_NOTEBOOKS = [
    ("gepa_banking77", "gepa_banking77_prompt_optimization.ipynb"),
    ("gepa_crafter_vlm", "gepa_crafter_vlm_verifier_optimization.ipynb"),
    ("rlm-mit", "gepa_oolong_rlm_prompt_optimization.ipynb"),
    ("image_style_matching", "graphgen_image_style_matching.ipynb"),
    ("style_matching", "style_matching_prompt_optimization.ipynb"),
]


def get_notebook_path(demo_dir: str, notebook_name: str) -> Path:
    """Get the full path to a notebook."""
    return DEMOS_PATH / demo_dir / notebook_name


def load_notebook(path: Path) -> dict:
    """Load and parse a notebook file."""
    with open(path) as f:
        return json.load(f)


@pytest.mark.unit
class TestAllDemoNotebooks:
    """Common validation tests for all demo notebooks."""

    @pytest.mark.parametrize("demo_dir,notebook_name", DEMO_NOTEBOOKS)
    def test_notebook_exists(self, demo_dir: str, notebook_name: str):
        """Verify the notebook file exists."""
        path = get_notebook_path(demo_dir, notebook_name)
        assert path.exists(), f"Notebook not found at {path}"

    @pytest.mark.parametrize("demo_dir,notebook_name", DEMO_NOTEBOOKS)
    def test_notebook_is_valid_json(self, demo_dir: str, notebook_name: str):
        """Verify the notebook is valid JSON with required structure."""
        path = get_notebook_path(demo_dir, notebook_name)
        notebook = load_notebook(path)

        assert "cells" in notebook, f"{notebook_name}: missing 'cells' key"
        assert "metadata" in notebook, f"{notebook_name}: missing 'metadata' key"
        assert len(notebook["cells"]) > 0, f"{notebook_name}: has no cells"

    @pytest.mark.parametrize("demo_dir,notebook_name", DEMO_NOTEBOOKS)
    def test_notebook_has_markdown_and_code_cells(self, demo_dir: str, notebook_name: str):
        """Verify the notebook has both markdown and code cells."""
        path = get_notebook_path(demo_dir, notebook_name)
        notebook = load_notebook(path)

        cell_types = [c["cell_type"] for c in notebook["cells"]]
        assert "markdown" in cell_types, f"{notebook_name}: should have markdown cells"
        assert "code" in cell_types, f"{notebook_name}: should have code cells"

    @pytest.mark.parametrize("demo_dir,notebook_name", DEMO_NOTEBOOKS)
    def test_notebook_code_cells_have_valid_syntax(self, demo_dir: str, notebook_name: str):
        """Verify all code cells have valid Python syntax."""
        path = get_notebook_path(demo_dir, notebook_name)
        notebook = load_notebook(path)

        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue

            source = "".join(cell["source"])

            # Skip cells that contain shell commands (!) - Jupyter magic
            if "!" in source:
                continue

            # Skip cells with IPython magic (%)
            if "%" in source:
                continue

            # Skip empty cells
            if not source.strip():
                continue

            # Skip cells with await at top level (valid in Jupyter but not standard Python)
            if "await " in source and not source.strip().startswith("async "):
                continue

            try:
                compile(source, f"<{notebook_name}:cell_{i}>", "exec")
            except SyntaxError as e:
                pytest.fail(f"{notebook_name} cell {i}: Syntax error - {e}")


@pytest.mark.unit
class TestCoreImportsAvailable:
    """Verify core imports used across notebooks are available."""

    def test_standard_library_imports(self):
        """Verify standard library imports are available."""
        imports = ["json", "asyncio", "os", "sys", "re"]
        for module_name in imports:
            try:
                __import__(module_name)
            except ImportError:
                pytest.fail(f"Standard library module '{module_name}' not importable")

    def test_third_party_imports(self):
        """Verify third-party imports are available."""
        imports = ["httpx", "nest_asyncio", "datasets", "openai"]
        for module_name in imports:
            try:
                __import__(module_name)
            except ImportError:
                pytest.fail(f"Third-party module '{module_name}' not importable")

    def test_synth_ai_core_imports(self):
        """Verify synth_ai core imports are available."""
        synth_imports = [
            ("synth_ai.core.utils.env", "mint_demo_api_key"),
            ("synth_ai.core.utils.urls", "BACKEND_URL_BASE"),
            ("synth_ai.sdk.localapi", "LocalAPIConfig"),
            ("synth_ai.sdk.localapi", "create_local_api"),
            ("synth_ai.sdk.localapi._impl.contracts", "RolloutMetrics"),
            ("synth_ai.sdk.localapi._impl.contracts", "RolloutRequest"),
            ("synth_ai.sdk.localapi._impl.contracts", "RolloutResponse"),
            ("synth_ai.sdk.localapi._impl.contracts", "TaskInfo"),
            ("synth_ai.core.tunnels", "TunnelBackend"),
            ("synth_ai.core.tunnels", "TunneledLocalAPI"),
            ("synth_ai.data.enums", "SuccessStatus"),
        ]

        for module_name, attr_name in synth_imports:
            try:
                module = __import__(module_name, fromlist=[attr_name])
                assert hasattr(module, attr_name), f"'{module_name}' missing '{attr_name}'"
            except ImportError as e:
                pytest.fail(f"Failed to import '{attr_name}' from '{module_name}': {e}")

    def test_synth_ai_optimization_imports(self):
        """Verify synth_ai optimization imports are available."""
        synth_imports = [
            ("synth_ai.sdk.optimization.internal.prompt_learning", "PromptLearningJob"),
            ("synth_ai.sdk.eval.job", "EvalJob"),
            ("synth_ai.sdk.eval.job", "EvalJobConfig"),
        ]

        for module_name, attr_name in synth_imports:
            try:
                module = __import__(module_name, fromlist=[attr_name])
                assert hasattr(module, attr_name), f"'{module_name}' missing '{attr_name}'"
            except ImportError as e:
                pytest.fail(f"Failed to import '{attr_name}' from '{module_name}': {e}")
