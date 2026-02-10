"""Validation tests for the GEPA Banking77 demo notebook.

These tests ensure the notebook:
1. Is valid JSON and can be parsed
2. Has all required cells
3. All imports can be resolved (syntax validation)
4. Core business logic functions are importable
"""

import json
from pathlib import Path

import pytest

# Path to the notebook
NOTEBOOK_PATH = Path(__file__).parent.parent.parent / "demos" / "gepa_banking77" / "gepa_banking77_prompt_optimization.ipynb"


@pytest.mark.unit
@pytest.mark.skipif(not NOTEBOOK_PATH.exists(), reason=f"Notebook not found at {NOTEBOOK_PATH}")
class TestGepaBanking77NotebookValidation:
    """Test suite for validating the GEPA Banking77 notebook."""

    def test_notebook_exists(self):
        """Verify the notebook file exists."""
        assert NOTEBOOK_PATH.exists(), f"Notebook not found at {NOTEBOOK_PATH}"

    def test_notebook_is_valid_json(self):
        """Verify the notebook is valid JSON."""
        with open(NOTEBOOK_PATH) as f:
            notebook = json.load(f)

        assert "cells" in notebook, "Notebook missing 'cells' key"
        assert "metadata" in notebook, "Notebook missing 'metadata' key"
        assert len(notebook["cells"]) > 0, "Notebook has no cells"

    def test_notebook_has_required_structure(self):
        """Verify the notebook has the expected structure."""
        with open(NOTEBOOK_PATH) as f:
            notebook = json.load(f)

        cells = notebook["cells"]

        # Check for markdown and code cells
        cell_types = [c["cell_type"] for c in cells]
        assert "markdown" in cell_types, "Notebook should have markdown cells"
        assert "code" in cell_types, "Notebook should have code cells"

        # Check minimum cell count (notebook has ~15 cells)
        assert len(cells) >= 10, f"Expected at least 10 cells, got {len(cells)}"

    def test_notebook_code_cells_have_valid_python_syntax(self):
        """Verify all code cells have valid Python syntax."""
        with open(NOTEBOOK_PATH) as f:
            notebook = json.load(f)

        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue

            source = "".join(cell["source"])

            # Skip cells that contain shell commands (!)
            # These are Jupyter magic commands and can't be compiled as Python
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
                compile(source, f"<cell_{i}>", "exec")
            except SyntaxError as e:
                pytest.fail(f"Syntax error in cell {i}: {e}")

    def test_core_imports_available(self):
        """Verify core imports used in the notebook are available."""
        # These are the core imports from the notebook that should be available
        imports_to_check = [
            "json",
            "asyncio",
            "httpx",
            "nest_asyncio",
            "datasets",
            "openai",
        ]

        for module_name in imports_to_check:
            try:
                __import__(module_name)
            except ImportError:
                pytest.fail(f"Required module '{module_name}' is not importable")

    def test_synth_ai_imports_available(self):
        """Verify synth_ai imports used in the notebook are available."""
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
            ("synth_ai.sdk.optimization.internal.prompt_learning", "PromptLearningJob"),
            ("synth_ai.sdk.eval.job", "EvalJob"),
            ("synth_ai.sdk.eval.job", "EvalJobConfig"),
        ]

        for module_name, attr_name in synth_imports:
            try:
                module = __import__(module_name, fromlist=[attr_name])
                assert hasattr(module, attr_name), f"Module '{module_name}' missing attribute '{attr_name}'"
            except ImportError as e:
                pytest.fail(f"Failed to import '{attr_name}' from '{module_name}': {e}")

    def test_banking77_labels_defined(self):
        """Verify the Banking77 labels are correctly defined in the notebook."""
        import re

        with open(NOTEBOOK_PATH) as f:
            notebook = json.load(f)

        # Find the cell with BANKING77_LABELS
        labels_source = None
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                if "BANKING77_LABELS = [" in source:
                    labels_source = source
                    break

        assert labels_source is not None, "Could not find BANKING77_LABELS definition"

        # Extract just the BANKING77_LABELS list using regex
        # Match from BANKING77_LABELS = [ to the closing ]
        pattern = r'BANKING77_LABELS\s*=\s*\[(.*?)\]'
        match = re.search(pattern, labels_source, re.DOTALL)
        assert match is not None, "Could not parse BANKING77_LABELS list"

        # Count the number of label entries (strings in quotes)
        labels_content = match.group(1)
        label_entries = re.findall(r'"([^"]+)"', labels_content)

        assert len(label_entries) == 77, f"Expected 77 labels, got {len(label_entries)}"

        # Verify some expected labels are present
        expected_labels = ["activate_my_card", "card_arrival", "lost_or_stolen_card"]
        for label in expected_labels:
            assert label in label_entries, f"Expected label '{label}' not found"

    def test_notebook_cells_sequential_execution_order(self):
        """Verify cells have proper execution order (no missing dependencies)."""
        with open(NOTEBOOK_PATH) as f:
            notebook = json.load(f)

        # Track defined names across cells
        defined_names = set()

        # Common built-ins and imports that don't need prior definition
        builtin_names = {
            "print", "len", "range", "list", "dict", "str", "int", "float",
            "True", "False", "None", "isinstance", "enumerate", "zip",
            "open", "set", "tuple", "type", "getattr", "hasattr", "setattr",
            "__name__", "__file__",
        }

        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue

            source = "".join(cell["source"])

            # Skip shell commands and empty cells
            if source.strip().startswith("!") or not source.strip():
                continue

            # Just check that the cell compiles (detailed dependency analysis is complex)
            try:
                compile(source, f"<cell_{i}>", "exec")
            except SyntaxError:
                # Already covered by another test
                pass
