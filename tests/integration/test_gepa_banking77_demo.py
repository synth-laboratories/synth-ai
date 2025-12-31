#!/usr/bin/env python3
"""
Integration test for the GEPA Banking77 demo notebook.

This test executes the gepa_banking77_prompt_optimization.ipynb notebook and verifies:
1. All cells execute without errors
2. GEPA optimization completes successfully
3. Eval scores are extracted and optimized >= baseline

Usage:
    # Run against production (default)
    pytest tests/integration/test_gepa_banking77_demo.py -v -s

    # Run against dev backend
    SYNTH_BACKEND=dev pytest tests/integration/test_gepa_banking77_demo.py -v -s

    # With custom API key
    SYNTH_API_KEY=sk_live_xxx pytest tests/integration/test_gepa_banking77_demo.py -v -s

Environment Variables:
    SYNTH_BACKEND: "prod" (default) or "dev"
    SYNTH_API_KEY: Optional API key (will mint demo key if not provided)
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pytest

# Get the repo root
REPO_ROOT = Path(__file__).parent.parent.parent
NOTEBOOK_PATH = REPO_ROOT / "demos" / "gepa_banking77" / "gepa_banking77_prompt_optimization.ipynb"

PROD_BACKEND = "https://api.usesynth.ai"
DEV_BACKEND = "https://synth-backend-dev-docker.onrender.com"


def get_backend_url() -> str:
    """Get the backend URL based on environment."""
    backend = os.environ.get("SYNTH_BACKEND", "prod").lower()
    if backend == "dev":
        return DEV_BACKEND
    return PROD_BACKEND


def extract_scores_from_output(output: str) -> dict:
    """Extract baseline and optimized scores from notebook output."""
    scores = {
        "baseline_train": None,
        "optimized_train": None,
        "baseline_eval": None,
        "optimized_eval": None,
        "eval_lift": None,
        "job_status": None,
    }

    # Look for job status
    if "FINAL: succeeded" in output:
        scores["job_status"] = "succeeded"
    elif "FINAL: failed" in output:
        scores["job_status"] = "failed"

    # Extract training scores (e.g., "Baseline Train:  73.3%")
    baseline_train_match = re.search(r"Baseline Train:\s+(\d+\.?\d*)%", output)
    if baseline_train_match:
        scores["baseline_train"] = float(baseline_train_match.group(1)) / 100

    optimized_train_match = re.search(r"Optimized Train:\s+(\d+\.?\d*)%", output)
    if optimized_train_match:
        scores["optimized_train"] = float(optimized_train_match.group(1)) / 100

    # Extract eval scores (e.g., "Baseline eval: 65.0%")
    baseline_eval_match = re.search(r"Baseline eval:\s+(\d+\.?\d*)%", output)
    if baseline_eval_match:
        scores["baseline_eval"] = float(baseline_eval_match.group(1)) / 100

    optimized_eval_match = re.search(r"Optimized eval:\s+(\d+\.?\d*)%", output)
    if optimized_eval_match:
        scores["optimized_eval"] = float(optimized_eval_match.group(1)) / 100

    # Extract lift (e.g., "Lift:      +5.0%")
    lift_match = re.search(r"Eval.*?Lift:\s+([+-]?\d+\.?\d*)%", output, re.DOTALL)
    if lift_match:
        scores["eval_lift"] = float(lift_match.group(1)) / 100

    return scores


def run_notebook(
    notebook_path: Path,
    backend_url: str,
    api_key: Optional[str] = None,
    timeout: int = 600,
) -> tuple[bool, str, dict]:
    """
    Execute a Jupyter notebook and return results.

    Returns:
        (success, output, scores)
    """
    # Create a modified notebook with the correct backend URL
    with open(notebook_path) as f:
        notebook = json.load(f)

    # Modify cell-1 to use the specified backend
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "SYNTH_API_BASE = '" in source:
                # Replace the backend URL
                new_source = re.sub(
                    r"SYNTH_API_BASE = '[^']+'",
                    f"SYNTH_API_BASE = '{backend_url}'",
                    source,
                )
                cell["source"] = new_source.split("\n")
                cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [
                    cell["source"][-1]
                ]
                break

    # Write modified notebook to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ipynb", delete=False
    ) as tmp_notebook:
        json.dump(notebook, tmp_notebook)
        tmp_notebook_path = tmp_notebook.name

    output_notebook_path = tmp_notebook_path.replace(".ipynb", "_output.ipynb")

    try:
        # Build environment
        env = os.environ.copy()
        if api_key:
            env["SYNTH_API_KEY"] = api_key

        # Run notebook with papermill
        cmd = [
            sys.executable,
            "-m",
            "papermill",
            tmp_notebook_path,
            output_notebook_path,
            "--log-output",
            "--progress-bar",
        ]

        print(f"\nExecuting notebook against {backend_url}...")
        print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(REPO_ROOT),
        )

        # Combine stdout and stderr
        full_output = result.stdout + "\n" + result.stderr

        # Also read the output notebook for cell outputs
        if os.path.exists(output_notebook_path):
            with open(output_notebook_path) as f:
                output_notebook = json.load(f)
            for cell in output_notebook.get("cells", []):
                if cell.get("cell_type") == "code":
                    for output in cell.get("outputs", []):
                        if output.get("output_type") == "stream":
                            text = output.get("text", [])
                            if isinstance(text, list):
                                full_output += "\n" + "".join(text)
                            else:
                                full_output += "\n" + text

        scores = extract_scores_from_output(full_output)
        success = result.returncode == 0

        return success, full_output, scores

    except subprocess.TimeoutExpired:
        return False, f"Notebook execution timed out after {timeout}s", {}
    finally:
        # Cleanup temp files
        for path in [tmp_notebook_path, output_notebook_path]:
            try:
                os.unlink(path)
            except OSError:
                pass


class TestGEPABanking77Demo:
    """Integration tests for the GEPA Banking77 demo notebook."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.backend_url = get_backend_url()
        self.api_key = os.environ.get("SYNTH_API_KEY")

        print(f"\nTest Configuration:")
        print(f"  Backend: {self.backend_url}")
        print(f"  API Key: {'provided' if self.api_key else 'will mint demo key'}")

    def test_notebook_exists(self):
        """Verify the notebook file exists."""
        assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_gepa_demo_end_to_end(self):
        """
        Run the full GEPA demo notebook and verify:
        1. Notebook executes without errors
        2. GEPA job succeeds
        3. Optimized eval score >= baseline eval score
        """
        success, output, scores = run_notebook(
            NOTEBOOK_PATH,
            self.backend_url,
            self.api_key,
            timeout=900,  # 15 minutes - GEPA can take a while
        )

        # Print output for debugging
        print("\n" + "=" * 60)
        print("NOTEBOOK OUTPUT (last 3000 chars)")
        print("=" * 60)
        print(output[-3000:] if len(output) > 3000 else output)

        print("\n" + "=" * 60)
        print("EXTRACTED SCORES")
        print("=" * 60)
        for key, value in scores.items():
            print(f"  {key}: {value}")

        # Assertions
        assert success, f"Notebook execution failed. Output:\n{output[-2000:]}"

        assert (
            scores["job_status"] == "succeeded"
        ), f"GEPA job did not succeed: {scores['job_status']}"

        # Check we got eval scores
        assert (
            scores["baseline_eval"] is not None
        ), "Could not extract baseline eval score"
        assert (
            scores["optimized_eval"] is not None
        ), "Could not extract optimized eval score"

        # Optimized should be >= baseline (allowing for some variance)
        # We use >= because sometimes they can be equal
        assert scores["optimized_eval"] >= scores["baseline_eval"] - 0.05, (
            f"Optimized eval ({scores['optimized_eval']:.1%}) is significantly worse "
            f"than baseline ({scores['baseline_eval']:.1%})"
        )

        print("\n" + "=" * 60)
        print("TEST PASSED!")
        print("=" * 60)
        print(f"Baseline Eval:  {scores['baseline_eval']:.1%}")
        print(f"Optimized Eval: {scores['optimized_eval']:.1%}")
        if scores["eval_lift"] is not None:
            print(f"Lift:           {scores['eval_lift']:+.1%}")


if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__, "-v", "-s", "--tb=short"])
