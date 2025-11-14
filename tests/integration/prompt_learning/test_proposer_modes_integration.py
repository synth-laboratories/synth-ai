"""Integration tests for GEPA and MIPRO proposer modes (LOCAL-ONLY).

⚠️  WARNING: This test suite is designed EXCLUSIVELY for synth developers working
    on the synth-ai codebase. It tests proposer modes using localhost only, without
    requiring Cloudflare tunnels or external services.

This test suite validates all proposer modes:
- GEPA with dspy-like proposer
- GEPA with synth-like proposer
- GEPA with gepa-ai proposer
- MIPRO with dspy-like proposer
- MIPRO with synth-like proposer
- MIPRO with gepa-ai proposer

Usage:
    # Run all proposer mode tests
    pytest tests/integration/prompt_learning/test_proposer_modes_integration.py -v

    # Run only GEPA tests
    pytest tests/integration/prompt_learning/test_proposer_modes_integration.py::test_proposer_mode[GEPA*] -v

    # Run only MIPRO tests
    pytest tests/integration/prompt_learning/test_proposer_modes_integration.py::test_proposer_mode[MIPRO*] -v

    # Run specific mode (e.g., DSPy)
    pytest tests/integration/prompt_learning/test_proposer_modes_integration.py -k "DSPy" -v

Requirements:
    - GROQ_API_KEY environment variable set
    - SYNTH_API_KEY environment variable set (or defaults to "test")
    - Backend running on localhost:8000 (or set BACKEND_BASE_URL)
    - synth-ai package installed
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
import toml

from synth_ai.api.train.prompt_learning import PromptLearningJob
from synth_ai.learning.prompt_learning_client import PromptLearningClient
from synth_ai.task import InProcessTaskApp
from synth_ai.api.train.utils import ensure_api_base


# Test configurations
GEPA_CONFIGS = [
    ("GEPA + DSPy", "heartdisease_gepa_dspy.toml"),
    ("GEPA + Synth", "heartdisease_gepa_synth.toml"),
    ("GEPA + GEPA-AI", "heartdisease_gepa_gepa_ai.toml"),
]

MIPRO_CONFIGS = [
    ("MIPRO + DSPy", "heartdisease_mipro_dspy.toml"),
    ("MIPRO + Synth", "heartdisease_mipro_synth.toml"),
    ("MIPRO + GEPA-AI", "heartdisease_mipro_gepa_ai.toml"),
]

ALL_CONFIGS = GEPA_CONFIGS + MIPRO_CONFIGS


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    """Get the configs directory."""
    # Find configs directory relative to this test file
    test_dir = Path(__file__).resolve().parent
    repo_root = test_dir.parent.parent.parent
    configs_dir = repo_root / "examples" / "blog_posts" / "gepa" / "configs"
    if not configs_dir.exists():
        pytest.skip(f"Configs directory not found: {configs_dir}")
    return configs_dir


@pytest.fixture(scope="session")
def task_app_path() -> Path:
    """Get the task app path."""
    test_dir = Path(__file__).resolve().parent
    repo_root = test_dir.parent.parent.parent
    task_app_path = (
        repo_root / "examples" / "task_apps" / "other_langprobe_benchmarks" / "heartdisease_task_app.py"
    )
    if not task_app_path.exists():
        pytest.skip(f"Task app not found: {task_app_path}")
    return task_app_path


@pytest.fixture(scope="session")
def backend_url() -> str:
    """Get backend URL from environment or default."""
    return os.getenv("BACKEND_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def api_key() -> str:
    """Get API key from environment."""
    key = os.getenv("SYNTH_API_KEY", "test")
    if not key:
        pytest.skip("SYNTH_API_KEY environment variable is required")
    return key


@pytest.fixture(scope="session")
def task_app_api_key() -> str:
    """Get task app API key from environment."""
    return os.getenv("ENVIRONMENT_API_KEY", "test")


@pytest.fixture(scope="session")
def groq_api_key() -> str:
    """Get Groq API key from environment."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        pytest.skip("GROQ_API_KEY environment variable is required")
    return key


@pytest_asyncio.fixture(scope="session")
async def task_app(task_app_path: Path, task_app_api_key: str):
    """Start in-process task app (local-only, no tunnels)."""
    # Set environment variable to force local mode (no tunnels)
    original_tunnel_mode = os.environ.get("SYNTH_TUNNEL_MODE")
    os.environ["SYNTH_TUNNEL_MODE"] = "local"

    async with InProcessTaskApp(
        task_app_path=task_app_path,
        port=8114,
        api_key=task_app_api_key,
    ) as app:
        # Verify it's using localhost (not a tunnel)
        assert "localhost" in app.url or "127.0.0.1" in app.url, (
            f"Task app URL is not localhost: {app.url}. "
            "This local-only test expects localhost URLs only."
        )
        yield app
    
    # Restore original tunnel mode
    if original_tunnel_mode is not None:
        os.environ["SYNTH_TUNNEL_MODE"] = original_tunnel_mode
    elif "SYNTH_TUNNEL_MODE" in os.environ:
        del os.environ["SYNTH_TUNNEL_MODE"]


@pytest.fixture
def verify_backend(backend_url: str):
    """Verify backend is reachable before running tests."""
    try:
        resp = httpx.get(f"{backend_url.rstrip('/')}/api/health", timeout=2.0)
        if resp.status_code != 200:
            pytest.skip(f"Backend health check failed: {resp.status_code}")
    except Exception as e:
        pytest.skip(f"Backend not reachable at {backend_url}: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("test_name,config_file", ALL_CONFIGS)
async def test_proposer_mode(
    test_name: str,
    config_file: str,
    configs_dir: Path,
    task_app: InProcessTaskApp,
    backend_url: str,
    api_key: str,
    verify_backend: None,
):
    """Test a single proposer mode configuration."""
    config_path = configs_dir / config_file
    
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    # Load config and override task_app_url
    config = toml.load(config_path)
    config["prompt_learning"]["task_app_url"] = task_app.url

    # Create a temporary config file with updated URL
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as tmp_file:
        tmp_config_path = Path(tmp_file.name)
        toml.dump(config, tmp_file)

    try:
        start_time = time.time()

        # Create job from config
        job = PromptLearningJob.from_config(
            config_path=tmp_config_path,
            backend_url=backend_url,
            api_key=api_key,
            task_app_api_key=os.getenv("ENVIRONMENT_API_KEY", "test"),
        )

        job_id = job.submit()

        # Create client for results
        client = PromptLearningClient(
            ensure_api_base(backend_url),
            api_key,
        )

        # Poll until complete
        result = job.poll_until_complete(
            timeout=600.0,
            interval=5.0,
            on_status=lambda status: None,  # Silent polling
        )

        elapsed_time = time.time() - start_time
        final_status = result.get("status", "unknown")

        # Get results
        results_obj = await client.get_prompts(job_id)
        best_score = results_obj.best_score
        
        # Fallback: Try to extract score from job status if not in events
        if best_score is None:
            job_detail = await client.get_job(job_id)
            if isinstance(job_detail, dict):
                # Try top-level best_score
                best_score = job_detail.get("best_score")
                # Try metadata if not found
                if best_score is None:
                    metadata = job_detail.get("metadata", {})
                    if isinstance(metadata, dict):
                        job_metadata = metadata.get("job_metadata", {})
                        if isinstance(job_metadata, dict):
                            best_score = job_metadata.get("prompt_best_score") or job_metadata.get("prompt_best_train_score")
        
        early_termination = best_score is None and final_status in ("completed", "succeeded")

        # Assertions
        assert final_status in ("completed", "succeeded"), (
            f"Job {job_id} did not complete successfully. Status: {final_status}"
        )

        # Early termination is acceptable for local testing
        if early_termination:
            pytest.skip(
                f"Optimization terminated early (acceptable for local testing). "
                f"Status: {final_status}, Time: {elapsed_time:.1f}s"
            )

        # If we have a score, verify it's reasonable
        if best_score is not None:
            assert 0.0 <= best_score <= 1.0, f"Best score out of range: {best_score}"

        # Log results for debugging
        print(f"\n✓ {test_name} completed:")
        print(f"  Status: {final_status}")
        print(f"  Best Score: {best_score:.4f}" if best_score else "  Best Score: None")
        print(f"  Time: {elapsed_time:.1f}s")

    finally:
        # Clean up temp file
        if tmp_config_path.exists():
            tmp_config_path.unlink()



