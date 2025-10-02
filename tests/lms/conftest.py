import os
import pytest

from synth_ai.config.base_url import get_backend_from_env


def pytest_addoption(parser):
    """Add command line options for test configuration."""
    parser.addoption(
        "--env-target",
        action="store",
        default="dev",
        choices=["dev", "prod"],
        help="Target environment for tests (dev or prod)"
    )


def _choose_backend_and_key(target: str) -> tuple[str, str]:
    target = (target or "dev").lower()
    if target == "prod":
        base = (
            os.getenv("PROD_BACKEND_URL")
            or os.getenv("BACKEND_URL")
            or os.getenv("LEARNING_V2_BASE_URL")
            or os.getenv("PROD_API_URL")
            or os.getenv("API_URL")
        )
        key = os.getenv("PROD_SYNTH_API_KEY") or os.getenv("SYNTH_API_KEY")
    else:
        base = (
            os.getenv("DEV_BACKEND_URL")
            or os.getenv("BACKEND_URL")
            or os.getenv("LEARNING_V2_BASE_URL")
            or os.getenv("DEV_API_URL")
            or os.getenv("API_URL")
        )
        key = os.getenv("SYNTH_API_KEY")

    if not base or not key:
        env_base, env_key = get_backend_from_env()
        base = base or env_base
        key = key or env_key
    if not base or not key:
        pytest.skip("Missing backend URL or API key in environment; see tests/lms/qwen3.txt")
    return base.rstrip("/"), key


@pytest.fixture(scope="session")
def test_target(pytestconfig) -> str:
    # Get the environment target from command line or environment variable
    env_target = os.getenv("TEST_TARGET")
    if env_target:
        return env_target

    # Get from pytest command line option
    return pytestconfig.getoption("--env-target", default="dev")


@pytest.fixture(scope="session")
def backend_base_url(test_target: str) -> str:
    base, _ = _choose_backend_and_key(test_target)
    return base


@pytest.fixture(scope="session")
def synth_api_key(test_target: str) -> str:
    _, key = _choose_backend_and_key(test_target)
    return key


@pytest.fixture(scope="session")
def auth_headers(synth_api_key: str) -> dict:
    return {"Authorization": f"Bearer {synth_api_key}"}
