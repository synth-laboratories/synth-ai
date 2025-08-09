import os
import pytest


# Prod-only base URL for all tests in this folder
PROD_BASE = "https://agent-learning.onrender.com/api/v1"


def _get_api_key() -> str:
    # Prefer explicit prod key, else fallback to generic key
    api_key = os.environ.get("SYNTH_API_KEY_PROD") or os.environ.get("SYNTH_API_KEY")
    if not api_key:
        pytest.skip("SYNTH_API_KEY_PROD or SYNTH_API_KEY is required for prod API tests")
    return api_key


@pytest.fixture(scope="session")
def base_url() -> str:
    return PROD_BASE


@pytest.fixture(scope="session")
def auth_headers() -> dict:
    api_key = _get_api_key()
    return {"Authorization": f"Bearer {api_key}"}


