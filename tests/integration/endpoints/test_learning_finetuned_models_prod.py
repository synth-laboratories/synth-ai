import os
import pytest

from synth_ai.learning import LearningClient


pytestmark = [pytest.mark.integration]


def _load_env_prod_only() -> None:
    prod_env = os.path.join(os.getcwd(), ".env.test.prod")
    if os.path.exists(prod_env):
        try:
            with open(prod_env, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
        except Exception:
            # best-effort env load
            pass


def _derive_backend_base_url_prod() -> str | None:
    # Prefer explicit base url if provided
    base = os.getenv("SYNTH_BASE_URL")
    if base:
        base = base.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return base
    # Prefer PROD_BACKEND_URL, then DEV_BACKEND_URL as last resort
    prod_backend = os.getenv("PROD_BACKEND_URL")
    if prod_backend:
        return prod_backend.rstrip("/")
    dev_backend = os.getenv("DEV_BACKEND_URL")
    if dev_backend:
        return dev_backend.rstrip("/")
    return None


@pytest.mark.asyncio
async def test_list_finetuned_models_prod() -> None:
    _load_env_prod_only()

    base_url = _derive_backend_base_url_prod()
    api_key = os.getenv("SYNTH_API_KEY")

    if not base_url or not api_key:
        pytest.skip("backend base URL and SYNTH_API_KEY required for prod test")

    client = LearningClient(base_url=base_url, api_key=api_key, timeout=60.0)
    models = await client.list_fine_tuned_models()

    assert isinstance(models, list)
    for m in models[:5]:
        assert isinstance(m, dict)
        if "id" in m:
            assert isinstance(m["id"], str)


