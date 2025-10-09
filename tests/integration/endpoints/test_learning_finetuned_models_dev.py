import os
import pytest

from synth_ai.learning import LearningClient


pytestmark = [pytest.mark.integration]


def _load_env_dev_first() -> None:
    env_candidates = [
        os.path.join(os.getcwd(), ".env.test.dev"),
        os.path.join(os.getcwd(), ".env.test.prod"),
        os.path.join(os.getcwd(), ".env.test"),
    ]
    for env_path in env_candidates:
        if not os.path.exists(env_path):
            continue
        try:
            with open(env_path, "r", encoding="utf-8") as f:
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


def _derive_backend_base_url() -> str | None:
    base = os.getenv("SYNTH_BASE_URL")
    if base:
        base = base.rstrip("/")
        # If /v1 is present, drop it to get backend base
        if base.endswith("/v1"):
            base = base[:-3]
        return base
    dev_backend = os.getenv("DEV_BACKEND_URL")
    if dev_backend:
        return dev_backend.rstrip("/")
    prod_backend = os.getenv("PROD_BACKEND_URL")
    if prod_backend:
        return prod_backend.rstrip("/")
    return None


@pytest.mark.asyncio
async def test_list_finetuned_models_dev() -> None:
    _load_env_dev_first()

    base_url = _derive_backend_base_url()
    api_key = os.getenv("SYNTH_API_KEY")

    if not base_url or not api_key:
        pytest.skip("backend base URL and SYNTH_API_KEY required for dev test")

    client = LearningClient(base_url=base_url, api_key=api_key, timeout=60.0)
    models = await client.list_fine_tuned_models()

    assert isinstance(models, list)
    # If present, ensure minimal schema
    for m in models[:5]:
        assert isinstance(m, dict)
        if "id" in m:
            assert isinstance(m["id"], str)


