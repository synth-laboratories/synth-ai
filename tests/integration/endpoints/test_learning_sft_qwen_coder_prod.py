import importlib
import os
from pathlib import Path

import pytest

from synth_ai._utils.http import HTTPError


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _load_env_prod_only() -> None:
    """Best-effort load of .env.test.prod for credentialed runs."""
    prod_env = Path(os.getcwd()) / ".env.test.prod"
    if not prod_env.exists():
        return
    try:
        for raw in prod_env.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    except Exception:
        # Allow tests to proceed even if the env file is malformed.
        pass


def _derive_backend_base_url_prod() -> str | None:
    """Pick the best available prod backend root (no trailing /v1)."""
    direct = os.getenv("SYNTH_BASE_URL")
    if direct:
        direct = direct.rstrip("/")
        return direct[:-3] if direct.endswith("/v1") else direct
    prod_backend = os.getenv("PROD_BACKEND_URL")
    if prod_backend:
        return prod_backend.rstrip("/")
    dev_backend = os.getenv("DEV_BACKEND_URL")
    if dev_backend:
        return dev_backend.rstrip("/")
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "default_filename", "variant"),
    [
        ("examples.qwen_coder.sft_lora_30b", "ft_model_id.txt", "lora"),
        ("examples.qwen_coder.sft_full_17b", "ft_model_id_full.txt", "full"),
    ],
)
async def test_sft_qwen_coder_prod(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module_name: str,
    default_filename: str,
    variant: str,
) -> None:
    """Invoke the Qwen coder SFT examples (LoRA + full) and assert they complete against prod backend."""
    _load_env_prod_only()

    backend_root = _derive_backend_base_url_prod()
    api_key = os.getenv("SYNTH_API_KEY")
    if not backend_root or not api_key:
        pytest.skip("SYNTH_API_KEY and PROD_BACKEND_URL (or SYNTH_BASE_URL) required for prod SFT test")

    backend_root = backend_root.rstrip("/")
    backend_api = backend_root if backend_root.endswith("/api") else f"{backend_root}/api"

    monkeypatch.setenv("BACKEND_BASE_URL", backend_api)
    monkeypatch.setenv("SYNTH_TIMEOUT", os.getenv("QWEN_CODER_SFT_TIMEOUT", "600"))

    out_path = tmp_path / f"{variant}_{default_filename}"
    monkeypatch.setenv("QWEN_CODER_FT_OUTPUT", str(out_path))

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - ensures visibility if module missing
        pytest.fail(f"Could not import coder SFT example ({variant}): {exc}")

    run_example = getattr(module, "main", None)
    if run_example is None:
        pytest.fail(f"Example module {module_name} has no main() coroutine")

    require_success = os.getenv("LEARNING_TEST_REQUIRE_SUCCESS", "0") == "1"

    try:
        await run_example()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        msg = str(exc)
        if code and ("SYNTH_API_KEY" in msg or "required" in msg):
            pytest.skip(f"Coder SFT example ({variant}) requires credentials: {msg or code}")
        if code:
            raise
    except HTTPError as err:
        if err.status in (401, 403):
            pytest.skip(f"Coder SFT example ({variant}) requires elevated credentials: {err}")
        raise
    except TimeoutError as exc:
        if require_success:
            raise
        pytest.xfail(f"Coder SFT example ({variant}) timed out before completion: {exc}")

    if not out_path.exists():
        message = f"Coder SFT example ({variant}) did not produce output file"
        if require_success:
            pytest.fail(message)
        pytest.xfail(message)

    contents = out_path.read_text(encoding="utf-8").strip()
    assert contents, f"Coder SFT example ({variant}) finished but output file is empty"
