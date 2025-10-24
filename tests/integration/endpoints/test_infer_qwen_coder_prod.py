import importlib
import os
from pathlib import Path

import httpx
import pytest

from synth_ai._utils.http import HTTPError


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _load_env_prod_only() -> None:
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
        # best-effort env load
        pass


def _derive_backend_base_url_prod() -> str | None:
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


async def _run_sft_example(
    module_name: str,
    *,
    backend_api: str,
    monkeypatch: pytest.MonkeyPatch,
    output_path: Path,
) -> Path:
    module = importlib.import_module(module_name)
    run_example = getattr(module, "main", None)
    if run_example is None:
        pytest.fail(f"Example module {module_name} has no main() coroutine")

    monkeypatch.setenv("BACKEND_BASE_URL", backend_api)
    monkeypatch.setenv("SYNTH_TIMEOUT", os.getenv("QWEN_CODER_SFT_TIMEOUT", "600"))
    monkeypatch.setenv("QWEN_CODER_FT_OUTPUT", str(output_path))

    try:
        await run_example()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        msg = str(exc)
        if code and ("SYNTH_API_KEY" in msg or "required" in msg):
            pytest.skip(f"SFT example requires credentials: {msg or code}")
        if code:
            raise
    except (HTTPError, httpx.HTTPStatusError) as err:
        if err.status in (401, 403):
            pytest.skip(f"SFT example requires elevated credentials: {err}")
        raise

    if not output_path.exists():
        pytest.fail(f"SFT example did not produce model id at {output_path}")

    contents = output_path.read_text(encoding="utf-8").strip()
    if not contents:
        pytest.fail("SFT example finished but model id output is empty")

    return output_path


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("variant", "infer_module", "sft_module", "default_filename", "model_override"),
    [
        (
            "base",
            "examples.qwen_coder.infer_prod_proxy",
            None,
            None,
            "Qwen/Qwen3-1.7B",
        ),
        (
            "lora",
            "examples.qwen_coder.infer_ft_smoke",
            "examples.qwen_coder.sft_lora_30b",
            "ft_model_id.txt",
            None,
        ),
        (
            "full",
            "examples.qwen_coder.infer_ft_smoke",
            "examples.qwen_coder.sft_full_17b",
            "ft_model_id_full.txt",
            None,
        ),
    ],
)
async def test_infer_qwen_coder_prod(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    variant: str,
    infer_module: str,
    sft_module: str | None,
    default_filename: str | None,
    model_override: str | None,
) -> None:
    _load_env_prod_only()

    backend_root = _derive_backend_base_url_prod()
    api_key = os.getenv("SYNTH_API_KEY")
    if not backend_root or not api_key:
        pytest.skip("SYNTH_API_KEY and PROD_BACKEND_URL (or SYNTH_BASE_URL) required for prod inference test")

    backend_root = backend_root.rstrip("/")
    backend_api = backend_root if backend_root.endswith("/api") else f"{backend_root}/api"

    require_success = os.getenv("LEARNING_TEST_REQUIRE_SUCCESS", "0") == "1"

    model_path: Path | None = None
    if sft_module:
        output_path = tmp_path / f"{variant}_ft_model_id.txt"
        model_path = await _run_sft_example(
            sft_module,
            backend_api=backend_api,
            monkeypatch=monkeypatch,
            output_path=output_path,
        )
    else:
        monkeypatch.setenv("BACKEND_BASE_URL", backend_api)

    monkeypatch.setenv("BACKEND_BASE_URL", backend_api)

    module = importlib.import_module(infer_module)
    run_infer = getattr(module, "main", None)
    if run_infer is None:
        pytest.fail(f"Inference module {infer_module} has no main() coroutine")

    if model_override:
        monkeypatch.setenv("MODEL", model_override)

    if model_path:
        monkeypatch.setenv("QWEN_CODER_FT_MODEL_PATH", str(model_path))
        output = tmp_path / f"{variant}_infer_output.txt"
        monkeypatch.setenv("QWEN_CODER_FT_INFER_OUTPUT", str(output))
        if default_filename:
            monkeypatch.setenv("QWEN_CODER_FT_FILENAME", default_filename)
            monkeypatch.setenv("QWEN_CODER_FT_INFER_FILENAME", f"{variant}_infer_smoke.txt")

    try:
        await run_infer()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        msg = str(exc)
        if code and ("SYNTH_API_KEY" in msg or "required" in msg):
            pytest.skip(f"Inference example ({variant}) requires credentials: {msg or code}")
        if code:
            raise
    except HTTPError as err:
        if err.status in (401, 403):
            pytest.skip(f"Inference example ({variant}) requires elevated credentials: {err}")
        raise
    except httpx.HTTPStatusError as err:
        status = getattr(err.response, "status_code", None)
        if status in (401, 403):
            pytest.skip(f"Inference example ({variant}) requires elevated credentials: {status}")
        raise
    except TimeoutError as exc:
        if require_success:
            raise
        pytest.xfail(f"Inference example ({variant}) timed out: {exc}")

    if model_path:
        infer_output = Path(os.getenv("QWEN_CODER_FT_INFER_OUTPUT", ""))
        if not infer_output:
            pytest.fail("Inference example did not provide output path env override")
        if not infer_output.exists():
            if require_success:
                pytest.fail(f"Inference example ({variant}) did not write output file")
            pytest.xfail(f"Inference example ({variant}) missing output file")
        else:
            contents = infer_output.read_text(encoding="utf-8").strip()
            assert contents, f"Inference example ({variant}) output file is empty"
