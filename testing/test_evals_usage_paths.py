from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import httpx
from synth_ai.sdk.container.auth import ensure_container_auth
from synth_ai.sdk.pools import ContainerPoolsClient

GITHUB_ROOT = Path(__file__).resolve().parents[2]
EVALS_ROOT = GITHUB_ROOT / "evals"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_container_auth_helper_fetches_existing_backend_env_key(
    backend_url: str,
    api_key: str,
    monkeypatch,
) -> None:
    monkeypatch.delenv("ENVIRONMENT_API_KEY", raising=False)
    monkeypatch.delenv("DEV_ENVIRONMENT_API_KEY", raising=False)

    seeded = httpx.post(
        f"{backend_url}/__test__/seed-env-key",
        json={"plaintext": "env_preseeded_from_backend"},
        timeout=5.0,
    )
    seeded.raise_for_status()

    resolved = ensure_container_auth(
        backend_base=backend_url,
        synth_api_key=api_key,
        upload=True,
    )

    assert resolved == "env_preseeded_from_backend"
    assert os.environ["ENVIRONMENT_API_KEY"] == "env_preseeded_from_backend"


def test_gepa_common_rollout_container_auth_flow(
    backend_url: str,
    api_key: str,
    monkeypatch,
) -> None:
    monkeypatch.delenv("ENVIRONMENT_API_KEY", raising=False)
    monkeypatch.delenv("DEV_ENVIRONMENT_API_KEY", raising=False)

    gepa_common = _load_module(
        "evals_prompt_opt_gepa_common",
        EVALS_ROOT / "prompt_opt" / "gepa" / "common.py",
    )

    env_key = gepa_common.ensure_rollout_container_auth(
        backend_url=backend_url,
        api_key=api_key,
        container_url="http://127.0.0.1:8102",
    )

    state = httpx.get(f"{backend_url}/__test__/state", timeout=5.0)
    state.raise_for_status()
    payload = state.json()

    assert env_key
    assert os.environ["ENVIRONMENT_API_KEY"] == env_key
    assert payload["env_key_upload_count"] == 1
    upload = payload["env_key_uploads"][0]
    assert upload["name"] == "ENVIRONMENT_API_KEY"
    assert upload["ciphertext_b64"]
    assert upload["ciphertext_b64"] != env_key

    second = gepa_common.ensure_rollout_container_auth(
        backend_url=backend_url,
        api_key=api_key,
        container_url="http://127.0.0.1:8102",
    )
    state_again = httpx.get(f"{backend_url}/__test__/state", timeout=5.0)
    state_again.raise_for_status()
    assert second == env_key
    assert state_again.json()["env_key_upload_count"] == 1


def test_evals_pool_runner_helpers_work_with_rewritten_pool_client(
    backend_url: str,
    api_key: str,
) -> None:
    pool_runner = _load_module(
        "evals_containers_common_pool_runner",
        EVALS_ROOT / "containers" / "common" / "pool_runner.py",
    )
    client = ContainerPoolsClient(api_key=api_key, backend_base=backend_url)

    pool_request = {
        "pool_id": "pool-evals-runner",
        "name": "evals-runner-pool",
        "target": "harbor",
        "runtime": "tblite",
        "owner": "tests",
    }

    created = pool_runner.ensure_pool(client, dict(pool_request))
    assert created["id"] == "pool-evals-runner"

    updated = pool_runner.ensure_pool(
        client,
        {**pool_request, "owner": "updated-owner", "status": "warming"},
    )
    assert updated["owner"] == "updated-owner"
    assert updated["status"] == "warming"

    rollout = client.create_rollout(
        "pool-evals-runner",
        {"task_id": "task-1", "mode": "eval", "messages": []},
    )
    waited = pool_runner.wait_for_rollout(
        client,
        pool_id="pool-evals-runner",
        rollout_id=rollout["id"],
        timeout_s=5.0,
    )

    assert waited["id"] == rollout["id"]
    assert waited["status"] == "completed"
