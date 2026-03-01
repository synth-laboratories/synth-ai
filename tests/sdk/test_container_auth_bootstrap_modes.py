from __future__ import annotations

from pathlib import Path

import pytest

from synth_ai.sdk.optimization.internal import builders as builders_module
from synth_ai.sdk.optimization.internal import prompt_learning as prompt_learning_module


def _stub_resolve(
    _name: str,
    *,
    cli_value: str | None = None,
    env_value: str | None = None,
    config_value: str | None = None,
    default: str | None = None,
    required: bool = False,
    docs_url: str | None = None,
) -> str | None:
    del docs_url
    value = cli_value or env_value or config_value or default
    if required and not value:
        raise ValueError("missing required value")
    return value


def test_prompt_learning_config_skips_env_key_bootstrap_when_signer_present(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _ensure(
        backend_base: str | None = None,
        synth_api_key: str | None = None,
        *,
        upload: bool = True,
        persist: bool | None = None,
    ) -> str:
        calls.append(
            {
                "backend_base": backend_base,
                "synth_api_key": synth_api_key,
                "upload": upload,
                "persist": persist,
            }
        )
        return "env-local"

    monkeypatch.setattr(prompt_learning_module, "has_container_token_signing_key", lambda: True)
    monkeypatch.setattr(prompt_learning_module, "ensure_container_auth", _ensure)

    cfg = prompt_learning_module.PromptLearningJobConfig(
        config_dict={"prompt_learning": {"container_url": "http://127.0.0.1:9001"}},
        backend_url="http://127.0.0.1:8080",
        api_key="sk_test",
    )

    assert cfg.container_api_key is None
    assert calls == []


def test_prompt_learning_config_skips_backend_key_bootstrap_without_signer(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _ensure(
        backend_base: str | None = None,
        synth_api_key: str | None = None,
        *,
        upload: bool = True,
        persist: bool | None = None,
    ) -> str:
        calls.append(
            {
                "backend_base": backend_base,
                "synth_api_key": synth_api_key,
                "upload": upload,
                "persist": persist,
            }
        )
        return "env-backend"

    monkeypatch.setattr(prompt_learning_module, "has_container_token_signing_key", lambda: False)
    monkeypatch.setattr(prompt_learning_module, "ensure_container_auth", _ensure)

    cfg = prompt_learning_module.PromptLearningJobConfig(
        config_dict={"prompt_learning": {"container_url": "http://127.0.0.1:9001"}},
        backend_url="http://127.0.0.1:8080",
        api_key="sk_test",
    )

    assert cfg.container_api_key is None
    assert calls == []


def test_prompt_learning_config_mipro_bootstraps_backend_key_with_signer(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _ensure(
        backend_base: str | None = None,
        synth_api_key: str | None = None,
        *,
        upload: bool = True,
        persist: bool | None = None,
    ) -> str:
        calls.append(
            {
                "backend_base": backend_base,
                "synth_api_key": synth_api_key,
                "upload": upload,
                "persist": persist,
            }
        )
        return "env-backend"

    monkeypatch.setattr(prompt_learning_module, "has_container_token_signing_key", lambda: True)
    monkeypatch.setattr(prompt_learning_module, "ensure_container_auth", _ensure)

    cfg = prompt_learning_module.PromptLearningJobConfig(
        config_dict={"prompt_learning": {"algorithm": "mipro", "container_url": "http://127.0.0.1:9001"}},
        backend_url="http://127.0.0.1:8080",
        api_key="sk_test",
    )

    assert cfg.container_api_key == "env-backend"
    assert calls == [
        {
            "backend_base": "http://127.0.0.1:8080",
            "synth_api_key": "sk_test",
            "upload": True,
            "persist": None,
        }
    ]


def test_prompt_learning_config_requires_signer_for_gepa_synthtunnel(monkeypatch) -> None:
    monkeypatch.setattr(prompt_learning_module, "has_container_token_signing_key", lambda: False)

    with pytest.raises(ValueError, match="GEPA SynthTunnel rollout auth requires"):
        prompt_learning_module.PromptLearningJobConfig(
            config_dict={
                "prompt_learning": {"algorithm": "gepa", "container_url": "https://st.usesynth.ai/s/rt_test"}
            },
            backend_url="http://127.0.0.1:8080",
            api_key="sk_test",
            container_worker_token="worker_token",
        )


def test_prompt_learning_config_requires_signer_for_gepa_non_local_url(monkeypatch) -> None:
    monkeypatch.setattr(prompt_learning_module, "has_container_token_signing_key", lambda: False)

    with pytest.raises(ValueError, match="non-local container_url requires"):
        prompt_learning_module.PromptLearningJobConfig(
            config_dict={
                "prompt_learning": {"algorithm": "gepa", "container_url": "https://eval.example.com/task"}
            },
            backend_url="http://127.0.0.1:8080",
            api_key="sk_test",
        )


class _StubGepaEvaluation:
    train_seeds = [0, 1]
    val_seeds = [2, 3]


class _StubGepaSection:
    evaluation = _StubGepaEvaluation()


class _StubPromptLearningConfig:
    algorithm = "gepa"
    container_url = "http://127.0.0.1:9001"
    container_id = None
    gepa = _StubGepaSection()
    mipro = object()

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_learning": {
                "algorithm": "gepa",
                "container_url": "http://127.0.0.1:9001",
                "policy": {"provider": "openai", "model": "gpt-4o-mini"},
                "gepa": {"evaluation": {"train_seeds": [0, 1], "val_seeds": [2, 3]}},
            }
        }


class _StubPromptLearningConfigForUrl:
    container_id = None
    gepa = _StubGepaSection()
    mipro = object()

    def __init__(self, container_url: str, *, algorithm: str = "gepa") -> None:
        self.algorithm = algorithm
        self.container_url = container_url

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_learning": {
                "algorithm": self.algorithm,
                "container_url": self.container_url,
                "policy": {"provider": "openai", "model": "gpt-4o-mini"},
                "gepa": {"evaluation": {"train_seeds": [0, 1], "val_seeds": [2, 3]}},
            }
        }


class _StubMiproPromptLearningConfig:
    algorithm = "mipro"
    container_url = "http://127.0.0.1:9001"
    container_id = None
    gepa = None
    mipro = object()

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_learning": {
                "algorithm": "mipro",
                "container_url": "http://127.0.0.1:9001",
                "policy": {"provider": "openai", "model": "gpt-4o-mini"},
                "mipro": {
                    "mode": "online",
                    "bootstrap_train_seeds": [0, 1],
                    "val_seeds": [2, 3],
                },
            }
        }


def test_builder_mapping_skips_env_key_bootstrap_when_signer_present(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _ensure(
        backend_base: str | None = None,
        synth_api_key: str | None = None,
        *,
        upload: bool = True,
        persist: bool | None = None,
    ) -> str:
        calls.append(
            {
                "backend_base": backend_base,
                "synth_api_key": synth_api_key,
                "upload": upload,
                "persist": persist,
            }
        )
        return "env-local"

    monkeypatch.setattr(builders_module, "has_container_token_signing_key", lambda: True)
    monkeypatch.setattr(builders_module, "ensure_container_auth", _ensure)
    monkeypatch.setattr(
        builders_module,
        "_normalize_mipro_section",
        lambda _pl_cfg, _cfg, source, prefer_model: None,
    )

    import synth_ai.sdk.optimization.internal.validators as validators_module

    monkeypatch.setattr(validators_module, "validate_prompt_learning_config", lambda *_a, **_k: None)
    monkeypatch.setattr(
        builders_module.PromptLearningConfig,
        "from_mapping",
        classmethod(lambda cls, _raw: _StubPromptLearningConfig()),
    )
    monkeypatch.setattr(
        builders_module.ConfigResolver,
        "resolve",
        staticmethod(_stub_resolve),
    )

    result = builders_module.build_prompt_learning_payload_from_mapping(
        raw_config={"prompt_learning": {"algorithm": "gepa", "container_url": "http://127.0.0.1:9001"}},
        task_url=None,
        overrides={},
        source_label="test",
    )
    assert result.task_url == "http://127.0.0.1:9001"
    assert calls == []


def test_builder_mapping_skips_bootstrap_when_signer_missing(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _ensure(
        backend_base: str | None = None,
        synth_api_key: str | None = None,
        *,
        upload: bool = True,
        persist: bool | None = None,
    ) -> str:
        calls.append(
            {
                "backend_base": backend_base,
                "synth_api_key": synth_api_key,
                "upload": upload,
                "persist": persist,
            }
        )
        return "env-backend"

    monkeypatch.setattr(builders_module, "has_container_token_signing_key", lambda: False)
    monkeypatch.setattr(builders_module, "ensure_container_auth", _ensure)
    monkeypatch.setattr(
        builders_module,
        "_normalize_mipro_section",
        lambda _pl_cfg, _cfg, source, prefer_model: None,
    )

    import synth_ai.sdk.optimization.internal.validators as validators_module

    monkeypatch.setattr(validators_module, "validate_prompt_learning_config", lambda *_a, **_k: None)
    monkeypatch.setattr(
        builders_module.PromptLearningConfig,
        "from_mapping",
        classmethod(lambda cls, _raw: _StubPromptLearningConfig()),
    )
    monkeypatch.setattr(
        builders_module.ConfigResolver,
        "resolve",
        staticmethod(_stub_resolve),
    )

    result = builders_module.build_prompt_learning_payload_from_mapping(
        raw_config={"prompt_learning": {"algorithm": "gepa", "container_url": "http://127.0.0.1:9001"}},
        task_url=None,
        overrides={},
        source_label="test",
    )
    assert result.task_url == "http://127.0.0.1:9001"
    assert calls == []


def test_builder_mapping_requires_signer_for_gepa_synthtunnel(monkeypatch) -> None:
    monkeypatch.setattr(builders_module, "has_container_token_signing_key", lambda: False)

    import synth_ai.sdk.optimization.internal.validators as validators_module

    monkeypatch.setattr(validators_module, "validate_prompt_learning_config", lambda *_a, **_k: None)
    monkeypatch.setattr(
        builders_module.PromptLearningConfig,
        "from_mapping",
        classmethod(
            lambda cls, _raw: _StubPromptLearningConfigForUrl(
                "https://st.usesynth.ai/s/rt_test", algorithm="gepa"
            )
        ),
    )
    monkeypatch.setattr(builders_module.ConfigResolver, "resolve", staticmethod(_stub_resolve))

    with pytest.raises(ValueError, match="GEPA SynthTunnel rollout auth requires"):
        builders_module.build_prompt_learning_payload_from_mapping(
            raw_config={
                "prompt_learning": {
                    "algorithm": "gepa",
                    "container_url": "https://st.usesynth.ai/s/rt_test",
                }
            },
            task_url=None,
            overrides={},
            source_label="test",
        )


def test_builder_mapping_requires_signer_for_gepa_non_local_url(monkeypatch) -> None:
    monkeypatch.setattr(builders_module, "has_container_token_signing_key", lambda: False)

    import synth_ai.sdk.optimization.internal.validators as validators_module

    monkeypatch.setattr(validators_module, "validate_prompt_learning_config", lambda *_a, **_k: None)
    monkeypatch.setattr(
        builders_module.PromptLearningConfig,
        "from_mapping",
        classmethod(
            lambda cls, _raw: _StubPromptLearningConfigForUrl(
                "https://eval.example.com/task", algorithm="gepa"
            )
        ),
    )
    monkeypatch.setattr(builders_module.ConfigResolver, "resolve", staticmethod(_stub_resolve))

    with pytest.raises(ValueError, match="non-local container_url requires"):
        builders_module.build_prompt_learning_payload_from_mapping(
            raw_config={
                "prompt_learning": {
                    "algorithm": "gepa",
                    "container_url": "https://eval.example.com/task",
                }
            },
            task_url=None,
            overrides={},
            source_label="test",
        )


def test_builder_mapping_mipro_bootstraps_backend_key_with_signer(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _ensure(
        backend_base: str | None = None,
        synth_api_key: str | None = None,
        *,
        upload: bool = True,
        persist: bool | None = None,
    ) -> str:
        calls.append(
            {
                "backend_base": backend_base,
                "synth_api_key": synth_api_key,
                "upload": upload,
                "persist": persist,
            }
        )
        return "env-backend"

    monkeypatch.setattr(builders_module, "has_container_token_signing_key", lambda: True)
    monkeypatch.setattr(builders_module, "ensure_container_auth", _ensure)
    monkeypatch.setattr(
        builders_module,
        "_normalize_mipro_section",
        lambda _pl_cfg, _cfg, source, prefer_model: None,
    )

    import synth_ai.sdk.optimization.internal.validators as validators_module

    monkeypatch.setattr(validators_module, "validate_prompt_learning_config", lambda *_a, **_k: None)
    monkeypatch.setattr(
        builders_module.PromptLearningConfig,
        "from_mapping",
        classmethod(lambda cls, _raw: _StubMiproPromptLearningConfig()),
    )
    monkeypatch.setattr(
        builders_module.ConfigResolver,
        "resolve",
        staticmethod(_stub_resolve),
    )

    result = builders_module.build_prompt_learning_payload_from_mapping(
        raw_config={"prompt_learning": {"algorithm": "mipro", "container_url": "http://127.0.0.1:9001"}},
        task_url=None,
        overrides={},
        source_label="test",
    )
    assert result.task_url == "http://127.0.0.1:9001"
    assert calls == [
        {
            "backend_base": None,
            "synth_api_key": None,
            "upload": True,
            "persist": None,
        }
    ]


def test_builder_file_mode_skips_env_key_bootstrap_when_signer_present(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, object]] = []

    def _ensure(
        backend_base: str | None = None,
        synth_api_key: str | None = None,
        *,
        upload: bool = True,
        persist: bool | None = None,
    ) -> str:
        calls.append(
            {
                "backend_base": backend_base,
                "synth_api_key": synth_api_key,
                "upload": upload,
                "persist": persist,
            }
        )
        return "env-local"

    monkeypatch.setattr(builders_module, "has_container_token_signing_key", lambda: True)
    monkeypatch.setattr(builders_module, "ensure_container_auth", _ensure)
    monkeypatch.setattr(
        builders_module,
        "_normalize_mipro_section",
        lambda _pl_cfg, _cfg, source, prefer_model: None,
    )

    import synth_ai.sdk.optimization.internal.validators as validators_module
    import synth_ai.sdk.optimization.internal.configs.prompt_learning as pl_configs_module

    monkeypatch.setattr(validators_module, "validate_prompt_learning_config", lambda *_a, **_k: None)
    monkeypatch.setattr(pl_configs_module, "load_toml", lambda _path: {"prompt_learning": {}})
    monkeypatch.setattr(
        builders_module.PromptLearningConfig,
        "from_path",
        classmethod(lambda cls, _path: _StubPromptLearningConfig()),
    )
    monkeypatch.setattr(
        builders_module.ConfigResolver,
        "resolve",
        staticmethod(_stub_resolve),
    )

    config_path = tmp_path / "cfg.toml"
    config_path.write_text("[prompt_learning]\nalgorithm='gepa'\n", encoding="utf-8")

    result = builders_module.build_prompt_learning_payload(
        config_path=Path(config_path),
        task_url=None,
        overrides={},
    )
    assert result.task_url == "http://127.0.0.1:9001"
    assert calls == []


def test_builder_file_mode_skips_bootstrap_when_signer_missing(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, object]] = []

    def _ensure(
        backend_base: str | None = None,
        synth_api_key: str | None = None,
        *,
        upload: bool = True,
        persist: bool | None = None,
    ) -> str:
        calls.append(
            {
                "backend_base": backend_base,
                "synth_api_key": synth_api_key,
                "upload": upload,
                "persist": persist,
            }
        )
        return "env-backend"

    monkeypatch.setattr(builders_module, "has_container_token_signing_key", lambda: False)
    monkeypatch.setattr(builders_module, "ensure_container_auth", _ensure)
    monkeypatch.setattr(
        builders_module,
        "_normalize_mipro_section",
        lambda _pl_cfg, _cfg, source, prefer_model: None,
    )

    import synth_ai.sdk.optimization.internal.validators as validators_module
    import synth_ai.sdk.optimization.internal.configs.prompt_learning as pl_configs_module

    monkeypatch.setattr(validators_module, "validate_prompt_learning_config", lambda *_a, **_k: None)
    monkeypatch.setattr(pl_configs_module, "load_toml", lambda _path: {"prompt_learning": {}})
    monkeypatch.setattr(
        builders_module.PromptLearningConfig,
        "from_path",
        classmethod(lambda cls, _path: _StubPromptLearningConfig()),
    )
    monkeypatch.setattr(
        builders_module.ConfigResolver,
        "resolve",
        staticmethod(_stub_resolve),
    )

    config_path = tmp_path / "cfg.toml"
    config_path.write_text("[prompt_learning]\nalgorithm='gepa'\n", encoding="utf-8")

    result = builders_module.build_prompt_learning_payload(
        config_path=Path(config_path),
        task_url=None,
        overrides={},
    )
    assert result.task_url == "http://127.0.0.1:9001"
    assert calls == []


