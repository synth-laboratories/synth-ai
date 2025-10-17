import pytest

from synth_ai.api.models.supported import (
    CORE_MODELS,
    EXPERIMENTAL_MODELS,
    SupportedModel,
    UnsupportedModelError,
    core_model_ids,
    ensure_allowed_model,
    ensure_supported_model,
    format_supported_models,
    is_core_model,
    is_experimental_model,
    is_supported_model,
    list_supported_models,
    normalize_model_identifier,
    supported_model_ids,
)


EXPECTED_QWEN3 = {
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
}


def test_supported_models_include_qwen3_variants():
    models = list_supported_models()
    assert all(isinstance(model, SupportedModel) for model in models)
    model_ids = {model.model_id for model in models}
    assert EXPECTED_QWEN3 <= model_ids


def test_supported_model_ids():
    model_ids = supported_model_ids()
    assert EXPECTED_QWEN3 <= set(model_ids)


def test_format_supported_models_outputs_table():
    output = format_supported_models()
    for model_id in EXPECTED_QWEN3:
        assert model_id in output
    assert "model_id | family | provider | lifecycle" in output


def test_ensure_supported_model_accepts_base_and_ft():
    assert ensure_supported_model("Qwen/Qwen3-0.6B") == "Qwen/Qwen3-0.6B"
    assert ensure_supported_model("qwen/qwen3-0.6b") == "Qwen/Qwen3-0.6B"
    assert ensure_supported_model("ft:Qwen/Qwen3-0.6B:job_123") == "Qwen/Qwen3-0.6B"
    assert ensure_supported_model("rl:Qwen/Qwen3-0.6B:policy-1") == "Qwen/Qwen3-0.6B"


def test_ensure_supported_model_rejects_unknown():
    with pytest.raises(UnsupportedModelError):
        ensure_supported_model("Unknown/Model")


def test_normalize_model_identifier_preserves_prefixes():
    base = normalize_model_identifier("qwen/qwen3-1.7b")
    assert base == "Qwen/Qwen3-1.7B"
    ft = normalize_model_identifier("ft:Qwen/Qwen3-1.7B:job")
    assert ft == "ft:Qwen/Qwen3-1.7B:job"
    rl = normalize_model_identifier("rl:Qwen/Qwen3-1.7B:policy")
    assert rl == "rl:Qwen/Qwen3-1.7B:policy"


def test_is_supported_model_matches_helper():
    assert is_supported_model("Qwen/Qwen3-4B")
    assert not is_supported_model("gpt-4")


def test_experimental_flags_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SDK_EXPERIMENTAL", raising=False)
    assert "Qwen/Qwen3-235B-A22B-Thinking-2507" in EXPERIMENTAL_MODELS
    assert is_experimental_model("Qwen/Qwen3-235B-A22B-Thinking-2507")
    assert not is_core_model("Qwen/Qwen3-235B-A22B-Thinking-2507")
    assert is_core_model("Qwen/Qwen3-8B")
    assert not is_experimental_model("Qwen/Qwen3-8B")
    assert set(core_model_ids()) == CORE_MODELS


def test_ensure_allowed_model_blocks_without_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SDK_EXPERIMENTAL", raising=False)
    with pytest.raises(UnsupportedModelError):
        ensure_allowed_model("Qwen/Qwen3-Coder-480B-A35B-Instruct")

    assert (
        ensure_allowed_model("Qwen/Qwen3-Coder-480B-A35B-Instruct", allow_experimental=True)
        == "Qwen/Qwen3-Coder-480B-A35B-Instruct"
    )


def test_ensure_allowed_model_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SDK_EXPERIMENTAL", "1")
    assert (
        ensure_allowed_model("Qwen/Qwen3-30B-A3B-Thinking-2507")
        == "Qwen/Qwen3-30B-A3B-Thinking-2507"
    )
