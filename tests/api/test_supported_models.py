import pytest

from synth_ai.sdk.api.models.supported import (
    CORE_MODELS,
    EXPERIMENTAL_MODELS,
    THINKING_MODELS,
    SupportedModel,
    UnsupportedModelError,
    core_model_ids,
    ensure_allowed_model,
    ensure_supported_model,
    format_supported_models,
    get_model_metadata,
    is_core_model,
    is_experimental_model,
    is_supported_model,
    list_supported_models,
    normalize_model_identifier,
    supported_model_ids,
    supports_thinking,
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


# Tests for instruct models
def test_instruct_models_supported():
    """Test that all instruct model variants are supported."""
    instruct_models = [
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-4B-Instruct-2507-FP8",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    ]
    for model_id in instruct_models:
        assert is_supported_model(model_id), f"{model_id} should be supported"


def test_thinking_models_supported():
    """Test that all thinking model variants are supported."""
    thinking_models = [
        "Qwen/Qwen3-4B-Thinking-2507",
        "Qwen/Qwen3-4B-Thinking-2507-FP8",
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
        "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
    ]
    for model_id in thinking_models:
        assert is_supported_model(model_id), f"{model_id} should be supported"


# Tests for thinking support detection
def test_supports_thinking_for_thinking_models():
    """Test that thinking models are correctly identified."""
    thinking_models = [
        "Qwen/Qwen3-4B-Thinking-2507",
        "Qwen/Qwen3-4B-Thinking-2507-FP8",
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
        "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
    ]
    for model_id in thinking_models:
        assert supports_thinking(model_id), f"{model_id} should support thinking"
        assert model_id in THINKING_MODELS


def test_supports_thinking_for_instruct_models():
    """Test that instruct models are correctly identified as NOT supporting thinking."""
    instruct_models = [
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-4B-Instruct-2507-FP8",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    ]
    for model_id in instruct_models:
        assert not supports_thinking(model_id), f"{model_id} should NOT support thinking"


def test_supports_thinking_with_prefixes():
    """Test that thinking detection works with fine-tuned prefixes."""
    assert supports_thinking("rl:Qwen/Qwen3-4B-Thinking-2507")
    assert supports_thinking("fft:Qwen/Qwen3-4B-Thinking-2507:job123")
    assert not supports_thinking("rl:Qwen/Qwen3-4B-Instruct-2507")
    assert not supports_thinking("fft:Qwen/Qwen3-4B-Instruct-2507:job123")


def test_supports_thinking_for_unknown_model():
    """Test that unknown models return False for thinking support."""
    assert not supports_thinking("Unknown/Model")
    assert not supports_thinking("gpt-4")


# Tests for model metadata
def test_get_model_metadata_for_thinking_model():
    """Test that metadata correctly reflects thinking support."""
    meta = get_model_metadata("Qwen/Qwen3-4B-Thinking-2507")
    assert meta is not None
    assert meta.supports_thinking is True
    assert "rl" in meta.training_modes
    assert "sft" in meta.training_modes


def test_get_model_metadata_for_instruct_model():
    """Test that metadata correctly reflects no thinking support for instruct models."""
    meta = get_model_metadata("Qwen/Qwen3-4B-Instruct-2507")
    assert meta is not None
    assert meta.supports_thinking is False
    assert "rl" in meta.training_modes
    assert "sft" in meta.training_modes


def test_get_model_metadata_with_prefix():
    """Test that metadata works with fine-tuned prefixes."""
    meta = get_model_metadata("rl:Qwen/Qwen3-4B-Instruct-2507")
    assert meta is not None
    assert meta.model_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert meta.supports_thinking is False


def test_get_model_metadata_for_unknown_model():
    """Test that unknown models return None."""
    meta = get_model_metadata("Unknown/Model")
    assert meta is None


# Tests for RL support
def test_instruct_models_support_rl():
    """Test that instruct models are in the RL support list."""
    rl_instruct_models = [
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-4B-Instruct-2507-FP8",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    ]
    for model_id in rl_instruct_models:
        meta = get_model_metadata(model_id)
        assert meta is not None
        assert "rl" in meta.training_modes, f"{model_id} should support RL"


def test_thinking_models_support_rl():
    """Test that thinking models are in the RL support list."""
    rl_thinking_models = [
        "Qwen/Qwen3-4B-Thinking-2507",
        "Qwen/Qwen3-4B-Thinking-2507-FP8",
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
    ]
    for model_id in rl_thinking_models:
        meta = get_model_metadata(model_id)
        assert meta is not None
        assert "rl" in meta.training_modes, f"{model_id} should support RL"


# Tests for SFT support
def test_all_2507_models_support_sft():
    """Test that all 2507 models (instruct, thinking, and base) support SFT."""
    sft_models = [
        "Qwen/Qwen3-4B-2507",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-4B-Instruct-2507-FP8",
        "Qwen/Qwen3-4B-Thinking-2507",
        "Qwen/Qwen3-4B-Thinking-2507-FP8",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
    ]
    for model_id in sft_models:
        meta = get_model_metadata(model_id)
        assert meta is not None
        assert "sft" in meta.training_modes, f"{model_id} should support SFT"
