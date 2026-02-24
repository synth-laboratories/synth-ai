from __future__ import annotations

from synth_ai.sdk.optimization.internal import validators


def test_container_field_normalizer_does_not_accept_task_app_aliases() -> None:
    normalized = validators._normalize_container_fields(  # type: ignore[attr-defined]
        {"prompt_learning": {"task_app_url": "https://legacy.example.com"}}
    )
    prompt_learning = normalized["prompt_learning"]
    assert "task_app_url" in prompt_learning
    assert "container_url" not in prompt_learning


def test_container_field_normalizer_keeps_canonical_container_url() -> None:
    normalized = validators._normalize_container_fields(  # type: ignore[attr-defined]
        {"prompt_learning": {"container_url": "https://canonical.example.com"}}
    )
    assert normalized["prompt_learning"]["container_url"] == "https://canonical.example.com"
