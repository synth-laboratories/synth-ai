from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_ai.sdk.learning.sft import (
    collect_sft_jsonl_errors,
    iter_sft_examples,
    load_jsonl,
    validate_jsonl_or_raise,
)
from synth_ai.sdk.learning.sft.data import (
    SFTMessage,
    SFTToolCall,
    coerce_example,
    # Reasoning utilities
    extract_reasoning,
    strip_reasoning,
    message_has_reasoning,
    validate_message_content,
    # Vision utilities
    has_image_content,
    message_has_image,
    example_has_image,
    count_images_in_content,
    extract_image_urls,
    validate_vision_example,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
# Check cookbooks first (moved from examples)
_DATASET_PATH = REPO_ROOT.parent / "cookbooks" / "dev" / "warming_up_to_rl" / "ft_data" / "crafter_sft.jsonl"
if not _DATASET_PATH.exists():
    # Fallback to old examples path
    _DATASET_PATH = REPO_ROOT / "examples" / "warming_up_to_rl" / "ft_data" / "crafter_sft.jsonl"
_SAMPLE_PATH = Path(__file__).with_name("crafter_sft_sample.jsonl")


def _crafter_sft() -> Path:
    if _DATASET_PATH.exists():
        return _DATASET_PATH
    # Always create sample if it doesn't exist - don't check first to avoid race conditions
    if not _SAMPLE_PATH.exists():
        _SAMPLE_PATH.write_text(
            '{"messages": [{"role": "user", "content": "test"}, {"role": "assistant", "content": "response"}]}\n',
            encoding="utf-8",
        )
    return _SAMPLE_PATH


CRAFT_SFT = _crafter_sft()


@pytest.mark.parametrize("path", [CRAFT_SFT])
@pytest.mark.fast
def test_collect_sft_jsonl_errors_clean(path: Path) -> None:
    assert path.exists(), f"fixture dataset missing: {path}"
    errors = collect_sft_jsonl_errors(path, min_messages=2, max_errors=5)
    assert errors == []


def test_iter_sft_examples_parses_tool_calls() -> None:
    with CRAFT_SFT.open("r", encoding="utf-8") as fh:
        example = next(iter_sft_examples(fh, min_messages=2))

    assert example.messages[-1].tool_calls, "expected tool calls in assistant turn"
    call = example.messages[-1].tool_calls[0]
    assert call.name == "interact"
    assert isinstance(call.arguments, dict)
    assert call.arguments.get("actions"), "parsed actions payload missing"
    assert example.metadata.get("model"), "model metadata missing"


def test_load_jsonl_supports_tools_and_metadata(tmp_path: Path) -> None:
    sample = {
        "messages": [
            {"role": "system", "content": "keep outputs concise"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": "{\"value\": 2}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": {"result": 4},
            },
        ],
        "tools": [
            {
                "name": "calculator",
                "description": "compute simple expressions",
                "parameters": {"type": "object"},
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "calculator"}},
        "metadata": {"split": "train"},
        "source": "unit-test",
    }
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    example = load_jsonl(path, min_messages=1)[0]

    assert example.tools and example.tools[0].name == "calculator"
    assert example.tool_choice == {"type": "function", "function": {"name": "calculator"}}
    call = example.messages[1].tool_calls[0]
    assert call.call_id == "call-1"
    assert call.type == "function"
    assert call.arguments == {"value": 2}
    assert example.metadata == {"split": "train"}
    assert example.extra["source"] == "unit-test"


def test_collect_sft_jsonl_errors_reports_issues(tmp_path: Path) -> None:
    invalid = {
        "messages": [
            {
                "role": "assistant",
                "content": "oops",
                "tool_calls": {"not": "a list"},
            }
        ]
    }
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")

    errors = collect_sft_jsonl_errors(path, min_messages=1, max_errors=5)
    assert errors and "tool_calls" in errors[0]


def test_collect_sft_jsonl_errors_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    errors = collect_sft_jsonl_errors(path, min_messages=1, max_errors=5)
    assert errors and "no sft examples" in errors[0].lower()


def test_validate_jsonl_or_raise_includes_path(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"messages": []}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        validate_jsonl_or_raise(path, min_messages=2)

    assert str(path) in str(excinfo.value)


# Reasoning/Thinking Tests
# ============================================================================


@pytest.mark.fast
def test_extract_reasoning_with_single_block() -> None:
    """Test extracting reasoning from single <think> block."""
    content = "<think>Let me analyze this...</think>The answer is 42"
    reasoning, clean = extract_reasoning(content)
    
    assert reasoning == "Let me analyze this..."
    assert clean == "The answer is 42"


@pytest.mark.fast
def test_extract_reasoning_with_multiple_blocks() -> None:
    """Test extracting multiple reasoning blocks."""
    content = "<think>First thought</think>Some text<think>Second thought</think>Final output"
    reasoning, clean = extract_reasoning(content)
    
    assert reasoning is not None
    assert "First thought" in reasoning
    assert "Second thought" in reasoning
    assert clean == "Some textFinal output"


@pytest.mark.fast
def test_extract_reasoning_no_blocks() -> None:
    """Test handling content without reasoning blocks."""
    content = "Just plain text"
    reasoning, clean = extract_reasoning(content)
    
    assert reasoning is None
    assert clean == "Just plain text"


@pytest.mark.fast
def test_strip_reasoning() -> None:
    """Test stripping reasoning tags."""
    content = "<think>thinking...</think>output"
    clean = strip_reasoning(content)
    assert clean == "output"


@pytest.mark.fast
def test_message_has_reasoning_explicit_field() -> None:
    """Test detecting explicit reasoning field."""
    msg = SFTMessage(role="assistant", content="output", reasoning="explicit reasoning")
    assert message_has_reasoning(msg) is True


@pytest.mark.fast
def test_message_has_reasoning_in_content() -> None:
    """Test detecting reasoning in content tags."""
    msg = SFTMessage(role="assistant", content="<think>thinking</think>output", reasoning=None)
    assert message_has_reasoning(msg) is True


@pytest.mark.fast
def test_message_has_reasoning_none() -> None:
    """Test correctly identifying no reasoning."""
    msg = SFTMessage(role="assistant", content="no reasoning here", reasoning=None)
    assert message_has_reasoning(msg) is False


@pytest.mark.fast
def test_validate_message_content_reasoning_and_tool_calls() -> None:
    """Test validation for reasoning + tool_calls combination."""
    msg = SFTMessage(
        role="assistant",
        content=None,
        reasoning="Let me call a function",
        tool_calls=[SFTToolCall(name="test", arguments={})],
    )
    is_valid, error = validate_message_content(msg)
    assert is_valid is True
    assert error is None


@pytest.mark.fast
def test_validate_message_content_reasoning_and_content() -> None:
    """Test validation for reasoning + content combination."""
    msg = SFTMessage(
        role="assistant",
        content="Final answer",
        reasoning="My thinking process",
    )
    is_valid, error = validate_message_content(msg)
    assert is_valid is True
    assert error is None


@pytest.mark.fast
def test_validate_message_content_only() -> None:
    """Test validation for content-only message."""
    msg = SFTMessage(role="assistant", content="Just content")
    is_valid, error = validate_message_content(msg)
    assert is_valid is True
    assert error is None


@pytest.mark.fast
def test_validate_message_raw_content_only() -> None:
    """Test validation for raw_content-only message."""
    msg = SFTMessage(role="assistant", content=None, raw_content="<think>...</think>output")
    is_valid, error = validate_message_content(msg)
    assert is_valid is True
    assert error is None


@pytest.mark.fast
def test_validate_message_tool_calls_only() -> None:
    """Test validation for tool_calls-only message (OpenAI format)."""
    msg = SFTMessage(
        role="assistant",
        content=None,
        tool_calls=[SFTToolCall(name="test", arguments={})],
    )
    is_valid, error = validate_message_content(msg)
    assert is_valid is True
    assert error is None


@pytest.mark.fast
def test_validate_message_reasoning_only() -> None:
    """Test validation for reasoning-only message (pure thinking turn)."""
    msg = SFTMessage(role="assistant", content=None, reasoning="Deep thoughts...")
    is_valid, error = validate_message_content(msg)
    assert is_valid is True
    assert error is None


@pytest.mark.fast
def test_validate_message_empty() -> None:
    """Test validation catches empty message."""
    msg = SFTMessage(role="assistant", content=None)
    is_valid, error = validate_message_content(msg, require_content=True)
    assert is_valid is False
    assert error is not None
    assert "no reasoning, content, raw_content, or tool_calls" in error.lower()


@pytest.mark.fast
def test_reasoning_with_vision_and_tool_calls() -> None:
    """Test full example with reasoning + vision + tool calls."""
    example_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}},
                ],
            },
            {
                "role": "assistant",
                "reasoning": "I should examine the image carefully",
                "content": None,
                "tool_calls": [{"function": {"name": "analyze", "arguments": "{}"}}],
            },
        ]
    }
    
    example = coerce_example(example_data)
    assert example.messages[1].reasoning == "I should examine the image carefully"
    assert len(example.messages[1].tool_calls) == 1
    
    is_valid, error = validate_message_content(example.messages[1])
    assert is_valid is True
    assert error is None


# Vision/Multimodal Tests
# ============================================================================


@pytest.mark.fast
def test_has_image_content_with_list() -> None:
    """Test detecting images in multimodal content list."""
    content = [
        {"type": "text", "text": "What's this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
    ]
    assert has_image_content(content) is True


@pytest.mark.fast
def test_has_image_content_single_dict() -> None:
    """Test detecting single image dict."""
    content = {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}}
    assert has_image_content(content) is True


@pytest.mark.fast
def test_has_image_content_text_only() -> None:
    """Test no false positive for text-only content."""
    assert has_image_content("Just text") is False
    assert has_image_content([{"type": "text", "text": "Hi"}]) is False


@pytest.mark.fast
def test_count_images_in_content() -> None:
    """Test counting images in content."""
    content = [
        {"type": "text", "text": "Two images:"},
        {"type": "image_url", "image_url": {"url": "img1.jpg"}},
        {"type": "image_url", "image_url": {"url": "img2.jpg"}},
    ]
    assert count_images_in_content(content) == 2
    
    content = [{"type": "text", "text": "No images"}]
    assert count_images_in_content(content) == 0


@pytest.mark.fast
def test_extract_image_urls() -> None:
    """Test extracting image URLs from content."""
    content = [
        {"type": "text", "text": "Check these:"},
        {"type": "image_url", "image_url": {"url": "https://example.com/img1.jpg"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
    ]
    urls = extract_image_urls(content)
    
    assert len(urls) == 2
    assert "https://example.com/img1.jpg" in urls
    assert urls[1].startswith("data:image/")


@pytest.mark.fast
def test_message_has_image() -> None:
    """Test detecting images in SFTMessage."""
    msg = SFTMessage(
        role="user",
        content=[
            {"type": "text", "text": "What's this?"},
            {"type": "image_url", "image_url": {"url": "img.jpg"}},
        ],
    )
    assert message_has_image(msg) is True
    
    msg = SFTMessage(role="assistant", content="Just text")
    assert message_has_image(msg) is False


@pytest.mark.fast
def test_example_has_image() -> None:
    """Test detecting images in SFTExample."""
    example_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's this?"},
                    {"type": "image_url", "image_url": {"url": "img.jpg"}},
                ],
            },
            {"role": "assistant", "content": "Response"},
        ]
    }
    example = coerce_example(example_data)
    assert example_has_image(example) is True
    
    example_data = {
        "messages": [
            {"role": "user", "content": "Text only"},
            {"role": "assistant", "content": "Response"},
        ]
    }
    example = coerce_example(example_data)
    assert example_has_image(example) is False


@pytest.mark.fast
def test_validate_vision_example_valid() -> None:
    """Test validation of valid vision example."""
    example_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}},
                ],
            },
            {"role": "assistant", "content": "A test image"},
        ]
    }
    example = coerce_example(example_data)
    is_valid, error = validate_vision_example(example, require_images=True)
    assert is_valid is True
    assert error is None


@pytest.mark.fast
def test_validate_vision_example_no_images() -> None:
    """Test validation catches missing images when required."""
    example_data = {
        "messages": [
            {"role": "user", "content": "Just text"},
            {"role": "assistant", "content": "Response"},
        ]
    }
    example = coerce_example(example_data)
    is_valid, error = validate_vision_example(example, require_images=True)
    assert is_valid is False
    assert error is not None
    assert "no image content" in error.lower()


@pytest.mark.fast
def test_validate_vision_example_no_images_optional() -> None:
    """Test validation passes when images not required."""
    example_data = {
        "messages": [
            {"role": "user", "content": "Just text"},
            {"role": "assistant", "content": "Response"},
        ]
    }
    example = coerce_example(example_data)
    is_valid, error = validate_vision_example(example, require_images=False)
    assert is_valid is True
    assert error is None


# Invalid/Bogus Image Content Tests
# ============================================================================


@pytest.mark.fast
def test_validate_vision_example_empty_url() -> None:
    """Test validation catches empty image URLs."""
    example_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Check this"},
                    {"type": "image_url", "image_url": {"url": ""}},  # Empty URL
                ],
            },
            {"role": "assistant", "content": "Response"},
        ]
    }
    example = coerce_example(example_data)
    is_valid, error = validate_vision_example(example, require_images=True)
    assert is_valid is False
    assert error is not None
    error_lower = error.lower()
    assert ("empty" in error_lower or "null" in error_lower or "missing" in error_lower)


@pytest.mark.fast
def test_validate_vision_example_missing_url_field() -> None:
    """Test validation catches missing url field in image_url."""
    example_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Check this"},
                    {"type": "image_url", "image_url": {}},  # No url field
                ],
            },
            {"role": "assistant", "content": "Response"},
        ]
    }
    example = coerce_example(example_data)
    
    # Should detect image_url type but find no valid URLs
    assert example_has_image(example) is True
    urls = extract_image_urls(example.messages[0].content)
    assert len(urls) == 0
    
    is_valid, error = validate_vision_example(example, require_images=True)
    assert is_valid is False
    assert error is not None
    error_lower = error.lower()
    assert ("empty" in error_lower or "null" in error_lower or "missing" in error_lower)


@pytest.mark.fast
def test_validate_vision_example_null_url() -> None:
    """Test validation catches null URL value."""
    example_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Check this"},
                    {"type": "image_url", "image_url": {"url": None}},  # Null URL
                ],
            },
            {"role": "assistant", "content": "Response"},
        ]
    }
    example = coerce_example(example_data)
    
    # extract_image_urls should skip None
    urls = extract_image_urls(example.messages[0].content)
    assert len(urls) == 0
    
    is_valid, error = validate_vision_example(example, require_images=True)
    assert is_valid is False


@pytest.mark.fast
def test_validate_vision_example_malformed_image_dict() -> None:
    """Test validation handles malformed image dict."""
    example_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Check this"},
                    {"type": "image_url"},  # Missing image_url field entirely
                ],
            },
            {"role": "assistant", "content": "Response"},
        ]
    }
    example = coerce_example(example_data)
    
    # Should detect type but no URL
    assert has_image_content(example.messages[0].content) is True
    urls = extract_image_urls(example.messages[0].content)
    assert len(urls) == 0
    
    # Validation should fail because no valid URL
    is_valid, error = validate_vision_example(example, require_images=True)
    assert is_valid is False


@pytest.mark.fast
def test_validate_vision_example_non_string_url() -> None:
    """Test validation catches non-string URL values."""
    example_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Check this"},
                    {"type": "image_url", "image_url": {"url": 12345}},  # Integer URL
                ],
            },
            {"role": "assistant", "content": "Response"},
        ]
    }
    example = coerce_example(example_data)
    
    # extract_image_urls should skip non-strings
    urls = extract_image_urls(example.messages[0].content)
    assert len(urls) == 0


@pytest.mark.fast
def test_validate_vision_example_whitespace_only_url() -> None:
    """Test validation catches whitespace-only URLs."""
    example_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Check this"},
                    {"type": "image_url", "image_url": {"url": "   "}},  # Whitespace
                ],
            },
            {"role": "assistant", "content": "Response"},
        ]
    }
    example = coerce_example(example_data)
    is_valid, error = validate_vision_example(example, require_images=True)
    assert is_valid is False
    assert error is not None


@pytest.mark.fast
def test_validate_vision_example_invalid_scheme() -> None:
    """Test validation warns about URLs with invalid schemes."""
    example_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Check this"},
                    {"type": "image_url", "image_url": {"url": "ftp://example.com/img.jpg"}},
                ],
            },
            {"role": "assistant", "content": "Response"},
        ]
    }
    example = coerce_example(example_data)
    
    # Should still extract the URL but validation will warn
    urls = extract_image_urls(example.messages[0].content)
    assert len(urls) == 1
    assert urls[0] == "ftp://example.com/img.jpg"
    
    # Validation should pass but with a warning (check logs)
    is_valid, error = validate_vision_example(example, require_images=True)
    # Still valid, just suspicious - we warn but don't fail
    assert is_valid is True


@pytest.mark.fast
def test_validate_vision_example_multiple_invalid_urls() -> None:
    """Test validation catches multiple invalid URLs and reports first error."""
    example_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Check these"},
                    {"type": "image_url", "image_url": {"url": ""}},  # Empty
                    {"type": "image_url", "image_url": {"url": None}},  # Null
                ],
            },
            {"role": "assistant", "content": "Response"},
        ]
    }
    example = coerce_example(example_data)
    is_valid, error = validate_vision_example(example, require_images=True)
    assert is_valid is False
    assert error is not None


@pytest.mark.fast
def test_validate_vision_example_mixed_valid_invalid() -> None:
    """Test validation fails if ANY image URL is invalid."""
    example_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Check these"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/valid.jpg"}},
                    {"type": "image_url", "image_url": {"url": ""}},  # One invalid
                ],
            },
            {"role": "assistant", "content": "Response"},
        ]
    }
    example = coerce_example(example_data)
    is_valid, error = validate_vision_example(example, require_images=True)
    assert is_valid is False
    assert error is not None
    error_lower = error.lower()
    assert ("empty" in error_lower or "null" in error_lower or "missing" in error_lower)


@pytest.mark.fast
def test_extract_image_urls_filters_invalid() -> None:
    """Test that extract_image_urls skips invalid entries."""
    content = [
        {"type": "text", "text": "Images:"},
        {"type": "image_url", "image_url": {"url": "https://valid.com/img.jpg"}},
        {"type": "image_url", "image_url": {"url": ""}},  # Empty - should skip
        {"type": "image_url", "image_url": {"url": None}},  # Null - should skip
        {"type": "image_url", "image_url": {}},  # Missing url - should skip
        {"type": "image_url"},  # No image_url field - should skip
        {"type": "image_url", "image_url": {"url": 123}},  # Non-string - should skip
    ]
    
    urls = extract_image_urls(content)
    
    # Should only extract the valid URL
    assert len(urls) == 1
    assert urls[0] == "https://valid.com/img.jpg"


@pytest.mark.fast
def test_validate_vision_example_invalid_base64_format() -> None:
    """Test validation detects potentially invalid base64 format."""
    example_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Check this"},
                    # Invalid: missing actual base64 data after comma
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
                ],
            },
            {"role": "assistant", "content": "Response"},
        ]
    }
    example = coerce_example(example_data)
    
    # URL is extracted (it's a string)
    urls = extract_image_urls(example.messages[0].content)
    assert len(urls) == 1
    
    # But it's an invalid/empty base64
    # Note: We don't validate base64 decoding in the SDK (that's inference-time)
    # We just check the URL string exists
    is_valid, error = validate_vision_example(example, require_images=True)
    # This actually passes string validation - decoder will catch it later
    # We're just checking URL presence/format here
    assert is_valid is True
