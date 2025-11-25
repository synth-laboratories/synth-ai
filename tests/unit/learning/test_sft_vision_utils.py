import json
import pytest

from synth_ai.sdk.learning.sft.data import (
    has_image_content,
    extract_image_urls,
    validate_vision_example,
    iter_vision_examples,
    SFTMessage,
    SFTExample,
)


def test_has_image_content_variants():
    assert has_image_content([
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "http://x/img.png"}},
    ]) is True
    assert has_image_content({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}) is True
    assert has_image_content({"type": "text", "text": "no image"}) is False
    assert has_image_content("plain text") is False


def test_extract_image_urls_filters_invalid_entries():
    content = [
        {"type": "image_url", "image_url": {"url": "https://ok/a.png"}},
        {"type": "image_url", "image_url": {"url": "   "}},
        {"type": "image_url", "image_url": {"url": None}},
        {"type": "text", "text": "ignore"},
        {"type": "image", "image": "data:image/png;base64,AAA"},
    ]
    urls = extract_image_urls(content)
    assert urls == ["https://ok/a.png", "data:image/png;base64,AAA"]


def test_validate_vision_example_detects_missing_urls():
    # Two image entries but only one valid URL
    ex = SFTExample(
        messages=[
            SFTMessage(
                role="user",
                content=[
                    {"type": "image_url", "image_url": {"url": "https://ok/a.png"}},
                    {"type": "image_url", "image_url": {"url": ""}},
                ],
            )
        ]
    )
    ok, err = validate_vision_example(ex, require_images=True)
    assert ok is False
    assert "image_url entries" in (err or "")


def test_validate_vision_example_passes_valid_urls():
    ex = SFTExample(
        messages=[
            SFTMessage(
                role="user",
                content=[
                    {"type": "image_url", "image_url": {"url": "https://ok/a.png"}},
                    {"type": "text", "text": "describe"},
                ],
            )
        ]
    )
    ok, err = validate_vision_example(ex, require_images=True)
    assert ok is True
    assert err is None


def test_iter_vision_examples_yields_only_valid(tmp_path):
    lines = [
        json.dumps({
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "https://ok/a.png"}},
                    {"type": "text", "text": "what is this"},
                ]}
            ]
        }),
        json.dumps({
            "messages": [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "   "}},
                ]}
            ]
        }),
    ]
    out = list(iter_vision_examples(lines, require_images=True))
    assert len(out) == 1
    assert out[0].messages[0].role == "user"

