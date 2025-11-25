from synth_ai.sdk.learning.sft.data import (
    SFTMessage,
    extract_reasoning,
    strip_reasoning,
    message_has_reasoning,
    validate_message_content,
)


def test_extract_and_strip_reasoning():
    r, clean = extract_reasoning("<think>A</think>B<think>C</think>")
    assert r == "A\n\nC"
    assert clean == "B"
    assert strip_reasoning("<think>X</think>Y") == "Y"


def test_message_has_reasoning_field_or_tag():
    m1 = SFTMessage(role="assistant", content="<think>plan</think> answer")
    assert message_has_reasoning(m1) is True
    m2 = SFTMessage(role="assistant", content="no tags", reasoning="explicit")
    assert message_has_reasoning(m2) is True
    m3 = SFTMessage(role="assistant", content="nope")
    assert message_has_reasoning(m3) is False


def test_validate_message_content_combinations():
    # reasoning + tool_calls
    m = SFTMessage(role="assistant", content=None, reasoning="think", tool_calls=[])
    ok, _ = validate_message_content(m)
    assert ok is True

    # content only
    m = SFTMessage(role="assistant", content="hi")
    ok, _ = validate_message_content(m)
    assert ok is True

    # raw_content only
    m = SFTMessage(role="assistant", content=None, raw_content="<think>a</think> b")
    ok, _ = validate_message_content(m)
    assert ok is True

    # empty (invalid when require_content=True)
    m = SFTMessage(role="assistant", content=None)
    ok, err = validate_message_content(m, require_content=True)
    assert ok is False
    assert err and "no reasoning" in err.lower()


