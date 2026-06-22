from synth_ai.managed_research.transport import (
    build_query_params,
    extract_next_cursor,
    preview_binary_payload,
)


def test_build_query_params_drops_none_and_empty_strings() -> None:
    assert build_query_params(project_id="proj_123", cursor=None, state="", limit=50) == {
        "project_id": "proj_123",
        "limit": 50,
    }


def test_extract_next_cursor_reads_common_payload_keys() -> None:
    assert extract_next_cursor({"next_cursor": "cur_123"}) == "cur_123"
    assert extract_next_cursor({"nextCursor": "cur_456"}) == "cur_456"
    assert extract_next_cursor({}) is None


def test_preview_binary_payload_returns_text_preview() -> None:
    preview = preview_binary_payload(b"hello world", media_type="text/plain")
    assert preview.media_type == "text/plain"
    assert preview.size_bytes == 11
    assert preview.preview_text == "hello world"
