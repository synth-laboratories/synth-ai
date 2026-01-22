import requests
import pytest

from synth_ai.sdk.optimization.internal.utils import parse_json_response


def _make_response(status_code: int, content: bytes) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = content
    response.headers["content-type"] = "application/json"
    response.url = "http://example.test"
    return response


def test_parse_json_response_success() -> None:
    response = _make_response(200, b'{"ok": true, "value": 3}')
    payload = parse_json_response(response, context="test")
    assert payload["ok"] is True
    assert payload["value"] == 3


def test_parse_json_response_raises_on_http_error() -> None:
    response = _make_response(500, b'{"error": "bad"}')
    with pytest.raises(RuntimeError, match="test failed"):
        parse_json_response(response, context="test")


def test_parse_json_response_raises_on_invalid_json() -> None:
    response = _make_response(200, b"not json")
    with pytest.raises(RuntimeError, match="invalid JSON"):
        parse_json_response(response, context="test")


def test_parse_json_response_raises_on_unexpected_type() -> None:
    response = _make_response(200, b'["not", "dict"]')
    with pytest.raises(RuntimeError, match="unexpected JSON type"):
        parse_json_response(response, context="test")
