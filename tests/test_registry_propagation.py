"""The post-upload registry check tolerates PyPI propagation lag (SYN-4373)."""

from __future__ import annotations

import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "verify_qualified_release", ROOT / ".github/verify_qualified_release.py"
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
SHA = "a" * 64
RECEIPT = {"version": "0.23.1", "distributions": {"synth_ai-0.23.1-py3-none-any.whl": SHA}}


def not_found():
    return urllib.error.HTTPError("https://pypi.org/x", 404, "Not Found", {}, io.BytesIO(b"{}"))


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def __call__(self):
        return self.now


def run(fetch):
    clock = Clock()
    result = None
    try:
        result = release.await_registry(RECEIPT, fetch=fetch, sleep=clock.sleep, clock=clock)
    finally:
        run.clock = clock
    return result


def test_404_then_200_is_treated_as_not_yet_propagated():
    calls = iter([not_found(), not_found(), True])

    def fetch():
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    assert run(fetch) is True
    assert run.clock.sleeps == [15, 30]


def test_404_forever_fails_after_the_window_with_a_rich_error():
    def fetch():
        raise not_found()

    with pytest.raises(ValueError) as error:
        run(fetch)
    message = str(error.value)
    assert "synth-ai 0.23.1" in message and "https://pypi.org/pypi/synth-ai/0.23.1/json" in message
    assert "HTTP 404" in message and "attempts" in message
    assert sum(run.clock.sleeps) <= release.REGISTRY_WINDOW_SECONDS


def test_other_http_errors_fail_immediately():
    def fetch():
        raise urllib.error.HTTPError("u", 500, "boom", {}, io.BytesIO(b""))

    with pytest.raises(urllib.error.HTTPError):
        run(fetch)
    assert run.clock.sleeps == []


def test_served_but_different_bytes_fail_without_retrying():
    def fetch():
        raise ValueError("registry has different bytes for this version")

    with pytest.raises(ValueError, match="different bytes"):
        run(fetch)
    assert run.clock.sleeps == []


def test_registry_state_checks_the_hashes_of_a_200(monkeypatch):
    body = json.dumps(
        {
            "info": {"version": "0.23.1"},
            "urls": [{"filename": "synth_ai-0.23.1-py3-none-any.whl", "digests": {"sha256": SHA}}],
        }
    ).encode()

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(release.urllib.request, "urlopen", lambda *a, **k: Response(body))
    assert release.registry_state(RECEIPT, require_published=True) is True
