"""Tests for container URL classification helpers.

Validates is_local_http_container_url from urls.py with a comprehensive matrix.
"""

from __future__ import annotations

import pytest

from synth_ai.core.utils.urls import is_local_http_container_url


class TestIsLocalHttpContainerUrl:
    @pytest.mark.parametrize("url", [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://0.0.0.0:9000",
        "http://host.docker.internal:8000",
        "http://localhost",
        "http://127.0.0.1",
        "http://0.0.0.0",
        "http://host.docker.internal",
        "http://localhost:8080/api",
    ])
    def test_local_urls_return_true(self, url):
        assert is_local_http_container_url(url) is True

    @pytest.mark.parametrize("url", [
        "https://localhost:8000",
        "http://example.com",
        "",
        "http://api.usesynth.ai",
        "ftp://localhost:8000",
        "http://192.168.1.1:8000",
        "not-a-url",
    ])
    def test_non_local_urls_return_false(self, url):
        assert is_local_http_container_url(url) is False

    def test_whitespace_trimmed(self):
        assert is_local_http_container_url("  http://localhost:8000  ") is True

    def test_case_insensitive_scheme(self):
        assert is_local_http_container_url("HTTP://localhost:8000") is True

    def test_case_insensitive_host(self):
        assert is_local_http_container_url("http://LOCALHOST:8000") is True
