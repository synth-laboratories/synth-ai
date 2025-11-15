"""Unit tests for Cloudflare tunnel URL parsing logic.

These tests verify that we can correctly extract tunnel URLs from cloudflared output,
including handling truncated URLs that may occur due to terminal wrapping.
"""

import re
import subprocess
from unittest.mock import Mock, patch

import pytest

from synth_ai.cloudflare import (
    _URL_RE,
    _URL_PARTIAL_RE,
    _URL_PARTIAL_RE2,
    open_quick_tunnel,
)


# Real output captured from cloudflared tunnel command
REAL_CLOUDFLARED_OUTPUT_STDERR = """2025-11-14T18:20:05Z INF Thank you for trying Cloudflare Tunnel. Doing so, without a Cloudflare account, is a quick way to experiment and try it out. However, be aware that these account-less Tunnels have no uptime guarantee, are subject to the Cloudflare Online Services Terms of Use (https://www.cloudflare.com/website-terms/), and Cloudflare reserves the right to investigate your use of Tunnels for violations of such terms. If you intend to use Tunnels in production you should use a pre-created named tunnel by following: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps
2025-11-14T18:20:05Z INF Requesting new quick Tunnel on trycloudflare.com...
2025-11-14T18:20:11Z INF +--------------------------------------------------------------------------------------------+
2025-11-14T18:20:11Z INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
2025-11-14T18:20:11Z INF |  https://operation-possibly-pal-experimental.trycloudflare.com                             |
2025-11-14T18:20:11Z INF +--------------------------------------------------------------------------------------------+
2025-11-14T18:20:11Z INF Cannot determine default configuration path. No file [config.yml config.yaml] in [~/.cloudflared ~/.cloudflare-warp ~/cloudflare-warp /etc/cloudflared /usr/local/etc/cloudflared]
2025-11-14T18:20:11Z INF Version 2025.11.1 (Checksum 53e8656c5207f9cba9ed4915ecaeffc2a8d6b401bef21e60ad543f02c135b4ae)
"""

# Simulated truncated outputs (as seen in user's terminal)
TRUNCATED_OUTPUT_1 = """2025-11-14T18:16:31Z INF +--------------------------------------------------------------------------------------------+
2025-11-14T18:16:31Z INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
2025-11-14T18:16:31Z INF |  https://nirvana-lasting-throw-boost.trycloudf
"""

TRUNCATED_OUTPUT_2 = """2025-11-14T18:16:53Z INF +--------------------------------------------------------------------------------------------+
2025-11-14T18:16:53Z INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
2025-11-14T18:16:53Z INF |  https://juvenile-structure-mixed-radar.tryclo
"""


class TestURLRegex:
    """Test URL regex patterns."""

    def test_full_url_regex(self):
        """Test that full URL regex matches complete URLs."""
        test_urls = [
            "https://operation-possibly-pal-experimental.trycloudflare.com",
            "https://nirvana-lasting-throw-boost.trycloudflare.com",
            "https://juvenile-structure-mixed-radar.trycloudflare.com",
            "https://test-123.trycloudflare.com",
        ]
        for url in test_urls:
            match = _URL_RE.search(url)
            assert match is not None, f"Should match {url}"
            assert match.group(0) == url

    def test_full_url_regex_with_context(self):
        """Test that full URL regex matches URLs in context."""
        test_cases = [
            (
                "2025-11-14T18:20:11Z INF |  https://operation-possibly-pal-experimental.trycloudflare.com                             |",
                "https://operation-possibly-pal-experimental.trycloudflare.com",
            ),
            (
                "Visit it at: https://test-123.trycloudflare.com",
                "https://test-123.trycloudflare.com",
            ),
        ]
        for text, expected_url in test_cases:
            match = _URL_RE.search(text)
            assert match is not None, f"Should match URL in: {text}"
            assert match.group(0) == expected_url

    def test_partial_url_regex_trycloudf(self):
        """Test partial URL regex for URLs ending with trycloudf."""
        test_cases = [
            "https://nirvana-lasting-throw-boost.trycloudf",
            "https://test-123.trycloudf",
            "Visit: https://example.trycloudf",
        ]
        for text in test_cases:
            match = _URL_PARTIAL_RE.search(text)
            assert match is not None, f"Should match partial URL: {text}"
            assert match.group(0).endswith(".trycloudf")

    def test_partial_url_regex_tryclo(self):
        """Test partial URL regex for URLs ending with tryclo."""
        test_cases = [
            "https://juvenile-structure-mixed-radar.tryclo",
            "https://test-123.tryclo",
            "Visit: https://example.tryclo",
        ]
        for text in test_cases:
            match = _URL_PARTIAL_RE2.search(text)
            assert match is not None, f"Should match partial URL: {text}"
            assert match.group(0).endswith(".tryclo")


class TestURLParsingFromOutput:
    """Test URL parsing from actual cloudflared output."""

    def test_parse_real_output(self):
        """Test parsing URL from real cloudflared stderr output."""
        match = _URL_RE.search(REAL_CLOUDFLARED_OUTPUT_STDERR)
        assert match is not None, "Should find URL in real output"
        assert match.group(0) == "https://operation-possibly-pal-experimental.trycloudflare.com"

    def test_parse_truncated_output_trycloudf(self):
        """Test parsing truncated URL ending with trycloudf."""
        # Should not match full URL
        full_match = _URL_RE.search(TRUNCATED_OUTPUT_1)
        assert full_match is None, "Should not match full URL in truncated output"

        # Should match partial URL
        partial_match = _URL_PARTIAL_RE.search(TRUNCATED_OUTPUT_1)
        assert partial_match is not None, "Should match partial URL"
        assert partial_match.group(0) == "https://nirvana-lasting-throw-boost.trycloudf"

        # Should be able to reconstruct
        partial_url = partial_match.group(0)
        reconstructed = partial_url + "lare.com"
        assert _URL_RE.match(reconstructed), "Should match reconstructed URL"
        assert reconstructed == "https://nirvana-lasting-throw-boost.trycloudflare.com"

    def test_parse_truncated_output_tryclo(self):
        """Test parsing truncated URL ending with tryclo."""
        # Should not match full URL
        full_match = _URL_RE.search(TRUNCATED_OUTPUT_2)
        assert full_match is None, "Should not match full URL in truncated output"

        # Should match partial URL
        partial_match = _URL_PARTIAL_RE2.search(TRUNCATED_OUTPUT_2)
        assert partial_match is not None, "Should match partial URL"
        assert partial_match.group(0) == "https://juvenile-structure-mixed-radar.tryclo"

        # Should be able to reconstruct
        partial_url = partial_match.group(0)
        reconstructed = partial_url + "udflare.com"
        assert _URL_RE.match(reconstructed), "Should match reconstructed URL"
        assert reconstructed == "https://juvenile-structure-mixed-radar.trycloudflare.com"


class TestURLReconstruction:
    """Test URL reconstruction logic (simulating the logic in open_quick_tunnel)."""

    def test_reconstruct_from_partial_trycloudf(self):
        """Test reconstructing URL from partial match ending with trycloudf."""
        # Simulate what happens in open_quick_tunnel when timeout occurs
        output_lines = []
        stderr_lines = TRUNCATED_OUTPUT_1.splitlines(keepends=True)
        
        # Accumulate output
        all_accumulated = ''.join(output_lines + stderr_lines)
        
        # Check for partial URL
        partial_match = _URL_PARTIAL_RE.search(all_accumulated)
        assert partial_match is not None, "Should find partial URL"
        
        partial_url = partial_match.group(0)
        assert partial_url == "https://nirvana-lasting-throw-boost.trycloudf"
        
        # Reconstruct
        test_url = partial_url + "lare.com"
        assert _URL_RE.match(test_url), "Reconstructed URL should match full pattern"
        assert test_url == "https://nirvana-lasting-throw-boost.trycloudflare.com"

    def test_reconstruct_from_partial_tryclo(self):
        """Test reconstructing URL from partial match ending with tryclo."""
        # Simulate what happens in open_quick_tunnel when timeout occurs
        output_lines = []
        stderr_lines = TRUNCATED_OUTPUT_2.splitlines(keepends=True)
        
        # Accumulate output
        all_accumulated = ''.join(output_lines + stderr_lines)
        
        # Check for partial URL
        partial_match = _URL_PARTIAL_RE2.search(all_accumulated)
        assert partial_match is not None, "Should find partial URL"
        
        partial_url = partial_match.group(0)
        assert partial_url == "https://juvenile-structure-mixed-radar.tryclo"
        
        # Reconstruct
        test_url = partial_url + "udflare.com"
        assert _URL_RE.match(test_url), "Reconstructed URL should match full pattern"
        assert test_url == "https://juvenile-structure-mixed-radar.trycloudflare.com"

    def test_parse_from_accumulated_output(self):
        """Test parsing URL from accumulated output (simulating line-by-line reading)."""
        # Simulate reading lines one by one
        lines = REAL_CLOUDFLARED_OUTPUT_STDERR.splitlines(keepends=True)
        accumulated = ""
        
        url = None
        for line in lines:
            accumulated += line
            match = _URL_RE.search(accumulated)
            if match:
                url = match.group(0)
                break
        
        assert url is not None, "Should find URL in accumulated output"
        assert url == "https://operation-possibly-pal-experimental.trycloudflare.com"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

