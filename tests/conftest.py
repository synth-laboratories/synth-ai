"""Pytest configuration for synth-ai tests."""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: integration tests requiring backend")
    config.addinivalue_line("markers", "unit: fast unit tests")
    config.addinivalue_line("markers", "slow: slow tests (>30s)")
