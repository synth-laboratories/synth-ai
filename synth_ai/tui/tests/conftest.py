"""Pytest configuration for TUI tests."""

import pytest


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark integration tests as slow."""
    for item in items:
        # Mark tests that start servers as slow
        if "server" in item.name.lower() or "integration" in item.name.lower():
            item.add_marker(pytest.mark.slow)
