"""
Pytest configuration for Environments tests.
"""

import pytest
import sys
from pathlib import Path

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture(autouse=True)
def cleanup_instances():
    """Automatically clean up environment instances after each test."""
    yield
    # Clean up any leftover instances
    try:
        from service.core_routes import instances

        instances.clear()
    except ImportError:
        pass


@pytest.fixture
def disable_external_environments(monkeypatch):
    """Disable loading of external environments for tests."""
    monkeypatch.setenv("EXTERNAL_ENVIRONMENTS", "")
    yield
