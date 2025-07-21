"""Configuration for tracing tests."""

import pytest
import asyncio
from unittest.mock import MagicMock


# Add any common test fixtures here
@pytest.fixture
def mock_system():
    """Mock system for tracing tests."""
    system = MagicMock()
    system.system_name = "test_system"
    system.system_id = "test_id"
    system.system_instance_id = "test_instance"
    return system


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
