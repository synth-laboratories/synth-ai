import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-classify tests:
    - Any test marked `slow` is also considered `integration` unless already marked otherwise.
    This keeps existing annotations working while enabling -m integration selection.
    """
    for item in items:
        if item.get_closest_marker("slow") and not item.get_closest_marker("integration"):
            item.add_marker(pytest.mark.integration)

