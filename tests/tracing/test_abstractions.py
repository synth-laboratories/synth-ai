"""
DEPRECATED: These tests use the v0 tracing API which has been removed.

These tests have been disabled as part of v0 removal.
The v0 tracing API has been replaced by tracing_v3.

If you need to test tracing functionality, please use tests/tracing_v3/ instead.
"""

import pytest

pytest.skip(
    "v0 tracing tests have been disabled as part of v0 removal. "
    "Please use tests/tracing_v3/ for tracing tests.",
    allow_module_level=True,
)
