"""Canonical J-refusals codes must have actionable SDK documentation."""

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "code",
    [
        "optimizer_topology_not_allowed",
        "optimizer_topology_unavailable",
        "optimizer_version_unavailable",
        "launch_request_too_large",
        "route_not_found",
        "billing_admin_required",
    ],
)
def test_journey_refusal_has_a_guide_entry(code):
    guide = (Path(__file__).parents[1] / "docs/errors.md").read_text()
    assert f"- `{code}`" in guide
