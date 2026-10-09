"""An expired owner-read credential refuses with a typed, re-issuable error."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from synth_ai.core.errors import SynthError
from synth_ai.sdk.research.owner_reads import (
    OwnerReadAccess,
    OwnerReadAccessExpired,
    OwnerReadScope,
)


def access(expires_at):
    scope = OwnerReadScope(str(uuid4()), str(uuid4()), str(uuid4()))
    return OwnerReadAccess(
        owner="sublinear",
        origin="http://sublinear:8011",
        scope=scope,
        path_prefix=f"/v1/runs/{scope.run_id}/planning",
        token="reader.fixture.jwt",
        expires_at=expires_at,
    )


def test_expired_access_raises_typed_error_naming_owner_run_and_expiry():
    expired_at = datetime.now(UTC) - timedelta(seconds=1)
    held = access(expired_at)
    with pytest.raises(OwnerReadAccessExpired) as caught:
        held.request("/graph", None)
    error = caught.value
    assert isinstance(error, SynthError) and isinstance(error, ValueError)
    assert (error.owner, error.run_id, error.expired_at) == (
        "sublinear",
        held.scope.run_id,
        expired_at,
    )
    assert "issue fresh access" in str(error)


def test_live_access_still_builds_the_scoped_request():
    held = access(datetime.now(UTC) + timedelta(minutes=5))
    path, query = held.request("/graph", None)
    assert path == held.path_prefix + "/graph"
    assert query == {
        "organization_id": held.scope.organization_id,
        "project_id": held.scope.project_id,
    }
