"""Programmatic archive bounds and credential-free upload observations."""

import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest
from synth_ai.mcp.catalog import BY_NAME, ALL_BY_NAME
from synth_ai.mcp.contribution_upload import MAXIMUM_ARCHIVE_BYTES, archive_bytes, progress
from synth_ai.types.uploads import ContributionUploadManifest
from tests.test_forge_workspaces import receipt_and_headers


def test_upload_observation_excludes_transfer_capabilities():
    capability, _ = receipt_and_headers()
    manifest = ContributionUploadManifest(
        byte_size=1, digest_sha256=hashlib.sha256(b"a").hexdigest(), media_type="application/zip"
    )
    from synth_ai.types.uploads import UploadStatus

    status = UploadStatus.model_validate_json(
        json.dumps(
            {
                "metadata": {**capability["metadata"], "contract": "index.v1"},
                "upload_id": "upload",
                "contribution_id": "draft",
                "state": "prepared",
                "manifest": manifest,
                "transfer_url": "https://storage.invalid/private-grant",
                "transfer_headers": {},
                "expires_at": (datetime.now(UTC) + timedelta(minutes=1)).isoformat(),
            },
            default=lambda value: value.model_dump(mode="json"),
        )
    )
    result = progress(status).model_dump_json()
    assert "private-grant" not in result and "transfer_" not in result
    assert "prepared" in result


def test_archive_strict_encoding_bound_and_closed_authority():
    assert archive_bytes(base64.b64encode(b"archive").decode()) == b"archive"
    for value in [
        "YQ==\n",
        "YQ",
        "!!!!",
        "",
        base64.b64encode(b"a" * (MAXIMUM_ARCHIVE_BYTES + 1)).decode(),
    ]:
        with pytest.raises(ValueError):
            archive_bytes(value)
    for name in ["prepare", "get", "finalize"]:
        tool = ALL_BY_NAME["index_contribution_upload_" + name]
        assert tool.scope == "index:intake" and not tool.paid
        assert not any(
            key in str(tool.output_schema()) for key in ["transfer_url", "transfer_headers"]
        )
    with pytest.raises(ValueError):
        ALL_BY_NAME["index_contribution_upload_finalize"].validate_arguments(
            {"upload_id": "upload", "idempotency_key": "key", "path": "/private/file"}
        )
