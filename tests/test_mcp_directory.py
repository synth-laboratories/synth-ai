import asyncio
import json
import os
from unittest.mock import AsyncMock

import pytest

from synth_ai._directory_source import DirectorySnapshot
from synth_ai.errors import DirectoryUploadCancelledError, InvalidRequestError
from synth_ai.mcp import directory
from synth_ai.mcp.server import call_tool


def test_grant_pins_root_through_parent_replacement(tmp_path):
    root = tmp_path / "parent" / "root"
    root.mkdir(parents=True)
    (root / "allowed.bin").write_bytes(b"\0\xff")
    with directory.grant_root(str(root)) as fd:
        assert fd is not None
        (tmp_path / "parent").rename(tmp_path / "original")
        root.mkdir(parents=True)
        (root / "secret.bin").write_bytes(b"secret")
        snapshot = DirectorySnapshot.prepare(fd)
        try:
            assert [f.name for f in snapshot.files] == ["allowed.bin"]
            snapshot.stream.seek(0)
            assert snapshot.stream.read() == b"\0\xff"
            os.fstat(fd)  # Snapshot borrows rather than consumes the capability.
        finally:
            snapshot.stream.close()
    assert fd is not None
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.parametrize("kind", ["ancestor", "root", "parent_traversal", "relative"])
def test_grant_refuses_links_and_ambiguous_paths(tmp_path, kind):
    root = tmp_path / "real"
    root.mkdir()
    (root / "child").mkdir()
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    path = {
        "ancestor": str(link / "child"),
        "root": str(link),
        "parent_traversal": str(root / ".." / "real"),
        "relative": "real",
    }[kind]
    with pytest.raises((OSError, ValueError)), directory.grant_root(path):
        pytest.fail("unsafe grant accepted")


async def test_no_ambient_directory_tool(tmp_path):
    client = AsyncMock()
    with pytest.raises(ValueError, match="Unknown tool"):
        await call_tool(client, directory.NAME, {"idempotency_key": "batch"})
    client.swarms.upload_directory.assert_not_called()


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"idempotency_key": " "},
        {"idempotency_key": "a" * 129},
        {"idempotency_key": "é" * 65},
        {"idempotency_key": "bad\nkey"},
        {"idempotency_key": "bad\rkey"},
        {"idempotency_key": "bad\0key"},
        {"idempotency_key": "batch", "directory": "../secret"},
    ],
)
async def test_local_tool_rejects_invalid_inputs_before_upload(arguments):
    properties = directory.SCHEMA["properties"]
    assert isinstance(properties, dict)
    assert properties["idempotency_key"]["maxLength"] == 128
    client = AsyncMock()
    with pytest.raises(InvalidRequestError):
        await call_tool(client, directory.NAME, arguments, upload_descriptor=10)
    client.swarms.upload_directory.assert_not_called()


async def test_granted_tool_uses_borrowed_descriptor():
    client = AsyncMock()
    client.swarms.upload_directory.return_value = []
    assert json.loads(
        await call_tool(client, directory.NAME, {"idempotency_key": "batch"}, upload_descriptor=10)
    ) == {"idempotency_key": "batch", "files": []}
    client.swarms.upload_directory.assert_awaited_once_with(10, idempotency_key="batch")


def test_cancellation_remains_cancellation_with_retry_custody():
    error = DirectoryUploadCancelledError("b.bin", (), "batch", "file-key")
    assert isinstance(error, asyncio.CancelledError)
    assert not isinstance(error, Exception)
    assert json.loads(directory.failure(error)) == {
        "code": "directory_upload_cancelled",
        "name": "b.bin",
        "completed_files": [],
        "idempotency_key": "batch",
        "file_idempotency_key": "file-key",
    }


def test_parallel_snapshots_have_independent_directory_cursors(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    names = [f"file-{n:03d}" for n in range(100)]
    for name in names:
        (tmp_path / name).write_bytes(b"data")
    with directory.grant_root(str(tmp_path)) as fd:
        assert fd is not None
        root_fd = fd

        def snapshot_names(_):
            snapshot = DirectorySnapshot.prepare(root_fd)
            try:
                return [f.name for f in snapshot.files]
            finally:
                snapshot.stream.close()

        with ThreadPoolExecutor(max_workers=4) as executor:
            assert list(executor.map(snapshot_names, range(8))) == [names] * 8
