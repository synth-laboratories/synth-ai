import asyncio
import hashlib
import os
import threading

import pytest

from synth_ai._directory_source import DirectorySnapshot, open_directory
from synth_ai._file_source import FileIO
from synth_ai.errors import DataIntegrityError


async def test_snapshot_preserves_relative_binary_bytes_after_original_changes(tmp_path):
    root = tmp_path / "package"
    (root / "nested").mkdir(parents=True)
    (root / "nested/policy.bin").write_bytes(b"\x00\xff\x80")
    (root / ".dockerignore").write_bytes(b"cache\n")
    io = FileIO()
    async with open_directory(root, io) as snapshot:
        assert [file.name for file in snapshot.files] == [".dockerignore", "nested/policy.bin"]
        (root / "nested/policy.bin").write_bytes(b"different")
        file = snapshot.files[1]
        assert file.digest_sha256 == hashlib.sha256(b"\x00\xff\x80").hexdigest()
        source = snapshot.source(file)
        assert b"".join([chunk async for chunk in source.chunks(io)]) == b"\x00\xff\x80"
    assert snapshot.stream.closed


@pytest.mark.parametrize("kind", ["root", "file", "directory", "fifo"])
def test_snapshot_refuses_symlinks_and_special_files(tmp_path, kind):
    root = tmp_path / "package"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_bytes(b"outside")
    if kind == "root":
        link = tmp_path / "link"
        link.symlink_to(root, target_is_directory=True)
        root = link
    elif kind == "file":
        (root / "link").symlink_to(outside / "secret")
    elif kind == "directory":
        (root / "link").symlink_to(outside, target_is_directory=True)
    else:
        os.mkfifo(root / "pipe")
    with pytest.raises((ValueError, OSError)):
        DirectorySnapshot.prepare(root)


@pytest.mark.parametrize(
    "name", ["bad name", "bad\\name", "policy\N{LATIN SMALL LETTER E WITH ACUTE}"]
)
def test_snapshot_refuses_names_outside_canonical_grammar(tmp_path, name):
    (tmp_path / name).write_bytes(b"x")
    with pytest.raises(ValueError, match="name"):
        DirectorySnapshot.prepare(tmp_path)


def test_snapshot_bounds_empty_directory_traversal(tmp_path, monkeypatch):
    import synth_ai._directory_source as module

    monkeypatch.setattr(module, "LAUNCH_MAX_FILES", 2)
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()
    with pytest.raises(ValueError, match="entries"):
        DirectorySnapshot.prepare(tmp_path)


def test_snapshot_bounds_total_and_individual_bytes(tmp_path, monkeypatch):
    import synth_ai._directory_source as module

    (tmp_path / "a").write_bytes(b"123")
    (tmp_path / "b").write_bytes(b"456")
    monkeypatch.setattr(module, "LAUNCH_MAX_BYTES", 5)
    with pytest.raises(ValueError, match="aggregate"):
        DirectorySnapshot.prepare(tmp_path)
    monkeypatch.setattr(module, "LAUNCH_MAX_BYTES", 10)
    monkeypatch.setattr(module, "FILE_MAX_BYTES", 2)
    with pytest.raises(ValueError, match="file"):
        DirectorySnapshot.prepare(tmp_path)


def test_directory_replaced_with_symlink_during_open_cannot_escape(tmp_path, monkeypatch):
    root, outside = tmp_path / "root", tmp_path / "outside"
    (root / "nested").mkdir(parents=True)
    outside.mkdir()
    (outside / "secret").write_bytes(b"outside")
    original = os.open

    def replace(name, flags, *args, **kwargs):
        if name == "nested":
            (root / "nested").rename(root / "original")
            (root / "nested").symlink_to(outside, target_is_directory=True)
        return original(name, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace)
    with pytest.raises(OSError):
        DirectorySnapshot.prepare(root)


def test_file_mutation_while_copying_refuses_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "policy"
    path.write_bytes(b"a" * 100_000)
    original, changed = os.read, False

    def read(descriptor, size):
        nonlocal changed
        chunk = original(descriptor, size)
        if not changed:
            changed = True
            path.write_bytes(b"b" * 100_001)
        return chunk

    monkeypatch.setattr(os, "read", read)
    with pytest.raises(DataIntegrityError):
        DirectorySnapshot.prepare(tmp_path)


async def test_cancel_joins_snapshot_worker_and_closes_spool(tmp_path, monkeypatch):
    (tmp_path / "policy").write_bytes(b"x")
    original = DirectorySnapshot.prepare
    started, release = threading.Event(), threading.Event()
    snapshots = []

    def prepare(path):
        snapshot = original(path)
        snapshots.append(snapshot)
        started.set()
        release.wait(2)
        return snapshot

    monkeypatch.setattr(DirectorySnapshot, "prepare", prepare)

    async def work():
        async with open_directory(tmp_path, FileIO()):
            pytest.fail("cancelled preparation returned")

    task = asyncio.create_task(work())
    await asyncio.to_thread(started.wait, 1)
    task.cancel()
    await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert snapshots[0].stream.closed
