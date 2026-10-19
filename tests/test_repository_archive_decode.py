"""Raw Git tar decoding proves the actual tree and refuses unsafe framing."""

import hashlib
import io
import tarfile

import pytest
from synth_ai.errors import ContractError
from synth_ai.sdk.core.archive_decode import extract_verified, windows_reserved
from synth_ai.sdk.core.commit_digest import (
    CommitDigestMismatch,
    commit_manifest_digest,
    verify_tree,
)
from synth_ai.sdk.core.types import RepositoryArchiveOptions

pytestmark = pytest.mark.unit


def fixture_tar(items):
    data = bytearray()
    manifest = []
    for path, mode, value in items:
        info = tarfile.TarInfo(path)
        info.mode = 0o755 if mode == "100755" else 0o644
        if mode == "160000":
            info.type = tarfile.DIRTYPE
            info.pax_headers = {"SYNTH.gitlink": value.decode()}
            content = b"gitlink:" + value
        elif mode == "120000":
            info.type = tarfile.SYMTYPE
            info.linkname = value.decode("utf-8", errors="surrogateescape")
            content = value
        else:
            info.size = len(value)
            content = value
        data.extend(
            info.tobuf(format=tarfile.PAX_FORMAT if mode == "160000" else tarfile.GNU_FORMAT)
        )
        if mode in {"100644", "100755"}:
            data.extend(value)
            data.extend(b"\x00" * (-len(value) % 512))
        manifest.append({"path": path, "mode": mode, "sha256": hashlib.sha256(content).hexdigest()})
    data.extend(b"\x00" * 1024)
    return bytes(data), commit_manifest_digest(manifest)


def test_native_format_tree_including_gitlinks_and_long_names(tmp_path):
    long = "unicode-λ-" + "long" * 40
    items = [
        ("nested/file", "100644", b"content\r\n"),
        ("exec", "100755", b"#!/bin/sh\n"),
        (long, "100644", b"long file"),
        ("link", "120000", long.encode()),
        ("submodule", "160000", b"a" * 40),
    ]
    raw, expected = fixture_tar(items)
    result = extract_verified(
        io.BytesIO(raw), tmp_path, expected, len(items), RepositoryArchiveOptions(), lambda: None
    )
    assert result == {"submodule": "a" * 40}
    assert verify_tree(tmp_path, expected, gitlinks=result) == expected
    assert (tmp_path / "nested/file").read_bytes() == b"content\r\n"
    assert (tmp_path / "link").read_bytes() == b"long file"
    assert list((tmp_path / "submodule").iterdir()) == []


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a//b", "a/./b", ".git/config"])
def test_archive_paths_refuse_before_writing_outside_the_private_root(tmp_path, path):
    info = tarfile.TarInfo(path)
    info.size = 1
    raw = info.tobuf(format=tarfile.GNU_FORMAT) + b"x" + b"\x00" * 511 + b"\x00" * 1024
    with pytest.raises(ContractError) as caught:
        extract_verified(
            io.BytesIO(raw), tmp_path, "a" * 64, 1, RepositoryArchiveOptions(), lambda: None
        )
    assert caught.value.code == "archive_path_invalid"


@pytest.mark.parametrize(
    "items",
    [
        [("link", "120000", b"../outside")],
        [("a", "120000", b"sub/.."), ("sub", "120000", b".")],
        [("a", "120000", b"a")],
        [("a", "120000", b"target"), ("a/file", "100644", b"must not follow a link")],
    ],
)
def test_escaping_cyclic_or_parent_symlinks_are_refused(tmp_path, items):
    raw, expected = fixture_tar(items)
    with pytest.raises(ContractError):
        extract_verified(
            io.BytesIO(raw),
            tmp_path,
            expected,
            len(items),
            RepositoryArchiveOptions(),
            lambda: None,
        )


def test_tampered_tree_or_trailing_payload_never_verifies(tmp_path):
    raw, expected = fixture_tar([("file", "100644", b"original")])
    with pytest.raises(CommitDigestMismatch):
        extract_verified(
            io.BytesIO(raw.replace(b"original", b"tampered")),
            tmp_path,
            expected,
            1,
            RepositoryArchiveOptions(),
            lambda: None,
        )
    (tmp_path / "file").unlink()
    with pytest.raises(ContractError):
        extract_verified(
            io.BytesIO(raw + b"hidden payload"),
            tmp_path,
            expected,
            1,
            RepositoryArchiveOptions(),
            lambda: None,
        )


def test_metadata_size_is_refused_before_reading_its_payload(tmp_path):
    info = tarfile.TarInfo("metadata")
    info.type = tarfile.XHDTYPE
    info.size = 1024 * 1024

    class HeaderOnly(io.BytesIO):
        def read(self, size=-1):
            assert size <= 512, "Never allocate the declared oversized metadata payload"
            return super().read(size)

    with pytest.raises(ContractError) as caught:
        extract_verified(
            HeaderOnly(info.tobuf()),
            tmp_path,
            "a" * 64,
            1,
            RepositoryArchiveOptions(),
            lambda: None,
        )
    assert caught.value.code == "archive_metadata_limit"


def test_directory_and_file_counts_are_separately_bounded(tmp_path):
    raw, expected = fixture_tar([("nested/file", "100644", b"value")])
    with pytest.raises(ContractError) as caught:
        extract_verified(
            io.BytesIO(raw),
            tmp_path,
            expected,
            1,
            RepositoryArchiveOptions(maximum_directories=0),
            lambda: None,
        )
    assert caught.value.code == "archive_directory_limit"


def test_gitlink_authority_metadata_cannot_hide_a_missing_or_populated_directory(tmp_path):
    with pytest.raises(ContractError):
        verify_tree(tmp_path, "a" * 64, gitlinks={"missing": "a" * 40})
    path = tmp_path / "submodule"
    path.mkdir()
    (path / "file").write_text("unexpected")
    with pytest.raises(ContractError):
        verify_tree(tmp_path, "a" * 64, gitlinks={"submodule": "a" * 40})


@pytest.mark.parametrize(
    "name", ["NUL", "con.txt", "COM1", "LPT¹.log", "file:stream", "file.", "file ", "a\x01b"]
)
def test_windows_devices_streams_and_normalized_aliases_are_refused(name):
    assert windows_reserved(name)
