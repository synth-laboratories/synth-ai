"""Workspace uploads send an explicit Git file mode for executable inputs.

The backend commits each upload with the mode it receives; a file that was
executable at the source must say ``100755`` on the wire or the actor
checkout gets it as ``100644`` (slot2 smoke 31, cg-rustc-wrapper).
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from click.testing import CliRunner
from synth_ai.mcp.research.tools.workspace_inputs import _file_upload
from synth_ai.sdk.research.contracts.workspaces import (
    WorkspaceFileMode,
    WorkspaceFilesBatchUploadRequest,
    WorkspaceFileUpload,
)


def test_mode_maps_from_posix_permission_bits() -> None:
    assert WorkspaceFileMode.from_posix_mode(0o100755) is WorkspaceFileMode.EXECUTABLE
    assert WorkspaceFileMode.from_posix_mode(0o744) is WorkspaceFileMode.EXECUTABLE
    assert WorkspaceFileMode.from_posix_mode(0o100644) is WorkspaceFileMode.REGULAR


def test_upload_wire_carries_mode_only_when_declared() -> None:
    executable = WorkspaceFileUpload(
        path=".cargo/cg-rustc-wrapper",
        content="#!/bin/sh\n",
        mode=WorkspaceFileMode.EXECUTABLE,
    )
    assert executable.to_wire()["mode"] == "100755"
    assert "mode" not in WorkspaceFileUpload(path="a.txt", content="x").to_wire()


def test_mcp_tool_accepts_and_validates_mode() -> None:
    upload = _file_upload({"path": "run.sh", "content": "x", "mode": "100755"}, index=0)
    assert upload.mode is WorkspaceFileMode.EXECUTABLE
    with pytest.raises(ValueError):
        _file_upload({"path": "run.sh", "content": "x", "mode": "100777"}, index=0)


def test_cli_upload_declares_source_executable_bit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from synth_ai.cli import research_projects

    wrapper = tmp_path / ".cargo" / "cg-rustc-wrapper"
    wrapper.parent.mkdir()
    wrapper.write_text("#!/bin/sh\n")
    wrapper.chmod(0o755)
    config = tmp_path / ".cargo" / "config.toml"
    config.write_text("[build]\n")
    config.chmod(0o644)

    captured: list[WorkspaceFilesBatchUploadRequest] = []

    class _Receipt:
        def to_wire(self) -> dict[str, bool]:
            return {"ok": True}

    class _Workspace:
        def upload_batches(self, project_id, request):  # noqa: ANN001, ANN202
            captured.append(request)
            return _Receipt()

    class _Client:
        class research:  # noqa: N801
            class projects:  # noqa: N801
                workspace = _Workspace()

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(research_projects, "_client", lambda *_args: _Client())
    result = CliRunner().invoke(
        research_projects.workspace_upload,
        [
            "proj_1",
            str(wrapper),
            str(config),
            "--root",
            str(tmp_path),
            "--api-key",
            "sk-test",
        ],
    )
    assert captured, result.output
    modes = {item.path: item.mode for item in captured[0].files}
    assert modes == {
        ".cargo/cg-rustc-wrapper": WorkspaceFileMode.EXECUTABLE,
        ".cargo/config.toml": WorkspaceFileMode.REGULAR,
    }
    assert base64.b64decode(captured[0].files[0].content) == b"#!/bin/sh\n"
