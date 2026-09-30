"""Explicit local helper that transfers selected files to a hosted upload session.

Hosted MCP ``index_contribution_upload_prepare`` returns a session (declared
asset digests and sizes) plus short-lived create-only storage targets. This
helper is the only component that touches the local disk: it reads exactly the
files the caller names through ``SelectedFileReader`` (root confinement, symlink
and traversal refusal, 64 MiB bound, credential scan), verifies them against the
declared digests, and PUTs them to the signed targets. It carries no backend
credential, follows no redirects, and never prints or stores signed URLs.

It does not finalize or submit; call ``index_contribution_upload_finalize`` and
``index_contribution_submit`` afterwards. Interrupted or expired transfers are
resolved by repeating the identical prepare call and running this helper again:
the same intent reuses one publication and only missing objects get targets.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from synth_ai.mcp.research.tools.index import read_selected_files
from synth_ai.sdk.index.contributions import ContributionUploadPrepared
from synth_ai.sdk.index.transfer import _transfer_content

_ACCEPTED = (200, 201, 204)
_REQUEST_TIMEOUT_SECONDS = 60.0
_URL = re.compile(r"https?://[^\s\"'<>]+")


class UploadSessionError(RuntimeError):
    """Base class; messages never contain signed URLs or credentials."""


class UploadSessionExpiredError(UploadSessionError):
    """Targets are expired or refused; prepare again with the identical arguments."""


class UploadSessionInterruptedError(UploadSessionError):
    """Transfer stopped part-way; ``uploaded_paths`` lists what already landed."""

    def __init__(self, message: str, uploaded_paths: Sequence[str]) -> None:
        super().__init__(message)
        self.uploaded_paths = tuple(uploaded_paths)


@dataclass(frozen=True, slots=True)
class UploadSessionReceipt:
    """Credential-free proof of what was transferred."""

    publication_id: str
    manifest_digest: str
    uploaded_paths: tuple[str, ...]
    already_stored_paths: tuple[str, ...]
    total_bytes: int


def redact(text: str) -> str:
    """Drop query strings, userinfo and fragments from every URL in ``text``."""

    def strip(match: re.Match[str]) -> str:
        parts = urlsplit(match.group(0))
        return f"{parts.scheme}://{parts.hostname or 'host'}{parts.path}?[redacted]"

    return _URL.sub(strip, text)


def transfer_selected_files(
    session_result: Mapping[str, Any],
    root: str,
    files: Mapping[str, str],
    *,
    client: httpx.Client | None = None,
    clock: Callable[[], float] | None = None,
) -> UploadSessionReceipt:
    """Transfer ``files`` ({declared logical path: path relative to root}).

    ``session_result`` is the JSON result of ``index_contribution_upload_prepare``.
    Nothing is read from disk when the session is expired or malformed; nothing
    is sent when any selected file differs from its declared digest or size.
    """
    session = session_result.get("session")
    if not isinstance(session, Mapping) or "prepared" not in session_result:
        raise UploadSessionError("Not an index_contribution_upload_prepare result")
    if (clock or time.time)() >= float(session.get("expires_at", 0)):
        raise UploadSessionExpiredError(
            "Upload session expired; repeat the identical prepare call for fresh targets"
        )
    try:
        prepared = ContributionUploadPrepared.model_validate(session_result["prepared"])
    except ValidationError as error:
        raise UploadSessionError("Prepared transfer is malformed") from error
    if str(prepared.transfer.publication_id) != session.get("publication_id"):
        raise UploadSessionError("Prepared transfer belongs to a different publication")

    content = read_selected_files(root, files)  # confinement, bounds, secret scan
    bodies = _transfer_content(prepared, content)  # declared digest/size for every asset

    owned = client is None
    http = client or httpx.Client(
        timeout=_REQUEST_TIMEOUT_SECONDS, follow_redirects=False, trust_env=False
    )
    uploaded: list[str] = []
    try:
        for target in prepared.transfer.upload_targets:
            http.cookies.clear()
            try:
                response = http.put(
                    target.upload_url,
                    headers=dict(target.required_headers),
                    content=bodies[target.logical_path],
                )
            except httpx.HTTPError as error:
                raise UploadSessionInterruptedError(
                    f"Transfer interrupted at {target.logical_path}: "
                    f"{type(error).__name__}; repeat prepare and re-run",
                    uploaded,
                ) from None
            if response.status_code in (400, 401, 403, 409):
                raise UploadSessionExpiredError(
                    f"Storage refused the target for {target.logical_path} "
                    f"(HTTP {response.status_code}); repeat prepare for fresh targets"
                )
            if response.status_code not in _ACCEPTED:
                raise UploadSessionInterruptedError(
                    f"Storage rejected {target.logical_path} with HTTP {response.status_code}",
                    uploaded,
                )
            uploaded.append(target.logical_path)
    finally:
        if owned:
            http.close()
    sent = set(uploaded)
    return UploadSessionReceipt(
        publication_id=str(prepared.transfer.publication_id),
        manifest_digest=str(prepared.transfer.manifest_digest),
        uploaded_paths=tuple(uploaded),
        already_stored_paths=tuple(sorted(set(bodies) - sent - {"contribution.json"})),
        total_bytes=sum(len(bodies[path]) for path in uploaded),
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    client: httpx.Client | None = None,
    clock: Callable[[], float] | None = None,
) -> int:
    """``python -m synth_ai.sdk.index.upload_session --session S.json --root DIR a=a.txt``."""
    parser = argparse.ArgumentParser(prog="upload_session", description=__doc__)
    parser.add_argument("--session", type=Path, required=True, help="prepare result JSON")
    parser.add_argument("--root", required=True, help="explicit directory holding the files")
    parser.add_argument(
        "files", nargs="+", metavar="LOGICAL=RELATIVE", help="declared path=file under root"
    )
    args = parser.parse_args(argv)
    try:
        selected = dict(item.split("=", 1) for item in args.files)
        receipt = transfer_selected_files(
            json.loads(args.session.read_text(encoding="utf-8")),
            args.root,
            selected,
            client=client,
            clock=clock,
        )
    except (UploadSessionError, ValueError, OSError) as error:
        print(redact(f"upload_session refused: {error}"), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "publication_id": receipt.publication_id,
                "manifest_digest": receipt.manifest_digest,
                "uploaded_paths": list(receipt.uploaded_paths),
                "already_stored_paths": list(receipt.already_stored_paths),
                "total_bytes": receipt.total_bytes,
                "next_steps": "Call index_contribution_upload_finalize, then index_contribution_submit.",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
