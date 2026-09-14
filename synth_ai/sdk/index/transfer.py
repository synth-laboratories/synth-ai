"""Explicit in-memory Contribution transfer, separate from authenticated API calls.

See sibling docs/drafts/synth-index-api-design-2026-09-12.md.
This bounded convenience path serves small reports/repos, not streaming datasets.
No filesystem discovery, automatic finalization, retries, or public release.
"""

import asyncio
from collections.abc import Mapping
from hashlib import sha256
from urllib.parse import urlsplit

import httpx

from .contributions import ContributionUploadPrepared
from .package import ContributionPackage


def _transfer_content(
    prepared: ContributionUploadPrepared, content: Mapping[str, bytes]
) -> dict[str, bytes]:
    package = ContributionPackage.model_validate_json(prepared.descriptor_json)
    declarations = {asset.object.logical_path: asset.object for asset in package.assets}
    if "contribution.json" in declarations or set(content) != set(declarations):
        raise ValueError("Transfer requires exactly the package's declared asset paths")
    if sum(len(value) for value in content.values()) > 64 * 1024 * 1024:
        raise ValueError("In-memory transfer exceeds 64 MiB; streaming transfer is required")
    for path, declaration in declarations.items():
        value = content[path]
        if not isinstance(value, bytes):
            raise ValueError("Transfer content must be immutable bytes")
        if (
            len(value) != declaration.size_bytes
            or sha256(value).hexdigest() != declaration.digest_sha256
        ):
            raise ValueError("Transfer content does not match its declared size and digest")
    bodies = dict(content)
    bodies["contribution.json"] = prepared.descriptor_json.encode("utf-8")
    seen = set()
    for target in prepared.transfer.upload_targets:
        if target.logical_path in seen or target.logical_path not in bodies:
            raise ValueError("Transfer target is duplicate or undeclared")
        seen.add(target.logical_path)
        if sha256(bodies[target.logical_path]).hexdigest() != target.digest_sha256:
            raise ValueError("Transfer target digest differs from supplied content")
        url = urlsplit(target.upload_url)
        local = url.hostname in {"127.0.0.1", "localhost", "::1"}
        if (
            not url.hostname
            or url.username
            or url.password
            or url.fragment
            or not (url.scheme == "https" or (url.scheme == "http" and local))
        ):
            raise ValueError("Transfer requires HTTPS or explicit loopback storage")
        if any(
            name.lower() in {"authorization", "cookie", "proxy-authorization", "host"}
            for name in target.required_headers
        ):
            raise ValueError(
                "Transfer instructions contain forbidden credential or routing headers"
            )
    return bodies


def upload_bytes_sync(prepared: ContributionUploadPrepared, content: Mapping[str, bytes]) -> None:
    """Blocking twin of ``upload_bytes`` with the same validation and client policy."""
    bodies = _transfer_content(prepared, content)
    with httpx.Client(timeout=30.0, follow_redirects=False, trust_env=False) as client:
        for target in prepared.transfer.upload_targets:
            client.cookies.clear()
            with client.stream(
                "PUT",
                target.upload_url,
                headers=target.required_headers,
                content=bodies[target.logical_path],
            ) as response:
                if response.status_code not in (200, 201, 204):
                    raise ValueError(
                        f"Artifact byte transfer failed with HTTP {response.status_code}"
                    )


async def upload_bytes(prepared: ContributionUploadPrepared, content: Mapping[str, bytes]) -> None:
    """Transfer supplied assets after validating every target; finalize separately.

    A fresh credential-free client prevents backend authorization/cookie leakage.
    Partial failures propagate; obtain fresh prepare instructions before retrying.
    Existing objects may be omitted by prepare, but finalize checks completeness.
    """
    bodies = _transfer_content(prepared, content)
    async with (
        asyncio.timeout(300),
        httpx.AsyncClient(timeout=30.0, follow_redirects=False, trust_env=False) as client,
    ):
        for target in prepared.transfer.upload_targets:
            # Targets are independent capabilities; do not carry response cookies
            # between them or buffer unneeded storage response bodies.
            client.cookies.clear()
            async with client.stream(
                "PUT",
                target.upload_url,
                headers=target.required_headers,
                content=bodies[target.logical_path],
            ) as response:
                if response.status_code not in (200, 201, 204):
                    raise ValueError(
                        f"Artifact byte transfer failed with HTTP {response.status_code}"
                    )
