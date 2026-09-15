"""Explicit in-memory Contribution transfer, separate from authenticated API calls.

See sibling docs/drafts/synth-index-api-design-2026-09-12.md.
This bounded convenience path serves small reports/repos, not streaming datasets.
No filesystem discovery, automatic finalization, retries, or public release.
"""

import asyncio
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import httpx

from .contributions import ContributionUploadPrepared
from .package import ContributionPackage


def _safe_file(root: Path, logical_path: str) -> Path:
    parts = PurePosixPath(logical_path).parts
    if (
        not parts
        or logical_path.startswith("/")
        or "\\" in logical_path
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"Unsafe package asset path: {logical_path}")
    path = root.joinpath(*parts)
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError(f"Package asset escapes its root: {logical_path}")
    return path


def _file_digest(path: Path) -> tuple[int, str]:
    digest = sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _file_blocks(handle):
    yield from iter(lambda: handle.read(1024 * 1024), b"")


def verify_package_directory(root: Path) -> ContributionPackage:
    """Verify declared package files without buffering a research arc in memory."""
    directory = root.resolve(strict=True)
    descriptor_path = _safe_file(directory, "contribution.json")
    if descriptor_path.stat().st_size > 1_048_576:
        raise ValueError("Contribution descriptor exceeds 1 MiB")
    package = ContributionPackage.model_validate_json(descriptor_path.read_bytes())
    declared = {"contribution.json"}
    total = 0
    for asset in package.assets:
        logical_path = asset.object.logical_path
        if logical_path in declared:
            raise ValueError("Package declares a duplicate or reserved path")
        declared.add(logical_path)
        size, digest = _file_digest(_safe_file(directory, logical_path))
        if size != asset.object.size_bytes or digest != asset.object.digest_sha256:
            raise ValueError(f"Package asset integrity mismatch: {logical_path}")
        total += size
    if total > 1024 * 1024 * 1024:
        raise ValueError("Package exceeds 1 GiB")
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != declared:
        raise ValueError(f"Package has undeclared or missing files: {sorted(actual ^ declared)}")
    return package


def _check_target(target, expected_digest: str) -> None:
    if target.digest_sha256 != expected_digest:
        raise ValueError("Transfer target digest differs from declared content")
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
        raise ValueError("Transfer instructions contain forbidden credential or routing headers")


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
        _check_target(target, sha256(bodies[target.logical_path]).hexdigest())
    return bodies


def upload_directory_sync(prepared: ContributionUploadPrepared, root: Path) -> None:
    """Stream declared files to fresh create-only targets; retry via prepare again.

    ``root`` retains the converter's original descriptor. The server-issued
    descriptor is transferred from ``prepared`` after server ID rebinding.
    """
    package = verify_package_directory(root)
    rebound = ContributionPackage.model_validate_json(prepared.descriptor_json)
    if package.assets != rebound.assets:
        raise ValueError("Server descriptor changes the converted asset declarations")
    directory = root.resolve(strict=True)
    paths = {asset.object.logical_path: asset.object.digest_sha256 for asset in rebound.assets}
    sizes = {asset.object.logical_path: asset.object.size_bytes for asset in rebound.assets}
    descriptor = prepared.descriptor_json.encode("utf-8")
    paths["contribution.json"] = sha256(descriptor).hexdigest()
    sizes["contribution.json"] = len(descriptor)
    seen = set()
    for target in prepared.transfer.upload_targets:
        if target.logical_path in seen or target.logical_path not in paths:
            raise ValueError("Transfer target is duplicate or undeclared")
        seen.add(target.logical_path)
        _check_target(target, paths[target.logical_path])
        required_length = next(
            (
                value
                for name, value in target.required_headers.items()
                if name.lower() == "content-length"
            ),
            None,
        )
        if required_length is not None and required_length != str(sizes[target.logical_path]):
            raise ValueError("Transfer target Content-Length differs from declared bytes")
    with httpx.Client(timeout=300.0, follow_redirects=False, trust_env=False) as client:
        for target in prepared.transfer.upload_targets:
            client.cookies.clear()
            headers = dict(target.required_headers)
            if not any(name.lower() == "content-length" for name in headers):
                headers["Content-Length"] = str(sizes[target.logical_path])
            if target.logical_path == "contribution.json":
                content = descriptor
                response = client.put(target.upload_url, headers=headers, content=content)
            else:
                with _safe_file(directory, target.logical_path).open("rb") as handle:
                    response = client.put(
                        target.upload_url,
                        headers=headers,
                        content=_file_blocks(handle),
                    )
            if response.status_code not in (200, 201, 204):
                raise ValueError(
                    f"Artifact byte transfer failed for {target.logical_path} with HTTP {response.status_code}"
                )


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
