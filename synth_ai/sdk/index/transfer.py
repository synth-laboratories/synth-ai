"""Explicit in-memory Contribution transfer, separate from authenticated API calls.

See sibling docs/drafts/synth-index-api-design-2026-09-12.md.
This bounded convenience path serves small reports/repos, not streaming datasets.
No filesystem discovery, automatic finalization, retries, or public release.
"""

import asyncio
from collections.abc import Iterator, Mapping
from hashlib import sha256
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import httpx

from .contributions import ContributionUploadPrepared
from .package import ContributionPackage

# One directory transfer is bounded end to end and per request, so a stalled
# storage endpoint cannot hold an intake open indefinitely.
DIRECTORY_TRANSFER_BUDGET_SECONDS = 900.0
DIRECTORY_REQUEST_BUDGET_SECONDS = 300.0
DIRECTORY_TRANSFER_CONCURRENCY = 4
_BLOCK_BYTES = 1024 * 1024
_ACCEPTED = (200, 201, 204)
# Signed create-only targets are short-lived: storage answers an expired or
# already-consumed capability with these, which is a reason to prepare again,
# not a reason to lose the objects already stored.
_EXPIRED = (400, 401, 403, 409)


class TransferTargetsExpired(RuntimeError):
    """Storage refused a signed target; prepare again and transfer what remains."""


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


def _file_blocks(handle) -> Iterator[bytes]:
    yield from iter(lambda: handle.read(_BLOCK_BYTES), b"")


def _plan_directory_transfer(
    prepared: ContributionUploadPrepared, root: Path
) -> tuple[Path, dict[str, int], bytes]:
    """Verify the directory against the server descriptor and check every target.

    Returns the resolved root, the declared size of each transferable path, and
    the exact descriptor bytes to send. Nothing is transferred until every target
    has been checked, so a bad instruction set fails before any byte leaves.
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
    return directory, sizes, descriptor


def _target_headers(target, size_bytes: int) -> dict[str, str]:
    headers = dict(target.required_headers)
    if not any(name.lower() == "content-length" for name in headers):
        headers["Content-Length"] = str(size_bytes)
    return headers


def _check_transfer_response(status_code: int, logical_path: str) -> None:
    if status_code in _ACCEPTED:
        return
    if status_code in _EXPIRED:
        raise TransferTargetsExpired(
            f"Storage refused the target for {logical_path} with HTTP {status_code}; "
            "prepare the upload again to obtain fresh targets"
        )
    raise ValueError(
        f"Artifact byte transfer failed for {logical_path} with HTTP {status_code}"
    )


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
    descriptor is transferred from ``prepared`` after server ID rebinding. A
    refused signed target raises ``TransferTargetsExpired`` so the caller can
    prepare again and transfer only what is still missing.
    """
    directory, sizes, descriptor = _plan_directory_transfer(prepared, root)
    with httpx.Client(
        timeout=DIRECTORY_REQUEST_BUDGET_SECONDS, follow_redirects=False, trust_env=False
    ) as client:
        for target in prepared.transfer.upload_targets:
            client.cookies.clear()
            headers = _target_headers(target, sizes[target.logical_path])
            if target.logical_path == "contribution.json":
                response = client.put(target.upload_url, headers=headers, content=descriptor)
            else:
                with _safe_file(directory, target.logical_path).open("rb") as handle:
                    response = client.put(
                        target.upload_url,
                        headers=headers,
                        content=_file_blocks(handle),
                    )
            _check_transfer_response(response.status_code, target.logical_path)


async def upload_directory(
    prepared: ContributionUploadPrepared,
    root: Path,
    *,
    concurrency: int = DIRECTORY_TRANSFER_CONCURRENCY,
) -> None:
    """Native async twin of ``upload_directory_sync``: streamed, bounded, credential-free.

    Each declared file is one separate storage request with its own create-only
    capability, streamed from disk in blocks instead of being buffered whole, so
    a research arc never has to fit in memory. File reads run in worker threads,
    which keeps the caller's event loop responsive during a large transfer. Each
    request uses its own fresh client that ignores the environment and carries no
    backend credential, so API authority is never forwarded to object storage and
    nothing storage returns is carried to the next target. The whole transfer is
    bounded, and so is each request; a refused signed target raises
    ``TransferTargetsExpired`` rather than being retried silently.
    """
    if not 1 <= concurrency <= 16:
        raise ValueError("Directory transfer concurrency must be between 1 and 16")
    directory, sizes, descriptor = _plan_directory_transfer(prepared, root)
    limit = asyncio.Semaphore(concurrency)

    async def stream_file(path: Path):
        handle = await asyncio.to_thread(path.open, "rb")
        try:
            while True:
                block = await asyncio.to_thread(handle.read, _BLOCK_BYTES)
                if not block:
                    return
                yield block
        finally:
            await asyncio.to_thread(handle.close)

    async def transfer(target) -> None:
        async with limit:
            headers = _target_headers(target, sizes[target.logical_path])
            content = (
                descriptor
                if target.logical_path == "contribution.json"
                else stream_file(_safe_file(directory, target.logical_path))
            )
            # One client per target: each is an independent capability, so no
            # cookie, connection or credential is shared between them.
            async with httpx.AsyncClient(
                timeout=DIRECTORY_REQUEST_BUDGET_SECONDS,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                response = await client.put(
                    target.upload_url, headers=headers, content=content
                )
            _check_transfer_response(response.status_code, target.logical_path)

    async with asyncio.timeout(DIRECTORY_TRANSFER_BUDGET_SECONDS):
        # One failure cancels the rest: a partial transfer is finished by
        # preparing again, never by continuing against stale instructions.
        async with asyncio.TaskGroup() as group:
            for target in prepared.transfer.upload_targets:
                group.create_task(transfer(target))


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
