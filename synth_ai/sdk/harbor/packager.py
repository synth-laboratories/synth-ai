"""HarborPackager - Package build context for Harbor deployments.

This module handles packaging a directory into a tar.gz archive suitable
for uploading to Harbor. It respects include/exclude globs and handles
file filtering.

Example:
    >>> from synth_ai.sdk.harbor import HarborBuildSpec
    >>> from synth_ai.sdk.harbor.packager import HarborPackager
    >>>
    >>> spec = HarborBuildSpec(
    ...     name="my-deployment",
    ...     dockerfile_path="./Dockerfile",
    ...     context_dir="./my-project",
    ... )
    >>>
    >>> packager = HarborPackager(spec)
    >>> context_base64 = packager.package()
    >>> print(f"Packaged {len(context_base64)} bytes")
"""

from __future__ import annotations

import base64
import fnmatch
import io
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .build_spec import HarborBuildSpec


class HarborPackager:
    """Package build context into tar.gz for Harbor upload.

    Handles file filtering based on include/exclude globs, creates a
    compressed tar archive, and returns it as base64.

    Attributes:
        spec: The HarborBuildSpec to package
        max_size_mb: Maximum allowed package size in MB (default: 100)
    """

    def __init__(self, spec: HarborBuildSpec, max_size_mb: int = 100) -> None:
        """Initialize packager with build spec.

        Args:
            spec: HarborBuildSpec defining what to package
            max_size_mb: Maximum package size in megabytes
        """
        self.spec = spec
        self.max_size_bytes = max_size_mb * 1024 * 1024

    def _should_include(self, rel_path: str) -> bool:
        """Check if a file should be included based on globs.

        Args:
            rel_path: Path relative to context_dir

        Returns:
            True if file should be included
        """
        # Check exclude patterns first
        for pattern in self.spec.exclude_globs:
            if fnmatch.fnmatch(rel_path, pattern):
                return False
            # Also check if any parent directory matches
            parts = Path(rel_path).parts
            for i in range(len(parts)):
                partial = "/".join(parts[: i + 1])
                if fnmatch.fnmatch(partial, pattern):
                    return False

        # Check include patterns
        return any(fnmatch.fnmatch(rel_path, pattern) for pattern in self.spec.include_globs)

    def _collect_files(self) -> list[tuple[Path, str]]:
        """Collect all files to include in the package.

        Returns:
            List of (absolute_path, archive_path) tuples
        """
        context_dir = Path(self.spec.context_dir).resolve()
        files: list[tuple[Path, str]] = []

        for path in context_dir.rglob("*"):
            if not path.is_file():
                continue

            rel_path = path.relative_to(context_dir)
            rel_str = str(rel_path).replace("\\", "/")  # Normalize for Windows

            if self._should_include(rel_str):
                files.append((path, rel_str))

        return files

    def package(self) -> str:
        """Package the build context into a base64-encoded tar.gz.

        Returns:
            Base64-encoded string of the tar.gz archive

        Raises:
            FileNotFoundError: If context_dir doesn't exist
            ValueError: If package exceeds max_size_mb
        """
        self.spec.validate_paths()

        files = self._collect_files()
        buffer = io.BytesIO()

        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for abs_path, archive_path in files:
                tar.add(abs_path, arcname=archive_path)

        data = buffer.getvalue()

        if len(data) > self.max_size_bytes:
            raise ValueError(
                f"Package size ({len(data) / 1024 / 1024:.1f} MB) exceeds "
                f"maximum ({self.max_size_bytes / 1024 / 1024:.1f} MB). "
                "Consider adding more patterns to exclude_globs."
            )

        return base64.b64encode(data).decode("ascii")

    def get_file_list(self) -> list[str]:
        """Get list of files that will be included in the package.

        Useful for debugging include/exclude patterns.

        Returns:
            List of relative paths that will be included
        """
        return [rel_path for _, rel_path in self._collect_files()]

    def get_package_info(self) -> dict[str, Any]:
        """Get information about the package without creating it.

        Returns:
            Dictionary with file count, estimated size, etc.
        """
        files = self._collect_files()
        total_size = sum(path.stat().st_size for path, _ in files)

        return {
            "file_count": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / 1024 / 1024,
            "context_dir": str(self.spec.context_dir),
            "dockerfile": str(self.spec.dockerfile_path),
        }
