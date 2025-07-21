from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel
import shutil


class Artifact(BaseModel, ABC):
    """Generic artifact representation."""

    artifact_type: str
    filename: str

    @abstractmethod
    def bytes(self) -> bytes:
        """Return raw bytes of the artifact."""
        ...

    def save(self, root: Path) -> None:
        """Save artifact bytes to `root/filename`."""
        p = root / self.filename
        p.write_bytes(self.bytes())


class FileArtifact(Artifact):
    """Artifact representing a file's contents."""

    artifact_type: str = "file"
    content: bytes

    def bytes(self) -> bytes:
        return self.content


class ArtifactStore:
    """Filesystem-based store for artifacts."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def add(self, art: Artifact) -> None:
        art.save(self.root)

    def list(self) -> list[Path]:
        return list(self.root.iterdir())

    def export_tar(self, out: Path) -> None:
        """Generate `out.tar.gz` from artifact directory."""
        shutil.make_archive(out.with_suffix(""), "gztar", self.root)
