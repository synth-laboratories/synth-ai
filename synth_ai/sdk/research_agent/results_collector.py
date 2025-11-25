from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable

from synth_ai.sdk.research_agent.container_builder import ContainerBackend


class ResultsCollector:
    """Collect and persist artifacts from a container run."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def collect_from_container(
        self,
        backend: ContainerBackend,
        container_id: str,
        patterns: Iterable[str],
    ) -> Dict[str, Path]:
        """Fetch artifacts via the backend and write them to disk."""
        artifacts = await backend.collect_artifacts(container_id, patterns)
        saved: Dict[str, Path] = {}
        for name, content in artifacts.items():
            path = self.output_dir / Path(name).name
            path.write_bytes(content)
            saved[name] = path
        return saved

    def create_manifest(self, saved_files: Dict[str, Path]) -> Path:
        """Create a manifest describing collected artifacts."""
        manifest = {
            "files": [
                {
                    "name": name,
                    "path": str(path.relative_to(self.output_dir)),
                    "size_bytes": path.stat().st_size,
                    "type": self._classify_file(name),
                }
                for name, path in saved_files.items()
            ],
            "collected_at": datetime.now(UTC).isoformat(),
            "total_size_bytes": sum(path.stat().st_size for path in saved_files.values()),
        }
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return manifest_path

    def _classify_file(self, filename: str) -> str:
        """Coarse file typing to make downstream filtering simpler."""
        lowered = filename.lower()
        if lowered.endswith(".json"):
            if "result" in lowered or "metric" in lowered:
                return "metrics"
            if "config" in lowered:
                return "config"
            return "data"
        if lowered.endswith(".log"):
            return "logs"
        if lowered.endswith(".md"):
            return "documentation"
        if lowered.endswith((".toml", ".yaml", ".yml")):
            return "config"
        if lowered == "diff.patch":
            return "code_changes"
        return "other"
