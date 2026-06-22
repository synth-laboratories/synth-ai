from __future__ import annotations

from pathlib import Path


def test_old_surface_archive_retains_container_tunnel_and_pool_code() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    archive_root = repo_root.parent / "research" / "old" / "synth_ai"
    expected_paths = [
        archive_root / "sdk" / "container" / "client.py",
        archive_root / "sdk" / "environment_pools.py",
        archive_root / "sdk" / "tunnels.py",
        archive_root / "core" / "tunnels" / "synth_tunnel.py",
    ]

    missing = [str(path) for path in expected_paths if not path.exists()]
    assert not missing, (
        f"expected archived legacy surface under research/old/synth_ai, missing: {missing}"
    )
