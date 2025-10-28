from __future__ import annotations

from importlib import metadata as _metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path


def _expected_version() -> str:
    try:
        return _metadata.version("synth-ai")
    except PackageNotFoundError:
        try:
            import tomllib as _toml  # Python 3.11+
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as _toml  # type: ignore[no-redef]
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with pyproject.open("rb") as fh:
            data = _toml.load(fh)
        return str(data["project"]["version"])


def test_version_matches_expected():
    import synth_ai

    assert getattr(synth_ai, "__version__", ""), "synth_ai.__version__ missing"
    assert synth_ai.__version__ == _expected_version()
