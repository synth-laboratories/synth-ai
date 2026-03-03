"""Load the compiled synth_ai_py Rust extension for local source checkouts.

This package exists because the repository contains a top-level `synth_ai_py/`
crate directory, which can shadow the installed wheel module during local
development. We explicitly load the extension artifact from common local build
locations so `import synth_ai_py` resolves to the actual bindings.
"""

from __future__ import annotations

from importlib.machinery import EXTENSION_SUFFIXES, ExtensionFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from types import ModuleType


def _candidate_extension_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    pkg_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []

    # First preference: the compiled extension shipped inside the installed wheel.
    for suffix in EXTENSION_SUFFIXES:
        candidates.append(pkg_dir / f"synth_ai_py{suffix}")

    # Local checkout fallbacks for developer workflows.
    candidates.extend(
        [
            root / "target/release/libsynth_ai_py.so",
            root / "target/release/libsynth_ai_py.dylib",
            root / "target/release/synth_ai_py.dll",
            root / "target/debug/libsynth_ai_py.so",
            root / "target/debug/libsynth_ai_py.dylib",
            root / "target/debug/synth_ai_py.dll",
        ]
    )

    # Deduplicate while preserving order.
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _load_extension() -> ModuleType:
    errors: list[str] = []
    for candidate in _candidate_extension_paths():
        if not candidate.exists():
            continue
        try:
            loader = ExtensionFileLoader("synth_ai_py", str(candidate))
            spec = spec_from_loader("synth_ai_py", loader)
            if spec is None:
                continue
            module = module_from_spec(spec)
            loader.exec_module(module)
            return module
        except Exception as exc:  # pragma: no cover
            errors.append(f"{candidate}: {exc}")
    raise ImportError(
        "Unable to load compiled synth_ai_py extension. "
        "Build/install the Rust binding first. "
        f"attempted={_candidate_extension_paths()} errors={errors}"
    )


_ext = _load_extension()

for _name in dir(_ext):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_ext, _name)

if hasattr(_ext, "__all__"):
    __all__ = _ext.__all__
