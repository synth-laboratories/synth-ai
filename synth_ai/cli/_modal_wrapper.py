from __future__ import annotations

import importlib
import sys


def main() -> int:
    # Apply Typer compatibility patch before Modal CLI bootstraps Click/Typer internals.
    try:
        module = importlib.import_module("synth_ai.cli._typer_patch")
    except Exception:
        module = None
    if module is not None:
        patch = getattr(module, "patch_typer_make_metavar", None)
        if callable(patch):
            patch()

    from modal.__main__ import main as modal_main

    # Present ourselves as the upstream `modal` CLI so Typer/Click parsing stays intact.
    if sys.argv:
        sys.argv[0] = "modal"
    else:
        sys.argv = ["modal"]

    result = modal_main()
    return result if result is not None else 0


if __name__ == "__main__":
    sys.exit(main())
