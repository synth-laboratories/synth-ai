from __future__ import annotations

import sys


def main() -> int:
    # Apply Typer compatibility patch before Modal CLI bootstraps Click/Typer internals.
    try:
        from ._typer_patch import patch_typer_make_metavar

        patch_typer_make_metavar()
    except Exception:
        pass

    from modal.__main__ import main as modal_main

    # Present ourselves as the upstream `modal` CLI so Typer/Click parsing stays intact.
    if sys.argv:
        sys.argv[0] = "modal"
    else:
        sys.argv = ["modal"]

    return modal_main()


if __name__ == "__main__":
    sys.exit(main())

