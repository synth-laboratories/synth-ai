#!/usr/bin/env python3
"""Backward-compatible entrypoint — implementation lives in ``readme_runs/readme_smoke.py``."""

from readme_runs.readme_smoke import main

if __name__ == "__main__":
    raise SystemExit(main())
