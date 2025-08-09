#!/usr/bin/env python3
"""
Entry point for Synth AI TUI dashboard.

Usage:
    python -m synth_ai.tui
    python -m synth_ai.tui --url sqlite+aiosqlite:///path/to/db
"""

from .dashboard import main

if __name__ == "__main__":
    main()