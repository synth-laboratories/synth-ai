#!/usr/bin/env python3
"""Allow running synth_ai.cli as a module: python -m synth_ai.cli."""

from . import cli

if __name__ == "__main__":
    cli()
