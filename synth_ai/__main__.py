#!/usr/bin/env python3
"""
Allow running synth_ai as a module: python -m synth_ai
"""

from .cli import cli

if __name__ == "__main__":
    cli()
