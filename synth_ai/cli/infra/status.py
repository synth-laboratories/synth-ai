#!/usr/bin/env python3
"""Compatibility wrapper for legacy status imports."""

from __future__ import annotations

import click

from synth_ai.cli.commands.status import register as _register_status


def register(cli: click.Group) -> None:
    """Register status subcommands on the provided CLI group."""
    _register_status(cli)


__all__ = ["register"]
