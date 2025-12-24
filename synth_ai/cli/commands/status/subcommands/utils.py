"""Shared helpers for status subcommands."""

from __future__ import annotations

from typing import Any

import click
import json


class StatusAPIClient:
    def __init__(self, config: Any) -> None:
        self.config = config

    async def __aenter__(self) -> "StatusAPIClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def list_jobs(self, **_: Any) -> list[dict[str, Any]]:
        return []

    async def get_job_events(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        return []

    async def get_file(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {}

    async def list_models(self, **_: Any) -> list[dict[str, Any]]:
        return []

    async def list_job_runs(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        return []

    async def list_files(self, **_: Any) -> list[dict[str, Any]]:
        return []


def print_json(data: Any) -> None:
    click.echo(json.dumps(data))
