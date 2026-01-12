"""Status command group."""

import asyncio
import contextlib
import json
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import click
import httpx

from synth_ai.core.urls import BACKEND_URL_BASE
from synth_ai.sdk.artifacts.config import resolve_backend_config
from synth_ai.sdk.session.client import AgentSessionClient
from synth_ai.sdk.session.exceptions import SessionNotFoundError
from synth_ai.sdk.session.models import AgentSession

DEFAULT_TIMEOUT = 30.0


@dataclass
class BackendConfig:
    base_url: str
    api_key: str | None
    timeout: float = DEFAULT_TIMEOUT


class StatusAPIError(RuntimeError):
    """Raised when status API calls fail."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def build_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}", "X-API-Key": api_key}


def resolve_status_config(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
) -> BackendConfig:
    if base_url is None:
        base_url = BACKEND_URL_BASE
    if api_key is None:
        api_key = os.getenv("SYNTH_API_KEY") or None
    return BackendConfig(
        base_url=base_url or "",
        api_key=api_key,
        timeout=timeout or DEFAULT_TIMEOUT,
    )


class StatusAPIClient:
    def __init__(self, config: BackendConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "StatusAPIClient":
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                headers=build_headers(self._config.api_key),
                timeout=self._config.timeout,
            )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        assert self._client is not None
        resp = await self._client.get(path, params=params)
        if resp.status_code >= 400:
            detail = resp.json().get("detail", "")
            raise StatusAPIError(detail or "Request failed", status_code=resp.status_code)
        return resp.json()

    async def _post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        assert self._client is not None
        resp = await self._client.post(path, json=payload or {})
        if resp.status_code >= 400:
            detail = resp.json().get("detail", "")
            raise StatusAPIError(detail or "Request failed", status_code=resp.status_code)
        return resp.json()

    async def list_jobs(
        self,
        *,
        status: str | None = None,
        job_type: str | None = None,
        created_after: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = status
        if job_type is not None:
            params["type"] = job_type
        if created_after is not None:
            params["created_after"] = created_after
        if limit is not None:
            params["limit"] = limit
        payload = await self._get("/learning/jobs", params=params or None)
        return payload.get("jobs", [])

    async def get_job(self, job_id: str) -> dict[str, Any]:
        return await self._get(f"/learning/jobs/{job_id}")

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        return await self._post(f"/learning/jobs/{job_id}/cancel")

    async def list_models(
        self,
        *,
        limit: int | None = None,
        model_type: str | None = None,
    ) -> list[dict[str, Any]]:
        if model_type:
            payload = await self._get(f"/learning/models/{model_type}")
            return payload.get("models", [])
        params = {"limit": limit} if limit is not None else None
        payload = await self._get("/learning/models", params=params)
        return payload.get("models", [])

    async def list_job_runs(self, job_id: str) -> list[dict[str, Any]]:
        payload = await self._get(f"/jobs/{job_id}/runs")
        return payload.get("runs", [])

    async def get_job_events(
        self, job_id: str, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        params = {"limit": limit} if limit is not None else None
        payload = await self._get(f"/learning/jobs/{job_id}/events", params=params)
        return payload.get("events", [])

    async def list_files(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        params = {"limit": limit} if limit is not None else None
        payload = await self._get("/learning/files", params=params)
        return payload.get("files", [])

    async def get_file(self, file_id: str) -> dict[str, Any]:
        return await self._get(f"/learning/files/{file_id}")


def print_json(data: Any) -> None:
    click.echo(json.dumps(data))


def _format_currency(value: Decimal) -> str:
    return f"${value:.2f}"


def _format_int(value: Decimal | int) -> str:
    return f"{int(value):,}"


def _session_exceeded(session: AgentSession) -> bool:
    if session.status == "limit_exceeded":
        return True
    return any(limit.current_usage >= limit.limit_value for limit in session.limits)


def _render_session(session: AgentSession) -> None:
    click.echo(f"Session: {session.session_id}")
    click.echo(f"Status: {session.status}")
    click.echo(f"Usage: {_format_currency(session.usage.cost_usd)}")
    click.echo(f"Tokens: {_format_int(session.usage.tokens)}")
    for limit in session.limits:
        if limit.metric_type == "cost_usd":
            click.echo(f"Limit: {_format_currency(limit.limit_value)}")
        elif limit.metric_type == "tokens":
            click.echo(f"Limit: {_format_int(limit.limit_value)}")
    if _session_exceeded(session):
        click.echo("✗ EXCEEDED")
        click.echo("WARNING: Session limits exceeded!")
    else:
        click.echo("✓ OK")


@click.group()
def status() -> None:
    """Inspect training jobs, files, models, and sessions."""


@status.command()
def summary() -> None:
    """Show a summary of jobs, models, and files."""

    async def _run() -> None:
        config = resolve_status_config()
        async with StatusAPIClient(config) as client:
            try:
                jobs = await client.list_jobs()
            except StatusAPIError:
                jobs = []
            with contextlib.suppress(StatusAPIError):
                await client.list_models()
            with contextlib.suppress(StatusAPIError):
                await client.list_files()

        click.echo("Training Jobs")
        for job in jobs:
            click.echo(str(job))

    asyncio.run(_run())


@status.group()
def jobs() -> None:
    """Inspect training jobs."""


@jobs.command()
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option("--status", default=None)
def list(as_json: bool, status: str | None) -> None:
    """List training jobs."""

    async def _run() -> None:
        config = resolve_status_config()
        async with StatusAPIClient(config) as client:
            data = await client.list_jobs(status=status)
            if as_json:
                print_json(data)
            else:
                click.echo(data)

    asyncio.run(_run())


@jobs.command()
@click.argument("job_id")
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option("--tail", default=None, type=int)
def logs(job_id: str, as_json: bool, tail: int | None) -> None:
    """Show job events."""

    async def _run() -> None:
        config = resolve_status_config()
        async with StatusAPIClient(config) as client:
            data = await client.get_job_events(job_id, limit=tail)
            if as_json:
                print_json(data)
            else:
                click.echo(data)

    asyncio.run(_run())


@status.group()
def models() -> None:
    """Inspect training models."""


@models.command(name="list")
@click.option("--limit", default=None, type=int)
@click.option("--type", "model_type", default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
def list_models(limit: int | None, model_type: str | None, as_json: bool) -> None:
    """List models."""

    async def _run() -> None:
        config = resolve_status_config()
        async with StatusAPIClient(config) as client:
            data = await client.list_models(limit=limit, model_type=model_type)
            if as_json:
                print_json(data)
            else:
                click.echo(data)

    asyncio.run(_run())


@status.group()
def runs() -> None:
    """Inspect job runs."""


@runs.command(name="list")
@click.argument("job_id")
@click.option("--json", "as_json", is_flag=True, default=False)
def runs_list(job_id: str, as_json: bool) -> None:
    """List runs for a job."""

    async def _run() -> None:
        config = resolve_status_config()
        async with StatusAPIClient(config) as client:
            data = await client.list_job_runs(job_id)
            if as_json:
                print_json(data)
            else:
                click.echo(data)

    asyncio.run(_run())


@status.group()
def files() -> None:
    """Inspect files."""


@files.command()
@click.argument("file_id")
@click.option("--json", "as_json", is_flag=True, default=False)
def get(file_id: str, as_json: bool) -> None:
    """Fetch a file record."""

    async def _run() -> None:
        config = resolve_status_config()
        async with StatusAPIClient(config) as client:
            data = await client.get_file(file_id)
            if as_json:
                print_json(data)
            else:
                click.echo(data)

    asyncio.run(_run())


@status.command()
@click.option("--session-id", default="", help="Session ID to inspect.")
def session(session_id: str) -> None:
    """Inspect agent session status."""

    async def _run() -> None:
        resolved_session_id = session_id or os.getenv("SYNTH_SESSION_ID", "")
        config = resolve_backend_config()
        client = AgentSessionClient(base_url=config.base_url, api_key=config.api_key)

        if resolved_session_id:
            try:
                agent_session = await client.get(resolved_session_id)
            except SessionNotFoundError:
                click.echo("Session not found")
                return
            _render_session(agent_session)
            return

        sessions = await client.list(status="active")
        if not sessions:
            click.echo("No active session found")
            return
        _render_session(sessions[0])

    asyncio.run(_run())


__all__ = [
    "BackendConfig",
    "StatusAPIClient",
    "StatusAPIError",
    "status",
]
