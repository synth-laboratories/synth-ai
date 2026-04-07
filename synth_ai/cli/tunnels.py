"""Tunnels CLI commands."""

from __future__ import annotations

import json
import os
from typing import Any

import click


def _client(api_key: str | None, backend_url: str | None):
    from synth_ai.sdk.tunnels import TunnelsClient

    return TunnelsClient(api_key=api_key, backend_base=backend_url)


def _echo(data: Any) -> None:
    click.echo(json.dumps(data, indent=2, default=str))


@click.group()
def tunnels() -> None:
    """Manage backend-owned tunnels and leases."""


@tunnels.command("list")
@click.option("--status", "status_filter", default=None)
@click.option("--include-deleted", is_flag=True, default=False)
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", default=os.getenv("SYNTH_BACKEND_URL"))
def list_tunnels(
    status_filter: str | None,
    include_deleted: bool,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _echo(
        _client(api_key, backend_url).list(
            status_filter=status_filter,
            include_deleted=include_deleted,
        )
    )


@tunnels.command("create")
@click.option("--subdomain", required=True)
@click.option("--local-port", required=True, type=int)
@click.option("--local-host", default="127.0.0.1")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", default=os.getenv("SYNTH_BACKEND_URL"))
def create_tunnel(
    subdomain: str,
    local_port: int,
    local_host: str,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _echo(
        _client(api_key, backend_url).create(
            subdomain=subdomain,
            local_port=local_port,
            local_host=local_host,
        )
    )


@tunnels.command("delete")
@click.argument("tunnel_id")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", default=os.getenv("SYNTH_BACKEND_URL"))
def delete_tunnel(tunnel_id: str, api_key: str | None, backend_url: str | None) -> None:
    _echo(_client(api_key, backend_url).delete(tunnel_id))


@tunnels.command("rotate")
@click.option("--local-port", default=8000, type=int)
@click.option("--local-host", default="127.0.0.1")
@click.option("--reason", default=None)
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", default=os.getenv("SYNTH_BACKEND_URL"))
def rotate_tunnel(
    local_port: int,
    local_host: str,
    reason: str | None,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _echo(
        _client(api_key, backend_url).rotate(
            local_port=local_port,
            local_host=local_host,
            reason=reason,
        )
    )


@tunnels.command("lease-create")
@click.option("--client-instance-id", required=True)
@click.option("--local-host", default="127.0.0.1")
@click.option("--local-port", required=True, type=int)
@click.option("--app-name", default=None)
@click.option("--provider", default="ngrok")
@click.option("--ttl", "requested_ttl_seconds", default=3600, type=int)
@click.option("--reuse/--no-reuse", default=True)
@click.option("--idempotency-key", default=None)
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", default=os.getenv("SYNTH_BACKEND_URL"))
def lease_create(
    client_instance_id: str,
    local_host: str,
    local_port: int,
    app_name: str | None,
    provider: str,
    requested_ttl_seconds: int,
    reuse: bool,
    idempotency_key: str | None,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _echo(
        _client(api_key, backend_url).create_lease(
            client_instance_id=client_instance_id,
            local_host=local_host,
            local_port=local_port,
            app_name=app_name,
            provider_preference=provider,
            requested_ttl_seconds=requested_ttl_seconds,
            reuse_connector=reuse,
            idempotency_key=idempotency_key,
        )
    )


@tunnels.command("lease-list")
@click.option("--client-instance-id", default=None)
@click.option("--include-expired", is_flag=True, default=False)
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", default=os.getenv("SYNTH_BACKEND_URL"))
def lease_list(
    client_instance_id: str | None,
    include_expired: bool,
    api_key: str | None,
    backend_url: str | None,
) -> None:
    _echo(
        _client(api_key, backend_url).list_leases(
            client_instance_id=client_instance_id,
            include_expired=include_expired,
        )
    )


@tunnels.command("lease-release")
@click.argument("lease_id")
@click.option("--api-key", envvar="SYNTH_API_KEY")
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", default=os.getenv("SYNTH_BACKEND_URL"))
def lease_release(lease_id: str, api_key: str | None, backend_url: str | None) -> None:
    _echo(_client(api_key, backend_url).release_lease(lease_id))
