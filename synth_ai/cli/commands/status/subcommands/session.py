"""Session status command for Synth CLI."""

from __future__ import annotations

import asyncio
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from synth_ai.cli.commands.status.config import resolve_backend_config
from synth_ai.cli.local.session import AgentSessionClient, SessionNotFoundError

console = Console()


@click.command("session", help="Show current agent session status and limits.")
@click.option(
    "--session-id",
    type=str,
    default=None,
    help="Session ID to check (default: uses active session from environment)"
)
@click.option(
    "--base-url",
    envvar="SYNTH_STATUS_BASE_URL",
    default=None,
    help="Synth backend base URL (defaults to environment configuration).",
)
@click.option(
    "--api-key",
    envvar="SYNTH_STATUS_API_KEY",
    default=None,
    help="API key for authenticated requests (falls back to Synth defaults).",
)
@click.option(
    "--timeout",
    default=30.0,
    show_default=True,
    type=float,
    help="HTTP request timeout in seconds.",
)
def session_status_cmd(
    session_id: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    timeout: float,
) -> None:
    """Show agent session status, limits, and current usage."""
    cfg = resolve_backend_config(base_url=base_url, api_key=api_key, timeout=timeout)
    
    async def _run() -> None:
        import os
        
        # Use provided session_id or try to get from environment
        if not session_id:
            session_id_env = os.getenv("SYNTH_SESSION_ID")
            if session_id_env:
                session_id_to_use = session_id_env
            else:
                # Try to get active session
                if not cfg.api_key:
                    console.print("[red]API key is required. Set SYNTH_API_KEY environment variable.[/red]")
                    return
                client = AgentSessionClient(f"{cfg.base_url}/api", cfg.api_key)
                try:
                    async with client._http:
                        sessions = await client.list(status="active", limit=1)
                        if sessions:
                            session_id_to_use = sessions[0].session_id
                        else:
                            console.print("[yellow]No active session found.[/yellow]")
                            console.print("Create a session with: [cyan]uvx synth-ai codex[/cyan] or [cyan]uvx synth-ai opencode[/cyan]")
                            return
                except Exception as e:
                    console.print(f"[red]Error fetching active session: {e}[/red]")
                    return
        else:
            session_id_to_use = session_id
        
        if not cfg.api_key:
            console.print("[red]API key is required. Set SYNTH_API_KEY environment variable.[/red]")
            return
        client = AgentSessionClient(f"{cfg.base_url}/api", cfg.api_key)
        try:
            session = await client.get(session_id_to_use)
            
            # Display session info
            table = Table(title=f"Session: {session.session_id[:16]}...", show_header=True, header_style="bold cyan")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Session ID", session.session_id)
            table.add_row("Status", f"[bold]{session.status}[/bold]")
            table.add_row("Created", session.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"))
            if session.ended_at:
                table.add_row("Ended", session.ended_at.strftime("%Y-%m-%d %H:%M:%S UTC"))
            if session.expires_at:
                table.add_row("Expires", session.expires_at.strftime("%Y-%m-%d %H:%M:%S UTC"))
            
            console.print(table)
            console.print()
            
            # Display usage summary
            usage_table = Table(title="Usage Summary", show_header=True, header_style="bold cyan")
            usage_table.add_column("Metric", style="cyan")
            usage_table.add_column("Value", style="green")
            
            usage_table.add_row("Tokens", f"{session.usage.tokens:,}")
            usage_table.add_row("Cost (USD)", f"${session.usage.cost_usd:.4f}")
            usage_table.add_row("GPU Hours", f"{session.usage.gpu_hours:.4f}")
            usage_table.add_row("API Calls", f"{session.usage.api_calls:,}")
            
            console.print(usage_table)
            console.print()
            
            # Display limits
            if session.limits:
                limits_table = Table(title="Session Limits", show_header=True, header_style="bold cyan")
                limits_table.add_column("Type", style="cyan")
                limits_table.add_column("Limit", style="green")
                limits_table.add_column("Used", style="yellow")
                limits_table.add_column("Remaining", style="green")
                limits_table.add_column("Status", style="bold")
                
                for limit in session.limits:
                    remaining = limit.remaining
                    status = "[green]✓ OK[/green]" if remaining > 0 else "[red]✗ EXCEEDED[/red]"
                    if limit.metric_type == "cost_usd":
                        limits_table.add_row(
                            "Cost (USD)",
                            f"${limit.limit_value:.2f}",
                            f"${limit.current_usage:.4f}",
                            f"${remaining:.4f}",
                            status
                        )
                    elif limit.metric_type == "tokens":
                        limits_table.add_row(
                            "Tokens",
                            f"{limit.limit_value:,.0f}",
                            f"{limit.current_usage:,.0f}",
                            f"{remaining:,.0f}",
                            status
                        )
                    elif limit.metric_type == "gpu_hours":
                        limits_table.add_row(
                            "GPU Hours",
                            f"{limit.limit_value:.2f}",
                            f"{limit.current_usage:.4f}",
                            f"{remaining:.4f}",
                            status
                        )
                    else:
                        limits_table.add_row(
                            limit.metric_type,
                            str(limit.limit_value),
                            str(limit.current_usage),
                            str(remaining),
                            status
                        )
                
                console.print(limits_table)
                
                # Warn if any limit exceeded
                exceeded = [limit for limit in session.limits if limit.current_usage >= limit.limit_value]
                if exceeded:
                    console.print()
                    console.print("[red][bold]⚠ WARNING: Session limits exceeded![/bold][/red]")
                    console.print("The session will reject new requests until limits are increased.")
                    for limit in exceeded:
                        console.print(f"  - {limit.metric_type}: {limit.current_usage} >= {limit.limit_value}")
            else:
                console.print("[yellow]No limits configured for this session.[/yellow]")
                
        except SessionNotFoundError:
            console.print(f"[red]Session not found: {session_id_to_use}[/red]")
        except Exception as e:
            console.print(f"[red]Error fetching session: {e}[/red]")
    
    asyncio.run(_run())

