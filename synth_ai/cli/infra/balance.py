#!/usr/bin/env python3
"""
CLI: check remaining credit balance from Synth backend.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Any, cast

import click
import requests
from requests import Response
from rich import box
from rich.console import Console
from rich.table import Table


def _load_base_url_module() -> tuple[str, Callable[[], tuple[str, str]]]:
    try:
        module = cast(Any, importlib.import_module("synth_ai.core.env"))
        default = cast(str, module.PROD_BASE_URL_DEFAULT)
        getter = cast(Callable[[], tuple[str, str]], module.get_backend_from_env)
        return str(default), getter
    except Exception:
        return "https://api.usesynth.ai", lambda: ("https://api.usesynth.ai", "")


PROD_BASE_URL_DEFAULT, _get_backend_from_env = _load_base_url_module()
PROD_BACKEND_BASE = f"{PROD_BASE_URL_DEFAULT.rstrip('/')}/api/v1"


def _get_default_base_url() -> str:
    # Prefer explicit backend variables; else default to prod backend
    for var in ("SYNTH_BACKEND_BASE_URL", "BACKEND_BASE_URL", "SYNTH_BASE_URL"):
        val = os.getenv(var)
        if val:
            return val
    base, _ = _get_backend_from_env()
    base = base.rstrip("/")
    if base.endswith("/api"):
        base = base[: -len("/api")]
    return f"{base}/api/v1"


def _ensure_api_v1_prefix(base_url: str) -> str:
    """Ensure the base URL includes the /api/v1 prefix.

    Accepts either a full prefix (http://host:port/api/v1) or a root
    service URL (http://host:port). If no '/api' segment is present, append
    '/api/v1'.
    """
    b = base_url.rstrip("/")
    if b.endswith("/api") or b.endswith("/api/v1") or "/api/" in b:
        return b
    return b + "/api/v1"


def _resolve_api_key(explicit_key: str | None) -> tuple[str | None, str | None]:
    if explicit_key:
        return explicit_key, "--api-key"
    # Try multiple env vars commonly used in this repo
    for var in ("SYNTH_BACKEND_API_KEY", "SYNTH_API_KEY", "DEFAULT_DEV_API_KEY"):
        val = os.getenv(var)
        if val:
            return val, var
    return None, None


def _auth_headers(api_key: str | None) -> dict[str, str]:
    key, _ = _resolve_api_key(api_key)
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def register(cli):
    @cli.command()
    @click.option(
        "--base-url",
        default=_get_default_base_url,
        show_default=True,
        help="Synth backend base URL (prefix like http://host:port/api/v1)",
    )
    @click.option(
        "--api-key",
        envvar="SYNTH_API_KEY",
        help="API key for the Synth backend (or set SYNTH_API_KEY)",
    )
    @click.option(
        "--usage/--no-usage",
        default=False,
        help="Also fetch recent usage summary",
    )
    def balance(base_url: str, api_key: str | None, usage: bool):
        """Show your remaining credit balance from the Synth backend."""
        console = Console()

        key_val, key_src = _resolve_api_key(api_key)
        if not key_val:
            console.print("[red]Missing API key.[/red] Set via --api-key or SYNTH_API_KEY env var.")
            return

        base = _ensure_api_v1_prefix(base_url)

        # No special-casing for modal.run domains; honor the provided base URL

        try:
            resp: Response = requests.get(
                f"{base}/balance/current",
                headers=_auth_headers(api_key),
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            cents = int(data.get("balance_cents", 0))
            dollars = float(data.get("balance_dollars", cents / 100.0))
            console.print(f"Balance: [bold]${dollars:,.2f}[/bold]")

            # Try to print compact spend summary for 24h and 7d
            try:
                u: Response = requests.get(
                    f"{base}/balance/usage/windows",
                    params={"hours": "24,168"},
                    headers=_auth_headers(api_key),
                    timeout=10,
                )
                if u.ok:
                    uj = u.json()
                    rows = uj.get("windows", [])
                    windows = {
                        int(r.get("window_hours")): r
                        for r in rows
                        if isinstance(r.get("window_hours"), int)
                    }

                    def _usd(c):
                        try:
                            return f"${(int(c) / 100):,.2f}"
                        except Exception:
                            return "$0.00"

                    if 24 in windows or 168 in windows:
                        t = Table(
                            title="Spend (Tokens vs GPU)", box=box.SIMPLE, header_style="bold"
                        )
                        t.add_column("Window")
                        t.add_column("Tokens", justify="right")
                        t.add_column("GPU", justify="right")
                        t.add_column("Total", justify="right")
                        for h, label in ((24, "24h"), (168, "7d")):
                            if h in windows:
                                w = windows[h]
                                t.add_row(
                                    label,
                                    _usd(w.get("token_spend_cents", 0)),
                                    _usd(w.get("gpu_spend_cents", 0)),
                                    _usd(w.get("total_spend_cents", 0)),
                                )
                        console.print(t)
                elif usage:
                    # Fallback to older summary if requested explicitly
                    u2: Response = requests.get(
                        f"{base}/balance/usage",
                        headers=_auth_headers(api_key),
                        timeout=10,
                    )
                    if u2.ok:
                        uj = u2.json()
                        cm = uj.get("current_month", {})
                        l30 = uj.get("last_30_days", {})
                        t = Table(title="Usage Summary", box=box.SIMPLE, header_style="bold")
                        t.add_column("Window")
                        t.add_column("Token Spend", justify="right")
                        t.add_column("GPU Spend", justify="right")
                        t.add_column("Total", justify="right")
                        t.add_row(
                            "Current Month",
                            f"${(cm.get('token_spend_cents', 0) / 100):,.2f}",
                            f"${(cm.get('gpu_spend_cents', 0) / 100):,.2f}",
                            f"${(cm.get('total_spend_cents', 0) / 100):,.2f}",
                        )
                        t.add_row(
                            "Last 30 days",
                            f"${(l30.get('token_spend_cents', 0) / 100):,.2f}",
                            f"${(l30.get('gpu_spend_cents', 0) / 100):,.2f}",
                            f"${(l30.get('total_spend_cents', 0) / 100):,.2f}",
                        )
                        console.print(t)
            except Exception:
                # Silent failure on usage summary
                pass

        except requests.HTTPError as e:
            try:
                detail = e.response.json().get("detail") if e.response else None
            except Exception:
                detail = None
            if e.response is not None and e.response.status_code == 401:
                key_dbg, key_src = _resolve_api_key(api_key)
                shown = (key_dbg[:6] + "…" + key_dbg[-4:]) if key_dbg else "<none>"
                console.print(
                    "[red]Unauthorized (401).[/red] The API key was not accepted by the backend."
                )
                console.print(
                    f"- Using base URL: {base}\n- API key (masked): {shown}\n- Key source: {key_src or '<none>'}\n- Ensure this key exists in the backend DB (table api_keys) and is active."
                )
                console.print(
                    "If running locally, you can seed a dev key by setting ENVIRONMENT=dev and ensuring the DB has no API keys (auto-seed path), or create one via your admin path."
                )
            else:
                console.print(f"[red]HTTP error:[/red] {e} {detail or ''}")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
