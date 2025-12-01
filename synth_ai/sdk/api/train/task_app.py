from __future__ import annotations

import socket
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

import click
import requests

from .utils import CLIResult, http_get, run_cli


@dataclass(slots=True)
class TaskAppHealth:
    ok: bool
    health_status: int | None
    task_info_status: int | None
    detail: str | None = None


def _resolve_hostname_with_explicit_resolvers(hostname: str) -> str:
    """
    Resolve hostname using explicit resolvers (1.1.1.1, 8.8.8.8) first,
    then fall back to system resolver.
    
    This fixes resolver path issues where system DNS is slow or blocking.
    """
    # Try Cloudflare / Google first via `dig`, then fall back to system resolver
    for resolver_ip in ("1.1.1.1", "8.8.8.8"):
        try:
            result = subprocess.run(
                ["dig", f"@{resolver_ip}", "+short", hostname],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if result.returncode == 0 and result.stdout.strip():
                first = result.stdout.strip().splitlines()[0].strip()
                if first:
                    return first
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            continue
    
    # Fallback: system resolver
    return socket.gethostbyname(hostname)


def _resolve_url_to_ip(url: str) -> tuple[str, str]:
    """
    Resolve URL's hostname to IP using explicit resolvers, return (ip_url, hostname).
    
    NOTE: For HTTPS URLs, we DON'T resolve to IP because SSL/TLS requires SNI
    (Server Name Indication) with the hostname. Connecting via IP causes handshake
    failures since the server (e.g., Cloudflare) doesn't know which cert to present.
    
    Returns:
        Tuple of (url_with_ip, original_hostname)
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    
    # Skip resolution for localhost
    if not hostname or hostname in ("localhost", "127.0.0.1"):
        return url, hostname or ""
    
    # Skip resolution for HTTPS URLs - SSL/TLS requires hostname for SNI
    # Connecting to IP directly causes "SSLV3_ALERT_HANDSHAKE_FAILURE" because
    # the server doesn't know which certificate to present
    if parsed.scheme == "https":
        return url, hostname
    
    # Only resolve HTTP URLs to IP
    try:
        resolved_ip = _resolve_hostname_with_explicit_resolvers(hostname)
        # Replace hostname with IP in URL
        new_parsed = parsed._replace(netloc=f"{resolved_ip}:{parsed.port}" if parsed.port else resolved_ip)
        ip_url = urlunparse(new_parsed)
        return ip_url, hostname
    except Exception:
        # If resolution fails, return original URL
        return url, hostname or ""


def _health_response_ok(resp: requests.Response | None) -> tuple[bool, str]:
    if resp is None:
        return False, ""
    status = resp.status_code
    if status == 200:
        return True, ""
    if status in {401, 403}:
        try:
            payload = resp.json()
        except ValueError:
            payload = {}
        prefix = payload.get("expected_api_key_prefix")
        detail = str(payload.get("detail", ""))
        if prefix or "expected prefix" in detail.lower():
            note = "auth-optional"
            if prefix:
                note += f" (expected-prefix={prefix})"
            return True, note
    return False, ""


def check_task_app_health(base_url: str, api_key: str, *, timeout: float = 10.0, max_retries: int = 5) -> TaskAppHealth:
    # Send ALL known environment keys so the server can authorize any valid one
    import os

    headers = {"X-API-Key": api_key}
    aliases = (os.getenv("ENVIRONMENT_API_KEY_ALIASES") or "").strip()
    keys: list[str] = [api_key]
    if aliases:
        keys.extend([p.strip() for p in aliases.split(",") if p.strip()])
    if keys:
        headers["X-API-Keys"] = ",".join(keys)
        headers.setdefault("Authorization", f"Bearer {api_key}")
    base = base_url.rstrip("/")
    detail_parts: list[str] = []

    def _is_dns_error(exc: requests.RequestException) -> bool:
        """Check if exception is a DNS resolution error."""
        exc_str = str(exc).lower()
        return any(phrase in exc_str for phrase in [
            "failed to resolve",
            "name resolution",
            "nodename nor servname",
            "name or service not known",
            "[errno 8]",
        ])

    health_resp: requests.Response | None = None
    health_ok = False

    # Resolve hostname to IP using explicit resolvers to avoid system DNS issues
    health_url = f"{base}/health"

    # Retry health check with exponential backoff for DNS errors
    # Re-resolve DNS on each retry attempt to handle DNS propagation delays
    for attempt in range(max_retries):
        # Re-resolve DNS on each attempt (DNS might not be ready yet)
        ip_health_url, original_hostname = _resolve_url_to_ip(health_url)
        use_ip_directly = ip_health_url != health_url  # True if we resolved to IP
        
        # Ensure Host header is set if we resolved to IP
        if use_ip_directly and original_hostname and original_hostname not in ("localhost", "127.0.0.1"):
            headers["Host"] = original_hostname
        
        try:
            # If using IP directly, disable SSL verification (cert is for hostname, not IP)
            if use_ip_directly:
                health_resp = requests.get(ip_health_url, headers=headers, timeout=timeout, verify=False)
            else:
                health_resp = http_get(ip_health_url, headers=headers, timeout=timeout)
            health_ok, note = _health_response_ok(health_resp)
            suffix = f" ({note})" if note else ""
            # On non-200, include brief JSON detail if present
            if not health_ok and health_resp is not None:
                try:
                    hjs = health_resp.json()
                    # pull a few helpful fields without dumping everything
                    expected = hjs.get("expected_api_key_prefix")
                    authorized = hjs.get("authorized")
                    detail = hjs.get("detail")
                    extras = []
                    if authorized is not None:
                        extras.append(f"authorized={authorized}")
                    if expected:
                        extras.append(f"expected_prefix={expected}")
                    if detail:
                        extras.append(f"detail={str(detail)[:80]}")
                    if extras:
                        suffix += " [" + ", ".join(extras) + "]"
                except Exception:
                    pass
            detail_parts.append(f"/health={health_resp.status_code}{suffix}")
            break  # Success, exit retry loop
        except requests.RequestException as exc:
            if _is_dns_error(exc) and attempt < max_retries - 1:
                # DNS error, retry with exponential backoff
                delay = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
                print(f"DNS resolution failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...", flush=True)
                time.sleep(delay)
                continue
            # Not a DNS error or final attempt, record and break
            detail_parts.append(f"/health_error={exc}")
            break

    task_resp: requests.Response | None = None
    task_ok = False

    # Resolve hostname to IP using explicit resolvers to avoid system DNS issues
    task_info_url = f"{base}/task_info"
    # Host header already set from health check above

    # Retry task_info check with exponential backoff for DNS errors
    # Re-resolve DNS on each retry attempt to handle DNS propagation delays
    for attempt in range(max_retries):
        # Re-resolve DNS on each attempt (DNS might not be ready yet)
        ip_task_info_url, task_info_hostname = _resolve_url_to_ip(task_info_url)
        use_ip_directly_task = ip_task_info_url != task_info_url  # True if we resolved to IP
        
        # Ensure Host header is set if we resolved to IP
        if use_ip_directly_task and task_info_hostname and task_info_hostname not in ("localhost", "127.0.0.1"):
            headers["Host"] = task_info_hostname
        
        try:
            # If using IP directly, disable SSL verification (cert is for hostname, not IP)
            if use_ip_directly_task:
                task_resp = requests.get(ip_task_info_url, headers=headers, timeout=timeout, verify=False)
            else:
                task_resp = http_get(ip_task_info_url, headers=headers, timeout=timeout)
            task_ok = bool(task_resp.status_code == 200)
            if not task_ok and task_resp is not None:
                try:
                    tjs = task_resp.json()
                    msg = tjs.get("detail") or tjs.get("status")
                    detail_parts.append(f"/task_info={task_resp.status_code} ({str(msg)[:80]})")
                except Exception:
                    detail_parts.append(f"/task_info={task_resp.status_code}")
            else:
                detail_parts.append(f"/task_info={task_resp.status_code}")
            break  # Success, exit retry loop
        except requests.RequestException as exc:
            if _is_dns_error(exc) and attempt < max_retries - 1:
                # DNS error, retry with exponential backoff
                # DNS will be re-resolved on next iteration
                delay = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
                print(f"DNS resolution failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...", flush=True)
                time.sleep(delay)
                continue
            # Not a DNS error or final attempt, record and break
            detail_parts.append(f"/task_info_error={exc}")
            break

    ok = bool(health_ok and task_ok)
    detail = ", ".join(detail_parts)
    return TaskAppHealth(
        ok=ok,
        health_status=None if health_resp is None else health_resp.status_code,
        task_info_status=None if task_resp is None else task_resp.status_code,
        detail=detail,
    )


@dataclass(slots=True)
class ModalSecret:
    name: str
    value: str


@dataclass(slots=True)
class ModalApp:
    app_id: str
    label: str
    url: str


def _run_modal(args: Iterable[str]) -> CLIResult:
    return run_cli(["modal", *args], timeout=30.0)


def list_modal_secrets(pattern: str | None = None) -> list[str]:
    result = _run_modal(["secret", "list"])
    if result.code != 0:
        raise click.ClickException(f"modal secret list failed: {result.stderr or result.stdout}")
    names: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("NAME"):
            continue
        parts = line.split()
        name = parts[0]
        if pattern and pattern.lower() not in name.lower():
            continue
        names.append(name)
    return names


def get_modal_secret_value(name: str) -> str:
    result = _run_modal(["secret", "get", name])
    if result.code != 0:
        raise click.ClickException(
            f"modal secret get {name} failed: {result.stderr or result.stdout}"
        )
    value = result.stdout.strip()
    if not value:
        raise click.ClickException(f"Secret {name} is empty")
    return value


def list_modal_apps(pattern: str | None = None) -> list[ModalApp]:
    result = _run_modal(["app", "list"])
    if result.code != 0:
        raise click.ClickException(f"modal app list failed: {result.stderr or result.stdout}")
    apps: list[ModalApp] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("APP"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        app_id, label, url = parts[0], parts[1], parts[-1]
        if pattern and pattern.lower() not in (label.lower() + url.lower() + app_id.lower()):
            continue
        apps.append(ModalApp(app_id=app_id, label=label, url=url))
    return apps


def format_modal_apps(apps: list[ModalApp]) -> str:
    rows = [f"{idx}) {app.label} {app.url}" for idx, app in enumerate(apps, start=1)]
    return "\n".join(rows)


def format_modal_secrets(names: list[str]) -> str:
    return "\n".join(f"{idx}) {name}" for idx, name in enumerate(names, start=1))


__all__ = [
    "ModalApp",
    "ModalSecret",
    "check_task_app_health",
    "format_modal_apps",
    "format_modal_secrets",
    "get_modal_secret_value",
    "list_modal_apps",
    "list_modal_secrets",
]
