"""Cloudflare CLI/bootstrap helpers and tunnel deployment utilities."""

import asyncio
import logging
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple
from urllib.parse import urlparse

import click
import httpx
import requests
import uvicorn
from starlette.types import ASGIApp

from synth_ai.core.apps.common import get_asgi_app, load_module
from synth_ai.core.cfgs import CFDeployCfg
from synth_ai.core.paths import REPO_ROOT, configure_import_paths
from synth_ai.core.telemetry import log_error, log_event, log_info
from synth_ai.core.urls import BACKEND_URL_BASE


def __resolve_env_var(key: str) -> str:
    """Lazy import to avoid circular dependency."""
    from synth_ai.cli.lib.env import resolve_env_var
    return resolve_env_var(key)


def __write_env_var_to_dotenv(key: str, value: str, **kwargs) -> None:
    """Lazy import to avoid circular dependency."""
    from synth_ai.cli.lib.env import write_env_var_to_dotenv
    write_env_var_to_dotenv(key, value, **kwargs)

logger = logging.getLogger(__name__)

# Constants
CLOUDFLARED_BIN_NAME = "cloudflared"
CLOUDFLARED_RELEASES = "https://updatecloudflared.com/launcher"
CLOUDFLARE_DOCS_URL = "https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation"

# Regex for parsing quick tunnel URLs
# Match partial URLs too (in case they're split across lines)
_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.I)
_URL_PARTIAL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudf", re.I)  # Partial match for truncated lines (ends with trycloudf)
_URL_PARTIAL_RE2 = re.compile(r"https://[a-z0-9-]+\.tryclo", re.I)  # Partial match for truncated lines (ends with tryclo)

# Global state - store tunnel process handles for cleanup
_TUNNEL_PROCESSES: dict[int, subprocess.Popen] = {}


@dataclass(slots=True)
class ManagedTunnelRecord:
    """Managed tunnel metadata returned by backend."""

    id: str
    hostname: str
    org_id: str
    org_name: Optional[str]
    local_host: str
    local_port: int
    metadata: dict[str, Any]
    raw: dict[str, Any]

    @property
    def url(self) -> str:
        if self.hostname.startswith(("http://", "https://")):
            return self.hostname
        return f"https://{self.hostname}"

    @property
    def subdomain(self) -> str:
        return self.hostname.split(".", 1)[0]

    def credential(self, key: str) -> Optional[str]:
        return _extract_credential(self.raw, key)


# ---------------------------------------------------------------------------
# Managed tunnel discovery helpers
# ---------------------------------------------------------------------------


async def fetch_managed_tunnels(synth_api_key: str) -> list[ManagedTunnelRecord]:
    """
    Fetch managed tunnels tied to the provided Synth API key.

    Raises:
        RuntimeError: If backend returns an error or unexpected payload.
    """
    url = f"{BACKEND_URL_BASE}/api/v1/tunnels/"
    headers = {"Authorization": f"Bearer {synth_api_key}"}
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Failed to list managed tunnels (status {exc.response.status_code}): {exc.response.text}"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Failed to reach Synth backend at {url}: {exc}") from exc

    if not isinstance(payload, list):
        raise RuntimeError("Unexpected tunnel API response: expected a list of tunnels.")

    records: list[ManagedTunnelRecord] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        hostname = entry.get("hostname")
        org_id = entry.get("org_id")
        tunnel_id = entry.get("id")
        if not hostname or not org_id or not tunnel_id:
            continue
        metadata = entry.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        records.append(
            ManagedTunnelRecord(
                id=str(tunnel_id),
                hostname=str(hostname),
                org_id=str(org_id),
                org_name=entry.get("org_name"),
                local_host=str(entry.get("local_host") or "127.0.0.1"),
                local_port=int(entry.get("local_port") or 8000),
                metadata=metadata,
                raw=entry,
            )
        )
    return records


def _select_existing_tunnel(
    tunnels: list[ManagedTunnelRecord],
    desired_subdomain: Optional[str],
) -> Optional[ManagedTunnelRecord]:
    if not tunnels:
        return None

    if desired_subdomain:
        target = _normalize_subdomain(desired_subdomain)
        for tunnel in tunnels:
            if _normalize_subdomain(tunnel.subdomain) == target or _normalize_subdomain(
                tunnel.hostname
            ) == target:
                print(
                    f"ℹ️  Using managed tunnel {tunnel.url} "
                    f"(matched subdomain '{tunnel.subdomain}')"
                )
                return tunnel
        _print_tunnel_choices(tunnels, header="Available managed tunnels:")
        raise RuntimeError(
            f"No managed tunnel matched subdomain '{desired_subdomain}'. "
            "Re-run with a valid --tunnel-subdomain."
        )

    if len(tunnels) == 1:
        tunnel = tunnels[0]
        print(
            f"ℹ️  Reusing existing managed tunnel for "
            f"{tunnel.org_name or tunnel.org_id}: {tunnel.url}"
        )
        return tunnel

    _print_tunnel_choices(
        tunnels,
        header=(
            "Multiple managed tunnels found. Please re-run with "
            "--tunnel-subdomain <subdomain> to choose one."
        ),
    )
    raise RuntimeError("Multiple managed tunnels available; selection required.")


def _print_tunnel_choices(
    tunnels: Iterable[ManagedTunnelRecord],
    header: Optional[str] = None,
) -> None:
    if header:
        print(header)
    for idx, tunnel in enumerate(tunnels, 1):
        label = tunnel.org_name or tunnel.org_id
        print(f"  {idx}. {label}: {tunnel.url} (subdomain '{tunnel.subdomain}')")


def _normalize_subdomain(value: str) -> str:
    value = value.strip().lower()
    if value.startswith("https://"):
        value = value[len("https://") :]
    elif value.startswith("http://"):
        value = value[len("http://") :]
    return value.split(".", 1)[0]


def _extract_credential(payload: dict[str, Any], key: str) -> Optional[str]:
    """Extract secret from various nested metadata structures."""

    def _dig(obj: Any, path: tuple[str, ...]) -> Optional[Any]:
        current = obj
        for part in path:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    candidate_paths: tuple[tuple[str, ...], ...] = (
        (key,),
        ("metadata", key),
        ("metadata", "secrets", key),
        ("metadata", "credentials", key),
        ("metadata", "cloudflare", key),
        ("metadata", "cloudflare", "secrets", key),
    )

    for path in candidate_paths:
        value = _dig(payload, path)
        if isinstance(value, str) and value:
            return value
    return None


# ---------------------------------------------------------------------------
# Cloudflared binary management
# ---------------------------------------------------------------------------


def get_cloudflared_path(prefer_system: bool = True) -> Optional[Path]:
    """Locate the cloudflared binary (managed bin dir, PATH, or common dirs)."""
    bin_dir = Path.home() / ".synth-ai" / "bin"
    candidate = bin_dir / CLOUDFLARED_BIN_NAME
    if candidate.exists() and os.access(candidate, os.X_OK):
        return candidate

    if prefer_system:
        resolved = shutil.which(CLOUDFLARED_BIN_NAME)
        if resolved:
            return Path(resolved)

    common = [
        Path("/usr/local/bin/cloudflared"),
        Path("/opt/homebrew/bin/cloudflared"),
        Path.home() / "bin" / "cloudflared",
    ]
    for path in common:
        if path.exists() and os.access(path, os.X_OK):
            return path
    return None


def ensure_cloudflared_installed(force: bool = False) -> Path:
    """Ensure cloudflared is installed in synth-ai's managed bin directory."""
    existing = get_cloudflared_path(prefer_system=not force)
    if existing and not force:
        return existing

    target_dir = Path.home() / ".synth-ai" / "bin"
    target_dir.mkdir(parents=True, exist_ok=True)

    url = _resolve_cloudflared_download_url()
    tmp_file = _download_file(url)

    if tmp_file.suffixes[-2:] == [".tar", ".gz"]:
        _extract_tarball(tmp_file, target_dir)
    elif tmp_file.suffix == ".gz":
        _extract_gzip(tmp_file, target_dir / CLOUDFLARED_BIN_NAME)
    else:
        shutil.move(str(tmp_file), str(target_dir / CLOUDFLARED_BIN_NAME))

    bin_path = target_dir / CLOUDFLARED_BIN_NAME
    bin_path.chmod(0o755)
    log_event("info", "cloudflared installed", ctx={"path": str(bin_path)})
    return bin_path


def require_cloudflared() -> Path:
    """Return cloudflared binary or raise ClickException with guidance."""
    path = get_cloudflared_path()
    if path:
        return path

    extra = ""
    if platform.system() == "Darwin":
        extra = "Try `brew install cloudflare/cloudflare/cloudflared`."
    elif platform.system() == "Linux":
        extra = "See Cloudflare docs for Linux packages."
    log_error("cloudflared not found", ctx={"hint": extra})
    raise click.ClickException(
        f"Cloudflared CLI missing. Install via Homebrew or follow {CLOUDFLARE_DOCS_URL}."
    )


def run_cloudflared_cmd(args: list[str], *, env: Optional[dict[str, str]] = None) -> subprocess.Popen:
    """Spawn cloudflared subprocess (mirrors synth_ai.core.integrations.modal.run_modal_cmd)."""
    bin_path = require_cloudflared()
    cmd = [str(bin_path), *args]
    log_event("info", "starting cloudflared", ctx={"cmd": cmd})
    try:
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env or os.environ.copy(),
        )
    except FileNotFoundError as exc:
        raise click.ClickException(f"cloudflared binary missing: {exc}") from exc
    except Exception as exc:
        raise click.ClickException(f"Failed to start cloudflared: {exc}") from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_cloudflared_download_url() -> str:
    system = platform.system().lower()
    arch = platform.machine().lower()
    mapping = {"darwin": "macos", "linux": "linux", "windows": "windows"}
    platform_key = mapping.get(system)
    if not platform_key:
        raise RuntimeError(f"Unsupported platform: {system}")

    arch_key = "amd64"
    if arch in ("arm64", "aarch64"):
        arch_key = "arm64"

    resp = requests.get(f"{CLOUDFLARED_RELEASES}/v1/{platform_key}/{arch_key}/versions/stable", timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    url = data.get("url")
    if not url:
        raise RuntimeError("Cloudflared release metadata missing URL")
    return url


def _download_file(url: str) -> Path:
    resp = requests.get(url, timeout=60.0, stream=True)
    resp.raise_for_status()
    suffix = Path(url.split("?")[0]).suffix or ".tmp"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
    return Path(tmp_path)


def _extract_tarball(archive_path: Path, target_dir: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(target_dir)
    archive_path.unlink(missing_ok=True)


def _extract_gzip(gz_path: Path, target: Path) -> None:
    import gzip

    # gzip.open ensures the bytes are decompressed while copying to target
    with gzip.open(gz_path, "rb") as gz_fh, open(target, "wb") as out_fh:
        shutil.copyfileobj(gz_fh, out_fh)
    gz_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tunnel process management
# ---------------------------------------------------------------------------


def open_quick_tunnel(port: int, wait_s: float = 10.0) -> Tuple[str, subprocess.Popen]:
    """
    Open a quick (ephemeral) Cloudflare tunnel.

    Args:
        port: Local port to tunnel to
        wait_s: Maximum time to wait for URL in seconds

    Returns:
        Tuple of (public_url, process_handle)

    Raises:
        RuntimeError: If tunnel fails to start or URL cannot be parsed
    """
    bin_path = require_cloudflared()
    
    # Verify cloudflared can run before attempting tunnel
    try:
        test_proc = subprocess.run(
            [str(bin_path), "--version"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if test_proc.returncode != 0:
            raise RuntimeError(
                f"cloudflared binary exists but fails to run (exit code {test_proc.returncode}). "
                f"STDOUT: {test_proc.stdout[:500] if test_proc.stdout else 'none'}. "
                f"STDERR: {test_proc.stderr[:500] if test_proc.stderr else 'none'}. "
                f"Try reinstalling: cloudflared update or brew reinstall cloudflared"
            )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            "cloudflared binary hangs when running --version. "
            "This suggests the binary is corrupted or incompatible with your system. "
            "Try reinstalling: cloudflared update or brew reinstall cloudflared"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to verify cloudflared binary: {e}. "
            f"Binary path: {bin_path}. "
            f"Try reinstalling: cloudflared update or brew reinstall cloudflared"
        ) from e
    
    # Capture stderr separately for better error diagnostics
    # Use --config /dev/null to prevent loading any user config file
    # This fixes issues where ~/.cloudflared/config.yml has ingress rules
    # for named tunnels that interfere with quick tunnels (returning 404)
    proc = subprocess.Popen(
        [str(bin_path), "tunnel", "--config", "/dev/null", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # Capture stderr separately
        text=True,
        bufsize=1,
    )

    start = time.time()
    url: Optional[str] = None
    output_lines: list[str] = []
    stderr_lines: list[str] = []
    
    # Use select for non-blocking I/O to avoid hanging on readline()
    import select

    # Stream both stdout and stderr to detect the trycloudflare URL
    # Note: cloudflared prints the URL to stderr, not stdout!
    while time.time() - start < wait_s:
        elapsed = time.time() - start
        remaining_time = wait_s - elapsed
        
        if remaining_time <= 0:
            break
            
        if proc.poll() is not None:
            # Process exited early - try to read all available output
            try:
                stdout, stderr = proc.communicate(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
            
            # Combine stdout and stderr for error message
            all_output = ""
            if stdout:
                all_output += f"STDOUT:\n{stdout}\n"
            if stderr:
                all_output += f"STDERR:\n{stderr}\n"
            if output_lines:
                all_output += f"Captured stdout:\n{''.join(output_lines)}\n"
            if stderr_lines:
                all_output += f"Captured stderr:\n{''.join(stderr_lines)}\n"
            
            # Check for rate limiting (429 Too Many Requests)
            is_rate_limited = False
            if stderr and "429" in stderr and "Too Many Requests" in stderr or stderr and "rate limit" in stderr.lower():
                is_rate_limited = True
            
            # Add diagnostic info
            if is_rate_limited:
                error_msg = (
                    "❌ RATE LIMIT ERROR: Cloudflare is blocking quick tunnel creation due to rate limiting.\n"
                    f"\n"
                    f"Error Details:\n"
                    f"  • Exit code: {proc.returncode}\n"
                    f"  • Status: 429 Too Many Requests\n"
                    f"  • Command: {' '.join([str(bin_path), 'tunnel', '--url', f'http://127.0.0.1:{port}'])}\n"
                    f"\n"
                    f"Why this happens:\n"
                    f"  Cloudflare limits how many quick (ephemeral) tunnels can be created\n"
                    f"  in a short time period. You've hit this limit.\n"
                    f"\n"
                    f"Solutions (in order of preference):\n"
                    f"  1. ⏰ WAIT: Wait 5-10 minutes for the rate limit to reset\n"
                    f"  2. 🔑 USE MANAGED TUNNEL: Set SYNTH_API_KEY env var to use managed tunnels (no rate limits)\n"
                    f"  3. ♻️  REUSE EXISTING: Set INTERCEPTOR_TUNNEL_URL env var to reuse an existing tunnel\n"
                    f"\n"
                    f"Full error output:\n"
                    f"{all_output[:1000]}"
                )
            else:
                error_msg = (
                    f"cloudflared exited early with code {proc.returncode}.\n"
                    f"Command: {' '.join([str(bin_path), 'tunnel', '--url', f'http://127.0.0.1:{port}'])}\n"
                    f"Binary path: {bin_path}\n"
                )
                if all_output:
                    error_msg += f"Output:\n{all_output[:1000]}"
                else:
                    error_msg += "No output captured. This usually means:\n"
                    error_msg += "  1. cloudflared binary is corrupted or wrong architecture\n"
                    error_msg += "  2. cloudflared needs to be updated (try: cloudflared update)\n"
                    error_msg += "  3. System-level issue preventing cloudflared from running\n"
                    error_msg += "  4. Port conflict or network issue\n"
                    error_msg += f"\nTry running manually: {bin_path} tunnel --url http://127.0.0.1:{port}"
            
            raise RuntimeError(error_msg)

        # Read from both stdout and stderr (cloudflared prints URL to stderr!)
        fds_to_check = []
        from contextlib import suppress

        if proc.stdout:
            with suppress(ValueError, OSError):
                fds_to_check.append(("stdout", proc.stdout.fileno(), proc.stdout))
        if proc.stderr:
            with suppress(ValueError, OSError):
                fds_to_check.append(("stderr", proc.stderr.fileno(), proc.stderr))
        
        if not fds_to_check:
            if time.time() - start >= wait_s:
                break
            time.sleep(0.05)
            continue

        # Use select to check if data is available (non-blocking)
        try:
            fds = [fd for _, fd, _ in fds_to_check]
            ready, _, _ = select.select(fds, [], [], min(0.1, remaining_time))
            
            if ready:
                # Check which file descriptors are ready
                for name, fd, stream in fds_to_check:
                    if fd in ready:
                        # Data is available, read a line
                        line = stream.readline()
                        if line:
                            # Capture output for diagnostics
                            if name == "stdout":
                                output_lines.append(line)
                            else:
                                stderr_lines.append(line)
                            
                            # Check current line for URL
                            match = _URL_RE.search(line)
                            if match:
                                url = match.group(0)
                                break
                            
                            # Check for partial URL (truncated line) - wait for more data
                            partial_match = _URL_PARTIAL_RE.search(line)
                            if partial_match:
                                # Found partial URL, wait a bit longer for the rest
                                # Read more lines to get the complete URL
                                for _ in range(5):  # Try reading up to 5 more lines
                                    if time.time() - start >= wait_s:
                                        break
                                    time.sleep(0.1)
                                    if proc.poll() is not None:
                                        break
                                    # Try to read more
                                    if stream in [s for _, _, s in fds_to_check]:
                                        try:
                                            more_line = stream.readline()
                                            if more_line:
                                                if name == "stdout":
                                                    output_lines.append(more_line)
                                                else:
                                                    stderr_lines.append(more_line)
                                                line += more_line
                                        except (OSError, ValueError):
                                            pass
                                
                                # Now check accumulated output for full URL
                                all_accumulated = ''.join(output_lines + stderr_lines)
                                match = _URL_RE.search(all_accumulated)
                                if match:
                                    url = match.group(0)
                                    break
                            
                            # Also check accumulated output (URL might be split across lines)
                            all_accumulated = ''.join(output_lines + stderr_lines)
                            match = _URL_RE.search(all_accumulated)
                            if match:
                                url = match.group(0)
                                break
                
                if url:
                    break
            else:
                # No data available, check timeout and continue
                if time.time() - start >= wait_s:
                    break
                time.sleep(0.05)
                continue
        except (ValueError, OSError) as e:
            # File descriptor not available or select failed - fall back to reading both streams
            # This can happen on Windows or if the file is closed
            _ = e  # Suppress unused variable warning
            if proc.stdout:
                line = proc.stdout.readline()
                if line:
                    output_lines.append(line)
                    match = _URL_RE.search(line)
                    if match:
                        url = match.group(0)
                        break
                    # Check for partial URL
                    partial_match = _URL_PARTIAL_RE.search(line)
                    if partial_match:
                        # Wait a bit and read more
                        time.sleep(0.2)
                        more_line = proc.stdout.readline()
                        if more_line:
                            output_lines.append(more_line)
                            line += more_line
            if proc.stderr:
                line = proc.stderr.readline()
                if line:
                    stderr_lines.append(line)
                    match = _URL_RE.search(line)
                    if match:
                        url = match.group(0)
                        break
                    # Check for partial URL
                    partial_match = _URL_PARTIAL_RE.search(line)
                    if partial_match:
                        # Wait a bit and read more
                        time.sleep(0.2)
                        more_line = proc.stderr.readline()
                        if more_line:
                            stderr_lines.append(more_line)
                            line += more_line
            
            # Check accumulated output
            all_accumulated = ''.join(output_lines + stderr_lines)
            match = _URL_RE.search(all_accumulated)
            if match:
                url = match.group(0)
                break
            
            if time.time() - start >= wait_s:
                break
            time.sleep(0.05)
            continue

    if not url:
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
        
        all_output = ""
        if stdout:
            all_output += f"STDOUT:\n{stdout}\n"
        if stderr:
            all_output += f"STDERR:\n{stderr}\n"
        if output_lines:
            all_output += f"Captured stdout:\n{''.join(output_lines)}\n"
        if stderr_lines:
            all_output += f"Captured stderr:\n{''.join(stderr_lines)}\n"
        
        # Try to extract URL from accumulated output even if timeout occurred
        all_accumulated = ''.join(output_lines + stderr_lines)
        if stdout:
            all_accumulated += stdout
        if stderr:
            all_accumulated += stderr
        
        # Check for partial URL and try to reconstruct
        if not url:
            # Try first partial pattern (ends with trycloudf)
            partial_match = _URL_PARTIAL_RE.search(all_accumulated)
            if partial_match:
                # Found partial URL - try to complete it
                partial_url = partial_match.group(0)
                # Partial match ends with "trycloudf", so we need "lare.com"
                test_url = partial_url + "lare.com"
                if _URL_RE.match(test_url):
                    url = test_url
                    logger.info(f"Reconstructed URL from partial match (trycloudf): {url}")
            
            # Try second partial pattern (ends with tryclo)
            if not url:
                partial_match2 = _URL_PARTIAL_RE2.search(all_accumulated)
                if partial_match2:
                    partial_url = partial_match2.group(0)
                    # Partial match ends with "tryclo", so we need "udflare.com"
                    test_url = partial_url + "udflare.com"
                    if _URL_RE.match(test_url):
                        url = test_url
                        logger.info(f"Reconstructed URL from partial match (tryclo): {url}")
        
        if url:
            return url, proc
        
        error_msg = (
            f"Failed to parse trycloudflare URL from cloudflared output after {wait_s}s.\n"
            f"Command: {' '.join([str(bin_path), 'tunnel', '--url', f'http://127.0.0.1:{port}'])}\n"
        )
        if all_output:
            error_msg += f"Output:\n{all_output[:1000]}"
        else:
            error_msg += "No output captured."
        
        raise RuntimeError(error_msg)

    return url, proc


async def resolve_hostname_with_explicit_resolvers(hostname: str) -> str:
    """
    Resolve hostname using explicit resolvers (1.1.1.1, 8.8.8.8) first,
    then fall back to system resolver.
    
    This fixes resolver path issues where system DNS is slow or blocking.
    
    Args:
        hostname: Hostname to resolve
    
    Returns:
        Resolved IP address
    
    Raises:
        socket.gaierror: If resolution fails with all resolvers
    """
    timeout = float(os.getenv("SYNTH_TUNNEL_DNS_TIMEOUT_PER_ATTEMPT_SECS", "5"))
    loop = asyncio.get_event_loop()
    
    # Try Cloudflare / Google first via `dig`, then fall back to system resolver
    for resolver_ip in ("1.1.1.1", "8.8.8.8"):
        try:
            result = await loop.run_in_executor(
                None,
                lambda ip=resolver_ip: subprocess.run(
                    ["dig", f"@{ip}", "+short", hostname],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                ),
            )
            if result.returncode == 0 and result.stdout.strip():
                first = result.stdout.strip().splitlines()[0].strip()
                if first:
                    logger.debug(f"Resolved via {resolver_ip}: {hostname} -> {first}")
                    return first
        except FileNotFoundError:
            logger.debug(f"dig not found, skipping {resolver_ip}")
            continue
        except Exception as e:
            logger.debug(f"Resolver {resolver_ip} failed: {e}")
            continue
    
    # Fallback: system resolver
    logger.debug(f"Falling back to system resolver for {hostname}")
    return await loop.run_in_executor(
        None,
        socket.gethostbyname,
        hostname,
    )


async def verify_tunnel_dns_resolution(
    tunnel_url: str,
    name: str = "tunnel",
    timeout_seconds: float = 60.0,
    api_key: Optional[str] = None,
) -> None:
    """
    Verify that a tunnel URL's hostname can be resolved via DNS (using public
    resolvers first) and that HTTP connectivity works by connecting directly
    to the resolved IP with the original Host header.
    
    This avoids depending on the system resolver for HTTP checks, which was
    causing [Errno 8] errors even after DNS resolved via explicit resolvers.
    
    Args:
        tunnel_url: The tunnel URL to verify (e.g., https://xxx.trycloudflare.com/v1)
        name: Human-readable name for logging
        timeout_seconds: Maximum time to wait for DNS resolution
        api_key: Optional API key for health check authentication (defaults to ENVIRONMENT_API_KEY env var)
    
    Raises:
        RuntimeError: If DNS resolution or HTTP connectivity fails after timeout
    """
    parsed = urlparse(tunnel_url)
    hostname = parsed.hostname
    if not hostname:
        logger.warning(f"No hostname in {name} tunnel URL: {tunnel_url}")
        return
    
    # Skip DNS check for localhost
    if hostname in ("localhost", "127.0.0.1"):
        logger.debug(f"Skipping DNS check for localhost {name}")
        return
    
    max_delay = 3.0
    delay = 0.5
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_seconds
    attempt = 0
    
    logger.info(f"Verifying DNS resolution for {name}: {hostname} (timeout {timeout_seconds:.0f}s)...")
    
    last_exc: Optional[Exception] = None
    
    while True:
        attempt += 1
        try:
            # 1. Resolve via explicit resolvers (1.1.1.1 / 8.8.8.8) → IP
            resolved_ip = await resolve_hostname_with_explicit_resolvers(hostname)
            logger.info(f"DNS resolution successful (attempt {attempt}): {hostname} -> {resolved_ip}")
            
            # 2. HTTP connectivity: hit the tunnel via the resolved IP, but keep Host header.
            #    This avoids depending on the system resolver, which is what gave you EAI_NONAME.
            try:
                scheme = parsed.scheme or "https"
                test_url = f"{scheme}://{resolved_ip}/health"
                headers = {"Host": hostname}
                
                # Include API key if provided (or from env var)
                if api_key is None:
                    # Try to load .env file if available
                    try:
                        from dotenv import load_dotenv
                        load_dotenv(override=False)
                    except ImportError:
                        pass
                    api_key = os.getenv("ENVIRONMENT_API_KEY")
                if api_key:
                    headers["X-API-Key"] = api_key
                
                # For Quick Tunnels, TLS cert is for *.trycloudflare.com, not the bare IP,
                # so we disable verification here; this is just a readiness probe.
                async with httpx.AsyncClient(timeout=5.0, verify=False) as client:
                    resp = await client.get(test_url, headers=headers)
                    # Accept 200 (OK), 400/401 (auth required - server is reachable), 404/405 (not found/method not allowed)
                    # All of these indicate the tunnel is working and the server is responding
                    if resp.status_code in (200, 400, 401, 404, 405):
                        logger.info(f"HTTP connectivity verified via IP: {test_url} -> {resp.status_code}")
                        return
                    else:
                        # 530 errors are common when tunnel is still establishing - be lenient
                        if resp.status_code == 530:
                            logger.debug("HTTP 530 (tunnel establishing) - will retry")
                            last_exc = RuntimeError("tunnel not ready yet (HTTP 530)")
                        else:
                            logger.warning(f"HTTP check returned unexpected status: {resp.status_code}")
                            last_exc = RuntimeError(f"unexpected HTTP status {resp.status_code}")
            except Exception as http_exc:
                logger.warning(f"HTTP connectivity check failed (attempt {attempt}): {http_exc}")
                last_exc = http_exc
            
            # DNS resolved, but HTTP check failed - wait and retry until deadline
            now = loop.time()
            if now >= deadline:
                break
            delay = min(delay * 2 if attempt > 1 else delay, max_delay)
            sleep_for = min(delay, max(0.0, deadline - now))
            logger.debug(f"Waiting {sleep_for:.1f}s before retry...")
            await asyncio.sleep(sleep_for)
            
        except socket.gaierror as e:
            logger.warning(f"DNS resolution failed (attempt {attempt}): {e}")
            last_exc = e
            now = loop.time()
            if now >= deadline:
                raise RuntimeError(
                    f"DNS resolution failed for {name} tunnel hostname {hostname} "
                    f"after {timeout_seconds:.0f}s. Tunnel URL: {tunnel_url}. Error: {e}"
                ) from e
            delay = min(delay * 2 if attempt > 1 else delay, max_delay)
            sleep_for = min(delay, max(0.0, deadline - now))
            logger.debug(f"Waiting {sleep_for:.1f}s before retry...")
            await asyncio.sleep(sleep_for)
        except Exception as e:
            logger.error(f"Unexpected error during DNS verification (attempt {attempt}): {e}")
            last_exc = e
            now = loop.time()
            if now >= deadline:
                raise RuntimeError(
                    f"DNS verification failed for {hostname} after {timeout_seconds:.0f}s: {e}"
                ) from e
            delay = min(delay * 2 if attempt > 1 else delay, max_delay)
            sleep_for = min(delay, max(0.0, deadline - now))
            await asyncio.sleep(sleep_for)
    
    # If we get here, we ran out of time with HTTP still failing
    raise RuntimeError(
        f"DNS succeeded but HTTP connectivity could not be confirmed for {hostname} "
        f"within {timeout_seconds:.0f}s. Last error: {last_exc}"
    )


async def open_quick_tunnel_with_dns_verification(
    port: int,
    *,
    wait_s: float = 10.0,
    max_retries: Optional[int] = None,
    dns_timeout_s: Optional[float] = None,
    api_key: Optional[str] = None,
) -> Tuple[str, subprocess.Popen]:
    """
    Open a quick Cloudflare tunnel with DNS verification and retry logic.
    
    This wraps open_quick_tunnel with DNS verification to ensure the tunnel
    is actually reachable before returning.
    
    Args:
        port: Local port to tunnel to
        wait_s: Maximum time to wait for URL in seconds
        max_retries: Maximum number of tunnel creation retries (default: from SYNTH_TUNNEL_MAX_RETRIES env var, or 2)
        dns_timeout_s: Maximum time to wait for DNS resolution (default: from SYNTH_TUNNEL_DNS_TIMEOUT_SECS env var, or 60)
        api_key: Optional API key for health check authentication (defaults to ENVIRONMENT_API_KEY env var)
    
    Returns:
        Tuple of (public_url, process_handle)
    
    Raises:
        RuntimeError: If tunnel creation or DNS verification fails after retries
    """
    max_retries = max_retries or int(os.getenv("SYNTH_TUNNEL_MAX_RETRIES", "2"))
    dns_timeout_s = dns_timeout_s or float(os.getenv("SYNTH_TUNNEL_DNS_TIMEOUT_SECS", "60"))
    
    # Get API key from parameter or env var
    if api_key is None:
        # Try to load .env file if available
        try:
            from dotenv import load_dotenv
            load_dotenv(override=False)
        except ImportError:
            pass
        api_key = os.getenv("ENVIRONMENT_API_KEY")
    
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        proc: Optional[subprocess.Popen] = None
        try:
            logger.info(f"Tunnel attempt {attempt}/{max_retries}")
            url, proc = open_quick_tunnel(port, wait_s=wait_s)
            logger.info(f"Tunnel URL obtained: {url}")
            
            # Give tunnel a moment to establish before verification
            # Cloudflare tunnels can take a few seconds to become fully ready
            logger.debug("Waiting 3s for tunnel to establish before verification...")
            await asyncio.sleep(3.0)
            
            # Verify DNS (this is where failures usually happen)
            await verify_tunnel_dns_resolution(url, timeout_seconds=dns_timeout_s, name=f"tunnel attempt {attempt}", api_key=api_key)
            
            logger.info("Tunnel verified and ready!")
            return url, proc
        except Exception as e:
            last_err = e
            # Check if this is a rate limit error and make it clearer
            error_str = str(e)
            is_rate_limit = "429" in error_str and "Too Many Requests" in error_str
            if is_rate_limit:
                logger.error(
                    f"❌ RATE LIMIT: Tunnel attempt {attempt}/{max_retries} failed due to Cloudflare rate limiting. "
                    f"This means too many quick tunnels were created recently. "
                    f"Wait 5-10 minutes or use managed tunnels (set SYNTH_API_KEY)."
                )
            else:
                logger.warning(f"Tunnel attempt {attempt} failed: {e}")
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
            if attempt < max_retries:
                logger.info("Retrying after 10s backoff...")
                await asyncio.sleep(10.0)
            else:
                break
    
    assert last_err is not None
    raise last_err


async def check_rate_limit_status(test_port: int = 19999) -> dict[str, Any]:
    """
    Check if Cloudflare is currently rate-limiting quick tunnel creation.
    
    This attempts to create a quick tunnel and checks for rate limit errors.
    
    Args:
        test_port: Port to use for test tunnel (should be available)
    
    Returns:
        dict with keys:
            - is_rate_limited: bool
            - exit_code: int | None
            - error_message: str | None
            - output: str
    """
    import http.server
    import socketserver
    import threading
    
    bin_path = require_cloudflared()
    
    # Start a dummy HTTP server
    server = None
    server_thread = None
    
    try:
        handler = http.server.SimpleHTTPRequestHandler
        server = socketserver.TCPServer(("127.0.0.1", test_port), handler)
        server.allow_reuse_address = True
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        await asyncio.sleep(0.5)
        
        # Try to create a tunnel (use --config /dev/null to ignore user config)
        proc = subprocess.Popen(
            [str(bin_path), "tunnel", "--config", "/dev/null", "--url", f"http://127.0.0.1:{test_port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        # Wait a few seconds
        start = time.time()
        output_lines = []
        stderr_lines = []
        
        while time.time() - start < 3.0:
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                if stdout:
                    output_lines.extend(stdout.splitlines())
                if stderr:
                    stderr_lines.extend(stderr.splitlines())
                break
            await asyncio.sleep(0.1)
        
        # Clean up
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        
        all_output = "\n".join(stderr_lines + output_lines)
        
        # Check for rate limit
        is_rate_limited = False
        if proc.returncode == 1 and (
            ("429" in all_output and "Too Many Requests" in all_output) or "rate limit" in all_output.lower()
        ):
            is_rate_limited = True
        
        return {
            "is_rate_limited": is_rate_limited,
            "exit_code": proc.returncode,
            "error_message": all_output if is_rate_limited else None,
            "output": all_output,
        }
        
    finally:
        if server:
            server.shutdown()
            server.server_close()
        if server_thread:
            server_thread.join(timeout=2.0)


def open_managed_tunnel(tunnel_token: str) -> subprocess.Popen:
    """
    Open a managed (named) Cloudflare tunnel using a token.

    Args:
        tunnel_token: Cloudflare tunnel token from backend API

    Returns:
        Process handle for the tunnel

    Raises:
        RuntimeError: If cloudflared is not installed
    """
    bin_path = require_cloudflared()
    # cloudflared v2023.4+ accepts --token for named tunnels
    return subprocess.Popen(
        [str(bin_path), "tunnel", "run", "--token", tunnel_token],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def stop_tunnel(proc: Optional[subprocess.Popen]) -> None:
    """
    Gracefully stop a tunnel process.

    Args:
        proc: Process handle to terminate, or None
    """
    if proc is None:
        return

    if proc.poll() is None:
        # Process is still running
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            # Force kill if graceful termination fails
            proc.kill()
            proc.wait()


def store_tunnel_credentials(
    tunnel_url: str,
    access_client_id: Optional[str] = None,
    access_client_secret: Optional[str] = None,
    env_file: Optional[Path] = None,
) -> None:
    """
    Store tunnel credentials in .env file for optimizer to use.

    Writes:
    - TASK_APP_URL=<tunnel_url>
    - CF_ACCESS_CLIENT_ID=<client_id> (if Access enabled)
    - CF_ACCESS_CLIENT_SECRET=<client_secret> (if Access enabled)

    Args:
        tunnel_url: Public tunnel URL (e.g., "https://cust-abc123.usesynth.ai")
        access_client_id: Cloudflare Access client ID (optional)
        access_client_secret: Cloudflare Access client secret (optional)
        env_file: Path to .env file (defaults to .env in current directory)
    """
    __write_env_var_to_dotenv(
        "TASK_APP_URL",
        tunnel_url,
        output_file_path=env_file,
        print_msg=True,
        mask_msg=False,
    )

    if access_client_id:
        __write_env_var_to_dotenv(
            "CF_ACCESS_CLIENT_ID",
            access_client_id,
            output_file_path=env_file,
            print_msg=True,
            mask_msg=True,
        )

    if access_client_secret:
        __write_env_var_to_dotenv(
            "CF_ACCESS_CLIENT_SECRET",
            access_client_secret,
            output_file_path=env_file,
            print_msg=True,
            mask_msg=True,
        )


# ---------------------------------------------------------------------------
# Tunnel deployment helpers
# ---------------------------------------------------------------------------


async def rotate_tunnel(
    synth_api_key: str,
    port: int,
    reason: Optional[str] = None,
    backend_url: Optional[str] = None,
) -> dict[str, Any]:
    """
    Rotate (delete + recreate) the org's managed tunnel via Synth backend API.

    This is useful when a tunnel becomes stale or inaccessible. It will:
    1. Delete any existing active tunnels for the org
    2. Create a fresh tunnel with a new auto-generated subdomain
    3. Wait for DNS propagation (up to 90s) before returning

    Args:
        synth_api_key: Synth API key for authentication
        port: Local port the new tunnel will forward to
        reason: Optional reason for rotation (for logging)
        backend_url: Optional backend URL (defaults to get_backend_url())

    Returns:
        Dict containing:
        - tunnel_token: Token for cloudflared
        - hostname: Public hostname (e.g., "task-8114-12345.usesynth.ai")
        - access_client_id: Cloudflare Access client ID (if Access enabled)
        - access_client_secret: Cloudflare Access client secret (if Access enabled)
        - dns_verified: True if backend verified DNS propagation (SDK can skip DNS verification)
        - metadata: Dict with dns_verified and dns_verified_at timestamp

    Raises:
        RuntimeError: If API request fails
    """
    from synth_ai.core.env import get_backend_url

    base_url = backend_url or get_backend_url()
    url = f"{base_url}/api/v1/tunnels/rotate"

    def mask_key(key: str) -> str:
        if len(key) > 14:
            return f"{key[:10]}...{key[-4:]}"
        return f"{key[:6]}..."

    try:
        # Backend now waits up to 90s for DNS propagation, so we need a longer timeout
        async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
            response = await client.post(
                url,
                headers={
                    "X-API-Key": synth_api_key,
                    "Authorization": f"Bearer {synth_api_key}",
                },
                json={
                    "local_port": port,
                    "local_host": "127.0.0.1",
                    "reason": reason,
                },
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        error_detail = exc.response.text
        try:
            import json
            error_json = json.loads(error_detail)
            error_detail = str(error_json.get("detail", error_detail))
        except Exception:
            pass

        raise RuntimeError(
            f"Backend API returned {exc.response.status_code} when rotating tunnel:\n"
            f"  Error: {error_detail}\n"
            f"  URL: {url}\n"
            f"  API Key: {mask_key(synth_api_key)}"
        ) from exc
    except httpx.ReadTimeout as exc:
        raise RuntimeError(
            f"Request timed out when rotating tunnel (backend waits for DNS propagation):\n"
            f"  URL: {url}\n"
            f"  Timeout: 180s\n"
            f"  This is usually temporary - try again in a moment"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Failed to connect to backend when rotating tunnel:\n"
            f"  URL: {url}\n"
            f"  Error: {exc}"
        ) from exc


async def create_tunnel(
    synth_api_key: str,
    port: int,
    subdomain: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create a managed Cloudflare tunnel via Synth backend API.

    The backend waits for DNS propagation (up to 90s) before returning.

    Args:
        synth_api_key: Synth API key for authentication
        port: Local port the tunnel will forward to
        subdomain: Optional custom subdomain (e.g., "my-company")

    Returns:
        Dict containing:
        - tunnel_token: Token for cloudflared
        - hostname: Public hostname (e.g., "cust-abc123.usesynth.ai")
        - access_client_id: Cloudflare Access client ID (if Access enabled)
        - access_client_secret: Cloudflare Access client secret (if Access enabled)
        - dns_verified: True if backend verified DNS propagation (SDK can skip DNS verification)
        - metadata: Dict with dns_verified and dns_verified_at timestamp

    Raises:
        RuntimeError: If API request fails
    """
    url = f"{BACKEND_URL_BASE}/api/v1/tunnels/"

    # Mask API key for error messages
    def mask_key(key: str) -> str:
        if len(key) > 14:
            return f"{key[:10]}...{key[-4:]}"
        return f"{key[:6]}..."

    try:
        # Use X-API-Key header (backend expects this format)
        # Also support Authorization header as fallback
        # Backend now waits up to 90s for DNS propagation, so we need a longer timeout
        async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
            response = await client.post(
                url,
                headers={
                    "X-API-Key": synth_api_key,
                    "Authorization": f"Bearer {synth_api_key}",  # Fallback
                },
                json={
                    "subdomain": subdomain or f"tunnel-{port}",
                    "local_port": port,
                    "local_host": "127.0.0.1",
                },
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        error_detail = exc.response.text
        try:
            import json
            error_json = json.loads(error_detail)
            error_detail = str(error_json.get("detail", error_detail))
        except Exception:
            pass
        
        # Provide helpful error message
        if exc.response.status_code == 401:
            raise RuntimeError(
                f"Authentication failed when creating tunnel:\n"
                f"  Status: {exc.response.status_code}\n"
                f"  Error: {error_detail}\n"
                f"  API Key used: {mask_key(synth_api_key)}\n"
                f"  URL: {url}\n"
                f"  This usually means:\n"
                f"    - The API key is invalid or expired\n"
                f"    - The backend is experiencing high load (PostgREST timeout)\n"
                f"    - Network connectivity issues\n"
                f"  Try:\n"
                f"    - Verify SYNTH_API_KEY is set correctly\n"
                f"    - Wait a moment and retry (backend may be under load)\n"
                f"    - Use tunnel_mode='quick' as a workaround"
            ) from exc
        else:
            raise RuntimeError(
                f"Backend API returned {exc.response.status_code} when creating tunnel:\n"
                f"  Error: {error_detail}\n"
                f"  URL: {url}\n"
                f"  API Key: {mask_key(synth_api_key)}"
            ) from exc
    except httpx.ReadTimeout as exc:
        raise RuntimeError(
            f"Request timed out when creating tunnel (backend waits for DNS propagation):\n"
            f"  URL: {url}\n"
            f"  API Key: {mask_key(synth_api_key)}\n"
            f"  Timeout: 180s\n"
            f"  This is usually temporary - try again in a moment"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Failed to connect to backend when creating tunnel:\n"
            f"  URL: {url}\n"
            f"  API Key: {mask_key(synth_api_key)}\n"
            f"  Error: {exc}\n"
            f"  Check network connectivity and backend availability"
        ) from exc


async def wait_for_health_check(
    host: str,
    port: int,
    api_key: str,
    timeout: float = 30.0,
) -> None:
    """
    Wait for task app health endpoint to be ready.

    Args:
        host: Host to check
        port: Port to check
        api_key: API key for authentication
        timeout: Maximum time to wait in seconds

    Raises:
        RuntimeError: If health check fails or times out
    """
    health_url = f"http://{host}:{port}/health"
    headers = {"X-API-Key": api_key}
    start = time.time()

    while time.time() - start < timeout:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(health_url, headers=headers)
                # Accept both 200 (success) and 400 (auth error means server is up)
                if response.status_code in (200, 400):
                    return
        except (httpx.RequestError, httpx.TimeoutException):
            pass

        await asyncio.sleep(0.5)

    raise RuntimeError(
        f"Health check failed: {health_url} not ready after {timeout}s. "
        "Make sure your task app has a /health endpoint."
    )


def _start_uvicorn_background(
    app: ASGIApp,
    host: str,
    port: int,
    daemon: bool = True,
) -> None:
    """
    Start uvicorn server in a background thread.

    Args:
        app: ASGI application
        host: Host to bind to
        port: Port to bind to
        daemon: If True, thread dies when main process exits. If False, thread keeps running.
    """
    import threading

    def serve():
        try:
            uvicorn.run(
                app,
                host=host,
                port=port,
                reload=False,
                log_level="info",
            )
        except Exception as exc:
            # Log error but don't raise (background thread)
            print(f"Uvicorn error: {exc}", flush=True)

    thread = threading.Thread(
        target=serve,
        name=f"synth-uvicorn-tunnel-{port}",
        daemon=daemon,
    )
    thread.start()


async def deploy_app_tunnel(
    cfg: CFDeployCfg,
    env_file: Optional[Path] = None,
    keep_alive: bool = False,
    wait: bool = False,
    health_check_timeout: float = 30.0,
) -> str:
    """
    Deploy task app via Cloudflare Tunnel.

    This function provides a clean abstraction that handles:
    1. Starting the local task app (uvicorn) in background
    2. Optionally waiting for health check (only if wait=True)
    3. Opening tunnel (quick or managed)
    4. Writing tunnel URL and Access credentials to .env
    5. Optionally keeping processes alive (blocking vs non-blocking mode)

    By default (wait=False), this function is non-blocking and returns immediately
    after starting the tunnel. This is designed for AI agent use to prevent indefinite stalls.
    Processes run in the background and will continue until explicitly stopped.

    When `wait=True` or `keep_alive=True`, this function blocks and keeps the tunnel running
    until interrupted (Ctrl+C). Use this for interactive use or when you need to wait
    for the deployment to complete.

    Args:
        cfg: Tunnel deployment configuration
        env_file: Optional path to .env file (defaults to .env in current directory)
        keep_alive: (Deprecated) If True, block and keep tunnel alive until interrupted.
                   Use `wait` instead.
        wait: If True, wait for health check and block until interrupted.
             If False (default), return immediately after deployment (background mode).
        health_check_timeout: Maximum time to wait for health check (only used if wait=True)

    Returns:
        Public tunnel URL

    Raises:
        RuntimeError: If deployment fails at any step

    Example:
        # Non-blocking (background mode, returns immediately) - DEFAULT
        url = await deploy_app_tunnel(cfg, wait=False)

        # Blocking (waits for health check and keeps tunnel alive)
        url = await deploy_app_tunnel(cfg, wait=True)
    """
    ctx: dict[str, Any] = {
        "mode": cfg.mode,
        "host": cfg.host,
        "port": cfg.port,
        "task_app_path": str(cfg.task_app_path) if cfg.task_app_path else None,
        "wait": wait,
    }
    log_info("deploy_app_tunnel invoked", ctx=ctx)

    ensure_cloudflared_installed()

    selected_managed: Optional[ManagedTunnelRecord] = None
    synth_api_key: Optional[str] = None

    if cfg.mode == "managed":
        synth_api_key = __resolve_env_var("SYNTH_API_KEY")
        tunnels = await fetch_managed_tunnels(synth_api_key)
        if tunnels:
            selected_managed = _select_existing_tunnel(tunnels, cfg.subdomain)
            if selected_managed:
                cfg.host = selected_managed.local_host or cfg.host
                cfg.port = selected_managed.local_port or cfg.port
        else:
            print("ℹ️  No managed tunnels found; provisioning a new managed tunnel.")

    # Load environment variables from env_file before starting uvicorn
    # This ensures all env vars (HF cache paths, dataset names, etc.) are available to the task app
    if env_file and env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(str(env_file), override=True)
            # Also explicitly set critical env vars to ensure they're available
            # Read the file directly to set vars even if dotenv fails
            try:
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            # Remove quotes if present
                            value = value.strip().strip('"').strip("'")
                            os.environ[key.strip()] = value
            except Exception as file_exc:
                logger.debug(f"Could not read env_file directly: {file_exc}")
            logger.debug(f"Loaded environment from {env_file}")
        except ImportError:
            logger.warning("python-dotenv not available, skipping env_file load")
            # Fallback: read file directly
            try:
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            value = value.strip().strip('"').strip("'")
                            os.environ[key.strip()] = value
            except Exception as file_exc:
                logger.warning(f"Failed to read env_file directly: {file_exc}")
        except Exception as exc:
            logger.warning(f"Failed to load env_file {env_file}: {exc}")
    
    os.environ["ENVIRONMENT_API_KEY"] = cfg.env_api_key
    if cfg.trace:
        os.environ["TASKAPP_TRACING_ENABLED"] = "1"
    else:
        os.environ.pop("TASKAPP_TRACING_ENABLED", None)

    configure_import_paths(cfg.task_app_path, REPO_ROOT)
    module = load_module(cfg.task_app_path, f"_synth_tunnel_task_app_{cfg.task_app_path.stem}")
    app = get_asgi_app(module)

    # Always use non-daemon thread so it survives when main process exits
    _start_uvicorn_background(app, cfg.host, cfg.port, daemon=False)
    
    # Only wait for health check if wait mode is enabled (for AI agents, skip to avoid stalls)
    if wait or keep_alive:
        await wait_for_health_check(cfg.host, cfg.port, cfg.env_api_key, timeout=health_check_timeout)
    else:
        # In background mode, give it a short moment to start, but don't wait for full health check
        # This prevents indefinite stalls while still allowing the server to start
        import asyncio
        await asyncio.sleep(1.0)  # Brief delay to let server start

    tunnel_proc: Optional[subprocess.Popen] = None
    try:
        if cfg.mode == "quick":
            # Quick tunnel: ephemeral, no backend API call
            url, tunnel_proc = open_quick_tunnel(cfg.port)
            _TUNNEL_PROCESSES[cfg.port] = tunnel_proc
            store_tunnel_credentials(url, None, None, env_file)
            # Record tunnel for scan command
            try:
                from synth_ai.cli.lib.tunnel_records import record_tunnel
                record_tunnel(
                    url=url,
                    port=cfg.port,
                    mode="quick",
                    pid=tunnel_proc.pid if tunnel_proc else None,
                    hostname=url.replace("https://", "").split("/")[0] if url.startswith("https://") else None,
                    local_host=cfg.host,
                    task_app_path=str(cfg.task_app_path) if cfg.task_app_path else None,
                )
            except Exception:
                pass  # Fail silently - records are optional
        else:
            # Managed tunnel: either reuse or provision via backend API
            if selected_managed:
                tunnel_token = selected_managed.credential("tunnel_token")
                if not tunnel_token:
                    raise RuntimeError(
                        "Managed tunnel metadata missing tunnel_token. "
                        "Delete the tunnel or contact Synth support."
                    )
                hostname = selected_managed.hostname
                access_client_id = selected_managed.credential("access_client_id")
                access_client_secret = selected_managed.credential("access_client_secret")
            else:
                if not synth_api_key:
                    synth_api_key = __resolve_env_var("SYNTH_API_KEY")
                data = await create_tunnel(synth_api_key, cfg.port, cfg.subdomain)
                tunnel_token = data["tunnel_token"]
                hostname = data["hostname"]
                access_client_id = data.get("access_client_id")
                access_client_secret = data.get("access_client_secret")

            tunnel_proc = open_managed_tunnel(str(tunnel_token))
            _TUNNEL_PROCESSES[cfg.port] = tunnel_proc

            url = hostname if hostname.startswith("http") else f"https://{hostname}"
            store_tunnel_credentials(url, access_client_id, access_client_secret, env_file)
            # Record tunnel for scan command
            try:
                from synth_ai.cli.lib.tunnel_records import record_tunnel
                record_tunnel(
                    url=url,
                    port=cfg.port,
                    mode="managed",
                    pid=tunnel_proc.pid if tunnel_proc else None,
                    hostname=hostname,
                    local_host=cfg.host,
                    task_app_path=str(cfg.task_app_path) if cfg.task_app_path else None,
                )
            except Exception:
                pass  # Fail silently - records are optional

        # If wait or keep_alive is True, block and keep processes alive until interrupted
        if wait or keep_alive:
            _keep_tunnel_alive(cfg.port, url)
        else:
            # Background mode: print URL and return immediately
            # Processes will keep running in background
            print(f"✓ Tunnel ready: {url}")
            print(f"⏳ Tunnel running in background (PID: {tunnel_proc.pid if tunnel_proc else 'N/A'})")
            print("   Press Ctrl+C in this process to stop, or use: pkill -f cloudflared")

        return url

    except Exception as exc:
        # Clean up tunnel process on error
        if tunnel_proc:
            stop_tunnel(tunnel_proc)
            _TUNNEL_PROCESSES.pop(cfg.port, None)
        # Remove record if it was created
        try:
            from synth_ai.cli.lib.tunnel_records import remove_tunnel_record
            remove_tunnel_record(cfg.port)
        except Exception:
            pass
        raise RuntimeError(f"Failed to deploy tunnel: {exc}") from exc


def _keep_tunnel_alive(port: int, url: str) -> None:
    """
    Keep tunnel processes alive until interrupted.

    This function blocks and monitors the tunnel process, similar to how
    local deployments block. Users can interrupt with Ctrl+C to stop.

    Args:
        port: Port the tunnel is running on
        url: Public tunnel URL (for display)
    """

    def signal_handler(signum, frame):  # noqa: ARG001
        """Handle SIGINT/SIGTERM to cleanup gracefully."""
        if port in _TUNNEL_PROCESSES:
            stop_tunnel(_TUNNEL_PROCESSES[port])
            _TUNNEL_PROCESSES.pop(port, None)
        sys.exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"✓ Tunnel ready: {url}")
    print("⏳ Keeping tunnel running... (Press Ctrl+C to stop)")

    try:
        # Monitor tunnel process and keep alive
        while True:
            if port in _TUNNEL_PROCESSES:
                proc = _TUNNEL_PROCESSES[port]
                if isinstance(proc, subprocess.Popen) and proc.poll() is not None:
                    print(f"❌ Tunnel process exited with code {proc.returncode}")
                    break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup on exit
        if port in _TUNNEL_PROCESSES:
            stop_tunnel(_TUNNEL_PROCESSES[port])
            _TUNNEL_PROCESSES.pop(port, None)
        print("\n🛑 Tunnel stopped")
