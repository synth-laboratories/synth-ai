"""Custom exceptions for the tunnel system.

All exceptions are designed to be actionable - they include clear
error messages and suggested remediation steps.
"""

from __future__ import annotations

from typing import Optional


class TunnelError(Exception):
    """Base exception for all tunnel-related errors."""

    def __init__(
        self,
        message: str,
        *,
        hint: Optional[str] = None,
        diagnostics_id: Optional[str] = None,
    ):
        self.message = message
        self.hint = hint
        self.diagnostics_id = diagnostics_id
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = [self.message]
        if self.hint:
            parts.append(f"\nHint: {self.hint}")
        if self.diagnostics_id:
            parts.append(f"\nDiagnostics ID: {self.diagnostics_id}")
        return "".join(parts)


class TunnelConfigurationError(TunnelError):
    """Error in tunnel configuration."""

    def __init__(self, message: str, *, hint: Optional[str] = None):
        super().__init__(
            message,
            hint=hint or "Check your tunnel configuration and environment variables.",
        )


class TunnelAPIError(TunnelError):
    """Error communicating with the backend API."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        hint: Optional[str] = None,
    ):
        self.status_code = status_code
        self.response_body = response_body

        if status_code == 401:
            hint = hint or "Your API key may be invalid or expired. Check SYNTH_API_KEY."
        elif status_code == 429:
            hint = hint or "Rate limited. Wait a moment and try again."
        elif status_code == 502:
            hint = hint or "Cloudflare provisioning failed. This is usually temporary - try again."
        elif status_code == 503:
            hint = hint or "Tunnel service is not configured. Contact support."

        super().__init__(message, hint=hint)


class LeaseError(TunnelError):
    """Error with tunnel lease operations."""

    def __init__(
        self,
        message: str,
        *,
        lease_id: Optional[str] = None,
        hint: Optional[str] = None,
    ):
        self.lease_id = lease_id
        super().__init__(
            message,
            hint=hint,
            diagnostics_id=f"lease:{lease_id[:8]}" if lease_id else None,
        )


class LeaseExpiredError(LeaseError):
    """The tunnel lease has expired."""

    def __init__(self, lease_id: str):
        super().__init__(
            f"Tunnel lease has expired: {lease_id[:8]}",
            lease_id=lease_id,
            hint="Create a new lease or extend the TTL.",
        )


class LeaseNotFoundError(LeaseError):
    """The tunnel lease was not found."""

    def __init__(self, lease_id: str):
        super().__init__(
            f"Tunnel lease not found: {lease_id[:8]}",
            lease_id=lease_id,
            hint="The lease may have expired or been released.",
        )


class ConnectorError(TunnelError):
    """Error with the cloudflared connector."""

    def __init__(
        self,
        message: str,
        *,
        hint: Optional[str] = None,
        diagnostics_id: Optional[str] = None,
    ):
        super().__init__(message, hint=hint, diagnostics_id=diagnostics_id)


class ConnectorNotInstalledError(ConnectorError):
    """cloudflared is not installed."""

    def __init__(self):
        super().__init__(
            "cloudflared is not installed",
            hint=(
                "Install cloudflared with: synth_ai.tunnels.ensure_cloudflared_installed()\n"
                "Or download from: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
            ),
        )


class ConnectorConnectionError(ConnectorError):
    """cloudflared failed to connect to Cloudflare edge."""

    def __init__(self, message: str, *, timeout: Optional[float] = None):
        hint = "Check your network connection and firewall settings."
        if timeout:
            hint = f"Connection timed out after {timeout}s. {hint}"
        super().__init__(message, hint=hint)


class ConnectorTokenError(ConnectorError):
    """Invalid tunnel token."""

    def __init__(self, message: str = "Invalid or expired tunnel token"):
        super().__init__(
            message,
            hint="The tunnel token may have been rotated. Try creating a new lease.",
        )


class GatewayError(TunnelError):
    """Error with the local gateway."""

    def __init__(
        self,
        message: str,
        *,
        port: Optional[int] = None,
        hint: Optional[str] = None,
    ):
        self.port = port
        super().__init__(message, hint=hint)


class GatewayPortInUseError(GatewayError):
    """The gateway port is already in use."""

    def __init__(self, port: int):
        super().__init__(
            f"Gateway port {port} is already in use",
            port=port,
            hint=f"Kill the process using port {port} or use a different port.",
        )


class GatewayStartError(GatewayError):
    """Failed to start the gateway."""

    def __init__(self, message: str, *, port: int):
        super().__init__(
            f"Failed to start gateway on port {port}: {message}",
            port=port,
            hint="Check if another process is using the port or if you have network permissions.",
        )


class LocalAppError(TunnelError):
    """Error with the local application."""

    def __init__(
        self,
        message: str,
        *,
        port: Optional[int] = None,
        hint: Optional[str] = None,
    ):
        self.port = port
        default_hint = "Make sure your local app is running and healthy."
        if port:
            default_hint = f"Make sure your local app is running on port {port} and responding to health checks."
        super().__init__(message, hint=hint or default_hint)


class LocalAppNotReadyError(LocalAppError):
    """The local application is not responding to health checks."""

    def __init__(self, port: int, *, timeout: Optional[float] = None):
        msg = f"Local app on port {port} is not responding"
        if timeout:
            msg = f"{msg} (waited {timeout}s)"
        super().__init__(msg, port=port)


class DNSResolutionError(TunnelError):
    """Error resolving tunnel DNS."""

    def __init__(self, hostname: str, *, timeout: Optional[float] = None):
        msg = f"DNS resolution failed for {hostname}"
        if timeout:
            msg = f"{msg} (waited {timeout}s)"
        super().__init__(
            msg,
            hint=(
                "DNS propagation can take a few seconds after cloudflared connects.\n"
                "If this persists, check that cloudflared is connected and your network can reach Cloudflare."
            ),
        )


class RateLimitError(TunnelError):
    """Rate limit exceeded (usually for quick tunnels)."""

    def __init__(self, *, retry_after: Optional[int] = None):
        msg = "Rate limit exceeded"
        hint = "Wait a few minutes before trying again."
        if retry_after:
            hint = f"Try again in {retry_after} seconds."
        super().__init__(
            msg,
            hint=f"{hint}\nConsider using managed tunnels instead of quick tunnels for more reliability.",
        )
