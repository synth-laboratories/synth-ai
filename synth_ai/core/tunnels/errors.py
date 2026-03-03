"""Custom exceptions for the tunnel system.

All exceptions are designed to be actionable - they include clear
error messages and suggested remediation steps.
"""

from __future__ import annotations

from enum import StrEnum
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
            hint = (
                hint
                or "Tunnel provisioning failed upstream. This is usually temporary - try again."
            )
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
    """Error with the tunnel connector runtime."""

    def __init__(
        self,
        message: str,
        *,
        hint: Optional[str] = None,
        diagnostics_id: Optional[str] = None,
    ):
        super().__init__(message, hint=hint, diagnostics_id=diagnostics_id)


class ConnectorNotInstalledError(ConnectorError):
    """A required tunnel connector is not installed."""

    def __init__(self):
        super().__init__(
            "Tunnel connector is not installed",
            hint="Install the required tunnel connector and retry.",
        )


class ConnectorConnectionError(ConnectorError):
    """Connector failed to establish an edge connection."""

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
                "DNS propagation can take a few seconds after tunnel creation.\n"
                "If this persists, verify your network resolver and managed tunnel host configuration."
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


class TunnelErrorCode(StrEnum):
    """Stable tunnel error taxonomy exposed by SDK surfaces."""

    PROVIDER_INVALID = "TUNNEL_PROVIDER_INVALID"
    PROVIDER_DEPRECATED_INPUT = "TUNNEL_PROVIDER_DEPRECATED_INPUT"
    URL_REQUIRED = "TUNNEL_URL_REQUIRED"
    URL_INVALID = "TUNNEL_URL_INVALID"
    URL_FORBIDDEN = "TUNNEL_URL_FORBIDDEN"
    AUTH_REQUIRED = "TUNNEL_AUTH_REQUIRED"
    AUTH_INVALID = "TUNNEL_AUTH_INVALID"
    LEASE_PENDING = "TUNNEL_LEASE_PENDING"
    LEASE_EXPIRED = "TUNNEL_LEASE_EXPIRED"
    AGENT_OFFLINE = "TUNNEL_AGENT_OFFLINE"
    ENDPOINT_UNREACHABLE = "TUNNEL_ENDPOINT_UNREACHABLE"
    CAPACITY_EXCEEDED = "TUNNEL_CAPACITY_EXCEEDED"
    PAYLOAD_TOO_LARGE = "TUNNEL_PAYLOAD_TOO_LARGE"
    TIMEOUT = "TUNNEL_TIMEOUT"
    INTERNAL = "TUNNEL_INTERNAL"


class TunnelProviderError(TunnelError):
    """Structured tunnel/provider error with stable code and backend metadata."""

    def __init__(
        self,
        message: str,
        *,
        code: TunnelErrorCode,
        status: int | None = None,
        request_id: str | None = None,
        backend_code: str | None = None,
        provider: str | None = None,
        hint: str | None = None,
    ):
        self.code = code
        self.status = status
        self.request_id = request_id
        self.backend_code = backend_code
        self.provider = provider
        super().__init__(message, hint=hint, diagnostics_id=request_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "status": self.status,
            "request_id": self.request_id,
            "backend_code": self.backend_code,
            "provider": self.provider,
            "message": self.message,
            "hint": self.hint,
        }


def map_problem_to_tunnel_error_code(
    problem_code: str | None, detail: str | None
) -> TunnelErrorCode:
    """Map backend problem+json codes/details into stable tunnel taxonomy codes."""
    code = (problem_code or "").strip().lower()
    detail_lc = (detail or "").strip().lower()

    if code in {"unauthorized", "forbidden"}:
        return TunnelErrorCode.AUTH_INVALID
    if code == "rate_limited":
        return TunnelErrorCode.CAPACITY_EXCEEDED
    if code == "upstream_error":
        return TunnelErrorCode.ENDPOINT_UNREACHABLE
    if code == "internal_error":
        return TunnelErrorCode.INTERNAL

    if "x-synthtunnel-worker-token" in detail_lc:
        return TunnelErrorCode.AUTH_REQUIRED
    if "no default managed ngrok url" in detail_lc:
        return TunnelErrorCode.URL_REQUIRED
    if "allow-listed" in detail_lc or "allowlisted" in detail_lc:
        return TunnelErrorCode.URL_FORBIDDEN
    if "invalid" in detail_lc and "url" in detail_lc:
        return TunnelErrorCode.URL_INVALID
    if "timed out" in detail_lc or "timeout" in detail_lc:
        return TunnelErrorCode.TIMEOUT
    if "agent_offline" in detail_lc or "agent offline" in detail_lc:
        return TunnelErrorCode.AGENT_OFFLINE

    return TunnelErrorCode.INTERNAL
