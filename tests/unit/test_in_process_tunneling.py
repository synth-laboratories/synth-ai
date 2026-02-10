"""Unit tests for in-process tunneling defaults and backend normalization."""

import pytest

from synth_ai.core.tunnels import TunnelBackend
from synth_ai.sdk.localapi._impl.in_process import (
    InProcessTaskApp,
    _normalize_tunnel_backend,
)


# =============================================================================
# Default Configuration Tests
# =============================================================================


@pytest.mark.unit
def test_in_process_defaults(tmp_path):
    """SynthTunnel should be the default tunnel mode."""
    app_path = tmp_path / "task_app.py"
    app_path.write_text("# stub")
    app = InProcessTaskApp(task_app_path=app_path)
    assert app.tunnel_mode == "synthtunnel"
    assert app.tunnel_backend is None


@pytest.mark.unit
def test_in_process_default_port(tmp_path):
    """Default port should be 8114."""
    app_path = tmp_path / "task_app.py"
    app_path.write_text("# stub")
    app = InProcessTaskApp(task_app_path=app_path)
    assert app.port == 8114


@pytest.mark.unit
def test_in_process_default_host(tmp_path):
    """Default host should be 127.0.0.1."""
    app_path = tmp_path / "task_app.py"
    app_path.write_text("# stub")
    app = InProcessTaskApp(task_app_path=app_path)
    assert app.host == "127.0.0.1"


# =============================================================================
# Tunnel Backend Normalization Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # SynthTunnel variants
        ("synthtunnel", TunnelBackend.SynthTunnel),
        ("synth_tunnel", TunnelBackend.SynthTunnel),
        ("synth-tunnel", TunnelBackend.SynthTunnel),
        ("SynthTunnel", TunnelBackend.SynthTunnel),
        ("SYNTHTUNNEL", TunnelBackend.SynthTunnel),
        # Cloudflare quick tunnel variants
        ("quick", TunnelBackend.CloudflareQuickTunnel),
        ("cloudflare_quick", TunnelBackend.CloudflareQuickTunnel),
        ("cloudflare-quick", TunnelBackend.CloudflareQuickTunnel),
        # Cloudflare managed lease variants
        ("cloudflare_managed_lease", TunnelBackend.CloudflareManagedLease),
        ("cloudflare-managed-lease", TunnelBackend.CloudflareManagedLease),
        ("managed_lease", TunnelBackend.CloudflareManagedLease),
        ("managed-lease", TunnelBackend.CloudflareManagedLease),
        # Cloudflare managed (legacy)
        ("cloudflare_managed", TunnelBackend.CloudflareManagedTunnel),
        ("cloudflare-managed", TunnelBackend.CloudflareManagedTunnel),
        ("managed", TunnelBackend.CloudflareManagedTunnel),
        # Localhost variants
        ("local", TunnelBackend.Localhost),
        ("localhost", TunnelBackend.Localhost),
    ],
)
def test_normalize_tunnel_backend(value, expected):
    """Test all supported string aliases normalize to correct TunnelBackend."""
    assert _normalize_tunnel_backend(value) == expected


@pytest.mark.unit
def test_normalize_tunnel_backend_enum_passthrough():
    """TunnelBackend enum values should pass through unchanged."""
    assert _normalize_tunnel_backend(TunnelBackend.SynthTunnel) == TunnelBackend.SynthTunnel
    assert _normalize_tunnel_backend(TunnelBackend.CloudflareQuickTunnel) == TunnelBackend.CloudflareQuickTunnel
    assert _normalize_tunnel_backend(TunnelBackend.CloudflareManagedLease) == TunnelBackend.CloudflareManagedLease
    assert _normalize_tunnel_backend(TunnelBackend.Localhost) == TunnelBackend.Localhost


@pytest.mark.unit
def test_normalize_tunnel_backend_invalid():
    """Invalid backend strings should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown tunnel backend"):
        _normalize_tunnel_backend("nope")

    with pytest.raises(ValueError, match="Unknown tunnel backend"):
        _normalize_tunnel_backend("invalid_backend")


@pytest.mark.unit
def test_normalize_tunnel_backend_whitespace():
    """Whitespace should be stripped from backend strings."""
    assert _normalize_tunnel_backend("  synthtunnel  ") == TunnelBackend.SynthTunnel
    assert _normalize_tunnel_backend("\tquick\n") == TunnelBackend.CloudflareQuickTunnel


# =============================================================================
# Tunnel Backend Override Tests
# =============================================================================


@pytest.mark.unit
def test_tunnel_backend_override(tmp_path):
    """Explicit tunnel_backend should be stored and override tunnel_mode."""
    app_path = tmp_path / "task_app.py"
    app_path.write_text("# stub")
    
    # String backend
    app = InProcessTaskApp(task_app_path=app_path, tunnel_backend="cloudflare_quick")
    assert app.tunnel_backend == "cloudflare_quick"
    
    # Enum backend
    app = InProcessTaskApp(task_app_path=app_path, tunnel_backend=TunnelBackend.CloudflareManagedLease)
    assert app.tunnel_backend == TunnelBackend.CloudflareManagedLease


# =============================================================================
# Input Validation Tests
# =============================================================================


@pytest.mark.unit
def test_invalid_port_range(tmp_path):
    """Ports outside valid range should raise ValueError."""
    app_path = tmp_path / "task_app.py"
    app_path.write_text("# stub")
    
    with pytest.raises(ValueError, match="Port must be in range"):
        InProcessTaskApp(task_app_path=app_path, port=80)  # Too low
    
    with pytest.raises(ValueError, match="Port must be in range"):
        InProcessTaskApp(task_app_path=app_path, port=70000)  # Too high


@pytest.mark.unit
def test_invalid_host(tmp_path):
    """Non-localhost hosts should raise ValueError for security."""
    app_path = tmp_path / "task_app.py"
    app_path.write_text("# stub")
    
    with pytest.raises(ValueError, match="Host must be one of"):
        InProcessTaskApp(task_app_path=app_path, host="0.0.0.1")


@pytest.mark.unit
def test_valid_hosts(tmp_path):
    """Valid localhost variants should be accepted."""
    app_path = tmp_path / "task_app.py"
    app_path.write_text("# stub")
    
    # All these should work
    InProcessTaskApp(task_app_path=app_path, host="127.0.0.1")
    InProcessTaskApp(task_app_path=app_path, host="localhost")
    InProcessTaskApp(task_app_path=app_path, host="0.0.0.0")


@pytest.mark.unit
def test_invalid_tunnel_mode(tmp_path):
    """Invalid tunnel_mode should raise ValueError."""
    app_path = tmp_path / "task_app.py"
    app_path.write_text("# stub")
    
    with pytest.raises(ValueError, match="tunnel_mode must be one of"):
        InProcessTaskApp(task_app_path=app_path, tunnel_mode="invalid")


@pytest.mark.unit
def test_valid_tunnel_modes(tmp_path):
    """All valid tunnel modes should be accepted."""
    app_path = tmp_path / "task_app.py"
    app_path.write_text("# stub")
    
    for mode in ("synthtunnel", "quick", "named", "local", "preconfigured"):
        app = InProcessTaskApp(task_app_path=app_path, tunnel_mode=mode)
        assert app.tunnel_mode == mode


@pytest.mark.unit
def test_multiple_inputs_error():
    """Providing multiple input methods should raise ValueError."""
    from fastapi import FastAPI
    
    app = FastAPI()
    
    with pytest.raises(ValueError, match="Must provide exactly one of"):
        InProcessTaskApp(app=app, config_factory=lambda: None)


@pytest.mark.unit
def test_no_inputs_error():
    """Providing no input method should raise ValueError."""
    with pytest.raises(ValueError, match="Must provide exactly one of"):
        InProcessTaskApp()


@pytest.mark.unit
def test_task_app_path_not_found():
    """Non-existent task app path should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        InProcessTaskApp(task_app_path="/nonexistent/path.py")


@pytest.mark.unit
def test_task_app_path_not_python(tmp_path):
    """Non-.py task app path should raise ValueError."""
    app_path = tmp_path / "task_app.txt"
    app_path.write_text("# stub")
    
    with pytest.raises(ValueError, match="must be a .py file"):
        InProcessTaskApp(task_app_path=app_path)
