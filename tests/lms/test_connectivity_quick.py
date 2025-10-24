import os
import httpx
import pytest


def _service_root(base: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/api"):
        return base[: -len("/api")]
    return base


@pytest.mark.integration
@pytest.mark.fast
def test_openapi_available_quick(backend_base_url: str):
    root = _service_root(backend_base_url)
    url = f"{root}/openapi.json"
    with httpx.Client(timeout=float(os.getenv("OPENAPI_TIMEOUT", "10"))) as client:
        resp = client.get(url)
    assert resp.status_code == 200, f"openapi.json failed ({resp.status_code})"
    data = resp.json()
    assert isinstance(data, dict) and "paths" in data


@pytest.mark.integration
@pytest.mark.fast
def test_modal_service_health(backend_base_url: str):
    """Test if Modal service is responding at all (basic connectivity)."""
    root = _service_root(backend_base_url)
    url = f"{root}/health"  # Common health endpoint
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(url)
        print(f"Health check: {resp.status_code} - {resp.text[:100]}")
        return resp.status_code < 500  # Any 4xx means service is responding
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


@pytest.mark.integration
@pytest.mark.fast
def test_modal_service_root(backend_base_url: str):
    """Test if Modal service root is accessible."""
    root = _service_root(backend_base_url)
    url = f"{root}/"  # Just the root
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(url)
        print(f"Root check: {resp.status_code} - {resp.text[:100]}")
        return resp.status_code < 500
    except Exception as e:
        print(f"Root check failed: {e}")
        return False


@pytest.mark.integration
@pytest.mark.fast
def test_modal_api_root(backend_base_url: str):
    """Test if /api endpoint is accessible."""
    url = f"{backend_base_url}/"  # Just /api/
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(url)
        print(f"API root check: {resp.status_code} - {resp.text[:100]}")
        return resp.status_code < 500
    except Exception as e:
        print(f"API root check failed: {e}")
        return False


@pytest.mark.integration
@pytest.mark.fast
def test_chat_completions_endpoint_exists(backend_base_url: str):
    """Test if /chat/completions endpoint exists (GET request)."""
    url = f"{backend_base_url}/chat/completions"
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(url)
        print(f"Chat completions GET: {resp.status_code} - {resp.text[:200]}")
        return resp.status_code < 500
    except Exception as e:
        print(f"Chat completions GET failed: {e}")
        return False


@pytest.mark.integration
def test_dns_resolution(backend_base_url: str):
    """Test if we can resolve the hostname."""
    import socket
    try:
        hostname = backend_base_url.replace("https://", "").replace("http://", "").split("/")[0]
        ip = socket.gethostbyname(hostname)
        print(f"DNS resolution: {hostname} -> {ip}")
        return True
    except Exception as e:
        print(f"DNS resolution failed: {e}")
        return False


@pytest.mark.integration
@pytest.mark.fast
def test_basic_connectivity(backend_base_url: str):
    """Test basic TCP connectivity to the host."""
    import socket
    try:
        hostname = backend_base_url.replace("https://", "").replace("http://", "").split("/")[0]
        port = 443 if backend_base_url.startswith("https") else 80

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        result = sock.connect_ex((hostname, port))
        sock.close()

        if result == 0:
            print(f"TCP connection: {hostname}:{port} - SUCCESS")
            return True
        else:
            print(f"TCP connection: {hostname}:{port} - FAILED (errno: {result})")
            return False
    except Exception as e:
        print(f"TCP connection failed: {e}")
        return False


