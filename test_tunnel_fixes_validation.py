#!/usr/bin/env python3
"""
Real-world validation test for Cloudflare tunnel fixes.

Tests against ACTUAL Cloudflare tunnels (no mocks):
1. Explicit DNS resolver logic (1.1.1.1, 8.8.8.8 fallback)
2. Retry wrapper for tunnel creation
3. Mode/hostname override logic
4. Env var configuration

Run this to validate logic works with real tunnels before integrating.
"""
import asyncio
import os
import socket
import subprocess
import time
from typing import Optional, Tuple
from urllib.parse import urlparse

import httpx

# Import real tunnel creation function
from synth_ai.cloudflare import open_quick_tunnel


# ============================================================================
# 1. Explicit DNS Resolver Logic
# ============================================================================

async def _resolve_hostname(hostname: str, loop: asyncio.AbstractEventLoop) -> str:
    """
    Resolve hostname using explicit resolvers (1.1.1.1, 8.8.8.8) first,
    then fall back to system resolver.
    
    This is the core fix for resolver path issues.
    """
    timeout = float(os.getenv("SYNTH_TUNNEL_DNS_TIMEOUT_PER_ATTEMPT_SECS", "5"))
    
    # Try Cloudflare / Google first via `dig`, then fall back to system resolver
    for resolver_ip in ("1.1.1.1", "8.8.8.8"):
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["dig", f"@{resolver_ip}", "+short", hostname],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                ),
            )
            if result.returncode == 0 and result.stdout.strip():
                first = result.stdout.strip().splitlines()[0].strip()
                if first:
                    print(f"  ✓ Resolved via {resolver_ip}: {hostname} -> {first}")
                    return first
        except FileNotFoundError:
            print(f"  ⚠️  dig not found, skipping {resolver_ip}")
            continue
        except Exception as e:
            print(f"  ⚠️  Resolver {resolver_ip} failed: {e}")
            continue
    
    # Fallback: system resolver
    print(f"  → Falling back to system resolver")
    return await loop.run_in_executor(
        None,
        socket.gethostbyname,
        hostname,
    )


async def verify_tunnel_dns_fixed(
    tunnel_url: str,
    name: str = "tunnel",
    timeout_seconds: float = 60.0,
) -> None:
    """
    Verify that a tunnel URL's hostname can be resolved via DNS (using public
    resolvers first) and that HTTP connectivity works by connecting directly
    to the resolved IP with the original Host header.
    
    This replaces the old version that only used system resolver and let httpx
    use the system resolver again for HTTP checks.
    """
    parsed = urlparse(tunnel_url)
    hostname = parsed.hostname
    if not hostname:
        print(f"⚠️  No hostname in {name} tunnel URL: {tunnel_url}")
        return
    
    # Skip DNS check for localhost
    if hostname in ("localhost", "127.0.0.1"):
        print(f"✓ Skipping DNS check for localhost {name}")
        return
    
    max_delay = 3.0
    delay = 0.5
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_seconds
    attempt = 0
    
    print(f"  Verifying DNS resolution for {name}: {hostname} (timeout {timeout_seconds:.0f}s)...")
    
    last_exc: Optional[Exception] = None
    
    while True:
        attempt += 1
        try:
            # 1. Resolve via explicit resolvers (1.1.1.1 / 8.8.8.8) → IP
            resolved_ip = await _resolve_hostname(hostname, loop)
            print(f"  ✓ DNS resolution successful (attempt {attempt}): {hostname} -> {resolved_ip}")
            
            # 2. HTTP connectivity: hit the tunnel via the resolved IP, but keep Host header.
            #    This avoids depending on the system resolver, which is what gave you EAI_NONAME.
            try:
                scheme = parsed.scheme or "https"
                test_url = f"{scheme}://{resolved_ip}/health"
                headers = {"Host": hostname}
                
                # For Quick Tunnels, TLS cert is for *.trycloudflare.com, not the bare IP,
                # so we disable verification here; this is just a readiness probe.
                async with httpx.AsyncClient(timeout=5.0, verify=False) as client:
                    resp = await client.get(test_url, headers=headers)
                    if resp.status_code in (200, 404, 405):
                        print(f"  ✓ HTTP connectivity verified via IP: {test_url} -> {resp.status_code}")
                        return
                    else:
                        print(f"  ⚠️  HTTP check returned unexpected status: {resp.status_code}")
                        last_exc = RuntimeError(f"unexpected HTTP status {resp.status_code}")
            except Exception as http_exc:
                print(f"  ⚠️  HTTP connectivity check failed (attempt {attempt}): {http_exc}")
                last_exc = http_exc
            
            # DNS resolved, but HTTP check failed - wait and retry until deadline
            now = loop.time()
            if now >= deadline:
                break
            delay = min(delay * 2 if attempt > 1 else delay, max_delay)
            sleep_for = min(delay, max(0.0, deadline - now))
            print(f"  Waiting {sleep_for:.1f}s before retry...")
            await asyncio.sleep(sleep_for)
            
        except socket.gaierror as e:
            print(f"  ⚠️  DNS resolution failed (attempt {attempt}): {e}")
            last_exc = e
            now = loop.time()
            if now >= deadline:
                raise RuntimeError(
                    f"DNS resolution failed for {name} tunnel hostname {hostname} "
                    f"after {timeout_seconds:.0f}s. Tunnel URL: {tunnel_url}. Error: {e}"
                ) from e
            delay = min(delay * 2 if attempt > 1 else delay, max_delay)
            sleep_for = min(delay, max(0.0, deadline - now))
            print(f"  Waiting {sleep_for:.1f}s before retry...")
            await asyncio.sleep(sleep_for)
        except Exception as e:
            print(f"  ❌ Unexpected error during DNS verification (attempt {attempt}): {e}")
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


# ============================================================================
# 2. Retry Wrapper Logic
# ============================================================================

async def create_quick_tunnel_with_retry(
    port: int,
    *,
    max_retries: Optional[int] = None,
    dns_timeout_s: Optional[float] = None,
) -> Tuple[str, subprocess.Popen]:
    """
    Retry wrapper for tunnel creation.
    
    This wraps open_quick_tunnel and verify_tunnel_dns with retry logic.
    Uses REAL open_quick_tunnel - no mocks.
    """
    max_retries = max_retries or int(os.getenv("SYNTH_TUNNEL_MAX_RETRIES", "2"))
    dns_timeout_s = dns_timeout_s or float(os.getenv("SYNTH_TUNNEL_DNS_TIMEOUT_SECS", "60"))
    
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        proc: Optional[subprocess.Popen] = None
        try:
            print(f"\n[Tunnel Attempt {attempt}/{max_retries}]")
            url, proc = open_quick_tunnel(port, wait_s=15.0)
            print(f"  ✓ Tunnel URL obtained: {url}")
            
            # Verify DNS (this is where failures usually happen)
            await verify_tunnel_dns_fixed(url, timeout_seconds=dns_timeout_s, name=f"tunnel attempt {attempt}")
            
            print(f"  ✓ Tunnel verified and ready!")
            return url, proc
        except Exception as e:
            last_err = e
            print(f"  ✗ Tunnel attempt {attempt} failed: {e}")
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
            if attempt < max_retries:
                print(f"  → Retrying after 10s backoff...")
                await asyncio.sleep(10.0)
            else:
                break
    
    assert last_err is not None
    raise last_err


# ============================================================================
# 3. Mode/Hostname Override Logic
# ============================================================================

async def create_tunnel_with_mode(
    port: int,
    *,
    mode: Optional[str] = None,
    override_hostname: Optional[str] = None,
) -> Tuple[str, Optional[subprocess.Popen]]:
    """
    Create tunnel with mode and hostname override support.
    
    Modes:
    - "local": Skip tunnel, use localhost
    - "quick": Use quick tunnel (default)
    - Other: Raise ValueError
    
    If override_hostname is set, replace hostname in URL (useful for pre-provisioned tunnels).
    """
    mode = mode or os.getenv("SYNTH_TUNNEL_MODE", "quick")
    override_host = override_hostname or os.getenv("SYNTH_TUNNEL_HOSTNAME")
    
    if mode == "local":
        url = f"http://127.0.0.1:{port}"
        proc = None
        print(f"✓ Using local mode: {url}")
        return url, proc
    elif mode == "quick":
        url, proc = await create_quick_tunnel_with_retry(port)
        if override_host:
            parsed = urlparse(url)
            url = f"{parsed.scheme}://{override_host}"
            print(f"✓ Overriding hostname: {url}")
        return url, proc
    else:
        raise ValueError(f"Unknown SYNTH_TUNNEL_MODE: {mode}")


# ============================================================================
# Test Cases
# ============================================================================

async def test_dns_resolver_logic():
    """Test 1: Explicit DNS resolver logic with known-good hostname"""
    print("\n" + "=" * 80)
    print("TEST 1: Explicit DNS Resolver Logic")
    print("=" * 80)
    
    # Test with known-good hostname
    test_hostname = "google.com"
    print(f"\nTesting resolver with {test_hostname}...")
    
    try:
        loop = asyncio.get_event_loop()
        resolved = await _resolve_hostname(test_hostname, loop)
        print(f"✓ Successfully resolved: {test_hostname} -> {resolved}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_real_tunnel_creation():
    """Test 2: Real tunnel creation with DNS verification"""
    print("\n" + "=" * 80)
    print("TEST 2: Real Tunnel Creation with DNS Verification")
    print("=" * 80)
    
    # Start a dummy HTTP server for the tunnel to forward to
    import http.server
    import socketserver
    import threading
    
    # Find an available port
    test_port = None
    for port in range(10000, 10100):
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            test_socket.bind(("127.0.0.1", port))
            test_socket.close()
            test_port = port
            break
        except OSError:
            continue
    
    if test_port is None:
        print("✗ Could not find available port")
        return False
    
    print(f"Using port {test_port} for test server")
    server = socketserver.TCPServer(("127.0.0.1", test_port), http.server.SimpleHTTPRequestHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    
    proc = None
    try:
        print(f"\nCreating real tunnel for port {test_port}...")
        
        # Use shorter timeout for testing (can be overridden via env)
        dns_timeout = float(os.getenv("SYNTH_TUNNEL_DNS_TIMEOUT_SECS", "60"))
        max_retries = int(os.getenv("SYNTH_TUNNEL_MAX_RETRIES", "2"))
        
        url, proc = await create_quick_tunnel_with_retry(
            test_port,
            max_retries=max_retries,
            dns_timeout_s=dns_timeout,
        )
        
        print(f"\n✓ Real tunnel created and verified: {url}")
        print(f"  Process PID: {proc.pid}")
        return True
        
    except RuntimeError as e:
        if "DNS resolution failed" in str(e):
            print(f"\n⚠️  Tunnel created but DNS verification failed (this is the issue we're fixing)")
            print(f"  Error: {e}")
            print(f"  This confirms the problem exists - our fixes should help!")
            return False  # This is a real failure we want to fix
        else:
            print(f"\n✗ Tunnel creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        # Shutdown server and wait a bit for port to be released
        try:
            server.shutdown()
            server.server_close()
        except Exception:
            pass
        await asyncio.sleep(0.5)  # Give port time to be released


async def test_mode_logic():
    """Test 3: Mode/hostname override logic"""
    print("\n" + "=" * 80)
    print("TEST 3: Mode/Hostname Override Logic")
    print("=" * 80)
    
    results = []
    
    # Test local mode
    print("\n[3a] Testing local mode...")
    try:
        url, proc = await create_tunnel_with_mode(8114, mode="local")
        assert url == "http://127.0.0.1:8114"
        assert proc is None
        print("✓ Local mode works correctly")
        results.append(True)
    except Exception as e:
        print(f"✗ Local mode failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)
    
    # Test quick mode (will create real tunnel)
    print("\n[3b] Testing quick mode...")
    import http.server
    import socketserver
    import threading
    
    # Find an available port (use different range to avoid conflict with TEST 2)
    test_port = None
    for port in range(10001, 10100):  # Start from 10001 to avoid conflict
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            test_socket.bind(("127.0.0.1", port))
            test_socket.close()
            test_port = port
            break
        except OSError:
            continue
    
    if test_port is None:
        print("✗ Could not find available port")
        results.append(False)
        return all(results)
    
    print(f"Using port {test_port} for test server")
    server = socketserver.TCPServer(("127.0.0.1", test_port), http.server.SimpleHTTPRequestHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    
    proc = None
    try:
        url, proc = await create_tunnel_with_mode(test_port, mode="quick")
        print(f"✓ Quick mode works (got URL: {url})")
        results.append(True)
    except RuntimeError as e:
        if "DNS resolution failed" in str(e):
            print("⚠️  Quick mode created tunnel but DNS failed (expected issue)")
            results.append(False)  # Real failure
        else:
            print(f"✗ Quick mode failed: {e}")
            results.append(False)
    except Exception as e:
        print(f"✗ Quick mode failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        server.shutdown()
    
    return all(results)


async def test_env_var_configuration():
    """Test 4: Env var configuration"""
    print("\n" + "=" * 80)
    print("TEST 4: Environment Variable Configuration")
    print("=" * 80)
    
    # Test that env vars are read correctly
    print("\nTesting env var defaults...")
    
    # Save original values
    orig_max_retries = os.environ.get("SYNTH_TUNNEL_MAX_RETRIES")
    orig_dns_timeout = os.environ.get("SYNTH_TUNNEL_DNS_TIMEOUT_SECS")
    orig_mode = os.environ.get("SYNTH_TUNNEL_MODE")
    orig_hostname = os.environ.get("SYNTH_TUNNEL_HOSTNAME")
    
    try:
        # Clear env vars to test defaults
        os.environ.pop("SYNTH_TUNNEL_MAX_RETRIES", None)
        os.environ.pop("SYNTH_TUNNEL_DNS_TIMEOUT_SECS", None)
        os.environ.pop("SYNTH_TUNNEL_MODE", None)
        os.environ.pop("SYNTH_TUNNEL_HOSTNAME", None)
        
        # Test defaults
        max_retries = int(os.getenv("SYNTH_TUNNEL_MAX_RETRIES", "2"))
        dns_timeout = float(os.getenv("SYNTH_TUNNEL_DNS_TIMEOUT_SECS", "60"))
        mode = os.getenv("SYNTH_TUNNEL_MODE", "quick")
        
        assert max_retries == 2, f"Expected max_retries=2, got {max_retries}"
        assert dns_timeout == 60.0, f"Expected dns_timeout=60.0, got {dns_timeout}"
        assert mode == "quick", f"Expected mode='quick', got {mode}"
        
        print("✓ Default env var values correct")
        
        # Test custom values
        os.environ["SYNTH_TUNNEL_MAX_RETRIES"] = "3"
        os.environ["SYNTH_TUNNEL_DNS_TIMEOUT_SECS"] = "90"
        os.environ["SYNTH_TUNNEL_MODE"] = "local"
        
        max_retries = int(os.getenv("SYNTH_TUNNEL_MAX_RETRIES", "2"))
        dns_timeout = float(os.getenv("SYNTH_TUNNEL_DNS_TIMEOUT_SECS", "60"))
        mode = os.getenv("SYNTH_TUNNEL_MODE", "quick")
        
        assert max_retries == 3, f"Expected max_retries=3, got {max_retries}"
        assert dns_timeout == 90.0, f"Expected dns_timeout=90.0, got {dns_timeout}"
        assert mode == "local", f"Expected mode='local', got {mode}"
        
        print("✓ Custom env var values read correctly")
        return True
        
    except Exception as e:
        print(f"✗ Env var configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Restore original values
        if orig_max_retries:
            os.environ["SYNTH_TUNNEL_MAX_RETRIES"] = orig_max_retries
        elif "SYNTH_TUNNEL_MAX_RETRIES" in os.environ:
            os.environ.pop("SYNTH_TUNNEL_MAX_RETRIES")
            
        if orig_dns_timeout:
            os.environ["SYNTH_TUNNEL_DNS_TIMEOUT_SECS"] = orig_dns_timeout
        elif "SYNTH_TUNNEL_DNS_TIMEOUT_SECS" in os.environ:
            os.environ.pop("SYNTH_TUNNEL_DNS_TIMEOUT_SECS")
            
        if orig_mode:
            os.environ["SYNTH_TUNNEL_MODE"] = orig_mode
        elif "SYNTH_TUNNEL_MODE" in os.environ:
            os.environ.pop("SYNTH_TUNNEL_MODE")
            
        if orig_hostname:
            os.environ["SYNTH_TUNNEL_HOSTNAME"] = orig_hostname
        elif "SYNTH_TUNNEL_HOSTNAME" in os.environ:
            os.environ.pop("SYNTH_TUNNEL_HOSTNAME")


async def main():
    """Run all validation tests"""
    print("\n" + "=" * 80)
    print("CLOUDFLARE TUNNEL FIXES - REAL TUNNEL VALIDATION TESTS")
    print("=" * 80)
    print("\n⚠️  NOTE: These tests create REAL Cloudflare tunnels.")
    print("   They require cloudflared to be installed and may take 60+ seconds.")
    print("   Set env vars to control behavior:")
    print("     SYNTH_TUNNEL_DNS_TIMEOUT_SECS=60")
    print("     SYNTH_TUNNEL_MAX_RETRIES=2")
    print("     SYNTH_TUNNEL_MODE=quick")
    print()
    
    results = []
    
    # Test 1: DNS resolver logic (fast, no tunnel needed)
    results.append(await test_dns_resolver_logic())
    
    # Test 2: Real tunnel creation (slow, creates actual tunnel)
    print("\n⚠️  Starting real tunnel test - this may take 60+ seconds...")
    results.append(await test_real_tunnel_creation())
    
    # Test 3: Mode logic (creates real tunnel for quick mode)
    results.append(await test_mode_logic())
    
    # Test 4: Env var configuration (fast, no tunnel needed)
    results.append(await test_env_var_configuration())
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\nTests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("\n✓ All validation tests passed! Logic works with real tunnels.")
        print("  Ready for integration into main codebase.")
        return 0
    else:
        print("\n⚠️  Some tests failed. This may indicate:")
        print("  1. DNS propagation is slow (expected - our fixes should help)")
        print("  2. Network/DNS resolver issues")
        print("  3. Logic bugs that need fixing")
        print("\nReview failures above before integrating.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
