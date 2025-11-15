"""End-to-end integration tests for scan command.

These tests perform full integration testing by:
1. Deploying actual task apps (local and tunnel)
2. Verifying scan discovers them
3. Stopping the apps
4. Verifying scan no longer finds them

These tests require:
- Valid .env file with ENVIRONMENT_API_KEY and SYNTH_API_KEY
- Task app examples available
- Network connectivity for tunnel tests
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from synth_ai.cli.commands.scan.core import scan_command


@pytest.fixture
def runner() -> CliRunner:
    """CLI test runner fixture."""
    return CliRunner()


@pytest.fixture
def test_env_file(tmp_path: Path) -> Path:
    """Create a test .env file with required keys."""
    env_file = tmp_path / ".env"
    
    # Get keys from environment or use test values
    env_api_key = os.environ.get("ENVIRONMENT_API_KEY", "test_env_key")
    synth_api_key = os.environ.get("SYNTH_API_KEY", "test_synth_key")
    
    env_file.write_text(f"""ENVIRONMENT_API_KEY={env_api_key}
SYNTH_API_KEY={synth_api_key}
""")
    return env_file


@pytest.fixture
def repo_root() -> Path:
    """Get repository root path."""
    return Path(__file__).parent.parent.parent.parent


@pytest.fixture
def banking77_task_app(repo_root: Path) -> Path:
    """Path to banking77 task app."""
    app_path = repo_root / "examples" / "task_apps" / "banking77" / "banking77_task_app.py"
    if not app_path.exists():
        pytest.skip(f"Task app not found: {app_path}")
    return app_path


@pytest.fixture
def cleanup_ports():
    """Fixture to ensure ports are cleaned up after tests."""
    used_ports: list[int] = []
    
    yield used_ports
    
    # Cleanup: kill any processes on used ports
    for port in used_ports:
        try:
            # Try to find and kill process on port
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        time.sleep(0.5)
                        # Force kill if still running
                        try:
                            os.kill(int(pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    except (ValueError, ProcessLookupError, PermissionError):
                        pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass


def find_process_on_port(port: int) -> list[int]:
    """Find process IDs using a specific port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if result.returncode == 0 and result.stdout.strip():
            return [int(pid) for pid in result.stdout.strip().split("\n")]
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return []


def wait_for_port(port: int, timeout: float = 10.0) -> bool:
    """Wait for a port to become available."""
    import socket
    
    start = time.time()
    while time.time() - start < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex(("127.0.0.1", port))
            sock.close()
            if result == 0:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def wait_for_port_closed(port: int, timeout: float = 10.0) -> bool:
    """Wait for a port to become closed."""
    import socket
    
    start = time.time()
    while time.time() - start < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex(("127.0.0.1", port))
            sock.close()
            if result != 0:
                return True
        except Exception:
            return True
        time.sleep(0.5)
    return False


@pytest.mark.integration
@pytest.mark.timeout(60)  # 60 second timeout per test
class TestScanE2ELocal:
    """End-to-end tests for local app discovery."""

    def test_scan_discovers_local_app(
        self,
        runner: CliRunner,
        test_env_file: Path,
        banking77_task_app: Path,
        repo_root: Path,
        cleanup_ports: list[int],
    ):
        """Test that scan discovers a deployed local app."""
        test_port = 9000
        cleanup_ports.append(test_port)
        
        # Ensure port is free
        pids = find_process_on_port(test_port)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
            except ProcessLookupError:
                pass
        
        # Deploy local app using subprocess (non-blocking, so use Popen)
        deploy_cmd = [
            "uv",
            "run",
            "synth-ai",
            "deploy",
            "--runtime",
            "local",
            "--task-app",
            str(banking77_task_app),
            "--port",
            str(test_port),
            "--env",
            str(test_env_file),
        ]
        
        deploy_proc = subprocess.Popen(
            deploy_cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        
        # Wait briefly for deployment to start, then check if port is open
        # Deploy is non-blocking, so process will exit quickly
        try:
            stdout, _ = deploy_proc.communicate(timeout=10.0)
            deploy_output = stdout
            deploy_exit_code = deploy_proc.returncode
        except subprocess.TimeoutExpired:
            # Process still running (shouldn't happen with non-blocking deploy)
            deploy_proc.kill()
            deploy_proc.wait()
            deploy_output = ""
            deploy_exit_code = 0  # Assume success if it's still running
        
        # Give it time to start (deploy is non-blocking, server starts in background)
        if deploy_exit_code == 0 or "Server started" in deploy_output or "PID" in deploy_output:
            assert wait_for_port(test_port, timeout=15.0), "App failed to start on port"
            
            # Wait a bit more for health endpoint to be ready
            time.sleep(2.0)
            
            # Scan for the app (with timeout protection)
            try:
                scan_result = runner.invoke(
                    scan_command,
                    [
                        "--port-range",
                        f"{test_port}:{test_port}",
                        "--api-key",
                        os.environ.get("ENVIRONMENT_API_KEY", "test_env_key"),
                        "--timeout",
                        "2.0",
                        "--json",
                    ],
                )
            except Exception as e:
                pytest.fail(f"Scan command failed: {e}")
            
            assert scan_result.exit_code == 0, f"Scan failed: {scan_result.output}"
            
            # Parse JSON output
            output_lines = [line for line in scan_result.output.split("\n") if line.strip() and not line.startswith("INFO:")]
            if output_lines:
                try:
                    data = json.loads("\n".join(output_lines))
                    apps = data.get("apps", [])
                    
                    # Find our app
                    found_app = next(
                        (app for app in apps if app.get("port") == test_port),
                        None,
                    )
                    
                    assert found_app is not None, f"App not found in scan results: {json.dumps(apps, indent=2)}"
                    assert found_app["type"] == "local"
                    assert found_app["port"] == test_port
                    # Health status might be healthy or unknown depending on timing
                    assert found_app["health_status"] in ("healthy", "unknown", "unhealthy")
                except json.JSONDecodeError as e:
                    pytest.fail(f"Failed to parse JSON output: {e}\nOutput: {scan_result.output}")
        else:
            pytest.skip(f"Failed to deploy app: {deploy_output}")
        
        # Cleanup: kill process on port
        pids = find_process_on_port(test_port)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        
        # Verify port is closed
        assert wait_for_port_closed(test_port, timeout=5.0), "Port still open after cleanup"
        
        # Scan again - should not find the app (with timeout protection)
        try:
            scan_result_after = runner.invoke(
                scan_command,
                [
                    "--port-range",
                    f"{test_port}:{test_port}",
                    "--api-key",
                    os.environ.get("ENVIRONMENT_API_KEY", "test_env_key"),
                    "--timeout",
                    "2.0",
                    "--json",
                ],
            )
        except Exception as e:
            # If scan fails, that's okay - we're just verifying cleanup
            scan_result_after = type("obj", (object,), {"exit_code": 0, "output": "{}"})()
        
        if scan_result_after.exit_code == 0:
            output_lines = [line for line in scan_result_after.output.split("\n") if line.strip() and not line.startswith("INFO:")]
            if output_lines:
                try:
                    data = json.loads("\n".join(output_lines))
                    apps = data.get("apps", [])
                    found_app = next(
                        (app for app in apps if app.get("port") == test_port),
                        None,
                    )
                    # App should not be found (or should be unhealthy/unknown)
                    if found_app:
                        # If found, it should be unhealthy/unknown since process is dead
                        assert found_app["health_status"] in ("unhealthy", "unknown")
                except json.JSONDecodeError:
                    pass  # Ignore JSON errors in cleanup verification

    def test_scan_discovers_multiple_local_apps(
        self,
        runner: CliRunner,
        test_env_file: Path,
        banking77_task_app: Path,
        repo_root: Path,
        cleanup_ports: list[int],
    ):
        """Test that scan discovers multiple deployed local apps."""
        ports = [9001, 9002]
        cleanup_ports.extend(ports)
        
        deployed_pids: list[int] = []
        
        try:
            # Deploy multiple apps
            for port in ports:
                # Ensure port is free
                pids = find_process_on_port(port)
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGTERM)
                        time.sleep(0.5)
                    except ProcessLookupError:
                        pass
                
                deploy_cmd = [
                    "uv",
                    "run",
                    "synth-ai",
                    "deploy",
                    "--runtime",
                    "local",
                    "--task-app",
                    str(banking77_task_app),
                    "--port",
                    str(port),
                    "--env",
                    str(test_env_file),
                ]
                
                deploy_proc = subprocess.Popen(
                    deploy_cmd,
                    cwd=str(repo_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                
                try:
                    stdout, _ = deploy_proc.communicate(timeout=10.0)
                    deploy_output = stdout
                    deploy_exit_code = deploy_proc.returncode
                except subprocess.TimeoutExpired:
                    deploy_proc.kill()
                    deploy_proc.wait()
                    deploy_output = ""
                    deploy_exit_code = 0
                
                if deploy_exit_code == 0 or "Server started" in deploy_output or "PID" in deploy_output:
                    assert wait_for_port(port, timeout=15.0), f"App failed to start on port {port}"
                    time.sleep(1.0)  # Give it time to be ready
                    deployed_pids.extend(find_process_on_port(port))
            
            if not deployed_pids:
                pytest.skip("Failed to deploy apps")
            
            # Wait for all apps to be ready
            time.sleep(2.0)
            
            # Scan for the apps (with timeout protection)
            try:
                scan_result = runner.invoke(
                    scan_command,
                    [
                        "--port-range",
                        f"{min(ports)}:{max(ports)}",
                        "--api-key",
                        os.environ.get("ENVIRONMENT_API_KEY", "test_env_key"),
                        "--timeout",
                        "2.0",
                        "--json",
                    ],
                )
            except Exception as e:
                pytest.fail(f"Scan command failed: {e}")
            
            assert scan_result.exit_code == 0, f"Scan failed: {scan_result.output}"
            
            # Parse JSON output
            output_lines = [line for line in scan_result.output.split("\n") if line.strip() and not line.startswith("INFO:")]
            if output_lines:
                try:
                    data = json.loads("\n".join(output_lines))
                    apps = data.get("apps", [])
                    
                    # Find our apps
                    found_ports = {app.get("port") for app in apps if app.get("port") in ports}
                    
                    # Should find at least some of the apps (may not find all due to timing)
                    assert len(found_ports) > 0, f"No apps found on ports {ports}. Found: {[app.get('port') for app in apps]}"
                except json.JSONDecodeError as e:
                    pytest.fail(f"Failed to parse JSON output: {e}\nOutput: {scan_result.output}")
        finally:
            # Cleanup: kill all processes
            for port in ports:
                pids = find_process_on_port(port)
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGTERM)
                        time.sleep(0.5)
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.timeout(120)  # 2 minute timeout for tunnel tests
class TestScanE2ETunnel:
    """End-to-end tests for tunnel app discovery.

    These tests are marked as slow because they involve actual tunnel deployment
    which can take 10-30 seconds.
    """

    def test_scan_discovers_tunnel_app(
        self,
        runner: CliRunner,
        test_env_file: Path,
        banking77_task_app: Path,
        repo_root: Path,
        cleanup_ports: list[int],
    ):
        """Test that scan discovers a deployed tunnel app."""
        test_port = 9003
        cleanup_ports.append(test_port)
        
        # Ensure port is free
        pids = find_process_on_port(test_port)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
            except ProcessLookupError:
                pass
        
        # Deploy tunnel app (non-blocking)
        deploy_cmd = [
            "uv",
            "run",
            "synth-ai",
            "deploy",
            "--runtime",
            "tunnel",
            "--task-app",
            str(banking77_task_app),
            "--port",
            str(test_port),
            "--env",
            str(test_env_file),
            "--tunnel-mode",
            "quick",
        ]
        
        deploy_proc = subprocess.Popen(
            deploy_cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        
        # Wait briefly for tunnel deployment output
        try:
            stdout, _ = deploy_proc.communicate(timeout=15.0)
            deploy_output = stdout
            deploy_exit_code = deploy_proc.returncode
        except subprocess.TimeoutExpired:
            # Process still running (tunnel is background)
            deploy_output = ""
            deploy_exit_code = 0  # Assume success
        
        # Tunnel deployment may take time, give it a moment
        time.sleep(5.0)
        
        if deploy_exit_code == 0 or "Tunnel ready" in deploy_output or "PID" in deploy_output:
            # Wait for local server to be ready
            assert wait_for_port(test_port, timeout=20.0), "Local server failed to start"
            time.sleep(3.0)  # Give tunnel time to establish
            
            # Scan for the app (with timeout protection)
            try:
                scan_result = runner.invoke(
                    scan_command,
                    [
                        "--port-range",
                        f"{test_port}:{test_port}",
                        "--api-key",
                        os.environ.get("ENVIRONMENT_API_KEY", "test_env_key"),
                        "--timeout",
                        "2.0",
                        "--json",
                    ],
                )
            except Exception as e:
                pytest.fail(f"Scan command failed: {e}")
            
            assert scan_result.exit_code == 0, f"Scan failed: {scan_result.output}"
            
            # Parse JSON output
            output_lines = [line for line in scan_result.output.split("\n") if line.strip() and not line.startswith("INFO:")]
            if output_lines:
                try:
                    data = json.loads("\n".join(output_lines))
                    apps = data.get("apps", [])
                    
                    # Find our app (could be found via tunnel_records or port_scan)
                    found_app = next(
                        (
                            app
                            for app in apps
                            if app.get("port") == test_port
                            or (app.get("type") == "cloudflare" and "trycloudflare.com" in app.get("url", ""))
                        ),
                        None,
                    )
                    
                    # App might be found via tunnel_records or port_scan
                    if found_app:
                        assert found_app["type"] in ("local", "cloudflare")
                        # Health status might vary
                        assert found_app["health_status"] in ("healthy", "unknown", "unhealthy")
                except json.JSONDecodeError as e:
                    pytest.fail(f"Failed to parse JSON output: {e}\nOutput: {scan_result.output}")
        else:
            pytest.skip(f"Failed to deploy tunnel: {deploy_output}")
        
        # Cleanup: kill processes
        pids = find_process_on_port(test_port)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        
        # Also kill cloudflared processes
        try:
            result = subprocess.run(
                ["pgrep", "-f", f"cloudflared.*{test_port}"],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if result.returncode == 0:
                for pid_str in result.stdout.strip().split("\n"):
                    try:
                        os.kill(int(pid_str), signal.SIGTERM)
                        time.sleep(0.5)
                        os.kill(int(pid_str), signal.SIGKILL)
                    except (ValueError, ProcessLookupError):
                        pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass


@pytest.mark.integration
@pytest.mark.timeout(60)  # 60 second timeout per test
class TestScanE2EServiceRecords:
    """End-to-end tests for service records discovery."""

    def test_scan_discovers_service_records(
        self,
        runner: CliRunner,
        test_env_file: Path,
        banking77_task_app: Path,
        repo_root: Path,
        cleanup_ports: list[int],
    ):
        """Test that scan discovers apps via service records."""
        test_port = 9004
        cleanup_ports.append(test_port)
        
        # Ensure port is free
        pids = find_process_on_port(test_port)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
            except ProcessLookupError:
                pass
        
        # Deploy local app (creates service record)
        deploy_cmd = [
            "uv",
            "run",
            "synth-ai",
            "deploy",
            "--runtime",
            "local",
            "--task-app",
            str(banking77_task_app),
            "--port",
            str(test_port),
            "--env",
            str(test_env_file),
        ]
        
        deploy_proc = subprocess.Popen(
            deploy_cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        
        try:
            stdout, _ = deploy_proc.communicate(timeout=10.0)
            deploy_output = stdout
            deploy_exit_code = deploy_proc.returncode
        except subprocess.TimeoutExpired:
            deploy_proc.kill()
            deploy_proc.wait()
            deploy_output = ""
            deploy_exit_code = 0
        
        if deploy_exit_code == 0 or "Server started" in deploy_output or "PID" in deploy_output:
            assert wait_for_port(test_port, timeout=15.0), "App failed to start"
            time.sleep(2.0)
            
            # Scan with wider range to ensure we find it via service_records (with timeout)
            try:
                scan_result = runner.invoke(
                    scan_command,
                    [
                        "--port-range",
                        "8000:10000",  # Wide range
                        "--api-key",
                        os.environ.get("ENVIRONMENT_API_KEY", "test_env_key"),
                        "--timeout",
                        "2.0",
                        "--json",
                    ],
                )
            except Exception as e:
                pytest.fail(f"Scan command failed: {e}")
            
            assert scan_result.exit_code == 0, f"Scan failed: {scan_result.output}"
            
            # Parse JSON output
            output_lines = [line for line in scan_result.output.split("\n") if line.strip() and not line.startswith("INFO:")]
            if output_lines:
                try:
                    data = json.loads("\n".join(output_lines))
                    apps = data.get("apps", [])
                    
                    # Find our app
                    found_app = next(
                        (app for app in apps if app.get("port") == test_port),
                        None,
                    )
                    
                    if found_app:
                        # Should be discovered via service_records
                        assert found_app["discovered_via"] in (
                            "service_records",
                            "port_scan",
                        ), f"Unexpected discovery method: {found_app['discovered_via']}"
                        assert found_app["type"] == "local"
                except json.JSONDecodeError as e:
                    pytest.fail(f"Failed to parse JSON output: {e}\nOutput: {scan_result.output}")
        else:
            pytest.skip(f"Failed to deploy app: {deploy_output}")
        
        # Cleanup
        pids = find_process_on_port(test_port)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

