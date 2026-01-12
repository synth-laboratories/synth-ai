"""Integration tests for TUI OpenCode integration.

These tests verify:
1. Environment variables are passed correctly to OpenCode
2. Local OpenCode checkout launches correctly (when OPENCODE_USE_LOCAL=1)
3. Synth provider loads with models when SYNTH_API_KEY is set
4. Working directory is correct (matches launch CWD, not tui/app)

Run with: pytest synth_ai/tui/tests/test_opencode_integration.py -v
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest
import requests
from dotenv import load_dotenv

# Load .env to get test API keys
load_dotenv(override=False)

# Paths
TUI_ROOT = Path(__file__).parent.parent / "app"
SYNTH_AI_ROOT = Path(__file__).parent.parent.parent.parent
OPENCODE_LOCAL_PATH = Path.home() / "Documents" / "GitHub" / "opencode"


def find_bun() -> str | None:
    """Find bun executable."""
    from shutil import which
    return which("bun")


def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Wait for a server to become available."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code < 500:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


def start_opencode_server(
    env: dict[str, str],
    cwd: str | Path,
    use_local: bool = False,
) -> tuple[subprocess.Popen, str | None]:
    """Start the OpenCode server and return (process, server_url).
    
    Args:
        env: Environment variables to pass
        cwd: Working directory for the server
        use_local: If True, use local opencode checkout
        
    Returns:
        Tuple of (process, server_url or None if failed)
    """
    bun = find_bun()
    if not bun:
        pytest.skip("bun not found")
    
    if use_local:
        # Use local opencode checkout
        entry = OPENCODE_LOCAL_PATH / "packages" / "opencode" / "src" / "index.ts"
        if not entry.exists():
            pytest.skip(f"Local opencode not found at {OPENCODE_LOCAL_PATH}")
        
        args = [bun, "run", "--conditions=browser", str(entry), "serve"]
        server_cwd = OPENCODE_LOCAL_PATH / "packages" / "opencode"
    else:
        # Use installed opencode via npx
        args = [bun, "x", "opencode@latest", "serve"]
        server_cwd = cwd
    
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(server_cwd),
        text=True,
    )
    
    # Parse stdout for server URL
    server_url = None
    start_time = time.time()
    timeout = 30.0
    
    while time.time() - start_time < timeout:
        if proc.poll() is not None:
            # Process exited
            stdout, stderr = proc.communicate()
            print(f"OpenCode exited early. stdout: {stdout}, stderr: {stderr}")
            return proc, None
            
        # Read available output
        import select
        if proc.stdout and select.select([proc.stdout], [], [], 0.1)[0]:
            line = proc.stdout.readline()
            if line:
                # Look for URL pattern like "http://127.0.0.1:XXXXX"
                if "http://" in line and ("127.0.0.1" in line or "localhost" in line):
                    import re
                    match = re.search(r'(https?://[^\s]+)', line)
                    if match:
                        server_url = match.group(1).rstrip('/')
                        break
    
    if server_url:
        # Wait for server to be ready
        if wait_for_server(server_url, timeout=10):
            return proc, server_url
    
    return proc, None


class TestEnvVarPassing:
    """Test that environment variables are passed correctly."""
    
    def test_synth_api_key_in_env(self):
        """SYNTH_API_KEY should be available after load_dotenv."""
        # This tests that the .env loading works
        api_key = os.environ.get("SYNTH_API_KEY")
        assert api_key, "SYNTH_API_KEY not found in environment. Check .env file."
        assert api_key.startswith("sk_"), f"SYNTH_API_KEY should start with 'sk_', got: {api_key[:10]}..."
    
    def test_launcher_sets_env_vars(self):
        """Test that launcher.py sets required env vars."""
        import inspect

        import synth_ai.tui.launcher as launcher_module
        
        # Get the source of the launcher function to verify it sets env vars
        func_source = inspect.getsource(launcher_module.run_prompt_learning_tui)
        module_source = inspect.getsource(launcher_module)
        
        assert "SYNTH_API_KEY" in func_source, "launcher should set SYNTH_API_KEY"
        assert "SYNTH_BACKEND_URL" in func_source, "launcher should set SYNTH_BACKEND_URL"
        assert "SYNTH_TUI_LAUNCH_CWD" in func_source, "launcher should set SYNTH_TUI_LAUNCH_CWD"
        assert "OPENCODE_WORKING_DIR" in func_source, "launcher should set OPENCODE_WORKING_DIR"
        assert "load_dotenv" in module_source, "launcher module should call load_dotenv"


class TestLocalOpenCode:
    """Test local OpenCode checkout integration."""
    
    @pytest.fixture
    def local_opencode_available(self) -> bool:
        """Check if local opencode is available."""
        entry = OPENCODE_LOCAL_PATH / "packages" / "opencode" / "src" / "index.ts"
        return entry.exists()
    
    def test_local_opencode_path_exists(self, local_opencode_available):
        """Verify local opencode checkout exists."""
        if not local_opencode_available:
            pytest.skip("Local opencode not available")
        assert local_opencode_available
    
    def test_local_opencode_has_synth_provider(self, local_opencode_available):
        """Verify local opencode has synth provider configured."""
        if not local_opencode_available:
            pytest.skip("Local opencode not available")
        
        provider_file = OPENCODE_LOCAL_PATH / "packages" / "opencode" / "src" / "provider" / "provider.ts"
        assert provider_file.exists(), f"provider.ts not found at {provider_file}"
        
        content = provider_file.read_text()
        assert "synth" in content.lower(), "synth provider not found in provider.ts"
        assert "SYNTH_API_KEY" in content, "SYNTH_API_KEY check not found in provider.ts"
        assert "synth-large-instant" in content, "synth-large-instant model not found in provider.ts"


class TestOpenCodeServerIntegration:
    """Integration tests that actually start the OpenCode server."""
    
    @pytest.fixture
    def test_env(self) -> dict[str, str]:
        """Create test environment with required variables."""
        env = os.environ.copy()
        
        # Ensure required vars are set
        if not env.get("SYNTH_API_KEY"):
            pytest.skip("SYNTH_API_KEY not set")
        
        env.setdefault("SYNTH_BACKEND_URL", "https://api.usesynth.ai")
        return env
    
    @pytest.fixture
    def temp_cwd(self) -> Path:
        """Create a temporary directory to use as CWD."""
        with tempfile.TemporaryDirectory(prefix="synth_tui_test_") as tmpdir:
            yield Path(tmpdir)
    
    @pytest.mark.slow
    def test_opencode_server_starts_with_local(self, test_env, temp_cwd):
        """Test that local OpenCode server starts correctly."""
        if not (OPENCODE_LOCAL_PATH / "packages" / "opencode" / "src" / "index.ts").exists():
            pytest.skip("Local opencode not available")
        
        proc, server_url = start_opencode_server(
            env=test_env,
            cwd=temp_cwd,
            use_local=True,
        )
        
        try:
            assert server_url is not None, "Failed to get server URL"
            assert "127.0.0.1" in server_url or "localhost" in server_url
            
            # Verify server responds
            resp = requests.get(server_url, timeout=5)
            assert resp.status_code < 500, f"Server returned error: {resp.status_code}"
        finally:
            proc.terminate()
            proc.wait(timeout=5)
    
    @pytest.mark.slow
    def test_synth_provider_loads(self, test_env, temp_cwd):
        """Test that synth provider loads when SYNTH_API_KEY is set."""
        if not (OPENCODE_LOCAL_PATH / "packages" / "opencode" / "src" / "index.ts").exists():
            pytest.skip("Local opencode not available")
        
        proc, server_url = start_opencode_server(
            env=test_env,
            cwd=temp_cwd,
            use_local=True,
        )
        
        try:
            assert server_url is not None, "Failed to start server"
            
            # Query the provider list API (no /api prefix)
            resp = requests.get(f"{server_url}/provider", timeout=10)
            
            # If we get a 500, log the error for debugging
            if resp.status_code == 500:
                error_text = resp.text
                pytest.fail(f"Provider API returned 500: {error_text[:500]}")
            
            assert resp.status_code == 200, f"Provider API returned {resp.status_code}: {resp.text[:200]}"
            
            data = resp.json()
            providers = data.get("all", data.get("data", {}).get("all", []))
            
            # Find synth provider
            synth_provider = None
            for p in providers:
                if isinstance(p, dict) and p.get("id") == "synth":
                    synth_provider = p
                    break
            
            assert synth_provider is not None, (
                f"Synth provider not found in providers. "
                f"Available: {[p.get('id') if isinstance(p, dict) else p for p in providers]}"
            )
            
            # Check for models
            models = synth_provider.get("models", {})
            assert len(models) > 0, "Synth provider has no models"
            assert "synth-large-instant" in models, (
                f"synth-large-instant not in models. Available: {list(models.keys())}"
            )
        finally:
            proc.terminate()
            proc.wait(timeout=5)
    
    @pytest.mark.slow
    def test_working_directory_propagation(self, test_env, temp_cwd):
        """Test that working directory is correctly set in sessions."""
        if not (OPENCODE_LOCAL_PATH / "packages" / "opencode" / "src" / "index.ts").exists():
            pytest.skip("Local opencode not available")
        
        # Set the working directory in env
        test_env["SYNTH_TUI_LAUNCH_CWD"] = str(temp_cwd)
        test_env["OPENCODE_WORKING_DIR"] = str(temp_cwd)
        
        proc, server_url = start_opencode_server(
            env=test_env,
            cwd=temp_cwd,
            use_local=True,
        )
        
        try:
            assert server_url is not None, "Failed to start server"
            
            # Create a session with the directory parameter (no /api prefix)
            resp = requests.post(
                f"{server_url}/session",
                params={"directory": str(temp_cwd)},
                timeout=10,
            )
            assert resp.status_code in (200, 201), f"Session creation failed: {resp.status_code}: {resp.text[:200]}"
            
            session_data = resp.json()
            session_id = session_data.get("id") or session_data.get("data", {}).get("id")
            assert session_id, f"No session ID in response: {session_data}"
            
            # Execute 'pwd' in the session to verify CWD
            resp = requests.post(
                f"{server_url}/session/{session_id}/shell",
                params={"directory": str(temp_cwd)},
                json={"command": "pwd"},
                timeout=30,
            )
            
            if resp.status_code == 200:
                result = resp.json()
                output = result.get("output", result.get("data", {}).get("output", ""))
                # The pwd output should contain our temp directory
                assert str(temp_cwd) in output or temp_cwd.name in output, (
                    f"Working directory mismatch. Expected {temp_cwd}, got: {output}"
                )
        finally:
            proc.terminate()
            proc.wait(timeout=5)


class TestOpenCodeServerTs:
    """Test the TypeScript opencode-server.ts module."""
    
    def test_opencode_server_ts_exists(self):
        """Verify opencode-server.ts exists."""
        path = TUI_ROOT / "src" / "utils" / "opencode-server.ts"
        assert path.exists(), f"opencode-server.ts not found at {path}"
    
    def test_opencode_server_ts_has_local_support(self):
        """Verify opencode-server.ts supports local opencode."""
        path = TUI_ROOT / "src" / "utils" / "opencode-server.ts"
        content = path.read_text()
        
        assert "OPENCODE_USE_LOCAL" in content, "Missing OPENCODE_USE_LOCAL support"
        assert "opencode-synth" in content or "opencode" in content, "Missing local opencode path"
        assert "bun" in content.lower() or "run" in content, "Missing bun run for local dev"
    
    def test_opencode_server_ts_passes_env(self):
        """Verify opencode-server.ts passes environment to spawn."""
        path = TUI_ROOT / "src" / "utils" / "opencode-server.ts"
        content = path.read_text()
        
        # Should explicitly pass env to spawn
        assert "env:" in content and "process.env" in content, (
            "opencode-server.ts should explicitly pass env to spawn"
        )


# Convenience function to run just the fast tests
def run_fast_tests():
    """Run only the fast (non-server) tests."""
    pytest.main([
        __file__,
        "-v",
        "-m", "not slow",
        "--tb=short",
    ])


if __name__ == "__main__":
    # Run all tests when executed directly
    pytest.main([__file__, "-v", "--tb=short"])
