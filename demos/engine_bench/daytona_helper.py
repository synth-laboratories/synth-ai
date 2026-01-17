"""
Daytona helper for running localapi servers in Daytona sandboxes.

This module provides utilities to:
1. Provision Daytona sandboxes
2. Upload localapi code to sandboxes
3. Run localapi servers inside sandboxes with preview URLs
4. Clean up sandboxes after use
"""

import asyncio
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse, urlunparse


def normalize_interceptor_base(inference_url: str) -> tuple[str, str | None]:
    """Normalize interceptor base URL for path-based routing.

    The interceptor route format is:
    /{trial_id}/{correlation_id}/responses

    Input URL format:
    https://host/api/interceptor/v1/{trial_id}/chat/completions?cid={correlation_id}

    Output base URL (for codex to append /responses):
    https://host/api/interceptor/v1/{trial_id}/{correlation_id}

    Returns:
    - base: URL with both trial_id and correlation_id in path
    - correlation_id: from query param 'cid'
    """
    parsed = urlparse(inference_url)
    cid_values = parse_qs(parsed.query).get("cid", [])
    correlation_id = cid_values[0] if cid_values else None

    base_path = parsed.path or ""

    # Strip endpoint suffixes
    for suffix in ["/v1/chat/completions", "/chat/completions", "/responses", "/v1/responses"]:
        if base_path.endswith(suffix):
            base_path = base_path[: -len(suffix)]
            break

    # Append correlation_id to path so route is /{trial_id}/{correlation_id}/responses
    if correlation_id:
        base_path = f"{base_path}/{correlation_id}"

    base = f"{parsed.scheme}://{parsed.netloc}{base_path}".rstrip("/")
    return base, correlation_id


try:
    from daytona_sdk import CodeLanguage, CreateSandboxFromImageParams, Daytona, DaytonaConfig

    DAYTONA_AVAILABLE = True
except ImportError:
    try:
        # Try alternate import path
        from daytona import CodeLanguage, CreateSandboxFromImageParams, Daytona, DaytonaConfig

        DAYTONA_AVAILABLE = True
    except ImportError:
        DAYTONA_AVAILABLE = False
        # Classes not defined - code must check DAYTONA_AVAILABLE before use


class DaytonaLocalapiRunner:
    """Manages running a localapi server in a Daytona sandbox.

    Supports snapshot-based caching for fast startup:
    1. First run: Create sandbox from base image, run setup, create snapshot
    2. Subsequent runs: Create sandbox from snapshot (~10s vs ~100s)
    """

    # Snapshot name for cached sandboxes (with full deps: Rust, synth-ai, engine-bench)
    SNAPSHOT_NAME = "synth-engine-bench-full-v3"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_url: str | None = None,
        target: str | None = None,
        image: str = "python:3.11-slim",
        localapi_port: int = 8000,
        use_snapshot: bool = True,  # Try to use cached snapshot
    ) -> None:
        """Initialize Daytona runner.

        Args:
            api_key: Daytona API key (or use DAYTONA_API_KEY env var)
            api_url: Daytona API URL (or use DAYTONA_API_URL env var)
            target: Target region (or use DAYTONA_TARGET env var)
            image: Base image for sandbox (default: ubuntu latest)
            localapi_port: Port for localapi to bind to (default: 8000)
        """
        if not DAYTONA_AVAILABLE:
            raise RuntimeError("Daytona SDK not available. Install with: pip install daytona")

        self.api_key = api_key or os.environ.get("DAYTONA_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "DAYTONA_API_KEY environment variable required. Set it or pass api_key parameter."
            )

        self.api_url = api_url or os.environ.get("DAYTONA_API_URL")
        self.target = target or os.environ.get("DAYTONA_TARGET")
        self.image = image
        self.localapi_port = localapi_port
        self.use_snapshot = use_snapshot

        config = DaytonaConfig(api_key=self.api_key)
        if self.api_url:
            config.api_url = self.api_url
        if self.target:
            config.target = self.target

        self.client = Daytona(config)
        self.sandbox = None
        self.sandbox_id = None
        self.preview_url = None
        self._localapi_process = None
        self._created_from_snapshot = False

    async def provision(self) -> str:
        """Provision a new Daytona sandbox.

        Uses snapshot caching for fast startup:
        1. Try to create from existing snapshot (fast: ~10s)
        2. Fall back to base image if snapshot not found (~60s+)

        Returns:
            Sandbox ID
        """
        # Try snapshot first if enabled
        if self.use_snapshot:
            try:
                print(f"[Daytona] Trying to create from snapshot: {self.SNAPSHOT_NAME}")
                from daytona_sdk import CreateSandboxFromSnapshotParams

                params = CreateSandboxFromSnapshotParams(
                    snapshot=self.SNAPSHOT_NAME,
                    language=CodeLanguage.PYTHON,
                    public=True,
                )
                self.sandbox = self.client.create(params)
                self.sandbox_id = self.sandbox.id
                self._created_from_snapshot = True
                print(f"[Daytona] ✅ Created from snapshot (fast path): {self.sandbox_id}")
            except Exception as e:
                print(f"[Daytona] Snapshot not found or failed: {e}")
                print("[Daytona] Falling back to base image...")
                self._created_from_snapshot = False

        # Fall back to base image
        if not self._created_from_snapshot:
            print(f"[Daytona] Creating from base image: {self.image}")
            params = CreateSandboxFromImageParams(
                image=self.image,
                language=CodeLanguage.PYTHON,
                public=True,
            )
            self.sandbox = self.client.create(params)
            self.sandbox_id = self.sandbox.id
            print(f"[Daytona] Sandbox provisioned: {self.sandbox_id}")

        # Get preview URL for the localapi port
        try:
            preview_info = self.sandbox.get_preview_link(self.localapi_port)
            print(f"[Daytona] Preview URL: {preview_info}")
            # Extract URL from preview info object
            if hasattr(preview_info, "url"):
                self.preview_url = preview_info.url
            elif isinstance(preview_info, dict):
                self.preview_url = preview_info.get("url")
            elif isinstance(preview_info, str):
                self.preview_url = preview_info
            else:
                # Fallback: construct expected format
                self.preview_url = (
                    f"https://{self.localapi_port}-{self.sandbox_id}.proxy.daytona.works"
                )

            if self.preview_url:
                print(f"[Daytona] Using preview URL: {self.preview_url}")
            else:
                print(f"[Daytona] Warning: No preview URL available for port {self.localapi_port}")
        except Exception as e:
            print(f"[Daytona] Warning: Could not get preview URL: {e}")
            # Fallback: construct expected format
            self.preview_url = f"https://{self.localapi_port}-{self.sandbox_id}.proxy.daytona.works"
            print(f"[Daytona] Using constructed preview URL: {self.preview_url}")

        assert self.sandbox_id is not None, "Sandbox ID should be set after provisioning"
        return self.sandbox_id

    async def _get_preview_url(self, port: int) -> dict[str, Any]:
        """Get preview URL for a port in the sandbox."""
        if not self.sandbox:
            raise RuntimeError("Sandbox not provisioned")

        # Try to get preview URL from sandbox
        # Daytona SDK may expose this via sandbox.get_preview_link() or similar
        try:
            # Try to get preview link from sandbox object
            if hasattr(self.sandbox, "get_preview_link"):
                preview_link = self.sandbox.get_preview_link(port)
                if preview_link:
                    return {
                        "url": preview_link,
                        "port": port,
                        "sandbox_id": self.sandbox_id,
                    }

            # Fallback: construct expected format
            preview_url = f"https://{port}-{self.sandbox_id}.proxy.daytona.work"
            return {
                "url": preview_url,
                "port": port,
                "sandbox_id": self.sandbox_id,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get preview URL: {e}") from e

    async def upload_localapi(
        self,
        localapi_path: Path,
        *,
        additional_files: list[Path] | None = None,
    ) -> None:
        """Upload localapi code to the sandbox.

        Args:
            localapi_path: Path to the localapi Python file
            additional_files: Optional list of additional files to upload
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox not provisioned. Call provision() first.")

        print(f"[Daytona] Uploading localapi: {localapi_path}")

        # Read localapi file
        localapi_content = localapi_path.read_text()
        localapi_bytes = localapi_content.encode("utf-8")

        # Upload to /app/localapi.py in sandbox using fs.upload_file
        await asyncio.to_thread(
            self.sandbox.fs.upload_file,
            localapi_bytes,
            "/app/localapi.py",
        )

        # Upload additional files if provided
        if additional_files:
            for file_path in additional_files:
                if file_path.is_file():
                    rel_path = f"/app/{file_path.name}"
                    content = file_path.read_bytes()
                    await asyncio.to_thread(
                        self.sandbox.fs.upload_file,
                        content,
                        rel_path,
                    )
                    print(f"[Daytona] Uploaded: {rel_path}")

        print("[Daytona] Localapi uploaded successfully")

    async def setup_environment(
        self,
        *,
        env_vars: dict[str, str] | None = None,
        install_commands: list[str] | None = None,
    ) -> None:
        """Set up environment in the sandbox.

        Args:
            env_vars: Environment variables to set
            install_commands: Commands to run for setup (e.g., pip install)
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox not provisioned. Call provision() first.")

        print("[Daytona] Setting up environment...")

        # Set environment variables
        if env_vars:
            env_file_content = "\n".join(f"export {k}='{v}'" for k, v in env_vars.items())
            env_bytes = env_file_content.encode("utf-8")
            await asyncio.to_thread(
                self.sandbox.fs.upload_file,
                env_bytes,
                "/app/.env",
            )
            print(f"[Daytona] Set {len(env_vars)} environment variables")

        # Run install commands
        if install_commands:
            for cmd in install_commands:
                print(f"[Daytona] Running: {cmd}")
                result = await asyncio.to_thread(
                    self.sandbox.process.exec,
                    cmd,
                )
                exit_code = (
                    getattr(result, "exit_code", None) or getattr(result, "return_code", None) or 0
                )
                output = getattr(result, "output", "") or getattr(result, "result", "") or ""
                if exit_code != 0:
                    print(f"[Daytona] Warning: Command failed: {cmd}")
                    print(f"[Daytona] Exit code: {exit_code}")
                    print(f"[Daytona] Output: {output[:500]}")

        print("[Daytona] Environment setup complete")

    async def start_localapi(
        self,
        *,
        host: str = "0.0.0.0",
        wait_for_health: bool = True,
        health_timeout: float = 60.0,
    ) -> str:
        """Start the localapi in the sandbox.

        Args:
            host: Host to bind to (must be 0.0.0.0 for Daytona preview URLs)
            wait_for_health: Whether to wait for health check
            health_timeout: Timeout for health check

        Returns:
            Localapi URL (preview URL if available, otherwise localhost)
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox not provisioned. Call provision() first.")

        if not self.preview_url:
            raise RuntimeError(
                "Preview URL not available. Cannot start localapi without preview URL."
            )

        print(f"[Daytona] Starting localapi on {host}:{self.localapi_port}")

        # Start localapi in background
        # Use uvicorn to run the FastAPI app
        # Source .env first to load environment variables
        cmd = (
            f"cd /app && "
            f"source .env 2>/dev/null || true && "
            f"nohup python3 -m uvicorn localapi:app "
            f"--host {host} --port {self.localapi_port} "
            f"--log-level info > /app/localapi.log 2>&1 &"
        )

        print(f"[Daytona] Running command: {cmd[:100]}...")

        # Run command (nohup makes it background)
        result = await asyncio.to_thread(
            self.sandbox.process.exec,
            cmd,
        )

        self._localapi_process = result

        # Give it a moment to start
        await asyncio.sleep(5.0)

        # Check logs
        try:
            log_result = await asyncio.to_thread(
                self.sandbox.process.exec,
                "cat /app/localapi.log 2>/dev/null | tail -50 || echo 'No log file'",
            )
            log_output = (
                getattr(log_result, "output", "") or getattr(log_result, "result", "") or ""
            )
            if log_output.strip():
                print(f"[Daytona] Localapi logs:\n{log_output[:500]}")
        except Exception as e:
            print(f"[Daytona] Could not read logs: {e}")

        # Wait for health check if requested
        if wait_for_health:
            print("[Daytona] Waiting for localapi to be ready...")
            await self._wait_for_health(health_timeout)

        print(f"[Daytona] Localapi started: {self.preview_url}")
        return self.preview_url

    async def _wait_for_health(self, timeout: float = 60.0) -> None:
        """Wait for localapi health check."""
        import time

        import httpx

        start = time.time()
        health_url = f"{self.preview_url}/health"

        while time.time() - start < timeout:
            try:
                response = httpx.get(health_url, timeout=5.0)
                if response.status_code in (200, 400):
                    print("[Daytona] Localapi health check passed")
                    return
            except Exception:  # noqa: S110
                pass
            await asyncio.sleep(1.0)

        raise RuntimeError(f"Health check failed after {timeout}s: {health_url}")

    async def create_snapshot(
        self, snapshot_name: str | None = None, image: str | None = None
    ) -> str:
        """Create a snapshot from a container image.

        Note: This creates a snapshot from a Docker image, NOT from a running sandbox.
        To create a cached snapshot:
        1. Build a Docker image with your deps pre-installed
        2. Push to a registry
        3. Call this method with the image URL

        Args:
            snapshot_name: Name for the snapshot (default: SNAPSHOT_NAME)
            image: Container image to create snapshot from (required)

        Returns:
            Snapshot name
        """
        name = snapshot_name or self.SNAPSHOT_NAME
        img = image or self.image

        print(f"[Daytona] Creating snapshot '{name}' from image: {img}")

        try:
            from daytona_sdk import CreateSnapshotParams

            params = CreateSnapshotParams(name=name, image=img)

            def on_logs(chunk: str) -> None:
                print(f"[Daytona] {chunk}", end="")

            await asyncio.to_thread(self.client.snapshot.create, params, on_logs=on_logs)
            print(f"\n[Daytona] ✅ Snapshot created: {name}")
            return name
        except Exception as e:
            print(f"[Daytona] ❌ Failed to create snapshot: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up the sandbox."""
        if self.sandbox and self.sandbox_id:
            print(f"[Daytona] Cleaning up sandbox: {self.sandbox_id}")
            try:
                await asyncio.to_thread(
                    self.client.delete,
                    self.sandbox,
                )
                print("[Daytona] Sandbox deleted")
            except Exception as e:
                print(f"[Daytona] Warning: Failed to delete sandbox: {e}")

        self.sandbox = None
        self.sandbox_id = None
        self.preview_url = None

    async def __aenter__(self) -> "DaytonaLocalapiRunner":
        """Async context manager entry."""
        await self.provision()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self.cleanup()


async def run_localapi_in_daytona(
    localapi_path: Path,
    *,
    api_key: str | None = None,
    env_vars: dict[str, str] | None = None,
    additional_files: list[Path] | None = None,
    install_commands: list[str] | None = None,
    localapi_port: int = 8000,
) -> tuple[str, DaytonaLocalapiRunner]:
    """Convenience function to run a localapi in Daytona.

    Args:
        localapi_path: Path to localapi Python file
        api_key: Daytona API key (or use DAYTONA_API_KEY env var)
        env_vars: Environment variables to set in sandbox
        additional_files: Additional files to upload
        install_commands: Commands to run for setup
        localapi_port: Port for localapi

    Returns:
        Tuple of (localapi_url, runner_instance)

    Example:
        >>> url, runner = await run_localapi_in_daytona(
        ...     Path("my_localapi.py"),
        ...     env_vars={"SYNTH_API_KEY": "sk_..."},
        ... )
        >>> # Use url for eval job
        >>> await runner.cleanup()
    """
    runner = DaytonaLocalapiRunner(
        api_key=api_key,
        localapi_port=localapi_port,
    )

    await runner.provision()

    # Upload localapi
    await runner.upload_localapi(
        localapi_path,
        additional_files=additional_files,
    )

    # Setup environment
    await runner.setup_environment(
        env_vars=env_vars,
        install_commands=install_commands,
    )

    # Start localapi
    url = await runner.start_localapi()

    return url, runner


class DaytonaRolloutRunner:
    """Runs a single engine_bench rollout in a dedicated Daytona sandbox.

    This enables true parallelism by giving each rollout its own sandbox.
    The snapshot should include: Rust, cargo, codex CLI, engine-bench repo.
    """

    SNAPSHOT_NAME = "synth-engine-bench-codex-v1"  # New snapshot with codex
    ENGINE_BENCH_PATH = "/engine-bench"  # Pre-cloned in snapshot

    def __init__(
        self,
        *,
        api_key: str | None = None,
        snapshot_name: str | None = None,
    ) -> None:
        if not DAYTONA_AVAILABLE:
            raise RuntimeError("Daytona SDK not available")

        self.api_key = api_key or os.environ.get("DAYTONA_API_KEY")
        if not self.api_key:
            raise RuntimeError("DAYTONA_API_KEY required")

        self.snapshot_name = snapshot_name or self.SNAPSHOT_NAME
        self.client = Daytona(DaytonaConfig(api_key=self.api_key))
        self.sandbox = None
        self.sandbox_id = None

    async def run_rollout(
        self,
        *,
        instance_id: str,
        instance_data: dict[str, Any],
        prompt: str,
        model: str = "codex-5.1-mini",
        timeout: int = 300,
        openai_api_key: str,
        inference_url: str | None = None,
        agent_type: str = "codex",  # "codex" or "opencode"
    ) -> dict[str, Any]:
        """Run a complete rollout in a dedicated sandbox.

        Args:
            instance_id: e.g. "df-023-tropius"
            instance_data: The instance JSON data
            prompt: Full prompt for the agent
            model: Model to use
            timeout: Agent timeout in seconds
            openai_api_key: API key for LLM calls
            inference_url: Optional interceptor URL
            agent_type: "codex" or "opencode"

        Returns:
            dict with: passed, total, output, success, error
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # 1. Provision sandbox from snapshot
            print(f"[DaytonaRollout:{instance_id}] Provisioning sandbox...")
            await self._provision()
            provision_time = asyncio.get_event_loop().time() - start_time
            print(
                f"[DaytonaRollout:{instance_id}] Sandbox ready in {provision_time:.1f}s: {self.sandbox_id}"
            )

            # 2. Setup instance files
            print(f"[DaytonaRollout:{instance_id}] Setting up instance...")
            await self._setup_instance(instance_id, instance_data)

            # 3. Write prompt to file
            await self._write_file("/app/prompt.txt", prompt)

            # 4. Run agent (codex or opencode)
            print(
                f"[DaytonaRollout:{instance_id}] Running {agent_type} agent (model={model}, timeout={timeout}s)..."
            )
            if agent_type == "opencode":
                agent_result = await self._run_opencode_agent(
                    model=model,
                    timeout=timeout,
                    openai_api_key=openai_api_key,
                    inference_url=inference_url,
                )
            else:
                agent_result = await self._run_codex_agent(
                    model=model,
                    timeout=timeout,
                    openai_api_key=openai_api_key,
                    inference_url=inference_url,
                )

            if not agent_result.get("success"):
                return {
                    "passed": 0,
                    "total": 1,
                    "output": agent_result.get("stderr", "Agent failed"),
                    "success": False,
                    "error": f"Agent failed: {agent_result.get('stderr', '')[:500]}",
                }

            # 5. Inject eval tests
            print(f"[DaytonaRollout:{instance_id}] Injecting eval tests...")
            await self._inject_eval_tests(instance_id)

            # 6. Run cargo test
            print(f"[DaytonaRollout:{instance_id}] Running cargo test...")
            test_result = await self._run_cargo_test(instance_id)

            total_time = asyncio.get_event_loop().time() - start_time
            print(
                f"[DaytonaRollout:{instance_id}] Complete in {total_time:.1f}s: {test_result['passed']}/{test_result['total']} tests"
            )

            return test_result

        except Exception as e:
            import traceback

            return {
                "passed": 0,
                "total": 1,
                "output": traceback.format_exc(),
                "success": False,
                "error": str(e),
            }
        finally:
            await self._cleanup()

    async def _provision(self) -> None:
        """Provision sandbox from snapshot."""
        from daytona_sdk import CreateSandboxFromSnapshotParams

        params = CreateSandboxFromSnapshotParams(
            snapshot=self.snapshot_name,
            language=CodeLanguage.PYTHON,
        )
        self.sandbox = await asyncio.to_thread(self.client.create, params)
        self.sandbox_id = self.sandbox.id

    async def _exec(self, cmd: str, timeout: int = 60) -> dict[str, Any]:
        """Execute command in sandbox."""
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self.sandbox.process.exec, cmd),
                timeout=timeout,
            )
            exit_code = getattr(result, "exit_code", 0) or getattr(result, "return_code", 0) or 0
            output = getattr(result, "output", "") or getattr(result, "result", "") or ""
            return {"success": exit_code == 0, "output": output, "exit_code": exit_code}
        except TimeoutError:
            return {"success": False, "output": f"Timeout after {timeout}s", "exit_code": -1}
        except Exception as e:
            return {"success": False, "output": str(e), "exit_code": -1}

    async def _write_file(self, path: str, content: str) -> None:
        """Write file to sandbox."""
        await asyncio.to_thread(
            self.sandbox.fs.upload_file,
            content.encode("utf-8"),
            path,
        )

    async def _read_file(self, path: str) -> str:
        """Read file from sandbox."""
        content = await asyncio.to_thread(
            self.sandbox.fs.download_file,
            path,
        )
        return content.decode("utf-8") if isinstance(content, bytes) else content

    async def _setup_instance(self, instance_id: str, instance_data: dict[str, Any]) -> None:
        """Setup instance files in sandbox."""
        # The scaffold is in /engine-bench/scaffold
        # Copy it to /app/tcg_expansions
        result = await self._exec(
            "cp -r /engine-bench/scaffold /app/tcg_expansions && ls -la /app/tcg_expansions/src/",
            timeout=30,
        )
        if not result["success"]:
            raise RuntimeError(f"Failed to copy scaffold: {result['output']}")

        # Get card file info
        card_file = instance_data.get("card_file", "")
        if card_file:
            # The stub file should exist in gold/stubs
            stub_file = f"/engine-bench/gold/stubs/{instance_id.replace('-', '_')}.rs"
            relative_path = card_file.replace("tcg_expansions/", "")
            target_path = f"/app/tcg_expansions/{relative_path}"

            # Ensure parent dir exists
            parent_dir = "/".join(target_path.split("/")[:-1])
            await self._exec(f"mkdir -p {parent_dir}")

            # Read gold stub and convert to todo!() stubs
            try:
                gold_content = await self._read_file(stub_file)
                import re

                stub_content = re.sub(
                    r"(pub fn \w+\([^)]*\)\s*->\s*\w+\s*\{)[^}]+\}", r"\1 todo!() }", gold_content
                )
                await self._write_file(target_path, stub_content)
            except Exception as e:
                print(f"[DaytonaRollout] Warning: Could not setup stub: {e}")

        # Setup expansion module if needed
        expansion = instance_id.split("-")[0]
        if expansion not in ["df", "cg"]:
            card_module = instance_id.replace("-", "_")
            # Add module declarations
            await self._exec(f"""
                echo "pub mod {card_module};" >> /app/tcg_expansions/src/{expansion}/cards/mod.rs 2>/dev/null || true
            """)

    async def _run_codex_agent(
        self,
        *,
        model: str,
        timeout: int,
        openai_api_key: str,
        inference_url: str | None = None,
    ) -> dict[str, Any]:
        """Run codex agent in sandbox."""
        # Remove any auth.json that might override API config
        await self._exec("rm -f /root/.codex/auth.json /root/.codex/auth.json.bak", timeout=10)

        # Build base URL for config - normalize from inference_url
        # The interceptor URL format: http://host/api/interceptor/openai/v1/responses?cid=xxx
        base_url = "https://api.openai.com/v1"
        if inference_url:
            base_url, _ = normalize_interceptor_base(inference_url)
            # For responses API, the base should end with /v1 or similar
            # The interceptor path is like /api/interceptor/openai/v1
            print(f"[DaytonaRollout] Codex base_url from inference_url: {base_url}")

        # Determine wire_api based on model - both codex and opencode use responses API
        wire_api = "responses"

        # Write config.toml - this sets the base URL for all API calls
        config_content = f'''# Auto-generated config for engine_bench Daytona sandbox
model = "{model}"
model_provider = "openai"

[model_providers.openai]
name = "OpenAI"
base_url = "{base_url}"
wire_api = "{wire_api}"
env_key = "OPENAI_API_KEY"
requires_openai_auth = false
request_max_retries = 4
stream_max_retries = 5
stream_idle_timeout_ms = 300000

[mcp]
enabled = false
'''
        await self._write_file("/root/.codex/config.toml", config_content)
        print(
            f"[DaytonaRollout] Configured codex config.toml: base_url={base_url} wire_api={wire_api}"
        )

        # Build env vars - OPENAI_BASE_URL is critical for codex to use interceptor
        env_vars = f'OPENAI_API_KEY="{openai_api_key}" OPENAI_MODEL="{model}"'
        if inference_url:
            env_vars += f' OPENAI_BASE_URL="{base_url}"'
            print(f"[DaytonaRollout] Setting OPENAI_BASE_URL={base_url}")

        # Run codex
        cmd = f"""
cd /app/tcg_expansions && \
{env_vars} \
codex exec --yolo --skip-git-repo-check -m {model} "$(cat /app/prompt.txt)"
"""
        result = await self._exec(cmd, timeout=timeout)
        return {
            "success": result["success"],
            "stdout": result["output"],
            "stderr": result["output"] if not result["success"] else "",
        }

    async def _run_opencode_agent(
        self,
        *,
        model: str,
        timeout: int,
        openai_api_key: str,
        inference_url: str | None = None,
    ) -> dict[str, Any]:
        """Run opencode agent in sandbox."""
        # Build base URL - normalize from inference_url (same as local version)
        base_url = "https://api.openai.com/v1"
        if inference_url:
            base_url, _ = normalize_interceptor_base(inference_url)
            print(f"[DaytonaRollout] OpenCode using interceptor base: {base_url}")

        # Strip provider prefix if present (e.g., "openai/gpt-4o-mini" -> "gpt-4o-mini")
        model_id = model.split("/", 1)[1] if "/" in model else model
        model_with_provider = f"openai/{model_id}"

        # Write opencode.json config - match the local version exactly
        config_content = f'''{{
  "$schema": "https://opencode.ai/config.json",
  "model": "{model_with_provider}",
  "provider": {{
    "openai": {{
      "name": "OpenAI",
      "npm": "@ai-sdk/openai",
      "options": {{
        "apiKey": "{openai_api_key}",
        "baseURL": "{base_url}"
      }},
      "models": {{
        "gpt-5-nano": {{}},
        "gpt-5.2": {{}},
        "gpt-4o": {{}},
        "gpt-4o-mini": {{}},
        "codex-5.1-mini": {{}}
      }}
    }}
  }},
  "permission": {{
    "*": "allow",
    "external_directory": "allow",
    "bash": "allow",
    "read": "allow",
    "write": "allow",
    "edit": "allow",
    "list": "allow",
    "glob": "allow",
    "grep": "allow"
  }}
}}'''
        await self._write_file("/app/tcg_expansions/opencode.json", config_content)
        print(
            f"[DaytonaRollout] OpenCode config written: model={model_with_provider} baseURL={base_url}"
        )

        # Also create AGENTS.md to prevent infinite loop
        await self._write_file(
            "/app/tcg_expansions/AGENTS.md",
            "# Agent Instructions\n\nSee the task prompt for instructions.\n",
        )

        # Run opencode using 'run' subcommand for non-interactive mode
        # Pass OPENAI_API_KEY in env (like local version does)
        cmd = f'''
cd /app/tcg_expansions && \
OPENAI_API_KEY="{openai_api_key}" \
/root/.opencode/bin/opencode run --format json --model {model_with_provider} "$(cat /app/prompt.txt)"
'''
        result = await self._exec(cmd, timeout=timeout)
        return {
            "success": result["success"],
            "stdout": result["output"],
            "stderr": result["output"] if not result["success"] else "",
        }

    async def _inject_eval_tests(self, instance_id: str) -> None:
        """Inject evaluation tests into the card file."""
        expansion = instance_id.split("-")[0]
        card_module = instance_id.replace("-", "_")
        card_file = f"/app/tcg_expansions/src/{expansion}/cards/{card_module}.rs"
        test_file = f"/engine-bench/gold/tests/{card_module}_eval.rs"

        # Check if test file exists
        result = await self._exec(f"test -f {test_file} && echo EXISTS")
        if "EXISTS" not in result["output"]:
            print(f"[DaytonaRollout] Warning: No eval tests found at {test_file}")
            return

        # Append tests to card file
        await self._exec(f"""
echo "" >> {card_file}
echo "// EVALUATION TESTS" >> {card_file}
cat {test_file} >> {card_file}
""")

    async def _run_cargo_test(self, instance_id: str) -> dict[str, Any]:
        """Run cargo test and parse results."""
        card_module = instance_id.replace("-", "_")
        test_filter = f"{card_module}::eval"

        result = await self._exec(
            f"cd /app/tcg_expansions && cargo test -- --test-threads=1 {test_filter} 2>&1",
            timeout=120,
        )

        output = result["output"]

        # Parse test results
        import re

        passed = 0
        total = 0

        # Look for "test result: ok. X passed"
        match = re.search(r"test result: \w+\. (\d+) passed", output)
        if match:
            passed = int(match.group(1))

        # Count total tests from "running X tests"
        match = re.search(r"running (\d+) tests?", output)
        if match:
            total = int(match.group(1))

        if total == 0:
            total = 1  # At least 1 test expected

        return {
            "passed": passed,
            "total": total,
            "output": output,
            "success": result["success"],
            "error": None if result["success"] else output[:500],
        }

    async def _cleanup(self) -> None:
        """Clean up sandbox."""
        if self.sandbox:
            try:
                await asyncio.to_thread(self.client.delete, self.sandbox)
            except Exception as e:
                print(f"[DaytonaRollout] Cleanup warning: {e}")
            self.sandbox = None
            self.sandbox_id = None


# Global pool for reusing Daytona clients
_daytona_client: Daytona | None = None


def get_daytona_client() -> Daytona:
    """Get or create a shared Daytona client."""
    global _daytona_client
    if _daytona_client is None:
        api_key = os.environ.get("DAYTONA_API_KEY")
        if not api_key:
            raise RuntimeError("DAYTONA_API_KEY required")
        _daytona_client = Daytona(DaytonaConfig(api_key=api_key))
    return _daytona_client


async def run_rollout_in_daytona(
    *,
    instance_id: str,
    instance_data: dict[str, Any],
    prompt: str,
    model: str = "codex-5.1-mini",
    timeout: int = 300,
    openai_api_key: str,
    inference_url: str | None = None,
    snapshot_name: str | None = None,
    agent_type: str = "codex",  # "codex" or "opencode"
) -> dict[str, Any]:
    """Convenience function to run a rollout in a Daytona sandbox.

    This is the main entry point for per-rollout sandbox execution.
    Each call provisions a new sandbox, runs the agent, and cleans up.

    Args:
        agent_type: "codex" or "opencode" - which agent CLI to use
    """
    # Apply INTERCEPTOR_TUNNEL_URL if set (for Daytona sandboxes to reach interceptor)
    interceptor_tunnel_url = os.environ.get("INTERCEPTOR_TUNNEL_URL")
    if interceptor_tunnel_url and inference_url:
        parsed = urlparse(inference_url)
        tunnel_parsed = urlparse(interceptor_tunnel_url)
        inference_url = urlunparse(
            (
                tunnel_parsed.scheme,
                tunnel_parsed.netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )
        print(f"[DaytonaRollout] Rewriting inference_url for tunnel: {inference_url[:80]}...")

    runner = DaytonaRolloutRunner(snapshot_name=snapshot_name)
    return await runner.run_rollout(
        instance_id=instance_id,
        instance_data=instance_data,
        prompt=prompt,
        model=model,
        timeout=timeout,
        openai_api_key=openai_api_key,
        agent_type=agent_type,
        inference_url=inference_url,
    )
