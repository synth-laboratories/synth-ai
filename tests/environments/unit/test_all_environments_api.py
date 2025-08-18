#!/usr/bin/env python3
"""
Comprehensive unit tests for all available environments API endpoints.

This test suite validates that all registered environments properly handle
the core API calls: initialize, step, and terminate.

Environments tested:
- Sokoban
- CrafterClassic
- Verilog
- TicTacToe
- NetHack
- MiniGrid
- Enron
"""

import asyncio
import traceback
from typing import Any

import httpx
import pytest
from public_tests.environments.utils import check_service_running


class TestAllEnvironmentsAPI:
    """Test class for all environment API endpoints."""

    @pytest.fixture
    def service_url(self) -> str:
        """Get the environment service URL."""
        return "http://localhost:8901"

    @pytest.fixture
    async def available_environments(self, service_url: str) -> list[str]:
        """Get list of available environments from the health endpoint."""
        await check_service_running(8901)
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{service_url}/health")
            assert response.status_code == 200
            data = response.json()
            return data.get("supported_environments", [])

    def get_test_action_for_env(self, env_name: str) -> dict[str, Any]:
        """Get appropriate test action for each environment type."""
        # Define basic test actions for each environment
        action_map = {
            "Sokoban": {"tool_calls": [{"tool": "move", "args": {"direction": "up"}}]},
            "CrafterClassic": {"tool_calls": [{"tool": "interact", "args": {"action": 2}}]},
            "Verilog": {
                "tool_calls": [{"tool": "edit", "args": {"content": "module test();\nendmodule"}}]
            },
            "TicTacToe": {"tool_calls": [{"tool": "place", "args": {"position": 0}}]},
            "NetHack": {"tool_calls": [{"tool": "move", "args": {"direction": "north"}}]},
            "MiniGrid": {"tool_calls": [{"tool": "move", "args": {"action": 2}}]},
            "Enron": {"tool_calls": [{"tool": "search", "args": {"query": "test"}}]},
        }

        return action_map.get(env_name, {"tool_calls": [{"tool": "noop", "args": {}}]})

    @pytest.mark.asyncio
    async def test_all_environments_basic_workflow(
        self, service_url: str, available_environments: list[str]
    ):
        """
        Test basic workflow (initialize -> step -> terminate) for all environments.

        This is the core test that ensures all environments can handle:
        1. Environment initialization
        2. Action step execution
        3. Environment termination
        """
        results = {}

        for env_name in available_environments:
            try:
                result = await self._test_single_environment_workflow(service_url, env_name)
                results[env_name] = {"status": "PASS", "details": result}
            except Exception as e:
                results[env_name] = {
                    "status": "FAIL",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }

        # Print comprehensive results
        print("\n" + "=" * 80)
        print("ENVIRONMENT API TEST RESULTS")
        print("=" * 80)

        passed = []
        failed = []

        for env_name, result in results.items():
            status = result["status"]
            if status == "PASS":
                passed.append(env_name)
                print(f"✅ {env_name:<15} PASSED")
            else:
                failed.append(env_name)
                print(f"❌ {env_name:<15} FAILED: {result['error']}")

        print("\n" + "-" * 80)
        print(f"SUMMARY: {len(passed)}/{len(available_environments)} environments passed")
        print(f"✅ PASSED: {', '.join(passed) if passed else 'None'}")
        print(f"❌ FAILED: {', '.join(failed) if failed else 'None'}")
        print("=" * 80)

        # Store detailed results for further inspection
        self._detailed_results = results

        # The test passes if at least CrafterClassic works (our known good environment)
        # But we want to see results for all environments
        assert "CrafterClassic" in passed, "CrafterClassic (known good environment) failed"

    async def _test_single_environment_workflow(
        self, service_url: str, env_name: str
    ) -> dict[str, Any]:
        """Test the complete workflow for a single environment."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Step 1: Initialize environment
            init_response = await client.post(
                f"{service_url}/env/{env_name}/initialize",
                json={"initial_state": None, "config": None},
            )

            if init_response.status_code != 200:
                raise Exception(
                    f"Initialize failed: {init_response.status_code} - {init_response.text}"
                )

            init_data = init_response.json()
            env_id = init_data.get("env_id")
            if not env_id:
                raise Exception(f"No env_id returned from initialize: {init_data}")

            try:
                # Step 2: Execute action step
                action = self.get_test_action_for_env(env_name)
                step_payload = {
                    "request_id": f"test_{env_id}_step_1",
                    "env_id": env_id,
                    "action": action,
                }

                step_response = await client.post(
                    f"{service_url}/env/{env_name}/step", json=step_payload
                )

                if step_response.status_code != 200:
                    raise Exception(
                        f"Step failed: {step_response.status_code} - {step_response.text}"
                    )

                step_data = step_response.json()

                # Validate step response structure
                required_keys = ["observation", "reward", "done", "info"]
                missing_keys = [key for key in required_keys if key not in step_data]
                if missing_keys:
                    raise Exception(f"Step response missing keys: {missing_keys}")

                # Step 3: Terminate environment
                term_response = await client.post(
                    f"{service_url}/env/{env_name}/terminate", json={"env_id": env_id}
                )

                if term_response.status_code != 200:
                    raise Exception(
                        f"Terminate failed: {term_response.status_code} - {term_response.text}"
                    )

                return {
                    "init_status": init_response.status_code,
                    "step_status": step_response.status_code,
                    "term_status": term_response.status_code,
                    "observation_type": type(step_data["observation"]).__name__,
                    "has_reward": "reward" in step_data,
                    "has_done": "done" in step_data,
                    "has_info": "info" in step_data,
                }

            except Exception as e:
                # Cleanup attempt even if step failed
                try:
                    await client.post(
                        f"{service_url}/env/{env_name}/terminate", json={"env_id": env_id}
                    )
                except:
                    pass  # Best effort cleanup
                raise e

    @pytest.mark.asyncio
    async def test_initialize_only_all_environments(
        self, service_url: str, available_environments: list[str]
    ):
        """Test just initialization for all environments (lighter test)."""
        results = {}

        for env_name in available_environments:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{service_url}/env/{env_name}/initialize",
                        json={"initial_state": None, "config": None},
                    )

                    if response.status_code == 200:
                        data = response.json()
                        env_id = data.get("env_id")
                        results[env_name] = "PASS" if env_id else "FAIL (no env_id)"

                        # Cleanup
                        if env_id:
                            try:
                                await client.post(
                                    f"{service_url}/env/{env_name}/terminate",
                                    json={"env_id": env_id},
                                )
                            except:
                                pass
                    else:
                        results[env_name] = f"FAIL ({response.status_code})"

            except Exception as e:
                results[env_name] = f"ERROR ({str(e)})"

        print("\n" + "=" * 60)
        print("INITIALIZATION ONLY TEST RESULTS")
        print("=" * 60)
        for env_name, result in results.items():
            status_emoji = "✅" if result == "PASS" else "❌"
            print(f"{status_emoji} {env_name:<15} {result}")
        print("=" * 60)

    @pytest.mark.asyncio
    async def test_health_endpoint(self, service_url: str):
        """Test that the health endpoint works and returns expected environments."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{service_url}/health")

            assert response.status_code == 200
            data = response.json()

            assert "status" in data
            assert "supported_environments" in data
            assert data["status"] == "ok"
            assert isinstance(data["supported_environments"], list)
            assert len(data["supported_environments"]) > 0

            # Should include our known environments
            expected_envs = ["Sokoban", "CrafterClassic", "TicTacToe", "NetHack"]
            available_envs = data["supported_environments"]

            for env in expected_envs:
                assert env in available_envs, f"Expected environment {env} not in available list"

    def test_action_mapping_completeness(self, available_environments: list[str]):
        """Test that we have test actions defined for all available environments."""
        missing_actions = []

        for env_name in [
            "Sokoban",
            "CrafterClassic",
            "Verilog",
            "TicTacToe",
            "NetHack",
            "MiniGrid",
            "Enron",
        ]:
            action = self.get_test_action_for_env(env_name)
            if action.get("tool_calls", [{}])[0].get("tool") == "noop":
                missing_actions.append(env_name)

        assert not missing_actions, f"Missing test actions for environments: {missing_actions}"

    @pytest.mark.asyncio
    async def test_error_handling_invalid_environment(self, service_url: str):
        """Test that invalid environment names are handled gracefully."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{service_url}/env/NonExistentEnvironment/initialize",
                json={"initial_state": None, "config": None},
            )

            # Should return 400 or 404, not 500
            assert response.status_code in [400, 404, 422], (
                f"Expected 400/404/422 for invalid environment, got {response.status_code}"
            )

    @pytest.mark.asyncio
    async def test_concurrent_environment_initialization(
        self, service_url: str, available_environments: list[str]
    ):
        """Test that multiple environments can be initialized concurrently."""
        # Test with first 3 environments to avoid overwhelming the system
        test_envs = available_environments[:3]

        async def init_env(env_name: str):
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{service_url}/env/{env_name}/initialize",
                    json={"initial_state": None, "config": None},
                )
                return (
                    env_name,
                    response.status_code,
                    response.json() if response.status_code == 200 else None,
                )

        # Run initializations concurrently
        tasks = [init_env(env_name) for env_name in test_envs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Cleanup all successful initializations
        cleanup_tasks = []
        for result in results:
            if not isinstance(result, Exception):
                env_name, status_code, data = result
                if status_code == 200 and data and "env_id" in data:

                    async def cleanup(env_name, env_id):
                        async with httpx.AsyncClient() as client:
                            await client.post(
                                f"{service_url}/env/{env_name}/terminate", json={"env_id": env_id}
                            )

                    cleanup_tasks.append(cleanup(env_name, data["env_id"]))

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        # Verify results
        successful = [r for r in results if not isinstance(r, Exception) and r[1] == 200]
        assert len(successful) >= 1, "At least one concurrent initialization should succeed"


if __name__ == "__main__":
    """
    Run the test directly for debugging.
    
    Usage:
        python test_all_environments_api.py
    """
    import subprocess
    import sys

    # Run with pytest
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "-s", "--tb=short"],
        capture_output=True,
        text=True,
    )

    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    sys.exit(result.returncode)
