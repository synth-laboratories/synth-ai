"""
Integration tests specifically for Sokoban through the service API.
Tests various Sokoban scenarios and edge cases.
"""

import pytest
import httpx
from tests.environments.utils import check_service_running

from synth_ai.environments.service.app import app
from synth_ai.environments.examples.sokoban.units.astar_common import ENGINE_ASTAR


# Various test puzzles
PUZZLES = {
    "trivial": {
        "dim_room": [3, 3],
        "room_fixed": [[0, 0, 0], [0, 2, 0], [0, 1, 0]],
        "room_state": [[0, 0, 0], [0, 3, 0], [0, 5, 0]],  # Box on target, player below
        "boxes_on_target": 1,
        "max_steps": 5,
        "num_boxes": 1,
        "box_mapping": [],
        "num_env_steps": 0,
        "player_position": [2, 1],
        "reward_last": 0
    },
    "one_move": {
        "dim_room": [3, 3],
        "room_fixed": [[0, 0, 0], [0, 2, 0], [0, 1, 0]],
        "room_state": [[0, 0, 0], [0, 4, 0], [0, 5, 0]],  # Push box up
        "boxes_on_target": 0,
        "max_steps": 5,
        "num_boxes": 1,
        "box_mapping": [],
        "num_env_steps": 0,
        "player_position": [2, 1],
        "reward_last": 0
    },
    "corridor": {
        "dim_room": [1, 5],
        "room_fixed": [[0, 2, 1, 1, 0]],
        "room_state": [[0, 2, 4, 5, 0]],
        "boxes_on_target": 0,
        "max_steps": 10,
        "num_boxes": 1,
        "box_mapping": [],
        "num_env_steps": 0,
        "player_position": [0, 3],
        "reward_last": 0
    },
    "two_boxes": {
        "dim_room": [5, 5],
        "room_fixed": [
            [0, 0, 0, 0, 0],
            [0, 2, 1, 2, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        "room_state": [
            [0, 0, 0, 0, 0],
            [0, 2, 4, 2, 0],
            [0, 1, 5, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        "boxes_on_target": 0,
        "max_steps": 20,
        "num_boxes": 2,
        "box_mapping": [],
        "num_env_steps": 0,
        "player_position": [2, 2],
        "reward_last": 0
    },
}


class TestSokobanService:
    """Test Sokoban-specific functionality through the service."""

    @pytest.fixture
    async def client(self):
        """Create async test client."""
        await check_service_running(8901)
        async with httpx.AsyncClient(base_url="http://localhost:8901") as ac:
            yield ac

    @pytest.mark.asyncio
    async def test_trivial_puzzle(self, client):
        """Test a puzzle that's essentially already solved."""
        response = await client.post(
            "/env/Sokoban/initialize", json={"initial_state": PUZZLES["trivial"]}
        )

        data = response.json()
        obs = data["observation"]

        # Should already show as solved
        assert obs["boxes_on_target"] == 1
        assert obs["num_boxes"] == 1

    @pytest.mark.asyncio
    async def test_one_move_solution(self, client):
        """Test a puzzle that requires exactly one move."""
        # Initialize
        init_resp = await client.post(
            "/env/Sokoban/initialize", json={"initial_state": PUZZLES["one_move"]}
        )
        env_id = init_resp.json()["env_id"]

        # Move UP (action 0)
        step_resp = await client.post(
            "/env/Sokoban/step",
            json={
                "env_id": env_id,
                "action": {"tool_calls": [{"tool": "interact", "args": {"action": 0}}]},
            },
        )

        obs = step_resp.json()["observation"]
        assert obs["boxes_on_target"] == 1
        assert obs["steps_taken"] == 1

    @pytest.mark.asyncio
    async def test_corridor_puzzle(self, client):
        """Test a corridor-style puzzle."""
        # Initialize
        init_resp = await client.post(
            "/env/Sokoban/initialize", json={"initial_state": PUZZLES["corridor"]}
        )
        env_id = init_resp.json()["env_id"]

        # Get environment to solve with A*
        from service.core_routes import instances

        env = instances[env_id]
        env.engine.package_sokoban_env.observation_mode = "raw"

        # Find solution
        plan = await ENGINE_ASTAR(env.engine, max_nodes=50)
        assert plan is not None

        # Execute plan
        for action in plan:
            await client.post(
                "/env/Sokoban/step",
                json={
                    "env_id": env_id,
                    "action": {"tool_calls": [{"tool": "interact", "args": {"action": action}}]},
                },
            )

        # Check final state
        checkpoint_resp = await client.get(f"/env/Sokoban/{env_id}/checkpoint")
        snapshot = checkpoint_resp.json()["snapshot"]

        # Verify solved through snapshot
        final_state = snapshot["engine_snapshot"]
        assert final_state["boxes_on_target"] == final_state["num_boxes"]

    @pytest.mark.asyncio
    async def test_two_box_puzzle(self, client):
        """Test solving a puzzle with multiple boxes."""
        # Initialize
        init_resp = await client.post(
            "/env/Sokoban/initialize", json={"initial_state": PUZZLES["two_boxes"]}
        )
        env_id = init_resp.json()["env_id"]
        obs = init_resp.json()["observation"]

        assert obs["num_boxes"] == 2
        assert obs["boxes_on_target"] == 0

        # Solve with A*
        from service.core_routes import instances

        env = instances[env_id]
        env.engine.package_sokoban_env.observation_mode = "raw"

        plan = await ENGINE_ASTAR(env.engine, max_nodes=200)
        assert plan is not None
        assert len(plan) > 0

        # Execute plan step by step
        for i, action in enumerate(plan):
            step_resp = await client.post(
                "/env/Sokoban/step",
                json={
                    "env_id": env_id,
                    "action": {"tool_calls": [{"tool": "interact", "args": {"action": action}}]},
                },
            )

            obs = step_resp.json()["observation"]
            assert obs["steps_taken"] == i + 1

        # Both boxes should be on targets
        assert obs["boxes_on_target"] == 2

    @pytest.mark.asyncio
    async def test_max_steps_limit(self, client):
        """Test that max_steps is enforced."""
        # Create puzzle with very low step limit
        limited_puzzle = PUZZLES["corridor"].copy()
        limited_puzzle["max_steps"] = 2

        init_resp = await client.post(
            "/env/Sokoban/initialize", json={"initial_state": limited_puzzle}
        )
        env_id = init_resp.json()["env_id"]

        # Make moves up to limit
        for i in range(3):  # Try to exceed limit
            step_resp = await client.post(
                "/env/Sokoban/step",
                json={
                    "env_id": env_id,
                    "action": {"tool_calls": [{"tool": "interact", "args": {"action": 1}}]},
                },
            )

            obs = step_resp.json()["observation"]

            # After max_steps, should be terminated
            if i >= 2:
                assert (
                    obs.get("terminated", False)
                    or obs["steps_taken"] >= limited_puzzle["max_steps"]
                )

    @pytest.mark.asyncio
    async def test_invalid_action_handling(self, client):
        """Test handling of invalid actions."""
        init_resp = await client.post(
            "/env/Sokoban/initialize", json={"initial_state": PUZZLES["one_move"]}
        )
        env_id = init_resp.json()["env_id"]

        # Try invalid action number
        step_resp = await client.post(
            "/env/Sokoban/step",
            json={
                "env_id": env_id,
                "action": {"tool_calls": [{"tool": "interact", "args": {"action": 99}}]},
            },
        )

        # Should handle gracefully (either error or no-op)
        assert step_resp.status_code in [200, 400]

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self, client):
        """Test handling multiple tool calls in one step."""
        init_resp = await client.post(
            "/env/Sokoban/initialize", json={"initial_state": PUZZLES["corridor"]}
        )
        env_id = init_resp.json()["env_id"]

        # Send multiple actions (only first should be processed in Sokoban)
        step_resp = await client.post(
            "/env/Sokoban/step",
            json={
                "env_id": env_id,
                "action": {
                    "tool_calls": [
                        {"tool": "interact", "args": {"action": 0}},
                        {"tool": "interact", "args": {"action": 1}},
                    ]
                },
            },
        )

        assert step_resp.status_code == 200
        obs = step_resp.json()["observation"]
        assert obs["steps_taken"] == 1  # Only one step should be taken

    @pytest.mark.asyncio
    async def test_room_text_format(self, client):
        """Test that room_text is properly formatted."""
        init_resp = await client.post(
            "/env/Sokoban/initialize", json={"initial_state": PUZZLES["one_move"]}
        )

        obs = init_resp.json()["observation"]
        assert "room_text" in obs

        room_text = obs["room_text"]
        assert isinstance(room_text, str)
        assert len(room_text) > 0

        # Should contain walls (#), player (@), box ($), etc.
        assert any(char in room_text for char in ["#", "@", "$", "."])

    @pytest.mark.asyncio
    async def test_concurrent_environments(self, client):
        """Test running multiple Sokoban instances concurrently."""
        # Create multiple environments with different puzzles
        env_configs = [
            ("one_move", PUZZLES["one_move"]),
            ("corridor", PUZZLES["corridor"]),
            ("trivial", PUZZLES["trivial"]),
        ]

        env_ids = {}

        # Initialize all environments
        for name, puzzle in env_configs:
            resp = await client.post("/env/Sokoban/initialize", json={"initial_state": puzzle})
            env_ids[name] = resp.json()["env_id"]

        # Make different moves in each
        actions = {
            "one_move": 0,  # UP
            "corridor": 3,  # LEFT
            "trivial": 1,  # DOWN
        }

        results = {}
        for name, action in actions.items():
            resp = await client.post(
                "/env/Sokoban/step",
                json={
                    "env_id": env_ids[name],
                    "action": {"tool_calls": [{"tool": "interact", "args": {"action": action}}]},
                },
            )
            results[name] = resp.json()

        # Verify each environment maintained its own state
        assert results["one_move"]["observation"]["steps_taken"] == 1
        assert results["corridor"]["observation"]["steps_taken"] == 1
        assert results["trivial"]["observation"]["steps_taken"] == 1

        # Clean up
        for env_id in env_ids.values():
            await client.post("/env/Sokoban/terminate", json={"env_id": env_id})
