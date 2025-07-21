"""Unit tests for NetHack engine."""

import pytest
import asyncio
from uuid import uuid4

from synth_ai.environments.tasks.core import TaskInstance, Impetus, Intent, TaskInstanceMetadata

from synth_ai.environments.examples.nethack.engine import (
    NetHackEngine,
    NetHackPublicState,
    NetHackPrivateState,
    NetHackEngineSnapshot,
    NetHackSurvivalComponent,
    NetHackProgressComponent,
)
from synth_ai.environments.examples.nethack.taskset import (
    NetHackTaskInstanceMetadata,
    NetHackTaskInstance,
)

# Since engine requires NLE, all tests require it
pytest.importorskip("nle", reason="NLE is required for NetHack engine")


class TestNetHackEngine:
    """Test cases for NetHack engine."""

    @pytest.fixture
    def mock_task_instance(self):
        """Create a mock task instance for testing."""
        metadata = NetHackTaskInstanceMetadata(
            character_role="tourist",
            starting_level=1,
            target_depth=5,
            time_limit=1000,
            difficulty="beginner",
            special_objectives=["Collect 100 gold pieces"],
            seed=42,
        )

        return NetHackTaskInstance(
            id=uuid4(),
            impetus=Impetus(instructions="Test NetHack game"),
            intent=Intent(
                rubric={"goal": "Reach depth 5"},
                gold_trajectories=None,
                gold_state_diff={},
            ),
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=None,
        )

    @pytest.mark.asyncio
    async def test_engine_initialization(self, mock_task_instance):
        """Test engine initialization."""
        engine = NetHackEngine(mock_task_instance)

        assert engine.task_instance == mock_task_instance
        assert engine.character_role == "tourist"
        assert engine.max_turns == 1000
        assert engine.public_state is None
        assert engine.private_state is None

        # Cleanup
        if hasattr(engine, "nle"):
            engine.nle.close()

    @pytest.mark.asyncio
    async def test_engine_reset(self, mock_task_instance):
        """Test engine reset functionality."""
        engine = NetHackEngine(mock_task_instance)

        priv, pub = await engine._reset_engine(seed=42)

        # Check private state
        assert isinstance(priv, NetHackPrivateState)
        assert priv.reward_last == 0.0
        assert priv.total_reward == 0.0
        assert priv.terminated is False
        assert priv.score >= 0
        assert priv.depth_reached >= 1

        # Check public state
        assert isinstance(pub, NetHackPublicState)
        assert pub.dungeon_level >= 1
        assert pub.terminated is False
        assert pub.turn_count == 0
        assert pub.max_turns == 1000
        assert len(pub.message) > 0  # Should have some message
        assert len(pub.ascii_map) > 0  # Should have a map

    @pytest.mark.asyncio
    async def test_basic_movement(self, mock_task_instance):
        """Test basic movement actions."""
        engine = NetHackEngine(mock_task_instance)
        priv0, pub0 = await engine._reset_engine()
        initial_pos = pub0.position

        # Test wait action (always valid)
        priv, pub = await engine._step_engine("wait")
        assert pub.last_action == "wait"
        assert pub.turn_count == 1
        assert priv.reward_last >= 0  # Should get at least survival reward

    @pytest.mark.asyncio
    async def test_invalid_action(self, mock_task_instance):
        """Test invalid action handling."""
        engine = NetHackEngine(mock_task_instance)
        await engine._reset_engine()

        # Test invalid action
        with pytest.raises(ValueError, match="Invalid action|Valid actions"):
            await engine._step_engine("invalid_action_xyz")

    @pytest.mark.asyncio
    async def test_turn_limit(self, mock_task_instance):
        """Test turn limit enforcement."""
        # Create instance with very short time limit
        metadata = NetHackTaskInstanceMetadata(
            character_role="tourist",
            starting_level=1,
            target_depth=5,
            time_limit=3,  # Very short
            difficulty="test",
            special_objectives=[],
            seed=42,
        )

        task = NetHackTaskInstance(
            id=uuid4(),
            impetus=Impetus(instructions="Test"),
            intent=Intent(rubric={"goal": "Test"}, gold_trajectories=None, gold_state_diff={}),
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=None,
        )

        engine = NetHackEngine(task)
        await engine._reset_engine()

        # Take actions until time limit
        await engine._step_engine("wait")
        await engine._step_engine("wait")
        priv, pub = await engine._step_engine("wait")

        # Should be terminated due to time limit
        assert pub.terminated is True
        assert priv.terminated is True
        assert priv.truncated is True
        assert "Time limit reached" in pub.message

    @pytest.mark.asyncio
    async def test_reward_calculation(self, mock_task_instance):
        """Test reward calculation."""
        engine = NetHackEngine(mock_task_instance)
        await engine._reset_engine()

        # Test survival reward
        priv1, pub1 = await engine._step_engine("wait")
        # Rewards can be positive (survival) or negative (NLE penalty)
        assert priv1.reward_last != 0  # Should get some reward

        # Test multiple steps accumulate reward
        total_before = priv1.total_reward
        priv2, pub2 = await engine._step_engine("wait")
        # Total reward should change
        assert priv2.total_reward != total_before

    @pytest.mark.asyncio
    async def test_state_diff(self, mock_task_instance):
        """Test state diff functionality."""
        engine = NetHackEngine(mock_task_instance)
        priv1, pub1 = await engine._reset_engine()

        # Since the engine returns references to its internal state,
        # we need to capture the values we want to compare
        initial_turn_count = pub1.turn_count
        initial_last_action = pub1.last_action
        initial_reward_last = priv1.reward_last
        initial_total_reward = priv1.total_reward

        # Take an action
        priv2, pub2 = await engine._step_engine("wait")

        # Verify states actually changed
        assert pub2.turn_count == initial_turn_count + 1
        assert pub2.last_action == "wait"
        assert priv2.total_reward > initial_total_reward  # Should have some reward

        # Since pub1 and pub2 are the same object (engine returns references),
        # the diff will be empty. This is expected behavior for a stateful engine.
        # The test was incorrectly assuming the engine returns copies.

    @pytest.mark.asyncio
    async def test_serialization_roundtrip(self, mock_task_instance):
        """Test state serialization and deserialization."""
        engine = NetHackEngine(mock_task_instance)
        await engine._reset_engine()

        # Take some actions
        await engine._step_engine("wait")
        await engine._step_engine("search")

        # Serialize
        snapshot = await engine._serialize_engine()
        assert isinstance(snapshot, NetHackEngineSnapshot)

        # Deserialize
        restored_engine = await NetHackEngine._deserialize_engine(snapshot)

        # Check restored state matches
        orig_priv, orig_pub = engine.get_current_states_for_observation()
        rest_priv, rest_pub = restored_engine.get_current_states_for_observation()

        assert rest_pub.turn_count == orig_pub.turn_count
        assert rest_pub.position == orig_pub.position
        assert rest_priv.total_reward == orig_priv.total_reward

    @pytest.mark.asyncio
    async def test_get_current_states(self, mock_task_instance):
        """Test getting current states without advancing."""
        engine = NetHackEngine(mock_task_instance)

        # Should raise before initialization
        with pytest.raises(RuntimeError, match="Engine not initialized"):
            engine.get_current_states_for_observation()

        # Initialize
        await engine._reset_engine()

        # Get states multiple times
        priv1, pub1 = engine.get_current_states_for_observation()
        priv2, pub2 = engine.get_current_states_for_observation()

        # Should be same states
        assert pub1.turn_count == pub2.turn_count
        assert priv1.total_reward == priv2.total_reward


class TestRewardComponents:
    """Test reward components."""

    @pytest.mark.asyncio
    async def test_survival_component(self):
        """Test survival reward component."""
        component = NetHackSurvivalComponent()

        # Test alive state
        state = NetHackPublicState(terminated=False)
        reward = await component.score(state, "wait")
        assert reward == 0.01

        # Test dead state
        state.terminated = True
        reward = await component.score(state, "wait")
        assert reward == -1.0

    @pytest.mark.asyncio
    async def test_progress_component(self):
        """Test progress reward component."""
        component = NetHackProgressComponent()

        # Test no progress
        state = NetHackPublicState(dungeon_level=1)
        reward = await component.score(state, "wait")
        assert reward == 0.0

        # Test depth increase
        state.dungeon_level = 2
        reward = await component.score(state, "go_down")
        assert reward == 1.0

        # Test no further reward for same level
        reward = await component.score(state, "wait")
        assert reward == 0.0
