import pytest
from synth_ai.environments.examples.red.engine import (
    PokemonRedEngine,
    BUTTON_MAP,
    PokemonRedEngineSnapshot,
)
from synth_ai.environments.examples.red.taskset import (
    INSTANCE as DEFAULT_TASK,
)


class TestPokemonRedEngine:
    """Test Pokemon Red engine functionality with REAL ROM"""

    @pytest.fixture
    def task_instance(self):
        """Create a task instance"""
        return DEFAULT_TASK

    def test_button_map_completeness(self):
        """Test that all expected buttons are mapped"""
        expected_buttons = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"]
        assert all(button in BUTTON_MAP for button in expected_buttons)

        # Each button should map to a string (PyBoy event name)
        for button, mapped in BUTTON_MAP.items():
            assert isinstance(mapped, str)
            assert mapped

    def test_engine_initialization(self, task_instance):
        """Test engine initialization with REAL ROM"""
        engine = PokemonRedEngine(task_instance)

        assert engine.task_instance == task_instance
        assert engine._total_reward == 0.0
        assert engine._step_count == 0
        assert engine._previous_state is None
        assert engine.emulator is not None  # Should have real PyBoy instance

    def test_rom_path_resolution(self, task_instance):
        """Test ROM path resolution logic"""
        engine = PokemonRedEngine(task_instance)
        rom_path = engine._get_rom_path()

        # Should find the actual ROM file
        assert rom_path.exists()
        assert rom_path.name == "pokemon_red.gb"

    @pytest.mark.asyncio
    async def test_press_button_real(self, task_instance):
        """Test button press functionality with real ROM"""
        engine = PokemonRedEngine(task_instance)

        # Test valid button press - should not raise exception
        engine._press_button("A", 1)

        # Test multiple frames
        engine._press_button("RIGHT", 3)

    @pytest.mark.asyncio
    async def test_press_button_invalid(self, task_instance):
        """Test invalid button press"""
        engine = PokemonRedEngine(task_instance)

        with pytest.raises(ValueError, match="Invalid button: INVALID"):
            engine._press_button("INVALID")

    @pytest.mark.asyncio
    async def test_extract_current_state_real(self, task_instance):
        """Test state extraction from real emulator"""
        engine = PokemonRedEngine(task_instance)
        state = engine._extract_current_state()

        # Should return a dictionary with expected keys (from actual state extraction)
        expected_keys = [
            "map_id",
            "player_x",
            "player_y",
            "badges",
            "party_hp_current",
            "party_hp_max",
            "party_level",
            "party_xp",
            "in_battle",
            "battle_outcome",
            "inventory_count",
            "menu_state",
            "warp_flag",
        ]
        for key in expected_keys:
            assert key in state

        # Values should be correct types
        assert isinstance(state["map_id"], int)
        assert isinstance(state["player_x"], int)
        assert isinstance(state["player_y"], int)
        assert isinstance(state["badges"], int)
        assert isinstance(state["in_battle"], bool)

    @pytest.mark.asyncio
    async def test_reset_engine_real(self, task_instance):
        """Test engine reset with real ROM"""
        engine = PokemonRedEngine(task_instance)

        priv, pub = await engine._reset_engine()

        assert engine._total_reward == 0.0
        assert engine._step_count == 0
        assert priv.reward_last_step == 0.0
        assert priv.total_reward == 0.0
        assert not priv.terminated

        # Public state should have real values
        assert isinstance(pub.map_id, int)
        assert isinstance(pub.player_x, int)
        assert isinstance(pub.player_y, int)

    @pytest.mark.asyncio
    async def test_step_engine_real(self, task_instance):
        """Test engine step execution with real ROM"""
        engine = PokemonRedEngine(task_instance)
        await engine._reset_engine()

        action = {"button": "A", "frames": 1}
        priv, pub = await engine._step_engine(action)

        assert engine._step_count == 1
        assert priv.step_count == 1
        assert isinstance(priv.reward_last_step, float)
        assert priv.total_reward == engine._total_reward

        # Should have actual game state
        assert isinstance(pub.map_id, int)
        assert isinstance(pub.badges, int)
        assert isinstance(pub.party_hp_current, int)

    @pytest.mark.asyncio
    async def test_button_sequence_real(self, task_instance):
        """Test a sequence of button presses with real ROM"""
        engine = PokemonRedEngine(task_instance)
        await engine._reset_engine()

        # Try a sequence of different buttons
        buttons = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"]

        for i, button in enumerate(buttons):
            action = {"button": button, "frames": 1}
            priv, pub = await engine._step_engine(action)

            assert engine._step_count == i + 1
            assert priv.step_count == i + 1

            # Game state should remain consistent
            assert isinstance(pub.map_id, int)
            assert isinstance(pub.player_x, int)
            assert isinstance(pub.player_y, int)

    @pytest.mark.asyncio
    async def test_serialization_real(self, task_instance):
        """Test engine serialization with real ROM"""
        engine = PokemonRedEngine(task_instance)
        await engine._reset_engine()

        # Take a few steps to change state
        await engine._step_engine({"button": "A", "frames": 1})
        await engine._step_engine({"button": "RIGHT", "frames": 1})

        snapshot = await engine._serialize_engine()

        assert isinstance(snapshot, PokemonRedEngineSnapshot)
        assert snapshot.total_reward == engine._total_reward
        assert snapshot.step_count == engine._step_count
        assert "_save_state_bytes" in snapshot.state_data

    @pytest.mark.asyncio
    async def test_rom_memory_access(self, task_instance):
        """Test that we can actually read ROM memory"""
        engine = PokemonRedEngine(task_instance)

        # Should be able to access memory
        assert engine.emulator is not None
        assert hasattr(engine.emulator, "memory")

        # Try reading some memory locations
        badge_flags = engine.emulator.memory[0xD356]
        player_x = engine.emulator.memory[0xD362]
        player_y = engine.emulator.memory[0xD361]

        # Should be valid integers (even if zero initially)
        assert isinstance(badge_flags, int)
        assert isinstance(player_x, int)
        assert isinstance(player_y, int)
